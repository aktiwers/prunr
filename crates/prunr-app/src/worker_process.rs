//! Subprocess worker entry point.
//!
//! Invoked by the parent process as `prunr --worker`. Reads commands from
//! stdin, processes images using ORT, writes events to stdout. If this
//! process OOMs, the parent detects the crash and retries with reduced
//! concurrency.

use std::io::{BufReader, BufWriter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use prunr_core::{
    create_engine_pool,
    OrtEngine, ProcessResult, ProgressStage, EdgeEngine,
};

use prunr_app::subprocess::protocol::*;
use prunr_app::subprocess::ipc::{read_message, write_message};
use prunr_app::gui::settings::LineMode;

/// Weighted memory semaphore for processing.
/// Instead of a binary lock, this tracks "pixel units" (1 unit = 1M pixels).
/// Small images run in parallel; large images throttle automatically.
/// Total capacity is set based on available RAM at worker startup.
struct WeightedSemaphore {
    state: Mutex<usize>,
    available: std::sync::Condvar,
    total_units: usize,
}

impl WeightedSemaphore {
    fn new(total_units: usize) -> Self {
        Self {
            state: Mutex::new(total_units),
            available: std::sync::Condvar::new(),
            total_units,
        }
    }

    /// Acquire units, returning a RAII guard that releases on drop (panic-safe).
    fn acquire(self: &Arc<Self>, weight: usize) -> SemaphoreGuard {
        let capped = weight.min(self.total_units);
        // Poison recovery: the semaphore state is just a unit counter. A
        // panicking user of the semaphore must not deadlock the worker pool.
        let mut units = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while *units < capped {
            units = self.available.wait(units).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *units -= capped;
        SemaphoreGuard { sem: self.clone(), acquired: capped }
    }
}

/// RAII guard: releases semaphore units on drop (including panics).
struct SemaphoreGuard {
    sem: Arc<WeightedSemaphore>,
    acquired: usize,
}

impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        // Poison recovery: drop must not panic (would abort under double-panic).
        let mut units = self.sem.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *units += self.acquired;
        self.sem.available.notify_all();
    }
}

/// Compose a SubjectOutline-mode output: builds the primary edge mask (from
/// the active scale) and, if `LineStyle::DualScale` is active, also builds
/// a Bold mask and dispatches to the dual-scale compose. Shared by the
/// full-pipeline SubjectOutline branch and the AddEdgeInference branch.
fn compose_subject_outline(
    edge_res: &prunr_core::EdgeInferenceResult,
    masked_rgba: &image::RgbaImage,
    edge: &prunr_core::EdgeSettings,
) -> image::RgbaImage {
    use prunr_core::{EdgeScale, LineStyle};
    let active = &edge_res.tensors[edge.edge_scale as usize];
    let primary_mask = prunr_core::tensor_to_edge_mask(
        active, edge_res.height, edge_res.width,
        masked_rgba.width(), masked_rgba.height(),
        edge.line_strength,
    );
    if let LineStyle::DualScale { fine_color, bold_color } = edge.line_style {
        let bold = &edge_res.tensors[EdgeScale::Bold as usize];
        let bold_mask = prunr_core::tensor_to_edge_mask(
            bold, edge_res.height, edge_res.width,
            masked_rgba.width(), masked_rgba.height(),
            edge.line_strength,
        );
        prunr_core::compose_edges_dual_styled(
            &primary_mask, &bold_mask, masked_rgba,
            edge.compose_mode,
            fine_color, bold_color,
            edge.edge_thickness,
        )
    } else {
        prunr_core::compose_edges_styled(
            &primary_mask, masked_rgba,
            edge.compose_mode,
            edge.line_style,
            edge.solid_line_color, edge.edge_thickness,
        )
    }
}

/// Pack an `EdgeInferenceResult`'s per-scale tensors into one LE byte buffer
/// for IPC. Matches the layout the parent expects in `read_edge_tensor_cache`.
fn pack_edge_tensors(res: &prunr_core::EdgeInferenceResult) -> Vec<u8> {
    let per_tensor_floats = (res.height as usize) * (res.width as usize);
    let mut bytes = Vec::with_capacity(per_tensor_floats * prunr_core::EDGE_SCALE_COUNT * 4);
    for t in &res.tensors {
        bytes.extend_from_slice(prunr_app::subprocess::ipc::f32s_as_le_bytes(t));
    }
    bytes
}

/// Calculate pixel weight for an image (1 unit = 1M pixels, minimum 1).
fn pixel_weight(image_bytes: &[u8]) -> usize {
    // Read dimensions from header without full decode
    let dims = image::ImageReader::new(std::io::Cursor::new(image_bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.into_dimensions().ok());
    match dims {
        Some((w, h)) => ((w as usize * h as usize) / 1_000_000).max(1),
        None => 4, // conservative fallback (~4MP)
    }
}

/// Entry point for `prunr --worker`.
pub fn run_worker() -> ! {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    let mut reader = BufReader::new(stdin.lock());
    let (evt_tx, evt_rx) = mpsc::channel::<SubprocessEvent>();

    // Writer thread: evt_rx → stdout (bincode frames)
    let writer_handle = std::thread::Builder::new()
        .name("worker-writer".into())
        .spawn(move || {
            let mut writer = BufWriter::new(stdout.lock());
            while let Ok(evt) = evt_rx.recv() {
                if write_message(&mut writer, &evt).is_err() {
                    break;
                }
            }
        })
        .expect("failed to spawn writer thread");

    // Read Init command
    let Ok(Some(SubprocessCommand::Init {
        model, jobs, mask, force_cpu, line_mode, edge, ipc_dir,
    })) = read_message::<_, SubprocessCommand>(&mut reader)
    else {
        let _ = evt_tx.send(SubprocessEvent::InitError {
            error: "Expected Init command".to_string(),
        });
        drop(evt_tx);
        let _ = writer_handle.join();
        std::process::exit(1);
    };

    // Detect backend
    let has_gpu = !OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    let cpu_only = force_cpu || !has_gpu;

    // Load edge engine if needed
    let needs_edge = line_mode != LineMode::Off;
    let needs_segmentation = line_mode != LineMode::EdgesOnly;

    let edge_engine: Option<Arc<EdgeEngine>> = if needs_edge {
        match EdgeEngine::new() {
            Ok(e) => Some(Arc::new(e)),
            Err(e) => {
                let _ = evt_tx.send(SubprocessEvent::InitError {
                    error: format!("Edge engine failed: {e}"),
                });
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Create segmentation engine pool
    let engines: Vec<Arc<OrtEngine>> = if needs_segmentation {
        match create_engine_pool(model, jobs, cpu_only) {
            Ok(e) => e,
            Err(e) => {
                let _ = evt_tx.send(SubprocessEvent::InitError {
                    error: format!("Engine pool failed: {e}"),
                });
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(1);
            }
        }
    } else {
        Vec::new()
    };

    let active_provider = OrtEngine::detect_active_provider();
    let _ = evt_tx.send(SubprocessEvent::Ready {
        active_provider: active_provider.clone(),
    });

    // Build rayon pool for parallel inference
    let pool_size = engines.len().max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_size)
        .build()
        // Fallback to a default (unsized) rayon pool if the sized builder
        // somehow fails. Default-pool construction failing means rayon itself
        // can't function; no useful recovery possible.
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    let cancel = Arc::new(AtomicBool::new(false));
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine_idx: usize = 0;

    let ipc_dir = Arc::new(ipc_dir);

    // Weighted memory semaphore: capacity in megapixel units.
    // ~64 units = allows parallel processing of many small images,
    // but throttles large images (40MP = 40 units → only ~1 at a time).
    let semaphore = Arc::new(WeightedSemaphore::new(64));

    // Main command loop
    loop {
        let cmd = match read_message::<_, SubprocessCommand>(&mut reader) {
            Ok(Some(cmd)) => cmd,
            Ok(None) => break, // stdin closed — parent exited
            Err(_) => break,
        };

        match cmd {
            SubprocessCommand::ProcessImage { item_id, image_path, chain_input } => {
                let engine = if !engines.is_empty() {
                    Some(engines[engine_idx % engines.len()].clone())
                } else {
                    None
                };
                engine_idx += 1;
                in_flight.fetch_add(1, Ordering::AcqRel);

                let evt_tx = evt_tx.clone();
                let cancel = cancel.clone();
                let in_flight = in_flight.clone();
                let edge_eng = edge_engine.clone();
                let provider = active_provider.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();

                pool.spawn(move || {
                    if cancel.load(Ordering::Acquire) {
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        return;
                    }

                    // Read image bytes from temp file
                    let img_bytes = match std::fs::read(&image_path) {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError {
                                item_id,
                                error: format!("Failed to read image: {e}"),
                            });
                            in_flight.fetch_sub(1, Ordering::AcqRel);
                            return;
                        }
                    };

                    // Read chain input if present
                    let chain_path = chain_input.as_ref().map(|ci| ci.path.clone());
                    let chain_img: Option<image::DynamicImage> = chain_input.and_then(|ci| {
                        let data = std::fs::read(&ci.path).ok()?;
                        let rgba = image::RgbaImage::from_raw(ci.width, ci.height, data)?;
                        Some(image::DynamicImage::ImageRgba8(rgba))
                    });

                    // Weighted semaphore: acquire pixel units proportional to image size.
                    // Small images run in parallel; large images throttle automatically.
                    // Guard releases on drop (panic-safe).
                    let weight = pixel_weight(&img_bytes);
                    let _sem_guard = sem.acquire(weight);

                    // For LineMode::Off we use the split pipeline (infer_only +
                    // tensor_to_mask + apply_mask) to capture the raw tensor for
                    // future Tier 2 mask reruns.  Other modes don't benefit from
                    // tensor caching so they use the monolithic pipeline.
                    let mut tensor_for_cache: Option<(Vec<f32>, u32, u32)> = None;
                    let mut edge_tensor_for_cache: Option<prunr_core::EdgeInferenceResult> = None;

                    let result = match line_mode {
                        LineMode::EdgesOnly => {
                            let decoded;
                            let img_ref = if let Some(ref img) = chain_img {
                                Ok(img as &image::DynamicImage)
                            } else {
                                match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => { decoded = img; Ok(&decoded as &image::DynamicImage) }
                                    Err(e) => Err(e),
                                }
                            };
                            img_ref.and_then(|img| {
                                // invariant: line_mode == EdgesOnly → needs_edge → edge_eng loaded.
                                let eng_ref = edge_eng.as_ref().unwrap();
                                eng_ref.infer_all_tensors(img).map(|res| {
                                    let active = &res.tensors[edge.edge_scale as usize];
                                    let rgba_image = prunr_core::finalize_edges(
                                        active, res.height, res.width, img, &edge,
                                    );
                                    edge_tensor_for_cache = Some(res);
                                    ProcessResult {
                                        rgba_image,
                                        active_provider: provider.clone(),
                                    }
                                })
                            })
                        }
                        LineMode::SubjectOutline => {
                            if let Some(ref img) = chain_img {
                                // Chain: chain input is already a masked RGBA from a prior
                                // tier, so run DexiNed on it directly. No seg cache — we
                                // don't have the seg tensor that produced the chain input.
                                // invariant: line_mode == SubjectOutline → needs_edge → edge_eng loaded.
                                let eng_ref = edge_eng.as_ref().unwrap();
                                eng_ref.infer_all_tensors(img).map(|res| {
                                    let active = &res.tensors[edge.edge_scale as usize];
                                    let rgba_image = prunr_core::finalize_edges(
                                        active, res.height, res.width, img, &edge,
                                    );
                                    edge_tensor_for_cache = Some(res);
                                    ProcessResult {
                                        rgba_image,
                                        active_provider: provider.clone(),
                                    }
                                })
                            } else {
                                // Non-chain: split pipeline so the seg tensor is captured
                                // for Tier 2 mask reruns and cross-mode transitions (the
                                // monolithic `process_image_with_mask` threw it away).
                                let Some(eng) = engine.as_ref() else {
                                    let _ = evt_tx.send(SubprocessEvent::ImageError {
                                        item_id,
                                        error: "segmentation engine not initialized".into(),
                                    });
                                    in_flight.fetch_sub(1, Ordering::AcqRel);
                                    if image_path.starts_with(ipc.as_ref()) {
                                        let _ = std::fs::remove_file(&image_path);
                                    }
                                    return;
                                };
                                let decoded = match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => img,
                                    Err(e) => {
                                        let _ = evt_tx.send(SubprocessEvent::ImageError {
                                            item_id, error: e.to_string(),
                                        });
                                        in_flight.fetch_sub(1, Ordering::AcqRel);
                                        if image_path.starts_with(ipc.as_ref()) {
                                            let _ = std::fs::remove_file(&image_path);
                                        }
                                        return;
                                    }
                                };
                                let original = &decoded;
                                if let Some(err) = prunr_core::check_large_image(original) {
                                    Err(err)
                                } else {
                                    let infer_evt_tx = evt_tx.clone();
                                    let infer_cancel = cancel.clone();
                                    let infer_progress = move |stage: ProgressStage, pct: f32| {
                                        if !infer_cancel.load(Ordering::Acquire) {
                                            let _ = infer_evt_tx.send(SubprocessEvent::Progress {
                                                item_id, stage, pct,
                                            });
                                        }
                                    };
                                    prunr_core::infer_only(
                                        original, eng, Some(infer_progress), Some(cancel.clone()),
                                    ).and_then(|ir| {
                                        if cancel.load(Ordering::Acquire) {
                                            return Err(prunr_core::CoreError::Cancelled);
                                        }
                                        let th = ir.tensor_height;
                                        let tw = ir.tensor_width;
                                        let active_provider = ir.active_provider.clone();
                                        let masked_rgba = prunr_core::postprocess_from_flat(
                                            &ir.tensor_data, th, tw, original, &mask, model,
                                        )?;
                                        // Seg tensor captured → Tier 2 mask reruns work in
                                        // SubjectOutline; Off ↔ SubjectOutline can reuse it.
                                        tensor_for_cache = Some((ir.tensor_data, th as u32, tw as u32));
                                        if cancel.load(Ordering::Acquire) {
                                            return Err(prunr_core::CoreError::Cancelled);
                                        }
                                        let masked_img = image::DynamicImage::ImageRgba8(masked_rgba.clone());
                                        // invariant: line_mode == SubjectOutline → needs_edge → edge_eng loaded.
                                        let eng_ref = edge_eng.as_ref().unwrap();
                                        let edge_res = eng_ref.infer_all_tensors(&masked_img)?;
                                        if cancel.load(Ordering::Acquire) {
                                            return Err(prunr_core::CoreError::Cancelled);
                                        }
                                        let rgba_image = compose_subject_outline(&edge_res, &masked_rgba, &edge);
                                        edge_tensor_for_cache = Some(edge_res);
                                        Ok(ProcessResult { rgba_image, active_provider })
                                    })
                                }
                            }
                        }
                        LineMode::Off => {
                            // Split pipeline: infer_only → tensor_to_mask → apply_mask
                            // This captures the raw tensor for future Tier 2 mask reruns.
                            let Some(eng) = engine.as_ref() else {
                                let _ = evt_tx.send(SubprocessEvent::ImageError {
                                    item_id,
                                    error: "segmentation engine not initialized".into(),
                                });
                                in_flight.fetch_sub(1, Ordering::AcqRel);
                                if image_path.starts_with(ipc.as_ref()) {
                                    let _ = std::fs::remove_file(&image_path);
                                }
                                if let Some(ref p) = chain_path {
                                    if p.starts_with(ipc.as_ref()) {
                                        let _ = std::fs::remove_file(p);
                                    }
                                }
                                return;
                            };
                            let decoded;
                            let original: &image::DynamicImage = if let Some(ref img) = chain_img {
                                img
                            } else {
                                match prunr_core::load_image_from_bytes(&img_bytes) {
                                    Ok(img) => { decoded = img; &decoded }
                                    Err(e) => {
                                        // Use the error directly
                                        let _ = evt_tx.send(SubprocessEvent::ImageError {
                                            item_id,
                                            error: e.to_string(),
                                        });
                                        in_flight.fetch_sub(1, Ordering::AcqRel);
                                        // Clean up temp files
                                        if image_path.starts_with(ipc.as_ref()) {
                                            let _ = std::fs::remove_file(&image_path);
                                        }
                                        if let Some(ref p) = chain_path {
                                            if p.starts_with(ipc.as_ref()) {
                                                let _ = std::fs::remove_file(p);
                                            }
                                        }
                                        return;
                                    }
                                }
                            };

                            // Check large image limit
                            if let Some(err) = prunr_core::check_large_image(original) {
                                Err(err)
                            } else {
                                // Progress callback for infer_only
                                let infer_evt_tx = evt_tx.clone();
                                let infer_cancel = cancel.clone();
                                let infer_progress = move |stage: ProgressStage, pct: f32| {
                                    if !infer_cancel.load(Ordering::Acquire) {
                                        let _ = infer_evt_tx.send(SubprocessEvent::Progress {
                                            item_id, stage, pct,
                                        });
                                    }
                                };

                                prunr_core::infer_only(
                                    original, eng, Some(infer_progress), Some(cancel.clone()),
                                ).and_then(|ir| {
                                    // Report postprocess stage
                                    if !cancel.load(Ordering::Acquire) {
                                        let _ = evt_tx.send(SubprocessEvent::Progress {
                                            item_id, stage: ProgressStage::Postprocess, pct: 0.8,
                                        });
                                    }

                                    let th = ir.tensor_height;
                                    let tw = ir.tensor_width;
                                    tracing::debug!(
                                        item_id, th, tw, tensor_len = ir.tensor_data.len(),
                                        provider = %ir.active_provider,
                                        "worker about to postprocess_from_flat",
                                    );
                                    let rgba_image = prunr_core::postprocess_from_flat(
                                        &ir.tensor_data, th, tw, original, &mask, model,
                                    )?;
                                    let (w, h) = (rgba_image.width(), rgba_image.height());
                                    let (mut a_min, mut a_max) = (255u8, 0u8);
                                    let mut a_sum = 0u64;
                                    for p in rgba_image.pixels() {
                                        let a = p.0[3];
                                        if a < a_min { a_min = a; }
                                        if a > a_max { a_max = a; }
                                        a_sum += a as u64;
                                    }
                                    let a_mean = a_sum / (w as u64 * h as u64).max(1);
                                    tracing::debug!(
                                        item_id, w, h, a_min, a_max, a_mean,
                                        "worker postprocess result alpha stats",
                                    );

                                    // Report alpha stage
                                    if !cancel.load(Ordering::Acquire) {
                                        let _ = evt_tx.send(SubprocessEvent::Progress {
                                            item_id, stage: ProgressStage::Alpha, pct: 0.95,
                                        });
                                    }

                                    // Stash tensor for cache output
                                    tensor_for_cache = Some((
                                        ir.tensor_data,
                                        th as u32,
                                        tw as u32,
                                    ));

                                    Ok(ProcessResult {
                                        rgba_image,
                                        active_provider: ir.active_provider,
                                    })
                                })
                            }
                        }
                    };

                    // Background color is applied by the parent at display/export time
                    // (non-destructive — result_rgba always stores the transparent version)

                    match result {
                        Ok(pr) => {
                            // Write tensor cache to temp file if available
                            let (tcp, tch, tcw) = if let Some((ref tdata, th, tw)) = tensor_for_cache {
                                let tp = ipc.join(format!("tensor_{item_id}.raw"));
                                let bytes = prunr_app::subprocess::ipc::f32s_as_le_bytes(tdata);
                                tracing::debug!(item_id, th, tw, tensor_len = tdata.len(), bytes_len = bytes.len(), path = %tp.display(), "writing seg tensor");
                                match std::fs::write(&tp, bytes) {
                                    Ok(()) => (Some(tp), Some(th), Some(tw)),
                                    Err(e) => {
                                        tracing::error!(item_id, %e, path = %tp.display(), "seg tensor write failed");
                                        (None, None, None)
                                    }
                                }
                            } else {
                                (None, None, None)
                            };

                            // Write DexiNed multi-scale cache (4 tensors concatenated
                            // into a single temp file; parent splits by EDGE_SCALE_COUNT).
                            let (ecp, ech, ecw) = if let Some(ref res) = edge_tensor_for_cache {
                                let ep = ipc.join(format!("edge_{item_id}.raw"));
                                match std::fs::write(&ep, pack_edge_tensors(res)) {
                                    Ok(()) => (Some(ep), Some(res.height), Some(res.width)),
                                    Err(_) => (None, None, None),
                                }
                            } else {
                                (None, None, None)
                            };

                            // Write result RGBA to temp file
                            let (w, h) = (pr.rgba_image.width(), pr.rgba_image.height());
                            let result_path = ipc.join(format!("result_{item_id}.raw"));
                            match std::fs::write(&result_path, pr.rgba_image.as_raw()) {
                                Ok(()) => {
                                    let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                        item_id,
                                        result_path,
                                        width: w,
                                        height: h,
                                        active_provider: pr.active_provider,
                                        tensor_cache_path: tcp,
                                        tensor_cache_height: tch,
                                        tensor_cache_width: tcw,
                                        edge_cache_path: ecp,
                                        edge_cache_height: ech,
                                        edge_cache_width: ecw,
                                    });
                                }
                                Err(e) => {
                                    let _ = evt_tx.send(SubprocessEvent::ImageError {
                                        item_id,
                                        error: format!("Failed to write result: {e}"),
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError {
                                item_id,
                                error: e.to_string(),
                            });
                        }
                    }

                    // Report RSS after each image
                    if let Some(stats) = memory_stats::memory_stats() {
                        let _ = evt_tx.send(SubprocessEvent::RssUpdate {
                            rss_bytes: stats.physical_mem as u64,
                        });
                    }

                    // Clean up temp files only (not user's original files).
                    // Files in the IPC temp dir were created by the parent for this subprocess.
                    if image_path.starts_with(ipc.as_ref()) {
                        let _ = std::fs::remove_file(&image_path);
                    }
                    if let Some(ref p) = chain_path {
                        if p.starts_with(ipc.as_ref()) {
                            let _ = std::fs::remove_file(p);
                        }
                    }

                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::RePostProcess {
                item_id, tensor_path, tensor_height, tensor_width,
                model, original_image_path, mask: repost_mask,
            } => {
                let evt_tx = evt_tx.clone();
                let in_flight = in_flight.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();

                in_flight.fetch_add(1, Ordering::AcqRel);
                pool.spawn(move || {
                    // Conservative weight — postprocess upscales tensor to original resolution
                    // which allocates significant memory for Lanczos3 + guided filter
                    let _sem_guard = sem.acquire(10);

                    // Read tensor from temp file
                    let tensor_result = (|| -> Result<image::RgbaImage, String> {
                        let raw_bytes = std::fs::read(&tensor_path)
                            .map_err(|e| format!("Failed to read tensor: {e}"))?;
                        let floats = prunr_app::subprocess::ipc::le_bytes_to_f32s(&raw_bytes);
                        let img_bytes = std::fs::read(&original_image_path)
                            .map_err(|e| format!("Failed to read original: {e}"))?;
                        let original = prunr_core::load_image_from_bytes(&img_bytes)
                            .map_err(|e| format!("Failed to decode original: {e}"))?;
                        prunr_core::postprocess_from_flat(
                            &floats, tensor_height as usize, tensor_width as usize,
                            &original, &repost_mask, model,
                        ).map_err(|e| e.to_string())
                    })();

                    match tensor_result {
                        Ok(rgba) => {
                            let (w, h) = (rgba.width(), rgba.height());
                            let result_path = ipc.join(format!("result_{item_id}.raw"));
                            if std::fs::write(&result_path, rgba.as_raw()).is_ok() {
                                let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                    item_id,
                                    result_path,
                                    width: w,
                                    height: h,
                                    active_provider: String::new(),
                                    tensor_cache_path: None,
                                    tensor_cache_height: None,
                                    tensor_cache_width: None,
                                    edge_cache_path: None,
                                    edge_cache_height: None,
                                    edge_cache_width: None,
                                });
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error: e });
                        }
                    }

                    // Clean up temp files
                    if tensor_path.starts_with(ipc.as_ref()) {
                        let _ = std::fs::remove_file(&tensor_path);
                    }
                    if original_image_path.starts_with(ipc.as_ref()) {
                        let _ = std::fs::remove_file(&original_image_path);
                    }

                    if let Some(stats) = memory_stats::memory_stats() {
                        let _ = evt_tx.send(SubprocessEvent::RssUpdate {
                            rss_bytes: stats.physical_mem as u64,
                        });
                    }

                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::AddEdgeInference {
                item_id, image_path, seg_tensor_path,
                seg_tensor_height, seg_tensor_width, model: cmd_model,
                mask: cmd_mask,
            } => {
                let evt_tx = evt_tx.clone();
                let in_flight = in_flight.clone();
                let sem = semaphore.clone();
                let ipc = ipc_dir.clone();
                let mask_settings = cmd_mask;
                let edge_settings = edge.clone();
                let edge_eng = edge_engine.clone();
                let cancel = cancel.clone();

                in_flight.fetch_add(1, Ordering::AcqRel);
                pool.spawn(move || {
                    let _sem_guard = sem.acquire(10);

                    // seg_tensor_path is both our input and the output we hand
                    // back as tensor_cache_path — the parent's read_tensor_cache
                    // reads-and-deletes it. So here we `read`, not `read_and_delete`.
                    let produce = (|| -> Result<(image::RgbaImage, prunr_core::EdgeInferenceResult), String> {
                        if cancel.load(Ordering::Acquire) {
                            return Err("cancelled".to_string());
                        }
                        let raw_bytes = std::fs::read(&seg_tensor_path)
                            .map_err(|e| format!("Failed to read seg tensor: {e}"))?;
                        let seg_data = prunr_app::subprocess::ipc::le_bytes_to_f32s(&raw_bytes);

                        let img_bytes = std::fs::read(&image_path)
                            .map_err(|e| format!("Failed to read image: {e}"))?;
                        let original = prunr_core::load_image_from_bytes(&img_bytes)
                            .map_err(|e| format!("Failed to decode image: {e}"))?;

                        if cancel.load(Ordering::Acquire) {
                            return Err("cancelled".to_string());
                        }
                        let masked_rgba = prunr_core::postprocess_from_flat(
                            &seg_data, seg_tensor_height as usize, seg_tensor_width as usize,
                            &original, &mask_settings, cmd_model,
                        ).map_err(|e| e.to_string())?;

                        let masked_img = image::DynamicImage::ImageRgba8(masked_rgba.clone());
                        let eng_ref = edge_eng.as_ref()
                            .ok_or_else(|| "edge engine not initialized in this subprocess".to_string())?;
                        let edge_res = eng_ref.infer_all_tensors(&masked_img)
                            .map_err(|e| e.to_string())?;

                        if cancel.load(Ordering::Acquire) {
                            return Err("cancelled".to_string());
                        }
                        let rgba_image = compose_subject_outline(&edge_res, &masked_rgba, &edge_settings);
                        Ok((rgba_image, edge_res))
                    })();

                    match produce {
                        Ok((rgba, edge_res)) => {
                            let (w, h) = (rgba.width(), rgba.height());
                            let result_path = ipc.join(format!("result_{item_id}.raw"));

                            let (ecp, ech, ecw) = {
                                let ep = ipc.join(format!("edge_{item_id}.raw"));
                                match std::fs::write(&ep, pack_edge_tensors(&edge_res)) {
                                    Ok(()) => (Some(ep), Some(edge_res.height), Some(edge_res.width)),
                                    Err(_) => (None, None, None),
                                }
                            };

                            if std::fs::write(&result_path, rgba.as_raw()).is_ok() {
                                let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                    item_id,
                                    result_path,
                                    width: w,
                                    height: h,
                                    // Empty active_provider — seg didn't run this
                                    // tier; parent treats it like a Tier 2 rerun
                                    // and keeps the existing backend label.
                                    active_provider: String::new(),
                                    tensor_cache_path: Some(seg_tensor_path.clone()),
                                    tensor_cache_height: Some(seg_tensor_height),
                                    tensor_cache_width: Some(seg_tensor_width),
                                    edge_cache_path: ecp,
                                    edge_cache_height: ech,
                                    edge_cache_width: ecw,
                                });
                            }
                        }
                        Err(e) => {
                            let _ = evt_tx.send(SubprocessEvent::ImageError { item_id, error: e });
                            if seg_tensor_path.starts_with(ipc.as_ref()) {
                                let _ = std::fs::remove_file(&seg_tensor_path);
                            }
                        }
                    }

                    if image_path.starts_with(ipc.as_ref()) {
                        let _ = std::fs::remove_file(&image_path);
                    }
                    if let Some(stats) = memory_stats::memory_stats() {
                        let _ = evt_tx.send(SubprocessEvent::RssUpdate {
                            rss_bytes: stats.physical_mem as u64,
                        });
                    }
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                });
            }

            SubprocessCommand::Cancel => {
                cancel.store(true, Ordering::Release);
                // Wait for in-flight to drain
                while in_flight.load(Ordering::Acquire) > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                let _ = evt_tx.send(SubprocessEvent::Finished);
                cancel.store(false, Ordering::Release);
            }

            SubprocessCommand::Shutdown => {
                // Wait for in-flight to drain
                while in_flight.load(Ordering::Acquire) > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                let _ = evt_tx.send(SubprocessEvent::Finished);
                drop(evt_tx);
                let _ = writer_handle.join();
                std::process::exit(0);
            }

            SubprocessCommand::Init { .. } => {
                // Duplicate Init — ignore
            }
        }
    }

    // stdin closed — clean exit
    drop(evt_tx);
    let _ = writer_handle.join();
    std::process::exit(0);
}
