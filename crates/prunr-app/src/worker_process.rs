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
    create_engine_pool, process_image_with_mask, process_image_from_decoded,
    OrtEngine, ProcessResult, ProgressStage, EdgeEngine,
};

use prunr_app::subprocess::protocol::*;
use prunr_app::subprocess::ipc::{read_message, write_message};
use prunr_app::gui::settings::LineMode;

/// Global lock for postprocessing (Lanczos3 resize).
/// AI inference can run in parallel, but high-res CPU resizing must be
/// serialized to prevent concurrent memory spikes from crashing the process.
static POSTPROCESS_LOCK: Mutex<()> = Mutex::new(());

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
    let init = match read_message::<_, SubprocessCommand>(&mut reader) {
        Ok(Some(SubprocessCommand::Init { model, jobs, mask, force_cpu, line_mode, line_strength, solid_line_color, bg_color })) => {
            (model, jobs, mask, force_cpu, line_mode, line_strength, solid_line_color, bg_color)
        }
        _ => {
            let _ = evt_tx.send(SubprocessEvent::InitError {
                error: "Expected Init command".to_string(),
            });
            drop(evt_tx);
            let _ = writer_handle.join();
            std::process::exit(1);
        }
    };

    let (model, jobs, mask, force_cpu, line_mode, line_strength, solid_line_color, bg_color) = init;

    // Detect backend
    let has_gpu = !OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
    let cpu_only = force_cpu || !has_gpu;

    // Load edge engine if needed
    let needs_edge = line_mode != LineMode::Off;
    let needs_segmentation = line_mode != LineMode::LinesOnly;

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
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    let cancel = Arc::new(AtomicBool::new(false));
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine_idx: usize = 0;

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

                    // Progress callback
                    let progress_evt_tx = evt_tx.clone();
                    let progress_cancel = cancel.clone();
                    let progress_cb = move |stage: ProgressStage, pct: f32| {
                        if !progress_cancel.load(Ordering::Acquire) {
                            let _ = progress_evt_tx.send(SubprocessEvent::Progress {
                                item_id, stage, pct,
                            });
                        }
                    };

                    // Serialize processing to prevent concurrent postprocess
                    // (Lanczos3 resize) spikes from causing OOM. AI inference
                    // still uses all CPU threads via ORT intra-op parallelism.
                    let _lock = POSTPROCESS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                    let result = match line_mode {
                        LineMode::LinesOnly => {
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
                                edge_eng.as_ref().unwrap().detect(img, line_strength, solid_line_color)
                                    .map(|rgba_image| ProcessResult {
                                        rgba_image,
                                        active_provider: provider.clone(),
                                    })
                            })
                        }
                        LineMode::AfterBgRemoval => {
                            if let Some(ref img) = chain_img {
                                edge_eng.as_ref().unwrap().detect(img, line_strength, solid_line_color)
                                    .map(|rgba_image| ProcessResult {
                                        rgba_image,
                                        active_provider: provider.clone(),
                                    })
                            } else {
                                let eng = engine.as_ref().expect("segmentation engine required");
                                process_image_with_mask(
                                    &img_bytes, eng, &mask,
                                    Some(progress_cb), Some(cancel.clone()),
                                ).and_then(|pr| {
                                    let img = image::DynamicImage::ImageRgba8(pr.rgba_image);
                                    edge_eng.as_ref().unwrap().detect(&img, line_strength, solid_line_color)
                                        .map(|rgba_image| ProcessResult {
                                            rgba_image,
                                            active_provider: pr.active_provider,
                                        })
                                })
                            }
                        }
                        LineMode::Off => {
                            let eng = engine.as_ref().expect("segmentation engine required");
                            if let Some(ref img) = chain_img {
                                process_image_from_decoded(
                                    img, eng, &mask,
                                    Some(progress_cb), Some(cancel.clone()),
                                )
                            } else {
                                process_image_with_mask(
                                    &img_bytes, eng, &mask,
                                    Some(progress_cb), Some(cancel.clone()),
                                )
                            }
                        }
                    };

                    // Apply background color if enabled
                    let result = result.map(|mut pr| {
                        if let Some(bg) = bg_color {
                            prunr_core::apply_background_color(&mut pr.rgba_image, bg);
                        }
                        pr
                    });

                    match result {
                        Ok(pr) => {
                            // Write result RGBA to temp file
                            let (w, h) = (pr.rgba_image.width(), pr.rgba_image.height());
                            let result_path = ipc_temp_dir().join(format!("result_{item_id}.raw"));
                            match std::fs::write(&result_path, pr.rgba_image.as_raw()) {
                                Ok(()) => {
                                    let _ = evt_tx.send(SubprocessEvent::ImageDone {
                                        item_id,
                                        result_path,
                                        width: w,
                                        height: h,
                                        active_provider: pr.active_provider,
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

                    // Clean up input temp files
                    let _ = std::fs::remove_file(&image_path);
                    if let Some(ref p) = chain_path {
                        let _ = std::fs::remove_file(p);
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
