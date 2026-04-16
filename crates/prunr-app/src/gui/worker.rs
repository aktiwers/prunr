use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use prunr_core::{MaskSettings, ModelKind, OrtEngine, ProgressStage, ProcessResult, process_image_with_mask, process_image_from_decoded, create_engine_pool, EdgeEngine};
use crate::gui::settings::LineMode;

/// A single image item to be processed.
pub type WorkItem = (u64, Arc<Vec<u8>>, Option<Arc<image::RgbaImage>>);

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<WorkItem>,
        model: ModelKind,
        jobs: usize,
        cancel: Arc<AtomicBool>,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        line_strength: f32,
        solid_line_color: Option<[u8; 3]>,
        bg_color: Option<[u8; 3]>,
        /// Channel for additional items admitted by the memory controller.
        /// The sender is dropped by the UI thread when all items are admitted.
        additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    },
}

pub enum WorkerResult {
    /// Per-stage progress for a specific batch item
    BatchProgress {
        item_id: u64,
        stage: ProgressStage,
        pct: f32,
    },
    BatchItemDone {
        item_id: u64,
        result: Result<ProcessResult, String>,
    },
    BatchComplete,
    Cancelled,
}

pub fn spawn_worker(
    ctx: egui::Context,
    prewarm_engine: Arc<std::sync::OnceLock<Arc<OrtEngine>>>,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("prunr-worker".into())
        .spawn(move || {
            while let Ok(msg) = msg_rx.recv() {
                match msg {
                    WorkerMessage::BatchProcess { items, model, jobs, cancel, mask, force_cpu, line_mode, line_strength, solid_line_color, bg_color, additional_items_rx } => {
                        let res_tx_batch = res_tx.clone();
                        let ctx_batch = ctx.clone();

                        let prewarm = prewarm_engine.clone();

                        std::thread::spawn(move || {
                            let has_gpu = !OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
                            let gpu_warming = !force_cpu && has_gpu && prewarm.get().is_none();
                            let cpu_only = force_cpu || prewarm.get().is_none();

                            // Report loading status — let user know if falling back to CPU
                            if let Some((first_id, _, _)) = items.first() {
                                let stage = if gpu_warming {
                                    ProgressStage::LoadingModelCpuFallback
                                } else {
                                    ProgressStage::LoadingModel
                                };
                                let _ = res_tx_batch.send(WorkerResult::BatchProgress {
                                    item_id: *first_id,
                                    stage,
                                    pct: 0.0,
                                });
                                ctx_batch.request_repaint();
                            }

                            let needs_edge = line_mode != LineMode::Off;
                            let needs_segmentation = line_mode != LineMode::LinesOnly;

                            // Load edge engine if needed
                            let edge_engine: Option<Arc<EdgeEngine>> = if needs_edge {
                                match EdgeEngine::new() {
                                    Ok(e) => Some(Arc::new(e)),
                                    Err(e) => {
                                        for (item_id, _, _) in &items {
                                            let _ = res_tx_batch.send(WorkerResult::BatchItemDone {
                                                item_id: *item_id,
                                                result: Err(e.to_string()),
                                            });
                                        }
                                        let _ = res_tx_batch.send(WorkerResult::BatchComplete);
                                        ctx_batch.request_repaint();
                                        return;
                                    }
                                }
                            } else {
                                None
                            };

                            // Load segmentation engines if needed.
                            // Reuse the prewarm engine if it matches the requested model
                            // (avoids loading a duplicate 250+ MB ORT session).
                            let engines: Vec<Arc<prunr_core::OrtEngine>> = if needs_segmentation {
                                let reused = prewarm.get()
                                    .filter(|e| e.model_kind() == model)
                                    .cloned();

                                match reused {
                                    Some(engine) => {
                                        // Prewarm was created with intra_threads=1 (for quick
                                        // startup check). For batch processing we need full
                                        // thread parallelism, so always create a proper pool.
                                        // The prewarm is only reused as fallback if pool
                                        // creation fails.
                                        create_engine_pool(model, jobs, cpu_only)
                                            .unwrap_or_else(|_| vec![engine])
                                    }
                                    None => {
                                        match create_engine_pool(model, jobs, cpu_only) {
                                            Ok(e) => e,
                                            Err(e) => {
                                                for (item_id, _, _) in &items {
                                                    let _ = res_tx_batch.send(WorkerResult::BatchItemDone {
                                                        item_id: *item_id,
                                                        result: Err(e.to_string()),
                                                    });
                                                }
                                                let _ = res_tx_batch.send(WorkerResult::BatchComplete);
                                                ctx_batch.request_repaint();
                                                return;
                                            }
                                        }
                                    }
                                }
                            } else {
                                Vec::new()
                            };

                            drop(prewarm);

                            let pool_size = if engines.is_empty() { jobs.max(1) } else { engines.len() };

                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(pool_size)
                                .build()
                                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                            run_streaming(
                                pool, pool_size, items, additional_items_rx,
                                &engines, &edge_engine, &cancel, mask, line_mode,
                                line_strength, solid_line_color, bg_color,
                                &res_tx_batch, &ctx_batch,
                            );

                            let final_msg = if !cancel.load(Ordering::Acquire) {
                                WorkerResult::BatchComplete
                            } else {
                                WorkerResult::Cancelled
                            };
                            if res_tx_batch.send(final_msg).is_err() {
                                eprintln!("worker: UI channel closed, batch result lost");
                            }
                            ctx_batch.request_repaint();
                        });
                    }
                }
            }
        })
        .expect("failed to spawn worker thread");

    (msg_tx, res_rx)
}

/// Process items with dynamic admission: initial items are spawned immediately,
/// and additional items arrive via `additional_rx` as the admission controller
/// releases budget. When `additional_rx` is None, all items are in `initial_items`.
fn run_streaming(
    pool: rayon::ThreadPool,
    pool_size: usize,
    initial_items: Vec<WorkItem>,
    additional_rx: Option<mpsc::Receiver<WorkItem>>,
    engines: &[Arc<OrtEngine>],
    edge_engine: &Option<Arc<EdgeEngine>>,
    cancel: &Arc<AtomicBool>,
    mask: MaskSettings,
    line_mode: LineMode,
    line_strength: f32,
    solid_line_color: Option<[u8; 3]>,
    bg_color: Option<[u8; 3]>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let (done_tx, done_rx) = mpsc::channel::<u64>();
    let shared_repaint = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
    let mut next_idx: usize = 0;

    // Closure to spawn a single item into the rayon pool.
    let spawn_one = |item_id: u64, img_bytes: Arc<Vec<u8>>, chain_input: Option<Arc<image::RgbaImage>>,
                          next_idx: &mut usize| {
        let engine = if !engines.is_empty() {
            Some(engines[*next_idx % engines.len()].clone())
        } else {
            None
        };
        *next_idx += 1;
        in_flight.fetch_add(1, Ordering::AcqRel);

        let done_tx = done_tx.clone();
        let edge_eng = edge_engine.clone();
        let cancel_item = cancel.clone();
        let res_tx_item = res_tx.clone();
        let ctx_item = ctx.clone();
        let repaint = shared_repaint.clone();

        pool.spawn(move || {
            if cancel_item.load(Ordering::Acquire) {
                let _ = done_tx.send(item_id);
                return;
            }
            let result = process_single_item(
                item_id, &img_bytes, chain_input, engine.as_ref(), edge_eng.as_ref(),
                &cancel_item, mask, line_mode, line_strength, solid_line_color,
                bg_color, &res_tx_item, &ctx_item, &repaint,
            );
            let _ = res_tx_item.send(WorkerResult::BatchItemDone {
                item_id,
                result: result.map_err(|e| e.to_string()),
            });
            ctx_item.request_repaint();
            let _ = done_tx.send(item_id);
        });
    };

    // Spawn initial items
    for (item_id, img_bytes, chain_input) in initial_items {
        spawn_one(item_id, img_bytes, chain_input, &mut next_idx);
    }

    // Dynamic admission loop.
    // Termination: in_flight == 0 && no more items coming.
    let mut additional_closed = additional_rx.is_none();
    loop {
        // Wait for a completion or timeout
        match done_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(_completed_id) => {
                in_flight.fetch_sub(1, Ordering::AcqRel);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if cancel.load(Ordering::Acquire) { break; }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Pull newly admitted items from the streaming channel
        if let Some(ref rx) = additional_rx {
            if !additional_closed {
                while in_flight.load(Ordering::Acquire) < pool_size {
                    match rx.try_recv() {
                        Ok((id, bytes, chain)) => spawn_one(id, bytes, chain, &mut next_idx),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            additional_closed = true;
                            break;
                        }
                    }
                }
            }
        }

        // All done?
        if in_flight.load(Ordering::Acquire) == 0 && additional_closed {
            break;
        }
    }
}

/// Process a single image item. Returns the result.
fn process_single_item(
    item_id: u64,
    img_bytes: &[u8],
    chain_input: Option<Arc<image::RgbaImage>>,
    engine: Option<&Arc<OrtEngine>>,
    edge_engine: Option<&Arc<EdgeEngine>>,
    cancel: &Arc<AtomicBool>,
    mask: MaskSettings,
    line_mode: LineMode,
    line_strength: f32,
    solid_line_color: Option<[u8; 3]>,
    bg_color: Option<[u8; 3]>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    shared_repaint: &Arc<std::sync::Mutex<std::time::Instant>>,
) -> Result<ProcessResult, prunr_core::CoreError> {
    let progress_tx = res_tx.clone();
    let progress_ctx = ctx.clone();
    let progress_cancel = cancel.clone();
    let repaint_lock = shared_repaint.clone();
    let progress_cb = move |stage: ProgressStage, pct: f32| {
        if !progress_cancel.load(Ordering::Acquire) {
            let _ = progress_tx.send(WorkerResult::BatchProgress {
                item_id, stage, pct,
            });
            if let Ok(mut last) = repaint_lock.try_lock() {
                if last.elapsed().as_millis() >= 33 {
                    progress_ctx.request_repaint();
                    *last = std::time::Instant::now();
                }
            }
        }
    };

    let chain_img: Option<image::DynamicImage> = chain_input.map(|rgba| {
        image::DynamicImage::ImageRgba8(Arc::unwrap_or_clone(rgba))
    });

    let result = match line_mode {
        LineMode::LinesOnly => {
            let decoded;
            let img_ref = if let Some(ref img) = chain_img {
                Ok(img as &image::DynamicImage)
            } else {
                match prunr_core::load_image_from_bytes(img_bytes) {
                    Ok(img) => { decoded = img; Ok(&decoded as &image::DynamicImage) }
                    Err(e) => Err(e),
                }
            };
            img_ref.and_then(|img| {
                edge_engine.unwrap().detect(img, line_strength, solid_line_color)
                    .map(|rgba_image| ProcessResult {
                        rgba_image,
                        active_provider: OrtEngine::detect_active_provider(),
                    })
            })
        }
        LineMode::AfterBgRemoval => {
            if let Some(ref img) = chain_img {
                edge_engine.unwrap().detect(img, line_strength, solid_line_color)
                    .map(|rgba_image| ProcessResult {
                        rgba_image,
                        active_provider: OrtEngine::detect_active_provider(),
                    })
            } else {
                let eng = engine.expect("segmentation engine required");
                process_image_with_mask(
                    img_bytes, eng, &mask,
                    Some(progress_cb), Some(cancel.clone()),
                ).and_then(|pr| {
                    let img = image::DynamicImage::ImageRgba8(pr.rgba_image);
                    edge_engine.unwrap().detect(&img, line_strength, solid_line_color)
                        .map(|rgba_image| ProcessResult {
                            rgba_image,
                            active_provider: pr.active_provider,
                        })
                })
            }
        }
        LineMode::Off => {
            let eng = engine.expect("segmentation engine required");
            if let Some(ref img) = chain_img {
                process_image_from_decoded(
                    img, eng, &mask,
                    Some(progress_cb), Some(cancel.clone()),
                )
            } else {
                process_image_with_mask(
                    img_bytes, eng, &mask,
                    Some(progress_cb), Some(cancel.clone()),
                )
            }
        }
    };

    // Apply background color if enabled
    result.map(|mut pr| {
        if let Some(bg) = bg_color {
            prunr_core::apply_background_color(&mut pr.rgba_image, bg);
        }
        pr
    })
}

