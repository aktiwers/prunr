use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use prunr_core::{MaskSettings, ModelKind, OrtEngine, ProgressStage, ProcessResult, process_image_with_mask, create_engine_pool, EdgeEngine};
use crate::gui::settings::LineMode;

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<(u64, Arc<Vec<u8>>, Option<Arc<image::RgbaImage>>)>,
        model: ModelKind,
        jobs: usize,
        cancel: Arc<AtomicBool>,
        mask: MaskSettings,
        force_cpu: bool,
        line_mode: LineMode,
        line_strength: f32,
        solid_line_color: Option<[u8; 3]>,
        bg_color: Option<[u8; 3]>,
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
    prewarm_engine: Arc<std::sync::OnceLock<OrtEngine>>,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("prunr-worker".into())
        .spawn(move || {
            while let Ok(msg) = msg_rx.recv() {
                match msg {
                    WorkerMessage::BatchProcess { items, model, jobs, cancel, mask, force_cpu, line_mode, line_strength, solid_line_color, bg_color } => {
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

                            // Load segmentation engines if needed
                            let engines: Vec<Arc<prunr_core::OrtEngine>> = if needs_segmentation {
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
                            } else {
                                Vec::new()
                            };

                            let pool_size = if engines.is_empty() { jobs.max(1) } else { engines.len() };

                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(pool_size)
                                .build()
                                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                            let cancel_batch = cancel.clone();

                            pool.scope(|s| {
                                for (idx, (item_id, img_bytes, chain_input)) in items.into_iter().enumerate() {
                                    let res_tx_item = res_tx_batch.clone();
                                    let ctx_item = ctx_batch.clone();
                                    let cancel_item = cancel_batch.clone();
                                    let mask_item = mask;
                                    let engine = if !engines.is_empty() {
                                        Some(engines[idx % engines.len()].clone())
                                    } else {
                                        None
                                    };

                                    let edge_eng = edge_engine.clone();
                                    let line_str = line_strength;
                                    let line_col = solid_line_color;
                                    let bg_col = bg_color;

                                    s.spawn(move |_| {
                                        if cancel_item.load(Ordering::Relaxed) {
                                            return;
                                        }
                                        let progress_tx = res_tx_item.clone();
                                        let progress_ctx = ctx_item.clone();
                                        let progress_cancel = cancel_item.clone();
                                        let last_repaint = std::cell::Cell::new(std::time::Instant::now());
                                        let progress_cb = move |stage: ProgressStage, pct: f32| {
                                            if !progress_cancel.load(Ordering::Relaxed) {
                                                let _ = progress_tx.send(WorkerResult::BatchProgress {
                                                    item_id, stage, pct,
                                                });
                                                if last_repaint.get().elapsed().as_millis() >= 33 {
                                                    progress_ctx.request_repaint();
                                                    last_repaint.set(std::time::Instant::now());
                                                }
                                            }
                                        };

                                        let result = match line_mode {
                                            LineMode::LinesOnly => {
                                                // Edge detection only, skip bg removal
                                                prunr_core::load_image_from_bytes(&img_bytes)
                                                    .and_then(|img| {
                                                        edge_eng.as_ref().unwrap().detect(&img, line_str, line_col)
                                                            .map(|rgba_image| ProcessResult {
                                                                rgba_image,
                                                                active_provider: OrtEngine::detect_active_provider(),
                                                            })
                                                    })
                                            }
                                            LineMode::AfterBgRemoval => {
                                                // BG removal first, then edge detection on result
                                                let eng = engine.as_ref().expect("segmentation engine required");
                                                process_image_with_mask(
                                                    &img_bytes, eng, &mask_item,
                                                    Some(progress_cb), Some(cancel_item),
                                                ).and_then(|pr| {
                                                    let img = image::DynamicImage::ImageRgba8(pr.rgba_image);
                                                    edge_eng.as_ref().unwrap().detect(&img, line_str, line_col)
                                                        .map(|rgba_image| ProcessResult {
                                                            rgba_image,
                                                            active_provider: pr.active_provider,
                                                        })
                                                })
                                            }
                                            LineMode::Off => {
                                                // Normal background removal
                                                let eng = engine.as_ref().expect("segmentation engine required");
                                                process_image_with_mask(
                                                    &img_bytes, eng, &mask_item,
                                                    Some(progress_cb), Some(cancel_item),
                                                )
                                            }
                                        };

                                        // Apply background color if enabled
                                        let result = result.map(|mut pr| {
                                            if let Some(bg) = bg_col {
                                                prunr_core::apply_background_color(&mut pr.rgba_image, bg);
                                            }
                                            pr
                                        });

                                        let _ = res_tx_item.send(WorkerResult::BatchItemDone {
                                            item_id,
                                            result: result.map_err(|e| e.to_string()),
                                        });
                                        ctx_item.request_repaint();
                                    });
                                }
                            });

                            if !cancel.load(Ordering::Relaxed) {
                                let _ = res_tx_batch.send(WorkerResult::BatchComplete);
                            } else {
                                let _ = res_tx_batch.send(WorkerResult::Cancelled);
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
