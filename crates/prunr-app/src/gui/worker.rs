use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use prunr_core::{MaskSettings, ModelKind, OrtEngine, ProgressStage, ProcessResult, process_image_with_mask};

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<(u64, Arc<Vec<u8>>)>,
        model: ModelKind,
        jobs: usize,
        cancel: Arc<AtomicBool>,
        mask: MaskSettings,
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
                    WorkerMessage::BatchProcess { items, model, jobs, cancel, mask } => {
                        let res_tx_batch = res_tx.clone();
                        let ctx_batch = ctx.clone();

                        let prewarm = prewarm_engine.clone();

                        std::thread::spawn(move || {
                            // Report "Loading model" for the first item
                            if let Some((first_id, _)) = items.first() {
                                let _ = res_tx_batch.send(WorkerResult::BatchProgress {
                                    item_id: *first_id,
                                    stage: ProgressStage::LoadingModel,
                                    pct: 0.0,
                                });
                                ctx_batch.request_repaint();
                            }

                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(jobs)
                                .build()
                                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                            let cancel_batch = cancel.clone();
                            // Take the pre-warmed engine if model matches (consumed once)
                            let prewarmed = prewarm.get().and_then(|e| {
                                if e.model_kind() == model { Some(()) } else { None }
                            });
                            let _ = prewarmed; // just to warm the cache; each worker needs its own engine

                            pool.scope(|s| {
                                for (item_id, img_bytes) in items {
                                    let res_tx_item = res_tx_batch.clone();
                                    let ctx_item = ctx_batch.clone();
                                    let cancel_item = cancel_batch.clone();
                                    let mask_item = mask;

                                    s.spawn(move |_| {
                                        if cancel_item.load(Ordering::Relaxed) {
                                            return;
                                        }
                                        // Each worker creates its own engine for true parallel
                                        // inference. After pre-warm, CoreML cache makes this fast.
                                        let intra_threads = (num_cpus::get() / jobs).max(1);
                                        let engine = match OrtEngine::new(model, intra_threads) {
                                            Ok(e) => e,
                                            Err(e) => {
                                                let _ = res_tx_item.send(WorkerResult::BatchItemDone {
                                                    item_id,
                                                    result: Err(e.to_string()),
                                                });
                                                ctx_item.request_repaint();
                                                return;
                                            }
                                        };
                                        let progress_tx = res_tx_item.clone();
                                        let progress_ctx = ctx_item.clone();
                                        let progress_cancel = cancel_item.clone();
                                        let progress_cb = move |stage: ProgressStage, pct: f32| {
                                            if !progress_cancel.load(Ordering::Relaxed) {
                                                let _ = progress_tx.send(WorkerResult::BatchProgress {
                                                    item_id, stage, pct,
                                                });
                                                progress_ctx.request_repaint();
                                            }
                                        };
                                        let result = process_image_with_mask(
                                            &img_bytes,
                                            &engine,
                                            &mask_item,
                                            Some(progress_cb),
                                            Some(cancel_item),
                                        );
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
