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

                        std::thread::spawn(move || {
                            // Report "Loading model" for the first item so the UI
                            // shows feedback during CoreML/CUDA model compilation
                            if let Some((first_id, _)) = items.first() {
                                let _ = res_tx_batch.send(WorkerResult::BatchProgress {
                                    item_id: *first_id,
                                    stage: ProgressStage::LoadingModel,
                                    pct: 0.0,
                                });
                                ctx_batch.request_repaint();
                            }

                            // Create engine ONCE — CoreML/CUDA compilation happens here.
                            // Sharing via Arc avoids recompiling per worker thread.
                            let intra_threads = (num_cpus::get() / jobs).max(1);
                            let engine = match OrtEngine::new(model, intra_threads) {
                                Ok(e) => Arc::new(e),
                                Err(e) => {
                                    // Report error for all items
                                    for (item_id, _) in &items {
                                        let _ = res_tx_batch.send(WorkerResult::BatchItemDone {
                                            item_id: *item_id,
                                            result: Err(e.to_string()),
                                        });
                                    }
                                    let _ = res_tx_batch.send(WorkerResult::BatchComplete);
                                    ctx_batch.request_repaint();
                                    return;
                                }
                            };

                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(jobs)
                                .build()
                                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                            let cancel_batch = cancel.clone();

                            pool.scope(|s| {
                                for (item_id, img_bytes) in items {
                                    let res_tx_item = res_tx_batch.clone();
                                    let ctx_item = ctx_batch.clone();
                                    let cancel_item = cancel_batch.clone();
                                    let mask_item = mask;
                                    let engine_ref = engine.clone();

                                    s.spawn(move |_| {
                                        if cancel_item.load(Ordering::Relaxed) {
                                            return;
                                        }
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
                                            &engine_ref,
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
