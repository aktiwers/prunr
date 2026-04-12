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

                            // GPU backends allocate VRAM per session — cap pool to
                            // avoid exhausting GPU memory with too many sessions.
                            let is_gpu = !OrtEngine::detect_active_provider().eq_ignore_ascii_case("CPU");
                            let pool_size = if is_gpu { jobs.min(2) } else { jobs };
                            let intra_threads = (num_cpus::get() / pool_size).max(1);
                            let mut engines: Vec<OrtEngine> = Vec::with_capacity(pool_size);

                            // The pre-warm thread (started at app launch) populates
                            // the CoreML/CUDA disk cache. We don't reuse its session
                            // directly — OnceLock can't move out — but all subsequent
                            // OrtEngine::new() calls are fast thanks to the warm cache.
                            let _ = prewarm.get(); // ensure pre-warm finished

                            while engines.len() < pool_size {
                                match OrtEngine::new(model, intra_threads) {
                                    Ok(e) => engines.push(e),
                                    Err(e) => {
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
                                }
                            }

                            // Wrap in Arc for sharing with rayon — each worker picks one by index
                            let engines: Vec<Arc<OrtEngine>> = engines.into_iter().map(Arc::new).collect();

                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(jobs)
                                .build()
                                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                            let cancel_batch = cancel.clone();

                            pool.scope(|s| {
                                for (idx, (item_id, img_bytes)) in items.into_iter().enumerate() {
                                    let res_tx_item = res_tx_batch.clone();
                                    let ctx_item = ctx_batch.clone();
                                    let cancel_item = cancel_batch.clone();
                                    let mask_item = mask;
                                    let engine = engines[idx % engines.len()].clone();

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
