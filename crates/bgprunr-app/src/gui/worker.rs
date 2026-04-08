use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use bgprunr_core::{ModelKind, OrtEngine, ProgressStage, ProcessResult, process_image};

pub enum WorkerMessage {
    ProcessImage {
        img_bytes: Arc<Vec<u8>>,
        model: ModelKind,
        cancel: Arc<AtomicBool>,
    },
    BatchProcess {
        items: Vec<(u64, Arc<Vec<u8>>)>,
        model: ModelKind,
        jobs: usize,
        cancel: Arc<AtomicBool>,
    },
    Quit,
}

pub enum WorkerResult {
    Progress(ProgressStage, f32),
    Done(ProcessResult),
    BatchItemDone {
        item_id: u64,
        result: Result<ProcessResult, String>,
    },
    BatchComplete,
    Cancelled,
    Error(String),
}

pub fn spawn_worker(
    ctx: egui::Context,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("bgprunr-worker".into())
        .spawn(move || {
            while let Ok(msg) = msg_rx.recv() {
                match msg {
                    WorkerMessage::ProcessImage { img_bytes, model, cancel } => {
                        // Create engine per invocation (matches CLI pattern)
                        let engine = match OrtEngine::new(model, 1) {
                            Ok(e) => e,
                            Err(e) => {
                                let _ = res_tx.send(WorkerResult::Error(e.to_string()));
                                ctx.request_repaint();
                                continue;
                            }
                        };

                        let cancel_clone = cancel.clone();
                        let res_tx_clone = res_tx.clone();
                        let ctx_clone = ctx.clone();

                        let result = process_image(
                            &img_bytes,
                            &engine,
                            Some(move |stage: ProgressStage, pct: f32| {
                                if !cancel_clone.load(Ordering::Relaxed) {
                                    let _ = res_tx_clone.send(WorkerResult::Progress(stage, pct));
                                    ctx_clone.request_repaint();
                                }
                            }),
                            Some(cancel.clone()),
                        );

                        if cancel.load(Ordering::Relaxed) {
                            let _ = res_tx.send(WorkerResult::Cancelled);
                        } else {
                            match result {
                                Ok(r) => { let _ = res_tx.send(WorkerResult::Done(r)); }
                                Err(bgprunr_core::CoreError::Cancelled) => {
                                    let _ = res_tx.send(WorkerResult::Cancelled);
                                }
                                Err(e) => { let _ = res_tx.send(WorkerResult::Error(e.to_string())); }
                            }
                        }
                        ctx.request_repaint();
                    }
                    WorkerMessage::BatchProcess { items, model, jobs, cancel } => {
                        // Process each item in parallel using rayon, sending results as they complete
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(jobs)
                            .build()
                            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

                        let res_tx_batch = res_tx.clone();
                        let ctx_batch = ctx.clone();
                        let cancel_batch = cancel.clone();

                        pool.scope(|s| {
                            for (item_id, img_bytes) in items {
                                let res_tx_item = res_tx_batch.clone();
                                let ctx_item = ctx_batch.clone();
                                let cancel_item = cancel_batch.clone();

                                s.spawn(move |_| {
                                    if cancel_item.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    // Each worker creates its own OrtEngine (matches Phase 2/3/4 pattern)
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
                                    let result = process_image(
                                        &img_bytes,
                                        &engine,
                                        None::<fn(ProgressStage, f32)>,
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
                            let _ = res_tx.send(WorkerResult::BatchComplete);
                        } else {
                            let _ = res_tx.send(WorkerResult::Cancelled);
                        }
                        ctx.request_repaint();
                    }
                    WorkerMessage::Quit => break,
                }
            }
        })
        .expect("failed to spawn worker thread");

    (msg_tx, res_rx)
}
