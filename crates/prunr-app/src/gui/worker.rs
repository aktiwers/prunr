use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use prunr_core::{MaskSettings, ModelKind, ProgressStage, ProcessResult};
use crate::gui::settings::LineMode;
use crate::subprocess::protocol::SubprocessEvent;
use crate::subprocess::manager::SubprocessManager;

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
        /// Channel for additional items admitted by the memory controller.
        additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    },
}

pub enum WorkerResult {
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
    /// Subprocess crashed — retrying with reduced concurrency.
    SubprocessRetry {
        reduced_jobs: usize,
        re_queued_count: usize,
    },
}

/// Spawn the worker bridge thread. Receives `WorkerMessage` from the UI,
/// translates to subprocess IPC, handles crash+retry, sends `WorkerResult` back.
pub fn spawn_worker(
    ctx: egui::Context,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("prunr-bridge".into())
        .spawn(move || {
            while let Ok(msg) = msg_rx.recv() {
                match msg {
                    WorkerMessage::BatchProcess {
                        items, model, jobs, cancel, mask, force_cpu,
                        line_mode, line_strength, solid_line_color,
                        additional_items_rx,
                    } => {
                        run_batch_with_retry(
                            items, model, jobs, &cancel, mask, force_cpu,
                            line_mode, line_strength, solid_line_color,
                            additional_items_rx,
                            &res_tx, &ctx,
                        );
                    }
                }
            }
        })
        .expect("failed to spawn bridge thread");

    (msg_tx, res_rx)
}

/// Run a batch with automatic retry on subprocess crash.
/// Reduces concurrency (jobs) on each crash: jobs → jobs/2 → 1.
/// If even 1 job crashes, marks remaining items as "insufficient memory".
fn run_batch_with_retry(
    initial_items: Vec<WorkItem>,
    model: ModelKind,
    initial_jobs: usize,
    cancel: &Arc<AtomicBool>,
    mask: MaskSettings,
    force_cpu: bool,
    line_mode: LineMode,
    line_strength: f32,
    solid_line_color: Option<[u8; 3]>,
    additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let mut pending: VecDeque<WorkItem> = initial_items.into();
    let mut completed: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut max_jobs = initial_jobs;

    // Report loading status
    if let Some((first_id, _, _)) = pending.front() {
        let _ = res_tx.send(WorkerResult::BatchProgress {
            item_id: *first_id,
            stage: ProgressStage::LoadingModel,
            pct: 0.0,
        });
        ctx.request_repaint();
    }

    loop {
        if cancel.load(Ordering::Acquire) {
            let _ = res_tx.send(WorkerResult::Cancelled);
            ctx.request_repaint();
            return;
        }

        if pending.is_empty() {
            // Check if more items coming from admission controller
            if let Some(ref rx) = additional_items_rx {
                match rx.try_recv() {
                    Ok(item) => pending.push_back(item),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // All items admitted and processed
                        let _ = res_tx.send(WorkerResult::BatchComplete);
                        ctx.request_repaint();
                        return;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // Wait a bit for more items
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    }
                }
            } else {
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }
        }

        // Spawn subprocess — cap engines at the number of pending items
        // (no point creating 4 engines for 2 images)
        let effective_jobs = max_jobs.min(pending.len());
        let (mut sub, _active_provider) = match SubprocessManager::spawn(
            model, effective_jobs, mask, force_cpu, line_mode,
            line_strength, solid_line_color,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Can't even spawn — report error for all pending
                for (id, _, _) in &pending {
                    let _ = res_tx.send(WorkerResult::BatchItemDone {
                        item_id: *id,
                        result: Err(e.clone()),
                    });
                }
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }
        };

        // Track items sent to this subprocess (for re-queue on crash)
        let mut sent_items: Vec<WorkItem> = Vec::new();

        // Send initial burst of items (up to max_jobs)
        let burst = max_jobs.min(pending.len());
        for _ in 0..burst {
            if let Some(item) = pending.pop_front() {
                if send_item_to_sub(&mut sub, &item).is_err() {
                    pending.push_front(item);
                    break;
                }
                sent_items.push(item);
            }
        }

        // Event loop: process results, admit more, handle crash
        let mut subprocess_finished = false;
        let mut subprocess_crashed = false;

        while !subprocess_finished {
            if cancel.load(Ordering::Acquire) {
                let _ = sub.send_cancel();
                // Wait for child to acknowledge
                std::thread::sleep(std::time::Duration::from_millis(200));
                sub.poll_events(); // drain
                let _ = res_tx.send(WorkerResult::Cancelled);
                ctx.request_repaint();
                return;
            }

            // Check if child is still alive
            if !sub.is_alive() {
                subprocess_crashed = true;
                break;
            }

            // Poll events from subprocess
            let events = sub.poll_events();
            for event in events {
                match event {
                    SubprocessEvent::Progress { item_id, stage, pct } => {
                        let _ = res_tx.send(WorkerResult::BatchProgress { item_id, stage, pct });
                        // Throttled repaint
                        ctx.request_repaint();
                    }
                    SubprocessEvent::ImageDone { item_id, result_path, width, height, active_provider, tensor_cache_path, .. } => {
                        // Read result from temp file and clean up
                        let result = std::fs::read(&result_path)
                            .ok()
                            .and_then(|data| image::RgbaImage::from_raw(width, height, data))
                            .map(|rgba_image| ProcessResult {
                                rgba_image,
                                active_provider: active_provider.clone(),
                            })
                            .ok_or_else(|| "Failed to read result from subprocess".to_string());
                        let _ = std::fs::remove_file(&result_path);
                        // Clean up tensor cache file if present (Phase E will store it instead)
                        if let Some(ref p) = tensor_cache_path {
                            let _ = std::fs::remove_file(p);
                        }

                        completed.insert(item_id);
                        sent_items.retain(|(id, _, _)| *id != item_id);

                        let _ = res_tx.send(WorkerResult::BatchItemDone { item_id, result });
                        ctx.request_repaint();

                        // Admit next item if RSS allows
                        if !sub.should_pause_admission() {
                            // Try from pending queue first
                            if let Some(item) = pending.pop_front() {
                                if send_item_to_sub(&mut sub, &item).is_ok() {
                                    sent_items.push(item);
                                } else {
                                    pending.push_front(item);
                                }
                            } else if let Some(ref rx) = additional_items_rx {
                                // Try streaming admission channel
                                if let Ok(item) = rx.try_recv() {
                                    if send_item_to_sub(&mut sub, &item).is_ok() {
                                        sent_items.push(item);
                                    } else {
                                        pending.push_back(item);
                                    }
                                }
                            }
                        }
                    }
                    SubprocessEvent::ImageError { item_id, error } => {
                        completed.insert(item_id);
                        sent_items.retain(|(id, _, _)| *id != item_id);
                        let _ = res_tx.send(WorkerResult::BatchItemDone {
                            item_id,
                            result: Err(error),
                        });
                        ctx.request_repaint();
                    }
                    SubprocessEvent::Finished => {
                        subprocess_finished = true;
                    }
                    SubprocessEvent::RssUpdate { .. } => {
                        // Handled internally by SubprocessManager
                    }
                    _ => {}
                }
            }

            // Check if all items done and no more coming
            if sent_items.is_empty() && pending.is_empty() {
                if let Some(ref rx) = additional_items_rx {
                    match rx.try_recv() {
                        Ok(item) => pending.push_back(item),
                        Err(mpsc::TryRecvError::Disconnected) => {
                            subprocess_finished = true;
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                    }
                } else {
                    subprocess_finished = true;
                }
            }

            if !subprocess_finished {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }

        if subprocess_crashed {
            let crash_reason = sub.crash_reason();

            // Re-queue ALL in-flight items (not just the one that crashed)
            let re_queued: Vec<WorkItem> = sent_items.into_iter()
                .filter(|(id, _, _)| !completed.contains(id))
                .collect();
            let re_count = re_queued.len();

            // Also drain any items from additional_items_rx into pending
            if let Some(ref rx) = additional_items_rx {
                while let Ok(item) = rx.try_recv() {
                    pending.push_back(item);
                }
            }

            // Put re-queued items back at the front
            for item in re_queued.into_iter().rev() {
                pending.push_front(item);
            }

            // Reduce concurrency
            let old_jobs = max_jobs;
            max_jobs = (max_jobs / 2).max(1);

            if old_jobs == 1 {
                // Already at minimum — these items genuinely can't be processed
                let err_msg = format!("{crash_reason} \u{2014} try a smaller model");
                for (id, _, _) in &pending {
                    if !completed.contains(id) {
                        let _ = res_tx.send(WorkerResult::BatchItemDone {
                            item_id: *id,
                            result: Err(err_msg.clone()),
                        });
                    }
                }
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }

            let _ = res_tx.send(WorkerResult::SubprocessRetry {
                reduced_jobs: max_jobs,
                re_queued_count: re_count,
            });
            ctx.request_repaint();

            // Clean up dead subprocess
            sub.kill();
            crate::subprocess::protocol::cleanup_ipc_temp();

            // Loop back to spawn a new subprocess with reduced concurrency
            continue;
        }

        // Subprocess finished normally — check if batch is done
        if pending.is_empty() {
            if let Some(ref rx) = additional_items_rx {
                match rx.try_recv() {
                    Err(mpsc::TryRecvError::Disconnected) => {
                        let _ = res_tx.send(WorkerResult::BatchComplete);
                        ctx.request_repaint();
                        return;
                    }
                    Ok(item) => pending.push_back(item),
                    Err(mpsc::TryRecvError::Empty) => {
                        // More items might be coming — loop back
                    }
                }
            } else {
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }
        }

        // Graceful shutdown of this subprocess before looping for more work
        let _ = sub.send_shutdown();
    }
}

/// Send a WorkItem to the subprocess via temp file IPC.
fn send_item_to_sub(sub: &mut SubprocessManager, item: &WorkItem) -> Result<(), String> {
    let (item_id, bytes, chain) = item;
    let chain_input = chain.as_ref().map(|rgba| {
        (rgba.as_ref(), rgba.width(), rgba.height())
    });
    sub.send_image(*item_id, bytes, chain_input)
}
