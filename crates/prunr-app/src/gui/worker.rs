use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use prunr_core::{MaskSettings, ModelKind, ProgressStage, ProcessResult};
use crate::gui::settings::LineMode;
use crate::subprocess::protocol::SubprocessEvent;
use crate::subprocess::manager::SubprocessManager;

/// A single image item to be processed (Tier 1: full pipeline).
pub type WorkItem = (u64, Arc<Vec<u8>>, Option<Arc<image::RgbaImage>>);

/// Raw tensor data from subprocess → parent (IPC transfer format).
pub struct TensorCache {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

/// Zstd-compressed tensor stored in BatchItem. Trades ~1ms decompress for ~3-4x RAM savings.
pub struct CompressedTensor {
    compressed: Vec<u8>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

impl CompressedTensor {
    /// Compress raw tensor data with zstd (level 1 for speed).
    /// Returns None if compression fails (caller skips caching).
    pub fn from_raw(tc: TensorCache) -> Option<Self> {
        let raw_bytes = crate::subprocess::ipc::f32s_to_le_bytes(&tc.data);
        let compressed = zstd::encode_all(raw_bytes.as_slice(), 1).ok()?;
        Some(Self { compressed, height: tc.height, width: tc.width, model: tc.model })
    }

    /// Decompress to raw f32 tensor data for Tier 2 dispatch.
    pub fn decompress(&self) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(self.compressed.as_slice()).ok()?;
        Some(crate::subprocess::ipc::le_bytes_to_f32s(&bytes))
    }

    /// Compressed size in bytes (for budget tracking).
    pub fn compressed_size(&self) -> usize {
        self.compressed.len()
    }
}

/// A Tier 2 re-postprocess item: cached tensor + original image bytes.
pub struct Tier2WorkItem {
    pub item_id: u64,
    pub tensor_data: Vec<f32>,
    pub tensor_height: u32,
    pub tensor_width: u32,
    pub model: ModelKind,
    pub original_bytes: Arc<Vec<u8>>,
    pub mask: MaskSettings,
}

/// Bundled processing settings — avoids passing 6+ individual fields.
pub struct ProcessingConfig {
    pub model: ModelKind,
    pub jobs: usize,
    pub mask: MaskSettings,
    pub force_cpu: bool,
    pub line_mode: LineMode,
    pub line_strength: f32,
    pub solid_line_color: Option<[u8; 3]>,
}

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<WorkItem>,
        /// Tier 2 items: re-postprocess from cached tensor (skip inference).
        tier2_items: Vec<Tier2WorkItem>,
        config: ProcessingConfig,
        cancel: Arc<AtomicBool>,
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
        /// Cached segmentation tensor from Tier 1 (for future mask-tier reruns).
        tensor_cache: Option<TensorCache>,
        /// Cached DexiNed output (for future edge-tier reruns on line_strength tweaks).
        /// Only populated when the run used edge detection.
        edge_cache: Option<TensorCache>,
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
                        items, tier2_items, config, cancel,
                        additional_items_rx,
                    } => {
                        run_batch_with_retry(
                            items, tier2_items, config, &cancel,
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
    initial_tier2: Vec<Tier2WorkItem>,
    config: ProcessingConfig,
    cancel: &Arc<AtomicBool>,
    additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let ProcessingConfig { model, jobs: initial_jobs, mask, force_cpu, line_mode, line_strength, solid_line_color } = config;
    let mut pending: VecDeque<WorkItem> = initial_items.into();
    let mut pending_tier2: VecDeque<Tier2WorkItem> = initial_tier2.into();
    let mut completed: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut max_jobs = initial_jobs;

    // Report loading status
    let first_id = pending.front().map(|(id, _, _)| *id)
        .or_else(|| pending_tier2.front().map(|t| t.item_id));
    if let Some(fid) = first_id {
        let _ = res_tx.send(WorkerResult::BatchProgress {
            item_id: fid,
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

        if pending.is_empty() && pending_tier2.is_empty() {
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

        // Spawn subprocess — cap engines at the number of Tier 1 items
        // (Tier 2 doesn't use engines; no point creating 4 for 2 images)
        let total_pending = pending.len() + pending_tier2.len();
        let effective_jobs = max_jobs.min(pending.len().max(1));
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
                        tensor_cache: None,
                        edge_cache: None,
                    });
                }
                for t2 in &pending_tier2 {
                    let _ = res_tx.send(WorkerResult::BatchItemDone {
                        item_id: t2.item_id,
                        result: Err(e.clone()),
                        tensor_cache: None,
                        edge_cache: None,
                    });
                }
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }
        };

        // Track items sent to this subprocess (for re-queue on crash)
        let mut sent_items: Vec<WorkItem> = Vec::new();
        let mut sent_tier2_ids: Vec<u64> = Vec::new();

        // Send initial burst: Tier 2 items first (faster), then Tier 1
        let burst = max_jobs.min(total_pending);
        let mut sent_count = 0;
        while sent_count < burst {
            if try_send_tier2(&mut sub, &mut pending_tier2, &mut sent_tier2_ids) {
                sent_count += 1;
            } else if let Some(item) = pending.pop_front() {
                if send_item_to_sub(&mut sub, &item).is_err() {
                    pending.push_front(item);
                    break;
                }
                sent_items.push(item);
                sent_count += 1;
            } else {
                break;
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
                // Any tempfiles written between drain and cancel honoring would leak.
                // Kill + cleanup mirrors the crash path.
                sub.kill();
                crate::subprocess::protocol::cleanup_ipc_temp();
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
                    SubprocessEvent::ImageDone {
                        item_id, result_path, width, height, active_provider,
                        tensor_cache_path, tensor_cache_height, tensor_cache_width,
                        edge_cache_path, edge_cache_height, edge_cache_width,
                    } => {
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

                        let read_tensor = |tp: &std::path::PathBuf, h: u32, w: u32| -> Option<TensorCache> {
                            let raw_bytes = std::fs::read(tp).ok()?;
                            let _ = std::fs::remove_file(tp);
                            let data = crate::subprocess::ipc::le_bytes_to_f32s(&raw_bytes);
                            Some(TensorCache { data, height: h, width: w, model })
                        };

                        // Read segmentation tensor cache (Tier 1 → Tier 2 mask reruns).
                        let tensor_cache = tensor_cache_path.as_ref().and_then(|tp| {
                            let th = tensor_cache_height?;
                            let tw = tensor_cache_width?;
                            read_tensor(tp, th, tw)
                        });

                        // Read DexiNed edge tensor cache (Tier 1 → Tier 2 edge reruns).
                        let edge_cache = edge_cache_path.as_ref().and_then(|tp| {
                            let th = edge_cache_height?;
                            let tw = edge_cache_width?;
                            read_tensor(tp, th, tw)
                        });

                        completed.insert(item_id);
                        sent_items.retain(|(id, _, _)| *id != item_id);
                        sent_tier2_ids.retain(|id| *id != item_id);

                        let _ = res_tx.send(WorkerResult::BatchItemDone { item_id, result, tensor_cache, edge_cache });
                        ctx.request_repaint();

                        // Admit next item if RSS allows
                        if !sub.should_pause_admission()
                            && !try_send_tier2(&mut sub, &mut pending_tier2, &mut sent_tier2_ids)
                        {
                            if let Some(item) = pending.pop_front() {
                                if send_item_to_sub(&mut sub, &item).is_ok() {
                                    sent_items.push(item);
                                } else {
                                    pending.push_front(item);
                                }
                            } else if let Some(ref rx) = additional_items_rx {
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
                        sent_tier2_ids.retain(|id| *id != item_id);
                        let _ = res_tx.send(WorkerResult::BatchItemDone {
                            item_id,
                            result: Err(error),
                            tensor_cache: None,
                            edge_cache: None,
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
            let all_sent_empty = sent_items.is_empty() && sent_tier2_ids.is_empty();
            if all_sent_empty && pending.is_empty() && pending_tier2.is_empty() {
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

            // Re-queue in-flight Tier 1 items (not just the one that crashed)
            let re_queued: Vec<WorkItem> = sent_items.into_iter()
                .filter(|(id, _, _)| !completed.contains(id))
                .collect();
            // Tier 2 in-flight items can't be re-queued (tensor data was consumed
            // by send_repostprocess). Report them as errors so they don't stay
            // stuck in Processing. The parent clears cached_tensor for errored
            // items, so next "Process" will use FullPipeline.
            for &tid in &sent_tier2_ids {
                if !completed.contains(&tid) {
                    let _ = res_tx.send(WorkerResult::BatchItemDone {
                        item_id: tid,
                        result: Err(crash_reason.clone()),
                        tensor_cache: None,
                        edge_cache: None,
                    });
                }
            }
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
                            tensor_cache: None,
                            edge_cache: None,
                        });
                    }
                }
                for t2 in &pending_tier2 {
                    let _ = res_tx.send(WorkerResult::BatchItemDone {
                        item_id: t2.item_id,
                        result: Err(err_msg.clone()),
                        tensor_cache: None,
                        edge_cache: None,
                    });
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
        if pending.is_empty() && pending_tier2.is_empty() {
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

/// Try to send a Tier 2 item to the subprocess. Returns true if sent.
fn try_send_tier2(
    sub: &mut SubprocessManager,
    pending_tier2: &mut VecDeque<Tier2WorkItem>,
    sent_tier2_ids: &mut Vec<u64>,
) -> bool {
    if let Some(t2) = pending_tier2.pop_front() {
        let tid = t2.item_id;
        if sub.send_repostprocess(
            t2.item_id, &t2.tensor_data, t2.tensor_height, t2.tensor_width,
            t2.model, &t2.original_bytes, t2.mask.clone(),
        ).is_ok() {
            sent_tier2_ids.push(tid);
            return true;
        }
        pending_tier2.push_front(t2);
    }
    false
}
