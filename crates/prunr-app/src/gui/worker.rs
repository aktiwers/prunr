use std::collections::VecDeque;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use prunr_core::{MaskSettings, EdgeSettings, EdgeScale, ModelKind, ProgressStage, ProcessResult, EDGE_SCALE_COUNT};
use crate::gui::settings::LineMode;
use crate::subprocess::protocol::{SubprocessEvent, CANCELLED_ERR_MSG};
use crate::subprocess::manager::SubprocessManager;

/// Maximum time the subprocess may stay silent with in-flight work before we
/// treat it as a hung crash. Real batches emit Progress events every few
/// hundred ms; 60s of silence means something is wrong.
const HANG_TIMEOUT: Duration = Duration::from_secs(60);

/// Graceful-shutdown budget at the end of a batch. Long enough for the child
/// to drop its model caches cleanly, short enough that a misbehaving worker
/// doesn't block the UI thread indefinitely.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// True when the subprocess is alive but hasn't produced events for too long.
/// Pure so it's unit-testable — the event loop calls this each iteration and
/// treats `true` as a crash, reusing the existing re-queue + retry path.
fn is_stalled(
    last_event_age: Duration,
    in_flight_count: usize,
    hang_timeout: Duration,
) -> bool {
    in_flight_count > 0 && last_event_age > hang_timeout
}

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
        let compressed = zstd::encode_all(
            crate::subprocess::ipc::f32s_as_le_bytes(&tc.data),
            1,
        ).ok()?;
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

/// Raw multi-scale DexiNed output for IPC transfer. `tensors` is indexed by
/// [`EdgeScale`] as `usize` — Fine=0, Balanced=1, Bold=2, Fused=3.
pub struct EdgeTensorCache {
    pub tensors: [Vec<f32>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

/// Zstd-compressed multi-scale DexiNed cache. All 4 scales are extracted from
/// one inference pass and compressed in parallel here, so scale switching in
/// live preview is a tensor lookup (one decompress, no re-inference).
pub struct CompressedEdgeTensors {
    tensors: [Vec<u8>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

impl CompressedEdgeTensors {
    /// Compress all 4 tensors with zstd level 1, in parallel via rayon.
    /// Runs off the UI thread (called from the BG-IO / subprocess-event path).
    /// Returns None if any compression fails.
    pub fn from_raw(raw: EdgeTensorCache) -> Option<Self> {
        use rayon::prelude::*;
        let parts: Vec<Vec<u8>> = raw.tensors.par_iter().map(|t| {
            zstd::encode_all(crate::subprocess::ipc::f32s_as_le_bytes(t), 1)
                .ok().unwrap_or_default()
        }).collect();
        if parts.iter().any(|p| p.is_empty()) { return None; }
        let [a, b, c, d] = parts.try_into().ok()?;
        Some(Self { tensors: [a, b, c, d], height: raw.height, width: raw.width })
    }

    /// Decompress one scale's tensor. Called per-dispatch in the hot path,
    /// so wrap the result in `Arc<Vec<f32>>` + a scale discriminator at the
    /// call site (BatchItem::volatile_edge_tensor) to amortise across a drag.
    pub fn decompress(&self, scale: EdgeScale) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(self.tensors[scale as usize].as_slice()).ok()?;
        Some(crate::subprocess::ipc::le_bytes_to_f32s(&bytes))
    }

    /// Total compressed bytes across all 4 scales (for budget tracking).
    pub fn compressed_size(&self) -> usize {
        self.tensors.iter().map(|t| t.len()).sum()
    }
}

/// Decompressed seg tensor + metadata — one Tier-2-ready input bundle.
/// Shared by `drag_export` (subject / mask layers) and `animation_sweep`.
pub struct SegBundle {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

impl CompressedTensor {
    /// Decompress into a bundle. `None` when the stored blob fails to decode.
    pub fn bundle(&self) -> Option<SegBundle> {
        Some(SegBundle {
            data: self.decompress()?,
            height: self.height,
            width: self.width,
            model: self.model,
        })
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

/// An AddEdgeInference work item: cached seg tensor + original image bytes.
/// Same shape as Tier2WorkItem — kept as a distinct type so the worker bridge
/// can't confuse them at dispatch time (different IPC commands).
pub struct AddEdgeWorkItem {
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
    pub edge: EdgeSettings,
}

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<WorkItem>,
        /// Tier 2 items: re-postprocess from cached tensor (skip inference).
        tier2_items: Vec<Tier2WorkItem>,
        /// AddEdgeInference items: cached seg tensor + DexiNed on masked image.
        /// Used for Off → SubjectOutline transitions so enabling the outline
        /// doesn't force a full seg re-inference.
        add_edge_items: Vec<AddEdgeWorkItem>,
        config: ProcessingConfig,
        cancels: super::processor::CancelRegistry,
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
        /// Multi-scale DexiNed cache — all 4 scales produced in one inference.
        /// Only populated when the run used edge detection.
        edge_cache: Option<EdgeTensorCache>,
    },
    BatchComplete,
    Cancelled,
    /// Subprocess crashed — retrying with reduced concurrency.
    SubprocessRetry {
        reduced_jobs: usize,
        re_queued_count: usize,
    },
    /// Subprocess reported the EP it actually built the session with. Sent
    /// once per subprocess spawn, before the first image completes, so the
    /// statusbar label corrects any mismatch with the startup guess from
    /// `OrtEngine::detect_active_provider` (e.g. CUDA driver installed but
    /// EP init fell back to DirectML at runtime).
    BackendReady(String),
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
                        items, tier2_items, add_edge_items, config, cancels,
                        additional_items_rx,
                    } => {
                        run_batch_with_retry(
                            items, tier2_items, add_edge_items, config, &cancels,
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

/// Batch-wide state carried across subprocess spawns. `max_jobs` halves on
/// each crash retry; `completed` lets the crash handler skip already-delivered
/// items when re-queueing in-flight work.
struct BatchRunState {
    pending: VecDeque<WorkItem>,
    pending_tier2: VecDeque<Tier2WorkItem>,
    pending_add_edge: VecDeque<AddEdgeWorkItem>,
    completed: std::collections::HashSet<u64>,
    max_jobs: usize,
}

/// Outcome of one subprocess's event loop. `Cancelled` means the user cancel
/// was already propagated (sub killed, IPC cleaned, `WorkerResult::Cancelled`
/// sent) — caller returns immediately.
enum EventLoopOutcome {
    Finished,
    Crashed(String),
    Cancelled,
}

/// What `poll_additional_items` found when the queues ran dry.
enum WaitAction {
    Continue, // more work in queues or just pushed
    Complete, // producer disconnected → batch done
    Sleep,    // producer still connected, nothing available yet
}

/// Run a batch with automatic retry on subprocess crash.
/// Reduces concurrency (jobs) on each crash: jobs → jobs/2 → 1.
/// If even 1 job crashes, marks remaining items as "insufficient memory".
#[tracing::instrument(
    skip_all,
    fields(
        tier1_count = initial_items.len(),
        tier2_count = initial_tier2.len(),
        // Snapshot of the user's requested job count — tracing fields are
        // captured at span entry. The effective count halves on each crash
        // retry; see the per-spawn span on `SubprocessManager::spawn`.
        initial_jobs = config.jobs,
    ),
)]
fn run_batch_with_retry(
    initial_items: Vec<WorkItem>,
    initial_tier2: Vec<Tier2WorkItem>,
    initial_add_edge: Vec<AddEdgeWorkItem>,
    config: ProcessingConfig,
    cancels: &super::processor::CancelRegistry,
    additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let ProcessingConfig { model, jobs: initial_jobs, mask, force_cpu, line_mode, edge } = config;
    let mut state = BatchRunState {
        pending: initial_items.into(),
        pending_tier2: initial_tier2.into(),
        pending_add_edge: initial_add_edge.into(),
        completed: std::collections::HashSet::new(),
        max_jobs: initial_jobs,
    };
    let additional_items_rx = additional_items_rx.as_ref();

    emit_loading_status(&state, res_tx, ctx);

    loop {
        if cancels.is_global_cancelled() {
            let _ = res_tx.send(WorkerResult::Cancelled);
            ctx.request_repaint();
            return;
        }

        match poll_additional_items(&mut state, additional_items_rx) {
            WaitAction::Complete => {
                let _ = res_tx.send(WorkerResult::BatchComplete);
                ctx.request_repaint();
                return;
            }
            WaitAction::Sleep => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            WaitAction::Continue => {}
        }

        let Some((mut sub, mut sent_items, mut sent_tier2_ids, mut sent_add_edge_ids)) = spawn_and_initial_burst(
            &mut state, model, mask, force_cpu, line_mode, edge, cancels, res_tx, ctx,
        ) else {
            return; // spawn failed — errors + BatchComplete already sent
        };

        let outcome = run_event_loop(
            &mut sub, &mut state, &mut sent_items, &mut sent_tier2_ids, &mut sent_add_edge_ids,
            additional_items_rx, cancels, model, res_tx, ctx,
        );

        match outcome {
            EventLoopOutcome::Cancelled => return,
            EventLoopOutcome::Crashed(reason) => {
                let should_retry = handle_crash_and_retry(
                    &mut state, sent_items, sent_tier2_ids, sent_add_edge_ids, reason,
                    &mut sub, additional_items_rx, res_tx, ctx,
                );
                if !should_retry {
                    return;
                }
            }
            EventLoopOutcome::Finished => {
                if !finalize_or_continue(&mut state, &mut sub, additional_items_rx, res_tx, ctx) {
                    return;
                }
            }
        }
    }
}

/// Emit the first-frame "Loading model..." status so the UI shows activity
/// before the subprocess reports its own progress.
fn emit_loading_status(
    state: &BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let first_id = state.pending.front().map(|(id, _, _)| *id)
        .or_else(|| state.pending_tier2.front().map(|t| t.item_id))
        .or_else(|| state.pending_add_edge.front().map(|a| a.item_id));
    if let Some(fid) = first_id {
        let _ = res_tx.send(WorkerResult::BatchProgress {
            item_id: fid,
            stage: ProgressStage::LoadingModel,
            pct: 0.0,
        });
        ctx.request_repaint();
    }
}

/// When the batch queues are empty, check if the admission controller has
/// more to send. Returns immediately — the caller owns the sleep/cancel cadence.
fn poll_additional_items(
    state: &mut BatchRunState,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
) -> WaitAction {
    if !state.pending.is_empty() || !state.pending_tier2.is_empty() || !state.pending_add_edge.is_empty() {
        return WaitAction::Continue;
    }
    let Some(rx) = additional_items_rx else {
        return WaitAction::Complete;
    };
    match rx.try_recv() {
        Ok(item) => {
            state.pending.push_back(item);
            WaitAction::Continue
        }
        Err(mpsc::TryRecvError::Disconnected) => WaitAction::Complete,
        Err(mpsc::TryRecvError::Empty) => WaitAction::Sleep,
    }
}

/// Spawn a subprocess and send the initial burst (Tier 2 first — no inference
/// needed — then Tier 1 up to `max_jobs`). On spawn failure, reports errors
/// for every pending item and sends `BatchComplete`; returns `None`.
fn spawn_and_initial_burst(
    state: &mut BatchRunState,
    model: ModelKind,
    mask: MaskSettings,
    force_cpu: bool,
    line_mode: LineMode,
    edge: EdgeSettings,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<(SubprocessManager, Vec<WorkItem>, Vec<u64>, Vec<u64>)> {
    // Drop anything cancelled before spawn — they don't count toward
    // `effective_jobs` and shouldn't occupy burst slots either.
    drop_cancelled_pending(state, cancels, res_tx, ctx);
    let total_pending = state.pending.len() + state.pending_tier2.len() + state.pending_add_edge.len();
    if total_pending == 0 {
        let _ = res_tx.send(WorkerResult::BatchComplete);
        ctx.request_repaint();
        return None;
    }
    // Cap engines at the number of Tier 1 items — Tier 2 + AddEdge don't
    // use the seg engine pool.
    let effective_jobs = state.max_jobs.min(state.pending.len().max(1));

    let (mut sub, active_provider) = match SubprocessManager::spawn(
        model, effective_jobs, mask, force_cpu, line_mode, edge,
    ) {
        Ok(s) => s,
        Err(e) => {
            emit_pending_errors(state, &e, res_tx);
            let _ = res_tx.send(WorkerResult::BatchComplete);
            ctx.request_repaint();
            return None;
        }
    };
    let _ = res_tx.send(WorkerResult::BackendReady(active_provider));
    ctx.request_repaint();

    let mut sent_items: Vec<WorkItem> = Vec::new();
    let mut sent_tier2_ids: Vec<u64> = Vec::new();
    let mut sent_add_edge_ids: Vec<u64> = Vec::new();
    let burst = state.max_jobs.min(total_pending);
    let mut sent_count = 0;
    while sent_count < burst {
        if try_send_tier2(&mut sub, &mut state.pending_tier2, &mut sent_tier2_ids) {
            sent_count += 1;
        } else if try_send_add_edge(&mut sub, &mut state.pending_add_edge, &mut sent_add_edge_ids) {
            sent_count += 1;
        } else if let Some(item) = state.pending.pop_front() {
            if send_item_to_sub(&mut sub, &item).is_err() {
                state.pending.push_front(item);
                break;
            }
            sent_items.push(item);
            sent_count += 1;
        } else {
            break;
        }
    }
    Some((sub, sent_items, sent_tier2_ids, sent_add_edge_ids))
}

/// Drive one subprocess through its lifecycle — poll events, admit more work
/// on ImageDone, detect crash / hang, stop on Finished or cancel.
#[allow(clippy::too_many_arguments)]
fn run_event_loop(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent_items: &mut Vec<WorkItem>,
    sent_tier2_ids: &mut Vec<u64>,
    sent_add_edge_ids: &mut Vec<u64>,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
    cancels: &super::processor::CancelRegistry,
    model: ModelKind,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> EventLoopOutcome {
    let mut last_event_at = Instant::now();
    let mut subprocess_finished = false;
    // Items we've already told the subprocess to cancel — avoids re-sending
    // `CancelItem` every iteration while we wait for the ImageError echo.
    let mut cancel_notified: std::collections::HashSet<u64> = std::collections::HashSet::new();

    while !subprocess_finished {
        if cancels.is_global_cancelled() {
            cancel_subprocess(sub, res_tx, ctx);
            return EventLoopOutcome::Cancelled;
        }

        // Notify subprocess of newly cancelled in-flight items so it can drop
        // them at the next dispatch check instead of finishing the work.
        forward_item_cancels(
            sub, sent_items, sent_tier2_ids, sent_add_edge_ids, cancels, &mut cancel_notified,
        );

        // Real-crash check first so a segfault is reported as such rather
        // than masked by the watchdog firing on the same silence.
        if !sub.is_alive() {
            return EventLoopOutcome::Crashed(sub.crash_reason());
        }

        let in_flight_count = sent_items.len() + sent_tier2_ids.len() + sent_add_edge_ids.len();
        if is_stalled(last_event_at.elapsed(), in_flight_count, HANG_TIMEOUT) {
            sub.kill();
            return EventLoopOutcome::Crashed(format!(
                "Worker stopped responding (no events for {}s)",
                HANG_TIMEOUT.as_secs(),
            ));
        }

        for event in sub.poll_events() {
            // Every event — Progress, ImageDone, ImageError, RssUpdate
            // (emitted per-completion, not on a timer) — proves liveness.
            last_event_at = Instant::now();
            if matches!(event, SubprocessEvent::Finished) {
                subprocess_finished = true;
                continue;
            }
            handle_subprocess_event(
                event, sub, state, sent_items, sent_tier2_ids, sent_add_edge_ids,
                additional_items_rx, cancels, model, res_tx, ctx,
            );
        }

        // If everything drained and no more coming, mark finished.
        let all_sent_empty = sent_items.is_empty() && sent_tier2_ids.is_empty() && sent_add_edge_ids.is_empty();
        if all_sent_empty && state.pending.is_empty() && state.pending_tier2.is_empty() && state.pending_add_edge.is_empty() {
            if let Some(rx) = additional_items_rx {
                match rx.try_recv() {
                    Ok(item) => state.pending.push_back(item),
                    Err(mpsc::TryRecvError::Disconnected) => subprocess_finished = true,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            } else {
                subprocess_finished = true;
            }
        }

        if !subprocess_finished {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    EventLoopOutcome::Finished
}

/// Route one `SubprocessEvent` to its handler. `Finished` is handled by the
/// event loop itself (sets the exit flag); `RssUpdate` is absorbed by
/// `SubprocessManager` internally.
#[allow(clippy::too_many_arguments)]
fn handle_subprocess_event(
    event: SubprocessEvent,
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent_items: &mut Vec<WorkItem>,
    sent_tier2_ids: &mut Vec<u64>,
    sent_add_edge_ids: &mut Vec<u64>,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
    cancels: &super::processor::CancelRegistry,
    model: ModelKind,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match event {
        SubprocessEvent::Progress { item_id, stage, pct } => {
            let _ = res_tx.send(WorkerResult::BatchProgress { item_id, stage, pct });
            ctx.request_repaint();
        }
        SubprocessEvent::ImageDone {
            item_id, result_path, width, height, active_provider,
            tensor_cache_path, tensor_cache_height, tensor_cache_width,
            edge_cache_path, edge_cache_height, edge_cache_width,
        } => {
            let result = read_result_image(&result_path, width, height, &active_provider);
            let tensor_cache = read_tensor_cache(
                tensor_cache_path.as_ref(), tensor_cache_height, tensor_cache_width, model,
            );
            let edge_cache = read_edge_tensor_cache(
                edge_cache_path.as_ref(), edge_cache_height, edge_cache_width,
            );

            state.completed.insert(item_id);
            sent_items.retain(|(id, _, _)| *id != item_id);
            sent_tier2_ids.retain(|id| *id != item_id);
            sent_add_edge_ids.retain(|id| *id != item_id);

            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id, result, tensor_cache, edge_cache,
            });
            ctx.request_repaint();

            admit_next_item(sub, state, sent_items, sent_tier2_ids, sent_add_edge_ids, additional_items_rx, cancels, res_tx, ctx);
        }
        SubprocessEvent::ImageError { item_id, error } => {
            state.completed.insert(item_id);
            sent_items.retain(|(id, _, _)| *id != item_id);
            sent_tier2_ids.retain(|id| *id != item_id);
            sent_add_edge_ids.retain(|id| *id != item_id);
            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id,
                result: Err(error),
                tensor_cache: None,
                edge_cache: None,
            });
            ctx.request_repaint();
        }
        SubprocessEvent::RssUpdate { .. } | SubprocessEvent::Finished => {}
        _ => {}
    }
}

/// Read the result RGBA from the subprocess's temp file and wrap it. Always
/// removes the temp file (success or parse failure) to keep `/dev/shm` tidy.
fn read_result_image(
    result_path: &std::path::Path,
    width: u32,
    height: u32,
    active_provider: &str,
) -> Result<ProcessResult, String> {
    let result = std::fs::read(result_path)
        .ok()
        .and_then(|data| image::RgbaImage::from_raw(width, height, data))
        .map(|rgba_image| ProcessResult {
            rgba_image,
            active_provider: active_provider.to_string(),
        })
        .ok_or_else(|| "Failed to read result from subprocess".to_string());
    let _ = std::fs::remove_file(result_path);
    result
}

/// Read a temp file and delete it in the same breath. Invariant shared by
/// every subprocess-IPC reader: the temp file is one-shot, always removed.
fn read_and_delete(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let _ = std::fs::remove_file(path);
    Some(bytes)
}

/// Read the segmentation tensor cache from disk.
fn read_tensor_cache(
    path: Option<&std::path::PathBuf>,
    height: Option<u32>,
    width: Option<u32>,
    model: ModelKind,
) -> Option<TensorCache> {
    let p = path?;
    let bytes = read_and_delete(p)?;
    let data = crate::subprocess::ipc::le_bytes_to_f32s(&bytes);
    let h = height?;
    let w = width?;
    let expected = (h as usize) * (w as usize);
    let head: Vec<f32> = data.iter().take(6).copied().collect();
    tracing::debug!(
        path = %p.display(), bytes_len = bytes.len(),
        tensor_len = data.len(), expected_len = expected,
        h, w, ?head,
        "parent read seg tensor",
    );
    if data.len() != expected {
        tracing::error!(
            path = %p.display(),
            got = data.len(), expected,
            "seg tensor length mismatch — mask will be garbage",
        );
    }
    Some(TensorCache { data, height: h, width: w, model })
}

/// Read the DexiNed multi-scale cache. Child concatenates 4 tensors into
/// one file (each h*w*4 bytes); parent splits into equal chunks.
fn read_edge_tensor_cache(
    path: Option<&std::path::PathBuf>,
    height: Option<u32>,
    width: Option<u32>,
) -> Option<EdgeTensorCache> {
    let h = height?;
    let w = width?;
    let raw_bytes = read_and_delete(path?)?;

    let per_tensor_bytes = (h as usize) * (w as usize) * std::mem::size_of::<f32>();
    let expected = per_tensor_bytes * EDGE_SCALE_COUNT;
    if raw_bytes.len() != expected {
        tracing::error!(
            got = raw_bytes.len(),
            expected,
            "edge cache file size mismatch — discarding",
        );
        return None;
    }

    let mut chunks = raw_bytes.chunks_exact(per_tensor_bytes);
    let mut next = || -> Option<Vec<f32>> {
        Some(crate::subprocess::ipc::le_bytes_to_f32s(chunks.next()?))
    };
    let a = next()?;
    let b = next()?;
    let c = next()?;
    let d = next()?;
    Some(EdgeTensorCache { tensors: [a, b, c, d], height: h, width: w })
}

/// After an ImageDone, pull one more item into the subprocess if RSS allows.
/// Prefers Tier 2 (no inference), then Tier 1 pending, then the admission
/// overflow channel.
#[allow(clippy::too_many_arguments)]
fn admit_next_item(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent_items: &mut Vec<WorkItem>,
    sent_tier2_ids: &mut Vec<u64>,
    sent_add_edge_ids: &mut Vec<u64>,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    drop_cancelled_pending(state, cancels, res_tx, ctx);
    if sub.should_pause_admission() {
        return;
    }
    if try_send_tier2(sub, &mut state.pending_tier2, sent_tier2_ids) {
        return;
    }
    if try_send_add_edge(sub, &mut state.pending_add_edge, sent_add_edge_ids) {
        return;
    }
    if let Some(item) = state.pending.pop_front() {
        if send_item_to_sub(sub, &item).is_ok() {
            sent_items.push(item);
        } else {
            state.pending.push_front(item);
        }
        return;
    }
    if let Some(rx) = additional_items_rx {
        if let Ok(item) = rx.try_recv() {
            if cancels.is_cancelled(item.0) {
                report_cancelled(item.0, state, res_tx, ctx);
            } else if send_item_to_sub(sub, &item).is_ok() {
                sent_items.push(item);
            } else {
                state.pending.push_back(item);
            }
        }
    }
}

fn drop_cancelled_pending(
    state: &mut BatchRunState,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let mut cancelled: Vec<u64> = Vec::new();
    for (id, _, _) in &state.pending {
        if cancels.is_cancelled(*id) { cancelled.push(*id); }
    }
    for w in &state.pending_tier2 {
        if cancels.is_cancelled(w.item_id) { cancelled.push(w.item_id); }
    }
    for w in &state.pending_add_edge {
        if cancels.is_cancelled(w.item_id) { cancelled.push(w.item_id); }
    }
    if cancelled.is_empty() { return; }
    state.pending.retain(|(id, _, _)| !cancels.is_cancelled(*id));
    state.pending_tier2.retain(|w| !cancels.is_cancelled(w.item_id));
    state.pending_add_edge.retain(|w| !cancels.is_cancelled(w.item_id));
    for id in cancelled {
        report_cancelled(id, state, res_tx, ctx);
    }
}

/// `notified` prevents re-sending `CancelItem` every 50ms while the worker
/// winds down the job.
fn forward_item_cancels(
    sub: &mut SubprocessManager,
    sent_items: &[WorkItem],
    sent_tier2_ids: &[u64],
    sent_add_edge_ids: &[u64],
    cancels: &super::processor::CancelRegistry,
    notified: &mut std::collections::HashSet<u64>,
) {
    let in_flight = sent_items.iter().map(|(id, _, _)| *id)
        .chain(sent_tier2_ids.iter().copied())
        .chain(sent_add_edge_ids.iter().copied());
    for id in in_flight {
        if cancels.is_cancelled(id) && notified.insert(id) {
            let _ = sub.send_cancel_item(id);
        }
    }
}

fn report_cancelled(
    item_id: u64,
    state: &mut BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    state.completed.insert(item_id);
    let _ = res_tx.send(WorkerResult::BatchItemDone {
        item_id,
        result: Err(CANCELLED_ERR_MSG.to_string()),
        tensor_cache: None,
        edge_cache: None,
    });
    ctx.request_repaint();
}

/// Cancel path: kill the child immediately and clean up. We used to send
/// `Cancel` + sleep 200ms to let the child emit `Finished`, but killing the
/// process is instant and orphaned IPC temps are swept by `cleanup_ipc_temp`
/// — the politeness delay just made "Cancel All" feel laggy.
fn cancel_subprocess(
    sub: &mut SubprocessManager,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    sub.kill();
    crate::subprocess::protocol::cleanup_ipc_temp();
    let _ = res_tx.send(WorkerResult::Cancelled);
    ctx.request_repaint();
}

/// Post-crash: re-queue in-flight Tier 1 work, error out in-flight Tier 2
/// (tensor data already consumed by IPC), halve `max_jobs`, and either
/// continue (return `true`) or bail if we were already at 1 job (return
/// `false` after emitting terminal errors + `BatchComplete`).
#[allow(clippy::too_many_arguments)]
fn handle_crash_and_retry(
    state: &mut BatchRunState,
    sent_items: Vec<WorkItem>,
    sent_tier2_ids: Vec<u64>,
    sent_add_edge_ids: Vec<u64>,
    crash_reason: String,
    sub: &mut SubprocessManager,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> bool {
    // Tier 2 + AddEdge can't be re-queued — their tensor data is owned by the
    // IPC temp file, already cleaned up. Parent errors the item so a follow-up
    // Process runs the full pipeline.
    for tid in sent_tier2_ids.into_iter().chain(sent_add_edge_ids) {
        if !state.completed.contains(&tid) {
            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id: tid,
                result: Err(crash_reason.clone()),
                tensor_cache: None,
                edge_cache: None,
            });
        }
    }

    let re_queued: Vec<WorkItem> = sent_items.into_iter()
        .filter(|(id, _, _)| !state.completed.contains(id))
        .collect();
    let re_count = re_queued.len();

    // Drain late additions before re-queuing so the original in-flight
    // order stays at the front.
    if let Some(rx) = additional_items_rx {
        while let Ok(item) = rx.try_recv() {
            state.pending.push_back(item);
        }
    }
    for item in re_queued.into_iter().rev() {
        state.pending.push_front(item);
    }

    let old_jobs = state.max_jobs;
    state.max_jobs = (state.max_jobs / 2).max(1);

    sub.kill();
    crate::subprocess::protocol::cleanup_ipc_temp();

    if old_jobs == 1 {
        let err_msg = format!("{crash_reason} \u{2014} try a smaller model");
        emit_pending_errors(state, &err_msg, res_tx);
        let _ = res_tx.send(WorkerResult::BatchComplete);
        ctx.request_repaint();
        return false;
    }

    let _ = res_tx.send(WorkerResult::SubprocessRetry {
        reduced_jobs: state.max_jobs,
        re_queued_count: re_count,
    });
    ctx.request_repaint();
    true
}

/// Emit `BatchItemDone(Err)` for every pending Tier 1 + Tier 2 item that
/// hasn't already completed. Used on spawn failure and at max-retries.
fn emit_pending_errors(
    state: &BatchRunState,
    err_msg: &str,
    res_tx: &mpsc::Sender<WorkerResult>,
) {
    for (id, _, _) in &state.pending {
        if !state.completed.contains(id) {
            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id: *id,
                result: Err(err_msg.to_string()),
                tensor_cache: None,
                edge_cache: None,
            });
        }
    }
    for t2 in &state.pending_tier2 {
        let _ = res_tx.send(WorkerResult::BatchItemDone {
            item_id: t2.item_id,
            result: Err(err_msg.to_string()),
            tensor_cache: None,
            edge_cache: None,
        });
    }
    for ae in &state.pending_add_edge {
        let _ = res_tx.send(WorkerResult::BatchItemDone {
            item_id: ae.item_id,
            result: Err(err_msg.to_string()),
            tensor_cache: None,
            edge_cache: None,
        });
    }
}

/// After a subprocess finishes normally, decide whether the batch is done or
/// another subprocess needs spawning. Returns `false` when the batch is done
/// (BatchComplete already sent); `true` to loop back and spawn again.
fn finalize_or_continue(
    state: &mut BatchRunState,
    sub: &mut SubprocessManager,
    additional_items_rx: Option<&mpsc::Receiver<WorkItem>>,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> bool {
    if state.pending.is_empty() && state.pending_tier2.is_empty() && state.pending_add_edge.is_empty() {
        if let Some(rx) = additional_items_rx {
            match rx.try_recv() {
                Err(mpsc::TryRecvError::Disconnected) => {
                    let _ = res_tx.send(WorkerResult::BatchComplete);
                    ctx.request_repaint();
                    return false;
                }
                Ok(item) => state.pending.push_back(item),
                Err(mpsc::TryRecvError::Empty) => {
                    // More might be coming — loop back.
                }
            }
        } else {
            let _ = res_tx.send(WorkerResult::BatchComplete);
            ctx.request_repaint();
            return false;
        }
    }

    // 5s covers model-cache teardown; if the worker ignores us, Drop's
    // shorter timeout + force-kill closes the gap.
    let _ = sub.shutdown_with_timeout(SHUTDOWN_TIMEOUT);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_stalled_false_when_no_work_in_flight() {
        // Idle subprocess waiting for admission is NOT a stall — we only flag
        // the subprocess as hung if it's sitting on work it hasn't reported.
        assert!(!is_stalled(Duration::from_secs(120), 0, Duration::from_secs(60)));
    }

    #[test]
    fn is_stalled_true_when_silence_exceeds_timeout_with_work() {
        assert!(is_stalled(Duration::from_secs(61), 1, Duration::from_secs(60)));
    }

    #[test]
    fn is_stalled_false_when_under_timeout() {
        assert!(!is_stalled(Duration::from_secs(59), 1, Duration::from_secs(60)));
    }

    #[test]
    fn is_stalled_at_exact_boundary_is_false() {
        // Strict `>` not `>=`: a subprocess that emits an event every
        // `hang_timeout` seconds on the dot is not considered stalled.
        assert!(!is_stalled(Duration::from_secs(60), 1, Duration::from_secs(60)));
    }

    #[test]
    fn is_stalled_scales_with_in_flight_count() {
        // Any non-zero in-flight count triggers the detector given silence.
        assert!(is_stalled(Duration::from_secs(90), 5, Duration::from_secs(60)));
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

/// Try to send an AddEdgeInference item to the subprocess. Returns true if sent.
fn try_send_add_edge(
    sub: &mut SubprocessManager,
    pending_add_edge: &mut VecDeque<AddEdgeWorkItem>,
    sent_add_edge_ids: &mut Vec<u64>,
) -> bool {
    if let Some(ae) = pending_add_edge.pop_front() {
        let tid = ae.item_id;
        if sub.send_add_edge_inference(
            ae.item_id, &ae.tensor_data, ae.tensor_height, ae.tensor_width,
            ae.model, &ae.original_bytes, ae.mask.clone(),
        ).is_ok() {
            sent_add_edge_ids.push(tid);
            return true;
        }
        pending_add_edge.push_front(ae);
    }
    false
}
