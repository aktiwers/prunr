use std::collections::VecDeque;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use prunr_core::{MaskSettings, EdgeSettings, EdgeScale, ModelKind, ProcessResult, EDGE_SCALE_COUNT};
use crate::gui::settings::LineMode;
use crate::subprocess::protocol::{SubprocessEvent, CANCELLED_ERR_MSG};
use crate::subprocess::manager::{SubprocessManager, WorkerLines};

/// Maximum time the subprocess may stay silent with in-flight work before we
/// treat it as a hung crash. Real batches emit Progress events every few
/// hundred ms; 60s of silence means something is wrong.
const HANG_TIMEOUT: Duration = Duration::from_secs(60);

/// Graceful-shutdown budget at the end of a batch. Long enough for the child
/// to drop its model caches cleanly, short enough that a misbehaving worker
/// doesn't block the UI thread indefinitely.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the worker that ran the last batch stays warm; the same as
/// the LaMa session and the SD worker.
const WARM_IDLE: Duration = Duration::from_secs(300);
const WARM_POLL: Duration = Duration::from_secs(30);

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
/// Cloning shares the bytes, so an export can take it to its thread.
#[derive(Clone)]
pub struct CompressedTensor {
    compressed: std::sync::Arc<[u8]>,
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
        Some(Self { compressed: compressed.into(), height: tc.height, width: tc.width, model: tc.model })
    }

    /// Decompress to raw f32 tensor data for Tier 2 dispatch.
    pub fn decompress(&self) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(&*self.compressed).ok()?;
        Some(crate::subprocess::ipc::le_bytes_to_f32s(&bytes))
    }

    /// Compressed size in bytes (for budget tracking).
    pub fn compressed_size(&self) -> usize {
        self.compressed.len()
    }
}

/// Raw multi-scale TEED output for IPC transfer. `tensors` is indexed by
/// [`EdgeScale`] as `usize`; `height` × `width` is the base size, a
/// scale's own size is `prunr_core::edge_scale_dims`.
pub struct EdgeTensorCache {
    pub tensors: [Vec<f32>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

/// Zstd-compressed multi-scale TEED cache. Every scale is kept and
/// compressed in parallel here, so scale switching in live preview is a
/// tensor lookup (one decompress, no re-inference).
/// Cloning shares the bytes.
#[derive(Clone)]
pub struct CompressedEdgeTensors {
    tensors: [std::sync::Arc<[u8]>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

impl CompressedEdgeTensors {
    /// Compress every scale's tensor with zstd level 1, in parallel via rayon.
    /// Runs off the UI thread (called from the BG-IO / subprocess-event path).
    /// Returns None if any compression fails.
    pub fn from_raw(raw: EdgeTensorCache) -> Option<Self> {
        use rayon::prelude::*;
        let parts: Vec<Vec<u8>> = raw.tensors.par_iter().map(|t| {
            zstd::encode_all(crate::subprocess::ipc::f32s_as_le_bytes(t), 1)
                .ok().unwrap_or_default()
        }).collect();
        if parts.iter().any(|p| p.is_empty()) { return None; }
        let tensors: [Vec<u8>; EDGE_SCALE_COUNT] = parts.try_into().ok()?;
        Some(Self { tensors: tensors.map(Into::into), height: raw.height, width: raw.width })
    }

    /// Decompress one scale's tensor. Called per-dispatch in the hot path,
    /// so wrap the result in `Arc<Vec<f32>>` + a scale discriminator at the
    /// call site (BatchItem::volatile_edge_tensor) to amortise across a drag.
    pub fn decompress(&self, scale: EdgeScale) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(&*self.tensors[scale as usize]).ok()?;
        Some(crate::subprocess::ipc::le_bytes_to_f32s(&bytes))
    }

    /// `scale`'s tensor size, height first.
    pub fn dims(&self, scale: EdgeScale) -> (u32, u32) {
        prunr_core::edge_scale_dims(scale, self.height, self.width)
    }

    /// The scales `edge` draws, decompressed for a compose. The other
    /// slots stay empty.
    pub fn for_compose(&self, edge: &EdgeSettings) -> Option<prunr_core::EdgeInferenceResult> {
        let mut tensors: [Vec<f32>; EDGE_SCALE_COUNT] = Default::default();
        for &scale in edge.drawn_scales() {
            tensors[scale as usize] = self.decompress(scale)?;
        }
        Some(prunr_core::EdgeInferenceResult { tensors, height: self.height, width: self.width })
    }

    /// Total compressed bytes across every scale (for budget tracking).
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

/// Work that starts from an item's cached seg tensor instead of a seg run:
/// a mask re-postprocess, or edge detection added on top of the mask.
pub struct CachedTensorWork {
    pub item_id: u64,
    pub tensor_data: Vec<f32>,
    pub tensor_height: u32,
    pub tensor_width: u32,
    pub model: ModelKind,
    pub original_bytes: Arc<Vec<u8>>,
    pub mask: MaskSettings,
}


/// Bundled processing settings — avoids passing 6+ individual fields.
/// Clone/PartialEq enable the pre-warm path to check whether the warm
/// subprocess was booted with the same Init-affecting config as the
/// incoming batch (and therefore safely reusable).
#[derive(Clone, PartialEq)]
pub struct ProcessingConfig {
    pub model: ModelKind,
    pub jobs: usize,
    pub mask: MaskSettings,
    pub force_cpu: bool,
    pub line_mode: LineMode,
    pub edge: EdgeSettings,
}

impl ProcessingConfig {
    /// Whether a worker built for `self` can run `next`: the same model
    /// on the same processor, with a pool at least as large. Mask, Sketch
    /// mode and line settings are not part of it; a batch sends those with
    /// `Configure`.
    pub fn reusable_by(&self, next: &ProcessingConfig) -> bool {
        self.model == next.model && self.force_cpu == next.force_cpu && self.jobs >= next.jobs
    }
}

pub enum WorkerMessage {
    BatchProcess {
        items: Vec<WorkItem>,
        /// Tier 2 items: re-postprocess from cached tensor (skip inference).
        tier2_items: Vec<CachedTensorWork>,
        /// AddEdgeInference items: cached seg tensor + TEED on masked image.
        /// Used for Off → SubjectOutline transitions so enabling the outline
        /// doesn't force a full seg re-inference.
        add_edge_items: Vec<CachedTensorWork>,
        config: Box<ProcessingConfig>,
        cancels: super::processor::CancelRegistry,
        /// Channel for additional items admitted by the memory controller.
        additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    },
    /// Drop any warm seg subprocess so its engine pool's resident RAM
    /// is reclaimed. Sent by the GUI on every `model_changed` so a
    /// previous backend (e.g. BiRefNetLite at ~2 GB) doesn't pin
    /// memory across the user's "I'm done with that model" signal.
    /// Idempotent; no-op when nothing is warm.
    ReleaseWarm,
}

pub enum WorkerResult {
    Report {
        item_id: u64,
        update: prunr_core::ProgressUpdate,
    },
    BatchItemDone {
        item_id: u64,
        result: Result<ProcessResult, String>,
        /// Cached segmentation tensor from Tier 1 (for future mask-tier reruns).
        tensor_cache: Option<TensorCache>,
        /// Multi-scale TEED cache — every scale, kept so a switch needs no run.
        /// Only populated when the run used edge detection.
        edge_cache: Option<EdgeTensorCache>,
    },
    /// The batch is over; `items` are the images it owned (given at the
    /// start or admitted while it ran).
    BatchComplete { items: Vec<u64> },
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
///
/// When `prewarm` is `Some`, the bridge boots a subprocess with that config
/// at startup. A batch reuses the warm worker when its model, processor
/// and pool size fit (`ProcessingConfig::reusable_by`) and sends it the
/// run's mask and Sketch settings; the worker that ran a batch stays warm
/// for `WARM_IDLE` after it, so the next Process skips the model load.
/// A model switch (`ReleaseWarm`), a crash or Cancel drops it.
pub fn spawn_worker(
    ctx: egui::Context,
    prewarm: Option<ProcessingConfig>,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("prunr-bridge".into())
        .spawn(move || {
            // Pre-warm: try to spawn a subprocess with the startup config
            // before any message arrives. Blocks this thread for the model-
            // load duration (UI stays responsive — bridge has its own
            // thread). If a Process click arrives during warm-up, it
            // queues on msg_rx and gets picked up once the sub is live.
            let mut warm: Option<(SubprocessManager, ProcessingConfig, Instant)> = None;
            if let Some(cfg) = prewarm {
                let ProcessingConfig { model, jobs, mask, force_cpu, line_mode, edge } = cfg.clone();
                let started = std::time::Instant::now();
                match SubprocessManager::spawn(model, jobs, mask, force_cpu, WorkerLines::app(line_mode, edge)) {
                    Ok((sub, provider)) => {
                        let elapsed_ms = started.elapsed().as_millis();
                        tracing::info!(
                            stage = "prewarm",
                            elapsed_ms,
                            model = ?model,
                            line_mode = ?line_mode,
                            "subprocess cold-start",
                        );
                        let _ = res_tx.send(WorkerResult::BackendReady(provider));
                        ctx.request_repaint();
                        warm = Some((sub, cfg, Instant::now()));
                    }
                    Err(err) => {
                        tracing::warn!(%err, "pre-warm subprocess failed, falling back to per-batch spawn");
                    }
                }
            }

            loop {
                let msg = match msg_rx.recv_timeout(WARM_POLL) {
                    Ok(msg) => msg,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if warm.as_ref().is_some_and(|(_, _, since)| since.elapsed() >= WARM_IDLE) {
                            if let Some((mut sub, cfg, _)) = warm.take() {
                                tracing::info!(model = ?cfg.model, "seg subprocess released after idling");
                                let _ = sub.shutdown_with_timeout(SHUTDOWN_TIMEOUT);
                            }
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                match msg {
                    WorkerMessage::ReleaseWarm => {
                        if let Some((mut sub, cfg, _)) = warm.take() {
                            tracing::info!(model = ?cfg.model, "seg subprocess released on model switch");
                            let _ = sub.shutdown_with_timeout(std::time::Duration::from_secs(2));
                        }
                    }
                    WorkerMessage::BatchProcess {
                        items, tier2_items, add_edge_items, config, cancels,
                        additional_items_rx,
                    } => {
                        // The pool this batch needs: one engine per image
                        // of its first burst, up to the jobs setting.
                        let needed = ProcessingConfig { jobs: config.jobs.min(items.len().max(1)), ..(*config).clone() };
                        let mut reuse: Option<(SubprocessManager, ProcessingConfig)> = None;
                        let respawn_reason = match warm.take() {
                            None => Some("no_warm_worker"),
                            Some((mut sub, cfg, _)) => {
                                if !sub.is_alive() {
                                    Some("warm_worker_dead")
                                } else if cfg.reusable_by(&needed) {
                                    reuse = Some((sub, cfg));
                                    None
                                } else {
                                    let _ = sub.shutdown_with_timeout(std::time::Duration::from_secs(2));
                                    Some("engines_differ")
                                }
                            }
                        };
                        let (decision, reason) = match respawn_reason {
                            None => ("reuse", "fits"),
                            Some(r) => ("respawn", r),
                        };
                        tracing::info!(
                            stage = "batch_dispatch",
                            decision,
                            reason,
                            model = ?config.model,
                            line_mode = ?config.line_mode,
                            "subprocess cold-start",
                        );
                        let state = BatchRunState::new(
                            items, tier2_items, add_edge_items, additional_items_rx, *config,
                        );
                        warm = run_batch_with_retry(reuse, state, &cancels, &res_tx, &ctx)
                            .map(|(sub, cfg)| (sub, cfg, Instant::now()));
                    }
                }
            }
        })
        .expect("failed to spawn bridge thread");

    (msg_tx, res_rx)
}

/// Batch-wide state carried across subprocess spawns. `max_jobs` starts at
/// `config.jobs` and halves on each crash retry; `completed` lets the crash
/// handler skip already-delivered items when re-queueing in-flight work.
struct BatchRunState {
    config: ProcessingConfig,
    pending: VecDeque<WorkItem>,
    pending_tier2: VecDeque<CachedTensorWork>,
    pending_add_edge: VecDeque<CachedTensorWork>,
    /// Items admitted by the memory controller after the batch started.
    additional_rx: Option<mpsc::Receiver<WorkItem>>,
    completed: std::collections::HashSet<u64>,
    /// Every image this batch was given or admitted, so its end frees
    /// only its own.
    owned: std::collections::HashSet<u64>,
    max_jobs: usize,
}

impl BatchRunState {
    fn new(
        items: Vec<WorkItem>,
        tier2_items: Vec<CachedTensorWork>,
        add_edge_items: Vec<CachedTensorWork>,
        additional_rx: Option<mpsc::Receiver<WorkItem>>,
        config: ProcessingConfig,
    ) -> Self {
        let owned = items.iter().map(|i| i.0)
            .chain(tier2_items.iter().map(|w| w.item_id))
            .chain(add_edge_items.iter().map(|w| w.item_id))
            .collect();
        Self {
            max_jobs: config.jobs,
            config,
            pending: items.into(),
            pending_tier2: tier2_items.into(),
            pending_add_edge: add_edge_items.into(),
            additional_rx,
            completed: std::collections::HashSet::new(),
            owned,
        }
    }

    /// An image admitted while the batch runs joins it.
    fn admit(&mut self, item: WorkItem) {
        self.owned.insert(item.0);
        self.pending.push_back(item);
    }

    fn complete_event(&self) -> WorkerResult {
        WorkerResult::BatchComplete { items: self.owned.iter().copied().collect() }
    }

    fn nothing_pending(&self) -> bool {
        self.pending.is_empty() && self.pending_tier2.is_empty() && self.pending_add_edge.is_empty()
    }
}

/// Work sent to the current subprocess that it hasn't reported back yet.
#[derive(Default)]
struct SentWork {
    items: Vec<WorkItem>,
    tier2_ids: Vec<u64>,
    add_edge_ids: Vec<u64>,
}

impl SentWork {
    fn len(&self) -> usize {
        self.items.len() + self.tier2_ids.len() + self.add_edge_ids.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn forget(&mut self, item_id: u64) {
        self.items.retain(|(id, _, _)| *id != item_id);
        self.tier2_ids.retain(|id| *id != item_id);
        self.add_edge_ids.retain(|id| *id != item_id);
    }

    fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.items.iter().map(|(id, _, _)| *id)
            .chain(self.tier2_ids.iter().copied())
            .chain(self.add_edge_ids.iter().copied())
    }
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
        tier1_count = state.pending.len(),
        tier2_count = state.pending_tier2.len(),
        // Snapshot of the user's requested job count — tracing fields are
        // captured at span entry. The effective count halves on each crash
        // retry; see the per-spawn span on `SubprocessManager::spawn`.
        initial_jobs = state.max_jobs,
    ),
)]
/// Runs the batch to its end and returns the worker that is still alive
/// afterwards, with the config it was built for, to stay warm for the
/// next batch; `None` after a crash, a Cancel or a failed spawn.
fn run_batch_with_retry(
    warm: Option<(SubprocessManager, ProcessingConfig)>,
    mut state: BatchRunState,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<(SubprocessManager, ProcessingConfig)> {
    // The worker in hand: the warm one, then the last spawn's; a crash
    // drops it and the next round spawns fresh.
    let mut live = warm;

    emit_loading_status(&state, res_tx, ctx);

    loop {
        if cancels.is_global_cancelled() {
            let _ = res_tx.send(WorkerResult::Cancelled);
            ctx.request_repaint();
            return live;
        }

        match poll_additional_items(&mut state) {
            WaitAction::Complete => {
                let _ = res_tx.send(state.complete_event());
                ctx.request_repaint();
                return live;
            }
            WaitAction::Sleep => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            WaitAction::Continue => {}
        }

        let Some(burst) = spawn_and_initial_burst(live.take(), &mut state, cancels, res_tx, ctx) else {
            return None; // spawn failed — errors + BatchComplete already sent
        };
        let InitialBurst { mut sub, mut sent, engines } = burst;

        let outcome = run_event_loop(&mut sub, &mut state, &mut sent, cancels, res_tx, ctx);

        match outcome {
            EventLoopOutcome::Cancelled => return None,
            EventLoopOutcome::Crashed(reason) => {
                let should_retry = handle_crash_and_retry(&mut state, sent, reason, &mut sub, res_tx, ctx);
                if !should_retry {
                    return None;
                }
            }
            EventLoopOutcome::Finished => {
                if finalize_or_continue(&mut state, res_tx, ctx) {
                    live = Some((sub, engines));
                } else {
                    return Some((sub, engines));
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
        let _ = res_tx.send(WorkerResult::Report {
            item_id: fid,
            update: prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel),
        });
        ctx.request_repaint();
    }
}

/// When the batch queues are empty, check if the admission controller has
/// more to send. Returns immediately — the caller owns the sleep/cancel cadence.
fn poll_additional_items(state: &mut BatchRunState) -> WaitAction {
    if !state.nothing_pending() {
        return WaitAction::Continue;
    }
    let Some(rx) = state.additional_rx.as_ref() else {
        return WaitAction::Complete;
    };
    match rx.try_recv() {
        Ok(item) => {
            state.admit(item);
            WaitAction::Continue
        }
        Err(mpsc::TryRecvError::Disconnected) => WaitAction::Complete,
        Err(mpsc::TryRecvError::Empty) => WaitAction::Sleep,
    }
}

/// What the initial dispatch burst sent to the freshly-spawned (or
/// pre-warmed) subprocess. Replaces a 4-tuple alias whose meaning was
/// pinned only by a comment.
struct InitialBurst {
    sub: SubprocessManager,
    sent: SentWork,
    /// What the worker's engines were built for.
    engines: ProcessingConfig,
}

/// Spawn a subprocess (or accept a warm one), give it the batch's run
/// settings, and send the initial burst (Tier 2 first — no inference
/// needed — then Tier 1 up to `max_jobs`). On spawn failure, reports
/// errors for every pending item and sends `BatchComplete`; returns `None`.
fn spawn_and_initial_burst(
    warm: Option<(SubprocessManager, ProcessingConfig)>,
    state: &mut BatchRunState,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<InitialBurst> {
    // Drop anything cancelled before spawn — they don't count toward
    // `effective_jobs` and shouldn't occupy burst slots either.
    drop_cancelled_pending(state, cancels, res_tx, ctx);
    let total_pending = state.pending.len() + state.pending_tier2.len() + state.pending_add_edge.len();
    if total_pending == 0 {
        let _ = res_tx.send(state.complete_event());
        ctx.request_repaint();
        return None;
    }
    // Cap engines at the number of Tier 1 items — Tier 2 + AddEdge don't
    // use the seg engine pool.
    let effective_jobs = state.max_jobs.min(state.pending.len().max(1));

    let (mut sub, engines) = match warm {
        Some(pair) => pair,
        None => match SubprocessManager::spawn(
            state.config.model, effective_jobs, state.config.mask, state.config.force_cpu,
            WorkerLines::app(state.config.line_mode, state.config.edge),
        ) {
            Ok((s, active_provider)) => {
                let _ = res_tx.send(WorkerResult::BackendReady(active_provider));
                ctx.request_repaint();
                (s, ProcessingConfig { jobs: effective_jobs, ..state.config.clone() })
            }
            Err(e) => {
                emit_pending_errors(state, &e, res_tx);
                let _ = res_tx.send(state.complete_event());
                ctx.request_repaint();
                return None;
            }
        },
    };
    // A dead worker fails here too; the event loop reports it as a crash.
    if let Err(err) = sub.send_configure(state.config.mask, state.config.line_mode, state.config.edge) {
        tracing::warn!(%err, "worker did not take the batch's run settings");
    }

    let mut sent = SentWork::default();
    let burst = state.max_jobs.min(total_pending);
    let mut sent_count = 0;
    while sent_count < burst {
        if try_send_tier2(&mut sub, &mut state.pending_tier2, &mut sent.tier2_ids)
            || try_send_add_edge(&mut sub, &mut state.pending_add_edge, &mut sent.add_edge_ids)
        {
            sent_count += 1;
        } else if let Some(item) = state.pending.pop_front() {
            if send_item_to_sub(&mut sub, &item).is_err() {
                state.pending.push_front(item);
                break;
            }
            sent.items.push(item);
            sent_count += 1;
        } else {
            break;
        }
    }
    Some(InitialBurst { sub, sent, engines })
}

/// Drive one subprocess through its lifecycle — poll events, admit more work
/// on ImageDone, detect crash / hang, stop on Finished or cancel.
fn run_event_loop(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    cancels: &super::processor::CancelRegistry,
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
        forward_item_cancels(sub, sent, cancels, &mut cancel_notified);

        // Real-crash check first so a segfault is reported as such rather
        // than masked by the watchdog firing on the same silence.
        if !sub.is_alive() {
            return EventLoopOutcome::Crashed(sub.crash_reason());
        }

        if is_stalled(last_event_at.elapsed(), sent.len(), HANG_TIMEOUT) {
            sub.kill();
            return EventLoopOutcome::Crashed(format!(
                "Worker stopped responding (no events for {}s)",
                HANG_TIMEOUT.as_secs(),
            ));
        }

        // Block up to 50 ms for the next event. Wakes immediately when
        // an event arrives so admission/pump latency tracks worker
        // throughput, not the old fixed 50 ms sleep tail. The watchdog
        // above runs once per iteration — well within HANG_TIMEOUT.
        for event in sub.poll_events_blocking(Duration::from_millis(50)) {
            // Every event — Progress, ImageDone, ImageError, RssUpdate
            // (emitted per-completion, not on a timer) — proves liveness.
            last_event_at = Instant::now();
            if matches!(event, SubprocessEvent::Finished) {
                subprocess_finished = true;
                continue;
            }
            handle_subprocess_event(event, sub, state, sent, cancels, res_tx, ctx);
        }

        // If everything drained and no more coming, mark finished.
        if sent.is_empty() && state.nothing_pending() {
            if let Some(rx) = state.additional_rx.as_ref() {
                match rx.try_recv() {
                    Ok(item) => state.admit(item),
                    Err(mpsc::TryRecvError::Disconnected) => subprocess_finished = true,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            } else {
                subprocess_finished = true;
            }
        }
    }
    EventLoopOutcome::Finished
}

/// Route one `SubprocessEvent` to its handler. `Finished` is handled by the
/// event loop itself (sets the exit flag); `RssUpdate` is absorbed by
/// `SubprocessManager` internally.
fn handle_subprocess_event(
    event: SubprocessEvent,
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match event {
        SubprocessEvent::Report { item_id, update } => {
            let _ = res_tx.send(WorkerResult::Report { item_id, update });
            ctx.request_repaint();
        }
        SubprocessEvent::ImageDone {
            item_id, result_path, width, height, active_provider,
            tensor_cache_path, tensor_cache_height, tensor_cache_width,
            edge_cache_path, edge_cache_height, edge_cache_width,
        } => {
            let result = read_result_image(&result_path, width, height, &active_provider);
            let tensor_cache = read_tensor_cache(
                tensor_cache_path.as_ref(), tensor_cache_height, tensor_cache_width, state.config.model,
            );
            let edge_cache = read_edge_tensor_cache(
                edge_cache_path.as_ref(), edge_cache_height, edge_cache_width,
            );

            state.completed.insert(item_id);
            sent.forget(item_id);

            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id, result, tensor_cache, edge_cache,
            });
            ctx.request_repaint();

            admit_next_item(sub, state, sent, cancels, res_tx, ctx);
        }
        SubprocessEvent::ImageError { item_id, error } => {
            state.completed.insert(item_id);
            sent.forget(item_id);
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
pub(super) fn read_and_delete(path: &std::path::Path) -> Option<Vec<u8>> {
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

/// Read the TEED multi-scale cache. Child concatenates every scale's
/// tensor into one file in `EdgeScale` order; parent splits it by each
/// scale's size.
fn read_edge_tensor_cache(
    path: Option<&std::path::PathBuf>,
    height: Option<u32>,
    width: Option<u32>,
) -> Option<EdgeTensorCache> {
    let h = height?;
    let w = width?;
    let raw_bytes = read_and_delete(path?)?;

    let scale_bytes = |scale: EdgeScale| {
        let (sh, sw) = prunr_core::edge_scale_dims(scale, h, w);
        (sh as usize) * (sw as usize) * std::mem::size_of::<f32>()
    };
    let expected: usize = EdgeScale::ALL.iter().map(|&s| scale_bytes(s)).sum();
    if raw_bytes.len() != expected {
        tracing::error!(
            got = raw_bytes.len(),
            expected,
            "edge cache file size mismatch — discarding",
        );
        return None;
    }

    let mut rest = raw_bytes.as_slice();
    let tensors = EdgeScale::ALL.map(|scale| {
        let (chunk, tail) = rest.split_at(scale_bytes(scale));
        rest = tail;
        crate::subprocess::ipc::le_bytes_to_f32s(chunk)
    });
    Some(EdgeTensorCache { tensors, height: h, width: w })
}

/// After an ImageDone, pull one more item into the subprocess if RSS allows.
/// Prefers Tier 2 (no inference), then Tier 1 pending, then the admission
/// overflow channel.
fn admit_next_item(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    drop_cancelled_pending(state, cancels, res_tx, ctx);
    if sub.should_pause_admission() {
        return;
    }
    if try_send_tier2(sub, &mut state.pending_tier2, &mut sent.tier2_ids) {
        return;
    }
    if try_send_add_edge(sub, &mut state.pending_add_edge, &mut sent.add_edge_ids) {
        return;
    }
    if let Some(item) = state.pending.pop_front() {
        if send_item_to_sub(sub, &item).is_ok() {
            sent.items.push(item);
        } else {
            state.pending.push_front(item);
        }
        return;
    }
    if let Some(rx) = state.additional_rx.as_ref() {
        if let Ok(item) = rx.try_recv() {
            if cancels.is_cancelled(item.0) {
                report_cancelled(item.0, state, res_tx, ctx);
            } else if send_item_to_sub(sub, &item).is_ok() {
                sent.items.push(item);
            } else {
                state.admit(item);
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
    sent: &SentWork,
    cancels: &super::processor::CancelRegistry,
    notified: &mut std::collections::HashSet<u64>,
) {
    for id in sent.ids() {
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
/// process is instant and orphaned IPC temps are swept by the prefix-scoped
/// cleanup — the politeness delay just made "Cancel All" feel laggy.
fn cancel_subprocess(
    sub: &mut SubprocessManager,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    sub.kill();
    crate::subprocess::protocol::cleanup_seg_pipeline_temps();
    let _ = res_tx.send(WorkerResult::Cancelled);
    ctx.request_repaint();
}

/// Post-crash: re-queue in-flight Tier 1 work, error out in-flight Tier 2
/// (tensor data already consumed by IPC), halve `max_jobs`, and either
/// continue (return `true`) or bail if we were already at 1 job (return
/// `false` after emitting terminal errors + `BatchComplete`).
fn handle_crash_and_retry(
    state: &mut BatchRunState,
    sent: SentWork,
    crash_reason: String,
    sub: &mut SubprocessManager,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> bool {
    // Tier 2 + AddEdge can't be re-queued — their tensor data is owned by the
    // IPC temp file, already cleaned up. Parent errors the item so a follow-up
    // Process runs the full pipeline.
    for tid in sent.tier2_ids.into_iter().chain(sent.add_edge_ids) {
        if !state.completed.contains(&tid) {
            let _ = res_tx.send(WorkerResult::BatchItemDone {
                item_id: tid,
                result: Err(crash_reason.clone()),
                tensor_cache: None,
                edge_cache: None,
            });
        }
    }

    let re_queued: Vec<WorkItem> = sent.items.into_iter()
        .filter(|(id, _, _)| !state.completed.contains(id))
        .collect();
    let re_count = re_queued.len();

    // Drain late additions before re-queuing so the original in-flight
    // order stays at the front.
    let late: Vec<WorkItem> = state.additional_rx.as_ref()
        .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
        .unwrap_or_default();
    for item in late {
        state.admit(item);
    }
    for item in re_queued.into_iter().rev() {
        state.pending.push_front(item);
    }

    let old_jobs = state.max_jobs;
    state.max_jobs = (state.max_jobs / 2).max(1);

    sub.kill();
    crate::subprocess::protocol::cleanup_seg_pipeline_temps();

    if old_jobs == 1 {
        let err_msg = format!("{crash_reason} \u{2014} try a smaller model");
        emit_pending_errors(state, &err_msg, res_tx);
        let _ = res_tx.send(state.complete_event());
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

/// After a subprocess finishes its sent work, decide whether the batch is
/// done or more is to come. Returns `false` when the batch is done
/// (BatchComplete already sent); `true` to loop back with the same worker.
fn finalize_or_continue(
    state: &mut BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> bool {
    if state.nothing_pending() {
        if let Some(rx) = state.additional_rx.as_ref() {
            match rx.try_recv() {
                Err(mpsc::TryRecvError::Disconnected) => {
                    let _ = res_tx.send(state.complete_event());
                    ctx.request_repaint();
                    return false;
                }
                Ok(item) => state.admit(item),
                Err(mpsc::TryRecvError::Empty) => {
                    // More might be coming — loop back.
                }
            }
        } else {
            let _ = res_tx.send(state.complete_event());
            ctx.request_repaint();
            return false;
        }
    }
    true
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
    pending_tier2: &mut VecDeque<CachedTensorWork>,
    sent_tier2_ids: &mut Vec<u64>,
) -> bool {
    if let Some(t2) = pending_tier2.pop_front() {
        let tid = t2.item_id;
        if sub.send_repostprocess(
            t2.item_id,
            &crate::subprocess::manager::TensorView {
                data: &t2.tensor_data,
                height: t2.tensor_height,
                width: t2.tensor_width,
            },
            t2.model, &t2.original_bytes, t2.mask,
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
    pending_add_edge: &mut VecDeque<CachedTensorWork>,
    sent_add_edge_ids: &mut Vec<u64>,
) -> bool {
    if let Some(ae) = pending_add_edge.pop_front() {
        let tid = ae.item_id;
        if sub.send_add_edge_inference(
            ae.item_id,
            &crate::subprocess::manager::TensorView {
                data: &ae.tensor_data,
                height: ae.tensor_height,
                width: ae.tensor_width,
            },
            ae.model, &ae.original_bytes, ae.mask,
        ).is_ok() {
            sent_add_edge_ids.push(tid);
            return true;
        }
        pending_add_edge.push_front(ae);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_worker_is_reused_for_the_same_engines_whatever_the_run_settings() {
        let built = ProcessingConfig {
            model: ModelKind::BiRefNetLite, jobs: 4, mask: MaskSettings::default(), force_cpu: false,
            line_mode: LineMode::Off, edge: super::super::item_settings::ItemSettings::default().edge_settings(),
        };
        let mut lines = built.clone();
        lines.line_mode = LineMode::SubjectOutline;
        lines.mask.gamma = 2.0;
        lines.edge.edge_scale = prunr_core::EdgeScale::Fine;
        lines.jobs = 1;
        assert!(built.reusable_by(&lines), "mask, Sketch and lines travel with the batch");
        assert!(built.reusable_by(&ProcessingConfig { jobs: 4, ..built.clone() }));
        assert!(!built.reusable_by(&ProcessingConfig { jobs: 5, ..built.clone() }), "a larger pool needs a new worker");
        assert!(!built.reusable_by(&ProcessingConfig { model: ModelKind::Silueta, ..built.clone() }));
        assert!(!built.reusable_by(&ProcessingConfig { force_cpu: true, ..built.clone() }));
    }

    #[test]
    fn a_batch_owns_what_it_was_given_and_what_it_admits_later() {
        let cfg = ProcessingConfig {
            model: ModelKind::Silueta, jobs: 2, mask: MaskSettings::default(), force_cpu: false,
            line_mode: LineMode::Off, edge: super::super::item_settings::ItemSettings::default().edge_settings(),
        };
        let bytes = Arc::new(Vec::new());
        let tier2 = CachedTensorWork {
            item_id: 3, tensor_data: Vec::new(), tensor_height: 1, tensor_width: 1,
            model: ModelKind::Silueta, original_bytes: Arc::clone(&bytes), mask: MaskSettings::default(),
        };
        let mut state = BatchRunState::new(vec![(1, Arc::clone(&bytes), None), (2, Arc::clone(&bytes), None)], vec![tier2], Vec::new(), None, cfg);
        state.admit((4, bytes, None));
        let WorkerResult::BatchComplete { mut items } = state.complete_event() else { panic!("a completion") };
        items.sort();
        assert_eq!(items, vec![1, 2, 3, 4]);
    }

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
