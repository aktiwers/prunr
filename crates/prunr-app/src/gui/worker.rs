use std::collections::VecDeque;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use prunr_core::{MaskSettings, EdgeSettings, EdgeScale, ModelKind, ProcessResult, EDGE_SCALE_COUNT};
use crate::gui::settings::LineMode;
use crate::subprocess::protocol::{EdgeCacheFile, TensorCacheFile, SpareOutcome, SubprocessEvent, CANCELLED_ERR_MSG};
use crate::subprocess::manager::{SubprocessManager, WorkerLines};
use crate::subprocess::ipc::read_and_delete;
pub(crate) use crate::subprocess::first_look::Shown;
use prunr_core::IDLE_RELEASE as WARM_IDLE;

/// Silence with work in flight before the subprocess counts as hung; real
/// batches report every few hundred ms.
const HANG_TIMEOUT: Duration = Duration::from_secs(60);

/// Graceful-shutdown budget when a warm worker is released. The bridge
/// waits it out, so a Process sent meanwhile waits too; a well-behaved
/// child exits well inside it.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);
/// How long a worker stays warm after a run of a model whose working memory
/// stays with it: BiRefNet keeps 5.5 GB on OpenVINO after its first image,
/// against 1.2 GB loaded. A minute still covers a tweak and Process again.
const WARM_IDLE_AFTER_HEAVY_RUN: Duration = Duration::from_secs(60);
/// A model counts as heavy from this registry working set up.
const HEAVY_WORKING_SET_MB: u32 = 2000;

/// The worker kept loaded between batches, and when it goes if unused.
struct WarmWorker {
    sub: SubprocessManager,
    spec: EngineSpec,
    release_at: Instant,
}

impl WarmWorker {
    fn release(mut self, reason: &'static str) {
        tracing::info!(model = ?self.spec.model, reason, "seg subprocess released");
        let _ = self.sub.shutdown_with_timeout(RELEASE_TIMEOUT);
    }
}

/// The engines a batch builds: one per image of its first burst, up to
/// the jobs it may run at once.
fn pool_jobs(jobs: usize, first_burst: usize) -> usize {
    jobs.min(first_burst.max(1))
}

/// When the worker that just ran a batch of `model` is released if unused.
fn idle_after_run(model: ModelKind) -> Duration {
    if super::memory::working_set_mb(model) >= HEAVY_WORKING_SET_MB { WARM_IDLE_AFTER_HEAVY_RUN } else { WARM_IDLE }
}

/// Alive but silent too long; the event loop treats it as a crash.
fn is_stalled(
    last_event_age: Duration,
    in_flight_count: usize,
    hang_timeout: Duration,
) -> bool {
    in_flight_count > 0 && last_event_age > hang_timeout
}

/// A full run on one image (Tier 1). Its file and start are handles,
/// read on the bridge thread when the run is sent.
#[derive(Clone)]
pub(crate) struct WorkItem {
    pub(crate) id: u64,
    /// The image's run this is (`BatchItem::job_run`).
    pub(crate) run: u64,
    pub(crate) source: super::item::ImageSource,
    pub(crate) image: Option<RunImage>,
}

/// What a sent full run reads: the file, the start it runs from (`None`:
/// the file) and a chained cut's start built for it.
struct RunInputs {
    bytes: Arc<Vec<u8>>,
    start: Option<Arc<image::RgbaImage>>,
    built: Option<Arc<image::RgbaImage>>,
}

impl WorkItem {
    fn read_inputs(&self) -> Result<RunInputs, String> {
        let bytes = self.source.load_bytes().map_err(super::item::source_unreadable)?;
        let (start, built) = match &self.image {
            Some(run) => run.resolve(self.id, || Ok(Arc::clone(&bytes)))?,
            None => (None, None),
        };
        Ok(RunInputs { bytes, start, built })
    }
}

/// What a full run reads besides its file.
#[derive(Clone)]
pub(crate) enum RunImage {
    /// An earlier image the run starts from instead of the file.
    Start(super::item::Pixels),
    /// A cut of another model's cut-out: `start`'s colours (`None`: the
    /// file's) seen through `cutout`'s transparency. Built on the bridge
    /// thread, which decodes the file anyway.
    Kept { start: Option<super::item::Pixels>, cutout: super::timeline::StoredImage },
}

/// The error an image fails with when the start it runs from is unreadable.
const START_UNREADABLE: &str = "The image to process could not be read";

/// A run's start, and for a chained cut the start built for it.
type ResolvedStart = (Option<Arc<image::RgbaImage>>, Option<Arc<image::RgbaImage>>);

impl RunImage {
    /// What the run starts from (`None`: its file) and, for a chained cut,
    /// the start built for it here; `file` is read only when that start's
    /// colours are the file's. A chained start that cannot be built runs
    /// from its plain start rather than not at all; a start that cannot
    /// be read fails the run rather than cut the file in its place.
    pub(crate) fn resolve(
        &self,
        item_id: u64,
        file: impl FnOnce() -> Result<Arc<Vec<u8>>, String>,
    ) -> Result<ResolvedStart, String> {
        let load = |start: &super::item::Pixels| start.load().ok_or_else(|| START_UNREADABLE.to_string());
        match self {
            RunImage::Start(start) => Ok((Some(load(start)?), None)),
            RunImage::Kept { start, cutout } => {
                let start = start.as_ref().map(load).transpose()?;
                Ok(match Self::kept(start.as_deref(), cutout, file) {
                    Ok(built) => (Some(Arc::clone(&built)), Some(built)),
                    Err(err) => {
                        tracing::warn!(item_id, %err, "chained cut runs from its plain start");
                        (start, None)
                    }
                })
            }
        }
    }

    /// `start`'s colours (the file's when `None`) seen through `cutout`'s
    /// transparency.
    fn kept(
        start: Option<&image::RgbaImage>,
        cutout: &super::timeline::StoredImage,
        file: impl FnOnce() -> Result<Arc<Vec<u8>>, String>,
    ) -> Result<Arc<image::RgbaImage>, String> {
        let mut image = match start {
            Some(start) => start.clone(),
            None => prunr_core::load_image_from_bytes(&file()?).map_err(|e| e.message())?.into_rgba8(),
        };
        let cutout = cutout.load().ok_or("the cut-out could not be read")?;
        if cutout.dimensions() != image.dimensions() {
            return Err(format!("the cut-out is {:?}, its start {:?}", cutout.dimensions(), image.dimensions()));
        }
        for (px, kept) in image.pixels_mut().zip(cutout.pixels()) {
            px.0[3] = kept.0[3];
        }
        Ok(Arc::new(image))
    }
}

/// A decompressed seg tensor: as the subprocess hands it over, or as
/// `CompressedTensor::bundle` unpacks it.
pub struct RawSegTensor {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

/// ~1 ms to decompress for ~3-4x less RAM. Cloning shares the bytes, so an
/// export can take it to its thread.
#[derive(Clone)]
pub struct CompressedTensor {
    compressed: std::sync::Arc<[u8]>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

impl CompressedTensor {
    /// zstd level 1, for speed. `None`: the caller skips caching.
    pub fn from_raw(tc: RawSegTensor) -> Option<Self> {
        let compressed = zstd::encode_all(
            crate::subprocess::ipc::f32s_as_le_bytes(&tc.data),
            1,
        ).ok()?;
        Some(Self { compressed: compressed.into(), height: tc.height, width: tc.width, model: tc.model })
    }

    pub fn decompress(&self) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(&*self.compressed).ok()?;
        Some(crate::subprocess::ipc::le_bytes_to_f32s(&bytes))
    }

    pub fn compressed_size(&self) -> usize {
        self.compressed.len()
    }

    /// Whether both share one stored tensor.
    pub fn same(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.compressed, &other.compressed)
    }
}

/// Raw multi-scale TEED output for IPC transfer. `tensors` is indexed by
/// [`EdgeScale`] as `usize`, a scale the run did not make left empty;
/// `height` × `width` is the base size, a scale's own size is
/// `prunr_core::edge_scale_dims`.
pub struct RawEdgeTensors {
    pub tensors: [Vec<f32>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
    /// The token of the worker's pass that makes the scales the run left
    /// for after its result.
    pub spare: Option<u64>,
}

/// Zstd-compressed multi-scale TEED cache: every scale the run made,
/// so a switch between them in live preview is a tensor lookup (one
/// decompress, no re-inference). A scale it lacks needs a new run.
/// Cloning shares the bytes.
#[derive(Clone)]
pub struct CompressedEdgeTensors {
    tensors: [Option<std::sync::Arc<[u8]>>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
    /// The pass still to fill the scales this run left for later.
    spare: Option<u64>,
}

impl CompressedEdgeTensors {
    /// `None` when the run made no scale or a compression fails.
    pub fn from_raw(raw: RawEdgeTensors) -> Option<Self> {
        use rayon::prelude::*;
        let parts: Vec<Option<std::sync::Arc<[u8]>>> = raw.tensors.par_iter().map(|t| {
            if t.is_empty() {
                return Ok(None);
            }
            zstd::encode_all(crate::subprocess::ipc::f32s_as_le_bytes(t), 1).map(|c| Some(c.into()))
        }).collect::<std::io::Result<_>>().ok()?;
        if parts.iter().all(Option::is_none) {
            return None;
        }
        let tensors = parts.try_into().ok()?;
        Some(Self { tensors, height: raw.height, width: raw.width, spare: raw.spare })
    }

    /// Whether both share `scale`'s stored tensor.
    pub fn same_scale(&self, other: &Self, scale: EdgeScale) -> bool {
        match (&self.tensors[scale as usize], &other.tensors[scale as usize]) {
            (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Whether the worker still makes the scales this run left for later.
    pub fn awaits_spare(&self) -> bool {
        self.spare.is_some()
    }

    /// Fill the scales the run left for later from `spare`, when this is
    /// still the cache that run made (`token`); `None`: they will not
    /// come. Returns whether the token matched.
    pub fn land_spare(&mut self, token: u64, spare: Option<Self>) -> bool {
        if self.spare != Some(token) {
            return false;
        }
        self.spare = None;
        if let Some(spare) = spare.filter(|s| (s.height, s.width) == (self.height, self.width)) {
            for (slot, made) in self.tensors.iter_mut().zip(spare.tensors) {
                if slot.is_none() {
                    *slot = made;
                }
            }
        }
        true
    }

    /// Whether the cache holds every one of `scales`.
    pub fn has_all(&self, scales: &[EdgeScale]) -> bool {
        scales.iter().all(|&s| self.tensors[s as usize].is_some())
    }

    /// `None` when the cache lacks it. Hot path: callers keep the result
    /// as `Arc<Vec<f32>>` keyed by scale to amortise across a drag.
    pub fn decompress(&self, scale: EdgeScale) -> Option<Vec<f32>> {
        let bytes = zstd::decode_all(&**self.tensors[scale as usize].as_ref()?).ok()?;
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

    pub fn compressed_size(&self) -> usize {
        self.tensors.iter().flatten().map(|t| t.len()).sum()
    }
}

pub type SegBundle = RawSegTensor;

impl CompressedTensor {
    /// Decompress into a bundle. `None` when the stored blob fails to decode.
    pub fn bundle(&self) -> Option<RawSegTensor> {
        Some(RawSegTensor {
            data: self.decompress()?,
            height: self.height,
            width: self.width,
            model: self.model,
        })
    }
}

/// Work that starts from an item's cached seg tensor instead of a seg run:
/// a mask re-postprocess, or edge detection added on top of the mask.
pub(crate) struct CachedTensorWork {
    pub(crate) item_id: u64,
    /// Unpacked, with the file read, on the bridge thread.
    pub(crate) tensor: CompressedTensor,
    pub(crate) source: super::item::ImageSource,
    pub(crate) mask: MaskSettings,
}


/// What a worker's engines are built for: the model, on which processor,
/// and the jobs they serve. A warm worker is kept and reused by this alone.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EngineSpec {
    pub model: ModelKind,
    pub jobs: usize,
    pub cpu_only: bool,
}

impl EngineSpec {
    /// Whether a worker built for `self` can run `next` on `provider`: the
    /// same model on the same processor, with a pool at least as large (two
    /// job counts can make the same pool: BiRefNet on the CPU builds one
    /// either way).
    pub fn reusable_by(&self, next: &EngineSpec, provider: &str) -> bool {
        let engines = |e: &EngineSpec| prunr_core::batch::engine_pool_size(e.model, e.jobs, e.cpu_only, provider);
        self.model == next.model && self.cpu_only == next.cpu_only && engines(self) >= engines(next)
    }
}

/// What a batch runs with besides its engines; the worker takes it with
/// `Configure`, so any warm worker can run any of these.
#[derive(Clone, PartialEq)]
pub struct RunSettings {
    pub mask: MaskSettings,
    pub line_mode: LineMode,
    pub edge: EdgeSettings,
}

impl RunSettings {
    /// The settings a worker holds between batches: each batch sends its own.
    fn neutral() -> Self {
        Self { mask: MaskSettings::default(), line_mode: LineMode::Off, edge: EdgeSettings::default() }
    }
}

/// A batch: the engines it needs and the settings it runs with.
#[derive(Clone)]
pub struct ProcessingConfig {
    pub engines: EngineSpec,
    pub run: RunSettings,
    /// The image on screen, when its run would show a first look; the
    /// worker learns of each change while the batch runs.
    pub first_look: Shown,
    /// The image on screen: its Sketch runs make the line scales they do
    /// not draw after their result. Kept current like `first_look`.
    pub on_screen: Shown,
}

pub(crate) enum WorkerMessage {
    BatchProcess {
        /// The id its completion names (`Processor::track_dispatch`).
        batch: u64,
        items: Vec<WorkItem>,
        tier2_items: Vec<CachedTensorWork>,
        /// Cached seg tensor + TEED on the masked image, so turning the
        /// outline on needs no seg re-run.
        add_edge_items: Vec<CachedTensorWork>,
        config: Box<ProcessingConfig>,
        cancels: super::processor::CancelRegistry,
        additional_items_rx: Option<mpsc::Receiver<WorkItem>>,
    },
    /// Drop any warm seg subprocess (BiRefNetLite holds ~2 GB). Idempotent.
    ReleaseWarm,
    /// Load this config's model now, replacing a warm worker that does
    /// not fit it: a switch to a cut-out model, so its first Process
    /// skips the load.
    Warm(EngineSpec),
    /// First looks went off (Settings › Animations, most often): the warm
    /// worker lets its first-look model go.
    FirstLookOff,
}

/// What reaches the app of a run's spare line scales.
pub enum SpareLanding {
    Made(CompressedEdgeTensors),
    /// Not made on purpose (a cancel, a shutdown, the image left the
    /// screen): nothing runs in their place.
    Skipped,
    /// The worker failed to make them, or died first.
    Failed,
}

pub enum WorkerResult {
    Report {
        item_id: u64,
        update: prunr_core::ProgressUpdate,
        /// When the bridge took it from the worker: the app may read it
        /// frames later.
        at: Instant,
    },
    BatchItemDone {
        /// The batch whose run this answers: a cancelled run's late
        /// answer names its own batch, not the image's next run.
        batch: u64,
        item_id: u64,
        result: Result<ProcessResult, String>,
        /// For later mask-only reruns; compressed on the bridge thread.
        seg: Option<CompressedTensor>,
        /// Only when the run used edge detection.
        edges: Option<CompressedEdgeTensors>,
    },
    /// The line scales a run left for after its result, under the token
    /// its `BatchItemDone`'s edge cache named.
    SpareEdgeScales { item_id: u64, token: u64, landing: SpareLanding },
    /// The image a chained cut starts from, built as it was sent.
    ChainStart { item_id: u64, run: u64, image: Arc<image::RgbaImage> },
    /// Batch `batch` is over.
    BatchComplete { batch: u64 },
    Cancelled,
    SubprocessRetry {
        reduced_jobs: usize,
        re_queued_count: usize,
    },
    /// The EP the subprocess actually built, once per spawn: corrects the
    /// startup guess (e.g. CUDA driver installed but EP init fell back).
    BackendReady(String),
}

/// The worker bridge thread: UI messages to subprocess IPC, with crash
/// retry. A batch reuses the warm worker when it fits; the worker that ran
/// a batch stays warm after it so the next Process skips the model load.
/// A model switch, a crash or Cancel drops it.
pub(crate) fn spawn_worker(
    ctx: egui::Context,
    prewarm: Option<EngineSpec>,
) -> (mpsc::Sender<WorkerMessage>, mpsc::Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>();

    std::thread::Builder::new()
        .name("prunr-bridge".into())
        .spawn(move || {
            // Blocks this thread for the model load; a Process sent
            // meanwhile queues on msg_rx.
            let mut warm: Option<WarmWorker> = prewarm.and_then(|spec| spawn_warm(spec, &res_tx, &ctx));

            loop {
                let wait = warm.as_ref().map_or(WARM_IDLE, |w| w.release_at.saturating_duration_since(Instant::now()));
                let msg = match msg_rx.recv_timeout(wait) {
                    Ok(msg) => msg,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Some(w) = warm.take_if(|w| Instant::now() >= w.release_at) {
                            w.release("idle");
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                match msg {
                    WorkerMessage::ReleaseWarm => {
                        if let Some(w) = warm.take() {
                            w.release("model switch");
                        }
                    }
                    WorkerMessage::FirstLookOff => {
                        if let Some(w) = warm.as_mut() {
                            let neutral = RunSettings::neutral();
                            let _ = w.sub.send_configure(neutral.mask, neutral.line_mode, neutral.edge, None, None);
                        }
                    }
                    WorkerMessage::Warm(spec) => {
                        if warm.as_ref().is_some_and(|w| w.spec.reusable_by(&spec, &prunr_core::OrtEngine::detect_active_provider())) {
                            continue;
                        }
                        if let Some(w) = warm.take() {
                            w.release("another model picked");
                        }
                        warm = spawn_warm(spec, &res_tx, &ctx);
                    }
                    WorkerMessage::BatchProcess {
                        batch, items, tier2_items, add_edge_items, config, cancels,
                        additional_items_rx,
                    } => {
                        let needed = EngineSpec { jobs: pool_jobs(config.engines.jobs, items.len()), ..config.engines };
                        let mut reuse: Option<(SubprocessManager, EngineSpec)> = None;
                        let respawn_reason = match warm.take() {
                            None => Some("no_warm_worker"),
                            Some(mut w) => {
                                if !w.sub.is_alive() {
                                    Some("warm_worker_dead")
                                } else if w.spec.reusable_by(&needed, &prunr_core::OrtEngine::detect_active_provider()) {
                                    reuse = Some((w.sub, w.spec));
                                    None
                                } else {
                                    w.release("engines differ");
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
                            model = ?config.engines.model,
                            line_mode = ?config.run.line_mode,
                            "subprocess cold-start",
                        );
                        let state = BatchRunState::new(
                            batch, items, tier2_items, add_edge_items, additional_items_rx, *config,
                        );
                        warm = run_batch_with_retry(reuse, state, &cancels, &res_tx, &ctx)
                            .map(|(sub, spec)| WarmWorker { release_at: Instant::now() + idle_after_run(spec.model), sub, spec });
                    }
                }
            }
        })
        .expect("failed to spawn bridge thread");

    (msg_tx, res_rx)
}

/// Batch-wide state carried across subprocess spawns. `max_jobs` halves on
/// each crash retry; `completed` keeps delivered items out of a re-queue.
struct BatchRunState {
    config: ProcessingConfig,
    pending: VecDeque<WorkItem>,
    pending_tier2: VecDeque<CachedTensorWork>,
    pending_add_edge: VecDeque<CachedTensorWork>,
    additional_rx: Option<mpsc::Receiver<WorkItem>>,
    completed: std::collections::HashSet<u64>,
    batch: u64,
    max_jobs: usize,
    /// What the worker was last told of `config.first_look` and
    /// `config.on_screen`.
    told_first_look: Option<u64>,
    told_on_screen: Option<u64>,
}

impl BatchRunState {
    fn new(
        batch: u64,
        items: Vec<WorkItem>,
        tier2_items: Vec<CachedTensorWork>,
        add_edge_items: Vec<CachedTensorWork>,
        additional_rx: Option<mpsc::Receiver<WorkItem>>,
        config: ProcessingConfig,
    ) -> Self {
        Self {
            max_jobs: config.engines.jobs,
            config,
            pending: items.into(),
            pending_tier2: tier2_items.into(),
            pending_add_edge: add_edge_items.into(),
            additional_rx,
            completed: std::collections::HashSet::new(),
            batch,
            told_first_look: None,
            told_on_screen: None,
        }
    }

    /// An image admitted while the batch runs joins it.
    fn admit(&mut self, item: WorkItem) {
        self.pending.push_back(item);
    }

    fn complete_event(&self) -> WorkerResult {
        WorkerResult::BatchComplete { batch: self.batch }
    }

    fn item_failed(&self, item_id: u64, error: String) -> WorkerResult {
        WorkerResult::BatchItemDone {
            batch: self.batch,
            item_id,
            result: Err(error),
            seg: None,
            edges: None,
        }
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
    /// Images whose result landed and whose spare line scales, under
    /// this token, are still to come. They keep the batch open (the
    /// worker is busy with them) but are not in `ids`: a cancel for the
    /// image is meant for a later run, and reaching the worker now it
    /// would linger and cancel that run as it starts.
    spares: Vec<(u64, u64)>,
}

impl SentWork {
    fn len(&self) -> usize {
        self.items.len() + self.tier2_ids.len() + self.add_edge_ids.len() + self.spares.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Work the worker runs now; owed spare scales hold no job slot.
    fn working(&self) -> usize {
        self.len() - self.spares.len()
    }

    /// Only spare scales are owed: losing the worker now loses no image,
    /// so it is no crash to retry or to run fewer jobs after.
    fn only_spares(&self) -> bool {
        !self.spares.is_empty() && self.len() == self.spares.len()
    }

    fn forget(&mut self, item_id: u64) {
        self.items.retain(|w| w.id != item_id);
        self.tier2_ids.retain(|id| *id != item_id);
        self.add_edge_ids.retain(|id| *id != item_id);
    }

    fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.items.iter().map(|w| w.id)
            .chain(self.tier2_ids.iter().copied())
            .chain(self.add_edge_ids.iter().copied())
    }

    /// The worker will not send the spare scales still owed: the app
    /// hears they will not come, `failed` (a crash) or skipped (a
    /// cancel), so a scale switch waiting for them stops waiting.
    fn abandon_spares(&mut self, failed: bool, res_tx: &mpsc::Sender<WorkerResult>) {
        for (item_id, token) in self.spares.drain(..) {
            let landing = if failed { SpareLanding::Failed } else { SpareLanding::Skipped };
            let _ = res_tx.send(WorkerResult::SpareEdgeScales { item_id, token, landing });
        }
    }
}

/// `Cancelled`: the cancel is already fully handled; the caller returns.
enum EventLoopOutcome {
    Finished,
    Crashed(crate::subprocess::manager::Crash),
    Cancelled,
}

/// What `poll_additional_items` found when the queues ran dry.
enum WaitAction {
    Continue, // more work in queues or just pushed
    Complete, // producer disconnected → batch done
    Sleep,    // producer still connected, nothing available yet
}

/// Retries a crash with jobs halved down to 1.
#[tracing::instrument(
    skip_all,
    fields(
        tier1_count = state.pending.len(),
        tier2_count = state.pending_tier2.len(),
        // Captured at span entry: the requested count, before any halving.
        initial_jobs = state.max_jobs,
    ),
)]
/// Runs the batch to its end and returns the worker that is still alive
/// afterwards, with the config it was built for, to stay warm for the
/// next batch; `None` after a crash, a Cancel or a failed spawn.
fn run_batch_with_retry(
    warm: Option<(SubprocessManager, EngineSpec)>,
    mut state: BatchRunState,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<(SubprocessManager, EngineSpec)> {
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
            EventLoopOutcome::Crashed(reason) if sent.only_spares() => {
                tracing::warn!(%reason, "worker lost while making spare line scales; no image to retry");
                sent.abandon_spares(true, res_tx);
                sub.kill();
                crate::subprocess::protocol::cleanup_seg_pipeline_temps();
            }
            EventLoopOutcome::Crashed(reason) => {
                handle_crash_and_retry(&mut state, sent, reason, &mut sub, res_tx, ctx);
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

fn emit_loading_status(
    state: &BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let first_id = state.pending.front().map(|w| w.id)
        .or_else(|| state.pending_tier2.front().map(|t| t.item_id))
        .or_else(|| state.pending_add_edge.front().map(|a| a.item_id));
    if let Some(fid) = first_id {
        let _ = res_tx.send(WorkerResult::Report {
            item_id: fid,
            update: prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel),
            at: Instant::now(),
        });
        ctx.request_repaint();
    }
}

/// Never blocks: the caller owns the sleep/cancel cadence.
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
/// pre-warmed) subprocess.
struct InitialBurst {
    sub: SubprocessManager,
    sent: SentWork,
    /// What the worker's engines were built for.
    engines: EngineSpec,
}

/// On spawn failure every pending item errors and the batch completes;
/// returns `None`.
fn spawn_and_initial_burst(
    warm: Option<(SubprocessManager, EngineSpec)>,
    state: &mut BatchRunState,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<InitialBurst> {
    drop_cancelled_pending(state, |id| cancels.is_cancelled(id), res_tx, ctx);
    let total_pending = state.pending.len() + state.pending_tier2.len() + state.pending_add_edge.len();
    if total_pending == 0 {
        let _ = res_tx.send(state.complete_event());
        ctx.request_repaint();
        return None;
    }
    // Cached-tensor work doesn't use the seg engine pool.
    let effective_jobs = pool_jobs(state.max_jobs, state.pending.len());

    let (mut sub, engines) = match warm {
        Some(pair) => pair,
        None => match SubprocessManager::spawn(
            state.config.engines.model, effective_jobs, state.config.run.mask, state.config.engines.cpu_only,
            WorkerLines::app(state.config.run.line_mode, state.config.run.edge),
        ) {
            Ok((s, active_provider)) => {
                let _ = res_tx.send(WorkerResult::BackendReady(active_provider));
                ctx.request_repaint();
                (s, EngineSpec { jobs: effective_jobs, ..state.config.engines })
            }
            Err(e) => {
                emit_pending_errors(state, &e, res_tx);
                let _ = res_tx.send(state.complete_event());
                ctx.request_repaint();
                return None;
            }
        },
    };
    configure(&mut sub, state);

    let mut sent = SentWork::default();
    top_up(&mut sub, state, &mut sent, cancels, res_tx, ctx);
    Some(InitialBurst { sub, sent, engines })
}

/// Give the worker the batch's run settings and the image on screen whose
/// run shows a first look. A dead worker fails here too; the event loop
/// reports it as a crash.
fn configure(sub: &mut SubprocessManager, state: &mut BatchRunState) {
    let run = &state.config.run;
    let shown = state.config.first_look.get();
    let on_screen = state.config.on_screen.get();
    if let Err(err) = sub.send_configure(run.mask, run.line_mode, run.edge, shown, on_screen) {
        tracing::warn!(%err, "worker did not take the batch's run settings");
    }
    state.told_first_look = shown;
    state.told_on_screen = on_screen;
}

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
    let mut cancel_notified: std::collections::HashSet<u64> = std::collections::HashSet::new();

    while !subprocess_finished {
        if cancels.is_global_cancelled() {
            sent.abandon_spares(false, res_tx);
            cancel_subprocess(sub, res_tx, ctx);
            return EventLoopOutcome::Cancelled;
        }

        forward_item_cancels(sub, sent, cancels, &mut cancel_notified);
        if state.config.first_look.get() != state.told_first_look || state.config.on_screen.get() != state.told_on_screen {
            configure(sub, state);
        }

        // Real-crash check first so a segfault is reported as such rather
        // than masked by the watchdog firing on the same silence.
        if let Some(reason) = sub.died() {
            return EventLoopOutcome::Crashed(reason);
        }

        if is_stalled(last_event_at.elapsed(), sent.len(), HANG_TIMEOUT) {
            sub.kill();
            tracing::warn!(silent_secs = HANG_TIMEOUT.as_secs(), "worker sent nothing; killed");
            return EventLoopOutcome::Crashed(crate::subprocess::manager::Crash::Unresponsive);
        }

        for event in sub.poll_events_blocking(Duration::from_millis(50)) {
            // Any event proves liveness.
            last_event_at = Instant::now();
            if matches!(event, SubprocessEvent::Finished) {
                subprocess_finished = true;
                continue;
            }
            handle_subprocess_event(event, state, sent, res_tx, ctx);
        }
        top_up(sub, state, sent, cancels, res_tx, ctx);

        if sent.is_empty() && matches!(poll_additional_items(state), WaitAction::Complete) {
            subprocess_finished = true;
        }
    }
    EventLoopOutcome::Finished
}

/// `Finished` is the event loop's own; `SubprocessManager` absorbs
/// `RssUpdate`.
fn handle_subprocess_event(
    event: SubprocessEvent,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match event {
        SubprocessEvent::Report { item_id, update } => {
            let _ = res_tx.send(WorkerResult::Report { item_id, update, at: Instant::now() });
            ctx.request_repaint();
        }
        SubprocessEvent::ImageDone {
            item_id, result_path, width, height, active_provider,
            tensor_cache, edge_cache: edge_cache_file, spare_scales,
        } => {
            let result = read_result_image(&result_path, width, height, &active_provider);
            let seg = read_raw_seg_tensor(tensor_cache.as_ref(), state.config.engines.model)
                .and_then(CompressedTensor::from_raw);
            let edges = read_raw_edge_tensors(edge_cache_file.as_ref())
                .and_then(|raw| CompressedEdgeTensors::from_raw(RawEdgeTensors { spare: spare_scales, ..raw }));

            state.completed.insert(item_id);
            sent.forget(item_id);
            if let Some(token) = spare_scales {
                sent.spares.push((item_id, token));
            }

            let _ = res_tx.send(WorkerResult::BatchItemDone {
                batch: state.batch,
                item_id, result, seg, edges,
            });
            ctx.request_repaint();
        }
        SubprocessEvent::ImageError { item_id, error } => {
            state.completed.insert(item_id);
            sent.forget(item_id);
            let _ = res_tx.send(state.item_failed(item_id, error));
            ctx.request_repaint();
        }
        SubprocessEvent::SpareEdgeScales { item_id, token, outcome } => {
            sent.spares.retain(|&owed| owed != (item_id, token));
            let landing = match outcome {
                SpareOutcome::Made(file) => read_raw_edge_tensors(Some(&file))
                    .and_then(CompressedEdgeTensors::from_raw)
                    .map_or(SpareLanding::Failed, SpareLanding::Made),
                SpareOutcome::Skipped => SpareLanding::Skipped,
                SpareOutcome::Failed => SpareLanding::Failed,
            };
            let _ = res_tx.send(WorkerResult::SpareEdgeScales { item_id, token, landing });
            ctx.request_repaint();
        }
        SubprocessEvent::Ready { .. }
        | SubprocessEvent::InpaintDone { .. }
        | SubprocessEvent::InpaintError { .. }
        | SubprocessEvent::RssUpdate { .. }
        | SubprocessEvent::Finished
        | SubprocessEvent::InitError { .. } => {}
    }
}

fn read_result_image(
    result_path: &std::path::Path,
    width: u32,
    height: u32,
    active_provider: &str,
) -> Result<ProcessResult, String> {
    crate::subprocess::ipc::read_result_rgba(result_path, width, height)
        .map(|rgba_image| ProcessResult {
            rgba_image,
            active_provider: active_provider.to_string(),
        })
        .ok_or_else(|| "The model's result could not be read. Process again.".to_string())
}

fn read_raw_seg_tensor(file: Option<&TensorCacheFile>, model: ModelKind) -> Option<RawSegTensor> {
    let TensorCacheFile { path: p, height: h, width: w } = file?;
    let (h, w) = (*h, *w);
    let bytes = read_and_delete(p)?;
    let data = crate::subprocess::ipc::le_bytes_to_f32s(&bytes);
    let expected = (h as usize) * (w as usize);
    if data.len() != expected {
        tracing::error!(
            path = %p.display(),
            got = data.len(), expected,
            "seg tensor length mismatch — mask will be garbage",
        );
    }
    Some(RawSegTensor { data, height: h, width: w, model })
}

/// Read the TEED planes the worker wrote, split by each listed scale's
/// size; the scales it did not write stay empty.
fn read_raw_edge_tensors(file: Option<&EdgeCacheFile>) -> Option<RawEdgeTensors> {
    let file = file?;
    let raw_bytes = read_and_delete(&file.path)?;

    let scale_bytes = |scale: EdgeScale| {
        let (sh, sw) = prunr_core::edge_scale_dims(scale, file.height, file.width);
        (sh as usize) * (sw as usize) * std::mem::size_of::<f32>()
    };
    let expected: usize = file.scales.iter().map(|&s| scale_bytes(s)).sum();
    if raw_bytes.len() != expected {
        tracing::error!(
            got = raw_bytes.len(),
            expected,
            "edge cache file size mismatch — discarding",
        );
        return None;
    }

    let mut tensors: [Vec<f32>; EDGE_SCALE_COUNT] = Default::default();
    let mut rest = raw_bytes.as_slice();
    for &scale in &file.scales {
        let (chunk, tail) = rest.split_at(scale_bytes(scale));
        rest = tail;
        tensors[scale as usize] = crate::subprocess::ipc::le_bytes_to_f32s(chunk);
    }
    Some(RawEdgeTensors { tensors, height: file.height, width: file.width, spare: None })
}

/// How much more work to send now. Up to the job limit; the memory pause
/// holds new work back only while some is running, as the worker reports
/// its memory when work ends: with none running the pause would never lift.
fn room_to_send(working: usize, max_jobs: usize, paused: bool) -> usize {
    if paused && working > 0 {
        return 0;
    }
    max_jobs.saturating_sub(working)
}

fn top_up(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    drop_cancelled_pending(state, |id| cancels.is_cancelled(id), res_tx, ctx);
    while room_to_send(sent.working(), state.max_jobs, sub.should_pause_admission()) > 0 {
        match send_next(sub, state, sent, cancels, res_tx, ctx) {
            SendOutcome::Sent | SendOutcome::Failed => {}
            SendOutcome::Held | SendOutcome::Nothing => break,
        }
    }
}

enum SendOutcome {
    Sent,
    /// The next work could not be read or handed over: it failed and went
    /// no further.
    Failed,
    /// The worker refused it; it stays at the front to retry.
    Held,
    Nothing,
}

/// Send the next pending work to the worker: cached-tensor work first,
/// then full runs, then an image the memory controller admitted since
/// (unless cancelled). Its inputs are read here, off the UI thread.
fn send_next(
    sub: &mut SubprocessManager,
    state: &mut BatchRunState,
    sent: &mut SentWork,
    cancels: &super::processor::CancelRegistry,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> SendOutcome {
    let cached = [
        (Lane::Tier2, SubprocessManager::send_repostprocess as SendCached),
        (Lane::AddEdge, SubprocessManager::send_add_edge_inference),
    ];
    for (lane, send) in cached {
        let (pending, ids) = match lane {
            Lane::Tier2 => (&mut state.pending_tier2, &mut sent.tier2_ids),
            Lane::AddEdge => (&mut state.pending_add_edge, &mut sent.add_edge_ids),
        };
        let Some(work) = pending.pop_front() else { continue };
        return match send_cached(sub, &work, send).map_err(|refusal| failure(refusal, sub)) {
            Ok(()) => {
                ids.push(work.item_id);
                SendOutcome::Sent
            }
            Err(Some(err)) => {
                report_failed(work.item_id, err, state, res_tx, ctx);
                SendOutcome::Failed
            }
            Err(None) => {
                pending.push_front(work);
                SendOutcome::Held
            }
        };
    }
    let (item, from_pending) = match state.pending.pop_front() {
        Some(item) => (item, true),
        None => match state.additional_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            Some(item) if cancels.is_cancelled(item.id) => {
                report_cancelled(item.id, state, res_tx, ctx);
                return SendOutcome::Nothing;
            }
            Some(item) => (item, false),
            None => return SendOutcome::Nothing,
        },
    };
    match send_item_to_sub(sub, &item, res_tx).map_err(|refusal| failure(refusal, sub)) {
        Ok(()) => {
            sent.items.push(item);
            SendOutcome::Sent
        }
        Err(Some(err)) => {
            report_failed(item.id, err, state, res_tx, ctx);
            SendOutcome::Failed
        }
        Err(None) => {
            if from_pending {
                state.pending.push_front(item);
            } else {
                state.admit(item);
            }
            SendOutcome::Held
        }
    }
}

#[derive(Clone, Copy)]
enum Lane {
    Tier2,
    AddEdge,
}

/// Why refused work fails (`Some`), or `None` to hold it: a live worker
/// that cannot take it (a full temp disk) would refuse it again forever,
/// while work a dead one refused is the crash path's to run again.
fn failure(refusal: Refused, sub: &mut SubprocessManager) -> Option<String> {
    match refusal {
        Refused::Unreadable(err) => Some(err),
        Refused::Worker(err) => sub.is_alive().then_some(err),
    }
}

/// Why work did not reach the worker.
enum Refused {
    Unreadable(String),
    /// Handing it over failed (a temp file, the pipe), with why.
    Worker(String),
}

fn drop_cancelled_pending(
    state: &mut BatchRunState,
    is_cancelled: impl Fn(u64) -> bool,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    // One look per image: a cancel landing mid-scan must not take an image
    // off the queue without its report.
    let mut cancelled: Vec<u64> = Vec::new();
    let mut keep = |id: u64| !is_cancelled(id) || { cancelled.push(id); false };
    state.pending.retain(|item| keep(item.id));
    state.pending_tier2.retain(|w| keep(w.item_id));
    state.pending_add_edge.retain(|w| keep(w.item_id));
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
    report_failed(item_id, CANCELLED_ERR_MSG.to_string(), state, res_tx, ctx);
}

fn report_failed(
    item_id: u64,
    error: String,
    state: &mut BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    state.completed.insert(item_id);
    let _ = res_tx.send(state.item_failed(item_id, error));
    ctx.request_repaint();
}

/// Kill the child at once: orphaned IPC temps are swept by the
/// prefix-scoped cleanup, and a polite shutdown made Cancel feel laggy.
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

/// Post-crash: error out in-flight Tier 2 (tensor data already consumed by
/// IPC), and either re-queue in-flight Tier 1 work with half the jobs or,
/// at one job, fail the image that crashed it. The batch goes on.
fn handle_crash_and_retry(
    state: &mut BatchRunState,
    mut sent: SentWork,
    crash_reason: crate::subprocess::manager::Crash,
    sub: &mut SubprocessManager,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    // Cached-tensor work can't be re-queued: its tensor's IPC temp file is
    // gone. Erroring it makes the next Process run the full pipeline.
    sent.abandon_spares(true, res_tx);
    for tid in sent.tier2_ids.into_iter().chain(sent.add_edge_ids) {
        if !state.completed.contains(&tid) {
            let _ = res_tx.send(state.item_failed(tid, crash_message(&crash_reason).to_string()));
        }
    }

    let re_queued: Vec<WorkItem> = sent.items.into_iter()
        .filter(|item| !state.completed.contains(&item.id))
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

    sub.kill();
    crate::subprocess::protocol::cleanup_seg_pipeline_temps();

    match on_crash(state.max_jobs) {
        OnCrash::FailInFlight => {
            tracing::warn!(reason = %crash_reason, "a worker crash failed the image it ran");
            let error = crash_message(&crash_reason);
            for item in re_queued {
                report_failed(item.id, error.to_string(), state, res_tx, ctx);
            }
        }
        OnCrash::RetryWith(jobs) => {
            for item in re_queued.into_iter().rev() {
                state.pending.push_front(item);
            }
            state.max_jobs = jobs;
            let _ = res_tx.send(WorkerResult::SubprocessRetry {
                reduced_jobs: jobs,
                re_queued_count: re_count,
            });
            ctx.request_repaint();
        }
    }
}

/// What a worker crash does to the images it was running.
#[derive(Debug, PartialEq)]
enum OnCrash {
    /// Run them again with fewer jobs: one of them, or running them
    /// together, crashed it.
    RetryWith(usize),
    /// At one job the image it ran is the cause: it fails, the batch
    /// goes on with a new worker.
    FailInFlight,
}

fn on_crash(jobs: usize) -> OnCrash {
    if jobs <= 1 { OnCrash::FailInFlight } else { OnCrash::RetryWith(jobs / 2) }
}

/// What an image a crash failed tells the user; the detail (a signal, an
/// exit code) goes to the log.
fn crash_message(crash: &crate::subprocess::manager::Crash) -> &'static str {
    use crate::subprocess::manager::Crash;
    match crash {
        Crash::OutOfMemory => "Ran out of memory. Try a smaller model or fewer images at once.",
        Crash::Unresponsive => "The model stopped responding. Process again.",
        Crash::Other(_) => "The model stopped unexpectedly. Process again; if it happens again, run prunr --doctor and send the report.",
    }
}

fn emit_pending_errors(
    state: &BatchRunState,
    err_msg: &str,
    res_tx: &mpsc::Sender<WorkerResult>,
) {
    for item in &state.pending {
        if !state.completed.contains(&item.id) {
            let _ = res_tx.send(state.item_failed(item.id, err_msg.to_string()));
        }
    }
    // Cached-tensor work is errored whether or not it completed.
    for work in state.pending_tier2.iter().chain(&state.pending_add_edge) {
        let _ = res_tx.send(state.item_failed(work.item_id, err_msg.to_string()));
    }
}

/// `false`: the batch is done (BatchComplete sent); `true`: loop back with
/// the same worker.
fn finalize_or_continue(
    state: &mut BatchRunState,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> bool {
    match poll_additional_items(state) {
        WaitAction::Complete => {
            let _ = res_tx.send(state.complete_event());
            ctx.request_repaint();
            false
        }
        WaitAction::Continue | WaitAction::Sleep => true,
    }
}

/// Boot a worker with `spec`'s model before any Process asks for it: at
/// start-up and on a switch to a cut-out model. Blocks the bridge thread
/// for the load; a Process sent meanwhile waits for it, as it would for
/// its own.
fn spawn_warm(
    spec: EngineSpec,
    res_tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) -> Option<WarmWorker> {
    let started = Instant::now();
    let neutral = RunSettings::neutral();
    let lines = WorkerLines::app(neutral.line_mode, neutral.edge);
    match SubprocessManager::spawn(spec.model, spec.jobs, neutral.mask, spec.cpu_only, lines) {
        Ok((sub, provider)) => {
            tracing::info!(stage = "prewarm", elapsed_ms = started.elapsed().as_millis(), model = ?spec.model, "subprocess cold-start");
            let _ = res_tx.send(WorkerResult::BackendReady(provider));
            ctx.request_repaint();
            Some(WarmWorker { sub, spec, release_at: Instant::now() + WARM_IDLE })
        }
        Err(err) => {
            tracing::warn!(%err, "pre-warm subprocess failed, falling back to per-batch spawn");
            None
        }
    }
}

/// A chained cut's start is built here and reported back.
fn send_item_to_sub(sub: &mut SubprocessManager, item: &WorkItem, res_tx: &mpsc::Sender<WorkerResult>) -> Result<(), Refused> {
    let RunInputs { bytes, start, built } = item.read_inputs().map_err(Refused::Unreadable)?;
    if let Some(image) = built {
        let _ = res_tx.send(WorkerResult::ChainStart { item_id: item.id, run: item.run, image });
    }
    let chain_input = start.as_deref().map(|rgba| (rgba, rgba.width(), rgba.height()));
    sub.send_image(item.id, &bytes, chain_input).map_err(Refused::Worker)
}

type SendCached = fn(&mut SubprocessManager, u64, &crate::subprocess::manager::TensorView<'_>, ModelKind, &[u8], MaskSettings) -> Result<(), String>;

impl CachedTensorWork {
    /// The unpacked tensor and the file. A corrupt cache or an unreadable
    /// file fails the image; its failure drops the cache, so the next
    /// Process starts over.
    fn read_inputs(&self) -> Result<(Vec<f32>, Arc<Vec<u8>>), String> {
        let data = self.tensor.decompress().ok_or("The kept cut-out could not be read. Process again.")?;
        let bytes = self.source.load_bytes().map_err(super::item::source_unreadable)?;
        Ok((data, bytes))
    }
}

fn send_cached(sub: &mut SubprocessManager, work: &CachedTensorWork, send: SendCached) -> Result<(), Refused> {
    let (data, bytes) = work.read_inputs().map_err(Refused::Unreadable)?;
    let tensor = crate::subprocess::manager::TensorView {
        data: &data,
        height: work.tensor.height,
        width: work.tensor.width,
    };
    send(sub, work.item_id, &tensor, work.tensor.model, &bytes, work.mask).map_err(Refused::Worker)
}

#[cfg(test)]
mod tests {

    /// A heavy model's worker lets go a minute after its run, a light
    /// one after five: BiRefNet keeps gigabytes of working memory loaded.
    #[test]
    fn a_heavy_models_worker_is_released_sooner_after_a_run() {
        use super::{idle_after_run, WARM_IDLE, WARM_IDLE_AFTER_HEAVY_RUN};
        assert_eq!(idle_after_run(ModelKind::BiRefNetLite), WARM_IDLE_AFTER_HEAVY_RUN);
        assert_eq!(idle_after_run(ModelKind::Silueta), WARM_IDLE);
        assert_eq!(idle_after_run(ModelKind::U2net), WARM_IDLE);
    }

    /// Owed spare scales keep the batch open but take no cancels: their
    /// run is over, and a cancel forwarded for the image would be one
    /// meant for its next run. A worker that will not send them (cancel,
    /// crash) tells the app they will not come, each under its own token.
    #[test]
    fn owed_spare_scales_hold_the_batch_and_are_abandoned_with_the_worker() {
        use super::SentWork;
        let mut sent = SentWork::default();
        sent.spares.push((4, 40));
        sent.spares.push((5, 50));
        assert_eq!(sent.len(), 2);
        assert!(sent.only_spares(), "a worker lost now loses no image");
        sent.tier2_ids.push(7);
        assert!(!sent.only_spares(), "an image in flight makes it a crash");
        sent.tier2_ids.clear();
        assert_eq!(sent.ids().count(), 0, "no cancel goes to a spare pass");
        let told = |failed: bool, sent: &mut SentWork| {
            let (tx, rx) = std::sync::mpsc::channel();
            sent.abandon_spares(failed, &tx);
            rx.try_iter().map(|r| match r {
                WorkerResult::SpareEdgeScales { item_id, token, landing } => (item_id, token, matches!(landing, super::SpareLanding::Failed)),
                _ => panic!("only spare-scale results"),
            }).collect::<Vec<_>>()
        };
        assert_eq!(told(false, &mut sent), vec![(4, 40, false), (5, 50, false)], "a cancel skips them");
        assert!(sent.is_empty());
        sent.spares.push((6, 60));
        assert_eq!(told(true, &mut sent), vec![(6, 60, true)], "a crash failed them");
    }

    /// A file of some scales lands those scales and leaves the others
    /// missing; a file whose size disagrees with its scales lands nothing.
    #[test]
    fn an_edge_cache_of_some_scales_lands_only_those() {
        use super::{read_raw_edge_tensors, CompressedEdgeTensors, EdgeCacheFile};
        use prunr_core::EdgeScale::{Bold, Fine, Fused};
        let dir = tempfile::tempdir().unwrap();
        let (h, w) = (2u32, 3u32);
        let plane = |v: f32, s: prunr_core::EdgeScale| {
            let (sh, sw) = prunr_core::edge_scale_dims(s, h, w);
            vec![v; (sh * sw) as usize]
        };
        let write = |name: &str, planes: &[Vec<f32>]| {
            let path = dir.path().join(name);
            let bytes: Vec<u8> = planes.iter().flat_map(|p| crate::subprocess::ipc::f32s_as_le_bytes(p).to_vec()).collect();
            std::fs::write(&path, bytes).unwrap();
            path
        };

        let path = write("drawn.raw", &[plane(0.25, Bold), plane(0.75, Fused)]);
        let file = EdgeCacheFile { path: path.clone(), height: h, width: w, scales: vec![Bold, Fused] };
        let raw = read_raw_edge_tensors(Some(&file)).expect("the drawn scales land");
        assert!(!path.exists(), "read and deleted");
        assert!(raw.tensors[Fine as usize].is_empty());
        let cache = CompressedEdgeTensors::from_raw(raw).expect("compresses");
        assert!(cache.has_all(&[Bold, Fused]) && !cache.has_all(&[Fine]));
        assert_eq!(cache.decompress(Fused), Some(plane(0.75, Fused)));
        assert_eq!(cache.decompress(Bold), Some(plane(0.25, Bold)));
        assert_eq!(cache.decompress(Fine), None);

        let short = write("short.raw", &[plane(0.5, Bold)]);
        let file = EdgeCacheFile { path: short, height: h, width: w, scales: vec![Bold, Fused] };
        assert!(read_raw_edge_tensors(Some(&file)).is_none());
    }


    /// A chained cut starts from its start's colours seen through the
    /// cut-out's transparency; the file is read only when the colours are
    /// the file's, and a cut-out of another size is refused.
    #[test]
    fn a_kept_run_image_takes_colours_from_its_start_and_alpha_from_the_cutout() {
        use super::RunImage;
        let cutout = crate::gui::timeline::Stored::new(Arc::new(image::RgbaImage::from_fn(2, 1, |x, _| image::Rgba([9, 9, 9, if x == 0 { 0 } else { 200 }]))));
        let start = Arc::new(image::RgbaImage::from_pixel(2, 1, image::Rgba([50, 60, 70, 255])));
        let resident = |image: &Arc<image::RgbaImage>| Some(crate::gui::item::Pixels::Resident(Arc::clone(image)));
        let kept = RunImage::Kept { start: resident(&start), cutout: Arc::clone(&cutout) };
        let (runs_from, built) = kept.resolve(1, || panic!("the colours are the start's")).unwrap();
        let built = built.expect("built");
        assert!(runs_from.is_some_and(|r| Arc::ptr_eq(&r, &built)), "the run starts from the built image");
        assert_eq!(built.get_pixel(0, 0).0, [50, 60, 70, 0]);
        assert_eq!(built.get_pixel(1, 0).0, [50, 60, 70, 200]);

        let mut png = Vec::new();
        image::RgbaImage::from_pixel(2, 1, image::Rgba([1, 2, 3, 255])).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let (_, from_file) = RunImage::Kept { start: None, cutout: Arc::clone(&cutout) }.resolve(2, || Ok(Arc::new(png))).unwrap();
        assert_eq!(from_file.expect("built from the file").get_pixel(1, 0).0, [1, 2, 3, 200]);

        let wide = Arc::new(image::RgbaImage::new(3, 1));
        let (runs_from, built) = RunImage::Kept { start: resident(&wide), cutout }.resolve(3, || unreachable!()).unwrap();
        assert!(built.is_none(), "a cut-out of another size is refused");
        assert!(runs_from.is_some_and(|r| Arc::ptr_eq(&r, &wide)), "and the run starts from its plain start");
    }

    /// A run's inputs are read on the bridge thread: an unreadable file,
    /// start or tensor cache fails the image there, with the words the
    /// image shows; a stored start is unpacked there too.
    #[test]
    fn a_runs_inputs_are_read_where_it_is_sent() {
        use super::{CachedTensorWork, RunImage, WorkItem};
        use crate::gui::item::{ImageSource, Pixels};
        let missing = ImageSource::file(std::path::PathBuf::from("/nonexistent/photo.png"));
        let file = ImageSource::Bytes(Arc::new(vec![1, 2, 3]));

        let gone = WorkItem { id: 1, run: 0, source: missing.clone(), image: None }.read_inputs();
        assert!(gone.is_err_and(|e| e.contains("moved or deleted")));

        let start = crate::gui::timeline::Stored::new(Arc::new(image::RgbaImage::from_pixel(1, 1, image::Rgba([5, 6, 7, 255]))));
        let stored = WorkItem { id: 2, run: 0, source: file.clone(), image: Some(RunImage::Start(Pixels::Parked(start))) };
        let inputs = stored.read_inputs().expect("reads");
        assert_eq!(inputs.bytes.as_slice(), &[1, 2, 3]);
        assert_eq!(inputs.start.expect("unpacked").get_pixel(0, 0).0, [5, 6, 7, 255]);

        let unreadable = WorkItem { id: 3, run: 0, source: file.clone(), image: Some(RunImage::Start(Pixels::Source(missing.clone()))) };
        assert!(unreadable.read_inputs().is_err(), "a start that can't be read fails the run, not cut the file instead");

        let tensor = CompressedTensor::from_raw(RawSegTensor { data: vec![0.5; 4], height: 2, width: 2, model: ModelKind::Silueta }).unwrap();
        let cached = CachedTensorWork { item_id: 4, tensor: tensor.clone(), source: file, mask: MaskSettings::default() };
        assert_eq!(cached.read_inputs().expect("reads").0, vec![0.5; 4]);
        let cached = CachedTensorWork { item_id: 5, tensor, source: missing, mask: MaskSettings::default() };
        assert!(cached.read_inputs().is_err_and(|e| e.contains("moved or deleted")));
    }

    use super::*;

    /// A worker is reused by its engines alone (a batch's mask, Sketch and
    /// line settings travel with it): the same model on the same
    /// processor, with a pool at least as large.
    #[test]
    fn a_worker_is_reused_for_the_same_engines() {
        let built = EngineSpec { model: ModelKind::BiRefNetLite, jobs: 4, cpu_only: false };
        let on = |a: &EngineSpec, b: &EngineSpec| a.reusable_by(b, "CPU");
        assert!(on(&built, &EngineSpec { jobs: 1, ..built }));
        assert!(on(&built, &built));
        assert!(!on(&built, &EngineSpec { model: ModelKind::Silueta, ..built }));
        assert!(!on(&built, &EngineSpec { cpu_only: true, ..built }));
        // On the CPU, Silueta builds an engine per job; BiRefNet one whatever the jobs.
        let with = |model, jobs| EngineSpec { model, jobs, ..built };
        assert!(!on(&with(ModelKind::Silueta, 4), &with(ModelKind::Silueta, 5)), "a larger pool needs a new worker");
        assert!(on(&with(ModelKind::BiRefNetLite, 1), &with(ModelKind::BiRefNetLite, 3)), "the same pool from other job counts");
        assert!(!with(ModelKind::BiRefNetLite, 1).reusable_by(&with(ModelKind::BiRefNetLite, 3), "CUDA"), "a GPU builds one per job, up to two");
    }

    #[test]
    fn a_batch_completion_names_its_batch() {
        let cfg = ProcessingConfig {
            engines: EngineSpec { model: ModelKind::Silueta, jobs: 2, cpu_only: false },
            run: RunSettings::neutral(),
            first_look: Shown::default(),
            on_screen: Shown::default(),
        };
        let source = crate::gui::item::ImageSource::Bytes(Arc::new(Vec::new()));
        let tensor = CompressedTensor::from_raw(RawSegTensor { data: vec![0.0], height: 1, width: 1, model: ModelKind::Silueta }).unwrap();
        let tier2 = CachedTensorWork { item_id: 3, tensor, source: source.clone(), mask: MaskSettings::default() };
        let work = |id| WorkItem { id, run: 0, source: source.clone(), image: None };
        let mut state = BatchRunState::new(7, vec![work(1), work(2)], vec![tier2], Vec::new(), None, cfg);
        state.admit(work(4));
        assert!(matches!(state.complete_event(), WorkerResult::BatchComplete { batch: 7 }));
    }

    /// A cancel that lands while the queues are scanned: every image
    /// taken off them is reported, or it would wait for an answer forever.
    #[test]
    fn every_image_dropped_as_cancelled_is_reported() {
        let cfg = ProcessingConfig {
            engines: EngineSpec { model: ModelKind::Silueta, jobs: 2, cpu_only: false },
            run: RunSettings::neutral(),
            first_look: Shown::default(),
            on_screen: Shown::default(),
        };
        let source = crate::gui::item::ImageSource::Bytes(Arc::new(Vec::new()));
        let tensor = CompressedTensor::from_raw(RawSegTensor { data: vec![0.0], height: 1, width: 1, model: ModelKind::Silueta }).unwrap();
        let tier2 = CachedTensorWork { item_id: 3, tensor, source: source.clone(), mask: MaskSettings::default() };
        let work = |id| WorkItem { id, run: 0, source: source.clone(), image: None };
        let mut state = BatchRunState::new(7, vec![work(1), work(2)], vec![tier2], Vec::new(), None, cfg);
        // Image 1 is cancelled from the start; 2 and 3 from their second look on.
        let looks = std::cell::RefCell::new(std::collections::HashMap::<u64, u32>::new());
        let is_cancelled = |id: u64| {
            let mut looks = looks.borrow_mut();
            let n = looks.entry(id).or_default();
            *n += 1;
            id == 1 || *n > 1
        };
        let (tx, rx) = mpsc::channel();
        drop_cancelled_pending(&mut state, is_cancelled, &tx, &egui::Context::default());
        drop(tx);
        let mut seen: Vec<u64> = rx.iter().filter_map(|r| match r {
            WorkerResult::BatchItemDone { item_id, .. } => Some(item_id),
            _ => None,
        }).collect();
        seen.extend(state.pending.iter().map(|w| w.id));
        seen.extend(state.pending_tier2.iter().map(|w| w.item_id));
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3], "each image is still queued or reported, once");
    }

    /// A crash halves the jobs and runs the images again; at one job the
    /// image it ran is the cause and fails alone, not the whole batch.
    #[test]
    fn a_crash_at_one_job_fails_only_the_image_it_ran() {
        assert_eq!(super::on_crash(4), super::OnCrash::RetryWith(2));
        assert_eq!(super::on_crash(2), super::OnCrash::RetryWith(1));
        assert_eq!(super::on_crash(1), super::OnCrash::FailInFlight);
        use crate::subprocess::manager::Crash;
        assert!(super::crash_message(&Crash::OutOfMemory).starts_with("Ran out of memory"));
        assert!(super::crash_message(&Crash::Unresponsive).starts_with("The model stopped responding"));
        assert!(super::crash_message(&Crash::Other("exited with code 3".into())).starts_with("The model stopped unexpectedly"));
    }

    /// Work goes out up to the job limit; the memory pause holds it back
    /// only while some runs, or a batch whose running images all failed
    /// would wait for a memory report that never comes.
    #[test]
    fn work_is_sent_up_to_the_limit_and_the_pause_needs_running_work() {
        assert_eq!(super::room_to_send(0, 2, false), 2, "the batch's first images");
        assert_eq!(super::room_to_send(1, 2, false), 1, "one finished or failed: one more");
        assert_eq!(super::room_to_send(2, 2, false), 0);
        assert_eq!(super::room_to_send(1, 2, true), 0, "paused while one runs");
        assert_eq!(super::room_to_send(0, 2, true), 2, "nothing running: the pause cannot lift itself");
        assert_eq!(super::room_to_send(3, 2, false), 0, "fewer jobs after a crash");
    }

    /// A worker is hung only when it sits on work past the timeout: an
    /// idle one waiting for admission is not, and the bound is strict, so
    /// one reporting every `hang_timeout` on the dot is not either.
    #[test]
    fn a_worker_is_stalled_only_with_work_past_the_timeout() {
        let s = Duration::from_secs;
        for (silence, in_flight, stalled) in [(120, 0, false), (61, 1, true), (59, 1, false), (60, 1, false), (90, 5, true)] {
            assert_eq!(is_stalled(s(silence), in_flight, s(60)), stalled, "{silence} s silent, {in_flight} in flight");
        }
    }

}
