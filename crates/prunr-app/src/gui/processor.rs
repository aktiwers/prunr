//! Processing-pipeline coordinator: subprocess worker channels, admission
//! control, live preview, and per-batch dispatch state.
//!
//! Owns:
//! - The worker bridge channels (`worker_tx` / `worker_rx`) — UI thread sends
//!   `WorkerMessage`, drains `WorkerResult` non-blockingly each frame.
//! - The shared cancellation flag (`Arc<AtomicBool>`) — read by the worker
//!   bridge to short-circuit a batch in flight.
//! - The in-process Tier 2 live-preview dispatcher.
//! - Admission controller state during streaming batches.
//! - The dispatch-time recipe snapshot (used to attribute completed results
//!   to the settings that produced them, even if the user keeps editing).
//!
//! Does NOT own:
//! - The worker bridge thread itself — that's spawned by `worker::spawn_worker`
//!   at app startup. We just hold the channel ends.
//! - `BatchManager` (per the cross-coordinator borrow rule). Methods that
//!   operate on the batch take `&mut BatchManager` per call, never as a field.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;

use prunr_core::ProcessingRecipe;

use super::inpaint_bridge::{InpaintBridgeMsg, InpaintBridgeResult, spawn_inpaint_bridge};
use super::live_preview::LivePreview;
use super::memory::AdmissionController;
use super::worker::{EngineSpec, WorkerMessage, WorkerResult, WorkItem};

#[derive(Clone)]
pub struct CancelRegistry {
    /// How many times Cancel All was pressed. A handle is cancelled when
    /// it moved past `since`, so a later dispatch's reset never takes back
    /// a Cancel All the running batch has not seen yet.
    global: Arc<AtomicU64>,
    since: u64,
    // Short-circuit for the common zero-cancel case: `is_cancelled` skips the
    // mutex entirely unless some per-item entry has been requested.
    has_per_item: Arc<AtomicBool>,
    per_item: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            global: Arc::new(AtomicU64::new(0)),
            since: 0,
            has_per_item: Arc::new(AtomicBool::new(false)),
            per_item: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn is_global_cancelled(&self) -> bool {
        self.global.load(Ordering::Acquire) != self.since
    }

    pub(crate) fn is_cancelled(&self, item_id: u64) -> bool {
        if self.is_global_cancelled() {
            return true;
        }
        if !self.has_per_item.load(Ordering::Acquire) {
            return false;
        }
        let guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(&item_id).is_some_and(|f| f.load(Ordering::Acquire))
    }

    pub(crate) fn request_global_cancel(&self) {
        self.global.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn request_item_cancel(&self, item_id: u64) {
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.entry(item_id)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .store(true, Ordering::Release);
        self.has_per_item.store(true, Ordering::Release);
    }

    /// A fresh start for the next dispatch: Cancel Alls before now no
    /// longer count for this handle (or its clones made after), while a
    /// batch already holding a clone still sees them.
    pub(crate) fn reset(&mut self) {
        self.since = self.global.load(Ordering::Acquire);
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.clear();
        self.has_per_item.store(false, Ordering::Release);
    }
}

/// Active dispatch's recipe + the set of items still expected to deliver.
/// All items in a batch share one recipe (the toolbar broadcasts current
/// settings at dispatch). `take_recipe` removes an item; the slot self-
/// cleans when the last pending item completes — so a late ImageDone after
/// a settings edit can't pick up the wrong recipe.
struct InFlightBatch {
    /// The id the worker's completion names it by.
    id: u64,
    recipe: ProcessingRecipe,
    pending: HashSet<u64>,
    /// Total items registered for this dispatch — frozen at
    /// `track_dispatch` time. Used by `current_dispatch_progress` to
    /// report "done / total" scoped to the *current* dispatch, so
    /// previously-Done items from prior dispatches don't inflate the
    /// counter when the user reprocesses a single item.
    total: usize,
}

/// How an inpaint stroke ended; it travels with the item id.
pub(crate) enum InpaintOutcome {
    /// `fill_scale` is how large the fill ran (1 is full size);
    /// `generation` is the dispatch's own, so a fresher stroke wins.
    Done { rgba: image::RgbaImage, fill_scale: f32, generation: u64 },
    /// Drives the "Erase cancelled" toast so the user gets explicit
    /// feedback that Esc / the Cancel button took effect (vs. a silent
    /// failure or a stale-result drop).
    Cancelled,
    /// A non-Cancelled error (e.g. the SD-bundle RAM guard refused to
    /// load), shown as a toast so the user sees WHY the stroke did
    /// nothing instead of guessing it was lost.
    Failed(String),
}

/// A finished fill that is still its item's newest stroke.
pub(crate) struct LandedFill {
    pub item_id: u64,
    pub rgba: image::RgbaImage,
    pub fill_scale: f32,
}

/// Result from a background SAM encoder thread. Carries the embedding or
/// a human-readable error string. The item_id lets pump_sam_encoder_results
/// route the result to the right BatchItem even if the user switched selection.
pub(crate) struct SamEncoderResult {
    pub(crate) item_id: u64,
    pub(crate) result: Result<prunr_core::sam::SamEmbedding, String>,
}

/// A face search landing: the faces found in `photo` with `model`.
pub(crate) struct FacesResult {
    pub(crate) item_id: u64,
    pub(crate) photo: super::item::FacePhoto,
    pub(crate) model: prunr_models::ModelId,
    pub(crate) result: Result<Vec<prunr_core::face::RestoredFace>, String>,
    /// The search was stopped (Cancel or Esc), not failed.
    pub(crate) stopped: bool,
}

/// Single-slot warm engine for an in-process model run (upscale,
/// colorize). Shared with the dispatch thread so session construction
/// (seconds on a cold EP) never runs on the GUI thread; single-slot
/// because caching several engines of one kind is YAGNI.
type WarmEngine =
    Arc<Mutex<Option<(WarmEngineKey, Arc<prunr_core::OrtEngine>)>>>;

/// What a warm engine was built for: the model, and whether the
/// user chose "CPU only" (an engine built under the other choice misses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WarmEngineKey {
    model: prunr_core::ModelKind,
    cpu_only: bool,
}

fn lock_warm_engine(
    slot: &WarmEngine,
) -> std::sync::MutexGuard<'_, Option<(WarmEngineKey, Arc<prunr_core::OrtEngine>)>> {
    slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The cached engine on a key hit; a mismatch evicts the slot and
/// returns `None`. Never constructs an engine.
fn cached_warm_engine(
    slot: &WarmEngine,
    key: WarmEngineKey,
) -> Option<Arc<prunr_core::OrtEngine>> {
    let mut guard = lock_warm_engine(slot);
    match guard.as_ref() {
        Some((cached, engine)) if *cached == key => Some(Arc::clone(engine)),
        Some(_) => {
            *guard = None;
            None
        }
        None => None,
    }
}

/// Cached engine for `key`, constructing and caching one when absent.
/// The lock is released during construction so the GUI thread's
/// `release_all_sessions` never waits on a session build.
fn ensure_warm_engine(
    slot: &WarmEngine,
    key: WarmEngineKey,
    intra_threads: usize,
) -> Result<Arc<prunr_core::OrtEngine>, prunr_core::CoreError> {
    if let Some(cached) = cached_warm_engine(slot, key) {
        return Ok(cached);
    }
    let fresh = Arc::new(prunr_core::OrtEngine::for_model(key.model.into(), intra_threads, key.cpu_only)?);
    *lock_warm_engine(slot) = Some((key, Arc::clone(&fresh)));
    Ok(fresh)
}

/// The slot's engine, taken out; it frees where the result drops.
fn take_warm_engine(slot: &WarmEngine) -> Option<(WarmEngineKey, Arc<prunr_core::OrtEngine>)> {
    lock_warm_engine(slot).take()
}

/// Images waiting for the whole-image run slot, and how far the queue is.
#[derive(Default)]
struct RunQueue {
    waiting: std::collections::VecDeque<u64>,
    started: u32,
    total: u32,
}

/// An in-process model run, one at a time (upscale, colorize): its
/// in-flight and cancel flags, the channel its result comes back on, and
/// its warm engine.
struct RunSlot<R> {
    active: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<R>,
    rx: mpsc::Receiver<R>,
    warm: WarmEngine,
}

impl<R> RunSlot<R> {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            active: Arc::new(AtomicBool::new(false)),
            cancel: Arc::new(AtomicBool::new(false)),
            tx,
            rx,
            warm: Arc::new(Mutex::new(None)),
        }
    }

    fn in_flight(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Raise the cancel flag of the run in flight; `false` when none is.
    fn cancel(&self) -> bool {
        if !self.in_flight() {
            return false;
        }
        self.cancel.store(true, Ordering::Release);
        true
    }

    fn drain(&self) -> Vec<R> {
        self.rx.try_iter().collect()
    }

    /// A run starts: it is in flight, not cancelled, and reports to
    /// `progress`. The run's thread ends it through the returned handle.
    fn begin(&self, progress: super::dispatch_progress::RunSink) -> RunDone<R> {
        self.active.store(true, Ordering::Release);
        self.cancel.store(false, Ordering::Release);
        RunDone { active: Arc::clone(&self.active), tx: self.tx.clone(), progress }
    }

    fn take_engine(&self) -> Option<(WarmEngineKey, Arc<prunr_core::OrtEngine>)> {
        take_warm_engine(&self.warm)
    }
}

/// The run thread's end of a `RunSlot`.
struct RunDone<R> {
    active: Arc<AtomicBool>,
    tx: mpsc::Sender<R>,
    progress: super::dispatch_progress::RunSink,
}

impl<R> RunDone<R> {
    fn deliver(self, result: R) {
        // Send the result BEFORE clearing the active flag. Reversed
        // ordering would let the UI thread see the run idle and enable
        // Process before the result has landed in the channel.
        let _ = self.tx.send(result);
        self.active.store(false, Ordering::Release);
        self.progress.end();
    }
}

/// SAM 2 encoder + decoder sessions, built on first use and reused for
/// every click until Magic Brush is deactivated. Both run on the CPU EP;
/// ~180 MB resident while cached.
pub(crate) struct SamSessions {
    encoder: Mutex<ort::session::Session>,
    decoder: Mutex<ort::session::Session>,
}

type SamSessionSlot = Arc<Mutex<Option<Arc<SamSessions>>>>;

fn build_sam_session(part: &str) -> Result<ort::session::Session, String> {
    let bytes = prunr_models::resolve_part_bytes(prunr_models::ModelId::Sam21HieraSmall, part)
        .map_err(|e| format!("resolve {part} bytes: {e:?}"))?;
    prunr_core::ort_runtime::session_builder()
        .map_err(|e| e.to_string())?
        .commit_from_memory(&bytes)
        .map_err(|e| {
            tracing::warn!(part, %e, "Magic Brush model did not load");
            format!("Could not load the Magic Brush model: {e}")
        })
}

/// Cached sessions, built outside the lock so a release on the GUI
/// thread never waits on the ~1 s encoder parse.
fn ensure_sam_sessions(slot: &SamSessionSlot) -> Result<Arc<SamSessions>, String> {
    if let Some(cached) = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() {
        return Ok(Arc::clone(cached));
    }
    let built = Arc::new(SamSessions {
        encoder: Mutex::new(build_sam_session("encoder")?),
        decoder: Mutex::new(build_sam_session("decoder")?),
    });
    let mut guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    // Two clicks racing on a cold cache both build; keep whichever landed.
    Ok(Arc::clone(guard.get_or_insert(built)))
}

/// Result from a background SAM decoder thread, or a human-readable error.
pub(crate) struct SamDecoderResult {
    pub(crate) item_id: u64,
    pub(crate) mode: prunr_core::selection::BrushMode,
    pub(crate) reading: prunr_core::MaskReading,
    pub(crate) result: Result<DecodedSelection, String>,
}

pub(crate) enum DecodedSelection {
    /// A click or stroke: the prompt's mask, still to be merged onto the
    /// current selection, with the logits it came from so a Confidence
    /// change can re-threshold the same stroke without the decoder.
    Fresh { mask: prunr_core::selection::MaskArtifact, output: Arc<prunr_core::sam::SamDecoderOutput> },
    /// A re-threshold: already merged onto the pre-stroke selection and
    /// hashed on the pool, ready to replace the stroke's selection.
    Retuned { merged: Arc<prunr_core::selection::MaskArtifact>, hash: u64, for_hash: u64 },
    /// A click with Depth on: the depth band's mask, still to be merged,
    /// with no logits to re-threshold.
    ByDepth(prunr_core::selection::MaskArtifact),
}

/// A Magic Brush click with Depth on, captured at click time like
/// `SamDecodeRequest`.
pub(crate) struct DepthSelectRequest {
    pub(crate) item_id: u64,
    pub(crate) depth: Arc<prunr_core::depth::DepthMap>,
    /// The click as fractions of the photo's width and height.
    pub(crate) at: (f32, f32),
    pub(crate) range: f32,
    pub(crate) mode: prunr_core::selection::BrushMode,
    pub(crate) source_dims: (u32, u32),
    pub(crate) reading: prunr_core::MaskReading,
}

/// Re-threshold the logits of the last stroke at a new confidence.
pub(crate) struct SamRethresholdRequest {
    pub(crate) item_id: u64,
    pub(crate) output: Arc<prunr_core::sam::SamDecoderOutput>,
    pub(crate) mode: prunr_core::selection::BrushMode,
    pub(crate) source_dims: (u32, u32),
    pub(crate) reading: prunr_core::MaskReading,
    /// The selection the stroke was merged onto, read on the job's thread.
    pub(crate) base: Option<super::item::Region>,
    /// The selection the retuned click committed: the retune replaces it
    /// only while it is still that click's.
    pub(crate) for_hash: u64,
}

/// Combine a stroke, painted or decoded, with the selection it was made
/// against: Add grows the selection and Subtract carves it away. What a
/// cut-out's selection then does is the brush's `CutoutAction`.
pub(crate) fn merge_stroke(
    existing: Option<Arc<prunr_core::selection::MaskArtifact>>,
    stroke: prunr_core::selection::MaskArtifact,
) -> prunr_core::selection::MaskArtifact {
    let Some(existing) = existing else { return stroke.into_selection() };
    match existing.merge_stroke(&stroke) {
        Ok(merged) => merged,
        Err(err) => {
            tracing::warn!(%err, "stroke size differs from the selection; keeping the stroke alone");
            stroke.into_selection()
        }
    }
}

/// One Magic Brush click or stroke, captured at request time so a chip
/// change while the decoder runs cannot re-polarise the result.
pub(crate) struct SamDecodeRequest {
    pub(crate) item_id: u64,
    pub(crate) embedding: Arc<prunr_core::sam::SamEmbedding>,
    pub(crate) prompt: prunr_core::sam::prompt::SamPrompt,
    pub(crate) mode: prunr_core::selection::BrushMode,
    pub(crate) source_dims: (u32, u32),
    pub(crate) reading: prunr_core::MaskReading,
}

/// A depth map landing for `item_id`, measured from the photo the item
/// had at `epoch`.
pub(crate) struct DepthResult {
    pub(crate) item_id: u64,
    pub(crate) epoch: u64,
    pub(crate) result: Result<Arc<prunr_core::depth::DepthMap>, String>,
}

/// What `release_and_admit` did with the images that fit.
#[derive(Default, Debug)]
pub(crate) struct Admitted {
    pub(crate) sent: Vec<u64>,
    /// Built but not sent: the worker had gone.
    pub(crate) lost: Vec<u64>,
}

/// One photo to measure, queued for the depth thread.
pub(crate) struct DepthJob {
    pub(crate) item_id: u64,
    pub(crate) epoch: u64,
    pub(crate) input: super::item::Pixels,
    pub(crate) cpu_only: bool,
    /// The faces a Faces search already found in this photo; `None`: the
    /// depth thread finds them.
    pub(crate) faces: Option<Vec<prunr_core::face::FacePlace>>,
}

/// The depth thread's queue and its warm engine. One thread measures the
/// photos one after another, so a batch never decodes them all at once.
struct DepthQueue {
    jobs: mpsc::Sender<DepthJob>,
    done: mpsc::Receiver<DepthResult>,
    warm: WarmEngine,
}

impl DepthQueue {
    fn spawn(progress: super::dispatch_progress::DispatchProgressSlot) -> Self {
        let (jobs, job_rx) = mpsc::channel::<DepthJob>();
        let (done_tx, done) = mpsc::channel();
        let warm: WarmEngine = Arc::new(Mutex::new(None));
        let engine_slot = Arc::clone(&warm);
        std::thread::spawn(move || {
            let mut faces = prunr_core::face::LazyFaceFinder::default();
            for job in job_rx {
                let (item_id, epoch) = (job.item_id, job.epoch);
                let result = caught(|| measure_depth(&engine_slot, &progress, job, &mut faces)).unwrap_or_else(Err);
                if let Err(err) = &result {
                    tracing::warn!(item_id, %err, "depth run failed");
                }
                let _ = done_tx.send(DepthResult { item_id, epoch, result });
            }
        });
        Self { jobs, done, warm }
    }
}

/// Run `f`; a panic in it (a model or driver bug) becomes its message, so
/// the run it was fails instead of the app.
fn caught<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let why = payload.downcast_ref::<&str>().map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        tracing::error!(%why, "a model run panicked");
        "The model stopped unexpectedly. Try again; if it repeats, try another model.".to_string()
    })
}

/// Measure one photo on the warm engine, with the faces in it for Auto
/// focus. The capsule shows it only when nothing else is showing, since
/// it cannot be cancelled.
fn measure_depth(
    engine_slot: &WarmEngine,
    slot: &super::dispatch_progress::DispatchProgressSlot,
    job: DepthJob,
    finder: &mut prunr_core::face::LazyFaceFinder,
) -> Result<Arc<prunr_core::depth::DepthMap>, String> {
    let DepthJob { input, cpu_only, faces, .. } = job;
    let kind = prunr_core::depth::MODEL;
    let descriptor = prunr_models::descriptor(kind.into()).ok_or("the depth model is not registered")?;
    admitted(descriptor)?;
    let (progress, run) = if slot.read().is_none() {
        let initial = super::dispatch_progress::DispatchProgress::new(super::dispatch_progress::ProgressKind::Depth, prunr_core::Step::EstimatingDepth.label());
        let run = slot.begin(initial);
        (prunr_core::Progress::new(Arc::new(run.clone())), Some(run))
    } else {
        (prunr_core::Progress::none(), None)
    };
    let threads = prunr_core::batch::ort_intra_threads(1);
    let result = input.read().map_err(prunr_core::CoreError::ImageFormat)
        .and_then(|photo| {
            let engine = ensure_warm_engine(engine_slot, WarmEngineKey { model: kind, cpu_only }, threads)?;
            let map = prunr_core::depth::estimate_depth(&engine, &photo, &progress)?;
            let faces = faces.unwrap_or_else(|| finder.places([photo.width(), photo.height()], |f| f.find(&photo)));
            Ok(map.with_faces(&faces))
        })
        .map(Arc::new)
        .map_err(|e| e.to_string());
    if let Some(run) = run {
        run.end();
    }
    result
}

/// Lay the photo an erase starts from under its tiles' previews, so the
/// preview is the whole image filling in and the result reveals over it.
/// One resample of the photo to the preview's size, off the UI thread;
/// the preview holds at most `TILE_PREVIEW_EDGE`² RGBA (9 MB).
fn preview_the_photo(progress: &prunr_core::Progress, photo: &image::RgbaImage) {
    let size = photo.dimensions();
    progress.tile_preview((0, 0, size.0, size.1), size, |w, h| {
        prunr_core::formats::preview_rgba(std::borrow::Cow::Borrowed(photo), (w, h), (0, 0, w, h))
    });
}

/// What made a picture: each lands on its own preview, whatever state
/// the item is in when it arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PictureFrom {
    /// The run the item awaits (a cut, an upscale, an erase, …).
    Run,
    FaceSearch,
}

/// A picture a run made for its item (a finished tile, …), on its way
/// to the canvas.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RunPicture {
    pub item_id: u64,
    pub from: PictureFrom,
    pub update: prunr_core::ProgressUpdate,
    /// A tile's pixels, made ready for upload on the run's thread; the
    /// tile in `update` keeps only its place.
    pub tile: Option<Arc<egui::ColorImage>>,
}

impl RunPicture {
    pub(crate) fn new(item_id: u64, from: PictureFrom, mut update: prunr_core::ProgressUpdate) -> Self {
        let tile = match &mut update {
            prunr_core::ProgressUpdate::TilePreview(t) => {
                let rgba = std::mem::take(&mut t.rgba);
                Some(Arc::new(egui::ColorImage::from_rgba_unmultiplied([t.width as usize, t.height as usize], &rgba)))
            }
            _ => None,
        };
        Self { item_id, from, update, tile }
    }
}

/// Where a tile picture lands: its item, maker and rectangle on a
/// preview of that size. `None` for any other picture.
fn tile_place(picture: &RunPicture) -> Option<(u64, PictureFrom, [u32; 6])> {
    match &picture.update {
        prunr_core::ProgressUpdate::TilePreview(t) => Some((picture.item_id, picture.from, [t.x, t.y, t.width, t.height, t.full_width, t.full_height])),
        _ => None,
    }
}

/// A run's listener: its pictures go to the canvas while the run says
/// they may still land; its steps and counts go to the progress pill,
/// when it has one.
#[derive(Clone)]
pub(crate) struct PictureSink {
    item_id: u64,
    from: PictureFrom,
    pill: Option<super::dispatch_progress::RunSink>,
    current: Arc<dyn Fn() -> bool + Send + Sync>,
    pictures: mpsc::Sender<RunPicture>,
}

impl PictureSink {
    /// A face search: its pictures for `item_id` land while `current`
    /// holds. Without a `pill` (another run shows on it) steps and counts
    /// go nowhere.
    pub(crate) fn of_face_search(
        item_id: u64,
        pill: Option<super::dispatch_progress::RunSink>,
        current: impl Fn() -> bool + Send + Sync + 'static,
        pictures: mpsc::Sender<RunPicture>,
    ) -> Self {
        Self { item_id, from: PictureFrom::FaceSearch, pill, current: Arc::new(current), pictures }
    }

    /// A run on the pill: its pictures land while it is the run shown.
    pub(crate) fn of_run(item_id: u64, run: super::dispatch_progress::RunSink, pictures: mpsc::Sender<RunPicture>) -> Self {
        let shown = run.clone();
        Self { item_id, from: PictureFrom::Run, pill: Some(run), current: Arc::new(move || shown.current()), pictures }
    }

    /// The run is over: the pill clears unless a newer run took it.
    pub(crate) fn end(&self) {
        if let Some(run) = &self.pill {
            run.end();
        }
    }

    /// A progress handle reporting here that asks for tile previews.
    pub(crate) fn into_progress(self) -> prunr_core::Progress {
        prunr_core::Progress::new(Arc::new(self)).with_tile_previews(crate::subprocess::protocol::TILE_PREVIEW_EDGE)
    }
}

impl prunr_core::ProgressSink for PictureSink {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        match update {
            update @ (prunr_core::ProgressUpdate::TilePreview(_) | prunr_core::ProgressUpdate::Faces(_)) => {
                if (self.current)() {
                    let _ = self.pictures.send(RunPicture::new(self.item_id, self.from, update));
                }
            }
            update => {
                if let Some(run) = &self.pill {
                    run.report(update);
                }
            }
        }
    }

    fn wants_previews(&self) -> bool {
        (self.current)()
    }
}

/// Result delivered from a background upscale thread back to the main thread.
/// Keyed on `item_id`; only one upscale is in flight at a time (gated by
/// `can_process_intent`), so generation-based stale-drop is not needed today.
/// The `recipe` is captured at dispatch time so the pump can stamp
/// `item.applied_recipe` against what actually ran — drift between the
/// running dispatch and a user mid-edit is invisible to the recipe.
pub(crate) struct UpscaleResult {
    pub item_id: u64,
    /// The image's run it answers.
    pub run: u64,
    pub result: Result<UpscaleOutput, prunr_core::CoreError>,
    pub recipe: ProcessingRecipe,
}

/// A finished upscale: the run kept for Refine, and the image to show
/// (Refine's knobs applied in the upscale thread, not on the UI thread).
pub(crate) struct UpscaleOutput {
    pub run: super::item::UpscaleCache,
    pub shown: Arc<image::RgbaImage>,
}

/// A finished colorize, stamped with the recipe it ran.
pub(crate) struct ColorizeResult {
    pub item_id: u64,
    /// The image's run it answers.
    pub run: u64,
    pub result: Result<ColorizeOutput, prunr_core::CoreError>,
    pub recipe: ProcessingRecipe,
}

/// The run kept for Strength, and the image to show at the dispatch's
/// strength (rendered in the colorize thread, not on the UI thread).
pub(crate) struct ColorizeOutput {
    pub run: super::item::ColorizeCache,
    pub shown: Arc<image::RgbaImage>,
}

/// Pure admission check: returns `true` when `free_ram_mb >= working_set_mb`.
/// Extracted as a free function so unit tests can exercise it without
/// constructing a `Processor`.
pub(crate) fn admission_check(working_set_mb: u32, free_ram_mb: u32) -> bool {
    free_ram_mb >= working_set_mb
}

/// Whether free RAM covers the model's working set, or why not; a
/// refusal is logged. The toolbar's `can_process_intent` gate should have
/// stopped the run already; this is the last guard at the dispatch layer.
fn admitted(descriptor: &prunr_models::ModelDescriptor) -> Result<(), String> {
    let free_mb = crate::hardware::available_ram_mb_throttled();
    if admission_check(descriptor.working_set_mb, free_mb) {
        return Ok(());
    }
    tracing::warn!(
        model = ?descriptor.id,
        working_set_mb = descriptor.working_set_mb,
        free_mb,
        "admission refused: insufficient RAM"
    );
    Err(format!("needs about {} MB of free memory", descriptor.working_set_mb))
}

/// The image the upscale model reads: the input, denoised first when
/// `pre_denoise` is above 0, else the same `Arc` (no copy).
pub(crate) fn build_inference_input(input: &Arc<image::RgbaImage>, pre_denoise: f32) -> Arc<image::RgbaImage> {
    if pre_denoise > 0.0 {
        Arc::new(prunr_core::denoise::apply_denoise(input, pre_denoise))
    } else {
        Arc::clone(input)
    }
}

/// Eraser-specific tuning passed from `BrushSettings` into the dispatch.
/// Bundled into a struct to keep `dispatch_inpaint` from sprawling.
#[derive(Clone, Debug)]
pub(crate) struct InpaintTuning {
    pub sharpen: f32,
    pub feather_px: f32,
    pub grow_px: f32,
    /// How large the hole is filled.
    pub fill_scale: prunr_core::inpaint::FillScale,
    /// Which eraser model runs (LaMaFp32, BigLaMa, …). For
    /// SD-family models, the choice between `SdV15InpaintFp16`
    /// (standard SD weights) and `SdV15LcmInpaintFp16` (LCM weights)
    /// is driven by the user's scheduler pick upstream
    /// (`Settings::lcm_routing_active`).
    pub model: prunr_models::ModelId,
    /// SD-only: text prompt; ignored for LaMa-family models.
    pub sd_prompt: String,
    pub sd_negative_prompt: String,
    pub sd_guidance_scale: f32,
    /// SD-only: scheduler kind. Carried through to the worker via
    /// `SdInpaintRequest::scheduler` so the right denoise math runs.
    pub sd_scheduler: super::brush_state::SdScheduler,
    /// SD-only: number of denoise steps.
    pub sd_steps: u32,
    /// RORem only: its denoise steps.
    pub rorem_steps: u32,
    /// SD-only: pinned RNG seed for reproducibility. `None` = random.
    pub sd_seed: Option<u64>,
    /// SD-only: post-gate TAESD selection — already resolved by the
    /// caller against user preference + install state. Dispatch
    /// consumes verbatim; no further gating, no scheduler coupling.
    pub use_taesd: bool,
    /// SD-only: how the pipeline runs, planned by the caller from the
    /// settings and the free RAM.
    pub sd_tuning: prunr_core::inpaint_sd::SdTuning,
    /// Settings › Processor is "CPU only": no session uses a GPU provider.
    pub cpu_only: bool,
}

impl InpaintTuning {
    /// The region a fill replaces: the painted one grown by Expand, plus
    /// the Feather ring.
    fn fill_region(&self, painted: &prunr_core::selection::MaskArtifact, (w, h): (u32, u32)) -> image::GrayImage {
        prunr_core::inpaint_blend::fill_region(&painted.region_mask(w, h), self.grow_px, self.feather_px)
    }
}

impl Default for InpaintTuning {
    fn default() -> Self {
        Self {
            sharpen: 0.0,
            feather_px: 0.0,
            grow_px: 0.0,
            fill_scale: Default::default(),
            model: prunr_models::ModelId::LaMaFp32,
            sd_prompt: String::new(),
            sd_negative_prompt: String::new(),
            sd_guidance_scale: 1.0,
            sd_scheduler: super::brush_state::SdScheduler::Lcm,
            sd_steps: 8,
            rorem_steps: prunr_core::inpaint_sd::ROREM_STEPS,
            sd_seed: None,
            use_taesd: false,
            sd_tuning: Default::default(),
            cpu_only: false,
        }
    }
}

/// The CFG an SD dispatch runs with.
///
/// `model` is the *actually-dispatched* model id (after
/// `Settings::lcm_routing_active` resolution upstream). The clamp
/// derives from the model, NOT from the user's scheduler request —
/// picking the LCM scheduler without the LCM bundle installed must NOT
/// clamp CFG against standard SD weights.
pub(crate) fn effective_cfg(model: prunr_models::ModelId, user_cfg: f32) -> f32 {
    // LCM weights are calibrated for low CFG (model card: 1.0–2.0).
    // Standard SD passes the user's CFG straight through.
    if model == prunr_models::ModelId::SdV15LcmInpaintFp16 {
        user_cfg.clamp(1.0, 2.0)
    } else {
        user_cfg
    }
}

pub(crate) struct Processor {
    pub(crate) worker_tx: mpsc::Sender<WorkerMessage>,
    pub(crate) worker_rx: mpsc::Receiver<WorkerResult>,
    /// Cancellation state shared with the worker bridge. `global` stops the
    /// whole batch; per-item entries drop individual items at the next
    /// dispatch check.
    pub(crate) cancels: CancelRegistry,
    pub(crate) live_preview: LivePreview,
    /// Active admission controller (present only during streaming batches).
    pub(crate) admission: Option<AdmissionController>,
    /// Sender for streaming additional items to the worker.
    pub(crate) admission_tx: Option<mpsc::Sender<WorkItem>>,
    /// The batch admission feeds: images it admits join that batch.
    admission_batch: Option<u64>,
    /// Re-runs that failed on images still showing an earlier result,
    /// since the last batch ended: their status stays Done, so the end of
    /// the batch must not read as all done.
    failed_reruns: u32,
    /// Re-runs asked for while admission was still feeding a batch, run
    /// once it is through: a batch opened meanwhile would take the
    /// admitted images' answers and wait behind them for good.
    deferred_runs: Vec<(Vec<u64>, Option<super::timeline::StepKind>)>,
    /// The batches sent and not finished, oldest first: one queued behind
    /// a running one keeps its own images and recipe.
    in_flight: Vec<InFlightBatch>,
    next_batch: u64,
    /// The progress run the in-flight batch shows through.
    seg_run: Option<super::dispatch_progress::RunSink>,
    /// Inpaint dispatch state. Per-item generation counter discards
    /// stale results when the user paints a fresh stroke before the
    /// previous one finishes. `inpaint_pending` is the count of
    /// dispatches not yet drained — the canvas reads it via
    /// `is_inpaint_in_flight` to show a progress overlay.
    inpaint_tx: mpsc::Sender<(u64, InpaintOutcome)>,
    inpaint_rx: mpsc::Receiver<(u64, InpaintOutcome)>,
    inpaint_latest_generation: HashMap<u64, u64>,
    inpaint_pending: HashMap<u64, u32>,
    /// Per-item cancel flag for the in-flight inpaint stroke. The flag
    /// is checked between LaMa tiles and between SD UNet steps; when
    /// set, the rayon job returns `CoreError::Cancelled` early and the
    /// drain path ignores the result. Cancel button + Esc key both
    /// flip the flag for the currently-selected item.
    inpaint_cancels: HashMap<u64, std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The run each in-flight eraser stroke reports to.
    inpaint_runs: HashMap<u64, PictureSink>,
    /// Channels to the dedicated SD-inpaint subprocess bridge thread.
    /// LaMa / Big-LaMa / MI-GAN stay on the in-process rayon path; only
    /// SD-family dispatches go through these. Bridge spawns the
    /// subprocess lazily and drops it on a 5-min idle timer.
    inpaint_bridge_tx: mpsc::Sender<InpaintBridgeMsg>,
    inpaint_bridge_rx: mpsc::Receiver<InpaintBridgeResult>,
    /// The upscale run: a single image is upscaled on a background
    /// thread; live tile progress is published to `dispatch_progress`
    /// (the canvas overlay reads it); the thread checks the cancel flag
    /// between tiles. Its warm engine is kept across consecutive upscales
    /// so the second click skips the 1-3s graph-optimization stall
    /// (Level3 for ESRGAN, Level2 for HAT); evicted on a switch away from
    /// the cached ModelKind and on full release.
    upscale: RunSlot<UpscaleResult>,
    /// The images still to run after the one in flight, when Process was
    /// pressed with several ticked: an upscale or colorize takes one image
    /// at a time.
    run_queue: Mutex<RunQueue>,
    /// Shared ORT `RunOptions` for the in-flight upscale session. The GUI
    /// thread calls `.terminate()` on this from `cancel_upscale()` to
    /// abort the running `Session::run` within ~50ms instead of waiting
    /// for the next tile boundary's cancel-flag check. Reset via
    /// `.unterminate()` before each new dispatch.
    ///
    /// Lazy because `RunOptions::new()` requires the ORT runtime to be
    /// loaded — `Processor::new()` runs from tests that don't init ORT,
    /// and an eager construction blocks on dlopen of the unloaded
    /// runtime (deadlock observed 2026-05-16 in the workspace lib test
    /// pass). First dispatch (which already requires ORT) initializes
    /// the slot; cancel before any dispatch is a no-op.
    upscale_run_options: std::sync::OnceLock<Arc<prunr_core::upscale::UpscaleRunOptions>>,
    /// The pictures runs make for their items (`PictureSink`).
    picture_tx: mpsc::Sender<RunPicture>,
    picture_rx: mpsc::Receiver<RunPicture>,
    /// The one progress snapshot every dispatch kind reports into; `None`
    /// when nothing runs.
    dispatch_progress: super::dispatch_progress::DispatchProgressSlot,
    /// The colorize run. Its cancel flag is read before and after the
    /// model's one run; its engine is released with every other session
    /// on a model switch.
    colorize: RunSlot<ColorizeResult>,
    /// SAM 2 encoder results channel. The rayon job sends here when
    /// the embedding is ready (or an error). Drained each frame from
    /// pump_sam_encoder_results (called via drain_background_channels).
    sam_encoder_tx: mpsc::Sender<SamEncoderResult>,
    sam_encoder_rx: mpsc::Receiver<SamEncoderResult>,
    /// SAM 2 decoder results channel. One result per click/stroke dispatch.
    sam_decoder_tx: mpsc::Sender<SamDecoderResult>,
    sam_decoder_rx: mpsc::Receiver<SamDecoderResult>,
    sam_sessions: SamSessionSlot,
    /// Whether dispatches run their models. A test app's runs start
    /// and stay in flight until the test lands or fails them with the
    /// `_for_test` helpers: a real run would build an ORT session the
    /// test process never initialises, and the pool worker it blocks is
    /// lost for every test after it.
    runs_models: bool,
    faces_tx: mpsc::Sender<FacesResult>,
    faces_rx: mpsc::Receiver<FacesResult>,
    /// The latest face search's stop flag, for Cancel and Esc.
    faces_stop: Mutex<Arc<AtomicBool>>,
    /// The latest face search's number: an earlier search's pictures are
    /// dropped.
    faces_latest: Arc<AtomicU64>,
    depth: DepthQueue,
    /// A cut-out model to load once the user stops switching, and when.
    seg_warm_due: Option<(std::time::Instant, EngineSpec)>,
    /// The image on screen whose run shows a first look at its subject,
    /// shared with the worker, which reads it as each run starts.
    pub(crate) first_look: super::worker::Shown,
    /// The image on screen, shared with the worker: only its Sketch runs
    /// make the line scales they do not draw.
    pub(crate) on_screen: super::worker::Shown,
    /// How long each cut-out model takes per image here, which decides
    /// whether a first look shows.
    pub(crate) cut_times: super::first_look::CutTimes,
}

/// How long the picked model must stay picked before it loads: clicking
/// through the list loads only the one it stops on.
const SEG_WARM_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(1);

impl Processor {
    pub(crate) fn new(
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        let (inpaint_tx, inpaint_rx) = mpsc::channel();
        let (inpaint_bridge_tx, inpaint_bridge_rx) = spawn_inpaint_bridge();
        let (picture_tx, picture_rx) = mpsc::channel();
        let (sam_encoder_tx, sam_encoder_rx) = mpsc::channel();
        let (sam_decoder_tx, sam_decoder_rx) = mpsc::channel();
        let (faces_tx, faces_rx) = mpsc::channel();
        let dispatch_progress = super::dispatch_progress::DispatchProgressSlot::new();
        let depth = DepthQueue::spawn(dispatch_progress.clone());
        Self {
            worker_tx,
            worker_rx,
            cancels: CancelRegistry::new(),
            live_preview: LivePreview::default(),
            admission: None,
            admission_tx: None,
            admission_batch: None,
            failed_reruns: 0,
            deferred_runs: Vec::new(),
            in_flight: Vec::new(),
            next_batch: 0,
            seg_run: None,
            inpaint_tx,
            inpaint_rx,
            inpaint_latest_generation: HashMap::new(),
            inpaint_pending: HashMap::new(),
            inpaint_cancels: HashMap::new(),
            inpaint_runs: HashMap::new(),
            inpaint_bridge_tx,
            inpaint_bridge_rx,
            upscale: RunSlot::new(),
            run_queue: Mutex::new(RunQueue::default()),
            upscale_run_options: std::sync::OnceLock::new(),
            picture_tx,
            picture_rx,
            dispatch_progress,
            colorize: RunSlot::new(),
            sam_encoder_tx,
            sam_encoder_rx,
            sam_decoder_tx,
            sam_decoder_rx,
            sam_sessions: Arc::new(Mutex::new(None)),
            runs_models: true,
            faces_tx,
            faces_rx,
            faces_stop: Mutex::default(),
            faces_latest: Arc::default(),
            depth,
            seg_warm_due: None,
            first_look: Default::default(),
            on_screen: Default::default(),
            cut_times: Default::default(),
        }
    }

    /// Load `spec`'s model in the worker once it has stayed picked for
    /// `SEG_WARM_DEBOUNCE`, so the first Process skips the load.
    pub(crate) fn schedule_seg_warm(&mut self, spec: EngineSpec, now: std::time::Instant) {
        self.seg_warm_due = Some((now + SEG_WARM_DEBOUNCE, spec));
    }

    /// Send the warm-up when it is due; how long until it is, if it waits.
    /// A run under way drops it: its own worker stays warm after.
    pub(crate) fn pump_seg_warm(&mut self, now: std::time::Instant) -> Option<std::time::Duration> {
        if self.is_run_in_flight() || !self.in_flight.is_empty() {
            self.seg_warm_due = None;
        }
        let (due, _) = self.seg_warm_due.as_ref()?;
        if let Some(wait) = due.checked_duration_since(now).filter(|w| !w.is_zero()) {
            return Some(wait);
        }
        let (_, spec) = self.seg_warm_due.take()?;
        let _ = self.worker_tx.send(WorkerMessage::Warm(spec));
        None
    }

    #[cfg(test)]
    pub(crate) fn stub_model_runs(&mut self) {
        self.runs_models = false;
    }

    /// Run a model job on the rayon pool; a test app (`stub_model_runs`)
    /// runs none.
    fn spawn_model(&self, job: impl FnOnce() + Send + 'static) {
        if self.runs_models {
            // rayon aborts the process on a job's panic.
            rayon::spawn(move || {
                let _ = caught(job);
            });
        }
    }

    /// Build the LaMa / Big-LaMa / MI-GAN session in the background, so
    /// the first stroke skips the 5-10 s build.
    pub(crate) fn prewarm_eraser(&self, id: prunr_models::ModelId, cpu_only: bool) {
        self.spawn_model(move || {
            if let Err(e) = prunr_core::inpaint::prewarm(id, cpu_only) {
                tracing::warn!(?id, %e, "Inpaint prewarm failed");
            }
        });
    }

    /// Queue a photo for a depth map; it lands through
    /// `pump_depth_results`, tagged with the job's epoch.
    pub(crate) fn dispatch_depth(&self, job: DepthJob) {
        let _ = self.depth.jobs.send(job);
    }

    pub(crate) fn pump_depth_results(&mut self) -> Vec<DepthResult> {
        self.depth.done.try_iter().collect()
    }

    /// A Magic Brush click with Depth on: everything near the clicked
    /// depth, made on the pool and landing like a decoded stroke.
    pub(crate) fn dispatch_depth_select(&self, req: DepthSelectRequest) {
        let tx = self.sam_decoder_tx.clone();
        rayon::spawn(move || {
            let (w, h) = req.source_dims;
            let around = req.depth.at(req.at.0, req.at.1);
            let mask = prunr_core::depth::select_by_depth(&req.depth, w, h, around, req.range, req.mode);
            let _ = tx.send(SamDecoderResult {
                item_id: req.item_id, mode: req.mode, reading: req.reading,
                result: Ok(DecodedSelection::ByDepth(mask)),
            });
        });
    }

    pub(super) fn dispatch_progress(&self) -> Option<super::dispatch_progress::DispatchProgress> {
        self.dispatch_progress.read()
    }

    #[cfg(test)]
    pub(crate) fn land_inpaint_for_test(&mut self, item_id: u64, rgba: image::RgbaImage) {
        self.land_inpaint_scaled_for_test(item_id, rgba, 1.0);
    }

    #[cfg(test)]
    pub(crate) fn land_inpaint_scaled_for_test(&mut self, item_id: u64, rgba: image::RgbaImage, fill_scale: f32) {
        let generation = *self.inpaint_latest_generation.entry(item_id).and_modify(|g| *g += 1).or_insert(1);
        let _ = self.inpaint_tx.send((item_id, InpaintOutcome::Done { rgba, fill_scale, generation }));
    }

    #[cfg(test)]
    pub(crate) fn fail_inpaint_for_test(&mut self, item_id: u64, error: &str) {
        let _ = self.inpaint_tx.send((item_id, InpaintOutcome::Failed(error.into())));
    }

    #[cfg(test)]
    pub(crate) fn show_progress_for_test(&self, progress: super::dispatch_progress::DispatchProgress) {
        self.dispatch_progress.set(Some(progress));
    }

    /// Show `text` as the background-removal run's current step.
    pub(crate) fn set_seg_step_text(&self, text: String) {
        if let Some(run) = &self.seg_run {
            run.set_step_text(text);
        }
    }

    /// Fold a background-removal worker report into its run.
    pub(crate) fn apply_seg_report(&self, update: prunr_core::ProgressUpdate) {
        if let Some(run) = &self.seg_run {
            prunr_core::ProgressSink::report(run, update);
        }
    }

    fn report_seg_count(&self) {
        if let (Some(run), Some((done, total))) = (&self.seg_run, self.current_dispatch_progress()) {
            let update = prunr_core::ProgressUpdate::Outer { done, total, unit: prunr_core::Unit::Image };
            prunr_core::ProgressSink::report(run, update);
        }
    }

    /// The batch is over or cancelled: stop showing it.
    pub(crate) fn end_seg_run(&mut self) {
        if let Some(run) = self.seg_run.take() {
            run.end();
        }
    }

    #[cfg(test)]
    pub(crate) fn set_upscale_in_flight_for_test(&self, active: bool) {
        self.upscale.active.store(active, Ordering::Release);
    }

    /// `true` while an upscale dispatch is in flight.
    pub fn is_upscale_in_flight(&self) -> bool {
        self.upscale.in_flight()
    }

    /// `true` while an upscale or a colorize is in flight.
    pub fn is_run_in_flight(&self) -> bool {
        self.upscale.in_flight() || self.colorize.in_flight()
    }

    /// Per-item generation counter ensures a fresh stroke supersedes the
    /// previous in-flight job at drain time — see `drain_inpaint_results`.
    /// SD-family models route through the inpaint subprocess bridge for
    /// process isolation; LaMa / Big-LaMa / MI-GAN stay on the in-process
    /// rayon path (low RAM footprint, no isolation pressure).
    pub(crate) fn dispatch_inpaint(
        &mut self,
        item_id: u64,
        image: super::item::Pixels,
        correction: super::item::Region,
        tuning: InpaintTuning,
    ) {
        let latest = self.inpaint_latest_generation.entry(item_id).or_insert(0);
        *latest += 1;
        let generation = *latest;
        *self.inpaint_pending.entry(item_id).or_insert(0) += 1;
        // Replace any prior cancel flag + progress sink — the new
        // dispatch supersedes its predecessor anyway, so wiring fresh
        // ones avoids a stale earlier-stroke cancel firing the moment
        // a new stroke starts (and avoids the capsule showing the prior
        // stroke's last step count for one frame).
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let initial = self.queued_initial(super::dispatch_progress::ProgressKind::Inpaint);
        let run = self.dispatch_progress.begin(initial);
        let sink = PictureSink::of_run(item_id, run.clone(), self.picture_sender());
        let progress = sink.clone().into_progress().with_cancel(cancel.clone());
        self.inpaint_cancels.insert(item_id, cancel.clone());
        self.inpaint_runs.insert(item_id, sink);
        if tuning.model.is_sd_family() {
            // The SD subprocess reports through the bridge to the run's sink.
            self.dispatch_inpaint_sd(item_id, generation, &image, &correction, &tuning, progress);
            return;
        }
        let tx = self.inpaint_tx.clone();
        self.spawn_model(move || {
            let fill = caught(|| {
                let (Some(image), Some(correction)) = (image.load(), correction.load()) else {
                    return Err(prunr_core::CoreError::ImageFormat("the image could not be read".into()));
                };
                preview_the_photo(&progress, &image);
                let mask = tuning.fill_region(&correction, image.dimensions());
                // This path only sees LaMa / Big-LaMa / MI-GAN.
                let filled = prunr_core::inpaint::process_lama(&image, &mask, tuning.model, tuning.cpu_only, tuning.fill_scale, &progress)?;
                progress.step(prunr_core::Step::Blending);
                let rgba = prunr_core::inpaint_blend::finalize_inpaint(filled.rgba, &image, &mask, tuning.feather_px, tuning.sharpen);
                Ok((rgba, filled.scale))
            });
            run.end();
            let outcome = match fill.unwrap_or_else(|why| Err(prunr_core::CoreError::Inference(why))) {
                Ok((rgba, fill_scale)) => InpaintOutcome::Done { rgba, fill_scale, generation },
                Err(prunr_core::CoreError::Cancelled) => {
                    tracing::info!(item_id, "inpaint cancelled by user");
                    InpaintOutcome::Cancelled
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::error!(item_id, e = %msg, "inpaint dispatch failed");
                    InpaintOutcome::Failed(msg)
                }
            };
            let _ = tx.send((item_id, outcome));
        });
    }

    /// Cancel every in-flight inpaint stroke. Used by `handle_cancel`
    /// (Esc) so it doesn't have to iterate `batch.items` on the GUI
    /// side — coordinator pattern: PrunrApp delegates, Processor owns
    /// the in-flight set. Idempotent and cheap (HashMap walk).
    pub(crate) fn cancel_all_inpaints(&self) {
        for flag in self.inpaint_cancels.values() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        if !self.inpaint_cancels.is_empty() {
            let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel);
        }
    }

    /// Cancel any in-flight inpaint stroke for `item_id`. The local
    /// flag and the capsule's "Stopping" show at once; for SD
    /// strokes the bridge kills the subprocess, ending every SD stroke.
    /// A LaMa stroke stops at its next tile; ORT has no per-op cancel
    /// hook. Idempotent.
    pub(crate) fn cancel_inpaint(&self, item_id: u64) {
        if let Some(flag) = self.inpaint_cancels.get(&item_id) {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel);
        self.mark_cancelling(super::dispatch_progress::ProgressKind::Inpaint);
    }

    /// Show that a run of `kind` is stopping, until it ends.
    fn mark_cancelling(&self, kind: super::dispatch_progress::ProgressKind) {
        self.dispatch_progress.update(|p| {
            if let Some(p) = p.as_mut().filter(|p| p.kind == kind) {
                p.cancelling = true;
            }
        });
    }

    /// Tell the inpaint bridge to drop its cached subprocess. Idempotent;
    /// no-op when the bridge is already idle. Called by
    /// `apply_toolbar_change` on every model switch so a previous
    /// SD-family model's ~200 MB residual subprocess (bundle was
    /// already released by `inpaint_sd::release` on dispatch
    /// completion, but the worker process itself stays alive for the
    /// 5-min idle window) doesn't pin RAM the user no longer wants
    /// allocated.
    pub(crate) fn release_inpaint_subprocess(&self) {
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Release);
    }

    /// The image on screen whose run would show a first look. Once first
    /// looks go off, the warm worker lets its first-look model go.
    pub(crate) fn show_first_look(&mut self, item: Option<u64>) {
        if self.first_look.set(item).is_some() && item.is_none() {
            let _ = self.worker_tx.send(WorkerMessage::FirstLookOff);
        }
    }

    /// Whether a batch of cut-outs is under way.
    pub(crate) fn batch_in_flight(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Tell the seg/edge worker bridge to drop its warm subprocess.
    /// Idempotent; no-op when nothing is warm. Symmetric with
    /// `release_inpaint_subprocess` — both fire on model switch so a
    /// previous model's engine pool (BiRefNetLite ~2 GB, U2Net
    /// ~800 MB, …) isn't held across the user's "I'm done with that
    /// model" signal.
    pub(crate) fn release_seg_warm(&mut self) {
        self.seg_warm_due = None;
        let _ = self.worker_tx.send(WorkerMessage::ReleaseWarm);
    }

    /// Drop every cached model session: the in-process eraser, upscale
    /// and colorize ones, and both workers'. On a model switch or a
    /// Processor change.
    pub(crate) fn release_all_sessions(&mut self) {
        let released = (
            prunr_core::inpaint::release_all(),
            self.upscale.take_engine(),
            self.colorize.take_engine(),
            take_warm_engine(&self.depth.warm),
        );
        // Freeing the sessions can take hundreds of ms; the caches are
        // already empty, so it waits on no one.
        super::reclaim::later(released);
        self.release_seg_warm();
        self.release_inpaint_subprocess();
    }

    #[cfg(test)]
    fn try_cached_upscale_engine(
        &mut self,
        model_kind: prunr_core::ModelKind,
        cpu_only: bool,
    ) -> Option<Arc<prunr_core::OrtEngine>> {
        cached_warm_engine(&self.upscale.warm, WarmEngineKey { model: model_kind, cpu_only })
    }

    /// Test-only accessor for injecting or inspecting the warm-engine slot.
    #[cfg(test)]
    pub(crate) fn warm_upscale_engine_for_test(
        &mut self,
    ) -> std::sync::MutexGuard<'_, Option<(WarmEngineKey, Arc<prunr_core::OrtEngine>)>> {
        lock_warm_engine(&self.upscale.warm)
    }

    /// SD inpaint dispatch via the dedicated subprocess bridge. Encodes
    /// the (already-Arc'd) image + binary mask to PNG temp files,
    /// writes them, and sends an `InpaintBridgeMsg::Dispatch`. The
    /// bridge handles subprocess lifecycle. Bridge results stream back
    /// via `pump_inpaint_subprocess` (called once per frame).
    fn dispatch_inpaint_sd(
        &mut self,
        item_id: u64,
        generation: u64,
        image: &super::item::Pixels,
        correction: &super::item::Region,
        tuning: &InpaintTuning,
        progress: prunr_core::Progress,
    ) {
        // PNG-encode + temp-file write is 150-300 ms per stroke at 4K
        // — moving it onto rayon keeps the egui frame loop responsive
        // mid-paint. Mask construction (`region_mask`, `grow_mask`)
        // is also CPU work and rides along.
        let image = image.clone();
        let correction = correction.clone();
        let tuning = tuning.clone();
        let bridge_tx = self.inpaint_bridge_tx.clone();
        let inpaint_tx = self.inpaint_tx.clone();
        // Cancel flag the parent already inserted into `inpaint_cancels`
        // — the rayon job checks it before any work and before the
        // bridge dispatch so a mid-encode Esc doesn't end up running
        // the SD bundle on a stroke the user already discarded.
        let cancel = self.inpaint_cancels.get(&item_id).cloned();
        rayon::spawn(move || {
            let cancelled = || cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire));
            let send_cancelled = || {
                let _ = inpaint_tx.send((item_id, InpaintOutcome::Cancelled));
            };
            if cancelled() { send_cancelled(); return; }
            let (Some(image), Some(correction)) = (image.load(), correction.load()) else {
                let _ = inpaint_tx.send((item_id, InpaintOutcome::Failed("the image could not be read".into())));
                return;
            };
            // Before the dispatch, so it lands under the worker's previews.
            preview_the_photo(&progress, &image);
            let mask = tuning.fill_region(&correction, image.dimensions());
            let sd_req = sd_request(&tuning);
            let dir = crate::subprocess::protocol::ipc_temp_dir();
            let image_path = crate::subprocess::protocol::IpcKind::InpaintImg.path_for_generation(dir, item_id, generation);
            let mask_path  = crate::subprocess::protocol::IpcKind::InpaintMask.path_for_generation(dir, item_id, generation);
            let res: Result<(), String> = (|| {
                crate::subprocess::ipc::write_image(&image_path, &image).map_err(|e| format!("write source: {e}"))?;
                crate::subprocess::ipc::write_image(&mask_path, &mask).map_err(|e| format!("write mask: {e}"))
            })();
            if let Err(e) = res {
                let _ = inpaint_tx.send((item_id, InpaintOutcome::Failed(e)));
                return;
            }
            // Esc landed during the encode — drop the work before the
            // bridge sees it. Temp files we just wrote are abandoned;
            // the next dispatch's generation-bump renames over them.
            if cancelled() {
                let _ = std::fs::remove_file(&image_path);
                let _ = std::fs::remove_file(&mask_path);
                send_cancelled();
                return;
            }
            let _ = bridge_tx.send(InpaintBridgeMsg::Dispatch {
                item_id, generation, model_id: tuning.model, image_path, mask_path, sd_req,
                feather_px: tuning.feather_px,
                sharpen: tuning.sharpen,
            });
        });
    }

    /// Drain bridge events, forward into the existing inpaint result
    /// channel + progress sinks. Called once per frame from app pump.
    pub(crate) fn pump_inpaint_subprocess(&mut self) {
        while let Ok(evt) = self.inpaint_bridge_rx.try_recv() {
            match evt {
                InpaintBridgeResult::Report { item_id, update } => self.report_inpaint(item_id, update),
                InpaintBridgeResult::Done { item_id, generation, rgba_path, width, height, fill_scale } => {
                    // The result file is read off the UI thread (a
                    // full-size image); the capsule ends once it is there.
                    let (tx, sink) = (self.inpaint_tx.clone(), self.inpaint_runs.get(&item_id).cloned());
                    rayon::spawn(move || {
                        // Stamp with the dispatch's own generation — drain
                        // drops it as stale if a fresher stroke bumped
                        // `inpaint_latest_generation` while this one was
                        // in the subprocess.
                        let outcome = match crate::subprocess::ipc::read_result_rgba(&rgba_path, width, height) {
                            Some(rgba) => InpaintOutcome::Done { rgba, fill_scale, generation },
                            None => InpaintOutcome::Failed(format!("eraser result missing or the wrong size: {}", rgba_path.display())),
                        };
                        let _ = tx.send((item_id, outcome));
                        if let Some(sink) = sink {
                            sink.end();
                        }
                    });
                }
                InpaintBridgeResult::Error { item_id, error } => {
                    // Translate bridge sentinels to user-facing text at
                    // this seam so `drain_inpaint_results` doesn't have
                    // to know about IPC strings.
                    use crate::subprocess::protocol::{is_cancel_error, MEMORY_PRESSURE_ABORT_MSG};
                    let outcome = if is_cancel_error(&error) {
                        InpaintOutcome::Cancelled
                    } else if error == MEMORY_PRESSURE_ABORT_MSG {
                        InpaintOutcome::Failed(format!(
                            "Erase stopped: memory ran low. Close other apps, or {}.",
                            prunr_core::inpaint_sd::LIGHTER_ERASER_HINT,
                        ))
                    } else {
                        InpaintOutcome::Failed(error)
                    };
                    let _ = self.inpaint_tx.send((item_id, outcome));
                    self.end_inpaint_run(item_id);
                }
            }
        }
    }

    /// An SD stroke's report reaches its run as an in-process one's does:
    /// a picture to the canvas, the rest to the pill.
    pub(crate) fn report_inpaint(&self, item_id: u64, update: prunr_core::ProgressUpdate) {
        if let Some(sink) = self.inpaint_runs.get(&item_id) {
            prunr_core::ProgressSink::report(sink, update);
        }
    }

    /// The SD stroke's run is over: clear the progress it showed.
    fn end_inpaint_run(&self, item_id: u64) {
        if let Some(sink) = self.inpaint_runs.get(&item_id) {
            sink.end();
        }
    }

    /// Drain in-flight inpaint results.
    ///
    /// Returns `(committed_results, cancelled_item_ids, errors)`.
    /// - `committed_results` only carries the latest-generation finished
    ///   strokes (stale ones are silently dropped).
    /// - `cancelled_item_ids` lists items whose stroke was cancelled
    ///   by the user — the GUI surfaces a toast for each so the user
    ///   gets explicit feedback that Esc/Cancel took effect.
    /// - `errors` carries the item and user-visible message for any non-Cancelled
    ///   dispatch failure (e.g. SD RAM-guard refusal). The GUI shows
    ///   each as an error toast — without this surface the user sees
    ///   the stroke "do nothing" with no idea why.
    pub(crate) fn drain_inpaint_results(&mut self) -> (Vec<LandedFill>, Vec<u64>, Vec<(u64, String)>) {
        let mut out = Vec::new();
        let mut cancelled = Vec::new();
        let mut errors = Vec::new();
        while let Ok((item_id, outcome)) = self.inpaint_rx.try_recv() {
            // Every drained result decrements pending — stale ones
            // count too, since the rayon job that produced them has
            // run to completion.
            if let Some(c) = self.inpaint_pending.get_mut(&item_id) {
                *c = c.saturating_sub(1);
            }
            // Reclaim the per-item cancel/progress entries once nothing
            // else is in flight for this id. Without this they accumulate
            // for the life of the session — `cancel_all_inpaints` walks
            // them all, and Esc-after-50-strokes ends up firing 50
            // no-op IPC cancels.
            if self.inpaint_pending.get(&item_id).copied().unwrap_or(0) == 0 {
                self.inpaint_cancels.remove(&item_id);
                self.inpaint_runs.remove(&item_id);
            }
            match outcome {
                InpaintOutcome::Cancelled => cancelled.push(item_id),
                InpaintOutcome::Failed(msg) => errors.push((item_id, msg)),
                InpaintOutcome::Done { rgba, fill_scale, generation } => {
                    if self.inpaint_latest_generation.get(&item_id) == Some(&generation) {
                        tracing::info!(item_id, fill_scale, "erase landed");
                        out.push(LandedFill { item_id, rgba, fill_scale });
                    }
                }
            }
        }
        (out, cancelled, errors)
    }

    /// True while a dispatched inpaint job hasn't drained yet for `item_id`.
    /// Canvas reads this to render a "Erasing..." overlay during LaMa work.
    pub(crate) fn is_inpaint_in_flight(&self, item_id: u64) -> bool {
        self.inpaint_pending.get(&item_id).copied().unwrap_or(0) > 0
    }

    /// True while ANY item has an in-flight inpaint job. Status bar
    /// reads this to override the "All done" text during LaMa work.
    pub(crate) fn any_inpaint_in_flight(&self) -> bool {
        self.inpaint_pending.values().any(|&c| c > 0)
    }

    #[cfg(test)]
    pub(crate) fn set_inpaint_in_flight_for_test(&mut self, item_id: u64, on: bool) {
        self.inpaint_pending.insert(item_id, u32::from(on));
    }

    /// Register a batch sent to the worker: `ids` land with `recipe`.
    /// Returns the batch's id, which its completion names. An image an
    /// older batch still lists (its run was cancelled) now awaits this
    /// batch only: the older run's late answer no longer lands.
    pub(crate) fn track_dispatch(
        &mut self,
        recipe: ProcessingRecipe,
        ids: impl IntoIterator<Item = u64>,
    ) -> u64 {
        let pending: HashSet<u64> = ids.into_iter().collect();
        let total = pending.len();
        self.in_flight.retain_mut(|b| {
            let before = b.pending.len();
            b.pending.retain(|id| !pending.contains(id));
            b.total -= before - b.pending.len();
            !b.pending.is_empty()
        });
        let id = self.next_batch;
        self.next_batch += 1;
        self.in_flight.push(InFlightBatch { id, recipe, pending, total });
        if self.seg_run.is_none() {
            let initial = super::dispatch_progress::DispatchProgress::new(super::dispatch_progress::ProgressKind::Seg, "Starting");
            self.seg_run = Some(self.dispatch_progress.begin(initial));
        }
        self.report_seg_count();
        id
    }

    /// `(done, total)` over the batches in flight, or `None` when idle.
    /// Scoped to them, so an image Done by an earlier run doesn't inflate
    /// the total ("1 of 2" when only one image was sent).
    pub(crate) fn current_dispatch_progress(&self) -> Option<(u32, u32)> {
        if self.in_flight.is_empty() {
            return None;
        }
        let (done, total) = self.in_flight.iter()
            .fold((0, 0), |(d, t), b| (d + b.total - b.pending.len(), t + b.total));
        Some((done as u32, total as u32))
    }

    /// Add a streamed (admission-pool) item to the batch admission feeds
    /// (without one, the newest batch). The `debug_assert` catches
    /// "admission ran without a tracked batch" in tests; release builds
    /// no-op so a late delivery can't take down a real batch.
    pub(crate) fn track_streamed(&mut self, id: u64) {
        let fed = match self.admission_batch {
            Some(batch) => self.in_flight.iter_mut().find(|b| b.id == batch),
            None => self.in_flight.last_mut(),
        };
        match fed {
            Some(b) => {
                if b.pending.insert(id) {
                    b.total += 1;
                    self.report_seg_count();
                }
            }
            None => debug_assert!(false, "track_streamed called without active batch"),
        }
    }

    pub(crate) fn note_failed_rerun(&mut self) {
        self.failed_reruns += 1;
    }

    /// The re-runs that failed since this was last asked.
    pub(crate) fn take_failed_reruns(&mut self) -> u32 {
        std::mem::take(&mut self.failed_reruns)
    }

    /// Admission feeds `batch` from now on.
    pub(crate) fn admission_feeds(&mut self, batch: u64) {
        self.admission_batch = Some(batch);
    }

    /// `batch` ended: when admission fed it, admission ends too, and the
    /// images it never admitted come back (they will not run).
    pub(crate) fn end_admission_with(&mut self, batch: u64) -> Vec<u64> {
        if self.admission_batch != Some(batch) {
            return Vec::new();
        }
        let waiting = self.admission.as_ref().map(|a| a.pending_ids().collect()).unwrap_or_default();
        self.clear_admission();
        waiting
    }

    /// Keep a re-run of `ids` for when admission is through.
    pub(crate) fn defer_run(&mut self, ids: Vec<u64>, step: Option<super::timeline::StepKind>) {
        self.deferred_runs.push((ids, step));
    }

    /// The re-runs kept while admission fed a batch, once it no longer does.
    pub(crate) fn take_deferred_runs(&mut self) -> Vec<(Vec<u64>, Option<super::timeline::StepKind>)> {
        if self.admission_still_feeding() {
            return Vec::new();
        }
        std::mem::take(&mut self.deferred_runs)
    }

    /// Take the recipe for an item batch `batch` answered. `None` when
    /// that batch no longer awaits it (cancelled, or a newer batch runs
    /// the image): the answer is stale. A batch whose last item lands is
    /// done.
    pub(crate) fn take_recipe(&mut self, batch: u64, id: u64) -> Option<ProcessingRecipe> {
        let at = self.in_flight.iter().position(|b| b.id == batch && b.pending.contains(&id))?;
        let fed = self.admission_batch == Some(batch);
        let batch = &mut self.in_flight[at];
        batch.pending.remove(&id);
        let recipe = batch.recipe.clone();
        // A batch admission still feeds stays open for the images to come.
        if batch.pending.is_empty() && !fed {
            self.in_flight.remove(at);
        }
        self.after_batch_change();
        Some(recipe)
    }

    /// The worker finished batch `id`: drop it and return the images it
    /// never answered.
    pub(crate) fn finish_batch(&mut self, id: u64) -> Vec<u64> {
        let Some(at) = self.in_flight.iter().position(|b| b.id == id) else { return Vec::new() };
        let batch = self.in_flight.remove(at);
        self.after_batch_change();
        batch.pending.into_iter().collect()
    }

    /// Drop every batch regardless of pending. Called on user cancel so a
    /// late delivery can't reattribute.
    pub(crate) fn drain_recipes(&mut self) {
        self.in_flight.clear();
        self.end_seg_run();
    }

    fn after_batch_change(&mut self) {
        if self.in_flight.is_empty() {
            self.end_seg_run();
        } else {
            self.report_seg_count();
        }
    }

    /// Whether the running batch still has images waiting for room in
    /// the RAM budget: a new batch opened now would orphan them.
    pub(crate) fn admission_still_feeding(&self) -> bool {
        self.admission.as_ref().is_some_and(AdmissionController::has_pending)
    }

    /// Drop admission state so no further items are admitted. Called on
    /// cancel (user or worker-side) and by the cancelled-message handler.
    /// Leaves the cancel registry untouched — that's owned by the caller's
    /// cancel protocol.
    pub(crate) fn clear_admission(&mut self) {
        self.admission = None;
        self.admission_tx = None;
        self.admission_batch = None;
    }

    /// Open streamed admission over the full runs queued on `ctrl`: what
    /// fits the RAM budget now is admitted, and its ids come back with
    /// the receiver the worker takes the rest from as `release_and_admit`
    /// sends them.
    pub(crate) fn open_admission(
        &mut self,
        mut ctrl: AdmissionController,
        mut work: impl FnMut(u64) -> Option<WorkItem>,
    ) -> (Vec<WorkItem>, mpsc::Receiver<WorkItem>) {
        let mut items = Vec::new();
        while let Some(id) = ctrl.try_admit_next() {
            match work(id) {
                Some(item) => items.push(item),
                // An image that can't be sent gives its share back at
                // once, so the next one can take it.
                None => ctrl.release(id),
            }
        }
        let (tx, rx) = mpsc::channel();
        self.admission = Some(ctrl);
        self.admission_tx = Some(tx);
        (items, rx)
    }

    /// `completed` frees its share of the budget, and the images that
    /// now fit are sent to the worker, each prepared by `work` (`None`:
    /// it can't be sent, and frees its share too). Admission ends once
    /// every image is through. Returns the images sent, and those that
    /// could not be (the worker is gone): neither started running here.
    pub(crate) fn release_and_admit(&mut self, completed: u64, mut work: impl FnMut(u64) -> Option<WorkItem>) -> Admitted {
        let mut admitted = Admitted::default();
        let Some(ctrl) = self.admission.as_mut() else { return admitted };
        ctrl.release(completed);
        while let Some(next) = ctrl.try_admit_next() {
            let Some(item) = work(next) else {
                ctrl.release(next);
                continue;
            };
            match self.admission_tx.as_ref() {
                Some(tx) if tx.send(item).is_ok() => admitted.sent.push(next),
                _ => {
                    ctrl.release(next);
                    admitted.lost.push(next);
                    break;
                }
            }
        }
        let complete = ctrl.is_complete();
        for &id in &admitted.sent {
            self.track_streamed(id);
        }
        if complete {
            let fed = self.admission_batch;
            self.clear_admission();
            // Every image answered: the batch it fed is done.
            if let Some(at) = fed.and_then(|b| self.in_flight.iter().position(|f| f.id == b && f.pending.is_empty())) {
                self.in_flight.remove(at);
                self.after_batch_change();
            }
        }
        admitted
    }

    /// Spawn a background thread that upscales the input and posts the
    /// result back through `pump_upscale_results`. The caller is
    /// responsible for having verified that the model is installed before
    /// dispatching.
    ///
    /// `model_id` is the model to run (`upscale::model_for_scale`), which
    /// may be Real-ESRGAN's 2x model for a 2x Real-ESRGAN upscale.
    ///
    /// Refused (logged, nothing spawned) when free RAM is below the
    /// model's working set.
    pub(crate) fn dispatch_upscale(
        &mut self,
        item_id: u64,
        run_id: u64,
        input: super::item::Pixels,
        model_id: prunr_models::ModelId,
        cpu_only: bool,
        recipe: ProcessingRecipe,
    ) -> Result<(), String> {
        // One upscale runs at a time, so it gets every core; "Parallel
        // jobs" splits cores between images of a batch, not this run.
        let intra_threads = prunr_core::batch::ort_intra_threads(1);
        let Some(descriptor) = prunr_models::descriptor(model_id) else {
            tracing::error!(model = ?model_id, "upscale dispatch: model not in REGISTRY");
            return Err("the model is not available".into());
        };
        admitted(descriptor)?;

        let output_scale = recipe.upscale.output_scale;
        let scale_factor = output_scale.factor();

        let cache_model_kind = match prunr_core::ModelKind::try_from(model_id) {
            Ok(k) => k,
            Err(id) => {
                tracing::error!(
                    model = ?id,
                    "upscale dispatch: ModelKind mapping missing — refusing dispatch"
                );
                return Err("the model is not available".into());
            }
        };

        // First dispatch initializes the shared RunOptions; subsequent
        // dispatches reuse it. ORT itself is initialised at app start.
        let run_options = self.upscale_run_options.get_or_init(|| {
            Arc::new(
                prunr_core::upscale::UpscaleRunOptions::new()
                    .expect("RunOptions::new requires an initialized ORT runtime"),
            )
        });
        // Clear any sticky terminate flag from the prior dispatch's cancel.
        let _ = run_options.unterminate();
        // The run shows "Loading the model" until the first tile reports.
        let run = self.dispatch_progress.begin(self.queued_initial(super::dispatch_progress::ProgressKind::Upscale));
        // Release stores in `begin` pair with the Acquire load in
        // `upscale::tiling::upscale_tiled` (cancel flag) and in
        // `is_upscale_in_flight` (active flag).
        let done = self.upscale.begin(run.clone());
        let progress = PictureSink::of_run(item_id, run, self.picture_sender())
            .into_progress()
            .with_cancel(Arc::clone(&self.upscale.cancel));

        let engine_slot = Arc::clone(&self.upscale.warm);
        // Shared with the GUI thread's `cancel_upscale()` — terminating
        // this RunOptions aborts the running `Session::run` mid-tile.
        let run_options_for_thread = Arc::clone(run_options);

        std::thread::spawn(move || {
            let result = caught(|| {
                // Decode and Denoise run before the model: the row-parallel
                // rayon inside apply_denoise must not nest in ORT's thread pool
                // (the pattern that deadlocked apply_background_color, b2306bb).
                let input = input.read().map_err(prunr_core::CoreError::ImageFormat).inspect_err(|err| {
                    tracing::error!(item_id, %err, "upscale dispatch: input unreadable");
                })?;
                let input_for_inference = build_inference_input(&input, recipe.upscale.pre_denoise());
                // Session construction runs here, not on the GUI thread: a cold
                // EP (OpenVINO compiles the graph) can take tens of seconds and
                // the window must stay responsive so Cancel is reachable. The
                // thread holds its own Arc, so a model swap that evicts the slot
                // mid-run never invalidates this session.
                let engine_for_thread = ensure_warm_engine(
                    &engine_slot, WarmEngineKey { model: cache_model_kind, cpu_only }, intra_threads,
                )
                .inspect_err(|err| tracing::error!(model = ?model_id, %err, "upscale dispatch: engine construction failed"))?;
                prunr_core::upscale::upscale_rgba_with_engine(
                    &input_for_inference,
                    &engine_for_thread,
                    model_id,
                    scale_factor,
                    &progress,
                    Some(&run_options_for_thread),
                )
                .map(|raw| {
                    let raw = Arc::new(raw);
                    let knobs = prunr_core::upscale::Refine::of_recipe(&recipe.upscale);
                    let (shown, reference) = super::live_preview::finish_upscale(&raw, &input_for_inference, None, &knobs, None);
                    UpscaleOutput { run: super::item::UpscaleCache::new(raw, input_for_inference, reference), shown }
                })
            })
            .unwrap_or_else(|why| Err(prunr_core::CoreError::Inference(why)));
            done.deliver(UpscaleResult { item_id, run: run_id, result, recipe });
        });
        Ok(())
    }

    /// Cancel any in-flight upscale dispatch.
    ///
    /// Two-stage abort:
    /// 1. `RunOptions::terminate()` signals ORT to abort the running
    ///    `Session::run` within ~50ms — much shorter than waiting for
    ///    the next tile boundary's cancel-flag check (5-15 s on CPU
    ///    EP). The worker thread sees a "terminate flag" error from
    ///    ORT, which `classify_run_error` maps to `CoreError::Cancelled`.
    /// 2. The atomic flag still gets set as a safety net — it handles
    ///    the case where the worker happens to be BETWEEN tiles when
    ///    cancel fires (no running session.run to terminate).
    ///
    /// Idempotent.
    pub(crate) fn cancel_upscale(&self) {
        // Esc and the model-change hook call this unconditionally.
        // Release pairs with Acquire in `upscale::tiling::upscale_tiled`.
        if !self.upscale.cancel() {
            return;
        }
        // The terminate call returns immediately; ORT aborts the running
        // session as soon as it checks between kernel launches.
        let terminate_ok = self
            .upscale_run_options
            .get()
            .is_some_and(|opts| opts.terminate().is_ok());
        // Not every EP honours terminate mid-run: OpenVINO measured 54 s
        // from terminate() to return on a cold 512 px tile (it only checks
        // the between-tiles flag). Say so instead of looking frozen.
        self.mark_cancelling(super::dispatch_progress::ProgressKind::Upscale);
        tracing::debug!(terminate_ok, "upscale cancel requested");
    }

    /// Where a `PictureSink` sends its run's pictures.
    pub(crate) fn picture_sender(&self) -> mpsc::Sender<RunPicture> {
        self.picture_tx.clone()
    }

    /// Drain the pictures runs have made since the last frame, oldest first.
    /// A tile painted again in the same place before the frame goes: only
    /// its newest picture is kept, in the order it came, so what lands on
    /// the canvas is the same and its upload is made once.
    pub(crate) fn pump_pictures(&mut self) -> Vec<RunPicture> {
        let mut pictures: Vec<RunPicture> = Vec::new();
        for picture in self.picture_rx.try_iter() {
            if let Some(place) = tile_place(&picture) {
                pictures.retain(|p| tile_place(p) != Some(place));
            }
            pictures.push(picture);
        }
        pictures
    }

    /// Drain completed upscale results from the background thread.
    /// Non-blocking; returns an empty `Vec` when nothing is ready. Caller
    /// applies each result to the matching `BatchItem`.
    pub(crate) fn pump_upscale_results(&mut self) -> Vec<UpscaleResult> {
        self.upscale.drain()
    }

    /// `true` while a colorize run is in flight.
    pub fn is_colorize_in_flight(&self) -> bool {
        self.colorize.in_flight()
    }

    #[cfg(test)]
    pub(crate) fn set_colorize_in_flight_for_test(&self, active: bool) {
        self.colorize.active.store(active, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn land_run_for_test(&self, engine: super::item::Engine, item_id: u64, run_id: u64, shown: Arc<image::RgbaImage>, recipe: ProcessingRecipe) {
        match engine {
            super::item::Engine::Upscale => {
                let run = super::item::UpscaleCache::new(Arc::clone(&shown), Arc::clone(&shown), None);
                let _ = self.upscale.tx.send(UpscaleResult { item_id, run: run_id, result: Ok(UpscaleOutput { run, shown }), recipe });
            }
            super::item::Engine::Colorize => {
                let n = (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize;
                let run = super::item::ColorizeCache::new(Arc::new(vec![0.0; 2 * n]), Arc::clone(&shown));
                let _ = self.colorize.tx.send(ColorizeResult { item_id, run: run_id, result: Ok(ColorizeOutput { run, shown }), recipe });
            }
            other => unreachable!("{other:?} is not an upscale or colorize"),
        }
    }

    /// Deliver an upscale or colorize for `item_id` that ended with `err`.
    #[cfg(test)]
    pub(crate) fn end_run_for_test(&self, engine: super::item::Engine, item_id: u64, run: u64, err: prunr_core::CoreError, recipe: ProcessingRecipe) {
        match engine {
            super::item::Engine::Upscale => { let _ = self.upscale.tx.send(UpscaleResult { item_id, run, result: Err(err), recipe }); }
            super::item::Engine::Colorize => { let _ = self.colorize.tx.send(ColorizeResult { item_id, run, result: Err(err), recipe }); }
            other => unreachable!("{other:?} is not an upscale or colorize"),
        }
    }

    /// Colorize `input` on a background thread; the result comes back
    /// through `pump_colorize_results`. Refused (logged) when free RAM is
    /// below the model's working set.
    pub(crate) fn dispatch_colorize(&mut self, item_id: u64, run_id: u64, input: super::item::Pixels, cpu_only: bool, recipe: ProcessingRecipe) -> Result<(), String> {
        let kind = super::settings::COLORIZER;
        let descriptor = prunr_models::descriptor(kind.into()).ok_or("the model is not available")?;
        admitted(descriptor)?;
        let intra_threads = prunr_core::batch::ort_intra_threads(1);
        let run = self.dispatch_progress.begin(self.queued_initial(super::dispatch_progress::ProgressKind::Colorize));
        let done = self.colorize.begin(run.clone());
        let progress = prunr_core::Progress::new(Arc::new(run)).with_cancel(Arc::clone(&self.colorize.cancel));
        let engine_slot = Arc::clone(&self.colorize.warm);
        std::thread::spawn(move || {
            let key = WarmEngineKey { model: kind, cpu_only };
            let result = caught(|| input.read().map_err(prunr_core::CoreError::ImageFormat).and_then(|input| {
                let engine = ensure_warm_engine(&engine_slot, key, intra_threads)?;
                let plane = prunr_core::colorize::colorize_plane(&engine, &input, &progress)?;
                let run = super::item::ColorizeCache::new(Arc::new(plane), input);
                let shown = Arc::new(run.render(recipe.colorize.strength()));
                Ok(ColorizeOutput { run, shown })
            }))
            .unwrap_or_else(|why| Err(prunr_core::CoreError::Inference(why)));
            if let Err(err) = &result {
                tracing::warn!(item_id, %err, "colorize run did not finish");
            }
            done.deliver(ColorizeResult { item_id, run: run_id, result, recipe });
        });
        Ok(())
    }

    /// Cancel the colorize run in flight. The model's one run is not
    /// interrupted; its result is dropped when it returns. Idempotent.
    pub(crate) fn cancel_colorize(&self) {
        if self.colorize.cancel() {
            self.mark_cancelling(super::dispatch_progress::ProgressKind::Colorize);
        }
    }

    /// Cancel the upscale or colorize in flight, whichever it is, and
    /// the images queued behind it: Esc, the capsule's Cancel and a model
    /// change end the whole batch. An undo on one image cancels only its
    /// own run (`cancel_upscale` / `cancel_colorize`) and leaves the
    /// queue to the others.
    pub(crate) fn cancel_runs(&self) {
        self.clear_run_queue();
        self.cancel_upscale();
        self.cancel_colorize();
    }

    fn run_queue(&self) -> std::sync::MutexGuard<'_, RunQueue> {
        self.run_queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Line up `ids` for the run slot, replacing any earlier queue.
    pub(crate) fn queue_runs(&self, ids: Vec<u64>) {
        let mut q = self.run_queue();
        q.total = ids.len() as u32;
        q.started = 0;
        q.waiting = ids.into();
    }

    pub(crate) fn queued_run_count(&self) -> usize {
        self.run_queue().waiting.len()
    }

    /// `item_id`, taken off the queue, waits at the back again.
    pub(crate) fn requeue_run(&self, item_id: u64) {
        let mut q = self.run_queue();
        q.started = q.started.saturating_sub(1);
        q.waiting.push_back(item_id);
    }

    /// The next queued image, to dispatch now; none while a run is in
    /// flight or the queue is empty.
    pub(crate) fn next_queued_run(&self) -> Option<u64> {
        if self.is_run_in_flight() || self.any_inpaint_in_flight() {
            return None;
        }
        let mut q = self.run_queue();
        let next = q.waiting.pop_front()?;
        q.started += 1;
        Some(next)
    }

    /// Where the queue stands, as the capsule's outer count: none when
    /// Process was pressed for one image.
    fn queued_run_position(&self) -> Option<(u32, u32)> {
        let q = self.run_queue();
        (q.total > 1).then_some((q.started.max(1), q.total))
    }

    pub(crate) fn clear_run_queue(&self) {
        *self.run_queue() = RunQueue::default();
    }

    /// A run's first snapshot: "Loading the model", counting images when
    /// it is one of several queued.
    fn queued_initial(&self, kind: super::dispatch_progress::ProgressKind) -> super::dispatch_progress::DispatchProgress {
        let mut initial = super::dispatch_progress::DispatchProgress::new(kind, prunr_core::Step::LoadingModel.label());
        initial.outer = self.queued_run_position();
        initial.outer_unit = initial.outer.map(|_| prunr_core::Unit::Image);
        initial
    }

    /// Drain finished colorize runs. Non-blocking.
    pub(crate) fn pump_colorize_results(&mut self) -> Vec<ColorizeResult> {
        self.colorize.drain()
    }

    /// Dispatch the SAM 2 encoder for `item_id`. Runs once per source image;
    /// the result is cached on `BatchItem.magic_brush_embedding`. Admission
    /// gate: refuses when `available_ram_mb < 800 MB`.
    ///
    /// Returns `Err(reason)` synchronously on admission failure (caller shows
    /// a toast). Returns `Ok(())` after spawning the background rayon job;
    /// the result arrives via `pump_sam_encoder_results`.
    pub(crate) fn dispatch_sam_encoder(
        &self,
        item_id: u64,
        source: std::sync::Arc<image::RgbaImage>,
        available_ram_mb: u32,
    ) -> Result<(), String> {
        const ENCODER_COST_MB: u32 = 800;
        if !admission_check(ENCODER_COST_MB, available_ram_mb) {
            return Err(format!(
                "Magic Brush requires ~{} MB free RAM; only {} MB available.",
                ENCODER_COST_MB, available_ram_mb,
            ));
        }
        let tx = self.sam_encoder_tx.clone();
        let sessions = Arc::clone(&self.sam_sessions);
        self.spawn_model(move || {
            let result = caught(|| ensure_sam_sessions(&sessions).and_then(|s| run_sam_encoder_inline(&s, &source)))
                .unwrap_or_else(Err);
            let _ = tx.send(SamEncoderResult { item_id, result });
        });
        Ok(())
    }

    /// Find and redraw the faces in `photo` with `model` on a thread; the
    /// capsule shows it, and the image the faces as they are found and
    /// redrawn, each blended in at `blend`. The photo is unpacked there,
    /// never here.
    pub(crate) fn dispatch_find_faces(&self, item_id: u64, photo: super::item::FacePhoto, model: prunr_models::ModelId, blend: prunr_core::face::FaceBlend) {
        let tx = self.faces_tx.clone();
        let (stop, current) = self.begin_faces_search();
        // The capsule belongs to a batch or an upscale while one runs; a
        // face search taking it would hide theirs and its Cancel.
        let run = self.dispatch_progress.read().is_none().then(|| {
            self.dispatch_progress.begin(super::dispatch_progress::DispatchProgress::new(super::dispatch_progress::ProgressKind::Faces, prunr_core::Step::FindingFaces.label()))
        });
        let progress = PictureSink::of_face_search(item_id, run.clone(), current, self.picture_sender())
            .into_progress()
            .with_cancel(Arc::clone(&stop));
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = caught(|| {
                photo.load()
                    .ok_or_else(|| "the photo is no longer there".to_string())
                    .and_then(|rgba| prunr_core::face::find_and_restore(&rgba, model, blend, &progress).map_err(|e| e.to_string()))
            })
            .unwrap_or_else(Err);
            tracing::info!(item_id, ?model, faces = result.as_ref().map_or(0, Vec::len), elapsed_ms = started.elapsed().as_millis() as u64, "face search ended");
            if let Some(run) = run {
                run.end();
            }
            let stopped = stop.load(Ordering::Acquire);
            let _ = tx.send(FacesResult { item_id, photo, model, result, stopped });
        });
    }

    /// A stop flag for the face search starting now, which Cancel and Esc
    /// reach, and whether its pictures may still land: while it is the
    /// latest search and not stopped.
    fn begin_faces_search(&self) -> (Arc<AtomicBool>, impl Fn() -> bool + Send + Sync + 'static) {
        let stop = Arc::new(AtomicBool::new(false));
        *self.faces_stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::clone(&stop);
        let search = self.faces_latest.fetch_add(1, Ordering::AcqRel) + 1;
        let (latest, stopped) = (Arc::clone(&self.faces_latest), Arc::clone(&stop));
        (stop, move || latest.load(Ordering::Acquire) == search && !stopped.load(Ordering::Acquire))
    }

    /// Stop the face search under way; it ends after the face it is on.
    pub(crate) fn cancel_faces(&self) {
        self.faces_stop.lock().unwrap_or_else(std::sync::PoisonError::into_inner).store(true, Ordering::Release);
    }

    pub(crate) fn pump_faces_results(&mut self) -> Vec<FacesResult> {
        self.faces_rx.try_iter().collect()
    }

    /// Build the SAM sessions (~180 MB resident, ~1 s parse) on the pool
    /// so the first Magic Brush use never waits for them.
    pub(crate) fn warm_sam_sessions(&self) {
        if self.sam_sessions_ready() {
            return;
        }
        let sessions = Arc::clone(&self.sam_sessions);
        self.spawn_model(move || {
            if let Err(err) = ensure_sam_sessions(&sessions) {
                tracing::warn!(%err, "SAM session warm-up failed");
            }
        });
    }

    /// Drop the warm Magic Brush sessions; a decode in flight keeps its own.
    pub(crate) fn release_sam_sessions(&self) {
        let released = self.sam_sessions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        super::reclaim::later(released);
    }

    pub(crate) fn sam_sessions_ready(&self) -> bool {
        self.sam_sessions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_some()
    }

    /// Dispatch the SAM 2 decoder for a click or stroke. No admission gate —
    /// the decoder model is small (~20 MB) and runs quickly. The embedding
    /// is the cached output from a prior `dispatch_sam_encoder` call.
    /// Decoding the logits into a source-resolution mask happens on the
    /// worker too, so the UI thread only merges the result; the logits
    /// come back with it for later re-thresholding.
    pub(crate) fn dispatch_sam_decoder(&self, req: SamDecodeRequest) {
        let tx = self.sam_decoder_tx.clone();
        let sessions = Arc::clone(&self.sam_sessions);
        self.spawn_model(move || {
            let (w, h) = req.source_dims;
            let started = std::time::Instant::now();
            let result = caught(|| ensure_sam_sessions(&sessions)
                .and_then(|s| run_sam_decoder_inline(&s, &req.embedding, &req.prompt))
                .map(|out| {
                    let decoded = started.elapsed();
                    let mask = prunr_core::sam::decode_to_mask_artifact(&out, w, h, req.reading, req.mode);
                    tracing::debug!(
                        item_id = req.item_id,
                        decoder_ms = decoded.as_millis() as u64,
                        mask_ms = (started.elapsed() - decoded).as_millis() as u64,
                        "SAM decoder ran"
                    );
                    DecodedSelection::Fresh { mask, output: Arc::new(out) }
                }))
                .unwrap_or_else(Err);
            let _ = tx.send(SamDecoderResult {
                item_id: req.item_id, mode: req.mode, reading: req.reading, result,
            });
        });
    }

    /// Re-threshold a kept decoder output on the pool, merge it onto the
    /// pre-stroke selection and hash it there, so the UI thread only swaps
    /// the plane in. The result arrives through the decoder channel.
    pub(crate) fn dispatch_sam_rethreshold(&self, req: SamRethresholdRequest) {
        let tx = self.sam_decoder_tx.clone();
        rayon::spawn(move || {
            let (w, h) = req.source_dims;
            let mask = prunr_core::sam::decode_to_mask_artifact(&req.output, w, h, req.reading, req.mode);
            let base = req.base.as_ref().and_then(super::item::Region::load);
            let merged = Arc::new(merge_stroke(base, mask));
            let hash = merged.content_hash();
            let _ = tx.send(SamDecoderResult {
                item_id: req.item_id, mode: req.mode, reading: req.reading,
                result: Ok(DecodedSelection::Retuned { merged, hash, for_hash: req.for_hash }),
            });
        });
    }

    pub(crate) fn pump_sam_encoder_results(&mut self) -> Vec<SamEncoderResult> {
        self.sam_encoder_rx.try_iter().collect()
    }

    pub(crate) fn pump_sam_decoder_results(&mut self) -> Vec<SamDecoderResult> {
        self.sam_decoder_rx.try_iter().collect()
    }
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    stage: &str,
) -> Result<Vec<f32>, String> {
    Ok(outputs[name]
        .try_extract_array::<f32>()
        .map_err(|e| format!("{stage}: extract {name}: {e}"))?
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset().0)
}

/// Run the SAM 2 encoder on the cached session: preprocess the source,
/// run inference, marshal the three named outputs into a SamEmbedding.
///
/// ORT session.run() lives here so `prunr_core::sam` stays ORT-free.
fn run_sam_encoder_inline(
    sessions: &SamSessions,
    source: &image::RgbaImage,
) -> Result<prunr_core::sam::SamEmbedding, String> {
    use ort::{inputs, value::Tensor};

    // Preprocess before taking the lock: it runs on the rayon pool, and a
    // rayon wait while holding the lock can pick up a queued encode that
    // then blocks on this same lock. Do NOT move rayon work under it.
    let input_vec = prunr_core::sam::preprocess::preprocess_for_sam(source);
    let mut session = sessions.encoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    // Shape: [1, 3, 1024, 1024] — NCHW, ImageNet-normalized f32.
    let arr = ndarray::Array4::from_shape_vec([1, 3, 1024, 1024], input_vec)
        .map_err(|e| format!("encoder input shape: {e}"))?;
    let tensor = Tensor::from_array(arr)
        .map_err(|e| format!("encoder input tensor: {e}"))?;

    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(inputs![input_name => &tensor])
        .map_err(|e| format!("encoder session.run: {e}"))?;

    // Address by NAME — vietanhdev's SAM 2 ONNX export emits outputs
    // in the order high_res_feats_0, high_res_feats_1, image_embed,
    // which is the reverse of the documented logical order. Indexing
    // by name keeps the dispatch immune to export-order changes.
    //
    // Expected shapes (from SAM 2 Hiera Small):
    //   image_embed      [1, 256, 64, 64]   — 1_048_576 f32
    //   high_res_feats_0 [1, 32, 256, 256]  — 2_097_152 f32
    //   high_res_feats_1 [1, 64, 128, 128]  — 1_048_576 f32
    let image_embed = extract_f32(&outputs, "image_embed", "encoder")?;
    let high_res_feats_0 = extract_f32(&outputs, "high_res_feats_0", "encoder")?;
    let high_res_feats_1 = extract_f32(&outputs, "high_res_feats_1", "encoder")?;

    let embedding = prunr_core::sam::SamEmbedding {
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
    };
    embedding.validate_shapes()
        .map_err(|e| format!("encoder output shape mismatch: {e}"))?;
    Ok(embedding)
}

/// Run the SAM 2 decoder on the cached session: pack the embedding +
/// prompt into 7 named inputs, run inference, marshal the two outputs
/// (masks + iou_predictions) into SamDecoderOutput.
///
/// ORT session.run() lives here so `prunr_core::sam` stays ORT-free.
fn run_sam_decoder_inline(
    sessions: &SamSessions,
    embedding: &prunr_core::sam::SamEmbedding,
    prompt: &prunr_core::sam::prompt::SamPrompt,
) -> Result<prunr_core::sam::SamDecoderOutput, String> {
    use ort::{inputs, value::{Tensor, TensorRef}};

    let mut session = sessions.decoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

    // The 16 MB embedding is borrowed for the run — no per-click copy.
    let embed_tensor = TensorRef::from_array_view(
        ([1usize, 256, 64, 64], embedding.image_embed.as_slice()),
    ).map_err(|e| format!("decoder: image_embed tensor: {e}"))?;
    let feats0_tensor = TensorRef::from_array_view(
        ([1usize, 32, 256, 256], embedding.high_res_feats_0.as_slice()),
    ).map_err(|e| format!("decoder: high_res_feats_0 tensor: {e}"))?;
    let feats1_tensor = TensorRef::from_array_view(
        ([1usize, 64, 128, 128], embedding.high_res_feats_1.as_slice()),
    ).map_err(|e| format!("decoder: high_res_feats_1 tensor: {e}"))?;

    // point_coords [1, N, 2]
    let n_points = prompt.point_labels.len();
    let coords_arr = ndarray::Array3::from_shape_vec(
        [1, n_points, 2],
        prompt.point_coords.clone(),
    ).map_err(|e| format!("decoder: point_coords shape: {e}"))?;
    let coords_tensor = Tensor::from_array(coords_arr)
        .map_err(|e| format!("decoder: point_coords tensor: {e}"))?;

    // point_labels [1, N]
    let labels_arr = ndarray::Array2::from_shape_vec(
        [1, n_points],
        prompt.point_labels.clone(),
    ).map_err(|e| format!("decoder: point_labels shape: {e}"))?;
    let labels_tensor = Tensor::from_array(labels_arr)
        .map_err(|e| format!("decoder: point_labels tensor: {e}"))?;

    // mask_input [1, 1, 256, 256]
    let mask_arr = ndarray::Array4::from_shape_vec(
        [1, 1, 256, 256],
        prompt.mask_input.clone(),
    ).map_err(|e| format!("decoder: mask_input shape: {e}"))?;
    let mask_tensor = Tensor::from_array(mask_arr)
        .map_err(|e| format!("decoder: mask_input tensor: {e}"))?;

    // has_mask_input scalar [1]
    let has_mask_arr = ndarray::Array1::from_vec(vec![prompt.has_mask_input]);
    let has_mask_tensor = Tensor::from_array(has_mask_arr)
        .map_err(|e| format!("decoder: has_mask_input tensor: {e}"))?;

    let outputs = session.run(inputs![
        "image_embed"       => embed_tensor,
        "high_res_feats_0"  => feats0_tensor,
        "high_res_feats_1"  => feats1_tensor,
        "point_coords"      => &coords_tensor,
        "point_labels"      => &labels_tensor,
        "mask_input"        => &mask_tensor,
        "has_mask_input"    => &has_mask_tensor,
    ]).map_err(|e| format!("decoder session.run: {e}"))?;

    // Outputs: masks [1, 3, 256, 256] and iou_predictions [1, 3].
    // Address by name for the same export-order resilience as the
    // encoder path above.
    let masks = extract_f32(&outputs, "masks", "decoder")?;
    let iou_raw = extract_f32(&outputs, "iou_predictions", "decoder")?;

    if iou_raw.len() != 3 {
        return Err(format!("decoder: expected 3 IoU predictions, got {}", iou_raw.len()));
    }
    let iou_predictions = [iou_raw[0], iou_raw[1], iou_raw[2]];

    Ok(prunr_core::sam::SamDecoderOutput { masks, iou_predictions })
}

/// The SD pipeline's request for an erase under `tuning`; `None` for an
/// eraser outside the SD family.
pub(crate) fn sd_request(tuning: &InpaintTuning) -> Option<prunr_core::inpaint_sd::SdInpaintRequest> {
    use prunr_core::inpaint_sd::{SchedulerKind, SdInpaintRequest};
    let base = SdInpaintRequest {
        seed: tuning.sd_seed,
        tuning: prunr_core::inpaint_sd::SdTuning { cpu_only: tuning.cpu_only, ..tuning.sd_tuning.clone() },
        fill_scale: tuning.fill_scale,
        ..Default::default()
    };
    match tuning.model {
        prunr_models::ModelId::SdV15InpaintFp16
        | prunr_models::ModelId::SdV15LcmInpaintFp16 => {
            Some(SdInpaintRequest {
                prompt: tuning.sd_prompt.clone(),
                negative_prompt: tuning.sd_negative_prompt.clone(),
                num_inference_steps: tuning.sd_steps,
                guidance_scale: effective_cfg(tuning.model, tuning.sd_guidance_scale),
                use_taesd: tuning.use_taesd,
                scheduler: tuning.sd_scheduler.into(),
                ..base
            })
        }
        // The prompt is baked in: the user's prompt, guidance and TAESD
        // do not apply.
        prunr_models::ModelId::RoremFp16 => Some(SdInpaintRequest {
            num_inference_steps: tuning.rorem_steps,
            scheduler: SchedulerKind::Lcm,
            ..base
        }),
        _ => None,
    }
}

#[cfg(test)]
mod panic_tests {
    /// A panic in a model run becomes that run's error; a run that
    /// returns is untouched. On the pool, a panicking job no longer takes
    /// the process down.
    #[test]
    fn a_panic_in_a_model_run_fails_the_run_not_the_app() {
        assert_eq!(super::caught(|| 7), Ok(7));
        let failed = super::caught(|| -> u32 { panic!("a driver bug") });
        assert!(failed.unwrap_err().starts_with("The model stopped unexpectedly"));
        let (tx, rx) = std::sync::mpsc::channel();
        rayon::spawn(move || {
            let result = super::caught(|| -> u32 { panic!("on the pool") });
            let _ = tx.send(result.is_err());
        });
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok(true));
    }
}

#[cfg(test)]
mod sd_request_tests {
    use super::*;

    /// RORem runs the steps its own chip sets, whatever the SD chips say;
    /// the SD 1.5 settings never reach it, the seed does (for Try another).
    #[test]
    fn rorem_runs_its_own_steps_and_keeps_the_seed() {
        let tuning = InpaintTuning {
            model: prunr_models::ModelId::RoremFp16,
            sd_prompt: "a cat".into(),
            sd_guidance_scale: 7.5,
            sd_scheduler: super::super::brush_state::SdScheduler::DpmPlusPlus2MKarras,
            sd_steps: 25,
            rorem_steps: 6,
            sd_seed: Some(42),
            use_taesd: true,
            ..Default::default()
        };
        let req = sd_request(&tuning).expect("RORem runs through the SD pipeline");
        assert_eq!(req.num_inference_steps, 6, "its own steps, not SD's");
        assert_eq!(req.scheduler, prunr_core::inpaint_sd::SchedulerKind::Lcm);
        assert!(req.prompt.is_empty() && req.guidance_scale <= 1.0 && !req.use_taesd);
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn non_sd_erasers_have_no_sd_request() {
        assert!(sd_request(&InpaintTuning::default()).is_none());
    }
}

#[cfg(test)]
mod sam_dispatch_tests {
    use super::*;

    /// Magic Brush end to end on the installed SAM 2 bundle: encode a
    /// photo, decode a click on its subject. Prints the timings and a
    /// digest of the candidates so two runtimes can be compared.
    #[test]
    #[ignore = "needs the SAM 2 bundle; seconds on a CPU"]
    fn sam_encodes_and_decodes_a_click() {
        let Some(path) = std::env::var_os("PRUNR_SAM_PHOTO") else { return };
        let photo = image::open(path).expect("photo").to_rgba8();
        let (w, h) = photo.dimensions();
        let sessions = ensure_sam_sessions(&SamSessionSlot::default()).expect("SAM 2 installed");
        let started = std::time::Instant::now();
        let embedding = run_sam_encoder_inline(&sessions, &photo).expect("encode");
        let encoded = started.elapsed();
        let prompt = prunr_core::sam::prompt::build_click_prompt(w as f32 / 2.0, h as f32 / 3.0, w, h);
        let decoded = run_sam_decoder_inline(&sessions, &embedding, &prompt).expect("decode");
        let positive = decoded.masks.iter().filter(|&&v| v > 0.0).count();
        eprintln!("sam encode {:.2}s decode {:.3}s iou {:?} positive {positive}",
            encoded.as_secs_f32(), (started.elapsed() - encoded).as_secs_f32(), decoded.iou_predictions);
        assert!(positive > 0, "the click selects something");
    }

    /// A stroke adds to or takes from the selection, on every model.
    #[test]
    fn a_stroke_adds_to_or_takes_from_the_selection() {
        use prunr_core::selection::MaskArtifact;
        let add = MaskArtifact::from_cells(2, 1, vec![100, 100]);
        let take = MaskArtifact::from_cells(2, 1, vec![-127, 0]);
        let selected = merge_stroke(None, add);
        assert!(selected.has_selected_region(), "a first Add stroke is the selection");
        let carved = merge_stroke(Some(Arc::new(selected)), take.clone());
        assert_eq!(carved.cells()[0], 0, "Subtract carves it away");
        assert_ne!(carved.cells()[1], 0, "the rest stays");
        assert!(!merge_stroke(None, take).has_selected_region(), "subtracting from nothing selects nothing");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::ModelId;

    /// An erase lays the photo it starts from under its tiles: the whole
    /// photo as one picture, at the preview's size, as it is when it fits.
    #[test]
    fn an_erase_lays_its_photo_under_its_tiles() {
        use prunr_core::progress::RecordingSink;
        use prunr_core::{Progress, ProgressUpdate};
        let sink = Arc::new(RecordingSink::default());
        let progress = Progress::new(sink.clone()).with_tile_previews(crate::subprocess::protocol::TILE_PREVIEW_EDGE);
        let small = image::RgbaImage::from_fn(30, 20, |x, y| image::Rgba([x as u8, y as u8, 7, 200]));
        preview_the_photo(&progress, &image::RgbaImage::from_pixel(3072, 1000, image::Rgba([10, 20, 30, 255])));
        preview_the_photo(&progress, &small);
        let pictures: Vec<_> = sink.updates().into_iter().map(|u| match u {
            ProgressUpdate::TilePreview(p) => p,
            u => panic!("{u:?}"),
        }).collect();
        assert_eq!((pictures[0].x, pictures[0].y, pictures[0].width, pictures[0].height, pictures[0].full_width, pictures[0].full_height), (0, 0, 1536, 500, 1536, 500));
        assert!(pictures[0].rgba.chunks(4).all(|p| p == [10, 20, 30, 255]));
        assert_eq!((pictures[1].width, pictures[1].height, pictures[1].full_width), (30, 20, 30));
        assert_eq!(pictures[1].rgba, small.into_raw(), "a photo that fits is laid as it is");
    }

    /// A tile painted again in the same place before the frame drains
    /// lands once, its newest picture where it came; other places, other
    /// items and other pictures all stay.
    #[test]
    fn a_tile_painted_twice_before_a_frame_lands_once() {
        use prunr_core::{ProgressUpdate, TilePreview};
        let tile = |x: u32, v: u8| ProgressUpdate::TilePreview(TilePreview { x, y: 0, width: 2, height: 2, full_width: 8, full_height: 8, rgba: vec![v; 16] });
        let run = |item_id, update| RunPicture::new(item_id, PictureFrom::Run, update);
        let face = |item_id, update| RunPicture::new(item_id, PictureFrom::FaceSearch, update);
        let mut p = fixture();
        let tx = p.picture_sender();
        for picture in [run(1, tile(0, 1)), run(1, tile(4, 2)), run(2, tile(0, 3)), face(1, tile(0, 5)), face(1, ProgressUpdate::Faces(Vec::new())), run(1, tile(0, 4))] {
            tx.send(picture).expect("the processor holds the receiver");
        }
        assert_eq!(
            p.pump_pictures(),
            vec![run(1, tile(4, 2)), run(2, tile(0, 3)), face(1, tile(0, 5)), face(1, ProgressUpdate::Faces(Vec::new())), run(1, tile(0, 4))],
        );
        assert!(p.pump_pictures().is_empty());
    }

    /// A run's pictures reach the canvas's channel while the run says
    /// they may land, and its steps reach the pill; a run another has
    /// replaced sends neither; a run without the pill still sends its
    /// pictures but leaves the pill to the run that has it.
    #[test]
    fn a_picture_sink_sends_pictures_while_its_run_is_current() {
        use super::super::dispatch_progress::{DispatchProgress, DispatchProgressSlot, ProgressKind};
        use prunr_core::{ProgressSink, ProgressUpdate, Step, TilePreview};
        let tile = || ProgressUpdate::TilePreview(TilePreview { x: 0, y: 0, width: 1, height: 1, full_width: 1, full_height: 1, rgba: vec![9; 4] });
        let slot = DispatchProgressSlot::new();
        let (tx, rx) = mpsc::channel();
        let first = PictureSink::of_run(4, slot.begin(DispatchProgress::new(ProgressKind::Inpaint, "Loading")), tx.clone());
        first.report(tile());
        first.report(ProgressUpdate::Step(Step::Erasing));
        let run = |item_id| RunPicture::new(item_id, PictureFrom::Run, tile());
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![run(4)]);
        assert_eq!(slot.read().map(|p| p.step_label), Some(Step::Erasing.label().into()));

        let second = PictureSink::of_run(5, slot.begin(DispatchProgress::new(ProgressKind::Upscale, "Loading")), tx.clone());
        first.report(tile());
        first.report(ProgressUpdate::Step(Step::Blending));
        assert!(rx.try_recv().is_err(), "a replaced run's pictures are dropped");
        assert_eq!(slot.read().map(|p| p.step_label), Some("Loading".into()));

        let open = std::sync::Arc::new(AtomicBool::new(true));
        let flag = open.clone();
        let no_pill = PictureSink::of_face_search(6, None, move || flag.load(Ordering::Acquire), tx);
        no_pill.report(ProgressUpdate::Step(Step::Finishing));
        no_pill.report(tile());
        assert_eq!(slot.read().map(|p| p.step_label), Some("Loading".into()), "the pill stays with its run");
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![RunPicture::new(6, PictureFrom::FaceSearch, tile())]);
        let previews = no_pill.clone().into_progress();
        assert!(previews.preview_edge().is_some());
        open.store(false, Ordering::Release);
        no_pill.report(tile());
        assert!(rx.try_recv().is_err(), "its own check stops its pictures");
        assert_eq!(previews.preview_edge(), None, "and stops a run making them");
        second.report(tile());
        assert_eq!(rx.try_iter().count(), 1);
    }

    /// Picking a cut-out model loads it once it has stayed picked a second:
    /// a quick second pick replaces the first, a run under way drops it
    /// (its own worker stays warm), and a release clears it.
    #[test]
    fn a_picked_cutout_model_loads_once_the_picking_stops() {
        use prunr_core::ModelKind;
        use std::time::{Duration, Instant};
        let (tx, rx) = mpsc::channel();
        let (_result_tx, result_rx) = mpsc::channel();
        let mut p = Processor::new(tx, result_rx);
        let cfg = |model| EngineSpec { model, jobs: 1, cpu_only: false };
        let warmed = |rx: &mpsc::Receiver<WorkerMessage>| rx.try_iter().filter_map(|m| match m {
            WorkerMessage::Warm(c) => Some(c.model),
            _ => None,
        }).collect::<Vec<_>>();
        let t0 = Instant::now();
        p.schedule_seg_warm(cfg(ModelKind::U2net), t0);
        p.schedule_seg_warm(cfg(ModelKind::BiRefNetLite), t0 + Duration::from_millis(300));
        let wait = p.pump_seg_warm(t0 + Duration::from_millis(900));
        assert_eq!(wait, Some(Duration::from_millis(400)), "the repaint lands when it is due");
        assert!(warmed(&rx).is_empty());
        assert_eq!(p.pump_seg_warm(t0 + Duration::from_millis(1300)), None);
        assert_eq!(warmed(&rx), vec![ModelKind::BiRefNetLite], "only the last pick loads, once");
        assert_eq!(p.pump_seg_warm(t0 + Duration::from_secs(5)), None);
        assert!(warmed(&rx).is_empty());

        p.schedule_seg_warm(cfg(ModelKind::Silueta), t0);
        p.track_dispatch(crate::gui::item_settings::ItemSettings::default().current_recipe(ModelKind::Silueta), [1]);
        assert_eq!(p.pump_seg_warm(t0 + Duration::from_secs(2)), None);
        assert!(warmed(&rx).is_empty(), "a run under way drops it");

        p.schedule_seg_warm(cfg(ModelKind::Silueta), t0);
        p.release_seg_warm();
        p.in_flight.clear();
        assert_eq!(p.pump_seg_warm(t0 + Duration::from_secs(2)), None);
        assert!(warmed(&rx).is_empty(), "a release clears it");
    }

    /// Cancel reaches the face search under way, and a new search starts
    /// with a flag of its own. A search's pictures land while it is the
    /// latest and not stopped.
    #[test]
    fn cancel_stops_the_latest_face_search() {
        let p = fixture();
        let (first, first_current) = p.begin_faces_search();
        assert!(first_current());
        p.cancel_faces();
        assert!(first.load(Ordering::Acquire));
        assert!(!first_current(), "a stopped search's pictures are dropped");
        let (next, next_current) = p.begin_faces_search();
        assert!(!next.load(Ordering::Acquire), "a new search is not born cancelled");
        assert!(next_current());
        let (_, newest_current) = p.begin_faces_search();
        assert!(!next_current() && newest_current(), "an earlier search's pictures are dropped");
    }

    /// A face search's pictures (the faces found, each one redrawn) reach
    /// the canvas's channel while it is the latest search, with no pill
    /// of its own when another run has it.
    #[test]
    fn a_face_searchs_pictures_flow_while_it_is_the_latest() {
        use prunr_core::{ProgressSink, ProgressUpdate};
        let p = fixture();
        let (_, current) = p.begin_faces_search();
        let sink = PictureSink::of_face_search(3, None, current, p.picture_sender());
        sink.report(ProgressUpdate::Faces(Vec::new()));
        assert_eq!(p.picture_rx.try_iter().collect::<Vec<_>>(), vec![RunPicture::new(3, PictureFrom::FaceSearch, ProgressUpdate::Faces(Vec::new()))]);
        let _next = p.begin_faces_search();
        sink.report(ProgressUpdate::Faces(Vec::new()));
        assert!(p.picture_rx.try_recv().is_err(), "a newer search took over");
    }

    #[test]
    fn is_upscale_in_flight_false_when_idle() {
        let p = fixture();
        assert!(!p.is_upscale_in_flight(),
            "idle Processor must not report an in-flight upscale");
    }

    #[test]
    fn is_upscale_in_flight_true_when_active_flag_set() {
        let p = fixture();
        p.upscale.active.store(true, Ordering::Release);
        assert!(p.is_upscale_in_flight());
    }

    /// The batch's image count comes from its own bookkeeping, steps from
    /// the worker; ending the batch must not clear another kind of run.
    /// A face search leaves the capsule to a batch already running, and
    /// a search that fails still lands.
    #[test]
    fn a_face_search_leaves_a_running_batchs_capsule_alone() {
        use crate::gui::dispatch_progress::{DispatchProgress, ProgressKind};
        let mut p = fixture();
        let _batch = p.dispatch_progress.begin(DispatchProgress::new(ProgressKind::Seg, "Finding the subject"));
        // Not a face model: the search fails at once, without loading anything.
        p.dispatch_find_faces(1, crate::gui::item::FacePhoto::Original(Arc::new(image::RgbaImage::new(8, 8))), prunr_models::ModelId::Silueta, prunr_core::face::FaceBlend::default());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let landed = loop {
            if let Some(r) = p.pump_faces_results().pop() { break r; }
            assert!(std::time::Instant::now() < deadline, "the search never landed");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert!(landed.result.is_err());
        assert_eq!(p.dispatch_progress().map(|d| d.kind), Some(ProgressKind::Seg), "the batch's capsule stays");
    }

    #[test]
    fn a_batch_counts_its_images_and_keeps_the_reported_step() {
        use crate::gui::dispatch_progress::{DispatchProgress, ProgressKind};
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [1, 2, 3].iter().copied());
        p.apply_seg_report(prunr_core::ProgressUpdate::Step(prunr_core::Step::FindingSubject));
        assert!(p.take_recipe(b, 1).is_some());
        let shown = p.dispatch_progress().unwrap();
        assert_eq!((shown.outer, shown.step_label.as_ref()), (Some((1, 3)), "Finding the subject"));
        p.track_streamed(4);
        assert_eq!(p.dispatch_progress().unwrap().outer, Some((1, 4)), "a streamed image grows the total");

        p.drain_recipes();
        assert!(p.dispatch_progress().is_none());
        let _upscale = p.dispatch_progress.begin(DispatchProgress::new(ProgressKind::Upscale, "Upscaling"));
        p.end_seg_run();
        p.apply_seg_report(prunr_core::ProgressUpdate::Step(prunr_core::Step::RefiningEdges));
        assert_eq!(p.dispatch_progress().unwrap().step_label.as_ref(), "Upscaling", "a finished batch reaches no other run");
    }

    /// A working set fits when it is no larger than free RAM; nothing
    /// fits in none.
    #[test]
    fn admission_fits_a_working_set_up_to_free_ram() {
        for (working_set, free, fits) in [(2000, 1500, false), (500, 1500, true), (1500, 1500, true), (600, 0, false)] {
            assert_eq!(admission_check(working_set, free), fits, "{working_set} MB in {free} MB free");
        }
    }

    /// LCM weights clamp CFG to [1.0, 2.0]; the LCM scheduler without the
    /// LCM bundle runs standard SD weights, so CFG is not clamped.
    #[test]
    fn only_lcm_weights_clamp_cfg() {
        for (model, cfg, want) in [
            (ModelId::SdV15LcmInpaintFp16, 7.5, 2.0),
            (ModelId::SdV15LcmInpaintFp16, 1.5, 1.5),
            (ModelId::SdV15InpaintFp16, 7.5, 7.5),
        ] {
            assert!((effective_cfg(model, cfg) - want).abs() < 1e-6, "{model:?} at {cfg}");
        }
    }


    fn fixture() -> Processor {
        let (tx, _rx_unused) = mpsc::channel::<WorkerMessage>();
        let (_tx_unused, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }


    /// Queued images come out in order, one per free slot, counted for
    /// the capsule; a cancel drops the rest.
    #[test]
    fn queued_runs_come_out_in_order_and_a_cancel_drops_them() {
        let p = fixture();
        assert_eq!(p.next_queued_run(), None);
        p.queue_runs(vec![4, 5, 6]);
        assert_eq!(p.next_queued_run(), Some(4));
        assert_eq!(p.queued_run_position(), Some((1, 3)));
        p.set_upscale_in_flight_for_test(true);
        assert_eq!(p.next_queued_run(), None, "the slot is taken");
        p.set_upscale_in_flight_for_test(false);
        assert_eq!(p.next_queued_run(), Some(5));
        assert_eq!(p.queued_run_position(), Some((2, 3)));
        p.cancel_upscale();
        assert_eq!(p.next_queued_run(), Some(6), "one image's own cancel leaves the others queued");
        p.cancel_runs();
        assert_eq!(p.next_queued_run(), None, "cancel drops the queue");
        p.queue_runs(vec![9]);
        assert_eq!(p.queued_run_position(), None, "one image counts nothing");
    }

    #[test]
    fn drain_filters_stale_generations() {
        let mut p = fixture();
        p.inpaint_latest_generation.insert(7, 2);
        p.inpaint_tx.send((7, InpaintOutcome::Done {
            rgba: image::RgbaImage::new(1, 1), fill_scale: 1.0, generation: 1,
        })).unwrap();
        p.inpaint_tx.send((7, InpaintOutcome::Done {
            rgba: image::RgbaImage::new(2, 2), fill_scale: 1.0, generation: 2,
        })).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert_eq!(drained.len(), 1, "stale generation=1 must be dropped");
        assert_eq!(drained[0].rgba.dimensions(), (2, 2), "the generation=2 fill lands");
        assert!(cancelled.is_empty(), "no cancellation events expected");
        assert!(errors.is_empty(), "no dispatch errors expected");
    }

    #[test]
    fn drain_routes_cancelled_results_to_cancellation_list() {
        let mut p = fixture();
        p.inpaint_latest_generation.insert(7, 5);
        p.inpaint_pending.insert(7, 1);
        p.inpaint_tx.send((7, InpaintOutcome::Cancelled)).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert!(drained.is_empty(),
            "cancelled stroke must not commit a result");
        assert_eq!(cancelled, vec![7],
            "cancellation event must surface for the toast");
        assert!(errors.is_empty(),
            "cancel must not be reported as a dispatch error");
    }

    #[test]
    fn drain_routes_dispatch_errors_to_error_list() {
        let mut p = fixture();
        p.inpaint_pending.insert(7, 1);
        p.inpaint_tx.send((7, InpaintOutcome::Failed(
            "SD inpaint refused to load: only 13.4 GB free".to_string(),
        ))).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert!(drained.is_empty(),
            "errored stroke must not commit a result");
        assert!(cancelled.is_empty(),
            "errored stroke is not a cancel");
        assert_eq!(errors.len(), 1,
            "dispatch error must surface for the user toast");
        assert_eq!(errors[0].0, 7);
        assert!(errors[0].1.contains("13.4 GB"));
    }

    /// An admission controller with `budget` and the images' costs queued.
    fn controller(budget: usize, costs: &[(u64, usize)]) -> AdmissionController {
        use crate::gui::memory::ImageMemCost;
        let mut ctrl = AdmissionController::with_budget(budget);
        ctrl.enqueue(costs.iter().map(|&(item_id, total)| ImageMemCost { item_id, total }).collect());
        ctrl
    }

    fn test_work(id: u64) -> WorkItem {
        WorkItem { id, run: 0, source: crate::gui::item::ImageSource::Bytes(Arc::new(Vec::new())), image: None }
    }

    #[test]
    fn clear_admission_drops_both_sides_but_leaves_cancels() {
        let mut p = fixture();
        let (tx, _rx) = mpsc::channel::<WorkItem>();
        p.admission_tx = Some(tx);
        p.cancels.request_global_cancel();
        assert!(p.admission_tx.is_some());

        p.clear_admission();

        assert!(p.admission.is_none());
        assert!(p.admission_tx.is_none());
        assert!(p.cancels.is_cancelled(999),
            "clear_admission must leave cancel registry untouched — that's the caller's protocol");
    }

    /// Streamed admission: images beyond the RAM budget go one at a time,
    /// the next sent as one completes and counted into the batch; one
    /// that can't be prepared gives its turn back; admission ends with
    /// the last one through.
    #[test]
    fn admission_streams_the_next_image_as_one_completes() {
        let mut p = fixture();
        let ctrl = controller(1, &[(1, 4), (2, 3), (3, 2)]);
        let work = |id: u64| Some(test_work(id));
        let (admitted, rx) = p.open_admission(ctrl, work);
        let admitted: Vec<u64> = admitted.into_iter().map(|w| w.id).collect();
        assert_eq!(admitted, vec![3], "only the smallest is forced in");
        p.track_dispatch(fixture_recipe(), admitted);

        p.release_and_admit(3, work);
        assert_eq!(rx.try_recv().map(|w| w.id), Ok(2), "the next smallest is sent");
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)), "and counted");
        assert!(p.admission.is_some(), "the batch is still open");

        p.release_and_admit(2, |_| None);
        assert!(rx.try_recv().is_err(), "an image that can't be prepared is not sent");
        assert!(p.admission.is_none() && p.admission_tx.is_none(), "and holds no budget: every image is through");
    }

    /// An image admitted at the start that cannot be sent frees its share
    /// right there, so the next image takes it and the batch can end.
    #[test]
    fn an_image_that_cannot_start_frees_its_admission_at_once() {
        let mut p = fixture();
        let ctrl = controller(4, &[(1, 3), (2, 3), (3, 3)]);
        let (items, _rx) = p.open_admission(ctrl, |id| (id != 1).then(|| test_work(id)));
        let sent: Vec<u64> = items.iter().map(|w| w.id).collect();
        assert_eq!(sent, vec![2], "1 fits first but can't be sent; its share lets 2 in");
        p.track_dispatch(fixture_recipe(), sent);
        let work = |id: u64| Some(test_work(id));
        p.release_and_admit(2, work);
        assert_eq!(_rx.try_recv().map(|w| w.id), Ok(3), "3 goes out behind 2");
        p.release_and_admit(3, work);
        assert!(p.admission.is_none(), "nothing holds the budget: 1's share was given back");
    }

    /// An image admission sends joins the batch admission feeds, not a
    /// newer batch opened meanwhile, so its answer is taken.
    #[test]
    fn an_admitted_image_joins_the_batch_admission_feeds() {
        let mut p = fixture();
        let ctrl = controller(1, &[(1, 2), (2, 1)]);
        let (first, _rx) = p.open_admission(ctrl, |id| Some(test_work(id)));
        let fed = p.track_dispatch(fixture_recipe(), first.iter().map(|w| w.id));
        p.admission_feeds(fed);
        let newer = p.track_dispatch(fixture_recipe(), [9]);
        assert!(p.release_and_admit(2, |id| Some(test_work(id))).sent == vec![1]);
        assert!(p.take_recipe(newer, 1).is_none(), "not the newer batch's");
        assert!(p.take_recipe(fed, 1).is_some(), "the fed batch takes its answer");
    }

    /// When the worker has gone, an admitted image is not sent: it comes
    /// back as lost, its share freed; and the batch admission fed ending
    /// hands back the images never admitted.
    #[test]
    fn a_batch_that_ends_hands_back_what_admission_still_held() {
        let mut p = fixture();
        let ctrl = controller(1, &[(1, 3), (2, 2), (3, 1)]);
        let (first, rx) = p.open_admission(ctrl, |id| Some(test_work(id)));
        let fed = p.track_dispatch(fixture_recipe(), first.iter().map(|w| w.id));
        p.admission_feeds(fed);
        drop(rx);
        let admitted = p.release_and_admit(first[0].id, |id| Some(test_work(id)));
        assert!(admitted.sent.is_empty() && admitted.lost.len() == 1, "{admitted:?}");
        assert_eq!(p.end_admission_with(fed + 1), Vec::<u64>::new(), "another batch's end leaves admission");
        assert_eq!(p.end_admission_with(fed).len(), 1, "the one never admitted comes back");
        assert!(p.admission.is_none() && !p.admission_still_feeding());
    }

    /// A re-run asked for while admission feeds waits, and goes once it is
    /// through.
    #[test]
    fn a_rerun_waits_while_admission_feeds() {
        let mut p = fixture();
        let ctrl = controller(1, &[(1, 2), (2, 1)]);
        let (first, _rx) = p.open_admission(ctrl, |id| Some(test_work(id)));
        p.track_dispatch(fixture_recipe(), first.iter().map(|w| w.id));
        p.defer_run(vec![7], None);
        assert!(p.take_deferred_runs().is_empty(), "held while admission feeds");
        p.clear_admission();
        assert_eq!(p.take_deferred_runs().len(), 1, "then it goes");
    }

    #[test]
    fn cancel_registry_clone_shares_state() {
        // Cloned into WorkerMessage::BatchProcess and read by the bridge —
        // a store on the parent must be visible via any clone.
        let r = CancelRegistry::new();
        let handle = r.clone();
        assert!(!handle.is_cancelled(5));
        r.request_global_cancel();
        assert!(handle.is_cancelled(5), "Clone must observe the global store");
    }

    #[test]
    fn cancel_registry_per_item_is_independent_of_global() {
        let r = CancelRegistry::new();
        r.request_item_cancel(42);
        assert!(r.is_cancelled(42));
        assert!(!r.is_cancelled(7), "per-item cancel must not leak to other ids");
    }

    /// Cancel All, then Process before the running batch looked: the new
    /// dispatch's reset must not take the cancel back from that batch.
    #[test]
    fn a_reset_never_takes_back_a_cancel_all_a_running_batch_has_not_seen() {
        let mut r = CancelRegistry::new();
        let running = r.clone();
        r.request_global_cancel();
        r.reset();
        assert!(running.is_global_cancelled(), "the running batch still sees its Cancel All");
        assert!(!r.is_global_cancelled(), "the next dispatch starts clean");
        let next = r.clone();
        assert!(!next.is_cancelled(3));
        r.request_global_cancel();
        assert!(next.is_global_cancelled(), "and a later Cancel All reaches it");
    }

    #[test]
    fn cancel_registry_reset_clears_all_flags() {
        let mut r = CancelRegistry::new();
        r.request_global_cancel();
        r.request_item_cancel(42);
        r.reset();
        assert!(!r.is_cancelled(42));
        assert!(!r.is_cancelled(99));
    }

    fn fixture_recipe() -> ProcessingRecipe {
        use prunr_core::{
            CompositeRecipe, EdgeRecipe, EdgeScale, ComposeMode, SubjectEffect, InferenceRecipe,
            InputTransform, LineStyle, MaskSettings, ModelKind, UpscaleRecipe,
        };
        ProcessingRecipe {
            inference: InferenceRecipe {
                model: ModelKind::Silueta,
                uses_segmentation: true,
                uses_edge_detection: false,
                input_transform: InputTransform::None,
            },
            edge: EdgeRecipe {
                line_strength_bits: 0.5f32.to_bits(),
                solid_line_color: None,
                edge_thickness: 0,
                edge_scale: EdgeScale::Fused,
                compose_mode: ComposeMode::LinesOnly,
                line_style: LineStyle::Solid,
                line_fade_pct: 0,
            },
            mask: (&MaskSettings { effect: SubjectEffect::None, ..Default::default() }).into(),
            composite: CompositeRecipe::default(),
            upscale: UpscaleRecipe::default(),
            colorize: Default::default(),
            depth: Default::default(),
        }
    }

    #[test]
    fn a_background_removal_run_shows_progress_from_the_start() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [10, 20].iter().copied());
        p.apply_seg_report(prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel));
        let shown = p.dispatch_progress().expect("the capsule shows before any image is done");
        assert_eq!((shown.outer, shown.step_label.as_ref()), (Some((0, 2)), "Loading the model"));
    }

    #[test]
    fn track_dispatch_then_take_returns_recipe_per_item() {
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [10, 20, 30].iter().copied());
        assert!(p.take_recipe(b, 10).is_some());
        assert!(p.take_recipe(b, 20).is_some());
        // Slot still alive while items remain.
        assert!(!p.in_flight.is_empty());
        assert!(p.take_recipe(b, 30).is_some());
        // Last item drains the slot.
        assert!(p.in_flight.is_empty());
    }

    /// A batch queued behind a running one keeps its own tracking: the
    /// first batch's image lands with that batch's recipe, and the first
    /// batch ending leaves the second tracked, its progress showing.
    #[test]
    fn a_queued_batch_keeps_its_tracking_when_the_first_ends() {
        let mut p = fixture();
        let mut second = fixture_recipe();
        second.mask = prunr_core::MaskRecipe::from(&prunr_core::MaskSettings { gamma: 2.0, ..Default::default() });
        let one = p.track_dispatch(fixture_recipe(), [1]);
        let two = p.track_dispatch(second.clone(), [2]);
        assert!(p.take_recipe(one, 1).is_some_and(|r| r != second), "the first batch's image keeps its own recipe");
        assert_eq!(p.current_dispatch_progress(), Some((0, 1)), "the first batch is done; the queued one shows on");
        assert_eq!(p.take_recipe(two, 2), Some(second));
        assert_eq!(p.current_dispatch_progress(), None);

        let first = p.track_dispatch(fixture_recipe(), [3, 4]);
        p.track_streamed(5);
        let queued = p.track_dispatch(fixture_recipe(), [6]);
        p.take_recipe(first, 3);
        let mut unanswered = p.finish_batch(first);
        unanswered.sort();
        assert_eq!(unanswered, vec![4, 5], "its end names its own unanswered images, an admitted one too");
        assert_eq!(p.current_dispatch_progress(), Some((0, 1)), "the queued one still shows");
        assert_eq!(p.finish_batch(queued), vec![6]);
        assert_eq!(p.current_dispatch_progress(), None);
    }

    #[test]
    fn take_recipe_unknown_id_is_none() {
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [1].iter().copied());
        assert!(p.take_recipe(b, 999).is_none(), "unknown id must not return a recipe");
        assert!(p.take_recipe(b + 1, 1).is_none(), "another batch's answer must not take it");
        // Tracked id still works.
        assert!(p.take_recipe(b, 1).is_some());
    }

    #[test]
    fn current_dispatch_progress_is_none_when_idle() {
        let p = fixture();
        assert!(p.current_dispatch_progress().is_none(),
            "idle Processor must report no in-flight dispatch progress");
    }

    #[test]
    fn current_dispatch_progress_reports_done_over_total() {
        // The user's "1 of 2" bug repro: a batch of two items was
        // tracked; one finished; counter must report (1, 2). Scoping
        // to in_flight (rather than whole-batch status_counts) ensures
        // a previously-Done item from a prior dispatch can't inflate
        // total.
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [10, 20].iter().copied());
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)),
            "fresh dispatch starts at 0/total");
        assert!(p.take_recipe(b, 10).is_some());
        assert_eq!(p.current_dispatch_progress(), Some((1, 2)),
            "one item delivered → 1/2");
        assert!(p.take_recipe(b, 20).is_some());
        // Last delivery self-drains the slot, so progress reports None
        // — the seg slot is cleared by the publisher on the next refresh.
        assert!(p.current_dispatch_progress().is_none(),
            "slot drains when the last item completes");
    }

    #[test]
    fn track_streamed_dedup_does_not_double_count() {
        // Re-registering an already-tracked id (defensive against a
        // worker echo) must not inflate total — only first-insert bumps.
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1, 2].iter().copied());
        p.track_streamed(1);
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)),
            "duplicate streamed id must not bump total");
    }

    #[test]
    fn track_streamed_inherits_batch_recipe() {
        // Admission-pool items are added after dispatch; they inherit the
        // current batch's recipe so a late ImageDone for a streamed id
        // still attributes correctly.
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [1].iter().copied());
        p.track_streamed(2);
        assert!(p.take_recipe(b, 1).is_some());
        assert!(p.take_recipe(b, 2).is_some(), "streamed item must have a recipe");
    }

    #[test]
    #[should_panic(expected = "track_streamed called without active batch")]
    fn track_streamed_without_dispatch_panics_in_debug() {
        let mut p = fixture();
        p.track_streamed(99);
    }

    #[test]
    fn drain_recipes_clears_pending() {
        let mut p = fixture();
        let b = p.track_dispatch(fixture_recipe(), [1, 2, 3].iter().copied());
        p.drain_recipes();
        assert!(p.take_recipe(b, 1).is_none(),
            "drain must drop the slot so late deliveries do not land");
    }

    /// A stroke stopped part-way by its cancel reports core's own text;
    /// it reads as a cancel ("Erase cancelled"), not as a failed erase.
    #[test]
    fn an_erase_stopped_part_way_is_a_cancel() {
        let mut p = fixture();
        let (tx, rx) = mpsc::channel();
        p.inpaint_bridge_rx = rx;
        tx.send(crate::gui::inpaint_bridge::InpaintBridgeResult::Error { item_id: 1, error: prunr_core::CoreError::Cancelled.message() }).unwrap();
        p.pump_inpaint_subprocess();
        assert!(matches!(p.inpaint_rx.try_recv(), Ok((1, InpaintOutcome::Cancelled))));
    }

    /// An image a cancelled run left in an older batch awaits only the
    /// newer batch that runs it again: the older run's answer is stale,
    /// the older batch's end does not name it, and its count drops it.
    #[test]
    fn a_newer_batch_takes_an_image_from_the_batch_that_ran_it() {
        let mut p = fixture();
        let old = p.track_dispatch(fixture_recipe(), [1, 2]);
        let new = p.track_dispatch(fixture_recipe(), [1]);
        assert!(p.take_recipe(old, 1).is_none(), "the old run's answer is stale");
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)), "image 1 is counted once");
        assert_eq!(p.finish_batch(old), vec![2]);
        assert!(p.take_recipe(new, 1).is_some(), "the new run's answer lands");

        let only = p.track_dispatch(fixture_recipe(), [3]);
        let again = p.track_dispatch(fixture_recipe(), [3]);
        assert!(p.finish_batch(only).is_empty(), "a batch left with nothing awaits nothing");
        assert_eq!(p.finish_batch(again), vec![3]);
    }

    // ── Brush gate boundary contract ───────────────────────────────
    //
    // `canvas::render` computes:
    //   brush_active = is_enabled && !inpaint_in_flight_for_selected && …
    //
    // If is_inpaint_in_flight returns true for the selected item, brush_active
    // must be false — so handle_brush_input is never entered. This test pins
    // the Processor half of that contract so a future decoupling of
    // is_inpaint_in_flight doesn't silently break the gate.

    #[test]
    fn inpaint_in_flight_blocks_brush_gate_for_selected_item() {
        let mut p = fixture();
        // Simulate an in-flight inpaint for item 42 by bumping the pending counter.
        *p.inpaint_pending.entry(42).or_insert(0) += 1;
        // The gate must block: is_inpaint_in_flight returns true.
        assert!(p.is_inpaint_in_flight(42),
            "pending count > 0 must report in-flight for the canvas brush gate");
        // A different item must not be affected.
        assert!(!p.is_inpaint_in_flight(99),
            "in-flight state must be item-scoped, not global");
    }

    #[test]
    fn a_cancel_marks_only_its_own_kind_of_run_as_stopping() {
        use super::super::dispatch_progress::{DispatchProgress, ProgressKind};
        let mut p = fixture();
        *p.inpaint_pending.entry(7).or_insert(0) += 1;
        let _run = p.dispatch_progress.begin(DispatchProgress::new(ProgressKind::Inpaint, "Erasing"));
        p.mark_cancelling(ProgressKind::Upscale);
        assert!(!p.dispatch_progress().unwrap().cancelling, "an upscale cancel leaves the eraser run alone");
        p.cancel_inpaint(7);
        assert!(p.dispatch_progress().unwrap().cancelling);
        assert!(p.is_inpaint_in_flight(7), "the stroke stays in flight until the worker stops");
    }

    #[test]
    fn zero_cancel_is_cancelled_does_not_touch_mutex() {
        // `is_cancelled` is called ~160×/s from the bridge loop. Until any
        // per-item entry is requested the mutex must stay cold — poisoning
        // the map from another thread and then calling `is_cancelled` on a
        // fresh registry must not panic.
        let r = CancelRegistry::new();
        // Poison the inner mutex from a panicking thread.
        let p = r.per_item.clone();
        let _ = std::thread::spawn(move || {
            let _guard = p.lock().unwrap();
            panic!("deliberate poison");
        }).join();
        // has_per_item is still false → no lock taken → no panic propagation.
        assert!(!r.is_cancelled(42));
    }
}

#[cfg(test)]
mod upscale_dispatch_tests {
    use super::build_inference_input;
    use std::sync::Arc;

    fn grey_image(w: u32, h: u32, v: u8) -> image::RgbaImage {
        image::RgbaImage::from_pixel(w, h, image::Rgba([v, v, v, 255]))
    }

    /// An upscale started before its photo is decoded decodes it in the
    /// upscale thread rather than being skipped.
    #[test]
    fn an_undecoded_upscale_input_decodes_from_its_source() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(grey_image(3, 2, 77)).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let source = crate::gui::item::ImageSource::Bytes(Arc::new(png));
        let img = crate::gui::item::Pixels::Source(source).read().expect("decodes");
        assert_eq!((img.dimensions(), img.get_pixel(0, 0).0[0]), ((3, 2), 77));
        let broken = crate::gui::item::ImageSource::Bytes(Arc::new(vec![1, 2, 3]));
        assert!(crate::gui::item::Pixels::Source(broken).read().is_err());
    }

    /// With Denoise off the input itself goes to the model (no pixel
    /// clone); with it on, a new buffer is made.
    #[test]
    fn dispatch_pre_process_copies_only_to_denoise() {
        let input = Arc::new(grey_image(4, 4, 128));
        assert!(Arc::ptr_eq(&input, &build_inference_input(&input, 0.0)), "no-op path must return the same Arc");
        assert!(!Arc::ptr_eq(&input, &build_inference_input(&input, 0.5)), "denoise path must return a new Arc");
    }

}

// ── Warm-engine cache boundary contract ──────────────────────────────────────
//
// These pin the cache-hit / cache-eviction / release semantics on the upscale
// run's warm engine without exercising real upscale inference: a sentinel
// Arc<OrtEngine> goes in through the test-only accessor, and the tests assert
// pointer identity or slot emptiness.
//
// OrtEngine construction needs the ORT dylib; the tests skip gracefully when
// none is installed.
#[cfg(test)]
mod warm_cache_tests {
    use super::*;
    use prunr_core::{ModelKind, OrtEngine};

    /// Construct one real engine (Silueta, ~4 MB bundled) and reuse it
    /// across the tests via OnceLock. Silueta bytes are always available —
    /// no download required. Returns None when ORT isn't available (no
    /// dylib found).
    fn sentinel_engine() -> Option<Arc<OrtEngine>> {
        prunr_core::ort_runtime::ensure_initialized().ok()?;
        static CELL: std::sync::OnceLock<Arc<OrtEngine>> = std::sync::OnceLock::new();
        Some(
            CELL.get_or_init(|| {
                Arc::new(
                    OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
                        .expect("Silueta is bundled and must construct without errors once ORT is init"),
                )
            })
            .clone(),
        )
    }

    /// A Processor whose warm slot holds the sentinel as a RealESRGAN x4
    /// engine built for Auto, or None (skip) without ORT.
    fn warm(test: &str) -> Option<(Processor, Arc<OrtEngine>)> {
        let Some(sentinel) = sentinel_engine() else {
            eprintln!("[{test}] SKIP: ORT runtime not found");
            return None;
        };
        let (tx, _rx) = mpsc::channel::<WorkerMessage>();
        let (_tx, rx) = mpsc::channel::<WorkerResult>();
        let mut p = Processor::new(tx, rx);
        *p.warm_upscale_engine_for_test() =
            Some((WarmEngineKey { model: ModelKind::RealEsrganX4Plus, cpu_only: false }, Arc::clone(&sentinel)));
        Some((p, sentinel))
    }

    #[test]
    fn cache_hit_reuses_arc_on_same_modelkind() {
        let Some((mut p, sentinel)) = warm("cache_hit_reuses_arc_on_same_modelkind") else { return };
        let got = p
            .try_cached_upscale_engine(ModelKind::RealEsrganX4Plus, false)
            .expect("cache hit must return the sentinel Arc");
        assert!(
            p.try_cached_upscale_engine(ModelKind::RealEsrganX4Plus, true).is_none(),
            "an engine built for Auto is not reused under CPU only",
        );
        assert!(Arc::ptr_eq(&sentinel, &got), "cache hit must return the SAME Arc that was inserted");
    }

    #[test]
    fn model_swap_evicts_cached_arc() {
        let Some((mut p, _)) = warm("model_swap_evicts_cached_arc") else { return };
        assert!(
            p.try_cached_upscale_engine(ModelKind::Nomos8kSchatL, false).is_none(),
            "cache miss on ModelKind mismatch must return None"
        );
        assert!(p.warm_upscale_engine_for_test().is_none(), "slot must be evicted after a ModelKind mismatch lookup");
    }

    #[test]
    fn release_upscale_engine_clears_slot() {
        let Some((mut p, _)) = warm("release_upscale_engine_clears_slot") else { return };
        drop(p.upscale.take_engine());
        assert!(p.warm_upscale_engine_for_test().is_none(), "releasing the upscale engine must clear the slot");
    }

    /// A model switch empties every session slot before it returns; only
    /// the free runs on another thread.
    #[test]
    fn releasing_all_sessions_empties_the_slots_at_once() {
        let Some((mut p, _)) = warm("releasing_all_sessions_empties_the_slots_at_once") else { return };
        p.release_all_sessions();
        assert!(p.warm_upscale_engine_for_test().is_none());
    }
}
