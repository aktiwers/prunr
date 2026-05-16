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
//! - The periodic history-cleanup timestamp.
//!
//! Does NOT own:
//! - The worker bridge thread itself — that's spawned by `worker::spawn_worker`
//!   at app startup. We just hold the channel ends.
//! - `BatchManager` (per the cross-coordinator borrow rule). Methods that
//!   operate on the batch take `&mut BatchManager` per call, never as a field.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use prunr_core::ProcessingRecipe;

use super::inpaint_bridge::{InpaintBridgeMsg, InpaintBridgeResult, spawn_inpaint_bridge};
use super::live_preview::LivePreview;
use super::memory::AdmissionController;
use super::worker::{WorkerMessage, WorkerResult, WorkItem};

#[derive(Clone)]
pub struct CancelRegistry {
    global: Arc<AtomicBool>,
    // Short-circuit for the common zero-cancel case: `is_cancelled` skips the
    // mutex entirely unless some per-item entry has been requested.
    has_per_item: Arc<AtomicBool>,
    per_item: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            global: Arc::new(AtomicBool::new(false)),
            has_per_item: Arc::new(AtomicBool::new(false)),
            per_item: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn is_global_cancelled(&self) -> bool {
        self.global.load(Ordering::Acquire)
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
        self.global.store(true, Ordering::Release);
    }

    pub(crate) fn request_item_cancel(&self, item_id: u64) {
        let mut guard = self.per_item.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.entry(item_id)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .store(true, Ordering::Release);
        self.has_per_item.store(true, Ordering::Release);
    }

    pub(crate) fn reset(&self) {
        self.global.store(false, Ordering::Release);
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
    recipe: ProcessingRecipe,
    pending: HashSet<u64>,
    /// Total items registered for this dispatch — frozen at
    /// `track_dispatch` time. Used by `current_dispatch_progress` to
    /// report "done / total" scoped to the *current* dispatch, so
    /// previously-Done items from prior dispatches don't inflate the
    /// counter when the user reprocesses a single item.
    total: usize,
}

pub(crate) struct InpaintResult {
    pub item_id: u64,
    pub rgba: image::RgbaImage,
    pub generation: u64,
    /// True when the worker exited via `CoreError::Cancelled`. Drives
    /// the "Erase cancelled" toast in `drain_inpaint_results` so the
    /// user gets explicit feedback that Esc / the Cancel button took
    /// effect (vs. a silent failure or a stale-result drop).
    pub cancelled: bool,
    /// Set when the worker returned a non-Cancelled error (e.g. the
    /// SD-bundle RAM guard refused to load). Surfaced as a toast in
    /// `drain_inpaint_results` so the user sees WHY the stroke did
    /// nothing instead of guessing it was lost.
    pub error: Option<String>,
}

/// Result delivered from a background upscale thread back to the main thread.
/// Keyed on `item_id`; only one upscale is in flight at a time (gated by
/// `can_process_intent`), so generation-based stale-drop is not needed today.
/// The `recipe` is captured at dispatch time so the pump can stamp
/// `item.applied_recipe` against what actually ran — drift between the
/// running dispatch and a user mid-edit is invisible to the recipe.
pub(crate) struct UpscaleResult {
    pub item_id: u64,
    pub result: Result<image::RgbaImage, prunr_core::CoreError>,
    pub recipe: ProcessingRecipe,
}

/// Pure admission check: returns `true` when `free_ram_mb >= working_set_mb`.
/// Extracted as a free function so unit tests can exercise it without
/// constructing a `Processor`.
pub(crate) fn admission_check(working_set_mb: u32, free_ram_mb: u32) -> bool {
    free_ram_mb >= working_set_mb
}

/// Build the input image for upscale inference, applying Tier-1 pre-process
/// knobs in the correct order: denoise FIRST, brightness_lift SECOND.
///
/// Order rationale: denoise runs on the raw input so the bilateral filter's
/// edge-preserving math sees the noise statistics it was tuned for; brightness
/// lift runs after so any subsequent amplification operates on the
/// already-smoothed signal (otherwise the lift would amplify whatever noise
/// the denoise was about to remove).
///
/// When neither knob is active (`pre_denoise == 0.0` and `brightness_lift == 0.0`),
/// returns an `Arc::clone` of the input — no allocation, no clone of the pixels.
/// The caller can verify this property with `Arc::ptr_eq` when both are zero.
pub(crate) fn build_inference_input(
    input: &Arc<image::RgbaImage>,
    pre_denoise: f32,
    brightness_lift: f32,
) -> Arc<image::RgbaImage> {
    if pre_denoise > 0.0 || brightness_lift != 0.0 {
        // ONE pixel clone covers both pre-process steps; both steps mutate
        // the scratch buffer in-place so no further allocation is needed.
        let mut scratch: image::RgbaImage = (**input).clone();
        if pre_denoise > 0.0 {
            scratch = prunr_core::denoise::apply_denoise(&scratch, pre_denoise);
        }
        if brightness_lift != 0.0 {
            prunr_core::denoise::apply_brightness_lift(&mut scratch, brightness_lift);
        }
        Arc::new(scratch)
    } else {
        // No pre-process knobs active: no clone, no allocation.
        Arc::clone(input)
    }
}

/// Apply the reciprocal brightness-lift transform to the upscaled output.
/// The lift was applied pre-inference to give the model more dark-region
/// signal; the inverse here restores the user's intended exposure.
///
/// No-op when `brightness_lift == 0.0`.
pub(crate) fn apply_post_inference(img: &mut image::RgbaImage, brightness_lift: f32) {
    if brightness_lift != 0.0 {
        prunr_core::denoise::apply_brightness_lift_inverse(img, brightness_lift);
    }
}

/// Eraser-specific tuning passed from `BrushSettings` into the dispatch.
/// Bundled into a struct to keep `dispatch_inpaint` from sprawling.
#[derive(Clone, Debug)]
pub(crate) struct InpaintTuning {
    pub sharpen: f32,
    pub feather_px: f32,
    pub grow_px: f32,
    /// Which inpaint backend to use (LaMaFp32, BigLaMa, …). For
    /// SD-family backends, the choice between `SdV15InpaintFp16`
    /// (standard SD weights) and `SdV15LcmInpaintFp16` (LCM weights)
    /// is driven by the user's scheduler pick upstream
    /// (`Settings::lcm_routing_active`).
    pub backend: prunr_models::ModelId,
    /// SD-only: text prompt; ignored for LaMa-family backends.
    pub sd_prompt: String,
    pub sd_negative_prompt: String,
    pub sd_guidance_scale: f32,
    /// SD-only: scheduler kind. Carried through to the worker via
    /// `SdInpaintRequest::scheduler` so the right denoise math runs.
    pub sd_scheduler: super::brush_state::SdScheduler,
    /// SD-only: number of denoise steps.
    pub sd_steps: u32,
    /// SD-only: pinned RNG seed for reproducibility. `None` = random.
    pub sd_seed: Option<u64>,
    /// SD-only: inpaint strength in [0, 1]. 1.0 = pure noise init,
    /// fully creative rewrite. <1.0 preserves the source proportionally.
    pub sd_strength: f32,
    /// LCM-only: Karras sigma schedule. Default false (linear, matches
    /// distillation training).
    pub sd_use_karras_sigmas: bool,
    /// SD-only: post-gate TAESD selection — already resolved by the
    /// caller against user preference + install state. Dispatch
    /// consumes verbatim; no further gating, no scheduler coupling.
    pub use_taesd: bool,
}

impl Default for InpaintTuning {
    fn default() -> Self {
        Self {
            sharpen: 0.0,
            feather_px: 0.0,
            grow_px: 0.0,
            backend: prunr_models::ModelId::LaMaFp32,
            sd_prompt: String::new(),
            sd_negative_prompt: String::new(),
            sd_guidance_scale: 1.0,
            sd_scheduler: super::brush_state::SdScheduler::Lcm,
            sd_steps: 8,
            sd_seed: None,
            sd_strength: 1.0,
            sd_use_karras_sigmas: false,
            use_taesd: false,
        }
    }
}

/// Pure result of resolving the SD dispatch decision from inputs.
/// Caller copies these into `SdInpaintRequest`. Stays out of the
/// rayon closure so the boundary test can exercise every input
/// combo without spawning anything.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SdDispatchPlan {
    pub use_taesd: bool,
    pub effective_cfg: f32,
}

/// Decide TAESD wiring + CFG clamping for an SD dispatch.
///
/// `backend` is the *actually-dispatched* model id (after
/// `Settings::lcm_routing_active` resolution upstream). `lcm` for
/// CFG-clamp purposes derives from the backend, NOT from the
/// user's scheduler request — picking the LCM scheduler without
/// the LCM bundle installed must NOT clamp CFG against standard
/// SD weights. (Bug #1 regression contract.)
///
/// `use_taesd_requested` is post-install-gate — the caller has
/// already resolved user preference + bundle availability.
/// Orthogonal to scheduler; works with standard SD and LCM checkpoints.
pub(crate) fn resolve_sd_dispatch(
    backend: prunr_models::ModelId,
    use_taesd_requested: bool,
    user_cfg: f32,
) -> SdDispatchPlan {
    let lcm_weights = backend == prunr_models::ModelId::SdV15LcmInpaintFp16;
    // LCM weights are calibrated for low CFG (model card: 1.0–2.0).
    // Standard SD passes the user's CFG straight through.
    let effective_cfg = if lcm_weights {
        user_cfg.clamp(1.0, 2.0)
    } else {
        user_cfg
    };
    SdDispatchPlan {
        use_taesd: use_taesd_requested,
        effective_cfg,
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
    /// In-flight batch: recipe + pending IDs. `None` between batches.
    in_flight: Option<InFlightBatch>,
    /// Last time periodic history cleanup ran.
    pub(crate) last_history_cleanup: Instant,
    /// Inpaint dispatch state. Per-item generation counter discards
    /// stale results when the user paints a fresh stroke before the
    /// previous one finishes. `inpaint_pending` is the count of
    /// dispatches not yet drained — the canvas reads it via
    /// `is_inpaint_in_flight` to show a progress overlay.
    inpaint_tx: mpsc::Sender<InpaintResult>,
    inpaint_rx: mpsc::Receiver<InpaintResult>,
    inpaint_latest_gen: HashMap<u64, u64>,
    inpaint_pending: HashMap<u64, u32>,
    /// Per-item cancel flag for the in-flight inpaint stroke. The flag
    /// is checked between LaMa tiles and between SD UNet steps; when
    /// set, the rayon job returns `CoreError::Cancelled` early and the
    /// drain path ignores the result. Cancel button + Esc key both
    /// flip the flag for the currently-selected item.
    inpaint_cancels: HashMap<u64, std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Per-item progress sink for the in-flight stroke. Worker writes
    /// `current` step between SD UNet iterations; the canvas banner
    /// reads `(current, total)` to show "Erasing — step N of M". Same
    /// lifetime as `inpaint_cancels` — both are replaced on every
    /// dispatch.
    inpaint_progress: HashMap<u64, std::sync::Arc<prunr_core::inpaint::InpaintProgress>>,
    /// Channels to the dedicated SD-inpaint subprocess bridge thread.
    /// LaMa / Big-LaMa / MI-GAN stay on the in-process rayon path; only
    /// SD-family dispatches go through these. Bridge spawns the
    /// subprocess lazily and drops it on a 5-min idle timer.
    inpaint_bridge_tx: mpsc::Sender<InpaintBridgeMsg>,
    inpaint_bridge_rx: mpsc::Receiver<InpaintBridgeResult>,
    /// In-process upscale dispatch state. A single image is upscaled on a
    /// background thread; live tile progress is published to
    /// `dispatch_progress` (the canvas overlay reads it). `upscale_active`
    /// is the in-flight predicate read by `is_upscale_in_flight`. Cancel
    /// is a per-dispatch atomic set by `cancel_upscale()`; the thread
    /// checks it between tiles.
    upscale_active: Arc<AtomicBool>,
    upscale_cancel: Arc<AtomicBool>,
    upscale_result_tx: mpsc::Sender<UpscaleResult>,
    upscale_result_rx: mpsc::Receiver<UpscaleResult>,
    /// Unified progress slot. One source of truth for the banner /
    /// modal widgets across every dispatch kind. Written by
    /// `dispatch_upscale`, `pump_inpaint_subprocess`, and the seg
    /// path's `refresh_batch_progress_status`; `None` when no dispatch
    /// is in flight.
    dispatch_progress: super::dispatch_progress::DispatchProgressSlot,
    /// Warm-cached upscale engine: kept alive across consecutive upscale
    /// dispatches so the second click skips the 1-3s graph-optimization
    /// stall (Level3 for ESRGAN, Level2 for HAT). Single-slot — caching
    /// multiple upscale engines simultaneously is YAGNI. Evicted on
    /// model swap (any switch away from the cached ModelKind) and on
    /// full release.
    warm_upscale_engine: Option<(prunr_core::ModelKind, Arc<prunr_core::OrtEngine>)>,
}

impl Processor {
    pub(crate) fn new(
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        let (inpaint_tx, inpaint_rx) = mpsc::channel();
        let (inpaint_bridge_tx, inpaint_bridge_rx) = spawn_inpaint_bridge();
        let (upscale_result_tx, upscale_result_rx) = mpsc::channel();
        Self {
            worker_tx,
            worker_rx,
            cancels: CancelRegistry::new(),
            live_preview: LivePreview::default(),
            admission: None,
            admission_tx: None,
            in_flight: None,
            last_history_cleanup: Instant::now(),
            inpaint_tx,
            inpaint_rx,
            inpaint_latest_gen: HashMap::new(),
            inpaint_pending: HashMap::new(),
            inpaint_cancels: HashMap::new(),
            inpaint_progress: HashMap::new(),
            inpaint_bridge_tx,
            inpaint_bridge_rx,
            upscale_active: Arc::new(AtomicBool::new(false)),
            upscale_cancel: Arc::new(AtomicBool::new(false)),
            upscale_result_tx,
            upscale_result_rx,
            dispatch_progress: super::dispatch_progress::DispatchProgressSlot::new(),
            warm_upscale_engine: None,
        }
    }

    /// Unified progress reader for the canvas overlay
    /// (`progress_widget::render_banner` / `render_modal`). `None`
    /// when no dispatch is in flight. Single source of truth across
    /// the seg, eraser, SD, and upscale paths.
    pub(super) fn dispatch_progress(&self) -> Option<super::dispatch_progress::DispatchProgress> {
        self.dispatch_progress.read()
    }

    /// Publish progress for the seg path. Called by
    /// `refresh_batch_progress_status` whenever batch state could have
    /// changed (per-frame during a batch; once when it completes).
    /// Skips the Mutex write when the new value is `None` and the slot
    /// is already empty — covers the steady-state idle case where this
    /// would otherwise grab the lock every frame.
    pub(crate) fn set_dispatch_progress(
        &self,
        progress: Option<super::dispatch_progress::DispatchProgress>,
    ) {
        if progress.is_none() && self.dispatch_progress.read().is_none() {
            return;
        }
        self.dispatch_progress.set(progress);
    }

    /// `true` while an upscale dispatch is in flight. The intent
    /// gates (`can_process_intent`, `apply_cancel_shortcut`) read this;
    /// the live tile counter is on `dispatch_progress` (the canvas
    /// banner / modal shows it directly).
    pub fn is_upscale_in_flight(&self) -> bool {
        self.upscale_active.load(Ordering::Acquire)
    }

    /// Per-item generation counter ensures a fresh stroke supersedes the
    /// previous in-flight job at drain time — see `drain_inpaint_results`.
    /// SD-family models route through the inpaint subprocess bridge for
    /// process isolation; LaMa / Big-LaMa / MI-GAN stay on the in-process
    /// rayon path (low RAM footprint, no isolation pressure).
    pub(crate) fn dispatch_inpaint(
        &mut self,
        item_id: u64,
        image: std::sync::Arc<image::RgbaImage>,
        correction: std::sync::Arc<prunr_core::brush::MaskCorrection>,
        tuning: InpaintTuning,
    ) {
        let generation = self.inpaint_latest_gen.entry(item_id).or_insert(0);
        *generation += 1;
        let gen = *generation;
        *self.inpaint_pending.entry(item_id).or_insert(0) += 1;
        // Replace any prior cancel flag + progress sink — the new
        // dispatch supersedes its predecessor anyway, so wiring fresh
        // ones avoids a stale earlier-stroke cancel firing the moment
        // a new stroke starts (and avoids the banner showing the prior
        // stroke's last step count for one frame).
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = std::sync::Arc::new(prunr_core::inpaint::InpaintProgress::new());
        self.inpaint_cancels.insert(item_id, cancel.clone());
        self.inpaint_progress.insert(item_id, progress.clone());
        if tuning.backend.is_sd_family() {
            self.dispatch_inpaint_sd(item_id, gen, &image, &correction, &tuning);
            return;
        }
        let tx = self.inpaint_tx.clone();
        rayon::spawn(move || {
            let raw_mask = correction.to_binary_mask(image.width(), image.height());
            // Pre-process: grow/erode the painted area before LaMa runs.
            let mask = if tuning.grow_px != 0.0 {
                prunr_core::inpaint::grow_mask(&raw_mask, tuning.grow_px.round() as i32)
            } else {
                raw_mask
            };
            // This path only sees LaMa / Big-LaMa / MI-GAN — sd_req is unread.
            let sd_req = None;
            let hooks = prunr_core::inpaint::InpaintHooks {
                cancel: Some(cancel.clone()),
                progress: Some(progress.clone()),
            };
            match prunr_core::inpaint::process_inpaint_with(&image, &mask, tuning.backend, sd_req, &hooks) {
                Ok(rgba) => {
                    let out = prunr_core::inpaint_blend::finalize_inpaint(
                        &rgba, &image, &mask, tuning.feather_px, tuning.sharpen,
                    );
                    let _ = tx.send(InpaintResult { item_id, rgba: out, generation: gen, cancelled: false, error: None });
                }
                Err(prunr_core::CoreError::Cancelled) => {
                    tracing::info!(item_id, "inpaint cancelled by user");
                    // Marker with cancelled=true so the drain path can
                    // surface a toast. Gen 0 keeps it treated as stale.
                    let _ = tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled: true,
                        error: None,
                    });
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::error!(item_id, e = %msg, "inpaint dispatch failed");
                    // Surface the error to the GUI as a toast — without
                    // it the user just sees the brush stroke "do nothing"
                    // (e.g. when the SD-bundle RAM guard refuses to
                    // load). Gen 0 still routes the result through the
                    // stale-drop branch in drain_inpaint_results.
                    let _ = tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled: false,
                        error: Some(msg),
                    });
                }
            }
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
        for &item_id in self.inpaint_cancels.keys() {
            let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel { item_id });
        }
    }

    /// Cancel any in-flight inpaint stroke for `item_id`. The local
    /// flag drives the "Cancelling…" banner state immediately; for SD
    /// strokes the bridge also forwards `CancelItem` to the subprocess
    /// so its inference loop sees the flag too. Latency to actually-
    /// stopping is one tile (LaMa) or one UNet step (SD); ORT has no
    /// per-op cancel hook. Idempotent.
    pub(crate) fn cancel_inpaint(&self, item_id: u64) {
        if let Some(flag) = self.inpaint_cancels.get(&item_id) {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Cancel { item_id });
    }

    /// Tell the inpaint bridge to drop its cached subprocess. Idempotent;
    /// no-op when the bridge is already idle. Called by
    /// `apply_toolbar_change` on every model switch so a previous
    /// SD-family backend's ~200 MB residual subprocess (bundle was
    /// already released by `inpaint_sd::release` on dispatch
    /// completion, but the worker process itself stays alive for the
    /// 5-min idle window) doesn't pin RAM the user no longer wants
    /// allocated.
    pub(crate) fn release_inpaint_subprocess(&self) {
        let _ = self.inpaint_bridge_tx.send(InpaintBridgeMsg::Release);
    }

    /// Tell the seg/edge worker bridge to drop its warm subprocess.
    /// Idempotent; no-op when nothing is warm. Symmetric with
    /// `release_inpaint_subprocess` — both fire on model switch so a
    /// previous backend's engine pool (BiRefNetLite ~2 GB, U2Net
    /// ~800 MB, …) isn't held across the user's "I'm done with that
    /// model" signal.
    pub(crate) fn release_seg_warm(&self) {
        let _ = self.worker_tx.send(WorkerMessage::ReleaseWarm);
    }

    /// Cache-lookup helper: returns the cached `Arc<OrtEngine>` on a
    /// ModelKind hit, evicts the slot on a ModelKind mismatch, and
    /// returns `None` in both miss cases. Never constructs a new engine.
    /// Called first by `ensure_upscale_engine`; if it returns `None`,
    /// the caller constructs and caches a fresh engine.
    fn try_cached_upscale_engine(
        &mut self,
        model_kind: prunr_core::ModelKind,
    ) -> Option<Arc<prunr_core::OrtEngine>> {
        match &self.warm_upscale_engine {
            Some((cached_kind, engine)) if *cached_kind == model_kind => {
                Some(Arc::clone(engine))
            }
            Some(_) => {
                // Different model — evict the old cache before the caller
                // constructs a fresh engine.
                self.warm_upscale_engine = None;
                None
            }
            None => None,
        }
    }

    /// Returns a cached `Arc<OrtEngine>` for the given ModelKind,
    /// constructing one if absent or if the cached ModelKind differs.
    /// Eviction on ModelKind mismatch drops the previous Arc (its
    /// strong_count goes to zero once outstanding dispatches release
    /// their clones).
    fn ensure_upscale_engine(
        &mut self,
        model_kind: prunr_core::ModelKind,
        intra_threads: usize,
        level: prunr_core::engine::GraphOptimizationLevel,
    ) -> Result<Arc<prunr_core::OrtEngine>, prunr_core::CoreError> {
        if let Some(cached) = self.try_cached_upscale_engine(model_kind) {
            return Ok(cached);
        }
        let fresh = Arc::new(
            prunr_core::OrtEngine::new_with_optimization_level(model_kind, intra_threads, level)?,
        );
        self.warm_upscale_engine = Some((model_kind, Arc::clone(&fresh)));
        Ok(fresh)
    }

    /// Drops the cached upscale engine. Call on:
    /// - User switches to any model (model dropdown swap — even
    ///   upscale→upscale; `ensure_upscale_engine` evicts on ModelKind
    ///   mismatch, so this is the switch-away-from-upscale case).
    /// - Hardware settings change.
    pub(crate) fn release_upscale_engine(&mut self) {
        self.warm_upscale_engine = None;
    }

    /// Test-only accessor for injecting or inspecting the warm-engine slot.
    #[cfg(test)]
    pub(crate) fn warm_upscale_engine_for_test(
        &mut self,
    ) -> &mut Option<(prunr_core::ModelKind, Arc<prunr_core::OrtEngine>)> {
        &mut self.warm_upscale_engine
    }

    /// SD inpaint dispatch via the dedicated subprocess bridge. Encodes
    /// the (already-Arc'd) image + binary mask to PNG temp files,
    /// writes them, and sends an `InpaintBridgeMsg::Dispatch`. The
    /// bridge handles subprocess lifecycle. Bridge results stream back
    /// via `pump_inpaint_subprocess` (called once per frame).
    fn dispatch_inpaint_sd(
        &mut self,
        item_id: u64,
        gen: u64,
        image: &std::sync::Arc<image::RgbaImage>,
        correction: &std::sync::Arc<prunr_core::brush::MaskCorrection>,
        tuning: &InpaintTuning,
    ) {
        // PNG-encode + temp-file write is 150-300 ms per stroke at 4K
        // — moving it onto rayon keeps the egui frame loop responsive
        // mid-paint. Mask construction (`to_binary_mask`, `grow_mask`)
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
                let _ = inpaint_tx.send(InpaintResult {
                    item_id,
                    rgba: image::RgbaImage::new(0, 0),
                    generation: 0,
                    cancelled: true,
                    error: None,
                });
            };
            if cancelled() { send_cancelled(); return; }
            let raw_mask = correction.to_binary_mask(image.width(), image.height());
            let mask = if tuning.grow_px != 0.0 {
                prunr_core::inpaint::grow_mask(&raw_mask, tuning.grow_px.round() as i32)
            } else {
                raw_mask
            };
            let sd_req = match tuning.backend {
                prunr_models::ModelId::SdV15InpaintFp16
                | prunr_models::ModelId::SdV15LcmInpaintFp16 => {
                    use prunr_core::inpaint_sd::SchedulerKind;
                    let scheduler: SchedulerKind = tuning.sd_scheduler.into();
                    let plan = resolve_sd_dispatch(
                        tuning.backend,
                        tuning.use_taesd,
                        tuning.sd_guidance_scale,
                    );
                    Some(prunr_core::inpaint_sd::SdInpaintRequest {
                        prompt: tuning.sd_prompt.clone(),
                        negative_prompt: tuning.sd_negative_prompt.clone(),
                        num_inference_steps: tuning.sd_steps,
                        guidance_scale: plan.effective_cfg,
                        seed: tuning.sd_seed,
                        use_taesd: plan.use_taesd,
                        scheduler,
                        strength: tuning.sd_strength,
                        use_karras_sigmas: tuning.sd_use_karras_sigmas,
                    })
                }
                _ => None,
            };
            let dir = crate::subprocess::protocol::ipc_temp_dir();
            let image_path = crate::subprocess::protocol::IpcKind::InpaintImg.path_for_gen(dir, item_id, gen);
            let mask_path  = crate::subprocess::protocol::IpcKind::InpaintMask.path_for_gen(dir, item_id, gen);
            let res: Result<(), String> = (|| {
                let img_bytes = prunr_core::encode_rgba_png(&image)
                    .map_err(|e| format!("encode source: {e:?}"))?;
                std::fs::write(&image_path, img_bytes)
                    .map_err(|e| format!("write source: {e}"))?;
                let mask_bytes = prunr_core::encode_gray_png(&mask)
                    .map_err(|e| format!("encode mask: {e:?}"))?;
                std::fs::write(&mask_path, mask_bytes)
                    .map_err(|e| format!("write mask: {e}"))
            })();
            if let Err(e) = res {
                let _ = inpaint_tx.send(InpaintResult {
                    item_id,
                    rgba: image::RgbaImage::new(0, 0),
                    generation: 0,
                    cancelled: false,
                    error: Some(e),
                });
                return;
            }
            // Esc landed during the encode — drop the work before the
            // bridge sees it. Temp files we just wrote are abandoned;
            // the next dispatch's gen-bump renames over them.
            if cancelled() {
                let _ = std::fs::remove_file(&image_path);
                let _ = std::fs::remove_file(&mask_path);
                send_cancelled();
                return;
            }
            let _ = bridge_tx.send(InpaintBridgeMsg::Dispatch {
                item_id, gen, model_id: tuning.backend, image_path, mask_path, sd_req,
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
                InpaintBridgeResult::Progress {
                    item_id, current, total, outer_current, outer_total,
                } => {
                    if let Some(p) = self.inpaint_progress.get(&item_id) {
                        p.set_total(total);
                        p.set_step(current);
                        p.set_outer_total(outer_total);
                        p.set_outer_step(outer_current);
                    }
                    // Mirror into the unified slot so the banner / modal
                    // sees the SD inpaint's two-level counter.
                    self.dispatch_progress.set(Some(
                        super::dispatch_progress::DispatchProgress::sd_inpaint(
                            outer_current, outer_total, (current, total),
                        ),
                    ));
                }
                InpaintBridgeResult::Done { item_id, gen, rgba_path, width, height } => {
                    let result = match super::worker::read_and_delete(&rgba_path) {
                        Some(b) => match image::load_from_memory(&b) {
                            Ok(img) => {
                                let rgba = img.to_rgba8();
                                debug_assert_eq!((rgba.width(), rgba.height()), (width, height));
                                // Stamp with the dispatch's own gen — drain
                                // drops it as stale if a fresher stroke
                                // bumped `inpaint_latest_gen` while this
                                // one was in the subprocess.
                                InpaintResult {
                                    item_id, rgba, generation: gen,
                                    cancelled: false, error: None,
                                }
                            }
                            Err(e) => InpaintResult {
                                item_id,
                                rgba: image::RgbaImage::new(0, 0),
                                generation: 0, cancelled: false,
                                error: Some(format!("decode SD result: {e}")),
                            },
                        },
                        None => InpaintResult {
                            item_id,
                            rgba: image::RgbaImage::new(0, 0),
                            generation: 0, cancelled: false,
                            error: Some(format!("read SD result missing: {}", rgba_path.display())),
                        },
                    };
                    let _ = self.inpaint_tx.send(result);
                    // Clear the unified slot — the dispatch is done from
                    // the bridge's perspective. The GUI's drain will
                    // surface the result on the next frame.
                    self.dispatch_progress.set(None);
                }
                InpaintBridgeResult::Error { item_id, error } => {
                    // Translate bridge sentinels to user-facing text at
                    // this seam so `drain_inpaint_results` doesn't have
                    // to know about IPC strings.
                    use crate::subprocess::protocol::{CANCELLED_ERR_MSG, MEMORY_PRESSURE_ABORT_MSG};
                    let cancelled = error == CANCELLED_ERR_MSG;
                    let user_error = if cancelled {
                        None
                    } else if error == MEMORY_PRESSURE_ABORT_MSG {
                        Some(
                            "Erase aborted — system memory low. \
                             Close other apps or use LaMa instead \
                             (Settings → Eraser)."
                                .to_string(),
                        )
                    } else {
                        Some(error)
                    };
                    let _ = self.inpaint_tx.send(InpaintResult {
                        item_id,
                        rgba: image::RgbaImage::new(0, 0),
                        generation: 0,
                        cancelled,
                        error: user_error,
                    });
                    self.dispatch_progress.set(None);
                }
            }
        }
    }

    /// Read the in-flight inpaint stroke's progress as
    /// `((outer_current, outer_total), (inner_current, inner_total))`.
    /// Used by the unified progress widget when the slot is empty but
    /// an in-process LaMa dispatch is mid-flight (LaMa writes
    /// `InpaintProgress` directly via the hooks and doesn't go
    /// through the subprocess bridge that publishes the slot).
    pub(crate) fn inpaint_progress_nested(&self, item_id: u64) -> ((u32, u32), (u32, u32)) {
        self.inpaint_progress.get(&item_id)
            .map(|p| p.read_nested())
            .unwrap_or(((0, 0), (0, 0)))
    }

    /// Drain in-flight inpaint results.
    ///
    /// Returns `(committed_results, cancelled_item_ids, errors)`.
    /// - `committed_results` only carries the latest-gen finished
    ///   strokes (stale ones are silently dropped).
    /// - `cancelled_item_ids` lists items whose stroke was cancelled
    ///   by the user — the GUI surfaces a toast for each so the user
    ///   gets explicit feedback that Esc/Cancel took effect.
    /// - `errors` carries the user-visible message for any non-Cancelled
    ///   dispatch failure (e.g. SD RAM-guard refusal). The GUI shows
    ///   each as an error toast — without this surface the user sees
    ///   the stroke "do nothing" with no idea why.
    pub(crate) fn drain_inpaint_results(&mut self) -> (Vec<InpaintResult>, Vec<u64>, Vec<String>) {
        let mut out = Vec::new();
        let mut cancelled = Vec::new();
        let mut errors = Vec::new();
        while let Ok(result) = self.inpaint_rx.try_recv() {
            let item_id = result.item_id;
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
                self.inpaint_progress.remove(&item_id);
            }
            if result.cancelled {
                cancelled.push(item_id);
                continue;
            }
            if let Some(msg) = result.error {
                errors.push(msg);
                continue;
            }
            let latest = self.inpaint_latest_gen.get(&item_id).copied().unwrap_or(0);
            if result.generation == latest && result.generation > 0 {
                out.push(result);
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

    /// True after Cancel/Esc clicked but before the worker's atomic
    /// Acquire load observes the flag. Drives the "Cancelling…" banner
    /// state — without this signal the click looks unacknowledged
    /// during the multi-second latency to the next worker checkpoint.
    pub(crate) fn is_inpaint_cancelling(&self, item_id: u64) -> bool {
        self.inpaint_cancels.get(&item_id)
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Register a batch's recipe + the IDs that should deliver against it.
    /// Replaces any prior in-flight state — callers ensure prior batches
    /// have completed before firing a new dispatch.
    pub(crate) fn track_dispatch(
        &mut self,
        recipe: ProcessingRecipe,
        ids: impl IntoIterator<Item = u64>,
    ) {
        let pending: HashSet<u64> = ids.into_iter().collect();
        let total = pending.len();
        self.in_flight = Some(InFlightBatch { recipe, pending, total });
    }

    /// `(done, total)` for the currently in-flight dispatch, or `None`
    /// when idle. `total` is the count registered at `track_dispatch`,
    /// `done = total - pending.len()`. Streamed-admission items bump
    /// `total` via `track_streamed`.
    ///
    /// Scopes the seg progress counter to the *current* dispatch so a
    /// previously-Done item from an earlier reprocess doesn't inflate
    /// the displayed total ("1 of 2" when only one item was dispatched).
    pub(crate) fn current_dispatch_progress(&self) -> Option<(u32, u32)> {
        let b = self.in_flight.as_ref()?;
        let done = b.total.saturating_sub(b.pending.len());
        Some((done as u32, b.total as u32))
    }

    /// Add a streamed (admission-pool) item to the current batch. The
    /// `debug_assert` catches the "admission ran without a tracked batch"
    /// invariant breach in tests; release builds silently no-op so a
    /// single late delivery can't take down a real batch.
    pub(crate) fn track_streamed(&mut self, id: u64) {
        match self.in_flight.as_mut() {
            Some(b) => {
                if b.pending.insert(id) {
                    b.total += 1;
                }
            }
            None => debug_assert!(false, "track_streamed called without active batch"),
        }
    }

    /// Take the recipe for a finished item. Returns `None` when the item
    /// wasn't in flight (late delivery after cancel/drain) — caller falls
    /// back. Self-cleans the in-flight slot when the last item completes.
    pub(crate) fn take_recipe(&mut self, id: u64) -> Option<ProcessingRecipe> {
        let batch = self.in_flight.as_mut()?;
        if !batch.pending.remove(&id) {
            return None;
        }
        let recipe = batch.recipe.clone();
        if batch.pending.is_empty() {
            self.in_flight = None;
        }
        Some(recipe)
    }

    /// Drop the in-flight slot regardless of pending. Called on user cancel
    /// or batch-complete signals so a late delivery can't reattribute.
    pub(crate) fn drain_recipes(&mut self) {
        self.in_flight = None;
    }

    /// Drop admission state so no further items are admitted. Called on
    /// cancel (user or worker-side) and by the cancelled-message handler.
    /// Leaves the cancel registry untouched — that's owned by the caller's
    /// cancel protocol.
    pub(crate) fn clear_admission(&mut self) {
        self.admission = None;
        self.admission_tx = None;
    }

    /// Spawn a background thread that upscales the input and posts the
    /// result back via `upscale_result_rx`. The caller is responsible for
    /// having verified that the model is installed before dispatching.
    ///
    /// Routing: `OutputScale::X4TwoPass` dispatches through `upscale_two_pass`
    /// (chains RealEsrganX2Plus twice for net 4× output); all other variants
    /// dispatch through `upscale_rgba` with the user-selected model.
    ///
    /// Pre-flight admission check refuses the dispatch when free RAM is
    /// below `descriptor.working_set_mb`; a warning is logged and the
    /// function returns without spawning. The toolbar's `can_process_intent`
    /// gate should have prevented reaching this path, but the check here
    /// is the last-resort guard at the dispatch layer.
    pub(crate) fn dispatch_upscale(
        &mut self,
        item_id: u64,
        input: Arc<image::RgbaImage>,
        model_id: prunr_models::ModelId,
        output_scale: prunr_core::OutputScale,
        intra_threads: usize,
        recipe: ProcessingRecipe,
    ) {
        let free_mb = (crate::hardware::available_ram_bytes_throttled() / (1024 * 1024)) as u32;
        if let Some(d) = prunr_models::descriptor(model_id) {
            if !admission_check(d.working_set_mb, free_mb) {
                tracing::warn!(
                    model = ?model_id,
                    working_set_mb = d.working_set_mb,
                    free_mb,
                    "upscale admission refused: insufficient RAM"
                );
                return;
            }
        }

        let use_two_pass = matches!(output_scale, prunr_core::OutputScale::X4TwoPass);
        let scale_factor = output_scale.factor();

        // Tier-1 pre-inference pass — runs OUTSIDE the ORT session, so the
        // row-parallel rayon inside apply_denoise is not nested in the inference
        // thread pool (the nested-rayon pattern that deadlocked apply_background_color
        // in b2306bb). Do not move this below upscale_active.store(true).
        let pre_denoise = recipe.upscale.pre_denoise();
        let brightness_lift = recipe.upscale.brightness_lift();
        let input_for_inference: Arc<image::RgbaImage> = build_inference_input(
            &input, pre_denoise, brightness_lift,
        );

        // Resolve ModelKind + optimization level so we can look up the
        // warm-engine cache. X4TwoPass always routes through RealEsrganX2Plus;
        // other variants use the user-selected model.
        let cache_model_kind = if use_two_pass {
            prunr_core::ModelKind::RealEsrganX2Plus
        } else {
            match prunr_core::ModelKind::try_from(model_id) {
                Ok(k) => k,
                Err(id) => {
                    tracing::error!(
                        model = ?id,
                        "upscale dispatch: ModelKind mapping missing — refusing dispatch"
                    );
                    return;
                }
            }
        };
        let cache_descriptor = match prunr_models::REGISTRY.iter().find(|d| {
            d.id == if use_two_pass { prunr_models::ModelId::RealEsrganX2Plus } else { model_id }
        }) {
            Some(d) => d,
            None => {
                tracing::error!(model = ?model_id, "upscale dispatch: model not in REGISTRY");
                return;
            }
        };
        let level = prunr_core::upscale::pick_optimization_level(cache_descriptor);

        let engine = match self.ensure_upscale_engine(cache_model_kind, intra_threads, level) {
            Ok(e) => e,
            Err(err) => {
                tracing::error!(model = ?model_id, %err, "upscale dispatch: engine construction failed");
                return;
            }
        };

        // Release stores pair with the Acquire load in
        // `upscale::tiling::upscale_tiled` (cancel flag) and the Acquire
        // load in `is_upscale_in_flight` (active flag). Without the
        // pairing, weakly-ordered architectures can delay propagation.
        self.upscale_active.store(true, Ordering::Release);
        self.upscale_cancel.store(false, Ordering::Release);
        // Seed the unified slot before the first tile so an early render
        // already shows "Upscaling — tile 0 of …" rather than the prior
        // dispatch's stale data.
        self.dispatch_progress.set(Some(
            super::dispatch_progress::DispatchProgress::upscale(
                0, 0, super::dispatch_progress::step_labels::LOADING_MODEL,
            ),
        ));

        let active_flag = Arc::clone(&self.upscale_active);
        let cancel_flag = Arc::clone(&self.upscale_cancel);
        let result_tx = self.upscale_result_tx.clone();
        let progress_slot = self.dispatch_progress.clone();
        // Clone the Arc so the worker thread owns a reference to the engine
        // independently of the cached slot. The slot may be evicted (e.g.
        // model swap mid-dispatch) without invalidating the in-flight session.
        let engine_for_thread = Arc::clone(&engine);

        std::thread::spawn(move || {
            let progress_slot_for_callback = progress_slot.clone();
            let on_tile = move |done, total| {
                progress_slot_for_callback.update(|p| {
                    if let Some(p) = p {
                        p.inner = (done, total);
                        p.step_label = std::borrow::Cow::Borrowed(
                            super::dispatch_progress::step_labels::TILE_INFERENCE,
                        );
                    }
                });
            };
            let result = if use_two_pass {
                prunr_core::upscale::upscale_two_pass_with_engine(
                    &input_for_inference,
                    &engine_for_thread,
                    on_tile,
                    Some(cancel_flag),
                    None,
                )
            } else {
                prunr_core::upscale::upscale_rgba_with_engine(
                    &input_for_inference,
                    &engine_for_thread,
                    model_id,
                    scale_factor,
                    on_tile,
                    Some(cancel_flag),
                    None,
                )
            };
            // Tier-1 post-inference: undo the brightness lift so the final
            // image has the user's intended exposure. Denoise has no inverse
            // — smoothing the input noise is the permanent intent.
            let result = result.map(|mut img| {
                apply_post_inference(&mut img, brightness_lift);
                img
            });
            // Send the result BEFORE clearing the active flag. Reversed
            // ordering would let the UI thread observe `is_in_flight=false`
            // and enable Process before the previous result has landed in
            // the channel.
            let _ = result_tx.send(UpscaleResult { item_id, result, recipe });
            active_flag.store(false, Ordering::Release);
            // Clear the unified slot — widgets fall back to idle state.
            progress_slot.set(None);
        });
    }

    /// Set the cancel flag for any in-flight upscale dispatch. The worker
    /// thread checks the flag between tiles; latency to actually stopping
    /// is one tile (~0.5 s at 4K with ESRGAN). Idempotent.
    pub(crate) fn cancel_upscale(&self) {
        // Release pairs with Acquire in `upscale::tiling::upscale_tiled`.
        self.upscale_cancel.store(true, Ordering::Release);
    }

    /// Drain completed upscale results from the background thread.
    /// Non-blocking; returns an empty `Vec` when nothing is ready. Caller
    /// applies each result to the matching `BatchItem`.
    pub(crate) fn pump_upscale_results(&mut self) -> Vec<UpscaleResult> {
        let mut out = Vec::new();
        while let Ok(r) = self.upscale_result_rx.try_recv() {
            out.push(r);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::ModelId;

    // ── is_upscale_in_flight + admission_check ─────────────────────────────

    #[test]
    fn is_upscale_in_flight_false_when_idle() {
        let p = fixture();
        assert!(!p.is_upscale_in_flight(),
            "idle Processor must not report an in-flight upscale");
    }

    #[test]
    fn is_upscale_in_flight_true_when_active_flag_set() {
        let p = fixture();
        p.upscale_active.store(true, Ordering::Release);
        assert!(p.is_upscale_in_flight());
    }

    #[test]
    fn set_dispatch_progress_none_over_none_is_a_no_op() {
        // The dirty-check inside `set_dispatch_progress` skips the
        // Mutex write when the slot is already None. Verify both:
        // (a) calling with None on an empty slot leaves it empty;
        // (b) calling with None on a populated slot clears it.
        use crate::gui::dispatch_progress::DispatchProgress;
        let p = fixture();

        // Empty → None: still empty.
        assert!(p.dispatch_progress().is_none(), "fixture starts empty");
        p.set_dispatch_progress(None);
        assert!(p.dispatch_progress().is_none(),
            "set(None) on already-empty slot must not flip state");

        // Populated → None: cleared.
        p.set_dispatch_progress(Some(DispatchProgress::seg(1, 3, "Processing 1/3")));
        assert!(p.dispatch_progress().is_some(), "Some was published");
        p.set_dispatch_progress(None);
        assert!(p.dispatch_progress().is_none(),
            "Some→None clear must write through the dirty-check");
    }

    #[test]
    fn admission_refuses_when_working_set_exceeds_free() {
        assert!(!admission_check(2000, 1500),
            "2000 MB working set must not fit in 1500 MB free RAM");
    }

    #[test]
    fn admission_allows_when_working_set_fits() {
        assert!(admission_check(500, 1500),
            "500 MB working set fits in 1500 MB free RAM");
    }

    #[test]
    fn admission_allows_exact_match() {
        assert!(admission_check(1500, 1500),
            "exact match (working_set == free_ram) must be allowed");
    }

    #[test]
    fn admission_refuses_when_free_ram_is_zero() {
        assert!(!admission_check(600, 0),
            "any positive working-set must be refused when free RAM is zero");
    }

    fn cfg_eq(a: f32, b: f32) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn standard_sd_ddim_taesd_off_passes_through() {
        let p = resolve_sd_dispatch(ModelId::SdV15InpaintFp16, false, 7.5);
        assert!(!p.use_taesd);
        assert!(cfg_eq(p.effective_cfg, 7.5),
            "standard SD must NOT clamp CFG");
    }

    #[test]
    fn standard_sd_ddim_taesd_on_when_available() {
        // Caller (sd_use_taesd_effective) has already gated by install;
        // here we simulate that with use_taesd_requested=true.
        let p = resolve_sd_dispatch(ModelId::SdV15InpaintFp16, true, 4.0);
        assert!(p.use_taesd);
        assert!(cfg_eq(p.effective_cfg, 4.0));
    }

    #[test]
    fn standard_sd_ddim_taesd_requested_but_install_gate_filtered() {
        // sd_use_taesd_effective returned false (bundle missing) so
        // caller threads use_taesd_requested=false. Plan stays orthogonal.
        let p = resolve_sd_dispatch(ModelId::SdV15InpaintFp16, false, 7.5);
        assert!(!p.use_taesd);
        assert!(cfg_eq(p.effective_cfg, 7.5));
    }

    #[test]
    fn lcm_weights_lcm_scheduler_clamps_cfg() {
        // user_cfg=7.5 with LCM bundle dispatched: clamp to 2.0.
        let p = resolve_sd_dispatch(ModelId::SdV15LcmInpaintFp16, false, 7.5);
        assert!(!p.use_taesd);
        assert!(cfg_eq(p.effective_cfg, 2.0),
            "LCM weights must clamp CFG to [1.0, 2.0]");
    }

    #[test]
    fn lcm_weights_with_taesd_clamps_cfg_and_keeps_taesd() {
        let p = resolve_sd_dispatch(ModelId::SdV15LcmInpaintFp16, true, 1.5);
        assert!(p.use_taesd);
        assert!(cfg_eq(p.effective_cfg, 1.5),
            "in-range CFG should round-trip unchanged through clamp");
    }

    /// Bug #1 regression: user picks LCM scheduler but the LCM
    /// bundle is NOT installed, so upstream resolves backend back
    /// to SdV15InpaintFp16. Old code clamped CFG to 1.0-2.0 because
    /// it gated on `matches!(scheduler, Lcm)`. New code gates on
    /// `backend == SdV15LcmInpaintFp16` so CFG passes through.
    /// Bug #1 regression: scheduler=LCM with bundle missing falls
    /// back to standard SD, so CFG must NOT clamp to 1.0–2.0.
    #[test]
    fn lcm_scheduler_without_lcm_bundle_does_not_clamp_cfg() {
        let p = resolve_sd_dispatch(ModelId::SdV15InpaintFp16, false, 7.5);
        assert!(cfg_eq(p.effective_cfg, 7.5),
            "CFG must NOT clamp when standard SD weights dispatch");
    }


    fn fixture() -> Processor {
        let (tx, _rx_unused) = mpsc::channel::<WorkerMessage>();
        let (_tx_unused, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }

    #[test]
    fn new_initialises_last_history_cleanup_recent() {
        // Verifies the periodic 600s cleanup gate isn't accidentally
        // triggered at startup — the Instant must be effectively-now.
        let p = fixture();
        assert!(p.last_history_cleanup.elapsed().as_secs() < 5);
    }

    #[test]
    fn drain_filters_stale_generations() {
        let mut p = fixture();
        p.inpaint_latest_gen.insert(7, 2);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 1, cancelled: false, error: None,
        }).unwrap();
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(1, 1), generation: 2, cancelled: false, error: None,
        }).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert_eq!(drained.len(), 1, "stale gen=1 must be dropped");
        assert_eq!(drained[0].generation, 2);
        assert!(cancelled.is_empty(), "no cancellation events expected");
        assert!(errors.is_empty(), "no dispatch errors expected");
    }

    #[test]
    fn drain_routes_cancelled_results_to_cancellation_list() {
        let mut p = fixture();
        p.inpaint_latest_gen.insert(7, 5);
        p.inpaint_pending.insert(7, 1);
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(0, 0),
            generation: 0, cancelled: true, error: None,
        }).unwrap();
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
        p.inpaint_tx.send(InpaintResult {
            item_id: 7, rgba: image::RgbaImage::new(0, 0),
            generation: 0, cancelled: false,
            error: Some("SD inpaint refused to load: only 13.4 GB free".to_string()),
        }).unwrap();
        let (drained, cancelled, errors) = p.drain_inpaint_results();
        assert!(drained.is_empty(),
            "errored stroke must not commit a result");
        assert!(cancelled.is_empty(),
            "errored stroke is not a cancel");
        assert_eq!(errors.len(), 1,
            "dispatch error must surface for the user toast");
        assert!(errors[0].contains("13.4 GB"));
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

    #[test]
    fn cancel_registry_reset_clears_all_flags() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        r.request_item_cancel(42);
        r.reset();
        assert!(!r.is_cancelled(42));
        assert!(!r.is_cancelled(99));
    }

    #[test]
    fn global_cancel_short_circuits_per_item_lookup() {
        let r = CancelRegistry::new();
        r.request_global_cancel();
        // Any id reports cancelled when global is set, even ones with no per-item entry.
        assert!(r.is_cancelled(u64::MAX));
    }

    fn fixture_recipe() -> ProcessingRecipe {
        use prunr_core::{
            CompositeRecipe, EdgeRecipe, EdgeScale, ComposeMode, FillStyle, InferenceRecipe,
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
            },
            mask: (&MaskSettings { fill_style: FillStyle::None, ..Default::default() }).into(),
            composite: CompositeRecipe::default(),
            upscale: UpscaleRecipe::default(),
            was_chain: false,
        }
    }

    #[test]
    fn track_dispatch_then_take_returns_recipe_per_item() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [10, 20, 30].iter().copied());
        assert!(p.take_recipe(10).is_some());
        assert!(p.take_recipe(20).is_some());
        // Slot still alive while items remain.
        assert!(p.in_flight.is_some());
        assert!(p.take_recipe(30).is_some());
        // Last item drains the slot.
        assert!(p.in_flight.is_none());
    }

    #[test]
    fn take_recipe_unknown_id_is_none() {
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        assert!(p.take_recipe(999).is_none(), "unknown id must not return a recipe");
        // Tracked id still works.
        assert!(p.take_recipe(1).is_some());
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
        p.track_dispatch(fixture_recipe(), [10, 20].iter().copied());
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)),
            "fresh dispatch starts at 0/total");
        assert!(p.take_recipe(10).is_some());
        assert_eq!(p.current_dispatch_progress(), Some((1, 2)),
            "one item delivered → 1/2");
        assert!(p.take_recipe(20).is_some());
        // Last delivery self-drains the slot, so progress reports None
        // — the seg slot is cleared by the publisher on the next refresh.
        assert!(p.current_dispatch_progress().is_none(),
            "slot drains when the last item completes");
    }

    #[test]
    fn current_dispatch_progress_includes_streamed_items() {
        // Admission-pool items bump the total so the user sees the
        // queue grow as the worker accepts more. Without this, a
        // streamed item would deliver and decrement done-from-total
        // → counter goes negative-ish (saturating_sub clamps it).
        let mut p = fixture();
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        assert_eq!(p.current_dispatch_progress(), Some((0, 1)));
        p.track_streamed(2);
        assert_eq!(p.current_dispatch_progress(), Some((0, 2)),
            "streamed item must bump total");
        assert!(p.take_recipe(1).is_some());
        assert_eq!(p.current_dispatch_progress(), Some((1, 2)));
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
        p.track_dispatch(fixture_recipe(), [1].iter().copied());
        p.track_streamed(2);
        assert!(p.take_recipe(1).is_some());
        assert!(p.take_recipe(2).is_some(), "streamed item must have a recipe");
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
        p.track_dispatch(fixture_recipe(), [1, 2, 3].iter().copied());
        p.drain_recipes();
        assert!(p.take_recipe(1).is_none(),
            "drain must drop the slot so late deliveries fall back");
    }

    // ── Brush gate boundary contract (M-GUI-9) ───────────────────────────────
    //
    // `canvas::render` computes:
    //   brush_active = is_enabled && !inpaint_in_flight_for_selected && …
    //
    // If is_inpaint_in_flight returns true for the selected item, brush_active
    // must be false — so handle_brush_input is never entered. This test pins
    // the Processor half of that contract so a future decoupling of
    // is_inpaint_in_flight / is_inpaint_cancelling doesn't silently break the gate.

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
    fn inpaint_cancelling_is_independent_of_in_flight_count() {
        // Cancelling starts before the worker has seen the flag; is_inpaint_in_flight
        // is still true during this window. A future refactor that decouples them
        // must not remove the separate is_inpaint_cancelling check.
        let mut p = fixture();
        *p.inpaint_pending.entry(7).or_insert(0) += 1;
        p.inpaint_cancels.insert(7, Arc::new(AtomicBool::new(true)));
        assert!(p.is_inpaint_in_flight(7), "in-flight must still be true during cancel");
        assert!(p.is_inpaint_cancelling(7), "cancelling must reflect the atomic flag");
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
    use super::{build_inference_input, apply_post_inference};
    use std::sync::Arc;

    fn grey_image(w: u32, h: u32, v: u8) -> image::RgbaImage {
        image::RgbaImage::from_pixel(w, h, image::Rgba([v, v, v, 255]))
    }

    #[test]
    fn dispatch_pre_process_no_knobs_returns_same_arc() {
        // When neither pre_denoise nor brightness_lift is active, the returned
        // Arc is the same pointer as the input (no clone, no allocation).
        let input = Arc::new(grey_image(4, 4, 128));
        let out = build_inference_input(&input, 0.0, 0.0);
        assert!(Arc::ptr_eq(&input, &out),
            "no-op path must return the same Arc (no pixel clone)");
    }

    #[test]
    fn dispatch_pre_process_denoise_positive_returns_different_arc() {
        // With pre_denoise > 0, a new buffer is allocated and processed.
        let input = Arc::new(grey_image(4, 4, 100));
        let out = build_inference_input(&input, 0.5, 0.0);
        assert!(!Arc::ptr_eq(&input, &out),
            "denoise path must return a new Arc (different buffer)");
    }

    #[test]
    fn dispatch_pre_process_brightness_lift_nonzero_returns_different_arc() {
        // With brightness_lift != 0, a new buffer is allocated.
        let input = Arc::new(grey_image(4, 4, 100));
        let out = build_inference_input(&input, 0.0, 1.0);
        assert!(!Arc::ptr_eq(&input, &out),
            "brightness_lift path must return a new Arc");
        // The lifted output should be brighter than the input.
        let mean_in: f64 = input.pixels().map(|p| p[0] as f64).sum::<f64>() / (4.0 * 4.0);
        let mean_out: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / (4.0 * 4.0);
        assert!(mean_out > mean_in,
            "positive EV lift must produce brighter pixels (mean_in={mean_in}, mean_out={mean_out})");
    }

    #[test]
    fn dispatch_post_process_brightness_lift_inverse_roundtrip() {
        // apply_brightness_lift then apply_brightness_lift_inverse should
        // recover the original within 2/255 for non-saturated pixels.
        let original = grey_image(4, 4, 100);
        let mut lifted = original.clone();
        prunr_core::denoise::apply_brightness_lift(&mut lifted, 1.0);
        apply_post_inference(&mut lifted, 1.0);
        for (orig_px, out_px) in original.pixels().zip(lifted.pixels()) {
            let diff = (orig_px[0] as i32 - out_px[0] as i32).unsigned_abs();
            // Skip pixels that clipped during the forward lift (orig=100 with 1 EV
            // won't clip, but be defensive).
            assert!(diff <= 2,
                "brightness_lift round-trip must recover within 2/255 (diff={diff})");
        }
    }

    #[test]
    fn dispatch_post_process_noop_when_lift_zero() {
        let original = grey_image(4, 4, 128);
        let mut img = original.clone();
        apply_post_inference(&mut img, 0.0);
        assert_eq!(img.as_raw(), original.as_raw(),
            "apply_post_inference must be a no-op when brightness_lift=0");
    }
}

// ── Warm-engine cache boundary contract ──────────────────────────────────────
//
// The four tests below pin the cache-hit / cache-eviction / release semantics
// on `Processor::warm_upscale_engine` without exercising real upscale inference.
// Tests 1-3 inject a sentinel Arc<OrtEngine> via the test-only accessor and
// assert pointer-identity or slot emptiness. Test 4 needs no engine at all.
//
// OrtEngine construction requires ort_runtime::init() (the ORT dylib must be
// loaded). Tests 1-3 call `ensure_ort_for_test()` which skips gracefully when
// the dylib isn't available — matching the pattern from prunr-core's
// integration test suite. Test 4 has no ORT dependency and always runs.
#[cfg(test)]
mod warm_cache_tests {
    use super::*;
    use prunr_core::{ModelKind, OrtEngine};
    use prunr_core::engine::GraphOptimizationLevel;

    fn make_processor() -> Processor {
        let (tx, _rx) = mpsc::channel::<WorkerMessage>();
        let (_tx, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }

    /// Call `ort_runtime::init()` once per process, then return whether
    /// the ORT runtime is available. `false` means the dylib wasn't found
    /// and the caller should skip. Matching the `skip_if_no_ort` pattern
    /// from `prunr-core/tests/test_common/mod.rs`.
    fn ensure_ort_for_test() -> bool {
        static INIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *INIT.get_or_init(|| crate::ort_runtime::init().is_ok())
    }

    /// Construct one real engine (Silueta, ~4 MB bundled) and reuse it
    /// across the three Arc-identity tests via OnceLock. Silueta bytes
    /// are always available — no download required. Returns None when
    /// ORT isn't available (no dylib found).
    fn sentinel_engine() -> Option<Arc<OrtEngine>> {
        if !ensure_ort_for_test() {
            return None;
        }
        static CELL: std::sync::OnceLock<Arc<OrtEngine>> = std::sync::OnceLock::new();
        Some(
            CELL.get_or_init(|| {
                Arc::new(
                    OrtEngine::new_cpu_only_with_optimization_level(
                        ModelKind::Silueta,
                        1,
                        GraphOptimizationLevel::Level1,
                    )
                    .expect("Silueta is bundled and must construct without errors once ORT is init"),
                )
            })
            .clone(),
        )
    }

    #[test]
    fn warm_upscale_engine_field_starts_empty() {
        let mut p = make_processor();
        assert!(
            p.warm_upscale_engine_for_test().is_none(),
            "Processor::new must not pre-populate the warm-engine cache"
        );
    }

    #[test]
    fn cache_hit_reuses_arc_on_same_modelkind() {
        let Some(sentinel) = sentinel_engine() else {
            eprintln!("[cache_hit_reuses_arc_on_same_modelkind] SKIP: ORT runtime not found");
            return;
        };
        let mut p = make_processor();
        *p.warm_upscale_engine_for_test() =
            Some((ModelKind::RealEsrganX4Plus, Arc::clone(&sentinel)));
        let got = p
            .try_cached_upscale_engine(ModelKind::RealEsrganX4Plus)
            .expect("cache hit must return the sentinel Arc");
        assert!(
            Arc::ptr_eq(&sentinel, &got),
            "cache hit must return the SAME Arc that was inserted"
        );
    }

    #[test]
    fn model_swap_evicts_cached_arc() {
        let Some(sentinel) = sentinel_engine() else {
            eprintln!("[model_swap_evicts_cached_arc] SKIP: ORT runtime not found");
            return;
        };
        let mut p = make_processor();
        *p.warm_upscale_engine_for_test() =
            Some((ModelKind::RealEsrganX4Plus, Arc::clone(&sentinel)));
        let got = p.try_cached_upscale_engine(ModelKind::Nomos8kSchatL);
        assert!(
            got.is_none(),
            "cache miss on ModelKind mismatch must return None"
        );
        assert!(
            p.warm_upscale_engine_for_test().is_none(),
            "slot must be evicted after a ModelKind mismatch lookup"
        );
    }

    #[test]
    fn release_upscale_engine_clears_slot() {
        let Some(sentinel) = sentinel_engine() else {
            eprintln!("[release_upscale_engine_clears_slot] SKIP: ORT runtime not found");
            return;
        };
        let mut p = make_processor();
        *p.warm_upscale_engine_for_test() =
            Some((ModelKind::RealEsrganX4Plus, Arc::clone(&sentinel)));
        p.release_upscale_engine();
        assert!(
            p.warm_upscale_engine_for_test().is_none(),
            "release_upscale_engine must clear the slot"
        );
    }
}

