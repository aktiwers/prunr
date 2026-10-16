//! In-process live-preview dispatcher for Tier 2 reruns.
//!
//! When the user tweaks a mask or edge knob in the adjustments toolbar,
//! `mark_tweak` registers a pending preview for the affected item. Every
//! frame, `tick` checks if `DEBOUNCE` has passed since the window's first
//! tweak and, if so, spawns a background thread to run the Tier 2
//! postprocess (mask) or edge finalize (edge) on the cached tensor.
//!
//! On an image larger than `QUICK_FRAME_SIDE`, a drag's in-between renders
//! are quick frames. The full-size render goes out when the slider is let
//! go (`flush`) or held still for `SETTLE_AFTER`, whichever comes first.
//!
//! Staleness handling: if multiple dispatches are in flight for the same item
//! (possible during a continuous drag — a new dispatch fires every DEBOUNCE
//! while earlier ones are still running), `drain_results` drops any result
//! whose generation doesn't match the most-recently-dispatched generation. The
//! cancel token is still held per-dispatch but is only triggered on
//! `cancel_all` (batch clear / shutdown) — letting in-flight dispatches
//! complete during a drag is what produces live updates mid-drag.
//!
//! **Do not reintroduce per-tweak cancellation.** It was removed on purpose:
//! `postprocess` doesn't check the cancel token at stage boundaries, so the
//! token only gates the final `tx.send`. Cancelling mid-drag therefore dropped
//! every dispatch's result before it could be drained, which is exactly the
//! "no live preview during drag" bug this module was rewritten to fix.
//!
//! No subprocess involved — Tier 2 is pure CPU work, so running it in-process
//! saves ~20-50ms of IPC overhead per tick and works even outside a batch.
//! The rayon thread pool handles parallel previews across multiple items.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use image::{DynamicImage, GrayImage, RgbaImage};

use prunr_core::depth::LineFade;
use prunr_core::{LineMode, ModelKind, postprocess_from_flat, tensor_to_edge_mask, thickened_edges, compose_edges_styled, compose_edges_dual_styled};

use crate::gui::item_settings::ItemSettings;
use crate::gui::worker::CompressedTensor;

/// Build the "masked base" for SubjectOutline live preview — run Tier 2 mask
/// from the cached seg tensor to reproduce the segmented subject, so edges
/// composite onto the masked subject instead of the raw photo. Returns None
/// on postprocess failure; caller falls back to the raw original.
fn build_masked_base(
    seg: &SegTensor,
    original: &DynamicImage,
    settings: &ItemSettings,
    correction: Option<&prunr_core::selection::MaskArtifact>,
) -> Option<DynamicImage> {
    let mask_settings = settings.mask_settings();
    let opts = prunr_core::PostprocessOpts::new(&mask_settings, seg.model)
        .with_correction(correction);
    postprocess_from_flat(
        &seg.data,
        seg.height as usize,
        seg.width as usize,
        original,
        &opts,
    )
    .ok()
    .map(DynamicImage::ImageRgba8)
}

/// Throttle cadence for live-preview dispatch during a continuous drag.
/// 150ms stays above the ~90ms baseline and ~240ms refine_edges worker
/// cost so dispatches don't pile up for the generation filter to discard.
/// Release / commit paths call `flush` instead, so the final result on
/// knob-release lands immediately.
pub const DEBOUNCE: Duration = Duration::from_millis(150);

/// How long a drag holds still before its full-size render is made, for
/// a user who waits with the slider held instead of letting go.
pub const SETTLE_AFTER: Duration = Duration::from_millis(300);

/// What kind of Tier 2 rerun a tweak needs. Two kinds because they touch
/// different cached tensors (segmentation vs TEED) and different pipeline
/// stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewKind {
    /// gamma / threshold / edge_shift / refine_edges → Tier 2 mask rerun.
    Mask,
    /// line_strength / solid_line_color → Tier 2 edge rerun.
    Edge,
    /// sharpen / ai_blend / color_match → Tier-2 upscale postprocess
    /// on the kept upscale run. No ORT inference — snapshot provides
    /// the cached buffer directly.
    UpscaleTier2,
    /// Strength → re-render the kept colorize run. No model run.
    ColorizeTier2,
    /// The unprocessed photo with what goes on it before any model runs
    /// (restored faces): shown in place of the photo, never a result.
    Photo,
}

impl PreviewKind {
    /// The kind that re-renders the photo and all drawn on it under
    /// `line_mode`: lines over the whole photo are an Edge render (a Mask
    /// render there would draw the photo without them); otherwise a Mask
    /// render, which also redraws Subject lines.
    pub fn photo_rerun(line_mode: LineMode) -> Self {
        if line_mode == LineMode::EdgesOnly { Self::Edge } else { Self::Mask }
    }
}

/// The upscale result to show: Refine over the model's output
/// (`prunr_core::upscale::refine`), then the restored faces. Faces go in
/// last: AI blend would mix them back toward the plain enlargement, and
/// Sharpen would sharpen redrawn detail.
pub(crate) fn finish_upscale(
    raw: &Arc<RgbaImage>,
    input: &RgbaImage,
    reference: Option<Arc<RgbaImage>>,
    knobs: &prunr_core::upscale::Refine,
    faces: Option<&FacedKey>,
) -> (Arc<RgbaImage>, Option<Arc<RgbaImage>>) {
    let (shown, reference) = prunr_core::upscale::refine(raw, input, reference, knobs);
    let shown = match faces {
        // The faces were found in `raw`; over a refined image the blend
        // takes the photo under each face from it, or it would undo Refine.
        Some(f) => {
            let refined = !Arc::ptr_eq(&shown, raw);
            let mut faced = Arc::unwrap_or_clone(shown);
            if refined {
                prunr_core::face::compose_faces_onto(&mut faced, &f.faces, f.blend);
            } else {
                prunr_core::face::compose_faces_into(&mut faced, &f.faces, f.blend);
            }
            Arc::new(faced)
        }
        None => shown,
    };
    (shown, reference)
}

/// A completed Tier 2 rerun delivered back to the UI thread.
pub struct PreviewResult {
    pub item_id: u64,
    pub rgba: Arc<RgbaImage>,
    /// The preview kind that produced this result. Used by
    /// `apply_completed_previews` to route kind-specific side-effects
    /// (e.g. advancing `applied_recipe.upscale` for UpscaleTier2 results).
    pub kind: PreviewKind,
    /// Generation counter — used to discard results from dispatches a
    /// newer one superseded.
    pub generation: u64,
    /// The item's timeline version at dispatch: the result lands only on
    /// the state it was computed for.
    pub version: u64,
    /// Where the preview started (`None`: the source).
    pub(crate) start: Option<crate::gui::timeline::RunStart>,
    /// The segmentation it cut with, if any.
    pub(crate) cutter: Option<crate::gui::item::Cutter>,
    /// The edge planes an Edge dispatch built (not the ones it reused), with
    /// the key they were built for, for the item's cache. `None` for the
    /// other kinds.
    pub new_edge: Option<(EdgePlanes, EdgePlaneKey)>,
    /// The Bold planes a dual-scale composition built, for the same cache.
    pub new_bold: Option<(EdgePlanes, EdgePlaneKey)>,
    /// Masked subject base built during this dispatch (SubjectOutline only),
    /// for the parent to cache so subsequent Edge tweaks whose mask recipe
    /// matches can skip `postprocess_from_flat`. Keyed by (MaskRecipe, model).
    pub new_masked_base: Option<(Arc<image::RgbaImage>, prunr_core::MaskRecipe, prunr_core::ModelKind)>,
    /// The upscale reference an UpscaleTier2 dispatch built, for the kept
    /// upscale run.
    pub new_reference: Option<Arc<image::RgbaImage>>,
    /// The faced photo this dispatch built, for the item to reuse.
    pub new_faced: Option<(Arc<DynamicImage>, FacedKey)>,
    /// The depth-looked photo this dispatch built, for the item to reuse.
    pub(crate) new_depth_photo: Option<(Arc<DynamicImage>, crate::gui::item::DepthPhotoKey)>,
    /// The image a quick frame shrank, for the drag's next frames.
    pub(crate) new_quick_photo: Option<QuickPhoto>,
    /// The `DynamicImage` of the render's photo it built, for the item to keep.
    pub(crate) new_original: Option<BuiltOriginal>,
    /// The correction plane the render built, for the item to keep.
    pub(crate) new_correction: Option<crate::gui::item::BuiltCorrection>,
    /// The tensors the render unpacked, for the item's hot cache.
    pub(crate) unpacked: Unpacked,
    /// MaskRecipe snapshot taken when this dispatch was kicked off. The
    /// parent updates `applied_recipe.mask` to match so the recipe-drift
    /// tripwire doesn't immediately re-fire on a result it already
    /// applied.
    pub applied_mask: prunr_core::MaskRecipe,
    /// The Depth chip's look this dispatch rendered, for `applied_recipe`.
    pub applied_depth: prunr_core::depth::DepthLook,
    /// Tier-2 knob values captured at dispatch time. `apply_completed_previews`
    /// patches `applied_recipe.upscale`'s four Tier-2 fields to match so
    /// resolve_tier sees Skip (not UpscaleTier2) on the next frame and
    /// mark_tweak stops firing between settled drags.
    /// `None` for non-UpscaleTier2 kinds.
    pub applied_tier2_knobs: Option<prunr_core::upscale::Refine>,
    /// The Strength a ColorizeTier2 dispatch rendered, patched into
    /// `applied_recipe.colorize` for the same reason. `None` otherwise.
    pub applied_color_strength: Option<f32>,
    /// `true` when no further tweaks are pending for this item at drain
    /// time — the drag has settled and this is the last result of the
    /// session. Callers gate heavy side-effects (sidebar thumb rebuild)
    /// on this so the mid-drag sidebar doesn't flicker.
    pub is_final: bool,
    /// A drag's in-between frame of a large image: shown, never kept as
    /// the result (see `QuickFrames`).
    pub quick: bool,
}

/// What an image shows of a drag's quick frames: the newest one's
/// texture, drawn over the image in place of the result. A quick frame is
/// never the result, its step or its thumbnail; it stays up until the
/// full-size render's texture lands, so the old result never flashes back.
#[derive(Default)]
pub(crate) struct QuickFrames {
    /// The newest quick frame drained (0: none); only its texture is
    /// taken, and the one before it stays up until it lands.
    generation: u64,
    texture: Option<egui::TextureHandle>,
    /// A full-size render was applied after the frame: it goes when that
    /// render's texture lands.
    superseded: bool,
}

impl QuickFrames {
    /// A quick frame of `generation` drained; its texture is on its way.
    pub(crate) fn expect(&mut self, generation: u64) {
        self.generation = generation;
        self.superseded = false;
    }

    /// A quick frame's texture lands: kept when it is the newest drained.
    pub(crate) fn land(&mut self, texture: egui::TextureHandle, generation: u64) {
        if generation == self.generation {
            self.texture = Some(texture);
        }
    }

    /// A full-size render was applied: the frame stays until its texture
    /// lands.
    pub(crate) fn supersede(&mut self) {
        self.superseded = true;
    }

    /// A result texture landed: a superseded frame goes.
    pub(crate) fn result_landed(&mut self) {
        if self.superseded {
            self.clear();
        }
    }

    /// Each frame: the frame goes once no drag works on the image any more
    /// (a cancel, a render that could not be made) and no full-size
    /// render's texture is on its way to replace it.
    pub(crate) fn settle(&mut self, drag_under_way: bool) {
        if !drag_under_way && !self.superseded {
            self.clear();
        }
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// The frame to draw in place of the result, when one is up.
    pub(crate) fn texture(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }
}

/// Source of timestamps for `mark_tweak` / `flush` / `tick`. Production
/// uses the system monotonic clock; tests inject a `MockClock` to drive
/// synthetic time without sleeping or poking private fields.
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

/// Production clock: just delegates to `Instant::now`.
pub struct SystemClock;
impl Clock for SystemClock {
    fn now(&self) -> Instant { Instant::now() }
}

/// State for a pending (debounced) preview dispatch.
struct Pending {
    /// When the throttle window opened: its first tweak, never re-armed.
    armed_at: Instant,
    /// The newest tweak, which the settle on stillness counts from.
    latest_tweak_at: Instant,
    kind: PreviewKind,
    /// The edit settled (slider released): the render is the one kept.
    settled: bool,
}

/// A drag whose last render was a quick frame: it owes the full-size
/// render, made once no tweak has come for `SETTLE_AFTER`.
struct Owed {
    kind: PreviewKind,
    latest_tweak_at: Instant,
}

/// State for an in-flight dispatch. Its cancel token fires on `cancel` /
/// `cancel_all` only.
struct InFlight {
    cancel: Arc<AtomicBool>,
    generation: u64,
}

pub struct LivePreview {
    pending: HashMap<u64, Pending>,
    in_flight: HashMap<u64, InFlight>,
    owed: HashMap<u64, Owed>,
    generation_counter: u64,
    result_tx: mpsc::Sender<Finished>,
    result_rx: mpsc::Receiver<Finished>,
    clock: Arc<dyn Clock>,
}

/// A dispatch that ran to its end: its result, or `None` when it could
/// make none (a missing tensor, a mask that would not build). Either way
/// it is no longer in flight.
pub(crate) struct Finished {
    item_id: u64,
    generation: u64,
    result: Option<PreviewResult>,
}

impl From<PreviewResult> for Finished {
    fn from(r: PreviewResult) -> Self {
        Self { item_id: r.item_id, generation: r.generation, result: Some(r) }
    }
}

impl Default for LivePreview {
    fn default() -> Self {
        Self::with_clock(Arc::new(SystemClock))
    }
}

impl LivePreview {
    /// Construct with an injected clock. Production callers go through
    /// `default()` (which uses `SystemClock`); tests pass a fake to drive
    /// synthetic timestamps.
    pub fn with_clock(clock: Arc<dyn Clock>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            pending: HashMap::new(),
            in_flight: HashMap::new(),
            owed: HashMap::new(),
            generation_counter: 0,
            result_tx: tx,
            result_rx: rx,
            clock,
        }
    }

    /// Pin `armed_at` on the first tweak in a window and never re-arm —
    /// `tick`'s removal-on-dispatch starts the next window.
    pub fn mark_tweak(&mut self, item_id: u64, kind: PreviewKind) {
        let now = self.clock.now();
        let is_new = !self.pending.contains_key(&item_id);
        self.pending
            .entry(item_id)
            .and_modify(|p| {
                p.kind = kind;
                p.settled = false;
                p.latest_tweak_at = now;
            })
            .or_insert_with(|| Pending { armed_at: now, latest_tweak_at: now, kind, settled: false });
        tracing::debug!(
            event = "live_preview_mark_tweak",
            item_id,
            ?kind,
            is_new_entry = is_new,
            pending_count = self.pending.len(),
            in_flight_count = self.in_flight.len(),
            "mark_tweak registered"
        );
    }

    /// Flush: expire the pending tweak timer so the next `tick` dispatches
    /// immediately, as the settled render. Used when an edit settles
    /// (slider released, checkbox toggled, color picked) so the user
    /// doesn't wait the full debounce. A drag whose last frame was quick
    /// and has nothing queued gets its full-size render queued.
    ///
    /// Idempotent — no-op if there's nothing pending or owed for `item_id`.
    pub fn flush(&mut self, item_id: u64) {
        let expired = self.expired_arm();
        if let Some(p) = self.pending.get_mut(&item_id) {
            p.settled = true;
            p.armed_at = expired;
        } else if let Some(owed) = self.owed.remove(&item_id) {
            self.pending.insert(item_id, Pending { armed_at: expired, latest_tweak_at: owed.latest_tweak_at, kind: owed.kind, settled: true });
        }
    }

    /// Queue `item_id`'s settled render for the next tick, outside any drag.
    pub(crate) fn render_now(&mut self, item_id: u64, kind: PreviewKind) {
        self.mark_tweak(item_id, kind);
        self.flush(item_id);
    }

    /// Dispatch `item_id`'s queued render at the next tick, as a drag's
    /// in-between frame: it is not settled.
    pub(crate) fn hurry(&mut self, item_id: u64) {
        let expired = self.expired_arm();
        if let Some(p) = self.pending.get_mut(&item_id) {
            p.armed_at = expired;
        }
    }

    /// An arm time already past DEBOUNCE, so the next tick dispatches.
    /// checked_sub guards against early-boot monotonic clock underflow.
    fn expired_arm(&self) -> Instant {
        let now = self.clock.now();
        now.checked_sub(DEBOUNCE + Duration::from_millis(10)).unwrap_or(now)
    }

    /// Per-frame tick. Returns the time until the next tween we're waiting on
    /// (useful for `ctx.request_repaint_after`), or `None` if nothing pending.
    /// That includes a held-still drag's settle: a pointer that does not
    /// move sends no input, so nothing else would wake the frame loop.
    ///
    /// `snapshot` provides per-item inputs (tensors, settings, original image)
    /// for any dispatches we fire this frame. A dispatch is quick when the
    /// edit has not settled (a drag's in-between frame) and its image is
    /// larger than a quick frame. A render that cannot be built yet waits
    /// for a later tick; one that never can (`Verdict::Never`) is dropped,
    /// so nothing waits for good.
    pub(crate) fn tick<F>(&mut self, mut snapshot: F) -> Option<Duration>
    where
        F: FnMut(u64, PreviewKind) -> Result<DispatchInputs, crate::gui::capability::Verdict>,
    {
        let now = self.clock.now();
        let mut ready_ids: Vec<u64> = Vec::new();
        let mut wait_for: Option<Duration> = None;
        let mut wait = |remaining: Duration| wait_for = Some(wait_for.map_or(remaining, |w: Duration| w.min(remaining)));

        let mut held_still: Vec<u64> = Vec::new();
        for (id, o) in self.owed.iter().filter(|(id, _)| !self.pending.contains_key(id)) {
            let held = now.saturating_duration_since(o.latest_tweak_at);
            if held >= SETTLE_AFTER {
                held_still.push(*id);
            } else {
                wait(SETTLE_AFTER - held);
            }
        }
        for id in held_still {
            self.flush(id);
        }

        for (id, p) in &self.pending {
            let elapsed = now.saturating_duration_since(p.armed_at);
            if elapsed >= DEBOUNCE {
                ready_ids.push(*id);
            } else {
                wait(DEBOUNCE - elapsed);
            }
        }

        for id in ready_ids {
            // Peek the pending kind without removing — if the snapshot isn't
            // ready yet (e.g. `source_rgba` decode in flight after a view
            // switch), we keep the pending entry so the next tick retries.
            // Dropping it here silently swallowed the tweak and left the
            // user thinking live preview was dead.
            let (kind, settled, latest_tweak_at) = match self.pending.get(&id) {
                Some(p) => (p.kind, p.settled, p.latest_tweak_at),
                None => continue,
            };
            tracing::debug!(
                event = "live_preview_tick_ready",
                item_id = id,
                ?kind,
                in_flight_count = self.in_flight.len(),
                "tick: item ready for dispatch"
            );
            let mut inputs = match snapshot(id, kind) {
                Ok(inputs) => inputs,
                Err(verdict) => {
                    let why = verdict.reason().map(crate::gui::capability::Reason::text);
                    tracing::debug!(event = "live_preview_tick_unbuilt", item_id = id, ?kind, ?why, "render not built");
                    if !verdict.allows_request() {
                        self.pending.remove(&id);
                        self.owed.remove(&id);
                    }
                    continue;
                }
            };
            self.pending.remove(&id);
            inputs.quick = !settled && inputs.can_be_quick();
            if inputs.quick {
                self.owed.insert(id, Owed { kind, latest_tweak_at });
            } else {
                self.owed.remove(&id);
            }
            self.generation_counter = self.generation_counter.wrapping_add(1);
            let generation = self.generation_counter;
            let cancel = Arc::new(AtomicBool::new(false));
            self.in_flight.insert(id, InFlight { cancel: cancel.clone(), generation });

            let tx = self.result_tx.clone();
            rayon::spawn(move || {
                let has_depth = inputs.depth.is_some();
                let plane_key = |scale: prunr_core::EdgeScale| EdgePlaneKey::new(&inputs.settings, scale, has_depth);
                let primary_key = plane_key(inputs.settings.edge_settings().primary_scale());
                let bold_key = plane_key(prunr_core::EdgeScale::Bold);
                let is_edge = matches!(inputs.kind, PreviewKind::Edge);
                let dispatch_kind = inputs.kind;
                let mask_recipe = prunr_core::MaskRecipe::from(&inputs.settings.mask_settings());
                let seg_model = inputs.seg_model();
                // Capture Tier-2 knob values at dispatch time. When the result
                // lands, apply_completed_previews patches only the 4 Tier-2
                // fields (sharpen/ai_blend/color_match) on
                // applied_recipe.upscale so the next resolve_tier diff sees Skip
                // instead of UpscaleTier2, stopping the per-frame mark_tweak loop
                // after the user stops dragging.
                let applied_tier2_knobs = inputs.upscale_tier2_knobs;
                let applied_color_strength = matches!(inputs.kind, PreviewKind::ColorizeTier2).then_some(inputs.settings.color_strength());
                let applied_depth = inputs.settings.depth;
                let quick = inputs.quick;
                let version = inputs.version;
                let start = inputs.start.clone();
                let faced_key = inputs.faces.as_ref().map(|f| f.key.clone());
                let started = std::time::Instant::now();
                let output = run_preview(inputs, &cancel);
                tracing::debug!(item_id = id, kind = ?dispatch_kind, ms = started.elapsed().as_millis() as u64, "live preview ran");
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                let Some(rgba) = output.rgba else {
                    let _ = tx.send(Finished { item_id: id, generation, result: None });
                    return;
                };
                let new_edge = is_edge.then_some((output.built_edge, primary_key));
                let new_bold = is_edge.then_some((output.built_bold, bold_key));
                let new_masked_base = output.built_masked_base
                    .zip(seg_model)
                    .map(|(base, model)| (base, mask_recipe, model));
                // `is_final` is set by `drain_results`, where the UI
                // thread can read `self.pending` atomically. The worker
                // ships a placeholder and doesn't care.
                let _ = tx.send(Finished::from(PreviewResult {
                    item_id: id, rgba, kind: dispatch_kind, generation, version, start,
                    cutter: seg_model.map(crate::gui::item::Cutter::Model),
                    new_edge, new_bold, new_masked_base,
                    new_reference: output.built_reference,
                    new_faced: output.built_faced.zip(faced_key),
                    new_depth_photo: output.built_depth_photo,
                    new_quick_photo: output.built_quick_photo,
                    new_original: output.built_original,
                    new_correction: output.built_correction,
                    unpacked: output.unpacked,
                    applied_mask: mask_recipe,
                    applied_depth,
                    applied_tier2_knobs,
                    applied_color_strength,
                    is_final: false,
                    quick,
                }));
            });
        }

        wait_for
    }

    /// Drain any completed previews. Returned results are already filtered
    /// to the latest generation; each result's `is_final` is set to `true`
    /// iff nothing more is coming for that item — no new tweak pending and
    /// no full-size render owed — i.e. the drag has settled and this is the
    /// last result of the session.
    pub fn drain_results(&mut self) -> Vec<PreviewResult> {
        let mut out = Vec::new();
        while let Ok(done) = self.result_rx.try_recv() {
            // Drop stale: if generation doesn't match the last dispatch for
            // this item, a newer tweak superseded it.
            let is_latest = self.in_flight
                .get(&done.item_id)
                .is_some_and(|f| f.generation == done.generation);
            if !is_latest {
                crate::gui::reclaim::later(done);
                continue;
            }
            self.in_flight.remove(&done.item_id);
            if let Some(mut r) = done.result {
                r.is_final = !self.pending.contains_key(&r.item_id) && !self.owed.contains_key(&r.item_id);
                out.push(r);
            }
        }
        out
    }

    /// The images with a render queued that reads the photo (not a Refine
    /// or Strength re-render of a kept run).
    pub(crate) fn pending_photo_renders(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.iter()
            .filter(|(_, p)| !matches!(p.kind, PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2))
            .map(|(id, _)| *id)
    }

    /// True while a preview waits for its debounce or runs, or a drag
    /// owes its full-size render.
    pub fn is_busy(&self) -> bool {
        !self.pending.is_empty() || !self.in_flight.is_empty() || !self.owed.is_empty()
    }

    /// True while `item_id` has a render queued, running, or owed.
    pub(crate) fn works_on(&self, item_id: u64) -> bool {
        self.pending.contains_key(&item_id) || self.in_flight.contains_key(&item_id) || self.owed.contains_key(&item_id)
    }

    /// True while a render for `item_id` waits for its debounce.
    pub(crate) fn is_pending_for(&self, item_id: u64) -> bool {
        self.pending.contains_key(&item_id)
    }

    /// True while a rayon dispatch hasn't drained yet.
    pub fn has_in_flight(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Drop the pending, running and owed previews of one item: what they
    /// would show belongs to a state it has left.
    pub fn cancel(&mut self, item_id: u64) {
        if let Some(f) = self.in_flight.remove(&item_id) {
            f.cancel.store(true, Ordering::Release);
        }
        self.pending.remove(&item_id);
        self.owed.remove(&item_id);
    }

    /// Cancel all in-flight + pending previews. Called on batch clear or shutdown.
    pub fn cancel_all(&mut self) {
        for f in self.in_flight.values() {
            f.cancel.store(true, Ordering::Release);
        }
        self.in_flight.clear();
        self.pending.clear();
        self.owed.clear();
    }
}

/// Inputs required to actually run a single preview. Snapshot captures these
/// on the UI thread (holding the item briefly) and hands them to the worker
/// so the worker doesn't need a `&mut BatchItem`.
pub struct DispatchInputs {
    pub kind: PreviewKind,
    /// The item's timeline version, carried to the result.
    pub version: u64,
    /// A drag's in-between frame: the depth look may be made at a smaller
    /// size, and the result is not kept for the next one.
    pub quick: bool,
    pub original: Arc<DynamicImage>,
    /// Where `original` comes from when the item had no `DynamicImage`
    /// of it yet: the render's thread makes it (a full copy, or an unpack
    /// first) and hands it back, so the UI thread never does.
    pub(crate) original_from: Option<OriginalFrom>,
    /// Where the preview starts when that is an earlier result (`None`: the
    /// source), recorded as the step's start on landing.
    pub(crate) start: Option<crate::gui::timeline::RunStart>,
    pub settings: ItemSettings,
    /// Segmentation tensor (decompressed) + dimensions + model. Required for
    /// `PreviewKind::Mask` *or* `PreviewKind::Edge` in SubjectOutline mode
    /// (the edge was computed on top of a segmented subject; preview reuses
    /// that as the base image).
    pub seg_tensor: Option<Arc<SegTensor>>,
    pub edge_tensor: Option<EdgeTensor>,
    /// Secondary edge tensor for multi-scale LineStyle variants
    /// (DualScale). Only populated when the style needs it, so single-scale
    /// dispatches skip the extra zstd decompress.
    pub secondary_edge_tensor: Option<EdgeTensor>,
    /// The tensors above still packed, unpacked on the render's thread.
    pub(crate) packed: Packed,
    /// The edge planes of previous dispatches that still match the current
    /// strength, scale and thickness, so a tweak redoes only what changed.
    pub cached_edge: EdgePlanes,
    /// Same for the Bold scale of a dual-scale style.
    pub cached_bold: EdgePlanes,
    /// SubjectOutline masked-subject base (output of `postprocess_from_flat`)
    /// from a previous dispatch whose mask recipe matches the current one.
    /// Populated when available so Edge tweaks skip Lanczos + guided filter.
    pub cached_masked_base: Option<Arc<image::RgbaImage>>,
    /// The item's selection, applied to the segmentation mask before the
    /// resize / refine / feather chain (resampled to tensor resolution
    /// inline). `None` when nothing is selected. Built on the render's
    /// thread when not built yet.
    pub(crate) correction: Option<crate::gui::item::Correction>,
    /// For `PreviewKind::UpscaleTier2` only — the kept upscale run, which
    /// Tier-2 re-finishes. `None` for other kinds.
    pub(crate) upscale: Option<crate::gui::item::UpscaleCache>,
    /// For `PreviewKind::UpscaleTier2` only — snapshot of the Refine knobs.
    pub upscale_tier2_knobs: Option<prunr_core::upscale::Refine>,
    /// For `PreviewKind::ColorizeTier2` only — the kept colorize run,
    /// re-rendered at `settings.color_strength()`.
    pub(crate) colorize: Option<crate::gui::item::ColorizeCache>,
    /// Restored faces to blend into `original` before anything else.
    pub faces: Option<FacesInput>,
    /// The photo's depth map, for the Depth chip's look and fading lines.
    pub(crate) depth: Option<DepthInput>,
    /// The image shrunk for an earlier quick frame.
    pub(crate) quick_photo: Option<QuickPhoto>,
}

/// What a preview reads from the depth map: the look to put on the photo
/// after the faces (No model only), and the photo it made last time when
/// that still matches.
#[derive(Clone)]
pub(crate) struct DepthInput {
    pub(crate) map: Arc<prunr_core::depth::DepthMap>,
    pub(crate) look: Option<prunr_core::depth::DepthLook>,
    pub(crate) photo: Option<Arc<DynamicImage>>,
}

impl DispatchInputs {
    /// The model whose segmentation the render re-cuts, unpacked or not.
    fn seg_model(&self) -> Option<ModelKind> {
        self.seg_tensor.as_ref().map(|s| s.model).or_else(|| self.packed.seg.as_ref().map(|s| s.model))
    }

    /// Unpack what came packed into the tensors a render reads; what was
    /// unpacked goes back for the item's hot cache.
    fn unpack(&mut self) -> Unpacked {
        let packed = std::mem::take(&mut self.packed);
        let seg = packed.seg.and_then(|ct| decompress_seg(&ct).map(|seg| (ct, Arc::new(seg))));
        if let Some((_, seg)) = &seg {
            self.seg_tensor = Some(Arc::clone(seg));
        }
        let Some(edges) = packed.edges else { return Unpacked { seg, edge: None } };
        let tensor = |scale| edges.decompress(scale).map(|data| {
            let (height, width) = edges.dims(scale);
            EdgeTensor { data: Arc::new(data), height, width }
        });
        let edge = packed.primary.and_then(|scale| tensor(scale).map(|t| (scale, t)));
        if packed.bold {
            self.secondary_edge_tensor = tensor(prunr_core::EdgeScale::Bold);
        }
        let edge = edge.map(|(scale, t)| {
            let data = Arc::clone(&t.data);
            self.edge_tensor = Some(t);
            (edges.clone(), scale, data)
        });
        Unpacked { seg, edge }
    }

    /// The correction plane; a render resolves a plane still to build
    /// before it reads one.
    fn correction_plane(&self) -> Option<&prunr_core::selection::MaskArtifact> {
        match self.correction.as_ref()? {
            crate::gui::item::Correction::Ready(plane) => Some(plane),
            crate::gui::item::Correction::Build(_) => None,
        }
    }

    /// A `kind` preview of `original` with `settings` and nothing cached,
    /// kept or found yet; callers set what their kind reads.
    pub(crate) fn new(kind: PreviewKind, version: u64, original: Arc<DynamicImage>, settings: ItemSettings) -> Self {
        Self {
            kind,
            version,
            quick: false,
            original,
            original_from: None,
            start: None,
            settings,
            seg_tensor: None,
            edge_tensor: None,
            secondary_edge_tensor: None,
            packed: Packed::default(),
            cached_edge: EdgePlanes::default(),
            cached_bold: EdgePlanes::default(),
            cached_masked_base: None,
            correction: None,
            upscale: None,
            upscale_tier2_knobs: None,
            colorize: None,
            faces: None,
            depth: None,
            quick_photo: None,
        }
    }

    /// Whether a drag's in-between frame of this render is a quick one:
    /// one of an image larger than a quick frame. An upscale with restored
    /// faces re-finishes at full size: the faces go back where the model's
    /// output has them.
    pub(crate) fn can_be_quick(&self) -> bool {
        let (w, h) = match self.kind {
            PreviewKind::Mask | PreviewKind::Edge => self.original_from.as_ref()
                .map_or((self.original.width(), self.original.height()), OriginalFrom::dims),
            PreviewKind::UpscaleTier2 if self.faces.is_some() => return false,
            PreviewKind::UpscaleTier2 => self.upscale.as_ref().map_or((0, 0), |u| u.raw.dimensions()),
            PreviewKind::ColorizeTier2 => self.colorize.as_ref().map_or((0, 0), |c| c.input.dimensions()),
            PreviewKind::Photo => return false,
        };
        w.max(h) > QUICK_FRAME_SIDE
    }
}

/// The faces found for a preview's photo and how they go back; `faced` is
/// the photo with them blended in, when a dispatch already built it for
/// this blend.
#[derive(Clone)]
pub struct FacesInput {
    pub key: FacedKey,
    pub faced: Option<Arc<DynamicImage>>,
}

/// What a faced photo was built from: these faces at this blend.
#[derive(Clone)]
pub struct FacedKey {
    pub faces: Arc<Vec<prunr_core::face::RestoredFace>>,
    pub blend: prunr_core::face::FaceBlend,
}

impl FacedKey {
    /// `photo` with these faces blended in.
    pub(crate) fn blend_into(&self, mut photo: RgbaImage) -> RgbaImage {
        prunr_core::face::compose_faces_into(&mut photo, &self.faces, self.blend);
        photo
    }

    pub fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.faces, &other.faces) && self.blend == other.blend
    }
}

/// What an edge plane was built from; a plane is reused only on an exact
/// match. Fade with distance applies to the dilated plane only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgePlaneKey {
    pub strength_bits: u32,
    pub scale: prunr_core::EdgeScale,
    pub thickness: u32,
    pub fade_pct: u8,
}

impl EdgePlaneKey {
    /// The key of `scale`'s planes under `settings`; lines fade only where
    /// the depth map is there to fade them by.
    pub fn new(settings: &ItemSettings, scale: prunr_core::EdgeScale, has_depth: bool) -> Self {
        Self {
            strength_bits: settings.line_strength.to_bits(),
            scale,
            thickness: u32::from(settings.edge_thickness),
            fade_pct: fade_pct(settings, has_depth),
        }
    }
}

/// What the undilated plane depends on: strength and scale, not thickness.
type EdgeBaseKey = (u32, prunr_core::EdgeScale);

impl EdgePlaneKey {
    fn base_key(self) -> EdgeBaseKey {
        (self.strength_bits, self.scale)
    }
}

/// The two planes of one edge scale: the undilated mask (tensor → sigmoid
/// → Lanczos) and the dilated one a composition draws. Handed to a
/// dispatch as what is cached, and handed back as what it built.
#[derive(Clone, Default)]
pub struct EdgePlanes {
    pub base: Option<Arc<GrayImage>>,
    pub dilated: Option<Arc<GrayImage>>,
}

/// An item's edge planes per scale, each plane with the key it was built
/// with: Bold (the second layer of the dual-scale style) and the newest
/// other scale, as many planes as two separate caches held. A thickness
/// drag keeps the base and redoes the dilation only; a colour or compose
/// drag keeps both.
#[derive(Default)]
pub(crate) struct EdgePlaneCache {
    slots: Vec<ScalePlanes>,
}

struct ScalePlanes {
    scale: prunr_core::EdgeScale,
    base: Option<(Arc<GrayImage>, EdgeBaseKey)>,
    dilated: Option<(Arc<GrayImage>, EdgePlaneKey)>,
}

impl EdgePlaneCache {
    /// The planes usable for `key`.
    pub(crate) fn lookup(&self, key: EdgePlaneKey) -> EdgePlanes {
        let Some(slot) = self.slots.iter().find(|s| s.scale == key.scale) else { return EdgePlanes::default() };
        EdgePlanes {
            base: slot.base.as_ref().filter(|(_, k)| *k == key.base_key()).map(|(m, _)| Arc::clone(m)),
            dilated: slot.dilated.as_ref().filter(|(_, k)| *k == key).map(|(m, _)| Arc::clone(m)),
        }
    }

    /// Keep what a dispatch built for `key`'s scale. A dispatch that built
    /// nothing (every plane was a hit) changes nothing.
    pub(crate) fn store(&mut self, built: EdgePlanes, key: EdgePlaneKey) {
        use prunr_core::EdgeScale::Bold;
        if built.base.is_none() && built.dilated.is_none() {
            return;
        }
        let i = match self.slots.iter().position(|s| s.scale == key.scale) {
            Some(i) => i,
            None => {
                // A new non-Bold scale replaces the old one.
                self.slots.retain(|s| s.scale == Bold || key.scale == Bold);
                self.slots.push(ScalePlanes { scale: key.scale, base: None, dilated: None });
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[i];
        if let Some(b) = built.base {
            slot.base = Some((b, key.base_key()));
        }
        if let Some(d) = built.dilated {
            slot.dilated = Some((d, key));
        }
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// One built dilated plane, for tests that check the cache is dropped.
    #[cfg(test)]
    pub(crate) fn store_placeholder(&mut self) {
        let key = EdgePlaneKey { strength_bits: 0, scale: prunr_core::EdgeScale::Fused, thickness: 0, fade_pct: 0 };
        self.store(EdgePlanes { base: None, dilated: Some(Arc::new(GrayImage::new(1, 1))) }, key);
    }
}

pub struct SegTensor {
    pub data: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub model: ModelKind,
}

pub struct EdgeTensor {
    pub data: Arc<Vec<f32>>,
    pub height: u32,
    pub width: u32,
}

/// Helper: decompress a `CompressedTensor` into the raw form the preview worker needs.
pub fn decompress_seg(ct: &CompressedTensor) -> Option<SegTensor> {
    Some(SegTensor {
        data: ct.decompress()?,
        height: ct.height,
        width: ct.width,
        model: ct.model,
    })
}

/// What `run_preview` produces. The built planes and `built_masked_base`
/// hold only what this dispatch constructed, so the parent caches them —
/// reused inputs pass through empty to avoid re-publishing.
#[derive(Default)]
struct RunOutput {
    rgba: Option<Arc<RgbaImage>>,
    built_edge: EdgePlanes,
    built_bold: EdgePlanes,
    built_masked_base: Option<Arc<image::RgbaImage>>,
    built_reference: Option<Arc<image::RgbaImage>>,
    built_faced: Option<Arc<DynamicImage>>,
    built_depth_photo: Option<(Arc<DynamicImage>, crate::gui::item::DepthPhotoKey)>,
    built_quick_photo: Option<QuickPhoto>,
    built_original: Option<BuiltOriginal>,
    built_correction: Option<crate::gui::item::BuiltCorrection>,
    unpacked: Unpacked,
}

/// Tensors a render unpacks on its thread: the segmentation, and of the
/// line scales the primary one and, for a dual-scale style, Bold.
#[derive(Default)]
pub(crate) struct Packed {
    pub(crate) seg: Option<CompressedTensor>,
    pub(crate) edges: Option<crate::gui::worker::CompressedEdgeTensors>,
    pub(crate) primary: Option<prunr_core::EdgeScale>,
    pub(crate) bold: bool,
}

/// What a render unpacked, with the packed form it came from, for the
/// item to keep while its caches are still those.
#[derive(Default, Clone)]
pub(crate) struct Unpacked {
    pub(crate) seg: Option<(CompressedTensor, Arc<SegTensor>)>,
    pub(crate) edge: Option<(crate::gui::worker::CompressedEdgeTensors, prunr_core::EdgeScale, Arc<Vec<f32>>)>,
}

/// The photo a render draws on, before it is a `DynamicImage`.
#[derive(Clone)]
pub(crate) enum OriginalFrom {
    /// The run's start image, maybe stored compressed; with its size.
    Start(crate::gui::timeline::StoredImage, (u32, u32)),
    /// The decoded photo.
    Photo(Arc<RgbaImage>),
}

impl OriginalFrom {
    fn dims(&self) -> (u32, u32) {
        match self {
            Self::Start(_, dims) => *dims,
            Self::Photo(rgba) => rgba.dimensions(),
        }
    }

    /// The pixels and their `DynamicImage`; `None` if a stored start can
    /// no longer be read.
    fn wrap(&self) -> Option<BuiltOriginal> {
        let (rgba, start) = match self {
            Self::Start(stored, _) => (stored.load()?, true),
            Self::Photo(rgba) => (Arc::clone(rgba), false),
        };
        let wrapped = Arc::new(DynamicImage::ImageRgba8((*rgba).clone()));
        Some(BuiltOriginal { rgba, wrapped, start })
    }
}

/// A `DynamicImage` a render made of its photo, and which photo it is.
#[derive(Clone)]
pub(crate) struct BuiltOriginal {
    pub(crate) rgba: Arc<RgbaImage>,
    pub(crate) wrapped: Arc<DynamicImage>,
    /// Made from the run's start image, not the decoded photo.
    pub(crate) start: bool,
}

/// Runs on a rayon worker thread.
///
/// Respects the user's full mask settings, including `refine_edges`. The
/// guided filter stage adds ~50-150ms on 4K when enabled, so drag cadence
/// slows accordingly — but the user opted into that cost when they toggled
/// Refine Edges on, and without running the filter the three refine knobs
/// (toggle, guided_radius, guided_epsilon) would be no-ops in live preview.
fn run_preview(mut inputs: DispatchInputs, cancel: &AtomicBool) -> RunOutput {
    if cancel.load(Ordering::Acquire) { return RunOutput::default(); }
    let (correction, built_correction) = inputs.correction.take().map(crate::gui::item::Correction::resolve).unzip();
    inputs.correction = correction.map(crate::gui::item::Correction::Ready);
    let unpacked = inputs.unpack();
    RunOutput { built_correction: built_correction.flatten(), unpacked, ..render(inputs, cancel) }
}

fn render(inputs: DispatchInputs, cancel: &AtomicBool) -> RunOutput {

    // UpscaleTier2: no seg/edge tensors involved — run postprocess directly
    // on the kept upscale run.
    if matches!(inputs.kind, PreviewKind::UpscaleTier2) {
        let Some(upscale) = inputs.upscale else {
            tracing::debug!(event = "upscale_tier2_run_abort", reason = "no upscale run kept");
            return RunOutput::default();
        };
        let Some(knobs) = inputs.upscale_tier2_knobs else {
            tracing::debug!(event = "upscale_tier2_run_abort", reason = "upscale_tier2_knobs is None");
            return RunOutput::default();
        };
        if inputs.quick {
            // Sharpen's radius is in output pixels; the plain enlargement
            // is made at the frame's size and not kept.
            let (raw, built_quick_photo) = QuickPhoto::shrink(&upscale.raw, inputs.quick_photo.as_ref());
            let knobs = prunr_core::upscale::Refine { sharpen_sigma: knobs.sharpen_sigma * raw.width() as f32 / upscale.raw.width() as f32, ..knobs };
            let (shown, _) = prunr_core::upscale::refine(&raw, &upscale.input, None, &knobs);
            return RunOutput { rgba: Some(shown), built_quick_photo, ..RunOutput::default() };
        }
        let had_reference = upscale.reference.is_some();
        let faces = inputs.faces.as_ref().map(|f| &f.key);
        let (shown, reference) = finish_upscale(&upscale.raw, &upscale.input, upscale.reference, &knobs, faces);
        let built_reference = reference.filter(|_| !had_reference);
        return RunOutput { rgba: Some(shown), built_reference, ..RunOutput::default() };
    }
    if matches!(inputs.kind, PreviewKind::ColorizeTier2) {
        let Some(run) = inputs.colorize else {
            tracing::debug!(event = "colorize_tier2_run_abort", reason = "no colorize run kept");
            return RunOutput::default();
        };
        let strength = inputs.settings.color_strength();
        if inputs.quick {
            let (input, built_quick_photo) = QuickPhoto::shrink(&run.input, inputs.quick_photo.as_ref());
            let shown = prunr_core::colorize::colorized(&input, &run.plane, strength);
            return RunOutput { rgba: Some(Arc::new(shown)), built_quick_photo, ..RunOutput::default() };
        }
        return RunOutput { rgba: Some(Arc::new(run.render(strength))), ..RunOutput::default() };
    }

    let mut inputs = inputs;
    let built_original = match inputs.original_from.take() {
        Some(from) => {
            let Some(built) = from.wrap() else { return RunOutput::default() };
            inputs.original = Arc::clone(&built.wrapped);
            Some(built)
        }
        None => None,
    };
    // Faces go into the photo before the mask, the effect and the lines.
    let built_faced = match inputs.faces.take() {
        Some(FacesInput { faced: Some(faced), .. }) => {
            inputs.original = faced;
            None
        }
        Some(FacesInput { key, faced: None }) => {
            let faced = Arc::new(DynamicImage::ImageRgba8(key.blend_into(inputs.original.to_rgba8())));
            inputs.original = Arc::clone(&faced);
            Some(faced)
        }
        None => None,
    };
    if cancel.load(Ordering::Acquire) { return RunOutput::default(); }
    if inputs.kind == PreviewKind::Photo {
        return RunOutput { rgba: Some(Arc::new(inputs.original.to_rgba8())), built_faced, built_original, ..RunOutput::default() };
    }
    let full_width = inputs.original.width();
    // A looked photo kept from a full-size render is the one to shrink;
    // otherwise the photo is, and the look goes on the small one.
    let kept_look = inputs.depth.as_ref()
        .is_some_and(|d| d.photo.is_some() && d.look.is_some_and(|l| l.is_active()));
    let mut built_quick_photo = None;
    if inputs.quick && !kept_look {
        (inputs.original, built_quick_photo) = QuickPhoto::shrink(&inputs.original, inputs.quick_photo.as_ref());
    }
    // The Depth chip's look goes on the faced photo, before the lines.
    let built_depth_photo = apply_depth_look(&mut inputs);
    if inputs.quick && kept_look {
        (inputs.original, built_quick_photo) = QuickPhoto::shrink(&inputs.original, inputs.quick_photo.as_ref());
    }
    if inputs.quick {
        shrink_to_quick_frame(&mut inputs, full_width);
    }
    if cancel.load(Ordering::Acquire) { return RunOutput::default(); }
    let output = run_photo_preview(inputs);
    RunOutput { built_faced, built_depth_photo, built_quick_photo, built_original, ..output }
}

/// The long side a drag's quick frames are made at. A larger image's
/// in-between frames are quick; its settled render is full size.
pub(crate) const QUICK_FRAME_SIDE: u32 = 2048;

/// An image shrunk for quick frames (the photo, a kept upscale's output
/// or a colorize run's input), and the one it was shrunk from, held
/// weakly so the pair never keeps a full-size image alive.
#[derive(Clone)]
pub(crate) struct QuickPhoto {
    from: std::sync::Weak<dyn std::any::Any + Send + Sync>,
    small: Small,
}

#[derive(Clone)]
enum Small {
    Photo(Arc<DynamicImage>),
    Image(Arc<RgbaImage>),
}

/// An image a quick frame is made from.
trait Shrinks: Sized + Send + Sync + 'static {
    fn long_side(&self) -> u32;
    fn shrunk(&self) -> Self;
    fn kept(small: &Small) -> Option<Arc<Self>>;
    fn keep(small: Arc<Self>) -> Small;
}

impl Shrinks for DynamicImage {
    fn long_side(&self) -> u32 {
        self.width().max(self.height())
    }

    fn shrunk(&self) -> Self {
        use prunr_core::formats::{fit_long_side, resize_to_rgba, ResizeFilter};
        let (qw, qh) = fit_long_side(self.width(), self.height(), QUICK_FRAME_SIDE);
        DynamicImage::ImageRgba8(resize_to_rgba(self, qw, qh, ResizeFilter::Bilinear))
    }

    fn kept(small: &Small) -> Option<Arc<Self>> {
        match small { Small::Photo(p) => Some(Arc::clone(p)), Small::Image(_) => None }
    }

    fn keep(small: Arc<Self>) -> Small {
        Small::Photo(small)
    }
}

impl Shrinks for RgbaImage {
    fn long_side(&self) -> u32 {
        self.width().max(self.height())
    }

    fn shrunk(&self) -> Self {
        use prunr_core::formats::{fit_long_side, resize_rgba, ResizeFilter};
        let (qw, qh) = fit_long_side(self.width(), self.height(), QUICK_FRAME_SIDE);
        resize_rgba(self, qw, qh, ResizeFilter::Bilinear)
    }

    fn kept(small: &Small) -> Option<Arc<Self>> {
        match small { Small::Image(i) => Some(Arc::clone(i)), Small::Photo(_) => None }
    }

    fn keep(small: Arc<Self>) -> Small {
        Small::Image(small)
    }
}

impl QuickPhoto {
    /// `image` no larger than a quick frame: the kept one when it was
    /// shrunk from `image`, else a fresh one, returned to keep.
    fn shrink<T: Shrinks>(image: &Arc<T>, kept: Option<&QuickPhoto>) -> (Arc<T>, Option<QuickPhoto>) {
        if image.long_side() <= QUICK_FRAME_SIDE {
            return (Arc::clone(image), None);
        }
        // The weak reference holds the allocation, so an equal address is
        // the same image, never a new one at a freed address.
        let hit = kept
            .filter(|k| std::ptr::eq(k.from.as_ptr().cast::<()>(), Arc::as_ptr(image).cast::<()>()))
            .and_then(|k| T::kept(&k.small));
        if let Some(small) = hit {
            return (small, None);
        }
        let small = Arc::new(image.shrunk());
        let from: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(image) as _;
        (Arc::clone(&small), Some(QuickPhoto { from: Arc::downgrade(&from), small: T::keep(small) }))
    }
}

/// Fit a quick frame's inputs to its smaller photo: the pixel-sized knobs
/// scale with it, and the planes kept from full-size renders don't fit.
fn shrink_to_quick_frame(inputs: &mut DispatchInputs, full_width: u32) {
    let scale = inputs.original.width() as f32 / full_width.max(1) as f32;
    inputs.settings = inputs.settings.at_scale(scale);
    inputs.cached_edge = EdgePlanes::default();
    inputs.cached_bold = EdgePlanes::default();
    inputs.cached_masked_base = None;
}

/// Swap `inputs.original` for the photo with the Depth chip's look on it:
/// the one made last time when it still matches, or a fresh one, which
/// is returned with its key for the item to keep.
fn apply_depth_look(inputs: &mut DispatchInputs) -> Option<(Arc<DynamicImage>, crate::gui::item::DepthPhotoKey)> {
    let depth = inputs.depth.as_ref()?;
    let look = depth.look.filter(prunr_core::depth::DepthLook::is_active)?;
    if let Some(photo) = &depth.photo {
        inputs.original = Arc::clone(photo);
        return None;
    }
    if inputs.quick {
        let looked = look.apply(&prunr_core::rgba_view(&inputs.original), &depth.map)?;
        inputs.original = Arc::new(DynamicImage::ImageRgba8(looked));
        return None;
    }
    let looked = look.apply(&prunr_core::rgba_view(&inputs.original), &depth.map)?;
    let photo = Arc::new(DynamicImage::ImageRgba8(looked));
    let key = crate::gui::item::DepthPhotoKey { base: Arc::clone(&inputs.original), depth: Arc::clone(&depth.map), look };
    inputs.original = Arc::clone(&photo);
    Some((photo, key))
}

/// `run_preview` for the kinds made from the photo (not an upscale).
fn run_photo_preview(inputs: DispatchInputs) -> RunOutput {
    // Pipeline shape follows the user's CURRENT `line_mode`, not tensor
    // presence. After a SubjectOutline → Off round-trip we (intentionally)
    // keep both tensors cached, so using tensor-presence as the selector
    // silently kept composing edges onto Off-mode previews.
    let is_subject_outline = inputs.settings.line_mode == LineMode::SubjectOutline
        && inputs.seg_tensor.is_some()
        && inputs.edge_tensor.is_some();

    // Resolve the masked base for SubjectOutline mode.
    // - Mask kind: always rebuild (the mask settings may have just changed).
    // - Edge kind: reuse cached base if Some (mask settings stable during drag).
    // When newly built, `built_masked_base` propagates back for caching.
    let (masked_base, built_masked_base) = if is_subject_outline {
        let fresh_base = || {
            // invariant: is_subject_outline → seg_tensor is Some.
            let seg = inputs.seg_tensor.as_ref().unwrap();
            build_masked_base(seg, &inputs.original, &inputs.settings, inputs.correction_plane())
                .map_or((None, None), |img| {
                    let arc = Arc::new(img.to_rgba8());
                    (Some(Arc::clone(&arc)), Some(arc))
                })
        };
        match inputs.kind {
            PreviewKind::Mask => fresh_base(),
            PreviewKind::Edge => inputs.cached_masked_base.clone()
                .map_or_else(fresh_base, |cached| (Some(cached), None)),
            PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2 | PreviewKind::Photo => unreachable!("run_preview finishes these kinds itself"),
        }
    } else {
        (None, None)
    };

    match inputs.kind {
        PreviewKind::Mask => {
            if !is_subject_outline {
                // Filter-only path: no seg tensor available (model=None, or
                // item not yet processed). Apply effect directly to the
                // raw source so the user sees the filter effect without
                // running inference.
                let Some(seg) = inputs.seg_tensor.as_ref() else {
                    let mut rgba = inputs.original.to_rgba8();
                    prunr_core::apply_subject_effect(&mut rgba, inputs.settings.effect);
                    return RunOutput { rgba: Some(Arc::new(rgba)), ..RunOutput::default() };
                };
                // Off mode with seg tensor: rebuild masked RGBA (no edge compose).
                let Some(masked) = build_masked_base(seg, &inputs.original, &inputs.settings, inputs.correction_plane())
                else { return RunOutput::default(); };
                return RunOutput { rgba: Some(Arc::new(masked.to_rgba8())), ..RunOutput::default() };
            }
            // SubjectOutline: compose subject + edges via the selected mode.
            // All modes are compose-time only — no re-inference.
            let Some(base_arc) = masked_base else { return RunOutput::default(); };
            // invariant: is_subject_outline → edge_tensor is Some.
            let edge = inputs.edge_tensor.as_ref().unwrap();
            let edge_settings = inputs.settings.edge_settings();
            let out_dims = (base_arc.width(), base_arc.height());
            let fade = line_fade(&inputs);
            let (mask, _) = resolve_edge_mask(&inputs.cached_edge, edge, out_dims, &edge_settings, fade);
            let (rgba, _) = dispatch_compose(
                &mask, &base_arc, &edge_settings,
                inputs.secondary_edge_tensor.as_ref(), &inputs.cached_bold,
                out_dims, fade,
            );
            RunOutput { rgba: Some(Arc::new(rgba)), built_masked_base, ..RunOutput::default() }
        }
        PreviewKind::Edge => {
            let Some(edge) = inputs.edge_tensor.as_ref() else { return RunOutput::default(); };
            let edge_settings = inputs.settings.edge_settings();

            // Base for compose: masked subject (SubjectOutline) or raw original (EdgesOnly).
            // We only need its dimensions for `resolve_edge_mask` — used to be
            // a full RGBA→DynamicImage clone (~64 MB at 4K, 10 Hz during drag).
            let out_dims = match masked_base.as_ref() {
                Some(arc) => (arc.width(), arc.height()),
                None => (inputs.original.width(), inputs.original.height()),
            };
            let fade = line_fade(&inputs);
            let (mask, built_edge) = resolve_edge_mask(&inputs.cached_edge, edge, out_dims, &edge_settings, fade);
            // SubjectOutline: the chosen composition over the subject.
            // EdgesOnly: the same compose over the photo, lines only.
            let (base, compose_settings) = match masked_base.as_deref() {
                Some(subject) => (std::borrow::Cow::Borrowed(subject), edge_settings),
                None => (
                    prunr_core::rgba_view(&inputs.original),
                    prunr_core::EdgeSettings { compose_mode: prunr_core::ComposeMode::LinesOnly, ..edge_settings },
                ),
            };
            let (rgba, built_bold) = dispatch_compose(
                &mask, &base, &compose_settings,
                inputs.secondary_edge_tensor.as_ref(), &inputs.cached_bold, out_dims, fade,
            );
            RunOutput { rgba: Some(Arc::new(rgba)), built_edge, built_bold, built_masked_base, ..RunOutput::default() }
        }
        PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2 | PreviewKind::Photo => unreachable!("run_preview finishes these kinds itself"),
    }
}

#[cfg(test)]
impl LivePreview {
    /// A drag on `item_id` whose last render was a quick frame, its latest
    /// tweak now.
    pub(crate) fn owe_for_test(&mut self, item_id: u64, kind: PreviewKind) {
        let latest_tweak_at = self.clock.now();
        self.owed.insert(item_id, Owed { kind, latest_tweak_at });
    }

    /// The kind of render queued for `item_id`.
    pub(crate) fn pending_kind_for_test(&self, item_id: u64) -> Option<PreviewKind> {
        self.pending.get(&item_id).map(|p| p.kind)
    }

    /// Whether the render queued for `item_id` is the settled one.
    pub(crate) fn pending_settled_for_test(&self, item_id: u64) -> Option<bool> {
        self.pending.get(&item_id).map(|p| p.settled)
    }
}

/// Pick between single-scale (`compose_edges_styled`) and dual-scale
/// (`compose_edges_dual_styled`) composition based on the selected
/// `LineStyle`. For dual-scale, `primary_mask` is the Fine layer and the
/// Bold layer comes from `cached_bold` or is built from `secondary_tensor`;
/// a freshly built Bold plane is returned for caching.
fn dispatch_compose(
    primary_mask: &GrayImage,
    base: &image::RgbaImage,
    edge_settings: &prunr_core::EdgeSettings,
    secondary_tensor: Option<&EdgeTensor>,
    cached_bold: &EdgePlanes,
    base_dims: (u32, u32),
    fade: Option<LineFade<'_>>,
) -> (image::RgbaImage, EdgePlanes) {
    use prunr_core::LineStyle;
    match (edge_settings.line_style, secondary_tensor) {
        (LineStyle::DualScale { fine_color, bold_color }, Some(secondary)) => {
            let (bold_mask, built) = resolve_edge_mask(cached_bold, secondary, base_dims, edge_settings, fade);
            let rgba = compose_edges_dual_styled(
                primary_mask, &bold_mask, base,
                edge_settings.compose_mode,
                fine_color, bold_color,
            );
            (rgba, built)
        }
        _ => (
            compose_edges_styled(
                primary_mask, base,
                edge_settings.compose_mode,
                edge_settings.line_style,
                edge_settings.solid_line_color,
            ),
            EdgePlanes::default(),
        ),
    }
}

/// The dilated edge plane for compositing, from the cached planes when
/// they match: a cached dilated plane costs nothing, a cached base costs
/// the dilation only, and otherwise the tensor goes through sigmoid,
/// Lanczos resize and dilation (tens of ms at 4K). Returns the planes it
/// built for caching.
fn resolve_edge_mask(
    cached: &EdgePlanes,
    edge: &EdgeTensor,
    out_dims: (u32, u32),
    edge_settings: &prunr_core::EdgeSettings,
    fade: Option<LineFade<'_>>,
) -> (Arc<GrayImage>, EdgePlanes) {
    if let Some(d) = &cached.dilated {
        return (Arc::clone(d), EdgePlanes::default());
    }
    let built_base = cached.base.is_none().then(|| {
        let (out_w, out_h) = out_dims;
        Arc::new(tensor_to_edge_mask(&edge.data, edge.height, edge.width, out_w, out_h, edge_settings.line_strength))
    });
    // one of the two is Some: built_base is None only when cached.base is Some
    let base = built_base.as_ref().or(cached.base.as_ref()).unwrap();
    let dilated = match (thickened_edges(base, edge_settings.edge_thickness), fade) {
        (thick, Some(fade)) => {
            let mut plane = thick.unwrap_or_else(|| (**base).clone());
            fade.apply(&mut plane);
            Arc::new(plane)
        }
        (Some(thick), None) => Arc::new(thick),
        (None, None) => Arc::clone(base),
    };
    (Arc::clone(&dilated), EdgePlanes { base: built_base, dilated: Some(dilated) })
}

/// How much a render fades its lines, in percent: only where the depth
/// map is there to fade them by. The plane cache key and the render read
/// this one value, so a cached plane always matches what is drawn.
fn fade_pct(settings: &ItemSettings, has_depth: bool) -> u8 {
    if has_depth { settings.line_fade_pct.min(100) } else { 0 }
}

/// Fade with distance for a render of `inputs`, when it fades and the
/// map is there.
fn line_fade(inputs: &DispatchInputs) -> Option<LineFade<'_>> {
    let depth = inputs.depth.as_ref()?;
    let pct = fade_pct(&inputs.settings, true);
    (pct > 0).then(|| LineFade { depth: &depth.map, amount: f32::from(pct) / 100.0 })
}

#[cfg(test)]
impl PreviewResult {
    /// A Mask render of `item_id` that landed as the final one, on version
    /// 0, having built and changed nothing else.
    pub(crate) fn mask_for_test(item_id: u64, rgba: Arc<RgbaImage>) -> Self {
        Self {
            item_id, rgba, kind: PreviewKind::Mask, generation: 0, version: 0, start: None, cutter: None,
            new_edge: None, new_bold: None, new_masked_base: None, new_reference: None, new_faced: None, new_depth_photo: None, new_quick_photo: None, new_original: None, new_correction: None, unpacked: Unpacked::default(),
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None, applied_color_strength: None, is_final: true, quick: false,
        }
    }
}

#[cfg(test)]
mod tests {


    use super::*;

    #[test]
    fn edge_planes_keep_bold_and_the_newest_other_scale() {
        use prunr_core::EdgeScale::{Bold, Fine, Fused};
        let key = |scale, thickness| EdgePlaneKey { strength_bits: 1, scale, thickness, fade_pct: 0 };
        let plane = || Some(Arc::new(GrayImage::new(1, 1)));
        let built = || EdgePlanes { base: plane(), dilated: plane() };
        let mut cache = EdgePlaneCache::default();
        cache.store(built(), key(Fine, 2));
        cache.store(built(), key(Bold, 2));
        assert!(cache.lookup(key(Fine, 2)).dilated.is_some(), "a dual-scale style keeps both layers");
        assert!(cache.lookup(key(Bold, 2)).dilated.is_some());
        let thicker = cache.lookup(key(Fine, 3));
        assert!(thicker.base.is_some() && thicker.dilated.is_none(), "a thickness change keeps only the base");
        cache.store(EdgePlanes::default(), key(Fused, 2));
        assert!(cache.lookup(key(Fine, 2)).base.is_some(), "a dispatch that built nothing evicts nothing");
        cache.store(built(), key(Fused, 2));
        assert!(cache.lookup(key(Fine, 2)).base.is_none(), "a new scale replaces the old one");
        assert!(cache.lookup(key(Bold, 2)).base.is_some() && cache.lookup(key(Fused, 2)).base.is_some(), "Bold stays");
    }
    use std::sync::Mutex;

    /// Test clock backed by a `Mutex<Instant>` so the test can advance
    /// time deterministically without sleeping. The trait already
    /// requires `Send + Sync`, so the Mutex is the natural fit.
    struct MockClock(Mutex<Instant>);

    impl MockClock {
        fn new(start: Instant) -> Arc<Self> {
            Arc::new(Self(Mutex::new(start)))
        }
        fn advance(&self, by: Duration) {
            let mut g = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            *g += by;
        }
    }

    impl Clock for MockClock {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }

    #[test]
    fn debounce_is_reasonable() {
        // Balance: short enough to feel responsive, long enough to coalesce
        // mid-drag change events into one dispatch.
        assert!(DEBOUNCE >= Duration::from_millis(100));
        assert!(DEBOUNCE <= Duration::from_millis(300));
    }

    #[test]
    fn mark_tweak_inserts_pending() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(42, PreviewKind::Mask);
        assert!(lp.pending.contains_key(&42));
    }

    #[test]
    fn has_in_flight_tracks_dispatch_lifecycle() {
        let mut lp = LivePreview::default();
        assert!(!lp.has_in_flight(), "fresh LivePreview is idle");
        lp.in_flight.insert(
            1,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 1 },
        );
        assert!(lp.has_in_flight());
        lp.cancel_all();
        assert!(!lp.has_in_flight(), "cancel_all clears in_flight");
    }

    #[test]
    fn mark_tweak_does_not_cancel_in_flight() {
        // Mid-drag behaviour: a running dispatch must be allowed to complete
        // so the user sees a preview update; `drain_results`' generation
        // filter drops results that were superseded.
        let mut lp = LivePreview::default();
        let cancel = Arc::new(AtomicBool::new(false));
        lp.in_flight.insert(
            7,
            InFlight { cancel: cancel.clone(), generation: 1 },
        );
        lp.mark_tweak(7, PreviewKind::Edge);
        assert!(
            !cancel.load(Ordering::Acquire),
            "in-flight dispatch must NOT be cancelled by a subsequent tweak",
        );
    }

    #[test]
    fn mark_tweak_does_not_reset_timer_mid_drag() {
        // Continuous-drag scenario: multiple tweaks within DEBOUNCE of each
        // other must not keep re-arming the timer, or the dispatch would
        // never fire until the user stopped moving.
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        let first_arm = lp.pending.get(&1).expect("armed").armed_at;

        // Simulate a second tweak "soon after" (short of DEBOUNCE). The timer
        // should NOT move — otherwise the dispatch would be pushed out.
        std::thread::sleep(Duration::from_millis(5));
        lp.mark_tweak(1, PreviewKind::Mask);
        let second_arm = lp.pending.get(&1).expect("still armed").armed_at;

        assert_eq!(
            first_arm, second_arm,
            "mid-drag tweaks must leave the original arm time in place",
        );
    }

    /// Slow frames put `elapsed >= DEBOUNCE` on every tweak. The broken
    /// shape re-armed `armed_at` in that branch and `tick` could
    /// never see a ready dispatch — mid-drag preview silently disabled.
    /// The current shape pins the timer on first tweak and only clears
    /// it via `tick`'s removal-on-dispatch.
    ///
    /// Drives synthetic time through `MockClock` so the contract holds
    /// regardless of CI runner speed.
    #[test]
    fn slow_frame_drag_dispatches_within_two_frames() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());

        // Frame 0: first tweak arrives, timer pinned at clock = T0.
        lp.mark_tweak(1, PreviewKind::Mask);
        assert!(lp.pending.contains_key(&1));

        // Frame 1: 200ms later (over the 150ms DEBOUNCE) — every
        // subsequent tweak in the drag arrives after `elapsed >= DEBOUNCE`.
        // Pre-fix this re-armed the timer; post-fix it leaves it alone.
        clock.advance(Duration::from_millis(200));
        lp.mark_tweak(1, PreviewKind::Mask);

        // tick at the same simulated `now` — DEBOUNCE elapsed since T0,
        // so the dispatch must be ready. `wait` is None because nothing's
        // pending after dispatch.
        let wait = lp.tick(|_, _| Err(WAIT));
        assert!(
            wait.is_none(),
            "tick at frame 1 (200ms after T0) must clear the pending entry"
        );
    }

    /// A drag's in-between frames are told apart from the release (the
    /// render then is quick): a flush settles the render, and a new tweak
    /// after it is a drag again.
    #[test]
    fn a_release_settles_the_render_and_a_new_tweak_does_not() {
        let mut lp = LivePreview::default();
        let settled = |lp: &LivePreview| lp.pending.get(&1).map(|p| p.settled);
        lp.mark_tweak(1, PreviewKind::Mask);
        assert_eq!(settled(&lp), Some(false));
        lp.flush(1);
        assert_eq!(settled(&lp), Some(true));
        lp.mark_tweak(1, PreviewKind::Mask);
        assert_eq!(settled(&lp), Some(false));
    }

    /// A render of an image `long_side` px on its long side, which a
    /// worker makes in no time (no tensor: the filters on a blank photo).
    fn photo_inputs(long_side: u32) -> DispatchInputs {
        let photo = Arc::new(image::DynamicImage::new_rgba8(long_side, 1));
        DispatchInputs::new(PreviewKind::Mask, 0, photo, ItemSettings::default())
    }

    fn large() -> Result<DispatchInputs, crate::gui::capability::Verdict> {
        Ok(photo_inputs(QUICK_FRAME_SIDE + 1))
    }

    const WAIT: crate::gui::capability::Verdict = crate::gui::capability::Verdict::Wait(crate::gui::capability::Reason::Decoding);

    /// A render that can never be built leaves the queue, and an owed
    /// settle with it: the busy pill would otherwise stay up for good.
    #[test]
    fn a_render_that_never_can_be_built_is_dropped() {
        use crate::gui::capability::{Reason, Verdict};
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.owe_for_test(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE + Duration::from_millis(1));
        lp.tick(|_, _| Err(WAIT));
        assert!(lp.is_busy(), "a render that waits is kept");
        lp.tick(|_, _| Err(Verdict::Never(Reason::ProcessFirst)));
        assert!(!lp.is_busy(), "nothing left: not pending, not owed");
    }

    /// The last dispatch for item 1: its generation and whether it was a
    /// quick frame (one that leaves the full-size render owed).
    fn dispatched(lp: &LivePreview) -> Option<(u64, bool)> {
        lp.in_flight.get(&1).map(|f| (f.generation, lp.owed.contains_key(&1)))
    }

    /// A drag on item 1 of a large photo, its first quick frame dispatched
    /// at `DEBOUNCE` after the tweak; the clock stands there.
    fn dragging() -> (Arc<MockClock>, LivePreview) {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, true))), "a drag's in-between frame is quick");
        (clock, lp)
    }

    /// A render that can make nothing (lines with no edge tensor) still
    /// ends: the image is no longer worked on, so nothing waits on it.
    #[test]
    fn a_render_that_makes_nothing_still_ends() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Edge);
        clock.advance(DEBOUNCE);
        lp.tick(|_, _| Ok(DispatchInputs::new(PreviewKind::Edge, 0, Arc::new(image::DynamicImage::new_rgb8(8, 8)), ItemSettings::default())));
        assert!(lp.works_on(1), "dispatched");
        let deadline = Instant::now() + Duration::from_secs(10);
        while lp.works_on(1) && Instant::now() < deadline {
            assert!(lp.drain_results().is_empty(), "nothing to show");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!lp.works_on(1) && !lp.is_busy(), "the render ended");
    }

    /// Held still with the slider down: the full-size render goes out
    /// `SETTLE_AFTER` after the last tweak, with no release.
    #[test]
    fn holding_a_drag_still_settles_it_without_a_release() {
        let (clock, mut lp) = dragging();
        let quick = dispatched(&lp);
        clock.advance(SETTLE_AFTER - DEBOUNCE - Duration::from_millis(1));
        lp.tick(|_, _| large());
        assert_eq!(dispatched(&lp), quick, "not yet");
        clock.advance(Duration::from_millis(1));
        lp.tick(|_, _| large());
        let settled = dispatched(&lp).expect("dispatched");
        assert!(!settled.1 && settled.0 > quick.expect("dispatched").0, "the full-size render goes out");
        assert!(lp.is_busy(), "and is under way");
    }

    /// Tweaks that keep coming never settle the drag, however long it
    /// lasts: every frame of it is quick.
    #[test]
    fn a_drag_that_keeps_moving_never_settles_mid_drag() {
        let (clock, mut lp) = dragging();
        for _ in 0..40 {
            lp.mark_tweak(1, PreviewKind::Mask);
            clock.advance(SETTLE_AFTER / 4);
            lp.tick(|_, _| large());
            assert!(matches!(dispatched(&lp), Some((_, true))), "still quick");
        }
    }

    /// A release settles at once: with a tweak queued, and with none
    /// queued (the release frame left the value as it was).
    #[test]
    fn a_release_settles_the_drag_at_once() {
        let (_clock, mut lp) = dragging();
        lp.flush(1);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, false))), "nothing queued: the owed render goes out");

        let (_clock, mut lp) = dragging();
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.flush(1);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, false))), "a queued tweak goes out settled");
    }

    /// The settle goes out once: holding still longer queues nothing more.
    #[test]
    fn a_drag_settles_once() {
        let (clock, mut lp) = dragging();
        clock.advance(SETTLE_AFTER);
        lp.tick(|_, _| large());
        let settled = dispatched(&lp);
        assert!(matches!(settled, Some((_, false))));
        for _ in 0..5 {
            clock.advance(SETTLE_AFTER);
            assert_eq!(lp.tick(|_, _| large()), None, "nothing to wait for");
            assert_eq!(dispatched(&lp), settled, "no second render");
        }
    }

    /// A tweak after the settle starts a new drag: quick frames again,
    /// then its own settle.
    #[test]
    fn a_tweak_after_a_settle_starts_a_new_quick_cycle() {
        let (clock, mut lp) = dragging();
        clock.advance(SETTLE_AFTER);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, false))));
        lp.mark_tweak(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, true))), "quick again");
        clock.advance(SETTLE_AFTER);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, false))), "and settles again");
    }

    /// A held-still pointer sends no input: the wait `tick` returns wakes
    /// the frame loop for the settle.
    #[test]
    fn the_wait_covers_the_settle_deadline() {
        let (clock, mut lp) = dragging();
        assert_eq!(lp.tick(|_, _| large()), Some(SETTLE_AFTER - DEBOUNCE), "from the last tweak");
        clock.advance(Duration::from_millis(100));
        assert_eq!(lp.tick(|_, _| large()), Some(SETTLE_AFTER - DEBOUNCE - Duration::from_millis(100)));
        lp.mark_tweak(1, PreviewKind::Mask);
        assert_eq!(lp.tick(|_, _| large()), Some(DEBOUNCE), "a queued tweak waits for its throttle first");
    }

    /// An image no larger than a quick frame renders full size mid-drag,
    /// as before quick frames: nothing is owed and nothing settles later.
    #[test]
    fn a_small_photo_has_no_quick_frames() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE);
        lp.tick(|_, _| Ok(photo_inputs(QUICK_FRAME_SIDE)));
        let full = dispatched(&lp);
        assert!(matches!(full, Some((_, false))), "full size");
        assert!(!lp.owed.contains_key(&1));
        clock.advance(SETTLE_AFTER * 2);
        assert_eq!(lp.tick(|_, _| Ok(photo_inputs(QUICK_FRAME_SIDE))), None);
        assert_eq!(dispatched(&lp), full, "no settle render");
    }

    /// Hurrying a drag's queued frame sends it out at the next tick, still
    /// as a drag frame.
    #[test]
    fn hurrying_a_queued_frame_does_not_settle_it() {
        let (_clock, mut lp) = dragging();
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.hurry(1);
        lp.tick(|_, _| large());
        assert!(matches!(dispatched(&lp), Some((_, true))));
        assert!(lp.works_on(1), "the drag still owes its settle");
        lp.cancel(1);
        assert!(!lp.works_on(1) && !lp.is_busy(), "a cancel drops the owed settle too");
    }

    /// A quick frame stays up from its texture's landing until a full-size
    /// render's texture lands: an older frame's late texture is dropped, a
    /// newer frame outlives an earlier render, and a drag that ends with
    /// no render coming takes its frame with it.
    #[test]
    fn a_quick_frame_stays_up_until_the_full_size_render_lands() {
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]), Default::default());
        let shown = |q: &QuickFrames| q.texture().map(|t| t.id());
        let mut q = QuickFrames::default();
        q.expect(5);
        q.expect(7);
        q.land(tex("5"), 5);
        assert_eq!(shown(&q), None, "an older frame's texture is dropped");
        let seven = tex("7");
        q.land(seven.clone(), 7);
        assert_eq!(shown(&q), Some(seven.id()));
        q.settle(true);
        assert_eq!(shown(&q), Some(seven.id()), "kept while the drag goes on");
        q.supersede();
        q.settle(false);
        assert_eq!(shown(&q), Some(seven.id()), "kept until the render's texture lands");
        q.expect(9);
        q.result_landed();
        assert_eq!(shown(&q), Some(seven.id()), "a newer frame outlives the earlier render");
        q.supersede();
        q.result_landed();
        assert_eq!(shown(&q), None, "the render's texture took over");
        q.land(tex("9"), 9);
        assert_eq!(shown(&q), None, "a texture landing after the render is dropped");
        q.expect(11);
        q.land(tex("11"), 11);
        q.settle(false);
        assert_eq!(shown(&q), None, "a drag ended with no render coming");
    }

    /// `tick` returning `None` (no remaining wait) is the readiness
    /// signal even when snapshot can't assemble inputs — the contract
    /// that matters here is that the entry is RECOGNISED as ready, not
    /// that the rayon dispatch actually fires (which depends on
    /// whether the snapshot has the source RGBA decoded etc).
    #[test]
    fn tick_recognises_ready_entry_after_synthetic_debounce() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Mask);

        // Before DEBOUNCE — wait is Some(>0).
        let wait = lp.tick(|_, _| Err(WAIT));
        assert!(wait.is_some(), "before DEBOUNCE tick should report a wait");

        // After DEBOUNCE — wait is None (entry is ready).
        clock.advance(DEBOUNCE + Duration::from_millis(1));
        let wait = lp.tick(|_, _| Err(WAIT));
        assert!(
            wait.is_none(),
            "after DEBOUNCE tick should report no remaining wait (entry is ready)"
        );
    }

    #[test]
    fn tick_before_debounce_does_not_dispatch() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        let wait = lp.tick(|_, _| Err(WAIT));
        assert!(wait.is_some(), "should still be pending");
        assert!(lp.in_flight.is_empty(), "no dispatch before debounce expires");
    }

    #[test]
    fn cancel_all_clears_state() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.in_flight.insert(
            2,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 1 },
        );
        lp.cancel_all();
        assert!(lp.pending.is_empty());
        assert!(lp.in_flight.is_empty());
    }

    #[test]
    fn stale_generation_result_is_dropped() {
        // In-flight carries generation 5 (latest dispatch). A stale worker
        // finishes with generation 1 — drain_results must drop it silently.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            42,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 5 },
        );
        lp.result_tx.send(PreviewResult { generation: 1, is_final: false, ..PreviewResult::mask_for_test(42, Arc::new(RgbaImage::new(1, 1))) }.into()).unwrap();

        let drained = lp.drain_results();
        assert!(drained.is_empty(), "stale generation must not be returned");
        assert!(
            lp.in_flight.contains_key(&42),
            "in-flight entry is preserved — the real dispatch hasn't completed",
        );
    }

    #[test]
    fn matching_generation_result_is_accepted_and_clears_in_flight() {
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            99,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 7 },
        );
        lp.result_tx.send(PreviewResult { generation: 7, is_final: false, ..PreviewResult::mask_for_test(99, Arc::new(RgbaImage::new(2, 2))) }.into()).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].item_id, 99);
        assert!(
            !lp.in_flight.contains_key(&99),
            "once a fresh result is accepted, the in-flight slot is cleared",
        );
    }

    #[test]
    fn result_for_unknown_item_is_dropped() {
        // No in-flight entry at all (user cancel_all'd or never dispatched).
        // A late result arriving for an unknown id must be dropped cleanly.
        let mut lp = LivePreview::default();
        lp.result_tx.send(PreviewResult { generation: 1, is_final: false, ..PreviewResult::mask_for_test(777, Arc::new(RgbaImage::new(1, 1))) }.into()).unwrap();
        let drained = lp.drain_results();
        assert!(drained.is_empty());
    }

    #[test]
    fn drain_sets_is_final_true_when_no_more_pending() {
        // User released the knob — no new pending entry for this item.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            11,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 3 },
        );
        lp.result_tx.send(PreviewResult { generation: 3, is_final: false, ..PreviewResult::mask_for_test(11, Arc::new(RgbaImage::new(1, 1))) }.into()).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].is_final, "empty pending at drain time → result is final");
    }

    #[test]
    fn drain_sets_is_final_false_while_user_still_tweaking() {
        // User is still dragging — a new tweak has landed in pending since
        // the dispatch was started.
        let mut lp = LivePreview::default();
        lp.in_flight.insert(
            22,
            InFlight { cancel: Arc::new(AtomicBool::new(false)), generation: 2 },
        );
        lp.pending.insert(
            22,
            Pending { armed_at: Instant::now(), latest_tweak_at: Instant::now(), kind: PreviewKind::Mask, settled: false },
        );
        lp.result_tx.send(PreviewResult { generation: 2, is_final: false, ..PreviewResult::mask_for_test(22, Arc::new(RgbaImage::new(1, 1))) }.into()).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].is_final, "pending for this item means drag is still active");
    }

    #[test]
    fn live_preview_mark_tweak_upscale_tier2_inserts_pending() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(55, PreviewKind::UpscaleTier2);
        let pending = lp.pending.get(&55).expect("pending entry must exist after mark_tweak");
        assert_eq!(pending.kind, PreviewKind::UpscaleTier2);
    }

    /// Refine's after-the-model knobs compare against the image the model
    /// read, never the original photo (an erase or a Denoise before the
    /// upscale would come back). With nothing on, the raw output shows
    /// uncopied; a kept reference is reused.
    #[test]
    fn finishing_an_upscale_compares_against_what_was_upscaled() {
        let raw = Arc::new(RgbaImage::from_pixel(8, 8, image::Rgba([200, 30, 30, 255])));
        let input = RgbaImage::from_pixel(2, 2, image::Rgba([20, 20, 220, 255]));
        let knobs = |ai_blend, color_match| prunr_core::upscale::Refine { sharpen: 0.0, sharpen_sigma: 2.0, ai_blend, color_match };

        let (shown, reference) = finish_upscale(&raw, &input, None, &knobs(1.0, false), None);
        assert!(Arc::ptr_eq(&shown, &raw), "nothing on: the raw output itself");
        assert!(reference.is_none(), "and no reference built");

        let (shown, reference) = finish_upscale(&raw, &input, None, &knobs(0.0, false), None);
        assert_eq!(shown.get_pixel(4, 4).0, [20, 20, 220, 255], "all blend: the enlarged input");
        let reference = reference.expect("built for AI blend");
        assert_eq!(reference.dimensions(), raw.dimensions());

        let (_, reused) = finish_upscale(&raw, &input, Some(Arc::clone(&reference)), &knobs(0.5, true), None);
        assert!(Arc::ptr_eq(&reused.expect("kept"), &reference), "a kept reference is reused");
    }

    #[test]
    fn live_preview_tier2_run_uses_the_kept_upscale_not_inference() {
        // A kept 2×2 upscale output: all channels = 100 (mid-grey).
        let mut raw_buf = RgbaImage::new(2, 2);
        for p in raw_buf.pixels_mut() {
            *p = image::Rgba([100u8, 100, 100, 255]);
        }
        let raw_arc = Arc::new(raw_buf);

        let knobs = prunr_core::upscale::Refine {
            sharpen: 0.0,
            sharpen_sigma: 2.0,
            ai_blend: 1.0,
            color_match: false,
        };

        // Calling run_preview with UpscaleTier2 + valid buffers must produce
        // an RGBA output. It must NOT attempt inference (no ORT session in this
        // process — the result is proof the code path avoids it).
        use crate::gui::item_settings::ItemSettings;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let output = run_preview(
            DispatchInputs {
                upscale: Some(crate::gui::item::UpscaleCache::new(raw_arc.clone(), raw_arc.clone(), None)),
                upscale_tier2_knobs: Some(knobs),
                ..DispatchInputs::new(
                    PreviewKind::UpscaleTier2, 0,
                    Arc::new(image::DynamicImage::ImageRgba8((*raw_arc).clone())), ItemSettings::default(),
                )
            },
            &cancel,
        );
        assert!(output.rgba.is_some(), "UpscaleTier2 dispatch must produce an RGBA result");
        assert!(output.built_edge.dilated.is_none(), "UpscaleTier2 must not build an edge mask");
        assert!(output.built_masked_base.is_none(), "UpscaleTier2 must not build a masked base");
    }

    /// A render from packed tensors unpacks them on its thread and draws
    /// what it draws from unpacked ones, handing back what it unpacked.
    #[test]
    fn a_render_from_packed_tensors_draws_the_same() {
        use crate::gui::item_settings::ItemSettings;
        use crate::gui::worker::{CompressedEdgeTensors, RawEdgeTensors, RawSegTensor};
        let photo = Arc::new(image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([90, 120, 150, 255]))));
        let seg: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let settings = ItemSettings { line_mode: LineMode::SubjectOutline, ..ItemSettings::default() };
        let primary = settings.edge_settings().primary_scale();
        let tensors = prunr_core::EdgeScale::ALL.map(|scale| {
            let (h, w) = prunr_core::edge_scale_dims(scale, 8, 8);
            (0..h * w).map(|i| (i % 7) as f32 / 7.0).collect::<Vec<f32>>()
        });
        let edges = CompressedEdgeTensors::from_raw(RawEdgeTensors { tensors, height: 8, width: 8, spare: None }).unwrap();
        let ct = CompressedTensor::from_raw(RawSegTensor { data: seg.clone(), height: 8, width: 8, model: ModelKind::Silueta }).unwrap();
        for kind in [PreviewKind::Mask, PreviewKind::Edge] {
            let (eh, ew) = edges.dims(primary);
            let unpacked = DispatchInputs {
                seg_tensor: Some(Arc::new(SegTensor { data: seg.clone(), height: 8, width: 8, model: ModelKind::Silueta })),
                edge_tensor: Some(EdgeTensor { data: Arc::new(edges.decompress(primary).unwrap()), height: eh, width: ew }),
                ..DispatchInputs::new(kind, 0, Arc::clone(&photo), settings)
            };
            let packed = DispatchInputs {
                packed: Packed { seg: Some(ct.clone()), edges: Some(edges.clone()), primary: Some(primary), bold: false },
                ..DispatchInputs::new(kind, 0, Arc::clone(&photo), settings)
            };
            assert_eq!(packed.seg_model(), Some(ModelKind::Silueta));
            let want = run_preview(unpacked, &AtomicBool::new(false)).rgba.expect("rendered");
            let out = run_preview(packed, &AtomicBool::new(false));
            assert_eq!(*out.rgba.expect("rendered"), *want, "{kind:?}");
            assert!(out.unpacked.seg.is_some_and(|(from, _)| from.same(&ct)));
            assert!(out.unpacked.edge.is_some_and(|(from, scale, _)| scale == primary && from.same_scale(&edges, primary)));
        }
    }

    /// A quick frame of every kind is made at `QUICK_FRAME_SIDE`, whatever
    /// full-size planes earlier renders kept; a settled render stays full
    /// size.
    #[test]
    fn a_quick_frame_is_made_small_and_a_settled_render_full_size() {
        use crate::gui::item_settings::ItemSettings;
        let (w, h) = (QUICK_FRAME_SIDE * 2, 600);
        let photo = Arc::new(image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(w, h, image::Rgba([90, 120, 150, 255]))));
        let full_plane = Arc::new(GrayImage::new(w, h));
        let settings = ItemSettings { edge_shift: 8.0, feather: 4.0, line_mode: LineMode::EdgesOnly, ..ItemSettings::default() };
        for kind in [PreviewKind::Mask, PreviewKind::Edge] {
            let inputs = |quick: bool| DispatchInputs {
                quick,
                seg_tensor: Some(Arc::new(SegTensor { data: (0..64).map(|i| i as f32 / 64.0).collect(), height: 8, width: 8, model: ModelKind::Silueta })),
                edge_tensor: Some(EdgeTensor { data: Arc::new(vec![0.5; 64]), height: 8, width: 8 }),
                cached_edge: EdgePlanes { base: Some(Arc::clone(&full_plane)), dilated: Some(Arc::clone(&full_plane)) },
                cached_masked_base: Some(Arc::new(RgbaImage::new(w, h))),
                ..DispatchInputs::new(kind, 0, Arc::clone(&photo), settings)
            };
            assert!(inputs(false).can_be_quick(), "{kind:?}");
            let quick = run_preview(inputs(true), &AtomicBool::new(false)).rgba.expect("rendered");
            assert_eq!(quick.dimensions(), (QUICK_FRAME_SIDE, 300), "{kind:?}");
            let settled = run_preview(inputs(false), &AtomicBool::new(false)).rgba.expect("rendered");
            assert_eq!(settled.dimensions(), (w, h), "{kind:?}");
        }
        let big = Arc::new(RgbaImage::from_pixel(w, h, image::Rgba([90, 120, 150, 255])));
        let knobs = prunr_core::upscale::Refine { sharpen: 0.5, sharpen_sigma: 2.0, ai_blend: 0.5, color_match: true };
        let upscale = |quick: bool| DispatchInputs {
            quick,
            upscale: Some(crate::gui::item::UpscaleCache::new(Arc::clone(&big), Arc::new(RgbaImage::new(w / 4, h / 4)), None)),
            upscale_tier2_knobs: Some(knobs),
            ..DispatchInputs::new(PreviewKind::UpscaleTier2, 0, Arc::clone(&photo), settings)
        };
        let colorize = |quick: bool| DispatchInputs {
            quick,
            colorize: Some(crate::gui::item::ColorizeCache::new(
                Arc::new(vec![0.0; 2 * (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize]),
                Arc::clone(&big),
            )),
            ..DispatchInputs::new(PreviewKind::ColorizeTier2, 0, Arc::clone(&photo), settings)
        };
        for (kind, inputs) in [("upscale", &upscale as &dyn Fn(bool) -> DispatchInputs), ("colorize", &colorize)] {
            assert!(inputs(false).can_be_quick(), "{kind}");
            let quick = run_preview(inputs(true), &AtomicBool::new(false)).rgba.expect("rendered");
            assert_eq!(quick.dimensions(), (QUICK_FRAME_SIDE, 300), "{kind}");
            let settled = run_preview(inputs(false), &AtomicBool::new(false)).rgba.expect("rendered");
            assert_eq!(settled.dimensions(), (w, h), "{kind}");
        }
        let faced = DispatchInputs {
            faces: Some(FacesInput { key: FacedKey { faces: Arc::new(Vec::new()), blend: Default::default() }, faced: None }),
            ..upscale(false)
        };
        assert!(!faced.can_be_quick(), "an upscale with restored faces re-finishes at full size");
    }

    /// A drag's quick frames shrink the image once: the next frame reuses
    /// it, another image does not, and the kept pair holds no full image.
    #[test]
    fn quick_frames_shrink_the_photo_once() {
        let photo = Arc::new(image::DynamicImage::new_rgb8(QUICK_FRAME_SIDE * 2, 10));
        let (small, kept) = QuickPhoto::shrink(&photo, None);
        assert_eq!((small.width(), small.height()), (QUICK_FRAME_SIDE, 5));
        let kept = kept.expect("a fresh shrink is kept");
        let (again, built) = QuickPhoto::shrink(&photo, Some(&kept));
        assert!(Arc::ptr_eq(&again, &small) && built.is_none(), "the next frame reuses it");
        let other = Arc::new(image::DynamicImage::new_rgb8(QUICK_FRAME_SIDE * 2, 10));
        let (fresh, built) = QuickPhoto::shrink(&other, Some(&kept));
        assert!(!Arc::ptr_eq(&fresh, &small) && built.is_some(), "another photo is shrunk anew");
        assert_eq!(Arc::strong_count(&photo), 1, "the kept pair holds the photo weakly");
        let tiny = Arc::new(image::DynamicImage::new_rgb8(8, 8));
        let (same, built) = QuickPhoto::shrink(&tiny, None);
        assert!(Arc::ptr_eq(&same, &tiny) && built.is_none(), "a small photo is its own quick frame");

        // An upscale's output or a colorize input the same way.
        let raw = Arc::new(RgbaImage::new(QUICK_FRAME_SIDE * 2, 10));
        let (small, kept) = QuickPhoto::shrink(&raw, None);
        let kept = kept.expect("kept");
        let (again, built) = QuickPhoto::shrink(&raw, Some(&kept));
        assert!(Arc::ptr_eq(&again, &small) && built.is_none(), "reused across frames");
        assert_eq!(Arc::strong_count(&raw), 1, "held weakly");
    }

    /// What a kept upscale or colorize run costs per drag frame at 24 MP.
    #[test]
    #[ignore]
    fn kept_run_frame_timing() {
        let photo = |w, h| image::RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8, 255]));
        let raw = Arc::new(photo(6000, 4000));
        let input = photo(1500, 1000);
        for knobs in [
            prunr_core::upscale::Refine { sharpen: 0.5, sharpen_sigma: 2.0, ai_blend: 1.0, color_match: false },
            prunr_core::upscale::Refine { sharpen: 0.5, sharpen_sigma: 2.0, ai_blend: 0.6, color_match: true },
        ] {
            let mut reference = None;
            for _ in 0..3 {
                let t = Instant::now();
                let (_, r) = prunr_core::upscale::refine(&raw, &input, reference.take(), &knobs);
                println!("upscale refine {knobs:?}: {:?}", t.elapsed());
                reference = r;
            }
        }
        let plane = vec![0.1f32; 2 * (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize];
        for _ in 0..3 {
            let t = Instant::now();
            let _ = prunr_core::colorize::colorized(&raw, &plane, 0.8);
            println!("colorize 24 MP: {:?}", t.elapsed());
        }
        use crate::gui::item_settings::ItemSettings;
        let input = Arc::new(input);
        let colorize = Arc::new(plane);
        let (mut kept_upscale, mut kept_colorize) = (None, None);
        for _ in 0..3 {
            for (label, inputs) in [
                ("upscale quick", DispatchInputs {
                    quick: true,
                    quick_photo: kept_upscale.clone(),
                    upscale: Some(crate::gui::item::UpscaleCache::new(Arc::clone(&raw), Arc::clone(&input), None)),
                    upscale_tier2_knobs: Some(prunr_core::upscale::Refine { sharpen: 0.5, sharpen_sigma: 2.0, ai_blend: 0.6, color_match: true }),
                    ..DispatchInputs::new(PreviewKind::UpscaleTier2, 0, Arc::new(image::DynamicImage::new_rgb8(1, 1)), ItemSettings::default())
                }),
                ("colorize quick", DispatchInputs {
                    quick: true,
                    quick_photo: kept_colorize.clone(),
                    colorize: Some(crate::gui::item::ColorizeCache::new(Arc::clone(&colorize), Arc::clone(&raw))),
                    ..DispatchInputs::new(PreviewKind::ColorizeTier2, 0, Arc::new(image::DynamicImage::new_rgb8(1, 1)), ItemSettings::default())
                }),
            ] {
                let t = Instant::now();
                let out = run_preview(inputs, &AtomicBool::new(false));
                println!("{label}: {:?}", t.elapsed());
                let kept = if label.starts_with("upscale") { &mut kept_upscale } else { &mut kept_colorize };
                *kept = out.built_quick_photo.or(kept.take());
            }
        }
    }

    /// What a 24 MP cut-out costs per drag frame, quick and settled.
    /// `cargo test --profile profiling -p prunr-app --lib quick_frame_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn quick_frame_timing() {
        use crate::gui::item_settings::ItemSettings;
        let photo = Arc::new(image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(6000, 4000, |x, y| image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8]))));
        let seg = Arc::new(SegTensor { data: (0..1024 * 1024).map(|i| ((i % 1024) as f32 / 512.0) - 1.0).collect(), height: 1024, width: 1024, model: ModelKind::BiRefNetLite });
        let mut kept = None;
        for refine_edges in [false, true] {
            let settings = ItemSettings { gamma: 1.4, feather: 2.0, refine_edges, ..ItemSettings::default() };
            for quick in [true, false, true, false, true, false] {
                let inputs = DispatchInputs { quick, seg_tensor: Some(Arc::clone(&seg)), quick_photo: kept.clone(), ..DispatchInputs::new(PreviewKind::Mask, 0, Arc::clone(&photo), settings) };
                let t = Instant::now();
                let out = run_preview(inputs, &AtomicBool::new(false));
                let took = t.elapsed();
                kept = out.built_quick_photo.or(kept);
                println!("refine {refine_edges} quick {quick}: {took:?} -> {:?}", out.rgba.expect("rendered").dimensions());
            }
        }
    }

    /// No model's photo preview puts the Depth chip's look on the photo,
    /// hands the looked photo back for the item, and reuses it next time.
    #[test]
    fn a_photo_preview_wears_the_depth_look_and_reuses_it() {
        use crate::gui::item_settings::ItemSettings;
        use prunr_core::depth::{DepthLook, DepthMap};
        let photo = Arc::new(image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 0, 255]))));
        let map = Arc::new(DepthMap::from_raw(1, 8, (0..8).map(|y| y as f32).collect()));
        let look = DepthLook { fog_pct: 100, ..DepthLook::default() };
        let settings = ItemSettings { depth: look, ..ItemSettings::default() };
        let inputs = |cached: Option<Arc<image::DynamicImage>>| DispatchInputs {
            depth: Some(DepthInput { map: Arc::clone(&map), look: Some(look), photo: cached }),
            ..DispatchInputs::new(PreviewKind::Mask, 0, Arc::clone(&photo), settings)
        };
        let cancel = AtomicBool::new(false);
        let first = run_preview(inputs(None), &cancel);
        let rgba = first.rgba.expect("rendered");
        assert!(rgba.get_pixel(4, 0).0[0] > 150, "the far top is fogged: {:?}", rgba.get_pixel(4, 0));
        assert!(rgba.get_pixel(4, 7).0[0] < 20, "the near bottom is not");
        let (looked, key) = first.built_depth_photo.expect("the looked photo comes back");
        assert!(Arc::ptr_eq(&key.base, &photo) && key.look == look);
        let again = run_preview(inputs(Some(Arc::clone(&looked))), &cancel);
        assert!(again.built_depth_photo.is_none(), "a kept looked photo is reused, not built again");
        assert_eq!(again.rgba.expect("rendered").as_raw(), rgba.as_raw());
        let plain = DispatchInputs { depth: Some(DepthInput { map, look: None, photo: None }), ..DispatchInputs::new(PreviewKind::Mask, 0, Arc::clone(&photo), settings) };
        assert_eq!(run_preview(plain, &cancel).rgba.expect("rendered").get_pixel(4, 0).0[0], 0, "no look outside No model");
    }

    /// Whole-photo lines fade with distance in the preview, and the plane
    /// key holds the fade so a cached unfaded plane is not reused.
    #[test]
    fn whole_photo_lines_fade_with_distance_in_the_preview() {
        use crate::gui::item_settings::ItemSettings;
        use prunr_core::LineMode;
        let (th, tw) = (8u32, 8u32);
        let photo = Arc::new(image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([90, 90, 90, 255]))));
        let edge = EdgeTensor { data: Arc::new(vec![1.0; (th * tw) as usize]), height: th, width: tw };
        let map = Arc::new(prunr_core::depth::DepthMap::from_raw(1, 8, (0..8).map(|y| y as f32).collect()));
        let settings = ItemSettings { line_mode: LineMode::EdgesOnly, line_fade_pct: 100, ..ItemSettings::default() };
        let render = |depth: Option<DepthInput>| {
            let inputs = DispatchInputs {
                edge_tensor: Some(EdgeTensor { data: Arc::clone(&edge.data), height: th, width: tw }),
                depth,
                ..DispatchInputs::new(PreviewKind::Edge, 0, Arc::clone(&photo), settings)
            };
            run_preview(inputs, &AtomicBool::new(false)).rgba.expect("rendered")
        };
        let faded = render(Some(DepthInput { map: Arc::clone(&map), look: None, photo: None }));
        let plain = render(None);
        assert!(faded.get_pixel(4, 0).0[3] < plain.get_pixel(4, 0).0[3] / 4, "far lines fade");
        let key = |has| EdgePlaneKey::new(&settings, prunr_core::EdgeScale::Fused, has);
        assert_ne!(key(true), key(false), "a faded plane is cached apart from an unfaded one");
        assert_eq!(key(true).base_key(), key(false).base_key(), "the undilated plane is shared");
    }

    /// A Strength preview re-renders the kept colorize run at the
    /// snapshot's strength, with no model run.
    #[test]
    fn colorize_tier2_renders_the_kept_run_at_the_new_strength() {
        use crate::gui::item_settings::ItemSettings;
        let n = (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize;
        let mut plane = vec![0.0; 2 * n];
        plane[n..].fill(40.0);
        let input = Arc::new(RgbaImage::from_pixel(4, 4, image::Rgba([128, 128, 128, 255])));
        let run = crate::gui::item::ColorizeCache::new(Arc::new(plane), Arc::clone(&input));
        let render = |color_strength_pct| {
            let inputs = DispatchInputs {
                colorize: Some(run.clone()),
                ..DispatchInputs::new(
                    PreviewKind::ColorizeTier2, 0,
                    Arc::new(image::DynamicImage::new_rgba8(0, 0)), ItemSettings { color_strength_pct, ..ItemSettings::default() },
                )
            };
            run_preview(inputs, &std::sync::atomic::AtomicBool::new(false)).rgba.expect("rendered")
        };
        let spread = |img: &RgbaImage| {
            let [r, g, b, _] = img.get_pixel(2, 2).0;
            r.max(g).max(b) - r.min(g).min(b)
        };
        assert!(spread(&render(0)) <= 1, "0 = grey");
        assert!(spread(&render(100)) > 20, "100% = the predicted colour");
        assert_eq!(*render(100), run.render(1.0));
    }

    /// A Sketch tweak previews exactly what Process makes: the same
    /// planes, scale and compose, for every line style, both modes and
    /// every Subject composition.
    #[test]
    fn the_sketch_preview_matches_process() {
        use crate::gui::item_settings::ItemSettings;
        use prunr_core::{ComposeMode, EdgeInferenceResult, EdgeScale, LineMode, LineStyle};
        let (th, tw) = (12u32, 16u32);
        let pattern = |scale: EdgeScale| -> Vec<f32> {
            let (sh, sw) = prunr_core::edge_scale_dims(scale, th, tw);
            (0..(sh * sw) as usize).map(|i| if (i * 7 + scale as usize * 3) % 5 < 2 { 0.9 } else { 0.0 }).collect()
        };
        let res = EdgeInferenceResult { tensors: EdgeScale::ALL.map(pattern), height: th, width: tw };
        let (w, h) = (40u32, 30u32);
        let photo = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 6) as u8, (y * 8) as u8, 90])));
        let subject = Arc::new(RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 6) as u8, (y * 8) as u8, 90, if x > 10 { 255 } else { 40 }])));
        let tensor = |scale: EdgeScale| {
            let (height, width) = res.dims(scale);
            EdgeTensor { data: Arc::new(res.tensors[scale as usize].clone()), height, width }
        };

        let cases = LineStyle::ALL.iter().flat_map(|style| {
            ComposeMode::ALL.iter().map(move |c| (LineMode::SubjectOutline, *style, *c))
                .chain(std::iter::once((LineMode::EdgesOnly, *style, ComposeMode::default())))
        });
        for (mode, style, compose) in cases {
            let settings = ItemSettings {
                line_mode: mode,
                line_style: style,
                compose_mode: compose,
                edge_scale: EdgeScale::Bold,
                edge_thickness: 2,
                solid_line_color: Some([200, 30, 30]),
                ..ItemSettings::default()
            };
            let edge = settings.edge_settings();
            let subject_mode = mode == LineMode::SubjectOutline;
            let preview = run_preview(
                DispatchInputs {
                    seg_tensor: subject_mode.then(|| Arc::new(SegTensor { data: vec![0.0; 4], height: 2, width: 2, model: ModelKind::Silueta })),
                    edge_tensor: Some(tensor(edge.primary_scale())),
                    secondary_edge_tensor: matches!(style, LineStyle::DualScale { .. }).then(|| tensor(EdgeScale::Bold)),
                    cached_masked_base: subject_mode.then(|| Arc::clone(&subject)),
                    ..DispatchInputs::new(PreviewKind::Edge, 0, Arc::new(photo.clone()), settings)
                },
                &std::sync::atomic::AtomicBool::new(false),
            )
            .rgba
            .unwrap_or_else(|| panic!("{mode:?} {style:?} {compose:?}: the preview renders"));
            let process = if subject_mode {
                prunr_core::compose_subject_outline(&res, &subject, &edge, None)
            } else {
                prunr_core::compose_whole_photo(&res, &photo, &edge, None)
            };
            assert!(preview.as_raw() == process.as_raw(), "{mode:?} {style:?} {compose:?}: preview differs from Process");
        }
    }
}

