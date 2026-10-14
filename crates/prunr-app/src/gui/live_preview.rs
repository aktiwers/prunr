//! In-process live-preview dispatcher for Tier 2 reruns.
//!
//! When the user tweaks a mask or edge knob in the adjustments toolbar,
//! `mark_tweak` registers a pending preview for the affected item. Every
//! frame, `tick` checks if 300ms have passed since the last tweak for that
//! item and, if so, spawns a background thread to run the Tier 2 postprocess
//! (mask) or edge finalize (edge) on the cached tensor.
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

use crate::gui::item_settings::ItemSettings;
use crate::gui::worker::CompressedTensor;

/// Throttle cadence for live-preview dispatch during a continuous drag.
/// 150ms stays above the ~90ms baseline and ~240ms refine_edges worker
/// cost so dispatches don't pile up for the generation filter to discard.
/// Release / commit paths call `flush` instead, so the final result on
/// knob-release lands immediately.
pub const DEBOUNCE: Duration = Duration::from_millis(150);

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
    /// Generation counter — used to discard results from stale dispatches
    /// that completed after a newer tweak cancelled them.
    pub generation: u64,
    /// The item's timeline version at dispatch: the result lands only on
    /// the state it was computed for.
    pub version: u64,
    /// Where the preview started (`None`: the source).
    pub(crate) start: Option<crate::gui::timeline::RunStart>,
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
    last_tweak_at: Instant,
    kind: PreviewKind,
    /// The edit settled (slider released): the render is the one kept.
    settled: bool,
}

/// State for an in-flight dispatch. Used to cancel when a new tweak arrives.
struct InFlight {
    cancel: Arc<AtomicBool>,
    generation: u64,
}

pub struct LivePreview {
    pending: HashMap<u64, Pending>,
    in_flight: HashMap<u64, InFlight>,
    generation_counter: u64,
    result_tx: mpsc::Sender<PreviewResult>,
    result_rx: mpsc::Receiver<PreviewResult>,
    clock: Arc<dyn Clock>,
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
            generation_counter: 0,
            result_tx: tx,
            result_rx: rx,
            clock,
        }
    }

    /// Pin `last_tweak_at` on the first tweak in a window and never re-arm —
    /// `tick`'s removal-on-dispatch starts the next window.
    pub fn mark_tweak(&mut self, item_id: u64, kind: PreviewKind) {
        let now = self.clock.now();
        let is_new = !self.pending.contains_key(&item_id);
        self.pending
            .entry(item_id)
            .and_modify(|p| {
                p.kind = kind;
                p.settled = false;
            })
            .or_insert_with(|| Pending { last_tweak_at: now, kind, settled: false });
        tracing::debug!(
            event = "upscale_tier2_mark_tweak",
            item_id,
            ?kind,
            is_new_entry = is_new,
            pending_count = self.pending.len(),
            in_flight_count = self.in_flight.len(),
            "mark_tweak registered"
        );
    }

    /// Flush: expire the pending tweak timer so the next `tick` dispatches
    /// immediately. Used when an edit settles (slider released, checkbox
    /// toggled, color picked) so the user doesn't wait the full debounce.
    ///
    /// Idempotent — no-op if there's no pending tweak for `item_id`.
    pub fn flush(&mut self, item_id: u64) {
        if let Some(p) = self.pending.get_mut(&item_id) {
            p.settled = true;
            // Anti-date past DEBOUNCE so next tick dispatches immediately.
            // checked_sub guards against early-boot monotonic clock underflow.
            let now = self.clock.now();
            p.last_tweak_at = now
                .checked_sub(DEBOUNCE + Duration::from_millis(10))
                .unwrap_or(now);
        }
    }

    /// Per-frame tick. Returns the time until the next tween we're waiting on
    /// (useful for `ctx.request_repaint_after`), or `None` if nothing pending.
    ///
    /// `snapshot` provides per-item inputs (tensors, settings, original image)
    /// for any dispatches we fire this frame, told whether the edit has
    /// settled (a release) or is a drag's in-between frame. Items that
    /// aren't in the snapshot (e.g. tensor cache evicted) are silently
    /// skipped.
    pub fn tick<F>(&mut self, mut snapshot: F) -> Option<Duration>
    where
        F: FnMut(u64, PreviewKind, bool) -> Option<DispatchInputs>,
    {
        let now = self.clock.now();
        let mut ready_ids: Vec<u64> = Vec::new();
        let mut wait_for: Option<Duration> = None;

        for (id, p) in &self.pending {
            let elapsed = now.saturating_duration_since(p.last_tweak_at);
            if elapsed >= DEBOUNCE {
                ready_ids.push(*id);
            } else {
                let remaining = DEBOUNCE - elapsed;
                wait_for = Some(wait_for.map_or(remaining, |w| w.min(remaining)));
            }
        }

        for id in ready_ids {
            // Peek the pending kind without removing — if the snapshot isn't
            // ready yet (e.g. `source_rgba` decode in flight after a view
            // switch), we keep the pending entry so the next tick retries.
            // Dropping it here silently swallowed the tweak and left the
            // user thinking live preview was dead.
            let (kind, settled) = match self.pending.get(&id) {
                Some(p) => (p.kind, p.settled),
                None => continue,
            };
            tracing::debug!(
                event = "upscale_tier2_tick_ready",
                item_id = id,
                ?kind,
                in_flight_count = self.in_flight.len(),
                "tick: item ready for dispatch"
            );
            let Some(inputs) = snapshot(id, kind, settled) else {
                tracing::debug!(
                    event = "upscale_tier2_tick_no_snapshot",
                    item_id = id,
                    ?kind,
                    "tick: snapshot returned None — dispatch skipped, will retry"
                );
                // Snapshot couldn't assemble inputs this frame. Possibilities:
                // (a) `source_rgba` is None (decode pending after view
                //     switch) — retry once the decode lands.
                // (b) required tensor cache is permanently absent (item
                //     never processed) — the retry is cheap and the pending
                //     entry stays inert until the user clicks Process or
                //     stops tweaking; no harm.
                continue;
            };
            self.pending.remove(&id);
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
                let seg_model = inputs.seg_tensor.as_ref().map(|s| s.model);
                // Capture Tier-2 knob values at dispatch time. When the result
                // lands, apply_completed_previews patches only the 4 Tier-2
                // fields (sharpen/ai_blend/color_match) on
                // applied_recipe.upscale so the next resolve_tier diff sees Skip
                // instead of UpscaleTier2, stopping the per-frame mark_tweak loop
                // after the user stops dragging.
                let applied_tier2_knobs = inputs.upscale_tier2_knobs;
                let applied_color_strength = matches!(inputs.kind, PreviewKind::ColorizeTier2).then_some(inputs.settings.color_strength());
                let applied_depth = inputs.settings.depth;
                let version = inputs.version;
                let start = inputs.start.clone();
                let faced_key = inputs.faces.as_ref().map(|f| f.key.clone());
                let started = std::time::Instant::now();
                let output = run_preview(inputs, &cancel);
                tracing::debug!(item_id = id, kind = ?dispatch_kind, ms = started.elapsed().as_millis() as u64, "live preview ran");
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                if let Some(rgba) = output.rgba {
                    let new_edge = is_edge.then_some((output.built_edge, primary_key));
                    let new_bold = is_edge.then_some((output.built_bold, bold_key));
                    let new_masked_base = output.built_masked_base
                        .zip(seg_model)
                        .map(|(base, model)| (base, mask_recipe, model));
                    // `is_final` is set by `drain_results`, where the UI
                    // thread can read `self.pending` atomically. The worker
                    // ships a placeholder and doesn't care.
                    let _ = tx.send(PreviewResult {
                        item_id: id, rgba, kind: dispatch_kind, generation, version, start,
                        new_edge, new_bold, new_masked_base,
                        new_reference: output.built_reference,
                        new_faced: output.built_faced.zip(faced_key),
                        new_depth_photo: output.built_depth_photo,
                        applied_mask: mask_recipe,
                        applied_depth,
                        applied_tier2_knobs,
                        applied_color_strength,
                        is_final: false,
                    });
                }
            });
        }

        wait_for
    }

    /// Drain any completed previews. Returned results are already filtered
    /// to the latest generation; each result's `is_final` is set to `true`
    /// iff no new tweak for that item is currently pending — i.e. the drag
    /// has settled and this is the last result of the session.
    pub fn drain_results(&mut self) -> Vec<PreviewResult> {
        let mut out = Vec::new();
        while let Ok(mut r) = self.result_rx.try_recv() {
            // Drop stale: if generation doesn't match the last dispatch for
            // this item, a newer tweak superseded it.
            let is_latest = self.in_flight
                .get(&r.item_id)
                .is_some_and(|f| f.generation == r.generation);
            if is_latest {
                self.in_flight.remove(&r.item_id);
                // If pending has a fresh entry for this item, the user is
                // still mid-drag (a tweak landed after the current dispatch
                // started). Empty pending = drag settled = final result.
                r.is_final = !self.pending.contains_key(&r.item_id);
                out.push(r);
            }
        }
        out
    }

    /// True while a preview waits for its debounce or runs.
    pub fn is_busy(&self) -> bool {
        !self.pending.is_empty() || !self.in_flight.is_empty()
    }

    /// True while a rayon dispatch hasn't drained yet.
    pub fn has_in_flight(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Drop the pending and running previews of one item: what they would
    /// show belongs to a state it has left.
    pub fn cancel(&mut self, item_id: u64) {
        if let Some(f) = self.in_flight.remove(&item_id) {
            f.cancel.store(true, Ordering::Release);
        }
        self.pending.remove(&item_id);
    }

    /// Cancel all in-flight + pending previews. Called on batch clear or shutdown.
    pub fn cancel_all(&mut self) {
        for f in self.in_flight.values() {
            f.cancel.store(true, Ordering::Release);
        }
        self.in_flight.clear();
        self.pending.clear();
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
    /// inline). `None` when nothing is selected.
    pub correction: Option<Arc<prunr_core::selection::MaskArtifact>>,
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
    /// A `kind` preview of `original` with `settings` and nothing cached,
    /// kept or found yet; callers set what their kind reads.
    pub(crate) fn new(kind: PreviewKind, version: u64, original: Arc<DynamicImage>, settings: ItemSettings) -> Self {
        Self {
            kind,
            version,
            quick: false,
            original,
            start: None,
            settings,
            seg_tensor: None,
            edge_tensor: None,
            secondary_edge_tensor: None,
            cached_edge: EdgePlanes::default(),
            cached_bold: EdgePlanes::default(),
            cached_masked_base: None,
            correction: None,
            upscale: None,
            upscale_tier2_knobs: None,
            colorize: None,
            faces: None,
            depth: None,
        }
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

/// Actually run the preview. Runs on a rayon worker thread. Returns the RGBA
/// plus, for Edge kind, the resized pre-dilation mask so the parent can cache
/// it (tied to the dispatch's line_strength).
///
/// Respects the user's full mask settings, including `refine_edges`. The
/// guided filter stage adds ~50-150ms on 4K when enabled, so drag cadence
/// slows accordingly — but the user opted into that cost when they toggled
/// Refine Edges on, and without running the filter the three refine knobs
/// (toggle, guided_radius, guided_epsilon) would be no-ops in live preview.
/// What `run_preview` produces. The built planes and `built_masked_base`
/// hold only what this dispatch constructed, so the parent caches them —
/// reused inputs pass through empty to avoid re-publishing.
struct RunOutput {
    rgba: Option<Arc<RgbaImage>>,
    built_edge: EdgePlanes,
    built_bold: EdgePlanes,
    built_masked_base: Option<Arc<image::RgbaImage>>,
    built_reference: Option<Arc<image::RgbaImage>>,
    built_faced: Option<Arc<DynamicImage>>,
    built_depth_photo: Option<(Arc<DynamicImage>, crate::gui::item::DepthPhotoKey)>,
}

impl RunOutput {
    fn empty() -> Self {
        Self { rgba: None, built_edge: EdgePlanes::default(), built_bold: EdgePlanes::default(), built_masked_base: None, built_reference: None, built_faced: None, built_depth_photo: None }
    }
}

fn run_preview(inputs: DispatchInputs, cancel: &AtomicBool) -> RunOutput {
    if cancel.load(Ordering::Acquire) { return RunOutput::empty(); }

    // UpscaleTier2: no seg/edge tensors involved — run postprocess directly
    // on the kept upscale run.
    if matches!(inputs.kind, PreviewKind::UpscaleTier2) {
        let Some(upscale) = inputs.upscale else {
            tracing::debug!(event = "upscale_tier2_run_abort", reason = "no upscale run kept");
            return RunOutput::empty();
        };
        let Some(knobs) = inputs.upscale_tier2_knobs else {
            tracing::debug!(event = "upscale_tier2_run_abort", reason = "upscale_tier2_knobs is None");
            return RunOutput::empty();
        };
        let had_reference = upscale.reference.is_some();
        let faces = inputs.faces.as_ref().map(|f| &f.key);
        let (shown, reference) = finish_upscale(&upscale.raw, &upscale.input, upscale.reference, &knobs, faces);
        let built_reference = reference.filter(|_| !had_reference);
        return RunOutput { rgba: Some(shown), built_reference, ..RunOutput::empty() };
    }
    if matches!(inputs.kind, PreviewKind::ColorizeTier2) {
        let Some(run) = inputs.colorize else {
            tracing::debug!(event = "colorize_tier2_run_abort", reason = "no colorize run kept");
            return RunOutput::empty();
        };
        return RunOutput { rgba: Some(Arc::new(run.render(inputs.settings.color_strength()))), ..RunOutput::empty() };
    }

    // Faces go into the photo before the mask, the effect and the lines.
    let mut inputs = inputs;
    let built_faced = match inputs.faces.take() {
        Some(FacesInput { faced: Some(faced), .. }) => {
            inputs.original = faced;
            None
        }
        Some(FacesInput { key, faced: None }) => {
            let mut photo = inputs.original.to_rgba8();
            prunr_core::face::compose_faces_into(&mut photo, &key.faces, key.blend);
            let faced = Arc::new(DynamicImage::ImageRgba8(photo));
            inputs.original = Arc::clone(&faced);
            Some(faced)
        }
        None => None,
    };
    if cancel.load(Ordering::Acquire) { return RunOutput::empty(); }
    // The Depth chip's look goes on the faced photo, before the lines.
    let built_depth_photo = apply_depth_look(&mut inputs);
    if cancel.load(Ordering::Acquire) { return RunOutput::empty(); }
    let output = run_photo_preview(inputs);
    RunOutput { built_faced, built_depth_photo, ..output }
}

/// The long side a drag's in-between depth look is made at.
const QUICK_LOOK_SIDE: u32 = 2048;

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
        let looked = look.apply_quick(&prunr_core::rgba_view(&inputs.original), &depth.map, QUICK_LOOK_SIDE)?;
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
        match inputs.kind {
            PreviewKind::Mask => {
                // invariant: is_subject_outline → seg_tensor is Some.
                let seg = inputs.seg_tensor.as_ref().unwrap();
                match build_masked_base(seg, &inputs.original, &inputs.settings, inputs.correction.as_deref()) {
                    Some(img) => {
                        let arc = Arc::new(img.to_rgba8());
                        (Some(arc.clone()), Some(arc))
                    }
                    None => (None, None),
                }
            }
            PreviewKind::Edge => {
                if let Some(cached) = inputs.cached_masked_base.clone() {
                    (Some(cached), None)
                } else {
                    // invariant: is_subject_outline → seg_tensor is Some.
                    let seg = inputs.seg_tensor.as_ref().unwrap();
                    match build_masked_base(seg, &inputs.original, &inputs.settings, inputs.correction.as_deref()) {
                        Some(img) => {
                            let arc = Arc::new(img.to_rgba8());
                            (Some(arc.clone()), Some(arc))
                        }
                        None => (None, None),
                    }
                }
            }
            PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2 => unreachable!("run_preview finishes the kept-run kinds itself"),
        }
    } else {
        (None, None)
    };

    match inputs.kind {
        PreviewKind::Mask => {
            if !is_subject_outline {
                // Filter-only path: no seg tensor available (model=None, or
                // item not yet processed). Apply fill_style directly to the
                // raw source so the user sees the filter effect without
                // running inference.
                let Some(seg) = inputs.seg_tensor.as_ref() else {
                    let mut rgba = inputs.original.to_rgba8();
                    prunr_core::apply_fill_style(&mut rgba, inputs.settings.fill_style);
                    return RunOutput { rgba: Some(Arc::new(rgba)), ..RunOutput::empty() };
                };
                // Off mode with seg tensor: rebuild masked RGBA (no edge compose).
                let Some(masked) = build_masked_base(seg, &inputs.original, &inputs.settings, inputs.correction.as_deref())
                else { return RunOutput::empty(); };
                return RunOutput { rgba: Some(Arc::new(masked.to_rgba8())), ..RunOutput::empty() };
            }
            // SubjectOutline: compose subject + edges via the selected mode.
            // All modes are compose-time only — no re-inference.
            let Some(base_arc) = masked_base else { return RunOutput::empty(); };
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
            RunOutput { rgba: Some(Arc::new(rgba)), built_masked_base, ..RunOutput::empty() }
        }
        PreviewKind::Edge => {
            let Some(edge) = inputs.edge_tensor.as_ref() else { return RunOutput::empty(); };
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
            RunOutput { rgba: Some(Arc::new(rgba)), built_edge, built_bold, built_masked_base, ..RunOutput::empty() }
        }
        PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2 => unreachable!("run_preview finishes the kept-run kinds itself"),
    }
}

impl LivePreview {
    /// Test-only spy: true iff `dispatch_brush_rerun` queued a preview for
    /// this item (i.e. `mark_tweak` was called with this item_id).
    #[cfg(test)]
    pub(crate) fn is_pending_for(&self, item_id: u64) -> bool {
        self.pending.contains_key(&item_id)
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
        let first_arm = lp.pending.get(&1).expect("armed").last_tweak_at;

        // Simulate a second tweak "soon after" (short of DEBOUNCE). The timer
        // should NOT move — otherwise the dispatch would be pushed out.
        std::thread::sleep(Duration::from_millis(5));
        lp.mark_tweak(1, PreviewKind::Mask);
        let second_arm = lp.pending.get(&1).expect("still armed").last_tweak_at;

        assert_eq!(
            first_arm, second_arm,
            "mid-drag tweaks must leave the original arm time in place",
        );
    }

    /// Slow frames put `elapsed >= DEBOUNCE` on every tweak. The broken
    /// shape re-armed `last_tweak_at` in that branch and `tick` could
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
        let wait = lp.tick(|_, _, _| None);
        assert!(
            wait.is_none(),
            "tick at frame 1 (200ms after T0) must clear the pending entry"
        );
    }

    /// `tick` returning `None` (no remaining wait) is the readiness
    /// signal even when snapshot can't assemble inputs — the contract
    /// that matters here is that the entry is RECOGNISED as ready, not
    /// that the rayon dispatch actually fires (which depends on
    /// whether the snapshot has the source RGBA decoded etc).
    /// A drag's in-between frames are told apart from the release: the
    /// snapshot sees `settled` only after a flush, and a new tweak after
    /// it is a drag again.
    #[test]
    fn a_release_settles_the_render_and_a_new_tweak_does_not() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        let mut seen = Vec::new();
        lp.mark_tweak(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE + Duration::from_millis(1));
        lp.tick(|_, _, settled| { seen.push(settled); None });
        lp.mark_tweak(1, PreviewKind::Mask);
        lp.flush(1);
        lp.tick(|_, _, settled| { seen.push(settled); None });
        lp.mark_tweak(1, PreviewKind::Mask);
        clock.advance(DEBOUNCE + Duration::from_millis(1));
        lp.tick(|_, _, settled| { seen.push(settled); None });
        assert_eq!(seen, vec![false, true, false]);
    }

    #[test]
    fn tick_recognises_ready_entry_after_synthetic_debounce() {
        let clock = MockClock::new(Instant::now());
        let mut lp = LivePreview::with_clock(clock.clone());
        lp.mark_tweak(1, PreviewKind::Mask);

        // Before DEBOUNCE — wait is Some(>0).
        let wait = lp.tick(|_, _, _| None);
        assert!(wait.is_some(), "before DEBOUNCE tick should report a wait");

        // After DEBOUNCE — wait is None (entry is ready).
        clock.advance(DEBOUNCE + Duration::from_millis(1));
        let wait = lp.tick(|_, _, _| None);
        assert!(
            wait.is_none(),
            "after DEBOUNCE tick should report no remaining wait (entry is ready)"
        );
    }

    #[test]
    fn tick_before_debounce_does_not_dispatch() {
        let mut lp = LivePreview::default();
        lp.mark_tweak(1, PreviewKind::Mask);
        let wait = lp.tick(|_, _, _| None);
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
        lp.result_tx.send(PreviewResult { version: 0, start: None,
            item_id: 42,
            rgba: Arc::new(RgbaImage::new(1, 1)),
            kind: PreviewKind::Mask,
            generation: 1,
            new_edge: None,
            new_bold: None,
            new_masked_base: None,
            new_reference: None,
            new_faced: None,
            new_depth_photo: None,
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None,
            applied_color_strength: None,
            is_final: false,
        }).unwrap();

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
        lp.result_tx.send(PreviewResult { version: 0, start: None,
            item_id: 99,
            rgba: Arc::new(RgbaImage::new(2, 2)),
            kind: PreviewKind::Mask,
            generation: 7,
            new_edge: None,
            new_bold: None,
            new_masked_base: None,
            new_reference: None,
            new_faced: None,
            new_depth_photo: None,
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None,
            applied_color_strength: None,
            is_final: false,
        }).unwrap();

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
        lp.result_tx.send(PreviewResult { version: 0, start: None,
            item_id: 777,
            rgba: Arc::new(RgbaImage::new(1, 1)),
            kind: PreviewKind::Mask,
            generation: 1,
            new_edge: None,
            new_bold: None,
            new_masked_base: None,
            new_reference: None,
            new_faced: None,
            new_depth_photo: None,
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None,
            applied_color_strength: None,
            is_final: false,
        }).unwrap();
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
        lp.result_tx.send(PreviewResult { version: 0, start: None,
            item_id: 11,
            rgba: Arc::new(RgbaImage::new(1, 1)),
            kind: PreviewKind::Mask,
            generation: 3,
            new_edge: None,
            new_bold: None,
            new_masked_base: None,
            new_reference: None,
            new_faced: None,
            new_depth_photo: None,
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None,
            applied_color_strength: None,
            is_final: false,
        }).unwrap();

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
            Pending { last_tweak_at: Instant::now(), kind: PreviewKind::Mask, settled: false },
        );
        lp.result_tx.send(PreviewResult { version: 0, start: None,
            item_id: 22,
            rgba: Arc::new(RgbaImage::new(1, 1)),
            kind: PreviewKind::Mask,
            generation: 2,
            new_edge: None,
            new_bold: None,
            new_masked_base: None,
            new_reference: None,
            new_faced: None,
            new_depth_photo: None,
            applied_depth: Default::default(),
            applied_mask: prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default()),
            applied_tier2_knobs: None,
            applied_color_strength: None,
            is_final: false,
        }).unwrap();

        let drained = lp.drain_results();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].is_final, "pending for this item means drag is still active");
    }

    #[test]
    fn preview_kind_upscale_tier2_added() {
        // UpscaleTier2 is a distinct variant, not aliased to Mask or Edge.
        assert_ne!(PreviewKind::UpscaleTier2, PreviewKind::Mask);
        assert_ne!(PreviewKind::UpscaleTier2, PreviewKind::Edge);
        // Copy + PartialEq + Eq — can be pattern-matched like the others.
        let kind = PreviewKind::UpscaleTier2;
        let copied = kind;
        assert_eq!(kind, copied);
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

