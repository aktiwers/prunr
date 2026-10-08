//! Unified progress signal for in-flight dispatches.
//!
//! One shared shape so the seg, eraser/SD, and upscale paths all
//! publish progress through the same struct, and one set of widgets
//! (banner / modal) renders it. Replaces three independent surfaces:
//!
//!   - `app.status.{pct, stage}` (seg / batch)
//!   - `prunr_core::inpaint::InpaintProgress` (eraser / SD)
//!   - `Processor::upscale_tile_progress` (upscale)
//!
//! Two-level counter: `outer` counts a per-stroke / per-batch unit
//! (SD tile-of-stroke, batch image-of-N); `inner` counts steps inside
//! the current outer unit (denoise step, postprocess stage, tile).
//! For dispatches with no nesting `outer` is `None` and `inner` is
//! the sole counter.
//!
//! Widgets choose: render the nested form `"Tile 2 of 3 — step 5 of 8"`,
//! or the flat form `"step 13 of 24"` derived from
//! `outer × inner_total + inner`. Either keeps the user aware of the
//! true remaining work; the prior surfaces silently reset `inner` at
//! each `outer` boundary, hiding the rest of the stroke.

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

/// Which dispatch is publishing — drives the headline label and the
/// per-step text source. New variants (depth, mat-cutting, …) extend
/// this enum and the widget match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressKind {
    /// Background-removal segmentation pipeline (BiRefNet, Silueta, …).
    Seg,
    /// Small inpaint models running in-process (LaMa, BigLaMa, MIGAN).
    Eraser,
    /// Stable Diffusion inpaint via the subprocess bridge.
    SdInpaint,
    /// Super-resolution upscale (Real-ESRGAN, HAT-L).
    Upscale,
}

impl ProgressKind {
    /// Verb form for the headline ("Erasing", "Upscaling", …).
    pub fn headline(self) -> &'static str {
        match self {
            ProgressKind::Seg => "Processing",
            ProgressKind::Eraser => "Erasing",
            ProgressKind::SdInpaint => "Erasing",
            ProgressKind::Upscale => "Upscaling",
        }
    }

    /// Singular noun for the inner counter ("tile" / "step"). The
    /// eraser + SD path uses "step" for the denoise loop; the upscale
    /// path uses "tile". Seg has multiple internal stages so "step"
    /// reads cleanly.
    pub fn inner_noun(self) -> &'static str {
        match self {
            ProgressKind::Upscale => "tile",
            ProgressKind::Seg | ProgressKind::Eraser | ProgressKind::SdInpaint => "step",
        }
    }

    /// Singular noun for the outer counter when set. `Some` only for
    /// kinds that meaningfully nest today (SD/Eraser tile-of-stroke);
    /// `None` for kinds that never set `outer`. Returning `Option`
    /// rather than a placeholder string means a caller reaching for
    /// the noun without first checking `outer.is_some()` fails at
    /// the type level instead of rendering a misleading "Tile 0 of 0".
    pub fn outer_noun(self) -> Option<&'static str> {
        match self {
            ProgressKind::SdInpaint | ProgressKind::Eraser => Some("tile"),
            ProgressKind::Seg | ProgressKind::Upscale => None,
        }
    }
}

/// Per-dispatch step-label strings. Defined once here so the dispatch
/// sites + tests + comments don't drift independently — a rename here
/// is a one-site edit, and grep-finding a label landing in the wrong
/// widget is trivial.
pub mod step_labels {
    pub const LOADING_MODEL: &str = "Loading model";
    pub const TILE_INFERENCE: &str = "Tile inference";
    /// Cancel was requested but the EP finishes its current tile first
    /// (OpenVINO ignores `RunOptions::terminate` mid-run).
    pub const CANCELLING: &str = "Cancelling… finishing current tile";
    pub const DENOISING: &str = "Denoising";
    pub const INPAINTING: &str = "Inpainting";
}

/// What a banner-Cancel click should target, derived purely from the
/// snapshot's `kind` plus an optional selected-item handle. Surfaces
/// the routing as data so the canvas closure can stay a one-line
/// `match`, and the four arms are unit-testable without a fixture
/// `PrunrApp`.
///
/// `Eraser` / `SdInpaint` need the item handle (per-stroke cancel);
/// when none is supplied the routing returns `None` (the banner
/// click is a no-op rather than collateral-damaging another item).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelTarget {
    InpaintForItem(u64),
    Upscale,
    SegBatchAndReset,
}

/// Pure routing: given the slot's `kind` and the active inpaint
/// item (if any), what should the banner Cancel button target?
pub fn cancel_target_for(kind: ProgressKind, active_inpaint_item: Option<u64>) -> Option<CancelTarget> {
    match kind {
        ProgressKind::Eraser | ProgressKind::SdInpaint => {
            active_inpaint_item.map(CancelTarget::InpaintForItem)
        }
        ProgressKind::Upscale => Some(CancelTarget::Upscale),
        ProgressKind::Seg => Some(CancelTarget::SegBatchAndReset),
    }
}

/// Translate the IPC sentinel `outer_total == 0` (= "no nesting") into
/// the `Option<(current, total)>` shape `DispatchProgress.outer` wants.
/// Only consumed by the two `DispatchProgress` builders in this module
/// — `pub(crate)` so the unit tests can pin the sentinel meaning, but
/// not part of the public surface.
pub(crate) fn outer_from_atomics(outer_current: u32, outer_total: u32) -> Option<(u32, u32)> {
    (outer_total > 0).then_some((outer_current, outer_total))
}

/// Snapshot of a single in-flight dispatch. Cheap to clone — only the
/// step_label can be a heap `String`, and short labels live as
/// `Cow::Borrowed(&'static str)`.
#[derive(Debug, Clone)]
pub struct DispatchProgress {
    pub kind: ProgressKind,
    /// Per-stroke / per-batch unit when applicable. `None` for
    /// non-tiled dispatches (Seg, Upscale today). `Some((current, total))`
    /// for SD strokes split into multiple 512² patches.
    pub outer: Option<(u32, u32)>,
    /// Step counter inside the current outer unit. `(0, 0)` means
    /// "indeterminate" — widgets render the spinner-only form.
    pub inner: (u32, u32),
    /// What the current step is doing, in human prose. Short fragments
    /// ("Decode", "Inference") are `Borrowed`; valued text
    /// ("Denoising at sigma 0.42") is `Owned`.
    pub step_label: Cow<'static, str>,
}

impl DispatchProgress {
    /// Builder for the seg / batch pipeline. `step` is whatever
    /// `BatchManager::progress().stage` reports ("Processing 3/5").
    pub fn seg(done: u32, total: u32, step: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind: ProgressKind::Seg,
            outer: None,
            inner: (done, total),
            step_label: step.into(),
        }
    }

    /// Builder for the upscale dispatch. `tile_total = 0` is OK at the
    /// "loading model" pre-dispatch point — the counter is then
    /// indeterminate.
    pub fn upscale(tile_done: u32, tile_total: u32, label: &'static str) -> Self {
        Self {
            kind: ProgressKind::Upscale,
            outer: None,
            inner: (tile_done, tile_total),
            step_label: Cow::Borrowed(label),
        }
    }

    /// Builder for the SD inpaint subprocess pump. Wraps the
    /// `outer_total == 0 → None` sentinel translation so call sites
    /// don't have to repeat it.
    pub fn sd_inpaint(outer_current: u32, outer_total: u32, inner: (u32, u32)) -> Self {
        Self {
            kind: ProgressKind::SdInpaint,
            outer: outer_from_atomics(outer_current, outer_total),
            inner,
            step_label: Cow::Borrowed(step_labels::DENOISING),
        }
    }

    /// Builder for the in-process LaMa dispatch (read-time synthesise
    /// from `InpaintProgress` in the canvas, since LaMa doesn't go
    /// through the subprocess bridge that publishes the slot).
    pub fn lama_inpaint(outer_current: u32, outer_total: u32, inner: (u32, u32)) -> Self {
        Self {
            kind: ProgressKind::Eraser,
            outer: outer_from_atomics(outer_current, outer_total),
            inner,
            step_label: Cow::Borrowed(step_labels::INPAINTING),
        }
    }
}

impl DispatchProgress {
    /// Render the counter in the flat form: `"step 13 of 24"` (counting
    /// inner across outer iterations). Returns `None` when there's no
    /// meaningful counter yet — widgets fall back to the spinner-only
    /// form.
    pub fn flat_counter_text(&self) -> Option<String> {
        let (cur, total) = self.flat_counter()?;
        Some(format!("{noun} {cur} of {total}", noun = self.kind.inner_noun()))
    }

    /// Render the counter in the nested form: `"Tile 2 of 3 — step 5
    /// of 8"`. Returns `None` when `outer` is `None` (no nesting) or
    /// when inner is indeterminate.
    pub fn nested_counter_text(&self) -> Option<String> {
        let (oc, ot) = self.outer?;
        let (ic, it) = self.inner;
        if it == 0 {
            return None;
        }
        // `outer_noun` is `Some` only when `outer.is_some()` makes
        // sense for the kind. We already short-circuited above when
        // `self.outer` was `None`; if `outer_noun` is `None` here the
        // kind set `outer` despite having no noun (mis-wiring at the
        // dispatch site) — fall back to "outer" for visibility.
        let outer_noun = capitalise_first(self.kind.outer_noun().unwrap_or("outer"));
        let inner_noun = self.kind.inner_noun();
        Some(format!(
            "{outer_noun} {oc} of {ot} \u{2014} {inner_noun} {ic} of {it}",
        ))
    }

    /// Flattened `(current, total)` across the outer × inner space.
    /// `None` when indeterminate. Monotonic — every step that lands
    /// advances `current` and never resets, even at outer boundaries.
    pub fn flat_counter(&self) -> Option<(u32, u32)> {
        let (ic, it) = self.inner;
        if it == 0 {
            return None;
        }
        match self.outer {
            // No nesting: inner is the full picture.
            None => Some((ic, it)),
            // Nested: total steps = outer_total * inner_total.
            // current = completed outer units × inner_total + current inner.
            // (outer_current is 1-based per convention; the completed
            // count is outer_current - 1.)
            Some((oc, ot)) => {
                if ot == 0 {
                    return None;
                }
                let completed_outer = oc.saturating_sub(1);
                let total_steps = ot.saturating_mul(it);
                let current_step = completed_outer.saturating_mul(it).saturating_add(ic);
                Some((current_step.min(total_steps), total_steps))
            }
        }
    }

    /// Fraction in `0.0..=1.0` for the progress-bar fill. `None` when
    /// the counter is indeterminate. Monotonic by construction.
    pub fn fraction(&self) -> Option<f32> {
        let (cur, total) = self.flat_counter()?;
        if total == 0 {
            return None;
        }
        Some((cur as f32 / total as f32).clamp(0.0, 1.0))
    }
}

/// Shared slot the dispatcher writes and the widgets read. `None` when
/// no dispatch is in flight. `Arc<Mutex<…>>` over atomics because the
/// payload includes a heap `Cow` for `step_label`; writes are
/// step-frequency (≤ a few Hz), reads are render-frequency (60 Hz),
/// and the lock is held for microseconds — contention is invisible.
#[derive(Debug, Clone, Default)]
pub struct DispatchProgressSlot {
    inner: Arc<Mutex<Option<DispatchProgress>>>,
}

impl DispatchProgressSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the active progress snapshot. Pass `None` to clear when
    /// the dispatch finishes (success, cancel, or error).
    pub fn set(&self, progress: Option<DispatchProgress>) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = progress;
    }

    /// Mutate the active progress in place. The closure runs with
    /// `&mut Option<DispatchProgress>` so callers can patch fields
    /// without rebuilding the whole struct (typical: bump `inner.0`
    /// and refresh `step_label`).
    pub fn update(&self, f: impl FnOnce(&mut Option<DispatchProgress>)) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut guard);
    }

    /// Snapshot the current progress for a widget. `None` when no
    /// dispatch is in flight.
    pub fn read(&self) -> Option<DispatchProgress> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.clone()
    }
}

/// Capitalise the first ASCII char — small helper for the nested-form
/// outer noun ("tile" → "Tile"). Stays inside the module rather than
/// pulling in a `heck`-style dependency for one site.
fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upscale(inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress {
            kind: ProgressKind::Upscale,
            outer: None,
            inner,
            step_label: Cow::Borrowed("Real-ESRGAN forward pass"),
        }
    }

    fn sd_nested(outer: (u32, u32), inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress {
            kind: ProgressKind::SdInpaint,
            outer: Some(outer),
            inner,
            // Use the constant rather than a literal so a `step_labels`
            // rename keeps both production and test in lockstep.
            step_label: Cow::Borrowed(step_labels::DENOISING),
        }
    }

    #[test]
    fn flat_counter_for_unnested_dispatch_is_inner() {
        let p = upscale((12, 49));
        assert_eq!(p.flat_counter(), Some((12, 49)));
        assert_eq!(p.flat_counter_text().as_deref(), Some("tile 12 of 49"));
    }

    #[test]
    fn flat_counter_for_nested_dispatch_accumulates_across_outer() {
        // Tile 2 of 3, step 5 of 8 → completed_outer = 1 tile (= 8 steps),
        // plus 5 current-tile steps = 13. Total = 3 × 8 = 24.
        let p = sd_nested((2, 3), (5, 8));
        assert_eq!(p.flat_counter(), Some((13, 24)));
        assert_eq!(p.flat_counter_text().as_deref(), Some("step 13 of 24"));
    }

    #[test]
    fn nested_counter_text_uses_outer_noun_capitalised() {
        let p = sd_nested((2, 3), (5, 8));
        assert_eq!(
            p.nested_counter_text().as_deref(),
            Some("Tile 2 of 3 \u{2014} step 5 of 8"),
        );
    }

    #[test]
    fn nested_counter_text_is_none_when_no_outer() {
        let p = upscale((12, 49));
        assert!(p.nested_counter_text().is_none());
    }

    #[test]
    fn indeterminate_inner_returns_none_counters() {
        let p = upscale((0, 0));
        assert!(p.flat_counter().is_none());
        assert!(p.flat_counter_text().is_none());
        assert!(p.fraction().is_none());
    }

    #[test]
    fn fraction_is_monotonic_across_outer_boundary() {
        // The original bug: end-of-tile-1 (step 8 of 8) and
        // start-of-tile-2 (step 0 of 8) should NOT swing fraction
        // backwards. We compute (8, 24) → 0.333 vs (8, 24) → 0.333.
        // (Tile 2 step 0 is "0 completed in current tile" → still
        // exactly 8 steps done overall.) Verify the next step inside
        // tile 2 advances: (9, 24) → 0.375.
        let end_of_t1 = sd_nested((1, 3), (8, 8));
        let start_of_t2 = sd_nested((2, 3), (0, 8));
        let mid_t2 = sd_nested((2, 3), (1, 8));
        let f_end = end_of_t1.fraction().unwrap();
        let f_start = start_of_t2.fraction().unwrap();
        let f_mid = mid_t2.fraction().unwrap();
        assert!(
            (f_end - 8.0 / 24.0).abs() < 1e-6,
            "end of tile 1 must report 8/24"
        );
        assert!(f_start >= f_end, "start of tile 2 must not regress");
        assert!(f_mid > f_start, "next step must advance");
    }

    #[test]
    fn fraction_clamps_to_one_at_completion() {
        let p = sd_nested((3, 3), (8, 8));
        assert!((p.fraction().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fraction_handles_zero_outer_total_gracefully() {
        let p = DispatchProgress {
            kind: ProgressKind::SdInpaint,
            outer: Some((0, 0)),
            inner: (1, 8),
            step_label: Cow::Borrowed("Denoising"),
        };
        assert!(p.fraction().is_none(),
            "zero outer_total is invalid input — must not divide by zero");
    }

    #[test]
    fn slot_set_and_read_round_trips() {
        let slot = DispatchProgressSlot::new();
        assert!(slot.read().is_none(), "default slot is empty");

        slot.set(Some(upscale((3, 10))));
        let p = slot.read().expect("read after set");
        assert_eq!(p.flat_counter(), Some((3, 10)));

        slot.set(None);
        assert!(slot.read().is_none(), "set None clears");
    }

    #[test]
    fn slot_update_mutates_in_place() {
        let slot = DispatchProgressSlot::new();
        slot.set(Some(upscale((1, 10))));
        slot.update(|p| {
            if let Some(p) = p {
                p.inner = (5, 10);
                p.step_label = Cow::Borrowed("After update");
            }
        });
        let p = slot.read().expect("read after update");
        assert_eq!(p.inner, (5, 10));
        assert_eq!(p.step_label.as_ref(), "After update");
    }

    #[test]
    fn dispatch_kind_headlines_and_nouns_are_distinct() {
        // Pin the strings — widgets concatenate them, so a rename would
        // be a user-visible label change.
        assert_eq!(ProgressKind::Seg.headline(), "Processing");
        assert_eq!(ProgressKind::Eraser.headline(), "Erasing");
        assert_eq!(ProgressKind::SdInpaint.headline(), "Erasing");
        assert_eq!(ProgressKind::Upscale.headline(), "Upscaling");
        assert_eq!(ProgressKind::Upscale.inner_noun(), "tile");
        assert_eq!(ProgressKind::Seg.inner_noun(), "step");
    }

    #[test]
    fn outer_noun_is_some_only_for_nesting_kinds() {
        // Pinning the variant set that has an outer dimension —
        // adding a new ProgressKind variant forces an explicit
        // decision here.
        assert_eq!(ProgressKind::SdInpaint.outer_noun(), Some("tile"));
        assert_eq!(ProgressKind::Eraser.outer_noun(), Some("tile"));
        assert_eq!(ProgressKind::Seg.outer_noun(), None);
        assert_eq!(ProgressKind::Upscale.outer_noun(), None);
    }

    #[test]
    fn outer_from_atomics_translates_sentinel() {
        // outer_total == 0 is the IPC sentinel for "no nesting"; the
        // GUI layer converts it to None via this helper. Both call
        // sites (SD pump + LaMa fallback) must agree on the sentinel
        // meaning — that's what this test pins.
        assert_eq!(outer_from_atomics(2, 3), Some((2, 3)));
        assert_eq!(outer_from_atomics(0, 0), None);
        // outer_current is ignored when outer_total is 0 — the sentinel
        // is solely on outer_total.
        assert_eq!(outer_from_atomics(99, 0), None);
    }

    #[test]
    fn seg_builder_accepts_owned_string_for_dynamic_stage() {
        // `BatchManager::progress()` returns `stage: String`. The
        // builder must accept that without forcing the caller into a
        // `.into()` dance.
        let stage: String = "Processing 3 of 5".to_string();
        let p = DispatchProgress::seg(3, 5, stage);
        assert_eq!(p.inner, (3, 5));
        assert_eq!(p.step_label.as_ref(), "Processing 3 of 5");
    }

    #[test]
    fn upscale_builder_uses_borrowed_label() {
        let p = DispatchProgress::upscale(12, 49, step_labels::TILE_INFERENCE);
        assert_eq!(p.kind, ProgressKind::Upscale);
        assert_eq!(p.outer, None);
        assert_eq!(p.inner, (12, 49));
        assert_eq!(p.step_label.as_ref(), "Tile inference");
    }

    #[test]
    fn sd_inpaint_builder_lifts_outer_sentinel() {
        // outer_total > 0 → outer is Some.
        let p = DispatchProgress::sd_inpaint(2, 3, (5, 8));
        assert_eq!(p.outer, Some((2, 3)));
        // outer_total == 0 → outer is None (single-tile stroke).
        let p_single = DispatchProgress::sd_inpaint(0, 0, (5, 8));
        assert_eq!(p_single.outer, None);
        assert_eq!(p_single.step_label.as_ref(), step_labels::DENOISING);
    }

    #[test]
    fn lama_inpaint_builder_sets_kind_label_and_outer() {
        // Tile-of-stroke nested: outer surfaces as Some.
        let p = DispatchProgress::lama_inpaint(1, 2, (3, 4));
        assert_eq!(p.kind, ProgressKind::Eraser);
        assert_eq!(p.outer, Some((1, 2)));
        assert_eq!(p.inner, (3, 4));
        assert_eq!(p.step_label.as_ref(), step_labels::INPAINTING);

        // Single-pass (no outer dim): outer is None.
        let p_single = DispatchProgress::lama_inpaint(0, 0, (0, 0));
        assert_eq!(p_single.outer, None);
    }

    #[test]
    fn cancel_target_eraser_needs_active_item() {
        assert_eq!(
            cancel_target_for(ProgressKind::Eraser, Some(42)),
            Some(CancelTarget::InpaintForItem(42)),
        );
        assert_eq!(
            cancel_target_for(ProgressKind::Eraser, None),
            None,
            "no active item → click is a no-op (the inpaint isn't ours to cancel)",
        );
    }

    #[test]
    fn cancel_target_sd_inpaint_needs_active_item() {
        assert_eq!(
            cancel_target_for(ProgressKind::SdInpaint, Some(7)),
            Some(CancelTarget::InpaintForItem(7)),
        );
        assert_eq!(cancel_target_for(ProgressKind::SdInpaint, None), None);
    }

    #[test]
    fn cancel_target_upscale_ignores_item() {
        // Upscale is single-flight on the Processor; the item handle
        // is irrelevant — there's only one upscale to cancel.
        assert_eq!(
            cancel_target_for(ProgressKind::Upscale, None),
            Some(CancelTarget::Upscale),
        );
        assert_eq!(
            cancel_target_for(ProgressKind::Upscale, Some(99)),
            Some(CancelTarget::Upscale),
        );
    }

    #[test]
    fn cancel_target_seg_routes_to_batch_reset() {
        // Seg banner Cancel → batch cancel, not an inpaint cancel.
        // Pins the H1 audit fix: a future refactor that maps
        // ProgressKind::Seg → cancel_all_inpaints() (the wrong call)
        // fails this assertion before users see a dropped click.
        assert_eq!(
            cancel_target_for(ProgressKind::Seg, None),
            Some(CancelTarget::SegBatchAndReset),
        );
        assert_eq!(
            cancel_target_for(ProgressKind::Seg, Some(1)),
            Some(CancelTarget::SegBatchAndReset),
            "seg routes to batch reset regardless of selection",
        );
    }

    #[test]
    fn seg_builder_borrowed_arm_works_for_static_str() {
        // The Owned arm is covered by the dynamic-stage test above; this
        // pins the Borrowed arm so the `impl Into<Cow>` signature can't
        // silently regress to `String`-only.
        let p = DispatchProgress::seg(0, 0, "Ready");
        assert_eq!(p.inner, (0, 0));
        assert_eq!(p.step_label.as_ref(), "Ready");
        assert!(matches!(p.step_label, Cow::Borrowed(_)),
            "static-str literal must produce Cow::Borrowed (zero alloc)");
    }
}
