//! Unified progress signal for in-flight dispatches.
//!
//! Every pipeline reports through `prunr_core::Progress`; a run started
//! with `DispatchProgressSlot::begin` folds those reports into one
//! `DispatchProgress` that one set of widgets renders.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
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

pub mod step_labels {
    /// Cancel was requested but the EP finishes its current tile first
    /// (OpenVINO ignores `RunOptions::terminate` mid-run).
    pub const CANCELLING: &str = "Cancelling… finishing current tile";
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
    /// What the counts count, as the pipeline reported it.
    pub outer_unit: Option<prunr_core::Unit>,
    pub inner_unit: Option<prunr_core::Unit>,
    /// The tile map: where the work happens, and how far each tile is.
    pub tiles: Vec<prunr_core::TileRect>,
    pub tile_states: Vec<TileState>,
    /// When work started moving and how far along it was then.
    pub pace: Option<(Instant, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState {
    Waiting,
    Running,
    Done,
}

/// Below this share done, or this long counting, an estimate is noise.
const PACE_MIN_SHARE: f32 = 0.02;
const PACE_MIN_TIME: Duration = Duration::from_secs(2);

impl DispatchProgress {
    pub fn new(kind: ProgressKind, step_label: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            outer: None,
            inner: (0, 0),
            step_label: step_label.into(),
            outer_unit: None,
            inner_unit: None,
            tiles: Vec::new(),
            tile_states: Vec::new(),
            pace: None,
        }
    }

    /// Fold one pipeline report into the snapshot the widgets draw.
    pub fn apply(&mut self, update: &prunr_core::ProgressUpdate) {
        self.apply_at(update, Instant::now());
    }

    pub(crate) fn apply_at(&mut self, update: &prunr_core::ProgressUpdate, now: Instant) {
        use prunr_core::ProgressUpdate;
        match update {
            // A cancelling run keeps saying so until it ends.
            ProgressUpdate::Step(step) if !self.is_cancelling() => self.step_label = Cow::Borrowed(step.label()),
            ProgressUpdate::Step(_) => {}
            &ProgressUpdate::Outer { done, total, unit } => {
                self.outer = Some((done, total));
                self.outer_unit = Some(unit);
                self.note_pace(now);
            }
            &ProgressUpdate::Inner { done, total, unit } => self.set_inner(done, total, unit, now),
            ProgressUpdate::Tiles(rects) => {
                self.tiles = rects.clone();
                self.tile_states = vec![TileState::Waiting; rects.len()];
            }
            &ProgressUpdate::Tile { index, done } => {
                if let Some(state) = self.tile_states.get_mut(index as usize) {
                    *state = if done { TileState::Done } else { TileState::Running };
                }
            }
        }
    }

    pub(crate) fn set_inner(&mut self, done: u32, total: u32, unit: prunr_core::Unit, now: Instant) {
        self.inner = (done, total);
        self.inner_unit = Some(unit);
        self.note_pace(now);
    }

    /// Pace runs from the last moment the share done still stood at its
    /// first value, so time spent before any work moves is never counted.
    fn note_pace(&mut self, now: Instant) {
        let Some(f) = self.fraction() else { return };
        match self.pace {
            Some((_, at_start)) if at_start != f => {}
            _ => self.pace = Some((now, f)),
        }
    }

    /// Time left at the pace since counting started; `None` until there
    /// is enough of it to say.
    pub fn remaining(&self, now: Instant) -> Option<Duration> {
        let (since, at_start) = self.pace?;
        let f = self.fraction()?;
        let elapsed = now.saturating_duration_since(since);
        let gained = f - at_start;
        if gained < PACE_MIN_SHARE || elapsed < PACE_MIN_TIME {
            return None;
        }
        Some(elapsed.mul_f32((1.0 - f) / gained))
    }

    /// "a few seconds left", "about 40 s left", "about 2 min left".
    pub fn remaining_text(&self, now: Instant) -> Option<String> {
        let secs = self.remaining(now)?.as_secs_f32();
        Some(if secs < 10.0 {
            "a few seconds left".to_string()
        } else if secs < 60.0 {
            format!("about {} s left", ((secs / 10.0).round() * 10.0) as u32)
        } else {
            format!("about {} min left", (secs / 60.0).ceil() as u32)
        })
    }

    /// A cancel was requested and the run is finishing its current step.
    pub fn is_cancelling(&self) -> bool {
        self.step_label == step_labels::CANCELLING
    }

    /// Builder for the seg / batch pipeline. `step` is whatever
    /// `BatchManager::progress().stage` reports ("Processing 3/5").
    pub fn seg(done: u32, total: u32, step: impl Into<Cow<'static, str>>) -> Self {
        let mut p = Self::new(ProgressKind::Seg, step);
        p.set_inner(done, total, prunr_core::Unit::Image, Instant::now());
        p
    }

    /// Builder for the upscale dispatch. `tile_total = 0` is OK at the
    /// "loading model" pre-dispatch point — the counter is then
    /// indeterminate.
    pub fn upscale(tile_done: u32, tile_total: u32, label: &'static str) -> Self {
        Self { inner: (tile_done, tile_total), ..Self::new(ProgressKind::Upscale, label) }
    }

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
        let running = (oc + 1).min(ot);
        Some(format!(
            "{outer_noun} {running} of {ot} \u{2014} {inner_noun} {ic} of {it}",
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
            // Nested: total steps = outer_total * inner_total; `oc` is
            // the outer units already done.
            Some((oc, ot)) => {
                if ot == 0 {
                    return None;
                }
                let completed_outer = oc;
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
    /// Bumped by `begin`, so a finished run's late reports cannot
    /// reach the run that replaced it.
    run: Arc<AtomicU64>,
}

/// The listener `begin` hands a run: its reports reach the slot only
/// while the slot still shows that run.
#[derive(Clone)]
pub struct RunSink {
    slot: DispatchProgressSlot,
    run: u64,
}

impl RunSink {
    fn current(&self) -> bool {
        self.slot.run.load(Ordering::Acquire) == self.run
    }

    /// The run is over: clear the slot unless a newer run took it.
    pub fn end(&self) {
        if self.current() {
            self.slot.set(None);
        }
    }
}

impl prunr_core::ProgressSink for RunSink {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        if !self.current() {
            return;
        }
        self.slot.update(|p| {
            if let Some(p) = p {
                p.apply(&update);
            }
        });
    }
}

impl DispatchProgressSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start showing a run and get the listener its pipeline reports to.
    pub fn begin(&self, initial: DispatchProgress) -> RunSink {
        let run = self.run.fetch_add(1, Ordering::AcqRel) + 1;
        self.set(Some(initial));
        RunSink { slot: self.clone(), run }
    }

    /// A progress handle for a run starting now, with its cancel flag.
    pub fn progress_for(&self, initial: DispatchProgress, cancel: Arc<std::sync::atomic::AtomicBool>) -> (prunr_core::Progress, RunSink) {
        let sink = self.begin(initial);
        (prunr_core::Progress::new(Arc::new(sink.clone())).with_cancel(cancel), sink)
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
        DispatchProgress { inner, ..DispatchProgress::new(ProgressKind::Upscale, "Upscaling") }
    }

    /// `outer` is (done, total), as the pipelines report it.
    fn sd_nested(outer: (u32, u32), inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress { outer: Some(outer), inner, ..DispatchProgress::new(ProgressKind::SdInpaint, prunr_core::Step::Denoising.label()) }
    }

    #[test]
    fn flat_counter_for_unnested_dispatch_is_inner() {
        let p = upscale((12, 49));
        assert_eq!(p.flat_counter(), Some((12, 49)));
        assert_eq!(p.flat_counter_text().as_deref(), Some("tile 12 of 49"));
    }

    #[test]
    fn flat_counter_for_nested_dispatch_accumulates_across_outer() {
        // One tile of three done (8 steps) plus 5 steps of the next = 13 of 24.
        let p = sd_nested((1, 3), (5, 8));
        assert_eq!(p.flat_counter(), Some((13, 24)));
        assert_eq!(p.flat_counter_text().as_deref(), Some("step 13 of 24"));
    }

    #[test]
    fn nested_counter_text_uses_outer_noun_capitalised() {
        let p = sd_nested((1, 3), (5, 8));
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
        let end_of_t1 = sd_nested((0, 3), (8, 8));
        let start_of_t2 = sd_nested((1, 3), (0, 8));
        let mid_t2 = sd_nested((1, 3), (1, 8));
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
        let p = sd_nested((3, 3), (0, 8));
        assert!((p.fraction().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fraction_handles_zero_outer_total_gracefully() {
        let p = DispatchProgress { outer: Some((0, 0)), inner: (1, 8), ..DispatchProgress::new(ProgressKind::SdInpaint, "Denoising") };
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
        let p = DispatchProgress::upscale(12, 49, prunr_core::Step::Upscaling.label());
        assert_eq!(p.kind, ProgressKind::Upscale);
        assert_eq!(p.outer, None);
        assert_eq!(p.inner, (12, 49));
        assert_eq!(p.step_label.as_ref(), "Upscaling");
    }

    #[test]
    fn the_tile_map_follows_tile_reports() {
        use prunr_core::{ProgressUpdate, TileRect};
        let mut p = upscale((0, 0));
        let half = |x| TileRect { x, y: 0.0, w: 0.5, h: 1.0 };
        p.apply(&ProgressUpdate::Tiles(vec![half(0.0), half(0.5)]));
        assert_eq!(p.tile_states, [TileState::Waiting; 2]);
        p.apply(&ProgressUpdate::Tile { index: 0, done: false });
        p.apply(&ProgressUpdate::Tile { index: 7, done: true });
        assert_eq!(p.tile_states, [TileState::Running, TileState::Waiting]);
        p.apply(&ProgressUpdate::Tile { index: 0, done: true });
        assert_eq!(p.tile_states, [TileState::Done, TileState::Waiting]);
        p.apply(&ProgressUpdate::Tiles(vec![half(0.0)]));
        assert_eq!(p.tile_states, [TileState::Waiting], "a new layout starts over");
    }

    #[test]
    fn the_estimate_leaves_out_the_wait_before_work_moves() {
        use prunr_core::Unit;
        let t0 = Instant::now();
        let s = Duration::from_secs;
        let mut p = DispatchProgress::new(ProgressKind::Seg, "Loading the model");
        p.set_inner(0, 10, Unit::Image, t0);
        p.set_inner(0, 10, Unit::Image, t0 + s(30));
        p.set_inner(1, 10, Unit::Image, t0 + s(31));
        assert_eq!(p.remaining(t0 + s(31)), None, "one second is too soon to say");
        p.set_inner(2, 10, Unit::Image, t0 + s(40));
        // 20% in the 10 s since work started: 80% takes 40 s more.
        assert_eq!(p.remaining(t0 + s(40)), Some(s(40)));
        assert_eq!(p.remaining_text(t0 + s(40)).as_deref(), Some("about 40 s left"));
    }

    #[test]
    fn the_estimate_reads_like_a_person_would_say_it() {
        let t0 = Instant::now();
        let at = |done, total, secs| {
            let p = DispatchProgress { inner: (done, total), pace: Some((t0, 0.0)), ..upscale((0, 0)) };
            p.remaining_text(t0 + Duration::from_secs(secs))
        };
        assert_eq!(at(1, 2, 5).as_deref(), Some("a few seconds left"));
        assert_eq!(at(1, 5, 5).as_deref(), Some("about 20 s left"));
        assert_eq!(at(1, 10, 20).as_deref(), Some("about 3 min left"));
        assert_eq!(at(0, 10, 20), None, "nothing done, nothing to go on");
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

    /// A run that was replaced, or has ended, can still have a report in
    /// flight; it must not land on whatever the slot shows next.
    #[test]
    fn a_late_report_never_reaches_the_next_run() {
        use prunr_core::{ProgressSink, ProgressUpdate, Unit};
        let slot = DispatchProgressSlot::new();
        let first = slot.begin(upscale((0, 4)));
        let second = slot.begin(upscale((0, 48)));
        first.report(ProgressUpdate::Inner { done: 4, total: 4, unit: Unit::Tile });
        assert_eq!(slot.read().unwrap().inner, (0, 48), "the replaced run is ignored");
        second.report(ProgressUpdate::Inner { done: 3, total: 48, unit: Unit::Tile });
        assert_eq!(slot.read().unwrap().inner, (3, 48));
        first.end();
        assert!(slot.read().is_some(), "ending a replaced run leaves the current one");
        second.end();
        assert!(slot.read().is_none());
    }

    #[test]
    fn a_report_updates_the_snapshot_but_never_hides_a_cancel() {
        use prunr_core::{ProgressUpdate, Step, Unit};
        let mut p = upscale((0, 48));
        p.apply(&ProgressUpdate::Step(Step::Upscaling));
        p.apply(&ProgressUpdate::Inner { done: 3, total: 48, unit: Unit::Tile });
        p.apply(&ProgressUpdate::Outer { done: 1, total: 2, unit: Unit::Pass });
        assert_eq!((p.step_label.as_ref(), p.inner, p.outer), ("Upscaling", (3, 48), Some((1, 2))));
        p.step_label = std::borrow::Cow::Borrowed(step_labels::CANCELLING);
        p.apply(&ProgressUpdate::Step(Step::Finishing));
        assert!(p.is_cancelling(), "a late step name must not hide the cancel");
    }

    /// The upscale cancel says so through its step label; the widgets
    /// read the same fact to swap the headline and drop the Esc hint.
    #[test]
    fn a_cancelling_run_reads_as_cancelling() {
        let mut p = upscale((1, 48));
        assert!(!p.is_cancelling());
        p.step_label = std::borrow::Cow::Borrowed(step_labels::CANCELLING);
        assert!(p.is_cancelling());
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
