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
//! The fraction runs across both levels (`outer × inner_total + inner`),
//! so it never swings back when the next outer unit starts.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

/// Which dispatch is publishing; decides what Cancel stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressKind {
    /// Background removal (BiRefNet, Silueta, …).
    Seg,
    /// Any eraser: LaMa family in-process, Stable Diffusion via its bridge.
    Inpaint,
    /// Super-resolution upscale (Real-ESRGAN, HAT-L).
    Upscale,
}

/// What Cancel stops. An eraser run needs the item it runs on; without
/// one the click does nothing rather than stop another item's stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelTarget {
    InpaintForItem(u64),
    Upscale,
    SegBatchAndReset,
}

pub fn cancel_target_for(kind: ProgressKind, active_inpaint_item: Option<u64>) -> Option<CancelTarget> {
    match kind {
        ProgressKind::Inpaint => active_inpaint_item.map(CancelTarget::InpaintForItem),
        ProgressKind::Upscale => Some(CancelTarget::Upscale),
        ProgressKind::Seg => Some(CancelTarget::SegBatchAndReset),
    }
}

/// Snapshot of a single in-flight dispatch, cloned once per frame: the
/// tile layout is shared, the rest is small.
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
    /// The current step, in the words the user reads.
    pub step_label: Cow<'static, str>,
    /// Cancel was asked for; the run finishes its current tile or step.
    pub cancelling: bool,
    /// What the counts count, as the pipeline reported it.
    pub outer_unit: Option<prunr_core::Unit>,
    pub inner_unit: Option<prunr_core::Unit>,
    /// The tile map: cells with overlaps split down the middle, and how
    /// far each one is.
    pub tiles: Arc<[prunr_core::TileRect]>,
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
            cancelling: false,
            outer_unit: None,
            inner_unit: None,
            tiles: Arc::new([]),
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
            ProgressUpdate::Step(step) => self.step_label = Cow::Borrowed(step.label()),
            &ProgressUpdate::Outer { done, total, unit } => {
                self.outer = Some((done, total));
                self.outer_unit = Some(unit);
                self.note_pace(now);
            }
            &ProgressUpdate::Inner { done, total, unit } => self.set_inner(done, total, unit, now),
            ProgressUpdate::Tiles(rects) => {
                self.tiles = split_overlaps(rects).into();
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
    fn remaining(&self, now: Instant) -> Option<Duration> {
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

    /// Builder for the seg / batch pipeline. `step` is whatever
    /// `BatchManager::progress().stage` reports ("Processing 3/5").
    pub fn seg(done: u32, total: u32, step: impl Into<Cow<'static, str>>) -> Self {
        let mut p = Self::new(ProgressKind::Seg, step);
        p.set_inner(done, total, prunr_core::Unit::Image, Instant::now());
        p
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

    /// "Crop 2 of 3 · step 5 of 20", or `None` while nothing is counted.
    /// The outer count names the unit in progress; the inner count what
    /// is done. A count of one says nothing and is left out.
    pub fn counter_text(&self) -> Option<String> {
        use prunr_core::Unit;
        let outer = self.outer.filter(|&(_, total)| total > 1).map(|(done, total)| {
            (self.outer_unit.unwrap_or(Unit::Crop), (done + 1).min(total), total)
        });
        let (done, total) = self.inner;
        let inner = (total > 1).then(|| (self.inner_unit.unwrap_or(Unit::Step), done, total));
        let parts: Vec<String> = outer.into_iter().chain(inner)
            .map(|(unit, n, total)| format!("{} {n} of {total}", unit.noun(1)))
            .collect();
        (!parts.is_empty()).then(|| capitalise_first(&parts.join(" \u{b7} ")))
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
    pub fn progress_for(&self, initial: DispatchProgress, cancel: Arc<AtomicBool>) -> (prunr_core::Progress, RunSink) {
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

/// Trim each tile where it overlaps a neighbour, at the middle of the
/// overlap, so overlapping tiles paint as cells that meet edge to edge.
fn split_overlaps(rects: &[prunr_core::TileRect]) -> Vec<prunr_core::TileRect> {
    rects.iter().map(|t| {
        let mut cell = *t;
        for o in rects {
            let (ox1, oy1) = (o.x + o.w, o.y + o.h);
            if std::ptr::eq(o, t) || o.x >= t.x + t.w || ox1 <= t.x || o.y >= t.y + t.h || oy1 <= t.y {
                continue;
            }
            if o.x < t.x {
                let mid = (t.x + ox1) / 2.0;
                if mid > cell.x { cell.w -= mid - cell.x; cell.x = mid; }
            }
            if o.x > t.x { cell.w = cell.w.min((o.x + t.x + t.w) / 2.0 - cell.x); }
            if o.y < t.y {
                let mid = (t.y + oy1) / 2.0;
                if mid > cell.y { cell.h -= mid - cell.y; cell.y = mid; }
            }
            if o.y > t.y { cell.h = cell.h.min((o.y + t.y + t.h) / 2.0 - cell.y); }
        }
        cell
    }).collect()
}

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
        DispatchProgress { outer: Some(outer), inner, ..DispatchProgress::new(ProgressKind::Inpaint, prunr_core::Step::Denoising.label()) }
    }

    #[test]
    fn counts_accumulate_across_the_outer_unit() {
        // One crop of three done (8 steps) plus 5 steps of the next = 13 of 24.
        assert_eq!(sd_nested((1, 3), (5, 8)).flat_counter(), Some((13, 24)));
        assert_eq!(upscale((12, 49)).flat_counter(), Some((12, 49)));
    }

    #[test]
    fn the_counter_names_the_reported_units_and_leaves_out_counts_of_one() {
        use prunr_core::Unit;
        let mut p = sd_nested((1, 3), (5, 8));
        assert_eq!(p.counter_text().as_deref(), Some("Crop 2 of 3 \u{b7} step 5 of 8"), "units default per level");
        p.outer_unit = Some(Unit::Pass);
        p.inner_unit = Some(Unit::Tile);
        assert_eq!(p.counter_text().as_deref(), Some("Pass 2 of 3 \u{b7} tile 5 of 8"));
        assert_eq!(sd_nested((0, 1), (5, 8)).counter_text().as_deref(), Some("Step 5 of 8"));
        assert_eq!(DispatchProgress::seg(0, 1, "x").counter_text(), None);
    }

    #[test]
    fn indeterminate_inner_returns_none_counters() {
        let p = upscale((0, 0));
        assert!(p.flat_counter().is_none());
        assert!(p.counter_text().is_none());
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
        let p = DispatchProgress { outer: Some((0, 0)), inner: (1, 8), ..DispatchProgress::new(ProgressKind::Inpaint, "Denoising") };
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
    fn overlapping_tiles_meet_in_the_middle_of_the_overlap() {
        use prunr_core::TileRect;
        // Two columns overlapping by 0.1, two rows overlapping by 0.2.
        let t = |x, y| TileRect { x, y, w: 0.55, h: 0.6 };
        let cells = split_overlaps(&[t(0.0, 0.0), t(0.45, 0.0), t(0.0, 0.4), t(0.45, 0.4)]);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(cells[0].w, 0.5) && close(cells[0].h, 0.5), "{:?}", cells[0]);
        assert!(close(cells[1].x, 0.5) && close(cells[1].x + cells[1].w, 1.0), "{:?}", cells[1]);
        assert!(close(cells[3].y, 0.5) && close(cells[3].y + cells[3].h, 1.0), "{:?}", cells[3]);
        let apart = [TileRect { x: 0.0, y: 0.0, w: 0.2, h: 0.2 }, TileRect { x: 0.5, y: 0.5, w: 0.2, h: 0.2 }];
        assert_eq!(split_overlaps(&apart), apart, "tiles that do not touch stay whole");
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
        assert_eq!(cancel_target_for(ProgressKind::Inpaint, Some(42)), Some(CancelTarget::InpaintForItem(42)));
        assert_eq!(cancel_target_for(ProgressKind::Inpaint, None), None, "another item's stroke is not ours to stop");
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
        p.cancelling = true;
        p.apply(&ProgressUpdate::Step(Step::Finishing));
        assert!(p.cancelling, "a late step must not hide the cancel");
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
        // Seg Cancel → batch cancel, not an inpaint cancel.
        // A future refactor that maps
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
