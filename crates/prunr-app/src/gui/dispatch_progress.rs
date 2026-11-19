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
    /// Colorize: one model run.
    Colorize,
    /// Restore faces: stops after the face it is on.
    Faces,
    /// Measuring a photo's depth: under a second, not cancellable.
    Depth,
}

/// What Cancel stops. An eraser run needs the item it runs on; without
/// one the click does nothing rather than stop another item's stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelTarget {
    /// The erase run the capsule shows, and the images queued after it.
    Inpaint,
    Upscale,
    Colorize,
    /// The cut-outs: the ticked ones when any ticked image awaits a cut,
    /// else all, as the toolbar's Cancel and Escape stop.
    Cuts,
    Faces,
}

pub fn cancel_target_for(kind: ProgressKind) -> Option<CancelTarget> {
    match kind {
        ProgressKind::Inpaint => Some(CancelTarget::Inpaint),
        ProgressKind::Upscale => Some(CancelTarget::Upscale),
        ProgressKind::Colorize => Some(CancelTarget::Colorize),
        ProgressKind::Seg => Some(CancelTarget::Cuts),
        ProgressKind::Faces => Some(CancelTarget::Faces),
        ProgressKind::Depth => None,
    }
}

/// Snapshot of a single in-flight dispatch, cloned once per frame: the
/// tile layout is shared, the rest is small.
#[derive(Debug, Clone)]
pub struct DispatchProgress {
    pub kind: ProgressKind,
    /// Per-batch / per-stroke unit: images of a batch or queue, crops of
    /// an SD stroke; `None` when the run counts one level only.
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
    /// The tile map: where the work happens, and how far each tile is.
    pub tiles: Arc<[prunr_core::TileRect]>,
    pub tile_states: Arc<Vec<TileState>>,
    /// Cells an earlier layout of this run finished (an erase fills
    /// region by region): they stay done.
    pub finished: Arc<[prunr_core::TileRect]>,
    /// When work started moving and how far along it was then.
    pub pace: Option<(Instant, f32)>,
    /// The image a one-image run works on (an upscale, an erase, a face
    /// search); `None` for a batch. Its tile map belongs over that image
    /// only.
    pub on: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeLeft {
    FewSeconds,
    Seconds(u32),
    Minutes(u32),
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
            tile_states: Arc::default(),
            finished: Arc::new([]),
            pace: None,
            on: None,
        }
    }

    /// This progress as a run on the image `item_id`.
    pub fn on(self, item_id: u64) -> Self {
        Self { on: Some(item_id), ..self }
    }

    /// Whether this progress belongs over the image `shown`: a batch's
    /// does everywhere, a one-image run's only over its own image.
    pub fn is_over(&self, shown: Option<u64>) -> bool {
        self.on.is_none() || self.on == shown
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
                // A run that re-lays one layout pass after pass adds no
                // second copy of a cell; the map draws every kept cell per frame.
                let fresh = self.tiles.iter().zip(self.tile_states.iter())
                    .filter(|&(c, s)| *s == TileState::Done && !self.finished.contains(c))
                    .map(|(c, _)| *c);
                self.finished = self.finished.iter().copied().chain(fresh).collect();
                self.tiles = rects.as_slice().into();
                self.tile_states = Arc::new(vec![TileState::Waiting; rects.len()]);
            }
            // Pictures for the image, not counted.
            ProgressUpdate::Subject(_) | ProgressUpdate::Faces(_) | ProgressUpdate::TilePreview(_) => {}
            &ProgressUpdate::Tile { index, done } => {
                if let Some(state) = Arc::make_mut(&mut self.tile_states).get_mut(index as usize) {
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

    /// Time left as a person would round it.
    pub fn time_left(&self, now: Instant) -> Option<TimeLeft> {
        let secs = self.remaining(now)?.as_secs_f32();
        Some(if secs < 10.0 {
            TimeLeft::FewSeconds
        } else if secs < 60.0 {
            TimeLeft::Seconds(((secs / 10.0).round() * 10.0) as u32)
        } else {
            TimeLeft::Minutes((secs / 60.0).ceil() as u32)
        })
    }

    /// "a few seconds left", "about 40 s left", "about 2 min left".
    pub fn remaining_text(&self, now: Instant) -> Option<String> {
        self.time_left(now).map(|t| match t {
            TimeLeft::FewSeconds => "a few seconds left".to_string(),
            TimeLeft::Seconds(s) => format!("about {s} s left"),
            TimeLeft::Minutes(m) => format!("about {m} min left"),
        })
    }

    /// Flattened `(current, total)` across the outer × inner space.
    /// `None` when indeterminate. Monotonic — every step that lands
    /// advances `current` and never resets, even at outer boundaries.
    pub fn flat_counter(&self) -> Option<(u32, u32)> {
        let (ic, it) = self.inner;
        match self.outer {
            None if it == 0 => None,
            None => Some((ic, it)),
            Some((_, 0)) => None,
            // Only the outer level counts (images of a batch).
            Some((oc, ot)) if it == 0 => Some((oc.min(ot), ot)),
            Some((oc, ot)) => {
                let total = ot.saturating_mul(it);
                Some((oc.saturating_mul(it).saturating_add(ic).min(total), total))
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
        (!parts.is_empty()).then(|| crate::gui::views::fmt::capitalized(&parts.join(" \u{b7} ")))
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

/// Every run under way, each with its progress, and which one the
/// capsule shows. Runs report to their own entry and end only it, so a
/// short run ending never hides a longer one that goes on. A Mutex rather
/// than atomics because the entries hold a heap `Cow`; the lock is held
/// for microseconds.
#[derive(Debug, Clone, Default)]
pub struct DispatchProgressSlot {
    runs: Arc<Mutex<Vec<LiveRun>>>,
    /// The number the next run takes.
    next: Arc<AtomicU64>,
}

#[derive(Debug)]
struct LiveRun {
    run: u64,
    progress: DispatchProgress,
    /// Started by itself (a face search, a depth measurement): shown only
    /// while no run the user started goes on.
    background: bool,
    /// Raised by a Cancel; the run's pipeline reads it through its
    /// `Progress` and stops at its next step.
    stop: Arc<AtomicBool>,
}

/// The listener `begin` hands a run, and its end. Clones share the run:
/// it ends at `end`, or when its last handle goes (its thread stopped by
/// a panic), whichever comes first.
#[derive(Clone)]
pub struct RunSink {
    live: Arc<RunToken>,
}

struct RunToken {
    slot: DispatchProgressSlot,
    run: u64,
    stop: Arc<AtomicBool>,
}

impl Drop for RunToken {
    fn drop(&mut self) {
        self.slot.remove(self.run);
    }
}

impl RunSink {
    fn run(&self) -> u64 {
        self.live.run
    }

    /// The run goes on, and no newer run of its kind works on its image:
    /// its pictures may still land.
    pub(crate) fn current(&self) -> bool {
        let runs = self.live.slot.lock();
        let Some(me) = runs.iter().find(|r| r.run == self.run()) else { return false };
        !runs.iter().any(|r| r.run > me.run && r.progress.kind == me.progress.kind && r.progress.on == me.progress.on)
    }

    /// The run's stop flag, for its `Progress` to read.
    pub(crate) fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.live.stop)
    }

    /// A Cancel reached the run.
    pub(crate) fn stopped(&self) -> bool {
        self.live.stop.load(Ordering::Acquire)
    }

    /// The run is over: its entry goes, the others stay.
    pub fn end(&self) {
        self.live.slot.remove(self.run());
    }

    /// Show `text` as the step, for wording only the app can give
    /// (which models are loading).
    pub fn set_step_text(&self, text: String) {
        self.live.slot.with_run(self.run(), |p| p.step_label = Cow::Owned(text));
    }
}

impl prunr_core::ProgressSink for RunSink {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        self.live.slot.with_run(self.run(), |p| p.apply(&update));
    }
}

impl DispatchProgressSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a run the user asked for and get the listener its pipeline
    /// reports to.
    pub fn begin(&self, initial: DispatchProgress) -> RunSink {
        self.add(initial, false)
    }

    /// Start a run that started by itself (a face search, a depth
    /// measurement): it shows only while nothing the user started runs,
    /// so it never hides a run whose Cancel the user may want.
    pub fn begin_background(&self, initial: DispatchProgress) -> RunSink {
        self.add(initial, true)
    }

    fn add(&self, progress: DispatchProgress, background: bool) -> RunSink {
        let run = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        let stop = Arc::new(AtomicBool::new(false));
        self.lock().push(LiveRun { run, progress, background, stop: Arc::clone(&stop) });
        RunSink { live: Arc::new(RunToken { slot: self.clone(), run, stop }) }
    }

    fn remove(&self, run: u64) {
        self.lock().retain(|r| r.run != run);
    }

    fn with_run(&self, run: u64, f: impl FnOnce(&mut DispatchProgress)) {
        if let Some(live) = self.lock().iter_mut().find(|r| r.run == run) {
            f(&mut live.progress);
        }
    }

    /// Show `progress` alone, with no run behind it; `None` shows none.
    #[cfg(test)]
    pub fn set(&self, progress: Option<DispatchProgress>) {
        let mut runs = self.lock();
        runs.clear();
        if let Some(progress) = progress {
            let run = self.next.fetch_add(1, Ordering::AcqRel) + 1;
            runs.push(LiveRun { run, progress, background: false, stop: Arc::default() });
        }
    }

    /// Stop every run of `kind` under way: each one's flag is raised and
    /// it shows as stopping until it ends.
    pub fn stop(&self, kind: ProgressKind) {
        self.stop_where(|r| r.progress.kind == kind);
    }

    /// Stop the runs of `kind` working on image `item`, and no other.
    pub fn stop_on(&self, kind: ProgressKind, item: u64) {
        self.stop_where(|r| r.progress.kind == kind && r.progress.on == Some(item));
    }

    fn stop_where(&self, which: impl Fn(&LiveRun) -> bool) {
        for live in self.lock().iter_mut().filter(|r| which(r)) {
            live.stop.store(true, Ordering::Release);
            live.progress.cancelling = true;
        }
    }

    /// The run the capsule shows; `None` when none is under way.
    pub fn read(&self) -> Option<DispatchProgress> {
        self.read_over(None)
    }

    /// The run the capsule shows over image `shown`: a run the user
    /// started before one that started by itself, then one working on
    /// `shown`, then the newest.
    pub fn read_over(&self, shown: Option<u64>) -> Option<DispatchProgress> {
        let runs = self.lock();
        let rank = |r: &LiveRun| (!r.background, shown.is_some() && r.progress.on == shown, r.run);
        runs.iter().max_by_key(|r| rank(r)).map(|r| r.progress.clone())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<LiveRun>> {
        self.runs.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch's progress shows over any image; a one-image run's only
    /// over its own, so switching images mid-run does not draw the run's
    /// tiles over the new one.
    #[test]
    fn a_runs_progress_shows_over_its_own_image_only() {
        let batch = DispatchProgress::new(ProgressKind::Seg, "x");
        assert!(batch.is_over(Some(1)) && batch.is_over(Some(2)) && batch.is_over(None));
        let run = DispatchProgress::new(ProgressKind::Upscale, "x").on(1);
        assert!(run.is_over(Some(1)));
        assert!(!run.is_over(Some(2)), "another image is shown");
        assert!(!run.is_over(None), "nothing is shown");
    }

    fn upscale(inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress { inner, ..DispatchProgress::new(ProgressKind::Upscale, "Upscaling") }
    }

    /// `outer` is (done, total), as the pipelines report it.
    fn sd_nested(outer: (u32, u32), inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress { outer: Some(outer), inner, ..DispatchProgress::new(ProgressKind::Inpaint, prunr_core::Step::Denoising.label()) }
    }

    #[test]
    fn a_batch_counted_only_at_the_outer_level_has_a_fraction() {
        let p = DispatchProgress { outer: Some((1, 4)), ..DispatchProgress::new(ProgressKind::Seg, "x") };
        assert_eq!(p.fraction(), Some(0.25));
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
        p.outer_unit = Some(Unit::Image);
        p.inner_unit = Some(Unit::Tile);
        assert_eq!(p.counter_text().as_deref(), Some("Image 2 of 3 \u{b7} tile 5 of 8"));
        assert_eq!(sd_nested((0, 1), (5, 8)).counter_text().as_deref(), Some("Step 5 of 8"));
        let one_image = DispatchProgress { outer: Some((0, 1)), ..DispatchProgress::new(ProgressKind::Seg, "x") };
        assert_eq!(one_image.counter_text(), None);
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

    /// A run that asks only when the slot is idle never takes it from the
    /// run showing, and shows once that run ends.
    /// A run started by itself (a face search, a depth measurement)
    /// shows only while nothing the user started runs, and a run the user
    /// starts shows over it.
    #[test]
    fn a_run_started_by_itself_shows_only_when_alone() {
        let slot = DispatchProgressSlot::new();
        let batch = slot.begin(upscale((1, 4)));
        let search = slot.begin_background(DispatchProgress::new(ProgressKind::Faces, "Finding faces"));
        assert_eq!(slot.read().map(|p| p.kind), Some(ProgressKind::Upscale), "the run started by hand stays shown");
        batch.end();
        assert_eq!(slot.read().map(|p| p.kind), Some(ProgressKind::Faces), "then the search shows");
        let _next = slot.begin(DispatchProgress::new(ProgressKind::Colorize, "Loading"));
        assert_eq!(slot.read().map(|p| p.kind), Some(ProgressKind::Colorize));
        drop(search);
    }

    /// A short run ending while a longer one goes on shows the longer one
    /// again, and the longer one's pictures go on landing meanwhile. One
    /// slot held the newest run only: a cut-out landing in a second
    /// blanked an erase's capsule and its tile previews for the rest of
    /// the erase.
    #[test]
    fn a_short_run_ending_shows_the_longer_one_again() {
        let slot = DispatchProgressSlot::new();
        let erase = slot.begin(DispatchProgress::new(ProgressKind::Inpaint, "Erasing").on(1));
        let cut = slot.begin(DispatchProgress::new(ProgressKind::Seg, "Finding the subject").on(2));
        assert!(erase.current(), "another kind of run on another image leaves it current");
        assert_eq!(slot.read_over(Some(1)).map(|p| p.kind), Some(ProgressKind::Inpaint), "over its own image, the erase");
        assert_eq!(slot.read_over(Some(2)).map(|p| p.kind), Some(ProgressKind::Seg));
        cut.end();
        assert_eq!(slot.read().map(|p| p.kind), Some(ProgressKind::Inpaint), "the erase shows again");

        let newer = slot.begin(DispatchProgress::new(ProgressKind::Inpaint, "Erasing").on(1));
        assert!(!erase.current() && newer.current(), "a newer run of its kind on its image replaces it");
    }

    /// A run whose every handle goes without ending it (its thread
    /// stopped by a panic) ends anyway: nothing stays up for good.
    #[test]
    fn a_run_whose_handles_all_go_ends() {
        let slot = DispatchProgressSlot::new();
        let run = slot.begin(upscale((0, 4)));
        let copy = run.clone();
        drop(run);
        assert!(slot.read().is_some(), "a handle is left");
        drop(copy);
        assert!(slot.read().is_none());
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

    /// A Cancel stops every run of its kind, and no other; one for an
    /// image stops only the runs of its kind on that image.
    #[test]
    fn a_cancel_stops_its_kinds_runs() {
        let slot = DispatchProgressSlot::new();
        let up = slot.begin(upscale((1, 10)));
        let faces = slot.begin_background(DispatchProgress::new(ProgressKind::Faces, "Finding faces"));
        slot.stop(ProgressKind::Faces);
        assert!(faces.stopped() && !up.stopped());
        assert!(!slot.read().expect("the upscale").cancelling);
        slot.stop(ProgressKind::Upscale);
        assert!(up.stopped() && slot.read().expect("the upscale").cancelling);

        let a = slot.begin(DispatchProgress::new(ProgressKind::Inpaint, "Erasing").on(1));
        let b = slot.begin(DispatchProgress::new(ProgressKind::Inpaint, "Erasing").on(2));
        slot.stop_on(ProgressKind::Inpaint, 1);
        assert!(a.stopped() && !b.stopped());
    }

    #[test]
    fn the_tile_map_follows_tile_reports() {
        use prunr_core::{ProgressUpdate, TileRect};
        let mut p = upscale((0, 0));
        let half = |x| TileRect { x, y: 0.0, w: 0.5, h: 1.0 };
        p.apply(&ProgressUpdate::Tiles(vec![half(0.0), half(0.5)]));
        assert_eq!(*p.tile_states, [TileState::Waiting; 2]);
        p.apply(&ProgressUpdate::Tile { index: 0, done: false });
        p.apply(&ProgressUpdate::Tile { index: 7, done: true });
        assert_eq!(*p.tile_states, [TileState::Running, TileState::Waiting]);
        p.apply(&ProgressUpdate::Tile { index: 0, done: true });
        assert_eq!(*p.tile_states, [TileState::Done, TileState::Waiting]);
        p.apply(&ProgressUpdate::Tiles(vec![half(0.5)]));
        assert_eq!(*p.tile_states, [TileState::Waiting], "a new layout starts over");
        assert_eq!(*p.finished, [half(0.0)], "what the last one finished stays done");
        p.apply(&ProgressUpdate::Tile { index: 0, done: true });
        p.apply(&ProgressUpdate::Tiles(vec![half(0.0)]));
        assert_eq!(*p.finished, [half(0.0), half(0.5)]);
    }

    /// A layout laid again for each pass of a run keeps one copy of each
    /// finished cell, however many passes.
    #[test]
    fn a_relaid_layout_keeps_one_copy_of_each_cell() {
        use prunr_core::{ProgressUpdate, TileRect};
        let mut p = upscale((0, 0));
        let layout = vec![TileRect { x: 0.0, y: 0.0, w: 0.5, h: 1.0 }, TileRect { x: 0.5, y: 0.0, w: 0.5, h: 1.0 }];
        for _ in 0..30 {
            p.apply(&ProgressUpdate::Tiles(layout.clone()));
            for index in 0..2 {
                p.apply(&ProgressUpdate::Tile { index, done: true });
            }
        }
        p.apply(&ProgressUpdate::Tiles(layout.clone()));
        assert_eq!(*p.finished, *layout);
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

    /// Each run's Cancel stops that run. A face search and an erase run
    /// once had a Cancel that did nothing; Seg must stop the cut-outs, not
    /// an erase. A depth run is too short to cancel.
    #[test]
    fn each_run_kind_cancels_its_own_run() {
        for (kind, target) in [
            (ProgressKind::Seg, Some(CancelTarget::Cuts)),
            (ProgressKind::Inpaint, Some(CancelTarget::Inpaint)),
            (ProgressKind::Upscale, Some(CancelTarget::Upscale)),
            (ProgressKind::Colorize, Some(CancelTarget::Colorize)),
            (ProgressKind::Faces, Some(CancelTarget::Faces)),
            (ProgressKind::Depth, None),
        ] {
            assert_eq!(cancel_target_for(kind), target, "{kind:?}");
        }
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
        p.apply(&ProgressUpdate::Outer { done: 1, total: 2, unit: Unit::Crop });
        assert_eq!((p.step_label.as_ref(), p.inner, p.outer), ("Upscaling", (3, 48), Some((1, 2))));
        p.cancelling = true;
        p.apply(&ProgressUpdate::Step(Step::Finishing));
        assert!(p.cancelling, "a late step must not hide the cancel");
    }

}
