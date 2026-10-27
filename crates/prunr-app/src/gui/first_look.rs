//! When the run of the image on screen shows a first look at its subject,
//! and how long each cut-out model takes per image here, which decides it.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use prunr_core::{LineMode, ModelKind, ProgressUpdate, Step};

/// The image on screen whose run shows a first look, if any: while
/// animations are on, for a cut-out by a model the first look pays for at
/// the time it is `expected` to take per image (`None`: not timed yet).
/// The worker settles whether a run is the one on screen as it starts.
pub(crate) fn first_look_for(
    shown: Option<u64>,
    model: Option<ModelKind>,
    line_mode: LineMode,
    animations: bool,
    expected: impl FnOnce(ModelKind) -> Option<Duration>,
) -> Option<u64> {
    let model = model.filter(|_| animations && line_mode != LineMode::EdgesOnly)?;
    shown.filter(|_| prunr_core::subject::first_look_pays(model, || expected(model)))
}

/// How long each cut-out model took per image this session, by the
/// provider its session landed on: from its subject search starting to
/// the subject found, the stretch a first look shows through. Kept in
/// memory only; a new session times its first run afresh.
#[derive(Default)]
pub(crate) struct CutTimes {
    took: Vec<(ModelKind, String, Duration)>,
    searching: HashMap<u64, Search>,
}

struct Search {
    since: Instant,
    took: Option<Duration>,
}

impl CutTimes {
    /// A report of `item_id`'s run, arrived `at`. A first look reports a
    /// search of its own first; the run's own comes last and is kept. A
    /// step repeated while its search is open is the worker saying the
    /// run is alive.
    pub(crate) fn note(&mut self, item_id: u64, update: &ProgressUpdate, at: Instant) {
        match update {
            ProgressUpdate::Step(Step::FindingSubject) => {
                let open = self.searching.get(&item_id).is_some_and(|s| s.took.is_none());
                if !open {
                    self.searching.insert(item_id, Search { since: at, took: None });
                }
            }
            ProgressUpdate::Subject(_) => {
                if let Some(search) = self.searching.get_mut(&item_id) {
                    search.took = Some(at.saturating_duration_since(search.since));
                }
            }
            _ => {}
        }
    }

    /// `item_id`'s run ended, by `landed`'s model with its result landing
    /// on its provider (`None`: no result): its time is what the next run
    /// expects.
    pub(crate) fn ended(&mut self, item_id: u64, landed: Option<(ModelKind, &str)>) {
        let took = self.searching.remove(&item_id).and_then(|s| s.took);
        let (Some(took), Some((model, provider))) = (took, landed.filter(|(_, p)| !p.is_empty())) else { return };
        match self.took.iter_mut().find(|(m, p, _)| *m == model && p == provider) {
            Some(entry) => entry.2 = took,
            None => self.took.push((model, provider.to_string(), took)),
        }
    }

    /// What a run of `model` on `provider` is expected to take per image:
    /// the last one's time, `None` before one has landed.
    pub(crate) fn expected(&self, model: ModelKind, provider: &str) -> Option<Duration> {
        self.took.iter().find(|(m, p, _)| *m == model && p == provider).map(|e| e.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprocess::first_look::FirstLook;
    use prunr_core::subject::FIRST_LOOK_AFTER;
    use ModelKind::{BiRefNetLite, Silueta, U2net};

    /// What happens in a batch, in order.
    #[derive(Clone, Copy, Debug)]
    enum At {
        /// The user puts this image on screen.
        Show(u64),
        /// This image's run starts.
        Start(u64),
        /// This image's run reaches its first look, if it takes one.
        Look(u64),
    }
    use At::{Look, Show, Start};

    struct Row {
        what: &'static str,
        model: ModelKind,
        line_mode: LineMode,
        animations: bool,
        expected: Option<Duration>,
        batch: &'static [At],
        /// The images Silueta runs for.
        looks: &'static [u64],
    }

    const OVER: Option<Duration> = Some(Duration::from_secs(5));
    const UNDER: Option<Duration> = Some(Duration::from_millis(400));
    /// Images 1 to 4 in a row, image 2 on screen throughout.
    const STEADY: &[At] = &[Show(2), Start(1), Look(1), Start(2), Look(2), Start(3), Look(3), Start(4), Look(4)];

    fn row(what: &'static str, model: ModelKind, expected: Option<Duration>, batch: &'static [At], looks: &'static [u64]) -> Row {
        Row { what, model, line_mode: LineMode::Off, animations: true, expected, batch, looks }
    }

    /// Run a row's batch: the app sets the image on screen whose run
    /// would show a first look, the worker settles it as each run starts.
    fn looked(r: &Row) -> Vec<u64> {
        let worker = FirstLook::default();
        let mut pending = HashMap::new();
        let mut looked = Vec::new();
        for &at in r.batch {
            match at {
                Show(id) => worker.show_for(first_look_for(Some(id), Some(r.model), r.line_mode, r.animations, |_| r.expected)),
                Start(id) => {
                    if let Some(look) = worker.for_run(id, r.model, 1) {
                        pending.insert(id, look);
                    }
                }
                Look(id) => {
                    if pending.remove(&id).is_some_and(|l| l.is_current()) {
                        looked.push(id);
                    }
                }
            }
        }
        looked
    }

    /// Which images run Silueta: only the one on screen as its run
    /// starts, for BiRefNet or U2Net with animations on, untimed or slower
    /// than the threshold here.
    #[test]
    fn the_first_look_decision_table() {
        let rows = [
            row("not timed yet: the image on screen looks", BiRefNetLite, None, STEADY, &[2]),
            row("slower than the threshold", BiRefNetLite, OVER, STEADY, &[2]),
            row("U2Net slower than the threshold", U2net, OVER, STEADY, &[2]),
            row("U2Net not timed yet", U2net, None, STEADY, &[2]),
            row("quicker than the threshold: a fast accelerator", BiRefNetLite, UNDER, STEADY, &[]),
            row("exactly the threshold", U2net, Some(FIRST_LOOK_AFTER), STEADY, &[]),
            row("Silueta itself never", Silueta, None, STEADY, &[]),
            Row { animations: false, ..row("animations off", BiRefNetLite, None, STEADY, &[]) },
            Row { animations: false, ..row("animations off, slow", U2net, OVER, STEADY, &[]) },
            Row { line_mode: LineMode::EdgesOnly, ..row("lines over the whole photo: nothing to cut", BiRefNetLite, None, STEADY, &[]) },
            row("a batch of N where you never switch looks once",
                BiRefNetLite, None, &[Show(3), Start(1), Look(1), Start(2), Look(2), Start(3), Look(3), Start(4), Look(4), Start(5), Look(5)], &[3]),
            row("switched to before its run starts: it looks when it starts",
                BiRefNetLite, None, &[Show(2), Start(1), Look(1), Show(3), Start(2), Look(2), Start(3), Look(3)], &[3]),
            row("switched to while already running: no look midway",
                BiRefNetLite, None, &[Show(1), Start(2), Show(2), Look(2), Start(3), Look(3)], &[]),
            row("switched away before its look: dropped as stale, the run goes on",
                BiRefNetLite, None, &[Show(2), Start(2), Show(4), Look(2), Start(3), Look(3)], &[]),
            row("switched away and back before its run starts",
                BiRefNetLite, None, &[Show(2), Show(4), Show(2), Start(2), Look(2)], &[2]),
            row("an image off screen never looks", U2net, None, &[Show(9), Start(1), Look(1), Start(2), Look(2)], &[]),
        ];
        for r in &rows {
            assert_eq!(looked(r), r.looks, "{}", r.what);
        }
    }

    /// Times run from the run's own subject search to its subject; a
    /// first look's search before it is not counted, and only a run that
    /// landed a result, on the provider it landed on, is kept.
    #[test]
    fn a_runs_time_is_its_own_search_on_the_provider_it_landed_on() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let subject = || ProgressUpdate::Subject(prunr_core::SubjectHint { width: 1, height: 1, coverage: vec![255] });
        let mut times = CutTimes::default();
        times.note(1, &ProgressUpdate::Step(Step::ReadingImage), at(0));
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(100));
        times.note(1, &subject(), at(900));
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(1000));
        times.note(1, &subject(), at(7000));
        times.ended(1, Some((BiRefNetLite, "CPU")));
        assert_eq!(times.expected(BiRefNetLite, "CPU"), Some(Duration::from_secs(6)), "the first look's 0.8 s is not the model's");
        assert_eq!(times.expected(BiRefNetLite, "DirectML"), None, "another provider is not timed yet");
        assert_eq!(times.expected(U2net, "CPU"), None);

        times.note(2, &ProgressUpdate::Step(Step::FindingSubject), at(0));
        times.note(2, &subject(), at(300));
        times.ended(2, Some((BiRefNetLite, "DirectML")));
        times.note(3, &ProgressUpdate::Step(Step::FindingSubject), at(0));
        times.note(3, &subject(), at(50));
        times.ended(3, None);
        times.note(4, &ProgressUpdate::Step(Step::FindingSubject), at(0));
        times.ended(4, Some((BiRefNetLite, "CPU")));
        assert_eq!(times.expected(BiRefNetLite, "DirectML"), Some(Duration::from_millis(300)));
        assert_eq!(times.expected(BiRefNetLite, "CPU"), Some(Duration::from_secs(6)), "a failed run, or one cut off before its subject, keeps the last time");
        assert!(times.searching.is_empty(), "nothing held for ended runs");
    }

    /// The worker repeats a long run's step every few seconds to show it
    /// is alive; a repeat is the same search, not a new one.
    #[test]
    fn a_repeated_step_does_not_restart_the_search() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let subject = || ProgressUpdate::Subject(prunr_core::SubjectHint { width: 1, height: 1, coverage: vec![255] });
        let mut times = CutTimes::default();
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(0));
        times.note(1, &subject(), at(500));
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(1000));
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(11_000));
        times.note(1, &ProgressUpdate::Step(Step::FindingSubject), at(21_000));
        times.note(1, &subject(), at(21_500));
        times.ended(1, Some((BiRefNetLite, "CPU")));
        assert_eq!(times.expected(BiRefNetLite, "CPU"), Some(Duration::from_millis(20_500)));
    }
}
