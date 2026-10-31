//! How a batch runs on a worker, in the app and the CLI alike: how much
//! work to send, when silence is a hang, what a crash does.

use std::time::Duration;

/// Silence with work in flight before the worker counts as hung; real
/// batches report every few hundred ms.
pub const HANG_TIMEOUT: Duration = Duration::from_secs(60);

/// How much more work to send now. Up to the job limit; the memory pause
/// holds new work back only while some is running, as the worker reports
/// its memory when work ends: with none running the pause would never lift.
pub fn room_to_send(working: usize, max_jobs: usize, paused: bool) -> usize {
    if paused && working > 0 {
        return 0;
    }
    max_jobs.saturating_sub(working)
}

/// The free RAM that stops a worker with work in flight, as `read_free`
/// reads it (only then): below `memory::STOP_BELOW_FREE_MB`. No reading:
/// it runs on.
pub fn low_free_memory(in_flight_count: usize, read_free: impl FnOnce() -> Option<u64>) -> Option<u64> {
    const LINE: u64 = prunr_core::memory::STOP_BELOW_FREE_MB * 1024 * 1024;
    if in_flight_count == 0 {
        return None;
    }
    read_free().filter(|&free| free < LINE)
}

/// Alive but silent too long; the event loop treats it as a crash.
pub fn is_stalled(last_event_age: Duration, in_flight_count: usize, hang_timeout: Duration) -> bool {
    in_flight_count > 0 && last_event_age > hang_timeout
}

/// What a worker crash does to the images it was running.
#[derive(Debug, PartialEq)]
pub enum OnCrash {
    /// Run them again with fewer jobs: one of them, or running them
    /// together, crashed it.
    RetryWith(usize),
    /// At one job the image it ran is the cause: it fails, the batch
    /// goes on with a new worker.
    FailInFlight,
}

pub fn on_crash(jobs: usize) -> OnCrash {
    if jobs <= 1 { OnCrash::FailInFlight } else { OnCrash::RetryWith(jobs / 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A crash halves the jobs and runs the images again; at one job the
    /// image it ran is the cause and fails alone, not the whole batch.
    #[test]
    fn a_crash_at_one_job_fails_only_the_image_it_ran() {
        assert_eq!(on_crash(4), OnCrash::RetryWith(2));
        assert_eq!(on_crash(2), OnCrash::RetryWith(1));
        assert_eq!(on_crash(1), OnCrash::FailInFlight);
    }

    /// Work goes out up to the job limit; the memory pause holds it back
    /// only while some runs, or a batch whose running images all failed
    /// would wait for a memory report that never comes.
    #[test]
    fn work_is_sent_up_to_the_limit_and_the_pause_needs_running_work() {
        assert_eq!(room_to_send(0, 2, false), 2, "the batch's first images");
        assert_eq!(room_to_send(1, 2, false), 1, "one finished or failed: one more");
        assert_eq!(room_to_send(2, 2, false), 0);
        assert_eq!(room_to_send(1, 2, true), 0, "paused while one runs");
        assert_eq!(room_to_send(0, 2, true), 2, "nothing running: the pause cannot lift itself");
        assert_eq!(room_to_send(3, 2, false), 0, "fewer jobs after a crash");
    }

    /// A run stops when free memory falls under the line while it works;
    /// an idle worker, or no reading, is left alone.
    #[test]
    fn a_run_stops_only_when_memory_runs_low_under_it() {
        let line = prunr_core::memory::STOP_BELOW_FREE_MB * 1024 * 1024;
        assert_eq!(low_free_memory(1, || Some(700 << 20)), Some(700 << 20));
        assert_eq!(low_free_memory(1, || Some(line - 1)), Some(line - 1));
        assert_eq!(low_free_memory(1, || Some(line)), None);
        assert_eq!(low_free_memory(0, || panic!("read only while work runs")), None);
        assert_eq!(low_free_memory(1, || None), None, "no reading");
        assert_eq!(low_free_memory(2, || Some(8 << 30)), None);
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
