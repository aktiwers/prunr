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
