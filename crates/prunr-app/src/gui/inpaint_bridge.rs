//! Dedicated bridge thread for the SD inpaint subprocess. It never serves
//! seg/edge work, so the engine pool is skipped at Init. The subprocess is
//! spawned lazily on the first SD dispatch and dropped after
//! `IDLE_RELEASE` idle, unless kept loaded, to release the ~5 GB bundle.
//! The LaMa family stays on the in-process rayon path.

use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::subprocess::manager::SubprocessManager;
use crate::subprocess::protocol::{InpaintJob, SubprocessEvent};
use prunr_core::IDLE_RELEASE;


/// Whether the idle eraser subprocess goes now: nothing in flight, idle
/// past the window, and the user did not ask to keep it loaded.
fn idle_release_due(no_in_flight: bool, idle: Duration, keep_loaded: bool) -> bool {
    no_in_flight && idle > IDLE_RELEASE && !keep_loaded
}

/// Below this much free RAM the watchdog force-kills the worker.
/// 1 GB sits below Linux's aggressive-reclaim band and above macOS's
/// memory-pressure warning, so it kicks in before swap-thrashing on either.
const MEMORY_PRESSURE_THRESHOLD_BYTES: u64 = 1024 * 1024 * 1024;

/// How long a released subprocess gets to exit before it is killed.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);

/// Parent → bridge messages.
pub enum InpaintBridgeMsg {
    /// One SD inpaint stroke; image + mask are temp-file PNGs the worker
    /// decodes. Its generation round-trips on `Done` so the GUI can drop a
    /// stroke a fresher one superseded.
    Dispatch(InpaintJob),
    /// Kill the subprocess; every in-flight stroke ends Cancelled.
    Cancel,
    /// Release the subprocess regardless of in-flight state: unlike Cancel,
    /// an idle warm one is killed too, freeing its ~200 MB. Sent on a switch
    /// to a non-SD model.
    Release,
}

/// Bridge → parent results: the SD subset of `SubprocessEvent`.
pub enum InpaintBridgeResult {
    Report { item_id: u64, update: prunr_core::ProgressUpdate },
    Done { item_id: u64, generation: u64, rgba_path: std::path::PathBuf, width: u32, height: u32, fill_scale: f32 },
    /// The stroke `generation` on `item_id`, and any before it, ended
    /// without a result.
    Error { item_id: u64, generation: u64, error: String },
}

/// The thread runs until the parent drops the sender.
pub fn spawn_inpaint_bridge() -> (
    mpsc::Sender<InpaintBridgeMsg>,
    mpsc::Receiver<InpaintBridgeResult>,
) {
    let (msg_tx, msg_rx) = mpsc::channel::<InpaintBridgeMsg>();
    let (res_tx, res_rx) = mpsc::channel::<InpaintBridgeResult>();

    std::thread::Builder::new()
        .name("prunr-inpaint-bridge".into())
        .spawn(move || run(msg_rx, res_tx))
        .expect("failed to spawn inpaint bridge");

    (msg_tx, res_rx)
}

/// An image and one of its strokes' generations.
type Stroke = (u64, u64);
/// Strokes sent or queued and not yet finished. Ordered, so one image's
/// strokes sit together.
type Strokes = BTreeSet<Stroke>;

/// Spawn lifecycle. `Spawning` holds buffered dispatches until the worker
/// reports Ready; the spawn runs on a helper thread polled with
/// `try_recv`, so a Cancel in the (up to 30 s) pre-Ready window doesn't
/// block the bridge.
enum SubState {
    Idle,
    Spawning {
        rx: mpsc::Receiver<Result<(SubprocessManager, String), String>>,
        /// Strokes dispatched mid-spawn, sent in order once `Ready` lands.
        pending: Vec<InpaintJob>,
        /// A Cancel arrived mid-spawn (already drained and toasted): kill the
        /// subprocess when it lands instead of promoting it.
        cancelled: bool,
    },
    Ready(SubprocessManager),
}

fn run(msg_rx: mpsc::Receiver<InpaintBridgeMsg>, res_tx: mpsc::Sender<InpaintBridgeResult>) {
    let mut sub_state = SubState::Idle;
    let mut last_used = Instant::now();
    let mut keep_loaded = false;
    let mut strokes = Strokes::new();

    loop {
        let received = match look_at_worker_every(&sub_state) {
            Some(wait) => msg_rx.recv_timeout(wait),
            None => msg_rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        };
        let first = match received {
            Ok(msg) => Some(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            // A sender gone mid-drain lands here on the next tick.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                shutdown_ready(&mut sub_state);
                return;
            }
        };
        for msg in first.into_iter().chain(std::iter::from_fn(|| msg_rx.try_recv().ok())) {
            match msg {
                InpaintBridgeMsg::Dispatch(job) => {
                    last_used = Instant::now();
                    keep_loaded = job.sd_req.as_ref().is_some_and(|r| r.tuning.keep_loaded);
                    strokes.insert((job.item_id, job.generation));
                    handle_dispatch(&mut sub_state, &mut strokes, &res_tx, job);
                }
                InpaintBridgeMsg::Cancel => {
                    // SD's UNet `run()` can't be interrupted mid-step, so only a kill
                    // frees RAM in time; it ends every in-flight stroke.
                    handle_cancel(&mut sub_state, &mut strokes, &res_tx);
                }
                InpaintBridgeMsg::Release => {
                    handle_release(&mut sub_state, &mut strokes, &res_tx);
                }
            }
        }

        if matches!(sub_state, SubState::Spawning { .. }) {
            poll_spawn_result(&mut sub_state, &mut strokes, &res_tx);
        }

        // Watchdog for `needs` underestimates and mid-load RAM shifts
        // the pre-flight gate can't see; only once Ready, since a spawning
        // worker has loaded nothing yet.
        if !strokes.is_empty() && matches!(sub_state, SubState::Ready(_)) {
            let free = crate::hardware::available_ram_bytes_throttled();
            if free > 0 && free < MEMORY_PRESSURE_THRESHOLD_BYTES {
                tracing::warn!(
                    free_mb = free / (1024 * 1024),
                    threshold_mb = MEMORY_PRESSURE_THRESHOLD_BYTES / (1024 * 1024),
                    in_flight = strokes.len(),
                    "memory-pressure abort: killing inpaint subprocess",
                );
                kill_ready_and_drain(
                    &mut sub_state, &mut strokes, &res_tx,
                    crate::subprocess::protocol::MEMORY_PRESSURE_ABORT_MSG,
                    "eraser worker killed by the memory-pressure watchdog",
                );
            }
        }

        if let SubState::Ready(s) = &mut sub_state {
            for evt in s.poll_events() {
                match evt {
                    SubprocessEvent::Report { item_id, update } => {
                        let _ = res_tx.send(InpaintBridgeResult::Report { item_id, update });
                    }
                    SubprocessEvent::InpaintDone { item_id, generation, rgba_path, width, height, fill_scale } => {
                        last_used = Instant::now();
                        strokes.remove(&(item_id, generation));
                        let _ = res_tx.send(InpaintBridgeResult::Done { item_id, generation, rgba_path, width, height, fill_scale });
                    }
                    SubprocessEvent::InpaintError { item_id, generation, error } => {
                        last_used = Instant::now();
                        strokes.remove(&(item_id, generation));
                        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, generation, error });
                    }
                    // An inpaint-only subprocess shouldn't emit seg events; ignore one
                    // rather than misroute it.
                    _ => {}
                }
            }
        }
        end_if_dead(&mut sub_state, &mut strokes, &res_tx);

        if matches!(sub_state, SubState::Ready(_))
            && idle_release_due(strokes.is_empty(), last_used.elapsed(), keep_loaded)
        {
            shutdown_ready(&mut sub_state);
            tracing::info!("inpaint subprocess released after idle window");
        }
    }
}

/// How long the bridge waits for a message before it looks at its worker
/// again: a message wakes it at once. With no worker, nothing changes
/// until a message comes.
fn look_at_worker_every(sub_state: &SubState) -> Option<Duration> {
    (!matches!(sub_state, SubState::Idle)).then_some(Duration::from_millis(100))
}

/// Idle: spawn off-thread and queue. Spawning: queue. Ready: send.
fn handle_dispatch(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    params: InpaintJob,
) {
    match sub_state {
        SubState::Ready(s) => {
            send_or_fail(s, params, strokes, res_tx);
        }
        SubState::Spawning { pending, .. } => {
            pending.push(params);
        }
        SubState::Idle => {
            *sub_state = kick_spawn(vec![params]);
        }
    }
}

/// Idle: no-op. Ready: kill. Spawning: toast at once and kill the
/// subprocess on arrival, so the user gets feedback in <100 ms instead of
/// after the up-to-30 s pre-Ready timeout.
fn handle_cancel(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    match sub_state {
        SubState::Idle => {}
        SubState::Spawning { cancelled, .. } => {
            if strokes.is_empty() {
                return;
            }
            *cancelled = true;
            fail_inflight(strokes, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
            tracing::info!("inpaint Cancel during spawn — pending result will be killed");
        }
        SubState::Ready(_) => {
            kill_ready_and_drain(
                sub_state, strokes, res_tx,
                crate::subprocess::protocol::CANCELLED_ERR_MSG,
                "eraser worker killed by Cancel",
            );
        }
    }
}

/// Unlike `handle_cancel`, an idle Ready subprocess is killed too (freeing
/// its ~200 MB is the point); in-flight strokes end with `CANCELLED_ERR_MSG`.
fn handle_release(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    match std::mem::replace(sub_state, SubState::Idle) {
        SubState::Idle => {}
        SubState::Spawning { pending: _, cancelled: _, rx } => {
            // Spawn is still in flight and the helper thread can't be
            // killed: stay Spawning with `cancelled` set so the
            // arriving subprocess is killed, and drain inflight.
            *sub_state = SubState::Spawning { rx, pending: Vec::new(), cancelled: true };
            fail_inflight(strokes, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
            tracing::info!("inpaint Release during spawn — pending result will be killed");
        }
        SubState::Ready(mut s) => {
            s.kill();
            tracing::info!("inpaint subprocess released on model switch");
            fail_inflight(strokes, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
        }
    }
}

/// Spawn on a one-shot helper thread so the bridge stays responsive to
/// Cancel during the Init handshake.
fn kick_spawn(pending: Vec<InpaintJob>) -> SubState {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("prunr-inpaint-spawn".into())
        .spawn(move || {
            let started = Instant::now();
            let result = SubprocessManager::spawn_inpaint_only();
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                ok = result.is_ok(),
                "inpaint subprocess spawn finished",
            );
            let _ = tx.send(result);
        })
        .expect("failed to spawn inpaint-spawn helper");
    SubState::Spawning { rx, pending, cancelled: false }
}

/// Non-blocking. Ok goes Ready and replays buffered dispatches, or, after
/// a mid-spawn Cancel, kills the subprocess and goes Idle; Err fails every
/// queued dispatch.
fn poll_spawn_result(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    let SubState::Spawning { rx, .. } = sub_state else { return };
    let result = match rx.try_recv() {
        Ok(r) => r,
        Err(mpsc::TryRecvError::Empty) => return,
        Err(mpsc::TryRecvError::Disconnected) => {
            // Helper thread panicked. Treat as spawn error.
            Err("inpaint spawn helper thread died".to_string())
        }
    };
    let SubState::Spawning { pending, cancelled, .. } =
        std::mem::replace(sub_state, SubState::Idle) else { return };
    match (result, cancelled) {
        (Ok((mut mgr, provider)), true) => {
            tracing::info!(provider = %provider,
                "inpaint spawn finished but Cancel had arrived — killing subprocess");
            mgr.kill();
        }
        (Ok((mut mgr, provider)), false) => {
            tracing::info!(provider = %provider, "inpaint subprocess ready");
            for params in pending {
                send_or_fail(&mut mgr, params, strokes, res_tx);
            }
            *sub_state = SubState::Ready(mgr);
        }
        (Err(e), true) => {
            // Spawn failed and user already got "Erase cancelled" —
            // don't double-toast with a spawn error.
            tracing::warn!(%e, "inpaint spawn failed (also cancelled by user)");
        }
        (Err(e), false) => {
            tracing::error!(%e, "inpaint subprocess spawn failed");
            // Every stroke in flight is still in `pending`: none was sent.
            drop(pending);
            fail_inflight(strokes, res_tx, &format!("Could not start the eraser: {e}"));
        }
    }
}

/// SIGKILL the Ready subprocess, end every in-flight stroke with
/// `error_msg`, and go Idle. No-op with nothing in flight.
fn kill_ready_and_drain(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    error_msg: &'static str,
    log_msg: &'static str,
) {
    if strokes.is_empty() {
        return;
    }
    if let SubState::Ready(mut s) = std::mem::replace(sub_state, SubState::Idle) {
        s.kill();
        tracing::info!("{log_msg}");
    }
    fail_inflight(strokes, res_tx, error_msg);
}

/// Send one stroke; one the subprocess won't take ends with its error.
fn send_or_fail(
    mgr: &mut SubprocessManager,
    job: InpaintJob,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    let stroke = (job.item_id, job.generation);
    if let Err(error) = mgr.send_inpaint(job) {
        let _ = res_tx.send(InpaintBridgeResult::Error { item_id: stroke.0, generation: stroke.1, error });
        strokes.remove(&stroke);
        remove_stroke_files(crate::subprocess::protocol::ipc_temp_dir(), stroke);
    }
}

/// A worker that died (killed for memory, crashed) ends the strokes it
/// had, with why, and the next stroke starts a new one. Its last events
/// are drained first, so a fill it finished still lands.
fn end_if_dead(
    sub_state: &mut SubState,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    let SubState::Ready(s) = sub_state else { return };
    let Some(reason) = s.died() else { return };
    tracing::warn!(%reason, in_flight = strokes.len(), "eraser worker ended");
    *sub_state = SubState::Idle;
    let why = match reason {
        crate::subprocess::manager::Crash::OutOfMemory => "The eraser ran out of memory. Try a lighter eraser or a smaller region.",
        _ => "The eraser stopped unexpectedly. Erase again.",
    };
    fail_inflight(strokes, res_tx, why);
}

/// The files a stroke that will never finish leaves: its source and mask,
/// and the result the worker may have begun. On Linux they sit in RAM
/// until quit.
fn remove_stroke_files(ipc_dir: &std::path::Path, (item_id, generation): Stroke) {
    use crate::subprocess::protocol::IpcKind;
    for kind in [IpcKind::InpaintImg, IpcKind::InpaintMask, IpcKind::InpaintOut] {
        let _ = std::fs::remove_file(kind.path_for_generation(ipc_dir, item_id, generation));
    }
}

/// End every in-flight stroke with `error`, once per image, and remove
/// their files.
fn fail_inflight(strokes: &mut Strokes, res_tx: &mpsc::Sender<InpaintBridgeResult>, error: &str) {
    end_strokes(crate::subprocess::protocol::ipc_temp_dir(), strokes, res_tx, error);
}

fn end_strokes(
    ipc_dir: &std::path::Path,
    strokes: &mut Strokes,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    error: &str,
) {
    let mut last_item = None;
    // Newest first, so the one error per image names its latest stroke.
    for stroke in std::mem::take(strokes).into_iter().rev() {
        remove_stroke_files(ipc_dir, stroke);
        if last_item.replace(stroke.0) != Some(stroke.0) {
            let _ = res_tx.send(InpaintBridgeResult::Error { item_id: stroke.0, generation: stroke.1, error: error.into() });
        }
    }
}

/// Shut a Ready subprocess down gracefully and go Idle.
fn shutdown_ready(sub_state: &mut SubState) {
    if let SubState::Ready(mut s) = std::mem::replace(sub_state, SubState::Idle) {
        let _ = s.shutdown_with_timeout(RELEASE_TIMEOUT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no eraser worker the bridge sleeps until it is asked for
    /// something, not ten times a second all session.
    #[test]
    fn the_bridge_sleeps_while_no_eraser_runs() {
        assert_eq!(look_at_worker_every(&SubState::Idle), None);
        let (_tx, rx) = mpsc::channel();
        let spawning = SubState::Spawning { rx, pending: Vec::new(), cancelled: false };
        assert!(look_at_worker_every(&spawning).is_some(), "a starting worker is looked at");
    }

    /// Every way a stroke ends without a result (Cancel, the memory
    /// watchdog, a model switch, a dead or unstartable worker) goes through
    /// `end_strokes`: each stroke's files go, two strokes on one image
    /// included, one error per image, and a finished stroke's result and
    /// others' files stay.
    #[test]
    fn ended_strokes_leave_no_files_and_spare_the_rest() {
        use crate::subprocess::protocol::IpcKind;
        let dir = tempfile::tempdir().unwrap();
        let ipc = dir.path();
        let files = |item, generation| {
            [IpcKind::InpaintImg, IpcKind::InpaintMask, IpcKind::InpaintOut]
                .map(|kind| kind.path_for_generation(ipc, item, generation))
        };
        let ended: Vec<_> = [(1, 3), (1, 4), (2, 1)].into_iter().flat_map(|(i, g)| files(i, g)).collect();
        let kept: Vec<_> = files(1, 2).into_iter().chain([IpcKind::Result.path_for(ipc, 1)]).collect();
        for path in ended.iter().chain(&kept) {
            std::fs::write(path, b"x").unwrap();
        }
        let mut strokes = Strokes::from([(1, 3), (1, 4), (2, 1)]);
        let (tx, rx) = mpsc::channel();
        end_strokes(ipc, &mut strokes, &tx, "stopped");
        assert!(strokes.is_empty());
        assert!(ended.iter().all(|p| !p.exists()), "every ended stroke's files are gone");
        assert!(kept.iter().all(|p| p.exists()), "a finished stroke's and others' files stay");
        let ended_items: Vec<(u64, u64)> = rx.try_iter().map(|r| match r {
            InpaintBridgeResult::Error { item_id, generation, .. } => (item_id, generation),
            _ => panic!("only errors"),
        }).collect();
        assert_eq!(ended_items, [(2, 1), (1, 4)], "one error per image, naming its latest stroke");
    }

    /// "Keep the Stable Diffusion eraser loaded" holds it until quit; only
    /// without it does an idle eraser go after the window.
    #[test]
    fn an_idle_eraser_goes_unless_kept_loaded() {
        let long = IDLE_RELEASE + Duration::from_secs(1);
        assert!(idle_release_due(true, long, false));
        assert!(!idle_release_due(true, long, true), "kept loaded: stays");
        assert!(!idle_release_due(false, long, false), "a fill in flight: stays");
        assert!(!idle_release_due(true, Duration::from_secs(1), false), "not idle long enough");
    }

    /// A worker that dies mid-stroke ends the stroke with why, and the
    /// next stroke starts a new worker: the stroke stayed busy for good.
    #[test]
    fn a_worker_that_dies_ends_its_strokes() {
        let child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .arg("--list")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("a stand-in worker");
        let mut state = SubState::Ready(SubprocessManager::for_test(child));
        let mut inflight = Strokes::from([(7, 3)]);
        let (tx, rx) = mpsc::channel();
        let deadline = Instant::now() + Duration::from_secs(10);
        while matches!(state, SubState::Ready(_)) && Instant::now() < deadline {
            end_if_dead(&mut state, &mut inflight, &tx);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(matches!(state, SubState::Idle), "the next stroke starts a new worker");
        assert!(inflight.is_empty());
        match rx.try_recv() {
            Ok(InpaintBridgeResult::Error { item_id: 7, error, .. }) => assert!(error.starts_with("The eraser stopped"), "{error}"),
            _ => panic!("the stroke did not end"),
        }
    }

    /// A worker that lives on with output that can no longer be read (a
    /// frame too large or garbled) ends its strokes like a dead one: the
    /// stroke stayed busy for good, the worker's memory held.
    #[cfg(unix)]
    #[test]
    fn a_worker_whose_output_breaks_ends_its_strokes() {
        let child = std::process::Command::new("sh")
            .args(["-c", "printf '\\377\\377\\377\\377'; sleep 30"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("a stand-in worker");
        let mut state = SubState::Ready(SubprocessManager::for_test(child));
        let mut inflight = Strokes::from([(7, 3)]);
        let (tx, rx) = mpsc::channel();
        let deadline = Instant::now() + Duration::from_secs(10);
        while matches!(state, SubState::Ready(_)) && Instant::now() < deadline {
            if let SubState::Ready(s) = &mut state {
                s.poll_events();
            }
            end_if_dead(&mut state, &mut inflight, &tx);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(matches!(state, SubState::Idle), "the next stroke starts a new worker");
        match rx.try_recv() {
            Ok(InpaintBridgeResult::Error { item_id: 7, error, .. }) => assert!(error.starts_with("The eraser stopped"), "{error}"),
            _ => panic!("the stroke did not end"),
        }
    }
}
