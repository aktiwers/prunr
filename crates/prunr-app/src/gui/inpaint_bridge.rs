//! Dedicated bridge thread for the SD inpaint subprocess. It never serves
//! seg/edge work, so the engine pool is skipped at Init. The subprocess is
//! spawned lazily on the first SD dispatch and dropped after
//! `IDLE_RELEASE` idle, unless kept loaded, to release the ~5 GB bundle.
//! The LaMa family stays on the in-process rayon path.

use std::collections::HashMap;
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
    /// decodes. `generation` round-trips on `Done` so the GUI can drop a
    /// stroke a fresher one superseded.
    Dispatch { job: InpaintJob, generation: u64 },
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
    Error { item_id: u64, error: String },
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
    // Per-item dispatch generation, threaded onto `Done` so the GUI can drop
    // a superseded stroke's result; the IPC doesn't carry it.
    let mut inflight_generations: HashMap<u64, u64> = HashMap::new();

    loop {
        // Blocks up to 100 ms, so a Dispatch or Cancel wakes the thread at once
        // while the subprocess drain below keeps a 100 ms cadence when idle.
        let first = match msg_rx.recv_timeout(Duration::from_millis(100)) {
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
                InpaintBridgeMsg::Dispatch { job, generation } => {
                    last_used = Instant::now();
                    keep_loaded = job.sd_req.as_ref().is_some_and(|r| r.tuning.keep_loaded);
                    inflight_generations.insert(job.item_id, generation);
                    handle_dispatch(&mut sub_state, &mut inflight_generations, &res_tx, job);
                }
                InpaintBridgeMsg::Cancel => {
                    // SD's UNet `run()` can't be interrupted mid-step, so only a kill
                    // frees RAM in time; it ends every in-flight stroke.
                    handle_cancel(&mut sub_state, &mut inflight_generations, &res_tx);
                }
                InpaintBridgeMsg::Release => {
                    handle_release(&mut sub_state, &mut inflight_generations, &res_tx);
                }
            }
        }

        if matches!(sub_state, SubState::Spawning { .. }) {
            poll_spawn_result(&mut sub_state, &mut inflight_generations, &res_tx);
        }

        // Watchdog for `working_set_mb` underestimates and mid-load RAM shifts
        // the pre-flight gate can't see; only once Ready, since a spawning
        // worker has loaded nothing yet.
        if !inflight_generations.is_empty() && matches!(sub_state, SubState::Ready(_)) {
            let free = crate::hardware::available_ram_bytes_throttled();
            if free > 0 && free < MEMORY_PRESSURE_THRESHOLD_BYTES {
                tracing::warn!(
                    free_mb = free / (1024 * 1024),
                    threshold_mb = MEMORY_PRESSURE_THRESHOLD_BYTES / (1024 * 1024),
                    in_flight = inflight_generations.len(),
                    "memory-pressure abort: killing inpaint subprocess",
                );
                kill_ready_and_drain(
                    &mut sub_state, &mut inflight_generations, &res_tx,
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
                    SubprocessEvent::InpaintDone { item_id, rgba_path, width, height, fill_scale } => {
                        last_used = Instant::now();
                        let generation = inflight_generations.remove(&item_id).unwrap_or(0);
                        let _ = res_tx.send(InpaintBridgeResult::Done { item_id, generation, rgba_path, width, height, fill_scale });
                    }
                    SubprocessEvent::InpaintError { item_id, error } => {
                        last_used = Instant::now();
                        inflight_generations.remove(&item_id);
                        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error });
                    }
                    // An inpaint-only subprocess shouldn't emit seg events; ignore one
                    // rather than misroute it.
                    _ => {}
                }
            }
        }
        end_if_dead(&mut sub_state, &mut inflight_generations, &res_tx);

        if let SubState::Ready(s) = &mut sub_state {
            if idle_release_due(s.in_flight_items().is_empty(), last_used.elapsed(), keep_loaded) {
                shutdown_ready(&mut sub_state);
                tracing::info!("inpaint subprocess released after idle window");
            }
        }
    }
}

/// Idle: spawn off-thread and queue. Spawning: queue. Ready: send.
fn handle_dispatch(
    sub_state: &mut SubState,
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    params: InpaintJob,
) {
    match sub_state {
        SubState::Ready(s) => {
            send_or_fail(s, params, inflight_generations, res_tx);
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
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    match sub_state {
        SubState::Idle => {}
        SubState::Spawning { cancelled, .. } => {
            if inflight_generations.is_empty() {
                return;
            }
            *cancelled = true;
            fail_inflight(inflight_generations, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
            tracing::info!("inpaint Cancel during spawn — pending result will be killed");
        }
        SubState::Ready(_) => {
            kill_ready_and_drain(
                sub_state, inflight_generations, res_tx,
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
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    match std::mem::replace(sub_state, SubState::Idle) {
        SubState::Idle => {}
        SubState::Spawning { pending: _, cancelled: _, rx } => {
            // Spawn is still in flight and the helper thread can't be
            // killed: stay Spawning with `cancelled` set so the
            // arriving subprocess is killed, and drain inflight.
            *sub_state = SubState::Spawning { rx, pending: Vec::new(), cancelled: true };
            fail_inflight(inflight_generations, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
            tracing::info!("inpaint Release during spawn — pending result will be killed");
        }
        SubState::Ready(mut s) => {
            s.kill();
            tracing::info!("inpaint subprocess released on model switch");
            fail_inflight(inflight_generations, res_tx, crate::subprocess::protocol::CANCELLED_ERR_MSG);
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
    inflight_generations: &mut HashMap<u64, u64>,
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
                send_or_fail(&mut mgr, params, inflight_generations, res_tx);
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
            for params in pending {
                let _ = res_tx.send(InpaintBridgeResult::Error {
                    item_id: params.item_id,
                    error: format!("Could not start the eraser: {e}"),
                });
                inflight_generations.remove(&params.item_id);
            }
        }
    }
}

/// SIGKILL the Ready subprocess, end every in-flight stroke with
/// `error_msg`, and go Idle. No-op with nothing in flight.
fn kill_ready_and_drain(
    sub_state: &mut SubState,
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    error_msg: &'static str,
    log_msg: &'static str,
) {
    if inflight_generations.is_empty() {
        return;
    }
    if let SubState::Ready(mut s) = std::mem::replace(sub_state, SubState::Idle) {
        s.kill();
        tracing::info!("{log_msg}");
    }
    fail_inflight(inflight_generations, res_tx, error_msg);
}

/// Send one stroke; one the subprocess won't take ends with its error.
fn send_or_fail(
    mgr: &mut SubprocessManager,
    job: InpaintJob,
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    let item_id = job.item_id;
    if let Err(error) = mgr.send_inpaint(job) {
        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error });
        inflight_generations.remove(&item_id);
    }
}

/// A worker that died (killed for memory, crashed) ends the strokes it
/// had, with why, and the next stroke starts a new one. Its last events
/// are drained first, so a fill it finished still lands.
fn end_if_dead(
    sub_state: &mut SubState,
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    let SubState::Ready(s) = sub_state else { return };
    let Some(reason) = s.died() else { return };
    tracing::warn!(%reason, in_flight = inflight_generations.len(), "eraser worker ended");
    *sub_state = SubState::Idle;
    remove_dead_stroke_files(crate::subprocess::protocol::ipc_temp_dir(), inflight_generations);
    let why = match reason {
        crate::subprocess::manager::Crash::OutOfMemory => "The eraser ran out of memory. Try a lighter eraser or a smaller region.",
        _ => "The eraser stopped unexpectedly. Erase again.",
    };
    fail_inflight(inflight_generations, res_tx, why);
}

/// The files the dead worker's strokes left: their source and mask, and
/// the result it may have begun. A stroke queued since keeps its own,
/// which a later generation names. On Linux they sit in RAM until quit.
fn remove_dead_stroke_files(ipc_dir: &std::path::Path, inflight_generations: &HashMap<u64, u64>) {
    use crate::subprocess::protocol::IpcKind;
    for (&item_id, &generation) in inflight_generations {
        let _ = std::fs::remove_file(IpcKind::InpaintImg.path_for_generation(ipc_dir, item_id, generation));
        let _ = std::fs::remove_file(IpcKind::InpaintMask.path_for_generation(ipc_dir, item_id, generation));
        let _ = std::fs::remove_file(IpcKind::InpaintOut.path_for(ipc_dir, item_id));
    }
}

/// End every in-flight stroke with `error`.
fn fail_inflight(
    inflight_generations: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    error: &str,
) {
    for (item_id, _generation) in inflight_generations.drain() {
        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error: error.into() });
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

    /// A dead worker's strokes leave no files behind; a stroke queued since
    /// (a later generation) and another worker's files are left alone.
    #[test]
    fn a_dead_eraser_leaves_no_files_and_spares_the_next_stroke() {
        use crate::subprocess::protocol::IpcKind;
        let dir = tempfile::tempdir().unwrap();
        let ipc = dir.path();
        let dead = [
            IpcKind::InpaintImg.path_for_generation(ipc, 1, 3),
            IpcKind::InpaintMask.path_for_generation(ipc, 1, 3),
            IpcKind::InpaintOut.path_for(ipc, 1),
        ];
        let kept = [
            IpcKind::InpaintImg.path_for_generation(ipc, 1, 4),
            IpcKind::InpaintMask.path_for_generation(ipc, 1, 4),
            IpcKind::InpaintImg.path_for_generation(ipc, 2, 1),
            IpcKind::Result.path_for(ipc, 1),
        ];
        for path in dead.iter().chain(&kept) {
            std::fs::write(path, b"x").unwrap();
        }
        remove_dead_stroke_files(ipc, &HashMap::from([(1, 3)]));
        assert!(dead.iter().all(|p| !p.exists()), "the dead stroke's files are gone");
        assert!(kept.iter().all(|p| p.exists()), "the queued stroke's and others' stay");
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
        let mut inflight = HashMap::from([(7, 3)]);
        let (tx, rx) = mpsc::channel();
        let deadline = Instant::now() + Duration::from_secs(10);
        while matches!(state, SubState::Ready(_)) && Instant::now() < deadline {
            end_if_dead(&mut state, &mut inflight, &tx);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(matches!(state, SubState::Idle), "the next stroke starts a new worker");
        assert!(inflight.is_empty());
        match rx.try_recv() {
            Ok(InpaintBridgeResult::Error { item_id: 7, error }) => assert!(error.starts_with("The eraser stopped"), "{error}"),
            _ => panic!("the stroke did not end"),
        }
    }
}
