//! Dedicated bridge thread for the SD inpaint subprocess.
//!
//! Mirrors `worker.rs`'s seg bridge but scoped to a single
//! `spawn_inpaint_only` subprocess: it never serves seg/edge work, so the
//! engine pool is skipped at Init. The subprocess is spawned lazily on
//! the first SD dispatch and dropped after `IDLE_RELEASE` idle, unless
//! kept loaded, to release the ~5 GB bundle resident set.
//!
//! `Processor` sends `InpaintBridgeMsg` and reads `InpaintBridgeResult`
//! through a pair of mpsc channels — same pattern as `WorkerMessage` /
//! `WorkerResult` for seg work.
//!
//! LaMa / Big-LaMa / MI-GAN stay on the in-process rayon path
//! (`Processor::dispatch_inpaint` short-circuits non-SD models). They
//! don't go near this bridge.

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::subprocess::manager::{InpaintJob, SubprocessManager};
use crate::subprocess::protocol::SubprocessEvent;
use prunr_core::inpaint_sd::SdInpaintRequest;
use prunr_models::ModelId;
use prunr_core::IDLE_RELEASE;


/// Whether the idle eraser subprocess goes now: nothing in flight, idle
/// past the window, and the user did not ask to keep it loaded.
fn idle_release_due(no_in_flight: bool, idle: Duration, keep_loaded: bool) -> bool {
    no_in_flight && idle > IDLE_RELEASE && !keep_loaded
}

/// Below this much free RAM the watchdog force-kills the worker.
/// 1 GB sits below Linux's aggressive-reclaim band and above macOS's
/// "compressed memory pressure" warning, so kicks before the kernel
/// starts swap-thrashing on either platform.
const MEMORY_PRESSURE_THRESHOLD_BYTES: u64 = 1024 * 1024 * 1024;

/// How long a released subprocess gets to exit before it is killed.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);

/// Parent → bridge messages.
pub enum InpaintBridgeMsg {
    /// Dispatch one SD inpaint stroke. Image + mask are temp-file
    /// PNGs the worker decodes (same shape as
    /// `SubprocessCommand::Inpaint`). `feather_px` / `sharpen`
    /// configure the worker-side post-process pass. `generation` is the
    /// dispatch-time generation counter — round-tripped on `Done` so
    /// the GUI can drop a stale stroke's result if a fresher stroke
    /// was dispatched while this one was in flight.
    Dispatch {
        item_id: u64,
        generation: u64,
        model_id: ModelId,
        image_path: std::path::PathBuf,
        mask_path: std::path::PathBuf,
        sd_req: Option<SdInpaintRequest>,
        feather_px: f32,
        sharpen: f32,
    },
    /// Kill the subprocess; every in-flight stroke ends Cancelled.
    Cancel,
    /// Release any cached subprocess regardless of in-flight state.
    /// Differs from Cancel in two ways:
    ///   1. Idle subprocess (no in-flight) is also killed — Cancel
    ///      keeps an idle warm subprocess alive for fast next-stroke,
    ///      Release frees the ~200 MB residual immediately.
    ///   2. Doesn't surface a "Cancelled" toast — Release is
    ///      driven by deterministic GUI signals (e.g. model
    ///      switch) rather than a user-facing Cancel button.
    ///
    /// Sent by `apply_toolbar_change` when the user picks a
    /// non-SD-family eraser model or a non-inpaint model.
    Release,
}

/// Bridge → parent results. One-to-one with the SD-specific subset of
/// `SubprocessEvent` — the bridge filters out seg events the inpaint-
/// only subprocess can't emit anyway.
pub enum InpaintBridgeResult {
    Report { item_id: u64, update: prunr_core::ProgressUpdate },
    Done { item_id: u64, generation: u64, rgba_path: std::path::PathBuf, width: u32, height: u32, fill_scale: f32 },
    Error { item_id: u64, error: String },
}

/// Spawn the inpaint bridge thread. Returns send/receive channels.
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

/// Spawn lifecycle. `Idle` and `Ready` are the steady states; `Spawning`
/// is a transient that holds buffered dispatches until the worker
/// reports Ready. The `Spawning` state exists specifically so a Cancel
/// during the (potentially 30 s) pre-Ready window doesn't block the
/// bridge thread — the spawn runs on a helper thread, the bridge polls
/// for the result via `rx.try_recv()` each loop tick.
enum SubState {
    Idle,
    Spawning {
        rx: mpsc::Receiver<Result<(SubprocessManager, String), String>>,
        /// Strokes dispatched mid-spawn, sent in order once `Ready` lands.
        pending: Vec<InpaintJob>,
        /// Set when a Cancel arrives mid-spawn. The bridge has already
        /// drained `inflight_generations` and surfaced "Erase cancelled" — when
        /// the spawn result lands we kill the new subprocess instead of
        /// promoting it.
        cancelled: bool,
    },
    Ready(SubprocessManager),
}

fn run(msg_rx: mpsc::Receiver<InpaintBridgeMsg>, res_tx: mpsc::Sender<InpaintBridgeResult>) {
    let mut sub_state = SubState::Idle;
    let mut last_used = Instant::now();
    let mut keep_loaded = false;
    // Per-item dispatch generation. Threaded onto `Done` events so the
    // GUI's drain path can drop a stale stroke's result if a fresher
    // stroke superseded it while this one was in flight. The IPC
    // doesn't carry generation — only the GUI cares about it.
    let mut inflight_generations: HashMap<u64, u64> = HashMap::new();

    loop {
        // Block up to 100 ms waiting for the next parent message. This
        // eliminates the fixed 100 ms sleep: a new Dispatch or Cancel
        // wakes the thread immediately instead of waiting for the next
        // poll tick. The subprocess event drain below still runs on the
        // same 100 ms cadence when idle (recv_timeout returns Empty).
        //
        // If recv_timeout returns a message we handle it inside the same
        // try_recv drain loop (first arm `msg`; rest via try_recv).
        let first = msg_rx.recv_timeout(Duration::from_millis(100));
        let mut pending_msg = match first {
            Ok(msg) => Some(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                shutdown_ready(&mut sub_state);
                return;
            }
        };
        // Drain parent messages (first is already resolved; rest via try_recv).
        loop {
            let msg = match pending_msg.take() {
                Some(m) => m,
                None => match msg_rx.try_recv() {
                    Ok(m) => m,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        shutdown_ready(&mut sub_state);
                        return;
                    }
                },
            };
            match msg {
                InpaintBridgeMsg::Dispatch { item_id, generation, model_id, image_path, mask_path, sd_req, feather_px, sharpen } => {
                    last_used = Instant::now();
                    keep_loaded = sd_req.as_ref().is_some_and(|r| r.tuning.keep_loaded);
                    inflight_generations.insert(item_id, generation);
                    let params = InpaintJob {
                        item_id, model_id, image_path, mask_path,
                        sd_req, feather_px, sharpen,
                    };
                    handle_dispatch(&mut sub_state, &mut inflight_generations, &res_tx, params);
                }
                InpaintBridgeMsg::Cancel => {
                    // SD's UNet `run()` is uninterruptible mid-step;
                    // flag-flip cancel can't free RAM in time. The
                    // kill applies to ALL in-flight strokes — one
                    // user, one Cancel button. Cancel-during-spawn
                    // surfaces "Erase cancelled" immediately and marks
                    // the spawn for kill-on-arrival.
                    handle_cancel(&mut sub_state, &mut inflight_generations, &res_tx);
                }
                InpaintBridgeMsg::Release => {
                    handle_release(&mut sub_state, &mut inflight_generations, &res_tx);
                }
            }
        }

        // Poll for spawn completion (non-blocking). If the helper
        // thread finished, transition state and send any pending
        // dispatches — or kill if Cancel arrived mid-spawn.
        if matches!(sub_state, SubState::Spawning { .. }) {
            poll_spawn_result(&mut sub_state, &mut inflight_generations, &res_tx);
        }

        // Watchdog: catches `working_set_mb` underestimates and
        // mid-load free-RAM shifts that the pre-flight gate can't see.
        // Only meaningful once the subprocess is Ready and dispatch
        // has actually started — during Spawning we haven't loaded
        // any model yet, so memory pressure is from somewhere else.
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
                    "inpaint subprocess killed by memory-pressure watchdog",
                );
            }
        }

        // Drain subprocess events when Ready.
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
                    // Inpaint-only subprocess shouldn't emit seg events,
                    // but if one slips through, ignore quietly rather
                    // than misroute it as an inpaint event.
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

/// Routes an incoming `Dispatch` based on current `SubState`. Handles
/// Idle (kicks an off-thread spawn + queues), Spawning (queues),
/// Ready (sends straight through).
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

/// Cancel during Idle is a no-op. Cancel during Ready kills the
/// subprocess. Cancel during Spawning surfaces the cancelled toast
/// immediately and marks the spawn for kill-on-arrival — the user
/// gets feedback in <100 ms instead of waiting up to 30 s for the
/// pre-Ready timeout.
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
                "inpaint subprocess killed by Cancel",
            );
        }
    }
}

/// Release any cached subprocess regardless of in-flight state. Differs
/// from `handle_cancel` in two ways: (1) an idle Ready subprocess is
/// also killed (no inflight check; the residual ~200 MB is the whole
/// reason the GUI sent Release), and (2) any in-flight strokes get the
/// `CANCELLED_ERR_MSG` sentinel so the user still sees a toast — the
/// GUI typically only sends Release on model-switch, where dropping
/// the in-flight stroke is the intent anyway.
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

/// Spawn `SubprocessManager::spawn_inpaint_only` on a one-shot helper
/// thread so the bridge stays responsive to Cancel during the Init
/// handshake. Returns the `Spawning` state with the result channel
/// and the queued dispatch list.
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

/// Non-blocking poll for spawn completion. On Ok with `cancelled =
/// false`, transitions to Ready and replays buffered dispatches.
/// On Ok with `cancelled = true`, kills the new subprocess and
/// returns to Idle (toasts already surfaced when Cancel was handled).
/// On Err, surfaces the error to every queued dispatch and returns
/// to Idle.
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
                    error: format!("inpaint subprocess spawn failed: {e}"),
                });
                inflight_generations.remove(&params.item_id);
            }
        }
    }
}

/// SIGKILL the Ready subprocess and surface `error_msg` on every
/// in-flight stroke. No-op when no stroke is in flight (next dispatch
/// re-spawns lazily). Caller must have asserted `SubState::Ready` —
/// the function transitions to `Idle` on its way out.
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
    if s.is_alive() {
        return;
    }
    let reason = s.crash_reason();
    tracing::warn!(%reason, in_flight = inflight_generations.len(), "eraser worker ended");
    *sub_state = SubState::Idle;
    fail_inflight(inflight_generations, res_tx, &format!("The eraser stopped: {reason}. Erase again."));
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
