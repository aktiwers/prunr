//! Dedicated bridge thread for the SD inpaint subprocess.
//!
//! Mirrors `worker.rs`'s seg bridge but scoped to a single
//! `spawn_inpaint_only` subprocess: it never serves seg/edge work, so the
//! engine pool is skipped at Init. The subprocess is spawned lazily on
//! the first SD dispatch and dropped after 5 minutes of inactivity to
//! release the ~5 GB bundle resident set.
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

use crate::subprocess::manager::SubprocessManager;
use crate::subprocess::protocol::SubprocessEvent;
use prunr_core::inpaint_sd::SdInpaintRequest;
use prunr_models::ModelId;

/// Idle window before the bridge drops its subprocess to reclaim RAM.
/// Mirrors `inpaint_sd::SdSession::IDLE_RELEASE`. 5 minutes balances
/// "user might paint again any second" vs "bundle is 5 GB and the OS
/// should get it back if they switched away."
const IDLE_RELEASE: Duration = Duration::from_secs(5 * 60);

/// Below this much free RAM the watchdog force-kills the worker.
/// 1 GB sits below Linux's aggressive-reclaim band and above macOS's
/// "compressed memory pressure" warning, so kicks before the kernel
/// starts swap-thrashing on either platform.
const MEMORY_PRESSURE_THRESHOLD_BYTES: u64 = 1024 * 1024 * 1024;

/// Parent → bridge messages.
pub enum InpaintBridgeMsg {
    /// Dispatch one SD inpaint stroke. Image + mask are temp-file
    /// PNGs the worker decodes (same shape as
    /// `SubprocessCommand::Inpaint`). `feather_px` / `sharpen`
    /// configure the worker-side post-process pass. `gen` is the
    /// dispatch-time generation counter — round-tripped on `Done` so
    /// the GUI can drop a stale stroke's result if a fresher stroke
    /// was dispatched while this one was in flight.
    Dispatch {
        item_id: u64,
        gen: u64,
        model_id: ModelId,
        image_path: std::path::PathBuf,
        mask_path: std::path::PathBuf,
        sd_req: Option<SdInpaintRequest>,
        feather_px: f32,
        sharpen: f32,
    },
    /// Cancel one in-flight stroke. Reuses `CancelItem` IPC; worker's
    /// handler flips both seg and inpaint cancel flags so this is the
    /// single cancel verb.
    Cancel { item_id: u64 },
}

/// Bridge → parent results. One-to-one with the SD-specific subset of
/// `SubprocessEvent` — the bridge filters out seg events the inpaint-
/// only subprocess can't emit anyway.
pub enum InpaintBridgeResult {
    Progress { item_id: u64, current: u32, total: u32 },
    Done { item_id: u64, gen: u64, rgba_path: std::path::PathBuf, width: u32, height: u32 },
    Error { item_id: u64, error: String },
}

/// Spawn the inpaint bridge thread. Returns send/receive channels.
/// The thread runs until `Shutdown` is sent or the parent drops the
/// receiver.
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

/// Buffered dispatch params kept while the subprocess is mid-spawn.
/// When `Ready` lands, the bridge replays each entry through
/// `send_inpaint` in order.
struct PendingDispatch {
    item_id: u64,
    model_id: ModelId,
    image_path: std::path::PathBuf,
    mask_path: std::path::PathBuf,
    sd_req: Option<SdInpaintRequest>,
    feather_px: f32,
    sharpen: f32,
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
        pending: Vec<PendingDispatch>,
        /// Set when a Cancel arrives mid-spawn. The bridge has already
        /// drained `inflight_gens` and surfaced "Erase cancelled" — when
        /// the spawn result lands we kill the new subprocess instead of
        /// promoting it.
        cancelled: bool,
    },
    Ready(SubprocessManager),
}

fn run(msg_rx: mpsc::Receiver<InpaintBridgeMsg>, res_tx: mpsc::Sender<InpaintBridgeResult>) {
    let mut sub_state = SubState::Idle;
    let mut last_used = Instant::now();
    // Per-item dispatch generation. Threaded onto `Done` events so the
    // GUI's drain path can drop a stale stroke's result if a fresher
    // stroke superseded it while this one was in flight. The IPC
    // doesn't carry gen — only the GUI cares about it.
    let mut inflight_gens: HashMap<u64, u64> = HashMap::new();

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
                if let SubState::Ready(mut s) = std::mem::replace(&mut sub_state, SubState::Idle) {
                    let _ = s.shutdown_with_timeout(Duration::from_secs(2));
                }
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
                        if let SubState::Ready(mut s) = std::mem::replace(&mut sub_state, SubState::Idle) {
                            let _ = s.shutdown_with_timeout(Duration::from_secs(2));
                        }
                        return;
                    }
                },
            };
            match msg {
                InpaintBridgeMsg::Dispatch { item_id, gen, model_id, image_path, mask_path, sd_req, feather_px, sharpen } => {
                    last_used = Instant::now();
                    inflight_gens.insert(item_id, gen);
                    let _ = gen;
                    let params = PendingDispatch {
                        item_id, model_id, image_path, mask_path,
                        sd_req, feather_px, sharpen,
                    };
                    handle_dispatch(&mut sub_state, &mut inflight_gens, &res_tx, params);
                }
                InpaintBridgeMsg::Cancel { item_id: _ } => {
                    // SD's UNet `run()` is uninterruptible mid-step;
                    // flag-flip cancel can't free RAM in time. The
                    // kill applies to ALL in-flight strokes — one
                    // user, one Cancel button. Cancel-during-spawn
                    // surfaces "Erase cancelled" immediately and marks
                    // the spawn for kill-on-arrival.
                    handle_cancel(&mut sub_state, &mut inflight_gens, &res_tx);
                }
            }
        }

        // Poll for spawn completion (non-blocking). If the helper
        // thread finished, transition state and send any pending
        // dispatches — or kill if Cancel arrived mid-spawn.
        if matches!(sub_state, SubState::Spawning { .. }) {
            poll_spawn_result(&mut sub_state, &mut inflight_gens, &res_tx);
        }

        // Watchdog: catches `working_set_mb` underestimates and
        // mid-load free-RAM shifts that the pre-flight gate can't see.
        // Only meaningful once the subprocess is Ready and dispatch
        // has actually started — during Spawning we haven't loaded
        // any model yet, so memory pressure is from somewhere else.
        if !inflight_gens.is_empty() && matches!(sub_state, SubState::Ready(_)) {
            let free = crate::hardware::available_ram_bytes_throttled();
            if free > 0 && free < MEMORY_PRESSURE_THRESHOLD_BYTES {
                tracing::warn!(
                    free_mb = free / (1024 * 1024),
                    threshold_mb = MEMORY_PRESSURE_THRESHOLD_BYTES / (1024 * 1024),
                    in_flight = inflight_gens.len(),
                    "memory-pressure abort: killing inpaint subprocess",
                );
                kill_ready_and_drain(
                    &mut sub_state, &mut inflight_gens, &res_tx,
                    crate::subprocess::protocol::MEMORY_PRESSURE_ABORT_MSG,
                    "inpaint subprocess killed by memory-pressure watchdog",
                );
            }
        }

        // Drain subprocess events when Ready.
        if let SubState::Ready(s) = &mut sub_state {
            for evt in s.poll_events() {
                match evt {
                    SubprocessEvent::InpaintProgress { item_id, current, total } => {
                        let _ = res_tx.send(InpaintBridgeResult::Progress { item_id, current, total });
                    }
                    SubprocessEvent::InpaintDone { item_id, rgba_path, width, height } => {
                        last_used = Instant::now();
                        let gen = inflight_gens.remove(&item_id).unwrap_or(0);
                        let _ = res_tx.send(InpaintBridgeResult::Done { item_id, gen, rgba_path, width, height });
                    }
                    SubprocessEvent::InpaintError { item_id, error } => {
                        last_used = Instant::now();
                        inflight_gens.remove(&item_id);
                        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error });
                    }
                    // Inpaint-only subprocess shouldn't emit seg events,
                    // but if one slips through, ignore quietly rather
                    // than misroute it as an inpaint event.
                    _ => {}
                }
            }
        }

        // Idle eviction: drop the subprocess after IDLE_RELEASE of no
        // dispatches. The `no_in_flight` gate protects long strokes;
        // only consider eviction in Ready state.
        if let SubState::Ready(s) = &mut sub_state {
            let no_in_flight = s.in_flight_items().is_empty();
            if no_in_flight && last_used.elapsed() > IDLE_RELEASE {
                if let SubState::Ready(mut owned) = std::mem::replace(&mut sub_state, SubState::Idle) {
                    let _ = owned.shutdown_with_timeout(Duration::from_secs(2));
                    tracing::info!("inpaint subprocess released after idle window");
                }
            }
        }
    }
}

/// Routes an incoming `Dispatch` based on current `SubState`. Handles
/// Idle (kicks an off-thread spawn + queues), Spawning (queues),
/// Ready (sends straight through).
fn handle_dispatch(
    sub_state: &mut SubState,
    inflight_gens: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    params: PendingDispatch,
) {
    match sub_state {
        SubState::Ready(s) => {
            if let Err(e) = s.send_inpaint(
                params.item_id, params.model_id, params.image_path, params.mask_path,
                params.sd_req, params.feather_px, params.sharpen,
            ) {
                let _ = res_tx.send(InpaintBridgeResult::Error { item_id: params.item_id, error: e });
                inflight_gens.remove(&params.item_id);
            }
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
    inflight_gens: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
) {
    match sub_state {
        SubState::Idle => {}
        SubState::Spawning { cancelled, .. } => {
            if inflight_gens.is_empty() {
                return;
            }
            *cancelled = true;
            for (id, _gen) in inflight_gens.drain() {
                let _ = res_tx.send(InpaintBridgeResult::Error {
                    item_id: id,
                    error: crate::subprocess::protocol::CANCELLED_ERR_MSG.into(),
                });
            }
            tracing::info!("inpaint Cancel during spawn — pending result will be killed");
        }
        SubState::Ready(_) => {
            kill_ready_and_drain(
                sub_state, inflight_gens, res_tx,
                crate::subprocess::protocol::CANCELLED_ERR_MSG,
                "inpaint subprocess killed by Cancel",
            );
        }
    }
}

/// Spawn `SubprocessManager::spawn_inpaint_only` on a one-shot helper
/// thread so the bridge stays responsive to Cancel during the Init
/// handshake. Returns the `Spawning` state with the result channel
/// and the queued dispatch list.
fn kick_spawn(pending: Vec<PendingDispatch>) -> SubState {
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
    inflight_gens: &mut HashMap<u64, u64>,
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
                if let Err(e) = mgr.send_inpaint(
                    params.item_id, params.model_id, params.image_path, params.mask_path,
                    params.sd_req, params.feather_px, params.sharpen,
                ) {
                    let _ = res_tx.send(InpaintBridgeResult::Error { item_id: params.item_id, error: e });
                    inflight_gens.remove(&params.item_id);
                }
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
                inflight_gens.remove(&params.item_id);
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
    inflight_gens: &mut HashMap<u64, u64>,
    res_tx: &mpsc::Sender<InpaintBridgeResult>,
    error_msg: &'static str,
    log_msg: &'static str,
) {
    if inflight_gens.is_empty() {
        return;
    }
    if let SubState::Ready(mut s) = std::mem::replace(sub_state, SubState::Idle) {
        s.kill();
        tracing::info!("{log_msg}");
    }
    for (id, _gen) in inflight_gens.drain() {
        let _ = res_tx.send(InpaintBridgeResult::Error {
            item_id: id,
            error: error_msg.into(),
        });
    }
}
