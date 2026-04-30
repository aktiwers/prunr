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

fn run(msg_rx: mpsc::Receiver<InpaintBridgeMsg>, res_tx: mpsc::Sender<InpaintBridgeResult>) {
    // None until the first Dispatch; spawned lazily so users who never
    // touch SD inpaint don't pay the subprocess + bundle cost.
    let mut sub: Option<SubprocessManager> = None;
    let mut last_used = Instant::now();
    // Per-item dispatch generation. Threaded onto `Done` events so the
    // GUI's drain path can drop a stale stroke's result if a fresher
    // stroke superseded it while this one was in flight. The IPC
    // doesn't carry gen — only the GUI cares about it.
    let mut inflight_gens: HashMap<u64, u64> = HashMap::new();
    // Poll cadence: 100 ms is fast enough that progress events feel
    // live (UNet steps are 1-3s on CPU) and slow enough that the idle
    // case doesn't burn CPU.
    let poll = Duration::from_millis(100);

    loop {
        // Drain pending parent messages without blocking — keeps the
        // event-pump responsive during a steady stream of dispatches.
        loop {
            match msg_rx.try_recv() {
                Ok(InpaintBridgeMsg::Dispatch { item_id, gen, model_id, image_path, mask_path, sd_req, feather_px, sharpen }) => {
                    last_used = Instant::now();
                    inflight_gens.insert(item_id, gen);
                    let s = match ensure_sub(&mut sub) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error: e });
                            inflight_gens.remove(&item_id);
                            continue;
                        }
                    };
                    if let Err(e) = s.send_inpaint(item_id, model_id, image_path, mask_path, sd_req, feather_px, sharpen) {
                        let _ = res_tx.send(InpaintBridgeResult::Error { item_id, error: e });
                        inflight_gens.remove(&item_id);
                    }
                }
                Ok(InpaintBridgeMsg::Cancel { item_id }) => {
                    if let Some(s) = sub.as_mut() {
                        let _ = s.send_cancel_item(item_id);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Parent dropped the sender (app exit). Tear down
                    // the subprocess and exit the bridge thread.
                    if let Some(mut s) = sub.take() {
                        let _ = s.shutdown_with_timeout(Duration::from_secs(2));
                    }
                    return;
                }
            }
        }

        // Drain subprocess events.
        if let Some(s) = sub.as_mut() {
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
        // dispatches. The `no_in_flight` gate is what protects long
        // strokes — `last_used` only updates on Dispatch / Done /
        // Error, not on Progress, so a 4m59s SD stroke would technically
        // hit the timer, but the in-flight check prevents eviction
        // mid-inference. Next dispatch pays the spawn-and-load cost.
        if let Some(s) = sub.as_mut() {
            let no_in_flight = s.in_flight_items().is_empty();
            if no_in_flight && last_used.elapsed() > IDLE_RELEASE {
                let mut owned = sub.take().unwrap(); // just checked Some above
                let _ = owned.shutdown_with_timeout(Duration::from_secs(2));
                tracing::info!("inpaint subprocess released after idle window");
            }
        }

        std::thread::sleep(poll);
    }
}

fn ensure_sub(sub: &mut Option<SubprocessManager>) -> Result<&mut SubprocessManager, String> {
    if sub.is_none() {
        let started = Instant::now();
        let (m, provider) = SubprocessManager::spawn_inpaint_only()
            .map_err(|e| format!("inpaint subprocess spawn failed: {e}"))?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            provider = %provider,
            "inpaint subprocess ready",
        );
        *sub = Some(m);
    }
    Ok(sub.as_mut().expect("just inserted"))
}
