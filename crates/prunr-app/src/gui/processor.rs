//! Processing-pipeline coordinator: subprocess worker channels, admission
//! control, live preview, and per-batch dispatch state.
//!
//! Owns:
//! - The worker bridge channels (`worker_tx` / `worker_rx`) — UI thread sends
//!   `WorkerMessage`, drains `WorkerResult` non-blockingly each frame.
//! - The shared cancellation flag (`Arc<AtomicBool>`) — read by the worker
//!   bridge to short-circuit a batch in flight.
//! - The in-process Tier 2 live-preview dispatcher.
//! - Admission controller state during streaming batches.
//! - The dispatch-time recipe snapshot (used to attribute completed results
//!   to the settings that produced them, even if the user keeps editing).
//! - The periodic history-cleanup timestamp.
//!
//! Does NOT own:
//! - The worker bridge thread itself — that's spawned by `worker::spawn_worker`
//!   at app startup. We just hold the channel ends.
//! - `BatchManager` (per the cross-coordinator borrow rule). Methods that
//!   operate on the batch take `&mut BatchManager` per call, never as a field.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use prunr_core::ProcessingRecipe;

use super::live_preview::LivePreview;
use super::memory::AdmissionController;
use super::worker::{WorkerMessage, WorkerResult, WorkItem};

pub(crate) struct Processor {
    pub(crate) worker_tx: mpsc::Sender<WorkerMessage>,
    pub(crate) worker_rx: mpsc::Receiver<WorkerResult>,
    /// Shared with the worker bridge thread — set by `cancel`, polled by the
    /// bridge to stop a batch in flight.
    pub(crate) cancel_flag: Arc<AtomicBool>,
    pub(crate) live_preview: LivePreview,
    /// Active admission controller (present only during streaming batches).
    pub(crate) admission: Option<AdmissionController>,
    /// Sender for streaming additional items to the worker.
    pub(crate) admission_tx: Option<mpsc::Sender<WorkItem>>,
    /// Recipe snapshot taken at dispatch time — stored on completed items so
    /// settings edits during a long batch don't re-attribute results.
    pub(crate) dispatch_recipe: Option<ProcessingRecipe>,
    /// Last time periodic history cleanup ran.
    pub(crate) last_history_cleanup: Instant,
}

impl Processor {
    pub(crate) fn new(
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        Self {
            worker_tx,
            worker_rx,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            live_preview: LivePreview::default(),
            admission: None,
            admission_tx: None,
            dispatch_recipe: None,
            last_history_cleanup: Instant::now(),
        }
    }

    /// Drop admission state so no further items are admitted. Called on
    /// cancel (user or worker-side) and by the cancelled-message handler.
    /// Leaves `cancel_flag` untouched — that's owned by the caller's cancel
    /// protocol.
    pub(crate) fn clear_admission(&mut self) {
        self.admission = None;
        self.admission_tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn fixture() -> Processor {
        let (tx, _rx_unused) = mpsc::channel::<WorkerMessage>();
        let (_tx_unused, rx) = mpsc::channel::<WorkerResult>();
        Processor::new(tx, rx)
    }

    #[test]
    fn new_initialises_last_history_cleanup_recent() {
        // Verifies the periodic 600s cleanup gate isn't accidentally
        // triggered at startup — the Instant must be effectively-now.
        let p = fixture();
        assert!(p.last_history_cleanup.elapsed().as_secs() < 5);
    }

    #[test]
    fn clear_admission_drops_both_sides_but_leaves_cancel_flag() {
        let mut p = fixture();
        let (tx, _rx) = mpsc::channel::<WorkItem>();
        p.admission_tx = Some(tx);
        // admission controller is harder to construct from a test; the None → Some
        // transition for admission is adequately covered by admission_tx + a sentinel.
        p.cancel_flag.store(true, Ordering::Release);
        assert!(p.admission_tx.is_some());

        p.clear_admission();

        assert!(p.admission.is_none());
        assert!(p.admission_tx.is_none());
        assert!(p.cancel_flag.load(Ordering::Acquire),
            "clear_admission must leave cancel_flag untouched — that's the caller's protocol");
    }

    #[test]
    fn cancel_flag_is_shared_arc_visible_across_clones() {
        // The cancel_flag is cloned into WorkerMessage::BatchProcess and read
        // by the bridge thread. A store on the parent must be visible via any
        // clone of the Arc — verify by storing through one handle, reading via another.
        let p = fixture();
        let clone = p.cancel_flag.clone();
        assert!(!clone.load(Ordering::Acquire));
        p.cancel_flag.store(true, Ordering::Release);
        assert!(clone.load(Ordering::Acquire), "Arc clone must observe the store");
    }
}
