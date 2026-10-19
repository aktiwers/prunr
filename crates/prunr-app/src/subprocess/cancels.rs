//! The worker's cancels: one flag per dispatch, registered as its command
//! arrives, so a `CancelItem` reaches the run it was sent for whether that
//! run is queued, starting or under way, and never a later run of the
//! same image.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

#[derive(Clone, Default)]
pub struct DispatchCancels(Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>>);

impl DispatchCancels {
    /// Register a dispatch for `item_id`; call as its command is read,
    /// before a later `CancelItem` can be.
    pub fn register(&self, item_id: u64) -> DispatchCancel {
        let flag = Arc::new(AtomicBool::new(false));
        self.0.lock().unwrap_or_else(PoisonError::into_inner).insert(item_id, flag.clone());
        DispatchCancel { item_id, flag, cancels: self.clone() }
    }

    /// Cancel the registered dispatch of `item_id`. With none (its run
    /// already ended) nothing is kept for later.
    pub fn cancel(&self, item_id: u64) {
        if let Some(flag) = self.0.lock().unwrap_or_else(PoisonError::into_inner).get(&item_id) {
            flag.store(true, Ordering::Release);
        }
    }
}

/// One dispatch's cancel; unregisters on drop.
pub struct DispatchCancel {
    item_id: u64,
    flag: Arc<AtomicBool>,
    cancels: DispatchCancels,
}

impl DispatchCancel {
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// The flag a run's progress polls.
    pub fn flag(&self) -> Arc<AtomicBool> {
        self.flag.clone()
    }
}

impl Drop for DispatchCancel {
    fn drop(&mut self) {
        let mut flags = self.cancels.0.lock().unwrap_or_else(PoisonError::into_inner);
        // A newer dispatch of the image may have registered since.
        if flags.get(&self.item_id).is_some_and(|f| Arc::ptr_eq(f, &self.flag)) {
            flags.remove(&self.item_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cancel_mid_run_leaves_the_images_next_run_alone() {
        let cancels = DispatchCancels::default();
        let first = cancels.register(7);
        assert!(!first.is_cancelled(), "nothing cancelled as the run starts");
        cancels.cancel(7);
        assert!(first.flag().load(Ordering::Acquire), "the run under way sees the cancel");
        drop(first);

        let next = cancels.register(7);
        assert!(!next.is_cancelled(), "the next run of the image starts uncancelled");
        assert!(!next.flag().load(Ordering::Acquire));
    }

    #[test]
    fn a_cancel_after_the_run_ended_reaches_no_later_run() {
        let cancels = DispatchCancels::default();
        drop(cancels.register(7));
        cancels.cancel(7);
        assert!(!cancels.register(7).is_cancelled());
    }

    #[test]
    fn an_ended_run_does_not_unregister_a_newer_run_of_the_image() {
        let cancels = DispatchCancels::default();
        let old = cancels.register(7);
        let new = cancels.register(7);
        drop(old);
        cancels.cancel(7);
        assert!(new.is_cancelled(), "the newer run still hears its cancel");
    }

    #[test]
    fn a_cancel_reaches_only_its_image() {
        let cancels = DispatchCancels::default();
        let seven = cancels.register(7);
        let eight = cancels.register(8);
        cancels.cancel(7);
        assert!(seven.is_cancelled());
        assert!(!eight.is_cancelled());
    }
}
