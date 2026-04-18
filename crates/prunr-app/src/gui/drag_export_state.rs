//! State for the OS drag-out feature (drag a thumbnail out of the sidebar
//! into Files / Finder / Explorer). The actual filesystem write of the PNG
//! lives in `gui/drag_export.rs`; this module owns the lifecycle state.
//!
//! The drag crate's completion callback runs on a separate thread, so the
//! "active" flag and "items" set are wrapped in `Arc<AtomicBool>` /
//! `Arc<Mutex<...>>`. Cloned handles are passed into the callback closure.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct DragExportState {
    /// One-shot: set by sidebar when a drag escapes the sidebar rect.
    /// Consumed by `ui()` which then invokes the platform drag crate.
    pub(crate) pending: Option<Vec<u64>>,
    /// True while an OS drag session is in progress. Flipped false by the
    /// drag crate's completion callback (runs on a separate thread).
    /// Read by sidebar to dim dragged thumbnails.
    pub(crate) active: Arc<AtomicBool>,
    /// Item IDs currently being dragged — sidebar reads this to dim the
    /// matching thumbnails. Shared with the drag callback thread.
    pub(crate) items: Arc<Mutex<HashSet<u64>>>,
    /// One-time flag: true if we've already shown the "Linux not supported"
    /// toast this session. Prevents repeat spam on every drag attempt.
    pub(crate) linux_notified: bool,
}

impl DragExportState {
    pub(crate) fn new() -> Self {
        Self {
            pending: None,
            active: Arc::new(AtomicBool::new(false)),
            items: Arc::new(Mutex::new(HashSet::new())),
            linux_notified: false,
        }
    }

    /// Clear active drag-out state (used on drag end, error, and Linux fallback).
    ///
    /// Takes the underlying primitives by reference rather than `&self` so the
    /// drag crate's completion callback — which runs on a separate thread —
    /// can call it via cloned `Arc`s without needing the whole `DragExportState`.
    ///
    /// Silent-skip on lock poisoning is intentional: the items set is ephemeral
    /// UI state. If a panic poisoned the lock, propagating that panic to the
    /// drag callback thread (or worse, to drop) is strictly worse than letting
    /// the dimmed thumbnails resolve naturally on the next drag attempt.
    pub(crate) fn reset(active: &AtomicBool, items: &Mutex<HashSet<u64>>) {
        active.store(false, Ordering::Release);
        if let Ok(mut set) = items.lock() {
            set.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_active_and_items() {
        let s = DragExportState::new();
        // Simulate an in-progress drag: active=true, items populated.
        s.active.store(true, Ordering::Release);
        s.items.lock().unwrap().extend([1u64, 2, 3]);
        assert!(s.active.load(Ordering::Acquire));
        assert_eq!(s.items.lock().unwrap().len(), 3);

        DragExportState::reset(&s.active, &s.items);
        assert!(!s.active.load(Ordering::Acquire));
        assert!(s.items.lock().unwrap().is_empty());
    }

    #[test]
    fn reset_does_not_touch_pending_or_linux_notified() {
        // Two flags are user-thread-only and not part of the cross-thread
        // reset — verify reset leaves them untouched.
        let mut s = DragExportState::new();
        s.pending = Some(vec![42]);
        s.linux_notified = true;

        DragExportState::reset(&s.active, &s.items);
        assert_eq!(s.pending, Some(vec![42]));
        assert!(s.linux_notified);
    }

    #[test]
    fn items_mutex_supports_insert_read_clear_round_trip() {
        let s = DragExportState::new();
        {
            let mut set = s.items.lock().unwrap();
            set.insert(7);
            set.insert(11);
        }
        {
            let set = s.items.lock().unwrap();
            assert!(set.contains(&7));
            assert!(set.contains(&11));
            assert_eq!(set.len(), 2);
        }
        s.items.lock().unwrap().clear();
        assert!(s.items.lock().unwrap().is_empty());
    }
}
