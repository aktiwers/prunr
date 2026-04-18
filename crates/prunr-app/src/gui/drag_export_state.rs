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
