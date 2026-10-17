//! State for the OS drag-out feature (drag a thumbnail out of the sidebar
//! into Files / Finder / Explorer). The actual filesystem write of the PNG
//! lives in `gui/drag_export.rs`; this module owns the lifecycle state.
//!
//! The drag crate's completion callback runs on a separate thread, so the
//! "active" flag and "items" set are wrapped in `Arc<AtomicBool>` /
//! `Arc<Mutex<...>>`. Cloned handles are passed into the callback closure.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::drag_export::{DragFiles, DragKey};

/// How long the pointer rests on a thumbnail before its drag files are
/// written ahead: long enough that sweeping down the list writes nothing.
pub(crate) const HOVER_DWELL: Duration = Duration::from_millis(250);

/// How many images' drag files are written at once. PNG encoding is
/// CPU-bound: on 8 cores a 12-24 MP image took 0.15 s alone, 0.04 s each
/// with four writers and 0.03 s with eight. Each running write also holds
/// a baked copy of its image.
pub(crate) fn writers() -> usize {
    static WRITERS: OnceLock<usize> = OnceLock::new();
    *WRITERS.get_or_init(|| (num_cpus::get() / 2).clamp(1, 4))
}

/// One item's drag files, written ahead of a drag on a worker.
struct Prewrite {
    key: DragKey,
    files: PrewriteFiles,
}

enum PrewriteFiles {
    Writing(JoinHandle<std::io::Result<Vec<PathBuf>>>),
    Written(Vec<PathBuf>),
}

fn joined(item_id: u64, h: JoinHandle<std::io::Result<Vec<PathBuf>>>) -> Option<Vec<PathBuf>> {
    match h.join() {
        Ok(Ok(paths)) => Some(paths),
        Ok(Err(err)) => {
            tracing::warn!(item_id, %err, "writing drag files ahead failed");
            None
        }
        Err(_) => {
            tracing::warn!(item_id, "drag file writer panicked");
            None
        }
    }
}

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
    /// The thumbnail under the pointer, since when, and whether it is
    /// pressed: a drag may follow, so its files are written ahead.
    hover: Option<(u64, Instant, bool)>,
    /// Files written ahead, kept per image until it changes so hovering
    /// it again writes nothing; the drag folder is emptied at exit.
    prewritten: HashMap<u64, Prewrite>,
    /// The hovered thumbnail whose drag files were all current when last
    /// checked: nothing is due until the pointer moves to another one.
    /// An edit made while it rests there is written by the drag itself.
    settled: Option<u64>,
    /// A drag out waiting for its files, written on workers: it starts
    /// once they are all written, if the button is still held.
    starting: Option<StartingDrag>,
}

/// The images a drag takes, and those whose files it asked a write for.
struct StartingDrag {
    ids: Vec<u64>,
    asked: HashSet<u64>,
}

/// Where a drag waiting for its files stands.
pub(crate) enum DragReady {
    /// No drag is waiting.
    Idle,
    Waiting,
    /// Every write has landed: the images dragged, their files, and the
    /// images whose write failed.
    Ready { ids: Vec<u64>, paths: Vec<PathBuf>, failed: Vec<u64> },
}

impl DragExportState {
    pub(crate) fn new() -> Self {
        Self {
            pending: None,
            active: Arc::new(AtomicBool::new(false)),
            items: Arc::new(Mutex::new(HashSet::new())),
            linux_notified: false,
            hover: None,
            prewritten: HashMap::new(),
            settled: None,
            starting: None,
        }
    }

    /// Note the thumbnail under the pointer this frame (`None` off the
    /// list). Returns when its files are due, for a repaint then.
    pub(crate) fn note_hover(&mut self, hovered: Option<(u64, bool)>, now: Instant) -> Option<Instant> {
        self.hover = match (hovered, self.hover) {
            (Some((id, pressed)), Some((was, since, _))) if id == was => Some((id, since, pressed)),
            (Some((id, pressed)), _) => Some((id, now, pressed)),
            (None, _) => None,
        };
        if self.hover.map(|(id, ..)| id) != self.settled {
            self.settled = None;
        }
        let (_, since, _) = self.hover?;
        Some(since + HOVER_DWELL)
    }

    /// The thumbnail whose drag files are due now: pressed, or rested on
    /// long enough, and no drag under way.
    pub(crate) fn due(&self, now: Instant) -> Option<u64> {
        let (id, since, pressed) = self.hover.filter(|(id, ..)| self.settled != Some(*id))?;
        let idle = self.pending.is_none() && !self.active.load(Ordering::Acquire);
        (idle && (pressed || now >= since + HOVER_DWELL)).then_some(id)
    }

    /// Whether `id`'s files written ahead (or being written) match `key`.
    pub(crate) fn has_current(&mut self, id: u64, key: &DragKey) -> bool {
        self.settle(id);
        self.prewritten.get(&id).is_some_and(|p| p.key.matches(key))
    }

    /// Collect a finished write of `id`'s files, dropping it when it
    /// failed so the next hover writes them again.
    fn settle(&mut self, id: u64) {
        let finished = matches!(self.prewritten.get(&id), Some(Prewrite { files: PrewriteFiles::Writing(h), .. }) if h.is_finished());
        if !finished {
            return;
        }
        if let Some(Prewrite { key, files: PrewriteFiles::Writing(h) }) = self.prewritten.remove(&id) {
            if let Some(paths) = joined(id, h) {
                self.prewritten.insert(id, Prewrite { key, files: PrewriteFiles::Written(paths) });
            }
        }
    }

    /// Write the files of `targets` ahead, `writers()` at a time, the
    /// hovered image first. True while writes run or wait their turn, so
    /// the caller comes back to start the next ones.
    pub(crate) fn queue_writes(
        &mut self,
        hovered: u64,
        targets: &[u64],
        key_of: impl Fn(u64) -> Option<DragKey>,
        files_of: impl Fn(u64) -> Option<DragFiles>,
    ) -> bool {
        let mut free = writers().saturating_sub(self.writes_running());
        let order = targets.iter().copied().filter(|&t| t == hovered)
            .chain(targets.iter().copied().filter(|&t| t != hovered));
        let mut more = false;
        for id in order {
            let Some(key) = key_of(id) else { continue };
            if self.has_current(id, &key) {
                more |= self.is_writing(id);
                continue;
            }
            // A stale write of these names finishes before the fresh one.
            if self.is_writing(id) || free == 0 {
                more = true;
                continue;
            }
            let Some(files) = files_of(id) else { continue };
            self.prewrite(id, key, files);
            free -= 1;
            more = true;
        }
        self.settled = (!more).then_some(hovered);
        more
    }

    fn writes_running(&self) -> usize {
        self.prewritten.values().filter(|p| matches!(&p.files, PrewriteFiles::Writing(h) if !h.is_finished())).count()
    }

    #[cfg(test)]
    pub(crate) fn prewrite_count(&self) -> usize {
        self.prewritten.len()
    }

    /// A write of `id`'s files is still running.
    pub(crate) fn is_writing(&self, id: u64) -> bool {
        self.prewritten.get(&id).is_some_and(|p| matches!(&p.files, PrewriteFiles::Writing(h) if !h.is_finished()))
    }

    /// Write `id`'s files on a worker, replacing any for another state of
    /// it. Waits while a stale write for it still runs: both write the
    /// same names, and the later one must win.
    pub(crate) fn prewrite(&mut self, id: u64, key: DragKey, files: DragFiles) {
        if self.is_writing(id) {
            return;
        }
        let handle = std::thread::spawn(move || files.write(&super::item::ExportDepth::default()));
        self.prewritten.insert(id, Prewrite { key, files: PrewriteFiles::Writing(handle) });
    }

    /// `id`'s drag files if they were written ahead for `key`, waiting for
    /// a write still running. `None` when the drag must write them now
    /// (none, stale or failed); a stale write still running is finished
    /// first so it cannot overwrite the fresh files afterwards.
    pub(crate) fn take_written(&mut self, id: u64, key: &DragKey) -> Option<Vec<PathBuf>> {
        let prewrite = self.prewritten.remove(&id)?;
        let paths = match prewrite.files {
            PrewriteFiles::Written(paths) => Some(paths),
            PrewriteFiles::Writing(h) => joined(id, h),
        };
        if !prewrite.key.matches(key) {
            return None;
        }
        let paths = paths?;
        self.prewritten.insert(id, Prewrite { key: prewrite.key, files: PrewriteFiles::Written(paths.clone()) });
        Some(paths)
    }

    /// A drag of `ids` waits for their files: each not written for its
    /// current state is written on a worker; `poll_start` says when they
    /// are all in.
    pub(crate) fn begin(&mut self, ids: Vec<u64>) {
        self.starting = Some(StartingDrag { ids, asked: HashSet::new() });
    }

    /// The waiting drag's files, once every write has landed. An image
    /// whose files are not current is written (once): `key_of` / `files_of`
    /// give its state and its files now.
    pub(crate) fn poll_start(
        &mut self,
        key_of: impl Fn(u64) -> Option<DragKey>,
        files_of: impl Fn(u64) -> Option<DragFiles>,
    ) -> DragReady {
        let Some(mut drag) = self.starting.take() else { return DragReady::Idle };
        let (mut paths, mut failed, mut waiting) = (Vec::new(), Vec::new(), false);
        for &id in &drag.ids {
            let Some(key) = key_of(id) else { continue };
            if self.is_writing(id) {
                waiting = true;
            } else if self.has_current(id, &key) {
                // Finished: taking it waits for nothing.
                match self.take_written(id, &key) {
                    Some(mut written) => paths.append(&mut written),
                    None => failed.push(id),
                }
            } else if !drag.asked.contains(&id) {
                // `writers()` at a time: each write holds a baked copy of
                // its image.
                if self.writes_running() < writers() {
                    drag.asked.insert(id);
                    if let Some(files) = files_of(id) {
                        self.prewrite(id, key, files);
                    }
                }
                waiting = true;
            } else {
                failed.push(id);
            }
        }
        if waiting {
            self.starting = Some(drag);
            return DragReady::Waiting;
        }
        DragReady::Ready { ids: drag.ids, paths, failed }
    }

    /// The button was let go before the files were in: no drag.
    pub(crate) fn cancel_start(&mut self) {
        self.starting = None;
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

    #[test]
    fn reset_works_from_a_separate_thread_via_cloned_arcs() {
        // The static `reset(&AtomicBool, &Mutex<...>)` signature exists
        // precisely so the OS drag-completion callback (which runs off-thread)
        // can call it via cloned Arcs without owning the full struct.
        // Verify by spawning a thread that does exactly that.
        let s = DragExportState::new();
        s.active.store(true, Ordering::Release);
        s.items.lock().unwrap().extend([10u64, 20, 30]);

        let active_clone = s.active.clone();
        let items_clone = s.items.clone();
        let handle = std::thread::spawn(move || {
            DragExportState::reset(&active_clone, &items_clone);
        });
        handle.join().expect("reset thread must not panic");

        assert!(!s.active.load(Ordering::Acquire), "parent must observe the off-thread store");
        assert!(s.items.lock().unwrap().is_empty(), "parent must observe the off-thread clear");
    }

    /// Sweeping the pointer down the list writes nothing; resting on a
    /// thumbnail, or pressing it, makes its files due; a drag under way
    /// holds them back.
    #[test]
    fn drag_files_are_due_after_a_rest_or_a_press() {
        let mut s = DragExportState::new();
        let t0 = Instant::now();
        s.note_hover(Some((1, false)), t0);
        assert_eq!(s.due(t0 + HOVER_DWELL / 2), None, "passing over");
        s.note_hover(Some((2, false)), t0 + HOVER_DWELL / 2);
        assert_eq!(s.due(t0 + HOVER_DWELL), None, "the rest starts again on another row");
        assert_eq!(s.due(t0 + HOVER_DWELL * 2), Some(2), "rested on");
        s.note_hover(Some((3, true)), t0 + HOVER_DWELL * 2);
        assert_eq!(s.due(t0 + HOVER_DWELL * 2), Some(3), "a press is due at once");
        s.active.store(true, Ordering::Release);
        assert_eq!(s.due(t0 + HOVER_DWELL * 3), None, "not while a drag runs");
        s.note_hover(None, t0);
        assert!(s.hover.is_none());
    }
}
