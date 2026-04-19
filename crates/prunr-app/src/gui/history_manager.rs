//! Per-item history management for `BatchItem` undo/redo.
//!
//! Two independent stacks live on each `BatchItem`:
//!
//! - **Result history** (`history` / `redo_stack`) — pixel-level undo for
//!   Process operations. Walked by Ctrl+Z / Ctrl+Y.
//! - **Preset history** (`preset_undo_stack` / `preset_redo_stack`) — the
//!   `(settings, applied_preset)` snapshot taken before each preset apply.
//!   Walked by Ctrl+Shift+Z / Ctrl+Shift+Y. Independent of the result
//!   history so an accidental preset pick can be rolled back without
//!   touching pixels.
//!
//! `HistoryManager` is a unit struct — every method is a free-function
//! over `&mut BatchItem`. State stays on `BatchItem` (RAII cleanup on
//! removal); this module owns the policy, not the data.
//!
//! Reset of UI-side caches (textures, thumbnails, decode flags) is the
//! caller's responsibility — those aren't history concerns.

use std::collections::VecDeque;

use super::item::{BatchItem, BatchStatus, HistoryEntry, PresetSnapshot};

/// Bound on the per-item preset undo/redo stacks. Each entry is ~100 bytes,
/// so 20 × ~100 = ~2 KB per image — a rounding error next to the result
/// history's megabyte-scale entries.
const PRESET_HISTORY_DEPTH: usize = 20;

/// Direction for preset history walk.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum HistoryDir {
    Undo,
    Redo,
}

/// Push onto a bounded undo/redo stack, dropping the oldest entry if it
/// exceeds `PRESET_HISTORY_DEPTH`.
fn push_bounded(stack: &mut VecDeque<PresetSnapshot>, snap: PresetSnapshot) {
    stack.push_back(snap);
    while stack.len() > PRESET_HISTORY_DEPTH {
        stack.pop_front();
    }
}

pub(crate) struct HistoryManager;

impl HistoryManager {
    /// True if `undo_result` on this item would change anything.
    pub(crate) fn can_undo(item: &BatchItem) -> bool {
        item.status == BatchStatus::Done
            && (!item.history.is_empty() || item.result_rgba.is_some())
    }

    /// True if `redo_result` on this item would change anything.
    pub(crate) fn can_redo(item: &BatchItem) -> bool {
        !item.redo_stack.is_empty()
    }

    /// Seed history with the source RGBA as an initial no-recipe entry.
    /// No-op if either stack is non-empty (history already exists) or the
    /// source hasn't been decoded yet.
    pub(crate) fn seed_with_source(item: &mut BatchItem) {
        if !item.history.is_empty() || !item.redo_stack.is_empty() {
            return;
        }
        if let Some(ref src_rgba) = item.source_rgba {
            item.history.push_back(HistoryEntry::new(src_rgba.clone(), None));
        }
    }

    /// Before reprocessing a Done item: archive the current result + recipe
    /// onto history so Ctrl+Z can restore it. Drain redo (a fresh process
    /// branches the timeline). Enforce `max_depth` by demoting the oldest
    /// entries — `HistoryEntry::cleanup` releases their backing Tier-3 disk
    /// files. In chain mode, keep `result_rgba` populated so the next chain
    /// step has its input.
    pub(crate) fn archive_current_result(item: &mut BatchItem, max_depth: usize, chain_mode: bool) {
        if item.status != BatchStatus::Done {
            return;
        }
        if let Some(current) = item.result_rgba.take() {
            if chain_mode {
                item.result_rgba = Some(current.clone());
            }
            item.history.push_back(HistoryEntry::new(current, item.applied_recipe.clone()));
            while item.history.len() > max_depth {
                if let Some(old) = item.history.pop_front() {
                    old.cleanup();
                }
            }
        }
        for entry in item.redo_stack.drain(..) {
            entry.cleanup();
        }
    }

    /// Walk the result history backward. Returns true iff anything was
    /// undone. Updates `result_rgba`, `applied_recipe`, `status` accordingly;
    /// pushes the current result onto `redo_stack` so Ctrl+Y can restore it.
    ///
    /// Precondition: caller checks `item.status == Done` if they want to
    /// gate by status (this method does the check internally and returns
    /// false if not Done — keeps callers branch-light).
    pub(crate) fn undo_result(item: &mut BatchItem) -> bool {
        if item.status != BatchStatus::Done {
            return false;
        }
        let current_recipe = item.applied_recipe.take();
        if item.history.is_empty() {
            // No history to walk back into; capture current as redo target,
            // then transition to Pending (the unprocessed source state).
            if let Some(current) = item.result_rgba.take() {
                item.redo_stack.push_back(HistoryEntry::new(current, current_recipe));
            }
            item.status = BatchStatus::Pending;
            item.result_rgba = None;
            return true;
        }
        if let Some(current) = item.result_rgba.take() {
            item.redo_stack.push_back(HistoryEntry::new(current, current_recipe));
        }
        if let Some(entry) = item.history.pop_back() {
            let (slot, recipe) = entry.into_parts();
            item.applied_recipe = recipe;
            item.result_rgba = slot.into_rgba();
        }
        // If decompression failed or the popped entry was the source seed
        // (history now empty), settle on the unprocessed state.
        if item.result_rgba.is_none() || item.history.is_empty() {
            item.status = BatchStatus::Pending;
            item.result_rgba = None;
            item.applied_recipe = None;
        }
        true
    }

    /// Walk the redo stack forward. Returns true iff anything was redone.
    /// Inverse of `undo_result`: pushes current to `history`, restores from
    /// redo top, transitions status to Done.
    pub(crate) fn redo_result(item: &mut BatchItem) -> bool {
        if item.redo_stack.is_empty() {
            return false;
        }
        let current_recipe = item.applied_recipe.take();
        if let Some(current) = item.result_rgba.take() {
            item.history.push_back(HistoryEntry::new(current, current_recipe));
        }
        if let Some(entry) = item.redo_stack.pop_back() {
            let (slot, recipe) = entry.into_parts();
            item.applied_recipe = recipe;
            item.result_rgba = slot.into_rgba();
        }
        item.status = BatchStatus::Done;
        true
    }

    /// Record a pre-apply snapshot before a preset is applied. The snapshot
    /// goes onto `preset_undo_stack` with depth enforcement; `preset_redo_stack`
    /// is cleared because a fresh apply branches the preset timeline.
    pub(crate) fn push_preset(item: &mut BatchItem, snap: PresetSnapshot) {
        push_bounded(&mut item.preset_undo_stack, snap);
        item.preset_redo_stack.clear();
    }

    /// Walk the preset history. Pops one snapshot from the directional
    /// stack, pushes the current `(settings, applied_preset)` onto the
    /// opposite stack, applies the popped snapshot to the item. Returns
    /// true iff a swap happened. Caller is responsible for deciding whether
    /// to reprocess (typical: only if `item.status == Done`).
    ///
    /// Side-effect: invalidates the edge cache if `line_mode` changed,
    /// because DexiNed's input depends on whether segmentation ran first.
    pub(crate) fn swap_preset(item: &mut BatchItem, dir: HistoryDir) -> bool {
        let popped = match dir {
            HistoryDir::Undo => item.preset_undo_stack.pop_back(),
            HistoryDir::Redo => item.preset_redo_stack.pop_back(),
        };
        let Some(snapshot) = popped else { return false };
        let current = PresetSnapshot {
            settings: item.settings,
            applied_preset: item.applied_preset.clone(),
        };
        match dir {
            HistoryDir::Undo => push_bounded(&mut item.preset_redo_stack, current),
            HistoryDir::Redo => push_bounded(&mut item.preset_undo_stack, current),
        }
        let old_line_mode = item.settings.line_mode;
        item.settings = snapshot.settings;
        item.applied_preset = snapshot.applied_preset;
        if item.settings.line_mode != old_line_mode {
            item.invalidate_edge_cache();
        }
        true
    }
}

#[cfg(test)]
mod preset_stack_tests {
    //! Tests for the `push_bounded` helper. Kept inline so the helper can
    //! stay private to this module.
    use super::*;
    use crate::gui::item_settings::ItemSettings;

    /// Build a single-attribute PresetSnapshot. `pub(super)` so the sibling
    /// `tests` module can reuse it (deduped per /simplify finding).
    pub(super) fn snap(gamma: f32) -> PresetSnapshot {
        let mut s = ItemSettings::default();
        s.gamma = gamma;
        PresetSnapshot { settings: s, applied_preset: String::new() }
    }

    #[test]
    fn push_bounded_drops_oldest_when_exceeding_depth() {
        let mut stack: VecDeque<PresetSnapshot> = VecDeque::new();
        for i in 0..(PRESET_HISTORY_DEPTH + 5) {
            push_bounded(&mut stack, snap(i as f32));
        }
        assert_eq!(stack.len(), PRESET_HISTORY_DEPTH);
        // Oldest 5 entries dropped; newest entry is the last pushed.
        assert_eq!(stack.front().unwrap().settings.gamma, 5.0);
        assert_eq!(
            stack.back().unwrap().settings.gamma,
            (PRESET_HISTORY_DEPTH + 4) as f32,
        );
    }

    #[test]
    fn push_bounded_below_depth_keeps_every_entry() {
        let mut stack: VecDeque<PresetSnapshot> = VecDeque::new();
        push_bounded(&mut stack, snap(1.0));
        push_bounded(&mut stack, snap(2.0));
        push_bounded(&mut stack, snap(3.0));
        assert_eq!(stack.len(), 3);
        assert_eq!(stack.front().unwrap().settings.gamma, 1.0);
        assert_eq!(stack.back().unwrap().settings.gamma, 3.0);
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the `HistoryManager` methods themselves — the policy layer
    //! over per-item history stacks. Build a `BatchItem` fixture, call the
    //! methods, assert state.
    use super::*;
    use super::preset_stack_tests::snap; // shared with the push_bounded tests
    use crate::gui::item::ImageSource;
    use crate::gui::item_settings::ItemSettings;
    use std::sync::Arc;

    fn fixture(id: u64) -> BatchItem {
        BatchItem::new(
            id,
            "test.png".into(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (10, 10),
            ItemSettings::default(),
            String::new(),
        )
    }

    fn rgba(r: u8) -> Arc<image::RgbaImage> {
        Arc::new(image::RgbaImage::from_pixel(2, 2, image::Rgba([r, 0, 0, 255])))
    }

    // ── can_undo / can_redo ─────────────────────────────────────────────

    #[test]
    fn can_undo_false_when_pending() {
        let mut item = fixture(1);
        item.status = BatchStatus::Pending;
        item.result_rgba = Some(rgba(7));
        assert!(!HistoryManager::can_undo(&item));
    }

    #[test]
    fn can_undo_true_on_done_with_result() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        item.result_rgba = Some(rgba(7));
        assert!(HistoryManager::can_undo(&item));
    }

    #[test]
    fn can_undo_true_on_done_with_nonempty_history() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        item.history.push_back(HistoryEntry::new(rgba(1), None));
        // result_rgba is None but history is non-empty → still undoable
        assert!(HistoryManager::can_undo(&item));
    }

    #[test]
    fn can_redo_false_until_undo() {
        let mut item = fixture(1);
        assert!(!HistoryManager::can_redo(&item));
        item.redo_stack.push_back(HistoryEntry::new(rgba(1), None));
        assert!(HistoryManager::can_redo(&item));
    }

    // ── seed_with_source ────────────────────────────────────────────────

    #[test]
    fn seed_with_source_no_op_if_history_non_empty() {
        let mut item = fixture(1);
        item.history.push_back(HistoryEntry::new(rgba(99), None));
        item.source_rgba = Some(rgba(0));
        let before = item.history.len();
        HistoryManager::seed_with_source(&mut item);
        assert_eq!(item.history.len(), before, "should not push when history non-empty");
    }

    #[test]
    fn seed_with_source_no_op_if_source_rgba_missing() {
        let mut item = fixture(1);
        // source_rgba defaults to None
        HistoryManager::seed_with_source(&mut item);
        assert!(item.history.is_empty(), "should not push without source_rgba");
    }

    #[test]
    fn seed_with_source_pushes_when_both_stacks_empty_and_source_present() {
        let mut item = fixture(1);
        item.source_rgba = Some(rgba(7));
        HistoryManager::seed_with_source(&mut item);
        assert_eq!(item.history.len(), 1);
        // Recipe is None for the source-seed entry — it's the unprocessed source.
        assert!(item.history.front().unwrap().recipe.is_none());
    }

    // ── archive_current_result ──────────────────────────────────────────

    #[test]
    fn archive_current_result_no_op_if_status_not_done() {
        let mut item = fixture(1);
        item.status = BatchStatus::Pending;
        item.result_rgba = Some(rgba(5));
        HistoryManager::archive_current_result(&mut item, 10, false);
        assert_eq!(item.history.len(), 0);
        assert!(item.result_rgba.is_some(), "result_rgba should be untouched");
    }

    #[test]
    fn archive_current_result_pushes_to_history_and_clears_redo() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        item.result_rgba = Some(rgba(5));
        item.redo_stack.push_back(HistoryEntry::new(rgba(99), None));

        HistoryManager::archive_current_result(&mut item, 10, false);

        assert_eq!(item.history.len(), 1, "current pushed onto history");
        assert!(item.redo_stack.is_empty(), "redo cleared on fresh process branch");
        assert!(item.result_rgba.is_none(), "non-chain mode releases result_rgba");
    }

    #[test]
    fn archive_current_result_chain_mode_keeps_result_rgba_for_next_step() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        item.result_rgba = Some(rgba(5));

        HistoryManager::archive_current_result(&mut item, 10, true);

        assert_eq!(item.history.len(), 1);
        assert!(item.result_rgba.is_some(), "chain mode keeps result_rgba populated");
    }

    #[test]
    fn archive_current_result_enforces_max_depth() {
        let mut item = fixture(1);
        // Pre-fill history to depth 5.
        for _ in 0..5 {
            item.history.push_back(HistoryEntry::new(rgba(0), None));
        }
        item.status = BatchStatus::Done;
        item.result_rgba = Some(rgba(99));

        HistoryManager::archive_current_result(&mut item, 3, false);

        // After: oldest popped to fit max_depth=3; latest pushed.
        assert_eq!(item.history.len(), 3);
    }

    // ── undo_result ─────────────────────────────────────────────────────

    #[test]
    fn undo_result_returns_false_if_status_not_done() {
        let mut item = fixture(1);
        item.status = BatchStatus::Pending;
        assert!(!HistoryManager::undo_result(&mut item));
    }

    #[test]
    fn undo_result_empty_history_transitions_to_pending() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        item.result_rgba = Some(rgba(7));

        assert!(HistoryManager::undo_result(&mut item));
        assert_eq!(item.status, BatchStatus::Pending);
        assert!(item.result_rgba.is_none());
        assert_eq!(item.redo_stack.len(), 1, "current went to redo stack");
    }

    #[test]
    fn undo_then_redo_round_trip_preserves_pixels() {
        let mut item = fixture(1);
        item.status = BatchStatus::Done;
        let original_pixels = rgba(42);
        item.result_rgba = Some(original_pixels.clone());
        // History layout: [source_seed (oldest), prior_result]. Two entries so
        // that after undo pops `prior_result`, history is non-empty and
        // status stays Done. (Single-entry history is the source-seed case
        // and walks back to Pending — covered by `undo_result_empty_history_*`.)
        item.history.push_back(HistoryEntry::new(rgba(0), None));   // source seed
        item.history.push_back(HistoryEntry::new(rgba(11), None));  // prior real result

        // Undo: pops prior_result onto current, current_42 goes to redo.
        assert!(HistoryManager::undo_result(&mut item));
        assert_eq!(item.status, BatchStatus::Done);
        let after_undo = item.result_rgba.as_ref().expect("undo restored a result");
        assert_eq!(after_undo.get_pixel(0, 0).0, [11, 0, 0, 255]);

        // Redo: original result restored byte-for-byte.
        assert!(HistoryManager::redo_result(&mut item));
        assert_eq!(item.status, BatchStatus::Done);
        let after_redo = item.result_rgba.as_ref().expect("redo restored result");
        assert_eq!(after_redo.as_raw(), original_pixels.as_raw());
    }

    // ── redo_result ─────────────────────────────────────────────────────

    #[test]
    fn redo_result_returns_false_if_redo_empty() {
        let mut item = fixture(1);
        // redo_stack empty by default
        assert!(!HistoryManager::redo_result(&mut item));
    }

    // ── push_preset / swap_preset ───────────────────────────────────────

    #[test]
    fn push_preset_adds_to_undo_and_clears_redo() {
        let mut item = fixture(1);
        item.preset_redo_stack.push_back(snap(99.0));
        HistoryManager::push_preset(&mut item, snap(1.5));
        assert_eq!(item.preset_undo_stack.len(), 1);
        assert_eq!(item.preset_undo_stack.front().unwrap().settings.gamma, 1.5);
        assert!(item.preset_redo_stack.is_empty(), "fresh apply branches the timeline");
    }

    #[test]
    fn swap_preset_returns_false_when_stack_empty() {
        let mut item = fixture(1);
        assert!(!HistoryManager::swap_preset(&mut item, HistoryDir::Undo));
        assert!(!HistoryManager::swap_preset(&mut item, HistoryDir::Redo));
    }

    #[test]
    fn swap_preset_undo_pops_one_and_pushes_current_to_redo() {
        let mut item = fixture(1);
        item.settings.gamma = 1.5;
        item.preset_undo_stack.push_back(snap(0.5));

        assert!(HistoryManager::swap_preset(&mut item, HistoryDir::Undo));

        // Snapshot from undo stack applied.
        assert_eq!(item.settings.gamma, 0.5);
        // Pre-swap state pushed to redo stack.
        assert_eq!(item.preset_redo_stack.len(), 1);
        assert_eq!(item.preset_redo_stack.front().unwrap().settings.gamma, 1.5);
    }

    #[test]
    fn swap_preset_invalidates_edge_cache_on_line_mode_change() {
        use crate::gui::item_settings::ItemSettings;
        use prunr_core::LineMode;
        let mut item = fixture(1);
        item.settings.line_mode = LineMode::Off;
        item.cached_edge_mask = Some((Arc::new(image::GrayImage::new(1, 1)), 0));
        let mut snap_with_edges = ItemSettings::default();
        snap_with_edges.line_mode = LineMode::EdgesOnly;
        item.preset_undo_stack.push_back(PresetSnapshot {
            settings: snap_with_edges,
            applied_preset: String::new(),
        });

        assert!(HistoryManager::swap_preset(&mut item, HistoryDir::Undo));
        assert_eq!(item.settings.line_mode, LineMode::EdgesOnly);
        assert!(item.cached_edge_mask.is_none(), "line_mode change must invalidate edge cache");
    }
}
