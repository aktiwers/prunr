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
    /// Seed history with the source RGBA as an initial no-recipe entry.
    /// No-op if either stack is non-empty (history already exists) or the
    /// source hasn't been decoded yet.
    pub fn seed_with_source(item: &mut BatchItem) {
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
    pub fn archive_current_result(item: &mut BatchItem, max_depth: usize, chain_mode: bool) {
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
    pub fn undo_result(item: &mut BatchItem) -> bool {
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
    pub fn redo_result(item: &mut BatchItem) -> bool {
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
    pub fn push_preset(item: &mut BatchItem, snap: PresetSnapshot) {
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
    pub fn swap_preset(item: &mut BatchItem, dir: HistoryDir) -> bool {
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

    fn snap(gamma: f32) -> PresetSnapshot {
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
