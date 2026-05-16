---
phase: 33
plan: "05"
subsystem: gui
tags: [selection, overlay, action-bar, keyboard, brush-chip, hot-path]
dependency_graph:
  requires: [33-01, 33-03, 33-04]
  provides: [selection-overlay-render, selection-action-bar, selection-keyboard-bindings]
  affects: [adjustments_toolbar, brush_chip, canvas, app]
tech_stack:
  added: []
  patterns: [off-thread-visualization, try_recv-drain, rayon-spawn, intent-return-pattern]
key_files:
  created:
    - crates/prunr-app/src/gui/views/selection_overlay.rs
    - crates/prunr-app/src/gui/views/selection_action_bar.rs
  modified:
    - crates/prunr-app/src/gui/brush_state.rs
    - crates/prunr-app/src/gui/background_io.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/views/brush_chip.rs
    - crates/prunr-app/src/gui/views/canvas.rs
    - crates/prunr-app/src/gui/views/mod.rs
decisions:
  - "Ctrl+C with active selection routes to selection-copy not whole-result copy; prevents silent loss of clipboard intent when selection is live"
  - "Escape clears selection before dismissing modals — priority order: upscale → inpaint stroke → batch processing → selection → modal"
  - "Cut archives once and applies alpha_cut + copy_to_rgba in one history entry (not two separate archive calls)"
  - "Protect chip gated on model_uses_seg && has_selection — meaningless outside segmentation context"
metrics:
  duration: "~120 min (across two sessions, continued from context boundary)"
  completed_date: "2026-05-16"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 9
---

# Phase 33 Plan 05: Selection UI Surface Summary

Selection visualization overlay, action bar, protect chip, keyboard bindings, and selection visualization pipeline — the user-facing surface of the selection-first refactor.

## Tasks

### Task 1: BrushSettings knobs + off-thread visualization pipeline

Commit: `7be3825`

- Extended `BrushSettings` with 5 new fields: `edge_feather` (0.0), `outline_thickness` (2.0), `outline_opacity` (1.0), `fill_opacity` (0.15), `magic_confidence_threshold` (0.5). All use `#[serde(default)]` for back-compat.
- Added `SelectionOutlineResult` and `SelectionTextureResult` payload structs to `background_io.rs`.
- Added two new channel pairs (`selection_outline_{tx,rx}`, `selection_texture_{tx,rx}`) to `BackgroundIO`.
- Implemented `request_selection_visualization`: rayon worker builds `outline_polyline` + ACCENT-tinted `ColorImage::new([w,h], pixels)`. Hash-stale-discard guard in drain.
- Added `commit_selection_and_dispatch` to `app.rs` as single-source-of-truth pairing commit + model dispatch + visualization spawn.
- Wired drain for both new channels in `drain_background_channels` (hash guard: drops if `item.selection_hash != result.hash`).
- Updated `canvas.rs` stroke commit to use `commit_selection_and_dispatch`.
- 4 unit tests covering default field values; serde back-compat test updated.

### Task 2: Selection overlay + action bar + UI wiring

Commit: `77699a4`

- `selection_overlay.rs`: 60 Hz zero-alloc render. Fill paints `selection_texture` with `fill_opacity` as tint alpha. Outline transforms `Vec<(u32,u32)>` to screen `Pos2` via `ox + px * iw` and draws `egui::Shape::line`. No I/O, no decode, no GPU upload. The per-frame `Vec<Pos2>` allocation (O(N) over pre-existing `Arc<Vec>`) is acceptable per CLAUDE.md.
- `selection_action_bar.rs`: Delete/Copy/Cut/Invert/Clear with 1px ACCENT top border, vertical separator between Cut and Invert, DESTRUCTIVE border on Delete. Keyboard hints in tooltips.
- `adjustments_toolbar.rs`: Added `selection_action: Option<SelectionAction>` and `protect_selection: Option<bool>` to `ToolbarChange`. Render accepts `has_selection` and `protect_selection`. Action bar rendered below all rows when selection is active (skipped in upscale mode). Protect chip rendered in right-aligned cluster when `model_uses_seg && has_selection`.
- `brush_chip.rs`: Added Selection section with 4 slider rows (edge feather, outline thickness, outline opacity, fill opacity) using verbatim copy from 33-UI-SPEC.md.
- `app.rs`: `handle_selection_action` implements Delete/Copy/Cut/Invert/Clear. `apply_toolbar_change` routes `selection_action` and `protect_selection` intents. `handle_keyboard_shortcuts` routes Del → Delete, Ctrl+X → Cut. `apply_cancel_shortcut` extended: Escape clears selection before dismissing modals. Ctrl+C routes to selection copy when selection is live.
- 3 pinning tests: `delete_action_alpha_cuts_result_and_pushes_history`, `invert_action_replaces_mask_with_inverse`, `cut_action_pushes_only_one_history_marker`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `ort::session::run_options` is a private module**
- Found during: Task 1 (build)
- Issue: `use ort::session::run_options::{NoSelectedOutputs, RunOptions}` failed — `run_options` is private in ort 2.0.0-rc.12
- Fix: Changed to `use ort::session::{NoSelectedOutputs, RunOptions}`
- Files modified: `crates/prunr-core/src/upscale/mod.rs`

**2. [Rule 1 - Bug] Missing `terminate` argument in `upscale_two_pass_with_engine`**
- Found during: Task 1 (build)
- Issue: Call site had 4 arguments but signature requires 5 (added `terminate: Option<Arc<AtomicBool>>`)
- Fix: Added `None` as the 5th argument
- Files modified: `crates/prunr-core/src/upscale/mod.rs`, `crates/prunr-app/src/gui/processor.rs`

**3. [Rule 1 - Bug] `egui::ColorImage` struct literal missing `source_size` field**
- Found during: Task 1 (build)
- Issue: egui 0.34.1 requires `source_size: Vec2` in struct literal; used `ColorImage { size, pixels }` which failed
- Fix: Replaced with `ColorImage::new([w, h], pixels)` constructor

## Self-Check

Files created:
- `/media/bolli/5c607cb1-5a5c-4c1b-a74a-d3060d86c222/Coding/Vibe/Private/BgPrunr/crates/prunr-app/src/gui/views/selection_overlay.rs` — FOUND
- `/media/bolli/5c607cb1-5a5c-4c1b-a74a-d3060d86c222/Coding/Vibe/Private/BgPrunr/crates/prunr-app/src/gui/views/selection_action_bar.rs` — FOUND
- `/media/bolli/5c607cb1-5a5c-4c1b-a74a-d3060d86c222/Coding/Vibe/Private/BgPrunr/.planning/phases/33-magic-brush/33-05-SUMMARY.md` — FOUND

Commits:
- `7be3825` (Task 1) — FOUND
- `77699a4` (Task 2) — FOUND

## Self-Check: PASSED
