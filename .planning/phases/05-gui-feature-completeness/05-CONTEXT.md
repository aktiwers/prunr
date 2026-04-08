# Phase 5: GUI Feature Completeness - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Full interactive experience for the egui desktop GUI: zoom/pan for edge inspection, before/after comparison toggle, batch sidebar for multi-image workflows, settings dialog with model selection and parallelism control, and a mask-aware reveal animation on background removal completion. No new input formats, no settings file persistence (Phase 6), no new inference capabilities.

</domain>

<decisions>
## Implementation Decisions

### Zoom & Pan
- Scroll-wheel zoom centers on cursor position (Photoshop/Figma style)
- Exponential zoom steps: each scroll tick multiplies by ~1.1x
- Zoom range: 10% to 2000% (supports pixel-level edge inspection)
- Space+drag to pan the image
- Ctrl/Cmd+0 fits image to window; if already fitted, toggles back to previous zoom level
- Ctrl/Cmd+1 shows 1:1 pixel size; if already at 1:1, toggles back to previous zoom level
- Current zoom percentage displayed in the status bar alongside image dimensions

### Before/After Toggle
- B key performs instant swap between original (source_texture) and result (result_texture)
- No split-slider — simple toggle is sufficient for edge comparison
- When viewing result, checkerboard pattern shows behind transparent areas (already implemented in canvas.rs)
- When viewing original in Done state, no checkerboard — just the original image on dark background

### Reveal Animation
- Mask-aware dissolve: only background pixels (alpha < threshold in result mask) dissolve away
- Subject stays 100% sharp throughout the animation — never goes semi-transparent
- Duration: 0.5–1s, smooth alpha interpolation on background regions
- Checkerboard gradually appears through dissolving background areas
- Any key press or mouse click immediately skips animation to final state
- No "press to skip" hint text displayed during animation
- Settings toggle to enable/disable the animation entirely (default: enabled)
- Animation plays once per processing completion, not on every before/after toggle

### Batch Sidebar
- Left side panel with vertical thumbnail strip
- Auto-shows when 2+ images are loaded, hidden for single image
- Can be manually toggled (sidebar visibility shortcut — Claude's discretion on key)
- Thumbnail for each image with corner status icon overlay: ○ pending, ◆ processing, ✓ done
- Clicking a thumbnail switches main canvas to that image
- When switching: show result if image is processed, show original if not yet processed
- B key toggles between original/result for the currently selected image (if processed)
- Drag-to-reorder items in sidebar
- [ and ] keys navigate to previous/next image in the sidebar list
- Each image's result is cached — switching does not re-run inference
- Dropping multiple images populates the sidebar queue

### Batch Processing
- "Process All" action processes all unprocessed images in the queue
- Uses batch_process() from prunr-core with rayon thread pool
- Parallelism level comes from settings (default: half available cores)
- Individual image progress shown via status icons in sidebar
- Overall batch progress in status bar

### Settings Dialog
- Centered modal overlay (same pattern as shortcuts overlay — dark semi-transparent background)
- Escape or X button closes it
- Ctrl/Cmd+, opens it
- Settings included:
  - Model selection: dropdown (silueta / u2net)
  - Auto-remove on import: checkbox (default: off)
  - Parallel inference jobs: slider from 1 to num_cpus::get(), default half cores
  - Reveal animation: checkbox to enable/disable (default: enabled)
  - Active inference backend: read-only label showing "CUDA (GPU)" / "CPU" / etc.
- All settings serve as defaults and must be remembered across application restarts
- Settings struct should be serializable (serde) — Phase 6 wires up file load/save, but the structure and defaults are established now

### Claude's Discretion
- Sidebar width and thumbnail dimensions
- Sidebar toggle keyboard shortcut
- Exact animation easing curve
- Drag-to-reorder visual feedback style (ghost item, insertion line, etc.)
- Settings dialog dimensions and internal layout
- Zoom step multiplier value (approximately 1.1x but exact value flexible)
- How batch "Process All" is triggered (toolbar button, shortcut, or context menu)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture
- `ARCHITECTURE.md` — GUI data flow, state machine diagram, threading model, keyboard shortcuts platform modifier section

### Existing GUI Code
- `crates/prunr-app/src/gui/app.rs` — PrunrApp struct with source_texture, result_texture, worker channels, state machine. All Phase 5 features extend this.
- `crates/prunr-app/src/gui/state.rs` — AppState enum (Empty/Loaded/Processing/Done). Needs extension for animation and batch states.
- `crates/prunr-app/src/gui/views/canvas.rs` — render_image_centered(), draw_checkerboard(), per-state rendering. Zoom/pan/before-after changes happen here.
- `crates/prunr-app/src/gui/worker.rs` — WorkerMessage/WorkerResult enums, spawn_worker(). Batch processing extends this.
- `crates/prunr-app/src/gui/theme.rs` — Color/spacing/layout constants. New UI elements must use these.
- `crates/prunr-app/src/gui/views/toolbar.rs` — Toolbar buttons. Needs batch-related buttons.
- `crates/prunr-app/src/gui/views/statusbar.rs` — Status bar. Needs zoom % display.
- `crates/prunr-app/src/gui/views/shortcuts.rs` — Shortcuts overlay pattern. Settings dialog follows same modal pattern.

### Core API
- `crates/prunr-core/src/lib.rs` — Public API: process_image, batch_process, OrtEngine, ModelKind, ProgressStage, ProcessResult
- `crates/prunr-core/src/batch.rs` — batch_process() with rayon, progress callback, indexed results

### Prior Phase Context
- `.planning/phases/04-gui-foundation/04-CONTEXT.md` — Phase 4 decisions: dark theme, fit-to-window default, worker thread pattern, toolbar layout, status bar format

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `canvas::draw_checkerboard()` — Already renders checkerboard behind transparent result. Reuse for zoom/pan views.
- `canvas::render_image_centered()` — Calculates fit-to-window scale. Will be reworked to support zoom/pan offset but the scaling logic is a starting point.
- `shortcuts::render()` — Modal overlay pattern with dark background. Clone pattern for settings dialog.
- `theme::OVERLAY_BG`, `theme::OVERLAY_BORDER` — Overlay visual constants already defined.
- `worker::spawn_worker()` — Single-image worker. Extend for batch or spawn additional workers.
- `app::PrunrApp` fields: `source_texture`, `result_texture`, `result_rgba` — Before/after toggle switches which texture renders.

### Established Patterns
- State machine in `state.rs` — All behavior gated on AppState. New states or sub-states needed for animation and batch.
- Worker communication via mpsc channels — UI polls with try_recv() each frame. Batch progress uses same pattern.
- Theme constants for all colors/spacing — New UI elements (sidebar, settings) must follow this.
- `pub(crate)` visibility for app fields accessed by view modules.
- OrtEngine created per invocation (Phase 2/3/4 pattern) — batch workers each create their own engine.

### Integration Points
- `app.rs logic()` — Keyboard shortcut handling section needs B, [, ], Ctrl+0, Ctrl+1, Space+drag, Ctrl+,
- `app.rs ui()` — Layout: add left panel for sidebar before CentralPanel
- `canvas.rs render()` — All four state renderers need zoom/pan transform applied
- `worker.rs` — Batch processing messages and results alongside existing single-image flow
- `statusbar.rs` — Add zoom percentage display

</code_context>

<specifics>
## Specific Ideas

- Zoom-toward-cursor is essential for edge inspection after background removal — users need to check if edges are clean
- Ctrl+0/Ctrl+1 toggle behavior (fit ↔ previous zoom) gives quick back-and-forth for overview vs detail
- Sidebar auto-show on multi-image keeps the single-image workflow clean and uncluttered
- When switching images in sidebar: show result if processed, original if not — natural workflow expectation
- Settings must persist across restarts — define serializable struct now, Phase 6 handles actual file I/O
- Reveal animation toggle in settings gives users who find it distracting a way to disable it

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-gui-feature-completeness*
*Context gathered: 2026-04-07*
