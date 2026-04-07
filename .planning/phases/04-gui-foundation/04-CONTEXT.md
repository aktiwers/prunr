# Phase 4: GUI Foundation - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

egui desktop GUI foundation — window, toolbar, image viewer, worker-thread inference, progress indicator, save/copy, keyboard shortcuts. No zoom/pan, no before/after, no batch sidebar, no reveal animation, no settings dialog — those are Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Window Layout
- **Minimal toolbar + canvas**: Thin top toolbar with [Open] [Remove BG] [Save] [Copy] buttons + Model selector dropdown
- Large image canvas area takes most of the window
- Status bar at bottom: state text ("Ready" / "Processing..." / "Done"), active backend ("CPU" / "CUDA"), image dimensions
- **Dark theme** — better for image editing, makes transparency checkerboard visible
- Default window size: 1280×800
- Remember last window size/position (store locally, Phase 6 adds full settings persistence)
- Window title: "BgPrunR" → "BgPrunR — filename.jpg" when image loaded

### Image Display
- **Empty state**: Centered drop zone hint with dashed border — "Drop an image here or press Ctrl+O"
- Adapts hint to platform: "Cmd+O" on macOS
- **Fit to window** by default — scale image to fit canvas, maintaining aspect ratio
- Dark background behind image (consistent with dark theme)

### Progress Indicator
- During inference: progress spinner/bar in status bar area with current stage name
- Toolbar buttons disabled during processing (except Escape to cancel)
- Status bar shows: "Processing... Inferring" → "Processing... Applying alpha" etc.

### Shortcut Overlay
- **Centered modal** with dark semi-transparent background
- Lists Phase 4 shortcuts only (6 core shortcuts):
  - Ctrl/Cmd+O — Open file
  - Ctrl/Cmd+R — Remove background
  - Ctrl/Cmd+S — Save
  - Ctrl/Cmd+C — Copy to clipboard
  - Escape — Cancel processing / Close overlay
  - ? — Show this help
- Press ? or Escape to dismiss
- Phase 5 will extend this list with zoom/pan/batch shortcuts

### Worker Thread Architecture
- Single long-lived worker thread communicating via std::sync::mpsc channels
- UI thread sends `WorkerMessage::ProcessImage(bytes, model)` to worker
- Worker sends `WorkerResult::Progress(stage, pct)` and `WorkerResult::Done(ProcessResult)` back
- UI thread polls via `try_recv()` in `App::update()` each frame
- Worker calls `ctx.request_repaint()` when progress updates or completes
- Escape sends `WorkerMessage::Cancel` — worker checks a cancel flag between stages

### Clipboard
- arboard crate with `wayland-data-control` feature for Wayland support
- Copy the processed RGBA image to clipboard as PNG
- Show brief status bar feedback: "Copied to clipboard"

### Claude's Discretion
- Exact egui widget layout code (ui.horizontal, ui.vertical, etc.)
- Status bar styling (colors, spacing)
- Drop zone visual design (dashed border style, text color)
- Progress bar vs spinner choice for status area
- arboard error handling details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture
- `ARCHITECTURE.md` — GUI data flow (Single Image section), state machine diagram (Empty→Loaded→Processing→Done), threading model, keyboard shortcuts platform modifier section

### Existing Code
- `crates/bgprunr-app/src/main.rs` — Current dispatch: `None => GUI stub`. Phase 4 replaces the stub with eframe::run_native
- `crates/bgprunr-app/src/cli.rs` — CLI module (stays as-is, GUI is a separate code path)
- `crates/bgprunr-app/Cargo.toml` — Needs eframe, arboard, rfd dependencies
- `crates/bgprunr-core/src/lib.rs` — Public API: process_image, OrtEngine, ModelKind, ProgressStage, ProcessResult

### Research
- `.planning/research/STACK.md` — egui/eframe 0.34.1, arboard 3.4
- `.planning/research/PITFALLS.md` — egui render-thread blocking (#3), texture re-upload (#8), Wayland clipboard (#7)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bgprunr_core::process_image()` — takes `&[u8]` image bytes, returns `ProcessResult` with progress callback
- `bgprunr_core::OrtEngine::new()` — model loading
- `bgprunr_core::formats::load_image_from_path()` — file loading
- `bgprunr_core::formats::encode_rgba_png()` — PNG encoding for save
- CLI `main.rs` dispatch pattern — `None => { /* GUI */ }` branch already exists

### Established Patterns
- `Modifiers::command` for platform-correct shortcuts (ARCHITECTURE.md)
- thiserror error enums
- Progress callback closure: `|stage: ProgressStage, pct: f32|`

### Integration Points
- `main.rs` `None` arm: replace stub with `eframe::run_native()` call
- GUI module (`gui/mod.rs`) created alongside existing `cli.rs`
- Worker thread uses the same `bgprunr_core::process_image()` that CLI uses
- `rfd` for native file dialogs (open/save)

</code_context>

<specifics>
## Specific Ideas

- The empty state drop zone should feel welcoming — the dashed border is a universal "drop here" visual cue
- Status bar is essential for showing what backend is active (users want to know if GPU is being used)
- Window title with filename matches OS conventions and helps when multiple windows are open

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-gui-foundation*
*Context gathered: 2026-04-07*
