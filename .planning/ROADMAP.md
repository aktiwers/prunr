# Roadmap: Prunr

## Overview

Prunr ships in six phases that follow the hard dependency graph imposed by the architecture: workspace scaffolding must exist before any crate compiles; the core inference engine must be correct and tested before any presentation layer exists; the CLI exercises the core API before the GUI adds threading complexity; the GUI is built in two passes (threading architecture first, then feature completeness); and distribution verification closes the loop on clean-machine reliability. Every requirement maps to the phase that first makes it possible to deliver it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Workspace Scaffolding** - Cargo workspace, CI matrix, model crate, and build pipeline foundation (completed 2026-04-06)
- [x] **Phase 2: Core Inference Engine** - ONNX inference pipeline verified correct against rembg, GPU + CPU fallback, batch API (completed 2026-04-06)
- [ ] **Phase 3: CLI Binary** - Full-featured CLI exercising the core API with batch, parallelism, and model selection
- [ ] **Phase 4: GUI Foundation** - egui app with worker-thread architecture, drag-and-drop, progress, save, copy, keyboard shortcuts
- [x] **Phase 5: GUI Feature Completeness** - Before/after view, zoom/pan, batch sidebar, settings dialog, reveal animation (completed 2026-04-07)
- [ ] **Phase 6: Distribution and Packaging** - Single-binary verification on clean VMs, SVG input, settings persistence, release artifacts
- [ ] **Phase 7: Iterative Processing** - Chain mode: process result of previous step instead of original, full undo/redo history stack, memory management

## Phase Details

### Phase 1: Workspace Scaffolding
**Goal**: The Cargo workspace structure, CI pipeline, and model embedding foundation exist — any developer can clone, build, and run a placeholder binary on all three platforms
**Depends on**: Nothing (first phase)
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04
**Success Criteria** (what must be TRUE):
  1. `cargo build` succeeds from the workspace root on Linux, macOS (x86_64 + aarch64), and Windows x86_64 without any manual setup steps
  2. GitHub Actions CI builds and tests all three platform targets in a single workflow run
  3. Model bytes are embedded via `include-bytes-zstd` in a dedicated `prunr-models` crate; a development feature flag loads models from the filesystem instead to avoid recompilation cost during development
  4. `cargo-dist` release pipeline produces a per-platform binary artifact in CI (even if the binary is a placeholder)
**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — Workspace manifests and crate stubs (Cargo.toml, prunr-core traits, prunr-models feature gate, placeholder binary)
- [ ] 01-02-PLAN.md — xtask fetch-models with SHA256 verification
- [ ] 01-03-PLAN.md — GitHub Actions CI matrix workflow (4 native platform targets)
- [ ] 01-04-PLAN.md — cargo-dist release pipeline + CI human verification gate

### Phase 2: Core Inference Engine
**Goal**: Users (and the CLI/GUI) can call `process_image()` and receive a pixel-accurate transparent PNG whose mask matches rembg Python output on the same input, with GPU used automatically when available
**Depends on**: Phase 1
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, LOAD-03, LOAD-04
**Success Criteria** (what must be TRUE):
  1. A reference test compares Prunr's output mask pixel-by-pixel against rembg Python output on three known test images and passes — this is a hard gate before any GUI or CLI work ships
  2. `process_image()` runs to completion on silueta and u2net models on CPU with no panic or data corruption
  3. When CUDA/CoreML/DirectML hardware is present, the active execution provider name is logged at session initialization and queryable via the public API (not silently falling back without notice)
  4. Calling `process_image()` on an image exceeding 8000px in either dimension returns a warning/prompt result rather than silently processing a huge tensor
  5. `batch_process()` accepts a progress callback and processes multiple images using a rayon thread pool with no thread oversubscription against ORT's intra-op pool
**Plans**: 6 plans

Plans:
- [ ] 02-01-PLAN.md — Types foundation: CoreError variants, ModelKind, ProgressStage, ProcessResult, Cargo deps
- [ ] 02-02-PLAN.md — Pure pipeline modules: preprocess.rs (rembg-exact Lanczos3 + max-pixel norm), postprocess.rs (min-max, no sigmoid), formats.rs
- [ ] 02-03-PLAN.md — OrtEngine session management + process_image() orchestration with progress callback
- [ ] 02-04-PLAN.md — batch_process() with rayon thread pool and ORT intra-op thread balancing
- [ ] 02-05-PLAN.md — Reference test infrastructure: scripts/generate_references.py, test image directories
- [ ] 02-06-PLAN.md — Integration test suite: CORE-05 pixel-accuracy hard gate + all requirement tests

### Phase 3: CLI Binary
**Goal**: A user with no GUI can process single images and batches via the terminal, select models, tune parallelism, and get correct exit codes — the full core API is exercised under real scripting conditions
**Depends on**: Phase 2
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05
**Success Criteria** (what must be TRUE):
  1. `prunr input.jpg -o output.png` produces a transparent PNG with the background removed and exits with code 0
  2. `prunr *.jpg --output-dir ./results/` processes all matching files in parallel and exits with code 0 (all success), 1 (total failure), or 2 (partial failure)
  3. `--model silueta` and `--model u2net` both select the correct embedded model and produce visibly different quality on a complex image
  4. `--jobs N` controls rayon parallelism and the binary does not spawn more threads than requested
  5. A progress bar (indicatif) updates per image during batch processing so the user knows the job is running
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Cargo deps + clap structs (Cli, Commands, RemoveArgs, CliModel, LargeImagePolicy) + process_image_unchecked core extension
- [ ] 03-02-PLAN.md — main.rs dispatch + run_remove() single-image and batch execution paths with indicatif progress + exit codes
- [ ] 03-03-PLAN.md — Human verification checkpoint: real-image end-to-end test of all CLI flags and exit codes

### Phase 4: GUI Foundation
**Goal**: A user can open the GUI, load an image by drag-and-drop or file picker, trigger inference, watch a progress indicator, and save or copy the result — the worker-thread threading architecture is in place and the UI never freezes
**Depends on**: Phase 3
**Requirements**: LOAD-01, LOAD-02, OUT-01, OUT-02, UX-01, UX-03, UX-04
**Success Criteria** (what must be TRUE):
  1. Dropping an image file onto the app window loads it without any dialog; using Ctrl/Cmd+O opens a file picker — both paths result in the image appearing in the viewer
  2. Pressing Ctrl/Cmd+R (or the Remove button) dispatches inference to a background worker thread; the UI remains responsive and shows a progress spinner for the full duration of inference — the window never freezes or appears crashed
  3. When inference completes, pressing Ctrl/Cmd+S opens a save dialog and writes a transparent PNG to the chosen location
  4. Pressing Ctrl/Cmd+C copies the processed image to the system clipboard and the result can be pasted into another application (including on Wayland)
  5. Pressing Escape during active inference cancels it; pressing ? shows a keyboard shortcut reference overlay
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — GUI module foundation: state machine, worker thread, theme constants, Cargo deps
- [ ] 04-02-PLAN.md — PrunrApp eframe integration, toolbar, canvas, status bar, shortcuts overlay, main.rs launch
- [ ] 04-03-PLAN.md — Human verification checkpoint: end-to-end GUI testing of all Phase 4 requirements

### Phase 5: GUI Feature Completeness
**Goal**: Users have the full interactive experience — before/after comparison, zoom and pan for edge inspection, batch sidebar for multi-image workflows, settings control, model selection, and the reveal animation on completion
**Depends on**: Phase 4
**Requirements**: VIEW-01, VIEW-02, VIEW-03, VIEW-04, VIEW-05, ANIM-01, ANIM-02, ANIM-03, BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05, BATCH-06, UX-02, UX-05
**Success Criteria** (what must be TRUE):
  1. Scrolling the mouse wheel zooms in/out on the image canvas; holding Space and dragging pans the view; Ctrl/Cmd+0 fits the image to the window and Ctrl/Cmd+1 shows it at 1:1 pixel size
  2. Pressing B toggles between the original and processed image; the transparency areas of the processed image are shown as a checkerboard pattern (not white or black)
  3. When background removal completes, removed pixels dissolve away in a 0.5–1s particle animation before settling into the checkerboard transparency view; pressing any key or clicking skips the animation immediately
  4. Dropping multiple images at once populates a sidebar queue; clicking a sidebar thumbnail switches the main view to that image without re-running inference; dragging items in the sidebar reorders them; pressing [ or ] navigates between images
  5. Ctrl/Cmd+, opens a settings dialog where the user can switch between silueta and u2net models, toggle auto-remove on import, and set the number of parallel inference jobs; the active inference backend (e.g., "CUDA (GPU)" or "CPU") is visible in the dialog
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — Foundation types, zoom/pan canvas rework, before/after toggle
- [ ] 05-02-PLAN.md — Settings dialog and reveal animation
- [ ] 05-03-PLAN.md — Batch sidebar, batch processing, queue management

### Phase 6: Distribution and Packaging
**Goal**: A user on a clean Linux, macOS, or Windows machine with no Rust, no ONNX Runtime, and no other prerequisites can download one binary, run it, and remove image backgrounds — the product is shippable
**Depends on**: Phase 5
**Requirements**: LOAD-03 (SVG via resvg)
**Success Criteria** (what must be TRUE):
  1. The release binary runs on a clean Windows x86_64 VM with no system ONNX Runtime DLL — it does not fail with a DLL-not-found error; ORT is statically linked or bundled via `copy-dylibs`
  2. The release binary runs on a clean macOS aarch64 machine and GPU inference uses CoreML when available (or falls back to CPU with a visible status message in settings)
  3. The release binary runs on a clean Linux x86_64 machine and processes an image end-to-end without any shared library errors
  4. Dropping an SVG file onto the app (or passing it to the CLI) rasterizes it via resvg and processes it identically to a raster input — no error or crash
  5. Settings (last-used model, parallelism) persist across application restarts on all three platforms
**Plans**: TBD

### Phase 7: Iterative Processing
**Goal**: Users can chain processing steps — each "Process" click operates on the previous result instead of the original image, with full undo/redo history through all layers. A toggle switches between "Process original" (default) and "Process result" (chain mode).
**Depends on**: Phase 5
**Requirements**: ITER-01, ITER-02, ITER-03, ITER-04
**Success Criteria** (what must be TRUE):
  1. A "Chain mode" toggle in General settings switches between processing the original source image and processing the current result
  2. In chain mode, clicking Process after a previous result uses the result RGBA as input to the pipeline (not the original source bytes)
  3. Each processing step pushes the previous result onto a history stack; Ctrl+Z walks backward through the history; Ctrl+Y walks forward
  4. The history stack has a configurable maximum depth (default 10) to prevent unbounded memory growth
  5. Switching chain mode off reverts to processing the original image; the history stack is preserved for undo/redo
  6. Works with all processing modes: background removal, line extraction (all 3 modes), and combinations
**Plans**: 3 plans

Plans:
- [ ] 07-01-PLAN.md — History stack data model and undo/redo (Vec-based history, depth limit)
- [ ] 07-02-PLAN.md — Chain mode: process result instead of original (toggle, worker pipeline, CLI)
- [ ] 07-03-PLAN.md — UI polish: history indicator, depth slider, status bar

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace Scaffolding | 4/4 | Complete   | 2026-04-06 |
| 2. Core Inference Engine | 6/6 | Complete   | 2026-04-06 |
| 3. CLI Binary | 2/3 | In Progress|  |
| 4. GUI Foundation | 2/3 | In Progress|  |
| 5. GUI Feature Completeness | 3/3 | Complete   | 2026-04-07 |
| 6. Distribution and Packaging | 0/TBD | Not started | - |
| 7. Iterative Processing | 0/3 | Not started | - |
