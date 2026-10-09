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
- [ ] **Phase 8: Adjustments Toolbar** - Persistent toolbar replacing modal settings, per-image settings, live-preview Tier 2 reruns, selection-as-apply semantics, rebindable hotkeys
- [ ] **Phase 10: Engineering Quality Bar** - Hardening / debt-paydown: tests, reliability, observability, god-object split, perf polish, dep hygiene. No user-visible features.

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

### Phase 8: Adjustments Toolbar
**Goal**: Transform Prunr's UX from a modal-settings model into a persistent-toolbar model with per-image settings and live preview. Showcases the tiered recipe pipeline (Tier 2 rerun) by making "tweak and see" feel instant. Selection in the sidebar becomes an "apply" action with visual revert via uncheck.
**Depends on**: Phase 7 (chain mode + history), tiered recipe pipeline (Phases A–E, shipped)
**Requirements**: TBD (UX-01..UX-N)
**Success Criteria** (what must be TRUE):
  1. Every mask/composite knob is adjustable from a persistent toolbar; the settings modal contains only app-wide config (parallel jobs, force CPU, dark checker, history depth, chain mode, live preview toggle, hotkeys)
  2. Each `BatchItem` owns its own `ItemSettings`; tweaking the toolbar updates only the current image, not all images
  3. Live preview: mask tweaks auto-rerun Tier 2 with 300ms debounce + cancel+restart; UI stays at 60fps during slider drag
  4. Selection semantics: checkbox check copies current toolbar → target + fires Tier 2 rerun + stores pre-check settings; uncheck reverts (if pre-check stored); Select All bulk-marks without triggering reruns
  5. Process / Process N Selected / Process All broadcast current toolbar and clear pre-check on processed items
  6. Footer logo animates as a global activity indicator (200ms delayed-show) for all in-flight work
  7. Hotkeys are rebindable in Settings → Hotkeys (default: H = sidebar, Shift+H = adjustments)
  8. New "Subject + Lines" output mode: segmentation + edge overlay with subject body preserved
  9. Performance targets met: slider release → result ≤1s on 4000×3000, cancel ≤20ms, chip popover first paint ≤16ms
**Plans**: 7 plans

Plans:
- [ ] 08-01-PLAN.md — Data model shift: ItemSettings struct, BatchItem refactor, AppSettings trim, DexiNed edge tensor caching, v1→v2 migration
- [ ] 08-02-PLAN.md — Adjustments toolbar UI (rows 2+3, chip+popover, lines popover, preset dropdown) — static, no live preview
- [ ] 08-03-PLAN.md — Live preview engine (Tier 2 mask + edge): debounce + cancel + restart, subprocess cancel path, adaptive-resolution
- [ ] 08-04-PLAN.md — GPU-side bg_color fill (Tier 3 → render-time, eliminates CPU pipeline for color changes)
- [ ] 08-05-PLAN.md — Selection-as-apply: thumbnail=view-only, checkbox=apply, pre_check flow, stale indicator, texture priority queue
- [ ] 08-06-PLAN.md — Footer activity indicator + Settings modal (General/Hotkeys tabs) + hotkey rebinding + preset save/load/apply
- [ ] 08-07-PLAN.md — Subject-with-outlines compositing + polish + docs

### Phase 10: Engineering Quality Bar
**Goal**: Bring the codebase from "ambitious app with real engineering in the hot paths" to production-grade quality: well-tested, observable, recoverable, modular. Closes every gap identified in the 2026-04-18 codebase review. No user-visible features.
**Depends on**: Nothing (orthogonal hardening pass; can run alongside feature phases)
**Requirements**: None (internal quality)
**Success Criteria** (what must be TRUE):
  1. Pure business logic has unit tests: `resolve_tier`, `AdmissionController` contention, `HistorySlot` tier demotion, every `SubprocessCommand`/`SubprocessEvent` bincode round-trip, live-preview stale-generation discard, `push_bounded` cap
  2. Every surviving `.unwrap()`/`.expect()` in non-test code has a one-line invariant comment; zero `.lock().unwrap()` — all mutex calls handle poison
  3. Subprocess hang is detected within 60s of silence with in-flight work; `send_shutdown` has a 5s timeout then force-kill
  4. All non-exception `eprintln!` migrated to `tracing` with structured fields; `RUST_LOG=prunr=debug` honoured
  5. `PrunrApp` holds ≤20 fields in ≤1500 LOC; business logic lives in `HistoryManager`, `DragExportState`, `BatchDispatcher`
  6. `ui()`, `run_batch_with_retry`, `BatchDispatcher::dispatch` each ≤60 lines; all helpers ≤80 lines
  7. Mask-to-alpha loop parallelized with before/after numbers recorded in ARCHITECTURE.md; reference pixel-accuracy test still passes
  8. Stale Cargo.toml comments removed; ORT RC pin documented; `include-bytes-zstd` stabilised; ARCHITECTURE.md has subprocess memory-reclaim rationale
**Plans**: 8 plans

Plans:
- [ ] 10-01-PLAN.md — Test safety net for pure business logic (resolve_tier, admission, history, IPC, live-preview, preset-bounded)
- [ ] 10-02-PLAN.md — Reliability hardening: unwrap/expect audit + mutex poison handling + test-only constructor gating
- [ ] 10-03-PLAN.md — Subprocess robustness: hang watchdog + shutdown timeout
- [ ] 10-04-PLAN.md — Observability: tracing migration from eprintln!
- [ ] 10-05-PLAN.md — Split PrunrApp into HistoryManager / DragExportState / BatchDispatcher
- [ ] 10-06-PLAN.md — Decompose ui(), run_batch_with_retry, BatchDispatcher::dispatch into ≤80-line helpers
- [ ] 10-07-PLAN.md — Performance polish: parallelize mask-to-alpha, dedupe conversions, record numbers
- [ ] 10-08-PLAN.md — Dep + doc hygiene: Cargo.toml comments, ORT pin policy, include-bytes-zstd, ARCHITECTURE.md

### Phase 28: SD VAE Orthogonality
**Goal**: Decouple TAESD VAE from LCM scheduler so all four valid SD pipeline combos (standard SD, standard SD + TAESD, LCM, LCM + TAESD) are reachable, fix two related dispatch correctness bugs, and surface a live "what's running" status to the user. Removes the artificial `lcm &&` gate that today hides combo #2 and silently degrades quality when LCM scheduler is picked without LCM weights installed.
**Depends on**: Phase 17 (Model Registry / on-demand bundles), Phase 27 (eraser toolbar + scheduler dropdown)
**Requirements**: None new (refines existing SD inpaint UX)
**Success Criteria** (what must be TRUE):
  1. `BrushSettings.sd_use_taesd: Option<bool>` exists and is consumed by dispatch; `processor.rs:430` no longer reads `lcm &&` as the TAESD gate
  2. SD chip popover renders a "Fast VAE (TAESD)" checkbox; greys out with "Download in Model Store" hint when bundle missing
  3. Scheduler dropdown greys out "LCM" with the same hint when `SdV15LcmInpaintFp16` is not installed; the gate predicate is shared with dispatch (one source of truth)
  4. `processor.rs:417` `lcm` flag derives from `backend == ModelId::SdV15LcmInpaintFp16`, not from the user's scheduler request — picking LCM scheduler without bundle installed no longer clamps CFG against standard SD weights
  5. SD chip popover shows a one-line live status: "Active: standard SD" / "standard SD + TAESD" / "LCM" / "LCM + TAESD"
  6. Model Store card descriptions for the LCM and TAESD bundles updated to reflect actual selection mechanism (scheduler dropdown + new checkbox)
  7. Dead code removed: `Settings.sd_fast_mode` field, `hardware::sd_fast_mode_auto_default`, `hardware::sd_fast_mode_active`, and their tests
  8. Boundary test enumerates all four (backend, use_taesd, scheduler) dispatch combos plus the two "bundle not installed" fallback rows; no test references the removed `sd_fast_mode` API
  9. Manual visual check on combo #2 (standard SD + TAESD) recorded in the phase summary — if visibly worse than full VAE, combo #2 is removed before shipping
**Plans**: 5 plans

Plans:
- [ ] 28-01-PLAN.md — Settings + dead code removal: add BrushSettings.sd_use_taesd field + sd_use_taesd_effective() predicate, delete Settings.sd_fast_mode + hardware sd_fast_mode_* helpers and their tests
- [ ] 28-02-PLAN.md — Dispatch correctness: extract pure resolve_sd_dispatch() helper, fix bug #1 (CFG-clamp keyed on backend, not scheduler) and bug #2's dispatch half (drop `lcm &&` TAESD coupling), boundary test enumerating all 7 rows
- [ ] 28-03-PLAN.md — UI gating predicate: add Settings::can_select_lcm_scheduler() and grey out the LCM entry in the scheduler + quality preset dropdowns when the LCM bundle is missing
- [ ] 28-04-PLAN.md — TAESD checkbox + status line: render the "Fast VAE" chip on the SD popover, add the live "Active: ..." status line via a pure 4-row truth-table helper, wire bs.sd_use_taesd_effective() into the dispatch InpaintTuning literal
- [ ] 28-05-PLAN.md — Model Store card text + manual visual check: update LCM and TAESD descriptions, run combo #2 (standard SD + TAESD) visual check, record outcome in 28-PHASE-SUMMARY.md

### Phase 29: Refine and Wire Presets to All Models
**Goal**: Extend the existing preset system from "ItemSettings only" to "every per-model setting the user can configure" — including all `BrushSettings` (radius/hardness/mode/shape/inpaint_*) and SD-tuning knobs (scheduler/steps/CFG/Karras/strength/prompt/negative/use_taesd/seed). Presets become per-model bundles; SD's per-scheduler tuning lives as a scheduler-keyed map within the SD entry. Auto-swap when the user changes models or schedulers; merge-save when overwriting a preset so other-model and other-scheduler entries are preserved. The "Prunr" preset stays factory-locked and serves as the universal fallback for missing models / fields.
**Depends on**: Phase 17 (preset filesystem store), Phase 27 (scheduler dropdown), Phase 28 (`sd_use_taesd` field — must exist before being preset-bundled)
**Requirements**: None new (refines existing preset UX)
**Success Criteria** (what must be TRUE):
  1. Preset JSON files carry a `format_version: 2` envelope with `models: HashMap<ModelId, ModelPreset>`; v1 files (current shape) auto-migrate on load by wrapping their content under the model the file's `model` field specifies
  2. `ModelPreset` contains `item_settings`, `brush`, and (for SD-family models) `sd: Option<SdPreset>`; `SdPreset` contains SD-level fields plus `active_scheduler` and `schedulers: HashMap<SdScheduler, SdSchedulerBundle>`
  3. Resolution algorithm: `(active preset, active model, [active scheduler])` → values; missing model entry falls back to Prunr's per-model defaults; missing field within an entry falls back per-field to Prunr defaults
  4. Save is merge-not-replace: overwriting preset Foo while on SD Inpaint + DDIM updates only `Foo.models[SdV15InpaintFp16].sd.schedulers[Ddim]` (and the SD-level fields), leaving other model entries and other scheduler bundles intact
  5. Top-right ↻ "Reset all knobs" applies the active preset's per-model values to the active item; resolves through the same lookup chain (preset → Prunr fallback)
  6. Brush popover "Reset brush" button resolves through the same chain, scoped to the brush-popover-visible knob subset
  7. Switching the active model auto-swaps the live `BrushSettings` (and `ItemSettings` for the new item) to the active preset's per-model values for the new model
  8. Switching the active scheduler within SD auto-swaps `sd_steps / sd_guidance_scale / sd_use_karras_sigmas / sd_strength` to the active preset's bundle for that scheduler (with factory fallback if not stored)
  9. Save path is dynamic: adding a new field to `ItemSettings`, `BrushSettings`, `SdPreset`, or `SdSchedulerBundle` is automatically picked up by the next save with no save-code change; `#[serde(default)]` plus the existing `loads_old_preset_with_unknown_fields` tripwire enforce forward-compat
 10. `SdQualityPreset` (Fast/Balanced/Quality) remains a UX shortcut on the SD chip; clicking it writes into the active preset's `sd.schedulers[<bundled scheduler>]` rather than directly into `Settings.brush`
 11. Boundary tests enumerate: (a) per-model resolution with Prunr fallback, (b) per-scheduler resolution with factory fallback, (c) merge-save preserves untouched entries, (d) v1 → v2 migration round-trips correctly, (e) cross-machine forward-compat — preset with unknown ModelId entries loads cleanly on older client
**Plans**: 5 plans

Plans:
- [ ] 29-01-PLAN.md — Schemas + Default impls + serde tripwire tests (foundation, pure data)
- [ ] 29-02-PLAN.md — Pure resolver `resolve_preset_for_model` + split/fuse brush helpers + boundary tests on every fallback row
- [ ] 29-03-PLAN.md — v1→v2 auto-migration in `presets_fs::load_all` + retype `Settings.presets` to `HashMap<String, PresetFile>` + round-trip test
- [ ] 29-04-PLAN.md — Merge-save (`save_merged`) preserves cross-model and cross-scheduler entries + boundary test
- [ ] 29-05-PLAN.md — UI wiring: top-right ↻ + brush popover Reset + model/scheduler auto-swap + SdQualityPreset write-through + dirty indicator + forward-compat unknown-ModelId test + human verification

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Phase 10 is orthogonal hardening and can be interleaved.

**Quality gate (Phase 10 internal):** After every subphase, run `/simplify` on files touched. After the full phase, run a **deep `/simplify`** across all files touched by 10-01 through 10-08 as one audit — looking for cross-subphase regressions.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace Scaffolding | 4/4 | Complete   | 2026-04-06 |
| 2. Core Inference Engine | 6/6 | Complete   | 2026-04-06 |
| 3. CLI Binary | 2/3 | In Progress|  |
| 4. GUI Foundation | 2/3 | In Progress|  |
| 5. GUI Feature Completeness | 3/3 | Complete   | 2026-04-07 |
| 6. Distribution and Packaging | 0/TBD | Not started | - |
| 7. Iterative Processing | 0/3 | Not started | - |
| 8. Adjustments Toolbar | 0/7 | Planning    | - |
| 10. Engineering Quality Bar | 5/8 | In Progress | (10-01..10-05 done) |
| 28. SD VAE Orthogonality | 2/5 | In Progress|  |
| 29. Refine and Wire Presets to All Models | 5/5 | Complete    | 2026-05-14 |
| 30. Upscale Tab v1 | 12/12 | Complete   | 2026-05-15 |
| 32. Upscale Refinement Knobs | 13/14 | Complete    | 2026-05-16 |

### Phase 30: Upscale Tab v1

**Goal:** A user can pick an upscale model from the Model Store, switch the active model to that upscaler, optionally chain after a BG-removal or inpaint pass, and produce a 4× output that flows through save / drag-export — all settings persist via the existing per-ModelId preset system.

**Depends on:** Phase 17 (model registry + on-demand downloads), Phase 29 (per-model presets — done)

**Requirements**: New (Phase 30 introduces 11 numbered success criteria below; referenced as "Criterion 1..11" in plan frontmatter)

**Success Criteria** (what must be TRUE):
  1. Two upscale models ship as `OnDemand` `ModelDescriptor` entries hosted on `github.com/aktiwers/prunr/releases`: Real-ESRGAN x4plus (BSD-3, self-exported with dynamic axes from xinntao `.pth`) and 4xNomos8kSCHAT-L (CC-BY-4.0, re-hosted from Phhofm's Drive)
  2. `ProcessingRecipe` carries an `UpscaleRecipe { model: Option<UpscaleModelId>, scale: u32 }` slot; `resolve_tier` returns a new `RequiredTier::UpscaleRerun` between `CompositeOnly` and `EdgeRerun` when only the upscale recipe changed
  3. `ModelDescriptor` gains three new fields: `tile_size_multiple: Option<u32>` (HAT requires 16-px multiples), `recommended_tile: Option<u32>`, `attribution_required: bool` — existing models unaffected (`None`/`false`)
  4. New `prunr-core/src/upscale/` module performs RGB inference through ONNX with overlap-blend tiling, pad-to-multiple wrapper for window-attention models, and Lanczos3 alpha companion for transparent inputs
  5. New `ModelCategory::Upscale` + filter chip visible in the Model Store; `is_upscale()` predicate routes the adjustments toolbar to a dedicated upscale toolbar (model + scale chips) when the active model is an upscaler
  6. Per-ModelId preset machinery automatically picks up the new `ItemSettings` upscale fields with no save-code change; preset round-trip test covers upscale fields; `ItemSettings` stays under 128-byte budget
  7. CC-BY-4.0 attribution is satisfied via a "Model credits" section in Settings that auto-lists installed attribution-required models with author + license link
  8. Upscale is gated out of live preview (10 Hz dispatch) — recipe-diff still triggers a single full dispatch on commit
  9. Stacking works: do BG removal in BiRefNet, switch active model to Real-ESRGAN with chain mode on, the upscale runs on the BG-removed result; the BiRefNet preset persists in its per-ModelId slot
 10. fp32 ONNX only for v1 (fp16 sibling deferred until GPU users request); upscale runs in-process via `OrtEngine` (subprocess routing deferred to v2 unless OOM analysis says otherwise)
 11. OOM protection: tile-based inference bounds per-tile working set; `working_set_mb` calibrated per model and gates admission like other models

**Plans:** 12/12 plans complete

Plans:
- [ ] 30-01-PLAN.md — ModelDescriptor + LicenseInfo extensions, ModelCategory::Upscale, ModelId variants (placeholder SHA)
- [ ] 30-02-PLAN.md — UpscaleRecipe + RequiredTier::UpscaleRerun + resolve_tier extension
- [ ] 30-03-PLAN.md — ItemSettings upscale fields + size-budget guard + preset round-trip
- [ ] 30-04-PLAN.md — prunr-core/src/upscale/ module — tiling (pad-to-multiple, overlap-blend) + alpha companion
- [ ] 30-05-PLAN.md — OrtEngine optimization-level override + upscale_rgba public entry gating on tile_size_multiple
- [ ] 30-06-PLAN.md — REGISTRY real SHA + SettingsModel upscale variants + end-to-end ONNX smoke test (requires PRECONDITIONS.md complete)
- [ ] 30-07-PLAN.md — Upscale toolbar (Surface 1) + scale chip + inline progress bar
- [ ] 30-08-PLAN.md — Model Store Upscale filter chip + empty state
- [ ] 30-09-PLAN.md — Chain-mode auto-on intent on ToolbarChange + app-side apply with has_result gate
- [ ] 30-10-PLAN.md — Settings "Model credits" tab (Surface 5)
- [ ] 30-11-PLAN.md — Processor::dispatch_upscale + handle_process_intent routing + live-preview gate + admission
- [ ] 30-12-PLAN.md — ARCHITECTURE.md update + human verification of all 11 criteria

### Phase 32: Upscale Refinement Knobs

**Goal:** Add a seven-knob refinement set to both upscale models (Real-ESRGAN x4plus, Nomos8kSCHAT-L) matching the existing chip / preset / recipe architecture. Tier-1 knobs re-run inference; Tier-2 knobs operate in real-time on a cached upscale RGBA buffer (mirrors BiRefNet's Tier-1/Tier-2 split). A new `RealESRGAN_x2plus` model entry powers a `4× (two-pass)` Output-Scale dropdown variant that runs the 2× model twice for noisy / low-light inputs.

**Depends on:** Phase 30 (upscale dispatch + REGISTRY infrastructure — done), Phase 29 (per-model presets — done)

**Requirements:** New (Phase 32 introduces 11 numbered success criteria below; referenced as "Criterion 1..11" in plan frontmatter)

**Success Criteria** (what must be TRUE):
  1. `UpscaleRecipe` carries seven new fields: `pre_denoise: f32 ∈ [0,1]`, `brightness_lift: f32 ∈ [-2,2]` (EV stops), `output_scale: OutputScale` (enum: `X2`, `X3`, `X4`, `X4TwoPass`), `sharpen: f32 ∈ [-1,1]`, `ai_blend: f32 ∈ [0,1]`, `saturation: f32 ∈ [-1,1]`, `color_match: bool`
  2. `resolve_tier` returns the existing `RequiredTier::UpscaleRerun` when any Tier-1 knob (`pre_denoise`, `brightness_lift`, `output_scale`) changes; returns a new `RequiredTier::UpscaleTier2` between `CompositeOnly` and `UpscaleRerun` when only Tier-2 knobs (`sharpen`, `ai_blend`, `saturation`, `color_match`) change
  3. `BatchItem` caches the raw model output as `upscale_raw: Option<Arc<RgbaImage>>` so Tier-2 dispatches re-apply postprocess without re-running inference; cache invalidates on any Tier-1 change or model switch
  4. New `prunr-core/src/denoise/` module ships classical median + bilateral filters that operate bbox-aware, sequential per-channel, and reuse buffers per CLAUDE.md `## RAM discipline` (canonical pattern from `guided_filter_alpha`)
  5. New `prunr-core/src/upscale/postprocess.rs` module ships sharpen (unsharp mask), AI-blend (lerp upscale ↔ bicubic-of-source), saturation (HSL-space), color-match (Reinhard Lab-space histogram transfer) as pure functions operating on `RgbaImage`
  6. `RealESRGAN_x2plus` registered as an `OnDemand` ModelDescriptor (`ModelCategory::Upscale`, `UpscaleModelKnobs { is_fp16: false, input_name: "data" }`); used only for the `X4TwoPass` dispatch path
  7. `X4TwoPass` dispatch runs `RealESRGAN_x2plus` twice (output of pass 1 → input of pass 2); the chip is hidden / dimmed in the dropdown when Nomos8k is selected (no 2× HAT variant ships)
  8. Adjustments toolbar gains 7 new chips in a dedicated "Refinement" row when the active model is upscale: pre-denoise, brightness-lift, output-scale (enum dropdown chip), sharpen, ai-blend, saturation, color-match — chip layout, theme, tooltip pattern matches existing knob chips (see `chip::popup_for`, `chip::slider_row_f32`)
  9. All seven knobs persist via the existing per-ModelId preset machinery — preset round-trip test covers each; `ItemSettings` stays under 128-byte budget after the additions
 10. Live preview (10 Hz Tier-2 dispatch) drives the real-time Tier-2 knob feel on the cached `upscale_raw` buffer — same debounced dispatch hook BiRefNet uses; Tier-1 knobs continue to gate out of live preview
 11. `prunr-core` unit tests cover each new pure function (denoise filters, postprocess ops, two-pass scheduler); `prunr-models` test asserts `X4TwoPass` recipe variant routes through `RealESRGAN_x2plus`

**Plans:** 13/14 plans complete

Plans:
- [x] 32-01-PLAN.md — Data layer: UpscaleRecipe + OutputScale + RequiredTier::UpscaleTier2 + resolve_tier extension
- [x] 32-02-PLAN.md — ItemSettings byte-budget rework (six new fields, output_scale replaces upscale_scale, edge_thickness/guided_radius narrowed)
- [ ] 32-03-PLAN.md — prunr-core/src/denoise/ module: histogram-window median + separable bilateral + apply_denoise
- [x] 32-04-PLAN.md — prunr-core/src/upscale/postprocess.rs: apply_sharpen + apply_ai_blend + apply_saturation + apply_color_match
- [x] 32-05-PLAN.md — RealEsrganX2Plus REGISTRY entry (PRECONDITIONS-gated) + upscale_two_pass scheduler
- [x] 32-06-PLAN.md — BatchItem.upscale_raw + bicubic_source caches + Processor dispatch branching + LivePreview UpscaleTier2 routing
- [x] 32-07-PLAN.md — Refinement chip row + render_output_scale_chip (4 variants + Nomos8k gating)
- [ ] 32-08-PLAN.md — ARCHITECTURE.md row + manual smoke + human verification

### Phase 33: Magic Brush — Selection-First Refactor + SAM-based Author

**Goal:** Decouple the brush from the active model by introducing a shared per-item `selection_mask` artifact — the single source of truth for "what's selected." Refactor the existing Paint Brush to author the selection (instead of writing to model-coupled state). Add Magic Brush as a second author backed by SAM 2 Hiera Small (Apache 2.0, ~46 MB, OnDemand) — click/stroke → SAM decoder → mask candidate → commit to shared selection. SAM's encoder/decoder split mirrors Phase 32's Tier-1/Tier-2 cache pattern (encoder cached on BatchItem, decoder runs per interaction). Per-model interpretation rules preserve existing immediate-feedback UX while enabling new alpha-cut / copy / cut actions on any active selection.

**Depends on:** Phase 32 (Tier-1/Tier-2 cache pattern + admission gate model — must complete first), Phase 30 (OnDemand model store + REGISTRY infrastructure — done), Phase 29 (per-model preset machinery — done)

**Requirements:** New (Phase 33 introduces 12 numbered success criteria below; referenced as "Criterion 1..12" in plan frontmatter)

**Success Criteria** (what must be TRUE):
  1. `BatchItem` carries `selection_mask: Option<MaskArtifact>` + `selection_hash: Option<u64>` as the single per-item selection; replaces the model-coupled `mask_correction` semantic
  2. Paint Brush refactored to author the shared `selection_mask` (strokes accumulate); no longer writes directly to model-specific correction state
  3. Magic Brush registered as a second selection author with **Click** + **Stroke** interaction modes and **Shift (add)** + **Alt (subtract)** modifiers
  4. `SAM2HieraSmall` registered as an `OnDemand` ModelDescriptor (`ModelCategory::Selection`, encoder + decoder as separate ONNX entries, Apache 2.0); MobileSAM available as a research-time fallback descriptor if SAM 2 ONNX export proves immature
  5. Encoder output cached on `BatchItem.magic_brush_embedding: Option<Arc<Tensor>>`, invalidates on source change; eager encoder run on Magic Brush activation with a "preparing..." spinner
  6. Per-model interpretation rule table enforced as code: BG-removal = continuous auto-apply + "Protect selection" toggle; SD inpaint = selection IS inpaint region; LaMa = selection IS erase region; no model loaded = mask-only actions enabled
  7. Shared action menu surfaces 5 model-free actions when a selection exists: Delete (alpha-cut), Copy to clipboard, Cut to clipboard, Clear, Invert — with keyboard bindings (Del, Ctrl+C, Ctrl+X, Esc, Enter; Shift/Alt modifiers for add/subtract)
  8. Selection lifecycle gates: survives Process clicks and model switches; cleared on image switch (per-`BatchItem` ownership)
  9. Selection visualization renders outline (default opacity 1.0, thickness 0–10 px) + low-opacity fill (default 0.15–0.20, range 0–1); marching ants deferred to Phase 35
 10. v1 knobs available on both brushes via shared brush settings: edge feather (0–20 px), outline thickness (0–10 px), outline opacity (0–1), fill opacity (0–1); Magic-specific confidence threshold knob
 11. Existing model-coupled brush behaviors regress-tested under the new selection-first dispatch — Paint Brush in BG-removal mode continues to produce immediate-feedback output identical to pre-refactor
 12. `prunr-core` unit tests cover the selection-mask data layout, per-model interpretation logic, and SAM decoder prompt construction; `prunr-models` test asserts SAM 2 Hiera Small (or fallback) routes encoder + decoder through separate `OrtEngine` instances

**Plans:** 7/7 plans complete

Plans:
- [ ] 33-01-PLAN.md — prunr-core::selection: MaskArtifact + add/sub/invert/alpha_cut/copy_to_rgba + outline polyline + feather refinement (Wave 1)
- [ ] 33-02-PLAN.md — prunr-models: ModelCategory::Selection + Sam2HieraSmall MultiPartOnDemand REGISTRY entry + boundary tests (Wave 1)
- [ ] 33-03-PLAN.md — BatchItem selection_* fields + magic_brush_embedding placeholder + lifecycle gates (commit_selection / clear_selection / invalidate_on_source_change) (Wave 2)
- [ ] 33-04-PLAN.md — Paint Brush migration to selection_mask + per-model interpretation rules + protect_selection toggle + BG-removal regression test (Wave 3)
- [ ] 33-05-PLAN.md — Selection visualization (60Hz overlay) + action bar (Delete/Copy/Cut/Invert/Clear) + shared brush knobs + off-thread outline/texture build (Wave 3)
- [ ] 33-06-PLAN.md — prunr-core::sam: preprocess_for_sam + 4 prompt builders (Click/Stroke/Shift/Alt) + decode_to_mask_artifact + SamEmbedding type (Wave 2)
- [ ] 33-07-PLAN.md — Magic Brush GUI integration: encoder dispatch (Processor) + click/stroke handlers + Preparing... overlay + manual smoke checkpoint (Wave 4)

---

## v0.5 Plan (set 2026-10-09)

**Release rule (user decision 2026-10-09):** no v0.5 binaries until Phases 34–38 are all done — bug fixes, UI consistency, performance, cleanup and docs ship together as v0.5.0 against the published v0.4.8. Pushing to master as we go is authorised; publishing is not.

**Order and why:** 34 (finish what is half-built) → 35 (UI, the largest block, so measurements later land on the final widgets) → 36 (structural cleanup before perf, so optimisations build on the final data structures) → 37 (perf) → 38 (docs + release). Recommended exception: one manual `workflow_dispatch` run of release.yml at the end of Phase 34 — artifacts only, no release — because its last six runs failed and the tree has since gained SAM bundling and a pinned toolchain. Awaiting user decision.

### Phase 34: Magic Brush Polish
**Goal:** Magic Brush and the shared selection feel finished: usable before a result exists, correct on image switch, every visible knob does something, bounded memory, smooth overlay.
**Success Criteria** (what must be TRUE):
  1. Paint and Magic toggles are enabled whenever the selected item has a decoded source; a stroke on a segmentation model with no tensor authors the selection without dispatching a rerun, and the selection is applied automatically when the first result lands
  2. Switching to another image while Magic is active dispatches the encoder for that image (or waits for its decode) without toggling the tool; Preparing… shows meanwhile
  3. Delete / Cut with no result operate on the source and produce a result; every Delete / Cut refreshes the result texture (today it does not)
  4. Edge feather feathers both the visualization and the Delete / Copy / Cut region; changing feather, fill opacity, outline opacity or thickness rebuilds the visualization
  5. Stroke undo depth is 32 and `cache_size()` counts the undo/redo planes
  6. The outline is baked into the selection texture (no per-frame rect list, no decimation); the overlay is one textured quad per frame
  7. The Phase 33 manual smoke checklist (`33-magic-brush/PENDING-MANUAL-VERIFICATION.md`) is run by the user and recorded
  8. `/simplify` run after each item; tests green; pushed

### Phase 35: UI Consistency
**Goal:** One visual and interaction system across all modes; settings a user can understand without the source.
**Approach:** (1) audit via `/gsd:ui-review` plus a manual walk of the four flows (BG removal, eraser, upscale, Magic Brush) producing an inventory of every chip, popover, label and tooltip; (2) a one-page design contract (hierarchy of settings: primary / refinement / advanced, naming rules, where timing hints appear, empty states, error states, keyboard conventions); (3) implementation in slices per toolbar row; (4) copy pass under the CLAUDE.md chip-copy rules.
**Candidates surfaced so far:** chips that differ in shape between rows; Add/Subtract + Strength hidden in Eraser but visible elsewhere; "Protect selection" only in seg mode with no explanation; Silueta on OpenVINO logs two warnings per load; tips list and shortcuts drift from the real bindings.

### Phase 36: Structural Cleanup (pre-perf)
**Goal:** Remove the duplicated mechanisms the 2026-10-08 reviews found, before measuring performance on them.
**Items:** DEFERRED L-1 (one ORT init + `session_builder()` + clippy `disallowed-methods`, then delete the tripwire), L-5 (one mask plane type; `DispatchInputs.correction` carries the plane), L-3 (one HTTP client in prunr-runtime-install), trail de-dup shared between Paint and Magic, ort rc.13 branch experiment (L-2), delete `rust_out`, prune DEFERRED.md of resolved items, update the CLAUDE.md commit-trailer wording.

### Phase 37: Performance
**Goal:** Push the hot paths to the limit without changing output; every quality or RAM trade surfaced for a decision.
**Method:** `cargo build --profile profiling` + samply on the four flows; criterion benches for kernels; golden suites as the bit-exact guard.
**Target list:** `apply_edge_shift` (separable / van Herk, replaces the 45 Bolt PRs), guided filter RAM (drop `mask_f` after stage one; box filters two at a time is a [TRADE]), selection texture upload per stroke (alpha-only or dirty-rect), SAM decode row-parallel, SAM sessions on the GPU EP ladder, OpenVINO upscale tile cap 256 (cancel latency; ~10–20 % throughput [TRADE]), live-preview tick allocations (32-DEFER-7), undo snapshots as bbox patches (L-6), startup (lazy model decompress), DEFERRED § J items.

### Phase 38: Docs and Release
**Goal:** Ship v0.5.0.
**Items:** ARCHITECTURE.md (brush/selection/Magic Brush sections are pre-Phase-33; upscale section lacks refinement knobs), README (upscale, Magic Brush, SAM, models table, toolchain), CHANGELOG since v0.4.8, bump workspace version to 0.5.0 (check the Info.plist / version-sync test), release workflow green on all three targets, tag.
