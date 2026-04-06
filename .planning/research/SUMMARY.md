# Project Research Summary

**Project:** BgPrunR
**Domain:** Pure Rust desktop AI background removal — local inference, CLI + GUI, single-binary distribution
**Researched:** 2026-04-06
**Confidence:** HIGH

## Executive Summary

BgPrunR is a local-first AI background removal desktop application that uses ONNX Runtime (via the `ort` crate) to run U2-Net/silueta models entirely on the user's machine, with no cloud dependency. Research confirms this is a well-understood problem domain: rembg (Python) has validated the model pipeline, the ONNX preprocessing contract is documented, and the Rust ecosystem now has production-ready bindings to make a pure-Rust implementation practical. The recommended architecture is a Cargo workspace with three crates — a shared inference core library, an egui/eframe GUI binary, and a clap CLI binary — structured so the inference pipeline is built and tested first, then wrapped in presentation layers.

The critical risk in this project is not the technology selection but the implementation order and correctness of the inference preprocessing pipeline. The ONNX models are sensitive to exact normalization constants, tensor layout (NCHW not NHWC), and pixel scaling order. Getting this wrong produces silently garbage output — the model runs but removes nothing useful. The mitigation is straightforward: build the core inference pipeline as an isolated, fully unit-tested library first, with a reference-comparison test against rembg's Python output on known images, before any GUI work begins. Architectural decisions that protect this boundary (worker threads for GUI, direct calls for CLI, no inference on the render thread) must be made from day one — retrofitting them is high-cost.

The single-binary distribution goal shapes every other decision: model bytes must be embedded at compile time using zstd compression, ONNX Runtime must be linked or bundled (not relied on from the system), and no runtime downloads are acceptable. This adds build complexity — longer compile times, a dedicated model crate to isolate recompilation cost, and per-platform CI runners for release artifacts — but the user experience payoff (download, run, no setup) is the primary competitive differentiator vs rembg and cloud-based alternatives.

## Key Findings

### Recommended Stack

The stack is built around three anchors: `ort 2.0.0-rc.12` for ONNX inference (the only mature Rust binding to ONNX Runtime 1.24 with GPU execution provider support), `egui + eframe 0.34.1` for the GUI (immediate-mode, cross-platform, built-in texture and drag-and-drop support with no retained-mode boilerplate), and the `image 0.25.8` crate as the central image type flowing through the pipeline. No async runtime is needed — inference is CPU/GPU-bound, and `rayon` provides all parallelism required for batch processing. The `include-bytes-zstd` proc-macro compresses model files at build time, reducing the u2net model from ~170MB to ~130MB and avoiding extreme compile-time memory pressure from raw `include_bytes!`.

Notably, `tokio` must not be added. GPU acceleration is handled by registering all three execution providers in the `ort` session builder (CUDA, CoreML, DirectML) and letting ONNX Runtime negotiate at runtime — no separate GPU binary is needed. CoreML on macOS requires compiling ORT from source (the prebuilt download does not include CoreML), which is a platform-specific build complexity to plan for.

**Core technologies:**
- `ort 2.0.0-rc.12`: ONNX inference — only mature Rust crate with GPU EP support; pin exact version (pre-release)
- `egui + eframe 0.34.1`: native GUI — immediate-mode eliminates image-tool state complexity; built-in GPU texture upload
- `image 0.25.8`: image I/O — standard library, handles all required formats, central DynamicImage type
- `rayon 1.11.0`: batch parallelism — work-stealing pool, zero config, no async overhead
- `ndarray 0.16.1`: tensor construction — required for building the [1,3,320,320] f32 ONNX input tensor
- `resvg 0.47.0`: SVG rasterization — pure Rust, zero native deps, rasterizes to pixmap before inference
- `arboard 3.6.1`: clipboard image copy — use `wayland-data-control` feature or Wayland clipboard silently fails
- `include-bytes-zstd` + `zstd 0.13.3`: model embedding — zstd compression at build time, decompression at first use
- `clap 4.5.x`: CLI argument parsing — derive macro, de-facto standard

See `.planning/research/STACK.md` for full dependency block, version compatibility matrix, and alternatives considered.

### Expected Features

Research confirms the feature set is well-scoped. The MVP is fully defined: all 14 P1 features are interdependent and must ship together. The three critical dependencies are: (1) the ONNX inference core must be complete before any other feature exists, (2) single-image removal must be solid before batch processing, and (3) the model embedding build pipeline must be proven before the model-selection UI makes sense.

The main anti-feature risks are: manual refinement brush (full editor scope) and background replacement (compositor complexity) — both are explicitly deferred to v2+. The competitive differentiation is clear: fully local/offline, single binary, no subscription, GPU-accelerated, SVG input support, and both CLI and GUI in one artifact.

**Must have (table stakes):**
- ONNX inference pipeline (silueta model, CPU) — the entire product is built on this
- Single-image background removal with drag-and-drop + file picker
- Transparent RGBA PNG output written to disk
- Before/after comparison view — primary quality assessment tool
- Zoom and pan on result — edge inspection at pixel level
- Progress indication during inference — u2net takes 3-15s on CPU
- Copy result to clipboard — paste directly into design tools
- Keyboard shortcuts for open, process, save, copy
- CLI mode with batch processing capability
- GPU acceleration with CPU fallback (CUDA + CoreML)
- Cross-platform builds: Linux x86_64, macOS aarch64, Windows x86_64
- Single-binary distribution (models embedded, no installer)
- u2net + silueta dual model with settings selection
- Large image warning and downscale prompt (>8000px guard)

**Should have (competitive):**
- SVG input via resvg — rare in competitors; useful for logo/icon workflows
- Parallelism tuning in settings — power users with CPU-intensive workflows
- BMP input support — legacy files, trivial to add
- Settings persistence — last-used model, output directory

**Defer (v2+):**
- Manual refinement brush — scope explosion; requires full editor pipeline
- Background replacement — compositor/layer complexity orthogonal to removal
- Additional ONNX models (BiRefNet, SAM) — evaluate quality vs. binary size post-launch
- Video background removal — fundamentally different pipeline, explicitly out of scope

See `.planning/research/FEATURES.md` for full competitor analysis and feature dependency graph.

### Architecture Approach

The architecture is a three-crate Cargo workspace: `bgprunr-core` (shared inference library with no binary targets), `bgprunr-gui` (egui/eframe desktop application), and `bgprunr-cli` (clap binary). The core library exposes two public functions — `process_image()` and `batch_process()` — and contains all inference logic, preprocessing, postprocessing, image I/O, and model management. The GUI dispatches all inference work to a background worker thread via mpsc channels, polling results with non-blocking `try_recv()` in the egui `update()` loop. The CLI calls core functions directly on the main thread (blocking is fine without a render loop), using rayon for batch parallelism. ONNX sessions are created once per model kind (lazy on first use) and reused across all inference calls — never created per image.

**Major components:**
1. `bgprunr-core` (lib crate) — inference pipeline, image I/O, model bytes, preprocessing, postprocessing, batch coordination
2. `bgprunr-gui` (bin crate) — egui/eframe app state, worker thread + mpsc channels, texture caching, UI panels
3. `bgprunr-cli` (bin crate) — clap arg parsing, direct core calls, indicatif progress bar, batch dispatch via rayon
4. `InferenceEngine` (in core) — ort Session lifecycle, execution provider negotiation (CUDA→CoreML→DirectML→CPU), session pool
5. `PreProcessor` / `PostProcessor` (in core) — isolated, independently testable pipeline stages; preprocessing is the correctness-critical component
6. `ModelRegistry` (in core) — static zstd-compressed model bytes via `include_bytes_zstd`, lazy session init

See `.planning/research/ARCHITECTURE.md` for data flow diagrams, code sketches for each pattern, and anti-patterns.

### Critical Pitfalls

1. **Preprocessing pipeline mismatch against rembg** — The model expects exact: decode → resize 320x320 bilinear → f32 → /255.0 → subtract ImageNet mean → /std → transpose HWC→CHW → batch dim → [1,3,320,320] tensor. Any deviation (wrong order, missing /255, wrong axis layout) produces silently garbage masks. Mitigation: build a unit test that compares BgPrunR's output mask pixel-by-pixel against rembg's Python output on 3 known test images — this test must pass before any GUI work begins.

2. **GPU execution provider silently falls back to CPU** — `ort` silently drops EPs it cannot initialize; no error is raised. Users with NVIDIA hardware get no benefit and cannot diagnose why. Mitigation: log the active EP name at session initialization, and surface it visibly in the GUI settings dialog as "Inference: CUDA (GPU)" or "Inference: CPU (no GPU detected)".

3. **Blocking the egui update loop with inference** — Calling `session.run()` inside `App::update()` freezes the window for 200ms–15s. The window appears crashed. Mitigation: architecture must use mpsc channels + worker thread from day one; retrofitting this is a HIGH-cost architectural change.

4. **Texture re-upload every frame** — Calling `ctx.load_texture()` inside `update()` without caching causes continuous GPU upload at 60fps. On large images this causes frame drops and battery drain. Mitigation: store `TextureHandle` in app state; only call `load_texture` on state transition to `Done`.

5. **Thread oversubscription: rayon + ONNX Runtime thread pools** — Naive rayon parallel batch × ORT intra-op threads = N×M threads competing for N cores, degrading throughput below sequential. Mitigation: set `intra_op_num_threads = total_cores / rayon_workers`; benchmark before wiring parallelism.

6. **Windows DLL hell** — System32 may contain an incompatible `onnxruntime.dll`. Binary works on dev machine but fails on clean Windows installs. Mitigation: use `ort`'s `copy-dylibs` feature or prefer static linking; test distribution on a clean Windows VM with no prerequisites installed.

7. **include_bytes! compile time with 170MB u2net model** — Embedding a 170MB blob causes minutes-long CI rebuilds even for small source changes. Mitigation: isolate model bytes in a dedicated `bgprunr-models` crate (recompiles only when model files change), use `include-bytes-zstd`, and cache that crate artifact in CI.

See `.planning/research/PITFALLS.md` for full pitfall descriptions, warning signs, recovery costs, and the pitfall-to-phase mapping.

## Implications for Roadmap

Based on combined research, the build order is dictated by hard dependencies: core inference must be proven correct before any presentation layer exists, and the workspace structure must be correct before any model is embedded. The pitfall-to-phase mapping from PITFALLS.md directly validates this ordering.

### Phase 0: Project Scaffolding and Workspace Setup

**Rationale:** The workspace structure, CI matrix, and model crate isolation must be established before any code is written. Getting the crate layout wrong at this stage is a disruptive mid-project refactor. The `include_bytes!` compile time pitfall specifically targets this phase.
**Delivers:** Cargo workspace with three crates, GitHub Actions CI matrix (Linux/macOS/Windows native runners), model crate scaffold with `include-bytes-zstd`, development feature flag for filesystem model loading, `cargo-dist` release pipeline skeleton.
**Addresses:** Single-binary distribution goal; CI pipeline foundation.
**Avoids:** include_bytes compile time pitfall (Pitfall 7); structural debt from monolithic `main.rs` pattern.
**Research flag:** Standard patterns — Cargo workspace setup is well-documented. No additional research needed.

### Phase 1: Core Inference Engine

**Rationale:** This is the highest-risk phase and the foundation of everything else. The preprocessing correctness must be locked in with a reference test before any GUI or CLI work begins. Architecture research explicitly states "Phase 1 (core inference) must be fully functional before any GUI or CLI work begins." Both the GPU EP silent fallback pitfall and the preprocessing mismatch pitfall are addressed here.
**Delivers:** `bgprunr-core` library with working `process_image()` on silueta model (CPU), verified against rembg Python output; active EP logged and queryable; `batch_process()` with progress callback; large image guard.
**Addresses:** ONNX inference pipeline (P1); GPU acceleration + CPU fallback (P1); large image warning (P1).
**Avoids:** Preprocessing mismatch (Pitfall 1 — critical); GPU silent fallback (Pitfall 2); session-per-image anti-pattern (Architecture Anti-Pattern 1); thread oversubscription (Pitfall 5 — design decision made here).
**Research flag:** Needs careful verification — preprocessing constants must match rembg exactly. Reference test against Python output is mandatory gate before Phase 2.

### Phase 2: CLI Binary

**Rationale:** The CLI is simpler than the GUI (no threading complexity, no texture management) and exercises the full `bgprunr-core` API under real conditions. Building CLI before GUI provides a working, distributable tool early and validates the core API boundary. Any API design problems surface here at low cost, before the GUI adds complexity.
**Delivers:** `bgprunr-cli` binary with single-image and batch commands, indicatif progress bar, model selection flag, cross-platform release artifacts.
**Addresses:** CLI mode (P1); batch processing (P1); cross-platform builds (P1).
**Avoids:** Tight coupling of core to any GUI-specific types (Architecture pattern: CLI direct-call with no channels).
**Research flag:** Standard patterns — clap + rayon batch CLI is well-documented. No additional research needed.

### Phase 3: GUI Foundation (Threading Architecture)

**Rationale:** The GUI must be built with the worker thread + mpsc channel pattern from day one. The blocking egui update loop pitfall has HIGH recovery cost if retrofitted. The texture caching pattern must also be established before the before/after view is added (which doubles the texture complexity). This phase delivers a working GUI with drag-and-drop, inference dispatch, progress display, and result display — but not yet the full comparison view.
**Delivers:** `bgprunr-gui` binary with: drag-and-drop + file picker, worker thread + mpsc channels, `InferenceState` enum, progress spinner, result display with TextureHandle caching, basic save/copy to clipboard, keyboard shortcuts.
**Addresses:** Drag-and-drop (P1); progress indication (P1); copy to clipboard (P1); save output file (P1); keyboard shortcuts (P1).
**Avoids:** Blocking egui update loop (Pitfall 3 — HIGH recovery cost); texture re-upload every frame (Pitfall 4); Wayland clipboard failure (Pitfall 8 — arboard setup done here).
**Research flag:** Standard patterns — egui worker thread + mpsc is well-documented with existing examples. Wayland clipboard testing requires explicit manual verification on a Wayland session.

### Phase 4: GUI Polish and Full Feature Parity

**Rationale:** With the threading architecture stable, the remaining GUI features (before/after comparison view, zoom/pan, model selection UI, settings dialog, batch processing UI, EP status indicator) can be added without architectural risk. These are UI layers on top of the already-working inference pipeline.
**Delivers:** Before/after split-slider comparison view; zoom and pan on result canvas; model selection (silueta vs u2net) in settings dialog; GPU/CPU status indicator; batch folder processing with per-file progress; parallelism settings; u2net model embedded and verified.
**Addresses:** Before/after comparison view (P1); zoom/pan (P1); model quality selection (P1); batch processing GUI (P1); GPU status visibility.
**Avoids:** Re-upload texture pitfall in before/after view (two textures — establish pattern from Phase 3 carries here); OOM on u2net model embedding (model crate isolation from Phase 0).
**Research flag:** Before/after slider widget may need implementation research — egui does not have a built-in split-slider; community solutions or custom widget implementation may be needed.

### Phase 5: Distribution and Packaging

**Rationale:** The Windows DLL hell pitfall is a distribution-phase problem requiring a clean-VM smoke test. This phase closes the gap between "works on dev machines" and "works on clean end-user machines." It also adds the SVG input differentiator (resvg) and the v1.x features (settings persistence, BMP input).
**Delivers:** Per-platform release artifacts verified on clean VMs (Linux, macOS arm64, Windows); ORT DLL bundling via `copy-dylibs` or static link; resvg SVG input; settings persistence; BMP input; final README with platform matrix.
**Addresses:** Single-binary distribution (P1 — full verification); SVG input (P2); BMP input (P2); settings persistence (P2); Windows DLL distribution (Pitfall 6).
**Avoids:** Windows DLL hell (Pitfall 6 — requires clean VM test, not dev machine test); "looks done but isn't" checklist from PITFALLS.md.
**Research flag:** macOS CoreML build may need research — CoreML EP requires compiling ORT from source, not using the prebuilt download strategy. This is the only remaining area with documented uncertainty.

### Phase Ordering Rationale

- **Core before CLI before GUI:** Hard dependency graph from the Cargo workspace — `bgprunr-core` must compile before either binary target. CLI exercises the API with less complexity than GUI, surfacing design problems cheaply.
- **Threading architecture in GUI Phase 3 before feature additions in Phase 4:** The blocking-update-loop pitfall has HIGH recovery cost. All GUI features assume the threading model is already in place.
- **Workspace scaffolding before any code (Phase 0):** The model crate isolation must predate any model embedding; retrofitting is disruptive.
- **Distribution verification last (Phase 5):** Clean-VM testing makes sense only after features are complete; but the DLL bundling configuration should be scaffolded in Phase 0.

### Research Flags

Phases needing deeper research during planning:
- **Phase 1 (Core Inference):** Preprocessing constants must be verified against the rembg source at planning time. The normalization pipeline is correctness-critical and not forgiving of approximation. Plan a reference test as the gate between Phase 1 and Phase 2.
- **Phase 4 (Before/After View):** egui does not ship a built-in split-slider comparison widget. Research community implementations (e.g., `egui-video`, custom `Painter` approach) or plan to implement a custom widget.
- **Phase 5 (macOS CoreML distribution):** Building ONNX Runtime from source with `--use_coreml` on macOS aarch64 for distribution is the least-documented step. May require a macOS native CI runner and significant build configuration.

Phases with standard patterns (skip research-phase):
- **Phase 0 (Scaffolding):** Cargo workspace setup, GitHub Actions multi-platform CI, `cargo-dist` — all well-documented with official guides.
- **Phase 2 (CLI):** clap derive + rayon batch + indicatif — completely standard Rust CLI patterns.
- **Phase 3 (GUI Threading):** egui worker thread + mpsc + `try_recv` in update loop — official egui pattern with documented examples.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified on crates.io and official docs as of 2026-04-06. One caveat: `ort 2.0.0-rc.12` is pre-release; pin exact version and monitor for stable 2.0.0. |
| Features | HIGH (table stakes), MEDIUM (differentiators) | Table stakes verified against multiple competitors. Differentiator value props based on competitor analysis and user review mining — directionally correct but market validation needed post-launch. |
| Architecture | HIGH | Patterns verified against official egui docs, Rust Book workspace docs, and ort source. Inference tensor flow confirmed against rembg preprocessing spec. |
| Pitfalls | HIGH | Most pitfalls verified against official docs, egui issue tracker, ort GitHub, and ONNX Runtime threading docs. GPU EP silent fallback behavior is documented and confirmed. |

**Overall confidence:** HIGH

### Gaps to Address

- **CoreML EP on macOS aarch64 (distribution build):** The prebuilt ORT download does not include CoreML. Building ORT from source with `--use_coreml` for macOS distribution is underdocumented in the context of the `ort` Rust crate. This needs a spike during Phase 5 planning (or earlier if macOS GPU acceleration is a hard Phase 1 requirement).
- **Before/after comparison widget:** egui does not ship a built-in split-slider. Research or implementation plan needed during Phase 4 planning. Community options exist but need evaluation.
- **ort 2.0.0 stable release timeline:** Research was conducted against `rc.12`. If 2.0.0 stable releases before development is complete, the exact version pin should be updated. Monitor [pykeio/ort releases](https://github.com/pykeio/ort/releases).
- **u2net model licensing for distribution:** The u2net ONNX model weights have their own license (MIT per the original repo, but verify the ONNX conversion used). Confirm license compatibility before distributing the embedded binary.

## Sources

### Primary (HIGH confidence)
- [ort crates.io](https://crates.io/crates/ort) — version 2.0.0-rc.12 confirmed 2026-03-05
- [pykeio/ort GitHub](https://github.com/pykeio/ort) — execution providers, linking strategies, session API
- [egui GitHub releases](https://github.com/emilk/egui/releases) — 0.34.1 confirmed latest March 2025
- [eframe crates.io](https://crates.io/crates/eframe) — version confirmed
- [image crates.io](https://crates.io/crates/image) — 0.25.8 confirmed
- [rayon docs.rs](https://docs.rs/crate/rayon/latest) — 1.11.0
- [resvg crates.io](https://crates.io/crates/resvg) — 0.47.0, 2026-02-09
- [arboard crates.io](https://crates.io/crates/arboard) — 3.6.1, Wayland note verified
- [include-bytes-zstd GitHub](https://github.com/daac-tools/include-bytes-zstd) — proc-macro approach
- [ndarray crates.io](https://crates.io/crates/ndarray/0.16.1) — 0.16.1 confirmed
- [zstd crates.io](https://crates.io/crates/zstd) — 0.13.3
- [Cargo Workspaces — The Rust Book](https://doc.rust-lang.org/cargo/reference/workspaces.html) — workspace structure
- [eframe App trait documentation (deepwiki)](https://deepwiki.com/membrane-io/egui/5-eframe-application-framework) — lifecycle, state management
- [ONNX Runtime threading docs](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) — intra-op thread pool
- [ONNX Runtime CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) — macOS limitations
- [egui texture discussions (emilk/egui #5718, #4932)](https://github.com/emilk/egui/discussions/5718) — texture re-upload
- [egui threading pattern (emilk/egui #484)](https://github.com/emilk/egui/discussions/484) — mpsc worker pattern
- [arboard — 1Password/arboard](https://github.com/1Password/arboard) — Wayland clipboard ownership
- [Rust include_bytes compile time (rust-lang/rust #65818)](https://github.com/rust-lang/rust/issues/65818) — known issue
- [U2-Net preprocessing (xuebinqin/U-2-Net #270)](https://github.com/xuebinqin/U-2-Net/issues/270) — preprocessing contract

### Secondary (MEDIUM confidence)
- [deepwiki.com pykeio/ort installation](https://deepwiki.com/pykeio/ort/2.1-installation-and-setup) — Cargo.toml feature flags
- [2025 Survey of Rust GUI Libraries (boringcactus)](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) — egui vs iced vs Slint comparison
- [rembg GitHub (danielgatis/rembg)](https://github.com/danielgatis/rembg) — model list, preprocessing, CLI
- [Snapclear.app](https://www.snapclear.app/) — offline desktop competitor features
- [Best offline background removers 2026 (MadFable)](https://www.madfable.com/blog/best-background-remover-offline-pc) — user expectations
- [Async vs worker threads for CPU work (wyeworks, 2025)](https://wyeworks.com/blog/2025/02/25/async-rust-when-to-use-it-when-to-avoid-it/) — CPU-bound vs async guidance
- [Bundling ONNX Runtime in Rust (blog.stark.pub)](https://blog.stark.pub/posts/bundling-onnxruntime-rust-nix/) — DLL/linking war stories

### Tertiary (LOW confidence / needs validation)
- [ONNX Runtime EP fallback discussion (onnx/onnx #6623)](https://github.com/onnx/onnx/discussions/6623) — silent fallback behavior; validate during Phase 1
- [ort execution providers docs (ort.pyke.io)](https://ort.pyke.io/perf/execution-providers) — 403 on direct fetch; content verified via WebSearch and DeepWiki; re-verify when site is accessible

---
*Research completed: 2026-04-06*
*Ready for roadmap: yes*
