---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Phase 3 context gathered
last_updated: "2026-04-07T00:18:39.033Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 10
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-06)

**Core value:** One-click local background removal that is fast, private, and works offline — your photos never leave your machine
**Current focus:** Phase 02 — core-inference-engine

## Current Position

Phase: 02 (core-inference-engine) — EXECUTING
Plan: 5 of 6

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01 P01 | 8 | 2 tasks | 14 files |
| Phase 01 P03 | 3 | 1 tasks | 1 files |
| Phase 01 P02 | 1 | 1 tasks | 1 files |
| Phase 01 P04 | 525537 | 2 tasks | 2 files |
| Phase 02 P01 | 4 | 2 tasks | 10 files |
| Phase 02 P02 | 5 | 2 tasks | 4 files |
| Phase 02 P04 | 2 | 1 tasks | 2 files |
| Phase 02 P05 | 2 | 1 tasks | 4 files |
| Phase 02 P06 | 12 | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Pre-roadmap: egui over iced/Slint for image tool texture/zoom/pan support
- Pre-roadmap: ort over tract/candle for exact rembg ONNX model compatibility
- Pre-roadmap: Embed models in binary via include-bytes-zstd for single-file UX
- Pre-roadmap: rayon for batch parallelism (no async runtime — inference is CPU/GPU-bound)
- Pre-roadmap: Cargo workspace with bgprunr-core, bgprunr-gui, bgprunr-cli crates
- [Phase 01]: Added #[allow(unused_imports)] to engine.rs — import required by spec but unused in stub; will be used in Phase 2 inference backend
- [Phase 01]: [Phase 01-03]: dtolnay/rust-toolchain used instead of deprecated actions-rs/toolchain in CI
- [Phase 01]: [Phase 01-03]: fail-fast: false in CI matrix so each platform failure is independently reported
- [Phase 01]: Bootstrap SHA256 mode: empty string constant skips verification and prints hash — developer hardens after first run by replacing empty strings with the printed SHA256 values
- [Phase 01]: rembg GitHub releases as model source (danielgatis/rembg/releases/download/v0.0.0/) — exact same models rembg uses for compatibility
- [Phase 01]: [Phase 01-04]: Manually authored release.yml for cargo-dist 0.31.0 rather than running cargo dist init interactively (no local install)
- [Phase 01]: [Phase 01-04]: release.yml mirrors ci.yml model cache pattern (hashFiles xtask/src/main.rs) for consistency
- [Phase 02]: include-bytes-zstd moved from build-dependencies to dependencies in bgprunr-models (proc-macro crate, not build script dep)
- [Phase 02]: ProgressStage uses Postprocess not Sigmoid — matches rembg min-max normalization per RESEARCH.md
- [Phase 02]: Dev workflow for bgprunr-core requires --features bgprunr-models/dev-models; production build needs xtask fetch-models to be run first
- [Phase 02-02]: From<image::ImageError> for CoreError implemented as inline impl mapping to ImageFormat(String) — required by formats.rs image loading
- [Phase 02-02]: DOWNSCALE_TARGET imported explicitly in test module rather than re-exported from formats.rs to avoid unused import lint
- [Phase 02]: [Phase 02-03]: OrtEngine wraps Mutex<Session> because ort::Session::run() requires &mut self — Mutex enables shared &OrtEngine references while satisfying mutability
- [Phase 02]: [Phase 02-03]: ndarray upgraded from 0.16 to 0.17 in workspace Cargo.toml to match ort 2.0-rc.12 dependency requirement
- [Phase 02]: [Phase 02-03]: dev-models and cuda features added to bgprunr-core Cargo.toml to propagate bgprunr-models feature flags properly
- [Phase 02]: Each rayon worker creates its own OrtEngine::new() — no Arc<Mutex<Session>> sharing, avoids contention
- [Phase 02]: ort_intra_threads = (num_cpus / jobs).max(1) — balances ORT and rayon thread counts to prevent CPU oversubscription
- [Phase 02]: [Phase 02-04]: Results collected as indexed (usize, Result) pairs then assigned to pre-allocated Vec to preserve input order with rayon work-stealing
- [Phase 02]: Test images not committed (copyrighted rembg assets, .gitignore excluded); only generated reference masks committed as ground truth
- [Phase 02]: [Phase 02-05]: generate_references.py locks alpha_matting=False, post_process_mask=False, model=u2net — exact rembg defaults required for valid CORE-05 comparison
- [Phase 02]: [Phase 02-05]: VERSIONS.txt written alongside masks by generate_references.py to capture rembg version at generation time
- [Phase Phase 02]: Import InferenceEngine trait explicitly in integration tests — active_provider() only accessible when trait is in scope
- [Phase Phase 02]: Reference mask resize with FilterType::Nearest for pixel-accurate comparison when rembg output dimensions differ from bgprunr output

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2: Preprocessing pipeline must exactly match rembg constants (NCHW layout, /255 → subtract ImageNet mean → /std). A pixel-accurate reference test against rembg Python output is a hard gate before CLI/GUI work begins.
- Phase 5: egui has no built-in split-slider widget for before/after comparison. Custom widget or community solution needed — research at plan time.
- Phase 6: macOS CoreML EP requires building ORT from source (prebuilt download excludes CoreML). Needs a macOS native CI runner and build configuration spike during Phase 6 planning.

## Session Continuity

Last session: 2026-04-07T00:18:39.029Z
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-cli-binary/03-CONTEXT.md
