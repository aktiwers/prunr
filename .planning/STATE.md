# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-06)

**Core value:** One-click local background removal that is fast, private, and works offline — your photos never leave your machine
**Current focus:** Phase 1 — Workspace Scaffolding

## Current Position

Phase: 1 of 6 (Workspace Scaffolding)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-04-06 — Roadmap created; ready to begin Phase 1 planning

Progress: [░░░░░░░░░░] 0%

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

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Pre-roadmap: egui over iced/Slint for image tool texture/zoom/pan support
- Pre-roadmap: ort over tract/candle for exact rembg ONNX model compatibility
- Pre-roadmap: Embed models in binary via include-bytes-zstd for single-file UX
- Pre-roadmap: rayon for batch parallelism (no async runtime — inference is CPU/GPU-bound)
- Pre-roadmap: Cargo workspace with bgprunr-core, bgprunr-gui, bgprunr-cli crates

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2: Preprocessing pipeline must exactly match rembg constants (NCHW layout, /255 → subtract ImageNet mean → /std). A pixel-accurate reference test against rembg Python output is a hard gate before CLI/GUI work begins.
- Phase 5: egui has no built-in split-slider widget for before/after comparison. Custom widget or community solution needed — research at plan time.
- Phase 6: macOS CoreML EP requires building ORT from source (prebuilt download excludes CoreML). Needs a macOS native CI runner and build configuration spike during Phase 6 planning.

## Session Continuity

Last session: 2026-04-06
Stopped at: Roadmap written; STATE.md and REQUIREMENTS.md traceability updated. Next action: run `/gsd:plan-phase 1`.
Resume file: None
