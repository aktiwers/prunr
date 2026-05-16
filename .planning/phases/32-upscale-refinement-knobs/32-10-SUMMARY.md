---
phase: 32-upscale-refinement-knobs
plan: "10"
subsystem: upscale
tags: [ort, ep-ladder, cuda, directml, coreml, upscale, registry, knobs]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: UpscaleModelKnobs, upscale dispatch (upscale_rgba, upscale_two_pass), tiling pipeline
  - phase: 30-upscale-tab-v1
    provides: Initial upscale dispatch (cpu_only constructor, hardcoded scale literals)
provides:
  - UpscaleModelKnobs.native_scale (u32) and .uses_window_attention (bool) in prunr-models
  - EP ladder constructor (OrtEngine::new_with_optimization_level) in upscale dispatch
  - Data-driven native_scale at run_upscale_native call sites (no more hardcoded 4 / 2)
  - Tiler overlap keyed on uses_window_attention (not tile_size_multiple.is_some())
  - Source-text contract test pinning the no-cpu-only rule
affects: [32-upscale-refinement-knobs, 30-upscale-tab-v1, prunr-core upscale]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Data-driven model knobs: add a field to UpscaleModelKnobs, populate REGISTRY, dispatch reads it — no code changes in prunr-core/src/upscale"
    - "Source-text contract test: include_str!(mod.rs) + assert!(!src.contains(banned_pattern)) for architecture invariants"
    - "uses_window_attention distinct from tile_size_multiple: conceptually different signals that happen to overlap on Nomos8k; x2plus has tile_size_multiple=Some(2) for pixel_unshuffle but is CNN/RRDB"

key-files:
  created: []
  modified:
    - crates/prunr-models/src/lib.rs
    - crates/prunr-core/src/upscale/mod.rs

key-decisions:
  - "EP ladder (new_with_optimization_level) replaces cpu_only in upscale dispatch — CUDA/DirectML/CoreML give 10-50x speedup with no correctness cost; OpenVINO's lazy-compile concern is an EP-layer decision, not a dispatch-site decision"
  - "uses_window_attention is a distinct field from tile_size_multiple: x2plus needs tile_size_multiple=Some(2) for pixel_unshuffle even-dimension requirement but is CNN/RRDB — 16-px overlap is correct, not 32-px"
  - "native_scale baked into REGISTRY eliminates hardcoded 4 and 2 from upscale/mod.rs — adding a 3x model is now a pure data edit"

patterns-established:
  - "Upscale knob extension pattern: add field to UpscaleModelKnobs → populate all REGISTRY entries → add unit tests per entry → dispatch reads via upscale_knobs(descriptor)?"
  - "Source-text contract test pattern: include_str!(mod.rs) assert in #[cfg(test)] block to pin banned patterns (cpu_only, specific literals)"

requirements-completed: ["Gap-6", "Gap-1"]

# Metrics
duration: 7min
completed: 2026-05-16
---

# Phase 32 Plan 10: Gap-1 + Gap-6 Closure Summary

**EP ladder replaces cpu-only upscale dispatch (10-50x GPU speedup); native_scale + uses_window_attention added to UpscaleModelKnobs, removing hardcoded scale literals and fixing x2plus over-wide tile overlap**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-16T02:08:35Z
- **Completed:** 2026-05-16T02:15:42Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Gap-1 closed: upscale dispatch now uses `OrtEngine::new_with_optimization_level` (EP ladder) instead of `new_cpu_only_with_optimization_level` — GPU machines get their full hardware capability for upscale inference
- Gap-6 closed: `native_scale: u32` and `uses_window_attention: bool` added to `UpscaleModelKnobs`; all three REGISTRY entries (x4plus, Nomos8k, x2plus) populated; hardcoded `4` and `2` literals removed from dispatch
- Tiler overlap now reads `knobs.uses_window_attention` instead of `tile_size_multiple.is_some()` — x2plus gets 16-px CNN overlap instead of 32-px HAT overlap, recovering ~7% tile throughput
- Source-text contract test `upscale_dispatch_does_not_force_cpu_only` pins the invariant via `include_str!` — any future reintroduction of `new_cpu_only` in mod.rs fails loudly

## Task Commits

1. **Task 1: Add native_scale + uses_window_attention to UpscaleModelKnobs** - `c131d87` (feat)
2. **Task 2: Switch upscale dispatch to EP ladder; read knobs.native_scale; key overlap on uses_window_attention** - `1ad5870` (fix)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `crates/prunr-models/src/lib.rs` — Extended `UpscaleModelKnobs` struct with `native_scale: u32` and `uses_window_attention: bool`; all three REGISTRY entries updated; four new tests added
- `crates/prunr-core/src/upscale/mod.rs` — Replaced `new_cpu_only_with_optimization_level` with `new_with_optimization_level` at two dispatch sites; `knobs.native_scale` at three call sites; overlap branch on `uses_window_attention`; source-text contract test added

## Decisions Made

- EP ladder is the right default per CLAUDE.md anti-patterns; the OpenVINO lazy-compile concern belongs at the EP-ladder layer, not at the dispatch site. Upscale was the only subsystem still opting out.
- `uses_window_attention` kept separate from `tile_size_multiple` because x2plus uses `tile_size_multiple=Some(2)` for pixel_unshuffle geometry (even H/W enforcement) but is CNN/RRDB architecture needing 16-px overlap — conflating them caused a silent 7% throughput regression.

## Deviations from Plan

None — plan executed exactly as written. The two sub-steps that appeared to fail during initial editing were actually re-applied correctly after verifying the file state.

## Issues Encountered

One Edit tool call appeared to succeed (no error) but the file wasn't changed — the old comment block surrounding the cpu_only call caused the string match to not find the exact text (the file had already been partially modified). Re-read the exact line range and re-applied the edit correctly.

## User Setup Required

None — no external service configuration required.

## GPU Speedup Note

Wall-clock speedup on a GPU machine: expected 10-50x for CUDA/DirectML/CoreML (per CLAUDE.md anti-patterns documentation, this was the cost of the cpu_only choice). Exact measurement requires a machine with a compatible GPU EP and a downloaded upscale model (both models are OnDemand). DEFER-5 is now closeable.

## Next Phase Readiness

- Gap-1 and Gap-6 closed; DEFER-5 is closeable (separate doc commit when requested)
- Adding a new upscale model is now purely a REGISTRY data edit: add `ModelDescriptor` with `native_scale`, `uses_window_attention`, `tile_size_multiple`, `uses_window_attention` populated — no code changes in `prunr-core/src/upscale`
- All 921 workspace lib tests pass; clippy clean

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-16*
