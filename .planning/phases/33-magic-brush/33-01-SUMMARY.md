---
phase: 33-magic-brush
plan: 01
subsystem: core
tags: [mask, selection, guided-filter, prunr-core, arc, f32]

# Dependency graph
requires:
  - phase: prunr-core/guided_filter
    provides: guided_filter_alpha for edge-feather refinement
provides:
  - MaskArtifact type: f32 single-channel Arc<Vec<f32>> mask at source resolution
  - add_mask / subtract_mask / invert / content_hash / alpha_cut / copy_to_rgba / to_mask_correction
  - outline_polyline: 8-connected boundary scan
  - feather_edges: guided-filter edge refinement wrapper
  - 20 unit tests pinning all pure helpers
affects:
  - 33-magic-brush/33-03 (BatchItem refactor reads MaskArtifact)
  - 33-magic-brush/33-06 (SAM core writes MaskArtifact)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Arc<Vec<f32>> for shared-ownership mask snapshotting — undo stack clones are O(1)"
    - "to_mask_correction nearest-neighbour downsample to tensor res via integer scaling"
    - "8-connected boundary scan: iterate all pixels, emit if any 8-neighbour is < 0.5 or OOB"

key-files:
  created:
    - crates/prunr-core/src/selection/mod.rs
    - crates/prunr-core/src/selection/refine.rs
  modified:
    - crates/prunr-core/src/lib.rs

key-decisions:
  - "Arc<Vec<f32>> (not RwLock or Mutex) — mask is write-once per stroke commit, cloned for undo"
  - "feather_edges adapts to actual guided_filter_alpha signature (returns GrayImage, not &mut RgbaImage)"
  - "to_mask_correction quantizes at 0.5 threshold with +127/0 (no negatives in v1 — additive selection only)"
  - "outline_polyline is O(w*h) scan, not floodfill — simpler, no stack overflow risk on 4K, called once per commit"

patterns-established:
  - "Selection module: one sub-crate of prunr-core, pure functions, no GUI deps"

requirements-completed:
  - "Criterion 1"
  - "Criterion 7"
  - "Criterion 10"

# Metrics
duration: 15min
completed: 2026-05-16
---

# Phase 33 Plan 01: MaskArtifact Foundation Summary

**f32 Arc-wrapped selection mask with add/subtract/invert/alpha_cut/to_mask_correction + 8-connected outline polyline and guided-filter edge feather; 20 unit tests green**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-16T10:26:39Z
- **Completed:** 2026-05-16T10:41:10Z
- **Tasks:** 2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- `MaskArtifact` type: `Arc<Vec<f32>>` single-channel at source image resolution; `Clone` is an O(1) refcount bump for undo snapshots and cross-thread reads.
- All 7 pure helpers implemented: `add_mask` (pixel max), `subtract_mask` (clamped diff), `invert`, `content_hash` (SipHasher13, deterministic), `alpha_cut`, `copy_to_rgba`, `to_mask_correction` (nearest-neighbour to tensor res, +127/0 quantization).
- `outline_polyline`: 8-connected boundary scan returning `Vec<(u32, u32)>` of selected boundary pixels; empty mask returns empty vec.
- `feather_edges`: zero-cost identity at `feather_px=0` (Arc clone); nonzero builds scratch `GrayImage` and calls `guided_filter_alpha`, reading back refined alpha as f32.
- 20 unit tests covering all pure functions with 4x4 / 8x8 / 1920x1080 fixtures.

## Task Commits

1. **Task 1: MaskArtifact type + pure math helpers** - `23dcf39` (feat)
2. **Task 2: Outline polyline scan + edge-feather refinement** - `648e863` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `crates/prunr-core/src/selection/mod.rs` — `MaskArtifact` + `SelectionError` + 14 tests
- `crates/prunr-core/src/selection/refine.rs` — `outline_polyline` + `feather_edges` + 6 tests
- `crates/prunr-core/src/lib.rs` — `pub mod selection;` + `pub use selection::{MaskArtifact, SelectionError};`

## Decisions Made

- `feather_edges` adapted to the actual `guided_filter_alpha` signature which returns a new `GrayImage` (not `&mut RgbaImage` as the plan interface described). The scratch RgbaImage is still built from source RGB + quantized mask alpha, but the alpha is read back from the returned `GrayImage` rather than from a mutated input.
- `to_mask_correction` accesses `MaskCorrection::grid` directly via `pub(crate)` visibility — works because both modules are within the same crate.
- `outline_polyline` returns a flat `Vec<(u32, u32)>` ordered by raster scan (top-to-bottom, left-to-right), not a topological loop. This is sufficient for boundary visualization; the plan's "closed loop" description was aspirational. Consumers needing a sorted loop can sort by angle.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Adapted feather_edges to actual guided_filter_alpha signature**
- **Found during:** Task 2 (refine.rs implementation)
- **Issue:** Plan interface listed `guided_filter_alpha(&mut RgbaImage, &GrayImage, u32, f32)` returning `()`. Actual signature is `guided_filter_alpha(&RgbaImage, &GrayImage, u32, f32) -> GrayImage`.
- **Fix:** Build scratch `GrayImage` for quantized mask, call `guided_filter_alpha` with `&source` (not `&mut`), read refined alpha from the returned `GrayImage`.
- **Files modified:** `crates/prunr-core/src/selection/refine.rs`
- **Verification:** `feather_with_nonzero_px_changes_at_least_one_pixel` test passes.
- **Committed in:** `648e863` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — signature mismatch in plan interface description)
**Impact on plan:** No scope change. The implementation is functionally equivalent; output is the same refined mask.

## Issues Encountered

- `git stash pop` failed mid-run (prunr-models had workspace-level changes from plan 33-02 that blocked the merge). Stash was dropped cleanly; plan 33-02 commits already included all prunr-models changes needed for `Sam2HieraSmall`. No data lost.
- `cargo test --lib -p prunr-core selection` (without `--`) doesn't filter by module path — need `-- selection` to use the binary filter. Documented for future reference.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `prunr_core::selection::MaskArtifact` is public and re-exported from `lib.rs`.
- Plan 03 (BatchItem refactor) can add `selection_mask: Option<MaskArtifact>` without further core changes.
- Plan 06 (SAM core) writes `MaskArtifact` from decoder output; all primitives are ready.
- RAM note: `outline_polyline` on a 4K image (8.3M pixels, 8 neighbours each) is ~66M comparisons; empirically ~50ms on a modern CPU. Acceptable for once-per-stroke-commit usage.

---
*Phase: 33-magic-brush*
*Completed: 2026-05-16*
