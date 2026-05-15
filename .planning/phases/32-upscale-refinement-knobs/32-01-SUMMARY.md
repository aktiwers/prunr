---
phase: 32-upscale-refinement-knobs
plan: 01
subsystem: core-pipeline
tags: [rust, recipe, upscale, tier-routing, prunr-core, knob-catalog]

# Dependency graph
requires:
  - phase: 30-upscale-tab-v1
    provides: UpscaleRecipe with model field + RequiredTier::UpscaleRerun + knob_catalog ordered() baseline
provides:
  - OutputScale enum (#[repr(u8)], 4 variants: X2/X3/X4/X4TwoPass)
  - UpscaleRecipe with 7 new fields (5 f32-as-u32-bits knobs + OutputScale + bool)
  - RequiredTier::UpscaleTier2 between CompositeOnly and UpscaleRerun
  - resolve_tier Tier-1/Tier-2 split for upscale recipes
  - only_tier2_changed() helper in recipe.rs
  - Updated ordered() table in knob_catalog.rs (UpscaleTier2 => 2, all higher tiers bumped)
affects:
  - 32-02 (ItemSettings mirrors these fields — explicit plan dependency)
  - 32-05 (dispatch uses OutputScale variants)
  - 32-06 (UpscaleTier2 live-preview wiring)
  - knob_catalog tests (ordered() table now has 8 entries)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "f32-as-u32-bits storage for float fields in recipe structs (pre_denoise_bits, etc.) — matches MaskRecipe::gamma_bits pattern"
    - "only_tier2_changed() helper above resolve_tier — single source of truth for Tier-1 vs Tier-2 upscale classification"
    - "Non-exhaustive match fences: UpscaleTier2 arms in app.rs route to same bucket as UpscaleRerun until plan 32-06 wires the Tier-2 live-preview"

key-files:
  created: []
  modified:
    - crates/prunr-core/src/recipe.rs
    - crates/prunr-core/src/lib.rs
    - crates/prunr-app/src/gui/knob_catalog.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/item_settings.rs

key-decisions:
  - "ai_blend_bits defaults to 1.0_f32.to_bits() (full AI), NOT 0 — zero means pure bicubic which would silently regress every Phase 30 upscale output"
  - "UpscaleTier2 in app.rs classify_candidates and preset dispatch routes to same skip bucket as UpscaleRerun — correct but slow until plan 32-06 adds the fast Tier-2 live-preview path"
  - "item_settings.rs current_recipe() maps legacy upscale_scale: u32 to OutputScale enum inline; plan 32-02 replaces the field entirely"

patterns-established:
  - "OutputScale #[repr(u8)] pattern: primitive repr enum for byte-budget-sensitive fields in ItemSettings"
  - "Tier-1/Tier-2 split helper pattern: only_tier2_changed() compares Tier-1 fields; returns bool for resolve_tier branch"

requirements-completed: [Criterion 1, Criterion 2]

# Metrics
duration: 8min
completed: 2026-05-15
---

# Phase 32 Plan 01: Data Layer Foundation Summary

**UpscaleRecipe extended with 7 knobs + #[repr(u8)] OutputScale enum + RequiredTier::UpscaleTier2 split between postprocess-only and re-inference paths**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-05-15T18:53:06Z
- **Completed:** 2026-05-15T19:01:06Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `OutputScale` enum (`X2/X3/X4/X4TwoPass`) with `#[repr(u8)]` at 1 byte — fits inside `ItemSettings` cache-line budget
- `UpscaleRecipe` extended with 5 float-as-bits knobs, `OutputScale`, and `color_match: bool`; `ai_blend_bits` defaults to `1.0_f32.to_bits()` so Phase 30 output is bit-identical when no new knobs are touched
- `RequiredTier::UpscaleTier2` inserted between `CompositeOnly` (1) and `UpscaleRerun` (3); `ordered()` table in `knob_catalog.rs` updated accordingly
- `only_tier2_changed()` helper isolates Tier-1 fields (`model/output_scale/pre_denoise_bits/brightness_lift_bits`); `resolve_tier` routes Tier-2-only changes to `UpscaleTier2`
- 13 new tests (4 for Task 1, 9 for Task 2); all 510 prunr-app + 290 prunr-core tests green; workspace clippy clean

## Task Commits

1. **Task 1: Add OutputScale enum + extend UpscaleRecipe with seven new fields** - `9d68a87` (feat)
2. **Task 2: Add RequiredTier::UpscaleTier2 + extend resolve_tier + update knob_catalog** - `686296c` (feat)

**Plan metadata:** `{metadata_commit}` (docs: complete plan)

## Exact UpscaleRecipe field list shipped

```rust
pub struct UpscaleRecipe {
    pub model: Option<prunr_models::ModelId>,
    pub output_scale: OutputScale,       // replaces scale: u32
    pub pre_denoise_bits: u32,           // Tier-1
    pub brightness_lift_bits: u32,       // Tier-1
    pub sharpen_bits: u32,               // Tier-2
    pub ai_blend_bits: u32,              // Tier-2, default = 1.0_f32.to_bits()
    pub saturation_bits: u32,            // Tier-2
    pub color_match: bool,               // Tier-2
}
```

## Exact ordered() table shipped

```rust
Skip => 0,
CompositeOnly => 1,
UpscaleTier2 => 2,   // NEW
UpscaleRerun => 3,   // was 2
EdgeRerun => 4,      // was 3
MaskRerun => 5,      // was 4
AddEdgeInference => 6, // was 5
FullPipeline => 7,   // was 6
```

## Match-site fences updated in prunr-app

| File | Approx line | Policy chosen for UpscaleTier2 arm |
|------|-------------|-------------------------------------|
| `app.rs` `classify_candidates` | ~1324 | Grouped with `Skip \| CompositeOnly \| UpscaleRerun` — routes to skip_count; correct (no seg/edge work needed), slow (Tier-2 live-preview added by plan 32-06) |
| `app.rs` preset dispatch | ~3389 | Grouped with `Skip \| CompositeOnly \| UpscaleRerun => DispatchKind::None` — same rationale |

Neither arm uses a phase/plan number in source per CLAUDE.md rules; the comment names the local invariant ("upscale Tier-2 live-preview wired separately in the upscale dispatch path").

## Files Created/Modified

- `crates/prunr-core/src/recipe.rs` — OutputScale enum, extended UpscaleRecipe, UpscaleTier2 variant, only_tier2_changed() helper, updated resolve_tier branch; 13 new tests
- `crates/prunr-core/src/lib.rs` — exports OutputScale
- `crates/prunr-app/src/gui/knob_catalog.rs` — ordered() table updated with UpscaleTier2 => 2
- `crates/prunr-app/src/gui/app.rs` — two UpscaleTier2 match arms added (classify_candidates + preset dispatch)
- `crates/prunr-app/src/gui/item_settings.rs` — current_recipe() updated to map upscale_scale: u32 → OutputScale; test updated to check output_scale instead of scale

## Decisions Made

- `ai_blend_bits` defaults to `1.0_f32.to_bits()`, NOT zero — zero = pure bicubic which would silently regress every Phase 30 upscale result
- `UpscaleTier2` in batch dispatch routes to the skip bucket (same as `UpscaleRerun`) until plan 32-06 adds the dedicated fast path; this is correct (the full upscale re-runs if the user hits Process) but plan 32-06 makes it fast via the cached `upscale_raw` buffer
- Kept `upscale_scale: u32` on `ItemSettings` (plan 32-02 removes it); bridged in `current_recipe()` with an inline mapping to `OutputScale` to unblock compilation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed item_settings.rs current_recipe() using removed UpscaleRecipe::scale field**
- **Found during:** Task 2 (workspace build after adding UpscaleTier2)
- **Issue:** `item_settings.rs:233` referenced `scale: self.upscale_scale` which no longer exists on `UpscaleRecipe`
- **Fix:** Replaced with `output_scale: if self.upscale_scale <= 2 { OutputScale::X2 } else { OutputScale::X4 }` plus `..UpscaleRecipe::default()` for the new fields; updated one test that checked `r.upscale.scale` to check `r.upscale.output_scale` instead
- **Files modified:** `crates/prunr-app/src/gui/item_settings.rs`
- **Verification:** `cargo test --workspace --lib` all green
- **Committed in:** 686296c (Task 2 commit)

**2. [Rule 3 - Blocking] Added UpscaleTier2 arms to two non-exhaustive match sites in app.rs**
- **Found during:** Task 2 (workspace build after adding UpscaleTier2 variant)
- **Issue:** Two `match tier` sites in `app.rs` did not cover `UpscaleTier2`
- **Fix:** Added arm to each site grouping `UpscaleTier2` with `UpscaleRerun` (same routing semantics until plan 32-06)
- **Files modified:** `crates/prunr-app/src/gui/app.rs`
- **Verification:** `cargo build --workspace` clean
- **Committed in:** 686296c (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking compilation issues from removing UpscaleRecipe::scale field and adding new enum variant)
**Impact on plan:** Both fixes necessary for workspace compilation. No scope creep — item_settings.rs field replacement is the plan 32-02 mandate; app.rs match fences are the standard "insert variant, fix all match sites" protocol.

## Issues Encountered

- `clippy::doc-lazy-continuation` error on a doc comment with `+` continuation line — fixed by rewording to remove the continuation.

## Next Phase Readiness

- Data layer locked: `UpscaleRecipe` shape and `RequiredTier` ordering are the foundation for Wave 2 (algorithms), Wave 3 (dispatch), and Wave 4 (GUI)
- Plan 32-02 can proceed: `ItemSettings` mirror of new fields; `upscale_scale: u32` replacement with `output_scale: OutputScale`
- All match fences across workspace compile; no stray `todo!()` or `unreachable!()` in production paths

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-15*
