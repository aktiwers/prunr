---
phase: 32-upscale-refinement-knobs
plan: "11"
subsystem: ui
tags: [upscale, recipe, item_settings, chip-copy, serde, defaults]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: UpscaleRecipe + ItemSettings with sharpen field (32-02), refinement chip row (32-07)
provides:
  - UpscaleRecipe::default() sharpen = 0.2 (mild unsharp mask out of box)
  - ItemSettings::default() sharpen = 0.2 with matching serde default
  - Chip copy style policy in CLAUDE.md under GUI state ownership
affects: [future chip phases, any code reading UpscaleRecipe::default]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "default_sharpen() function + #[serde(default = \"default_sharpen\")] mirrors default_ai_blend pattern for non-zero serde defaults"
    - "Chip tooltip policy: silence on timing; active-voice 'Requires reprocessing.' for Tier-1 knobs only"

key-files:
  created: []
  modified:
    - crates/prunr-core/src/recipe.rs
    - crates/prunr-app/src/gui/item_settings.rs
    - CLAUDE.md

key-decisions:
  - "sharpen serde default uses #[serde(default = \"default_sharpen\")] not #[serde(default)]: ensures old presets without the field load with 0.2, matching the new Default::default() value and the default_ai_blend pattern"
  - "refinement_row.rs tooltips were already compliant from 32-07; no changes needed — policy CLAUDE.md captures the anti-pattern to prevent recurrence"
  - "existing upscale_recipe_default_matches_existing_behaviour test updated in-place (not renamed) to preserve bisect history"

patterns-established:
  - "Chip copy style: no timing/plumbing vocabulary in chip tooltips; silence = real-time, explicit note = requires reprocessing"

requirements-completed: ["Gap-4", "Gap-8"]

# Metrics
duration: 25min
completed: 2026-05-16
---

# Phase 32 Plan 11: Gap-4 + Gap-8 Closure Summary

**Default sharpen raised 0.0→0.2 in both UpscaleRecipe and ItemSettings (with matching serde default), plus chip-copy policy codified in CLAUDE.md preventing "Updates live" / "Tier-2" vocabulary from re-entering tooltips**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-16
- **Completed:** 2026-05-16
- **Tasks:** 2 of 2
- **Files modified:** 3

## Accomplishments

- Gap 4 closed: fresh users processing their first Real-ESRGAN image get 0.2 sharpen out of the box — matches Upscayl's perceived quality bar without any slider interaction
- Gap 8 closed: chip-copy policy written in CLAUDE.md so the next chip phase has a durable rule to check against; cites the 32-07 anti-patterns as the canonical bad examples
- 5 new tests pin the default value at every layer: recipe bits, recipe accessor, ItemSettings field, serde missing-field behavior, current_recipe round-trip

## Task Commits

1. **Task 1: Default sharpen to 0.2 in UpscaleRecipe + ItemSettings** - `c4d32b4` (feat)
2. **Task 2: Add chip-copy style policy to CLAUDE.md** - `ab63c55` (docs)

## Files Created/Modified

- `crates/prunr-core/src/recipe.rs` — `sharpen_bits: 0.2_f32.to_bits()` in `UpscaleRecipe::default()`; updated `upscale_recipe_default_matches_existing_behaviour` test; added `upscale_recipe_default_sharpen_is_0_2` test
- `crates/prunr-app/src/gui/item_settings.rs` — `sharpen: 0.2` in `ItemSettings::default()`; `default_sharpen()` function; `#[serde(default = "default_sharpen")]` on field; updated 2 existing tests; added 3 new tests
- `CLAUDE.md` — `### Chip copy style` subsection under `### View-layer helper menu` (inside `## GUI state ownership`)

## Decisions Made

- Used `#[serde(default = "default_sharpen")]` (not bare `#[serde(default)]`) so old presets loading without the field get 0.2, not `f32::default()` = 0.0. This matches the `default_ai_blend` pattern already in place for `ai_blend`. The forward-compat tripwire tests in `presets_fs::tests` confirm the behavior.
- Did not rewrite refinement_row.rs tooltips — 32-07 already shipped compliant tooltips with rich contextual descriptions. Zero banned terms present. CLAUDE.md captures the policy for future phases.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] serde default for sharpen required explicit default function**

- **Found during:** Task 1 (running tests after changing ItemSettings::default())
- **Issue:** `#[serde(default)]` on `f32` uses `f32::default()` = 0.0, not `ItemSettings::default().sharpen` = 0.2. Three forward-compat tests in `presets_fs::tests` failed: `loads_empty_preset_as_defaults`, `loads_preset_with_unknown_fields`, `migrate_v1_empty_object_wraps_default_item_settings`.
- **Fix:** Added `fn default_sharpen() -> f32 { 0.2 }` and changed attribute to `#[serde(default = "default_sharpen")]`. Updated `serde_loads_old_preset_missing_phase32_fields` to assert 0.2 (correct new behavior).
- **Files modified:** `crates/prunr-app/src/gui/item_settings.rs`
- **Verification:** All 924 workspace lib tests pass; forward-compat tripwire tests pass.
- **Committed in:** c4d32b4 (Task 1)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in serde default strategy)
**Impact on plan:** Essential for correctness — without the explicit default function, old presets would have loaded sharpen=0 instead of sharpen=0.2, defeating the purpose of the default change. No scope creep.

## Issues Encountered

The git stash during a baseline verification check reverted my recipe.rs and item_settings.rs changes. I re-applied all edits from scratch. No data lost.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Gap 4 + Gap 8 closed; 32-11 complete
- The chip-copy policy in CLAUDE.md is the durable artifact for all future chip phases
- Any reviewer can grep `Requires reprocessing` + `Tier-2` to verify compliance

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-16*
