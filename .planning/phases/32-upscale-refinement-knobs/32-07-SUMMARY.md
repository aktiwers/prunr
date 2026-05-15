---
phase: 32-upscale-refinement-knobs
plan: "07"
subsystem: gui-views
tags: [upscale, chip, refinement, toolbar, egui]

dependency_graph:
  requires: ["32-02", "32-05", "32-06"]
  provides:
    - render_output_scale_chip (4-variant OutputScale dropdown with Nomos8k gating)
    - output_scale_label (single source of truth for user-visible scale strings)
    - render_refinement_row (6-chip refinement row: pre-denoise/brightness-lift/sharpen/ai-blend/saturation/color-match)
  affects:
    - adjustments_toolbar.rs (refinement_row wired after render_upscale_row in upscale_mode branch)
    - upscale_toolbar.rs (updated to call render_output_scale_chip with model_id)

tech_stack:
  added: []
  patterns:
    - "output_scale_label() as single source of truth: chip face and popover selectable_label rows both read this function"
    - "x4twopass_available imported from prunr_core::upscale — not redefined; tests call the imported function directly"
    - "4-px ui.add_space gap between Tier-1 and Tier-2 chips instead of separator line — lighter visual grouping"

key_files:
  created:
    - crates/prunr-app/src/gui/views/refinement_row.rs
  modified:
    - crates/prunr-app/src/gui/views/upscale_chip.rs
    - crates/prunr-app/src/gui/views/upscale_toolbar.rs
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/views/mod.rs

key_decisions:
  - "x4twopass_available imported from prunr_core::upscale, not redefined — single source of truth rule; tests pin the imported predicate"
  - "OutputScale::factor() called via method (value.factor()), not via free function — OutputScale::factor() already existed in recipe.rs"
  - "output_scale_factor NOT added to upscale_chip.rs — test renamed to output_scale_factor_via_method and exercises the method"
  - "4-px gap between Tier-1 (pre-denoise/brightness-lift) and Tier-2 (sharpen/ai-blend/saturation/color-match) chips for visual grouping without a separator line"
  - "render_refinement_row return value (bool) not plumbed into ToolbarChange — dispatch tier is determined by recipe diff at next tick, not by explicit chip signals"

metrics:
  duration_minutes: 4
  completed_date: "2026-05-15"
  tasks_completed: 2
  tasks_planned: 2
  files_modified: 4
  files_created: 1
  tests_added: 3
---

# Phase 32 Plan 07: Refinement Chip Row Summary

**Upscale refinement row: seven chips surfaced via canonical chip helpers; output-scale chip upgraded to 4-variant dropdown with Nomos8k gating**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-15T23:14:25Z
- **Completed:** 2026-05-15T23:18:00Z
- **Tasks:** 2
- **Files modified/created:** 5

## Task Commits

| Task | Name | Commit | Key files |
|---|---|---|---|
| 1 | Replace render_scale_chip with render_output_scale_chip (4 variants + Nomos8k gating) | ca8f2cc | upscale_chip.rs, upscale_toolbar.rs |
| 2 | Create refinement_row.rs with six chips + wire into adjustments_toolbar | 93a2fbf | refinement_row.rs, mod.rs, adjustments_toolbar.rs |

## Accomplishments

- `render_scale_chip` (2-state) replaced with `render_output_scale_chip` (4-state: X2/X3/X4/X4TwoPass); X4TwoPass row dims for non-RealEsrganX4Plus models via `x4twopass_available` imported from `prunr_core::upscale`
- `output_scale_label()` is the single source of truth for user-visible strings ("2×"/"3×"/"4×"/"4× (two-pass)"); both the chip face and popover selectable_label rows read it
- Three pinning tests: `x4twopass_available_only_for_real_esrgan`, `output_scale_label_pinned`, `output_scale_factor_via_method`
- `render_refinement_row` creates six chips in two visual groups: Tier-1 (pre-denoise, brightness-lift) then 4-px gap then Tier-2 (sharpen, ai-blend, saturation, color-match)
- All 34 chip:: calls use canonical helpers (chip_button, popup_for, slider_row_f32, chip_tooltip, reset_button); zero hand-rolled equivalents
- Row wired into `adjustments_toolbar::render` in the `upscale_mode` branch after `render_upscale_row`
- 918 workspace tests pass; clippy clean

## Icon Substitutions

No substitutions needed. All six icons guessed by the plan are confirmed in `egui_material_icons v0.6.0`:
- `ICON_BLUR_ON` (`\u{e3a5}`) — pre-denoise
- `ICON_BRIGHTNESS_6` (`\u{e3ab}`) — brightness-lift
- `ICON_DEBLUR` (`\u{eb77}`) — sharpen
- `ICON_PSYCHOLOGY` (`\u{ea4a}`) — ai-blend
- `ICON_PALETTE` — saturation (reused from adjustments_toolbar bg chip)
- `ICON_TUNE` — color-match (reused from adjustments_toolbar line-strength chip)

## Chip Layout Notes

Row 2 (refinement_row.rs) renders left-aligned with a 4-px `ui.add_space` gap between the Tier-1 pair and the Tier-2 quad. No overflow observed at standard window widths. The right-aligned reset/preset cluster stays on Row 1 (upscale_toolbar.rs) where the model dropdown lives.

## Tooltip Copy Shipped

| Chip | Tooltip label | Tier note in body |
|---|---|---|
| Pre-denoise | "Pre-denoise" | "Re-runs inference on change (Tier-1)." |
| Brightness Lift | "Brightness Lift" | "Re-runs inference on change (Tier-1)." |
| Sharpen | "Sharpen" | "Real-time (Tier-2). Range: -1 to +1." |
| AI Blend | "AI Blend" | "Real-time (Tier-2)." |
| Saturation | "Saturation" | "Real-time (Tier-2)." |
| Color Match | "Color Match" | "Real-time (Tier-2)." |

## Deviations from Plan

### Plan draft contained duplicate helpers — NOT implemented

The plan's `<action>` code included `pub fn x4twopass_available(...)` and `pub fn output_scale_factor(...)` in `upscale_chip.rs`. Per the `<critical_brief_use_existing_helpers>` in the execution context, these were not added:

- `x4twopass_available` is imported from `prunr_core::upscale` (added in plan 32-05)
- `OutputScale::factor()` is the existing method on the enum (added in plan 32-01's /simplify pass)

The test renamed `output_scale_factor_pinned` → `output_scale_factor_via_method` to accurately describe what it tests (the method, not a free function).

### slider_row_f32 signature mismatch in plan draft

The plan's draft code called `chip::slider_row_f32(ui, "Strength", value, 0.0..=1.0, 0.05)` with a f32 step value as the 5th argument. The actual `slider_row_f32` signature is `(ui, label, value, range, logarithmic: bool, format: impl Fn(f32) -> String)`. All six slider calls were written with the correct signature.

## Self-Check: PASSED

Files exist:
- FOUND: crates/prunr-app/src/gui/views/refinement_row.rs
- FOUND: crates/prunr-app/src/gui/views/upscale_chip.rs
- FOUND: crates/prunr-app/src/gui/views/upscale_toolbar.rs
- FOUND: crates/prunr-app/src/gui/views/adjustments_toolbar.rs
- FOUND: crates/prunr-app/src/gui/views/mod.rs

Commits exist:
- FOUND: ca8f2cc (Task 1)
- FOUND: 93a2fbf (Task 2)

All workspace tests: 537 prunr-app + 330 prunr-core + 31 prunr-models + 20 prunr-runtime-install = 918 total, 0 failed.
Clippy: clean.
