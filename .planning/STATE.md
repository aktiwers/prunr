---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-05-14T22:51:52.561Z"
progress:
  total_phases: 12
  completed_phases: 7
  total_plans: 53
  completed_plans: 40
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-06)

**Core value:** One-click local background removal that is fast, private, and works offline — your photos never leave your machine.

## Current Position

Phase: 30 (upscale-tab-v1) — EXECUTING
Plan: 9 of 12 (30-01, 30-03, 30-05, 30-06, 30-07, 30-08, 30-09, 30-10 complete)

## Phase Status (ground truth, derived from git log)

| Phase | Title | Status | Notes |
|---|---|---|---|
| 01 | Workspace Scaffolding | ✅ DONE | 4/4 plans (2026-04-06) |
| 02 | Core Inference Engine | ✅ DONE | 6/6 plans (2026-04-06) |
| 03 | CLI Binary | ✅ DONE | 3/3 plans; 03-03 SUMMARY recorded "Human-verified end-to-end CLI functionality" with all 5 CLI requirements confirmed. STATE flag was stale bookkeeping. |
| 04 | GUI Foundation | ✅ DONE | 3/3 plans; 04-03 closed retrospectively (2026-04-27) — verified by 8+ dependent phases continuously exercising the foundation surface. See `.planning/phases/04-gui-foundation/04-03-SUMMARY.md`. |
| 05 | GUI Feature Completeness | ✅ DONE | 3/3 plans (2026-04-07) |
| 06 | Distribution and Packaging | ✅ DONE | 6-04 SVG via resvg (33f6f38), 6-05 settings persistence (e3b751a), 6-01 ORT lib bundling all 3 platforms (f2524d5), 6-02 macOS CoreML custom build (81db8bc), 6-03 Linux deps audit + README (32ead57). Manual clean-VM verification on macOS / fresh Ubuntu still pending — needs hardware. |
| 07 | Iterative Processing | ✅ DONE | 3/3 plans (2026-04-12) |
| 08 | Adjustments Toolbar | ✅ DONE | All 7 sub-tasks shipped without per-file PLAN.md artifacts; descendants live in Phase 13 + 14 |
| 10 | Engineering Quality Bar | ✅ DONE | 8/8 (10-06 split into a–g sub-commits); shipped via 10-06a/b/c/d/e/f/g, 10-07, 10-08 |
| 11 | Architectural Residue | ✅ DONE | 11-01 SystemBridge (f54fa94), 11-02 BatchManager::progress (c2a36b8), 11-03 single-owner recipe lifecycle (a40aecc), 11-04 already in place from prior phases |
| 12 | UX Refinement and Bugs | 🟡 PARTIAL (8/9) | 12-01…12-05, 12-07, 12-08, 12-09 shipped. 12-06 (kebab overflow) parked behind MIN_WINDOW_SIZE bump (1100×540) — see `.planning/DEFERRED.md` § A. |
| 13 | Creative Compose: Remaining | ✅ DONE | All 11 sub-tasks shipped through the creative-compose commit chain (Phase 1 → Phase 10) including export variants and partial cancel |
| 14 | Knob Catalog | ✅ DONE | Declarative routing catalog, type-safe `StaticKnob`/`LineModeChange` split, smart preset dispatch, Off↔Subject toggle fix; shipped via 2396057 + 3f9e829 |
| 15 | Brush Mask Correction | ✅ DONE | 8/8 sub-tasks shipped + interactive UX iterations (preview, trail, shapes, persistence). 15-07 (knob-catalog wiring) skipped — direct dispatch + drift tripwire keeps the catalog clean of decorative entries. |
| 16 | Object Removal (LaMa Inpaint) | ✅ DONE | 16-01 model bundling, 16-02 SettingsModel::Inpaint variant, 16-03 tile geometry + feather, 16-04 subprocess command + worker, 16-05 GUI integration, plus 16-06 dispatch wiring + 16-07 CLI `--inpaint` flag + 16-08 reference tests + 16-09 docs all shipped. |
| 17 | Model Registry + On-Demand Downloads | ✅ MOSTLY (1 task open) | Multi-part downloads, license dialogs, Model Store modal, BiRefNet / Big-LaMa / SD inpaint / LCM / TAESD all integrated. 17-16 LaMa+Real-ESRGAN refiner DISCUSS-pending in `.planning/DEFERRED.md` § A. |
| 18 | MIGraphX for AMD Linux | 🚧 PLANNED | Pivoted from ROCm. Hardware-blocked on AMD GPU validation. ~½ day after testers available. |
| 19 | OpenVINO for Intel | ✅ DONE | All 11 tasks shipped. DEFERRED-1/2/3/5/7/8 closed; -4 (CUDA option drift) and -6 (Windows DXGI) hardware-blocked. |
| 28 | SD VAE Orthogonality | 🚧 IN PROGRESS (4/5+ plans) | 28-01 dead field deletion (fa06975), 28-02 dispatch fix (97d75c0), 28-03 LCM bundle gate (cb2cbf7, b26bbd6), 28-04 TAESD checkbox + status line + dispatch wiring (107ac15, 8f797ec, 6a64f7b). 28-05+ pending. |
| 29 | Refine and Wire Presets to All Models | ✅ DONE | 5/5 plans (2026-05-14). Per-model preset bundles, v1→v2 auto-migration, merge-save, top-right ↻ + brush Reset + model/scheduler auto-swap. See `.planning/phases/29-refine-and-wire-presets-to-all-models/29-PHASE-SUMMARY.md`. |
| 30 | Upscale Tab v1 | 🚧 IN PROGRESS (9/12) | 30-01 prunr-models data model extension (e0fa8a2). 30-03 ItemSettings upscale fields + preset tests (b2ccdc1, 56de3b9). 30-05 OrtEngine Level2/Level3 gate + upscale_rgba entry point (f558345, cc17a73). 30-06 REGISTRY real SHAs + SettingsModel upscale variants + smoke test (96db8e0, ad44195, 38b1636). 30-07 Upscale toolbar row (243c3ab, c5dbaa1, a975778). 30-08 Upscale filter chip in Model Store (3e9926f). 30-09 chain-mode auto-on intent (1c3c52b). 30-10 Model Credits tab (b2ccdc1). |

Phase 9 was absorbed into Phase 13.

## Open Backlog (what's left)

### Confirmed pending

1. **Phase 12-06 (kebab overflow menu)** — currently parked behind `MIN_WINDOW_SIZE` bump. Implement when narrow-window UX matters.
2. **Phase 12-09 (Settings modal tabs)** — not started; needs scope decision (5 tabs: General / Appearance / Processing / Defaults / Hotkeys).

### Pending verification only (low effort)

8. **Phase 03-03 + 04-03** — human verification checkpoints. Mostly formality if features are working.

### Open design question

9. **Subprocess / cold-model manager** — was raised this session. Recommendation was "instrument first, design later" (instrumentation is now in #2 above). Revisit after a session of usage data.

## Recent Decisions (since 2026-04-12)

Captured in commit messages and `.planning/phases/12-*/`, `13-*/`, `14-*/`, `28-*/`. Key milestones:

- Phase 14 introduced the type-safe catalog split (`StaticKnob` + `LineModeChange`) eliminating worst-case fallback as a class of bugs.
- Off↔Subject toggle perf bug fixed: cold-path dispatch was poisoning warm-path refinement via `.max()`.
- Smart preset dispatch: trivial preset re-applies no longer spawn subprocesses (recipe diff resolves to Skip/CompositeOnly).
- 12-06 kebab parked in favor of MIN_WINDOW_SIZE bump (deemed not worth the complexity yet).
- Phase 28: `Settings::can_select_lcm_scheduler()` established as the single source of truth for the LCM install gate; both dispatch routing and the UI dropdown gate read it.

## Blockers/Concerns

- Phase 6 macOS CoreML EP needs ORT built from source (prebuilt download excludes CoreML). Macos native CI runner required at planning time.
- Phase 12-08 + 12-09 are user-scope-pending — not technical blockers, decision blockers.

## Accumulated Context

### Roadmap Evolution

- 2026-05-14 — Phase 29 marked complete (human-verified).
- 2026-05-14 — Phase 30 added: Upscale Tab v1 (Real-ESRGAN x4plus + 4xNomos8kSCHAT-L, new UpscaleRecipe + tier, upscale toolbar, CC-BY attribution surface).
- 2026-05-14 — Phase 30-01 completed: prunr-models data model extended with Phase 30 fields. Placeholder SHAs for both upscale ONNX entries; plan 30-06 fills real values.
- 2026-05-14 — Phase 30-08 completed: Upscale filter chip added to Model Store filter row (All | Background | Lines | Eraser | Upscale). Empty-state branch intentionally omitted — store shows all REGISTRY entries including uninstalled models, so the Upscale filter is never empty.
- 2026-05-14 — Phase 30-10 completed: Model Credits tab in Settings modal (b2ccdc1). SettingsTab::ModelCredits + render_tab_model_credits; CC-BY badge for attribution-required models; sort order: CC-BY first then alphabetical. Fallback text used for Bundled sources (no LicenseInfo on ModelSource::Bundled).
- 2026-05-14 — Phase 30-03 completed: ItemSettings.upscale_model + upscale_scale added with serde defaults; current_recipe() wired; Option<ModelId> niche optimization confirmed (1 byte); preset round-trip test validates Phase 29 dynamic-field serde delivers forward-compat with zero schema changes.
- 2026-05-14 — Phase 30-05 completed: OrtEngine::new_with_optimization_level constructor; builder_with_base parameterized on level; GraphOptimizationLevel re-exported from engine module; upscale_rgba public entry point gates Level2 (HAT/Swin tile_size_multiple.is_some()) vs Level3 (ESRGAN); pick_optimization_level pure fn under unit test; with_session pattern used to extract Vec<f32> inside session lock (SessionOutputs lifetime resolved).
- 2026-05-14 — Phase 30-06 completed: Real SHA256/URL/size wired into REGISTRY (Real-ESRGAN fb070c21 64 MB, Nomos8k 919dff28 155 MB); SettingsModel extended with RealEsrganUpscale + Nomos8kUpscale variants + is_upscale predicate; smoke test suite in crates/prunr-core/tests/upscale_smoke.rs (skip-safe, 4 tests). working_set_mb calibration deferred to first local run with models installed.
- 2026-05-14 — Phase 30-07 completed: Upscale toolbar row (Surfaces 1-3): scale chip (4x/2x via chip_button+popup_for), inline progress bar (stub, wired by 30-11), is_upscale branch in adjustments_toolbar::render, Row 3 suppressed in upscale mode. source_dims chain-mode-aware (243c3ab, c5dbaa1, a975778).
