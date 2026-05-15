---
phase: 30-upscale-tab-v1
plan: 11
subsystem: gui/processor + gui/app + gui/live_preview
tags: [upscale, dispatch, routing, admission, live-preview, cancel]
dependency_graph:
  requires: [30-02, 30-03, 30-05, 30-06, 30-07, 30-09]
  provides: [upscale-dispatch-wiring, tile-progress, admission-gate, live-preview-gate]
  affects: [processor.rs, app.rs, live_preview.rs, views/statusbar.rs]
tech_stack:
  added: []
  patterns:
    - "Arc<AtomicBool/U32> for cross-thread tile-progress + cancel (mirrors inpaint pattern)"
    - "mpsc channel for upscale result delivery (mirrors inpaint_tx/rx)"
    - "pure fn admission_check(working_set_mb, free_ram_mb) for testable gate"
    - "pure fn can_process_upscale(is_available, is_in_flight, item_loaded) for testable gate"
    - "pure fn should_dispatch_live_preview(tier) contract-pinning helper in live_preview.rs"
key_files:
  created: []
  modified:
    - crates/prunr-app/src/gui/processor.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/live_preview.rs
    - crates/prunr-app/src/gui/views/statusbar.rs
decisions:
  - "WorkKind variant: no new variant added — upscale dispatches on a plain std::thread, not the rayon/worker path; the existing admission system is bypassed in favor of the pre-flight check inside dispatch_upscale itself"
  - "admission_check semantics: free_ram_mb >= working_set_mb (exact match allowed); no safety margin beyond the descriptor's working_set_mb value"
  - "UpscaleResult struct: generation field removed — only one upscale in flight at a time (gated by can_process_intent), so stale-drop by generation is unnecessary overhead"
  - "Generation for dispatch: item_id used as stable unique marker in dispatch_upscale_intent; dispatcher stores none internally"
  - "should_dispatch_live_preview: #[allow(dead_code)] with durable local rationale — contract-pinning helper; actual production gate is in app.rs apply_knob_change which already maps UpscaleRerun -> DispatchKind::None"
metrics:
  duration_minutes: 10
  completed_date: "2026-05-14"
  tasks_completed: 3
  files_modified: 4
  tests_added: 11
---

# Phase 30 Plan 11: Upscale Dispatch Integration Summary

Wire the upscale pipeline into the Processor intent-routing layer: background
thread dispatch with tile-progress atomics, admission gate, cancel support,
status bar tile counter, live-preview gate, and boundary tests.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1a | Processor upscale atomic fields + admission_check + real tile_progress | 5c37039 |
| 1b | dispatch_upscale + cancel_upscale + pump_upscale_results | fdd99f0 |
| 2 | handle_process_intent + can_process_intent + status bar + cancel wiring | 64dddb1 |
| 3 | Live preview gate for upscale (should_dispatch_live_preview + tests) | 15a923c |

## What Was Built

**Processor (`processor.rs`):**
- Four `Arc<Atomic*>` fields: `upscale_active`, `upscale_cancel`, `upscale_tile_done`, `upscale_tile_total`
- `mpsc::channel()` for `UpscaleResult` delivery
- `upscale_tile_progress() -> Option<(u32, u32)>`: replaces the stub from plan 30-07; reads atomics, returns `None` when idle
- `dispatch_upscale`: pre-flight admission check (`working_set_mb` from REGISTRY vs free RAM), spawns background thread calling `prunr_core::upscale::upscale_rgba` with tile-progress callbacks and cancel flag
- `cancel_upscale`: sets the cancel atomic (latency: one tile — ORT has no per-op cancel hook)
- `pump_upscale_results`: non-blocking `try_recv` drain, returns `Vec<UpscaleResult>`
- `admission_check(working_set_mb, free_ram_mb) -> bool`: pure fn under 5 unit tests

**App routing (`app.rs`):**
- `handle_process_intent`: now routes `is_upscale()` branch first via `dispatch_upscale_intent`, then existing inpaint/seg paths; wrapped by `can_process_intent()` guard at entry
- `dispatch_upscale_intent`: resolves input (chain mode: `result_rgba` if present, else `source_rgba`; non-chain: `source_rgba`), calls `processor.dispatch_upscale`
- `can_process_intent`: upscale gate checks `is_available` + `!is_in_flight` + `item_loaded` via pure `can_process_upscale` helper
- `can_process_upscale(is_available, is_in_flight, item_loaded) -> bool`: pure fn under 1 unit test (4 assertions)
- `pump_upscale_results`: drain per-frame, apply `result_rgba`, queue tex_prep + thumbnail, toast on cancel/error
- `handle_cancel`: `cancel_upscale()` called when `is_upscale()` is active
- `apply_cancel_shortcut`: Esc checks `upscale_tile_progress().is_some()` first

**Status bar (`statusbar.rs`):**
- `"Upscaling \u{2014} tile {done}/{total}"` (em-dash) displayed when `upscale_tile_progress()` returns `Some`; takes priority over Erasing and batch-processing text

**Live preview (`live_preview.rs`):**
- `should_dispatch_live_preview(tier: RequiredTier) -> bool`: `UpscaleRerun` returns `false` alongside `FullPipeline` and `AddEdgeInference`; cheap tiers return `true`
- 4 unit tests pin the contract
- Zero references to `upscale_rgba` or `dispatch_upscale` anywhere in the file

## Output Record (per plan spec)

1. **WorkKind variant**: No new variant needed — upscale uses a plain `std::thread`, not the rayon/worker/subprocess paths. The `WorkKind` enum is a registry for the batch-worker path; upscale bypasses it entirely.

2. **admission_check semantics**: `free_ram_mb >= working_set_mb` — exact match (working_set == free) is allowed. No multiplier or safety margin added beyond what `ModelDescriptor.working_set_mb` already encodes. The values in REGISTRY (600 MB for RealESRGAN, 1200 MB for Nomos8kSCHAT-L) already include a conservative headroom factor.

3. **Cancel-during-tile latency**: Worst-case = one tile duration. The cancel flag is polled by `upscale_rgba` between tiles. A single tile at 4K with ESRGAN takes roughly 0.3–1.5 s depending on hardware. There is no cancel hook inside ORT's per-op inference loop.

4. **live_preview.rs grep for upscale dispatch references**: Returns zero matches for `upscale_rgba` and `dispatch_upscale` — only the doc comment string appears. Criterion 8 is enforced.

## Deviations from Plan

### Auto-fixed Issues

None significant — plan executed as written.

**Minor structural deviation:** `UpscaleResult` struct dropped the `generation` field (plan included it). Rationale: the gate in `can_process_intent` ensures only one upscale is in flight at a time; stale-drop by generation adds complexity without a corresponding bug class to prevent. The struct is simpler and the tests still pin the single-flight contract.

**Minor structural deviation:** `should_dispatch_live_preview` is a private `#[allow(dead_code)]` fn rather than `pub(crate)`. The production dispatch gate is in `app.rs apply_knob_change` which maps `UpscaleRerun → DispatchKind::None` — the live_preview.rs fn is a contract-pinning test helper with durable local rationale on the allow attribute.

## Self-Check: PASSED

- processor.rs: upscale_active field present (6 occurrences) ✓
- processor.rs: admission_check fn present ✓
- processor.rs: dispatch_upscale fn present ✓
- app.rs: is_upscale() present (3 occurrences) ✓
- app.rs: dispatch_upscale call site in dispatch_upscale_intent ✓
- statusbar.rs: "Upscaling — tile" format string present ✓
- live_preview.rs: UpscaleRerun match arm present ✓
- live_preview.rs: no upscale_rgba/dispatch_upscale call sites ✓
- Commits: 5c37039, fdd99f0, 64dddb1, 15a923c — all found in git log ✓
- cargo test --workspace --lib: 790 passed, 0 failed ✓
- cargo clippy --workspace -- -D warnings: clean ✓
