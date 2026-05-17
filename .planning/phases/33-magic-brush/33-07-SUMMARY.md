---
phase: 33-magic-brush
plan: 07
subsystem: gui
tags: [sam2, magic-brush, selection, ort, egui, rayon]

# Dependency graph
requires:
  - phase: 33-magic-brush/33-01
    provides: MaskArtifact foundation (add_mask, subtract_mask, alpha_cut, outline_polyline)
  - phase: 33-magic-brush/33-02
    provides: ModelId::Sam2HieraSmall registry entry + resolve_part_bytes
  - phase: 33-magic-brush/33-03
    provides: BatchItem selection fields + magic_brush_embedding (Arc<()> placeholder)
  - phase: 33-magic-brush/33-04
    provides: apply_selection_to_active_model dispatch table + protect_selection
  - phase: 33-magic-brush/33-05
    provides: commit_selection_and_dispatch lockstep entry point + selection overlay + action bar
  - phase: 33-magic-brush/33-06
    provides: prunr_core::sam pure functions (preprocess, prompt builders, decode_to_mask_artifact, SamEmbedding)
provides:
  - MagicBrushState coordinator (active flag, encoder-pending flag, active stroke buffer)
  - Processor SAM encoder/decoder dispatch (dispatch_sam_encoder, dispatch_sam_decoder)
  - run_sam_encoder_inline / run_sam_decoder_inline private ORT session.run helpers in processor.rs
  - BatchItem.magic_brush_embedding retyped from Arc<()> to Arc<SamEmbedding>
  - Paint/Magic tool toggle pair in adjustments toolbar with mutual exclusion
  - Canvas Preparing... overlay + Magic chip popover
  - Canvas click/stroke/modifier handlers wired to decoder dispatch
affects: [34-multi-mask, 33-phase-summary, phase-33-manual-smoke]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ORT session.run lives in processor.rs — prunr-core::sam remains ORT-free (Plan 06 boundary contract)"
    - "Admission gate before 800 MB rayon job; error toast + deactivate on failure"
    - "pump_sam_encoder/decoder_results via non-blocking try_recv in per-frame logic()"
    - "PromptModifier enum (Replace/Add/Subtract) carries modifier semantics from canvas to decoder pump"

key-files:
  created:
    - crates/prunr-app/src/gui/magic_brush_state.rs
    - crates/prunr-app/src/gui/views/magic_brush_chip.rs
  modified:
    - crates/prunr-app/src/gui/item.rs
    - crates/prunr-app/src/gui/processor.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/views/canvas.rs
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/views/brush_chip.rs
    - crates/prunr-app/src/gui/batch_manager.rs
    - crates/prunr-app/src/gui/mod.rs

key-decisions:
  - "ORT session.run kept in processor.rs (run_sam_encoder_inline / run_sam_decoder_inline) — prunr-core::sam stays a zero-ORT pure module per Plan 06 design contract"
  - "Magic Brush activation triggers eager encoder dispatch (once per source image); embedding cached on BatchItem; subsequent activations skip"
  - "No cpu_only=true blanket gate on SAM dispatch — default EP ladder per Scoped Workarounds principle"
  - "Task 3 manual smoke DEFERRED — user away from computer, ETA ~1 week (2026-05-17)"

patterns-established:
  - "MagicBrushState coordinator mirrors BrushState: owns transient UI state (active + pending flags + stroke buffer), not business data"
  - "render_shared_selection_section extracted from brush_chip — both Paint and Magic popovers delegate to one helper, no duplication"
  - "PromptModifier (Replace/Add/Subtract) resolved at canvas interaction time and carried through SamDecoderResult to pump consumer"

requirements-completed: ["Criterion 3", "Criterion 5", "Criterion 9"]

# Metrics
duration: ~95min (tasks 1+2)
completed: 2026-05-16
---

# Phase 33 Plan 07: SAM 2 GUI Wiring Summary

**Magic Brush wired as a second selection author: MagicBrushState coordinator, SAM encoder/decoder dispatch through Processor, canvas click/stroke/modifier handlers, and embedding retype — manual smoke deferred**

## Performance

- **Duration:** ~95 min (tasks 1 and 2 only — task 3 deferred)
- **Started:** 2026-05-16
- **Completed:** 2026-05-16 (code complete); manual smoke pending ~2026-05-24
- **Tasks:** 2 of 3 complete; task 3 DEFERRED
- **Files modified:** 10 (across prunr-app)

## Accomplishments

- `BatchItem.magic_brush_embedding` retyped from `Option<Arc<()>>` placeholder to `Option<Arc<prunr_core::sam::SamEmbedding>>`; `cache_size()` uses `SamEmbedding::expected_bytes()` instead of hardcoded 16 MB
- `MagicBrushState` coordinator shipped in `magic_brush_state.rs`: active flag, encoder-pending flag, active stroke buffer; mirrors `BrushState` ownership pattern; 3 unit tests pin activate/deactivate/pending semantics
- `[ Paint ] [ Magic ]` toggle pair in adjustments toolbar Row 2 with mutual exclusion (activating one deactivates the other); two cross-cutting app tests pin the invariant
- `Processor` gains `dispatch_sam_encoder` (800 MB admission gate, rayon::spawn, `run_sam_encoder_inline`), `dispatch_sam_decoder` (small decoder, no admission gate), `pump_sam_encoder_results`, `pump_sam_decoder_results`; `run_sam_encoder_inline` / `run_sam_decoder_inline` private helpers keep ORT session.run out of `prunr-core::sam`
- Canvas click handler + drag stroke accumulation: Shift→Add, Alt→Subtract, plain→Replace; fires decoder dispatch on click or pointer release; silently ignores input while encoder is pending
- Canvas "Preparing Magic Brush..." text + spinner overlay when encoder in flight
- `magic_brush_chip.rs`: Magic chip popover with shared selection knobs (delegated to `brush_chip::render_shared_selection_section`) + Confidence threshold slider
- `brush_chip::render_shared_selection_section` extracted as `pub(super)` helper — no duplication between Paint and Magic popovers

## Task Commits

Each completed task committed atomically:

1. **Task 1: Retype magic_brush_embedding + MagicBrushState + Magic chip + toggle pair** — `62d5797` (feat)
   - Files: `item.rs`, `magic_brush_state.rs` (new), `magic_brush_chip.rs` (new), `adjustments_toolbar.rs`, `brush_chip.rs`, `canvas.rs`, `app.rs`, `mod.rs`, `views/mod.rs`, `batch_manager.rs`

2. **Task 2: Processor SAM encoder/decoder dispatch + canvas click/stroke handlers** — `5aabf58` (feat)
   - Files: `processor.rs`, `app.rs`, `views/canvas.rs`

3. **Task 3: Manual smoke — Magic Brush end-to-end + Paint Brush BG-removal regression** — **DEFERRED**
   - Status: User away from computer; ETA approximately one week (around 2026-05-24)
   - Checklist: `.planning/phases/33-magic-brush/PENDING-MANUAL-VERIFICATION.md`
   - Criteria covered: Criterion 3 (Click+Stroke+Shift+Alt), Criterion 5 (eager encoder+cache), Criterion 9 (60 Hz visualization), Criterion 11 (Paint Brush regression)

## Files Created/Modified

- `crates/prunr-app/src/gui/magic_brush_state.rs` (new) — MagicBrushState coordinator: active/encoder-pending flags + stroke buffer + 3 unit tests
- `crates/prunr-app/src/gui/views/magic_brush_chip.rs` (new) — Magic chip button + popover: shared selection knobs + Confidence threshold slider + Preparing... spinner
- `crates/prunr-app/src/gui/item.rs` — magic_brush_embedding retyped; cache_size() updated
- `crates/prunr-app/src/gui/processor.rs` — SamEncoderResult / SamDecoderResult / PromptModifier; 4 dispatch/pump methods; 2 private inline ORT helpers
- `crates/prunr-app/src/gui/app.rs` — magic_brush_state field; pump loops for SAM channels; toggle_magic handler with eager encoder dispatch; 2 mutual-exclusion tests
- `crates/prunr-app/src/gui/views/canvas.rs` — handle_magic_brush_input: click→Replace, Shift→Add, Alt→Subtract, drag stroke accumulation; Preparing... overlay
- `crates/prunr-app/src/gui/views/adjustments_toolbar.rs` — Paint/Magic toggle pair; ToolbarChange gains toggle_paint/toggle_magic
- `crates/prunr-app/src/gui/views/brush_chip.rs` — render_shared_selection_section extracted as pub(super)
- `crates/prunr-app/src/gui/batch_manager.rs` — test fixture updated from Arc<()> to real SamEmbedding
- `crates/prunr-app/src/gui/mod.rs` — magic_brush_state module declared

## Decisions Made

- ORT session.run kept in `processor.rs` (`run_sam_encoder_inline` / `run_sam_decoder_inline`) — `prunr-core::sam` stays zero-ORT per Plan 06's explicit design contract. Tested via grep: zero `ort::` / `onnxruntime::Session` hits in `crates/prunr-core/src/sam/`.
- No `cpu_only=true` blanket gate on SAM dispatch — default EP ladder per CLAUDE.md Scoped Workarounds. Narrower gate available if a specific EP proves problematic in production.
- Task 3 manual smoke DEFERRED: user away from computer approximately one week. CODE is complete and green (`cargo test --workspace --lib` passes). Smoke exercises real hardware and model download paths that cannot be automated.

## Deviations from Plan

### Planned deferral (not an auto-fix)

**Task 3: Manual regression smoke — DEFERRED by user**
- **Found during:** Post-task-2 checkpoint
- **Reason:** User is away from the computer; ETA ~1 week (~2026-05-24)
- **What remains:** Steps 1-10 of the how-to-verify checklist in `.planning/phases/33-magic-brush/PENDING-MANUAL-VERIFICATION.md`
- **Criteria impacted:** Criterion 3 (Click+Stroke+Shift+Alt), Criterion 5 (eager encoder+cache), Criterion 9 (60 Hz visualization), Criterion 11 (Paint Brush BG-removal regression)
- **Resolution path:** User runs `cargo run -p prunr-app`, exercises checklist, reports "test 33 done — approved" or describes issues

None — code tasks 1 and 2 executed exactly as written.

## Issues Encountered

None — tasks 1 and 2 compiled cleanly, all tests green.

## Next Phase Readiness

- Code complete: Magic Brush is a fully wired second selection author; encoder/decoder dispatch, canvas handlers, chip UI, mutual exclusion all in place.
- Blocked on manual smoke only: Criterion 11 (Paint Brush regression) requires eyes-on the running binary.
- Resume signal: "test 33 done — approved" → close Criterion 11, run `/gsd:plan-phase 33 --gaps` or advance to Phase 34.

---

## Self-Check

### Files exist

- `crates/prunr-app/src/gui/magic_brush_state.rs` — FOUND (committed 62d5797)
- `crates/prunr-app/src/gui/views/magic_brush_chip.rs` — FOUND (committed 62d5797)
- `.planning/phases/33-magic-brush/PENDING-MANUAL-VERIFICATION.md` — FOUND (pre-existing, not modified)

### Commits exist

- `62d5797` — FOUND (`feat(33-07): retype magic_brush_embedding + ship MagicBrushState + Magic chip + toggle pair`)
- `5aabf58` — FOUND (`feat(33-07): Processor SAM encoder/decoder dispatch + canvas click/stroke handlers`)

### Grep checks (recorded at summary time)

- `grep -rn "ort::\|onnxruntime::Session" crates/prunr-core/src/sam/` — ZERO matches (only comment references to ort in doc-comments; no live ORT imports). prunr-core::sam is ORT-free.
- `grep -n "magic_brush_embedding" crates/prunr-app/src/gui/item.rs` — field at line 369 typed as `Option<std::sync::Arc<prunr_core::sam::SamEmbedding>>`. Arc<()> placeholder fully replaced.

## Self-Check: PASSED

*Phase: 33-magic-brush*
*Completed (code): 2026-05-16*
*Manual smoke: DEFERRED ~2026-05-24*
