# Phase 33 — Pending Manual Verification

**Deferred:** 2026-05-17 (user away from computer, ETA ~1 week)
**Closes:** Phase 33 Criterion 11 (Paint Brush BG-removal feel unchanged)
**Plan:** 33-07 SAM 2 GUI wiring — task 3 of 3

## What's been verified by code

- All 7 plans complete with atomic commits.
- 558+ prunr-app tests, 991+ workspace tests green at last sample.
- Paint Brush regression test (33-04) pins the per-model dispatch table behaviour.
- Hot-path discipline grep-verified for `selection_overlay.rs`.

## What needs hands-on confirmation

Run `cargo run -p prunr-app`, load an image, exercise:

1. **Background removal baseline** — run a default BG-removal pass; result should be visually identical to pre-Phase-33 output. No subtle alpha, halo, or edge differences.

2. **Paint Brush feel** — switch to Paint, draw strokes on the canvas:
   - No lag while drawing.
   - No visual artifacts during or after a stroke.
   - Strokes commit to the mask correctly (visible in the selection overlay).
   - Undo / redo of strokes behaves as before.

3. **Tool toggle exclusivity** — `[ Paint ] [ Magic ]` are mutually exclusive:
   - Clicking Paint while Magic is active deactivates Magic (and vice versa).
   - There is no "both active" state at any frame.

4. **Paint-mode click semantics** — while Paint is active, canvas click / drag must draw brush strokes, NOT fire SAM Magic clicks.

5. **Magic Brush eager encoder** — switch to Magic with an item loaded:
   - "Preparing Magic Brush..." overlay appears immediately.
   - Overlay clears when the SAM encoder finishes.
   - If system has < 800 MB free RAM, an error toast appears instead (do not test this — environmental).

6. **Magic Brush modifier semantics** — with embedding ready:
   - Plain click → mask replaces current selection.
   - Shift + click → mask ADDS to selection.
   - Alt + click → mask SUBTRACTS from selection.

7. **Keyboard bindings on the selection action bar** (closes Criterion 7) — with a selection active and no text field focused:
   - `Del` → Delete (alpha-cut).
   - `Ctrl+C` → Copy to clipboard.
   - `Ctrl+X` → Cut to clipboard.
   - `Enter` → Invert.  ← *added 2026-05-17 to close gap*
   - `Esc` → Clear.

## What to report back

Either:
- "approved — Phase 33 smoke clean" → I'll close Criterion 11, mark phase verified, and finalize.
- "issues: <describe>" → I'll route to gap-closure planning (`/gsd:plan-phase 33 --gaps`).

## Resume command

When ready, just say "test 33 done — approved" or describe issues. No need to re-invoke the workflow — STATE.md tracks the open verification.
