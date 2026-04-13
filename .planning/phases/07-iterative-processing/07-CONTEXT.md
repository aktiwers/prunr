# Phase 7: Iterative Processing - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Add "chain mode" to Prunr that allows processing the result of a previous processing step instead of always starting from the original source image. Each process step chains from the previous result, with full undo/redo history through all layers.

</domain>

<decisions>
## Implementation Decisions

### Chain Mode Toggle
- A toggle in General settings: "Process original" (default) vs "Process result" (chain mode)
- When chain mode is on, clicking Process uses the current `result_rgba` as input
- When chain mode is off, Process always uses the original `source_bytes`
- Persisted in settings.json

### History Stack
- Replace single `undo_result_rgba: Option<Arc<RgbaImage>>` with `Vec<Arc<RgbaImage>>` history
- Each process step pushes the current result onto the history before overwriting
- Ctrl+Z pops the last result from history and restores it
- Ctrl+Y pushes current result and moves forward (redo stack)
- Max depth configurable (default 10) — oldest entries dropped when exceeded

### Worker Pipeline Changes
- WorkerMessage gains a new field: source image bytes OR result RGBA
- In chain mode, the worker receives the result_rgba converted to bytes/DynamicImage
- The edge detection AfterBgRemoval mode already does this internally — generalize that pattern

### Memory Management
- 4K RGBA image = ~48MB. At 10 history depth = ~480MB per image
- For batch with 18 images × 10 depth = ~8.6GB — need per-image limits
- History depth setting in General tab with a sensible default
- Could also limit total memory (e.g., 2GB) and evict oldest entries across all images

### Claude's Discretion
- Exact UI placement of history depth slider
- Whether to show a visual indicator of history depth (e.g., "3/10 layers")
- Whether redo stack is cleared on new processing (standard behavior)

</decisions>

<canonical_refs>
## Canonical References

### Core Data Model
- `crates/prunr-app/src/gui/app.rs` — BatchItem struct (undo_result_rgba field), process_items(), handle_undo/redo
- `crates/prunr-app/src/gui/settings.rs` — Settings struct, LineMode enum
- `crates/prunr-app/src/gui/worker.rs` — WorkerMessage, processing pipeline dispatch

### Processing Pipeline
- `crates/prunr-core/src/pipeline.rs` — process_image_with_mask (takes &[u8] bytes)
- `crates/prunr-core/src/edge.rs` — EdgeEngine::detect (takes &DynamicImage)
- `crates/prunr-core/src/formats.rs` — encode_rgba_png, apply_background_color

</canonical_refs>

<specifics>
## Specific Ideas

- The worker already handles DynamicImage input for edge detection (AfterBgRemoval mode wraps RgbaImage → DynamicImage). The chain mode generalizes this: any mode can receive either source_bytes or result_rgba.
- The undo/redo system already exists with single-level undo. Extending to a stack is a data structure change, not an architectural one.
- Consider: should "Process All" in batch mode chain from each image's individual result, or process all from originals? Probably per-image — each image has its own history.

</specifics>

<deferred>
## Deferred Ideas

- Layer naming/labels (e.g., "BG removal → Lines → Threshold adjustment")
- Layer panel showing the full history visually
- Selective layer deletion (remove a middle layer and reprocess)
- Branching history (multiple paths from the same state)

</deferred>

---

*Phase: 07-iterative-processing*
*Context gathered: 2026-04-13*
