# Phase 2: Core Inference Engine - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

ONNX inference pipeline in bgprunr-core: `process_image()` returns a pixel-accurate transparent PNG whose mask matches rembg Python output on the same input. GPU acceleration with CPU fallback. Batch API with parallel processing. Large image handling. No CLI or GUI — this is the shared library that both consume.

</domain>

<decisions>
## Implementation Decisions

### Reference Test Strategy
- Use rembg's own test images from their GitHub repo — same images they test against
- Use rembg's default settings (model: u2net, alpha matting: off, no post-processing)
- A `scripts/generate_references.py` script runs rembg with their defaults on test images, saves reference masks to `tests/references/`
- Reference outputs are committed to the repo as ground truth
- Comparison tolerance: 95% pixel match — accounts for floating-point rounding between Python numpy and Rust ndarray
- This is a **hard gate**: reference test must pass before any CLI or GUI work ships

### Progress Reporting
- Callback closure API: `process_image(img, |stage, pct| { ... })` — caller provides a closure
- **Fine-grained stages**: Decode → Resize → Normalize → Infer → Sigmoid → Threshold → Alpha
- Zero-cost when callback is unused (Option<F> with None)
- CLI uses callback to drive indicatif progress bar
- GUI uses callback to send progress over mpsc channel to UI thread

### Large Image Handling
- Core returns a `LargeImageWarning` result when image exceeds 8000px in either dimension
- The **caller decides** what to do: GUI shows dialog (process anyway / downscale), CLI checks `--large-image=downscale|process` flag
- Downscale target: 4096px max dimension (preserving aspect ratio)
- Default behavior (no flag): prompt the user for a choice
- Core provides a `downscale_image(img, max_dim)` utility function

### Batch Parallelism
- `batch_process()` accepts `--jobs N` parameter with **default of 1** (sequential)
- ORT intra-op threads auto-calculated: `num_cpus / rayon_workers` — prevents thread oversubscription
- Per-image callback: `batch_process(images, |image_idx, stage, pct| { ... })` — same fine-grained stages as single image, plus image index
- Results cached: returns `Vec<ProcessResult>` so callers can access all results after batch completes

### Claude's Discretion
- Exact preprocessing constants (ImageNet mean/std values from rembg source code)
- ORT session configuration (memory arena, optimization level)
- ndarray tensor manipulation details
- Error variants to add to CoreError for inference-specific failures

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture
- `ARCHITECTURE.md` — Inference pipeline detail (step 1-10), GPU EP strategy, threading model, data flow diagrams

### Existing Code
- `crates/bgprunr-core/src/engine.rs` — InferenceEngine trait (currently just `active_provider()`, needs `process_image()` and `batch_process()` methods)
- `crates/bgprunr-core/src/types.rs` — CoreError enum (needs inference-specific variants)
- `crates/bgprunr-core/src/lib.rs` — Module exports
- `crates/bgprunr-models/src/lib.rs` — Model byte loading (SILUETA_BYTES, U2NET_BYTES + dev-models feature)

### Research
- `.planning/research/STACK.md` — ort 2.0.0-rc.12 API, ndarray 0.16, execution provider feature flags
- `.planning/research/PITFALLS.md` — Preprocessing mismatch (#1 risk), GPU EP silent fallback, thread oversubscription
- `.planning/research/ARCHITECTURE.md` — Session management, one Session per model, preprocessing constants

### rembg Reference
- `https://github.com/danielgatis/rembg` — Source of truth for preprocessing pipeline, default settings, test images

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `InferenceEngine` trait in `engine.rs`: Currently has `active_provider() -> &str`. Phase 2 adds `process_image()` and `batch_process()` methods.
- `CoreError` in `types.rs`: Has `Io` and `Model` variants. Phase 2 adds `Inference`, `ImageFormat`, `LargeImage` variants.
- `bgprunr-models` crate: `model_bytes(name: &str) -> &[u8]` function already provides model bytes. Phase 2 uses this to create ORT sessions.

### Established Patterns
- thiserror enums with `#[from]` conversions — follow this pattern for new error types
- Trait-based abstractions — `InferenceEngine` trait, concrete `OrtEngine` implementation
- `dev-models` feature flag — tests use filesystem models, production uses embedded

### Integration Points
- `bgprunr-core` is consumed by `bgprunr-app` (the single binary) — Phase 3 (CLI) and Phase 4 (GUI) depend on the API designed here
- `process_image()` signature must work for both single-image CLI and GUI worker thread use cases
- `batch_process()` must work for both CLI batch mode and GUI batch queue

</code_context>

<specifics>
## Specific Ideas

- The preprocessing pipeline must exactly match rembg's Python implementation: HWC→CHW transposition, divide by 255.0, then ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Use rembg's same defaults when generating reference outputs — don't customize settings that would make comparison invalid
- The callback closure pattern `process_image(img, |stage, pct| { ... })` matches Rust idioms and is zero-cost for callers that don't need progress

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-core-inference-engine*
*Context gathered: 2026-04-07*
