---
phase: 02-core-inference-engine
verified: 2026-04-06T00:00:00Z
status: passed
score: 7/7 requirements verified, 5/5 success criteria met
re_verification: false
---

# Phase 2: Core Inference Engine Verification Report

**Phase Goal:** Users (and the CLI/GUI) can call `process_image()` and receive a pixel-accurate transparent PNG whose mask matches rembg Python output on the same input, with GPU used automatically when available
**Verified:** 2026-04-06
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Reference test passes for all 3 test images at >=95% pixel match | VERIFIED | `test_rembg_reference` in reference_test.rs; 3 reference masks present in tests/references/; stated all 43 tests pass |
| 2 | `process_image()` runs to completion on silueta and u2net on CPU | VERIFIED | `test_process_image_produces_valid_rgba_png`, `test_model_selection_silueta_and_u2net` in reference_test.rs; full pipeline wired in pipeline.rs |
| 3 | Active execution provider name is queryable via public API | VERIFIED | `active_provider()` on `InferenceEngine` trait; `test_active_provider_queryable` integration test; `detect_active_provider()` cfg-chain in engine.rs |
| 4 | Images exceeding 8000px return a warning result rather than silently processing | VERIFIED | `check_large_image()` called in `process_image()` before preprocessing; `test_large_image_warning` and `test_process_image_large_returns_err`; `LARGE_IMAGE_LIMIT = 8000` |
| 5 | `batch_process()` accepts a progress callback and uses a rayon thread pool with no thread oversubscription | VERIFIED | `ort_intra_threads()` formula `(num_cpus / workers).max(1)` in batch.rs; `ThreadPoolBuilder` with `num_threads(jobs)`; per-worker `OrtEngine::new()` (no shared sessions) |

**Score:** 5/5 success criteria verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/prunr-core/src/types.rs` | CoreError, ModelKind, ProgressStage, ProcessResult, constants | VERIFIED | 5 CoreError variants (Io, Model, Inference, ImageFormat, LargeImage), ModelKind (Silueta/U2net), ProgressStage (6 variants), ProcessResult (rgba_bytes/active_provider), LARGE_IMAGE_LIMIT=8000, DOWNSCALE_TARGET=4096 |
| `crates/prunr-core/src/preprocess.rs` | preprocess() → Array4<f32> [1,3,320,320] | VERIFIED | Lanczos3 resize, max(max_pixel, 1e-6) normalization, NCHW layout, 4 passing unit tests |
| `crates/prunr-core/src/postprocess.rs` | postprocess() → RgbaImage at original dims | VERIFIED | Min-max normalization only (no sigmoid, no threshold), Lanczos3 mask resize, 4 unit tests; "NO sigmoid" documented in code comments |
| `crates/prunr-core/src/formats.rs` | load_image_from_{path,bytes}, check_large_image, downscale_image, encode_rgba_png | VERIFIED | All 5 functions present, magic-byte format detection, aspect-ratio-preserving Lanczos3 downscale, PNG magic byte validation in test |
| `crates/prunr-core/src/engine.rs` | OrtEngine with session management, active_provider(), InferenceEngine trait | VERIFIED | Session behind Mutex<Session> (soundness improvement over plan's session() accessor), with_session() closure API, cfg-chain EP detection, InferenceEngine trait impl |
| `crates/prunr-core/src/pipeline.rs` | process_image() orchestration with 6 progress stages | VERIFIED | All 6 ProgressStage variants reported in order, check_large_image guard before tensor allocation, input name queried at runtime via session.inputs()[0].name() |
| `crates/prunr-core/src/batch.rs` | batch_process() with rayon parallelism and ORT thread balancing | VERIFIED | ThreadPoolBuilder, ort_intra_threads formula, per-worker OrtEngine (no Arc<Mutex<Session>>), order-preserving Vec fill pattern |
| `crates/prunr-core/src/lib.rs` | Public re-exports for all Phase 2 types and functions | VERIFIED | All exports present: OrtEngine, InferenceEngine, process_image, batch_process, all format functions, all types and constants |
| `crates/prunr-core/tests/reference_test.rs` | Integration test suite with CORE-05 hard gate | VERIFIED | 9 test functions present, test_rembg_reference uses U2net + 95% tolerance, all 7 requirements covered |
| `scripts/generate_references.py` | Reference mask generator with exact rembg settings | VERIFIED | u2net, alpha_matting=False, post_process_mask=False, VERSIONS.txt recording, clear error messages |
| `tests/references/` | 3 reference masks committed | VERIFIED | car-1_u2net_mask.png, car-2_u2net_mask.png, car-3_u2net_mask.png, VERSIONS.txt all present |
| `tests/test_images/` | 3 test images + README | VERIFIED | car-1.jpg, car-2.jpg, car-3.jpg, README.md all present |
| `crates/prunr-core/Cargo.toml` | ort, ndarray, image, rayon, num_cpus dependencies | VERIFIED | All deps declared; dev-models feature gate correctly wired to prunr-models/dev-models |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `lib.rs` | `types.rs` | `pub use types::{CoreError, ModelKind, ProgressStage, ProcessResult}` | WIRED | All 4 types + 2 constants exported |
| `lib.rs` | `pipeline.rs` | `pub use pipeline::process_image` | WIRED | Confirmed in lib.rs line 14 |
| `lib.rs` | `batch.rs` | `pub use batch::batch_process` | WIRED | Confirmed in lib.rs line 15 |
| `lib.rs` | `engine.rs` | `pub use engine::{InferenceEngine, OrtEngine}` | WIRED | Confirmed in lib.rs line 9 |
| `pipeline.rs` | `preprocess.rs` | `preprocess::preprocess()` | WIRED | `use crate::preprocess::preprocess` + called at line 53 |
| `pipeline.rs` | `postprocess.rs` | `postprocess::postprocess()` | WIRED | `use crate::postprocess::postprocess` + called at line 81 |
| `pipeline.rs` | `engine.rs` | `engine.with_session()` | WIRED | `with_session` closure wraps all ORT inference calls at line 58 |
| `engine.rs` | `prunr-models` | `prunr_models::silueta_bytes()` / `u2net_bytes()` | WIRED | cfg-gated model byte loading for both dev-models and production paths |
| `batch.rs` | `pipeline.rs` | `pipeline::process_image` per image | WIRED | `use crate::pipeline::process_image` + called in par_iter at line 84 |
| `batch.rs` | `rayon::ThreadPoolBuilder` | `build_batch_pool(jobs)` | WIRED | `ThreadPoolBuilder::new().num_threads(jobs.max(1)).build()` |
| `batch.rs` | `num_cpus` | `ort_intra_threads` calculation | WIRED | `num_cpus::get()` in `ort_intra_threads()` |
| `reference_test.rs` | `prunr_core::process_image` | Calls process_image on test images | WIRED | `use prunr_core::process_image` + called in test_rembg_reference |
| `reference_test.rs` | `tests/references/` | Loads reference masks for pixel comparison | WIRED | `references_dir()` resolves to workspace root + tests/references; 3 mask files present |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CORE-01 | 02-01, 02-03, 02-06 | User can remove background from single image, receive transparent PNG | SATISFIED | `process_image()` returns `ProcessResult { rgba_bytes, active_provider }`; PNG magic bytes verified in tests; transparency verified in `test_process_image_produces_valid_rgba_png` |
| CORE-02 | 02-01, 02-04, 02-06 | User can select between silueta and u2net models | SATISFIED | `ModelKind` enum with both variants; `OrtEngine::new(ModelKind, usize)` branches on model kind; `test_model_selection_silueta_and_u2net` verifies both load |
| CORE-03 | 02-03, 02-06 | Inference uses GPU when available, falls back to CPU | SATISFIED | EP priority list (CUDA→CoreML→DirectML→CPU) in `OrtEngine::new_from_bytes`; `active_provider()` returns current EP name; `test_active_provider_queryable` verifies non-empty string |
| CORE-04 | 02-01, 02-03, 02-06 | User sees progress indicator while inference runs | SATISFIED | All 6 `ProgressStage` variants (Decode, Resize, Normalize, Infer, Postprocess, Alpha) reported in `process_image()`; `test_progress_callback_all_stages` verifies >=5 stages including Infer |
| CORE-05 | 02-02, 02-05, 02-06 | Pipeline produces pixel-accurate results matching rembg Python output | SATISFIED | `test_rembg_reference` hard gate at 95% pixel match (tolerance ±5/255) on 3 car images; Lanczos3 + max-pixel norm + min-max postprocess verified; all 43 tests stated to pass |
| LOAD-03 | 02-02, 02-06 | App accepts PNG, JPEG, WebP, BMP (SVG deferred to Phase 6) | SATISFIED | `load_image_from_bytes()` uses magic-byte format detection; `image` crate with jpeg, png, webp, bmp features; `test_format_support_png_jpeg_webp_bmp` integration test |
| LOAD-04 | 02-01, 02-02, 02-06 | User prompted to downscale if image exceeds 8000px | SATISFIED | `check_large_image()` returns `CoreError::LargeImage { width, height, limit }` for >8000px; called in `process_image()` before tensor allocation; `downscale_image()` with Lanczos3 provided for callers; `test_large_image_warning` and `test_downscale_image_preserves_aspect_ratio` |

**All 7 phase requirements: SATISFIED**

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None found | — | — | — |

Checks performed:
- No TODO/FIXME/HACK/PLACEHOLDER comments in any implementation file
- No `sigmoid`, `threshold`, or `0.5` in postprocess.rs (only in test/comment strings about the absence of sigmoid)
- No `Arc<Mutex<Session>>` in batch.rs (sessions are per-worker, not shared)
- No `return null` / empty implementations / stub bodies in any module
- No hardcoded input tensor name (queried via `session.inputs()[0].name()` at runtime)

### Notable Implementation Divergences from Plan (Non-Blocking)

The actual implementation diverges from the plan spec in one intentional way:

**engine.rs: `with_session()` closure API instead of plan's `session() -> &Session`**

The plan specified `pub(crate) fn session(&self) -> &Session`. The implementation instead uses:
```rust
pub(crate) fn with_session<T, F>(&self, f: F) -> Result<T, CoreError>
where F: FnOnce(&mut Session) -> Result<T, CoreError>
```
This is a soundness improvement: ORT's `Session::run()` requires `&mut self`, and the plan's `session() -> &Session` would have required unsafe or awkward workarounds. The `Mutex<Session>` + closure pattern is the correct solution. All downstream code (pipeline.rs) uses this correctly.

### Human Verification Required

| # | Test | Expected | Why Human |
|---|------|----------|-----------|
| 1 | Run `cargo test -p prunr-core --features dev-models` and inspect CORE-05 output | Lines like `CORE-05 [car-1]: 97.x% pixel match ... PASS` for all 3 cars | Numeric pixel match percentages are runtime outputs; the prompt states all 43 tests pass including test_rembg_reference, which is accepted as ground truth |
| 2 | On a machine with CUDA GPU, verify `active_provider()` returns "CUDA" | `Active execution provider: CUDA` printed by test_active_provider_queryable | Requires actual GPU hardware; not verifiable in this environment |

Note: Item 1 is marked as human-needed only for completeness. The prompt explicitly states test_rembg_reference passed, which is accepted as authoritative.

### Gaps Summary

No gaps. All 7 requirements are satisfied, all must-have artifacts exist and are substantive (no stubs), all key links are wired. The test suite (34 unit + 9 integration = 43 total) passes in full per the provided test run confirmation.

---

_Verified: 2026-04-06_
_Verifier: Claude (gsd-verifier)_
