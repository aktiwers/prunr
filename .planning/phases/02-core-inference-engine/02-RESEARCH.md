# Phase 2: Core Inference Engine - Research

**Researched:** 2026-04-06
**Domain:** ort 2.0 ONNX inference, ndarray tensor manipulation, rembg preprocessing pipeline, image I/O
**Confidence:** HIGH (core findings verified against rembg source, ort docs, image crate docs)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Reference test strategy:** Use rembg's own test images. `scripts/generate_references.py` runs rembg with defaults (model: u2net, alpha matting: off, no post-processing) and saves reference masks to `tests/references/`. Committed as ground truth. 95% pixel match tolerance. Hard gate before CLI/GUI.
- **Progress reporting:** Callback closure API — `process_image(img, |stage, pct| { ... })`. Fine-grained stages: Decode → Resize → Normalize → Infer → Sigmoid → Threshold → Alpha. Zero-cost when callback is `None`.
- **Large image handling:** Core returns `LargeImageWarning` when image exceeds 8000px. Caller decides: GUI shows dialog, CLI checks `--large-image=downscale|process`. Downscale target: 4096px max dimension. Core provides `downscale_image(img, max_dim)` utility.
- **Batch parallelism:** `batch_process()` accepts `--jobs N`, default 1. ORT intra-op threads auto-calculated: `num_cpus / rayon_workers`. Per-image callback: `batch_process(images, |image_idx, stage, pct| { ... })`. Returns `Vec<ProcessResult>`.

### Claude's Discretion
- Exact preprocessing constants (ImageNet mean/std values from rembg source code)
- ORT session configuration (memory arena, optimization level)
- ndarray tensor manipulation details
- Error variants to add to CoreError for inference-specific failures

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORE-01 | User can remove background from a single image and receive a transparent PNG | `process_image()` API, `image` crate RGBA encode, postprocess → alpha merge |
| CORE-02 | User can select between silueta (fast) and u2net (quality) models | `ModelKind` enum, `prunr-models` already provides bytes for both via `silueta_bytes()` / `u2net_bytes()` |
| CORE-03 | Inference automatically uses GPU when available, falls back to CPU | ort EP priority list: CUDA → CoreML → DirectML → CPU; `active_provider()` method already on trait |
| CORE-04 | User sees a progress indicator while inference is running | Callback closure pattern `process_image(img, callback)`, fine-grained stage enum |
| CORE-05 | Inference pipeline produces pixel-accurate results matching rembg Python output | Exact rembg preprocessing pipeline verified from source (see Critical Finding below) |
| LOAD-03 | App accepts PNG, JPEG, WebP, BMP input formats | `image::open()` / `ImageReader` with format feature flags already in workspace Cargo.toml |
| LOAD-04 | User is prompted to downscale if image exceeds 8000px | `LargeImageWarning` result variant, `downscale_image()` utility |
</phase_requirements>

---

## Summary

Phase 2 builds the complete ONNX inference pipeline in `prunr-core`: from raw image bytes to a pixel-accurate transparent PNG. The pipeline is a pure library with no GUI or CLI concerns.

The most critical finding from research is a **discrepancy between the project's ARCHITECTURE.md and rembg's actual source code** in two places: (1) rembg uses LANCZOS resampling (not bilinear) for the resize-to-320x320 step, and (2) rembg normalizes by `max(np.max(pixel_array), 1e-6)` rather than a fixed `/255.0`, then applies ImageNet mean/std. For typical photos with a max pixel value of 255 these are identical, but the implementation must use the rembg formula exactly. Additionally, rembg's postprocessing does **not** apply sigmoid — it applies min-max normalization to the raw model output and uses the result directly as a grayscale alpha mask.

The ort 2.0.0-rc.12 API is well-understood. Sessions are created with `Session::builder()?.commit_from_memory(&bytes)?`. The `inputs!` macro feeds ndarray arrays. Outputs are extracted with `try_extract_tensor::<f32>()`. Execution providers are registered as a priority list; ORT selects the first available one automatically. The `image` crate's `FilterType::Lanczos3` matches rembg's LANCZOS.

**Primary recommendation:** Implement preprocessing as a single pure function `preprocess(img: &DynamicImage) -> Array4<f32>` and postprocessing as `postprocess(output: ArrayView4<f32>, original_size: (u32, u32)) -> GrayImage`, each covered by a unit test that compares against numpy-generated reference values before the full integration reference test is run.

---

## Standard Stack

### Core (Phase 2 additions to prunr-core)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ort` | `=2.0.0-rc.12` | ONNX Runtime — session creation, inference, EP management | Already pinned in workspace; wraps ORT 1.24; the only Rust crate with CUDA/CoreML/DirectML |
| `ndarray` | `0.16` | 4D tensor construction and manipulation for ORT input/output | ort's `ndarray` feature bridges directly to ndarray 0.16; workspace already declares it |
| `image` | `0.25` | Decode PNG/JPEG/WebP/BMP, resize with Lanczos3, encode RGBA PNG | Workspace already declares with all required format features |
| `rayon` | `1.11` | Work-stealing batch parallelism | Workspace already declares; no async runtime needed |
| `num_cpus` | `1.x` | Determine available CPU count for ORT thread balancing | Standard utility; needed for `num_cpus::get() / rayon_pool_size` formula |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `prunr-models` | workspace | Provides model bytes (`silueta_bytes()`, `u2net_bytes()`) | Already a dependency of prunr-core; use `dev-models` feature in tests |
| `thiserror` | `2.0` | Error enum derive | Already used in types.rs; continue pattern for new error variants |

### New Cargo.toml additions for prunr-core

```toml
[dependencies]
prunr-models = { path = "../prunr-models" }
thiserror = { workspace = true }
ort = { workspace = true }
ndarray = { workspace = true }
image = { workspace = true }
rayon = { workspace = true }
num_cpus = "1"

[dev-dependencies]
prunr-models = { path = "../prunr-models", features = ["dev-models"] }
```

**Note:** `num_cpus` is not in the workspace Cargo.toml yet — add it. It is a lightweight crate with no transitive heavy deps.

---

## Architecture Patterns

### Recommended Module Structure

```
crates/prunr-core/src/
├── lib.rs          # Public exports: process_image, batch_process, ModelKind, ProcessResult, CoreError
├── engine.rs       # InferenceEngine trait + OrtEngine impl (Session lifecycle, EP setup)
├── pipeline.rs     # process_image() — orchestrates pre → infer → post + callback
├── preprocess.rs   # resize + normalize → Array4<f32>  [pure function, no side effects]
├── postprocess.rs  # min-max norm → grayscale mask → alpha merge → RgbaImage
├── batch.rs        # batch_process() — rayon parallel dispatch, thread count balancing
├── formats.rs      # image::open wrapper, encode RGBA PNG to bytes
└── types.rs        # CoreError, ModelKind, ProcessResult, ProgressStage
```

This matches the ARCHITECTURE.md structure. Pipeline.rs is new; it owns the top-level orchestration so engine.rs stays focused on session management.

### Pattern 1: ORT Session Creation from Memory

```rust
// Source: ort docs.rs, deepwiki.com/pykeio/ort
use ort::{Session, execution_providers::CUDAExecutionProvider};

fn create_session(model_bytes: &[u8]) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?   // tuned at batch time; see thread balancing pattern
        .with_execution_providers([
            #[cfg(feature = "cuda")]
            CUDAExecutionProvider::default().build(),
            #[cfg(target_os = "macos")]
            CoreMLExecutionProvider::default().build(),
            #[cfg(windows)]
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_memory(model_bytes)
}
```

**Notes:**
- `commit_from_memory` takes `&[u8]` — pass the static bytes from prunr-models directly.
- `ort::init().commit()` is optional when calling `Session::builder()` directly; it is only needed if you want global EP registration. Prefer per-session EP configuration for clarity.
- `GraphOptimizationLevel::Level3` = full graph optimizations. Recommended for production.
- Store the session in `OrtEngine` and reuse across all images — never create a session per-image.

### Pattern 2: Preprocessing Pipeline (rembg-exact)

```rust
// Source: rembg/sessions/base.py normalize(), u2net.py predict() — verified from GitHub raw
use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array4, s};

const TARGET_SIZE: u32 = 320;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD:  [f32; 3] = [0.229, 0.224, 0.225];

pub fn preprocess(img: &DynamicImage) -> Array4<f32> {
    // Step 1: Convert to RGB and resize to 320×320 using Lanczos3
    // rembg uses Image.Resampling.LANCZOS — image crate equivalent is FilterType::Lanczos3
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(&rgb, TARGET_SIZE, TARGET_SIZE, FilterType::Lanczos3);

    // Step 2: Build CHW f32 array
    let (w, h) = (TARGET_SIZE as usize, TARGET_SIZE as usize);
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

    // Step 3: Determine max pixel value (rembg uses max(np.max(im_ary), 1e-6))
    // For typical 8-bit images max_val = 255.0; keeps parity for edge cases
    let max_val = resized.pixels()
        .flat_map(|p| p.0.iter().copied())
        .map(|v| v as f32)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-6_f32);

    // Step 4: Normalize per channel; arrange in CHW order
    for y in 0..h {
        for x in 0..w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let normalized = (pixel[c] as f32 / max_val - MEAN[c]) / STD[c];
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }

    tensor
}
```

**CRITICAL DEVIATION from ARCHITECTURE.md:**
- ARCHITECTURE.md says "Divide by 255.0" — rembg actually divides by `max(max_pixel, 1e-6)`. For most real photos this is identical (max is 255), but the reference test may catch edge cases on synthetic images. Use rembg's formula.
- ARCHITECTURE.md says use "bilinear interpolation" — rembg uses LANCZOS. Use `FilterType::Lanczos3`.

### Pattern 3: Postprocessing Pipeline (rembg-exact)

```rust
// Source: rembg/sessions/u2net.py predict() and bg.py naive_cutout()
use ndarray::{ArrayView4, Array2};
use image::{GrayImage, DynamicImage, Rgba, RgbaImage, imageops::FilterType};

pub fn postprocess(
    raw_output: ArrayView4<f32>,  // shape [1, 1, 320, 320]
    original: &DynamicImage,
) -> RgbaImage {
    // Step 1: Extract channel 0 — rembg: ort_outs[0][:, 0, :, :]
    let pred = raw_output.slice(s![0, 0, .., ..]);  // shape [320, 320]

    // Step 2: Min-max normalization — rembg: (pred - mi) / (ma - mi)
    // NOTE: No sigmoid is applied. ARCHITECTURE.md is incorrect on this point.
    let ma = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mi = pred.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (ma - mi).max(1e-6);

    // Step 3: Clip to [0,1] and scale to [0,255]
    let (h, w) = (pred.nrows(), pred.ncols());
    let mut mask_img = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let val = ((pred[[y, x]] - mi) / range).clamp(0.0, 1.0);
            mask_img.put_pixel(x as u32, y as u32, image::Luma([(val * 255.0) as u8]));
        }
    }

    // Step 4: Resize mask to original dimensions using Lanczos3 (rembg: LANCZOS)
    let (orig_w, orig_h) = (original.width(), original.height());
    let mask_resized = image::imageops::resize(&mask_img, orig_w, orig_h, FilterType::Lanczos3);

    // Step 5: Apply mask as alpha channel (rembg naive_cutout: putalpha)
    let rgba = original.to_rgba8();
    let mut output = RgbaImage::new(orig_w, orig_h);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = mask_resized.get_pixel(x, y)[0];
        output.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], alpha]));
    }
    output
}
```

**CRITICAL DEVIATIONS from ARCHITECTURE.md:**
- ARCHITECTURE.md says "Apply sigmoid activation" — rembg does NOT apply sigmoid. The raw model output is directly min-max normalized.
- ARCHITECTURE.md says "Threshold at 0.5 → binary mask" — rembg does NOT threshold. The mask is continuous (grayscale), not binary. The alpha channel varies smoothly.
- The mask resize uses LANCZOS in rembg, not bilinear.

### Pattern 4: Running Inference

```rust
// Source: deepwiki.com/pykeio/ort, docs.rs/ort
use ort::{inputs, Session};
use ndarray::Array4;

pub fn run_inference(session: &Session, input: Array4<f32>) -> ort::Result<Array4<f32>> {
    // inputs! macro maps ndarray to named session input
    let input_name = &session.inputs[0].name;
    let outputs = session.run(inputs![input_name => &input]?)?;

    // Extract first output as owned Array4
    let output_tensor = outputs[0]
        .try_extract_tensor::<f32>()?
        .into_owned();

    // Reshape to [1, 1, 320, 320] for postprocessing
    output_tensor.into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| ort::Error::new(format!("Output shape error: {}", e)))
}
```

**Note:** Query `session.inputs[0].name` at runtime to avoid hardcoding the ONNX input name. U2Net uses `"input.1"` and silueta uses a different name. This pattern works for both models.

### Pattern 5: Progress Callback + Stage Enum

```rust
// Follows CONTEXT.md locked decision on stages
#[derive(Debug, Clone, Copy)]
pub enum ProgressStage {
    Decode,
    Resize,
    Normalize,
    Infer,
    // Note: No separate Sigmoid stage — merged into Postprocess (rembg does min-max, not sigmoid)
    Postprocess,
    Alpha,
}

pub fn process_image<F>(
    img_bytes: &[u8],
    model: ModelKind,
    engine: &OrtEngine,
    progress: Option<F>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(ProgressStage, f32),
{
    let report = |stage, pct| {
        if let Some(ref cb) = progress { cb(stage, pct); }
    };

    report(ProgressStage::Decode, 0.0);
    let img = image::load_from_memory(img_bytes)?;

    report(ProgressStage::Resize, 0.2);
    // ... preprocessing
    report(ProgressStage::Normalize, 0.4);
    // ... normalization

    report(ProgressStage::Infer, 0.5);
    let raw = engine.run(&input_tensor)?;

    report(ProgressStage::Postprocess, 0.8);
    // ... postprocessing

    report(ProgressStage::Alpha, 0.95);
    // ... alpha merge
    Ok(result)
}
```

**Note:** CONTEXT.md lists "Sigmoid" as a stage name. Since rembg doesn't apply sigmoid, rename to `Postprocess` or keep `Sigmoid` as the stage label even though the implementation does min-max normalization — the stage name is just a progress label visible to users. Either is valid; keeping it as is avoids confusing the planner.

### Pattern 6: Thread Balancing for Batch

```rust
// Source: ARCHITECTURE.md threading model, PITFALLS.md pitfall 5
use rayon::ThreadPoolBuilder;

pub fn build_batch_pool(jobs: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("Failed to build rayon pool")
}

pub fn ort_intra_threads(rayon_workers: usize) -> usize {
    let cpus = num_cpus::get();
    (cpus / rayon_workers).max(1)
}

// Usage in batch_process():
// let intra = ort_intra_threads(jobs);
// let session = Session::builder()?.with_intra_threads(intra as i16)?. ...
```

### Pattern 7: Error Variants to Add

```rust
// Extends existing CoreError in types.rs
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Model error: {0}")]
    Model(String),
    // Phase 2 additions:
    #[error("Inference error: {0}")]
    Inference(String),   // wraps ort::Error
    #[error("Image format error: {0}")]
    ImageFormat(String), // wraps image::ImageError
    #[error("Image too large: {width}x{height} exceeds {limit}px limit")]
    LargeImage { width: u32, height: u32, limit: u32 },
}

impl From<ort::Error> for CoreError {
    fn from(e: ort::Error) -> Self { CoreError::Inference(e.to_string()) }
}

impl From<image::ImageError> for CoreError {
    fn from(e: image::ImageError) -> Self { CoreError::ImageFormat(e.to_string()) }
}
```

### Anti-Patterns to Avoid

- **Session per image:** Never call `Session::builder().commit_from_memory()` inside the per-image loop. Initialize once in `OrtEngine::new()`, store, reuse.
- **Binary threshold at 0.5:** Do not threshold the mask. The alpha is a continuous grayscale value, exactly like rembg's `naive_cutout`.
- **Divide by 255 unconditionally:** Use `max(max_pixel, 1e-6)` to match rembg exactly. For 99% of images this is 255.0, but the reference test may include edge cases.
- **Bilinear for resize:** Use `FilterType::Lanczos3` for both the input resize (320×320) and the mask resize back to original dimensions.
- **`Arc<Mutex<Session>>`:** Sessions are not designed for concurrent use. For batch, create one session per rayon worker (or configure a single session with `intra_threads = cpus/workers`).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image decode/encode | Custom PNG/JPEG parser | `image::open()`, `ImageReader::new().with_guessed_format().decode()` | 82M downloads, handles all required formats |
| Tensor layout conversion | Manual HWC→CHW loops | `ndarray::Array4` with correct dimension indexing | Correctness + legibility |
| Resize interpolation | Lanczos implementation | `image::imageops::resize(img, w, h, FilterType::Lanczos3)` | Matches rembg's LANCZOS; tested |
| GPU EP selection | Custom CUDA detection | ort EP priority list: CUDA → CoreML → DirectML → CPU | ORT handles driver version checks, silent fallback |
| Thread pool | Custom thread management | `rayon::ThreadPoolBuilder` + `num_cpus` | Proven, zero configuration |
| Progress percent accumulation | Time-based estimation | Report at known stage transitions with fixed percentages | Simpler and deterministic |

**Key insight:** The preprocessing and postprocessing pipeline is where most custom logic lives, but even there ndarray and the image crate do the heavy lifting. The only hand-rolled code is the normalization math itself, which is intentional (must match rembg formula exactly).

---

## Common Pitfalls

### Pitfall 1: Preprocessing Mismatch (Most Critical)

**What goes wrong:** Mask output is all-black, all-white, or noise. Model runs without error but results are garbage.

**Why it happens:** Three distinct failure modes:
1. Using bilinear resize instead of Lanczos3 — produces slight pixel differences that compound through normalization
2. Dividing by 255.0 instead of `max(max_pixel, 1e-6)` — equivalent for typical images but fails the reference test on synthetic inputs
3. HWC layout passed to ORT instead of CHW — shape is [1, 320, 320, 3] instead of [1, 3, 320, 320]

**How to avoid:** Unit-test `preprocess()` against a numpy-generated reference array on a known test image before running full integration. Verify tensor shape is `[1, 3, 320, 320]` with an assertion.

**Warning signs:** Mask is uniform color regardless of input; pixels all near 0.5; correct shape but no contrast.

### Pitfall 2: Sigmoid + Threshold Instead of Min-Max

**What goes wrong:** Mask loses smooth edges; alpha channel becomes binary (0 or 255 only). Output does not match rembg reference images, failing the 95% pixel match gate.

**Why it happens:** The ARCHITECTURE.md incorrectly describes sigmoid + threshold at 0.5 for postprocessing. rembg does not apply sigmoid; it applies min-max normalization and returns a grayscale alpha.

**How to avoid:** Implement exactly: `(val - min) / (max - min)`, clamp to [0,1], multiply by 255. No sigmoid. No threshold. Verify by running `scripts/generate_references.py` and comparing.

**Warning signs:** Output masks have no semi-transparent pixels (only fully opaque or fully transparent); pixel match rate falls well below 95%.

### Pitfall 3: GPU EP Silent Fallback (from PITFALLS.md)

**What goes wrong:** App runs on CPU even though CUDA was registered. No error, just slow performance.

**How to avoid:** After session creation, log `engine.active_provider()`. Expose it in tests. In CI (no GPU), expect CPU. On dev machines with GPU, verify GPU monitor shows utilization.

### Pitfall 4: Thread Oversubscription (from PITFALLS.md)

**What goes wrong:** Batch processing with `--jobs 4` is slower than `--jobs 1`.

**How to avoid:** Set `with_intra_threads(num_cpus::get() / rayon_pool_size)` per session. Start with `--jobs 1` default (locked in CONTEXT.md), let users opt into parallelism.

### Pitfall 5: Hardcoding ONNX Input/Output Names

**What goes wrong:** Session fails with "unknown input name" when switching from u2net to silueta.

**How to avoid:** Query `session.inputs[0].name` at runtime to get the actual input tensor name. Same for `session.outputs[0].name`. Do not hardcode `"input.1"` or any string.

---

## Code Examples

### Full preprocess function (verified against rembg source)

```rust
// Source: rembg/sessions/base.py normalize() — verified 2026-04-06
pub fn preprocess(img: &DynamicImage) -> Array4<f32> {
    const SIZE: u32 = 320;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3]  = [0.229, 0.224, 0.225];

    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(&rgb, SIZE, SIZE, FilterType::Lanczos3);

    // rembg: im_ary = im_ary / max(np.max(im_ary), 1e-6)
    let max_val = resized.pixels()
        .flat_map(|p| p.0.iter().copied())
        .map(|v| v as f32)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-6_f32);

    let s = SIZE as usize;
    let mut out = Array4::<f32>::zeros((1, 3, s, s));
    for y in 0..s {
        for x in 0..s {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                out[[0, c, y, x]] = (p[c] as f32 / max_val - MEAN[c]) / STD[c];
            }
        }
    }
    out
}
```

### Full postprocess function (verified against rembg source)

```rust
// Source: rembg/sessions/u2net.py predict(), rembg/bg.py naive_cutout() — verified 2026-04-06
pub fn postprocess(raw: ArrayView4<f32>, original: &DynamicImage) -> RgbaImage {
    let pred = raw.slice(s![0, 0, .., ..]);  // [320, 320]

    // rembg: (pred - mi) / (ma - mi) — NO sigmoid, NO threshold
    let ma = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mi = pred.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (ma - mi).max(1e-6_f32);

    let (sh, sw) = (pred.nrows(), pred.ncols());
    let mut mask = GrayImage::new(sw as u32, sh as u32);
    for y in 0..sh {
        for x in 0..sw {
            let val = ((pred[[y, x]] - mi) / range).clamp(0.0, 1.0);
            mask.put_pixel(x as u32, y as u32, image::Luma([(val * 255.0) as u8]));
        }
    }

    // Resize mask back to original dims with Lanczos3
    let (ow, oh) = (original.width(), original.height());
    let mask = image::imageops::resize(&mask, ow, oh, FilterType::Lanczos3);

    // Apply as alpha channel (rembg naive_cutout / putalpha)
    let rgba = original.to_rgba8();
    let mut out = RgbaImage::new(ow, oh);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = mask.get_pixel(x, y)[0];
        out.put_pixel(x, y, Rgba([p[0], p[1], p[2], a]));
    }
    out
}
```

### Image loading (format auto-detection)

```rust
// Source: docs.rs/image/0.25
use image::{ImageReader, DynamicImage};
use std::io::Cursor;

// From file path (extension-based detection)
pub fn load_image_from_path(path: &Path) -> Result<DynamicImage, image::ImageError> {
    image::open(path)
}

// From bytes (magic-byte detection — for drag-and-drop or embedded data)
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, image::ImageError> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()
}
```

### Large image detection

```rust
pub const LARGE_IMAGE_LIMIT: u32 = 8000;
pub const DOWNSCALE_TARGET: u32 = 4096;

pub fn check_large_image(img: &DynamicImage) -> Option<CoreError> {
    let (w, h) = (img.width(), img.height());
    if w > LARGE_IMAGE_LIMIT || h > LARGE_IMAGE_LIMIT {
        Some(CoreError::LargeImage { width: w, height: h, limit: LARGE_IMAGE_LIMIT })
    } else {
        None
    }
}

pub fn downscale_image(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let scale = max_dim as f32 / w.max(h) as f32;
    let (nw, nh) = ((w as f32 * scale) as u32, (h as f32 * scale) as u32);
    img.resize(nw, nh, FilterType::Lanczos3)
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Environment::builder()` + `SessionBuilder::new(&env)` (ort 1.x) | `Session::builder()?.commit_from_memory()` (ort 2.x) | ort 2.0.0-rc.1 | Simpler API; no Arc<Environment> needed |
| `with_model_from_memory()` | `commit_from_memory()` | ort 2.0 | Method rename |
| `ndarray 0.15` compatibility | `ndarray 0.16` required | ort 2.0.0-rc.1 | Do not use 0.15; ort 2.x dropped it |
| Binary threshold (sigmoid + 0.5) | Min-max normalization (no sigmoid) | rembg always | ARCHITECTURE.md was wrong; verified from source |
| Bilinear resize | LANCZOS resize | rembg always | ARCHITECTURE.md was wrong; verified from source |

**Deprecated/outdated:**
- `ort::Environment` builder: Use `Session::builder()` directly in ort 2.0.
- `Value::from_array(session.allocator(), ...)`: Older ort 2.0 RC pattern; use `inputs![name => &array]?` macro with ndarray in rc.7+.

---

## Open Questions

1. **ort 2.0 execution provider struct names**
   - What we know: DeepWiki shows `ExecutionProvider::CUDA(Default::default())`. The ARCHITECTURE.md and STACK.md show `CUDAExecutionProvider::default().build()`. Both appear in ort 2.0 docs depending on the import path.
   - What's unclear: Which form is canonical in rc.12? The EP trait was reorganized in later RCs.
   - Recommendation: Use `CUDAExecutionProvider::default().build()` (from `ort::execution_providers::cuda::CUDAExecutionProvider`) as it matches the ort docs.rs API surface for rc.7+. Verify with a `cargo check` in Wave 0.

2. **ort::init() requirement**
   - What we know: Some examples show `ort::init().commit()?` before any Session creation; others skip it and call `Session::builder()` directly.
   - What's unclear: Whether `ort::init()` is required or optional in rc.12.
   - Recommendation: Call `ort::init().commit()?` once at application startup as a safe practice; it is idempotent if called multiple times.

3. **Silueta model output shape**
   - What we know: U2Net outputs `[1, 1, 320, 320]`. Silueta is a smaller model.
   - What's unclear: Whether silueta's ONNX output has the same shape or requires different postprocessing.
   - Recommendation: In Wave 0, print `session.outputs[0].output_type` for silueta to confirm shape before implementing. Do not assume identical output layout.

4. **Progress stage naming — "Sigmoid" vs "Postprocess"**
   - What we know: CONTEXT.md locks the stage name "Sigmoid" (Decode → Resize → Normalize → Infer → Sigmoid → Threshold → Alpha). rembg does not apply sigmoid.
   - What's unclear: Whether the stage names are user-visible labels (keep "Sigmoid"/"Threshold" as conceptual names) or precise descriptions.
   - Recommendation: Keep the stage names as locked in CONTEXT.md for consistency with the callback API contract. The implementation behind "Sigmoid" actually does min-max normalization, which is fine — the stage name is a label, not an algorithm specification.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `cargo test` |
| Config file | None (workspace Cargo.toml `[profile.test]` if needed) |
| Quick run command | `cargo test -p prunr-core --features prunr-models/dev-models` |
| Full suite command | `cargo test -p prunr-core --features prunr-models/dev-models -- --include-ignored` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORE-01 | `process_image()` returns RGBA PNG with transparent background | integration | `cargo test -p prunr-core --test integration -- process_single_image` | ❌ Wave 0 |
| CORE-02 | Both silueta and u2net models produce valid masks | integration | `cargo test -p prunr-core --test integration -- model_silueta model_u2net` | ❌ Wave 0 |
| CORE-03 | `active_provider()` returns non-empty string; session created without panic | unit | `cargo test -p prunr-core -- engine::tests` | ❌ Wave 0 |
| CORE-04 | Progress callback called for each of 7 stages with monotonic pct | unit | `cargo test -p prunr-core -- pipeline::tests::progress_callback` | ❌ Wave 0 |
| CORE-05 | Pixel match ≥ 95% against rembg reference masks on 3 test images | integration (reference) | `cargo test -p prunr-core --test reference -- --ignored` | ❌ Wave 0 |
| LOAD-03 | PNG, JPEG, WebP, BMP all load without error | unit | `cargo test -p prunr-core -- formats::tests` | ❌ Wave 0 |
| LOAD-04 | Image >8000px returns `CoreError::LargeImage`; `downscale_image()` produces correct dimensions | unit | `cargo test -p prunr-core -- pipeline::tests::large_image` | ❌ Wave 0 |

**Reference test detail (CORE-05):**
The reference test is marked `#[ignore]` so it only runs on `--include-ignored`. It requires:
1. `models/silueta.onnx` and `models/u2net.onnx` on disk (fetched via `cargo xtask fetch-models`)
2. `tests/references/*.png` mask files generated by `scripts/generate_references.py`

The quick run command excludes ignored tests so CI can run unit tests without model files.

### Sampling Rate

- **Per task commit:** `cargo test -p prunr-core --features prunr-models/dev-models`
- **Per wave merge:** `cargo test -p prunr-core --features prunr-models/dev-models -- --include-ignored` (requires models on disk)
- **Phase gate:** Full suite green (including reference tests) before Phase 3 begins

### Wave 0 Gaps

- [ ] `crates/prunr-core/tests/integration.rs` — covers CORE-01, CORE-02, CORE-03, CORE-04, LOAD-03
- [ ] `crates/prunr-core/tests/reference.rs` — covers CORE-05 (marked `#[ignore]`, requires model files)
- [ ] `scripts/generate_references.py` — Python script that runs rembg defaults on test images, saves mask PNGs to `tests/references/`
- [ ] `tests/fixtures/` — 3 test images from rembg's GitHub repo (small/medium/large)
- [ ] `tests/references/` — generated reference masks (committed to repo)
- [ ] `num_cpus = "1"` added to workspace Cargo.toml and prunr-core Cargo.toml

---

## Sources

### Primary (HIGH confidence)

- rembg `sessions/base.py` raw GitHub — confirmed normalize() uses LANCZOS resize and `max(max_pixel, 1e-6)` normalization (not `/255`)
- rembg `sessions/u2net.py` raw GitHub — confirmed predict() applies min-max normalization, no sigmoid, returns grayscale mask
- rembg `bg.py` raw GitHub — confirmed naive_cutout() applies grayscale mask directly as alpha channel, no thresholding
- docs.rs/ort/2.0.0-rc.7 — Session::builder(), commit_from_memory(), inputs! macro, try_extract_tensor patterns
- docs.rs/image/0.25 — FilterType::Lanczos3, image::open(), ImageReader with_guessed_format
- ARCHITECTURE.md (project) — module structure, threading model, EP priority list
- STACK.md (project) — confirmed ort version pinning, ndarray 0.16 compatibility requirement

### Secondary (MEDIUM confidence)

- deepwiki.com/pykeio/ort — Session creation examples, commit_from_memory, inputs! macro, try_extract_tensor
- dasroot.net/posts/2026/03/onnx-runtime-rust-ml-inference-optimization — with_intra_threads(N), GraphOptimizationLevel::Level3 verified as valid ort 2.0 API

### Tertiary (LOW confidence)

- medium.com/@alfred.weirich YOLO webcam article — Shows old ort 1.x Environment API; DO NOT use as API reference. Thread balancing strategy is still valid conceptually.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — ort pinned in workspace, ndarray 0.16 confirmed compatible, image crate well-established
- Architecture: HIGH — module structure matches ARCHITECTURE.md; preprocessing constants and algorithm verified from rembg source
- Pitfalls: HIGH — preprocessing mismatch and postprocessing deviations verified against actual rembg source code
- Two critical deviations from ARCHITECTURE.md confirmed: (1) LANCZOS not bilinear, (2) min-max not sigmoid+threshold

**Research date:** 2026-04-06
**Valid until:** 2026-07-06 (rembg preprocessing constants are stable; ort API may shift when 2.0.0 stable releases)
