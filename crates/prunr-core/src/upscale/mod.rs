//! Tile-based ONNX upscale dispatch. RGB through the model with
//! overlap-blend; alpha resampled with Lanczos3 in parallel.

mod tiling;
mod alpha;
pub mod postprocess;

pub use postprocess::{apply_sharpen, apply_ai_blend, apply_saturation, apply_color_match};

pub use alpha::upscale_alpha_lanczos3;
pub use tiling::{plan_upscale_tiles, upscale_tiled, TilingConfig, UpscaleTilePlacement};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use half::f16;
use image::{RgbImage, RgbaImage};
use ort::{inputs, value::Tensor};
use ort::session::{NoSelectedOutputs, RunOptions};
use ort::value::TensorElementType;

use crate::CoreError;
use crate::engine::{GraphOptimizationLevel, OrtEngine};
use crate::types::ModelKind;

/// Shared `RunOptions` handle used by callers that want to terminate a
/// running upscale session mid-tile. The owning thread holds the `Arc`
/// and passes it through the dispatch path; another thread (typically
/// the GUI's Cancel handler) calls `terminate()` on the same `RunOptions`
/// to abort the C++ `Session::run` within ~50ms instead of waiting for
/// the next tile-boundary cancel-flag check.
pub type UpscaleRunOptions = RunOptions<NoSelectedOutputs>;

/// Select the ORT graph optimization level for a given model descriptor.
///
/// Window-attention transformers (HAT, Swin) use Level2 because Level3
/// bakes the first tile's input shape into the graph during session init.
/// A subsequent tile with different dimensions then hits an irrecoverable
/// shape-mismatch (observed against 4xNomos8kSCHAT-L: Level3 panics with
/// `Attempting to get index by a name which does not exist:
/// InsertedPrecisionFreeCast_…`; Level2 succeeds across all tile sizes).
/// Exposed `pub` so the Processor's warm-engine cache can derive the
/// same level without duplicating the branch.
pub fn pick_optimization_level(descriptor: &prunr_models::ModelDescriptor) -> GraphOptimizationLevel {
    if descriptor.tile_size_multiple.is_some() {
        GraphOptimizationLevel::Level2
    } else {
        GraphOptimizationLevel::Level3
    }
}

/// Per-model upscale knobs from the REGISTRY. The descriptor's
/// `upscale: Option<UpscaleModelKnobs>` is `Some` only for
/// upscale-category entries; this helper unwraps with an
/// `Inference` error so a non-upscale `ModelId` reaching here fails
/// loud at the dispatch boundary rather than at the ORT session.
fn upscale_knobs(descriptor: &prunr_models::ModelDescriptor)
    -> Result<&prunr_models::UpscaleModelKnobs, CoreError>
{
    descriptor.upscale.as_ref().ok_or_else(|| {
        CoreError::Inference(format!(
            "{:?} is not an upscale model (no UpscaleModelKnobs in REGISTRY)",
            descriptor.id
        ))
    })
}

/// CHW f32-or-f16 → HWC u8 inline pack, run inside the session lock so
/// the ORT-borrowed slice doesn't need an `into_owned()` copy. `is_fp16`
/// selects which dtype to extract; both paths normalise to f32 before
/// the clamp + cast.
fn pack_output(
    value: &ort::value::DynValue,
    plane: usize,
    packed: &mut [u8],
    is_fp16: bool,
) -> Result<(), CoreError> {
    fn write_u8(packed: &mut [u8], plane: usize, slice: &[f32]) -> Result<(), CoreError> {
        if slice.len() < 3 * plane {
            return Err(CoreError::Inference(format!(
                "upscale: output tensor too small: {} < {}",
                slice.len(),
                3 * plane
            )));
        }
        for idx in 0..plane {
            packed[3 * idx]     = (slice[idx] * 255.0).clamp(0.0, 255.0) as u8;
            packed[3 * idx + 1] = (slice[plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
            packed[3 * idx + 2] = (slice[2 * plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
        }
        Ok(())
    }

    if is_fp16 {
        let arr = value
            .try_extract_array::<f16>()
            .map_err(|e| CoreError::Inference(format!("upscale: output extract (f16): {e}")))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("upscale: output reshape (f16): {e}")))?;
        // f16 → f32 conversion is required before clamp/cast; the f16
        // payload from ORT is borrowed, so the owned f32 buffer here is
        // unavoidable.
        let owned: ndarray::Array4<f32> = arr.mapv(|x| x.to_f32());
        let slice = owned.as_slice().ok_or_else(|| {
            CoreError::Inference("upscale: output tensor not contiguous (f16)".into())
        })?;
        write_u8(packed, plane, slice)
    } else {
        let arr = value
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("upscale: output extract: {e}")))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("upscale: output reshape: {e}")))?;
        let slice = arr.as_slice().ok_or_else(|| {
            CoreError::Inference("upscale: output tensor not contiguous".into())
        })?;
        write_u8(packed, plane, slice)
    }
}

/// Run a model at its native output scale (no downscale). Internal only.
///
/// `native_scale` is the number of times larger the model output is than
/// the input (e.g. 4 for RealESRGAN-x4plus, 2 for RealESRGAN-x2plus).
/// The value is read from the REGISTRY and passed in by callers rather than
/// inferred here, so neither this function nor `upscale_rgba` needs to know
/// which model it is running.
///
/// Alpha is NOT handled here — it is composed by the public callers
/// (`upscale_rgba` for single-pass, `upscale_two_pass` for the final pass).
fn run_upscale_native<F>(
    input: &RgbaImage,
    engine: &OrtEngine,
    descriptor: &prunr_models::ModelDescriptor,
    native_scale: u32,
    on_tile_done: F,
    cancel: Option<Arc<AtomicBool>>,
    terminate: Option<&Arc<UpscaleRunOptions>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(u32, u32),
{
    let tile_size = descriptor
        .recommended_tile
        .ok_or_else(|| CoreError::Inference(format!(
            "model {:?} has no recommended_tile -- upscale cannot dispatch",
            descriptor.id
        )))?;

    let tile_multiple = descriptor.tile_size_multiple;
    let knobs = upscale_knobs(descriptor)?;
    let overlap = knobs.tile_overlap();

    let input_name = knobs.input_name;
    // Probe the actual loaded session's input dtype. The REGISTRY's
    // `knobs.is_fp16` flag describes the MAIN ONNX URL; when the engine
    // promoted to an fp16 sibling via `optimized_variant_bytes`, the
    // graph expects fp16 inputs even though the REGISTRY knob still
    // reads false. Without this probe the dispatch would feed fp32
    // tensors into an fp16 graph and ORT errors with "Unexpected input
    // data type" (observed 2026-05-16 against Siax-CX / Superscale fp16
    // variants on OpenVINO).
    let is_fp16 = engine.with_session(|session| {
        let dtype = session
            .inputs()
            .first()
            .map(|outlet| outlet.dtype())
            .and_then(|vt| vt.tensor_type());
        Ok(matches!(dtype, Some(TensorElementType::Float16)))
    })?;
    let ns = native_scale;

    let run_tile = |rgb_tile: &RgbImage, padded_w: u32, padded_h: u32| -> Result<RgbImage, CoreError> {
        const INV_255: f32 = 1.0 / 255.0;
        let pixel_count = (padded_w * padded_h) as usize;
        let shape = [1, 3, padded_h as usize, padded_w as usize];

        let out_h = padded_h * ns;
        let out_w = padded_w * ns;
        let plane = (out_w * out_h) as usize;
        let mut packed = vec![0u8; 3 * plane];

        // The two branches keep the f16/f32 buffers + tensor scoped to
        // the relevant arm so the unused dtype's buffer doesn't sit
        // allocated during inference. When `terminate` is Some, the
        // session.run is wired through RunOptions so another thread can
        // abort the running tile — on EPs that honour it (CPU does;
        // OpenVINO only returns at the tile boundary).
        if is_fp16 {
            let mut input_data = vec![f16::ZERO; 3 * pixel_count];
            for (i, p) in rgb_tile.pixels().enumerate() {
                let [r, g, b] = p.0;
                input_data[i] = f16::from_f32(r as f32 * INV_255);
                input_data[pixel_count + i] = f16::from_f32(g as f32 * INV_255);
                input_data[2 * pixel_count + i] = f16::from_f32(b as f32 * INV_255);
            }
            let arr = ndarray::Array4::from_shape_vec(shape, input_data)
                .map_err(|e| CoreError::Inference(format!("upscale: input shape: {e}")))?;
            let tensor = Tensor::from_array(arr)
                .map_err(|e| CoreError::Inference(format!("upscale: input tensor: {e}")))?;
            engine.with_session(|session| {
                let outputs = match terminate {
                    Some(opts) => session.run_with_options(inputs![input_name => &tensor], opts.as_ref()),
                    None => session.run(inputs![input_name => &tensor]),
                }
                .map_err(classify_run_error)?;
                pack_output(&outputs[0], plane, &mut packed, is_fp16)
            })?;
        } else {
            let mut input_data = vec![0.0_f32; 3 * pixel_count];
            for (i, p) in rgb_tile.pixels().enumerate() {
                let [r, g, b] = p.0;
                input_data[i] = r as f32 * INV_255;
                input_data[pixel_count + i] = g as f32 * INV_255;
                input_data[2 * pixel_count + i] = b as f32 * INV_255;
            }
            let arr = ndarray::Array4::from_shape_vec(shape, input_data)
                .map_err(|e| CoreError::Inference(format!("upscale: input shape: {e}")))?;
            let tensor = Tensor::from_array(arr)
                .map_err(|e| CoreError::Inference(format!("upscale: input tensor: {e}")))?;
            engine.with_session(|session| {
                let outputs = match terminate {
                    Some(opts) => session.run_with_options(inputs![input_name => &tensor], opts.as_ref()),
                    None => session.run(inputs![input_name => &tensor]),
                }
                .map_err(classify_run_error)?;
                pack_output(&outputs[0], plane, &mut packed, is_fp16)
            })?;
        }

        RgbImage::from_raw(out_w, out_h, packed).ok_or_else(|| {
            CoreError::Inference("upscale: from_raw failed (packed length mismatch)".into())
        })
    };

    let cfg = TilingConfig { tile_size, tile_multiple, overlap };
    upscale_tiled(input, ns, cfg, run_tile, on_tile_done, cancel)
}

/// Map an ORT run-time error to a `CoreError`. ORT reports a session
/// aborted via `RunOptions::terminate()` as a generic error containing
/// "terminate flag" in its message; surface that as `Cancelled` so the
/// dispatcher's existing cancellation handling (toast, no error log)
/// fires instead of treating it as a real inference failure.
fn classify_run_error(err: ort::Error) -> CoreError {
    let msg = err.to_string();
    if msg.contains("terminate flag") || msg.contains("Exiting due to terminate") {
        CoreError::Cancelled
    } else {
        CoreError::Inference(format!("upscale: inference failed: {msg}"))
    }
}

/// Engine-parameterized upscale. Caller constructs and owns the
/// `OrtEngine`; used by the Processor's warm-engine cache so the
/// 1-3 s Level3 graph-optimization stall is paid only on the first
/// dispatch.
///
/// Behavior:
///   - scale = 4: runs the model at its native 4× scale and returns.
///   - scale = 2: for 4× models, runs at 4× then Lanczos3 downscales to 2×.
///     For native-2× models (x2plus), runs at native scale directly.
///   - Alpha is upscaled independently via Lanczos3.
///
/// Peak working-set RAM: same as `upscale_rgba` (see that doc).
pub fn upscale_rgba_with_engine<F>(
    input: &RgbaImage,
    engine: &OrtEngine,
    model_id: prunr_models::ModelId,
    scale: u32,
    on_tile_done: F,
    cancel: Option<Arc<AtomicBool>>,
    terminate: Option<&Arc<UpscaleRunOptions>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(u32, u32),
{
    let descriptor = prunr_models::REGISTRY
        .iter()
        .find(|d| d.id == model_id)
        .ok_or_else(|| CoreError::Model(format!("{model_id:?} not found in REGISTRY")))?;

    let knobs = upscale_knobs(descriptor)?;
    let native_result = run_upscale_native(input, engine, descriptor, knobs.native_scale, on_tile_done, cancel, terminate)?;
    Ok(fit_to_scale(native_result, input, scale))
}

/// Bring the model's native output to `scale ×` the input: returned as
/// is when the sizes already agree, otherwise Lanczos3-resampled (down
/// for 2× or 3× from a 4× model, up when the factor exceeds the model's
/// native scale). Alpha is resampled from the input directly.
fn fit_to_scale(native: RgbaImage, input: &RgbaImage, scale: u32) -> RgbaImage {
    let (w, h) = (input.width() * scale, input.height() * scale);
    if native.dimensions() == (w, h) {
        return native;
    }
    let rgb = crate::formats::resize_rgb_lanczos3(&image::DynamicImage::ImageRgba8(native), w, h);
    let alpha = upscale_alpha_lanczos3(input, w, h);
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in out.enumerate_pixels_mut() {
        let rgb = rgb.get_pixel(x, y).0;
        *p = image::Rgba([rgb[0], rgb[1], rgb[2], alpha.get_pixel(x, y).0[0]]);
    }
    out
}

/// Convenience wrapper: constructs an engine internally then delegates to
/// `upscale_rgba_with_engine`. Preserves the existing public API for
/// callers that don't need the warm-engine cache (CLI, tests).
///
/// Peak working-set RAM (additive to the `upscale_tiled` accumulators
/// documented in `tiling.rs`):
///   - Tile input scratch: `3 × padded_w × padded_h × 4 bytes` (f32 CHW).
///     ~3 MB at 512-tile.
///   - Tile output scratch: `3 × out_w × out_h bytes` (u8 HWC).
///     ~12 MB at 512→2048 4× tile.
///   - scale=2 with a 4× model only: the full 4× RgbaImage (~16× input
///     bytes) lives until the Lanczos3 downscale completes.
///   - ONNX session: model-dependent (see `ModelDescriptor.working_set_mb`).
pub fn upscale_rgba<F>(
    input: &RgbaImage,
    model_id: prunr_models::ModelId,
    scale: u32,
    intra_threads: usize,
    on_tile_done: F,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(u32, u32),
{
    let descriptor = prunr_models::REGISTRY
        .iter()
        .find(|d| d.id == model_id)
        .ok_or_else(|| CoreError::Model(format!("{model_id:?} not found in REGISTRY")))?;

    let level = pick_optimization_level(descriptor);

    let model_kind = ModelKind::try_from(model_id).map_err(|id| {
        CoreError::Inference(format!(
            "{id:?} has no ModelKind mapping — upscale dispatch requires a seg/upscale variant"
        ))
    })?;
    let engine = OrtEngine::new_with_optimization_level(model_kind, intra_threads, level)?;
    upscale_rgba_with_engine(input, &engine, model_id, scale, on_tile_done, cancel, None)
}

/// Engine-parameterized two-pass 4× upscale via `RealEsrganX2Plus`
/// chained against itself. Caller constructs and owns the `OrtEngine`;
/// used by the Processor's warm-engine cache so the graph-optimization
/// stall is paid only on the first dispatch.
///
/// Pass 1: input → 2× via x2plus.
/// Pass 2: 2× output → 4× via x2plus.
///
/// The engine must have been constructed for `ModelKind::RealEsrganX2Plus`.
///
/// Peak RAM: same as `upscale_two_pass` (see that doc).
pub fn upscale_two_pass_with_engine<F>(
    input: &RgbaImage,
    engine: &OrtEngine,
    on_tile_done: F,
    cancel: Option<Arc<AtomicBool>>,
    terminate: Option<&Arc<UpscaleRunOptions>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(u32, u32) + Clone,
{
    use std::sync::atomic::Ordering;

    // Honor cancel before allocating the first pass.
    if let Some(c) = cancel.as_ref() {
        if c.load(Ordering::Acquire) {
            return Err(CoreError::Cancelled);
        }
    }

    let model_id = prunr_models::ModelId::RealEsrganX2Plus;
    let descriptor = prunr_models::REGISTRY
        .iter()
        .find(|d| d.id == model_id)
        .ok_or_else(|| CoreError::Model("RealEsrganX2Plus not found in REGISTRY".into()))?;

    let knobs = upscale_knobs(descriptor)?;
    // Pass 1: source → 2×
    let intermediate = run_upscale_native(
        input,
        engine,
        descriptor,
        knobs.native_scale,
        on_tile_done.clone(),
        cancel.clone(),
        terminate,
    )?;

    // Honor cancel between passes.
    if let Some(c) = cancel.as_ref() {
        if c.load(Ordering::Acquire) {
            return Err(CoreError::Cancelled);
        }
    }

    // Pass 2: intermediate → 4× (intermediate is moved, not cloned).
    let final_rgb = run_upscale_native(
        &intermediate,
        engine,
        descriptor,
        knobs.native_scale,
        on_tile_done,
        cancel,
        terminate,
    )?;
    drop(intermediate);

    // Compose alpha into final_rgb in place: take RGB from pass-2 output,
    // overwrite its alpha with the Lanczos3-upscaled alpha from the original
    // input. Mutating final_rgb saves a parallel 4× RgbaImage allocation.
    let out_w = input.width() * 4;
    let out_h = input.height() * 4;
    let alpha_4x = upscale_alpha_lanczos3(input, out_w, out_h);
    let mut out = final_rgb;
    for (x, y, p) in out.enumerate_pixels_mut() {
        p.0[3] = alpha_4x.get_pixel(x, y).0[0];
    }
    Ok(out)
}

/// Convenience wrapper: constructs an engine internally then delegates to
/// `upscale_two_pass_with_engine`. Preserves the existing public API for
/// callers that don't need the warm-engine cache (CLI, tests).
///
/// Pass 1: input → 2× via x2plus.
/// Pass 2: 2× output → 4× via x2plus.
///
/// The intermediate 2× buffer is moved into the pass-2 input, not cloned —
/// large-image RAM accounting depends on this. Pass-1's `RgbaImage` is freed
/// as soon as pass-2 begins consuming it.
///
/// Peak RAM during pass-2 (per PRECONDITIONS.md two-pass smoke test, 963 MB
/// peak on 512→1024 pass-2):
///   input_bytes × 4   // pass-1 output (2× linear = 4× pixels)
///   + input_bytes × 16 // pass-2 output (4× linear = 16× pixels)
///   + ~64 MB           // x2plus ORT session (same session reused for both passes)
///
/// Compose stage allocates a 4× GrayImage for alpha (input × 4) and mutates
/// pass-2's RgbaImage in place — no parallel 4× RgbaImage allocation.
///
/// Progress callback fires from BOTH passes. The `(done, total)` pair
/// reflects per-pass tile counts — total is per-pass, not combined.
///
/// Callable only via the `OutputScale::X4TwoPass` recipe variant.
/// Nomos8k cannot use this path — its scale is fixed at 4×; the chip UI
/// dims X4TwoPass for non-RealEsrgan models via `x4twopass_available`.
pub fn upscale_two_pass<F>(
    input: &RgbaImage,
    intra_threads: usize,
    on_tile_done: F,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(u32, u32) + Clone,
{
    use std::sync::atomic::Ordering as _Ordering;

    // Honor cancel before constructing the engine.
    if let Some(c) = cancel.as_ref() {
        if c.load(_Ordering::Acquire) {
            return Err(CoreError::Cancelled);
        }
    }

    let model_id = prunr_models::ModelId::RealEsrganX2Plus;
    let descriptor = prunr_models::REGISTRY
        .iter()
        .find(|d| d.id == model_id)
        .ok_or_else(|| CoreError::Model("RealEsrganX2Plus not found in REGISTRY".into()))?;

    let level = pick_optimization_level(descriptor);
    let engine = OrtEngine::new_with_optimization_level(
        ModelKind::RealEsrganX2Plus,
        intra_threads,
        level,
    )?;
    upscale_two_pass_with_engine(input, &engine, on_tile_done, cancel, None)
}

/// Returns `true` when `model_id` supports `OutputScale::X4TwoPass`
/// (chains `RealEsrganX2Plus` twice). Currently only `RealEsrganX4Plus`.
pub fn x4twopass_available(model_id: prunr_models::ModelId) -> bool {
    model_id == prunr_models::ModelId::RealEsrganX4Plus
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pick_optimization_level` gates Level2 on `tile_size_multiple.is_some()`.
    /// Without this, HAT-family models panic on the second tile in the same
    /// session due to Level3 shape baking during graph optimisation.
    #[test]
    fn pick_optimization_level_uses_level2_for_hat_family() {
        let nomos = prunr_models::REGISTRY
            .iter()
            .find(|d| d.id == prunr_models::ModelId::Nomos8kSchatL)
            .expect("Nomos8kSchatL registered");
        assert!(
            matches!(pick_optimization_level(nomos), GraphOptimizationLevel::Level2),
            "Nomos8kSchatL must use Level2 (has tile_size_multiple)"
        );

        let esrgan = prunr_models::REGISTRY
            .iter()
            .find(|d| d.id == prunr_models::ModelId::RealEsrganX4Plus)
            .expect("RealEsrganX4Plus registered");
        assert!(
            matches!(pick_optimization_level(esrgan), GraphOptimizationLevel::Level3),
            "RealEsrganX4Plus must use Level3 (no tile_size_multiple)"
        );
    }

    /// `is_fp16` selects the output-extraction dtype in `pack_output`.
    /// A regression that flipped Nomos8kSCHAT-L's flag to `false` (or
    /// RealESRGAN's to `true`) would surface as a runtime ORT
    /// "Unexpected input data type" panic on the first dispatch —
    /// caught here at compile-time-of-the-test instead.
    #[test]
    fn fit_to_scale_resamples_to_every_offered_factor() {
        let input = RgbaImage::from_pixel(4, 6, image::Rgba([10, 20, 30, 200]));
        let native4 = RgbaImage::from_pixel(16, 24, image::Rgba([40, 50, 60, 255]));
        assert_eq!(fit_to_scale(native4.clone(), &input, 4).dimensions(), (16, 24));
        assert_eq!(fit_to_scale(native4.clone(), &input, 3).dimensions(), (12, 18));
        let two = fit_to_scale(native4, &input, 2);
        assert_eq!(two.dimensions(), (8, 12));
        assert_eq!(two.get_pixel(3, 3).0, [40, 50, 60, 200], "RGB from the model, alpha from the input");
        // A native-2× model asked for 4× resamples up instead of returning the 2× plane.
        let native2 = RgbaImage::from_pixel(8, 12, image::Rgba([1, 2, 3, 255]));
        assert_eq!(fit_to_scale(native2, &input, 4).dimensions(), (16, 24));
    }

    #[test]
    fn upscale_knobs_fp16_matches_export() {
        let nomos = prunr_models::REGISTRY
            .iter()
            .find(|d| d.id == prunr_models::ModelId::Nomos8kSchatL)
            .expect("Nomos8kSchatL registered");
        let nomos_knobs = upscale_knobs(nomos).expect("Nomos8kSchatL has upscale knobs");
        assert!(
            nomos_knobs.is_fp16,
            "Nomos8kSchatL exports fp16 — dispatch must pack f16 tensors"
        );
        assert_eq!(
            nomos_knobs.input_name, "input",
            "Nomos8kSchatL (Phhofm HAT-L) names the input tensor `input`"
        );

        let esrgan = prunr_models::REGISTRY
            .iter()
            .find(|d| d.id == prunr_models::ModelId::RealEsrganX4Plus)
            .expect("RealEsrganX4Plus registered");
        let esrgan_knobs = upscale_knobs(esrgan).expect("RealEsrganX4Plus has upscale knobs");
        assert!(
            !esrgan_knobs.is_fp16,
            "RealEsrganX4Plus exports f32 — dispatch must NOT pack f16"
        );
        assert_eq!(
            esrgan_knobs.input_name, "data",
            "RealEsrganX4Plus names the input tensor `data`"
        );
    }

    // ── Two-pass scheduler tests ──────────────────────────────────────────────

    /// `upscale_two_pass` takes no model_id parameter — RealEsrganX2Plus is
    /// hardcoded. This test confirms the signature does not accept a model_id
    /// argument by verifying the function compiles with only (input, threads,
    /// callback, cancel) — a compile-error proof in the signature itself.
    #[test]
    #[allow(clippy::type_complexity)] // the explicit fn-pointer type IS the test — it pins the signature
    fn two_pass_has_no_model_id_parameter() {
        // If this test compiles, the signature is correct: no model_id arg.
        let _fn: fn(
            &RgbaImage,
            usize,
            fn(u32, u32),
            Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        ) -> Result<RgbaImage, CoreError> = |input, threads, cb, cancel| {
            upscale_two_pass(input, threads, cb, cancel)
        };
    }

    /// Cancel before dispatch must return Cancelled without loading the model.
    #[test]
    fn two_pass_cancel_during_pass1_returns_cancelled() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel = Arc::new(AtomicBool::new(true));
        let input = RgbaImage::from_pixel(8, 8, image::Rgba([128, 64, 32, 255]));
        let result = upscale_two_pass(&input, 1, |_, _| {}, Some(cancel.clone()));

        match result {
            Err(CoreError::Cancelled) => {}
            other => panic!("expected Err(Cancelled), got {:?}", other),
        }
        let _ = cancel.load(Ordering::Acquire);
    }

    /// `x4twopass_available` gates the X4TwoPass path: true only for
    /// RealEsrganX4Plus (the user-facing model that signals ESRGAN architecture).
    #[test]
    fn x4twopass_available_predicate() {
        assert!(
            x4twopass_available(prunr_models::ModelId::RealEsrganX4Plus),
            "x4twopass must be available when user has RealEsrganX4Plus selected"
        );
        assert!(
            !x4twopass_available(prunr_models::ModelId::Nomos8kSchatL),
            "x4twopass must not be available for Nomos8k — scale is fixed at 4×"
        );
        assert!(
            !x4twopass_available(prunr_models::ModelId::RealEsrganX2Plus),
            "RealEsrganX2Plus is an internal model — never the user's selected model"
        );
        assert!(
            !x4twopass_available(prunr_models::ModelId::Silueta),
            "Silueta is a segmentation model — x4twopass irrelevant"
        );
    }

    /// x2plus knobs match the x4plus export convention (same RRDB architecture,
    /// same export script → same tensor names and dtype).
    #[test]
    fn x2plus_knobs_fp16_false_and_input_data() {
        let x2 = prunr_models::REGISTRY
            .iter()
            .find(|d| d.id == prunr_models::ModelId::RealEsrganX2Plus)
            .expect("RealEsrganX2Plus registered");
        let knobs = upscale_knobs(x2).expect("RealEsrganX2Plus has upscale knobs");
        assert!(!knobs.is_fp16, "x2plus exports fp32 — must NOT pack f16 tensors");
        assert_eq!(knobs.input_name, "data", "x2plus names the input tensor `data`");
    }
}
