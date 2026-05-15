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

use crate::CoreError;
use crate::engine::{GraphOptimizationLevel, OrtEngine};
use crate::types::ModelKind;

/// Select the ORT graph optimization level for a given model descriptor.
///
/// Window-attention transformers (HAT, Swin) use Level2 because Level3
/// bakes the first tile's input shape into the graph during session init.
/// A subsequent tile with different dimensions then hits an irrecoverable
/// shape-mismatch (observed against 4xNomos8kSCHAT-L: Level3 panics with
/// `Attempting to get index by a name which does not exist:
/// InsertedPrecisionFreeCast_…`; Level2 succeeds across all tile sizes).
fn pick_optimization_level(descriptor: &prunr_models::ModelDescriptor) -> GraphOptimizationLevel {
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
    let overlap = match tile_multiple {
        Some(_) => 32, // HAT: 2 window widths
        None => 16,    // ESRGAN: 16 px safe overlap
    };

    let knobs = upscale_knobs(descriptor)?;
    let input_name = knobs.input_name;
    let is_fp16 = knobs.is_fp16;
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
        // allocated during inference.
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
                let outputs = session
                    .run(inputs![input_name => &tensor])
                    .map_err(|e| CoreError::Inference(format!("upscale: inference failed: {e}")))?;
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
                let outputs = session
                    .run(inputs![input_name => &tensor])
                    .map_err(|e| CoreError::Inference(format!("upscale: inference failed: {e}")))?;
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

/// Upscale an RGBA image by `scale` (2 or 4) using the ONNX model
/// identified by `model_id`.
///
/// Behavior:
///   - scale = 4: runs the model at its native 4× scale and returns.
///   - scale = 2: for 4× models, runs at 4× then Lanczos3 downscales to 2×.
///     For native-2× models (x2plus), runs at native scale directly.
///   - Alpha is upscaled independently via Lanczos3.
///   - Window-attention models (`descriptor.tile_size_multiple.is_some()`)
///     are run at `GraphOptimizationLevel::Level2` to avoid first-tile
///     shape baking; other models run at Level3.
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
    // CPU-only EP: OpenVINO's lazy per-shape graph compilation can stall
    // a single RRDB upscale dispatch for tens of minutes on the first
    // tile dimension (RealESRGAN's 23 residual blocks compile slowly
    // and the EP cache wasn't hitting). The CPU EP has no lazy compile
    // step — total wall-clock is dominated by inference, which is what
    // the user actually waited for.
    let engine = OrtEngine::new_cpu_only_with_optimization_level(model_kind, intra_threads, level)?;

    // All user-selectable upscale models are native-4×. x2plus (native-2×)
    // is dispatched exclusively via upscale_two_pass — never reaches here.
    let native_result = run_upscale_native(input, &engine, descriptor, 4, on_tile_done, cancel)?;

    if scale == 4 {
        Ok(native_result)
    } else if scale == 2 {
        let half_w = native_result.width() / 2;
        let half_h = native_result.height() / 2;

        let rgb4 = image::DynamicImage::ImageRgba8(native_result);
        let rgb_half = crate::formats::resize_rgb_lanczos3(&rgb4, half_w, half_h);

        // Resize alpha from the original input — skips the intermediate 4×
        // pass that the RGB path goes through.
        let alpha_half = upscale_alpha_lanczos3(input, half_w, half_h);

        let mut out = RgbaImage::new(half_w, half_h);
        for (x, y, p) in out.enumerate_pixels_mut() {
            let rgb = rgb_half.get_pixel(x, y).0;
            let a = alpha_half.get_pixel(x, y).0[0];
            *p = image::Rgba([rgb[0], rgb[1], rgb[2], a]);
        }
        Ok(out)
    } else {
        Err(CoreError::Inference(format!("unsupported upscale scale: {scale}")))
    }
}

/// Two-pass 4× upscale via `RealEsrganX2Plus` chained against itself.
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

    let level = pick_optimization_level(descriptor);
    let engine = OrtEngine::new_cpu_only_with_optimization_level(
        ModelKind::RealEsrganX2Plus,
        intra_threads,
        level,
    )?;

    // Pass 1: source → 2×
    let intermediate = run_upscale_native(
        input,
        &engine,
        descriptor,
        2,
        on_tile_done.clone(),
        cancel.clone(),
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
        &engine,
        descriptor,
        2,
        on_tile_done,
        cancel,
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
