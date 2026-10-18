//! Tile-based ONNX upscale dispatch. RGB through the model with
//! overlap-blend; alpha resampled with Lanczos3 in parallel.

mod tiling;
mod alpha;
pub mod postprocess;

pub use postprocess::{sharpened, apply_ai_blend, apply_color_match, refine, Refine};

pub use alpha::upscale_alpha_lanczos3;
pub use tiling::{plan_upscale_tiles, upscale_tiled, TilingConfig, UpscaleTilePlacement};

use std::sync::Arc;

use crate::progress::Progress;

use half::f16;
use image::{RgbImage, RgbaImage};
use ort::{inputs, value::{DynValue, Tensor}};
use ort::session::{NoSelectedOutputs, RunOptions};
use ort::value::TensorElementType;

use crate::CoreError;
use crate::math::round_u8;
use crate::engine::{GraphOptimizationLevel, InferenceEngine, OrtEngine};

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
pub(crate) fn pick_optimization_level(descriptor: &prunr_models::ModelDescriptor) -> GraphOptimizationLevel {
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
            packed[3 * idx]     = round_u8(slice[idx] * 255.0);
            packed[3 * idx + 1] = round_u8(slice[plane + idx] * 255.0);
            packed[3 * idx + 2] = round_u8(slice[2 * plane + idx] * 255.0);
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

/// OpenVINO honours `RunOptions::terminate` only between tiles, so its
/// tile sets the cancel latency: 51 s at 512 on an HD 530. Other EPs
/// abort mid-tile and keep the model's tile. A multiple of 16, so window
/// attention models stay aligned.
const OPENVINO_TILE_CAP: u32 = 256;

fn tile_for_provider(recommended: u32, provider: &str) -> u32 {
    if provider == crate::engine::OPENVINO_NAME {
        recommended.min(OPENVINO_TILE_CAP)
    } else {
        recommended
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
/// Alpha is NOT handled here — the public callers compose it.
fn run_upscale_native(
    input: &RgbaImage,
    engine: &OrtEngine,
    descriptor: &prunr_models::ModelDescriptor,
    native_scale: u32,
    progress: &Progress,
    terminate: Option<&Arc<UpscaleRunOptions>>,
) -> Result<RgbaImage, CoreError> {
    let tile_size = descriptor
        .recommended_tile
        .map(|tile| tile_for_provider(tile, engine.active_provider()))
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
    // loaded the fp16 sibling on a GPU, the
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
        let shape = [1, 3, padded_h as usize, padded_w as usize];

        let out_h = padded_h * ns;
        let out_w = padded_w * ns;
        let plane = (out_w * out_h) as usize;
        let mut packed = vec![0u8; 3 * plane];

        // Only the session's dtype gets a buffer. When `terminate` is
        // Some, the session.run is wired through RunOptions so another
        // thread can abort the running tile — on EPs that honour it (CPU
        // does; OpenVINO only returns at the tile boundary).
        let tensor = if is_fp16 {
            chw_tensor(rgb_tile, shape, |v| f16::from_f32(v * INV_255))?
        } else {
            chw_tensor(rgb_tile, shape, |v| v * INV_255)?
        };
        engine.with_session(|session| {
            let outputs = match terminate {
                Some(opts) => session.run_with_options(inputs![input_name => &tensor], opts.as_ref()),
                None => session.run(inputs![input_name => &tensor]),
            }
            .map_err(classify_run_error)?;
            pack_output(&outputs[0], plane, &mut packed, is_fp16)
        })?;

        RgbImage::from_raw(out_w, out_h, packed).ok_or_else(|| {
            CoreError::Inference("upscale: from_raw failed (packed length mismatch)".into())
        })
    };

    let cfg = TilingConfig { tile_size, tile_multiple, overlap };
    upscale_tiled(input, ns, cfg, run_tile, progress)
}

/// `rgb` as a planar CHW tensor of `shape`, each channel value mapped
/// through `convert`.
fn chw_tensor<T>(rgb: &RgbImage, shape: [usize; 4], convert: impl Fn(f32) -> T) -> Result<DynValue, CoreError>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + Default + 'static,
{
    let pixel_count = shape[2] * shape[3];
    let mut data = vec![T::default(); 3 * pixel_count];
    for (i, p) in rgb.pixels().enumerate() {
        let [r, g, b] = p.0;
        data[i] = convert(r as f32);
        data[pixel_count + i] = convert(g as f32);
        data[2 * pixel_count + i] = convert(b as f32);
    }
    let arr = ndarray::Array4::from_shape_vec(shape, data)
        .map_err(|e| CoreError::Inference(format!("upscale: input shape: {e}")))?;
    Tensor::from_array(arr)
        .map(|t| t.into_dyn())
        .map_err(|e| CoreError::Inference(format!("upscale: input tensor: {e}")))
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
pub fn upscale_rgba_with_engine(
    input: &RgbaImage,
    engine: &OrtEngine,
    model_id: prunr_models::ModelId,
    scale: u32,
    progress: &Progress,
    terminate: Option<&Arc<UpscaleRunOptions>>,
) -> Result<RgbaImage, CoreError> {
    let descriptor = prunr_models::descriptor(model_id)
        .ok_or_else(|| CoreError::Model(format!("{model_id:?} not found in REGISTRY")))?;

    let knobs = upscale_knobs(descriptor)?;
    let native_result = run_upscale_native(input, engine, descriptor, knobs.native_scale, progress, terminate)?;
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
    let mut out = crate::formats::resize_rgba(&native, w, h, crate::formats::ResizeFilter::Lanczos3);
    drop(native);
    let alpha = upscale_alpha_lanczos3(input, w, h);
    for (p, &a) in out.pixels_mut().zip(alpha.as_raw()) {
        p.0[3] = a;
    }
    out
}

/// What an upscale of an `in_w` × `in_h` image by `scale` with `model`
/// holds at its peak, besides the input and the model: the tiler's
/// buffers at the model's own scale, or the resample to `scale` when
/// that is larger.
pub fn peak_bytes(model: prunr_models::ModelId, in_w: u32, in_h: u32, scale: u32) -> u64 {
    let descriptor = prunr_models::descriptor(model);
    let native = descriptor.and_then(|d| d.upscale.as_ref()).map_or(scale, |k| k.native_scale);
    // A model without a tile size refuses to run; count the whole height.
    let tile = descriptor.and_then(|d| d.recommended_tile).unwrap_or(in_h);
    let tiled = tiling::tiled_peak_bytes(in_w, in_h, native, tile);
    if native == scale {
        return tiled;
    }
    let pixels = |s: u32| u64::from(in_w) * u64::from(in_h) * u64::from(s) * u64::from(s);
    tiled.max(pixels(native) * 4 + pixels(scale) * 5)
}

/// Builds its own engine; for callers without a warm engine (CLI, tests).
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
pub fn upscale_rgba(
    input: &RgbaImage,
    model_id: prunr_models::ModelId,
    scale: u32,
    intra_threads: usize,
    progress: &Progress,
) -> Result<RgbaImage, CoreError> {
    let engine = OrtEngine::for_model(model_id, intra_threads, false)?;
    upscale_rgba_with_engine(input, &engine, model_id, scale, progress, None)
}

/// The model to run for `scale` when `model` is picked: Real-ESRGAN's own
/// 2x model for a 2x Real-ESRGAN upscale when it is installed (about 4x
/// faster than 4x and a downscale, as Real-ESRGAN does), else `model`.
pub fn model_for_scale(model: prunr_models::ModelId, scale: crate::OutputScale, x2plus_installed: bool) -> prunr_models::ModelId {
    use prunr_models::ModelId;
    if model == ModelId::RealEsrganX4Plus && scale == crate::OutputScale::X2 && x2plus_installed {
        ModelId::RealEsrganX2Plus
    } else {
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The peak counts the model's own scale: a 2x from a 4x model holds
    /// the 4x image before it is brought down.
    #[test]
    fn the_peak_is_at_the_models_own_scale() {
        use prunr_models::ModelId;
        let mb = |b: u64| b / (1024 * 1024);
        let four = peak_bytes(ModelId::RealEsrganX4Plus, 4096, 4096, 4);
        assert_eq!(mb(four), 512 + 1024 + 256, "a 512 band, the output and its alpha");
        assert_eq!(peak_bytes(ModelId::RealEsrganX4Plus, 4096, 4096, 2), four, "the tiler at 4x outweighs the resample");
        assert_eq!(mb(peak_bytes(ModelId::RealEsrganX2Plus, 4096, 4096, 2)), 128 + 256 + 64);
    }

    #[test]
    fn a_real_esrgan_2x_runs_the_native_2x_model_when_installed() {
        use crate::OutputScale;
        use prunr_models::ModelId;
        assert_eq!(model_for_scale(ModelId::RealEsrganX4Plus, OutputScale::X2, true), ModelId::RealEsrganX2Plus);
        assert_eq!(model_for_scale(ModelId::RealEsrganX4Plus, OutputScale::X2, false), ModelId::RealEsrganX4Plus, "not installed");
        assert_eq!(model_for_scale(ModelId::RealEsrganX4Plus, OutputScale::X4, true), ModelId::RealEsrganX4Plus);
        assert_eq!(model_for_scale(ModelId::Nomos8kSchatL, OutputScale::X2, true), ModelId::Nomos8kSchatL, "other models resample");
    }

    #[test]
    fn openvino_tiles_cap_at_256_for_a_prompt_cancel() {
        assert_eq!(tile_for_provider(512, "OpenVINO"), 256);
        assert_eq!(tile_for_provider(256, "OpenVINO"), 256);
        assert_eq!(tile_for_provider(512, "CPU"), 512);
        assert_eq!(tile_for_provider(512, "CUDA"), 512);
        assert_eq!(OPENVINO_TILE_CAP % 16, 0, "window-attention models need a multiple of 16");
    }

    /// `pick_optimization_level` gates Level2 on `tile_size_multiple.is_some()`.
    /// Without this, HAT-family models panic on the second tile in the same
    /// session due to Level3 shape baking during graph optimisation.
    #[test]
    fn pick_optimization_level_uses_level2_for_hat_family() {
        let nomos = prunr_models::descriptor(prunr_models::ModelId::Nomos8kSchatL)
            .expect("Nomos8kSchatL registered");
        assert!(
            matches!(pick_optimization_level(nomos), GraphOptimizationLevel::Level2),
            "Nomos8kSchatL must use Level2 (has tile_size_multiple)"
        );

        let esrgan = prunr_models::descriptor(prunr_models::ModelId::RealEsrganX4Plus)
            .expect("RealEsrganX4Plus registered");
        assert!(
            matches!(pick_optimization_level(esrgan), GraphOptimizationLevel::Level3),
            "RealEsrganX4Plus must use Level3 (no tile_size_multiple)"
        );
    }

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

    /// `is_fp16` selects the output-extraction dtype in `pack_output`.
    /// A regression that flipped Nomos8kSCHAT-L's flag to `false` (or
    /// RealESRGAN's to `true`) would surface as a runtime ORT
    /// "Unexpected input data type" panic on the first dispatch —
    /// caught here at compile-time-of-the-test instead.
    #[test]
    fn upscale_knobs_fp16_matches_export() {
        let nomos = prunr_models::descriptor(prunr_models::ModelId::Nomos8kSchatL)
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

        let esrgan = prunr_models::descriptor(prunr_models::ModelId::RealEsrganX4Plus)
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

    /// x2plus knobs match the x4plus export convention (same RRDB architecture,
    /// same export script → same tensor names and dtype).
    #[test]
    fn x2plus_knobs_fp16_false_and_input_data() {
        let x2 = prunr_models::descriptor(prunr_models::ModelId::RealEsrganX2Plus)
            .expect("RealEsrganX2Plus registered");
        let knobs = upscale_knobs(x2).expect("RealEsrganX2Plus has upscale knobs");
        assert!(!knobs.is_fp16, "x2plus exports fp32 — must NOT pack f16 tensors");
        assert_eq!(knobs.input_name, "data", "x2plus names the input tensor `data`");
    }
}
