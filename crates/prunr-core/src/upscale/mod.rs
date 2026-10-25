//! ONNX upscale dispatch: RGB through the model in overlap-blended tiles,
//! or part by part for a model that normalizes over its whole input;
//! alpha resampled with Lanczos3 apart.

mod tiling;
mod alpha;
pub(crate) mod blocks;
pub mod postprocess;

pub use postprocess::{sharpened, apply_ai_blend, apply_color_match, refine, Reference, Refine};

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
use crate::engine::{GraphOptimizationLevel, OrtEngine};

/// Shared `RunOptions`: another thread calls `terminate()` on it to abort
/// the running tile within ~50 ms instead of at the next tile boundary.
pub type UpscaleRunOptions = RunOptions<NoSelectedOutputs>;

/// Window-attention transformers (HAT, Swin) use Level2: Level3 bakes the
/// first tile's input shape into the graph, and a later tile of another
/// size fails (4xNomos8kSCHAT-L: `InsertedPrecisionFreeCast_…` not found).
pub(crate) fn pick_optimization_level(descriptor: &prunr_models::ModelDescriptor) -> GraphOptimizationLevel {
    if descriptor.tile_size_multiple.is_some() {
        GraphOptimizationLevel::Level2
    } else {
        GraphOptimizationLevel::Level3
    }
}

/// Errors on a non-upscale model, so it fails at the dispatch boundary
/// rather than in the ORT session.
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

/// CHW f32-or-f16 → HWC u8, run inside the session lock so the
/// ORT-borrowed slice needs no `into_owned()` copy.
fn pack_output(
    value: &ort::value::DynValue,
    plane: usize,
    packed: &mut [u8],
    is_fp16: bool,
) -> Result<(), CoreError> {
    let extract_err = |e: ort::Error| CoreError::Inference(format!("upscale: output extract: {e}"));
    let widened: Vec<f32>;
    let slice: &[f32] = if is_fp16 {
        widened = value.try_extract_tensor::<f16>().map_err(extract_err)?.1.iter().map(|x| x.to_f32()).collect();
        &widened
    } else {
        value.try_extract_tensor::<f32>().map_err(extract_err)?.1
    };
    if slice.len() < 3 * plane {
        return Err(CoreError::Inference(format!("upscale: output tensor too small: {} < {}", slice.len(), 3 * plane)));
    }
    for idx in 0..plane {
        packed[3 * idx]     = round_u8(slice[idx] * 255.0);
        packed[3 * idx + 1] = round_u8(slice[plane + idx] * 255.0);
        packed[3 * idx + 2] = round_u8(slice[2 * plane + idx] * 255.0);
    }
    Ok(())
}

/// OpenVINO honours `RunOptions::terminate` only between tiles, so its
/// tile sets the cancel latency: 51 s at 512 on an HD 530. Other EPs
/// abort mid-tile and keep the model's tile. A multiple of 16, so window
/// attention models stay aligned.
const OPENVINO_TILE_CAP: u32 = 256;

/// On unified memory CoreML's GPU buffers are the RAM: Real-ESRGAN x4 at
/// a 512 px tile peaked at 11.6 GB on a 16 GB MacBook Air (4.3 GB on its
/// CPU). A multiple of 16, as OpenVINO's.
const COREML_TILE_CAP: u32 = 256;

pub fn tile_for_provider(recommended: u32, provider: &str) -> u32 {
    if provider == crate::engine::OPENVINO_NAME {
        recommended.min(OPENVINO_TILE_CAP)
    } else if provider == crate::engine::COREML_NAME {
        recommended.min(COREML_TILE_CAP)
    } else {
        recommended
    }
}

/// Upscale on a caller-owned engine, so a warm engine skips the 1-3 s
/// Level3 graph-optimization stall. Runs at the model's native scale,
/// then Lanczos3-resamples to `scale` when they differ; alpha is
/// resampled separately. Peak RAM as `upscale_rgba`.
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
    if let Some(net) = engine.parts() {
        let running = blocks::Running { net, terminate: terminate.map(Arc::as_ref) };
        let room = parted_room(crate::sysmem::available_ram_bytes(), working_set_bytes(model_id), net.resident_bytes());
        let pre = blocks::pre_norm_within(net.layout(), input.width(), input.height(), knobs.native_scale, room);
        tracing::info!(?model_id, ?pre, room, "upscale in parts");
        let rgb = blocks::upscale_by_parts(&running, input, knobs.native_scale, pre, progress)?;
        let mut native: RgbaImage = image::buffer::ConvertBuffer::convert(&rgb);
        drop(rgb);
        if knobs.native_scale == scale {
            alpha::take_input_alpha(&mut native, input);
            return Ok(native);
        }
        return Ok(fit_to_scale(native, input, scale));
    }
    let tile_size = descriptor
        .recommended_tile
        .map(|tile| tile_for_provider(tile, engine.active_provider()))
        .ok_or_else(|| CoreError::Inference(format!(
            "model {:?} has no recommended_tile -- upscale cannot dispatch",
            descriptor.id
        )))?;

    let tile_multiple = descriptor.tile_size_multiple;
    let overlap = knobs.tile_overlap();

    let input_name = knobs.input_name;
    // Probe the loaded session, not `knobs.is_fp16`: the engine may have
    // loaded the fp16 sibling on a GPU, and fp32 tensors into an fp16
    // graph fail with "Unexpected input data type".
    let is_fp16 = engine.with_session(|session| {
        let dtype = session
            .inputs()
            .first()
            .map(|outlet| outlet.dtype())
            .and_then(|vt| vt.tensor_type());
        Ok(matches!(dtype, Some(TensorElementType::Float16)))
    })?;
    let ns = knobs.native_scale;

    let run_tile = |rgb_tile: &RgbImage, padded_w: u32, padded_h: u32| -> Result<RgbImage, CoreError> {
        const INV_255: f32 = 1.0 / 255.0;
        let shape = [1, 3, padded_h as usize, padded_w as usize];

        let out_h = padded_h * ns;
        let out_w = padded_w * ns;
        let plane = (out_w * out_h) as usize;
        let mut packed = vec![0u8; 3 * plane];

        let tensor = if is_fp16 {
            chw_tensor(rgb_tile, shape, |v| f16::from_f32(v * INV_255))?
        } else {
            chw_tensor(rgb_tile, shape, |v| v * INV_255)?
        };
        engine.with_session_for(&shape, |session| {
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
    let native = upscale_tiled(input, ns, cfg, run_tile, progress)?;
    Ok(fit_to_scale(native, input, scale))
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

/// ORT reports a `terminate()` abort only as a generic error mentioning
/// "terminate flag"; it maps to `Cancelled`, not an inference failure.
fn classify_run_error(err: ort::Error) -> CoreError {
    let msg = err.to_string();
    if msg.contains("terminate flag") || msg.contains("Exiting due to terminate") {
        CoreError::Cancelled
    } else {
        CoreError::Inference(format!("upscale: inference failed: {msg}"))
    }
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
    alpha::take_input_alpha(&mut out, input);
    out
}

/// What an upscale of an `in_w` × `in_h` image by `scale` with `model`
/// needs free: its buffers at their peak and the model's working set.
pub fn needed_bytes(model: prunr_models::ModelId, in_w: u32, in_h: u32, scale: u32) -> u64 {
    peak_bytes(model, in_w, in_h, scale) + working_set_bytes(model)
}

/// What a parted run may still take of `free`: the parts' sessions are
/// in memory already (`resident`), so only the rest of the working set is
/// still to come. Nothing readable: none, as the app's gates refuse then.
fn parted_room(free: Option<u64>, working_set: u64, resident: u64) -> u64 {
    free.map_or(0, |free| free.saturating_sub(working_set.saturating_sub(resident)))
}

fn working_set_bytes(model: prunr_models::ModelId) -> u64 {
    prunr_models::descriptor(model).map_or(0, |d| u64::from(d.needs.host_cpu().host_mb)) * 1024 * 1024
}

/// What an upscale of an `in_w` × `in_h` image by `scale` with `model`
/// holds at its peak, besides the input and the model: the tiler's
/// buffers, or the least a part-by-part run can hold, at the model's own
/// scale, or the resample to `scale` when that is larger.
fn peak_bytes(model: prunr_models::ModelId, in_w: u32, in_h: u32, scale: u32) -> u64 {
    let descriptor = prunr_models::descriptor(model);
    let native = descriptor.and_then(|d| d.upscale.as_ref()).map_or(scale, |k| k.native_scale);
    let run = match descriptor.and_then(|d| d.part_layout()) {
        Some(layout) => blocks::least_peak_bytes(layout, in_w, in_h, native),
        None => {
            // A model without a tile size refuses to run; count the whole height.
            let tile = descriptor.and_then(|d| d.recommended_tile).unwrap_or(in_h);
            tiling::tiled_peak_bytes(in_w, in_h, native, tile)
        }
    };
    if native == scale {
        return run;
    }
    let pixels = |s: u32| u64::from(in_w) * u64::from(in_h) * u64::from(s) * u64::from(s);
    run.max(pixels(native) * 4 + pixels(scale) * 5)
}

/// Builds its own engine; for callers without a warm engine (CLI, tests).
///
/// Peak working-set RAM of a tiled model (`peak_bytes` counts a model run
/// in parts too), additive to the tiler's accumulators:
///   - Tile input scratch: `3 × padded_w × padded_h × 4 bytes` (f32 CHW).
///     ~3 MB at 512-tile.
///   - Tile output scratch: `3 × out_w × out_h bytes` (u8 HWC).
///     ~12 MB at 512→2048 4× tile.
///   - scale=2 with a 4× model only: the full 4× RgbaImage (~16× input
///     bytes) lives until the Lanczos3 downscale completes.
///   - ONNX session: model-dependent (see `ModelDescriptor.needs`).
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

    /// A parted run's room leaves out what its sessions already hold.
    #[test]
    fn a_parted_run_counts_only_the_working_set_still_to_come() {
        assert_eq!(parted_room(Some(10), 4, 1), 7);
        assert_eq!(parted_room(Some(10), 4, 6), 10, "sessions past the working set leave all of it");
        assert_eq!(parted_room(Some(2), 4, 0), 0);
        assert_eq!(parted_room(None, 4, 1), 0, "nothing readable: no room");
    }

    /// What an upscale needs free is its peak and the model's working set:
    /// for a model run in parts the least it can hold, whatever the Scale.
    #[test]
    fn an_upscale_needs_its_peak_and_the_models_working_set() {
        use prunr_models::ModelId;
        let working_set = working_set_bytes;
        let nomos2 = ModelId::FourXNomos2RealPlksr;
        // The features and the image (268 B a pixel), and half-precision
        // blocks' outputs or two bands of 250 of the 1000 rows: 128 B.
        assert_eq!(needed_bytes(nomos2, 1000, 1000, 4), 1_000_000 * 268 + 2 * 250 * 1000 * 256 + working_set(nomos2));
        assert_eq!(needed_bytes(nomos2, 1000, 1000, 2), needed_bytes(nomos2, 1000, 1000, 4), "the run at 4x outweighs the resample");
        let x4 = ModelId::RealEsrganX4Plus;
        assert_eq!(needed_bytes(x4, 4096, 4096, 4), peak_bytes(x4, 4096, 4096, 4) + working_set(x4));
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
        assert_eq!(tile_for_provider(512, "CoreML"), 256);
        assert_eq!(COREML_TILE_CAP % 16, 0, "window-attention models need a multiple of 16");
        assert_eq!(OPENVINO_TILE_CAP % 16, 0, "window-attention models need a multiple of 16");
    }

    #[test]
    fn every_providers_tile_is_a_multiple_the_model_takes() {
        for descriptor in prunr_models::REGISTRY {
            let (Some(tile), Some(multiple)) = (descriptor.recommended_tile, descriptor.tile_size_multiple) else { continue };
            for provider in ["CPU", "CUDA", "OpenVINO", "CoreML", "DirectML", "MIGraphX"] {
                assert_eq!(tile_for_provider(tile, provider) % multiple, 0, "{:?} on {provider}", descriptor.id);
            }
        }
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
}
