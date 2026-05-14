//! Tile-based ONNX upscale dispatch. RGB through the model with
//! overlap-blend; alpha resampled with Lanczos3 in parallel.

mod tiling;
mod alpha;

pub use alpha::upscale_alpha_lanczos3;
pub use tiling::{plan_upscale_tiles, upscale_tiled, TilingConfig, UpscaleTilePlacement};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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

/// Tensor input name for an upscale model. RealESRGAN exports as `"data"`;
/// HAT-family (Phhofm Nomos8kSCHAT-L) exports as `"input"`. New upscale
/// variants must extend this match — the `unreachable!` fires loud if a
/// non-upscale `ModelId` is routed here.
fn upscale_input_name(id: prunr_models::ModelId) -> &'static str {
    match id {
        prunr_models::ModelId::RealEsrganX4Plus => "data",
        prunr_models::ModelId::Nomos8kSchatL => "input",
        other => unreachable!("upscale_input_name called with non-upscale ModelId: {other:?}"),
    }
}

/// Upscale an RGBA image by `scale` (2 or 4) using the ONNX model
/// identified by `model_id`.
///
/// Behavior:
///   - scale = 4: runs the model once, returns the 4× output.
///   - scale = 2: runs the model at 4× then downscales with Lanczos3 to
///     halve dimensions. This path holds the full 4× RGBA buffer briefly
///     before the downscale; for a 4K input that is ~768 MB of transient
///     scratch on top of the tiling accumulators below. The cost is
///     accepted for v1 — see `30-DEFERRED.md` for the native 2× option.
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
///   - scale=2 only: the full 4× RgbaImage (~16× input bytes) lives
///     until the Lanczos3 downscale completes.
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

    let tile_size = descriptor
        .recommended_tile
        .ok_or_else(|| CoreError::Inference(format!(
            "model {model_id:?} has no recommended_tile -- upscale cannot dispatch"
        )))?;

    let tile_multiple = descriptor.tile_size_multiple;
    let overlap = match tile_multiple {
        Some(_) => 32, // HAT: 2 window widths
        None => 16,    // ESRGAN: 16 px safe overlap
    };

    let model_kind = ModelKind::from(model_id);
    let engine = OrtEngine::new_with_optimization_level(model_kind, intra_threads, level)?;

    let input_name = upscale_input_name(model_id);

    let run_tile = |rgb_tile: &RgbImage, padded_w: u32, padded_h: u32| -> Result<RgbImage, CoreError> {
        const INV_255: f32 = 1.0 / 255.0;
        let pixel_count = (padded_w * padded_h) as usize;
        let mut input_data = vec![0.0_f32; 3 * pixel_count];
        for (i, p) in rgb_tile.pixels().enumerate() {
            let [r, g, b] = p.0;
            input_data[i] = r as f32 * INV_255;
            input_data[pixel_count + i] = g as f32 * INV_255;
            input_data[2 * pixel_count + i] = b as f32 * INV_255;
        }

        let arr = ndarray::Array4::from_shape_vec(
            [1, 3, padded_h as usize, padded_w as usize],
            input_data,
        )
        .map_err(|e| CoreError::Inference(format!("upscale: input shape: {e}")))?;

        let tensor = Tensor::from_array(arr)
            .map_err(|e| CoreError::Inference(format!("upscale: input tensor: {e}")))?;

        let out_h = padded_h * 4;
        let out_w = padded_w * 4;
        let plane = (out_w * out_h) as usize;

        // Pack CHW f32 → HWC u8 directly from the ORT-borrowed slice while
        // still inside the session lock. Avoids one full output-tensor copy
        // (~50 MB per tile at 512→2048 4×).
        let mut packed = vec![0u8; 3 * plane];
        engine.with_session(|session| {
            let outputs = session
                .run(inputs![input_name => &tensor])
                .map_err(|e| CoreError::Inference(format!("upscale: inference failed: {e}")))?;

            let arr = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| CoreError::Inference(format!("upscale: output extract: {e}")))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| CoreError::Inference(format!("upscale: output reshape: {e}")))?;

            let slice = arr.as_slice().ok_or_else(|| {
                CoreError::Inference("upscale: output tensor not contiguous".into())
            })?;

            if slice.len() < 3 * plane {
                return Err(CoreError::Inference(format!(
                    "upscale: output tensor too small: {} < {}",
                    slice.len(),
                    3 * plane
                )));
            }

            for idx in 0..plane {
                let r = (slice[idx] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (slice[plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (slice[2 * plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
                packed[3 * idx] = r;
                packed[3 * idx + 1] = g;
                packed[3 * idx + 2] = b;
            }
            Ok(())
        })?;

        RgbImage::from_raw(out_w, out_h, packed).ok_or_else(|| {
            CoreError::Inference("upscale: from_raw failed (packed length mismatch)".into())
        })
    };

    let cfg = TilingConfig { tile_size, tile_multiple, overlap };
    let scale4_result = upscale_tiled(input, 4, cfg, run_tile, on_tile_done, cancel)?;

    if scale == 4 {
        Ok(scale4_result)
    } else if scale == 2 {
        let half_w = scale4_result.width() / 2;
        let half_h = scale4_result.height() / 2;

        let rgb4 = image::DynamicImage::ImageRgba8(scale4_result);
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
}
