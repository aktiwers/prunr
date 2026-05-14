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
/// A second tile with different dimensions then hits an irrecoverable
/// shape-mismatch. Empirically confirmed against 4xNomos8kSCHAT-L
/// (PRECONDITIONS.md P-3): Level3 panics with
/// "Attempting to get index by a name which does not exist:
/// InsertedPrecisionFreeCast_..."; Level2 succeeds across all tile sizes.
fn pick_optimization_level(descriptor: &prunr_models::ModelDescriptor) -> GraphOptimizationLevel {
    if descriptor.tile_size_multiple.is_some() {
        GraphOptimizationLevel::Level2
    } else {
        GraphOptimizationLevel::Level3
    }
}

/// Tensor input name for an upscale model. Confirmed from PRECONDITIONS.md:
/// RealESRGAN exports as `"data"`, Nomos8kSCHAT-L (Phhofm) exports as `"input"`.
fn upscale_input_name(id: prunr_models::ModelId) -> &'static str {
    match id {
        prunr_models::ModelId::RealEsrganX4Plus => "data",
        prunr_models::ModelId::Nomos8kSchatL => "input",
        _ => "input",
    }
}

/// Upscale an RGBA image by `scale` (2 or 4) using the ONNX model
/// identified by `model_id`.
///
/// Behavior:
///   - scale = 4: runs the model once, returns the 4x output.
///   - scale = 2: runs the model at 4x then downscales with
///     Lanczos3 to halve dimensions.
///   - Alpha is upscaled independently via Lanczos3.
///   - Window-attention models (descriptor.tile_size_multiple.is_some())
///     are run at GraphOptimizationLevel::Level2 to avoid first-tile
///     shape baking; other models run at Level3.
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
        let pixel_count = (padded_w * padded_h) as usize;
        let mut input_data = vec![0.0_f32; 3 * pixel_count];
        for (i, p) in rgb_tile.pixels().enumerate() {
            let [r, g, b] = p.0;
            let h_idx = (i as u32) / padded_w;
            let w_idx = (i as u32) % padded_w;
            let plane = pixel_count;
            let pos = (h_idx * padded_w + w_idx) as usize;
            input_data[pos] = r as f32 / 255.0;
            input_data[plane + pos] = g as f32 / 255.0;
            input_data[2 * plane + pos] = b as f32 / 255.0;
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

        // Run inference and extract the output as owned data inside the
        // session lock. `SessionOutputs` borrows from the session, so we
        // must convert to `Vec<f32>` before the lock guard drops.
        let output_vec: Vec<f32> = engine.with_session(|session| {
            let outputs = session
                .run(inputs![input_name => &tensor])
                .map_err(|e| CoreError::Inference(format!("upscale: inference failed: {e}")))?;

            let arr = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| CoreError::Inference(format!("upscale: output extract: {e}")))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| CoreError::Inference(format!("upscale: output reshape: {e}")))?;

            // Convert to owned Vec inside the closure so the borrow from
            // `outputs` ends before the session lock releases.
            Ok(arr.into_owned().into_raw_vec_and_offset().0)
        })?;

        // Hoist slice extraction outside per-pixel loop. `output_vec` is
        // already owned so as_slice is always Some.
        let view_slice = &output_vec;

        if view_slice.len() < 3 * plane {
            return Err(CoreError::Inference(format!(
                "upscale: output tensor too small: {} < {}",
                view_slice.len(),
                3 * plane
            )));
        }

        let mut out_img = RgbImage::new(out_w, out_h);
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = (y * out_w + x) as usize;
                let r = (view_slice[idx] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (view_slice[plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (view_slice[2 * plane + idx] * 255.0).clamp(0.0, 255.0) as u8;
                out_img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        Ok(out_img)
    };

    let cfg = TilingConfig { tile_size, tile_multiple, overlap };
    let scale4_result = upscale_tiled(input, 4, cfg, run_tile, on_tile_done, cancel)?;

    if scale == 4 {
        Ok(scale4_result)
    } else if scale == 2 {
        let half_w = scale4_result.width() / 2;
        let half_h = scale4_result.height() / 2;

        // Downscale RGB channels from 4x to 2x.
        let rgb4 = image::DynamicImage::ImageRgba8(scale4_result);
        let rgb_half = crate::formats::resize_rgb_lanczos3(&rgb4, half_w, half_h);

        // Upscale alpha from input to 2x directly (skips the intermediate 4x).
        let alpha_half = crate::formats::resize_gray_lanczos3(
            &image::GrayImage::from_fn(input.width(), input.height(), |x, y| {
                image::Luma([input.get_pixel(x, y).0[3]])
            }),
            half_w,
            half_h,
        );

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
    /// This pins the invariant empirically confirmed in PRECONDITIONS.md P-3:
    /// Level3 panics with shape-mismatch on HAT-family models when the second
    /// tile has different dimensions than the first.
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
