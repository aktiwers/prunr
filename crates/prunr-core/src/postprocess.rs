use image::{DynamicImage, GrayImage, RgbaImage};
use ndarray::ArrayView4;
use rayon::prelude::*;

use crate::formats::resize_gray_lanczos3;
use crate::guided_filter::guided_filter_alpha;
use crate::types::{MaskSettings, ModelKind};

/// Postprocess raw ONNX model output into a transparent RGBA image.
/// Allocates the RGBA buffer once and reuses it for guided filter (if enabled)
/// and final mask application — avoids two 4×width×height allocations.
pub fn postprocess(raw: ArrayView4<f32>, original: &DynamicImage, mask_settings: &MaskSettings, model: ModelKind) -> RgbaImage {
    let mut rgba = original.to_rgba8();
    let mask = tensor_to_mask_with_rgba(raw, &rgba, mask_settings, model);
    apply_mask_inplace(&mut rgba, &mask);
    rgba
}

/// Postprocess from a flat f32 tensor slice. Used by subprocess paths where
/// tensor data arrives via IPC as a Vec<f32> that needs reshaping to [1,1,H,W].
pub fn postprocess_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &DynamicImage,
    mask_settings: &MaskSettings,
    model: ModelKind,
) -> Result<RgbaImage, crate::types::CoreError> {
    let view = ArrayView4::from_shape((1, 1, tensor_h, tensor_w), tensor)
        .map_err(|e| crate::types::CoreError::Inference(format!("Tensor reshape: {e}")))?;
    Ok(postprocess(view, original, mask_settings, model))
}

/// Convert raw ONNX tensor to a full-resolution grayscale mask (Tier 2).
/// Applies normalization, gamma, threshold, resize, edge shift, and guided filter.
pub fn tensor_to_mask(raw: ArrayView4<f32>, original: &DynamicImage, mask_settings: &MaskSettings, model: ModelKind) -> GrayImage {
    // Materializing rgba here is wasteful when refine_edges is false; callers on
    // the hot path should use `postprocess()` which shares the RGBA buffer.
    let rgba = if mask_settings.refine_edges { Some(original.to_rgba8()) } else { None };
    tensor_to_mask_core(raw, original.width(), original.height(), rgba.as_ref(), mask_settings, model)
}

/// Same as `tensor_to_mask` but reuses an already-materialized RGBA buffer.
fn tensor_to_mask_with_rgba(raw: ArrayView4<f32>, rgba: &RgbaImage, mask_settings: &MaskSettings, model: ModelKind) -> GrayImage {
    tensor_to_mask_core(raw, rgba.width(), rgba.height(), Some(rgba), mask_settings, model)
}

fn tensor_to_mask_core(raw: ArrayView4<f32>, ow: u32, oh: u32, rgba_for_guided: Option<&RgbaImage>, mask_settings: &MaskSettings, model: ModelKind) -> GrayImage {
    let pred = raw.slice(ndarray::s![0, 0, .., ..]);

    let use_sigmoid = matches!(model, ModelKind::BiRefNetLite);

    // rembg models need min-max stats; BiRefNet uses sigmoid instead
    let (mi, range, uniform_val) = if !use_sigmoid {
        let (mi, ma) = pred.iter().cloned().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        );
        let r = ma - mi;
        if r < 1e-6 {
            // Uniform output — use the absolute value to decide:
            // rembg models output ~0 for background, ~1 for foreground
            // after min-max normalization. A uniform value > 0.5 means
            // "everything is foreground" → full opacity.
            (mi, 1.0, Some(if ma > 0.5 { 1.0f32 } else { 0.0 }))
        } else {
            (mi, r, None)
        }
    } else {
        (0.0, 1.0, None)
    };

    let (sh, sw) = (pred.nrows(), pred.ncols());
    let contiguous;
    let pred_slice = match pred.as_slice() {
        Some(s) => s,
        None => { contiguous = pred.as_standard_layout(); contiguous.as_slice().unwrap() }
    };
    let gamma = mask_settings.gamma;
    let threshold = mask_settings.threshold;

    // Short-circuit: uniform output → fill with constant, skip per-pixel loop
    let mask_buf = if let Some(uv) = uniform_val {
        let mut val = uv;
        if gamma != 1.0 { val = val.powf(gamma); }
        if let Some(t) = threshold { val = if val >= t { 1.0 } else { 0.0 }; }
        vec![(val * 255.0) as u8; sw * sh]
    } else {
        let inv_range = 1.0 / range;
        let mut buf = vec![0u8; sw * sh];
        for i in 0..sh * sw {
            let raw_val = pred_slice[i];
            let mut val = if use_sigmoid {
                1.0 / (1.0 + (-raw_val).exp())
            } else {
                ((raw_val - mi) * inv_range).clamp(0.0, 1.0)
            };

            if gamma != 1.0 {
                val = val.powf(gamma);
            }
            if let Some(t) = threshold {
                val = if val >= t { 1.0 } else { 0.0 };
            }

            buf[i] = (val * 255.0) as u8;
        }
        buf
    };
    let mask = GrayImage::from_raw(sw as u32, sh as u32, mask_buf)
        .expect("mask buffer size matches dimensions");

    // Resize mask back to original dimensions (SIMD-accelerated Lanczos3)
    let mut mask = resize_gray_lanczos3(&mask, ow, oh);

    // Edge shift: positive erodes (shrinks foreground), negative dilates (expands it)
    if mask_settings.edge_shift.abs() > 0.01 {
        apply_edge_shift(&mut mask, mask_settings.edge_shift);
    }

    if mask_settings.refine_edges {
        if let Some(rgba) = rgba_for_guided {
            const GUIDED_RADIUS: u32 = 8;
            const GUIDED_EPSILON: f32 = 1e-4;
            mask = guided_filter_alpha(rgba, &mask, GUIDED_RADIUS, GUIDED_EPSILON);
        }
    }

    mask
}

/// Apply a grayscale mask as the alpha channel on the original image (Tier 3).
pub fn apply_mask(original: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let mut rgba = original.to_rgba8();
    apply_mask_inplace(&mut rgba, mask);
    rgba
}

/// Write the mask into an existing RGBA buffer's alpha channel in place.
/// Used by `postprocess()` to avoid a second full-resolution RGBA allocation.
fn apply_mask_inplace(rgba: &mut RgbaImage, mask: &GrayImage) {
    let mask_raw = mask.as_raw();
    let out_raw = rgba.as_mut();
    for (pixel, &alpha) in out_raw.chunks_mut(4).zip(mask_raw.iter()) {
        pixel[3] = alpha;
    }
}

/// Erode (positive shift) or dilate (negative shift) the mask.
///
/// Uses iterative box-blur approximation: blur the mask, then re-threshold.
/// Each iteration shifts the boundary by ~1px.
fn apply_edge_shift(mask: &mut GrayImage, shift: f32) {
    let iterations = shift.abs().round() as u32;
    if iterations == 0 { return; }
    let erode = shift > 0.0;
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let wi = w as i32;
    let hi = h as i32;

    let mut a = mask.as_raw().clone();
    let mut b = vec![0u8; w * h];
    let use_par = h >= 512;

    for _ in 0..iterations {
        let process_row = |(y, row): (usize, &mut [u8])| {
            let yi = y as i32;
            for x in 0..w {
                let xi = x as i32;
                let mut extremum: u8 = if erode { 255 } else { 0 };
                for dy in -1i32..=1 {
                    let ny = (yi + dy).clamp(0, hi - 1) as usize;
                    for dx in -1i32..=1 {
                        let nx = (xi + dx).clamp(0, wi - 1) as usize;
                        let v = a[ny * w + nx];
                        if erode {
                            extremum = extremum.min(v);
                        } else {
                            extremum = extremum.max(v);
                        }
                    }
                }
                row[x] = extremum;
            }
        };

        if use_par {
            b.par_chunks_mut(w).enumerate().for_each(process_row);
        } else {
            b.chunks_mut(w).enumerate().for_each(process_row);
        }
        std::mem::swap(&mut a, &mut b);
    }

    mask.as_mut().copy_from_slice(&a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, Rgb};
    use ndarray::Array4;

    fn make_raw_tensor(val: f32) -> Array4<f32> {
        Array4::from_elem((1, 1, 320, 320), val)
    }

    fn solid_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([100, 150, 200])))
    }

    #[test]
    fn test_postprocess_output_dimensions() {
        let raw = make_raw_tensor(0.5);
        let original = solid_rgb(640, 480);
        let result = postprocess(raw.view(), &original, &MaskSettings::default(), ModelKind::Silueta);
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_zero() {
        // All-zero tensor: mi = ma = 0, range = 1e-6
        // (0 - 0) / 1e-6 = 0 -> alpha = 0
        let raw = make_raw_tensor(0.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &MaskSettings::default(), ModelKind::Silueta);
        // All alpha values should be 0
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 0, "Expected alpha=0 for all-zero tensor");
        }
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_one() {
        // All-one tensor: uniform high confidence → foreground → alpha=255
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &MaskSettings::default(), ModelKind::Silueta);
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 255, "Expected alpha=255 for uniform high-confidence tensor");
        }
    }

    #[test]
    fn test_postprocess_continuous_alpha() {
        // Tensor with gradient 0..1 should produce multiple distinct alpha values
        let mut raw = Array4::<f32>::zeros((1, 1, 320, 320));
        for y in 0..320_usize {
            for x in 0..320_usize {
                raw[[0, 0, y, x]] = (y * 320 + x) as f32 / (320.0 * 320.0);
            }
        }
        let original = solid_rgb(320, 320);
        let result = postprocess(raw.view(), &original, &MaskSettings::default(), ModelKind::Silueta);
        let unique_alphas: std::collections::HashSet<u8> =
            result.enumerate_pixels().map(|(_, _, p)| p[3]).collect();
        assert!(
            unique_alphas.len() > 10,
            "Expected many distinct alpha values, got {}",
            unique_alphas.len()
        );
    }
}
