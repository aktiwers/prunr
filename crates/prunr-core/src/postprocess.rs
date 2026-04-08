use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage, imageops::FilterType};
use ndarray::ArrayView4;

use crate::types::MaskSettings;

/// Postprocess raw ONNX model output into a transparent RGBA image.
///
/// Pipeline:
/// 1. Slice channel 0: output[0, 0, :, :] -> shape [320, 320]
/// 2. Min-max normalize to [0, 1]
/// 3. Apply gamma curve (mask_settings.gamma)
/// 4. Apply optional binary threshold (mask_settings.threshold)
/// 5. Scale to u8 grayscale mask, resize to original dimensions (Lanczos3)
/// 6. Apply edge shift (erode/dilate via distance-based approach)
/// 7. Apply mask as alpha channel
pub fn postprocess(raw: ArrayView4<f32>, original: &DynamicImage, mask_settings: &MaskSettings) -> RgbaImage {
    let pred = raw.slice(ndarray::s![0, 0, .., ..]);  // [320, 320]

    let ma = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mi = pred.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (ma - mi).max(1e-6_f32);

    let (sh, sw) = (pred.nrows(), pred.ncols());
    let mut mask = GrayImage::new(sw as u32, sh as u32);
    for y in 0..sh {
        for x in 0..sw {
            let mut val = ((pred[[y, x]] - mi) / range).clamp(0.0, 1.0);

            // Gamma curve: >1 = more aggressive, <1 = gentler
            if mask_settings.gamma != 1.0 {
                val = val.powf(mask_settings.gamma);
            }

            // Binary threshold
            if let Some(t) = mask_settings.threshold {
                val = if val >= t { 1.0 } else { 0.0 };
            }

            mask.put_pixel(x as u32, y as u32, Luma([(val * 255.0) as u8]));
        }
    }

    // Resize mask back to original dimensions using Lanczos3
    let (ow, oh) = (original.width(), original.height());
    let mut mask = image::imageops::resize(&mask, ow, oh, FilterType::Lanczos3);

    // Edge shift: positive erodes (shrinks foreground), negative dilates (expands it)
    if mask_settings.edge_shift.abs() > 0.01 {
        apply_edge_shift(&mut mask, mask_settings.edge_shift);
    }

    // Apply as alpha channel
    let rgba = original.to_rgba8();
    let mut out = RgbaImage::new(ow, oh);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = mask.get_pixel(x, y)[0];
        out.put_pixel(x, y, Rgba([p[0], p[1], p[2], a]));
    }
    out
}

/// Erode (positive shift) or dilate (negative shift) the mask.
///
/// Uses iterative box-blur approximation: blur the mask, then re-threshold.
/// Each iteration shifts the boundary by ~1px.
fn apply_edge_shift(mask: &mut GrayImage, shift: f32) {
    let iterations = shift.abs().round() as u32;
    if iterations == 0 { return; }
    let erode = shift > 0.0;

    let mut buf = mask.clone();
    for i in 0..iterations {
        let (src, dst) = if i % 2 == 0 { (&*mask, &mut buf) } else { (&buf, &mut *mask) };
        let (w, h) = (src.width(), src.height());
        for y in 0..h {
            for x in 0..w {
                let mut extremum: u8 = if erode { 255 } else { 0 };
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                        let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                        let v = src.get_pixel(nx, ny)[0];
                        if erode {
                            extremum = extremum.min(v);
                        } else {
                            extremum = extremum.max(v);
                        }
                    }
                }
                dst.put_pixel(x, y, Luma([extremum]));
            }
        }
    }
    // If last iteration wrote to buf, copy back
    if iterations % 2 == 1 {
        mask.clone_from(&buf);
    }
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
        let result = postprocess(raw.view(), &original, &MaskSettings::default());
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_zero() {
        // All-zero tensor: mi = ma = 0, range = 1e-6
        // (0 - 0) / 1e-6 = 0 -> alpha = 0
        let raw = make_raw_tensor(0.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &MaskSettings::default());
        // All alpha values should be 0
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 0, "Expected alpha=0 for all-zero tensor");
        }
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_one() {
        // All-one tensor: mi = ma = 1, range = 1e-6
        // (1 - 1) / 1e-6 = 0 -> alpha = 0
        // Note: uniform tensor always produces 0 alpha (no dynamic range)
        // This is mathematically correct for min-max on constant input
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &MaskSettings::default());
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 0, "Expected alpha=0 for uniform tensor (no range)");
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
        let result = postprocess(raw.view(), &original, &MaskSettings::default());
        let unique_alphas: std::collections::HashSet<u8> =
            result.enumerate_pixels().map(|(_, _, p)| p[3]).collect();
        assert!(
            unique_alphas.len() > 10,
            "Expected many distinct alpha values, got {}",
            unique_alphas.len()
        );
    }
}
