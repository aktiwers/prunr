//! SAM 2 encoder input builder.
//!
//! RGBA source → 1024×1024 ImageNet-normalized f32 tensor in NCHW layout.
//!
//! Peak working set:
//! | Stage        | RAM    |
//! |--------------|--------|
//! | Input RGBA   | caller-owned, not counted |
//! | Resized RGBA | 1024×1024×4 ≈ 4 MB scratch |
//! | NCHW output  | 1024×1024×3×4 ≈ 12 MB returned |
//! | Total peak   | ~15 MB transient |

use super::SAM_ENCODER_INPUT;
use crate::preprocess::{IMAGENET_MEAN, IMAGENET_STD};

/// Build the SAM 2 encoder input tensor from a source RGBA image.
///
/// 1. Stretch-resize to 1024×1024 via SIMD Lanczos3, straight from the
///    borrowed RGBA (no full-resolution copy).
/// 2. Drop alpha — SAM 2 encoder is RGB-only.
/// 3. Normalize per-channel: `(pixel/255.0 - mean) / std`.
/// 4. Layout: NCHW — output[0..n] = all R values, output[n..2n] = G, output[2n..3n] = B,
///    where n = 1024*1024.
///
/// Returns a `Vec<f32>` of length 3_145_728 (3 × 1024 × 1024).
pub fn preprocess_for_sam(source: &image::RgbaImage) -> Vec<f32> {
    let resized = crate::formats::resize_rgba(
        source,
        SAM_ENCODER_INPUT,
        SAM_ENCODER_INPUT,
        crate::formats::ResizeFilter::Lanczos3,
    );
    // One 256-entry table per channel: the same expression per entry,
    // so the values are bit-identical to computing it per pixel.
    let table = |c: usize| -> [f32; 256] {
        std::array::from_fn(|v| (v as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c])
    };
    let (tr, tg, tb) = (table(0), table(1), table(2));
    let n = (SAM_ENCODER_INPUT * SAM_ENCODER_INPUT) as usize;
    let mut out = vec![0.0f32; 3 * n];
    let (r, gb) = out.split_at_mut(n);
    let (g, b) = gb.split_at_mut(n);
    for (((r, g), b), px) in r.iter_mut().zip(g).zip(b).zip(resized.as_raw().chunks_exact(4)) {
        *r = tr[px[0] as usize];
        *g = tg[px[1] as usize];
        *b = tb[px[2] as usize];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn preprocess_returns_correct_length() {
        let img = image::RgbaImage::new(64, 64);
        let out = preprocess_for_sam(&img);
        assert_eq!(out.len(), 3 * 1024 * 1024);
    }

    #[test]
    fn preprocess_normalizes_white_pixel_correctly() {
        let img = image::RgbaImage::from_pixel(64, 64, image::Rgba([255, 255, 255, 255]));
        let out = preprocess_for_sam(&img);
        let n = 1024 * 1024;
        // R channel: (1.0 - 0.485) / 0.229 ≈ 2.249
        assert!(
            approx_eq(out[0], 2.249, 0.01),
            "R white expected ~2.249, got {}",
            out[0]
        );
        // G channel: (1.0 - 0.456) / 0.224 ≈ 2.429
        assert!(
            approx_eq(out[n], 2.429, 0.01),
            "G white expected ~2.429, got {}",
            out[n]
        );
        // B channel: (1.0 - 0.406) / 0.225 ≈ 2.640
        assert!(
            approx_eq(out[2 * n], 2.640, 0.01),
            "B white expected ~2.640, got {}",
            out[2 * n]
        );
    }

    #[test]
    fn preprocess_normalizes_black_pixel_correctly() {
        let img = image::RgbaImage::from_pixel(64, 64, image::Rgba([0, 0, 0, 255]));
        let out = preprocess_for_sam(&img);
        let n = 1024 * 1024;
        // R = (0.0 - 0.485) / 0.229 ≈ -2.118
        assert!(
            approx_eq(out[0], -2.118, 0.01),
            "R black expected ~-2.118, got {}",
            out[0]
        );
        // G = (0.0 - 0.456) / 0.224 ≈ -2.036
        assert!(
            approx_eq(out[n], -2.036, 0.01),
            "G black expected ~-2.036, got {}",
            out[n]
        );
        // B = (0.0 - 0.406) / 0.225 ≈ -1.804
        assert!(
            approx_eq(out[2 * n], -1.804, 0.01),
            "B black expected ~-1.804, got {}",
            out[2 * n]
        );
    }

    #[test]
    fn preprocess_layout_is_nchw_not_nhwc() {
        // 1024×1024 all-red image: R post-norm ≈ 2.249, G ≈ -2.036, B ≈ -1.804
        let img = image::RgbaImage::from_pixel(1024, 1024, image::Rgba([255, 0, 0, 255]));
        let out = preprocess_for_sam(&img);
        let n = 1024 * 1024;
        // out[0..n] should all be ≈ R-norm value
        assert!(approx_eq(out[0], 2.249, 0.01), "out[0] = {} (expected R-norm ~2.249)", out[0]);
        // out[n..2n] should all be ≈ G-norm value (0 red → G=0 → (0-0.456)/0.224 ≈ -2.036)
        assert!(approx_eq(out[n], -2.036, 0.01), "out[n] = {} (expected G-norm ~-2.036)", out[n]);
        // out[2n..3n] should all be ≈ B-norm value
        assert!(approx_eq(out[2 * n], -1.804, 0.01), "out[2n] = {} (expected B-norm ~-1.804)", out[2 * n]);
        // NHWC would interleave: out[1] would be G-norm; in NCHW out[1] is still R-norm
        assert!(
            approx_eq(out[1], 2.249, 0.01),
            "out[1] = {} — expected ~2.249 (R-norm for NCHW); if NHWC this would be G-norm",
            out[1]
        );
    }
}
