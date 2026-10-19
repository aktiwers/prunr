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

    /// ImageNet mean/std per channel, laid out NCHW at 1024²: white
    /// reads (1 - mean) / std, black -mean / std, and an all-red image
    /// keeps R in the first plane (NHWC would put G at out[1]).
    #[test]
    fn preprocess_normalizes_per_channel_in_nchw() {
        let n = 1024 * 1024;
        let at = |rgb: [u8; 3], side: u32| preprocess_for_sam(&image::RgbaImage::from_pixel(side, side, image::Rgba([rgb[0], rgb[1], rgb[2], 255])));
        let white = at([255, 255, 255], 64);
        assert_eq!(white.len(), 3 * n);
        let black = at([0, 0, 0], 64);
        let red = at([255, 0, 0], 1024);
        for (out, want) in [(&white, [2.249, 2.429, 2.640]), (&black, [-2.118, -2.036, -1.804]), (&red, [2.249, -2.036, -1.804])] {
            for (c, w) in want.into_iter().enumerate() {
                assert!(approx_eq(out[c * n], w, 0.01), "channel {c}: {} vs {w}", out[c * n]);
            }
        }
        assert!(approx_eq(red[1], 2.249, 0.01), "out[1] is still R: {}", red[1]);
    }
}
