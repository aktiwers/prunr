//! Colorize: DDColor predicts the a/b colour channels of a photo from
//! its lightness at 512×512; the photo keeps its own full-resolution
//! lightness and takes the predicted colour, scaled by a strength.

use image::RgbaImage;
use rayon::prelude::*;

use crate::color::{lab_to_srgb, srgb_lightness};
use crate::math::{bilinear_sample, bilinear_taps, Tap};
use crate::progress::{Progress, Step};
use crate::{CoreError, OrtEngine};

/// Edge length of the model's square input and of its colour output.
pub const COLORIZE_INPUT: u32 = 512;

/// The strongest colour Strength allows: half again the prediction.
pub const STRENGTH_MAX: f32 = 1.5;

/// The predicted colour: `a` then `b`, each `COLORIZE_INPUT²` values in
/// Lab units, row-major.
pub type ColorPlane = Vec<f32>;

/// The model's input: the photo's lightness alone, as grey RGB in
/// [0, 1], NCHW at `COLORIZE_INPUT²`. A colour photo loses its colour
/// here, so the model sees what a black-and-white print would show.
///
/// Peak working set: the 512² resized copy (1 MB) and the 3 MB tensor.
pub fn colorize_input(src: &RgbaImage) -> Vec<f32> {
    let resized = crate::formats::resize_rgba(
        src,
        COLORIZE_INPUT,
        COLORIZE_INPUT,
        crate::formats::ResizeFilter::Lanczos3,
    );
    let n = (COLORIZE_INPUT * COLORIZE_INPUT) as usize;
    let mut out = vec![0.0f32; 3 * n];
    for (i, px) in resized.as_raw().chunks_exact(4).enumerate() {
        let (r, g, b) = lab_to_srgb(srgb_lightness(px[0], px[1], px[2]), 0.0, 0.0);
        out[i] = r as f32 / 255.0;
        out[n + i] = g as f32 / 255.0;
        out[2 * n + i] = b as f32 / 255.0;
    }
    out
}

/// Run the colorizer on `src`: the predicted colour at
/// `COLORIZE_INPUT²`, to hand to `colorized` at any strength.
///
/// Peak working set: the 3 MB input tensor, the 2 MB plane and the
/// session's own scratch (`ModelDescriptor.working_set_mb`).
pub fn colorize_plane(engine: &OrtEngine, src: &RgbaImage, progress: &Progress) -> Result<ColorPlane, CoreError> {
    progress.step(Step::Colorizing);
    let side = COLORIZE_INPUT as usize;
    engine.run_single_f32("colorize", "input", [1, 3, side, side], colorize_input(src), 2 * side * side, progress)
}

/// `src` with its lightness kept and the colour taken from `plane`,
/// scaled by `strength` (0 = grey, 1 = as predicted, above 1 more
/// saturated). Alpha is unchanged.
///
/// Peak working set:
/// | Buffer        | Size            |
/// |---------------|-----------------|
/// | result        | w·h·4 bytes     |
/// | column taps   | w·12 bytes      |
/// | row taps      | h·12 bytes      |
/// 24 MP: ~96 MB, all of it the result.
pub fn colorized(src: &RgbaImage, plane: &[f32], strength: f32) -> RgbaImage {
    let n = (COLORIZE_INPUT * COLORIZE_INPUT) as usize;
    debug_assert_eq!(plane.len(), 2 * n);
    let (a_plane, b_plane) = plane.split_at(n);
    let (w, h) = src.dimensions();
    let (xs, ys) = (bilinear_taps(COLORIZE_INPUT, w), bilinear_taps(COLORIZE_INPUT, h));
    let stride = COLORIZE_INPUT as usize;
    let row_len = w as usize * 4;
    let mut out = vec![0u8; src.as_raw().len()];
    out.par_chunks_exact_mut(row_len)
        .zip(src.as_raw().par_chunks_exact(row_len))
        .zip(ys.par_iter())
        .for_each(|((row, src_row), ty)| {
            let sample = |p: &[f32], tx: &Tap| bilinear_sample(p, stride, tx, ty);
            for ((px, s), tx) in row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)).zip(&xs) {
                let (r, g, b) = lab_to_srgb(
                    srgb_lightness(s[0], s[1], s[2]),
                    sample(a_plane, tx) * strength,
                    sample(b_plane, tx) * strength,
                );
                px.copy_from_slice(&[r, g, b, s[3]]);
            }
        });
    // same dimensions as `src`, so the buffer length matches
    RgbaImage::from_raw(w, h, out).expect("buffer sized from src")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::srgb_to_lab;

    fn plane(a: f32, b: f32) -> ColorPlane {
        let n = (COLORIZE_INPUT * COLORIZE_INPUT) as usize;
        let mut p = vec![a; 2 * n];
        p[n..].fill(b);
        p
    }

    fn spread(px: &image::Rgba<u8>) -> u8 {
        let [r, g, b, _] = px.0;
        r.max(g).max(b) - r.min(g).min(b)
    }

    #[test]
    fn the_input_is_the_photos_lightness_as_grey() {
        let src = RgbaImage::from_pixel(40, 30, image::Rgba([200, 40, 40, 255]));
        let input = colorize_input(&src);
        let n = (COLORIZE_INPUT * COLORIZE_INPUT) as usize;
        assert_eq!(input.len(), 3 * n);
        let (l, _, _) = srgb_to_lab(200, 40, 40);
        let (grey, _, _) = lab_to_srgb(l, 0.0, 0.0);
        for c in 0..3 {
            assert!((input[c * n + n / 2] - grey as f32 / 255.0).abs() < 1.5 / 255.0, "channel {c}");
        }
    }

    #[test]
    fn zero_strength_or_no_colour_leaves_grey_at_the_photos_lightness() {
        let src = RgbaImage::from_pixel(33, 17, image::Rgba([90, 160, 210, 128]));
        for out in [colorized(&src, &plane(30.0, -20.0), 0.0), colorized(&src, &plane(0.0, 0.0), 1.0)] {
            for px in out.pixels() {
                assert!(spread(px) <= 1, "{px:?}");
                assert_eq!(px.0[3], 128, "alpha kept");
            }
            let (l_src, _, _) = srgb_to_lab(90, 160, 210);
            let p = out.get_pixel(5, 5).0;
            let (l_out, _, _) = srgb_to_lab(p[0], p[1], p[2]);
            assert!((l_src - l_out).abs() < 1.0, "lightness kept: {l_src} vs {l_out}");
        }
    }

    #[test]
    fn strength_scales_the_predicted_colour() {
        let src = RgbaImage::from_pixel(8, 8, image::Rgba([128, 128, 128, 255]));
        let p = plane(0.0, 40.0);
        let chroma = |s: f32| {
            let px = colorized(&src, &p, s).get_pixel(4, 4).0;
            srgb_to_lab(px[0], px[1], px[2]).2
        };
        let (half, full) = (chroma(0.5), chroma(1.0));
        assert!((full - 40.0).abs() < 2.0, "{full}");
        assert!((half - 20.0).abs() < 2.0, "{half}");
    }

    #[test]
    fn colour_is_sampled_where_the_pixel_sits() {
        let n = COLORIZE_INPUT as usize;
        let mut p = plane(0.0, 0.0);
        for y in 0..n {
            for x in n / 2..n {
                p[y * n + x] = 50.0;
            }
        }
        let src = RgbaImage::from_pixel(1000, 10, image::Rgba([120, 120, 120, 255]));
        let out = colorized(&src, &p, 1.0);
        let a_at = |x| {
            let px = out.get_pixel(x, 5).0;
            srgb_to_lab(px[0], px[1], px[2]).1
        };
        assert!(a_at(100).abs() < 1.5, "left half grey");
        assert!((a_at(900) - 50.0).abs() < 3.0, "right half coloured: {}", a_at(900));
    }
}
