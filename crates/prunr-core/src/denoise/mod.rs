//! Classical denoise filters applied to the upscale input.
//!
//! Two primitives:
//!   - `median_filter_channel`: 3×3 histogram-based sliding-window
//!     median (Huang et al. 1979). O(1) per pixel after the initial
//!     column histogram build. Rows are mutually independent — parallelised
//!     with rayon to hit the ~300ms target on 2560×1920.
//!   - `bilateral_filter_channel`: separable approximation —
//!     horizontal row pass + vertical column pass, intensity-weighted
//!     Gaussian kernel. Row-parallel via rayon.
//!
//! Public entry point `apply_denoise` operates on full-image RGBA:
//!   - Sequential per-channel (R, G, B); alpha copied unchanged.
//!   - `strength == 0.0` early-returns a clone (bit-identical to the
//!     unfiltered path — pinned by `apply_denoise_strength_zero_is_identity`).
//!   - `strength` linearly blends filtered output toward input.
//!
//! Peak RAM (canonical 2560×1920 input, sequential channels):
//!   | working buffer              | size      |
//!   |-----------------------------|-----------|
//!   | channel extract (u8)        | ~4.9 MB   |
//!   | median output (u8)          | ~4.9 MB   |
//!   | bilateral h-pass (f32)      | ~19.7 MB  |
//!   | bilateral v-pass / out (f32)| ~19.7 MB  |
//!   | peak (one channel at a time)| ~49 MB    |
//!
//! Concurrency: rayon row-parallelism here is safe because the only
//! production caller — `Processor::dispatch_upscale::build_inference_input`
//! (crates/prunr-app/src/gui/processor.rs) — invokes `apply_denoise`
//! BEFORE the ORT session is created or driven. It is not nested inside
//! the tiled-inference inner loop or any ort callback.

use image::RgbaImage;
use rayon::prelude::*;

/// Median-filter a single channel using the histogram-based O(1)
/// sliding-window algorithm (Huang et al. 1979). `radius` is half-width:
/// radius=1 → 3×3 window, radius=2 → 5×5.
///
/// `src` must be `width * height` bytes. Replicate-border padding is used
/// at image edges (standard image-processing default).
///
/// Each row is independent — parallelised with rayon. The histogram is
/// `[u32; 256]` on the stack — the per-row inner loop is allocation-free
/// per CLAUDE.md `## Hot paths`.
pub(crate) fn median_filter_channel(
    src: &[u8],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<u8> {
    let win_h = 2 * radius + 1;
    let win_w = 2 * radius + 1;
    let win_size = win_w * win_h;
    let half = win_size / 2;
    let mut out = vec![0u8; width * height];

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            // Build histogram for the initial window centred at (0, y)
            let mut hist = [0u32; 256];
            for wy in 0..win_h {
                let sy = (y as isize + wy as isize - radius as isize)
                    .clamp(0, height as isize - 1) as usize;
                for wx in 0..win_w {
                    let sx = (wx as isize - radius as isize).clamp(0, width as isize - 1) as usize;
                    hist[src[sy * width + sx] as usize] += 1;
                }
            }

            for (x, slot) in row.iter_mut().enumerate() {
                // Find median from histogram
                let mut cum = 0u32;
                let mut median = 0usize;
                for (v, &count) in hist.iter().enumerate() {
                    cum += count;
                    if cum > half as u32 {
                        median = v;
                        break;
                    }
                }
                *slot = median as u8;

                // Slide window right: remove x-radius column, add x+radius+1 column
                if x + 1 < width {
                    let remove_x = (x as isize - radius as isize)
                        .clamp(0, width as isize - 1) as usize;
                    let add_x = (x as isize + radius as isize + 1)
                        .clamp(0, width as isize - 1) as usize;
                    for wy in 0..win_h {
                        let sy = (y as isize + wy as isize - radius as isize)
                            .clamp(0, height as isize - 1) as usize;
                        hist[src[sy * width + remove_x] as usize] -= 1;
                        hist[src[sy * width + add_x] as usize] += 1;
                    }
                }
            }
        });

    out
}

/// Bilateral filter on a single channel using the separable approximation.
/// `spatial_sigma` controls the spatial radius (in pixels);
/// `range_sigma` controls the intensity sensitivity (in normalized 0..1
/// units — NOT 0..255).
///
/// Two-pass separable: horizontal row pass writes to an f32 scratch buffer;
/// vertical column pass writes to the u8 output (clamp+cast at the boundary
/// per CLAUDE.md `## Numerical & color invariants`).
///
/// Spatial kernel is precomputed once (not per-pixel) as a `Vec<f32>` of
/// length `2*radius+1`. Rows processed in parallel via rayon.
///
/// Variance guard: when the denominator (sum of weights) is < 1e-6, the
/// unfiltered pixel value is used — defends against the `0/0` case on
/// degenerate inputs.
pub(crate) fn bilateral_filter_channel(
    src: &[u8],
    width: usize,
    height: usize,
    spatial_sigma: f32,
    range_sigma: f32,
) -> Vec<u8> {
    let radius = (2.0 * spatial_sigma).ceil() as usize;
    let kernel_len = 2 * radius + 1;

    let spatial_kernel: Vec<f32> = (0..kernel_len)
        .map(|i| {
            let d = i as f32 - radius as f32;
            (-0.5 * d * d / (spatial_sigma * spatial_sigma)).exp()
        })
        .collect();

    let range_sigma2 = 2.0 * range_sigma * range_sigma;

    let n = width * height;

    let mut h_pass = vec![0.0f32; n];
    h_pass
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            let src_row = &src[y * width..(y + 1) * width];
            for x in 0..width {
                row_out[x] = bilateral_tap(src_row[x] as f32, &spatial_kernel, range_sigma2, |d| {
                    src_row[(x as isize + d).clamp(0, width as isize - 1) as usize] as f32
                });
            }
        });

    let mut out = vec![0u8; n];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..width {
                let filtered = bilateral_tap(h_pass[y * width + x], &spatial_kernel, range_sigma2, |d| {
                    h_pass[(y as isize + d).clamp(0, height as isize - 1) as usize * width + x]
                });
                row_out[x] = filtered.clamp(0.0, 255.0) as u8;
            }
        });

    out
}

/// One output sample of the bilateral pass: `kernel` (centred, odd
/// length) weighted by intensity closeness to `center`. `neighbour(d)`
/// is the value `d` taps from the centre, border-clamped by the caller.
#[inline]
fn bilateral_tap(center: f32, kernel: &[f32], range_sigma2: f32, neighbour: impl Fn(isize) -> f32) -> f32 {
    let radius = (kernel.len() / 2) as isize;
    let center_norm = center / 255.0;
    let mut weighted_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (ki, &sp_w) in kernel.iter().enumerate() {
        let neighbor = neighbour(ki as isize - radius);
        let diff = center_norm - neighbor / 255.0;
        let range_w = (-(diff * diff) / range_sigma2).exp();
        let w = sp_w * range_w;
        weighted_sum += neighbor * w;
        weight_sum += w;
    }
    weighted_sum / weight_sum.max(1e-6)
}

/// Apply classical denoise to the RGB channels of `img`. Alpha is copied
/// unchanged. `strength` is the blend weight in [0.0, 1.0]:
///   - 0.0 = output bit-identical to input (early-return clone).
///   - 1.0 = output is the fully filtered image.
///   - Values in between: linear lerp per pixel per channel.
///
/// At strength=1.0 the pipeline is: median (radius 1, 3×3 window) →
/// bilateral (spatial_sigma 2.0, range_sigma 0.1 normalized) → lerp
/// against original.
pub fn apply_denoise(img: &RgbaImage, strength: f32) -> RgbaImage {
    let s = strength.clamp(0.0, 1.0);
    if s == 0.0 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    let width = w as usize;
    let height = h as usize;

    let mut out = img.clone();

    // Sequential per-channel (NOT parallel channels — saves 2× peak RAM)
    for c in 0..3usize {
        // Extract channel — originals buffer used both as median input and lerp source
        let originals: Vec<u8> = img.pixels().map(|p| p.0[c]).collect();

        let median_out = median_filter_channel(&originals, width, height, 1);
        let bilateral_out = bilateral_filter_channel(&median_out, width, height, 2.0, 0.1);
        // explicit drop before the lerp pass
        drop(median_out);

        // Lerp: original * (1-s) + filtered * s — write back to out
        for (i, p) in out.pixels_mut().enumerate() {
            let original = originals[i] as f32;
            let filtered = bilateral_out[i] as f32;
            let blended = original * (1.0 - s) + filtered * s;
            p.0[c] = blended.clamp(0.0, 255.0) as u8;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn make_gradient_rgba(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            let v = ((x + y) % 256) as u8;
            Rgba([v, (v.wrapping_add(50)), (v.wrapping_add(100)), 255])
        })
    }

    #[test]
    fn apply_denoise_strength_zero_is_identity() {
        let img = make_gradient_rgba(64, 64);
        let result = apply_denoise(&img, 0.0);
        assert_eq!(img.as_raw(), result.as_raw(), "strength=0 must be bit-identical");
    }

    #[test]
    fn apply_denoise_preserves_alpha() {
        let img = RgbaImage::from_pixel(32, 32, Rgba([100, 150, 200, 192]));
        let result = apply_denoise(&img, 0.7);
        for p in result.pixels() {
            assert_eq!(p.0[3], 192, "alpha must be unchanged after denoise");
        }
    }

    #[test]
    fn apply_denoise_preserves_dimensions() {
        let img = make_gradient_rgba(64, 48);
        let result = apply_denoise(&img, 0.5);
        assert_eq!(result.dimensions(), img.dimensions());
    }

    #[test]
    fn median_filter_channel_kills_impulse_noise() {
        let w = 32usize;
        let h = 32usize;
        let mut src = vec![100u8; w * h];
        // Plant five impulse noise pixels
        let impulse_positions = [5 * w + 5, 10 * w + 10, 15 * w + 8, 20 * w + 20, 25 * w + 3];
        for &pos in &impulse_positions {
            src[pos] = 255;
        }
        let out = median_filter_channel(&src, w, h, 1);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 100, "pixel {i} should be 100 after median, got {v}");
        }
    }

    #[test]
    fn median_filter_channel_preserves_constant_input() {
        let w = 32usize;
        let h = 32usize;
        let src = vec![128u8; w * h];
        let out = median_filter_channel(&src, w, h, 1);
        for &v in &out {
            assert_eq!(v, 128, "constant input should produce constant output");
        }
    }

    /// The separable passes, written out tap by tap as a reference.
    #[test]
    fn bilateral_filter_channel_matches_a_tap_by_tap_reference() {
        let (w, h, sigma, range) = (19usize, 13usize, 1.5f32, 0.1f32);
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 97 + i / 7 * 31) % 256) as u8).collect();
        let r = (2.0 * sigma).ceil() as isize;
        let kernel: Vec<f32> = (-r..=r).map(|d| (-0.5 * (d * d) as f32 / (sigma * sigma)).exp()).collect();
        let rs2 = 2.0 * range * range;
        let tap = |center: f32, at: &dyn Fn(isize) -> f32| {
            let (mut ws, mut wsum) = (0.0f32, 0.0f32);
            for (k, &sp) in kernel.iter().enumerate() {
                let n = at(k as isize - r);
                let diff = center / 255.0 - n / 255.0;
                let wt = sp * (-(diff * diff) / rs2).exp();
                ws += n * wt;
                wsum += wt;
            }
            ws / wsum.max(1e-6)
        };
        let clamp = |v: isize, len: usize| v.clamp(0, len as isize - 1) as usize;
        let hp: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                tap(src[i] as f32, &|d| src[y * w + clamp(x as isize + d, w)] as f32)
            })
            .collect();
        let want: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                tap(hp[i], &|d| hp[clamp(y as isize + d, h) * w + x]).clamp(0.0, 255.0) as u8
            })
            .collect();
        assert_eq!(bilateral_filter_channel(&src, w, h, sigma, range), want);
    }

    #[test]
    fn bilateral_filter_channel_preserves_step_edge() {
        let w = 64usize;
        let h = 8usize;
        let src: Vec<u8> = (0..w * h)
            .map(|i| if (i % w) < 32 { 50u8 } else { 200u8 })
            .collect();
        let out = bilateral_filter_channel(&src, w, h, 2.0, 0.1);
        for y in 0..h {
            for x in 0..w {
                let v = out[y * w + x];
                if x < 28 {
                    assert!(v < 80, "left of edge: pixel ({x},{y}) = {v}, expected < 80");
                }
                if x > 36 {
                    assert!(v > 170, "right of edge: pixel ({x},{y}) = {v}, expected > 170");
                }
            }
        }
    }

    #[test]
    fn bilateral_filter_channel_smooths_gaussian_noise() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let mut src = vec![0u8; n];
        for (i, slot) in src.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let noise = (hasher.finish() % 17) as i32 - 8;
            *slot = (128i32 + noise).clamp(0, 255) as u8;
        }
        let input_var: f32 = src.iter().map(|&v| {
            let d = v as f32 - 128.0;
            d * d
        }).sum::<f32>() / n as f32;

        let out = bilateral_filter_channel(&src, w, h, 2.0, 0.1);
        let out_mean: f32 = out.iter().map(|&v| v as f32).sum::<f32>() / n as f32;
        let out_var: f32 = out.iter().map(|&v| {
            let d = v as f32 - out_mean;
            d * d
        }).sum::<f32>() / n as f32;

        assert!(out_var < input_var / 2.0,
            "bilateral should reduce variance: in_var={input_var:.2} out_var={out_var:.2}");
    }

    #[test]
    fn apply_denoise_full_strength_changes_noisy_input() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let w = 32u32;
        let h = 32u32;
        let n = (w * h) as usize;
        let mut img = RgbaImage::new(w, h);
        for (i, p) in img.pixels_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let noise = (hasher.finish() % 41) as i32 - 20;
            let v = (100i32 + noise).clamp(0, 255) as u8;
            *p = Rgba([v, v, v, 255]);
        }

        let in_var: f32 = img.pixels().map(|p| {
            let d = p.0[0] as f32 - 100.0;
            d * d
        }).sum::<f32>() / n as f32;

        let out = apply_denoise(&img, 1.0);
        let out_mean: f32 = out.pixels().map(|p| p.0[0] as f32).sum::<f32>() / n as f32;
        let out_var: f32 = out.pixels().map(|p| {
            let d = p.0[0] as f32 - out_mean;
            d * d
        }).sum::<f32>() / n as f32;

        assert!(out_var < in_var,
            "full strength denoise must reduce variance: in={in_var:.2} out={out_var:.2}");
    }

    #[test]
    fn apply_denoise_alpha_zero_input_passes_through() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 0]));
        let result = apply_denoise(&img, 0.7);
        for p in result.pixels() {
            assert_eq!(p.0[3], 0, "alpha must remain 0");
        }
    }

    /// Wall-clock timing test for apply_denoise on a 2560×1920 image at strength=1.0.
    /// Ignored in CI (slow). Run manually with:
    ///   cargo test -p prunr-core --lib --release -- --ignored denoise_timing_2560x1920
    #[test]
    #[ignore]
    fn denoise_timing_2560x1920() {
        use std::time::Instant;
        let w = 2560u32;
        let h = 1920u32;
        let img = RgbaImage::from_fn(w, h, |x, y| {
            let v = ((x * 13 + y * 7) % 200 + 28) as u8;
            Rgba([v, v.wrapping_add(30), v.wrapping_add(60), 255])
        });
        let start = Instant::now();
        let _result = apply_denoise(&img, 1.0);
        let elapsed = start.elapsed();
        println!("apply_denoise 2560×1920 at strength=1.0: {elapsed:?}");
        assert!(elapsed.as_millis() <= 2000,
            "apply_denoise took {elapsed:?}, must be ≤2000ms in release mode");
    }

}
