//! Guided filter for alpha mask refinement.
//!
//! Uses the original image as a guide to refine the AI-generated mask,
//! producing better edges around fine detail (hair, leaves, etc.).
//!
//! Reference: He, Sun, Tang — "Guided Image Filtering" (2013)

use image::{GrayImage, RgbaImage};
use rayon::prelude::*;

/// Minimum number of pixels to justify rayon overhead in the box filter.
const PAR_THRESHOLD: usize = 64 * 64;

/// Rows per parallel band of the box filter's column pass.
const BAND_ROWS: usize = 64;

/// Refine an alpha mask using a guided filter.
pub fn guided_filter_alpha(
    guide: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
) -> GrayImage {
    let (w, h) = (mask.width(), mask.height());
    let n = (w * h) as usize;

    // Convert guide to grayscale luminance [0,1] and mask to [0,1],
    // and compute element-wise products for box_filter in one pass.
    let mut guide_f = vec![0.0f32; n];
    let mut mask_f = vec![0.0f32; n];
    let mut ii = vec![0.0f32; n]; // I*I
    let mut ip = vec![0.0f32; n]; // I*p

    const INV_255: f32 = 1.0 / 255.0;
    const REC601_R: f32 = 0.299;
    const REC601_G: f32 = 0.587;
    const REC601_B: f32 = 0.114;
    // `guide: &RgbaImage` is part of the signature — 4 bytes per pixel
    // (RGBA). Hardcoded so `par_chunks_exact(4)` lowers to a tight
    // stride without runtime division.
    const RGBA_BYTES_PER_PIXEL: usize = 4;
    debug_assert_eq!(guide.as_raw().len(), n * RGBA_BYTES_PER_PIXEL);

    guide_f.par_iter_mut()
        .zip(mask_f.par_iter_mut())
        .zip(ii.par_iter_mut())
        .zip(ip.par_iter_mut())
        .zip(guide.as_raw().par_chunks_exact(RGBA_BYTES_PER_PIXEL))
        .zip(mask.as_raw().par_iter())
        .for_each(|(((((gf, mf), iiv), ipv), gp), &mp)| {
            let g = (REC601_R * gp[0] as f32 + REC601_G * gp[1] as f32 + REC601_B * gp[2] as f32) * INV_255;
            let m = mp as f32 * INV_255;
            *gf = g;
            *mf = m;
            *iiv = g * g;
            *ipv = g * m;
        });

    // --- Box filter calls 1-4 in parallel ---
    // Four row-sum scratches alive at once is the peak of the whole
    // filter. Do NOT pair them to save planes: measured slower (an open
    // RAM-for-speed trade, see ARCHITECTURE.md).
    let ((mean_i, mean_p), (mean_ii, mean_ip)) = rayon::join(
        || rayon::join(|| box_filter(&guide_f, w, h, radius), || box_filter(&mask_f, w, h, radius)),
        || rayon::join(|| box_filter(&ii, w, h, radius), || box_filter(&ip, w, h, radius)),
    );
    drop(mask_f);

    // Compute a and b element-wise: a = cov_ip / (var_i + eps), b = mean_p - a * mean_i.
    // Reuse ii/ip buffers for a/b to avoid allocation.
    // `var_i.max(0.0)` guards against sub-epsilon negative variance from f32
    // rounding on flat regions — would otherwise feed `/(var+eps)` a value
    // below `eps` and amplify noise on completely uniform input.
    let mut a_buf = ii; // reuse
    let mut b_buf = ip; // reuse
    a_buf
        .par_iter_mut()
        .zip(b_buf.par_iter_mut())
        .zip(
            mean_ii
                .par_iter()
                .zip(mean_ip.par_iter())
                .zip(mean_i.par_iter().zip(mean_p.par_iter())),
        )
        .for_each(|((a, b), ((mii, mip), (mi, mp)))| {
            let var_i = (mii - mi * mi).max(0.0);
            let cov_ip = mip - mi * mp;
            *a = cov_ip / (var_i + epsilon);
            *b = mp - *a * mi;
        });

    drop((mean_ii, mean_ip));

    // --- Box filter calls 5-6 in parallel, into the two means that are
    // no longer needed (saves faulting in two fresh planes) ---
    let (mut mean_a, mut mean_b) = (mean_i, mean_p);
    rayon::join(
        || box_filter_into(&a_buf, w, h, radius, &mut mean_a),
        || box_filter_into(&b_buf, w, h, radius, &mut mean_b),
    );
    drop((a_buf, b_buf));

    // Output: q = mean_a * I_original + mean_b. Reuse the `guide_f`
    // luminance buffer instead of recomputing 0.299·R + 0.587·G +
    // 0.114·B per output pixel — saves ~3 mul + 2 add per pixel
    // (~36 M FMAs at 4K); every other plane is already freed.
    let mut out = GrayImage::new(w, h);
    out.as_mut()
        .par_iter_mut()
        .zip(mean_a.par_iter())
        .zip(mean_b.par_iter())
        .zip(guide_f.par_iter())
        .for_each(|(((slot, &ma), &mb), &gf)| {
            let val = (ma * gf + mb).clamp(0.0, 1.0);
            *slot = (val * 255.0) as u8;
        });
    out
}

/// O(1)-per-pixel box mean, windows clipped at the borders and averaged
/// over the pixels they cover.
///
/// Returns a newly-allocated output buffer. For inner-loop callers that
/// run many filters in sequence, use [`box_filter_into`] to reuse a
/// caller-owned output buffer and skip the per-call allocation.
pub(crate) fn box_filter(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    box_filter_into(src, w, h, radius, &mut out);
    out
}

/// Three box passes of `radius` over `buf`, close to a Gaussian; the
/// result is left in `buf`. `scratch` is grown to `buf`'s length and
/// reused as each pass's output.
pub(crate) fn box_blur3_into(buf: &mut Vec<f32>, w: u32, h: u32, radius: u32, scratch: &mut Vec<f32>) {
    scratch.resize(buf.len(), 0.0);
    for _ in 0..3 {
        box_filter_into(buf, w, h, radius, scratch);
        std::mem::swap(buf, scratch);
    }
}

/// Buffer-reusing variant: writes the filtered result into `dst` (which
/// must equal `src.len()`). One `src`-sized scratch holds the row sums.
///
/// Two sliding-window passes (rows, then columns) with f64 running sums.
/// Do NOT go back to an f32 integral image: its sums grow with the area
/// (~1e11 for squared 0-255 values over a 1500 px region), where f32
/// steps by thousands, and the guided filters' mean(g²) − mean(g)² turned
/// into noise that painted garbage pixels along erase seams.
pub(crate) fn box_filter_into(src: &[f32], w: u32, h: u32, radius: u32, dst: &mut [f32]) {
    let (w, h, r) = (w as usize, h as usize, radius as usize);
    debug_assert_eq!(dst.len(), src.len(), "box_filter_into: dst.len() must equal src.len()");
    if w == 0 || h == 0 {
        return;
    }

    // Pass 1: each pixel's sum over its row window.
    let mut row_sums = vec![0.0f32; w * h];
    let row_pass = |(out, row): (&mut [f32], &[f32])| {
        let mut acc: f64 = row[..(r + 1).min(w)].iter().map(|&v| f64::from(v)).sum();
        for x in 0..w {
            out[x] = acc as f32;
            if x + r + 1 < w {
                acc += f64::from(row[x + r + 1]);
            }
            if x >= r {
                acc -= f64::from(row[x - r]);
            }
        }
    };
    if w * h >= PAR_THRESHOLD {
        row_sums.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(row_pass);
    } else {
        row_sums.chunks_mut(w).zip(src.chunks(w)).for_each(row_pass);
    }

    // Pass 2: slide down the columns of row sums, a band of rows at a
    // time so bands run in parallel; each band starts its running sums
    // afresh, so drift never spans more than one band.
    let band = BAND_ROWS.max(4 * r);
    let column_pass = |(bi, out): (usize, &mut [f32])| {
        let y0 = bi * band;
        let mut acc = vec![0.0f64; w];
        for y in y0.saturating_sub(r)..(y0 + r + 1).min(h) {
            for (a, &v) in acc.iter_mut().zip(&row_sums[y * w..(y + 1) * w]) {
                *a += f64::from(v);
            }
        }
        for (j, out_row) in out.chunks_mut(w).enumerate() {
            let y = y0 + j;
            let rows = ((y + r).min(h - 1) + 1 - y.saturating_sub(r)) as f64;
            for (x, (o, &a)) in out_row.iter_mut().zip(&acc).enumerate() {
                let cols = ((x + r).min(w - 1) + 1 - x.saturating_sub(r)) as f64;
                *o = (a / (rows * cols)) as f32;
            }
            if y + r + 1 < h {
                for (a, &v) in acc.iter_mut().zip(&row_sums[(y + r + 1) * w..(y + r + 2) * w]) {
                    *a += f64::from(v);
                }
            }
            if y >= r {
                for (a, &v) in acc.iter_mut().zip(&row_sums[(y - r) * w..(y - r + 1) * w]) {
                    *a -= f64::from(v);
                }
            }
        }
    };
    if w * h >= PAR_THRESHOLD {
        dst.par_chunks_mut(band * w).enumerate().for_each(column_pass);
    } else {
        dst.chunks_mut(band * w).enumerate().for_each(column_pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    /// A guided filter takes variance as mean(g²) − mean(g)². On a big
    /// region of large values (0-255 squared) that difference must stay
    /// exact to a fraction of a level, or a flat photo reads as a wildly
    /// varying one and the filter paints garbage.
    #[test]
    fn box_means_stay_exact_over_a_large_region_of_large_values() {
        let (w, h) = (1500u32, 1500u32);
        let g: Vec<f32> = (0..w * h).map(|i| 190.0 + ((i % w) * 7 + (i / w) * 13) as f32 % 5.0).collect();
        let gg: Vec<f32> = g.iter().map(|v| v * v).collect();
        let (mean, mean_sq) = (box_filter(&g, w, h, 6), box_filter(&gg, w, h, 6));
        for (i, (m, msq)) in mean.iter().zip(&mean_sq).enumerate() {
            let var = msq - m * m;
            assert!((-0.5..=4.5).contains(&var), "pixel {i}: variance {var}, data spans 0..4");
        }
    }

    /// Windows are clipped at the borders and averaged over the pixels
    /// they cover.
    #[test]
    fn box_means_average_what_the_window_covers() {
        let src: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let out = box_filter(&src, 5, 4, 1);
        let at = |x: usize, y: usize| out[y * 5 + x];
        assert!((at(0, 0) - (0.0 + 1.0 + 5.0 + 6.0) / 4.0).abs() < 1e-5);
        assert!((at(2, 1) - (1.0 + 2.0 + 3.0 + 6.0 + 7.0 + 8.0 + 11.0 + 12.0 + 13.0) / 9.0).abs() < 1e-5);
        assert!((at(4, 3) - (13.0 + 14.0 + 18.0 + 19.0) / 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_guided_filter_preserves_dimensions() {
        let guide = RgbaImage::from_pixel(64, 48, Rgba([128, 128, 128, 255]));
        let mask = GrayImage::from_pixel(64, 48, Luma([200]));
        let result = guided_filter_alpha(&guide, &mask, 4, 0.01);
        assert_eq!(result.width(), 64);
        assert_eq!(result.height(), 48);
    }

    #[test]
    fn test_guided_filter_uniform_mask_unchanged() {
        let guide = RgbaImage::from_pixel(32, 32, Rgba([100, 150, 200, 255]));
        let mask = GrayImage::from_pixel(32, 32, Luma([255]));
        let result = guided_filter_alpha(&guide, &mask, 4, 0.01);
        for p in result.pixels() {
            assert!(p[0] >= 250, "Expected ~255, got {}", p[0]);
        }
    }

    /// Regression: without the `var_i.max(0.0)` guard, f32 rounding on
    /// completely uniform input drives `var_i` and `cov_ip` to sub-epsilon
    /// negative values; the unguarded division amplifies that noise. With
    /// the guard, flat input produces flat output. `epsilon = 1e-4` matches
    /// the lowest production caller (`item_settings.rs` defaults sit in
    /// `[1e-4, 2e-3]`) so the contract holds in band, not just at extreme
    /// amplification.
    #[test]
    fn test_guided_filter_flat_input_produces_flat_output() {
        let guide = RgbaImage::from_pixel(48, 48, Rgba([128, 128, 128, 255]));
        let mask = GrayImage::from_pixel(48, 48, Luma([200]));
        let result = guided_filter_alpha(&guide, &mask, 4, 1e-4);
        let (min, max) = result.pixels().map(|p| p[0])
            .fold((u8::MAX, u8::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        assert!(max - min <= 1, "flat input produced spread {min}..={max}");
    }

    #[test]
    fn box_blur3_is_three_box_passes() {
        let (w, h) = (23u32, 17u32);
        let data: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 251) as f32).collect();
        let want = box_filter(&box_filter(&box_filter(&data, w, h, 2), w, h, 2), w, h, 2);
        let (mut buf, mut scratch) = (data, Vec::new());
        box_blur3_into(&mut buf, w, h, 2, &mut scratch);
        assert_eq!(buf, want);
    }

    #[test]
    fn test_box_filter_uniform_is_identity() {
        let data = vec![0.5f32; 16];
        let out = box_filter(&data, 4, 4, 1);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.01, "Expected ~0.5, got {v}");
        }
    }

    #[test]
    fn test_box_filter_correctness_3x3() {
        // 3x3 image, radius 1 => each interior pixel averages all 9 neighbors
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let out = box_filter(&data, 3, 3, 1);
        // Center pixel (1,1): mean of all 9 = 5.0
        assert!((out[4] - 5.0).abs() < 1e-5, "center={}", out[4]);
        // Corner (0,0): mean of [1,2,4,5] = 3.0
        assert!((out[0] - 3.0).abs() < 1e-5, "corner={}", out[0]);
    }

    #[test]
    fn test_parallel_box_filter_matches_sequential() {
        // Generate a non-trivial image and verify parallel results match
        let w = 200u32;
        let h = 150u32;
        let n = (w * h) as usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin().abs()).collect();
        let result = box_filter(&data, w, h, 5);
        assert_eq!(result.len(), n);
        // Every output should be non-negative (input is non-negative)
        for (i, &v) in result.iter().enumerate() {
            assert!(v >= -1e-6, "Negative value at {i}: {v}");
            assert!(v <= 1.0 + 1e-6, "Value > 1 at {i}: {v}");
        }
    }

    /// Naïve O(N²·(2r+1)²) reference: each pixel = mean of its (2r+1)² window
    /// clamped to image bounds. No integral image, no fast paths, no parallel
    /// tricks — just the definition of a box filter.
    fn brute_force_box(data: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
        let (wi, hi) = (w as i64, h as i64);
        let r = radius as i64;
        let mut out = vec![0.0_f32; data.len()];
        for y in 0..hi {
            for x in 0..wi {
                let mut sum = 0.0_f32;
                let mut count = 0u32;
                for dy in -r..=r {
                    let yy = y + dy;
                    if yy < 0 || yy >= hi { continue; }
                    for dx in -r..=r {
                        let xx = x + dx;
                        if xx < 0 || xx >= wi { continue; }
                        sum += data[(yy * wi + xx) as usize];
                        count += 1;
                    }
                }
                out[(y * wi + x) as usize] = sum / count as f32;
            }
        }
        out
    }

    /// Pins the interior fast-path against brute-force reference on a grid
    /// big enough to exercise all three lookup regions (left margin, interior,
    /// right margin) AND all three row regions (top margin, interior, bottom
    /// margin). 16×16 + r=2 gives x_start=3, x_end=14 — interior is 11×11
    /// pixels of fast path, surrounded by 2-wide margin in every direction.
    #[test]
    fn test_box_filter_interior_fast_path_matches_brute_force() {
        let (w, h, r) = (16u32, 16u32, 2u32);
        let data: Vec<f32> = (0..(w * h) as usize)
            .map(|i| ((i as f32) * 0.137).sin())
            .collect();
        let actual = box_filter(&data, w, h, r);
        let expected = brute_force_box(&data, w, h, r);
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            assert!((a - e).abs() < 1e-4,
                "box_filter mismatch at ({x},{y}): actual={a} expected={e}");
        }
    }

    /// Same brute-force check on a wider image so the column-chunked
    /// vertical prefix sum (COL_CHUNK = 32) walks a full chunk plus a
    /// partial chunk in the same pass.
    #[test]
    fn test_box_filter_chunked_vertical_pass_matches_brute_force() {
        let (w, h, r) = (48u32, 24u32, 3u32);
        let data: Vec<f32> = (0..(w * h) as usize)
            .map(|i| ((i as f32) * 0.071 + (i as f32).sqrt()).cos())
            .collect();
        let actual = box_filter(&data, w, h, r);
        let expected = brute_force_box(&data, w, h, r);
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            assert!((a - e).abs() < 1e-4,
                "box_filter mismatch at ({x},{y}): actual={a} expected={e}");
        }
    }
}
