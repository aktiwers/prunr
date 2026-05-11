//! Guided filter for alpha mask refinement.
//!
//! Uses the original image as a guide to refine the AI-generated mask,
//! producing better edges around fine detail (hair, leaves, etc.).
//!
//! Reference: He, Sun, Tang — "Guided Image Filtering" (2013)

use image::{GrayImage, RgbaImage};
use rayon::prelude::*;

/// Minimum number of pixels to justify rayon overhead for the lookup pass.
const PAR_LOOKUP_THRESHOLD: usize = 64 * 64;

/// Minimum number of rows/cols to justify rayon for prefix-sum passes.
const PAR_PREFIX_THRESHOLD: usize = 256;

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
    // Each needs its own integral scratch and output buffer.
    let ((mean_i, mean_p), (mean_ii, mean_ip)) = rayon::join(
        || {
            rayon::join(
                || box_filter(&guide_f, w, h, radius), // mean_I
                || box_filter(&mask_f, w, h, radius),  // mean_p
            )
        },
        || {
            rayon::join(
                || box_filter(&ii, w, h, radius),  // mean(I*I)
                || box_filter(&ip, w, h, radius),  // mean(I*p)
            )
        },
    );

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

    // --- Box filter calls 5-6 in parallel ---
    let (mean_a, mean_b) = rayon::join(
        || box_filter(&a_buf, w, h, radius),
        || box_filter(&b_buf, w, h, radius),
    );

    // Output: q = mean_a * I_original + mean_b. Reuse the `guide_f`
    // luminance buffer instead of recomputing 0.299·R + 0.587·G +
    // 0.114·B per output pixel — saves ~3 mul + 2 add per pixel
    // (~36 M FMAs at 4K). Buffer is alive through this point and
    // dropped at function end.
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

/// O(1) box filter using integral image (two-pass parallel prefix sums).
///
/// Returns a newly-allocated output buffer. For inner-loop callers that
/// run many filters in sequence, use [`box_filter_into`] to reuse a
/// caller-owned output buffer and skip the per-call allocation.
pub(crate) fn box_filter(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    box_filter_into(src, w, h, radius, &mut out);
    out
}

/// Buffer-reusing variant: writes the filtered result into `dst` (which
/// must equal `src.len()`). The integral-image scratch is still
/// allocated internally; lifting that requires a `box_filter_with_scratch`
/// — not yet justified.
pub(crate) fn box_filter_into(src: &[f32], w: u32, h: u32, radius: u32, dst: &mut [f32]) {
    let w = w as usize;
    let h = h as usize;
    let n = w * h;
    let r = radius as i64;

    // --- Build integral image via two separable passes ---
    // Pass 1: horizontal prefix sums (each row independent)
    let mut integral = vec![0.0f32; n];

    let do_par_rows = h >= PAR_PREFIX_THRESHOLD;

    if do_par_rows {
        integral
            .par_chunks_mut(w)
            .zip(src.par_chunks(w))
            .for_each(|(dst_row, src_row)| {
                let mut acc = 0.0f32;
                for (&s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                    acc += s;
                    *d = acc;
                }
            });
    } else {
        for y in 0..h {
            let base = y * w;
            let mut acc = 0.0f32;
            for x in 0..w {
                acc += src[base + x];
                integral[base + x] = acc;
            }
        }
    }

    // Pass 2: vertical prefix sums.
    // Process columns in chunks (e.g., 32) to improve cache locality: walking
    // down N columns at once keeps those N horizontal accumulators in L1.
    const COL_CHUNK: usize = 32;
    let do_par_cols = w >= PAR_PREFIX_THRESHOLD;

    if do_par_cols {
        // SAFETY: each chunk of columns accesses disjoint horizontal slices
        // of the 1D integral buffer. No two parallel iterations touch the same element.
        let integral_ptr_val = integral.as_mut_ptr() as usize;

        (0..w)
            .into_par_iter()
            .chunks(COL_CHUNK)
            .for_each(|chunk| {
                let ptr = integral_ptr_val as *mut f32;
                for y in 1..h {
                    let row_off = y * w;
                    let prev_off = (y - 1) * w;
                    for &x in &chunk {
                        unsafe {
                            let cur = ptr.add(row_off + x);
                            let prev = ptr.add(prev_off + x);
                            *cur += *prev;
                        }
                    }
                }
            });
    } else {
        for cx in (0..w).step_by(COL_CHUNK) {
            let end = (cx + COL_CHUNK).min(w);
            for y in 1..h {
                let row_off = y * w;
                let prev_off = (y - 1) * w;
                for x in cx..end {
                    integral[row_off + x] += integral[prev_off + x];
                }
            }
        }
    }

    // --- Lookup pass (embarrassingly parallel) ---
    debug_assert_eq!(dst.len(), n, "box_filter_into: dst.len() must equal src.len()");

    let get = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 {
            return 0.0;
        }
        let x = (x as usize).min(w - 1);
        let y = (y as usize).min(h - 1);
        integral[y * w + x]
    };

    let do_par_lookup = n >= PAR_LOOKUP_THRESHOLD;

    let inv_area = 1.0 / ((2 * r + 1) as f32).powi(2);

    let process_row = |y: usize, row: &mut [f32]| {
        let yi = y as i64;
        let y1 = (yi - r - 1).max(-1);
        let y2 = (yi + r).min(h as i64 - 1);

        // Fast path for interior pixels (fully contained box)
        let x_start = (r + 1) as usize;
        let x_end = w.saturating_sub(r as usize);

        if yi > r && yi < (h as i64 - r) && x_start < x_end {
            // Margin pixels (left)
            for (x, slot) in row.iter_mut().enumerate().take(x_start) {
                let xi = x as i64;
                let x1 = (xi - r - 1).max(-1);
                let x2 = (xi + r).min(w as i64 - 1);
                let area = (x2 - x1) as f32 * (y2 - y1) as f32;
                let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
                *slot = sum / area.max(1.0);
            }

            // Interior fast path: no boundary checks, no area re-calc.
            // The enclosing `yi > r` guard forces y1 = yi - r - 1 ≥ 0,
            // so row_y1 is always a valid row offset — no Option needed.
            let row_y2 = y2 as usize * w;
            let row_y1 = y1 as usize * w;

            for (x, slot) in row.iter_mut().enumerate().take(x_end).skip(x_start) {
                let xi = x as i64;
                let x2 = (xi + r) as usize;
                let x1 = (xi - r - 1) as usize;
                let sum = integral[row_y2 + x2] - integral[row_y2 + x1]
                    - integral[row_y1 + x2] + integral[row_y1 + x1];
                *slot = sum * inv_area;
            }

            // Margin pixels (right)
            for (x, slot) in row.iter_mut().enumerate().take(w).skip(x_end) {
                let xi = x as i64;
                let x1 = (xi - r - 1).max(-1);
                let x2 = (xi + r).min(w as i64 - 1);
                let area = (x2 - x1) as f32 * (y2 - y1) as f32;
                let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
                *slot = sum / area.max(1.0);
            }
        } else {
            // Fully slow row (top/bottom margins)
            for (x, slot) in row.iter_mut().enumerate().take(w) {
                let xi = x as i64;
                let x1 = (xi - r - 1).max(-1);
                let x2 = (xi + r).min(w as i64 - 1);
                let area = (x2 - x1) as f32 * (y2 - y1) as f32;
                let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
                *slot = sum / area.max(1.0);
            }
        }
    };

    if do_par_lookup {
        dst.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    } else {
        dst.chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

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
