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

    // Convert guide to grayscale luminance [0,1] and mask to [0,1]
    let mut guide_f = vec![0.0f32; n];
    let mut mask_f = vec![0.0f32; n];

    let wu = w as usize;
    guide_f
        .par_chunks_mut(wu)
        .zip(mask_f.par_chunks_mut(wu))
        .enumerate()
        .for_each(|(y, (grow, mrow))| {
            for x in 0..wu {
                let gp = guide.get_pixel(x as u32, y as u32);
                grow[x] = (0.299 * gp[0] as f32 + 0.587 * gp[1] as f32 + 0.114 * gp[2] as f32)
                    / 255.0;
                mrow[x] = mask.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
            }
        });

    // Prepare element-wise products needed for box_filter calls.
    // guide_f*guide_f and guide_f*mask_f can be computed in parallel.
    let mut ii = vec![0.0f32; n]; // I*I
    let mut ip = vec![0.0f32; n]; // I*p
    ii.par_iter_mut()
        .zip(ip.par_iter_mut())
        .zip(guide_f.par_iter().zip(mask_f.par_iter()))
        .for_each(|((ii_v, ip_v), (&g, &m))| {
            *ii_v = g * g;
            *ip_v = g * m;
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

    // Compute a and b element-wise: a = cov_ip / (var_i + eps), b = mean_p - a * mean_i
    // Reuse ii/ip buffers for a/b to avoid allocation.
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
            let var_i = mii - mi * mi;
            let cov_ip = mip - mi * mp;
            *a = cov_ip / (var_i + epsilon);
            *b = mp - *a * mi;
        });

    // --- Box filter calls 5-6 in parallel ---
    let (mean_a, mean_b) = rayon::join(
        || box_filter(&a_buf, w, h, radius),
        || box_filter(&b_buf, w, h, radius),
    );

    // Output: q = mean_a * I_original + mean_b
    // Recompute luminance from guide to avoid keeping guide_f alive.
    let mut out = GrayImage::new(w, h);
    let out_buf = out.as_mut();
    out_buf
        .par_chunks_mut(wu)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..wu {
                let idx = y * wu + x;
                let gp = guide.get_pixel(x as u32, y as u32);
                let lum = (0.299 * gp[0] as f32 + 0.587 * gp[1] as f32 + 0.114 * gp[2] as f32)
                    / 255.0;
                let val = (mean_a[idx] * lum + mean_b[idx]).clamp(0.0, 1.0);
                row[x] = (val * 255.0) as u8;
            }
        });
    out
}

/// O(1) box filter using integral image (two-pass parallel prefix sums).
///
/// Returns a newly-allocated output buffer.
pub(crate) fn box_filter(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
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
            .enumerate()
            .for_each(|(y, row)| {
                let src_base = y * w;
                let mut acc = 0.0f32;
                for x in 0..w {
                    acc += src[src_base + x];
                    row[x] = acc;
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

    // Pass 2: vertical prefix sums (each column independent)
    let do_par_cols = w >= PAR_PREFIX_THRESHOLD;

    if do_par_cols {
        // SAFETY: each column x accesses indices {x, x+w, x+2w, ...} which are disjoint
        // across different x values. No two parallel iterations touch the same element.
        let integral_ptr = integral.as_mut_ptr();
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        let sp = SendPtr(integral_ptr);

        (0..w).into_par_iter().for_each(|x| {
            for y in 1..h {
                unsafe {
                    let cur = sp.0.add(y * w + x);
                    let prev = sp.0.add((y - 1) * w + x);
                    *cur += *prev;
                }
            }
            let _ = &sp;
        });
    } else {
        for x in 0..w {
            for y in 1..h {
                integral[y * w + x] += integral[(y - 1) * w + x];
            }
        }
    }

    // --- Lookup pass (embarrassingly parallel) ---
    let mut out = vec![0.0f32; n];

    let get = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 {
            return 0.0;
        }
        let x = (x as usize).min(w - 1);
        let y = (y as usize).min(h - 1);
        integral[y * w + x]
    };

    let do_par_lookup = n >= PAR_LOOKUP_THRESHOLD;

    if do_par_lookup {
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let yi = y as i64;
            for x in 0..w {
                let xi = x as i64;
                let x1 = (xi - r - 1).max(-1);
                let y1 = (yi - r - 1).max(-1);
                let x2 = (xi + r).min(w as i64 - 1);
                let y2 = (yi + r).min(h as i64 - 1);
                let area = (x2 - x1) as f32 * (y2 - y1) as f32;
                let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
                row[x] = sum / area.max(1.0);
            }
        });
    } else {
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let x1 = (x - r - 1).max(-1);
                let y1 = (y - r - 1).max(-1);
                let x2 = (x + r).min(w as i64 - 1);
                let y2 = (y + r).min(h as i64 - 1);
                let area = (x2 - x1) as f32 * (y2 - y1) as f32;
                let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
                out[y as usize * w + x as usize] = sum / area.max(1.0);
            }
        }
    }

    out
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
}
