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

/// The smallest regularization the filter takes; the app's is 1e-4.
const MIN_EPSILON: f32 = 1e-6;

/// Refine an alpha mask using a guided filter.
pub fn guided_filter_alpha(
    guide: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
) -> GrayImage {
    let (w, h) = (mask.width(), mask.height());
    let n = (w * h) as usize;
    // A zero (or NaN) epsilon divides zero by zero on a flat region.
    let epsilon = if epsilon >= MIN_EPSILON { epsilon } else { MIN_EPSILON };

    let mut guide_f = vec![0.0f32; n];
    let mut mask_f = vec![0.0f32; n];
    let mut ii = vec![0.0f32; n]; // I*I
    let mut ip = vec![0.0f32; n]; // I*p

    const INV_255: f32 = 1.0 / 255.0;
    const REC601_R: f32 = 0.299;
    const REC601_G: f32 = 0.587;
    const REC601_B: f32 = 0.114;
    // A constant so `par_chunks_exact` lowers to a fixed stride.
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

    // Four row-sum scratches alive at once is the peak of the whole
    // filter. Do NOT pair them to save planes: measured slower (an open
    // RAM-for-speed trade).
    let ((mean_i, mean_p), (mean_ii, mean_ip)) = rayon::join(
        || rayon::join(|| box_filter(&guide_f, w, h, radius), || box_filter(&mask_f, w, h, radius)),
        || rayon::join(|| box_filter(&ii, w, h, radius), || box_filter(&ip, w, h, radius)),
    );
    drop(mask_f);

    // f32 rounding on flat regions can push var_i below zero, which
    // `/(var + eps)` would amplify into noise: hence `.max(0.0)`.
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

    // Into the two spent means: saves faulting in two fresh planes.
    let (mut mean_a, mut mean_b) = (mean_i, mean_p);
    rayon::join(
        || box_filter_into(&a_buf, w, h, radius, &mut mean_a),
        || box_filter_into(&b_buf, w, h, radius, &mut mean_b),
    );
    drop((a_buf, b_buf));

    // q = mean_a · I + mean_b, reusing the `guide_f` luminance rather
    // than recomputing it per pixel.
    let mut out = GrayImage::new(w, h);
    out.as_mut()
        .par_iter_mut()
        .zip(mean_a.par_iter())
        .zip(mean_b.par_iter())
        .zip(guide_f.par_iter())
        .for_each(|(((slot, &ma), &mb), &gf)| {
            let val = (ma * gf + mb).clamp(0.0, 1.0);
            *slot = crate::math::round_u8(val * 255.0);
        });
    out
}

/// Rows per band of `guided_filter_alpha_colour`.
const COLOUR_BAND_ROWS: usize = 128;

/// `guided_filter_alpha` following the photo's colour, not only its
/// brightness (He et al.'s colour guide: a 3 × 3 covariance per pixel), so
/// an edge between two colours of the same brightness is kept too.
///
/// Worked a band of `COLOUR_BAND_ROWS` rows at a time, each with its
/// windows' margin, so the planes held at once are a band's, not the
/// photo's.
///
/// Peak working set besides the output mask (w·h bytes): about 22 f32
/// planes of w × (`COLOUR_BAND_ROWS` + 4·radius) rows: 84 MB for a 6000 px
/// wide photo at radius 8.
pub fn guided_filter_alpha_colour(guide: &RgbaImage, mask: &GrayImage, radius: u32, epsilon: f32) -> GrayImage {
    colour_guided_in_bands(guide, mask, radius, epsilon, COLOUR_BAND_ROWS)
}

fn colour_guided_in_bands(guide: &RgbaImage, mask: &GrayImage, radius: u32, epsilon: f32, band_rows: usize) -> GrayImage {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let mut out = GrayImage::new(mask.width(), mask.height());
    if w == 0 || h == 0 {
        return out;
    }
    let r = radius as usize;
    let epsilon = f64::from(if epsilon >= MIN_EPSILON { epsilon } else { MIN_EPSILON });
    let band_rows = band_rows.max(1);
    for y0 in (0..h).step_by(band_rows) {
        let y1 = (y0 + band_rows).min(h);
        // A row's coefficients average over r rows each way, and each
        // coefficient's means over r more: the band reads 2r beyond.
        let (in0, in1) = (y0.saturating_sub(2 * r), (y1 + 2 * r).min(h));
        let band = colour_band(guide, mask, in0..in1, radius, epsilon);
        let from = (y0 - in0) * w;
        out.as_mut()[y0 * w..y1 * w].copy_from_slice(&band[from..from + (y1 - y0) * w]);
    }
    out
}

/// The colour-guided filter over `rows` alone, as if they were the whole
/// image: exact for every row at least 2·radius inside them, and at the
/// image's own top and bottom.
fn colour_band(guide: &RgbaImage, mask: &GrayImage, rows: std::ops::Range<usize>, radius: u32, epsilon: f64) -> Vec<u8> {
    const INV_255: f32 = 1.0 / 255.0;
    let w = mask.width() as usize;
    let (bw, bh) = (mask.width(), rows.len() as u32);
    let n = w * rows.len();
    let pixels = &guide.as_raw()[rows.start * w * 4..rows.end * w * 4];
    let channel = |c: usize| -> Vec<f32> { pixels.par_chunks_exact(4).map(|p| f32::from(p[c]) * INV_255).collect() };
    let (gr, gg, gb) = (channel(0), channel(1), channel(2));
    let p: Vec<f32> = mask.as_raw()[rows.start * w..rows.end * w].par_iter().map(|&v| f32::from(v) * INV_255).collect();
    let mut scratch = vec![0.0f32; n];
    let mut mean = |x: &[f32], y: Option<&[f32]>| -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        match y {
            None => box_filter_into(x, bw, bh, radius, &mut out),
            Some(y) => {
                scratch.par_iter_mut().zip(x.par_iter().zip(y)).for_each(|(s, (a, b))| *s = a * b);
                box_filter_into(&scratch, bw, bh, radius, &mut out);
            }
        }
        out
    };
    let (mr, mg, mb, mp) = (mean(&gr, None), mean(&gg, None), mean(&gb, None), mean(&p, None));
    let (mrp, mgp, mbp) = (mean(&gr, Some(&p)), mean(&gg, Some(&p)), mean(&gb, Some(&p)));
    let (mrr, mrg, mrb) = (mean(&gr, Some(&gr)), mean(&gr, Some(&gg)), mean(&gr, Some(&gb)));
    let (mgg, mgb, mbb) = (mean(&gg, Some(&gg)), mean(&gg, Some(&gb)), mean(&gb, Some(&gb)));
    drop(p);
    // a = (Σ + εI)⁻¹ · cov(I, p), b = mean(p) − a · mean(I); in f64, as the
    // cofactors subtract near-equal products on flat regions.
    let coefficients: Vec<[f32; 4]> = (0..n).into_par_iter().map(|i| {
        let (r, g, b, q) = (f64::from(mr[i]), f64::from(mg[i]), f64::from(mb[i]), f64::from(mp[i]));
        let cov = [f64::from(mrp[i]) - r * q, f64::from(mgp[i]) - g * q, f64::from(mbp[i]) - b * q];
        // f32 rounding on flat regions can take a variance below zero.
        let s11 = (f64::from(mrr[i]) - r * r).max(0.0) + epsilon;
        let s22 = (f64::from(mgg[i]) - g * g).max(0.0) + epsilon;
        let s33 = (f64::from(mbb[i]) - b * b).max(0.0) + epsilon;
        let (s12, s13, s23) = (f64::from(mrg[i]) - r * g, f64::from(mrb[i]) - r * b, f64::from(mgb[i]) - g * b);
        let (c11, c12, c13) = (s22 * s33 - s23 * s23, s13 * s23 - s12 * s33, s12 * s23 - s13 * s22);
        let (c22, c23, c33) = (s11 * s33 - s13 * s13, s13 * s12 - s11 * s23, s11 * s22 - s12 * s12);
        let det = s11 * c11 + s12 * c12 + s13 * c13;
        if det <= 0.0 {
            return [0.0, 0.0, 0.0, q as f32];
        }
        let a = [
            (c11 * cov[0] + c12 * cov[1] + c13 * cov[2]) / det,
            (c12 * cov[0] + c22 * cov[1] + c23 * cov[2]) / det,
            (c13 * cov[0] + c23 * cov[1] + c33 * cov[2]) / det,
        ];
        [a[0] as f32, a[1] as f32, a[2] as f32, (q - a[0] * r - a[1] * g - a[2] * b) as f32]
    }).collect();
    drop((mr, mg, mb, mp, mrp, mgp, mbp, mrr, mrg, mrb, mgg, mgb, mbb));
    let part = |k: usize| -> Vec<f32> { coefficients.par_iter().map(|c| c[k]).collect() };
    let (ar, ag, ab, bb) = (part(0), part(1), part(2), part(3));
    drop(coefficients);
    let (ar, ag, ab, bb) = (mean(&ar, None), mean(&ag, None), mean(&ab, None), mean(&bb, None));
    (0..n).into_par_iter().map(|i| {
        let q = ar[i] * gr[i] + ag[i] * gg[i] + ab[i] * gb[i] + bb[i];
        crate::math::round_u8(q.clamp(0.0, 1.0) * 255.0)
    }).collect()
}

/// O(1)-per-pixel box mean, windows clipped at the borders and averaged
/// over the pixels they cover.
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

/// [`box_filter`] into `dst` (same length as `src`). One `src`-sized
/// scratch holds the row sums.
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

    /// An epsilon of zero (a hand-edited preset) on a flat region divides
    /// zero by zero; it filters as the app's small one does, not into NaN
    /// bands.
    #[test]
    fn a_zero_epsilon_on_a_flat_region_filters_as_a_small_one() {
        let guide = RgbaImage::from_pixel(16, 12, Rgba([120, 120, 120, 255]));
        let mask = GrayImage::from_fn(16, 12, |x, _| Luma([if x < 8 { 255 } else { 0 }]));
        let small = guided_filter_alpha(&guide, &mask, 2, 1e-4);
        for eps in [0.0, f32::NAN] {
            let out = guided_filter_alpha(&guide, &mask, 2, eps);
            assert!(out.pixels().zip(small.pixels()).all(|(a, b)| a[0].abs_diff(b[0]) <= 1), "eps {eps}");
        }
    }

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

    /// The colour guide keeps an edge between two colours of the same
    /// brightness, which the brightness guide blurs.
    #[test]
    fn the_colour_guide_keeps_an_edge_of_equal_brightness() {
        let (w, h) = (40u32, 20u32);
        let guide = RgbaImage::from_fn(w, h, |x, _| if x < w / 2 { image::Rgba([200, 50, 50, 255]) } else { image::Rgba([50, 126, 50, 255]) });
        let mask = GrayImage::from_fn(w, h, |x, _| image::Luma([if x < w / 2 { 255 } else { 0 }]));
        let colour = guided_filter_alpha_colour(&guide, &mask, 4, 1e-4);
        let brightness = guided_filter_alpha(&guide, &mask, 4, 1e-4);
        let (inside, outside) = (w / 2 - 3, w / 2 + 2);
        assert!(colour.get_pixel(inside, 10)[0] >= 230 && colour.get_pixel(outside, 10)[0] <= 25, "colour keeps the edge");
        assert!(brightness.get_pixel(outside, 10)[0] > 40, "brightness alone blurs it");
    }

    /// Bands give the same mask as one pass over the whole image: each
    /// band reads the margin its windows need.
    #[test]
    fn colour_bands_match_one_pass() {
        let (w, h) = (37u32, 61u32);
        let guide = RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x ^ y) * 5 % 256) as u8, 255]));
        let mask = GrayImage::from_fn(w, h, |x, y| image::Luma([if (x + y) % 17 < 9 { 230 } else { 20 }]));
        for radius in [1, 3, 8] {
            let whole = colour_guided_in_bands(&guide, &mask, radius, 1e-4, h as usize);
            for band_rows in [1, 5, 16] {
                let banded = colour_guided_in_bands(&guide, &mask, radius, 1e-4, band_rows);
                let worst = whole.as_raw().iter().zip(banded.as_raw()).map(|(a, b)| a.abs_diff(*b)).max();
                assert!(worst <= Some(1), "radius {radius}, bands of {band_rows}: off by {worst:?}");
            }
        }
    }

    /// Against the brute-force reference: a 16×16 grid where windows clip
    /// at every border around an unclipped 12×12 interior, and a wider
    /// than tall one with a larger radius, so rows and columns clip
    /// differently.
    #[test]
    fn test_box_filter_matches_brute_force() {
        let index = |n: u32| (0..n).map(|i| i as f32);
        for (w, h, r, data) in [
            (16u32, 16u32, 2u32, index(16 * 16).map(|i| (i * 0.137).sin()).collect::<Vec<f32>>()),
            (48, 24, 3, index(48 * 24).map(|i| (i * 0.071 + i.sqrt()).cos()).collect()),
        ] {
            let actual = box_filter(&data, w, h, r);
            let expected = brute_force_box(&data, w, h, r);
            for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
                let (x, y) = (i as u32 % w, i as u32 / w);
                assert!((a - e).abs() < 1e-4, "{w}x{h} r{r}: box_filter mismatch at ({x},{y}): actual={a} expected={e}");
            }
        }
    }
}
