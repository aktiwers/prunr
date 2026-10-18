//! Tiny shared math primitives.

use rayon::prelude::*;

/// Cubic Hermite smoothstep on `t ∈ [0, 1]`. Caller clamps the input.
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// The nearest byte to `v` on the 0-255 scale, clamped. A plain `as u8`
/// truncates, darkening every value by half a step on average. Not
/// `f32::round`: without SSE4.1 that is a libm call per channel.
#[inline]
pub fn round_u8(v: f32) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}

/// Logistic sigmoid, logit → probability.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One bilinear tap along an axis: the two source indices and the
/// weight of the second.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tap {
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) t: f32,
}

/// Half-pixel-centred taps from `out_len` output samples onto a
/// `src_len`-long axis, clamped at the ends.
pub(crate) fn bilinear_taps(src_len: u32, out_len: u32) -> Vec<Tap> {
    let last = src_len.max(1) as usize - 1;
    (0..out_len)
        .map(|o| {
            let x = ((o as f32 + 0.5) * src_len as f32 / out_len as f32 - 0.5).clamp(0.0, last as f32);
            let lo = x.floor() as usize;
            Tap { lo, hi: (lo + 1).min(last), t: x - lo as f32 }
        })
        .collect()
}

/// `plane` (rows `stride` long) sampled bilinearly between the taps.
#[inline]
pub(crate) fn bilinear_sample(plane: &[f32], stride: usize, tx: &Tap, ty: &Tap) -> f32 {
    let (r0, r1) = (ty.lo * stride, ty.hi * stride);
    let top = plane[r0 + tx.lo] + (plane[r0 + tx.hi] - plane[r0 + tx.lo]) * tx.t;
    let bot = plane[r1 + tx.lo] + (plane[r1 + tx.hi] - plane[r1 + tx.lo]) * tx.t;
    top + (bot - top) * ty.t
}

/// A whole percent (capped at 100) as a fraction, 0..=1.
#[inline]
pub fn pct(v: u8) -> f32 {
    f32::from(v.min(100)) / 100.0
}

/// Single-pass min / max / mean over a slice.
///
/// Returns `(f32::INFINITY, f32::NEG_INFINITY, 0.0)` for an empty slice —
/// the unchanged sentinel encodes "no data" without forcing callers to
/// branch on `len()` themselves before logging the stats.
#[inline]
pub fn slice_stats(s: &[f32]) -> (f32, f32, f32) {
    let (lo, hi, sum) = if s.len() >= 512 * 512 {
        s.par_iter()
            .fold(
                || (f32::INFINITY, f32::NEG_INFINITY, 0.0f32),
                |(lo, hi, sum), &v| (lo.min(v), hi.max(v), sum + v),
            )
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY, 0.0f32),
                |(l1, h1, s1), (l2, h2, s2)| (l1.min(l2), h1.max(h2), s1 + s2),
            )
    } else {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for &v in s {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
            sum += v;
        }
        (lo, hi, sum)
    };
    let mean = if s.is_empty() { 0.0 } else { sum / s.len() as f32 };
    (lo, hi, mean)
}

/// Distance from every pixel of a `w × h` grid to the nearest `target`
/// pixel: a forward and a backward chamfer pass, √2 on the diagonals so the
/// metric stays near Euclidean. With no target anywhere every distance is
/// `w + h`, longer than any real one.
pub fn distance_to_nearest(target: &[bool], w: usize, h: usize) -> Vec<f32> {
    const DIAG: f32 = std::f32::consts::SQRT_2;
    let far = (w + h) as f32;
    let mut d: Vec<f32> = target.iter().map(|&t| if t { 0.0 } else { far }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0.0 { continue; }
            let mut best = d[i];
            if x > 0 { best = best.min(d[i - 1] + 1.0); }
            if y > 0 { best = best.min(d[i - w] + 1.0); }
            if x > 0 && y > 0 { best = best.min(d[i - w - 1] + DIAG); }
            if x + 1 < w && y > 0 { best = best.min(d[i - w + 1] + DIAG); }
            d[i] = best;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0.0 { continue; }
            let mut best = d[i];
            if x + 1 < w { best = best.min(d[i + 1] + 1.0); }
            if y + 1 < h { best = best.min(d[i + w] + 1.0); }
            if x + 1 < w && y + 1 < h { best = best.min(d[i + w + 1] + DIAG); }
            if x > 0 && y + 1 < h { best = best.min(d[i + w - 1] + DIAG); }
            d[i] = best;
        }
    }
    d
}

/// A buffer's base pointer shared across parallel iterations that each
/// write their own bytes of it. A closure must capture the whole struct
/// (`let _ = &ptr;`): Rust 2021 disjoint capture would otherwise take the
/// bare `*mut u8` field, which is not `Sync`.
#[derive(Clone, Copy)]
pub(crate) struct SendMutPtr(pub(crate) *mut u8);
// SAFETY: every user partitions its writes so no two parallel
// iterations touch the same byte.
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_u8_rounds_and_clamps() {
        assert_eq!([-3.0, 0.49, 0.5, 99.99, 254.6, 300.0, f32::NAN].map(round_u8), [0, 0, 1, 100, 255, 255, 0]);
    }

    #[test]
    fn slice_stats_empty_returns_sentinel() {
        let (lo, hi, mean) = slice_stats(&[]);
        assert_eq!(lo, f32::INFINITY);
        assert_eq!(hi, f32::NEG_INFINITY);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn slice_stats_single_value() {
        let (lo, hi, mean) = slice_stats(&[3.5]);
        assert_eq!(lo, 3.5);
        assert_eq!(hi, 3.5);
        assert_eq!(mean, 3.5);
    }

    #[test]
    fn slice_stats_mixed_signs() {
        let (lo, hi, mean) = slice_stats(&[-2.0, 0.0, 2.0, 4.0]);
        assert_eq!(lo, -2.0);
        assert_eq!(hi, 4.0);
        assert_eq!(mean, 1.0);
    }

    #[test]
    fn slice_stats_all_equal() {
        let (lo, hi, mean) = slice_stats(&[7.0; 5]);
        assert_eq!(lo, 7.0);
        assert_eq!(hi, 7.0);
        assert_eq!(mean, 7.0);
    }

    #[test]
    fn slice_stats_large_parallel() {
        let n = 1024 * 1024;
        let mut data = vec![1.0f32; n];
        data[0] = -5.0;
        data[n - 1] = 10.0;
        // sum = (n-2)*1.0 + (-5.0) + 10.0 = n - 2 + 5 = n + 3
        let (lo, hi, mean) = slice_stats(&data);
        assert_eq!(lo, -5.0);
        assert_eq!(hi, 10.0);
        assert_eq!(mean, (n as f32 + 3.0) / n as f32);
    }

    #[test]
    fn a_percent_is_a_capped_fraction() {
        assert_eq!((pct(0), pct(50), pct(100), pct(255)), (0.0, 0.5, 1.0, 1.0));
    }

    #[test]
    fn a_bilinear_sample_weighs_the_four_neighbours() {
        let plane = [0.0, 1.0, 2.0, 3.0];
        let tap = |lo, hi, t| Tap { lo, hi, t };
        assert_eq!(bilinear_sample(&plane, 2, &tap(0, 1, 0.5), &tap(0, 1, 0.5)), 1.5);
        assert_eq!(bilinear_sample(&plane, 2, &tap(1, 1, 0.0), &tap(0, 0, 0.0)), 1.0);
    }

    #[test]
    fn distance_to_nearest_is_zero_on_targets_and_grows_away() {
        let mut target = vec![false; 25];
        target[12] = true; // centre of 5x5
        let d = distance_to_nearest(&target, 5, 5);
        assert_eq!(d[12], 0.0);
        assert_eq!(d[11], 1.0);
        assert!((d[6] - std::f32::consts::SQRT_2).abs() < 1e-6, "diagonal neighbour");
        assert!(d[0] > d[6], "farther corners are farther");
        assert!(distance_to_nearest(&[false; 4], 2, 2).iter().all(|&v| v == 4.0), "no target: w + h");
    }
}
