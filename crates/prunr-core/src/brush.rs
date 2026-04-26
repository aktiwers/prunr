//! Pure mask-correction brush.
//!
//! `MaskCorrection` is a signed magnitude grid the size of the model's
//! output mask. Positive values push toward foreground, negative toward
//! background. `apply_correction` runs in postprocess BEFORE the guided
//! filter so refine still feathers strokes naturally.

use serde::{Deserialize, Serialize};

use crate::math::smoothstep;

/// Range of a single brush stamp's contribution into the i8 grid.
const STAMP_SCALE: f32 = 127.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushMode {
    Add,
    Subtract,
}

impl BrushMode {
    #[inline]
    fn sign(self) -> f32 {
        match self {
            BrushMode::Add => 1.0,
            BrushMode::Subtract => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushShape {
    Circle,
    Square,
    Line,
}

/// Stamp parameters shared by every paint primitive.
#[derive(Clone, Copy, Debug)]
pub struct Stamp {
    /// 0.0 = full smoothstep falloff, 1.0 = hard edges.
    pub hardness: f32,
    /// 0.0 = no effect, 1.0 = full magnitude.
    pub strength: f32,
    pub mode: BrushMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskCorrection {
    pub width: u16,
    pub height: u16,
    /// Direct mutation can violate the `width × height == grid.len()`
    /// invariant. External writers go through `paint_circle` /
    /// `paint_square` / `paint_line` / `merge`.
    pub(crate) grid: Vec<i8>,
}

impl MaskCorrection {
    pub fn empty(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            grid: vec![0; (width as usize) * (height as usize)],
        }
    }

    /// O(n). Caller-controlled — `apply_correction` does NOT short-circuit
    /// on empty (the saturating-add loop is fast enough that a pre-scan
    /// pays for itself only when the correction stays empty across many
    /// dispatches, which is not the brush-session common case).
    pub fn is_empty(&self) -> bool {
        self.grid.iter().all(|&v| v == 0)
    }

    fn dims_match(&self, mask_len: usize) -> bool {
        (self.width as usize) * (self.height as usize) == mask_len
    }

    /// Project the signed-magnitude grid onto a binary `GrayImage` at the
    /// target image dimensions. Any non-zero cell paints 255; zero cells
    /// stay 0. Resamples via nearest-neighbour when grid resolution
    /// differs from the image; the equal-size path runs as a tight
    /// `cells()`-vs-pixel-buffer pair iteration.
    pub fn to_binary_mask(&self, target_w: u32, target_h: u32) -> image::GrayImage {
        let cw = self.width as u32;
        let ch = self.height as u32;
        let mut out = image::GrayImage::new(target_w, target_h);
        if cw == target_w && ch == target_h {
            for (px, &v) in out.as_mut().iter_mut().zip(self.grid.iter()) {
                *px = if v != 0 { 255 } else { 0 };
            }
            return out;
        }
        let cw_us = cw as usize;
        let buf = out.as_mut();
        for y in 0..target_h {
            let gy = ((y as u64 * ch as u64) / target_h as u64) as usize;
            let row_base = gy * cw_us;
            let out_row = (y as usize) * (target_w as usize);
            for x in 0..target_w {
                let gx = ((x as u64 * cw as u64) / target_w as u64) as usize;
                if self.grid[row_base + gx] != 0 {
                    buf[out_row + x as usize] = 255;
                }
            }
        }
        out
    }
}

/// In-place multiplicative correction in normalized [0, 1] mask space.
/// Applied BEFORE gamma/threshold so subsequent gamma slider tweaks
/// modulate the painted regions naturally — a 50% subtract stroke
/// halves the local mask, then a higher gamma further attenuates.
///
/// Semantics per cell:
/// - `g > 0` (add direction):    `m → lerp(m, 1.0, g/127)`
/// - `g < 0` (subtract direction): `m → m * (1 + g/127)` (toward 0)
/// - `g == 0`:                   no-op
///
/// Caller passes the post-normalize, pre-gamma mask in [0, 1].
/// Skips silently on dimension mismatch.
pub fn apply_correction(mask: &mut [f32], correction: &MaskCorrection) {
    if !correction.dims_match(mask.len()) {
        tracing::warn!(
            mask_len = mask.len(),
            expected = (correction.width as usize) * (correction.height as usize),
            "apply_correction: dimension mismatch, skipping"
        );
        return;
    }
    for (m, &g) in mask.iter_mut().zip(correction.grid.iter()) {
        if g == 0 {
            continue;
        }
        let s = (g as f32) / 127.0;
        if s > 0.0 {
            *m += (1.0 - *m) * s;
        } else {
            *m *= 1.0 + s;
        }
    }
}

/// Generic stamp painter. The distance function decides shape:
/// euclidean for `paint_circle`, chebyshev (`max`) for `paint_square`.
///
/// Overlapping stamps keep the strongest magnitude in the active mode's
/// direction — painting twice over the same pixel doesn't double up.
fn stamp_with<D>(
    target: &mut MaskCorrection,
    cx: f32, cy: f32,
    outer: f32,
    stamp: Stamp,
    distance: D,
)
where
    D: Fn(f32, f32) -> f32,
{
    if outer <= 0.0 || stamp.strength <= 0.0 {
        return;
    }
    let w_i = target.width as i32;
    let h_i = target.height as i32;
    let inner = outer * stamp.hardness.clamp(0.0, 1.0);
    let span = (outer - inner).max(1e-6);
    let sign = stamp.mode.sign();
    let strength = stamp.strength.clamp(0.0, 1.0);

    let xmin = ((cx - outer).floor() as i32).max(0);
    let xmax = ((cx + outer).ceil() as i32 + 1).min(w_i);
    let ymin = ((cy - outer).floor() as i32).max(0);
    let ymax = ((cy + outer).ceil() as i32 + 1).min(h_i);
    if xmin >= xmax || ymin >= ymax {
        return;
    }

    let grid = &mut target.grid;
    for y in ymin..ymax {
        for x in xmin..xmax {
            let dx = (x as f32 + 0.5) - cx;
            let dy = (y as f32 + 0.5) - cy;
            let dist = distance(dx, dy);
            if dist > outer {
                continue;
            }
            let intensity = if dist <= inner {
                1.0
            } else {
                smoothstep((outer - dist) / span)
            };
            let value = (intensity * strength * STAMP_SCALE * sign).round() as i32;
            let idx = (y * w_i + x) as usize;
            let prev = grid[idx] as i32;
            let combined = match stamp.mode {
                BrushMode::Add => prev.max(value),
                BrushMode::Subtract => prev.min(value),
            };
            grid[idx] = combined.clamp(-127, 127) as i8;
        }
    }
}

pub fn paint_circle(target: &mut MaskCorrection, cx: f32, cy: f32, radius: f32, stamp: Stamp) {
    stamp_with(target, cx, cy, radius, stamp, |dx, dy| (dx * dx + dy * dy).sqrt());
}

/// Chebyshev-distance variant: `hardness=1` is a sharp square,
/// `hardness=0` softens toward a diamond.
pub fn paint_square(target: &mut MaskCorrection, cx: f32, cy: f32, half_size: f32, stamp: Stamp) {
    stamp_with(target, cx, cy, half_size, stamp, |dx, dy| dx.abs().max(dy.abs()));
}

/// Thick line from `(x1, y1)` to `(x2, y2)`. Caller is responsible
/// for invocation cadence — the Line tool calls this once at
/// `commit_stroke`, not per pointer event.
pub fn paint_line(
    target: &mut MaskCorrection,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    radius: f32,
    stamp: Stamp,
) {
    if radius <= 0.0 || stamp.strength <= 0.0 {
        return;
    }
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    let step = (radius * 0.5).max(0.5);
    let n = ((len / step).ceil() as i32).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        paint_circle(target, x1 + dx * t, y1 + dy * t, radius, stamp);
    }
}

/// Merge `addition` into `target` using max-magnitude in the additive
/// direction — positive stamps from either source keep the strongest
/// foreground push; negative stamps keep the strongest background push.
/// Untouched cells in `addition` (== 0) leave `target` unchanged.
///
/// Skips silently on dimension mismatch.
pub fn merge(target: &mut MaskCorrection, addition: &MaskCorrection) {
    if target.width != addition.width || target.height != addition.height {
        tracing::warn!(
            target_dims = format!("{}x{}", target.width, target.height),
            addition_dims = format!("{}x{}", addition.width, addition.height),
            "merge: dimension mismatch, skipping"
        );
        return;
    }
    for (t, &a) in target.grid.iter_mut().zip(addition.grid.iter()) {
        if a == 0 {
            continue;
        }
        if a > 0 {
            *t = (*t).max(a);
        } else {
            *t = (*t).min(a);
        }
    }
}

/// Stable hash of the correction's content (width, height, grid bytes).
/// Uses `DefaultHasher` (`SipHasher13`) which is deterministic across
/// runs, so persisted hashes survive load.
pub fn content_hash(c: &MaskCorrection) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    c.width.hash(&mut h);
    c.height.hash(&mut h);
    c.grid.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(mask: &[f32], expected: &[f32], tol: f32) -> bool {
        mask.len() == expected.len()
            && mask.iter().zip(expected).all(|(a, b)| (a - b).abs() < tol)
    }

    #[test]
    fn empty_correction_is_no_op() {
        let c = MaskCorrection::empty(10, 10);
        let mut mask = vec![0.5f32; 100];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn to_binary_mask_marks_painted_cells() {
        let mut c = MaskCorrection::empty(64, 64);
        paint_circle(
            &mut c, 32.0, 32.0, 4.0,
            Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Subtract },
        );
        let mask = c.to_binary_mask(64, 64);
        assert_eq!(mask.get_pixel(32, 32).0[0], 255, "centre of stroke must be 255");
        assert_eq!(mask.get_pixel(0, 0).0[0], 0, "untouched corner stays 0");
    }

    #[test]
    fn to_binary_mask_resizes_to_target() {
        let mut c = MaskCorrection::empty(32, 32);
        paint_circle(
            &mut c, 16.0, 16.0, 4.0,
            Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Subtract },
        );
        let mask = c.to_binary_mask(64, 64);
        assert_eq!(mask.dimensions(), (64, 64));
        assert_eq!(mask.get_pixel(32, 32).0[0], 255);
    }

    #[test]
    fn dimension_mismatch_skipped() {
        let c = MaskCorrection::empty(5, 5);
        let mut mask = vec![0.4f32; 100];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 0.4));
    }

    #[test]
    fn full_add_drives_to_one() {
        let mut c = MaskCorrection::empty(2, 2);
        c.grid = vec![127, 127, 127, 127];
        let mut mask = vec![0.3f32; 4];
        apply_correction(&mut mask, &c);
        assert!(approx(&mask, &[1.0, 1.0, 1.0, 1.0], 1e-6));
    }

    #[test]
    fn full_subtract_drives_to_zero() {
        let mut c = MaskCorrection::empty(2, 2);
        c.grid = vec![-127, -127, -127, -127];
        let mut mask = vec![0.95f32; 4];
        apply_correction(&mut mask, &c);
        assert!(approx(&mask, &[0.0, 0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn half_subtract_halves_the_value() {
        // s = -64/127 ≈ -0.504, so m → m * (1 - 0.504) = m * 0.496.
        let mut c = MaskCorrection::empty(2, 1);
        c.grid = vec![-64, -64];
        let mut mask = vec![1.0f32, 0.6];
        apply_correction(&mut mask, &c);
        let expected_factor = 1.0 - 64.0 / 127.0;
        assert!(approx(&mask, &[expected_factor, 0.6 * expected_factor], 1e-5));
    }

    #[test]
    fn half_add_lerps_toward_one() {
        // s = +64/127 ≈ 0.504, so m → m + (1 - m) * 0.504.
        let mut c = MaskCorrection::empty(2, 1);
        c.grid = vec![64, 64];
        let mut mask = vec![0.0f32, 0.5];
        apply_correction(&mut mask, &c);
        let s = 64.0 / 127.0;
        assert!(approx(&mask, &[s, 0.5 + 0.5 * s], 1e-5));
    }

    #[test]
    fn apply_correction_non_uniform_grid() {
        let mut c = MaskCorrection::empty(3, 1);
        c.grid = vec![64, 0, -64];
        let mut mask = vec![0.5f32, 0.5, 0.5];
        apply_correction(&mut mask, &c);
        let s = 64.0 / 127.0;
        assert!(approx(&mask, &[0.5 + 0.5 * s, 0.5, 0.5 * (1.0 - s)], 1e-5));
    }

    #[test]
    fn paint_circle_centered_pixel_only() {
        let mut c = MaskCorrection::empty(5, 5);
        paint_circle(&mut c, 2.5, 2.5, 0.5, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert_eq!(c.grid[12], 127);
        let neighbours = [c.grid[11], c.grid[13], c.grid[7], c.grid[17]];
        assert!(neighbours.iter().all(|&v| v == 0), "only center should be hit, got {:?}", neighbours);
    }

    #[test]
    fn paint_circle_radius_10_covers_disc() {
        let mut c = MaskCorrection::empty(32, 32);
        paint_circle(&mut c, 16.0, 16.0, 10.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert_eq!(c.grid[16 * 32 + 16], 127);
        assert_eq!(c.grid[0], 0);
        let nonzero = c.grid.iter().filter(|&&v| v != 0).count();
        let area = std::f32::consts::PI * 100.0;
        assert!(
            (nonzero as f32 - area).abs() < area * 0.2,
            "covered {} pixels, expected ~{}",
            nonzero,
            area as i32
        );
    }

    #[test]
    fn paint_circle_subtract_writes_negative() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Subtract });
        assert_eq!(c.grid[4 * 8 + 4], -127);
    }

    #[test]
    fn paint_circle_overlapping_keeps_strongest() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        let after_first = c.grid[4 * 8 + 4];
        paint_circle(&mut c, 4.0, 4.0, 2.0, Stamp { hardness: 0.5, strength: 1.0, mode: BrushMode::Add });
        let after_second = c.grid[4 * 8 + 4];
        assert_eq!(after_first, 127);
        assert_eq!(after_second, 127, "second weaker stroke must not lower the strong stamp");
    }

    #[test]
    fn paint_circle_zero_radius_no_op() {
        let mut c = MaskCorrection::empty(4, 4);
        paint_circle(&mut c, 2.0, 2.0, 0.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert!(c.grid.iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_zero_strength_no_op() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 3.0, Stamp { hardness: 1.0, strength: 0.0, mode: BrushMode::Add });
        assert!(c.grid.iter().all(|&v| v == 0), "strength = 0 produces no stamp");
    }

    #[test]
    fn paint_circle_half_strength_halves_stamp() {
        let mut full = MaskCorrection::empty(8, 8);
        paint_circle(&mut full, 4.0, 4.0, 3.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        let mut half = MaskCorrection::empty(8, 8);
        paint_circle(&mut half, 4.0, 4.0, 3.0, Stamp { hardness: 1.0, strength: 0.5, mode: BrushMode::Add });
        let center_full = full.grid[4 * 8 + 4];
        let center_half = half.grid[4 * 8 + 4];
        assert_eq!(center_full, 127, "full strength stamps the maximum");
        // Half-strength halves the magnitude (within rounding).
        assert!(
            (center_half as i32 - 64).abs() <= 1,
            "half strength should land near 64, got {}",
            center_half
        );
    }

    #[test]
    fn paint_circle_outside_bounds_no_panic() {
        let mut c = MaskCorrection::empty(4, 4);
        paint_circle(&mut c, -10.0, -10.0, 5.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        paint_circle(&mut c, 100.0, 100.0, 5.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert!(c.grid.iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_hardness_zero_falls_off_smoothly() {
        let mut c = MaskCorrection::empty(16, 16);
        // Half-pixel offset places the center exactly on pixel (8, 8), so
        // we can compare the perfectly-radial profile.
        paint_circle(&mut c, 8.5, 8.5, 6.0, Stamp { hardness: 0.0, strength: 1.0, mode: BrushMode::Add });
        let center = c.grid[8 * 16 + 8];
        let mid = c.grid[8 * 16 + 11];
        let edge = c.grid[8 * 16 + 13];
        assert_eq!(center, 127, "center should be full-strength at zero distance");
        assert!(
            mid > 0 && mid < 127,
            "mid-radius pixel should be partial, got {}",
            mid
        );
        assert!(
            edge < mid,
            "edge pixel must be weaker than mid under smoothstep (edge={}, mid={})",
            edge, mid
        );
    }

    #[test]
    fn paint_circle_corner_clamps_safely() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_circle(&mut c, 0.5, 0.5, 4.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert_eq!(c.grid[0], 127, "in-frame center pixel is hit");
        let in_disc_corner = c.grid[2 * 8 + 2];
        let outside = c.grid[7 * 8 + 7];
        assert!(in_disc_corner > 0, "pixel inside the truncated disc should be painted");
        assert_eq!(outside, 0, "pixel outside the disc should be untouched");
    }

    #[test]
    fn is_empty_detects_zero_grid() {
        let c = MaskCorrection::empty(8, 8);
        assert!(c.is_empty());
        let mut c2 = c.clone();
        c2.grid[3] = 1;
        assert!(!c2.is_empty());
    }

    #[test]
    fn merge_takes_max_magnitude_per_direction() {
        let mut a = MaskCorrection::empty(2, 1);
        a.grid = vec![10, -50];
        let mut b = MaskCorrection::empty(2, 1);
        b.grid = vec![80, -20];
        merge(&mut a, &b);
        assert_eq!(a.grid, vec![80, -50]);
    }

    #[test]
    fn merge_skips_zero_cells() {
        let mut a = MaskCorrection::empty(3, 1);
        a.grid = vec![5, -5, 0];
        let b_zero = MaskCorrection::empty(3, 1);
        let snapshot = a.grid.clone();
        merge(&mut a, &b_zero);
        assert_eq!(a.grid, snapshot, "zero addition leaves target intact");
    }

    #[test]
    fn merge_dim_mismatch_is_no_op() {
        let mut a = MaskCorrection::empty(4, 4);
        a.grid[0] = 33;
        let b = MaskCorrection::empty(2, 2);
        merge(&mut a, &b);
        assert_eq!(a.grid[0], 33);
    }

    #[test]
    fn content_hash_changes_on_grid_edit() {
        let mut a = MaskCorrection::empty(8, 8);
        let h1 = content_hash(&a);
        a.grid[5] = 1;
        let h2 = content_hash(&a);
        assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_changes_on_dim_change() {
        let a = MaskCorrection::empty(8, 8);
        let b = MaskCorrection::empty(16, 4);
        assert_ne!(content_hash(&a), content_hash(&b));
    }

    #[test]
    fn content_hash_deterministic_across_calls() {
        let mut a = MaskCorrection::empty(4, 4);
        a.grid[3] = 7;
        a.grid[10] = -7;
        assert_eq!(content_hash(&a), content_hash(&a));
    }

    #[test]
    fn paint_square_fills_chebyshev_disc() {
        let mut c = MaskCorrection::empty(16, 16);
        paint_square(&mut c, 8.0, 8.0, 3.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        // Inside the 6×6 chebyshev ball: full strength.
        assert_eq!(c.grid[8 * 16 + 8], 127);
        assert_eq!(c.grid[6 * 16 + 6], 127);
        assert_eq!(c.grid[10 * 16 + 10], 127);
        // Outside: untouched.
        assert_eq!(c.grid[2 * 16 + 2], 0);
        assert_eq!(c.grid[14 * 16 + 14], 0);
    }

    #[test]
    fn paint_line_zero_length_stamps_one_circle() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_line(&mut c, 4.0, 4.0, 4.0, 4.0, 1.5, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        // Same as a single paint_circle stamp at (4, 4).
        assert!(c.grid[4 * 8 + 4] > 0, "zero-length line still stamps a circle");
    }

    #[test]
    fn paint_line_covers_intermediate_pixels() {
        let mut c = MaskCorrection::empty(32, 32);
        paint_line(&mut c, 4.0, 16.0, 28.0, 16.0, 1.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert!(c.grid[16 * 32 + 16] > 0, "midpoint of a horizontal line must be painted");
        assert_eq!(c.grid[5 * 32 + 16], 0, "well above the line stays untouched");
    }

    #[test]
    fn paint_line_diagonal_has_no_gaps() {
        let mut c = MaskCorrection::empty(32, 32);
        paint_line(&mut c, 0.5, 0.5, 24.5, 24.5, 1.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        for d in 1..=23 {
            assert!(c.grid[(d * 32 + d) as usize] > 0, "diagonal pixel ({d}, {d}) must be painted");
        }
    }

    #[test]
    fn paint_square_hardness_zero_falls_off_smoothly() {
        let mut c = MaskCorrection::empty(16, 16);
        paint_square(&mut c, 8.5, 8.5, 6.0, Stamp { hardness: 0.0, strength: 1.0, mode: BrushMode::Add });
        let center = c.grid[8 * 16 + 8];
        let mid = c.grid[8 * 16 + 11];
        let edge = c.grid[8 * 16 + 13];
        assert_eq!(center, 127, "chebyshev = 0 at center is full strength");
        assert!(mid > 0 && mid < 127, "mid radius is partial under smoothstep, got {mid}");
        assert!(edge < mid, "outer pixel weaker than mid (edge={edge}, mid={mid})");
    }
}
