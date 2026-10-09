//! Pure mask-correction brush.
//!
//! Strokes paint into a `MaskArtifact` (signed `i8` plane). Positive
//! cells push toward foreground, negative toward background.
//! `apply_correction` runs in postprocess BEFORE the guided filter so
//! refine still feathers strokes naturally.

use serde::{Deserialize, Serialize};

use crate::math::smoothstep;
use crate::selection::MaskArtifact;

/// Magnitude of a fully painted cell; the scale of every i8 mask plane.
pub const CELL_MAX: i8 = 127;
const STAMP_SCALE: f32 = CELL_MAX as f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushMode {
    Add,
    Subtract,
}

impl BrushMode {
    /// `+1` pushes toward subject, `-1` toward background.
    #[inline]
    pub(crate) fn sign(self) -> i8 {
        match self {
            BrushMode::Add => 1,
            BrushMode::Subtract => -1,
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
/// Caller passes the post-normalize, pre-gamma mask in [0, 1] along with
/// its 2D dims. The correction is usually at source resolution while the
/// mask is at tensor resolution; it is nearest-neighbour resampled inline.
pub fn apply_correction(mask: &mut [f32], mask_w: usize, mask_h: usize, correction: &MaskArtifact) {
    debug_assert_eq!(mask.len(), mask_w * mask_h, "apply_correction: mask len != w*h");
    correction.for_each_resampled(mask_w as u32, mask_h as u32, |i, g| apply_one(&mut mask[i], g));
}

#[inline]
fn apply_one(m: &mut f32, g: i8) {
    if g == 0 { return; }
    let s = (g as f32) / STAMP_SCALE;
    if s > 0.0 {
        *m += (1.0 - *m) * s;
    } else {
        *m *= 1.0 + s;
    }
}

/// Generic stamp painter. The distance function decides shape:
/// euclidean for `paint_circle`, chebyshev (`max`) for `paint_square`.
///
/// Overlapping stamps keep the strongest magnitude in the active mode's
/// direction — painting twice over the same pixel doesn't double up.
fn stamp_with<D>(
    target: &mut MaskArtifact,
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
    let sign = f32::from(stamp.mode.sign());
    let strength = stamp.strength.clamp(0.0, 1.0);

    let xmin = ((cx - outer).floor() as i32).max(0);
    let xmax = ((cx + outer).ceil() as i32 + 1).min(w_i);
    let ymin = ((cy - outer).floor() as i32).max(0);
    let ymax = ((cy + outer).ceil() as i32 + 1).min(h_i);
    if xmin >= xmax || ymin >= ymax {
        return;
    }

    let grid = target.cells_mut();
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
            grid[idx] = combined.clamp(-(CELL_MAX as i32), CELL_MAX as i32) as i8;
        }
    }
}

pub fn paint_circle(target: &mut MaskArtifact, cx: f32, cy: f32, radius: f32, stamp: Stamp) {
    stamp_with(target, cx, cy, radius, stamp, |dx, dy| (dx * dx + dy * dy).sqrt());
}

/// Chebyshev-distance variant: `hardness=1` is a sharp square,
/// `hardness=0` softens toward a diamond.
pub fn paint_square(target: &mut MaskArtifact, cx: f32, cy: f32, half_size: f32, stamp: Stamp) {
    stamp_with(target, cx, cy, half_size, stamp, |dx, dy| dx.abs().max(dy.abs()));
}

/// Thick line from `(x1, y1)` to `(x2, y2)`. Caller is responsible
/// for invocation cadence — the Line tool calls this once at
/// `commit_stroke`, not per pointer event.
pub fn paint_line(
    target: &mut MaskArtifact,
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

/// How a newer stroke cell lands on an existing one: zero leaves the
/// existing cell, same sign keeps the stronger magnitude (painting twice
/// does not double up), opposite sign lets the newer stroke win.
#[inline]
pub(crate) fn merge_cell(existing: i8, newer: i8) -> i8 {
    match newer {
        0 => existing,
        n if n > 0 => existing.max(n),
        n => existing.min(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(mask: &[f32], expected: &[f32], tol: f32) -> bool {
        mask.len() == expected.len()
            && mask.iter().zip(expected).all(|(a, b)| (a - b).abs() < tol)
    }

    fn plane(w: u32, h: u32, cells: Vec<i8>) -> MaskArtifact {
        MaskArtifact::from_cells(w, h, cells)
    }

    fn add(hardness: f32, strength: f32) -> Stamp {
        Stamp { hardness, strength, mode: BrushMode::Add }
    }

    #[test]
    fn empty_correction_is_no_op() {
        let c = MaskArtifact::new_empty(10, 10);
        let mut mask = vec![0.5f32; 100];
        apply_correction(&mut mask, 10, 10, &c);
        assert!(mask.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn dimension_mismatch_resamples_inline() {
        // 5×5 correction with one painted cell at (2, 2): nearest-neighbour
        // resample into a 10×10 mask lands paint in a 2×2 block at
        // (4..6, 4..6) instead of being dropped.
        let mut cells = vec![0i8; 25];
        cells[2 * 5 + 2] = 127;
        let c = plane(5, 5, cells);
        let mut mask = vec![0.0f32; 100];
        apply_correction(&mut mask, 10, 10, &c);
        let painted = mask.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(painted, 4);
        assert!(mask.iter().all(|&v| v == 0.0 || (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn full_add_drives_to_one() {
        let c = plane(2, 2, vec![127; 4]);
        let mut mask = vec![0.3f32; 4];
        apply_correction(&mut mask, 2, 2, &c);
        assert!(approx(&mask, &[1.0, 1.0, 1.0, 1.0], 1e-6));
    }

    #[test]
    fn full_subtract_drives_to_zero() {
        let c = plane(2, 2, vec![-127; 4]);
        let mut mask = vec![0.95f32; 4];
        apply_correction(&mut mask, 2, 2, &c);
        assert!(approx(&mask, &[0.0, 0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn half_subtract_halves_the_value() {
        // s = -64/127 ≈ -0.504, so m → m * (1 - 0.504) = m * 0.496.
        let c = plane(2, 1, vec![-64, -64]);
        let mut mask = vec![1.0f32, 0.6];
        apply_correction(&mut mask, 2, 1, &c);
        let expected_factor = 1.0 - 64.0 / 127.0;
        assert!(approx(&mask, &[expected_factor, 0.6 * expected_factor], 1e-5));
    }

    #[test]
    fn half_add_lerps_toward_one() {
        // s = +64/127 ≈ 0.504, so m → m + (1 - m) * 0.504.
        let c = plane(2, 1, vec![64, 64]);
        let mut mask = vec![0.0f32, 0.5];
        apply_correction(&mut mask, 2, 1, &c);
        let s = 64.0 / 127.0;
        assert!(approx(&mask, &[s, 0.5 + 0.5 * s], 1e-5));
    }

    #[test]
    fn apply_correction_non_uniform_grid() {
        let c = plane(3, 1, vec![64, 0, -64]);
        let mut mask = vec![0.5f32, 0.5, 0.5];
        apply_correction(&mut mask, 3, 1, &c);
        let s = 64.0 / 127.0;
        assert!(approx(&mask, &[0.5 + 0.5 * s, 0.5, 0.5 * (1.0 - s)], 1e-5));
    }

    #[test]
    fn paint_circle_centered_pixel_only() {
        let mut c = MaskArtifact::new_empty(5, 5);
        paint_circle(&mut c, 2.5, 2.5, 0.5, add(1.0, 1.0));
        let g = c.cells();
        assert_eq!(g[12], 127);
        let neighbours = [g[11], g[13], g[7], g[17]];
        assert!(neighbours.iter().all(|&v| v == 0), "only center should be hit, got {:?}", neighbours);
    }

    #[test]
    fn paint_circle_radius_10_covers_disc() {
        let mut c = MaskArtifact::new_empty(32, 32);
        paint_circle(&mut c, 16.0, 16.0, 10.0, add(1.0, 1.0));
        let g = c.cells();
        assert_eq!(g[16 * 32 + 16], 127);
        assert_eq!(g[0], 0);
        let nonzero = g.iter().filter(|&&v| v != 0).count();
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
        let mut c = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Subtract });
        assert_eq!(c.cells()[4 * 8 + 4], -127);
    }

    #[test]
    fn paint_circle_overlapping_keeps_strongest() {
        let mut c = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 2.0, add(1.0, 1.0));
        let after_first = c.cells()[4 * 8 + 4];
        paint_circle(&mut c, 4.0, 4.0, 2.0, add(0.5, 1.0));
        let after_second = c.cells()[4 * 8 + 4];
        assert_eq!(after_first, 127);
        assert_eq!(after_second, 127, "second weaker stroke must not lower the strong stamp");
    }

    #[test]
    fn paint_circle_zero_radius_no_op() {
        let mut c = MaskArtifact::new_empty(4, 4);
        paint_circle(&mut c, 2.0, 2.0, 0.0, add(1.0, 1.0));
        assert!(c.cells().iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_zero_strength_no_op() {
        let mut c = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 3.0, add(1.0, 0.0));
        assert!(c.cells().iter().all(|&v| v == 0), "strength = 0 produces no stamp");
    }

    #[test]
    fn paint_circle_half_strength_halves_stamp() {
        let mut full = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut full, 4.0, 4.0, 3.0, add(1.0, 1.0));
        let mut half = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut half, 4.0, 4.0, 3.0, add(1.0, 0.5));
        let center_full = full.cells()[4 * 8 + 4];
        let center_half = half.cells()[4 * 8 + 4];
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
        let mut c = MaskArtifact::new_empty(4, 4);
        paint_circle(&mut c, -10.0, -10.0, 5.0, add(1.0, 1.0));
        paint_circle(&mut c, 100.0, 100.0, 5.0, add(1.0, 1.0));
        assert!(c.cells().iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_hardness_zero_falls_off_smoothly() {
        let mut c = MaskArtifact::new_empty(16, 16);
        // Half-pixel offset places the center exactly on pixel (8, 8), so
        // we can compare the perfectly-radial profile.
        paint_circle(&mut c, 8.5, 8.5, 6.0, add(0.0, 1.0));
        let g = c.cells();
        let center = g[8 * 16 + 8];
        let mid = g[8 * 16 + 11];
        let edge = g[8 * 16 + 13];
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
        let mut c = MaskArtifact::new_empty(8, 8);
        paint_circle(&mut c, 0.5, 0.5, 4.0, add(1.0, 1.0));
        let g = c.cells();
        assert_eq!(g[0], 127, "in-frame center pixel is hit");
        let in_disc_corner = g[2 * 8 + 2];
        let outside = g[7 * 8 + 7];
        assert!(in_disc_corner > 0, "pixel inside the truncated disc should be painted");
        assert_eq!(outside, 0, "pixel outside the disc should be untouched");
    }

    #[test]
    fn paint_square_fills_chebyshev_disc() {
        let mut c = MaskArtifact::new_empty(16, 16);
        paint_square(&mut c, 8.0, 8.0, 3.0, add(1.0, 1.0));
        let g = c.cells();
        // Inside the 6×6 chebyshev ball: full strength.
        assert_eq!(g[8 * 16 + 8], 127);
        assert_eq!(g[6 * 16 + 6], 127);
        assert_eq!(g[10 * 16 + 10], 127);
        // Outside: untouched.
        assert_eq!(g[2 * 16 + 2], 0);
        assert_eq!(g[14 * 16 + 14], 0);
    }

    #[test]
    fn paint_line_zero_length_stamps_one_circle() {
        let mut c = MaskArtifact::new_empty(8, 8);
        paint_line(&mut c, 4.0, 4.0, 4.0, 4.0, 1.5, add(1.0, 1.0));
        // Same as a single paint_circle stamp at (4, 4).
        assert!(c.cells()[4 * 8 + 4] > 0, "zero-length line still stamps a circle");
    }

    #[test]
    fn paint_line_covers_intermediate_pixels() {
        let mut c = MaskArtifact::new_empty(32, 32);
        paint_line(&mut c, 4.0, 16.0, 28.0, 16.0, 1.0, add(1.0, 1.0));
        let g = c.cells();
        assert!(g[16 * 32 + 16] > 0, "midpoint of a horizontal line must be painted");
        assert_eq!(g[5 * 32 + 16], 0, "well above the line stays untouched");
    }

    #[test]
    fn paint_line_diagonal_has_no_gaps() {
        let mut c = MaskArtifact::new_empty(32, 32);
        paint_line(&mut c, 0.5, 0.5, 24.5, 24.5, 1.0, add(1.0, 1.0));
        for d in 1..=23 {
            assert!(c.cells()[(d * 32 + d) as usize] > 0, "diagonal pixel ({d}, {d}) must be painted");
        }
    }

    #[test]
    fn paint_square_hardness_zero_falls_off_smoothly() {
        let mut c = MaskArtifact::new_empty(16, 16);
        paint_square(&mut c, 8.5, 8.5, 6.0, add(0.0, 1.0));
        let g = c.cells();
        let center = g[8 * 16 + 8];
        let mid = g[8 * 16 + 11];
        let edge = g[8 * 16 + 13];
        assert_eq!(center, 127, "chebyshev = 0 at center is full strength");
        assert!(mid > 0 && mid < 127, "mid radius is partial under smoothstep, got {mid}");
        assert!(edge < mid, "outer pixel weaker than mid (edge={edge}, mid={mid})");
    }
}
