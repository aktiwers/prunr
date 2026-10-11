//! Pure stroke painters for a `MaskArtifact`: circle, square and line
//! stamps with hardness falloff and strength.

use serde::{Deserialize, Serialize};

use crate::math::smoothstep;
use crate::selection::{BrushMode, MaskArtifact, FULL};

const STAMP_SCALE: f32 = FULL as f32;

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
            grid[idx] = combined.clamp(-(FULL as i32), FULL as i32) as i8;
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

/// Thick line from `(x1, y1)` to `(x2, y2)`: the Line tool's one
/// segment, and a Circle stroke between two pointer samples.
pub fn paint_line(
    target: &mut MaskArtifact,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    radius: f32,
    stamp: Stamp,
) {
    along((x1, y1), (x2, y2), radius, stamp, |x, y| paint_circle(target, x, y, radius, stamp));
}

/// Square stamps from `(x1, y1)` to `(x2, y2)`: a Square stroke between
/// two pointer samples.
pub fn paint_square_line(
    target: &mut MaskArtifact,
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    half_size: f32,
    stamp: Stamp,
) {
    along((x1, y1), (x2, y2), half_size, stamp, |x, y| paint_square(target, x, y, half_size, stamp));
}

/// Stamp centres from `a` to `b` half a radius apart. Overlapping stamps
/// keep the stronger cell, so the spacing never darkens a stroke.
fn along(a: (f32, f32), b: (f32, f32), radius: f32, stamp: Stamp, mut paint: impl FnMut(f32, f32)) {
    if radius <= 0.0 || stamp.strength <= 0.0 {
        return;
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    let step = (radius * 0.5).max(0.5);
    let n = ((len / step).ceil() as i32).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        paint(a.0 + dx * t, a.1 + dy * t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(hardness: f32, strength: f32) -> Stamp {
        Stamp { hardness, strength, mode: BrushMode::Add }
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
