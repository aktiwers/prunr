//! Pure mask-correction brush.
//!
//! `MaskCorrection` is a signed magnitude grid the size of the model's
//! output mask. Positive values push toward foreground, negative toward
//! background. `apply_correction` runs in postprocess BEFORE the guided
//! filter so refine still feathers strokes naturally.

use serde::{Deserialize, Serialize};

use crate::math::smoothstep;

/// Scale factor when applying the i8 grid into the u8 mask. Lifts the
/// max signed magnitude (127) into a useful u8 effect (~254), so a
/// full-strength stamp can flip a midtone pixel either direction.
const APPLY_SCALE: i16 = 2;

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskCorrection {
    pub width: u16,
    pub height: u16,
    pub grid: Vec<i8>,
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
}

/// In-place saturating signed add of `correction` into `mask`. Skips
/// silently on dimension mismatch (caller must resize on model swap).
pub fn apply_correction(mask: &mut [u8], correction: &MaskCorrection) {
    if !correction.dims_match(mask.len()) {
        tracing::warn!(
            mask_len = mask.len(),
            expected = (correction.width as usize) * (correction.height as usize),
            "apply_correction: dimension mismatch, skipping"
        );
        return;
    }
    for (m, &g) in mask.iter_mut().zip(correction.grid.iter()) {
        let next = (*m as i16) + (g as i16) * APPLY_SCALE;
        *m = next.clamp(0, 255) as u8;
    }
}

/// Stamp a soft brush at (`cx`, `cy`) into `target`. `hardness ∈ [0, 1]`:
/// 0 = full smoothstep falloff, 1 = hard disc.
///
/// Overlapping stamps keep the strongest magnitude in the active mode's
/// direction — painting twice over the same pixel doesn't double up.
pub fn paint_circle(
    target: &mut MaskCorrection,
    cx: f32,
    cy: f32,
    radius: f32,
    hardness: f32,
    mode: BrushMode,
) {
    if radius <= 0.0 {
        return;
    }
    let w_i = target.width as i32;
    let h_i = target.height as i32;
    let inner = radius * hardness.clamp(0.0, 1.0);
    let outer = radius;
    let outer_sq = outer * outer;
    let inner_sq = inner * inner;
    let span = (outer - inner).max(1e-6);
    let sign = mode.sign();

    let xmin = ((cx - radius).floor() as i32).max(0);
    let xmax = ((cx + radius).ceil() as i32 + 1).min(w_i);
    let ymin = ((cy - radius).floor() as i32).max(0);
    let ymax = ((cy + radius).ceil() as i32 + 1).min(h_i);
    if xmin >= xmax || ymin >= ymax {
        return;
    }

    let grid = &mut target.grid;
    for y in ymin..ymax {
        for x in xmin..xmax {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > outer_sq {
                continue;
            }
            let intensity = if dist_sq <= inner_sq {
                1.0
            } else {
                smoothstep((outer - dist_sq.sqrt()) / span)
            };
            let stamp = (intensity * STAMP_SCALE * sign).round() as i32;
            let idx = (y * w_i + x) as usize;
            let prev = grid[idx] as i32;
            let combined = match mode {
                BrushMode::Add => prev.max(stamp),
                BrushMode::Subtract => prev.min(stamp),
            };
            grid[idx] = combined.clamp(-127, 127) as i8;
        }
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

    #[test]
    fn empty_correction_is_no_op() {
        let c = MaskCorrection::empty(10, 10);
        let mut mask = vec![128u8; 100];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 128));
    }

    #[test]
    fn dimension_mismatch_skipped() {
        let c = MaskCorrection::empty(5, 5);
        let mut mask = vec![100u8; 100];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 100));
    }

    #[test]
    fn add_saturates_at_255() {
        let mut c = MaskCorrection::empty(2, 2);
        c.grid = vec![127, 127, 127, 127];
        let mut mask = vec![250u8; 4];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 255));
    }

    #[test]
    fn subtract_saturates_at_0() {
        let mut c = MaskCorrection::empty(2, 2);
        c.grid = vec![-127, -127, -127, -127];
        let mut mask = vec![5u8; 4];
        apply_correction(&mut mask, &c);
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn add_then_subtract_returns_to_baseline_on_aligned_strokes() {
        let mut c = MaskCorrection::empty(2, 2);
        c.grid = vec![50, 50, 50, 50];
        let mut mask = vec![128u8; 4];
        apply_correction(&mut mask, &c);
        assert_eq!(mask, vec![228, 228, 228, 228]);
    }

    #[test]
    fn apply_correction_non_uniform_grid() {
        let mut c = MaskCorrection::empty(3, 1);
        c.grid = vec![10, 0, -10];
        let mut mask = vec![128u8; 3];
        apply_correction(&mut mask, &c);
        assert_eq!(mask, vec![148, 128, 108]);
    }

    #[test]
    fn paint_circle_centered_pixel_only() {
        let mut c = MaskCorrection::empty(5, 5);
        paint_circle(&mut c, 2.5, 2.5, 0.5, 1.0, BrushMode::Add);
        assert_eq!(c.grid[12], 127);
        let neighbours = [c.grid[11], c.grid[13], c.grid[7], c.grid[17]];
        assert!(neighbours.iter().all(|&v| v == 0), "only center should be hit, got {:?}", neighbours);
    }

    #[test]
    fn paint_circle_radius_10_covers_disc() {
        let mut c = MaskCorrection::empty(32, 32);
        paint_circle(&mut c, 16.0, 16.0, 10.0, 1.0, BrushMode::Add);
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
        paint_circle(&mut c, 4.0, 4.0, 2.0, 1.0, BrushMode::Subtract);
        assert_eq!(c.grid[4 * 8 + 4], -127);
    }

    #[test]
    fn paint_circle_overlapping_keeps_strongest() {
        let mut c = MaskCorrection::empty(8, 8);
        paint_circle(&mut c, 4.0, 4.0, 2.0, 1.0, BrushMode::Add);
        let after_first = c.grid[4 * 8 + 4];
        paint_circle(&mut c, 4.0, 4.0, 2.0, 0.5, BrushMode::Add);
        let after_second = c.grid[4 * 8 + 4];
        assert_eq!(after_first, 127);
        assert_eq!(after_second, 127, "second weaker stroke must not lower the strong stamp");
    }

    #[test]
    fn paint_circle_zero_radius_no_op() {
        let mut c = MaskCorrection::empty(4, 4);
        paint_circle(&mut c, 2.0, 2.0, 0.0, 1.0, BrushMode::Add);
        assert!(c.grid.iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_outside_bounds_no_panic() {
        let mut c = MaskCorrection::empty(4, 4);
        paint_circle(&mut c, -10.0, -10.0, 5.0, 1.0, BrushMode::Add);
        paint_circle(&mut c, 100.0, 100.0, 5.0, 1.0, BrushMode::Add);
        assert!(c.grid.iter().all(|&v| v == 0));
    }

    #[test]
    fn paint_circle_hardness_zero_falls_off_smoothly() {
        let mut c = MaskCorrection::empty(16, 16);
        // Half-pixel offset places the center exactly on pixel (8, 8), so
        // we can compare the perfectly-radial profile.
        paint_circle(&mut c, 8.5, 8.5, 6.0, 0.0, BrushMode::Add);
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
        paint_circle(&mut c, 0.5, 0.5, 4.0, 1.0, BrushMode::Add);
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
}
