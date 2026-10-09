//! Selection-mask artifact + pure action primitives.
//!
//! `MaskArtifact` is the one signed `i8` plane: the Paint Brush paints
//! into it, the Magic Brush decodes into it, the item keeps it as the
//! selection, and postprocess applies it to the segmentation mask. It
//! lives at source-image resolution, wrapped in `Arc<Vec<i8>>` so undo
//! snapshots and cross-thread reads are refcount bumps, not memcpy.
//!
//! Each cell is a signed coverage in `-FULL..=FULL`:
//! - magnitude / `FULL` = how selected the pixel is. Soft values come
//!   from brush hardness and strength; Magic Brush writes full magnitude.
//! - sign = what a segmentation correction does with the pixel:
//!   positive pushes it toward subject (`BrushMode::Add`), negative
//!   toward background (`BrushMode::Subtract`).
//!
//! Consumers fall into two groups, each with one definition:
//! - binary region consumers (outline, fill, inpaint region, bounding
//!   box) use `is_selected`, i.e. at least half coverage;
//! - continuous consumers (Delete / Copy alpha, the segmentation
//!   correction, invert) use the coverage or the signed cell as-is.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod refine;

/// Magnitude of a fully selected cell.
pub const FULL: i8 = 127;

/// Cells at or above this magnitude count as selected for region
/// consumers (≈ 0.5 coverage).
const SELECTED_THRESHOLD: u8 = 64;

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

/// How a newer stroke cell lands on an existing one: zero leaves the
/// existing cell, same sign keeps the stronger magnitude (painting twice
/// does not double up), opposite sign lets the newer stroke win.
#[inline]
fn merge_cell(existing: i8, newer: i8) -> i8 {
    match newer {
        0 => existing,
        n if n > 0 => existing.max(n),
        n => existing.min(n),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaskArtifact {
    pub width: u32,
    pub height: u32,
    /// `width × height` cells; writers go through the brush painters or
    /// `from_cells`, which pin that invariant.
    data: Arc<Vec<i8>>,
    /// False only when every cell is known to be zero. Painting sets it
    /// without looking, so it can be true for a blank plane; the
    /// dispatch path only uses it to skip the correction work.
    painted: bool,
}

/// Returned by add_mask / subtract_mask when dimensions disagree.
#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("mask dimension mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    DimensionMismatch { lhs: (u32, u32), rhs: (u32, u32) },
}

impl MaskArtifact {
    /// All-zero mask at source image resolution.
    pub fn new_empty(width: u32, height: u32) -> Self {
        let cells = vec![0; (width as usize) * (height as usize)];
        Self { width, height, data: Arc::new(cells), painted: false }
    }

    pub fn from_cells(width: u32, height: u32, cells: Vec<i8>) -> Self {
        debug_assert_eq!(cells.len(), (width as usize) * (height as usize), "cell count != width × height");
        let painted = cells.iter().any(|&v| v != 0);
        Self { width, height, data: Arc::new(cells), painted }
    }

    pub fn cells(&self) -> &[i8] {
        &self.data
    }

    /// Mutable cells; copies the plane first only when a snapshot still
    /// shares it, so the mid-stroke buffer paints in place.
    pub(crate) fn cells_mut(&mut self) -> &mut [i8] {
        self.painted = true;
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// True when every cell is zero, so applying the plane changes
    /// nothing. O(1): decided when the plane is built.
    pub fn is_blank(&self) -> bool {
        !self.painted
    }

    #[inline]
    pub fn is_selected(v: i8) -> bool {
        v.unsigned_abs() >= SELECTED_THRESHOLD
    }

    /// Unsigned coverage in `0.0..=1.0`.
    #[inline]
    pub fn coverage(v: i8) -> f32 {
        v.unsigned_abs() as f32 / FULL as f32
    }

    /// True when at least one cell is selected — the gate for region
    /// actions and inpaint dispatch. Distinct from "all cells zero": a
    /// stroke below half coverage still feeds the segmentation correction.
    pub fn has_selected_region(&self) -> bool {
        self.data.iter().any(|&v| Self::is_selected(v))
    }

    fn map_with(&self, other: &Self, f: impl Fn(i8, i8) -> i8) -> Result<Self, SelectionError> {
        if self.width != other.width || self.height != other.height {
            return Err(SelectionError::DimensionMismatch {
                lhs: (self.width, self.height),
                rhs: (other.width, other.height),
            });
        }
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect();
        Ok(Self::from_cells(self.width, self.height, data))
    }

    fn map(&self, f: impl Fn(i8) -> i8) -> Self {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Self::from_cells(self.width, self.height, data)
    }

    /// Union with `other` as the newer stroke or candidate, using the
    /// brush merge rule (`merge_cell`). Shift modifier.
    pub fn add_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        self.map_with(other, merge_cell)
    }

    /// Region difference: `|self| - |other|` clamped at zero, keeping
    /// `self`'s sign. Alt modifier.
    pub fn subtract_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        self.map_with(other, |a, b| {
            let mag = a.unsigned_abs().saturating_sub(b.unsigned_abs()) as i8;
            if a < 0 { -mag } else { mag }
        })
    }

    /// Region inversion: `FULL - |v|`, signed by `mode`.
    pub fn invert(&self, mode: BrushMode) -> Self {
        let sign = mode.sign();
        self.map(|v| sign * (FULL - v.unsigned_abs() as i8))
    }

    /// Stable content hash. Uses DefaultHasher (SipHasher13) — deterministic
    /// across runs so persisted hashes survive load.
    pub fn content_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.data.as_slice().hash(&mut h);
        h.finish()
    }

    fn scale_alpha(&self, rgba: &mut image::RgbaImage, factor: impl Fn(f32) -> f32) {
        // Fully selected and untouched cells are the bulk of any plane;
        // only soft edges pay for the float multiply.
        let (at_zero, at_full) = (factor(0.0), factor(1.0));
        for (pixel, &m) in rgba.pixels_mut().zip(self.data.iter()) {
            let f = match m.unsigned_abs() {
                0 => at_zero,
                v if v >= FULL as u8 => at_full,
                _ => factor(Self::coverage(m)),
            };
            if f >= 1.0 {
                continue;
            }
            pixel.0[3] = if f <= 0.0 { 0 } else { (pixel.0[3] as f32 * f).round() as u8 };
        }
    }

    /// Scales alpha by `1 - coverage` — a fully selected pixel becomes
    /// transparent, soft edges fade. RGB is preserved.
    pub fn alpha_cut(&self, rgba: &mut image::RgbaImage) {
        self.scale_alpha(rgba, |c| 1.0 - c);
    }

    /// Returns a copy of `source` with alpha scaled by coverage — pixels
    /// outside the selection become transparent. Basis of clipboard Copy/Cut.
    pub fn copy_to_rgba(&self, source: &image::RgbaImage) -> image::RgbaImage {
        let mut out = source.clone();
        self.scale_alpha(&mut out, |c| c);
        out
    }

    /// Visit every cell of an `out_w × out_h` nearest-neighbour resample
    /// as `(output index, cell)`, row by row.
    fn for_each_resampled(&self, out_w: u32, out_h: u32, mut f: impl FnMut(usize, i8)) {
        if (out_w, out_h) == (self.width, self.height) {
            for (i, &v) in self.data.iter().enumerate() {
                f(i, v);
            }
            return;
        }
        if self.width == 0 || self.height == 0 {
            return;
        }
        let (sw, sh) = (self.width as u64, self.height as u64);
        let xs: Vec<usize> = (0..out_w as u64).map(|tx| ((tx * sw) / out_w as u64) as usize).collect();
        for ty in 0..out_h as u64 {
            let src_row = &self.data[(((ty * sh) / out_h as u64) * sw) as usize..];
            let row_base = (ty * out_w as u64) as usize;
            for (dx, &sx) in xs.iter().enumerate() {
                f(row_base + dx, src_row[sx]);
            }
        }
    }

    /// The selected region as a binary mask (255 where `is_selected`),
    /// nearest-neighbour resampled to `w × h`. This is the inpaint region —
    /// the same contour the overlay and outline show.
    pub fn region_mask(&self, w: u32, h: u32) -> image::GrayImage {
        let mut out = image::GrayImage::new(w, h);
        let px = out.as_mut();
        self.for_each_resampled(w, h, |i, v| px[i] = if Self::is_selected(v) { 255 } else { 0 });
        out
    }

    /// In-place multiplicative correction of a `mask_w × mask_h` mask in
    /// normalized [0, 1] space, applied BEFORE gamma/threshold so later
    /// gamma tweaks modulate the painted regions naturally — a 50%
    /// subtract stroke halves the local mask, then a higher gamma further
    /// attenuates. The plane is resampled to the mask's size inline.
    ///
    /// Per cell: `g > 0` does `m → lerp(m, 1.0, g/FULL)`, `g < 0` does
    /// `m → m * (1 + g/FULL)`, `g == 0` leaves `m`.
    pub fn apply_to_mask(&self, mask: &mut [f32], mask_w: usize, mask_h: usize) {
        debug_assert_eq!(mask.len(), mask_w * mask_h, "apply_to_mask: mask len != w*h");
        self.for_each_resampled(mask_w as u32, mask_h as u32, |i, g| {
            if g == 0 {
                return;
            }
            let s = (g as f32) / FULL as f32;
            let m = &mut mask[i];
            if s > 0.0 {
                *m += (1.0 - *m) * s;
            } else {
                *m *= 1.0 + s;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::{paint_circle, Stamp};

    fn mask(w: u32, h: u32, data: Vec<i8>) -> MaskArtifact {
        MaskArtifact::from_cells(w, h, data)
    }

    fn approx(mask: &[f32], expected: &[f32], tol: f32) -> bool {
        mask.len() == expected.len()
            && mask.iter().zip(expected).all(|(a, b)| (a - b).abs() < tol)
    }

    #[test]
    fn blank_plane_is_a_no_op_correction() {
        let c = MaskArtifact::new_empty(10, 10);
        assert!(c.is_blank());
        let mut m = vec![0.5f32; 100];
        c.apply_to_mask(&mut m, 10, 10);
        assert!(m.iter().all(|&v| v == 0.5));
        assert!(!mask(2, 1, vec![0, 1]).is_blank());
        assert!(mask(2, 1, vec![0, 0]).is_blank());
    }

    #[test]
    fn apply_resamples_a_smaller_plane_inline() {
        // One painted cell in a 5×5 plane lands on a 2×2 block of a
        // 10×10 mask instead of being dropped.
        let mut cells = vec![0i8; 25];
        cells[2 * 5 + 2] = 127;
        let c = mask(5, 5, cells);
        let mut m = vec![0.0f32; 100];
        c.apply_to_mask(&mut m, 10, 10);
        assert_eq!(m.iter().filter(|&&v| v > 0.0).count(), 4);
        assert!(m.iter().all(|&v| v == 0.0 || (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn apply_full_add_and_subtract_saturate() {
        let mut m = vec![0.3f32; 4];
        mask(2, 2, vec![127; 4]).apply_to_mask(&mut m, 2, 2);
        assert!(approx(&m, &[1.0; 4], 1e-6));
        let mut m = vec![0.95f32; 4];
        mask(2, 2, vec![-127; 4]).apply_to_mask(&mut m, 2, 2);
        assert!(approx(&m, &[0.0; 4], 1e-6));
    }

    #[test]
    fn apply_half_strength_scales_and_lerps() {
        // s = 64/127 ≈ 0.504: subtract does m * (1 - s), add does
        // m + (1 - m) * s, zero leaves m.
        let s = 64.0 / 127.0;
        let mut m = vec![1.0f32, 0.6, 0.5];
        mask(3, 1, vec![-64, -64, 0]).apply_to_mask(&mut m, 3, 1);
        assert!(approx(&m, &[1.0 - s, 0.6 * (1.0 - s), 0.5], 1e-5));
        let mut m = vec![0.0f32, 0.5];
        mask(2, 1, vec![64, 64]).apply_to_mask(&mut m, 2, 1);
        assert!(approx(&m, &[s, 0.5 + 0.5 * s], 1e-5));
    }

    #[test]
    fn empty_mask_has_zero_data_of_correct_length() {
        let m = MaskArtifact::new_empty(1920, 1080);
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.cells().len(), 1920 * 1080);
        assert!(m.cells().iter().all(|&v| v == 0));
        assert!(!m.has_selected_region());
    }

    #[test]
    fn selected_threshold_is_half_coverage() {
        assert!(!MaskArtifact::is_selected(63));
        assert!(MaskArtifact::is_selected(64));
        assert!(MaskArtifact::is_selected(-64));
        assert!(MaskArtifact::is_selected(FULL));
        assert!((MaskArtifact::coverage(-FULL) - 1.0).abs() < 1e-6);
        assert!(!mask(2, 1, vec![40, -63]).has_selected_region());
        assert!(mask(2, 1, vec![0, -64]).has_selected_region());
    }

    #[test]
    fn add_mask_same_sign_keeps_stronger_magnitude() {
        let a = mask(2, 2, vec![30, 100, 0, -40]);
        let b = mask(2, 2, vec![90, 20, 0, -80]);
        let r = a.add_mask(&b).unwrap();
        assert_eq!(r.cells(), &[90, 100, 0, -80]);
    }

    #[test]
    fn add_mask_opposite_sign_lets_newer_stroke_win() {
        // An Add stroke painted over a Subtract region restores it, and
        // vice versa.
        let existing = mask(2, 1, vec![-100, 80]);
        let newer = mask(2, 1, vec![30, -10]);
        let r = existing.add_mask(&newer).unwrap();
        assert_eq!(r.cells(), &[30, -10]);
    }

    #[test]
    fn add_mask_zero_cells_are_transparent() {
        let a = mask(2, 1, vec![-50, 0]);
        let b = mask(2, 1, vec![0, 70]);
        let r = a.add_mask(&b).unwrap();
        assert_eq!(r.cells(), &[-50, 70]);
    }

    #[test]
    fn subtract_mask_reduces_magnitude_and_keeps_sign() {
        let a = mask(3, 1, vec![100, -40, 60]);
        let b = mask(3, 1, vec![30, 127, -60]);
        let r = a.subtract_mask(&b).unwrap();
        assert_eq!(r.cells(), &[70, 0, 0]);
    }

    #[test]
    fn invert_flips_coverage_with_requested_sign() {
        let m = mask(2, 2, vec![0, 64, FULL, -32]);
        assert_eq!(m.invert(BrushMode::Subtract).cells(), &[-127, -63, 0, -95]);
        assert_eq!(m.invert(BrushMode::Add).cells(), &[127, 63, 0, 95]);
    }

    #[test]
    fn content_hash_is_deterministic() {
        let m = mask(4, 4, vec![64; 16]);
        assert_eq!(m.content_hash(), m.content_hash());
    }

    #[test]
    fn content_hash_changes_when_data_changes() {
        let mut data = vec![64i8; 16];
        let m1 = mask(4, 4, data.clone());
        data[7] = -64;
        let m2 = mask(4, 4, data);
        assert_ne!(m1.content_hash(), m2.content_hash());
    }

    #[test]
    fn content_hash_changes_when_dims_change() {
        let m1 = mask(4, 4, vec![0; 16]);
        let m2 = mask(8, 2, vec![0; 16]);
        assert_ne!(m1.content_hash(), m2.content_hash());
    }

    #[test]
    fn add_mask_dimension_mismatch_returns_err() {
        let a = MaskArtifact::new_empty(4, 4);
        let b = MaskArtifact::new_empty(8, 8);
        assert!(matches!(
            a.add_mask(&b),
            Err(SelectionError::DimensionMismatch { lhs: (4, 4), rhs: (8, 8) })
        ));
    }

    #[test]
    fn subtract_mask_dimension_mismatch_returns_err() {
        let a = MaskArtifact::new_empty(4, 4);
        let b = MaskArtifact::new_empty(2, 8);
        assert!(matches!(
            a.subtract_mask(&b),
            Err(SelectionError::DimensionMismatch { lhs: (4, 4), rhs: (2, 8) })
        ));
    }

    #[test]
    fn alpha_cut_zeros_alpha_in_selected_region_regardless_of_sign() {
        let m = mask(2, 2, vec![FULL, -FULL, 0, 0]);
        let mut rgba = image::RgbaImage::from_pixel(2, 2, image::Rgba([200u8, 100, 50, 255]));
        m.alpha_cut(&mut rgba);
        assert_eq!(rgba.get_pixel(0, 0).0, [200, 100, 50, 0]);
        assert_eq!(rgba.get_pixel(1, 0).0[3], 0);
        assert_eq!(rgba.get_pixel(0, 1).0[3], 255);
        assert_eq!(rgba.get_pixel(1, 1).0[3], 255);
    }

    #[test]
    fn alpha_cut_soft_edge_scales_alpha_by_coverage() {
        let m = mask(1, 1, vec![-64]);
        let mut rgba = image::RgbaImage::from_pixel(1, 1, image::Rgba([0u8, 0, 0, 254]));
        m.alpha_cut(&mut rgba);
        // 254 * (1 - 64/127) = 126
        assert_eq!(rgba.get_pixel(0, 0).0[3], 126);
    }

    #[test]
    fn copy_to_rgba_keeps_selected_alpha_and_clears_the_rest() {
        let m = mask(2, 2, vec![FULL, 0, 0, -FULL]);
        let source = image::RgbaImage::from_pixel(2, 2, image::Rgba([10u8, 20, 30, 200]));
        let out = m.copy_to_rgba(&source);
        assert_eq!(out.get_pixel(0, 0).0[3], 200);
        assert_eq!(out.get_pixel(1, 0).0[3], 0);
        assert_eq!(out.get_pixel(0, 1).0[3], 0);
        assert_eq!(out.get_pixel(1, 1).0[3], 200);
    }

    /// A soft stroke keeps hardness, strength and direction in the plane
    /// `apply_correction` reads, cell for cell.
    #[test]
    fn paint_stroke_keeps_sign_and_softness() {
        let mut stroke = MaskArtifact::new_empty(32, 32);
        let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
        paint_circle(&mut stroke, 16.0, 16.0, 10.0, stamp);
        assert!(stroke.cells().iter().any(|&v| v < 0), "subtract stroke must be negative");
        assert!(
            stroke.cells().iter().any(|&v| v < 0 && v > -FULL),
            "soft falloff must survive as intermediate magnitudes"
        );
    }

    /// Painting into a plane an undo snapshot still shares must not
    /// change the snapshot.
    #[test]
    fn painting_a_shared_plane_leaves_the_snapshot_untouched() {
        let mut live = MaskArtifact::new_empty(8, 8);
        let snapshot = live.clone();
        paint_circle(&mut live, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert!(snapshot.cells().iter().all(|&v| v == 0));
        assert!(live.has_selected_region());
    }

    /// The inpaint region is the thresholded contour the overlay shows,
    /// not every cell the brush touched: a soft stroke's faint falloff is
    /// excluded from both, consistently.
    #[test]
    fn region_mask_uses_the_same_selected_predicate_as_the_overlay() {
        let mut selection = MaskArtifact::new_empty(32, 32);
        let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
        paint_circle(&mut selection, 16.0, 16.0, 10.0, stamp);
        let region = selection.region_mask(32, 32);
        let mut faint_touched = 0;
        for (&cell, px) in selection.cells().iter().zip(region.pixels()) {
            assert_eq!(px.0[0] == 255, MaskArtifact::is_selected(cell));
            if cell != 0 && !MaskArtifact::is_selected(cell) {
                faint_touched += 1;
            }
        }
        assert!(faint_touched > 0, "a soft stroke must have cells below the threshold");
    }

    #[test]
    fn region_mask_resamples_to_the_requested_size() {
        let mut data = vec![0i8; 16];
        data[0] = FULL;
        data[1] = FULL;
        data[4] = -FULL;
        data[5] = -FULL;
        let region = mask(4, 4, data).region_mask(2, 2);
        assert_eq!(region.as_raw().as_slice(), &[255, 0, 0, 0]);
    }
}
