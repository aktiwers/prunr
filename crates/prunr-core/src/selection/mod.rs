//! Selection-mask artifact + pure action primitives.
//!
//! `MaskArtifact` is the single shared representation written by both
//! Paint Brush and Magic Brush: a signed `i8` plane at source-image
//! resolution, wrapped in `Arc<Vec<i8>>` so undo snapshots and
//! cross-thread reads are refcount bumps, not memcpy.
//!
//! Each cell is a signed coverage in `-CELL_MAX..=CELL_MAX`:
//! - magnitude / `CELL_MAX` = how selected the pixel is. Soft values come
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
//!
//! The layout is bit-compatible with `brush::MaskCorrection`, so a Paint
//! stroke round-trips to the postprocess correction without loss.

use std::sync::Arc;

use crate::brush::{merge_cell, BrushMode, MaskCorrection};

pub mod refine;

/// Magnitude of a fully selected cell.
pub const FULL: i8 = crate::brush::CELL_MAX;

/// Cells at or above this magnitude count as selected for region
/// consumers (≈ 0.5 coverage).
const SELECTED_THRESHOLD: u8 = 64;

#[derive(Debug, Clone)]
pub struct MaskArtifact {
    pub width: u32,
    pub height: u32,
    pub data: Arc<Vec<i8>>,
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
        Self {
            width,
            height,
            data: Arc::new(vec![0; (width as usize) * (height as usize)]),
        }
    }

    /// A committed Paint stroke, cell-for-cell. The correction already
    /// carries sign (mode), hardness falloff and strength.
    pub fn from_correction(correction: MaskCorrection) -> Self {
        Self {
            width: correction.width as u32,
            height: correction.height as u32,
            data: Arc::new(correction.grid),
        }
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
        Ok(Self { width: self.width, height: self.height, data: Arc::new(data) })
    }

    fn map(&self, f: impl Fn(i8) -> i8) -> Self {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Self { width: self.width, height: self.height, data: Arc::new(data) }
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

    fn resample<T: Copy>(&self, out_w: u32, out_h: u32, cell: impl Fn(i8) -> T, out: &mut [T]) {
        if (out_w, out_h) == (self.width, self.height) {
            for (o, &v) in out.iter_mut().zip(self.data.iter()) {
                *o = cell(v);
            }
            return;
        }
        let (sw, sh) = (self.width as u64, self.height as u64);
        let xs: Vec<usize> = (0..out_w as u64).map(|tx| ((tx * sw) / out_w as u64) as usize).collect();
        for ty in 0..out_h as u64 {
            let src_row = &self.data[(((ty * sh) / out_h as u64) * sw) as usize..];
            let row = &mut out[(ty * out_w as u64) as usize..][..out_w as usize];
            for (o, &sx) in row.iter_mut().zip(xs.iter()) {
                *o = cell(src_row[sx]);
            }
        }
    }

    /// Nearest-neighbour resample to model tensor resolution, cell values
    /// carried as-is (sign, hardness falloff and strength intact).
    pub fn to_mask_correction(&self, tensor_w: u16, tensor_h: u16) -> MaskCorrection {
        let mut correction = MaskCorrection::empty(tensor_w, tensor_h);
        self.resample(tensor_w as u32, tensor_h as u32, |v| v, &mut correction.grid);
        correction
    }

    /// The selected region as a binary mask (255 where `is_selected`),
    /// nearest-neighbour resampled to `w × h`. This is the inpaint region —
    /// the same contour the overlay and outline show.
    pub fn region_mask(&self, w: u32, h: u32) -> image::GrayImage {
        let mut out = image::GrayImage::new(w, h);
        self.resample(w, h, |v| if Self::is_selected(v) { 255 } else { 0 }, out.as_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::{paint_circle, Stamp};

    fn mask(w: u32, h: u32, data: Vec<i8>) -> MaskArtifact {
        MaskArtifact { width: w, height: h, data: Arc::new(data) }
    }

    #[test]
    fn empty_mask_has_zero_data_of_correct_length() {
        let m = MaskArtifact::new_empty(1920, 1080);
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.data.len(), 1920 * 1080);
        assert!(m.data.iter().all(|&v| v == 0));
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
        assert_eq!(r.data.as_slice(), &[90, 100, 0, -80]);
    }

    #[test]
    fn add_mask_opposite_sign_lets_newer_stroke_win() {
        // An Add stroke painted over a Subtract region restores it, and
        // vice versa.
        let existing = mask(2, 1, vec![-100, 80]);
        let newer = mask(2, 1, vec![30, -10]);
        let r = existing.add_mask(&newer).unwrap();
        assert_eq!(r.data.as_slice(), &[30, -10]);
    }

    #[test]
    fn add_mask_zero_cells_are_transparent() {
        let a = mask(2, 1, vec![-50, 0]);
        let b = mask(2, 1, vec![0, 70]);
        let r = a.add_mask(&b).unwrap();
        assert_eq!(r.data.as_slice(), &[-50, 70]);
    }

    #[test]
    fn subtract_mask_reduces_magnitude_and_keeps_sign() {
        let a = mask(3, 1, vec![100, -40, 60]);
        let b = mask(3, 1, vec![30, 127, -60]);
        let r = a.subtract_mask(&b).unwrap();
        assert_eq!(r.data.as_slice(), &[70, 0, 0]);
    }

    #[test]
    fn invert_flips_coverage_with_requested_sign() {
        let m = mask(2, 2, vec![0, 64, FULL, -32]);
        assert_eq!(m.invert(BrushMode::Subtract).data.as_slice(), &[-127, -63, 0, -95]);
        assert_eq!(m.invert(BrushMode::Add).data.as_slice(), &[127, 63, 0, 95]);
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

    #[test]
    fn to_mask_correction_downsamples_with_nearest_neighbour() {
        let mut data = vec![0i8; 16];
        for y in 0..2usize {
            for x in 0..2usize {
                data[y * 4 + x] = -90;
            }
        }
        let m = mask(4, 4, data);
        let corr = m.to_mask_correction(2, 2);
        assert_eq!((corr.width, corr.height), (2, 2));
        assert_eq!(corr.grid.as_slice(), &[-90, 0, 0, 0]);
    }

    /// A soft stroke survives the trip stroke → selection → correction
    /// bit-for-bit, so hardness, strength and direction all reach
    /// `apply_correction` unchanged.
    #[test]
    fn paint_stroke_round_trips_through_selection_losslessly() {
        let mut stroke = MaskCorrection::empty(32, 32);
        let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
        paint_circle(&mut stroke, 16.0, 16.0, 10.0, stamp);
        let selection = MaskArtifact::from_correction(stroke.clone());
        assert!(selection.data.iter().any(|&v| v < 0), "subtract stroke must be negative");
        assert!(
            selection.data.iter().any(|&v| v < 0 && v > -FULL),
            "soft falloff must survive as intermediate magnitudes"
        );
        assert_eq!(selection.to_mask_correction(32, 32), stroke);
    }

    /// The inpaint region is the thresholded contour the overlay shows,
    /// not every cell the brush touched: a soft stroke's faint falloff is
    /// excluded from both, consistently.
    #[test]
    fn region_mask_uses_the_same_selected_predicate_as_the_overlay() {
        let mut stroke = MaskCorrection::empty(32, 32);
        let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
        paint_circle(&mut stroke, 16.0, 16.0, 10.0, stamp);
        let selection = MaskArtifact::from_correction(stroke);
        let region = selection.region_mask(32, 32);
        let mut faint_touched = 0;
        for (&cell, px) in selection.data.iter().zip(region.pixels()) {
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
