//! Selection-mask artifact + pure action primitives.
//!
//! `MaskArtifact` is the single shared representation written by both
//! Paint Brush and Magic Brush: a signed `i8` plane at source-image
//! resolution, wrapped in `Arc<Vec<i8>>` so undo snapshots and
//! cross-thread reads are refcount bumps, not memcpy.
//!
//! Each cell is a signed coverage in `-127..=127`:
//! - magnitude / 127 = how selected the pixel is. Soft values come from
//!   brush hardness and strength; Magic Brush writes full magnitude.
//! - sign = what a segmentation correction does with the pixel:
//!   positive pushes it toward subject (`BrushMode::Add`), negative
//!   toward background (`BrushMode::Subtract`). Region actions
//!   (Delete / Copy / Cut, outline, inpaint region) ignore the sign.
//!
//! The layout is bit-compatible with `brush::MaskCorrection`, so a Paint
//! stroke round-trips to the postprocess correction without loss — the
//! Phase 15 brush feel (soft edges, strength, per-stroke direction) is
//! preserved through the shared selection.

use std::sync::Arc;

use crate::brush::{BrushMode, MaskCorrection};

pub mod refine;

/// Magnitude of a fully selected cell.
pub const FULL: i8 = 127;

/// Cells at or above this magnitude count as selected for region
/// actions and visualization (≈ 0.5 coverage).
pub const SELECTED_THRESHOLD: u8 = 64;

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

#[inline]
fn mode_sign(mode: BrushMode) -> i8 {
    match mode {
        BrushMode::Add => 1,
        BrushMode::Subtract => -1,
    }
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
    pub fn from_correction(correction: &MaskCorrection) -> Self {
        Self {
            width: correction.width as u32,
            height: correction.height as u32,
            data: Arc::new(correction.grid.clone()),
        }
    }

    /// True when the cell counts as selected for region actions.
    #[inline]
    pub fn is_selected(v: i8) -> bool {
        v.unsigned_abs() >= SELECTED_THRESHOLD
    }

    /// Unsigned coverage in `0.0..=1.0`.
    #[inline]
    pub fn coverage(v: i8) -> f32 {
        v.unsigned_abs() as f32 / FULL as f32
    }

    fn check_dims(&self, other: &Self) -> Result<(), SelectionError> {
        if self.width != other.width || self.height != other.height {
            return Err(SelectionError::DimensionMismatch {
                lhs: (self.width, self.height),
                rhs: (other.width, other.height),
            });
        }
        Ok(())
    }

    fn map_with(&self, other: &Self, f: impl Fn(i8, i8) -> i8) -> Result<Self, SelectionError> {
        self.check_dims(other)?;
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect();
        Ok(Self { width: self.width, height: self.height, data: Arc::new(data) })
    }

    /// Union, `other` being the newer stroke or candidate. Same-sign
    /// overlap keeps the stronger magnitude (painting twice does not
    /// double up); opposite-sign overlap lets the newer stroke win, so an
    /// Add stroke over a Subtract region restores it. Shift modifier.
    pub fn add_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        self.map_with(other, |a, b| {
            let same_sign = (a < 0) == (b < 0);
            let keep_existing = b == 0
                || (a != 0 && same_sign && a.unsigned_abs() >= b.unsigned_abs());
            if keep_existing { a } else { b }
        })
    }

    /// Region difference: `|self| - |other|` clamped at zero, keeping
    /// `self`'s sign. Alt modifier.
    pub fn subtract_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        self.map_with(other, |a, b| {
            let mag = a.unsigned_abs().saturating_sub(b.unsigned_abs()) as i8;
            if a < 0 { -mag } else { mag }
        })
    }

    /// Region inversion: `127 - |v|`, signed by `mode`.
    pub fn invert(&self, mode: BrushMode) -> Self {
        let sign = mode_sign(mode);
        self.map(|v| sign * (FULL - v.unsigned_abs() as i8))
    }

    /// Same coverage, re-signed by `mode`. Used for Magic Brush
    /// candidates, which carry no direction of their own.
    pub fn with_mode(&self, mode: BrushMode) -> Self {
        let sign = mode_sign(mode);
        self.map(|v| sign * v.unsigned_abs() as i8)
    }

    fn map(&self, f: impl Fn(i8) -> i8) -> Self {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Self { width: self.width, height: self.height, data: Arc::new(data) }
    }

    /// True when no cell is selected.
    pub fn is_empty(&self) -> bool {
        !self.data.iter().any(|&v| Self::is_selected(v))
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

    /// Bounding box `(x0, y0, x1, y1)`, inclusive, of the selected cells.
    /// `None` when nothing is selected.
    pub fn selected_bbox(&self) -> Option<(u32, u32, u32, u32)> {
        let w = self.width as usize;
        let mut bbox: Option<(u32, u32, u32, u32)> = None;
        for (i, &v) in self.data.iter().enumerate() {
            if !Self::is_selected(v) {
                continue;
            }
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            bbox = Some(match bbox {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
        bbox
    }

    /// Scales alpha by `1 - coverage` — a fully selected pixel becomes
    /// transparent, soft edges fade. RGB is preserved.
    pub fn alpha_cut(&self, rgba: &mut image::RgbaImage) {
        let buf = rgba.as_flat_samples_mut().samples;
        for (i, &m) in self.data.iter().enumerate() {
            if m != 0 {
                let a = &mut buf[i * 4 + 3];
                *a = (*a as f32 * (1.0 - Self::coverage(m))).round() as u8;
            }
        }
    }

    /// Returns a copy of `source` with alpha scaled by coverage — pixels
    /// outside the selection become transparent. Basis of clipboard Copy/Cut.
    pub fn copy_to_rgba(&self, source: &image::RgbaImage) -> image::RgbaImage {
        let mut out = source.clone();
        let buf = out.as_flat_samples_mut().samples;
        for (i, &m) in self.data.iter().enumerate() {
            let a = &mut buf[i * 4 + 3];
            *a = (*a as f32 * Self::coverage(m)).round() as u8;
        }
        out
    }

    /// Nearest-neighbour resample to model tensor resolution, cell values
    /// carried as-is (sign, hardness falloff and strength intact).
    pub fn to_mask_correction(&self, tensor_w: u16, tensor_h: u16) -> MaskCorrection {
        let tw = tensor_w as u32;
        let th = tensor_h as u32;
        let sw = self.width;
        let sh = self.height;
        let mut correction = MaskCorrection::empty(tensor_w, tensor_h);
        for ty in 0..th {
            let sy = ((ty as u64 * sh as u64) / th as u64) as u32;
            let row_base = (ty * tw) as usize;
            let src_row = (sy * sw) as usize;
            for tx in 0..tw {
                let sx = ((tx as u64 * sw as u64) / tw as u64) as u32;
                correction.grid[row_base + tx as usize] = self.data[src_row + sx as usize];
            }
        }
        correction
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
        assert!(m.is_empty());
    }

    #[test]
    fn selected_threshold_is_half_coverage() {
        assert!(!MaskArtifact::is_selected(63));
        assert!(MaskArtifact::is_selected(64));
        assert!(MaskArtifact::is_selected(-64));
        assert!(MaskArtifact::is_selected(FULL));
        assert!((MaskArtifact::coverage(-FULL) - 1.0).abs() < 1e-6);
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
        // An Add stroke (positive) painted over a Subtract region restores it,
        // and vice versa — matches MaskCorrection::merge from Phase 15.
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
        let inv = m.invert(BrushMode::Subtract);
        assert_eq!(inv.data.as_slice(), &[-127, -63, 0, -95]);
        let inv = m.invert(BrushMode::Add);
        assert_eq!(inv.data.as_slice(), &[127, 63, 0, 95]);
    }

    #[test]
    fn with_mode_resigns_without_changing_coverage() {
        let m = mask(3, 1, vec![127, -50, 0]);
        assert_eq!(m.with_mode(BrushMode::Subtract).data.as_slice(), &[-127, -50, 0]);
        assert_eq!(m.with_mode(BrushMode::Add).data.as_slice(), &[127, 50, 0]);
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
    fn selected_bbox_covers_selected_cells_only() {
        let mut data = vec![0i8; 16];
        data[5] = -FULL; // (1, 1)
        data[11] = FULL; // (3, 2)
        data[12] = 10; // (0, 3): below threshold — ignored
        let m = mask(4, 4, data);
        assert_eq!(m.selected_bbox(), Some((1, 1, 3, 2)));
        assert_eq!(MaskArtifact::new_empty(4, 4).selected_bbox(), None);
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

    /// The Phase 15 contract: a soft Subtract stroke survives the trip
    /// stroke → selection → correction bit-for-bit, so hardness, strength
    /// and direction all reach `apply_correction` unchanged.
    #[test]
    fn paint_stroke_round_trips_through_selection_losslessly() {
        let mut stroke = MaskCorrection::empty(32, 32);
        let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
        paint_circle(&mut stroke, 16.0, 16.0, 10.0, stamp);
        let selection = MaskArtifact::from_correction(&stroke);
        assert!(selection.data.iter().any(|&v| v < 0), "subtract stroke must be negative");
        assert!(
            selection.data.iter().any(|&v| v < 0 && v > -FULL),
            "soft falloff must survive as intermediate magnitudes"
        );
        let back = selection.to_mask_correction(32, 32);
        assert_eq!(back.grid, stroke.grid);
    }
}
