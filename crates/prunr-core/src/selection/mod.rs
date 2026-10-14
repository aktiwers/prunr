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

use std::sync::{Arc, OnceLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod refine;

/// An inclusive rectangle of cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl CellRect {
    pub fn width(self) -> u32 {
        self.x1 - self.x0 + 1
    }

    pub fn height(self) -> u32 {
        self.y1 - self.y0 + 1
    }

    pub fn union(self, o: Self) -> Self {
        Self { x0: self.x0.min(o.x0), y0: self.y0.min(o.y0), x1: self.x1.max(o.x1), y1: self.y1.max(o.y1) }
    }

    /// Grown by `margin` on every side, clamped to a `w × h` plane.
    pub fn grown(self, margin: u32, w: u32, h: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: (self.x1 + margin).min(w - 1),
            y1: (self.y1 + margin).min(h - 1),
        }
    }
}

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

/// Cells per parallel hash chunk; part of the hash format.
const HASH_CHUNK: usize = 1 << 20;

/// `a` with a correction cell `g` applied: `g > 0` lifts it toward
/// opaque by `g / FULL`, `g < 0` scales it toward clear, 0 leaves it.
#[inline]
fn corrected_alpha(a: u8, g: i8) -> u8 {
    if g == 0 {
        return a;
    }
    let s = g as f32 / FULL as f32;
    let a = a as f32 / 255.0;
    let out = if s > 0.0 { a + (1.0 - a) * s } else { a * (1.0 + s) };
    (out * 255.0).round().clamp(0.0, 255.0) as u8
}

fn scan_flags(cells: &[i8]) -> (bool, bool) {
    let painted = cells.iter().any(|&v| v != 0);
    (painted, painted && cells.iter().any(|&v| MaskArtifact::is_selected(v)))
}

/// How a newer stroke cell lands on an existing one: zero leaves the
/// existing cell, same sign keeps the stronger magnitude (painting twice
/// does not double up), opposite sign moves it toward the newer stroke by
/// that stroke's strength, as paint would: a full stroke flips it, a
/// soft edge only nudges it.
#[inline]
fn merge_cell(existing: i8, newer: i8) -> i8 {
    match newer {
        0 => existing,
        n if (n > 0) == (existing > 0) || existing == 0 => if n > 0 { existing.max(n) } else { existing.min(n) },
        n => {
            let weight = f32::from(n.unsigned_abs()) / f32::from(FULL as u8);
            let moved = f32::from(existing) + (f32::from(n) - f32::from(existing)) * weight;
            moved.round().clamp(-f32::from(FULL as u8), f32::from(FULL as u8)) as i8
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskArtifact {
    pub width: u32,
    pub height: u32,
    /// `width × height` cells; writers go through the brush painters or
    /// `from_cells`, which pin that invariant.
    data: Arc<Vec<i8>>,
    /// `(any cell non-zero, any cell selected)`, computed once per
    /// plane: at build, or on first query after painting (painting
    /// resets it). Always exact, O(1) after the first read.
    flags: OnceLock<(bool, bool)>,
}

/// Equal by size and cells; the flags are derived from the cells and may
/// not have been computed yet.
impl PartialEq for MaskArtifact {
    fn eq(&self, other: &Self) -> bool {
        (self.width, self.height) == (other.width, other.height) && self.data == other.data
    }
}

/// Returned by add_mask / merge_stroke when dimensions disagree.
#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("mask dimension mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    DimensionMismatch { lhs: (u32, u32), rhs: (u32, u32) },
}

impl MaskArtifact {
    /// All-zero mask at source image resolution.
    pub fn new_empty(width: u32, height: u32) -> Self {
        let cells = vec![0; (width as usize) * (height as usize)];
        Self { width, height, data: Arc::new(cells), flags: OnceLock::from((false, false)) }
    }

    pub fn from_cells(width: u32, height: u32, cells: Vec<i8>) -> Self {
        debug_assert_eq!(cells.len(), (width as usize) * (height as usize), "cell count != width × height");
        let flags = OnceLock::from(scan_flags(&cells));
        Self { width, height, data: Arc::new(cells), flags }
    }

    pub fn cells(&self) -> &[i8] {
        &self.data
    }

    /// Mutable cells; copies the plane first only when a snapshot still
    /// shares it, so the mid-stroke buffer paints in place.
    pub(crate) fn cells_mut(&mut self) -> &mut [i8] {
        self.flags = OnceLock::new();
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    fn flags(&self) -> (bool, bool) {
        *self.flags.get_or_init(|| scan_flags(&self.data))
    }

    /// True when every cell is zero, so applying the plane changes
    /// nothing. O(1) after the first read.
    pub fn is_blank(&self) -> bool {
        !self.flags().0
    }

    /// The bounding rectangle of the cells that differ from `other`;
    /// `None` when the planes are identical or differ in size.
    pub fn diff_bbox(&self, other: &Self) -> Option<CellRect> {
        if (self.width, self.height) != (other.width, other.height) {
            return None;
        }
        let w = self.width as usize;
        self.data
            .par_chunks(w)
            .zip(other.data.par_chunks(w))
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(y, (a, b))| {
                let differs = |(p, q): (&i8, &i8)| p != q;
                // the row differs, so both positions exist
                let x0 = a.iter().zip(b).position(differs).unwrap_or(0) as u32;
                let x1 = a.iter().zip(b).rposition(differs).unwrap_or(0) as u32;
                CellRect { x0, y0: y as u32, x1, y1: y as u32 }
            })
            .reduce_with(CellRect::union)
    }

    /// The cells inside `r` as their own plane.
    pub fn crop(&self, r: CellRect) -> Self {
        let (w, cw) = (self.width as usize, r.width() as usize);
        let mut cells = Vec::with_capacity(cw * r.height() as usize);
        for y in r.y0..=r.y1 {
            let start = y as usize * w + r.x0 as usize;
            cells.extend_from_slice(&self.data[start..start + cw]);
        }
        Self::from_cells(r.width(), r.height(), cells)
    }

    /// A copy of this plane with `patch` written over the cells at `r`;
    /// the inverse of `crop`. `patch` must be `r`'s size.
    pub fn paste(&self, r: CellRect, patch: &Self) -> Self {
        debug_assert_eq!((patch.width, patch.height), (r.width(), r.height()), "patch != rect size");
        let (w, cw) = (self.width as usize, r.width() as usize);
        let mut cells = self.data.as_ref().clone();
        for (row, y) in (r.y0..=r.y1).enumerate() {
            let start = y as usize * w + r.x0 as usize;
            cells[start..start + cw].copy_from_slice(&patch.data[row * cw..(row + 1) * cw]);
        }
        // Flags on first query: an undo that is not read again skips the scan.
        Self { width: self.width, height: self.height, data: Arc::new(cells), flags: OnceLock::new() }
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
    /// O(1) after the first read: the toolbar asks every frame.
    pub fn has_selected_region(&self) -> bool {
        self.flags().1
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

    /// `stroke` combined with this selection, which is only how much is
    /// selected: an Add stroke grows it and a Subtract stroke carves it
    /// away. It is kept negative, so as it is it reads as an erase.
    pub fn merge_stroke(&self, stroke: &Self) -> Result<Self, SelectionError> {
        self.map_with(stroke, |a, b| {
            let held = a.unsigned_abs();
            let mag = if b >= 0 { held.max(b as u8) } else { held.saturating_sub(b.unsigned_abs()) };
            -(mag as i8)
        })
    }

    /// The same area as a restore: every cell's amount, made positive.
    pub fn as_restore(&self) -> Self {
        self.map(|v| v.unsigned_abs().min(FULL as u8) as i8)
    }

    /// Region inversion: `FULL - |v|`, kept negative like any selection.
    pub fn invert(&self) -> Self {
        self.map(|v| -(FULL - v.unsigned_abs() as i8))
    }

    /// Stable content hash: SipHash-1-3 (`DefaultHasher`) over fixed
    /// 1 MiB chunks in parallel, then over the chunk digests and the
    /// dimensions. Deterministic across runs and machines, so persisted
    /// hashes survive load; the chunking is part of the format.
    pub fn content_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let digests: Vec<u64> = self
            .data
            .par_chunks(HASH_CHUNK)
            .map(|chunk| {
                let mut h = DefaultHasher::new();
                chunk.hash(&mut h);
                h.finish()
            })
            .collect();
        let mut h = DefaultHasher::new();
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        digests.hash(&mut h);
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

    /// The selected part of `source` (a plane of its size): cropped to
    /// the selection's bounds, alpha scaled by coverage. What clipboard
    /// Copy and Cut carry; `None` when nothing is selected.
    pub fn copy_to_rgba(&self, source: &image::RgbaImage) -> Option<image::RgbaImage> {
        let r = self.bounds()?;
        let mut out = image::imageops::crop_imm(source, r.x0, r.y0, r.width(), r.height()).to_image();
        self.crop(r).scale_alpha(&mut out, |c| c);
        Some(out)
    }

    /// The bounding rectangle of the non-zero cells; `None` on a blank plane.
    pub fn bounds(&self) -> Option<CellRect> {
        let w = self.width as usize;
        if self.is_blank() || w == 0 {
            return None;
        }
        self.data
            .par_chunks(w)
            .enumerate()
            .filter_map(|(y, row)| {
                let x0 = row.iter().position(|&v| v != 0)? as u32;
                // the row has a non-zero cell, found just above
                let x1 = row.iter().rposition(|&v| v != 0).unwrap_or(0) as u32;
                Some(CellRect { x0, y0: y as u32, x1, y1: y as u32 })
            })
            .reduce_with(CellRect::union)
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

    /// Nearest-neighbour resample to `w × h`, cell values carried as-is,
    /// by the same index rule as `apply_to_alpha`.
    #[cfg(test)]
    pub fn resampled(&self, w: u32, h: u32) -> Self {
        let mut cells = vec![0i8; (w as usize) * (h as usize)];
        self.for_each_resampled(w, h, |i, v| cells[i] = v);
        Self { width: w, height: h, data: Arc::new(cells), flags: OnceLock::new() }
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

    /// Apply the plane to a `mask_w × mask_h` alpha mask in place,
    /// resampled to its size inline. Per cell, with `s = g / FULL`:
    /// `g > 0` lifts alpha toward opaque (`a + (1 - a) s`), `g < 0`
    /// scales it toward clear (`a (1 + s)`), `g == 0` leaves it.
    pub fn apply_to_alpha(&self, alpha: &mut [u8], mask_w: usize, mask_h: usize) {
        debug_assert_eq!(alpha.len(), mask_w * mask_h, "apply_to_alpha: mask len != w*h");
        if self.is_blank() {
            return;
        }
        if (mask_w, mask_h) == (self.width as usize, self.height as usize) {
            // Full resolution on every live tick: rows in parallel, and the
            // untouched rows (most of a typical selection) skipped.
            alpha.par_chunks_mut(mask_w.max(1)).zip(self.data.par_chunks(mask_w.max(1))).for_each(|(row, cells)| {
                if cells.iter().all(|&g| g == 0) {
                    return;
                }
                for (a, &g) in row.iter_mut().zip(cells) {
                    *a = corrected_alpha(*a, g);
                }
            });
            return;
        }
        self.for_each_resampled(mask_w as u32, mask_h as u32, |i, g| alpha[i] = corrected_alpha(alpha[i], g));
    }

    /// The plane a first stroke makes on an empty selection: its Add part,
    /// kept negative.
    pub fn into_selection(self) -> Self {
        self.map(|b| if b > 0 { -b } else { 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::{paint_circle, Stamp};

    fn mask(w: u32, h: u32, data: Vec<i8>) -> MaskArtifact {
        MaskArtifact::from_cells(w, h, data)
    }


    #[test]
    fn a_selection_as_a_restore_keeps_its_amounts_positive() {
        let selection = mask(4, 1, vec![-127, -60, 0, 40]);
        assert_eq!(selection.as_restore().cells(), &[127, 60, 0, 40]);
    }

    /// Add grows a selection, Subtract carves it away, and what stays is
    /// stored as an erase.
    #[test]
    fn a_stroke_merges_by_amount() {
        let held = mask(4, 1, vec![-127, -127, -60, 0]);
        let add = mask(4, 1, vec![0, 100, 100, 127]);
        let sub = mask(4, 1, vec![-127, -100, 0, -127]);
        assert_eq!(held.merge_stroke(&add).unwrap().cells(), &[-127, -127, -100, -127], "add grows the selection");
        assert_eq!(held.merge_stroke(&sub).unwrap().cells(), &[0, -27, -60, 0], "subtract carves it away");
        assert!(!MaskArtifact::new_empty(4, 1).merge_stroke(&sub).unwrap().has_selected_region(), "nothing to carve");
        for stroke in [add, sub] {
            let first = MaskArtifact::new_empty(4, 1).merge_stroke(&stroke).unwrap();
            assert_eq!(stroke.into_selection(), first, "a first stroke equals merging onto nothing");
        }
    }

    #[test]
    fn blank_plane_is_a_no_op_correction() {
        let c = MaskArtifact::new_empty(10, 10);
        assert!(c.is_blank());
        let mut m = vec![128u8; 100];
        c.apply_to_alpha(&mut m, 10, 10);
        assert!(m.iter().all(|&v| v == 128));
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
        let mut m = vec![0u8; 100];
        c.apply_to_alpha(&mut m, 10, 10);
        assert_eq!(m.iter().filter(|&&v| v > 0).count(), 4);
        assert!(m.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn apply_full_add_and_subtract_saturate() {
        let mut m = vec![77u8; 4];
        mask(2, 2, vec![127; 4]).apply_to_alpha(&mut m, 2, 2);
        assert_eq!(m, [255; 4]);
        let mut m = vec![242u8; 4];
        mask(2, 2, vec![-127; 4]).apply_to_alpha(&mut m, 2, 2);
        assert_eq!(m, [0; 4]);
    }

    #[test]
    fn apply_half_strength_scales_and_lerps() {
        // s = 64/127 ≈ 0.504: subtract does m * (1 - s), add does
        // m + (1 - m) * s, zero leaves m.
        let s = 64.0 / 127.0;
        let byte = |v: f32| (v * 255.0).round() as u8;
        let mut m = vec![255u8, 153, 128];
        mask(3, 1, vec![-64, -64, 0]).apply_to_alpha(&mut m, 3, 1);
        assert_eq!(m, [byte(1.0 - s), byte(0.6 * (1.0 - s)), 128]);
        let mut m = vec![0u8, 255];
        mask(2, 1, vec![64, 64]).apply_to_alpha(&mut m, 2, 1);
        assert_eq!(m, [byte(s), 255]);
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
    fn add_mask_opposite_sign_moves_toward_the_newer_stroke_by_its_strength() {
        // A full restore over an erase flips it; a faint one (a soft brush
        // edge, a feathered rim) only nudges it, as paint would.
        let existing = mask(4, 1, vec![-100, 80, -127, 127]);
        let newer = mask(4, 1, vec![30, -10, 127, -127]);
        let r = existing.add_mask(&newer).unwrap();
        assert_eq!(r.cells(), &[-69, 73, 127, -127]);
    }

    #[test]
    fn add_mask_zero_cells_are_transparent() {
        let a = mask(2, 1, vec![-50, 0]);
        let b = mask(2, 1, vec![0, 70]);
        let r = a.add_mask(&b).unwrap();
        assert_eq!(r.cells(), &[-50, 70]);
    }

    #[test]
    fn invert_flips_coverage_and_keeps_it_negative() {
        let m = mask(2, 2, vec![0, 64, FULL, -32]);
        assert_eq!(m.invert().cells(), &[-127, -63, 0, -95]);
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
    fn merge_stroke_dimension_mismatch_returns_err() {
        let a = MaskArtifact::new_empty(4, 4);
        let b = MaskArtifact::new_empty(2, 8);
        assert!(matches!(
            a.merge_stroke(&b),
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
        let out = m.copy_to_rgba(&source).expect("a selection");
        assert_eq!(out.get_pixel(0, 0).0[3], 200);
        assert_eq!(out.get_pixel(1, 0).0[3], 0);
        assert_eq!(out.get_pixel(0, 1).0[3], 0);
        assert_eq!(out.get_pixel(1, 1).0[3], 200);
    }

    #[test]
    fn copy_to_rgba_carries_only_the_selections_bounds() {
        let mut cells = vec![0; 5 * 4];
        cells[5 + 1] = FULL;
        cells[2 * 5 + 3] = FULL / 2;
        let source = image::RgbaImage::from_fn(5, 4, |x, y| image::Rgba([x as u8, y as u8, 0, 255]));
        let out = mask(5, 4, cells).copy_to_rgba(&source).expect("a selection");
        assert_eq!(out.dimensions(), (3, 2), "columns 1-3, rows 1-2");
        assert_eq!(out.get_pixel(0, 0).0, [1, 1, 0, 255], "the selected pixel, where it was in the bounds");
        assert_eq!(out.get_pixel(2, 1).0[..2], [3, 2], "the half-selected one");
        assert!(out.get_pixel(2, 1).0[3] > 100 && out.get_pixel(2, 1).0[3] < 155);
        assert_eq!(out.get_pixel(1, 0).0[3], 0, "an unselected pixel inside the bounds is clear");
        assert!(MaskArtifact::new_empty(5, 4).copy_to_rgba(&source).is_none(), "nothing selected, nothing to copy");
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
    fn flags_follow_every_edit() {
        let faint = mask(2, 1, vec![0, 30]);
        assert!(!faint.is_blank() && !faint.has_selected_region());
        let mut live = MaskArtifact::new_empty(8, 8);
        assert!(live.is_blank());
        paint_circle(&mut live, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 0.1, mode: BrushMode::Add });
        assert!(!live.is_blank() && !live.has_selected_region(), "10 % strength never reaches half coverage");
        paint_circle(&mut live, 4.0, 4.0, 2.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        assert!(live.has_selected_region());
    }

    /// Pins the hash format (SipHash-1-3 over 1 MiB chunks, then over the
    /// digests and the dimensions): persisted hashes must survive a load.
    #[test]
    fn content_hash_format_is_pinned() {
        let n = HASH_CHUNK + 1;
        let cells: Vec<i8> = (0..n).map(|i| (i % 251) as i8).collect();
        let hash = mask(n as u32, 1, cells.clone()).content_hash();
        assert_eq!(hash, PINNED_HASH, "hash format changed: {hash:#x}");
        let mut flipped = cells;
        flipped[HASH_CHUNK] ^= 1;
        assert_ne!(mask(n as u32, 1, flipped).content_hash(), hash);
    }
    const PINNED_HASH: u64 = 0xa774abdc0c587095;

    #[test]
    fn resampled_applies_like_the_inline_resample() {
        let mut data = vec![0i8; 64];
        data[2 * 8 + 3] = -90;
        data[5 * 8 + 6] = 120;
        let m = mask(8, 8, data);
        let (mut inline, mut pre) = (vec![128u8; 9], vec![128u8; 9]);
        m.apply_to_alpha(&mut inline, 3, 3);
        m.resampled(3, 3).apply_to_alpha(&mut pre, 3, 3);
        assert_eq!(inline, pre);
    }

    #[test]
    fn diff_bbox_and_crop_round_trip() {
        let a = mask(6, 5, vec![0; 30]);
        let mut cells = vec![0i8; 30];
        cells[6 + 2] = 50;
        cells[3 * 6 + 4] = -70;
        let b = mask(6, 5, cells);
        assert_eq!(a.diff_bbox(&a), None);
        assert_eq!(a.diff_bbox(&MaskArtifact::new_empty(5, 5)), None, "size mismatch");
        let r = a.diff_bbox(&b).unwrap();
        assert_eq!(r, CellRect { x0: 2, y0: 1, x1: 4, y1: 3 });
        assert_eq!(r.grown(1, 6, 5), CellRect { x0: 1, y0: 0, x1: 5, y1: 4 });
        assert_eq!(r.grown(9, 6, 5), CellRect { x0: 0, y0: 0, x1: 5, y1: 4 });
        let c = b.crop(r);
        assert_eq!((c.width, c.height), (3, 3));
        assert_eq!(c.cells(), &[50, 0, 0, 0, 0, 0, 0, 0, -70]);
    }

    #[test]
    fn paste_undoes_a_change_inside_its_rect() {
        let a = mask(6, 5, (0..30).map(|i| i as i8).collect());
        let mut cells = a.cells().to_vec();
        cells[6 + 2] = 99;
        cells[3 * 6 + 4] = -70;
        let b = mask(6, 5, cells);
        let r = a.diff_bbox(&b).unwrap();
        assert_eq!(b.paste(r, &a.crop(r)), a, "the old cells restore the old plane");
        assert_eq!(a.paste(r, &b.crop(r)), b, "and the new ones the new plane");
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
