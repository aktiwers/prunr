//! Selection-mask artifact + pure action primitives.
//!
//! `MaskArtifact` is the single shared representation written by both
//! Paint Brush and Magic Brush. f32 single-channel at source
//! image resolution, wrapped in `Arc<Vec<f32>>` so undo snapshots and
//! cross-thread reads are refcount bumps, not memcpy of up to ~32 MB
//! (4K image).
//!
//! Convention: 0.0 = unselected, 1.0 = fully selected. Straight (non-
//! premultiplied) semantics throughout — matches the prunr-core sRGB rule.

use std::sync::Arc;

pub mod refine;

#[derive(Debug, Clone)]
pub struct MaskArtifact {
    pub width: u32,
    pub height: u32,
    pub data: Arc<Vec<f32>>,
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
            data: Arc::new(vec![0.0_f32; (width as usize) * (height as usize)]),
        }
    }

    /// Additive selection: pixel-wise max(self, other). Shift modifier.
    pub fn add_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        if self.width != other.width || self.height != other.height {
            return Err(SelectionError::DimensionMismatch {
                lhs: (self.width, self.height),
                rhs: (other.width, other.height),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.max(b))
            .collect();
        Ok(Self { width: self.width, height: self.height, data: Arc::new(data) })
    }

    /// Subtractive selection: pixel-wise (self - other).max(0.0). Alt modifier.
    pub fn subtract_mask(&self, other: &Self) -> Result<Self, SelectionError> {
        if self.width != other.width || self.height != other.height {
            return Err(SelectionError::DimensionMismatch {
                lhs: (self.width, self.height),
                rhs: (other.width, other.height),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a - b).max(0.0))
            .collect();
        Ok(Self { width: self.width, height: self.height, data: Arc::new(data) })
    }

    /// Inverts the selection: 1.0 - value per pixel.
    pub fn invert(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&v| 1.0 - v).collect();
        Self { width: self.width, height: self.height, data: Arc::new(data) }
    }

    /// Stable content hash. Uses DefaultHasher (SipHasher13) — deterministic
    /// across runs so persisted hashes survive load.
    pub fn content_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        for &v in self.data.iter() {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }

    /// Zeros the alpha channel for every pixel where mask >= 0.5.
    /// RGB values are preserved; pixels outside the selection are unchanged.
    pub fn alpha_cut(&self, rgba: &mut image::RgbaImage) {
        let samples = rgba.as_flat_samples_mut();
        let buf = samples.samples;
        for (i, &m) in self.data.iter().enumerate() {
            if m >= 0.5 {
                buf[i * 4 + 3] = 0;
            }
        }
    }

    /// Returns a new RgbaImage copying source pixels inside the selection
    /// (mask >= 0.5); pixels outside get alpha = 0. Basis of clipboard Copy/Cut.
    pub fn copy_to_rgba(&self, source: &image::RgbaImage) -> image::RgbaImage {
        let mut out = source.clone();
        for (i, pixel) in out.pixels_mut().enumerate() {
            if self.data[i] < 0.5 {
                pixel.0[3] = 0;
            }
        }
        out
    }

    /// Nearest-neighbour downsample to model tensor resolution.
    /// Output: +127 where upsampled mask >= 0.5, else 0.
    /// No negative entries in v1 — selection is additive only.
    pub fn to_mask_correction(
        &self,
        tensor_w: u16,
        tensor_h: u16,
    ) -> crate::brush::MaskCorrection {
        let tw = tensor_w as u32;
        let th = tensor_h as u32;
        let sw = self.width;
        let sh = self.height;
        let mut correction = crate::brush::MaskCorrection::empty(tensor_w, tensor_h);
        // Nearest-neighbour: map each tensor pixel to its source pixel.
        for ty in 0..th {
            let sy = ((ty as u64 * sh as u64) / th as u64) as u32;
            let row_base = (ty * tw) as usize;
            let src_row = (sy * sw) as usize;
            for tx in 0..tw {
                let sx = ((tx as u64 * sw as u64) / tw as u64) as u32;
                let src_val = self.data[src_row + sx as usize];
                correction.grid[row_base + tx as usize] = if src_val >= 0.5 { 127 } else { 0 };
            }
        }
        correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_has_zero_data_of_correct_length() {
        let m = MaskArtifact::new_empty(1920, 1080);
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.data.len(), 1920 * 1080);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn add_mask_takes_pixel_max() {
        let mut a_data = vec![0.0f32; 4];
        a_data[0] = 0.3;
        a_data[1] = 0.8;
        let a = MaskArtifact { width: 2, height: 2, data: Arc::new(a_data) };

        let mut b_data = vec![0.0f32; 4];
        b_data[0] = 0.7;
        b_data[1] = 0.2;
        let b = MaskArtifact { width: 2, height: 2, data: Arc::new(b_data) };

        let result = a.add_mask(&b).unwrap();
        assert!((result.data[0] - 0.7).abs() < 1e-6, "max(0.3, 0.7) = 0.7");
        assert!((result.data[1] - 0.8).abs() < 1e-6, "max(0.8, 0.2) = 0.8");
        assert!((result.data[2] - 0.0).abs() < 1e-6);
        assert!((result.data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn subtract_mask_clamps_at_zero() {
        let a = MaskArtifact {
            width: 2,
            height: 1,
            data: Arc::new(vec![0.8f32, 0.3]),
        };
        let b = MaskArtifact {
            width: 2,
            height: 1,
            data: Arc::new(vec![0.5f32, 0.9]),
        };
        let result = a.subtract_mask(&b).unwrap();
        assert!((result.data[0] - 0.3).abs() < 1e-6, "0.8 - 0.5 = 0.3");
        assert!((result.data[1] - 0.0).abs() < 1e-6, "(0.3 - 0.9).max(0.0) = 0.0");
    }

    #[test]
    fn invert_inverts_each_pixel() {
        let m = MaskArtifact {
            width: 2,
            height: 2,
            data: Arc::new(vec![0.0f32, 0.5, 1.0, 0.25]),
        };
        let inv = m.invert();
        let expected = [1.0f32, 0.5, 0.0, 0.75];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((inv.data[i] - exp).abs() < 1e-6, "pixel {i}: expected {exp}, got {}", inv.data[i]);
        }
    }

    #[test]
    fn content_hash_is_deterministic() {
        let m = MaskArtifact {
            width: 4,
            height: 4,
            data: Arc::new(vec![0.5f32; 16]),
        };
        let h1 = m.content_hash();
        let h2 = m.content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_changes_when_data_changes() {
        let mut data = vec![0.5f32; 16];
        let m1 = MaskArtifact { width: 4, height: 4, data: Arc::new(data.clone()) };
        data[7] = 0.9;
        let m2 = MaskArtifact { width: 4, height: 4, data: Arc::new(data) };
        assert_ne!(m1.content_hash(), m2.content_hash());
    }

    #[test]
    fn content_hash_changes_when_dims_change() {
        let m1 = MaskArtifact { width: 4, height: 4, data: Arc::new(vec![0.0f32; 16]) };
        let m2 = MaskArtifact { width: 8, height: 2, data: Arc::new(vec![0.0f32; 16]) };
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
    fn alpha_cut_zeros_alpha_in_selected_region() {
        // 2x2 mask: top-left selected (1.0), rest not.
        let mask = MaskArtifact {
            width: 2,
            height: 2,
            data: Arc::new(vec![1.0f32, 0.0, 0.0, 0.0]),
        };
        let mut rgba = image::RgbaImage::from_pixel(2, 2, image::Rgba([200u8, 100, 50, 255]));
        mask.alpha_cut(&mut rgba);
        // Pixel (0,0) should have alpha=0, RGB preserved.
        let p00 = rgba.get_pixel(0, 0);
        assert_eq!(p00.0[3], 0, "selected pixel alpha must be zeroed");
        assert_eq!(p00.0[0], 200, "RGB must be preserved");
    }

    #[test]
    fn alpha_cut_preserves_alpha_outside_selection() {
        let mask = MaskArtifact {
            width: 2,
            height: 2,
            data: Arc::new(vec![1.0f32, 0.0, 0.0, 0.0]),
        };
        let mut rgba = image::RgbaImage::from_pixel(2, 2, image::Rgba([200u8, 100, 50, 255]));
        mask.alpha_cut(&mut rgba);
        // Pixel (1,0), (0,1), (1,1) — mask < 0.5 — alpha unchanged.
        assert_eq!(rgba.get_pixel(1, 0).0[3], 255);
        assert_eq!(rgba.get_pixel(0, 1).0[3], 255);
        assert_eq!(rgba.get_pixel(1, 1).0[3], 255);
    }

    #[test]
    fn copy_to_rgba_zeros_alpha_outside_selection() {
        let mask = MaskArtifact {
            width: 2,
            height: 2,
            data: Arc::new(vec![1.0f32, 0.0, 0.0, 1.0]),
        };
        let source = image::RgbaImage::from_pixel(2, 2, image::Rgba([10u8, 20, 30, 200]));
        let out = mask.copy_to_rgba(&source);
        // Pixels 0 and 3 are selected: alpha preserved from source.
        assert_eq!(out.get_pixel(0, 0).0[3], 200);
        // Pixels 1 and 2 are unselected: alpha = 0.
        assert_eq!(out.get_pixel(1, 0).0[3], 0);
        assert_eq!(out.get_pixel(0, 1).0[3], 0);
        // Pixel (1,1) is selected: alpha preserved.
        assert_eq!(out.get_pixel(1, 1).0[3], 200);
    }

    #[test]
    fn to_mask_correction_downsamples_with_nearest_neighbour() {
        // 4x4 mask with all-ones in top-left 2x2 quadrant.
        let mut data = vec![0.0f32; 16];
        // Rows 0..2, cols 0..2.
        for y in 0..2usize {
            for x in 0..2usize {
                data[y * 4 + x] = 1.0;
            }
        }
        let mask = MaskArtifact { width: 4, height: 4, data: Arc::new(data) };
        // Downsample to 2x2 tensor.
        let corr = mask.to_mask_correction(2, 2);
        assert_eq!(corr.width, 2);
        assert_eq!(corr.height, 2);
        // Tensor pixel (0,0) maps to source (0,0) → 1.0 → 127.
        assert_eq!(corr.grid[0], 127);
        // Tensor pixel (1,0) maps to source (2,0) → 0.0 → 0.
        assert_eq!(corr.grid[1], 0);
        // Tensor pixel (0,1) maps to source (0,2) → 0.0 → 0.
        assert_eq!(corr.grid[2], 0);
    }

    #[test]
    fn to_mask_correction_quantizes_to_plus_127_or_zero() {
        // Mask with values 0.4 and 0.6 — threshold at 0.5.
        let mask = MaskArtifact {
            width: 2,
            height: 1,
            data: Arc::new(vec![0.4f32, 0.6]),
        };
        let corr = mask.to_mask_correction(2, 1);
        assert_eq!(corr.grid[0], 0, "0.4 < 0.5 → 0");
        assert_eq!(corr.grid[1], 127, "0.6 >= 0.5 → 127");
    }
}
