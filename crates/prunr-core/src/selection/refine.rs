//! Selection-mask refinement: outline polyline extraction (8-connected
//! boundary scan) + edge feathering via `guided_filter_alpha`.
//!
//! Working set: outline scan allocates one Vec<(u32, u32)> sized to
//! the boundary pixel count (typically << image area). Feather goes
//! through `guided_filter_alpha` which returns a fresh GrayImage at
//! source resolution — RAM cost is one GrayImage + one scratch RgbaImage.

use super::MaskArtifact;
use std::sync::Arc;

/// 8-connected boundary scan. Returns source-pixel coords of pixels
/// where mask >= 0.5 AND any of the 8 neighbours is < 0.5 or out of bounds.
/// Empty mask returns Vec::new(). Complexity is O(w*h).
pub fn outline_polyline(mask: &MaskArtifact) -> Vec<(u32, u32)> {
    let (w, h) = (mask.width, mask.height);
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if mask.data[idx] < 0.5 {
                continue;
            }
            let mut is_boundary = false;
            'outer: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx as u32 >= w || ny as u32 >= h {
                        is_boundary = true;
                        break 'outer;
                    }
                    let nidx = (ny as u32 * w + nx as u32) as usize;
                    if mask.data[nidx] < 0.5 {
                        is_boundary = true;
                        break 'outer;
                    }
                }
            }
            if is_boundary {
                out.push((x, y));
            }
        }
    }
    out
}

/// Edge-aware feather. feather_px=0 returns an identity clone (cheap Arc bump).
/// Otherwise builds a scratch RgbaImage from source + quantized mask alpha
/// and calls `guided_filter_alpha`, then reads back the refined alpha as f32.
/// Does NOT mutate inputs.
pub fn feather_edges(
    mask: &MaskArtifact,
    source: &image::RgbaImage,
    feather_px: u32,
) -> MaskArtifact {
    if feather_px == 0 {
        return mask.clone();
    }
    let (w, h) = (source.width(), source.height());
    let mut mask_lo = image::GrayImage::new(w, h);
    for (i, pixel) in mask_lo.pixels_mut().enumerate() {
        pixel.0[0] = if mask.data[i] >= 0.5 { 255 } else { 0 };
    }
    // eps=1e-3 matches the standard refinement default for 8-bit alpha.
    let refined_gray = crate::guided_filter::guided_filter_alpha(source, &mask_lo, feather_px, 1e-3);
    let refined: Vec<f32> = refined_gray
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    MaskArtifact {
        width: mask.width,
        height: mask.height,
        data: Arc::new(refined),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mask(w: u32, h: u32, data: Vec<f32>) -> MaskArtifact {
        MaskArtifact { width: w, height: h, data: Arc::new(data) }
    }

    #[test]
    fn outline_empty_mask_returns_empty_vec() {
        let mask = MaskArtifact::new_empty(8, 8);
        assert!(outline_polyline(&mask).is_empty());
    }

    #[test]
    fn outline_full_mask_returns_border_pixels() {
        // All-ones 4x4 mask: boundary = all pixels on the 4 edges.
        let mask = make_mask(4, 4, vec![1.0f32; 16]);
        let pts = outline_polyline(&mask);
        // Every pixel is on the border in a 4x4 grid (none are surrounded
        // by 8 in-bounds neighbours all at >= 0.5, since 4x4 has no interior).
        // 4x4 has interior pixels at (1,1),(2,1),(1,2),(2,2) only if 8-connected
        // — wait, (1,1): neighbours include (0,0),(0,1),(0,2),(1,0),(2,0),(1,2),(2,1),(2,2)
        // all in bounds and all 1.0 — so (1,1) is NOT a boundary pixel.
        // Only true boundary pixels (any neighbour out-of-bounds or < 0.5).
        assert!(!pts.is_empty(), "full mask must have border pixels");
        // Corner (0,0) must be in the outline (it has out-of-bounds neighbours).
        assert!(pts.contains(&(0, 0)));
        // Interior pixel (1,1) in a 4x4 all-ones mask is surrounded on all 8
        // sides by in-bounds pixels that are all >= 0.5 — NOT a boundary.
        assert!(!pts.contains(&(1, 1)), "fully interior pixel must not be in outline");
    }

    #[test]
    fn outline_centre_2x2_in_4x4_returns_2x2_perimeter() {
        // 4x4 mask, central 2x2 pixels at (1,1),(2,1),(1,2),(2,2) set to 1.0.
        let mut data = vec![0.0f32; 16];
        for &(x, y) in &[(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
            data[y * 4 + x] = 1.0;
        }
        let mask = make_mask(4, 4, data);
        let pts = outline_polyline(&mask);
        // All 4 pixels should be boundary (each has at least one 0.0 neighbour).
        assert_eq!(pts.len(), 4, "all 4 central pixels are boundary, got {:?}", pts);
        for &coord in &[(1u32, 1u32), (2, 1), (1, 2), (2, 2)] {
            assert!(pts.contains(&coord), "expected {:?} in outline", coord);
        }
    }

    #[test]
    fn outline_isolated_pixel_returns_single_point() {
        // Single pixel at (3,3) in an 8x8 mask.
        let mut data = vec![0.0f32; 64];
        data[3 * 8 + 3] = 1.0;
        let mask = make_mask(8, 8, data);
        let pts = outline_polyline(&mask);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0], (3, 3));
    }

    /// Two disjoint selected regions produce boundary points from BOTH —
    /// the row-major scan visits the whole mask and the renderer (in
    /// `selection_overlay::render_selection_overlay`) emits one filled
    /// rect per pixel, never connecting them. This test pins the boundary
    /// scan's output; the no-connecting-line invariant is upheld at the
    /// render side by virtue of NOT using `Shape::line` (see the comment
    /// in `selection_overlay.rs` for the design rationale).
    #[test]
    fn outline_two_disjoint_regions_returns_points_from_both() {
        // 16x16 mask, two non-overlapping 2x2 squares far apart.
        // Region A: (1,1)..(2,2). Region B: (12,12)..(13,13).
        let mut data = vec![0.0f32; 256];
        for (rx, ry) in [(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
            data[ry * 16 + rx] = 1.0;
        }
        for (rx, ry) in [(12usize, 12usize), (13, 12), (12, 13), (13, 13)] {
            data[ry * 16 + rx] = 1.0;
        }
        let mask = make_mask(16, 16, data);
        let pts = outline_polyline(&mask);
        // Both regions contribute boundary points.
        assert!(
            pts.iter().any(|&(x, y)| x <= 2 && y <= 2),
            "expected at least one boundary point from region A: {:?}", pts,
        );
        assert!(
            pts.iter().any(|&(x, y)| x >= 12 && y >= 12),
            "expected at least one boundary point from region B: {:?}", pts,
        );
    }

    #[test]
    fn feather_with_zero_px_returns_identity_clone() {
        let mask = make_mask(4, 4, vec![0.5f32; 16]);
        let source = image::RgbaImage::from_pixel(4, 4, image::Rgba([128u8, 64, 32, 200]));
        let result = feather_edges(&mask, &source, 0);
        // Data should be identical.
        assert_eq!(result.data.len(), mask.data.len());
        for (a, b) in result.data.iter().zip(mask.data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn feather_with_nonzero_px_changes_at_least_one_pixel() {
        // 8x8 mask with left half selected.
        let mut data = vec![0.0f32; 64];
        for y in 0..8usize {
            for x in 0..4usize {
                data[y * 8 + x] = 1.0;
            }
        }
        let mask = make_mask(8, 8, data.clone());
        let source = image::RgbaImage::new(8, 8);
        let result = feather_edges(&mask, &source, 2);
        // The guided filter should produce a different result from the input
        // for at least some pixels.
        let changed = result
            .data
            .iter()
            .zip(data.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-3);
        assert!(changed, "feather with radius > 0 must alter at least one pixel");
    }
}
