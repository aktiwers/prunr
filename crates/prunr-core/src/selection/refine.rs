//! Selection-mask refinement: boundary scan for the outline, and
//! edge feathering via `guided_filter_alpha`.
//!
//! Both operate on the unsigned coverage (`MaskArtifact::is_selected`);
//! feathering carries each cell's sign through unchanged.

use super::MaskArtifact;
use std::sync::Arc;

/// 8-connected boundary scan. Returns source-pixel coords of selected
/// pixels with at least one unselected or out-of-bounds neighbour.
/// Empty mask returns Vec::new(). Complexity is O(w*h).
pub fn outline_polyline(mask: &MaskArtifact) -> Vec<(u32, u32)> {
    let (w, h) = (mask.width, mask.height);
    let selected = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h
            && MaskArtifact::is_selected(mask.data[(y as u32 * w + x as u32) as usize])
    };
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !MaskArtifact::is_selected(mask.data[(y * w + x) as usize]) {
                continue;
            }
            let (xi, yi) = (x as i32, y as i32);
            let is_boundary = (-1..=1).any(|dy| {
                (-1..=1).any(|dx| (dx != 0 || dy != 0) && !selected(xi + dx, yi + dy))
            });
            if is_boundary {
                out.push((x, y));
            }
        }
    }
    out
}

/// Edge-aware feather. `feather_px == 0` or an empty selection returns a
/// clone (one Arc bump). Otherwise the binary region (`is_selected`) is
/// refined by `guided_filter_alpha` against `source`, and the result
/// replaces the magnitude of every cell inside the selection's bounding
/// box plus a `3 * feather_px` margin. Cells keep their own sign; cells
/// the filter newly reaches take the selection's dominant sign.
///
/// Peak working set is proportional to the bbox, not the image:
/// bbox_area × ~30 B (RGBA guide crop + two gray planes + the guided
/// filter's f32 scratch) plus one i8 copy of the full mask.
///
/// | bbox      | transient |
/// |-----------|-----------|
/// | 512²      | ~8 MB     |
/// | 2048²     | ~125 MB   |
/// | full 4K   | ~250 MB   |
pub fn feather_edges(
    mask: &MaskArtifact,
    source: &image::RgbaImage,
    feather_px: u32,
) -> MaskArtifact {
    if feather_px == 0 {
        return mask.clone();
    }
    let (w, h) = (mask.width, mask.height);
    if source.dimensions() != (w, h) {
        tracing::warn!(
            mask = ?(w, h), source = ?source.dimensions(),
            "feather_edges: guide dimensions differ from mask — skipping"
        );
        return mask.clone();
    }
    let Some((x0, y0, x1, y1)) = mask.selected_bbox() else {
        return mask.clone();
    };
    let margin = 3 * feather_px;
    let bx0 = x0.saturating_sub(margin);
    let by0 = y0.saturating_sub(margin);
    let bx1 = (x1 + margin).min(w - 1);
    let by1 = (y1 + margin).min(h - 1);
    let (bw, bh) = (bx1 - bx0 + 1, by1 - by0 + 1);

    let guide = image::imageops::crop_imm(source, bx0, by0, bw, bh).to_image();
    let mut binary = image::GrayImage::new(bw, bh);
    let (mut positive, mut negative) = (0usize, 0usize);
    for y in 0..bh {
        for x in 0..bw {
            let v = mask.data[((by0 + y) * w + bx0 + x) as usize];
            if MaskArtifact::is_selected(v) {
                binary.put_pixel(x, y, image::Luma([255]));
                if v < 0 { negative += 1 } else { positive += 1 }
            }
        }
    }
    // eps=1e-3 matches the standard refinement default for 8-bit alpha.
    let refined = crate::guided_filter::guided_filter_alpha(&guide, &binary, feather_px, 1e-3);
    let dominant: i8 = if negative > positive { -1 } else { 1 };

    let mut out: Vec<i8> = (*mask.data).clone();
    for y in 0..bh {
        for x in 0..bw {
            let idx = ((by0 + y) * w + bx0 + x) as usize;
            let v = out[idx];
            let magnitude = (refined.get_pixel(x, y).0[0] as f32 / 255.0 * super::FULL as f32)
                .round() as i8;
            let sign = if v < 0 { -1 } else if v > 0 { 1 } else { dominant };
            out[idx] = sign * magnitude;
        }
    }
    MaskArtifact { width: w, height: h, data: Arc::new(out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::FULL;

    fn make_mask(w: u32, h: u32, data: Vec<i8>) -> MaskArtifact {
        MaskArtifact { width: w, height: h, data: Arc::new(data) }
    }

    #[test]
    fn outline_empty_mask_returns_empty_vec() {
        let mask = MaskArtifact::new_empty(8, 8);
        assert!(outline_polyline(&mask).is_empty());
    }

    #[test]
    fn outline_full_mask_returns_border_pixels_only() {
        let mask = make_mask(4, 4, vec![FULL; 16]);
        let pts = outline_polyline(&mask);
        assert_eq!(pts.len(), 12, "4x4 full mask: 12 border pixels, 4 interior");
        assert!(pts.contains(&(0, 0)));
        assert!(!pts.contains(&(1, 1)), "fully interior pixel must not be in outline");
    }

    #[test]
    fn outline_centre_2x2_in_4x4_returns_2x2_perimeter() {
        let mut data = vec![0i8; 16];
        for &(x, y) in &[(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
            data[y * 4 + x] = FULL;
        }
        let mask = make_mask(4, 4, data);
        let pts = outline_polyline(&mask);
        assert_eq!(pts.len(), 4, "all 4 central pixels are boundary, got {:?}", pts);
        for &coord in &[(1u32, 1u32), (2, 1), (1, 2), (2, 2)] {
            assert!(pts.contains(&coord), "expected {:?} in outline", coord);
        }
    }

    #[test]
    fn outline_isolated_pixel_returns_single_point() {
        let mut data = vec![0i8; 64];
        data[3 * 8 + 3] = -FULL;
        let mask = make_mask(8, 8, data);
        assert_eq!(outline_polyline(&mask), vec![(3, 3)]);
    }

    #[test]
    fn outline_ignores_cells_below_the_selected_threshold() {
        let mut data = vec![0i8; 64];
        data[3 * 8 + 3] = 40;
        let mask = make_mask(8, 8, data);
        assert!(outline_polyline(&mask).is_empty());
    }

    /// Two disjoint selected regions produce boundary points from BOTH;
    /// the renderer never connects them (it draws per pixel, not a line).
    #[test]
    fn outline_two_disjoint_regions_returns_points_from_both() {
        let mut data = vec![0i8; 256];
        for (rx, ry) in [(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
            data[ry * 16 + rx] = FULL;
        }
        for (rx, ry) in [(12usize, 12usize), (13, 12), (12, 13), (13, 13)] {
            data[ry * 16 + rx] = FULL;
        }
        let mask = make_mask(16, 16, data);
        let pts = outline_polyline(&mask);
        assert!(pts.iter().any(|&(x, y)| x <= 2 && y <= 2), "region A missing: {:?}", pts);
        assert!(pts.iter().any(|&(x, y)| x >= 12 && y >= 12), "region B missing: {:?}", pts);
    }

    #[test]
    fn feather_with_zero_px_returns_identity_clone() {
        let mask = make_mask(4, 4, vec![64; 16]);
        let source = image::RgbaImage::from_pixel(4, 4, image::Rgba([128u8, 64, 32, 200]));
        let result = feather_edges(&mask, &source, 0);
        assert_eq!(result.data, mask.data);
    }

    #[test]
    fn feather_with_nonzero_px_softens_the_edge_and_keeps_sign() {
        // 16x16 mask, left half selected as a Subtract region.
        let mut data = vec![0i8; 256];
        for y in 0..16usize {
            for x in 0..8usize {
                data[y * 16 + x] = -FULL;
            }
        }
        let mask = make_mask(16, 16, data.clone());
        let source = image::RgbaImage::new(16, 16);
        let result = feather_edges(&mask, &source, 2);
        let changed = result.data.iter().zip(data.iter()).any(|(&a, &b)| a != b);
        assert!(changed, "feather with radius > 0 must alter at least one pixel");
        assert!(
            result.data.iter().all(|&v| v <= 0),
            "a Subtract selection must stay non-positive after feathering"
        );
        assert!(
            result.data.iter().any(|&v| v < 0 && v > -FULL),
            "feathered edge must contain intermediate magnitudes"
        );
    }

    #[test]
    fn feather_leaves_cells_far_from_the_selection_untouched() {
        // Single selected pixel in a corner; the opposite corner is well
        // outside bbox + margin and must be byte-identical.
        let mut data = vec![0i8; 64 * 64];
        data[2 * 64 + 2] = FULL;
        let mask = make_mask(64, 64, data.clone());
        let source = image::RgbaImage::from_pixel(64, 64, image::Rgba([90u8, 90, 90, 255]));
        let result = feather_edges(&mask, &source, 1);
        assert_eq!(result.data[63 * 64 + 63], 0);
        assert_eq!(result.data[40 * 64 + 40], 0);
    }
}
