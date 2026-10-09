//! Selection-mask refinement: boundary scan for the outline, and
//! edge feathering via `guided_filter_alpha`. Both work on the
//! `is_selected` region; feathering keeps each cell's sign and gives
//! newly reached cells the selection's dominant sign.

use super::{MaskArtifact, FULL};
use std::sync::Arc;

/// 8-connected boundary scan. Returns source-pixel coords of selected
/// pixels with at least one unselected or out-of-bounds neighbour.
/// Empty mask returns Vec::new(). Complexity is O(w*h).
pub fn outline_polyline(mask: &MaskArtifact) -> Vec<(u32, u32)> {
    let (w, h) = (mask.width as usize, mask.height as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    // One byte per cell so the neighbour probes are plain compares, and a
    // per-row "all three vertical neighbours selected" line so each cell
    // needs three reads instead of eight.
    let sel: Vec<u8> = mask.data.iter().map(|&v| MaskArtifact::is_selected(v) as u8).collect();
    let zero = vec![0u8; w];
    let row = |y: usize| -> &[u8] { &sel[y * w..(y + 1) * w] };
    let mut vertical = vec![0u8; w];
    let mut out = Vec::new();
    for y in 0..h {
        let (up, mid, down) = (
            if y > 0 { row(y - 1) } else { &zero },
            row(y),
            if y + 1 < h { row(y + 1) } else { &zero },
        );
        for (((v, &u), &m), &d) in vertical.iter_mut().zip(up).zip(mid).zip(down) {
            *v = u & m & d;
        }
        for x in 0..w {
            if mid[x] == 0 {
                continue;
            }
            let left = x > 0 && vertical[x - 1] != 0;
            let right = x + 1 < w && vertical[x + 1] != 0;
            if !(left && vertical[x] != 0 && right) {
                out.push((x as u32, y as u32));
            }
        }
    }
    out
}

/// 1 for every cell within `radius` cells (Chebyshev) of a boundary
/// pixel, 0 elsewhere; `radius == 0` marks the boundary pixels only.
/// O(boundary × (2r+1)²) — r is at most 5 for a 10 px outline.
pub fn outline_band(mask: &MaskArtifact, radius: u32) -> Vec<u8> {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let mut band = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return band;
    }
    let r = radius as usize;
    for (x, y) in outline_polyline(mask) {
        let (x, y) = (x as usize, y as usize);
        let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
        for row in y.saturating_sub(r)..=(y + r).min(h - 1) {
            band[row * w + x0..=row * w + x1].fill(1);
        }
    }
    band
}

/// Bounding box `(x0, y0, x1, y1)`, inclusive, of the selected cells.
fn selected_bbox(mask: &MaskArtifact) -> Option<(u32, u32, u32, u32)> {
    let w = mask.width as usize;
    let mut bbox: Option<(u32, u32, u32, u32)> = None;
    for (y, row) in mask.data.chunks_exact(w.max(1)).enumerate() {
        let Some(first) = row.iter().position(|&v| MaskArtifact::is_selected(v)) else { continue };
        // just found one, so rposition is Some
        let last = row.iter().rposition(|&v| MaskArtifact::is_selected(v)).unwrap_or(first);
        let (first, last, y) = (first as u32, last as u32, y as u32);
        bbox = Some(match bbox {
            None => (first, y, last, y),
            Some((x0, y0, x1, _)) => (x0.min(first), y0, x1.max(last), y),
        });
    }
    bbox
}

/// Edge-aware feather. `feather_px == 0` or an empty selection returns a
/// clone (one Arc bump). Otherwise the selected region is refined by
/// `guided_filter_alpha` against `source`, and the result replaces the
/// magnitude of every cell inside the selection's bounding box plus a
/// `3 * feather_px` margin.
///
/// Peak working set is proportional to the bbox, not the image:
/// bbox_area × ~54 B (RGBA guide crop, two gray planes, and the guided
/// filter's eight f32 planes plus integral scratch) plus one i8 copy of
/// the full mask. Measured on a 4K source:
///
/// | bbox      | transient |
/// |-----------|-----------|
/// | 512²      | ~15 MB    |
/// | 2048²     | ~220 MB   |
/// | full 4K   | ~430 MB   |
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
    let Some((x0, y0, x1, y1)) = selected_bbox(mask) else {
        return mask.clone();
    };
    let margin = 3 * feather_px;
    let bx0 = x0.saturating_sub(margin);
    let by0 = y0.saturating_sub(margin);
    let bx1 = (x1 + margin).min(w - 1);
    let by1 = (y1 + margin).min(h - 1);
    let (bw, bh) = (bx1 - bx0 + 1, by1 - by0 + 1);
    let bw_us = bw as usize;
    let crop_row = |y: u32| -> std::ops::Range<usize> {
        let start = ((by0 + y) * w + bx0) as usize;
        start..start + bw_us
    };

    let mut guide = image::RgbaImage::new(bw, bh);
    let mut binary = image::GrayImage::new(bw, bh);
    let (mut positive, mut negative) = (0usize, 0usize);
    for y in 0..bh {
        let r = crop_row(y);
        guide.as_mut()[y as usize * bw_us * 4..][..bw_us * 4]
            .copy_from_slice(&source.as_raw()[r.start * 4..r.end * 4]);
        let cells = &mask.data[r];
        for (b, &v) in binary.as_mut()[y as usize * bw_us..][..bw_us].iter_mut().zip(cells) {
            if MaskArtifact::is_selected(v) {
                *b = 255;
                if v < 0 { negative += 1 } else { positive += 1 }
            }
        }
    }
    let dominant: i8 = if negative > positive { -1 } else { 1 };
    // eps=1e-3 matches the standard refinement default for 8-bit alpha.
    let refined = crate::guided_filter::guided_filter_alpha(&guide, &binary, feather_px, 1e-3);
    let magnitude: Vec<i8> = (0..=255u32)
        .map(|v| (v as f32 / 255.0 * FULL as f32).round() as i8)
        .collect();

    let mut out: Vec<i8> = (*mask.data).clone();
    for y in 0..bh {
        let r = crop_row(y);
        let refined_row = &refined.as_raw()[y as usize * bw_us..][..bw_us];
        for (cell, &rv) in out[r].iter_mut().zip(refined_row) {
            let sign = match cell.signum() {
                0 => dominant,
                s => s,
            };
            *cell = sign * magnitude[rv as usize];
        }
    }
    MaskArtifact { width: w, height: h, data: Arc::new(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn outline_band_dilates_the_boundary_by_the_radius() {
        let mut data = vec![0i8; 64];
        data[3 * 8 + 3] = FULL;
        let mask = make_mask(8, 8, data);
        let thin = outline_band(&mask, 0);
        assert_eq!(thin.iter().filter(|&&b| b == 1).count(), 1);
        let wide = outline_band(&mask, 1);
        assert_eq!(wide.iter().filter(|&&b| b == 1).count(), 9, "radius 1 = 3×3 block");
        assert_eq!(wide[2 * 8 + 2], 1);
        assert_eq!(wide[5 * 8 + 5], 0);
    }

    #[test]
    fn selected_bbox_covers_selected_cells_only() {
        let mut data = vec![0i8; 16];
        data[5] = -FULL; // (1, 1)
        data[11] = FULL; // (3, 2)
        data[12] = 10; // (0, 3): below threshold — ignored
        assert_eq!(selected_bbox(&make_mask(4, 4, data)), Some((1, 1, 3, 2)));
        assert_eq!(selected_bbox(&MaskArtifact::new_empty(4, 4)), None);
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
}
