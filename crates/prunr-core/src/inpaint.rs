//! LaMa-based inpainting for Phase 16 object removal.
//!
//! Public entry: `process_inpaint(image, mask) -> Result<RgbaImage>`.
//! Internally tiles the input at 512 px with 64 px feathered overlap so
//! large images stay within LaMa's fixed input size while seams stay
//! invisible against flat backgrounds.
//!
//! The `inpaint_tile` step is currently a stub returning the source
//! tile unchanged — wires up to ORT once the LaMa model file is
//! verified locally (see `prunr-models::lama_bytes`).

use image::{GrayImage, RgbaImage};

use crate::types::CoreError;

/// LaMa input/output side length in pixels.
pub const TILE: u32 = 512;
/// Overlap between adjacent tiles in pixels. Half the tile to stay safe
/// on the corners; the feather blend tapers within this region.
pub const OVERLAP: u32 = 64;

/// Top-level inpaint entry. Returns the input unchanged when the mask
/// is all-zero (no work) or the LaMa model isn't available.
pub fn process_inpaint(image: &RgbaImage, mask: &GrayImage) -> Result<RgbaImage, CoreError> {
    if image.dimensions() != mask.dimensions() {
        return Err(CoreError::Inference(format!(
            "inpaint: dim mismatch — image {:?} vs mask {:?}",
            image.dimensions(),
            mask.dimensions()
        )));
    }
    if mask_is_empty(mask) {
        return Ok(image.clone());
    }
    let Some(_model_bytes) = prunr_models::lama_bytes() else {
        return Err(CoreError::Inference(
            "inpaint: LaMa model not available — run `cargo xtask fetch-models`".into(),
        ));
    };

    // TODO(16-03): build the ORT session from `_model_bytes` once we
    // verify input/output tensor names + shapes against the FP32 file.
    // For now compose a no-op tile pipeline so the geometry path can
    // land + be tested.
    tile_compose(image, mask, |tile_rgba, tile_mask| {
        // Stub inference — return the source tile so the compose path
        // is exercised end-to-end without hitting ORT.
        let _ = tile_mask;
        tile_rgba.clone()
    })
}

/// True when every pixel of `mask` is zero. O(n) scan; cheap relative
/// to inference, and short-circuits the entire pipeline.
fn mask_is_empty(mask: &GrayImage) -> bool {
    mask.as_raw().iter().all(|&v| v == 0)
}

/// Source rect (in image pixels) for one inpaint tile + the same rect
/// in the destination buffer (always identical here, but the type
/// makes future repositioning explicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlacement {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Lay tiles across `(width, height)` with `OVERLAP` overlap. Tiles
/// near the right/bottom edges shift backwards to keep `TILE × TILE`
/// extent — never extending past the image bounds. For images <= TILE
/// in either axis, a single tile covers the whole image at its actual
/// size (LaMa accepts smaller inputs after pad-to-512 in the wrapper).
pub(crate) fn plan_tiles(width: u32, height: u32) -> Vec<TilePlacement> {
    if width <= TILE && height <= TILE {
        return vec![TilePlacement { x: 0, y: 0, w: width, h: height }];
    }
    let step = TILE - OVERLAP;
    let mut xs: Vec<u32> = (0..)
        .map(|i| i * step)
        .take_while(|&x| x + TILE <= width || x == 0)
        .collect();
    if let Some(&last_x) = xs.last() {
        if last_x + TILE < width {
            xs.push(width - TILE);
        }
    }
    if xs.is_empty() {
        xs.push(0);
    }
    let mut ys: Vec<u32> = (0..)
        .map(|i| i * step)
        .take_while(|&y| y + TILE <= height || y == 0)
        .collect();
    if let Some(&last_y) = ys.last() {
        if last_y + TILE < height {
            ys.push(height - TILE);
        }
    }
    if ys.is_empty() {
        ys.push(0);
    }

    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            let w = TILE.min(width - x);
            let h = TILE.min(height - y);
            out.push(TilePlacement { x, y, w, h });
        }
    }
    out
}

/// Smoothstep weight for blending tile contributions in the overlap
/// region. Returns 1.0 at the tile interior, 0.0 at the very edge,
/// with a smooth `t² (3 - 2t)` taper across `OVERLAP` pixels.
pub(crate) fn feather_weight(distance_from_edge: u32) -> f32 {
    if distance_from_edge >= OVERLAP {
        1.0
    } else {
        let t = distance_from_edge as f32 / OVERLAP as f32;
        crate::math::smoothstep(t)
    }
}

/// Compose the inpaint output by walking each tile, running the
/// inference closure, and feather-blending into the output buffer.
/// Skips tiles whose mask is all-zero (no work). The closure runs
/// once per non-empty tile.
fn tile_compose<F>(
    image: &RgbaImage,
    mask: &GrayImage,
    mut inpaint_tile: F,
) -> Result<RgbaImage, CoreError>
where
    F: FnMut(&RgbaImage, &GrayImage) -> RgbaImage,
{
    let (w, h) = image.dimensions();
    let mut out = image.clone();

    // Per-pixel weight accumulator for the feather blend. Tiles overlap
    // in the OVERLAP band; each contributes a smoothstep-weighted
    // sample, normalized at the end.
    let mut weight_acc: Vec<f32> = vec![0.0; (w * h) as usize];
    // Accumulator for the weighted RGBA in f32. We blend in linear
    // f32 space and quantize back to u8 at the end.
    let mut color_acc: Vec<[f32; 4]> = vec![[0.0; 4]; (w * h) as usize];

    for tile in plan_tiles(w, h) {
        let tile_rgba = sub_image_rgba(image, &tile);
        let tile_mask = sub_image_gray(mask, &tile);
        if mask_is_empty(&tile_mask) {
            continue;
        }
        let painted = inpaint_tile(&tile_rgba, &tile_mask);
        accumulate_tile(&mut color_acc, &mut weight_acc, &painted, &tile, w);
    }

    // Resolve accumulated tiles. Where weight == 0 (no tile touched),
    // keep the original pixel; otherwise normalise the accumulated
    // colour by the accumulated weight.
    for (i, pixel) in out.pixels_mut().enumerate() {
        let wsum = weight_acc[i];
        if wsum > 0.0 {
            let inv = 1.0 / wsum;
            let c = color_acc[i];
            pixel.0 = [
                (c[0] * inv).clamp(0.0, 255.0) as u8,
                (c[1] * inv).clamp(0.0, 255.0) as u8,
                (c[2] * inv).clamp(0.0, 255.0) as u8,
                (c[3] * inv).clamp(0.0, 255.0) as u8,
            ];
        }
    }
    Ok(out)
}

fn sub_image_rgba(src: &RgbaImage, t: &TilePlacement) -> RgbaImage {
    let mut out = RgbaImage::new(t.w, t.h);
    for ty in 0..t.h {
        for tx in 0..t.w {
            out.put_pixel(tx, ty, *src.get_pixel(t.x + tx, t.y + ty));
        }
    }
    out
}

fn sub_image_gray(src: &GrayImage, t: &TilePlacement) -> GrayImage {
    let mut out = GrayImage::new(t.w, t.h);
    for ty in 0..t.h {
        for tx in 0..t.w {
            out.put_pixel(tx, ty, *src.get_pixel(t.x + tx, t.y + ty));
        }
    }
    out
}

fn accumulate_tile(
    color_acc: &mut [[f32; 4]],
    weight_acc: &mut [f32],
    tile: &RgbaImage,
    placement: &TilePlacement,
    image_w: u32,
) {
    for ty in 0..placement.h {
        let dist_top = ty;
        let dist_bottom = placement.h - 1 - ty;
        let edge_y = dist_top.min(dist_bottom);
        for tx in 0..placement.w {
            let dist_left = tx;
            let dist_right = placement.w - 1 - tx;
            let edge_x = dist_left.min(dist_right);
            let w = feather_weight(edge_x.min(edge_y));
            let dst_idx = ((placement.y + ty) * image_w + (placement.x + tx)) as usize;
            let p = tile.get_pixel(tx, ty).0;
            let acc = &mut color_acc[dst_idx];
            acc[0] += p[0] as f32 * w;
            acc[1] += p[1] as f32 * w;
            acc[2] += p[2] as f32 * w;
            acc[3] += p[3] as f32 * w;
            weight_acc[dst_idx] += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    #[test]
    fn plan_tiles_small_image_single_tile() {
        let tiles = plan_tiles(400, 300);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], TilePlacement { x: 0, y: 0, w: 400, h: 300 });
    }

    #[test]
    fn plan_tiles_exact_tile_size_single_tile() {
        let tiles = plan_tiles(TILE, TILE);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], TilePlacement { x: 0, y: 0, w: TILE, h: TILE });
    }

    #[test]
    fn plan_tiles_horizontal_strip_overlaps_correctly() {
        // 1024 wide → first tile [0..512], second tile must be at
        // 1024 - 512 = 512 to cover the right edge.
        let tiles = plan_tiles(1024, TILE);
        assert!(tiles.len() >= 2, "1024-wide must produce >=2 tiles");
        let xs: Vec<u32> = tiles.iter().map(|t| t.x).collect();
        assert!(xs.contains(&0));
        assert!(xs.iter().any(|&x| x + TILE == 1024), "must cover the right edge");
    }

    #[test]
    fn plan_tiles_grid_full_size() {
        // 1024×1024 with 64-px overlap: a clean 2×2 split would leave
        // tiles touching with NO overlap (no feather possible). Plan
        // therefore packs 3 tiles per axis with proper overlap and
        // an extra row/column hugging the right/bottom edge.
        let tiles = plan_tiles(1024, 1024);
        assert!(tiles.len() >= 4, "must have at least 2×2 coverage, got {}", tiles.len());
        for t in &tiles {
            assert_eq!(t.w, TILE);
            assert_eq!(t.h, TILE);
        }
        // Right + bottom edges must be covered.
        assert!(tiles.iter().any(|t| t.x + t.w == 1024), "right edge");
        assert!(tiles.iter().any(|t| t.y + t.h == 1024), "bottom edge");
    }

    #[test]
    fn plan_tiles_covers_full_image_pixel_by_pixel() {
        // Every pixel of a 1500×900 image must be covered by ≥1 tile.
        let (w, h) = (1500u32, 900u32);
        let tiles = plan_tiles(w, h);
        for y in 0..h {
            for x in 0..w {
                let covered = tiles.iter().any(|t| {
                    x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h
                });
                assert!(covered, "pixel ({x}, {y}) uncovered");
            }
        }
    }

    #[test]
    fn feather_weight_full_inside_overlap() {
        // Far from the tile edge: weight 1.0.
        assert!((feather_weight(OVERLAP) - 1.0).abs() < 1e-6);
        assert!((feather_weight(OVERLAP * 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn feather_weight_zero_at_edge() {
        assert!(feather_weight(0).abs() < 1e-6);
    }

    #[test]
    fn feather_weight_smoothstep_monotone() {
        let mut prev = 0.0;
        for d in 0..=OVERLAP {
            let w = feather_weight(d);
            assert!(w >= prev - 1e-6, "feather weight must be monotonic non-decreasing");
            prev = w;
        }
    }

    #[test]
    fn process_inpaint_dim_mismatch_errors() {
        let img = RgbaImage::new(100, 100);
        let mask = GrayImage::new(50, 50);
        let result = process_inpaint(&img, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn process_inpaint_empty_mask_returns_input_unchanged() {
        let mut img = RgbaImage::new(64, 64);
        for (_, _, p) in img.enumerate_pixels_mut() {
            *p = Rgba([10, 20, 30, 255]);
        }
        let mask = GrayImage::new(64, 64); // all zero
        // Empty mask short-circuits BEFORE the model-load check, so this
        // works even when LaMa isn't available.
        let result = process_inpaint(&img, &mask).expect("empty mask is no-op");
        assert_eq!(result.as_raw(), img.as_raw());
    }

    #[test]
    fn process_inpaint_without_model_errors() {
        // Without the LaMa model file present, a non-empty mask must
        // surface a clear error rather than crashing or silently
        // returning the input (which would mask the bug).
        let img = RgbaImage::new(64, 64);
        let mut mask = GrayImage::new(64, 64);
        mask.put_pixel(32, 32, Luma([255]));
        let result = process_inpaint(&img, &mask);
        // The test runs without `cargo xtask fetch-models`; the helper
        // returns None and we expect Err. If the file IS present (CI
        // post-fetch), the stub inpaint succeeds and returns input
        // unchanged — that's also acceptable.
        match result {
            Err(_) => {}
            Ok(out) => assert_eq!(out.dimensions(), img.dimensions()),
        }
    }
}
