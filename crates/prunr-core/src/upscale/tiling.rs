//! Tile-based upscale with overlap-blend.
//!
//! Peak working-set RAM (per dispatch, scale=4, fp32 RGB accumulator +
//! weight buffer + per-tile transient scratch + alpha companion buffers).
//! Numbers are for the OUTPUT buffers; the input image and the model
//! session are separate.
//!
//! | Input        | RGB accum f32 | Weight buf f32 | Per-tile transient¹ | Alpha src+dst | Total      |
//! |--------------|---------------|----------------|---------------------|---------------|------------|
//! | 1024 × 1024  | 192 MB        | 64 MB          | ~10 MB              | 5 MB          | ~272 MB    |
//! | 2048 × 2048  | 768 MB        | 256 MB         | ~10 MB              | 20 MB         | ~1054 MB   |
//! | 4096 × 4096  | 3072 MB       | 1024 MB        | ~10 MB              | 80 MB         | ~4186 MB   |
//!
//! ¹ Per-tile transient = padded RGB tile + inferred RGB output (lives only
//! during one `run_tile` call, dropped before the next tile). The blend
//! reads `inferred` directly with an offset (no separate trimmed copy).
//!
//! 4K → 16K upscale is the limit on a 16 GB machine. The Processor's
//! working_set_mb admission gate refuses to dispatch when free RAM is
//! below this peak.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use image::{RgbImage, RgbaImage};

use crate::CoreError;
use super::alpha::upscale_alpha_lanczos3;

/// Tile dispatch configuration shared by the planner and the tiler.
#[derive(Debug, Clone, Copy)]
pub struct TilingConfig {
    /// Tile edge length in input pixels.
    pub tile_size: u32,
    /// Window-size multiple constraint: both tile W and H are padded to the
    /// nearest multiple before the model sees them (HAT-L window_size = 16).
    /// `None` means no constraint.
    pub tile_multiple: Option<u32>,
    /// Overlap in input pixels between adjacent tiles. Smoothstep-blended.
    pub overlap: u32,
}

/// Location and dimensions of one tile within the input image.
///
/// Carries the model-padded dimensions (`padded_w` / `padded_h`) alongside
/// the input-space dimensions; `inpaint::TilePlacement` does not need
/// padding so it is a distinct, narrower type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscaleTilePlacement {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Padded width sent to the model (>= w, multiple of tile_multiple).
    pub padded_w: u32,
    /// Padded height sent to the model (>= h, multiple of tile_multiple).
    pub padded_h: u32,
}

/// Plan a grid of overlapping tiles that covers the entire image.
///
/// `cfg.tile_multiple`: if `Some(m)`, each tile is padded so its dimensions
/// are multiples of `m` (required for transformer models with a window
/// constraint such as HAT's 16-px window). The tiler pads bottom-right
/// and the caller trims the output back to `w × h`.
pub fn plan_upscale_tiles(
    width: u32,
    height: u32,
    cfg: TilingConfig,
) -> Vec<UpscaleTilePlacement> {
    let TilingConfig { tile_size, tile_multiple, overlap } = cfg;
    let step = tile_size.saturating_sub(overlap).max(1);

    let mut xs: Vec<u32> = (0..width).step_by(step as usize).collect();
    // Ensure the last column reaches the right edge.
    if let Some(&last) = xs.last() {
        if last + tile_size < width {
            xs.push(width.saturating_sub(tile_size));
        }
    }
    // Deduplicate and sort (last column may coincide with an earlier step).
    xs.sort_unstable();
    xs.dedup();

    let mut ys: Vec<u32> = (0..height).step_by(step as usize).collect();
    if let Some(&last) = ys.last() {
        if last + tile_size < height {
            ys.push(height.saturating_sub(tile_size));
        }
    }
    ys.sort_unstable();
    ys.dedup();

    let mut tiles = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            let w = tile_size.min(width - x);
            let h = tile_size.min(height - y);
            let (padded_w, padded_h) = match tile_multiple {
                Some(m) if m > 0 => (
                    w.div_ceil(m) * m,
                    h.div_ceil(m) * m,
                ),
                _ => (w, h),
            };
            tiles.push(UpscaleTilePlacement { x, y, w, h, padded_w, padded_h });
        }
    }
    tiles
}

/// Upscale an RGBA image tile-by-tile, blending overlapping regions with a
/// smoothstep weight function to hide tile seams.
///
/// `run_tile(rgb_tile, padded_w, padded_h)` must return an RGB image of
/// dimensions `scale * padded_w × scale * padded_h`. The caller pads
/// bottom-right before the call; `upscale_tiled` reads the unpadded region
/// of the result directly during blend (no separate trim allocation).
///
/// Alpha is upscaled independently via Lanczos3 and merged into the final
/// `RgbaImage`.
///
/// `on_tile_done(done, total)` is called after each tile completes.
///
/// If `cancel` is set and its flag becomes `true`, returns
/// `Err(CoreError::Cancelled)` before the next tile's inference.
pub fn upscale_tiled<F, G>(
    input: &RgbaImage,
    scale: u32,
    cfg: TilingConfig,
    run_tile: F,
    on_tile_done: G,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(&RgbImage, u32, u32) -> Result<RgbImage, CoreError>,
    G: Fn(u32, u32),
{
    let TilingConfig { overlap, .. } = cfg;
    let (in_w, in_h) = input.dimensions();
    let out_w = in_w * scale;
    let out_h = in_h * scale;

    let tiles = plan_upscale_tiles(in_w, in_h, cfg);
    let total = tiles.len() as u32;

    // RGB f32 accumulator (channels interleaved: R, G, B per pixel).
    let pixel_count = (out_w as usize) * (out_h as usize);
    let mut rgb_accum: Vec<f32> = vec![0.0; pixel_count * 3];
    let mut weight_buf: Vec<f32> = vec![0.0; pixel_count];

    // Sequential: ort sessions are Mutex-guarded internally and nested
    // rayon inside the subprocess worker path has caused deadlocks
    // historically (see `apply_background_color` for the same rule).
    for (tile_idx, tile) in tiles.iter().enumerate() {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err(CoreError::Cancelled);
        }

        let rgb_tile = extract_rgb_tile(input, tile);
        let padded_tile = if tile.padded_w != tile.w || tile.padded_h != tile.h {
            pad_rgb_tile(&rgb_tile, tile.padded_w, tile.padded_h)
        } else {
            rgb_tile
        };
        let inferred = run_tile(&padded_tile, tile.padded_w, tile.padded_h)?;

        // Blend the unpadded `tile.w * scale × tile.h * scale` region of
        // `inferred` into the accumulator with smoothstep weights. The taper
        // only applies on sides where an adjacent tile exists — image-
        // boundary sides always get weight 1.0 so corner pixels are never
        // zeroed by the taper.
        let out_tile_x = tile.x * scale;
        let out_tile_y = tile.y * scale;
        let out_tile_w = tile.w * scale;
        let out_tile_h = tile.h * scale;
        let has_left = tile.x > 0;
        let has_top = tile.y > 0;
        let has_right = tile.x + tile.w < in_w;
        let has_bottom = tile.y + tile.h < in_h;
        let edge = overlap * scale;

        for py in 0..out_tile_h {
            for px in 0..out_tile_w {
                let wx = blend_weight_sided(px, edge, out_tile_w, has_left, has_right);
                let wy = blend_weight_sided(py, edge, out_tile_h, has_top, has_bottom);
                let w = wx * wy;

                let out_x = out_tile_x + px;
                let out_y = out_tile_y + py;
                let out_idx = (out_y as usize) * (out_w as usize) + (out_x as usize);

                let tile_pixel = inferred.get_pixel(px, py).0;
                rgb_accum[out_idx * 3] += tile_pixel[0] as f32 * w;
                rgb_accum[out_idx * 3 + 1] += tile_pixel[1] as f32 * w;
                rgb_accum[out_idx * 3 + 2] += tile_pixel[2] as f32 * w;
                weight_buf[out_idx] += w;
            }
        }

        on_tile_done(tile_idx as u32 + 1, total);
    }

    let alpha = upscale_alpha_lanczos3(input, out_w, out_h);

    let mut out_img = RgbaImage::new(out_w, out_h);
    for py in 0..out_h {
        for px in 0..out_w {
            let idx = (py as usize) * (out_w as usize) + (px as usize);
            // Guard against zero weight on unreachable pixels — f32 division
            // by 0 would otherwise produce inf and saturate the channel.
            let inv_w = 1.0 / weight_buf[idx].max(1e-6);
            let r = (rgb_accum[idx * 3] * inv_w).clamp(0.0, 255.0) as u8;
            let g = (rgb_accum[idx * 3 + 1] * inv_w).clamp(0.0, 255.0) as u8;
            let b = (rgb_accum[idx * 3 + 2] * inv_w).clamp(0.0, 255.0) as u8;
            let a = alpha.get_pixel(px, py).0[0];
            out_img.put_pixel(px, py, image::Rgba([r, g, b, a]));
        }
    }

    Ok(out_img)
}

fn extract_rgb_tile(input: &RgbaImage, tile: &UpscaleTilePlacement) -> RgbImage {
    let cropped = image::imageops::crop_imm(input, tile.x, tile.y, tile.w, tile.h).to_image();
    image::DynamicImage::ImageRgba8(cropped).to_rgb8()
}

fn pad_rgb_tile(src: &RgbImage, padded_w: u32, padded_h: u32) -> RgbImage {
    let mut dst = RgbImage::new(padded_w, padded_h);
    image::imageops::overlay(&mut dst, src, 0, 0);
    dst
}

/// Smoothstep weight for a single axis, respecting whether adjacent tiles
/// exist on each side. Image-boundary sides (`has_low = false` or
/// `has_high = false`) receive weight 1.0 rather than tapering toward 0 —
/// there is no neighboring tile there, so the taper would zero pixels that
/// have no alternative contributor.
fn blend_weight_sided(x: u32, edge: u32, size: u32, has_low: bool, has_high: bool) -> f32 {
    let t_low = if has_low && x < edge {
        x as f32 / edge as f32
    } else {
        1.0
    };
    let t_high = if has_high && x + edge >= size {
        (size - x) as f32 / edge as f32
    } else {
        1.0
    };
    let t = t_low.min(t_high).clamp(0.0, 1.0);
    crate::math::smoothstep(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every input pixel must be covered by at least one tile.
    #[test]
    fn plan_upscale_tiles_covers_input_with_overlap() {
        let width = 1024u32;
        let height = 768u32;
        let tiles = plan_upscale_tiles(width, height, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 });

        let mut covered = vec![false; (width * height) as usize];
        for tile in &tiles {
            for ty in 0..tile.h {
                for tx in 0..tile.w {
                    let px = tile.x + tx;
                    let py = tile.y + ty;
                    covered[(py * width + px) as usize] = true;
                }
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "some pixels not covered by any tile"
        );
    }

    /// The last row/column tile must have padded dimensions that are a
    /// multiple of the given tile_multiple.
    #[test]
    fn plan_upscale_tiles_pads_to_multiple_when_required() {
        let tiles = plan_upscale_tiles(250, 200, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 16 });
        for tile in &tiles {
            assert_eq!(
                tile.padded_w % 16, 0,
                "padded_w {} not a multiple of 16",
                tile.padded_w
            );
            assert_eq!(
                tile.padded_h % 16, 0,
                "padded_h {} not a multiple of 16",
                tile.padded_h
            );
        }
    }

    /// Pad-to-multiple round-trip: a 250×200 tile padded to 256×208 (nearest
    /// multiples of 16), inferred at 4× to 1024×832, then trimmed back to
    /// 1000×800 (250*4 × 200*4) must produce exactly the expected dimensions.
    #[test]
    fn pad_to_multiple_roundtrip_preserves_dimensions() {
        let tiles = plan_upscale_tiles(250, 200, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 0 });
        assert_eq!(tiles.len(), 1, "expected single tile");
        let tile = tiles[0];
        assert_eq!(tile.w, 250);
        assert_eq!(tile.h, 200);
        assert_eq!(tile.padded_w, 256);
        assert_eq!(tile.padded_h, 208);

        let scale = 4u32;
        let run_tile = |rgb: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            // Nearest-neighbor upscale to scale*pw × scale*ph.
            let out = RgbImage::from_fn(scale * pw, scale * ph, |x, y| {
                *rgb.get_pixel(x / scale, y / scale)
            });
            Ok(out)
        };

        let input = RgbaImage::from_pixel(250, 200, image::Rgba([128, 64, 32, 255]));
        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 0 }, run_tile, |_, _| {}, None)
            .expect("upscale_tiled failed");

        assert_eq!(result.width(), 250 * scale);
        assert_eq!(result.height(), 200 * scale);
    }

    /// On a uniform-color input, overlap-blend must not introduce drift.
    /// Every output pixel should match the input color within ±1 LSB.
    #[test]
    fn upscale_tiled_seamless_with_uniform_stub() {
        let scale = 4u32;
        let r = 128u8;
        let g = 64u8;
        let b = 200u8;
        let input = RgbaImage::from_pixel(512, 512, image::Rgba([r, g, b, 255]));

        let run_tile = |rgb: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            let out = RgbImage::from_fn(scale * pw, scale * ph, |x, y| {
                *rgb.get_pixel(x / scale, y / scale)
            });
            Ok(out)
        };

        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 }, run_tile, |_, _| {}, None)
            .expect("upscale_tiled failed");

        assert_eq!(result.width(), 512 * scale);
        assert_eq!(result.height(), 512 * scale);

        for py in 0..result.height() {
            for px in 0..result.width() {
                let p = result.get_pixel(px, py).0;
                assert!(
                    (p[0] as i32 - r as i32).abs() <= 1,
                    "R drift at ({px},{py}): expected ~{r}, got {}",
                    p[0]
                );
                assert!(
                    (p[1] as i32 - g as i32).abs() <= 1,
                    "G drift at ({px},{py}): expected ~{g}, got {}",
                    p[1]
                );
                assert!(
                    (p[2] as i32 - b as i32).abs() <= 1,
                    "B drift at ({px},{py}): expected ~{b}, got {}",
                    p[2]
                );
            }
        }
    }

    /// `on_tile_done` must be called exactly N times with (1,N), (2,N), ..., (N,N).
    #[test]
    fn upscale_tiled_emits_correct_tile_progress() {
        let scale = 4u32;
        let input = RgbaImage::from_pixel(512, 512, image::Rgba([0, 0, 0, 255]));

        let tiles = plan_upscale_tiles(512, 512, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 });
        let expected_total = tiles.len() as u32;

        let calls = std::cell::RefCell::new(Vec::new());

        let run_tile = |rgb: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            Ok(RgbImage::from_fn(scale * pw, scale * ph, |x, y| {
                *rgb.get_pixel(x / scale, y / scale)
            }))
        };

        upscale_tiled(
            &input, scale,
            TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 },
            run_tile,
            |done, total| calls.borrow_mut().push((done, total)),
            None,
        ).expect("upscale_tiled failed");

        let calls = calls.into_inner();
        assert_eq!(calls.len() as u32, expected_total, "on_tile_done call count mismatch");
        for (i, &(done, total)) in calls.iter().enumerate() {
            assert_eq!(done, i as u32 + 1, "done counter incorrect at step {i}");
            assert_eq!(total, expected_total, "total incorrect at step {i}");
        }
    }
}
