//! Tile-based upscale with overlap-blend.
//!
//! Peak working-set RAM (per dispatch, scale=4, 512 tiles). Numbers are
//! for the OUTPUT buffers; the input image and the model session are
//! separate.
//!
//! | Input        | Band f32 (one tile row of sums + weights) | Output RGBA | Alpha | Per-tile transient¹ | Total    |
//! |--------------|-------------------------------------------|-------------|-------|---------------------|----------|
//! | 1024 × 1024  | 128 MB                                    | 64 MB       | 16 MB | ~10 MB              | ~218 MB  |
//! | 2048 × 2048  | 256 MB                                    | 256 MB      | 64 MB | ~10 MB              | ~586 MB  |
//! | 4096 × 4096  | 512 MB                                    | 1024 MB     | 256 MB| ~10 MB              | ~1802 MB |
//!
//! ¹ Per-tile transient = padded RGB tile + inferred RGB output (lives only
//! during one `run_tile` call, dropped before the next tile). The blend
//! reads `inferred` directly with an offset (no separate trimmed copy).
//!
//! The band is freed before the alpha is made, so the total overstates
//! by the alpha. The upscale dispatch refuses to start when free RAM is
//! below this peak (`super::peak_bytes`).

use crate::progress::{Progress, Step, TileRect, Unit};

/// What `upscale_tiled` holds at its peak, besides the input and the
/// model: one tile row of colour sums and weights (16 bytes a pixel),
/// the output (4) and, as it finishes, the enlarged alpha (1).
pub(super) fn tiled_peak_bytes(in_w: u32, in_h: u32, scale: u32, tile_size: u32) -> u64 {
    let out_w = u64::from(in_w) * u64::from(scale);
    let out_h = u64::from(in_h) * u64::from(scale);
    let band_rows = u64::from(tile_size.min(in_h)) * u64::from(scale);
    band_rows * out_w * 16 + out_w * out_h * 5
}

use image::{GrayImage, RgbImage, RgbaImage};

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

    let (xs, ys) = (axis_starts(width, tile_size, step), axis_starts(height, tile_size, step));

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

/// Tile starts along one axis of length `len`, the last one moved in so
/// its tile reaches the far edge.
fn axis_starts(len: u32, tile_size: u32, step: u32) -> Vec<u32> {
    let mut starts: Vec<u32> = (0..len).step_by(step as usize).collect();
    if let Some(&last) = starts.last() {
        if last + tile_size < len {
            starts.push(len.saturating_sub(tile_size));
        }
    }
    // The edge tile may coincide with an earlier step.
    starts.sort_unstable();
    starts.dedup();
    starts
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
/// Reports the tile layout and each tile through `progress`, and returns
/// `Err(CoreError::Cancelled)` before the next tile once it is cancelled.
pub fn upscale_tiled<F>(
    input: &RgbaImage,
    scale: u32,
    cfg: TilingConfig,
    run_tile: F,
    progress: &Progress,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(&RgbImage, u32, u32) -> Result<RgbImage, CoreError>,
{
    let TilingConfig { tile_size, overlap, .. } = cfg;
    let (in_w, in_h) = input.dimensions();
    let out_w = in_w * scale;
    let out_h = in_h * scale;

    let tiles = plan_upscale_tiles(in_w, in_h, cfg);
    let total = tiles.len() as u32;
    progress.step(Step::Upscaling);
    progress.tiles(tiles.iter().map(|t| TileRect::of_pixels(t.x, t.y, t.w, t.h, in_w, in_h)).collect());
    progress.inner(0, total, Unit::Tile);

    // Tiles run a row at a time, so output rows above the next tile row
    // are final: the accumulator holds one tile row of output, finished
    // into `out_img` and slid up as the rows advance.
    let row_len = out_w as usize;
    let band_rows = (tile_size.min(in_h) * scale) as usize;
    let mut band = Band { top: 0, rgb: vec![0.0; band_rows * row_len * 3], weight: vec![0.0; band_rows * row_len] };
    let mut out_img = RgbaImage::new(out_w, out_h);

    // Sequential: ort sessions are Mutex-guarded internally and nested
    // rayon inside the subprocess worker path deadlocks.
    for (tile_idx, tile) in tiles.iter().enumerate() {
        progress.check_cancelled()?;
        progress.tile(tile_idx as u32, false);

        let rgb_tile = extract_rgb_tile(input, tile);
        let padded_tile = crate::formats::pad_mirrored(&rgb_tile, tile.padded_w, tile.padded_h);
        let inferred = run_tile(&padded_tile, tile.padded_w, tile.padded_h)?;
        band.finish_above(tile.y * scale, &mut out_img);

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
                let out_idx = ((out_tile_y + py - band.top) as usize) * row_len + (out_x as usize);

                let tile_pixel = inferred.get_pixel(px, py).0;
                band.rgb[out_idx * 3] += tile_pixel[0] as f32 * w;
                band.rgb[out_idx * 3 + 1] += tile_pixel[1] as f32 * w;
                band.rgb[out_idx * 3 + 2] += tile_pixel[2] as f32 * w;
                band.weight[out_idx] += w;
            }
        }

        progress.tile_preview((tile.x, tile.y, tile.w, tile.h), (in_w, in_h), |w, h| preview_tile(input, &inferred, tile, scale, w, h));
        drop(inferred);
        progress.inner(tile_idx as u32 + 1, total, Unit::Tile);
        progress.tile(tile_idx as u32, true);
    }
    progress.step(Step::Finishing);
    band.finish_above(out_h, &mut out_img);
    drop(band);

    let alpha = upscale_alpha_lanczos3(input, out_w, out_h);
    for (px, a) in out_img.pixels_mut().zip(alpha.as_raw()) {
        px.0[3] = *a;
    }
    Ok(out_img)
}

/// The weighted colour sums of the output rows still taking tiles, from
/// output row `top` down.
struct Band {
    top: u32,
    rgb: Vec<f32>,
    weight: Vec<f32>,
}

impl Band {
    /// Write the rows above output row `until` into `out` and slide the
    /// band down to start there. No tile reaches them any more.
    fn finish_above(&mut self, until: u32, out: &mut RgbaImage) {
        let Some(done) = until.checked_sub(self.top).filter(|&n| n > 0) else { return };
        let row_len = out.width() as usize;
        let done_px = (done as usize * row_len).min(self.weight.len());
        let start = self.top as usize * row_len * 4;
        let rows = &mut out.as_mut()[start..start + done_px * 4];
        for ((px, rgb), w) in rows.chunks_exact_mut(4).zip(self.rgb.chunks_exact(3)).zip(&self.weight) {
            // Guard against zero weight on unreachable pixels — f32 division
            // by 0 would otherwise produce inf and saturate the channel.
            let inv_w = 1.0 / w.max(1e-6);
            for k in 0..3 {
                px[k] = crate::math::round_u8(rgb[k] * inv_w);
            }
        }
        self.rgb.copy_within(done_px * 3.., 0);
        let kept = self.rgb.len() - done_px * 3;
        self.rgb[kept..].fill(0.0);
        self.weight.copy_within(done_px.., 0);
        let kept = self.weight.len() - done_px;
        self.weight[kept..].fill(0.0);
        self.top = until;
    }
}

/// The finished `tile` at `width × height` for its preview: the model's
/// output resampled with Lanczos3, with the input's alpha. Costs one
/// resample of the tile, a small share of its inference.
fn preview_tile(input: &RgbaImage, inferred: &RgbImage, tile: &UpscaleTilePlacement, scale: u32, width: u32, height: u32) -> Vec<u8> {
    // A padded tile is copied out first: the filter would pull the padding in.
    let unpadded = (tile.padded_w != tile.w || tile.padded_h != tile.h)
        .then(|| image::imageops::crop_imm(inferred, 0, 0, tile.w * scale, tile.h * scale).to_image());
    let rgb = crate::formats::resize_rgb_image_lanczos3(unpadded.as_ref().unwrap_or(inferred), width, height);
    let alpha = GrayImage::from_fn(tile.w, tile.h, |ax, ay| image::Luma([input.get_pixel(tile.x + ax, tile.y + ay)[3]]));
    let alpha = crate::formats::resize_gray_lanczos3(&alpha, width, height);
    rgb.pixels().zip(alpha.pixels()).flat_map(|(c, a)| [c[0], c[1], c[2], a[0]]).collect()
}

fn extract_rgb_tile(input: &RgbaImage, tile: &UpscaleTilePlacement) -> RgbImage {
    let cropped = image::imageops::crop_imm(input, tile.x, tile.y, tile.w, tile.h).to_image();
    image::DynamicImage::ImageRgba8(cropped).to_rgb8()
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
    #[test]
    fn the_peak_matches_the_table_and_does_not_overflow() {
        let mb = |b: u64| b / (1024 * 1024);
        assert_eq!(mb(tiled_peak_bytes(1024, 1024, 4, 512)), 208, "the table's row less the per-tile scratch");
        assert_eq!(mb(tiled_peak_bytes(4096, 4096, 4, 512)), 1792);
        assert_eq!(tiled_peak_bytes(50_000, 50_000, 2, 512), 1024 * 100_000 * 16 + 100_000 * 100_000 * 5);
    }

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
        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 0 }, run_tile, &Progress::none())
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

        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 }, run_tile, &Progress::none())
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

    /// The tiler reports a contract-clean run: the step, the layout of
    /// every tile inside the image, the count from zero, and each tile
    /// started, counted and finished in order.
    #[test]
    fn upscale_tiled_reports_every_tile() {
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        let scale = 4u32;
        let input = RgbaImage::from_pixel(512, 512, image::Rgba([0, 0, 0, 255]));
        let cfg = TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16 };
        let expected_total = plan_upscale_tiles(512, 512, cfg).len() as u32;
        let run_tile = |rgb: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            Ok(RgbImage::from_fn(scale * pw, scale * ph, |x, y| *rgb.get_pixel(x / scale, y / scale)))
        };
        let sink = std::sync::Arc::new(RecordingSink::default());
        upscale_tiled(&input, scale, cfg, run_tile, &Progress::new(sink.clone())).expect("upscale_tiled failed");

        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        let counts: Vec<u32> = updates.iter().filter_map(|u| match u {
            ProgressUpdate::Inner { done, total, .. } if *total == expected_total => Some(*done),
            _ => None,
        }).collect();
        assert_eq!(counts, (0..=expected_total).collect::<Vec<_>>(), "the count shows before the first tile runs");
        assert!(updates.iter().any(|u| matches!(u, ProgressUpdate::Tiles(t) if t.len() as u32 == expected_total)));
    }

    /// Asked for previews, the tiler sends each finished tile once, in
    /// its square of a preview at most the asked size, in the model's
    /// colours with the input's alpha; not asked, it sends none.
    #[test]
    fn previews_show_each_finished_tile_in_its_square() {
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        let scale = 4u32;
        let input = RgbaImage::from_pixel(600, 300, image::Rgba([0, 0, 0, 128]));
        let cfg = TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 16 };
        let tiles = plan_upscale_tiles(600, 300, cfg).len();
        let run_tile = |_: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            Ok(RgbImage::from_pixel(scale * pw, scale * ph, image::Rgb([10, 200, 30])))
        };
        let sink = std::sync::Arc::new(RecordingSink::default());
        upscale_tiled(&input, scale, cfg, run_tile, &Progress::new(sink.clone()).with_tile_previews(300)).expect("upscale_tiled failed");
        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        let previews: Vec<_> = updates.iter().filter_map(|u| match u { ProgressUpdate::TilePreview(p) => Some(p), _ => None }).collect();
        assert_eq!(previews.len(), tiles, "one preview per tile");
        let mut covered = vec![false; 300 * 150];
        for p in &previews {
            assert_eq!((p.full_width, p.full_height), (300, 150), "the longer side fits the asked size");
            for (i, px) in p.rgba.chunks(4).enumerate() {
                assert!(px[0].abs_diff(10) <= 1 && px[1].abs_diff(200) <= 1 && px[2].abs_diff(30) <= 1 && px[3].abs_diff(128) <= 1, "{px:?}");
                let (cx, cy) = (p.x + i as u32 % p.width, p.y + i as u32 / p.width);
                covered[(cy * 300 + cx) as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "the squares cover the preview");

        let quiet = std::sync::Arc::new(RecordingSink::default());
        upscale_tiled(&input, scale, cfg, run_tile, &Progress::new(quiet.clone())).expect("upscale_tiled failed");
        assert!(!quiet.updates().iter().any(|u| matches!(u, ProgressUpdate::TilePreview(_))), "no preview unless asked");
    }

    /// The plain whole-image accumulator: the oracle.
    fn full_buffer_reference(input: &RgbaImage, scale: u32, cfg: TilingConfig, run_tile: impl Fn(&RgbImage, u32, u32) -> RgbImage) -> RgbaImage {
        let (in_w, in_h) = input.dimensions();
        let (out_w, out_h) = (in_w * scale, in_h * scale);
        let mut acc = vec![0f32; (out_w * out_h * 3) as usize];
        let mut wts = vec![0f32; (out_w * out_h) as usize];
        for tile in plan_upscale_tiles(in_w, in_h, cfg) {
            let rgb = extract_rgb_tile(input, &tile);
            let padded = crate::formats::pad_mirrored(&rgb, tile.padded_w, tile.padded_h);
            let inferred = run_tile(&padded, tile.padded_w, tile.padded_h);
            let (tw, th, edge) = (tile.w * scale, tile.h * scale, cfg.overlap * scale);
            for py in 0..th {
                for px in 0..tw {
                    let w = blend_weight_sided(px, edge, tw, tile.x > 0, tile.x + tile.w < in_w) * blend_weight_sided(py, edge, th, tile.y > 0, tile.y + tile.h < in_h);
                    let i = ((tile.y * scale + py) * out_w + tile.x * scale + px) as usize;
                    let c = inferred.get_pixel(px, py).0;
                    for k in 0..3 {
                        acc[i * 3 + k] += c[k] as f32 * w;
                    }
                    wts[i] += w;
                }
            }
        }
        let alpha = upscale_alpha_lanczos3(input, out_w, out_h);
        RgbaImage::from_fn(out_w, out_h, |x, y| {
            let i = (y * out_w + x) as usize;
            let inv = 1.0 / wts[i].max(1e-6);
            let c = |k: usize| crate::math::round_u8(acc[i * 3 + k] * inv);
            image::Rgba([c(0), c(1), c(2), alpha.get_pixel(x, y)[0]])
        })
    }

    /// Finishing rows a band at a time gives the very bytes the whole-image
    /// accumulator gave: several tile rows, overlap, padding, the last row
    /// moved in, and tiles that disagree where they overlap.
    #[test]
    fn the_banded_accumulator_matches_the_whole_image_one() {
        let (w, h, scale) = (300u32, 700u32, 2u32);
        let input = RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 7 + y) as u8, (y * 3) as u8, (x ^ y) as u8, (x + y * 5) as u8]));
        let cfg = TilingConfig { tile_size: 128, tile_multiple: Some(16), overlap: 16 };
        // Each tile shifts its colours by its first pixel, so overlaps blend.
        let run = |rgb: &RgbImage, pw: u32, ph: u32| {
            let shift = rgb.get_pixel(0, 0)[0] / 8;
            RgbImage::from_fn(scale * pw, scale * ph, |x, y| image::Rgb(rgb.get_pixel(x / scale, y / scale).0.map(|v| v.wrapping_add(shift))))
        };
        let banded = upscale_tiled(&input, scale, cfg, |t, pw, ph| Ok(run(t, pw, ph)), &Progress::none()).unwrap();
        assert!(banded == full_buffer_reference(&input, scale, cfg, run), "bit-identical");
    }

    /// A flat image comes back flat across the overlaps: the blend's
    /// weighted mean is not quite the value in f32, and truncating it
    /// would land one low there.
    #[test]
    fn overlaps_give_back_a_flat_colour_exactly() {
        let cfg = TilingConfig { tile_size: 64, tile_multiple: None, overlap: 24 };
        let run = |rgb: &RgbImage, pw: u32, ph: u32| Ok(RgbImage::from_fn(2 * pw, 2 * ph, |x, y| *rgb.get_pixel(x / 2, y / 2)));
        for v in [1u8, 3, 77, 101, 199, 254] {
            let input = RgbaImage::from_pixel(150, 150, image::Rgba([v, v, v, 255]));
            let out = upscale_tiled(&input, 2, cfg, run, &Progress::none()).unwrap();
            assert!(out.pixels().all(|p| p[0] == v), "{v} drifted");
        }
    }

    #[test]
    fn a_cancelled_tiler_stops_before_the_next_tile() {
        let input = RgbaImage::from_pixel(64, 64, image::Rgba([0, 0, 0, 255]));
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let ran = std::cell::Cell::new(false);
        let run_tile = |rgb: &RgbImage, _: u32, _: u32| { ran.set(true); Ok(rgb.clone()) };
        let result = upscale_tiled(&input, 1, TilingConfig { tile_size: 32, tile_multiple: None, overlap: 0 }, run_tile, &Progress::none().with_cancel(flag));
        assert!(matches!(result, Err(CoreError::Cancelled)) && !ran.get());
    }
}
