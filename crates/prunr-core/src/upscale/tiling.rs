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

#[derive(Debug, Clone, Copy)]
pub struct TilingConfig {
    /// Tile edge length in input pixels.
    pub tile_size: u32,
    /// Tile W and H are padded to a multiple of this before the model sees
    /// them (HAT-L window_size = 16).
    pub tile_multiple: Option<u32>,
    /// In input pixels; smoothstep-blended.
    pub overlap: u32,
    /// Tile starts land on a multiple of this (1: anywhere): a model that
    /// packs each 2 x 2 block of pixels into one (Real-ESRGAN x2) must see
    /// the photo's own blocks in every tile, or the seams show.
    pub align: u32,
}

/// One tile in input space, plus the padded size the model sees.
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

/// Overlapping tiles covering the image, all one size: per axis the
/// fewest tiles of at most `cfg.tile_size` that overlap their neighbours
/// by at least `cfg.overlap`, made equal and spread evenly. No tile is an
/// edge sliver: one would see little of the picture, and CoreML runs a
/// new shape ~0.5 s slower and keeps only the last one warm. With
/// `cfg.tile_multiple` each tile is padded bottom-right to that multiple,
/// and the output trimmed back to `w × h`.
pub fn plan_upscale_tiles(
    width: u32,
    height: u32,
    cfg: TilingConfig,
) -> Vec<UpscaleTilePlacement> {
    let TilingConfig { tile_size, tile_multiple, overlap, align } = cfg;
    let ((xs, w), (ys, h)) = (axis_tiles(width, tile_size, overlap, align), axis_tiles(height, tile_size, overlap, align));
    let (padded_w, padded_h) = match tile_multiple {
        Some(m) if m > 0 => (w.div_ceil(m) * m, h.div_ceil(m) * m),
        _ => (w, h),
    };
    ys.iter()
        .flat_map(|&y| xs.iter().map(move |&x| UpscaleTilePlacement { x, y, w, h, padded_w, padded_h }))
        .collect()
}

/// One axis of length `len`: the tiles' starts, each a multiple of
/// `align`, and their common length.
fn axis_tiles(len: u32, tile_size: u32, overlap: u32, align: u32) -> (Vec<u32>, u32) {
    if len <= tile_size {
        return (vec![0], len);
    }
    let step = tile_size.saturating_sub(overlap).max(1);
    // ≥ 2: len > tile_size
    let count = len.saturating_sub(overlap).div_ceil(step);
    // count × size ≥ len + (count - 1) × overlap, so neighbours overlap
    // by at least `overlap` and size ≤ tile_size.
    let size = (len + (count - 1) * overlap).div_ceil(count);
    let span = len - size;
    let align = align.max(1);
    let starts: Vec<u32> = (0..count).map(|i| i * span / (count - 1) / align * align).collect();
    // Starts only move back, so neighbours overlap at least as much; the
    // last tile grows by what its start gave up, to reach the edge.
    let size = len - starts[starts.len() - 1];
    (starts, size)
}

/// Upscale tile by tile, smoothstep-blending overlaps to hide seams; alpha
/// is Lanczos3-upscaled separately.
///
/// `run_tile(rgb_tile, padded_w, padded_h)` must return an RGB image of
/// `scale * padded_w × scale * padded_h`. Returns `Cancelled` before the
/// next tile once `progress` is cancelled.
#[cfg(test)]
fn upscale_tiled<F>(
    input: &RgbaImage,
    scale: u32,
    cfg: TilingConfig,
    run_tile: F,
    progress: &Progress,
) -> Result<RgbaImage, CoreError>
where
    F: Fn(&RgbImage, u32, u32) -> Result<RgbImage, CoreError>,
{
    upscale_tiled_as(Step::Upscaling, input, scale, cfg, run_tile, progress)
}

/// The smallest tile a run that ran out of memory tries: below it the
/// overlap is most of each tile.
const MIN_RETRY_TILE: u32 = 128;

/// The next tile down a ladder of halves, a multiple of `multiple`, no
/// smaller than `MIN_RETRY_TILE`: halves, so the sizes a provider compiles
/// for repeat from run to run.
pub fn smaller_tile(tile: u32, multiple: Option<u32>) -> Option<u32> {
    let m = multiple.unwrap_or(1).max(1);
    let half = tile / 2 / m * m;
    (half >= MIN_RETRY_TILE).then_some(half)
}

/// `upscale_tiled` at `cfg`'s tile, and when a tile runs out of memory on
/// a graphics card (`on_gpu`), again at each smaller tile that splits the
/// image further, until one fits or none is left; with the tile it ran on.
pub fn upscale_tiled_in_less_memory<F>(
    input: &RgbaImage,
    scale: u32,
    mut cfg: TilingConfig,
    on_gpu: bool,
    run_tile: F,
    progress: &Progress,
) -> Result<(RgbaImage, u32), CoreError>
where
    F: Fn(&RgbImage, u32, u32) -> Result<RgbImage, CoreError>,
{
    let (w, h) = input.dimensions();
    let mut step = Step::Upscaling;
    loop {
        match upscale_tiled_as(step, input, scale, cfg, &run_tile, progress) {
            Ok(out) => return Ok((out, cfg.tile_size)),
            Err(e) if on_gpu && e.is_out_of_memory() => {
                let tiles = plan_upscale_tiles(w, h, cfg).len();
                let Some(smaller) = smaller_tile(cfg.tile_size, cfg.tile_multiple)
                    .map(|tile_size| TilingConfig { tile_size, ..cfg })
                    .filter(|c| plan_upscale_tiles(w, h, *c).len() > tiles)
                else {
                    return Err(e);
                };
                tracing::warn!(%e, from = cfg.tile_size, to = smaller.tile_size, "upscale ran out of graphics memory; smaller tiles");
                cfg = smaller;
                step = Step::UpscalingInLessMemory;
            }
            Err(e) => return Err(e),
        }
    }
}

fn upscale_tiled_as<F>(
    step: Step,
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
    progress.step(step);
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

    super::alpha::take_input_alpha(&mut out_img, input);
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

/// The finished `tile` at `width × height` for its preview.
fn preview_tile(input: &RgbaImage, inferred: &RgbImage, tile: &UpscaleTilePlacement, scale: u32, width: u32, height: u32) -> Vec<u8> {
    // A padded tile is copied out first: the filter would pull the padding in.
    let unpadded = (tile.padded_w != tile.w || tile.padded_h != tile.h)
        .then(|| image::imageops::crop_imm(inferred, 0, 0, tile.w * scale, tile.h * scale).to_image());
    preview_with_alpha(input, unpadded.as_ref().unwrap_or(inferred), (tile.x, tile.y, tile.w, tile.h), width, height)
}

/// `rgb`, the model's output for the `rect` (x, y, w, h) of `input`, at
/// `width × height` for its preview: resampled with Lanczos3, with the
/// input's alpha. Costs one resample of the rect, a small share of its
/// inference.
pub(super) fn preview_with_alpha(input: &RgbaImage, rgb: &RgbImage, (x, y, w, h): (u32, u32, u32, u32), width: u32, height: u32) -> Vec<u8> {
    let rgb = crate::formats::resize_rgb_image_lanczos3(rgb, width, height);
    let alpha = GrayImage::from_fn(w, h, |ax, ay| image::Luma([input.get_pixel(x + ax, y + ay)[3]]));
    let alpha = crate::formats::resize_gray_lanczos3(&alpha, width, height);
    rgb.pixels().zip(alpha.pixels()).flat_map(|(c, a)| [c[0], c[1], c[2], a[0]]).collect()
}

fn extract_rgb_tile(input: &RgbaImage, tile: &UpscaleTilePlacement) -> RgbImage {
    let cropped = image::imageops::crop_imm(input, tile.x, tile.y, tile.w, tile.h).to_image();
    image::DynamicImage::ImageRgba8(cropped).to_rgb8()
}

/// Smoothstep weight for one axis. Sides without a neighbouring tile stay
/// at 1.0: tapering there would zero pixels no other tile covers.
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
        let tiles = plan_upscale_tiles(width, height, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16, align: 1 });

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

    /// Per axis the fewest tiles that overlap by `overlap`, all one size
    /// and spread evenly: no edge sliver, and no more pixels run than
    /// the overlaps need.
    #[test]
    fn tiles_are_equal_and_few() {
        let cfg = TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16, align: 1 };
        assert_eq!(plan_upscale_tiles(256, 256, cfg).len(), 1, "a picture one tile big is one tile");
        assert_eq!(axis_tiles(1000, 256, 16, 1), (vec![0, 196, 393, 590, 787], 213));
        assert_eq!(axis_tiles(4000, 512, 16, 1), (vec![0, 442, 885, 1327, 1770, 2213, 2655, 3098, 3541], 459));
        for len in [257, 496, 497, 700, 1000, 1024, 3000, 4000, 4032, 6000] {
            let (starts, size) = axis_tiles(len, 256, 16, 1);
            assert!(size <= 256, "{len}: {size}");
            assert_eq!((starts[0], starts.last().map(|s| s + size)), (0, Some(len)), "{len}: from edge to edge");
            assert!(starts.windows(2).all(|p| p[1] > p[0] && p[0] + size >= p[1] + 16), "{len}: {starts:?} × {size}");
            assert_eq!(starts.len() as u32, (len - 16).div_ceil(240), "{len}: the fewest tiles");
        }
        let tiles = plan_upscale_tiles(1000, 700, cfg);
        assert!(tiles.iter().all(|t| (t.w, t.h) == (tiles[0].w, tiles[0].h)), "one shape: {tiles:?}");
    }

    /// Pad-to-multiple round-trip: a 250×200 tile padded to 256×208 (nearest
    /// multiples of 16), inferred at 4× to 1024×832, then trimmed back to
    /// 1000×800 (250*4 × 200*4) must produce exactly the expected dimensions.
    #[test]
    fn pad_to_multiple_roundtrip_preserves_dimensions() {
        let tiles = plan_upscale_tiles(250, 200, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 0, align: 1 });
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
        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 0, align: 1 }, run_tile, &Progress::none())
            .expect("upscale_tiled failed");

        assert_eq!(result.width(), 250 * scale);
        assert_eq!(result.height(), 200 * scale);
    }

    /// Aligned tiles start on the multiple, still reach the far edge, and
    /// still overlap by at least the overlap asked.
    #[test]
    fn aligned_tiles_start_on_the_multiple_and_cover_the_axis() {
        for (len, tile, overlap, align) in [(1024, 512, 16, 2), (1000, 512, 64, 2), (777, 256, 32, 2), (1500, 512, 96, 4), (300, 512, 16, 2)] {
            let (starts, size) = axis_tiles(len, tile, overlap, align);
            assert!(starts.iter().all(|s| s % align == 0), "{len}/{tile}/{overlap}: {starts:?}");
            assert_eq!(starts.last().copied().unwrap_or(0) + size, len, "{len}/{tile}/{overlap}: reaches the edge");
            assert!(starts.windows(2).all(|w| w[0] + size >= w[1] + overlap), "{len}/{tile}/{overlap}: overlap kept");
            assert!(size < tile + align, "{len}/{tile}/{overlap}: {size}");
        }
    }

    /// Tiles go down by halves on a multiple of the model's window, and
    /// stop above the smallest worth running.
    #[test]
    fn a_smaller_tile_is_half_on_the_models_multiple() {
        assert_eq!(smaller_tile(512, Some(16)), Some(256));
        assert_eq!(smaller_tile(256, Some(16)), Some(128));
        assert_eq!(smaller_tile(128, Some(16)), None);
        assert_eq!(smaller_tile(500, Some(16)), Some(240));
        assert_eq!(smaller_tile(300, None), Some(150));
    }

    /// A card short of memory for a 512 px tile gets the image again in
    /// 256 px tiles, the same picture, its progress counted afresh for
    /// the new layout; on the processor, or with no smaller tile that
    /// splits the image further, the error stands.
    #[test]
    fn a_tile_out_of_memory_on_a_card_runs_again_in_smaller_tiles() {
        use std::sync::Arc;
        let scale = 2;
        let input = RgbaImage::from_fn(600, 400, |x, y| image::Rgba([(x % 256) as u8, (y % 256) as u8, 7, 255]));
        let upscale = |rgb: &RgbImage, pw: u32, ph: u32| RgbImage::from_fn(scale * pw, scale * ph, |x, y| *rgb.get_pixel(x / scale, y / scale));
        let fits_256 = |rgb: &RgbImage, pw: u32, ph: u32| -> Result<RgbImage, CoreError> {
            if pw.max(ph) > 256 {
                return Err(CoreError::Inference("upscale: inference failed: CUDA error: out of memory".into()));
            }
            Ok(upscale(rgb, pw, ph))
        };
        let cfg = TilingConfig { tile_size: 512, tile_multiple: Some(16), overlap: 16, align: 1 };
        let sink = Arc::new(crate::progress::RecordingSink::default());
        let (retried, tile) = upscale_tiled_in_less_memory(&input, scale, cfg, true, fits_256, &Progress::new(sink.clone())).expect("fits in 256 px tiles");
        assert_eq!(tile, 256, "the tile it ran on, for the next image to start at");
        let direct = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, ..cfg }, |t, w, h| Ok(upscale(t, w, h)), &Progress::none()).unwrap();
        assert!(retried == direct, "the same picture as 256 px tiles from the start");
        crate::progress::check_contract(&sink.updates()).expect("the retry is counted afresh");
        assert!(sink.updates().contains(&crate::ProgressUpdate::Step(Step::UpscalingInLessMemory)), "and says so");

        let on_cpu = upscale_tiled_in_less_memory(&input, scale, cfg, false, fits_256, &Progress::none());
        assert!(on_cpu.is_err_and(|e| e.is_out_of_memory()), "the processor's memory is not a card's");
        let small = RgbaImage::new(100, 100);
        let never = |_: &RgbImage, _: u32, _: u32| -> Result<RgbImage, CoreError> { Err(CoreError::Inference("out of memory".into())) };
        assert!(upscale_tiled_in_less_memory(&small, scale, cfg, true, never, &Progress::none()).is_err(), "one tile at any size: nothing smaller to try");
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

        let result = upscale_tiled(&input, scale, TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16, align: 1 }, run_tile, &Progress::none())
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
        let cfg = TilingConfig { tile_size: 256, tile_multiple: None, overlap: 16, align: 1 };
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
        let cfg = TilingConfig { tile_size: 256, tile_multiple: Some(16), overlap: 16, align: 1 };
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
        let alpha = crate::upscale::upscale_alpha_lanczos3(input, out_w, out_h);
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
        let cfg = TilingConfig { tile_size: 128, tile_multiple: Some(16), overlap: 16, align: 1 };
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
        let cfg = TilingConfig { tile_size: 64, tile_multiple: None, overlap: 24, align: 1 };
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
        let result = upscale_tiled(&input, 1, TilingConfig { tile_size: 32, tile_multiple: None, overlap: 0, align: 1 }, run_tile, &Progress::none().with_cancel(flag));
        assert!(matches!(result, Err(CoreError::Cancelled)) && !ran.get());
    }
}
