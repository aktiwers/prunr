use image::{DynamicImage, GrayImage, RgbaImage};
use ndarray::ArrayView4;
use rayon::prelude::*;

use crate::brush::MaskCorrection;
use crate::formats::resize_gray_lanczos3;
use crate::guided_filter::guided_filter_alpha;
use crate::types::{MaskSettings, ModelKind};

/// Bundle of stable inputs shared by every postprocess entry point.
/// Adding a new optional field here is preferable to adding another
/// positional parameter to six functions.
#[derive(Clone, Copy)]
pub struct PostprocessOpts<'a> {
    pub mask_settings: &'a MaskSettings,
    pub model: ModelKind,
    pub correction: Option<&'a MaskCorrection>,
}

impl<'a> PostprocessOpts<'a> {
    pub fn new(mask_settings: &'a MaskSettings, model: ModelKind) -> Self {
        Self { mask_settings, model, correction: None }
    }
    pub fn with_correction(mut self, correction: Option<&'a MaskCorrection>) -> Self {
        self.correction = correction;
        self
    }
}

/// Postprocess raw ONNX model output into a transparent RGBA image.
/// Allocates the RGBA buffer once and reuses it for guided filter (if enabled)
/// and final mask application — avoids two 4×width×height allocations.
pub fn postprocess(raw: ArrayView4<f32>, original: &DynamicImage, opts: &PostprocessOpts<'_>) -> RgbaImage {
    let mut rgba = original.to_rgba8();
    let mask = tensor_to_mask_with_rgba(raw, &rgba, opts);
    apply_mask_inplace(&mut rgba, &mask);
    apply_fill_style(&mut rgba, opts.mask_settings.fill_style);
    apply_bg_effect(&mut rgba, original, opts.mask_settings.bg_effect);
    rgba
}

/// Postprocess from a flat f32 tensor slice. Used by subprocess paths where
/// tensor data arrives via IPC as a Vec<f32> that needs reshaping to [1,1,H,W].
pub fn postprocess_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &DynamicImage,
    opts: &PostprocessOpts<'_>,
) -> Result<RgbaImage, crate::types::CoreError> {
    let view = ArrayView4::from_shape((1, 1, tensor_h, tensor_w), tensor)
        .map_err(|e| crate::types::CoreError::Inference(format!("Tensor reshape: {e}")))?;
    Ok(postprocess(view, original, opts))
}

/// Flat-slice variant of `tensor_to_mask`. Reshapes `[1,1,H,W]` and forwards.
pub fn tensor_to_mask_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &DynamicImage,
    opts: &PostprocessOpts<'_>,
) -> Result<GrayImage, crate::types::CoreError> {
    let view = ArrayView4::from_shape((1, 1, tensor_h, tensor_w), tensor)
        .map_err(|e| crate::types::CoreError::Inference(format!("Tensor reshape: {e}")))?;
    Ok(tensor_to_mask(view, original, opts))
}

/// Convert raw ONNX tensor to a full-resolution grayscale mask (Tier 2).
/// Applies normalization, gamma, threshold, resize, edge shift, and guided filter.
pub fn tensor_to_mask(raw: ArrayView4<f32>, original: &DynamicImage, opts: &PostprocessOpts<'_>) -> GrayImage {
    // Materializing rgba here is wasteful when refine_edges is false; callers on
    // the hot path should use `postprocess()` which shares the RGBA buffer.
    let rgba = if opts.mask_settings.refine_edges { Some(original.to_rgba8()) } else { None };
    tensor_to_mask_core(raw, original.width(), original.height(), rgba.as_ref(), opts)
}

/// Same as `tensor_to_mask` but reuses an already-materialized RGBA buffer.
fn tensor_to_mask_with_rgba(raw: ArrayView4<f32>, rgba: &RgbaImage, opts: &PostprocessOpts<'_>) -> GrayImage {
    tensor_to_mask_core(raw, rgba.width(), rgba.height(), Some(rgba), opts)
}

fn tensor_to_mask_core(raw: ArrayView4<f32>, ow: u32, oh: u32, rgba_for_guided: Option<&RgbaImage>, opts: &PostprocessOpts<'_>) -> GrayImage {
    let mask_settings = opts.mask_settings;
    let model = opts.model;
    let correction = opts.correction;
    let pred = raw.slice(ndarray::s![0, 0, .., ..]);

    let use_sigmoid = matches!(model, ModelKind::BiRefNetLite);

    // rembg models need min-max stats; BiRefNet uses sigmoid instead
    let (mi, range, uniform_val) = if !use_sigmoid {
        let (mi, ma) = pred.iter().cloned().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        );
        let r = ma - mi;
        if r < 1e-6 {
            // Uniform output — use the absolute value to decide:
            // rembg models output ~0 for background, ~1 for foreground
            // after min-max normalization. A uniform value > 0.5 means
            // "everything is foreground" → full opacity.
            (mi, 1.0, Some(if ma > 0.5 { 1.0f32 } else { 0.0 }))
        } else {
            (mi, r, None)
        }
    } else {
        (0.0, 1.0, None)
    };

    let (sh, sw) = (pred.nrows(), pred.ncols());
    let contiguous;
    let pred_slice = match pred.as_slice() {
        Some(s) => s,
        // invariant: as_standard_layout() produces a contiguous view, so as_slice() is Some.
        None => { contiguous = pred.as_standard_layout(); contiguous.as_slice().unwrap() }
    };
    let gamma = mask_settings.gamma;
    let threshold = mask_settings.threshold;

    // Short-circuit: uniform output → fill with constant, skip per-pixel loop
    let mut mask_buf = if let Some(uv) = uniform_val {
        let mut val = uv;
        if gamma != 1.0 { val = val.powf(gamma); }
        if let Some(t) = threshold { val = if val >= t { 1.0 } else { 0.0 }; }
        vec![(val * 255.0) as u8; sw * sh]
    } else {
        let inv_range = 1.0 / range;
        let mut buf = vec![0u8; sw * sh];
        for i in 0..sh * sw {
            let raw_val = pred_slice[i];
            let mut val = if use_sigmoid {
                1.0 / (1.0 + (-raw_val).exp())
            } else {
                ((raw_val - mi) * inv_range).clamp(0.0, 1.0)
            };

            if gamma != 1.0 {
                val = val.powf(gamma);
            }
            if let Some(t) = threshold {
                val = if val >= t { 1.0 } else { 0.0 };
            }

            buf[i] = (val * 255.0) as u8;
        }
        buf
    };

    // Brush correction runs at model resolution, BEFORE resize/guided
    // filter, so the guided filter snaps stroke edges to the image's
    // color edges.
    if let Some(corr) = correction {
        crate::brush::apply_correction(&mut mask_buf, corr);
    }

    let mask = GrayImage::from_raw(sw as u32, sh as u32, mask_buf)
        .expect("mask buffer size matches dimensions");

    // Resize mask back to original dimensions (SIMD-accelerated Lanczos3)
    let mut mask = resize_gray_lanczos3(&mask, ow, oh);

    // Edge shift: positive erodes (shrinks foreground), negative dilates (expands it)
    if mask_settings.edge_shift.abs() > 0.01 {
        apply_edge_shift(&mut mask, mask_settings.edge_shift);
    }

    if mask_settings.refine_edges {
        if let Some(rgba) = rgba_for_guided {
            mask = guided_filter_alpha(
                rgba,
                &mask,
                mask_settings.guided_radius,
                mask_settings.guided_epsilon,
            );
        }
    }

    // Feather runs AFTER refine_edges: guided filter snaps the mask to color
    // edges first (sharpening); feather is the final softening pass on top.
    // Running feather first and then refine would have them fight each other.
    if mask_settings.feather >= 0.1 {
        feather_mask(&mut mask, mask_settings.feather);
    }

    mask
}

/// Apply a grayscale mask as the alpha channel on the original image (Tier 3).
pub fn apply_mask(original: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let mut rgba = original.to_rgba8();
    apply_mask_inplace(&mut rgba, mask);
    rgba
}

/// Transform the RGB channels of `rgba` according to `style`. Alpha is
/// preserved. Spatial variants (`Pixelate`) allocate a scratch copy of the
/// buffer so reads don't race writes; per-pixel variants mutate in place.
pub fn apply_fill_style(rgba: &mut RgbaImage, style: crate::types::FillStyle) {
    use crate::types::FillStyle;
    match style {
        FillStyle::None => {}
        FillStyle::Desaturate => {
            for p in rgba.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                p.0[0] = y; p.0[1] = y; p.0[2] = y;
            }
        }
        FillStyle::Invert => {
            for p in rgba.pixels_mut() {
                p.0[0] = 255 - p.0[0];
                p.0[1] = 255 - p.0[1];
                p.0[2] = 255 - p.0[2];
            }
        }
        FillStyle::Duotone { dark, light } => {
            for p in rgba.pixels_mut() {
                let t = luma_u8(p.0[0], p.0[1], p.0[2]) as u16;
                let inv = 255 - t;
                p.0[0] = ((dark[0] as u16 * inv + light[0] as u16 * t) / 255) as u8;
                p.0[1] = ((dark[1] as u16 * inv + light[1] as u16 * t) / 255) as u8;
                p.0[2] = ((dark[2] as u16 * inv + light[2] as u16 * t) / 255) as u8;
            }
        }
        FillStyle::Sepia => {
            for p in rgba.pixels_mut() {
                let (r, g, b) = (p.0[0] as u32, p.0[1] as u32, p.0[2] as u32);
                // Standard sepia coefficients (scaled ×1000 for integer math).
                p.0[0] = ((r * 393 + g * 769 + b * 189) / 1000).min(255) as u8;
                p.0[1] = ((r * 349 + g * 686 + b * 168) / 1000).min(255) as u8;
                p.0[2] = ((r * 272 + g * 534 + b * 131) / 1000).min(255) as u8;
            }
        }
        FillStyle::Threshold { level } => {
            for p in rgba.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                let v = if y >= level { 255 } else { 0 };
                p.0[0] = v; p.0[1] = v; p.0[2] = v;
            }
        }
        FillStyle::Posterize { levels } => {
            let n = levels.max(2) as u16 - 1;
            for p in rgba.pixels_mut() {
                for i in 0..3 {
                    let v = p.0[i] as u16;
                    p.0[i] = ((v * n / 255) * 255 / n) as u8;
                }
            }
        }
        FillStyle::Solarize { pivot } => {
            for p in rgba.pixels_mut() {
                for i in 0..3 {
                    if p.0[i] >= pivot {
                        p.0[i] = 255 - p.0[i];
                    }
                }
            }
        }
        FillStyle::HueShift { degrees } => {
            for p in rgba.pixels_mut() {
                let (h, s, v) = rgb_to_hsv(p.0[0], p.0[1], p.0[2]);
                let new_h = (h as i32 + degrees as i32).rem_euclid(360) as u16;
                let (r, g, b) = hsv_to_rgb(new_h, s, v);
                p.0[0] = r; p.0[1] = g; p.0[2] = b;
            }
        }
        FillStyle::Saturate { percent } => {
            // Move each channel toward/away from luma by `percent / 100`.
            // percent=0 → grayscale, 100 → unchanged, >100 → punchy.
            let factor = percent.min(300) as i32;
            for p in rgba.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]) as i32;
                for i in 0..3 {
                    let v = p.0[i] as i32;
                    let shifted = y + ((v - y) * factor / 100);
                    p.0[i] = shifted.clamp(0, 255) as u8;
                }
            }
        }
        FillStyle::ColorSplash { keep_hue, tolerance } => {
            let keep = (keep_hue % 360) as i32;
            let tol = tolerance.min(180) as i32;
            for p in rgba.pixels_mut() {
                let (h, _, _) = rgb_to_hsv(p.0[0], p.0[1], p.0[2]);
                let dist = hue_distance(h as i32, keep);
                if dist > tol {
                    let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                    p.0[0] = y; p.0[1] = y; p.0[2] = y;
                }
            }
        }
        FillStyle::Pixelate { block_size } => {
            if block_size < 2 { return; }
            let (w, h) = rgba.dimensions();
            // Scratch copy: reads sample the block top-left, writes fill the
            // whole block. In-place would race the read once the first block
            // fills forward.
            let src = rgba.clone();
            for y in 0..h {
                for x in 0..w {
                    let bx = (x / block_size) * block_size;
                    let by = (y / block_size) * block_size;
                    let sample = src.get_pixel(bx, by);
                    let dst = rgba.get_pixel_mut(x, y);
                    dst.0[0] = sample.0[0];
                    dst.0[1] = sample.0[1];
                    dst.0[2] = sample.0[2];
                }
            }
        }
        FillStyle::CrossProcess { shadow, highlight } => {
            // Split-tone by luma: pixels below 128 bend toward `shadow`,
            // above bend toward `highlight`. Preserves midtones, lifts
            // shadows + warms highlights (or whatever the user picked).
            for p in rgba.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                let (target, t) = if y < 128 {
                    (shadow, (128 - y) as u16) // 0..=128
                } else {
                    (highlight, (y - 128) as u16) // 0..=127
                };
                let w = t.min(128) * 2; // scale to 0..=256 then clamp
                let w = w.min(255);
                let inv = 255 - w;
                p.0[0] = ((p.0[0] as u16 * inv + target[0] as u16 * w) / 255) as u8;
                p.0[1] = ((p.0[1] as u16 * inv + target[1] as u16 * w) / 255) as u8;
                p.0[2] = ((p.0[2] as u16 * inv + target[2] as u16 * w) / 255) as u8;
            }
        }
        FillStyle::ChannelSwap { variant } => {
            use crate::types::ChannelSwapVariant;
            for p in rgba.pixels_mut() {
                let [r, g, b, _] = p.0;
                let (nr, ng, nb) = match variant {
                    ChannelSwapVariant::Grb => (g, r, b),
                    ChannelSwapVariant::Brg => (b, r, g),
                    ChannelSwapVariant::Rbg => (r, b, g),
                    ChannelSwapVariant::Bgr => (b, g, r),
                    ChannelSwapVariant::Gbr => (g, b, r),
                };
                p.0[0] = nr; p.0[1] = ng; p.0[2] = nb;
            }
        }
        FillStyle::Halftone { dot_spacing } => {
            // Classic halftone: overlay a lattice where dot radius scales
            // inversely with luma. `dot_spacing` = centre-to-centre pitch.
            // Uses max pitch clamp to keep the test suite well-defined at
            // corner cases.
            let spacing = dot_spacing.clamp(2, 32);
            let (w, h) = rgba.dimensions();
            let half = (spacing / 2) as i32;
            for y in 0..h {
                for x in 0..w {
                    let px = rgba.get_pixel(x, y);
                    let luma = luma_u8(px.0[0], px.0[1], px.0[2]);
                    // Dot radius squared: dark (luma=0) → full cell, light → tiny.
                    let max_r_sq = (half * half) as u32;
                    let r_sq = max_r_sq * (255 - luma as u32) / 255;
                    let cx = ((x as i32) / spacing as i32) * spacing as i32 + half;
                    let cy = ((y as i32) / spacing as i32) * spacing as i32 + half;
                    let dx = x as i32 - cx;
                    let dy = y as i32 - cy;
                    let dist_sq = (dx * dx + dy * dy) as u32;
                    let inside = dist_sq <= r_sq;
                    let dst = rgba.get_pixel_mut(x, y);
                    let v = if inside { 0 } else { 255 };
                    dst.0[0] = v; dst.0[1] = v; dst.0[2] = v;
                }
            }
        }
        FillStyle::GradientMap { stops } => {
            // Map luma 0..=255 through 4 colour stops at 0, 85, 170, 255.
            // Linear interp between adjacent stops.
            for p in rgba.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]) as u16;
                let (lo, hi, t) = if y <= 85 {
                    (stops[0], stops[1], y * 255 / 85)
                } else if y <= 170 {
                    (stops[1], stops[2], (y - 85) * 255 / 85)
                } else {
                    (stops[2], stops[3], (y - 170) * 255 / 85)
                };
                let t = t.min(255);
                let inv = 255 - t;
                p.0[0] = ((lo[0] as u16 * inv + hi[0] as u16 * t) / 255) as u8;
                p.0[1] = ((lo[1] as u16 * inv + hi[1] as u16 * t) / 255) as u8;
                p.0[2] = ((lo[2] as u16 * inv + hi[2] as u16 * t) / 255) as u8;
            }
        }
    }
}

/// Rec. 709 luma of an 8-bit RGB triple. Used by every grayscale / duotone /
/// threshold / splash variant.
#[inline]
fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 2126 + g as u32 * 7152 + b as u32 * 722) / 10000) as u8
}

/// Composite a derived backdrop into the transparent areas of `rgba` per the
/// selected `effect`. Output becomes fully opaque where the effect applies.
/// Always allocates a scratch backdrop the size of the source — acceptable
/// cost because bg effects only run at postprocess tier, never on the
/// per-tick live-preview path (live preview keeps transparency so the
/// canvas's GPU-rect bg render stays instant).
pub fn apply_bg_effect(rgba: &mut RgbaImage, source: &DynamicImage, effect: crate::types::BgEffect) {
    use crate::types::BgEffect;
    let backdrop: RgbaImage = match effect {
        BgEffect::None => return,
        BgEffect::BlurredSource { radius } => {
            let r = radius.clamp(1, 64) as f32;
            image::imageops::blur(&source.to_rgba8(), r)
        }
        BgEffect::InvertedSource => {
            let mut img = source.to_rgba8();
            for p in img.pixels_mut() {
                p.0[0] = 255 - p.0[0];
                p.0[1] = 255 - p.0[1];
                p.0[2] = 255 - p.0[2];
            }
            img
        }
        BgEffect::DesaturatedSource => {
            let mut img = source.to_rgba8();
            for p in img.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                p.0[0] = y; p.0[1] = y; p.0[2] = y;
            }
            img
        }
    };

    // Alpha-blend foreground over backdrop, force fully opaque output where
    // the subject mask left partial alpha. Anywhere the subject was fully
    // opaque already, the formula reduces to a no-op.
    debug_assert_eq!(rgba.dimensions(), backdrop.dimensions());
    let raw = rgba.as_mut();
    let bd = backdrop.as_raw();
    for (i, px) in raw.chunks_exact_mut(4).enumerate() {
        let a = px[3] as u16;
        if a == 255 { continue; }
        let inv = 255 - a;
        let bd_i = i * 4;
        px[0] = ((px[0] as u16 * a + bd[bd_i]     as u16 * inv) / 255) as u8;
        px[1] = ((px[1] as u16 * a + bd[bd_i + 1] as u16 * inv) / 255) as u8;
        px[2] = ((px[2] as u16 * a + bd[bd_i + 2] as u16 * inv) / 255) as u8;
        px[3] = 255;
    }
}

/// Convert RGB (0-255) to HSV with H in 0..=359°, S and V in 0..=255.
/// Called per-pixel on the 4K live-preview hot path — `#[inline]` lets LLVM
/// fold it through the caller's arithmetic.
#[inline]
pub(crate) fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (u16, u8, u8) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;
    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let h = if delta == 0.0 {
        0.0
    } else if max == rf {
        60.0 * (((gf - bf) / delta).rem_euclid(6.0))
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h as u16 % 360, (s * 255.0) as u8, (v * 255.0) as u8)
}

/// Convert HSV (H 0..=359°, S / V 0..=255) to RGB (0..=255).
#[inline]
pub(crate) fn hsv_to_rgb(h: u16, s: u8, v: u8) -> (u8, u8, u8) {
    let sf = s as f32 / 255.0;
    let vf = v as f32 / 255.0;
    let c = vf * sf;
    let hp = (h % 360) as f32 / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (rp, gp, bp) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = vf - c;
    (
        ((rp + m) * 255.0) as u8,
        ((gp + m) * 255.0) as u8,
        ((bp + m) * 255.0) as u8,
    )
}

/// Shortest angular distance between two hues in degrees, 0..=180.
#[inline]
fn hue_distance(a: i32, b: i32) -> i32 {
    let d = (a - b).rem_euclid(360);
    if d > 180 { 360 - d } else { d }
}

/// Write the mask into an existing RGBA buffer's alpha channel in place.
/// Used by `postprocess()` to avoid a second full-resolution RGBA allocation.
///
/// Above `PAR_THRESHOLD` pixels the work splits across rows — each rayon
/// task owns a disjoint horizontal strip, matching the buffer's row-major
/// memory layout so writes stay cache-friendly. Below the threshold the
/// serial loop wins: rayon's fan-out costs more than this loop does on a
/// small image. The loop is memory-bandwidth-bound, so the speedup ceiling
/// is ~1.1-1.2× on 4K regardless of core count.
fn apply_mask_inplace(rgba: &mut RgbaImage, mask: &GrayImage) {
    const PAR_THRESHOLD: usize = 512 * 512;

    let width = rgba.width() as usize;
    let row_bytes = width * 4;
    let mask_stride = width;
    let mask_raw = mask.as_raw();
    let out_raw = rgba.as_mut();

    if mask_raw.len() >= PAR_THRESHOLD {
        out_raw
            .par_chunks_mut(row_bytes)
            .zip(mask_raw.par_chunks(mask_stride))
            .for_each(|(px_row, mask_row)| {
                for (pixel, &alpha) in px_row.chunks_mut(4).zip(mask_row.iter()) {
                    pixel[3] = alpha;
                }
            });
    } else {
        for (pixel, &alpha) in out_raw.chunks_mut(4).zip(mask_raw.iter()) {
            pixel[3] = alpha;
        }
    }
}

/// Erode (positive shift) or dilate (negative shift) the mask.
///
/// Integer iterations of a 3×3 min/max filter; each shifts the boundary by
/// ~1px. Fractional shifts run `floor` full iterations then linearly blend
/// with one extra iteration for sub-pixel precision (e.g. 2.5 = 50% 2-iter +
/// 50% 3-iter).
/// Gaussian blur approximation via 3-pass O(1) box filter. ~10× faster than
/// `image::imageops::blur` and rayon-parallel. For σ → radius, the 3-box
/// theorem gives `σ² = 3·(2r+1)²/12` → `r ≈ σ`. Visually indistinguishable
/// from a true Gaussian on a single-channel alpha mask.
fn feather_mask(mask: &mut GrayImage, sigma: f32) {
    let (w, h) = (mask.width(), mask.height());
    let radius = sigma.round().max(1.0) as u32;
    let raw = mask.as_raw();
    let mut buf: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    for _ in 0..3 {
        buf = crate::guided_filter::box_filter(&buf, w, h, radius);
    }
    let out = mask.as_mut();
    for (dst, src) in out.iter_mut().zip(buf.iter()) {
        *dst = src.clamp(0.0, 255.0).round() as u8;
    }
}

/// Dilate a grayscale mask by N pixels (0 is a fast no-op).
/// Thin wrapper around `apply_edge_shift` that hides the "negative = dilate" sign convention.
pub(crate) fn dilate_mask(mask: &mut GrayImage, pixels: u32) {
    if pixels == 0 { return; }
    apply_edge_shift(mask, -(pixels as f32));
}

fn apply_edge_shift(mask: &mut GrayImage, shift: f32) {
    let abs = shift.abs();
    if abs < 0.01 { return; }
    let erode = shift > 0.0;
    let full = abs.floor() as u32;
    let frac = abs - full as f32;
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let wi = w as i32;
    let hi = h as i32;
    let use_par = h >= 512;

    let mut a = mask.as_raw().clone();
    let mut b = vec![0u8; w * h];

    let step = |src: &[u8], dst: &mut [u8]| {
        let process_row = |(y, row): (usize, &mut [u8])| {
            let yi = y as i32;
            for x in 0..w {
                let xi = x as i32;
                let mut extremum: u8 = if erode { 255 } else { 0 };
                for dy in -1i32..=1 {
                    let ny = (yi + dy).clamp(0, hi - 1) as usize;
                    for dx in -1i32..=1 {
                        let nx = (xi + dx).clamp(0, wi - 1) as usize;
                        let v = src[ny * w + nx];
                        if erode {
                            extremum = extremum.min(v);
                        } else {
                            extremum = extremum.max(v);
                        }
                    }
                }
                row[x] = extremum;
            }
        };
        if use_par {
            dst.par_chunks_mut(w).enumerate().for_each(process_row);
        } else {
            dst.chunks_mut(w).enumerate().for_each(process_row);
        }
    };

    for _ in 0..full {
        step(&a, &mut b);
        std::mem::swap(&mut a, &mut b);
    }

    if frac >= 0.01 {
        // Blend `a` (N iterations) with one more iteration for sub-pixel shift.
        step(&a, &mut b);
        let inv = 1.0 - frac;
        let blend = |a_byte: &mut u8, b_byte: u8| {
            *a_byte = (*a_byte as f32 * inv + b_byte as f32 * frac + 0.5) as u8;
        };
        if use_par {
            a.par_iter_mut().zip(b.par_iter()).for_each(|(a, &b)| blend(a, b));
        } else {
            a.iter_mut().zip(b.iter()).for_each(|(a, &b)| blend(a, b));
        }
    }

    mask.as_mut().copy_from_slice(&a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, Rgb};
    use ndarray::Array4;

    fn make_raw_tensor(val: f32) -> Array4<f32> {
        Array4::from_elem((1, 1, 320, 320), val)
    }

    fn solid_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([100, 150, 200])))
    }

    #[test]
    fn test_postprocess_output_dimensions() {
        let raw = make_raw_tensor(0.5);
        let original = solid_rgb(640, 480);
        let result = postprocess(raw.view(), &original, &PostprocessOpts::new(&MaskSettings::default(), ModelKind::Silueta));
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_zero() {
        // All-zero tensor: mi = ma = 0, range = 1e-6
        // (0 - 0) / 1e-6 = 0 -> alpha = 0
        let raw = make_raw_tensor(0.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &PostprocessOpts::new(&MaskSettings::default(), ModelKind::Silueta));
        // All alpha values should be 0
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 0, "Expected alpha=0 for all-zero tensor");
        }
    }

    #[test]
    fn test_postprocess_no_sigmoid_uniform_one() {
        // All-one tensor: uniform high confidence → foreground → alpha=255
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(32, 32);
        let result = postprocess(raw.view(), &original, &PostprocessOpts::new(&MaskSettings::default(), ModelKind::Silueta));
        for (_, _, p) in result.enumerate_pixels() {
            assert_eq!(p[3], 255, "Expected alpha=255 for uniform high-confidence tensor");
        }
    }

    #[test]
    fn test_tensor_to_mask_from_flat_matches_view_path() {
        // Flat-slice path must produce byte-identical output to the ArrayView4 path.
        let mut flat = vec![0.0f32; 320 * 320];
        for y in 0..320_usize {
            for x in 0..320_usize {
                flat[y * 320 + x] = (y * 320 + x) as f32 / (320.0 * 320.0);
            }
        }
        let original = solid_rgb(128, 96);
        let mask = MaskSettings::default();

        let opts = PostprocessOpts::new(&mask, ModelKind::Silueta);
        let view_mask = {
            let arr = ndarray::Array4::from_shape_vec((1, 1, 320, 320), flat.clone()).unwrap();
            tensor_to_mask(arr.view(), &original, &opts)
        };
        let flat_mask = tensor_to_mask_from_flat(&flat, 320, 320, &original, &opts)
            .expect("flat path succeeds");
        assert_eq!(view_mask.as_raw(), flat_mask.as_raw());
    }

    #[test]
    fn test_tensor_to_mask_from_flat_rejects_size_mismatch() {
        let flat = vec![0.0f32; 10];
        let original = solid_rgb(16, 16);
        let mask = MaskSettings::default();
        let r = tensor_to_mask_from_flat(&flat, 320, 320, &original, &PostprocessOpts::new(&mask, ModelKind::Silueta));
        assert!(r.is_err(), "size mismatch must return Err");
    }

    #[test]
    fn test_postprocess_continuous_alpha() {
        // Tensor with gradient 0..1 should produce multiple distinct alpha values
        let mut raw = Array4::<f32>::zeros((1, 1, 320, 320));
        for y in 0..320_usize {
            for x in 0..320_usize {
                raw[[0, 0, y, x]] = (y * 320 + x) as f32 / (320.0 * 320.0);
            }
        }
        let original = solid_rgb(320, 320);
        let result = postprocess(raw.view(), &original, &PostprocessOpts::new(&MaskSettings::default(), ModelKind::Silueta));
        let unique_alphas: std::collections::HashSet<u8> =
            result.enumerate_pixels().map(|(_, _, p)| p[3]).collect();
        assert!(
            unique_alphas.len() > 10,
            "Expected many distinct alpha values, got {}",
            unique_alphas.len()
        );
    }

    #[test]
    fn brush_correction_subtract_drives_mask_to_zero_at_painted_pixels() {
        use crate::brush::{paint_circle, BrushMode, MaskCorrection};
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(320, 320);
        let mask_settings = MaskSettings::default();

        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let baseline = tensor_to_mask(raw.view(), &original, &opts);
        let baseline_at_center = baseline.get_pixel(160, 160)[0];
        assert!(baseline_at_center > 200, "baseline center should be foreground (got {})", baseline_at_center);

        let mut correction = MaskCorrection::empty(320, 320);
        paint_circle(&mut correction, 160.0, 160.0, 20.0, 1.0, BrushMode::Subtract);
        let corrected = tensor_to_mask(raw.view(), &original, &opts.with_correction(Some(&correction)));

        let center_after = corrected.get_pixel(160, 160)[0];
        let edge_after = corrected.get_pixel(10, 10)[0];
        // Full-strength subtract drives 255→1 (i8 grid range × 2 = ±254
        // effect). The 1/255 residue is imperceptible.
        assert!(center_after <= 1, "subtract stroke should drive painted pixel to ~0, got {}", center_after);
        assert_eq!(edge_after, baseline.get_pixel(10, 10)[0], "untouched pixels should match baseline");
    }

    #[test]
    fn brush_correction_dim_mismatch_skipped() {
        use crate::brush::MaskCorrection;
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(320, 320);
        let mask_settings = MaskSettings::default();

        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let baseline = tensor_to_mask(raw.view(), &original, &opts);
        let wrong_size = MaskCorrection::empty(64, 64);
        let with_bad = tensor_to_mask(raw.view(), &original, &opts.with_correction(Some(&wrong_size)));

        assert_eq!(baseline.as_raw(), with_bad.as_raw(), "dim mismatch must skip silently, not corrupt the mask");
    }

    /// Time `f` N times (with a few warm-up runs) and `eprintln!` min /
    /// median / mean. Shared by the two `#[ignore]` benches below.
    fn bench_report(label: &str, warmup: usize, iterations: usize, mut f: impl FnMut()) {
        for _ in 0..warmup {
            f();
        }
        let mut samples: Vec<u128> = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            f();
            samples.push(start.elapsed().as_micros());
        }
        samples.sort_unstable();
        let min = samples[0];
        let median = samples[iterations / 2];
        let mean = samples.iter().sum::<u128>() / iterations as u128;
        eprintln!(
            "{label} ({iterations} iters): min={:.3}ms median={:.3}ms mean={:.3}ms",
            min as f64 / 1000.0,
            median as f64 / 1000.0,
            mean as f64 / 1000.0,
        );
    }

    /// Isolated bench for the row-parallel `apply_mask_inplace`. Skips the
    /// RGBA alloc / Lanczos resize that dominate `postprocess_4k_bench`, so
    /// the number reflects only the mask-to-alpha loop.
    ///   `cargo test -p prunr-core --release apply_mask_inplace_4k_bench -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn apply_mask_inplace_4k_bench() {
        let (w, h) = (4000u32, 3000u32);
        let mask = GrayImage::from_fn(w, h, |x, _| image::Luma([(x & 0xFF) as u8]));
        let mut rgba = RgbaImage::new(w, h);
        for (i, px) in rgba.as_mut().chunks_mut(4).enumerate() {
            px[0] = (i & 0xFF) as u8;
            px[1] = ((i >> 8) & 0xFF) as u8;
            px[2] = ((i >> 16) & 0xFF) as u8;
            px[3] = 0;
        }

        bench_report(
            &format!("apply_mask_inplace_4k_bench ({w}x{h})"),
            3,
            20,
            || apply_mask_inplace(&mut rgba, &mask),
        );
    }

    /// Timed bench for the whole postprocess pipeline on a Silueta-shaped
    /// tensor + 4000×3000 source.
    ///   `cargo test -p prunr-core --release postprocess_4k_bench -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn postprocess_4k_bench() {
        let mut tensor = vec![0.0f32; 320 * 320];
        for y in 0..320 {
            for x in 0..320 {
                tensor[y * 320 + x] = (y * 320 + x) as f32 / (320.0 * 320.0);
            }
        }
        let original = solid_rgb(4000, 3000);
        let mask = MaskSettings::default();

        bench_report("postprocess_4k_bench (4000x3000, 320x320 tensor)", 2, 12, || {
            let _ = postprocess_from_flat(&tensor, 320, 320, &original, &PostprocessOpts::new(&mask, ModelKind::Silueta))
                .expect("postprocess succeeds");
        });
    }
}
