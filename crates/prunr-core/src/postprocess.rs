use image::{DynamicImage, GrayImage, RgbaImage};
use ndarray::ArrayView4;
use rayon::prelude::*;

use crate::selection::MaskArtifact;
use crate::formats::resize_gray_lanczos3;
use crate::guided_filter::guided_filter_alpha;
use crate::types::{MaskSettings, ModelKind};

/// Pixel-count cutover for serial → row-parallel kernels in this module.
/// Below ~512² rayon's fan-out costs more than the loop saves.
const ROW_PAR_THRESHOLD: usize = 512 * 512;

/// Bundle of stable inputs shared by every postprocess entry point.
/// Adding a new optional field here is preferable to adding another
/// positional parameter to six functions.
#[derive(Clone, Copy)]
pub struct PostprocessOpts<'a> {
    pub mask_settings: &'a MaskSettings,
    pub model: ModelKind,
    pub correction: Option<&'a MaskArtifact>,
}

impl<'a> PostprocessOpts<'a> {
    pub fn new(mask_settings: &'a MaskSettings, model: ModelKind) -> Self {
        Self { mask_settings, model, correction: None }
    }
    pub fn with_correction(mut self, correction: Option<&'a MaskArtifact>) -> Self {
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

/// How a segmentation model's raw output reads as subject coverage in
/// 0..1: sigmoid for BiRefNet (whose output is logits), then stretched
/// over the output's range. rembg's birefnet_general.py and BiRefNet's
/// inference.py both stretch after the sigmoid, not before.
pub(crate) struct SegNormaliser {
    use_sigmoid: bool,
    min: f32,
    inv_range: f32,
    /// A flat output: every value is this, subject (1) or not (0).
    pub(crate) uniform: Option<f32>,
}

impl SegNormaliser {
    /// From the raw output's own minimum and maximum.
    pub(crate) fn new(lo: f32, hi: f32, model: ModelKind) -> Self {
        let use_sigmoid = matches!(model, ModelKind::BiRefNetLite);
        // Sigmoid is monotonic, so min(sigmoid(x)) == sigmoid(min(x)):
        // one exp per bound instead of one per value.
        let (lo, hi) = if use_sigmoid { (sigmoid(lo), sigmoid(hi)) } else { (lo, hi) };
        let range = hi - lo;
        if range < 1e-6 {
            // Uniform output: a value past one half means all subject.
            let uniform = Some(if hi > 0.5 { 1.0 } else { 0.0 });
            return Self { use_sigmoid, min: lo, inv_range: 1.0, uniform };
        }
        Self { use_sigmoid, min: lo, inv_range: 1.0 / range, uniform: None }
    }

    /// Coverage of one raw value, ignoring `uniform`.
    #[inline]
    pub(crate) fn stretch(&self, raw: f32) -> f32 {
        let v = if self.use_sigmoid { sigmoid(raw) } else { raw };
        ((v - self.min) * self.inv_range).clamp(0.0, 1.0)
    }

    /// Coverage of one raw value.
    #[inline]
    pub(crate) fn apply(&self, raw: f32) -> f32 {
        self.uniform.unwrap_or_else(|| self.stretch(raw))
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
/// Applies normalization, gamma, resize, edge shift, guided filter, feather and contrast.
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

fn tensor_to_mask_core(
    raw: ArrayView4<f32>,
    ow: u32,
    oh: u32,
    rgba_for_guided: Option<&RgbaImage>,
    opts: &PostprocessOpts<'_>,
) -> GrayImage {
    let mask_settings = opts.mask_settings;
    let model = opts.model;
    let correction = opts.correction;
    let pred = raw.slice(ndarray::s![0, 0, .., ..]);

    let (sh, sw) = (pred.nrows(), pred.ncols());
    let contiguous;
    let pred_slice = match pred.as_slice() {
        Some(s) => s,
        // invariant: as_standard_layout() produces a contiguous view, so as_slice() is Some.
        None => {
            contiguous = pred.as_standard_layout();
            contiguous.as_slice().unwrap()
        }
    };

    // Both branches fold over the prediction values to get min/max for stretch.
    // For rembg models the fold is over raw logits; for BiRefNet the fold is over
    // sigmoid'd values — canonical rembg birefnet_general.py and BiRefNet/inference.py
    // both apply (x - min) / (max - min) after sigmoid, not before.
    let (lo, hi) = if pred_slice.len() >= ROW_PAR_THRESHOLD {
        pred_slice
            .par_iter()
            .fold(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |(lo, hi), &v| (lo.min(v), hi.max(v)),
            )
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |(l1, h1), (l2, h2)| (l1.min(l2), h1.max(h2)),
            )
    } else {
        pred_slice.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), &v| (lo.min(v), hi.max(v)),
        )
    };

    let norm = SegNormaliser::new(lo, hi, model);
    let uniform_val = norm.uniform;
    let gamma = mask_settings.gamma;

    let mut mask_buf = vec![0u8; sw * sh];
    let normalize = |raw_val: f32| norm.stretch(raw_val);
    let finalise = |val: f32| -> u8 {
        let mut v = val;
        if gamma != 1.0 { v = v.powf(gamma); }
        (v * 255.0) as u8
    };

    if let Some(uv) = uniform_val {
        mask_buf.fill(finalise(uv));
    } else if sw * sh >= ROW_PAR_THRESHOLD {
        // Fused walk skips a ~410 KB f32 scratch (4 MB at BiRefNet 1024²)
        // that an intermediate normalize-then-finalise would allocate.
        mask_buf
            .par_iter_mut()
            .zip(pred_slice.par_iter())
            .for_each(|(dst, &src)| {
                *dst = finalise(normalize(src));
            });
    } else {
        for i in 0..sh * sw {
            mask_buf[i] = finalise(normalize(pred_slice[i]));
        }
    }

    let mask = GrayImage::from_raw(sw as u32, sh as u32, mask_buf)
        .expect("mask buffer size matches dimensions");

    // Resize mask back to original dimensions (SIMD-accelerated Lanczos3)
    let mut mask = resize_gray_lanczos3(&mask, ow, oh);

    // Edge shift: positive erodes (shrinks foreground), negative dilates (expands it)
    if mask_settings.edge_shift.abs() > 0.01 {
        crate::morphology::shift_mask(&mut mask, mask_settings.edge_shift);
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
    if mask_settings.feather > 0.0 {
        feather_mask(&mut mask, mask_settings.feather);
    }

    if mask_settings.contrast > 0.0 {
        apply_contrast(&mut mask, mask_settings.contrast);
    }

    // Strokes go last, at full resolution: they are exact intent, and
    // refining or feathering them would pull them onto nearby edges.
    if let Some(corr) = correction {
        corr.apply_to_alpha(&mut mask, ow as usize, oh as usize);
    }

    mask
}

/// Steepen the mask's edge around half coverage at full resolution:
/// `contrast` 0 leaves it, 1 cuts it hard, like Select and Mask's
/// Contrast in Photoshop.
pub fn apply_contrast(mask: &mut GrayImage, contrast: f32) {
    let c = contrast.clamp(0.0, 1.0);
    let lut: [u8; 256] = std::array::from_fn(|v| {
        if c >= 1.0 {
            return if v >= 128 { 255 } else { 0 };
        }
        let a = v as f32 / 255.0;
        (((a - 0.5) / (1.0 - c) + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
    });
    mask.as_mut().par_iter_mut().for_each(|v| *v = lut[*v as usize]);
}

/// Apply a grayscale mask as the alpha channel on the original image (Tier 3).
pub fn apply_mask(original: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let mut rgba = original.to_rgba8();
    apply_mask_inplace(&mut rgba, mask);
    rgba
}

/// Helper for parallelizing coordinate-independent pixel mutations.
fn apply_pixelwise<F>(rgba: &mut RgbaImage, f: F)
where
    F: Fn(&mut [u8]) + Sync + Send,
{
    let (w, h) = rgba.dimensions();
    if (w * h) as usize >= ROW_PAR_THRESHOLD {
        rgba.as_mut().par_chunks_exact_mut(4).for_each(f);
    } else {
        for p in rgba.pixels_mut() {
            f(&mut p.0);
        }
    }
}

/// Transform the RGB channels of `rgba` according to `style`. Alpha is
/// preserved. Spatial variants (`Pixelate`) allocate a scratch copy of the
/// buffer so reads don't race writes; per-pixel variants mutate in place.
pub fn apply_fill_style(rgba: &mut RgbaImage, style: crate::types::FillStyle) {
    use crate::types::FillStyle;
    match style {
        FillStyle::None => {}
        FillStyle::Desaturate => {
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                p[0] = y; p[1] = y; p[2] = y;
            });
        }
        FillStyle::Invert => {
            apply_pixelwise(rgba, |p| {
                p[0] = 255 - p[0];
                p[1] = 255 - p[1];
                p[2] = 255 - p[2];
            });
        }
        FillStyle::Duotone { dark, light } => {
            apply_pixelwise(rgba, |p| {
                let t = luma_u8(p[0], p[1], p[2]) as u16;
                let inv = 255 - t;
                p[0] = ((dark[0] as u16 * inv + light[0] as u16 * t) / 255) as u8;
                p[1] = ((dark[1] as u16 * inv + light[1] as u16 * t) / 255) as u8;
                p[2] = ((dark[2] as u16 * inv + light[2] as u16 * t) / 255) as u8;
            });
        }
        FillStyle::Sepia => {
            apply_pixelwise(rgba, |p| {
                let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
                // Standard sepia coefficients (scaled ×1000 for integer math).
                p[0] = ((r * 393 + g * 769 + b * 189) / 1000).min(255) as u8;
                p[1] = ((r * 349 + g * 686 + b * 168) / 1000).min(255) as u8;
                p[2] = ((r * 272 + g * 534 + b * 131) / 1000).min(255) as u8;
            });
        }
        FillStyle::Threshold { level } => {
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                let v = if y >= level { 255 } else { 0 };
                p[0] = v; p[1] = v; p[2] = v;
            });
        }
        FillStyle::Posterize { levels } => {
            let n = levels.max(2) as u16 - 1;
            apply_pixelwise(rgba, |p| {
                for v in p.iter_mut().take(3) {
                    let v_u16 = *v as u16;
                    *v = ((v_u16 * n / 255) * 255 / n) as u8;
                }
            });
        }
        FillStyle::Solarize { pivot } => {
            apply_pixelwise(rgba, |p| {
                for v in p.iter_mut().take(3) {
                    if *v >= pivot {
                        *v = 255 - *v;
                    }
                }
            });
        }
        FillStyle::HueShift { degrees } => {
            apply_pixelwise(rgba, |p| {
                let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
                let new_h = (h as i32 + degrees as i32).rem_euclid(360) as u16;
                let (r, g, b) = hsv_to_rgb(new_h, s, v);
                p[0] = r; p[1] = g; p[2] = b;
            });
        }
        FillStyle::Saturate { percent } => {
            // Move each channel toward/away from luma by `percent / 100`.
            // percent=0 → grayscale, 100 → unchanged, >100 → punchy.
            let factor = percent.min(300) as i32;
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]) as i32;
                for v in p.iter_mut().take(3) {
                    let v_i32 = *v as i32;
                    let shifted = y + ((v_i32 - y) * factor / 100);
                    *v = shifted.clamp(0, 255) as u8;
                }
            });
        }
        FillStyle::ColorSplash { keep_hue, tolerance } => {
            let keep = (keep_hue % 360) as i32;
            let tol = tolerance.min(180) as i32;
            apply_pixelwise(rgba, |p| {
                let (h, _, _) = rgb_to_hsv(p[0], p[1], p[2]);
                let dist = hue_distance(h as i32, keep);
                if dist > tol {
                    let y = luma_u8(p[0], p[1], p[2]);
                    p[0] = y; p[1] = y; p[2] = y;
                }
            });
        }
        FillStyle::Pixelate { block_size } => {
            if block_size < 2 { return; }
            let (w, h) = rgba.dimensions();
            // Sample one pixel per block (top-left corner) into a small
            // LUT before any mutation. Replaces the previous full-image
            // `rgba.clone()` (~48 MB at 4K) with a block-grid scratch
            // (~144 KB at block_size=16, ~9 MB at block_size=2).
            let bs = block_size;
            let blocks_x = w.div_ceil(bs);
            let blocks_y = h.div_ceil(bs);
            let mut lut: Vec<[u8; 3]> =
                Vec::with_capacity((blocks_x * blocks_y) as usize);
            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    let p = rgba.get_pixel(bx * bs, by * bs);
                    lut.push([p.0[0], p.0[1], p.0[2]]);
                }
            }
            // Row-parallel: postprocess runs in the subprocess worker,
            // never nested inside rayon (so par_chunks_mut won't deadlock).
            use rayon::prelude::*;
            let row_stride = (w * 4) as usize;
            rgba.as_mut().par_chunks_mut(row_stride).enumerate().for_each(|(y, row)| {
                let row_block = ((y as u32) / bs) * blocks_x;
                for x in 0..w as usize {
                    let s = lut[(row_block + (x as u32) / bs) as usize];
                    let p = x * 4;
                    row[p] = s[0];
                    row[p + 1] = s[1];
                    row[p + 2] = s[2];
                }
            });
        }
        FillStyle::CrossProcess { shadow, highlight } => {
            // Split-tone by luma: pixels below 128 bend toward `shadow`,
            // above bend toward `highlight`. Preserves midtones, lifts
            // shadows + warms highlights (or whatever the user picked).
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                let (target, t) = if y < 128 {
                    (shadow, (128 - y) as u16) // 0..=128
                } else {
                    (highlight, (y - 128) as u16) // 0..=127
                };
                let w = (t.min(128) * 2).min(255); // scale to 0..=256 then clamp
                let inv = 255 - w;
                p[0] = ((p[0] as u16 * inv + target[0] as u16 * w) / 255) as u8;
                p[1] = ((p[1] as u16 * inv + target[1] as u16 * w) / 255) as u8;
                p[2] = ((p[2] as u16 * inv + target[2] as u16 * w) / 255) as u8;
            });
        }
        FillStyle::ChannelSwap { variant } => {
            use crate::types::ChannelSwapVariant;
            apply_pixelwise(rgba, |p| {
                let [r, g, b, _] = [p[0], p[1], p[2], p[3]];
                let (nr, ng, nb) = match variant {
                    ChannelSwapVariant::Grb => (g, r, b),
                    ChannelSwapVariant::Brg => (b, r, g),
                    ChannelSwapVariant::Rbg => (r, b, g),
                    ChannelSwapVariant::Bgr => (b, g, r),
                    ChannelSwapVariant::Gbr => (g, b, r),
                };
                p[0] = nr; p[1] = ng; p[2] = nb;
            });
        }
        FillStyle::Halftone { dot_spacing } => {
            let spacing = dot_spacing.clamp(2, 32) as i32;
            let (w, h) = rgba.dimensions();
            let half = spacing / 2;
            let max_r_sq = (half * half) as u32;
            let row_stride = (w * 4) as usize;

            // Read each pixel's luma BEFORE writing the same pixel. The
            // earlier bug read cell-top-left luma, which had already been
            // mutated by the time later rows ran; reading `(x, y)` itself
            // is safe because the write at `(x, y)` happens after.
            let process_row = |(y, row): (usize, &mut [u8])| {
                let cy = ((y as i32) / spacing) * spacing + half;
                let dy = y as i32 - cy;
                for x in 0..w as usize {
                    let p = x * 4;
                    let luma = luma_u8(row[p], row[p + 1], row[p + 2]);
                    let r_sq = max_r_sq * (255 - luma as u32) / 255;
                    let cx = ((x as i32) / spacing) * spacing + half;
                    let dx = x as i32 - cx;
                    let dist_sq = (dx * dx + dy * dy) as u32;
                    let v = if dist_sq <= r_sq { 0u8 } else { 255 };
                    row[p] = v; row[p + 1] = v; row[p + 2] = v;
                }
            };

            if (w * h) as usize >= ROW_PAR_THRESHOLD {
                rgba.as_mut().par_chunks_mut(row_stride).enumerate().for_each(process_row);
            } else {
                rgba.as_mut().chunks_mut(row_stride).enumerate().for_each(process_row);
            }
        }
        FillStyle::GradientMap { stops } => {
            // Map luma 0..=255 through 4 colour stops at 0, 85, 170, 255.
            // Linear interp between adjacent stops.
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]) as u16;
                let (lo, hi, t) = if y <= 85 {
                    (stops[0], stops[1], y * 255 / 85)
                } else if y <= 170 {
                    (stops[1], stops[2], (y - 85) * 255 / 85)
                } else {
                    (stops[2], stops[3], (y - 170) * 255 / 85)
                };
                let t = t.min(255);
                let inv = 255 - t;
                p[0] = ((lo[0] as u16 * inv + hi[0] as u16 * t) / 255) as u8;
                p[1] = ((lo[1] as u16 * inv + hi[1] as u16 * t) / 255) as u8;
                p[2] = ((lo[2] as u16 * inv + hi[2] as u16 * t) / 255) as u8;
            });
        }
    }
}

/// Rec. 709 luma of an 8-bit RGB triple. Used by every grayscale / duotone /
/// threshold / splash variant.
#[inline]
fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 2126 + g as u32 * 7152 + b as u32 * 722) / 10000) as u8
}

/// The picture `effect` makes from `source`, opaque, at the source's
/// size; `None` for no effect. It goes behind a result like a background
/// image does.
///
/// Peak working set: the source as RGBA plus the output (2 × w·h·4 bytes;
/// 96 MB at 12 MP).
pub fn effect_backdrop(source: &RgbaImage, effect: crate::types::BgEffect) -> Option<RgbaImage> {
    use crate::types::BgEffect;
    let mut img = match effect {
        BgEffect::None => return None,
        // fast_blur is a 3-pass box approximation of a Gaussian, ~3-4x
        // faster than imageops::blur and alike at these radii.
        BgEffect::BlurredSource { radius } => image::imageops::fast_blur(source, radius.clamp(1, 64) as f32),
        BgEffect::DesaturatedSource => {
            let mut img = source.clone();
            for p in img.pixels_mut() {
                let y = luma_u8(p.0[0], p.0[1], p.0[2]);
                p.0[0] = y; p.0[1] = y; p.0[2] = y;
            }
            img
        }
    };
    for p in img.pixels_mut() {
        p.0[3] = 255;
    }
    Some(img)
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
/// Above `ROW_PAR_THRESHOLD` pixels the work splits across rows — each
/// rayon task owns a disjoint horizontal strip, matching the buffer's
/// row-major memory layout so writes stay cache-friendly. Below the
/// threshold the serial loop wins: rayon's fan-out costs more than this
/// loop does on a small image. The loop is memory-bandwidth-bound, so
/// the speedup ceiling is ~1.1-1.2× on 4K regardless of core count.
fn apply_mask_inplace(rgba: &mut RgbaImage, mask: &GrayImage) {
    let width = rgba.width() as usize;
    let row_bytes = width * 4;
    let mask_stride = width;
    let mask_raw = mask.as_raw();
    let out_raw = rgba.as_mut();

    if mask_raw.len() >= ROW_PAR_THRESHOLD {
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

/// Feather: a Gaussian blur approximated by a 3-pass box filter (the
/// 3-box theorem gives `r ≈ σ`). A fractional `sigma` blends the blurs at
/// the two whole radii around it, so the slider is continuous and values
/// under 1 px soften too. Peak working set: two u8 copies of the mask's
/// bbox (plus the blur margin) when `sigma` is fractional, plus the bbox
/// scratch of `box_blur`.
fn feather_mask(mask: &mut GrayImage, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let lo = sigma.floor() as u32;
    let t = sigma - lo as f32;
    if t < 1e-3 {
        box_blur(mask, lo);
        return;
    }
    // Both blurs stay inside the wider one's reach; blend only there.
    let Some(reach) = crate::inpaint::mask_bbox(mask, 1, 3 * (lo + 1)) else { return };
    let (x0, y0, w) = (reach.x as usize, reach.y as usize, mask.width() as usize);
    let mut wider = image::imageops::crop_imm(mask, reach.x, reach.y, reach.w, reach.h).to_image();
    let mut narrow = wider.clone();
    box_blur(&mut wider, lo + 1);
    box_blur(&mut narrow, lo);
    let out = mask.as_mut();
    for (row, (n, wd)) in narrow.as_raw().chunks(reach.w as usize).zip(wider.as_raw().chunks(reach.w as usize)).enumerate() {
        let dst = &mut out[(y0 + row) * w + x0..][..reach.w as usize];
        for (d, (&a, &b)) in dst.iter_mut().zip(n.iter().zip(wd)) {
            *d = (a as f32 * (1.0 - t) + b as f32 * t).round() as u8;
        }
    }
}

/// Bbox-cropped 3-box blur at `radius`: pixels outside the `mask +
/// 3·radius` margin can't receive any weight from inside, so the
/// full-image variant would just walk zeros there. Bit-exact output,
/// ~96 MB → bbox-sized peak working set on 4K masks with small regions.
fn box_blur(mask: &mut GrayImage, radius: u32) {
    if radius == 0 { return; }
    // 3-box kernel total reach is 3·radius from each source pixel.
    let margin = 3 * radius;
    let Some(bbox) = crate::inpaint::mask_bbox(mask, 1, margin) else {
        return;
    };
    let raw_w = mask.width() as usize;
    let bw = bbox.w as usize;
    let bh = bbox.h as usize;
    let bn = bw * bh;
    let mut buf = vec![0.0f32; bn];
    let mut tmp = vec![0.0f32; bn];
    let raw = mask.as_raw();
    for by in 0..bh {
        let src_off = (bbox.y as usize + by) * raw_w + bbox.x as usize;
        let dst_off = by * bw;
        for bx in 0..bw {
            buf[dst_off + bx] = raw[src_off + bx] as f32;
        }
    }
    crate::guided_filter::box_filter_into(&buf, bbox.w, bbox.h, radius, &mut tmp);
    crate::guided_filter::box_filter_into(&tmp, bbox.w, bbox.h, radius, &mut buf);
    crate::guided_filter::box_filter_into(&buf, bbox.w, bbox.h, radius, &mut tmp);
    let out = mask.as_mut();
    for by in 0..bh {
        let dst_off = (bbox.y as usize + by) * raw_w + bbox.x as usize;
        let src_off = by * bw;
        for bx in 0..bw {
            out[dst_off + bx] = tmp[src_off + bx].clamp(0.0, 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Gamma above 1 removes more of the soft edge and below 1 keeps
    /// more; 1 leaves the mask as the model gave it.
    #[test]
    fn gamma_darkens_or_lightens_the_soft_edge() {
        let tensor = ndarray::Array4::from_shape_fn((1, 1, 1, 64), |(_, _, _, x)| x as f32 / 63.0 * 8.0 - 4.0);
        let original = DynamicImage::ImageRgb8(RgbImage::new(64, 1));
        let mean = |gamma: f32| {
            let settings = MaskSettings { gamma, ..MaskSettings::default() };
            let m = tensor_to_mask(tensor.view(), &original, &PostprocessOpts::new(&settings, ModelKind::Silueta));
            m.as_raw().iter().map(|&v| v as f32).sum::<f32>() / 64.0
        };
        assert!(mean(2.0) < mean(1.0) && mean(1.0) < mean(0.5), "{} {} {}", mean(2.0), mean(1.0), mean(0.5));
    }

    /// Feather is continuous: half a pixel softens an edge, and 1.5 px
    /// lands between 1 and 2.
    #[test]
    fn feather_is_continuous_below_and_between_whole_pixels() {
        let edge = || GrayImage::from_fn(16, 1, |x, _| image::Luma([if x < 8 { 255 } else { 0 }]));
        let at = |sigma: f32| { let mut m = edge(); feather_mask(&mut m, sigma); m.get_pixel(8, 0)[0] };
        assert!(at(0.5) > 0, "half a pixel softens the edge");
        let (one, mid, two) = (at(1.0), at(1.5), at(2.0));
        assert!(one.min(two) <= mid && mid <= one.max(two), "1.5 lies between 1 and 2: {one} {mid} {two}");
        assert_eq!(at(0.0), 0);
    }

    /// Contrast steepens the edge around half coverage and leaves the
    /// rest: 0 changes nothing, 100 % cuts it hard at full resolution.
    #[test]
    fn contrast_steepens_the_edge_and_cuts_it_at_full() {
        let ramp = || GrayImage::from_raw(5, 1, vec![0, 64, 128, 192, 255]).unwrap();
        let mut none = ramp();
        apply_contrast(&mut none, 0.0);
        assert_eq!(none.as_raw(), ramp().as_raw());
        let mut half = ramp();
        apply_contrast(&mut half, 0.5);
        assert!(half.as_raw()[1] < 64 && half.as_raw()[3] > 192, "steeper: {:?}", half.as_raw());
        let mut hard = ramp();
        apply_contrast(&mut hard, 1.0);
        assert_eq!(hard.as_raw(), &[0, 0, 255, 255, 255]);
    }

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
        use crate::brush::paint_circle;
        use crate::selection::BrushMode;
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(320, 320);
        let mask_settings = MaskSettings::default();

        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let baseline = tensor_to_mask(raw.view(), &original, &opts);
        let baseline_at_center = baseline.get_pixel(160, 160)[0];
        assert!(baseline_at_center > 200, "baseline center should be foreground (got {})", baseline_at_center);

        let mut correction = MaskArtifact::new_empty(320, 320);
        paint_circle(&mut correction, 160.0, 160.0, 20.0, crate::brush::Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Subtract });
        let corrected = tensor_to_mask(raw.view(), &original, &opts.with_correction(Some(&correction)));

        let center_after = corrected.get_pixel(160, 160)[0];
        let edge_after = corrected.get_pixel(10, 10)[0];
        // Full-strength subtract drives 255→1 (i8 grid range × 2 = ±254
        // effect). The 1/255 residue is imperceptible.
        assert!(center_after <= 1, "subtract stroke should drive painted pixel to ~0, got {}", center_after);
        assert_eq!(edge_after, baseline.get_pixel(10, 10)[0], "untouched pixels should match baseline");
    }

    /// A stroke is the user's exact intent: it lands pixel for pixel where
    /// it was painted, after edge refinement, which would otherwise pull
    /// its outline onto nearby image edges.
    #[test]
    fn a_stroke_lands_exactly_where_painted_even_with_refined_edges() {
        let (w, h) = (64u32, 64u32);
        let tensor = Array4::from_shape_fn((1, 1, 16, 16), |(_, _, y, _)| if y < 4 { 10.0 } else { -10.0 });
        let original = DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, _| if x < 32 { Rgb([10, 10, 10]) } else { Rgb([240, 240, 240]) }));
        let mask_settings = MaskSettings { refine_edges: true, ..MaskSettings::default() };
        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let block = |x: u32, y: u32| (10..50).contains(&x) && (42..56).contains(&y);
        let cells = (0..w * h).map(|i| if block(i % w, i / w) { crate::selection::FULL } else { 0 }).collect();
        let restore = MaskArtifact::from_cells(w, h, cells);

        let mask = tensor_to_mask(tensor.view(), &original, &opts.with_correction(Some(&restore)));

        for y in 36..62 {
            for x in 0..w {
                let expected = if block(x, y) { 255 } else { 0 };
                assert_eq!(mask.get_pixel(x, y)[0], expected, "pixel ({x}, {y})");
            }
        }
    }

    #[test]
    fn brush_correction_dim_mismatch_skipped() {
        let raw = make_raw_tensor(1.0);
        let original = solid_rgb(320, 320);
        let mask_settings = MaskSettings::default();

        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let baseline = tensor_to_mask(raw.view(), &original, &opts);
        let wrong_size = MaskArtifact::new_empty(64, 64);
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

    /// Halftone regression: on a uniform-luma input, pixels at the
    /// same offset within their respective cells must produce
    /// identical output. The pre-fix version read from the already-
    /// mutated buffer, so cells past row 0 saw corrupted "luma" (the
    /// halftone's own 0/255 output) and produced different radii.
    /// Comparing rows at multiples of `spacing` exercises this: the
    /// cell-relative offset is identical, so the math should be too.
    #[test]
    fn halftone_uniform_input_is_cell_invariant() {
        let spacing = 8u32;
        let mut img = image::RgbaImage::from_pixel(32, 32, image::Rgba([100, 100, 100, 255]));
        apply_fill_style(
            &mut img,
            crate::types::FillStyle::Halftone { dot_spacing: spacing },
        );
        // Rows 0, 8, 16, 24 all have dy = -half from their cell centre.
        for &probe_y in &[spacing, spacing * 2, spacing * 3] {
            for x in 0..32 {
                let a = img.get_pixel(x, 0).0;
                let b = img.get_pixel(x, probe_y).0;
                assert_eq!(
                    a, b,
                    "uniform halftone diverges at ({x}, 0) vs ({x}, {probe_y})",
                );
            }
        }
    }

    /// Filter-only mode (`SettingsModel::None` + `LineMode::Off`) routes
    /// through `apply_fill_style` directly on the source RGB. The toolbar
    /// chip claims every `FillStyle::ALL` variant produces a different
    /// output; without a test that's an unverified assertion. Run each
    /// variant on a colourful 4×4 patch and assert:
    ///
    /// - `FillStyle::None` is bit-exact identity (no-op contract).
    /// - Every other variant differs from the input (variant did something).
    /// - All non-None variants produce pairwise-distinct outputs (no two
    ///   variants accidentally collapse to the same bytes).
    #[test]
    fn fill_style_all_variants_produce_distinct_output() {
        use crate::types::FillStyle;

        // 4×4 patch covering primaries + grays so every channel-touching
        // variant has substrate to differ on.
        let mut src = image::RgbaImage::new(4, 4);
        let palette: [[u8; 4]; 16] = [
            [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255],
            [0, 255, 255, 255], [255, 0, 255, 255], [128, 128, 128, 255], [200, 100, 50, 255],
            [50, 100, 200, 255], [180, 220, 100, 255], [100, 50, 180, 255], [240, 240, 240, 255],
            [20, 20, 20, 255], [128, 64, 192, 255], [192, 128, 64, 255], [64, 192, 128, 255],
        ];
        for (i, px) in palette.iter().enumerate() {
            src.put_pixel((i % 4) as u32, (i / 4) as u32, image::Rgba(*px));
        }

        let mut outputs: Vec<(String, image::RgbaImage)> = FillStyle::ALL.iter()
            .map(|style| {
                let mut img = src.clone();
                apply_fill_style(&mut img, *style);
                (style.name().to_string(), img)
            })
            .collect();

        // None is identity.
        let none_idx = outputs.iter().position(|(n, _)| n == "None")
            .expect("FillStyle::None must be in ALL");
        assert_eq!(outputs[none_idx].1, src, "FillStyle::None must be bit-exact identity");

        // Every other variant differs from input.
        for (name, img) in outputs.iter().filter(|(n, _)| n != "None") {
            assert_ne!(img, &src, "FillStyle::{name} produced bit-exact-identity output (variant is dead)");
        }

        // No two variants produce the same output.
        outputs.sort_by(|a, b| a.0.cmp(&b.0));
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i].1, outputs[j].1,
                    "FillStyle::{} and FillStyle::{} collapse to identical bytes",
                    outputs[i].0, outputs[j].0,
                );
            }
        }
    }

    /// Pixelate overhang: image dimensions not a multiple of `block_size`.
    /// `blocks_x = w.div_ceil(bs)`; the rightmost block should sample at
    /// `(blocks_x-1) * bs` which must be inside the image.
    #[test]
    fn pixelate_handles_overhang_dimensions() {
        // 17×17 image, block_size=8 → blocks = 3×3, last block sample at (16, 16).
        let mut img = image::RgbaImage::new(17, 17);
        for y in 0..17 {
            for x in 0..17 {
                img.put_pixel(x, y, image::Rgba([(x * 13) as u8, (y * 17) as u8, 0, 255]));
            }
        }
        let pre = img.clone();
        apply_fill_style(&mut img, crate::types::FillStyle::Pixelate { block_size: 8 });
        // Sanity: corner blocks sampled at (0,0), (8,0), (16,0), etc.
        // Top-left block (0..8, 0..8) all same colour as pre[(0,0)].
        let expected_tl = pre.get_pixel(0, 0).0;
        assert_eq!(img.get_pixel(0, 0).0[..3], expected_tl[..3]);
        assert_eq!(img.get_pixel(7, 7).0[..3], expected_tl[..3]);
        // Rightmost block (16..17, 0..8) samples at (16, 0).
        let expected_r = pre.get_pixel(16, 0).0;
        assert_eq!(img.get_pixel(16, 0).0[..3], expected_r[..3]);
        // Bottom-right block (16..17, 16..17) samples at (16, 16).
        let expected_br = pre.get_pixel(16, 16).0;
        assert_eq!(img.get_pixel(16, 16).0[..3], expected_br[..3]);
    }

    /// BiRefNet sigmoid path must apply min-max stretch after sigmoid.
    ///
    /// A logit tensor that, after sigmoid, clusters in [0.35, 0.65] would
    /// produce u8 values in [89, 166] without the stretch — never reaching
    /// full black or white. With the stretch the output must span [0, 255].
    #[test]
    fn birefnet_sigmoid_stretch_reaches_full_range() {
        // logit_lo → sigmoid ≈ 0.35, logit_hi → sigmoid ≈ 0.65
        let logit_lo: f32 = -0.6190; // sigmoid(-0.619) ≈ 0.35
        let logit_hi: f32 = 0.6190;  // sigmoid( 0.619) ≈ 0.65

        let h = 4usize;
        let w = 4usize;
        let n = h * w;
        // Alternate low/high logits so the tensor has both extremes.
        let data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { logit_lo } else { logit_hi }).collect();
        let raw = ndarray::Array4::from_shape_vec((1, 1, h, w), data).unwrap();

        let original = solid_rgb(w as u32, h as u32);
        let mask_settings = MaskSettings::default();
        let opts = PostprocessOpts::new(&mask_settings, ModelKind::BiRefNetLite);
        let mask = tensor_to_mask(raw.view(), &original, &opts);

        let pixels: Vec<u8> = mask.pixels().map(|p| p[0]).collect();
        let got_min = *pixels.iter().min().unwrap();
        let got_max = *pixels.iter().max().unwrap();

        // After stretch the extremes must reach near-0 and near-255.
        // Allow a 2-gray tolerance for f32 rounding through the pipeline.
        assert!(got_min <= 2,  "min after sigmoid stretch should be ≈0, got {got_min}");
        assert!(got_max >= 253, "max after sigmoid stretch should be ≈255, got {got_max}");
    }

    #[test]
    fn an_effect_backdrop_is_an_opaque_picture_of_the_photo() {
        use crate::types::BgEffect;
        let photo = RgbaImage::from_fn(8, 4, |x, _| {
            if x < 4 { image::Rgba([200, 40, 40, 0]) } else { image::Rgba([20, 20, 220, 255]) }
        });
        assert!(effect_backdrop(&photo, BgEffect::None).is_none());
        for effect in BgEffect::ALL.iter().filter(|e| **e != BgEffect::None) {
            let b = effect_backdrop(&photo, *effect).expect("an effect makes a picture");
            assert_eq!(b.dimensions(), (8, 4));
            assert!(b.pixels().all(|p| p.0[3] == 255), "{effect:?} is opaque");
        }
        let grey = effect_backdrop(&photo, BgEffect::DesaturatedSource).unwrap();
        let p = grey.get_pixel(6, 1).0;
        assert!(p[0] == p[1] && p[1] == p[2], "black and white");
        let blur = effect_backdrop(&photo, BgEffect::BlurredSource { radius: 2 }).unwrap();
        let (left, right) = (blur.get_pixel(3, 2).0[2], blur.get_pixel(4, 2).0[2]);
        assert!(left > 40 && right < 220, "the two halves bleed into each other: {left} {right}");
    }
}

