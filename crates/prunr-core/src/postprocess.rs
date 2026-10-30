use image::{DynamicImage, GrayImage, RgbaImage};
use ndarray::ArrayView4;
use rayon::prelude::*;

use crate::color::lerp_rgb;
use crate::math::sigmoid;
use crate::selection::MaskArtifact;
use crate::formats::resize_gray_lanczos3;
use crate::guided_filter::{guided_filter_alpha, guided_filter_alpha_colour};
use crate::types::{MaskSettings, ModelKind, SubjectEffect};

/// Pixel-count cutover for serial → row-parallel kernels in this module.
/// Below ~512² rayon's fan-out costs more than the loop saves.
const ROW_PAR_THRESHOLD: usize = 512 * 512;

/// Bundle of stable inputs shared by every postprocess entry point.
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

/// Raw ONNX output → transparent RGBA. One RGBA buffer serves both the
/// guided filter and the final mask application.
pub fn postprocess(raw: ArrayView4<f32>, original: &impl crate::formats::AsRgba, opts: &PostprocessOpts<'_>) -> RgbaImage {
    let mut rgba = original.as_rgba().into_owned();
    let mask = tensor_to_mask_with_rgba(raw, &rgba, opts);
    finish_into(&mut rgba, &mask, opts.mask_settings);
    rgba
}

/// The cut-out's look on a photo already cut: `mask` as its alpha, then
/// the subject effect and the decoration.
fn finish_into(rgba: &mut RgbaImage, mask: &GrayImage, settings: &MaskSettings) {
    apply_mask_inplace(rgba, mask);
    apply_subject_effect(rgba, settings.effect);
    crate::decorate::apply_decoration(rgba, &settings.decoration);
}

/// The mask `postprocess_from_flat` cuts with, for a caller that keeps
/// it: it holds while the tensor, the photo and the mask settings other
/// than the look are unchanged. Refused when its work does not fit, as
/// the whole pass is.
pub fn mask_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &RgbaImage,
    opts: &PostprocessOpts<'_>,
) -> Result<GrayImage, crate::types::CoreError> {
    crate::pipeline::room_for_postprocess(original.width(), original.height(), opts.mask_settings.refine_edges)?;
    Ok(tensor_to_mask_with_rgba(flat_view(tensor, tensor_h, tensor_w)?, original, opts))
}

/// The cut-out of `original` from a mask `mask_from_flat` made, with
/// `settings`' look. Peak working set: the output (w·h·4 bytes) and what
/// the decoration takes.
pub fn finish_with_mask(original: &RgbaImage, mask: &GrayImage, settings: &MaskSettings) -> RgbaImage {
    let mut rgba = original.clone();
    finish_into(&mut rgba, mask, settings);
    rgba
}

/// `postprocess` from a flat `[1,1,H,W]` tensor, as it arrives over IPC:
/// a first run, a re-run or a live preview. Refused when its full-size
/// work does not fit (`pipeline::room_for_postprocess`), so turning on
/// Refine edges cannot outgrow memory after the run itself fitted.
pub fn postprocess_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &impl crate::formats::AsRgba,
    opts: &PostprocessOpts<'_>,
) -> Result<RgbaImage, crate::types::CoreError> {
    let (w, h) = original.size();
    crate::pipeline::room_for_postprocess(w, h, opts.mask_settings.refine_edges)?;
    let view = flat_view(tensor, tensor_h, tensor_w)?;
    Ok(postprocess(view, original, opts))
}

fn flat_view(tensor: &[f32], h: usize, w: usize) -> Result<ArrayView4<'_, f32>, crate::types::CoreError> {
    ArrayView4::from_shape((1, 1, h, w), tensor)
        .map_err(|e| crate::types::CoreError::Inference(format!("Tensor reshape: {e}")))
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

/// Flat-slice variant of `tensor_to_mask`. Reshapes `[1,1,H,W]` and forwards.
pub fn tensor_to_mask_from_flat(
    tensor: &[f32],
    tensor_h: usize,
    tensor_w: usize,
    original: &DynamicImage,
    opts: &PostprocessOpts<'_>,
) -> Result<GrayImage, crate::types::CoreError> {
    let view = flat_view(tensor, tensor_h, tensor_w)?;
    Ok(tensor_to_mask(view, original, opts))
}

/// Convert raw ONNX tensor to a full-resolution grayscale mask (Tier 2).
/// Applies normalization, gamma, resize, edge shift, guided filter, feather and contrast.
pub fn tensor_to_mask(raw: ArrayView4<f32>, original: &DynamicImage, opts: &PostprocessOpts<'_>) -> GrayImage {
    // Converts to RGBA even when nothing needs it: hot-path callers use
    // `postprocess()`, which shares the RGBA buffer.
    let owned;
    let base = match original.as_rgba8() {
        Some(rgba) => Some(rgba),
        None if opts.mask_settings.refine_edges || original.color().has_alpha() => {
            owned = original.to_rgba8();
            Some(&owned)
        }
        None => None,
    };
    tensor_to_mask_core(raw, original.width(), original.height(), base, opts)
}

/// Same as `tensor_to_mask` but reuses an already-materialized RGBA buffer.
fn tensor_to_mask_with_rgba(raw: ArrayView4<f32>, rgba: &RgbaImage, opts: &PostprocessOpts<'_>) -> GrayImage {
    tensor_to_mask_core(raw, rgba.width(), rgba.height(), Some(rgba), opts)
}

fn tensor_to_mask_core(
    raw: ArrayView4<f32>,
    ow: u32,
    oh: u32,
    base: Option<&RgbaImage>,
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
        None => {
            contiguous = pred.as_standard_layout();
            // invariant: as_standard_layout() produces a contiguous view, so as_slice() is Some.
            contiguous.as_slice().unwrap()
        }
    };

    let (lo, hi, _) = crate::math::slice_stats(pred_slice);

    let norm = SegNormaliser::new(lo, hi, model);
    let uniform_val = norm.uniform;
    // A gamma that is not a number (a scripted value) cuts as gamma 1.
    let gamma = if mask_settings.gamma.is_finite() { mask_settings.gamma } else { 1.0 };

    let mut mask_buf = vec![0u8; sw * sh];
    let normalize = |raw_val: f32| norm.stretch(raw_val);
    let finalise = |val: f32| -> u8 {
        let mut v = val;
        if gamma != 1.0 { v = v.powf(gamma); }
        crate::math::round_u8(v * 255.0)
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

    let mut mask = resize_gray_lanczos3(&mask, ow, oh);

    // Edge shift: positive erodes (shrinks foreground), negative dilates
    // (expands it). Clamped here, where every source of the setting meets:
    // the shift's planes grow with it.
    if mask_settings.edge_shift.abs() > 0.01 {
        let max = MaskSettings::EDGE_SHIFT_MAX_PX;
        crate::morphology::shift_mask(&mut mask, mask_settings.edge_shift.clamp(-max, max));
    }

    if mask_settings.refine_edges {
        if let Some(rgba) = base {
            let (radius, epsilon) = (mask_settings.guided_radius, mask_settings.guided_epsilon);
            mask = match mask_settings.refine_guide {
                crate::RefineGuide::Brightness => guided_filter_alpha(rgba, &mask, radius, epsilon),
                crate::RefineGuide::Colour => guided_filter_alpha_colour(rgba, &mask, radius, epsilon),
            };
        }
    }

    // Feather runs AFTER refine: sharpen-then-soften; the other order
    // makes the two fight.
    if mask_settings.feather > 0.0 {
        feather_mask(&mut mask, mask_settings.feather);
    }

    if mask_settings.contrast > 0.0 {
        apply_contrast(&mut mask, mask_settings.contrast);
    }

    // A cut of an earlier cut-out only takes more away: what that image
    // hides stays hidden, whatever the model sees in the colours under it.
    if let Some(base) = base {
        keep_hidden(&mut mask, base);
    }

    // Strokes go last, at full resolution: they are exact intent, and
    // refining or feathering them would pull them onto nearby edges.
    if let Some(corr) = correction {
        corr.apply_to_alpha(&mut mask, ow as usize, oh as usize);
    }

    mask
}

/// Scale `mask` by `base`'s alpha: a pixel the base hides stays hidden,
/// a half-transparent one keeps at most its share. Opaque rows cost a scan.
fn keep_hidden(mask: &mut GrayImage, base: &RgbaImage) {
    let width = base.width() as usize;
    if width == 0 || mask.dimensions() != base.dimensions() {
        return;
    }
    let scale = |m: &mut [u8], px: &[u8]| {
        for (v, p) in m.iter_mut().zip(px.chunks_exact(4)) {
            if p[3] != 255 {
                *v = ((*v as u16 * p[3] as u16 + 127) / 255) as u8;
            }
        }
    };
    if width * base.height() as usize >= ROW_PAR_THRESHOLD {
        mask.as_mut().par_chunks_mut(width).zip(base.as_raw().par_chunks(width * 4)).for_each(|(m, px)| scale(m, px));
    } else {
        mask.as_mut().chunks_mut(width).zip(base.as_raw().chunks(width * 4)).for_each(|(m, px)| scale(m, px));
    }
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
pub fn apply_subject_effect(rgba: &mut RgbaImage, style: SubjectEffect) {
    match style {
        SubjectEffect::None => {}
        SubjectEffect::Desaturate => {
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                p[0] = y; p[1] = y; p[2] = y;
            });
        }
        SubjectEffect::Invert => {
            apply_pixelwise(rgba, |p| {
                p[0] = 255 - p[0];
                p[1] = 255 - p[1];
                p[2] = 255 - p[2];
            });
        }
        SubjectEffect::Duotone { dark, light } => gradient_map(rgba, &[dark, light]),
        SubjectEffect::Sepia => {
            apply_pixelwise(rgba, |p| {
                let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
                // Standard sepia coefficients (scaled ×1000 for integer math).
                p[0] = ((r * 393 + g * 769 + b * 189) / 1000).min(255) as u8;
                p[1] = ((r * 349 + g * 686 + b * 168) / 1000).min(255) as u8;
                p[2] = ((r * 272 + g * 534 + b * 131) / 1000).min(255) as u8;
            });
        }
        SubjectEffect::Threshold { level } => {
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                let v = if y >= level { 255 } else { 0 };
                p[0] = v; p[1] = v; p[2] = v;
            });
        }
        SubjectEffect::Posterize { levels } => {
            let n = posterize_steps(levels);
            apply_pixelwise(rgba, |p| {
                for v in p.iter_mut().take(3) {
                    *v = posterize_u8(*v, n);
                }
            });
        }
        SubjectEffect::Solarize { pivot } => {
            apply_pixelwise(rgba, |p| {
                for v in p.iter_mut().take(3) {
                    if *v >= pivot {
                        *v = 255 - *v;
                    }
                }
            });
        }
        SubjectEffect::HueShift { degrees } => {
            apply_pixelwise(rgba, |p| {
                let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
                let new_h = (h as i32 + degrees as i32).rem_euclid(360) as u16;
                let (r, g, b) = hsv_to_rgb(new_h, s, v);
                p[0] = r; p[1] = g; p[2] = b;
            });
        }
        SubjectEffect::Saturate { percent } => {
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
        SubjectEffect::ColorSplash { keep_hue, tolerance } => {
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
        SubjectEffect::Pixelate { block_size } => {
            if block_size < 2 { return; }
            let (w, h) = rgba.dimensions();
            // Sample one pixel per block (top-left corner) into a small
            // LUT before any mutation, not a full copy: ~144 KB at
            // block_size=16, ~9 MB at block_size=2.
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
        SubjectEffect::CrossProcess { shadow, highlight } => {
            // Split-tone by luma: pixels below 128 bend toward `shadow`,
            // above toward `highlight`; midtones stay.
            apply_pixelwise(rgba, |p| {
                let y = luma_u8(p[0], p[1], p[2]);
                let (target, t) = if y < 128 {
                    (shadow, (128 - y) as u16) // 0..=128
                } else {
                    (highlight, (y - 128) as u16) // 0..=127
                };
                let w = (t.min(128) * 2).min(255); // scale to 0..=256 then clamp
                let rgb = lerp_rgb([p[0], p[1], p[2]], target, w);
                p[..3].copy_from_slice(&rgb);
            });
        }
        SubjectEffect::Halftone { dot_spacing } => {
            let spacing = dot_spacing.clamp(*SubjectEffect::DOT_SPACING_PX.start(), *SubjectEffect::DOT_SPACING_PX.end());
            let radii_sq = halftone_radii_sq(spacing);
            let (w, h) = rgba.dimensions();
            let row_stride = (w * 4) as usize;
            // From the pixel's centre to its cell's: the table is solved
            // for a disc centred in the cell, so an even spacing must not
            // round the centre to a pixel.
            let from_centre = |v: usize| (v % spacing as usize) as f32 + 0.5 - spacing as f32 / 2.0;

            // Read each pixel's own luma before writing it: a cell's
            // other pixels may already be overwritten by earlier rows.
            let process_row = |(y, row): (usize, &mut [u8])| {
                let dy = from_centre(y);
                for x in 0..w as usize {
                    let p = x * 4;
                    let luma = luma_u8(row[p], row[p + 1], row[p + 2]);
                    let dx = from_centre(x);
                    let dist_sq = dx * dx + dy * dy;
                    let v = if dist_sq < radii_sq[luma as usize] { 0u8 } else { 255 };
                    row[p] = v; row[p + 1] = v; row[p + 2] = v;
                }
            };

            if (w * h) as usize >= ROW_PAR_THRESHOLD {
                rgba.as_mut().par_chunks_mut(row_stride).enumerate().for_each(process_row);
            } else {
                rgba.as_mut().chunks_mut(row_stride).enumerate().for_each(process_row);
            }
        }
        SubjectEffect::GradientMap { stops } => gradient_map(rgba, &stops),
    }
}

/// Each luma's squared dot radius in cells `spacing` px wide (within
/// `SubjectEffect::DOT_SPACING_PX`), worked out once per spacing.
fn halftone_radii_sq(spacing: u32) -> &'static [f32; 256] {
    const SPACINGS: usize = *SubjectEffect::DOT_SPACING_PX.end() as usize + 1;
    static TABLES: [std::sync::OnceLock<[f32; 256]>; SPACINGS] = [const { std::sync::OnceLock::new() }; SPACINGS];
    TABLES[spacing as usize].get_or_init(|| halftone_radii_table(spacing))
}

/// A dot covers its cell's dark share: past the inscribed circle it grows
/// into the corners, and black prints solid. White prints no dot.
fn halftone_radii_table(spacing: u32) -> [f32; 256] {
    let s = f64::from(spacing);
    let edge = s / 2.0;
    let coverage = |r: f64| {
        let disc = std::f64::consts::PI * r * r;
        // The four caps of the disc past the cell's sides.
        let caps = if r <= edge { 0.0 } else { 4.0 * (r * r * (edge / r).acos() - edge * (r * r - edge * edge).sqrt()) };
        (disc - caps) / (s * s)
    };
    std::array::from_fn(|luma| match luma {
        255 => 0.0,
        0 => f32::INFINITY,
        _ => {
            let dark = (255 - luma) as f64 / 255.0;
            // Coverage rises to 1 at the cell's corners and no farther.
            let (mut lo, mut hi) = (0.0, edge * std::f64::consts::SQRT_2);
            for _ in 0..40 {
                let mid = (lo + hi) / 2.0;
                if coverage(mid) < dark { lo = mid } else { hi = mid }
            }
            (hi * hi) as f32
        }
    })
}

/// Map each pixel's luma through `stops` spaced evenly from black (the
/// first) to white (the last), blending linearly between neighbours.
fn gradient_map(rgba: &mut RgbaImage, stops: &[[u8; 3]]) {
    debug_assert!(stops.len() >= 2);
    let segments = (stops.len() - 1) as u16;
    let span = 255 / segments;
    apply_pixelwise(rgba, |p| {
        let y = luma_u8(p[0], p[1], p[2]) as u16;
        let i = (y / span).min(segments - 1) as usize;
        let t = ((y - i as u16 * span) * 255 / span).min(255);
        p[..3].copy_from_slice(&lerp_rgb(stops[i], stops[i + 1], t));
    });
}

/// Rec. 709 luma of an 8-bit RGB triple.
#[inline]
pub(crate) fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 2126 + g as u32 * 7152 + b as u32 * 722) / 10000) as u8
}

/// The step count `posterize_u8` takes for `levels` output levels.
#[inline]
pub(crate) fn posterize_steps(levels: u8) -> u16 {
    levels.max(2) as u16 - 1
}

/// `v` snapped to the nearest of `n + 1` evenly spaced levels, each
/// written rounded.
#[inline]
pub(crate) fn posterize_u8(v: u8, n: u16) -> u8 {
    let (v, n) = (v as u32, n as u32);
    let level = (v * n + 127) / 255;
    ((level * 255 + n / 2) / n) as u8
}

/// The picture `effect` makes from `source`, opaque, at the source's
/// size; `None` for no effect, and for a lens blur without `depth`. A
/// lens blur focuses on the subject of `cutout` (`depth::backdrop_focus`).
/// It goes behind a result like a background image does. Blurred photo
/// and Lens blur blur a clean plate of the photo (`clean_plate`), so the
/// subject's colour does not halo the backdrop around the cut-out.
///
/// Peak working set: the source as RGBA plus the output (2 × w·h·4 bytes;
/// 96 MB at 12 MP), the clean plate (w·h·5 while it is made); a lens
/// blur, `depth::lens_blur`'s (~4 × w·h·4).
pub fn effect_backdrop(
    source: &RgbaImage,
    effect: crate::types::BgEffect,
    depth: Option<&crate::depth::DepthMap>,
    cutout: Option<&RgbaImage>,
) -> Option<RgbaImage> {
    use crate::types::BgEffect;
    let plate = || cutout.and_then(|cutout| clean_plate(source, cutout));
    let mut img = match effect {
        BgEffect::None => return None,
        BgEffect::BlurredSource { radius } => crate::blur::blur_rgba(plate().as_ref().unwrap_or(source), radius.clamp(1, 64) as f32),
        BgEffect::DesaturatedSource => {
            let mut img = source.clone();
            apply_subject_effect(&mut img, SubjectEffect::Desaturate);
            img
        }
        BgEffect::LensBlur { amount_pct } => {
            let depth = depth?;
            let focus = crate::depth::Focus::Depth(crate::depth::backdrop_focus(depth, cutout));
            crate::depth::lens_blur(plate().as_ref().unwrap_or(source), depth, focus, crate::math::pct(amount_pct))
        }
    };
    for p in img.pixels_mut() {
        p.0[3] = 255;
    }
    Some(img)
}

/// Subject alpha up to this is background a backdrop is made of.
const PLATE_BACKGROUND_MAX: u8 = 8;

/// Background kept around the subject's box for the fill to draw from.
const PLATE_MARGIN: u32 = 32;

/// The fill is made at 1/`PLATE_STEP` of the photo's size: it only goes
/// under the subject and is blurred after, so it needs no detail, and a
/// rebuild after each stroke costs about what the blur does.
const PLATE_STEP: u32 = 4;

/// `source` with the subject of `cutout` filled smoothly from the
/// background around it (`inpaint_blend::push_pull`), so a blur of it
/// carries no subject colour past the subject's edge. A cut-out of
/// another size (an upscale) is read at the photo's. `None` with no
/// subject, or no background to fill from.
///
/// Works on the subject's box plus `PLATE_MARGIN`: the alpha at full
/// size, the background's colour and share per `PLATE_STEP`² cell, the
/// fill per channel at that size. Peak working set: the output
/// (w·h·4 bytes) and the alpha (w·h); the cells are 1/16 of the box. A
/// cut-out of another size adds its own alpha while it is resized (an
/// upscale by k: w·h·k² bytes, 190 MB for 4× of 12 MP).
fn clean_plate(source: &RgbaImage, cutout: &RgbaImage) -> Option<RgbaImage> {
    let (w, h) = source.dimensions();
    let alpha = crate::formats::alpha_plane(cutout);
    let alpha = if alpha.dimensions() == (w, h) { alpha } else { resize_gray_lanczos3(&alpha, w, h) };
    let b = crate::inpaint::mask_bbox(&alpha, PLATE_BACKGROUND_MAX + 1, PLATE_MARGIN)?;
    let (cw, ch) = (b.w.div_ceil(PLATE_STEP) as usize, b.h.div_ceil(PLATE_STEP) as usize);
    let step = PLATE_STEP as usize;
    // The box's rows: the alpha's and the RGBA bytes', start to end.
    let (cols, width) = (b.x as usize..(b.x + b.w) as usize, w as usize);
    let rows = || (b.y as usize..(b.y + b.h) as usize).enumerate();
    let alpha_row = |y: usize| &alpha.as_raw()[y * width + cols.start..y * width + cols.end];
    let byte_span = |y: usize| (y * width + cols.start) * 4..(y * width + cols.end) * 4;
    let mut count = vec![0u32; cw * ch];
    let mut sum = vec![[0.0f32; 3]; cw * ch];
    for (by, y) in rows() {
        let cells = (by / step) * cw;
        for (bx, (&a, px)) in alpha_row(y).iter().zip(source.as_raw()[byte_span(y)].chunks_exact(4)).enumerate() {
            if a > PLATE_BACKGROUND_MAX {
                continue;
            }
            let cell = cells + bx / step;
            count[cell] += 1;
            for c in 0..3 {
                sum[cell][c] += f32::from(px[c]);
            }
        }
    }
    if count.iter().all(|&n| n == 0) {
        return None;
    }
    let per_cell = (step * step) as f32;
    let weight: Vec<f32> = count.iter().map(|&n| n as f32 / per_cell).collect();
    let fills: Vec<Vec<f32>> = (0..3)
        .map(|c| {
            let mean: Vec<f32> = sum.iter().zip(&count).map(|(s, &n)| if n > 0 { s[c] / n as f32 } else { 0.0 }).collect();
            crate::inpaint_blend::push_pull(&mean, &weight, cw, ch)
        })
        .collect();
    drop((count, sum, weight));
    let (xs, ys) = (crate::math::bilinear_taps(cw as u32, b.w), crate::math::bilinear_taps(ch as u32, b.h));
    let mut out = source.clone();
    let bytes: &mut [u8] = &mut out;
    for (by, y) in rows() {
        for (bx, (&a, px)) in alpha_row(y).iter().zip(bytes[byte_span(y)].chunks_exact_mut(4)).enumerate() {
            if a <= PLATE_BACKGROUND_MAX {
                continue;
            }
            for (c, plane) in fills.iter().enumerate() {
                px[c] = crate::math::round_u8(crate::math::bilinear_sample(plane, cw, &xs[bx], &ys[by]));
            }
        }
    }
    Some(out)
}

/// Put `effect`'s picture of `photo` under what `rgba` leaves transparent,
/// covering it at any result size (an upscale included); a lens blur
/// focuses on `rgba`'s own subject. False when there is no picture.
pub fn apply_bg_effect(rgba: &mut RgbaImage, photo: &RgbaImage, effect: crate::types::BgEffect, depth: Option<&crate::depth::DepthMap>) -> bool {
    let Some(picture) = effect_backdrop(photo, effect, depth, Some(rgba)) else { return false };
    crate::formats::apply_background_image(rgba, &DynamicImage::ImageRgba8(picture), crate::types::BgImageFit::Cover);
    true
}

/// Convert RGB (0-255) to HSV with H in 0..=359°, S and V in 0..=255.
/// Inlined: it runs per pixel on the 4K live-preview hot path.
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
    (h as u16 % 360, crate::math::round_u8(s * 255.0), crate::math::round_u8(v * 255.0))
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
        crate::math::round_u8((rp + m) * 255.0),
        crate::math::round_u8((gp + m) * 255.0),
        crate::math::round_u8((bp + m) * 255.0),
    )
}

/// Shortest angular distance between two hues in degrees, 0..=180.
#[inline]
fn hue_distance(a: i32, b: i32) -> i32 {
    let d = (a - b).rem_euclid(360);
    if d > 180 { 360 - d } else { d }
}

/// Write the mask into an existing RGBA buffer's alpha channel in place.
///
/// Row-parallel above `ROW_PAR_THRESHOLD`. Memory-bound: ~1.1-1.2× at 4K
/// whatever the core count.
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
    if sigma.is_nan() || sigma <= 0.0 {
        return;
    }
    // Past the image's size a blur is the blur of its size.
    let sigma = sigma.min(mask.width().max(mask.height()) as f32);
    let lo = sigma.floor() as u32;
    let t = sigma - lo as f32;
    if t < 1e-3 {
        box_blur(mask, lo);
        return;
    }
    // Both blurs stay inside the wider one's reach; blend only there.
    let Some(reach) = crate::inpaint::mask_bbox(mask, 1, BLUR_CROP_MARGIN * (lo + 1)) else { return };
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

/// Radii of margin a 3-box blur's crop needs to be bit-exact: the passes
/// reach 3 radii, and a crop edge, where the windows are clipped and
/// divide by fewer pixels, must lie one radius past that, over zeros.
pub(crate) const BLUR_CROP_MARGIN: u32 = 4;

/// 3-box blur at `radius`, cropped to the mask's bbox plus its margin:
/// bit-exact, with a bbox-sized peak instead of ~96 MB at 4K.
fn box_blur(mask: &mut GrayImage, radius: u32) {
    if radius == 0 { return; }
    let Some(bbox) = crate::inpaint::mask_bbox(mask, 1, BLUR_CROP_MARGIN * radius) else {
        return;
    };
    let raw_w = mask.width() as usize;
    let bw = bbox.w as usize;
    let bh = bbox.h as usize;
    let bn = bw * bh;
    let mut buf = vec![0.0f32; bn];
    let raw = mask.as_raw();
    for by in 0..bh {
        let src_off = (bbox.y as usize + by) * raw_w + bbox.x as usize;
        let dst_off = by * bw;
        for bx in 0..bw {
            buf[dst_off + bx] = raw[src_off + bx] as f32;
        }
    }
    crate::guided_filter::box_blur3_into(&mut buf, bbox.w, bbox.h, radius, &mut Vec::new());
    let out = mask.as_mut();
    for by in 0..bh {
        let dst_off = (bbox.y as usize + by) * raw_w + bbox.x as usize;
        let src_off = by * bw;
        for bx in 0..bw {
            out[dst_off + bx] = buf[src_off + bx].clamp(0.0, 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blurred photo and Lens blur blur the background alone: the subject's
    /// colour does not spread past its edge onto the backdrop around it,
    /// the halo it drew there. With no cut-out the whole photo blurs.
    #[test]
    fn a_blurred_backdrop_carries_no_subject_colour_past_its_edge() {
        use crate::types::BgEffect;
        let (red, blue) = (image::Rgba([220, 30, 30, 255]), image::Rgba([30, 30, 220, 255]));
        let photo = RgbaImage::from_fn(64, 32, |x, _| if x < 32 { red } else { blue });
        let cutout = RgbaImage::from_fn(64, 32, |x, _| if x < 32 { red } else { image::Rgba([0, 0, 0, 0]) });
        let blur = |cutout| effect_backdrop(&photo, BgEffect::BlurredSource { radius: 8 }, None, cutout).unwrap();
        let outside = blur(Some(&cutout)).get_pixel(34, 16).0;
        assert!(outside[0] < 40 && outside[2] > 200, "background only: {outside:?}");
        assert!(blur(None).get_pixel(34, 16).0[0] > 80, "no cut-out, no subject to leave out");
    }

    /// The plate changes the photo only under the subject, and fills it
    /// with the background's colours, also from a cut-out of another size.
    #[test]
    fn a_clean_plate_fills_only_under_the_subject() {
        let photo = RgbaImage::from_fn(96, 64, |x, y| {
            let inside = (32..64).contains(&x) && (16..48).contains(&y);
            if inside { image::Rgba([250, 20, 20, 255]) } else { image::Rgba([20, (y * 3) as u8, 200, 255]) }
        });
        let subject = |w: u32, h: u32| RgbaImage::from_fn(w, h, |x, y| {
            let (x, y) = (x * 96 / w, y * 64 / h);
            image::Rgba([0, 0, 0, if (32..64).contains(&x) && (16..48).contains(&y) { 255 } else { 0 }])
        });
        for cutout in [subject(96, 64), subject(192, 128)] {
            let plate = clean_plate(&photo, &cutout).expect("a subject with background around it");
            for (x, y, p) in plate.enumerate_pixels() {
                let deep_inside = (36..60).contains(&x) && (20..44).contains(&y);
                let outside = !(30..66).contains(&x) || !(14..50).contains(&y);
                if outside {
                    assert_eq!(p, photo.get_pixel(x, y), "({x},{y}) is the photo");
                } else if deep_inside {
                    assert!(p.0[0] < 60 && p.0[2] > 150, "({x},{y}) holds the background's colour: {:?}", p.0);
                }
            }
        }
        assert!(clean_plate(&photo, &RgbaImage::new(96, 64)).is_none(), "no subject, no plate");
    }

    /// What the clean plate costs beside the blur it goes into.
    /// `cargo test --profile profiling -p prunr-core --lib time_the_clean_plate -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn time_the_clean_plate() {
        let (w, h) = (4000u32, 3000u32);
        let photo = RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8, 255]));
        let cutout = RgbaImage::from_fn(w, h, |x, y| {
            let (dx, dy) = (x as f32 - 2000.0, y as f32 - 1500.0);
            image::Rgba([0, 0, 0, if dx * dx / 1.0e6 + dy * dy / 1.2e6 < 1.0 { 255 } else { 0 }])
        });
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let plate = clean_plate(&photo, &cutout);
            let plated = t.elapsed();
            let t = std::time::Instant::now();
            let _ = crate::blur::blur_rgba(&photo, 24.0);
            eprintln!("plate {plated:?} (made: {}), blur {:?}", plate.is_some(), t.elapsed());
        }
    }


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

    /// A disc in a canvas far wider than the blur's reach.
    fn disc_in_canvas() -> GrayImage {
        GrayImage::from_fn(96, 80, |x, y| {
            let (dx, dy) = (x as f32 - 40.0, y as f32 - 36.0);
            image::Luma([if dx * dx + dy * dy < 81.0 { 255 } else if dx * dx + dy * dy < 100.0 { 128 } else { 0 }])
        })
    }

    /// The 3-box blur over the whole plane, rounded as `box_blur` writes it.
    fn whole_plane_blur(mask: &GrayImage, radius: u32) -> Vec<u8> {
        let mut buf: Vec<f32> = mask.as_raw().iter().map(|&v| v as f32).collect();
        crate::guided_filter::box_blur3_into(&mut buf, mask.width(), mask.height(), radius, &mut Vec::new());
        buf.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8).collect()
    }

    fn assert_same(got: &[u8], want: &[u8], label: String) {
        let off: Vec<u8> = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).filter(|&d| d > 0).collect();
        assert!(off.is_empty(), "{label}: {} pixels off, by up to {}", off.len(), off.iter().max().unwrap_or(&0));
    }

    /// A feather that is not a number leaves the mask; one past the
    /// image's size blurs as one of its size, without overflow.
    #[test]
    fn a_feather_not_a_number_or_past_the_image_is_sound() {
        let mut m = disc_in_canvas();
        feather_mask(&mut m, f32::NAN);
        assert_eq!(m, disc_in_canvas(), "NaN feathers nothing");
        let mut widest = disc_in_canvas();
        feather_mask(&mut widest, 96.0);
        for sigma in [1e30, f32::INFINITY] {
            let mut m = disc_in_canvas();
            feather_mask(&mut m, sigma);
            assert_eq!(m, widest, "{sigma}");
        }
    }

    /// A gamma that is not a number cuts as gamma 1, not as a clear mask.
    #[test]
    fn a_gamma_not_a_number_cuts_as_one() {
        let tensor = ndarray::Array4::from_shape_fn((1, 1, 1, 64), |(_, _, _, x)| x as f32 / 63.0 * 8.0 - 4.0);
        let original = DynamicImage::ImageRgb8(RgbImage::new(64, 1));
        let mask = |gamma: f32| {
            let settings = MaskSettings { gamma, ..MaskSettings::default() };
            tensor_to_mask(tensor.view(), &original, &PostprocessOpts::new(&settings, ModelKind::Silueta))
        };
        assert_eq!(mask(f32::NAN), mask(1.0));
    }

    /// Cropping the blur to the mask's reach changes no pixel: the crop
    /// edge, where the box windows are clipped, lies where the whole-image
    /// blur reads only zeros.
    #[test]
    fn the_cropped_blur_equals_the_whole_image_blur() {
        for radius in 1..=6 {
            let mut cropped = disc_in_canvas();
            box_blur(&mut cropped, radius);
            assert_same(cropped.as_raw(), &whole_plane_blur(&disc_in_canvas(), radius), format!("radius {radius}"));
        }
    }

    #[test]
    fn a_fractional_feather_equals_the_whole_image_blend() {
        for sigma in [0.4f32, 1.5, 2.25, 4.7] {
            let lo = sigma.floor() as u32;
            let t = sigma - lo as f32;
            let narrow = if lo == 0 { disc_in_canvas().into_raw() } else { whole_plane_blur(&disc_in_canvas(), lo) };
            let wider = whole_plane_blur(&disc_in_canvas(), lo + 1);
            let want: Vec<u8> = narrow.iter().zip(&wider).map(|(&a, &b)| (a as f32 * (1.0 - t) + b as f32 * t).round() as u8).collect();
            let mut got = disc_in_canvas();
            feather_mask(&mut got, sigma);
            assert_same(got.as_raw(), &want, format!("sigma {sigma}"));
        }
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
        // A flat output at or below one half reads as no subject.
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

    /// Finishing from the kept mask makes the same cut-out as the whole
    /// pass, whatever the look.
    #[test]
    fn a_cut_out_finished_from_its_mask_is_the_whole_pass() {
        let flat: Vec<f32> = (0..64 * 64).map(|i| (i % 64) as f32 / 64.0).collect();
        let original = RgbaImage::from_fn(96, 80, |x, y| image::Rgba([(x * 2) as u8, (y * 3) as u8, 90, 255]));
        let looks = [
            MaskSettings { refine_edges: true, feather: 1.5, ..MaskSettings::default() },
            MaskSettings { effect: crate::types::SubjectEffect::Sepia, ..MaskSettings::default() },
            MaskSettings {
                decoration: crate::types::Decoration { outline_width: 2.0, shadow: crate::types::ShadowKind::Drop, ..Default::default() },
                ..MaskSettings::default()
            },
        ];
        for settings in looks {
            let opts = PostprocessOpts::new(&settings, ModelKind::Silueta);
            let whole = postprocess_from_flat(&flat, 64, 64, &original, &opts).expect("fits");
            let mask = mask_from_flat(&flat, 64, 64, &original, &opts).expect("fits");
            assert_eq!(finish_with_mask(&original, &mask, &settings), whole, "{settings:?}");
        }
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

    /// A cut of an earlier cut-out only takes more away: what its image
    /// hides stays hidden, half-transparent pixels keep at most their share,
    /// and a Restore stroke still brings the colours under it back.
    #[test]
    fn a_cut_keeps_what_its_start_image_hides() {
        use crate::brush::paint_circle;
        use crate::selection::BrushMode;
        let raw = make_raw_tensor(1.0);
        let mut start = image::RgbaImage::from_pixel(320, 320, image::Rgba([90, 90, 90, 255]));
        for (x, _, p) in start.enumerate_pixels_mut() {
            if x < 100 { p.0[3] = 0 } else if x < 200 { p.0[3] = 128 }
        }
        let start = DynamicImage::ImageRgba8(start);
        let mask_settings = MaskSettings::default();
        let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);
        let opaque = tensor_to_mask(raw.view(), &solid_rgb(320, 320), &opts).get_pixel(250, 160)[0];

        let mask = tensor_to_mask(raw.view(), &start, &opts);
        assert_eq!(mask.get_pixel(50, 160)[0], 0, "hidden stays hidden");
        let half = mask.get_pixel(150, 160)[0];
        assert!((126..=130).contains(&half), "half-transparent keeps its share of the model's {opaque}, got {half}");
        assert_eq!(mask.get_pixel(250, 160)[0], opaque, "opaque follows the model alone");
        let rgba = postprocess(raw.view(), &start, &opts);
        assert_eq!(rgba.get_pixel(50, 160)[3], 0, "the composite agrees");

        let mut restore = MaskArtifact::new_empty(320, 320);
        paint_circle(&mut restore, 50.0, 160.0, 20.0, crate::brush::Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        let restored = tensor_to_mask(raw.view(), &start, &opts.with_correction(Some(&restore)));
        assert!(restored.get_pixel(50, 160)[0] >= 254, "a Restore stroke wins over the hidden start");
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

    /// A halftone's ink covers each grey's dark share: black prints
    /// solid, white prints nothing, and mid grey about half.
    #[test]
    fn halftone_ink_follows_the_darkness() {
        let ink = |grey: u8| {
            let mut img = image::RgbaImage::from_pixel(64, 64, image::Rgba([grey, grey, grey, 255]));
            apply_subject_effect(&mut img, crate::types::SubjectEffect::Halftone { dot_spacing: 8 });
            img.pixels().filter(|p| p[0] == 0).count() as f32 / (64.0 * 64.0)
        };
        assert_eq!(ink(0), 1.0, "black is solid");
        assert_eq!(ink(255), 0.0, "white has no dots");
        let mid = ink(128);
        assert!((mid - 0.5).abs() < 0.01, "mid grey inks {mid}");
        assert!(ink(64) > ink(128) && ink(128) > ink(192), "darker greys ink more");
        // Each grey misses its share by whole pixels of an 8 × 8 cell, but
        // not all one way.
        let bias = (1..255u8).map(|g| ink(g) - f32::from(255 - g) / 255.0).sum::<f32>() / 254.0;
        assert!(bias.abs() < 0.01, "greys ink {bias:+} off on average");
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
        apply_subject_effect(
            &mut img,
            crate::types::SubjectEffect::Halftone { dot_spacing: spacing },
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
    /// through `apply_subject_effect` directly on the source RGB. The toolbar
    /// chip claims every `SubjectEffect::ALL` variant produces a different
    /// output; without a test that's an unverified assertion. Run each
    /// variant on a colourful 4×4 patch and assert:
    ///
    /// - `SubjectEffect::None` is bit-exact identity (no-op contract).
    /// - Every other variant differs from the input (variant did something).
    /// - All non-None variants produce pairwise-distinct outputs (no two
    ///   variants accidentally collapse to the same bytes).
    #[test]
    fn subject_effect_all_variants_produce_distinct_output() {
        use crate::types::SubjectEffect;

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

        let mut outputs: Vec<(String, image::RgbaImage)> = SubjectEffect::ALL.iter()
            .map(|style| {
                let mut img = src.clone();
                apply_subject_effect(&mut img, *style);
                (style.name().to_string(), img)
            })
            .collect();

        // None is identity.
        let none_idx = outputs.iter().position(|(n, _)| n == "None")
            .expect("SubjectEffect::None must be in ALL");
        assert_eq!(outputs[none_idx].1, src, "SubjectEffect::None must be bit-exact identity");

        // Every other variant differs from input.
        for (name, img) in outputs.iter().filter(|(n, _)| n != "None") {
            assert_ne!(img, &src, "SubjectEffect::{name} produced bit-exact-identity output (variant is dead)");
        }

        // No two variants produce the same output.
        outputs.sort_by(|a, b| a.0.cmp(&b.0));
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i].1, outputs[j].1,
                    "SubjectEffect::{} and SubjectEffect::{} collapse to identical bytes",
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
        apply_subject_effect(&mut img, crate::types::SubjectEffect::Pixelate { block_size: 8 });
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
        let depth = crate::depth::DepthMap::from_raw(2, 2, vec![0.0, 0.0, 1.0, 1.0]);
        assert!(effect_backdrop(&photo, BgEffect::None, Some(&depth), None).is_none());
        assert!(effect_backdrop(&photo, BgEffect::LensBlur { amount_pct: 50 }, None, None).is_none(), "a lens blur waits for depth");
        for effect in BgEffect::ALL.iter().filter(|e| **e != BgEffect::None) {
            let b = effect_backdrop(&photo, *effect, Some(&depth), None).expect("an effect makes a picture");
            assert_eq!(b.dimensions(), (8, 4));
            assert!(b.pixels().all(|p| p.0[3] == 255), "{effect:?} is opaque");
        }
        let grey = effect_backdrop(&photo, BgEffect::DesaturatedSource, None, None).unwrap();
        let p = grey.get_pixel(6, 1).0;
        assert!(p[0] == p[1] && p[1] == p[2], "black and white");
        let blur = effect_backdrop(&photo, BgEffect::BlurredSource { radius: 2 }, None, None).unwrap();
        let (left, right) = (blur.get_pixel(3, 2).0[2], blur.get_pixel(4, 2).0[2]);
        assert!(left > 40 && right < 220, "the two halves bleed into each other: {left} {right}");
    }

    /// Duotone and Gradient map share one evenly-spaced gradient map; at
    /// Each value goes to its nearest level: two levels split at
    /// mid-grey, where rounding down sent everything under white to black.
    #[test]
    fn posterize_snaps_each_value_to_its_nearest_level() {
        let two = posterize_steps(2);
        assert_eq!((posterize_u8(127, two), posterize_u8(128, two), posterize_u8(254, two)), (0, 255, 255));
        let five = posterize_steps(5);
        let levels: std::collections::BTreeSet<u8> = (0..=255u8).map(|v| posterize_u8(v, five)).collect();
        assert_eq!(levels.into_iter().collect::<Vec<_>>(), [0, 64, 128, 191, 255]);
        assert_eq!((posterize_u8(90, five), posterize_u8(100, five)), (64, 128));
    }

    /// every grey level it gives what each computed on its own before.
    #[test]
    fn posterize_keeps_black_and_white_and_snaps_to_its_levels() {
        for levels in [0u8, 1, 2, 4, 255] {
            let n = posterize_steps(levels);
            assert_eq!(posterize_u8(0, n), 0);
            assert_eq!(posterize_u8(255, n), 255);
            let distinct: std::collections::BTreeSet<u8> = (0..=255u8).map(|v| posterize_u8(v, n)).collect();
            assert_eq!(distinct.len(), n as usize + 1, "levels {levels}");
        }
    }

    #[test]
    fn duotone_and_gradient_map_blend_to_the_nearest_byte() {
        let blend = |lo: [u8; 3], hi: [u8; 3], t: u16| -> [u8; 3] {
            let inv = 255 - t;
            std::array::from_fn(|c| ((lo[c] as u16 * inv + hi[c] as u16 * t + 127) / 255) as u8)
        };
        let (dark, light) = ([10, 20, 30], [240, 200, 100]);
        let stops = [[0, 0, 80], [200, 0, 120], [255, 140, 0], [255, 255, 200]];
        for y in 0..=255u8 {
            let mut img = RgbaImage::from_pixel(1, 1, image::Rgba([y, y, y, 255]));
            gradient_map(&mut img, &[dark, light]);
            assert_eq!(img.get_pixel(0, 0).0[..3], blend(dark, light, y as u16), "duotone at {y}");

            let y16 = y as u16;
            let (lo, hi, t) = if y16 <= 85 { (stops[0], stops[1], y16 * 255 / 85) }
                else if y16 <= 170 { (stops[1], stops[2], (y16 - 85) * 255 / 85) }
                else { (stops[2], stops[3], (y16 - 170) * 255 / 85) };
            let mut img = RgbaImage::from_pixel(1, 1, image::Rgba([y, y, y, 255]));
            gradient_map(&mut img, &stops);
            assert_eq!(img.get_pixel(0, 0).0[..3], blend(lo, hi, t.min(255)), "gradient map at {y}");
        }
    }
}

