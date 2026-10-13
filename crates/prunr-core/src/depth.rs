//! Depth: Depth Anything V2 predicts how near each part of a photo is.
//! The map is small (the model's resolution) and kept per photo; the
//! effects sample it at the photo's own size.

use image::{GrayImage, ImageBuffer, Luma, RgbaImage};
use ort::{inputs, value::Tensor};
use rayon::prelude::*;

use crate::math::{bilinear_taps, smoothstep, Tap};
use crate::progress::{Progress, Step};
use crate::selection::{BrushMode, MaskArtifact, FULL};
use crate::{CoreError, OrtEngine};

/// The model's long side; a multiple of its 14-pixel patch.
pub const DEPTH_LONG_SIDE: u32 = 518;
const PATCH: u32 = 14;

/// A relative depth map: 1 = the nearest point in the photo, 0 = the
/// farthest. Only the order is meaningful, not distances.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    width: u32,
    height: u32,
    near: Vec<f32>,
}

impl DepthMap {
    /// A map from raw model output (larger = nearer), stretched to 0..1.
    pub fn from_raw(width: u32, height: u32, mut raw: Vec<f32>) -> Self {
        debug_assert_eq!(raw.len(), (width * height) as usize);
        let (lo, hi, _) = crate::math::slice_stats(&raw);
        let span = (hi - lo).max(1e-6);
        raw.iter_mut().for_each(|v| *v = ((*v - lo) / span).clamp(0.0, 1.0));
        Self { width, height, near: raw }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn byte_size(&self) -> usize {
        self.near.len() * std::mem::size_of::<f32>()
    }

    /// Nearness at a point given as fractions of the photo's width and
    /// height.
    pub fn at(&self, fx: f32, fy: f32) -> f32 {
        let x = ((fx * self.width as f32) as u32).min(self.width - 1);
        let y = ((fy * self.height as f32) as u32).min(self.height - 1);
        self.near[(y * self.width + x) as usize]
    }

    /// Fills `out`, a `w × h` image with `per_px` values per pixel, row
    /// by row in parallel: `f` gets the row index, the map's nearness
    /// along that row (sampled bilinearly) and the row to write.
    fn fill_rows<T: Send>(&self, w: u32, h: u32, out: &mut [T], per_px: usize, f: impl Fn(usize, &[f32], &mut [T]) + Sync) {
        let (xs, ys) = (bilinear_taps(self.width, w), bilinear_taps(self.height, h));
        let stride = self.width as usize;
        out.par_chunks_exact_mut(w as usize * per_px).zip(ys.par_iter()).enumerate().for_each(|(y, (row, ty))| {
            let near: Vec<f32> = xs.iter().map(|tx| self.sample(stride, tx, ty)).collect();
            f(y, &near, row);
        });
    }

    fn sample(&self, stride: usize, tx: &Tap, ty: &Tap) -> f32 {
        let p = &self.near;
        let (r0, r1) = (ty.lo * stride, ty.hi * stride);
        let top = p[r0 + tx.lo] + (p[r0 + tx.hi] - p[r0 + tx.lo]) * tx.t;
        let bot = p[r1 + tx.lo] + (p[r1 + tx.hi] - p[r1 + tx.lo]) * tx.t;
        top + (bot - top) * ty.t
    }

    /// The map at `w × h` as a 16-bit grey image, white = near, for
    /// export to other tools.
    pub fn to_gray16(&self, w: u32, h: u32) -> ImageBuffer<Luma<u16>, Vec<u16>> {
        let mut out = vec![0u16; (w * h) as usize];
        self.fill_rows(w, h, &mut out, 1, |_, near, row| {
            for (v, d) in row.iter_mut().zip(near) {
                *v = (d * 65535.0).round() as u16;
            }
        });
        // sized w × h above
        ImageBuffer::from_raw(w, h, out).expect("buffer sized w × h")
    }
}

/// The model's input size for a `w × h` photo: the long side at
/// `DEPTH_LONG_SIDE`, both sides multiples of the patch.
fn input_size(w: u32, h: u32) -> (u32, u32) {
    let scale = DEPTH_LONG_SIDE as f32 / w.max(h).max(1) as f32;
    let snap = |v: u32| (((v as f32 * scale) / PATCH as f32).round() as u32).max(1) * PATCH;
    (snap(w), snap(h))
}

/// Estimate the depth of `src`.
///
/// Peak working set: the resized copy and its NCHW tensor at
/// `DEPTH_LONG_SIDE` (~4 MB), the model's own scratch
/// (`ModelDescriptor.working_set_mb`), and the returned map (~1 MB).
pub fn estimate_depth(engine: &OrtEngine, src: &RgbaImage, progress: &Progress) -> Result<DepthMap, CoreError> {
    progress.step(Step::EstimatingDepth);
    let (iw, ih) = input_size(src.width(), src.height());
    let resized = crate::formats::resize_rgba(src, iw, ih, crate::formats::ResizeFilter::Lanczos3);
    let n = (iw * ih) as usize;
    let mut chw = vec![0.0f32; 3 * n];
    for (i, px) in resized.as_raw().chunks_exact(4).enumerate() {
        for c in 0..3 {
            chw[c * n + i] = (px[c] as f32 / 255.0 - crate::preprocess::IMAGENET_MEAN[c]) / crate::preprocess::IMAGENET_STD[c];
        }
    }
    drop(resized);
    let input = ndarray::Array4::from_shape_vec((1, 3, ih as usize, iw as usize), chw)
        .map_err(|e| CoreError::Inference(format!("depth: input shape: {e}")))?;
    let tensor = Tensor::from_array(input).map_err(|e| CoreError::Inference(format!("depth: input tensor: {e}")))?;
    if progress.is_cancelled() {
        return Err(CoreError::Cancelled);
    }
    let raw = engine.with_session(|session| {
        let outputs = session.run(inputs!["pixel_values" => &tensor])
            .map_err(|e| CoreError::Inference(format!("depth: inference failed: {e}")))?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(format!("depth: output: {e}")))?;
        Ok(data.to_vec())
    })?;
    if raw.len() != n {
        return Err(CoreError::Inference(format!("depth: output has {} values, expected {n}", raw.len())));
    }
    if progress.is_cancelled() {
        return Err(CoreError::Cancelled);
    }
    Ok(DepthMap::from_raw(iw, ih, raw))
}

/// What stays sharp under `lens_blur`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Focus {
    /// Everything at this nearness (0 = far, 1 = near); blur grows with
    /// the depth difference.
    Depth(f32),
    /// Tilt-shift: a horizontal band centred at this fraction of the
    /// height, `half_height` thick on each side; blur grows away from it.
    Band { center: f32, half_height: f32 },
}

/// Depth difference (or band distance) at which the blur is full.
const FULL_BLUR_AT: f32 = 0.5;
/// Blur levels between sharp and the full blur.
const LEVELS: usize = 4;

/// A camera-lens look: `src` blurred more the farther each pixel is from
/// `focus`. `amount` in 0..=1 sets the strongest blur, up to 1% of the
/// photo's long side. Alpha is blurred alike.
///
/// Peak working set (24 MP photo):
/// | Buffer                     | Size        |
/// |----------------------------|-------------|
/// | result                     | 96 MB       |
/// | previous and current level | 2 × 96 MB   |
/// | per-pixel level positions  | 96 MB       |
/// Total ~384 MB beside `src`; levels are made one at a time, not kept.
pub fn lens_blur(src: &RgbaImage, depth: &DepthMap, focus: Focus, amount: f32) -> RgbaImage {
    let (w, h) = src.dimensions();
    let sigma_max = amount.clamp(0.0, 1.0) * w.max(h) as f32 / 100.0;
    if sigma_max < 0.3 {
        return src.clone();
    }
    // Where each pixel falls between the levels: 0 = sharp, LEVELS = full.
    let mut pos = vec![0.0f32; (w * h) as usize];
    depth.fill_rows(w, h, &mut pos, 1, |y, near, row| {
        for (p, &d) in row.iter_mut().zip(near) {
            let off = match focus {
                Focus::Depth(f) => (d - f).abs(),
                Focus::Band { center, half_height } => ((y as f32 + 0.5) / h as f32 - center).abs() - half_height,
            };
            *p = smoothstep((off / FULL_BLUR_AT).clamp(0.0, 1.0)) * LEVELS as f32;
        }
    });
    let mut out = src.clone();
    let mut prev = src.clone();
    for k in 1..=LEVELS {
        let cur = crate::blur::blur_rgba(src, sigma_max * k as f32 / LEVELS as f32);
        let lo = (k - 1) as f32;
        out.as_mut()
            .par_chunks_exact_mut(4)
            .zip(prev.as_raw().par_chunks_exact(4))
            .zip(cur.as_raw().par_chunks_exact(4))
            .zip(pos.par_iter())
            .for_each(|(((o, p), c), &t)| {
                if t > lo {
                    let f = (t - lo).min(1.0);
                    for i in 0..4 {
                        o[i] = (p[i] as f32 + (c[i] as f32 - p[i] as f32) * f).round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
        prev = cur;
    }
    out
}

/// Haze that thickens with distance: each pixel moves toward `color` by
/// `amount` times how far it is. Alpha is unchanged.
///
/// Peak working set: the result (w·h·4 bytes) and one row of nearness.
pub fn depth_fog(src: &RgbaImage, depth: &DepthMap, amount: f32, color: [u8; 3]) -> RgbaImage {
    let (w, h) = src.dimensions();
    let amount = amount.clamp(0.0, 1.0);
    let row_len = w as usize * 4;
    let mut out = vec![0u8; src.as_raw().len()];
    depth.fill_rows(w, h, &mut out, 4, |y, near, row| {
        let src_row = &src.as_raw()[y * row_len..(y + 1) * row_len];
        for ((o, px), &d) in row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)).zip(near) {
            let f = amount * smoothstep(1.0 - d);
            let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round().clamp(0.0, 255.0) as u8;
            o.copy_from_slice(&[mix(px[0], color[0]), mix(px[1], color[1]), mix(px[2], color[2]), px[3]]);
        }
    });
    // sized from src
    RgbaImage::from_raw(w, h, out).expect("buffer sized from src")
}

/// How far `aerial_perspective` moves the farthest pixel at full amount:
/// b* toward blue, the share of chroma removed, the share of the way to
/// white L is lifted. The nearest pixel warms by a quarter of the shift.
const AERIAL_BLUE: f32 = 12.0;
const AERIAL_DESATURATE: f32 = 0.6;
const AERIAL_LIFT: f32 = 0.15;
const AERIAL_WARM_SHARE: f32 = 0.25;

/// Colour that recedes: far pixels turn cooler and paler, the way
/// distant hills do, and near ones a touch warmer. Works in Lab; L is
/// lifted by an affine map with a positive slope at each depth, so no
/// two pixels of one depth swap lightness order. Alpha is unchanged.
///
/// Peak working set: the result (w·h·4 bytes) and one row of nearness.
pub fn aerial_perspective(src: &RgbaImage, depth: &DepthMap, amount: f32) -> RgbaImage {
    let (w, h) = src.dimensions();
    let amount = amount.clamp(0.0, 1.0);
    if amount == 0.0 {
        return src.clone();
    }
    let row_len = w as usize * 4;
    let mut out = vec![0u8; src.as_raw().len()];
    depth.fill_rows(w, h, &mut out, 4, |y, near, row| {
        let src_row = &src.as_raw()[y * row_len..(y + 1) * row_len];
        for ((o, px), &d) in row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)).zip(near) {
            let far = amount * smoothstep(1.0 - d);
            let warm = amount * AERIAL_WARM_SHARE * smoothstep(d);
            let (l, a, b) = crate::color::srgb_to_lab(px[0], px[1], px[2]);
            let keep = 1.0 - AERIAL_DESATURATE * far;
            let l = l + AERIAL_LIFT * far * (100.0 - l);
            let a = a * keep + 0.5 * AERIAL_BLUE * warm;
            let b = b * keep - AERIAL_BLUE * far + AERIAL_BLUE * warm;
            let (r, g, bl) = crate::color::lab_to_srgb(l, a, b);
            o.copy_from_slice(&[r, g, bl, px[3]]);
        }
    });
    // sized from src
    RgbaImage::from_raw(w, h, out).expect("buffer sized from src")
}

/// A selection of everything within `range` of nearness `around`, for a
/// `w × h` photo, pushed toward `mode`.
///
/// Peak working set: the w·h cells (1 byte each).
pub fn select_by_depth(depth: &DepthMap, w: u32, h: u32, around: f32, range: f32, mode: BrushMode) -> MaskArtifact {
    let mut cells = vec![0i8; (w * h) as usize];
    let value = mode.sign() * FULL;
    depth.fill_rows(w, h, &mut cells, 1, |_, near, row| {
        for (c, &d) in row.iter_mut().zip(near) {
            *c = if (d - around).abs() <= range { value } else { 0 };
        }
    });
    MaskArtifact::from_cells(w, h, cells)
}

/// Fade with distance for one render: the map and how much (0..=1).
#[derive(Clone, Copy, Debug)]
pub struct LineFade<'a> {
    pub depth: &'a DepthMap,
    pub amount: f32,
}

impl LineFade<'_> {
    /// Fade `plane` (see `fade_lines_by_depth`).
    pub fn apply(&self, plane: &mut GrayImage) {
        fade_lines_by_depth(plane, self.depth, self.amount);
    }
}

/// Lines lighter with distance: each line pixel keeps `1 - fade × far`
/// of its strength, so the nearest lines stay full and, at `fade` 1,
/// the farthest vanish.
pub fn fade_lines_by_depth(plane: &mut GrayImage, depth: &DepthMap, fade: f32) {
    let fade = fade.clamp(0.0, 1.0);
    if fade == 0.0 {
        return;
    }
    let (w, h) = plane.dimensions();
    depth.fill_rows(w, h, plane.as_mut(), 1, |_, near, row| {
        for (v, &d) in row.iter_mut().zip(near) {
            *v = (*v as f32 * (1.0 - fade * (1.0 - d))).round() as u8;
        }
    });
}

/// The haze `depth_fog` draws toward in the app and the CLI: a light
/// grey-blue, like air over distance.
pub const FOG_COLOR: [u8; 3] = [214, 220, 228];

/// A tilt-shift band's half height, as a fraction of the photo's height.
pub const BAND_HALF_HEIGHT: f32 = 0.12;

/// `DepthLook::focus` units per photo side.
const FOCUS_SCALE: f32 = 10_000.0;

/// The longest side of the grid `subject_nearness` samples.
const SUBJECT_SAMPLES: u32 = 512;

/// How the Depth chip blurs a photo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LensBlurMode {
    #[default]
    Off,
    /// Sharp at one depth, blurring with the depth difference.
    Point,
    /// Sharp in a horizontal band, blurring above and below it.
    TiltShift,
}

/// What the Depth chip does to a photo before anything else draws on it:
/// a lens blur and a haze, both read from its depth map. Whole percent
/// and ten-thousandths keep it `Eq` and `Hash` for the recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DepthLook {
    pub blur: LensBlurMode,
    /// Blur strength, 0-100.
    pub blur_pct: u8,
    /// The focus point as ten-thousandths of the photo's width and
    /// height; `None` focuses automatically.
    pub focus: Option<[u16; 2]>,
    /// Haze strength, 0-100.
    pub fog_pct: u8,
    /// Aerial perspective strength, 0-100.
    pub aerial_pct: u8,
}

impl DepthLook {
    /// Whether it changes the photo at all.
    pub fn is_active(&self) -> bool {
        self.blurs() || self.fog_pct > 0 || self.aerial_pct > 0
    }

    fn blurs(&self) -> bool {
        self.blur != LensBlurMode::Off && self.blur_pct > 0
    }

    pub fn focus_point(&self) -> Option<(f32, f32)> {
        self.focus.map(|[x, y]| (f32::from(x) / FOCUS_SCALE, f32::from(y) / FOCUS_SCALE))
    }

    /// Focus on a point given as fractions of the photo's width and height.
    pub fn set_focus(&mut self, fx: f32, fy: f32) {
        let unit = |v: f32| (v.clamp(0.0, 1.0) * FOCUS_SCALE).round() as u16;
        self.focus = Some([unit(fx), unit(fy)]);
    }

    /// What stays sharp: the depth at the focus point (the nearest tenth
    /// of the photo when none is set), or a band through it (the middle
    /// when none is set).
    pub fn lens_focus(&self, depth: &DepthMap) -> Focus {
        let point = self.focus_point();
        match self.blur {
            LensBlurMode::TiltShift => Focus::Band { center: point.map_or(0.5, |(_, y)| y), half_height: BAND_HALF_HEIGHT },
            LensBlurMode::Off | LensBlurMode::Point => Focus::Depth(point.map_or_else(|| auto_focus(depth), |(x, y)| depth.at(x, y))),
        }
    }

    /// `photo` blurred, then hazed, then graded for aerial perspective;
    /// `None` when it has none of them. The blur goes first so the haze
    /// and the grade stay smooth over what it softened.
    ///
    /// Peak working set: `lens_blur`'s (~4 photos); each later stage holds
    /// its input and its output (2 photos).
    pub fn apply(&self, photo: &RgbaImage, depth: &DepthMap) -> Option<RgbaImage> {
        if !self.is_active() {
            return None;
        }
        let pct = |v: u8| f32::from(v.min(100)) / 100.0;
        let mut out: Option<RgbaImage> = None;
        if self.blurs() {
            out = Some(lens_blur(photo, depth, self.lens_focus(depth), pct(self.blur_pct)));
        }
        if self.fog_pct > 0 {
            out = Some(depth_fog(out.as_ref().unwrap_or(photo), depth, pct(self.fog_pct), FOG_COLOR));
        }
        if self.aerial_pct > 0 {
            out = Some(aerial_perspective(out.as_ref().unwrap_or(photo), depth, pct(self.aerial_pct)));
        }
        out
    }
}

/// The nearness that is nearer than 90% of the photo: where a camera
/// with nothing else to go on would focus.
pub fn auto_focus(depth: &DepthMap) -> f32 {
    let mut near = depth.near.clone();
    percentile(&mut near, 0.9).unwrap_or(1.0)
}

/// The median nearness under the opaque part of `cutout` (its subject),
/// sampled on a grid; `None` when nothing is opaque.
pub fn subject_nearness(depth: &DepthMap, cutout: &RgbaImage) -> Option<f32> {
    let (w, h) = cutout.dimensions();
    let step = (w.max(h) / SUBJECT_SAMPLES).max(1);
    let mut under: Vec<f32> = (0..h).step_by(step as usize)
        .flat_map(|y| (0..w).step_by(step as usize).map(move |x| (x, y)))
        .filter(|&(x, y)| cutout.get_pixel(x, y).0[3] >= 128)
        .map(|(x, y)| depth.at((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32))
        .collect();
    percentile(&mut under, 0.5)
}

/// What a lens-blur background focuses on: the cut-out's subject, or
/// `auto_focus` without one.
pub fn backdrop_focus(depth: &DepthMap, cutout: Option<&RgbaImage>) -> f32 {
    cutout.and_then(|c| subject_nearness(depth, c)).unwrap_or_else(|| auto_focus(depth))
}

fn percentile(values: &mut [f32], q: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let k = ((values.len() - 1) as f32 * q).round() as usize;
    let (_, v, _) = values.select_nth_unstable_by(k, f32::total_cmp);
    Some(*v)
}

/// The map at `w × h` as a 16-bit grey PNG (white = near).
pub fn encode_depth_png(depth: &DepthMap, w: u32, h: u32) -> Result<Vec<u8>, CoreError> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma16(depth.to_gray16(w, h)).write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Near at the bottom, far at the top, like a ground plane.
    fn ramp(w: u32, h: u32) -> DepthMap {
        let raw = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
        DepthMap::from_raw(w, h, raw)
    }

    fn checker(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| if (x / 2 + y / 2) % 2 == 0 { image::Rgba([255, 255, 255, 255]) } else { image::Rgba([0, 0, 0, 255]) })
    }

    fn contrast(img: &RgbaImage, y: u32) -> u8 {
        let row: Vec<u8> = (0..img.width()).map(|x| img.get_pixel(x, y).0[0]).collect();
        row.iter().max().unwrap_or(&0) - row.iter().min().unwrap_or(&0)
    }

    #[test]
    fn the_map_is_stretched_to_near_one_far_zero() {
        let d = ramp(4, 10);
        assert_eq!(d.at(0.5, 0.0), 0.0);
        assert_eq!(d.at(0.5, 0.99), 1.0);
    }

    #[test]
    fn input_sides_are_patch_multiples_with_the_long_side_at_518() {
        assert_eq!(input_size(6000, 4000), (518, 350));
        assert_eq!(input_size(800, 1200), (350, 518));
        let (w, h) = input_size(37, 3);
        assert_eq!((w % PATCH, h % PATCH), (0, 0));
    }

    #[test]
    fn no_amount_leaves_the_photo_as_is() {
        let src = checker(64, 64);
        assert_eq!(lens_blur(&src, &ramp(8, 8), Focus::Depth(0.5), 0.0), src);
    }

    #[test]
    fn the_focused_depth_stays_sharp_and_the_rest_blurs() {
        let src = checker(200, 200);
        let out = lens_blur(&src, &ramp(20, 20), Focus::Depth(1.0), 1.0);
        assert!(contrast(&out, 198) > 200, "near row in focus: {}", contrast(&out, 198));
        assert!(contrast(&out, 2) < 60, "far row blurred: {}", contrast(&out, 2));
    }

    #[test]
    fn tilt_shift_keeps_its_band_sharp_whatever_the_depth() {
        let src = checker(200, 200);
        let flat = DepthMap::from_raw(2, 2, vec![0.0, 0.0, 0.0, 1.0]);
        let out = lens_blur(&src, &flat, Focus::Band { center: 0.5, half_height: 0.1 }, 1.0);
        assert!(contrast(&out, 100) > 200, "band sharp");
        assert!(contrast(&out, 2) < 60, "top blurred");
    }

    #[test]
    fn fog_thickens_with_distance_and_keeps_alpha() {
        let src = RgbaImage::from_pixel(10, 100, image::Rgba([0, 0, 0, 77]));
        let out = depth_fog(&src, &ramp(10, 100), 1.0, [200, 210, 220]);
        let (far, near) = (out.get_pixel(5, 0).0, out.get_pixel(5, 99).0);
        assert!(far[0] > 190 && near[0] < 10, "far {far:?} near {near:?}");
        assert_eq!(far[3], 77);
        assert_eq!(depth_fog(&src, &ramp(10, 100), 0.0, [200, 210, 220]), src);
    }

    #[test]
    fn selecting_by_depth_takes_only_the_band_around_the_click() {
        let m = select_by_depth(&ramp(10, 100), 10, 100, 0.5, 0.1, BrushMode::Add);
        let at = |y: usize| m.cells()[y * 10 + 3];
        assert_eq!((at(50), at(5), at(95)), (FULL, 0, 0));
        let sub = select_by_depth(&ramp(10, 100), 10, 100, 0.5, 0.1, BrushMode::Subtract);
        assert_eq!(sub.cells()[50 * 10 + 3], -FULL);
    }

    #[test]
    fn faded_lines_keep_near_strength_and_lose_far() {
        let mut plane = GrayImage::from_pixel(10, 100, Luma([200]));
        fade_lines_by_depth(&mut plane, &ramp(10, 100), 1.0);
        assert!(plane.get_pixel(3, 99).0[0] >= 198 && plane.get_pixel(3, 0).0[0] <= 2);
    }

    #[test]
    fn the_export_spans_the_full_16_bit_range() {
        let img = ramp(4, 50).to_gray16(4, 50);
        assert_eq!(img.get_pixel(1, 0).0[0], 0);
        assert_eq!(img.get_pixel(1, 49).0[0], 65535);
    }

    #[test]
    fn auto_focus_is_the_nearest_tenth() {
        let f = auto_focus(&ramp(1, 101));
        assert!((f - 0.9).abs() < 0.02, "{f}");
    }

    #[test]
    fn the_subject_is_focused_at_its_median_depth() {
        let depth = ramp(10, 100);
        let mut cut = RgbaImage::new(10, 100);
        for y in 70..80 {
            for x in 0..10 {
                cut.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
            }
        }
        let n = subject_nearness(&depth, &cut).expect("opaque pixels");
        assert!((n - 0.75).abs() < 0.03, "{n}");
        assert_eq!(subject_nearness(&depth, &RgbaImage::new(10, 100)), None);
        assert_eq!(backdrop_focus(&depth, Some(&RgbaImage::new(10, 100))), auto_focus(&depth));
    }

    #[test]
    fn a_look_focuses_where_it_is_told_or_automatically() {
        let depth = ramp(10, 100);
        let mut look = DepthLook { blur: LensBlurMode::Point, blur_pct: 50, ..DepthLook::default() };
        assert_eq!(look.lens_focus(&depth), Focus::Depth(auto_focus(&depth)));
        look.set_focus(0.5, 0.25);
        assert_eq!(look.focus_point(), Some((0.5, 0.25)));
        assert_eq!(look.lens_focus(&depth), Focus::Depth(depth.at(0.5, 0.25)));
        look.blur = LensBlurMode::TiltShift;
        assert_eq!(look.lens_focus(&depth), Focus::Band { center: 0.25, half_height: BAND_HALF_HEIGHT });
    }

    #[test]
    fn an_idle_look_leaves_the_photo_alone() {
        let src = checker(20, 20);
        let depth = ramp(4, 4);
        assert!(DepthLook::default().apply(&src, &depth).is_none());
        let no_amount = DepthLook { blur: LensBlurMode::Point, ..DepthLook::default() };
        assert!(no_amount.apply(&src, &depth).is_none());
        let fog = DepthLook { fog_pct: 100, ..DepthLook::default() };
        assert_eq!(fog.apply(&src, &depth), Some(depth_fog(&src, &depth, 1.0, FOG_COLOR)));
        let aerial = DepthLook { aerial_pct: 100, ..DepthLook::default() };
        assert_eq!(aerial.apply(&src, &depth), Some(aerial_perspective(&src, &depth, 1.0)));
    }

    #[test]
    fn aerial_perspective_cools_and_pales_the_far_and_warms_the_near() {
        let src = RgbaImage::from_pixel(10, 100, image::Rgba([150, 110, 80, 99]));
        assert_eq!(aerial_perspective(&src, &ramp(10, 100), 0.0), src);
        let out = aerial_perspective(&src, &ramp(10, 100), 1.0);
        let lab = |p: image::Rgba<u8>| crate::color::srgb_to_lab(p.0[0], p.0[1], p.0[2]);
        let (_, a0, b0) = lab(*src.get_pixel(0, 0));
        let (_, af, bf) = lab(*out.get_pixel(5, 0));
        let (_, an, bn) = lab(*out.get_pixel(5, 99));
        assert!(bf < b0 && af.hypot(bf) < a0.hypot(b0), "far is cooler and paler: {af} {bf}");
        assert!(bn > b0 && an >= a0, "near is warmer: {an} {bn}");
        assert_eq!(out.get_pixel(5, 0).0[3], 99);
        let grey = RgbaImage::from_pixel(10, 100, image::Rgba([128, 128, 128, 255]));
        let (_, _, b) = lab(*aerial_perspective(&grey, &ramp(10, 100), 1.0).get_pixel(5, 0));
        assert!(b < 0.0, "a far grey turns blue: {b}");
    }

    #[test]
    fn aerial_perspective_keeps_lightness_order_at_one_depth() {
        let src = RgbaImage::from_fn(256, 4, |x, _| image::Rgba([x as u8, x as u8, x as u8, 255]));
        let out = aerial_perspective(&src, &DepthMap::from_raw(2, 2, vec![0.0, 0.0, 0.0, 1.0]), 1.0);
        let l = |x: u32| crate::color::srgb_lightness(out.get_pixel(x, 0).0[0], out.get_pixel(x, 0).0[1], out.get_pixel(x, 0).0[2]);
        assert!((1..256).all(|x| l(x) >= l(x - 1) - 0.5));
    }

    #[test]
    fn the_depth_png_is_16_bit_at_the_asked_size() {
        let bytes = encode_depth_png(&ramp(4, 50), 8, 20).expect("encode");
        let img = image::load_from_memory(&bytes).expect("decode");
        assert_eq!((img.width(), img.height()), (8, 20));
        assert_eq!(img.color(), image::ColorType::L16);
    }
}
