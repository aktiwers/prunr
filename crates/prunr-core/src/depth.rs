//! Depth: Depth Anything V2 predicts how near each part of a photo is.
//! The map is small (the model's resolution) and kept per photo; the
//! effects sample it at the photo's own size.

use image::{imageops, GrayImage, ImageBuffer, Luma, RgbaImage};
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

    /// Calls `f(row, nearness per column)` for each row of a `w × h`
    /// photo, sampling the map bilinearly. Rows run in parallel; the
    /// row buffer is per call, so nothing photo-sized is allocated.
    fn for_each_row(&self, w: u32, h: u32, f: impl Fn(usize, &[f32]) + Sync) {
        let (xs, ys) = (bilinear_taps(self.width, w), bilinear_taps(self.height, h));
        let stride = self.width as usize;
        ys.par_iter().enumerate().for_each(|(y, ty)| {
            let row: Vec<f32> = xs.iter().map(|tx| self.sample(stride, tx, ty)).collect();
            f(y, &row);
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
        let rows: Vec<(usize, Vec<u16>)> = {
            let collected = std::sync::Mutex::new(Vec::with_capacity(h as usize));
            self.for_each_row(w, h, |y, near| {
                let row = near.iter().map(|v| (v * 65535.0).round() as u16).collect();
                collected.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((y, row));
            });
            collected.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        for (y, row) in rows {
            out[y * w as usize..(y + 1) * w as usize].copy_from_slice(&row);
        }
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
    {
        let rows = std::sync::Mutex::new(&mut pos);
        depth.for_each_row(w, h, |y, near| {
            let row: Vec<f32> = near.iter().map(|&d| {
                let off = match focus {
                    Focus::Depth(f) => (d - f).abs(),
                    Focus::Band { center, half_height } => ((y as f32 + 0.5) / h as f32 - center).abs() - half_height,
                };
                smoothstep((off / FULL_BLUR_AT).clamp(0.0, 1.0)) * LEVELS as f32
            }).collect();
            let mut guard = rows.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            guard[y * w as usize..(y + 1) * w as usize].copy_from_slice(&row);
        });
    }
    let mut out = src.clone();
    let mut prev = src.clone();
    for k in 1..=LEVELS {
        let cur = imageops::fast_blur(src, sigma_max * k as f32 / LEVELS as f32);
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
    let out = std::sync::Mutex::new(vec![0u8; src.as_raw().len()]);
    depth.for_each_row(w, h, |y, near| {
        let src_row = &src.as_raw()[y * row_len..(y + 1) * row_len];
        let row: Vec<u8> = src_row.chunks_exact(4).zip(near).flat_map(|(px, &d)| {
            let f = amount * smoothstep(1.0 - d);
            let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round().clamp(0.0, 255.0) as u8;
            [mix(px[0], color[0]), mix(px[1], color[1]), mix(px[2], color[2]), px[3]]
        }).collect();
        out.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[y * row_len..(y + 1) * row_len].copy_from_slice(&row);
    });
    let out = out.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    // sized from src
    RgbaImage::from_raw(w, h, out).expect("buffer sized from src")
}

/// A selection of everything within `range` of nearness `around`, for a
/// `w × h` photo, pushed toward `mode`.
///
/// Peak working set: the w·h cells (1 byte each).
pub fn select_by_depth(depth: &DepthMap, w: u32, h: u32, around: f32, range: f32, mode: BrushMode) -> MaskArtifact {
    let row_len = w as usize;
    let cells = std::sync::Mutex::new(vec![0i8; (w * h) as usize]);
    let value = mode.sign() * FULL;
    depth.for_each_row(w, h, |y, near| {
        let row: Vec<i8> = near.iter().map(|&d| if (d - around).abs() <= range { value } else { 0 }).collect();
        cells.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[y * row_len..(y + 1) * row_len].copy_from_slice(&row);
    });
    MaskArtifact::from_cells(w, h, cells.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner))
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
    let rows = std::sync::Mutex::new(plane.as_mut());
    depth.for_each_row(w, h, |y, near| {
        let mut guard = rows.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let row = &mut guard[y * w as usize..(y + 1) * w as usize];
        for (v, &d) in row.iter_mut().zip(near) {
            *v = (*v as f32 * (1.0 - fade * (1.0 - d))).round() as u8;
        }
    });
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
}
