//! Face restoration. YuNet finds each face and its five landmarks; the
//! face is aligned to the FFHQ 512 template and a restorer (GFPGAN v1.4
//! or RestoreFormer++) redraws it. Finding and redrawing run once
//! (`find_and_restore`); `compose_faces` blends what they kept back into
//! the photo at a strength, cheaply enough to follow a slider.
//!
//! The blend is where this differs from a plain fade between the photo
//! and the restored face, which doubles every edge halfway:
//! - the restored face is colour-matched to the photo's face first, so
//!   skin tone and lighting stay the photo's;
//! - strength works by frequency: up to half it brings in the restored
//!   fine detail over the photo's own shading and shape, above half it
//!   also takes the restored shading;
//! - a face larger than the restorer's 512 px takes its detail only as
//!   far as that resolution reaches;
//! - it is pasted through a soft face-shaped mask from the template, not
//!   a feathered square that smears into hair and background.

use image::{Rgb, RgbImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::Session, value::Tensor};

use crate::progress::{Progress, Step, Unit};
use crate::types::CoreError;

/// The restorers' input and output side.
pub const FACE_SIZE: u32 = 512;

/// The five landmarks (eyes, nose tip, mouth corners, image-left first)
/// of the FFHQ-aligned 512 face both restorers were trained on.
const TEMPLATE: [[f32; 2]; 5] = [
    [192.981_38, 239.947_08],
    [318.902_77, 240.193_6],
    [256.634_16, 314.019_35],
    [201.261_17, 371.410_43],
    [313.089_05, 371.151_18],
];

/// The detector sees the photo at most this large: a face 20 px wide in
/// a 20 MP group photo is still about 10 px at this size, YuNet's floor.
const DETECT_MAX_SIDE: u32 = 1280;
const SCORE_MIN: f32 = 0.7;
const NMS_IOU: f32 = 0.3;
/// Eyes closer than this, in photo pixels, are too small to redraw.
const MIN_EYE_DIST_PX: f32 = 8.0;

/// Where the photo's low frequencies end, in crop pixels: shading and
/// face shape below, skin, hair and eye detail above.
const DETAIL_SIGMA: f32 = 6.0;

/// One face the detector found, in photo pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub score: f32,
    /// x0, y0, x1, y1.
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
}

/// A face found and redrawn, prepared for blending at any strength: the
/// aligned crop of the photo and the restorer's version colour-matched to
/// it, each split into shading (`*_low`) and detail. About 3 MB per face;
/// what a strength change does is then a per-pixel mix.
#[derive(Clone)]
pub struct RestoredFace {
    to_crop: Similarity,
    original: RgbImage,
    restored: RgbImage,
    original_low: RgbImage,
    restored_low: RgbImage,
}

impl RestoredFace {
    fn new(to_crop: Similarity, original: RgbImage, restored: RgbImage) -> Self {
        let mask = face_mask(false);
        let planes = |img: &RgbImage| -> [Vec<f32>; 3] { std::array::from_fn(|c| img.pixels().map(|p| f32::from(p[c])).collect()) };
        let original_planes = planes(&original);
        let restored_planes = colour_matched(planes(&restored), &original_planes, mask);
        let low = |p: &[Vec<f32>; 3]| -> [Vec<f32>; 3] { std::array::from_fn(|c| gaussian(&p[c], FACE_SIZE, DETAIL_SIGMA)) };
        let image = |p: &[Vec<f32>; 3]| RgbImage::from_fn(FACE_SIZE, FACE_SIZE, |x, y| {
            let i = (y * FACE_SIZE + x) as usize;
            Rgb(std::array::from_fn(|c| p[c][i].round().clamp(0.0, 255.0) as u8))
        });
        Self {
            to_crop,
            original_low: image(&low(&original_planes)),
            restored_low: image(&low(&restored_planes)),
            restored: image(&restored_planes),
            original,
        }
    }
}

/// Find every face in `image` and redraw it with `id`'s restorer, loading
/// the models for this one photo.
///
/// Peak working set: the detector's input (≤ 1280² × 3 f32, 20 MB), the
/// restorer session (~1.2 GB for GFPGAN while it runs), and two 512²
/// RGB crops per face (1.5 MB each face).
pub fn find_and_restore(image: &RgbaImage, id: prunr_models::ModelId, progress: &Progress) -> Result<Vec<RestoredFace>, CoreError> {
    FaceRestorer::new(id)?.find_and_restore(image, progress)
}

/// A face model loaded once for several photos (a command-line batch).
/// The restorer (~1.2 GB for GFPGAN) loads on the first face and stays
/// until this is dropped.
pub struct FaceRestorer {
    id: prunr_models::ModelId,
    detector: Session,
    restorer_path: std::path::PathBuf,
    restorer: Option<Session>,
}

impl FaceRestorer {
    pub fn new(id: prunr_models::ModelId) -> Result<Self, CoreError> {
        let parts = prunr_models::multi_part_paths(id)
            .ok_or_else(|| CoreError::Inference(prunr_models::not_installed_error(id)))?;
        let part = |key: &str| parts.iter().find(|(k, _)| *k == key).map(|(_, p)| p.clone())
            .ok_or_else(|| CoreError::Inference(format!("face model {id:?}: no {key} part")));
        Ok(Self { id, detector: load(&part("detector")?)?, restorer_path: part("restorer")?, restorer: None })
    }

    pub fn find_and_restore(&mut self, image: &RgbaImage, progress: &Progress) -> Result<Vec<RestoredFace>, CoreError> {
        progress.step(Step::FindingFaces);
        let faces = detect(&mut self.detector, image)?;
        if progress.is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        if self.restorer.is_none() {
            progress.step(Step::LoadingModel);
            self.restorer = Some(load(&self.restorer_path)?);
        }
        let restorer = self.restorer.as_mut().ok_or_else(|| CoreError::Inference(format!("face model {:?}: restorer not loaded", self.id)))?;
        progress.step(Step::RestoringFaces);
        let total = faces.len() as u32;
        let mut out = Vec::with_capacity(faces.len());
        for (i, face) in faces.iter().enumerate() {
            if progress.is_cancelled() {
                return Err(CoreError::Cancelled);
            }
            progress.outer(i as u32, total, Unit::Face);
            let to_crop = Similarity::fit(&face.landmarks, &TEMPLATE);
            let original = warp_to_crop(image, &to_crop);
            let restored = restore(restorer, &original)?;
            out.push(RestoredFace::new(to_crop, original, restored));
        }
        progress.outer(total, total, Unit::Face);
        Ok(out)
    }
}

fn load(path: &std::path::Path) -> Result<Session, CoreError> {
    crate::ort_runtime::session_builder()?
        .commit_from_file(path)
        .map_err(|e| CoreError::Inference(format!("face model {}: {e}", path.display())))
}

/// YuNet over the photo, scaled to `DETECT_MAX_SIDE` and padded to a
/// multiple of 32. Returns faces in photo pixels, best first.
fn detect(session: &mut Session, image: &RgbaImage) -> Result<Vec<Face>, CoreError> {
    let (w, h) = image.dimensions();
    let scale = (DETECT_MAX_SIDE as f32 / w.max(h) as f32).min(1.0);
    let (sw, sh) = (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
    let small = crate::formats::resize_rgba(image, sw, sh, crate::formats::ResizeFilter::Bilinear);
    let (pw, ph) = (sw.div_ceil(32) * 32, sh.div_ceil(32) * 32);
    // YuNet reads BGR at 0-255.
    let mut input = Array4::<f32>::zeros((1, 3, ph as usize, pw as usize));
    for (x, y, p) in small.enumerate_pixels() {
        for (c, v) in [p[2], p[1], p[0]].into_iter().enumerate() {
            input[[0, c, y as usize, x as usize]] = f32::from(v);
        }
    }
    let tensor = Tensor::from_array(input).map_err(|e| CoreError::Inference(format!("face detector: input: {e}")))?;
    let outputs = session.run(inputs!["input" => &tensor])
        .map_err(|e| CoreError::Inference(format!("face detector: {e}")))?;
    let get = |name: String| -> Result<Vec<f32>, CoreError> {
        let (_, data) = outputs[name.as_str()].try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(format!("face detector: {name}: {e}")))?;
        Ok(data.to_vec())
    };
    let mut faces = Vec::new();
    for stride in [8u32, 16, 32] {
        let level = Level {
            stride,
            cols: pw / stride,
            cls: get(format!("cls_{stride}"))?,
            obj: get(format!("obj_{stride}"))?,
            bbox: get(format!("bbox_{stride}"))?,
            kps: get(format!("kps_{stride}"))?,
        };
        faces.extend(level.decode(1.0 / scale));
    }
    Ok(keep_distinct(faces))
}

/// One YuNet output level: per-anchor scores, box and landmark offsets
/// in units of `stride` from the anchor's grid cell.
struct Level {
    stride: u32,
    cols: u32,
    cls: Vec<f32>,
    obj: Vec<f32>,
    bbox: Vec<f32>,
    kps: Vec<f32>,
}

impl Level {
    /// The faces above `SCORE_MIN`, mapped to photo pixels by `to_photo`.
    fn decode(&self, to_photo: f32) -> Vec<Face> {
        let s = self.stride as f32;
        (0..self.cls.len())
            .filter_map(|i| {
                let score = (self.cls[i].clamp(0.0, 1.0) * self.obj[i].clamp(0.0, 1.0)).sqrt();
                if score < SCORE_MIN {
                    return None;
                }
                let (r, c) = ((i as u32 / self.cols) as f32, (i as u32 % self.cols) as f32);
                let b = &self.bbox[i * 4..i * 4 + 4];
                let (cx, cy) = ((c + b[0]) * s, (r + b[1]) * s);
                let (bw, bh) = (b[2].exp() * s, b[3].exp() * s);
                let k = &self.kps[i * 10..i * 10 + 10];
                let landmarks = std::array::from_fn(|j| [(k[2 * j] + c) * s * to_photo, (k[2 * j + 1] + r) * s * to_photo]);
                let bbox = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0].map(|v| v * to_photo);
                Some(Face { score, bbox, landmarks })
            })
            .collect()
    }
}

/// Best first, dropping any face that overlaps a better one and any too
/// small to redraw.
fn keep_distinct(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        let [l, r, ..] = f.landmarks;
        let eye_dist = ((l[0] - r[0]).powi(2) + (l[1] - r[1]).powi(2)).sqrt();
        if eye_dist >= MIN_EYE_DIST_PX && kept.iter().all(|k| iou(&k.bbox, &f.bbox) < NMS_IOU) {
            kept.push(f);
        }
    }
    kept
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Rotation, uniform scale and shift: `q = [[a, -b], [b, a]]·p + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Similarity {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
}

impl Similarity {
    /// The least-squares similarity taking `from` onto `to`.
    fn fit(from: &[[f32; 2]; 5], to: &[[f32; 2]; 5]) -> Self {
        let mean = |pts: &[[f32; 2]; 5]| pts.iter().fold([0.0, 0.0], |m, p| [m[0] + p[0] / 5.0, m[1] + p[1] / 5.0]);
        let (mp, mq) = (mean(from), mean(to));
        let (mut num_a, mut num_b, mut den) = (0.0, 0.0, 0.0);
        for (p, q) in from.iter().zip(to) {
            let (px, py, qx, qy) = (p[0] - mp[0], p[1] - mp[1], q[0] - mq[0], q[1] - mq[1]);
            num_a += px * qx + py * qy;
            num_b += px * qy - py * qx;
            den += px * px + py * py;
        }
        let den = den.max(f32::EPSILON);
        let (a, b) = (num_a / den, num_b / den);
        Self { a, b, tx: mq[0] - (a * mp[0] - b * mp[1]), ty: mq[1] - (b * mp[0] + a * mp[1]) }
    }

    fn apply(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        [self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty]
    }

    /// Photo pixels to crop pixels.
    fn scale(&self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    fn inverse(&self) -> Self {
        let s2 = (self.a * self.a + self.b * self.b).max(f32::EPSILON);
        let (a, b) = (self.a / s2, -self.b / s2);
        Self { a, b, tx: -(a * self.tx - b * self.ty), ty: -(b * self.tx + a * self.ty) }
    }
}

/// The 512 crop the restorers expect: each crop pixel sampled from the
/// photo through the inverse alignment, edges clamped.
fn warp_to_crop(image: &RgbaImage, to_crop: &Similarity) -> RgbImage {
    let to_photo = to_crop.inverse();
    RgbImage::from_fn(FACE_SIZE, FACE_SIZE, |u, v| {
        let [x, y] = to_photo.apply([u as f32, v as f32]);
        let p = sample_rgba(image, x, y);
        Rgb([p[0], p[1], p[2]].map(|c| c.round().clamp(0.0, 255.0) as u8))
    })
}

/// Bilinear sample of the RGB channels, edges clamped.
fn sample_rgba(image: &RgbaImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let (x, y) = (x.clamp(0.0, w - 1.0), y.clamp(0.0, h - 1.0));
    let (x0, y0) = (x.floor(), y.floor());
    let (x1, y1) = ((x0 + 1.0).min(w - 1.0), (y0 + 1.0).min(h - 1.0));
    let (fx, fy) = (x - x0, y - y0);
    let px = |x: f32, y: f32| image.get_pixel(x as u32, y as u32);
    let (p00, p10, p01, p11) = (px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1));
    std::array::from_fn(|c| {
        let top = f32::from(p00[c]) * (1.0 - fx) + f32::from(p10[c]) * fx;
        let bottom = f32::from(p01[c]) * (1.0 - fx) + f32::from(p11[c]) * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// Run the restorer on one aligned crop: RGB in [-1, 1] in and out.
fn restore(session: &mut Session, crop: &RgbImage) -> Result<RgbImage, CoreError> {
    let n = FACE_SIZE as usize;
    let mut input = Array4::<f32>::zeros((1, 3, n, n));
    for (x, y, p) in crop.enumerate_pixels() {
        for c in 0..3 {
            input[[0, c, y as usize, x as usize]] = f32::from(p[c]) / 127.5 - 1.0;
        }
    }
    let tensor = Tensor::from_array(input).map_err(|e| CoreError::Inference(format!("face restorer: input: {e}")))?;
    let outputs = session.run(inputs!["input" => &tensor])
        .map_err(|e| CoreError::Inference(format!("face restorer: {e}")))?;
    let (_, data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| CoreError::Inference(format!("face restorer: output: {e}")))?;
    let plane = n * n;
    Ok(RgbImage::from_fn(FACE_SIZE, FACE_SIZE, |x, y| {
        let i = y as usize * n + x as usize;
        Rgb(std::array::from_fn(|c| ((data[c * plane + i] + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8))
    }))
}

/// How the found faces go back into the photo; all of it is cheap to
/// change.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FaceBlend {
    /// 0 = the photo, 1 = fully restored.
    pub strength: f32,
    /// Only the largest face (a portrait with people behind).
    pub largest_only: bool,
    /// Hair and ears too, not just the face.
    pub hair: bool,
}

impl Default for FaceBlend {
    fn default() -> Self {
        Self { strength: 0.5, largest_only: false, hair: false }
    }
}

/// `image` with the faces blended back as `blend` says. Pure and
/// crop-local: tens of ms per face.
pub fn compose_faces(image: &RgbaImage, faces: &[RestoredFace], blend: FaceBlend) -> RgbaImage {
    let mut out = image.clone();
    compose_faces_into(&mut out, faces, blend);
    out
}

/// `compose_faces` into an image the caller owns, saving the copy.
pub fn compose_faces_into(out: &mut RgbaImage, faces: &[RestoredFace], blend: FaceBlend) {
    let strength = blend.strength.clamp(0.0, 1.0);
    if strength == 0.0 || faces.is_empty() {
        return;
    }
    let mask = face_mask(blend.hair);
    // The largest face in the photo is the one its alignment shrinks most.
    let largest = faces.iter().min_by(|a, b| a.to_crop.scale().total_cmp(&b.to_crop.scale()));
    let chosen: Vec<&RestoredFace> = if blend.largest_only { largest.into_iter().collect() } else { faces.iter().collect() };
    use rayon::prelude::*;
    let blended: Vec<[Vec<f32>; 3]> = chosen.par_iter().map(|face| blend_face(face, strength)).collect();
    for (face, blended) in chosen.iter().zip(&blended) {
        paste(out, face, blended, mask);
    }
}

/// The crop's face area, soft-edged, as 0-1 weights over the 512 crop:
/// an ellipse from brow to chin on the FFHQ template, or with `hair` one
/// that takes in the hair and ears. Built once per kind.
fn face_mask(hair: bool) -> &'static [f32] {
    static FACE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    static WITH_HAIR: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    if hair { WITH_HAIR.get_or_init(|| build_face_mask(true)) } else { FACE.get_or_init(|| build_face_mask(false)) }
}

fn build_face_mask(hair: bool) -> Vec<f32> {
    let n = FACE_SIZE as usize;
    let (cx, cy, rx, ry) = if hair { (256.0f32, 250.0f32, 225.0f32, 250.0f32) } else { (256.0, 280.0, 170.0, 215.0) };
    let hard: Vec<f32> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            let d = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2);
            if d <= 1.0 { 1.0 } else { 0.0 }
        })
        .collect();
    gaussian(&hard, FACE_SIZE, 14.0)
}

/// Gaussian blur approximated by three box passes.
fn gaussian(plane: &[f32], side: u32, sigma: f32) -> Vec<f32> {
    // Three box passes of radius r have a variance of r² + r ≈ sigma².
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as u32;
    let mut out = plane.to_vec();
    for _ in 0..3 {
        out = crate::guided_filter::box_filter(&out, side, side, r);
    }
    out
}

/// The face at `strength`, in crop space, as RGB planes.
fn blend_face(face: &RestoredFace, strength: f32) -> [Vec<f32>; 3] {
    let detail_take = (2.0 * strength).min(1.0) * detail_reach(face);
    let shading_take = (2.0 * strength - 1.0).max(0.0);
    let (o, r, o_low, r_low) = (face.original.as_raw(), face.restored.as_raw(), face.original_low.as_raw(), face.restored_low.as_raw());
    std::array::from_fn(|c| {
        (c..o.len())
            .step_by(3)
            .map(|i| {
                let (o, r, o_low, r_low) = (f32::from(o[i]), f32::from(r[i]), f32::from(o_low[i]), f32::from(r_low[i]));
                let low = o_low + (r_low - o_low) * shading_take;
                let (o_detail, r_detail) = (o - o_low, r - r_low);
                low + o_detail + (r_detail - o_detail) * detail_take
            })
            .collect()
    })
}

/// How far the restored detail reaches, 0.25-1. The restorer draws at
/// 512 px; a face larger than that in the photo has finer detail of its
/// own than the restoration, so it takes the restored detail only in
/// proportion. A face's own sharpness can't decide this: an upscaler's
/// invented edges measure as sharp as real detail.
fn detail_reach(face: &RestoredFace) -> f32 {
    face.to_crop.scale().clamp(0.25, 1.0)
}

/// `restored` shifted to the original's mean brightness and colour over
/// the face area, so the restorer's tone drift goes. Only the mean: a
/// soft photo has little spread, and matching it would flatten the very
/// detail the restorer added.
/// rationale: YCbCr is a linear mix of the sRGB values, so this stays in
/// straight sRGB; it only separates brightness from colour.
fn colour_matched(restored: [Vec<f32>; 3], original: &[Vec<f32>; 3], mask: &[f32]) -> [Vec<f32>; 3] {
    let to_ycc = |p: &[Vec<f32>; 3]| -> [Vec<f32>; 3] {
        std::array::from_fn(|c| (0..p[0].len()).map(|i| {
            let (r, g, b) = (p[0][i], p[1][i], p[2][i]);
            match c {
                0 => 0.299 * r + 0.587 * g + 0.114 * b,
                1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b,
                _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b,
            }
        }).collect())
    };
    let (r_ycc, o_ycc) = (to_ycc(&restored), to_ycc(original));
    let weight: f32 = mask.iter().sum::<f32>().max(f32::EPSILON);
    let mean = |plane: &[f32]| plane.iter().zip(mask).map(|(v, m)| v * m).sum::<f32>() / weight;
    let matched: [Vec<f32>; 3] = std::array::from_fn(|c| {
        let shift = mean(&o_ycc[c]) - mean(&r_ycc[c]);
        r_ycc[c].iter().map(|v| v + shift).collect()
    });
    std::array::from_fn(|c| (0..matched[0].len()).map(|i| {
        let (y, cb, cr) = (matched[0][i], matched[1][i], matched[2][i]);
        match c {
            0 => y + 1.402 * cr,
            1 => y - 0.344_136 * cb - 0.714_136 * cr,
            _ => y + 1.772 * cb,
        }
    }).collect())
}

/// Paste the blended crop into `out` through the face mask, over the
/// photo pixels the crop covers. Alpha is left as it was.
fn paste(out: &mut RgbaImage, face: &RestoredFace, blended: &[Vec<f32>; 3], mask: &[f32]) {
    let to_photo = face.to_crop.inverse();
    let corners = [[0.0, 0.0], [FACE_SIZE as f32, 0.0], [0.0, FACE_SIZE as f32], [FACE_SIZE as f32, FACE_SIZE as f32]].map(|p| to_photo.apply(p));
    let (w, h) = out.dimensions();
    let lo = |i: usize, max: u32| (corners.iter().map(|p| p[i]).fold(f32::MAX, f32::min).floor().max(0.0) as u32).min(max);
    let hi = |i: usize, max: u32| (corners.iter().map(|p| p[i]).fold(f32::MIN, f32::max).ceil().max(0.0) as u32).min(max);
    let n = FACE_SIZE as usize;
    let sample = |plane: &[f32], u: f32, v: f32| -> f32 {
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let at = |x: f32, y: f32| plane[(y as usize).min(n - 1) * n + (x as usize).min(n - 1)];
        let top = at(u0, v0) * (1.0 - fu) + at(u0 + 1.0, v0) * fu;
        let bottom = at(u0, v0 + 1.0) * (1.0 - fu) + at(u0 + 1.0, v0 + 1.0) * fu;
        top * (1.0 - fv) + bottom * fv
    };
    for y in lo(1, h)..hi(1, h) {
        for x in lo(0, w)..hi(0, w) {
            let [u, v] = face.to_crop.apply([x as f32, y as f32]);
            if !(0.0..(FACE_SIZE - 1) as f32).contains(&u) || !(0.0..(FACE_SIZE - 1) as f32).contains(&v) {
                continue;
            }
            let m = sample(mask, u, v);
            if m <= 0.0 {
                continue;
            }
            let p = out.get_pixel_mut(x, y);
            for c in 0..3 {
                let value = f32::from(p[c]) * (1.0 - m) + sample(&blended[c], u, v) * m;
                p[c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn a_similarity_fit_recovers_a_known_one() {
        let truth = Similarity { a: 0.8, b: 0.3, tx: 12.0, ty: -7.0 };
        let from = TEMPLATE.map(|p| [p[0] * 0.5 + 40.0, p[1] * 0.7 + 10.0]);
        let to = from.map(|p| truth.apply(p));
        let fit = Similarity::fit(&from, &to);
        for (got, want) in [fit.a, fit.b, fit.tx, fit.ty].into_iter().zip([truth.a, truth.b, truth.tx, truth.ty]) {
            assert!((got - want).abs() < 1e-3, "{fit:?}");
        }
        let back = fit.inverse().apply(fit.apply([100.0, 50.0]));
        assert!((back[0] - 100.0).abs() < 1e-3 && (back[1] - 50.0).abs() < 1e-3);
    }

    /// Two detections of one face keep the better; a far one stays; one
    /// with eyes a few pixels apart is dropped.
    #[test]
    fn overlapping_and_tiny_faces_are_dropped() {
        let face = |score, x: f32, eye: f32| Face {
            score,
            bbox: [x, 0.0, x + 100.0, 100.0],
            landmarks: [[x + 30.0, 40.0], [x + 30.0 + eye, 40.0], [x + 50.0, 60.0], [x + 35.0, 80.0], [x + 65.0, 80.0]],
        };
        let kept = keep_distinct(vec![face(0.8, 0.0, 40.0), face(0.9, 5.0, 40.0), face(0.75, 300.0, 40.0), face(0.95, 600.0, 3.0)]);
        assert_eq!(kept.iter().map(|f| f.score).collect::<Vec<_>>(), [0.9, 0.75]);
    }

    /// YuNet's offsets are in units of the stride from the anchor cell.
    #[test]
    fn a_level_decodes_boxes_and_landmarks_in_photo_pixels() {
        let cols = 4;
        let n = 8;
        let mut level = Level { stride: 8, cols, cls: vec![0.0; n], obj: vec![0.0; n], bbox: vec![0.0; n * 4], kps: vec![0.0; n * 10] };
        let i = 5; // row 1, column 1
        level.cls[i] = 0.81;
        level.obj[i] = 1.0;
        level.bbox[i * 4..i * 4 + 4].copy_from_slice(&[0.5, 0.5, 0.0, 0.0]);
        level.kps[i * 10..i * 10 + 2].copy_from_slice(&[0.25, 0.75]);
        let faces = level.decode(2.0);
        assert_eq!(faces.len(), 1);
        let f = faces[0];
        assert!((f.score - 0.9).abs() < 1e-5);
        assert_eq!(f.bbox, [16.0, 16.0, 32.0, 32.0], "centre (1.5·8, 1.5·8), side 8, doubled");
        assert_eq!(f.landmarks[0], [20.0, 28.0]);
    }

    fn face_of(original: RgbImage, restored: RgbImage) -> RestoredFace {
        // Crop pixel (u, v) is photo pixel (u, v) + (10, 20).
        RestoredFace::new(Similarity { a: 1.0, b: 0.0, tx: -10.0, ty: -20.0 }, original, restored)
    }

    /// "Largest only" keeps the face the alignment shrinks most.
    #[test]
    fn largest_only_restores_the_biggest_face() {
        let image = RgbaImage::from_pixel(1400, 700, Rgba([120, 100, 90, 255]));
        let flat = RgbImage::from_pixel(512, 512, Rgb([120, 100, 90]));
        let stripes = RgbImage::from_fn(512, 512, |x, _| if x % 2 == 0 { Rgb([160, 140, 130]) } else { Rgb([80, 60, 50]) });
        let small = RestoredFace::new(Similarity { a: 2.0, b: 0.0, tx: -2000.0, ty: 0.0 }, flat.clone(), stripes.clone());
        let big = RestoredFace::new(Similarity { a: 1.0, b: 0.0, tx: 0.0, ty: 0.0 }, flat, stripes);
        let out = compose_faces(&image, &[small, big], FaceBlend { strength: 1.0, largest_only: true, hair: false });
        let changed = |x: u32, y: u32| out.get_pixel(x, y) != image.get_pixel(x, y);
        assert!(changed(256, 280), "the big face");
        assert!(!changed(1000 + 128, 140), "not the small one");
    }

    /// A face twice the restorer's size in the photo takes half its
    /// detail; one smaller takes it all.
    #[test]
    fn a_large_face_takes_the_restored_detail_in_proportion() {
        let flat = RgbImage::from_pixel(512, 512, Rgb([100, 100, 100]));
        let at = |a: f32| detail_reach(&RestoredFace::new(Similarity { a, b: 0.0, tx: 0.0, ty: 0.0 }, flat.clone(), flat.clone()));
        assert_eq!(at(0.5), 0.5);
        assert_eq!(at(1.6), 1.0);
        assert_eq!(at(0.1), 0.25, "never below a quarter");
    }

    #[test]
    fn strength_zero_leaves_the_photo_as_it_was() {
        let image = RgbaImage::from_pixel(600, 600, Rgba([10, 20, 30, 200]));
        let face = face_of(RgbImage::from_pixel(512, 512, Rgb([10, 20, 30])), RgbImage::from_pixel(512, 512, Rgb([200, 0, 0])));
        assert_eq!(compose_faces(&image, &[face], FaceBlend { strength: 0.0, ..FaceBlend::default() }), image);
    }

    /// At full strength the face takes the restored detail, at the
    /// photo's own colour; outside the face nothing changes; alpha stays.
    #[test]
    fn full_strength_brings_the_detail_in_the_photos_colour() {
        let image = RgbaImage::from_pixel(600, 600, Rgba([120, 100, 90, 200]));
        let original = RgbImage::from_pixel(512, 512, Rgb([120, 100, 90]));
        // A restored face greyer than the photo, with fine stripes.
        let restored = RgbImage::from_fn(512, 512, |x, _| if x % 2 == 0 { Rgb([150, 150, 150]) } else { Rgb([110, 110, 110]) });
        let out = compose_faces(&image, &[face_of(original, restored)], FaceBlend { strength: 1.0, ..FaceBlend::default() });
        let (a, b) = (out.get_pixel(10 + 256, 20 + 280), out.get_pixel(10 + 257, 20 + 280));
        assert!(a[0].abs_diff(b[0]) > 10, "the stripes came in: {a:?} {b:?}");
        let mean_r = (u32::from(a[0]) + u32::from(b[0])) / 2;
        assert!(mean_r.abs_diff(120) <= 6, "the photo's colour, not the restorer's grey: {a:?} {b:?}");
        assert_eq!(out.get_pixel(5, 5), image.get_pixel(5, 5), "outside the face");
        assert_eq!(a[3], 200, "alpha stays");
    }
}
