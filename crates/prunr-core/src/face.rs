//! Face restoration. YuNet finds each face and its five landmarks; the
//! face is aligned to the FFHQ 512 template and a restorer (GFPGAN v1.4
//! or RestoreFormer++) redraws it. Finding and redrawing run once
//! (`find_and_restore`); `compose_faces` blends what they kept back into
//! the photo at a strength, cheaply enough to follow a slider. While they
//! run, the faces are reported as soon as they are found
//! (`ProgressUpdate::Faces`) and each redrawn one as a preview of how it
//! blends in, so a listener can show them one at a time.
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

use crate::progress::{Progress, ProgressUpdate, Step, Unit};
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

/// A mask weight below this moves no pixel by half a level, so the blend
/// leaves the pixel as it was.
const HALF_LEVEL: f32 = 0.5 / 255.0;

/// One face the detector found, in photo pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub score: f32,
    /// x0, y0, x1, y1.
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
}

/// Where a found face sits in its photo: its alignment onto the
/// restorers' crop, its box, and the photo's size. Enough to outline the
/// face, tell the largest and focus on it before it is redrawn.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FacePlace {
    to_crop: Similarity,
    /// The detector's box, x0, y0, x1, y1 in photo pixels.
    bbox: [f32; 4],
    /// The photo it was found in, width and height.
    photo_size: [u32; 2],
}

impl FacePlace {
    pub fn of(face: &Face, photo_size: [u32; 2]) -> Self {
        Self { to_crop: Similarity::fit(&face.landmarks, &TEMPLATE), bbox: face.bbox, photo_size }
    }

    /// The face's box as fractions of the photo (x0, y0, x1, y1), within it.
    pub fn bounds(&self) -> [f32; 4] {
        let [w, h] = self.photo_size.map(|v| v.max(1) as f32);
        let [x0, y0, x1, y1] = self.bbox;
        [x0 / w, y0 / h, x1 / w, y1 / h].map(|v| v.clamp(0.0, 1.0))
    }

    /// The area its redraw covers in the photo: the crop's face ellipse
    /// (with `hair`, the larger one) taken back through the alignment.
    pub fn area(&self, hair: bool) -> FaceArea {
        let (cx, cy, rx, ry) = face_ellipse(hair);
        let back = self.to_crop.inverse();
        let scale = back.scale();
        FaceArea { center: back.apply([cx, cy]), radii: [rx * scale, ry * scale], angle: back.b.atan2(back.a), photo_size: self.photo_size }
    }

    /// The photo pixels a blend with `hair` can change, as (x, y, w, h):
    /// where the soft face mask reaches, clipped to the photo.
    fn reach(&self, hair: bool) -> Option<(u32, u32, u32, u32)> {
        let [x0, y0, x1, y1] = photo_span(&self.to_crop.inverse(), face_mask(hair).reach, self.photo_size, 1.0);
        (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
    }
}

/// A face found and redrawn, prepared for blending at any strength: the
/// aligned crop of the photo and the restorer's version colour-matched to
/// it, each split into shading (`*_low`) and detail. About 3 MB per face;
/// what a strength change does is then a per-pixel mix.
#[derive(Clone)]
pub struct RestoredFace {
    place: FacePlace,
    original: RgbImage,
    restored: RgbImage,
    original_low: RgbImage,
    restored_low: RgbImage,
}

impl RestoredFace {
    fn new(place: FacePlace, original: RgbImage, restored: RgbImage) -> Self {
        let mask = &face_mask(false).weights;
        let original_planes = crop_planes(&original);
        let restored_planes = colour_matched(crop_planes(&restored), &original_planes, mask);
        Self {
            place,
            original_low: crop_image(&shading(&original_planes)),
            restored_low: crop_image(&shading(&restored_planes)),
            restored: crop_image(&restored_planes),
            original,
        }
    }
}

impl RestoredFace {
    pub fn place(&self) -> &FacePlace {
        &self.place
    }

    /// A face at `place` that the restorer left as it was.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn unchanged_for_test(place: FacePlace) -> Self {
        let grey = RgbImage::from_pixel(FACE_SIZE, FACE_SIZE, Rgb([128, 128, 128]));
        Self::new(place, grey.clone(), grey)
    }
}

/// An ellipse in photo pixels, turned by `angle` radians, in a photo of
/// `photo_size`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceArea {
    pub center: [f32; 2],
    pub radii: [f32; 2],
    pub angle: f32,
    pub photo_size: [u32; 2],
}

/// The face "Largest" redraws: the one its alignment shrinks most.
pub fn largest_face<'a>(faces: impl IntoIterator<Item = &'a FacePlace>) -> Option<usize> {
    faces.into_iter().enumerate().min_by(|(_, a), (_, b)| a.to_crop.scale().total_cmp(&b.to_crop.scale())).map(|(i, _)| i)
}

/// Find every face in `image` and redraw it with `id`'s restorer. The
/// models stay loaded for the next search and free after
/// `crate::IDLE_RELEASE` unused, or when another face model is asked for.
/// `blend` is how previews show each face blended in; what is returned
/// is the same whatever it is.
///
/// Peak working set: the detector's input (≤ 1280² × 3 f32, 20 MB), the
/// restorer session (~1.2 GB for GFPGAN while it runs), and two 512²
/// RGB crops per face (1.5 MB each face). With previews asked for, one
/// face blended at a time on top: its 512² RGB f32 planes (3 MB) and its
/// preview, at most the preview's own size (1536² × 4 = 9 MB for a face
/// filling the photo; a few hundred KB for a face in a group).
pub fn find_and_restore(image: &RgbaImage, id: prunr_models::ModelId, blend: FaceBlend, progress: &Progress) -> Result<Vec<RestoredFace>, CoreError> {
    static SWEEPER: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    SWEEPER.get_or_init(|| crate::inpaint_sd::spawn_idle_sweeper("face-idle-sweeper", warm_restorers, "faces: released the idle face model"));
    let restorer = crate::inpaint_sd::warm_entry(warm_restorers(), id, || FaceRestorer::new(id).map(std::sync::Mutex::new))?;
    let found = restorer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).find_and_restore(image, blend, progress);
    crate::inpaint_sd::warm_touch(warm_restorers(), &id);
    found
}

type WarmRestorers = std::collections::HashMap<prunr_models::ModelId, crate::inpaint_sd::CacheEntry<std::sync::Arc<std::sync::Mutex<FaceRestorer>>>>;

fn warm_restorers() -> &'static std::sync::Mutex<WarmRestorers> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<WarmRestorers>> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// The face detector built into the app (YuNet), loaded once for as many
/// photos as its owner looks at.
pub struct FaceFinder {
    detector: Session,
}

impl FaceFinder {
    pub fn new() -> Result<Self, CoreError> {
        Self::on(true).map(|(finder, _)| finder)
    }

    /// The detector on the first graphics device that builds it, unless
    /// `cpu_only`, with where it landed.
    pub fn on(cpu_only: bool) -> Result<(Self, crate::engine::SessionPick), CoreError> {
        let bytes = prunr_models::resolve_bytes(prunr_models::ModelId::YuNet)
            .ok_or_else(|| CoreError::Inference("the face finder is missing from this build".into()))?;
        let (detector, pick) = crate::engine::session_on_ladder(prunr_models::ModelId::YuNet, cpu_only, |mut builder| {
            builder.commit_from_memory(&bytes).map_err(|e| format!("face finder: {e}"))
        })?;
        Ok((Self { detector }, pick))
    }

    /// The faces in `image`, best first.
    pub fn find(&mut self, image: &RgbaImage) -> Result<Vec<Face>, CoreError> {
        detect(&mut self.detector, image.dimensions(), |w, h| crate::formats::resize_rgba(image, w, h, crate::formats::ResizeFilter::Bilinear))
    }

    /// `find` for a decoded photo in any layout: it is shrunk as it is,
    /// never expanded to a full-size RGBA copy first.
    pub fn find_decoded(&mut self, photo: &image::DynamicImage) -> Result<Vec<Face>, CoreError> {
        use image::GenericImageView;
        detect(&mut self.detector, photo.dimensions(), |w, h| crate::formats::resize_to_rgba(photo, w, h, crate::formats::ResizeFilter::Bilinear))
    }
}

/// A face finder built at the first photo that asks and kept for the
/// rest; one that fails to build is tried again at the next photo.
#[derive(Default)]
pub struct LazyFaceFinder(Option<FaceFinder>);

impl LazyFaceFinder {
    /// Where the faces `find` sees in a photo of `photo_size` sit; none
    /// when the finder fails, which is logged.
    pub fn places(&mut self, photo_size: [u32; 2], find: impl FnOnce(&mut FaceFinder) -> Result<Vec<Face>, CoreError>) -> Vec<FacePlace> {
        if self.0.is_none() {
            self.0 = FaceFinder::new().map_err(|err| tracing::warn!(%err, "face finder unavailable")).ok();
        }
        let Some(finder) = self.0.as_mut() else { return Vec::new() };
        match find(finder) {
            Ok(faces) => faces.iter().map(|f| FacePlace::of(f, photo_size)).collect(),
            Err(err) => {
                tracing::warn!(%err, "face finder failed");
                Vec::new()
            }
        }
    }
}

/// A face model loaded once for several photos (a command-line batch).
/// The restorer (~1.2 GB for GFPGAN) loads on the first face and stays
/// until this is dropped.
pub struct FaceRestorer {
    id: prunr_models::ModelId,
    finder: FaceFinder,
    restorer: Option<Redrawer>,
}

impl FaceRestorer {
    pub fn new(id: prunr_models::ModelId) -> Result<Self, CoreError> {
        restorer_path(id)?;
        Ok(Self { id, finder: FaceFinder::new()?, restorer: None })
    }

    /// `find_and_restore` with the models this holds.
    pub fn find_and_restore(&mut self, image: &RgbaImage, blend: FaceBlend, progress: &Progress) -> Result<Vec<RestoredFace>, CoreError> {
        progress.step(Step::FindingFaces);
        let faces = self.finder.find(image)?;
        progress.check_cancelled()?;
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        // Before the restorer loads, which takes seconds the first time.
        let places = report_found(image, &faces, progress);
        let restorer = match &mut self.restorer {
            Some(r) => r,
            None => {
                progress.step(Step::LoadingModel);
                self.restorer.insert(Redrawer::on(self.id, true)?.0)
            }
        };
        restore_each(image, &places, blend, progress, |crop| restorer.redraw(crop))
    }
}

/// Where each detected face sits in `image`, reported as soon as known.
fn report_found(image: &RgbaImage, faces: &[Face], progress: &Progress) -> Vec<FacePlace> {
    let places: Vec<FacePlace> = faces.iter().map(|f| FacePlace::of(f, [image.width(), image.height()])).collect();
    progress.report(ProgressUpdate::Faces(places.clone()));
    places
}

/// Redraw the face at each of `places` in `image` with `redraw`, in
/// order, reporting each one `blend` redraws as a preview once it is done.
fn restore_each(
    image: &RgbaImage,
    places: &[FacePlace],
    blend: FaceBlend,
    progress: &Progress,
    mut redraw: impl FnMut(&RgbImage) -> Result<RgbImage, CoreError>,
) -> Result<Vec<RestoredFace>, CoreError> {
    progress.step(Step::RestoringFaces);
    let total = places.len() as u32;
    let redrawn = blend.redrawn(places);
    let mut out = Vec::with_capacity(places.len());
    for (i, place) in places.iter().enumerate() {
        progress.check_cancelled()?;
        progress.outer(i as u32, total, Unit::Face);
        let original = warp_to_crop(image, &place.to_crop);
        let restored = redraw(&original)?;
        let face = RestoredFace::new(*place, original, restored);
        if redrawn(i) && !progress.is_cancelled() {
            report_preview(image, &face, blend, progress);
        }
        out.push(face);
    }
    progress.outer(total, total, Unit::Face);
    Ok(out)
}

/// Report `face` as `blend` draws it into `image`, as a preview to lay
/// over the image shown: the blended face, with the soft face mask (times
/// the photo's alpha) as its alpha, over the pixels the mask reaches.
/// Over the photo it shows what the blend makes there and leaves the rest
/// as it is; over a cut-out of the photo, the cut-out shows round the face.
fn report_preview(image: &RgbaImage, face: &RestoredFace, blend: FaceBlend, progress: &Progress) {
    let strength = blend.strength.clamp(0.0, 1.0);
    let Some(rect) = face.place.reach(blend.hair).filter(|_| strength > 0.0) else { return };
    progress.tile_preview(rect, image.dimensions(), |w, h| {
        let blended = blend_face(face, &Base::stored(face), strength);
        let mask = &face_mask(blend.hair).weights;
        let (x0, y0) = (rect.0 as f32, rect.1 as f32);
        let (step_x, step_y) = (rect.2 as f32 / w as f32, rect.3 as f32 / h as f32);
        let (last_x, last_y) = (image.width() - 1, image.height() - 1);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for py in 0..h {
            for px in 0..w {
                // The preview pixel's centre, in photo pixels.
                let (x, y) = (x0 + (px as f32 + 0.5) * step_x - 0.5, y0 + (py as f32 + 0.5) * step_y - 0.5);
                let [u, v] = face.place.to_crop.apply([x, y]);
                if !in_crop(u, v) {
                    rgba.extend([0; 4]);
                    continue;
                }
                let photo_alpha = f32::from(image.get_pixel((x.round().max(0.0) as u32).min(last_x), (y.round().max(0.0) as u32).min(last_y))[3]);
                let alpha = (sample_crop(mask, u, v) * photo_alpha).round().clamp(0.0, 255.0) as u8;
                let [r, g, b] = std::array::from_fn(|c| sample_crop(&blended[c], u, v).round().clamp(0.0, 255.0) as u8);
                rgba.extend([r, g, b, alpha]);
            }
        }
        rgba
    });
}

/// A face restorer's network (GFPGAN, RestoreFormer++), loaded.
pub struct Redrawer(Session);

impl Redrawer {
    /// `id`'s network on the first graphics device that builds it, unless
    /// `cpu_only`, with where it landed. No optimised-graph cache:
    /// measured on GFPGAN (CPU), loading the cached graph saved 0.2 s of a
    /// 4.2 s first search for 340 MB of disk.
    pub fn on(id: prunr_models::ModelId, cpu_only: bool) -> Result<(Self, crate::engine::SessionPick), CoreError> {
        let path = restorer_path(id)?;
        let (session, pick) = crate::engine::session_on_ladder(id, cpu_only, |mut builder| {
            builder.commit_from_file(&path).map_err(|e| format!("face model {}: {e}", path.display()))
        })?;
        Ok((Self(session), pick))
    }

    /// One aligned `FACE_SIZE` crop, redrawn.
    pub fn redraw(&mut self, crop: &RgbImage) -> Result<RgbImage, CoreError> {
        restore(&mut self.0, crop)
    }
}

fn restorer_path(id: prunr_models::ModelId) -> Result<std::path::PathBuf, CoreError> {
    prunr_models::multi_part_paths(id)
        .ok_or_else(|| CoreError::Inference(prunr_models::not_installed_error(id)))?
        .into_iter()
        .find_map(|(key, path)| (key == "restorer").then_some(path))
        .ok_or_else(|| CoreError::Inference(format!("face model {id:?}: no restorer part")))
}

/// YuNet over a `w × h` photo, which `shrink(sw, sh)` gives scaled to
/// `DETECT_MAX_SIDE`, padded to a multiple of 32. Returns faces in photo
/// pixels, best first.
fn detect(session: &mut Session, (w, h): (u32, u32), shrink: impl FnOnce(u32, u32) -> RgbaImage) -> Result<Vec<Face>, CoreError> {
    let scale = (DETECT_MAX_SIDE as f32 / w.max(h) as f32).min(1.0);
    let (sw, sh) = (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
    let small = shrink(sw, sh);
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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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

impl FaceBlend {
    /// Whether it redraws face `i` of `faces`: every face, or with
    /// `largest_only` the largest.
    pub fn redrawn<'a>(&self, faces: impl IntoIterator<Item = &'a FacePlace>) -> impl Fn(usize) -> bool {
        let largest = self.largest_only.then(|| largest_face(faces)).flatten();
        let all = !self.largest_only;
        move |i| all || largest == Some(i)
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
    compose(out, faces, blend, false);
}

/// `compose_faces_into` over an image made from the one the faces were
/// found in (an upscale after Refine): the photo under each face is taken
/// from `out` again and the restoration moved to its colour, so what
/// Refine did there stays. A few ms more per face.
pub fn compose_faces_onto(out: &mut RgbaImage, faces: &[RestoredFace], blend: FaceBlend) {
    compose(out, faces, blend, true);
}

fn compose(out: &mut RgbaImage, faces: &[RestoredFace], blend: FaceBlend, retake: bool) {
    let strength = blend.strength.clamp(0.0, 1.0);
    if strength == 0.0 || faces.is_empty() {
        return;
    }
    let mask = &face_mask(blend.hair).weights;
    let redrawn = blend.redrawn(faces.iter().map(RestoredFace::place));
    let chosen: Vec<&RestoredFace> = faces.iter().enumerate().filter(|&(i, _)| redrawn(i)).map(|(_, f)| f).collect();
    use rayon::prelude::*;
    let target: &RgbaImage = out;
    let blended: Vec<[Vec<f32>; 3]> = chosen.par_iter().map(|face| {
        let base = if retake { Base::retaken(face, target) } else { Base::stored(face) };
        blend_face(face, &base, strength)
    }).collect();
    for (face, blended) in chosen.iter().zip(&blended) {
        paste(out, face, blended, mask);
    }
}

/// The crop's face area, soft-edged: an ellipse from brow to chin on the
/// FFHQ template, or with `hair` one that takes in the hair and ears.
struct FaceMask {
    /// 0-1 over the 512 crop.
    weights: Vec<f32>,
    /// Where in the crop (u0, v0, u1, v1) a sample of the weights can
    /// reach `HALF_LEVEL`: outside it a blend changes no pixel.
    reach: [f32; 4],
}

fn face_mask(hair: bool) -> &'static FaceMask {
    static FACE: std::sync::OnceLock<FaceMask> = std::sync::OnceLock::new();
    static WITH_HAIR: std::sync::OnceLock<FaceMask> = std::sync::OnceLock::new();
    let build = || {
        let weights = build_face_mask(hair);
        let n = FACE_SIZE as usize;
        let reach = weights.iter().enumerate().filter(|&(_, &w)| w >= HALF_LEVEL).fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |r, (i, _)| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            [r[0].min(x), r[1].min(y), r[2].max(x), r[3].max(y)]
        });
        // A bilinear sample reads the next pixel too.
        FaceMask { weights, reach: [reach[0] - 1.0, reach[1] - 1.0, reach[2] + 1.0, reach[3] + 1.0] }
    };
    if hair { WITH_HAIR.get_or_init(build) } else { FACE.get_or_init(build) }
}

/// The face ellipse on the FFHQ crop: centre and radii, brow to chin, or
/// with `hair` taking in the hair and ears.
fn face_ellipse(hair: bool) -> (f32, f32, f32, f32) {
    if hair { (256.0, 250.0, 225.0, 250.0) } else { (256.0, 280.0, 170.0, 215.0) }
}

fn build_face_mask(hair: bool) -> Vec<f32> {
    let n = FACE_SIZE as usize;
    let (cx, cy, rx, ry) = face_ellipse(hair);
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
    crate::guided_filter::box_blur3_into(&mut out, side, side, r, &mut Vec::new());
    out
}

/// The face at `strength`, in crop space, as RGB planes.
fn blend_face(face: &RestoredFace, base: &Base<'_>, strength: f32) -> [Vec<f32>; 3] {
    let detail_take = (2.0 * strength).min(1.0) * detail_reach(face);
    let shading_take = (2.0 * strength - 1.0).max(0.0);
    let (o, r, o_low, r_low) = (base.original.as_raw(), face.restored.as_raw(), base.original_low.as_raw(), face.restored_low.as_raw());
    std::array::from_fn(|c| {
        let shift = base.shift[c];
        (c..o.len())
            .step_by(3)
            .map(|i| {
                let (o, r, o_low, r_low) = (f32::from(o[i]), f32::from(r[i]) + shift, f32::from(o_low[i]), f32::from(r_low[i]) + shift);
                let low = o_low + (r_low - o_low) * shading_take;
                let (o_detail, r_detail) = (o - o_low, r - r_low);
                low + o_detail + (r_detail - o_detail) * detail_take
            })
            .collect()
    })
}

/// The photo under a face that a blend starts from, and how far its
/// colour moved from the one the restoration was matched to.
struct Base<'a> {
    original: std::borrow::Cow<'a, RgbImage>,
    original_low: std::borrow::Cow<'a, RgbImage>,
    shift: [f32; 3],
}

impl<'a> Base<'a> {
    fn stored(face: &'a RestoredFace) -> Self {
        Self { original: std::borrow::Cow::Borrowed(&face.original), original_low: std::borrow::Cow::Borrowed(&face.original_low), shift: [0.0; 3] }
    }

    /// The photo under `face` taken from `image` again. Shading and colour
    /// are a mean shift apart, so moving the restoration by the change in
    /// the face's mean colour keeps it matched.
    fn retaken(face: &RestoredFace, image: &RgbaImage) -> Self {
        let original = warp_to_crop(image, &face.place.to_crop);
        let original_low = crop_image(&shading(&crop_planes(&original)));
        let mask = &face_mask(false).weights;
        let weight = mask_weight(mask);
        let mean = |img: &RgbImage, c: usize| masked_mean(img.pixels().map(|p| f32::from(p[c])), mask, weight);
        let shift = std::array::from_fn(|c| mean(&original, c) - mean(&face.original, c));
        Self { original: std::borrow::Cow::Owned(original), original_low: std::borrow::Cow::Owned(original_low), shift }
    }
}

/// How far the restored detail reaches, 0.25-1. The restorer draws at
/// 512 px; a face larger than that in the photo has finer detail of its
/// own than the restoration, so it takes the restored detail only in
/// proportion. A face's own sharpness can't decide this: an upscaler's
/// invented edges measure as sharp as real detail.
fn detail_reach(face: &RestoredFace) -> f32 {
    face.place.to_crop.scale().clamp(0.25, 1.0)
}

fn crop_planes(img: &RgbImage) -> [Vec<f32>; 3] {
    std::array::from_fn(|c| img.pixels().map(|p| f32::from(p[c])).collect())
}

fn crop_image(planes: &[Vec<f32>; 3]) -> RgbImage {
    RgbImage::from_fn(FACE_SIZE, FACE_SIZE, |x, y| {
        let i = (y * FACE_SIZE + x) as usize;
        Rgb(std::array::from_fn(|c| planes[c][i].round().clamp(0.0, 255.0) as u8))
    })
}

/// A crop's shading layer: each plane low-passed below the detail scale.
fn shading(planes: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    std::array::from_fn(|c| gaussian(&planes[c], FACE_SIZE, DETAIL_SIGMA))
}

/// The divisor for `masked_mean` over `mask`.
fn mask_weight(mask: &[f32]) -> f32 {
    mask.iter().sum::<f32>().max(f32::EPSILON)
}

fn masked_mean(values: impl Iterator<Item = f32>, mask: &[f32], weight: f32) -> f32 {
    values.zip(mask).map(|(v, m)| v * m).sum::<f32>() / weight
}

/// The photo pixels, as (x0, y0, x1, y1) clipped to `size`, covered by the
/// crop rectangle `[u0, v0, u1, v1]` taken through `to_photo`; `pad` is
/// added past the far edge.
fn photo_span(to_photo: &Similarity, [u0, v0, u1, v1]: [f32; 4], [w, h]: [u32; 2], pad: f32) -> [u32; 4] {
    let corners = [[u0, v0], [u1, v0], [u0, v1], [u1, v1]].map(|p| to_photo.apply(p));
    let lo = |i: usize, max: u32| (corners.iter().map(|p| p[i]).fold(f32::MAX, f32::min).floor().max(0.0) as u32).min(max);
    let hi = |i: usize, max: u32| ((corners.iter().map(|p| p[i]).fold(f32::MIN, f32::max).ceil() + pad).max(0.0) as u32).min(max);
    [lo(0, w), lo(1, h), hi(0, w), hi(1, h)]
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
    let weight = mask_weight(mask);
    let mean = |plane: &[f32]| masked_mean(plane.iter().copied(), mask, weight);
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
    let side = FACE_SIZE as f32;
    let [x0, y0, x1, y1] = photo_span(&face.place.to_crop.inverse(), [0.0, 0.0, side, side], [out.width(), out.height()], 0.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let [u, v] = face.place.to_crop.apply([x as f32, y as f32]);
            if !in_crop(u, v) {
                continue;
            }
            let m = sample_crop(mask, u, v);
            if m <= 0.0 {
                continue;
            }
            let p = out.get_pixel_mut(x, y);
            for c in 0..3 {
                let value = f32::from(p[c]) * (1.0 - m) + sample_crop(&blended[c], u, v) * m;
                p[c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Whether crop point (`u`, `v`) has the four pixels a bilinear sample reads.
fn in_crop(u: f32, v: f32) -> bool {
    let inside = 0.0..(FACE_SIZE - 1) as f32;
    inside.contains(&u) && inside.contains(&v)
}

/// Bilinear sample of a 512² crop plane, edges clamped.
fn sample_crop(plane: &[f32], u: f32, v: f32) -> f32 {
    let n = FACE_SIZE as usize;
    let (u0, v0) = (u.floor(), v.floor());
    let (fu, fv) = (u - u0, v - v0);
    let at = |x: f32, y: f32| plane[(y as usize).min(n - 1) * n + (x as usize).min(n - 1)];
    let top = at(u0, v0) * (1.0 - fu) + at(u0 + 1.0, v0) * fu;
    let bottom = at(u0, v0 + 1.0) * (1.0 - fu) + at(u0 + 1.0, v0 + 1.0) * fu;
    top * (1.0 - fv) + bottom * fv
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

    fn placed(to_crop: Similarity, photo_size: [u32; 2]) -> FacePlace {
        FacePlace { to_crop, bbox: [0.0; 4], photo_size }
    }

    /// A textured 900 × 600 photo with two faces as the detector reports
    /// them: one at the crop's own size at (20, 10), one half that size
    /// at (600, 50), right of the first.
    fn two_faces() -> (RgbaImage, Vec<Face>) {
        let image = RgbaImage::from_fn(900, 600, |x, y| Rgba([(x % 200) as u8 + 30, (y % 180) as u8 + 40, 120, 255]));
        let face = |scale: f32, dx: f32, dy: f32| Face { score: 0.9, bbox: [0.0; 4], landmarks: TEMPLATE.map(|[x, y]| [x * scale + dx, y * scale + dy]) };
        (image, vec![face(1.0, 20.0, 10.0), face(0.5, 600.0, 50.0)])
    }

    /// What the stand-in restorer draws: a fine check, about as much new
    /// detail as a restorer adds to skin, nothing like the photo.
    fn checks(_: &RgbImage) -> Result<RgbImage, CoreError> {
        Ok(RgbImage::from_fn(FACE_SIZE, FACE_SIZE, |x, y| if (x / 2 + y / 2) % 2 == 0 { Rgb([150, 140, 105]) } else { Rgb([115, 100, 135]) }))
    }

    fn previews(updates: &[ProgressUpdate]) -> Vec<&crate::progress::TilePreview> {
        updates.iter().filter_map(|u| match u { ProgressUpdate::TilePreview(p) => Some(p), _ => None }).collect()
    }

    /// The faces are reported as soon as they are found, then each face,
    /// once redrawn, as a preview over its own part of the photo, in the
    /// order they were found.
    #[test]
    fn found_faces_are_reported_then_previewed_one_at_a_time() {
        use crate::progress::{check_contract, RecordingSink};
        let (image, faces) = two_faces();
        let sink = std::sync::Arc::new(RecordingSink::default());
        let progress = Progress::new(sink.clone()).with_tile_previews(1000);
        progress.step(Step::FindingFaces);
        let places = report_found(&image, &faces, &progress);
        let center = places[0].area(false).center;
        assert!((center[0] - 276.0).abs() < 0.01 && (center[1] - 290.0).abs() < 0.01, "where it was found: {center:?}");
        let restored = restore_each(&image, &places, FaceBlend::default(), &progress, checks).expect("restored");
        assert_eq!(restored.len(), 2);

        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        let face = |n| ProgressUpdate::Outer { done: n, total: 2, unit: Unit::Face };
        let without_pixels: Vec<_> = updates.iter().map(|u| match u {
            ProgressUpdate::TilePreview(p) => format!("preview at {},{}", p.x, p.y),
            u => format!("{u:?}"),
        }).collect();
        let at = |p: &FacePlace| {
            let (x, y, ..) = p.reach(false).expect("on the photo");
            format!("preview at {x},{y}")
        };
        assert_eq!(without_pixels, [
            format!("{:?}", ProgressUpdate::Step(Step::FindingFaces)),
            format!("{:?}", ProgressUpdate::Faces(places.clone())),
            format!("{:?}", ProgressUpdate::Step(Step::RestoringFaces)),
            format!("{:?}", face(0)), at(&places[0]),
            format!("{:?}", face(1)), at(&places[1]),
            format!("{:?}", face(2)),
        ]);
        let shown = previews(&updates);
        assert!(shown.iter().all(|p| (p.full_width, p.full_height) == (900, 600)), "at the photo's own size");

        let halved = std::sync::Arc::new(RecordingSink::default());
        let progress = Progress::new(halved.clone()).with_tile_previews(450);
        restore_each(&image, &places, FaceBlend::default(), &progress, checks).expect("restored");
        let updates = halved.updates();
        let small = previews(&updates)[0];
        let alpha = |x: u32, y: u32| small.rgba[((y * small.width + x) * 4 + 3) as usize];
        assert_eq!((small.full_width, small.full_height), (450, 300));
        assert_eq!(alpha(small.width / 2, small.height / 2), 255, "the face's middle");
        assert_eq!((alpha(0, 0), alpha(small.width - 1, small.height - 1)), (0, 0), "clear round the face");
    }

    /// Each preview laid over the photo shows what the blend makes of it
    /// (the face at its strength, through its soft mask), and the blend
    /// changes nothing outside the previews.
    #[test]
    fn a_face_preview_over_the_photo_is_the_blend() {
        use crate::progress::RecordingSink;
        let (image, faces) = two_faces();
        for blend in [FaceBlend::default(), FaceBlend { strength: 1.0, hair: true, largest_only: false }] {
            let sink = std::sync::Arc::new(RecordingSink::default());
            let progress = Progress::new(sink.clone()).with_tile_previews(1000);
            let places = report_found(&image, &faces, &Progress::none());
            let restored = restore_each(&image, &places, blend, &progress, checks).expect("restored");
            let blended = compose_faces(&image, &restored, blend);
            let mut laid = image.clone();
            for p in previews(&sink.updates()) {
                for (i, px) in p.rgba.chunks_exact(4).enumerate() {
                    let (x, y) = (p.x + i as u32 % p.width, p.y + i as u32 / p.width);
                    let a = f32::from(px[3]) / 255.0;
                    let under = laid.get_pixel_mut(x, y);
                    for c in 0..3 {
                        under[c] = (f32::from(under[c]) * (1.0 - a) + f32::from(px[c]) * a).round() as u8;
                    }
                }
            }
            let worst = laid.pixels().zip(blended.pixels()).map(|(a, b)| (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0)).max();
            assert!(worst <= Some(2), "{blend:?}: the previews differ from the blend by {worst:?}");
            assert_ne!(blended, image, "the blend changed the faces");
        }
    }

    /// Under "Largest" only the face it redraws is previewed; every face is
    /// still redrawn, so turning it off needs no new search.
    #[test]
    fn largest_only_previews_only_the_largest_face() {
        use crate::progress::RecordingSink;
        let (image, faces) = two_faces();
        let sink = std::sync::Arc::new(RecordingSink::default());
        let progress = Progress::new(sink.clone()).with_tile_previews(1000);
        let places = report_found(&image, &faces, &Progress::none());
        let blend = FaceBlend { largest_only: true, ..FaceBlend::default() };
        let restored = restore_each(&image, &places, blend, &progress, checks).expect("restored");
        assert_eq!(restored.len(), 2);
        let (x, y, ..) = places[0].reach(false).expect("on the photo");
        assert_eq!(previews(&sink.updates()).iter().map(|p| (p.x, p.y)).collect::<Vec<_>>(), [(x, y)], "the crop-sized face only");
    }

    /// No preview is made unless one is asked for, nor when the blend
    /// redraws nothing.
    #[test]
    fn no_preview_unless_asked_or_when_nothing_is_redrawn() {
        use crate::progress::RecordingSink;
        let (image, faces) = two_faces();
        let places = report_found(&image, &faces, &Progress::none());
        let unasked = std::sync::Arc::new(RecordingSink::default());
        restore_each(&image, &places, FaceBlend::default(), &Progress::new(unasked.clone()), checks).expect("restored");
        assert!(previews(&unasked.updates()).is_empty(), "not asked");
        let at_zero = std::sync::Arc::new(RecordingSink::default());
        let progress = Progress::new(at_zero.clone()).with_tile_previews(1000);
        restore_each(&image, &places, FaceBlend { strength: 0.0, ..FaceBlend::default() }, &progress, checks).expect("restored");
        assert!(previews(&at_zero.updates()).is_empty(), "strength 0 redraws nothing");
    }

    /// A stop asked for while a face is redrawn ends the search once that
    /// face is done, without showing it.
    #[test]
    fn a_stop_ends_the_search_after_the_face_under_way() {
        use crate::progress::RecordingSink;
        let (image, faces) = two_faces();
        let places = report_found(&image, &faces, &Progress::none());
        let sink = std::sync::Arc::new(RecordingSink::default());
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = Progress::new(sink.clone()).with_tile_previews(1000).with_cancel(stop.clone());
        let mut redrawn = 0;
        let result = restore_each(&image, &places, FaceBlend::default(), &progress, |crop| {
            redrawn += 1;
            stop.store(true, std::sync::atomic::Ordering::Release);
            checks(crop)
        });
        assert!(matches!(result, Err(CoreError::Cancelled)));
        assert_eq!(redrawn, 1, "the face under way finishes, the next does not start");
        assert!(previews(&sink.updates()).is_empty());
    }

    fn face_of(original: RgbImage, restored: RgbImage) -> RestoredFace {
        // Crop pixel (u, v) is photo pixel (u, v) + (10, 20).
        RestoredFace::new(placed(Similarity { a: 1.0, b: 0.0, tx: -10.0, ty: -20.0 }, [1000, 1000]), original, restored)
    }

    /// A face's area is the crop's ellipse back in the photo: moved,
    /// scaled and turned with the alignment; "Largest" names the biggest.
    #[test]
    fn a_face_area_is_the_crop_ellipse_back_in_the_photo() {
        let flat = || RgbImage::from_pixel(512, 512, Rgb([120, 100, 90]));
        // Photo to crop: halve, then move by (-100, -50): the face is twice the crop's size.
        let face = RestoredFace::new(placed(Similarity { a: 0.5, b: 0.0, tx: -100.0, ty: -50.0 }, [2000, 1500]), flat(), flat());
        let area = face.place().area(false);
        let (cx, cy, rx, ry) = face_ellipse(false);
        assert!((area.center[0] - (cx + 100.0) * 2.0).abs() < 1e-3 && (area.center[1] - (cy + 50.0) * 2.0).abs() < 1e-3, "{area:?}");
        assert!((area.radii[0] - rx * 2.0).abs() < 1e-3 && (area.radii[1] - ry * 2.0).abs() < 1e-3);
        assert!(area.angle.abs() < 1e-6 && area.photo_size == [2000, 1500]);
        assert!(face.place().area(true).radii[0] > area.radii[0], "with hair, wider");
        let turned = RestoredFace::new(placed(Similarity { a: 0.0, b: 1.0, tx: 0.0, ty: 0.0 }, [2000, 1500]), flat(), flat());
        assert!((turned.place().area(false).angle + std::f32::consts::FRAC_PI_2).abs() < 1e-5, "a quarter turn back");
        assert_eq!(largest_face([face.place(), turned.place()]), Some(0), "the one the crop shrinks");
        assert_eq!(largest_face(&[]), None);
    }

    /// "Largest only" keeps the face the alignment shrinks most.
    #[test]
    fn largest_only_restores_the_biggest_face() {
        let image = RgbaImage::from_pixel(1400, 700, Rgba([120, 100, 90, 255]));
        let flat = RgbImage::from_pixel(512, 512, Rgb([120, 100, 90]));
        let stripes = RgbImage::from_fn(512, 512, |x, _| if x % 2 == 0 { Rgb([160, 140, 130]) } else { Rgb([80, 60, 50]) });
        let small = RestoredFace::new(placed(Similarity { a: 2.0, b: 0.0, tx: -2000.0, ty: 0.0 }, [1400, 700]), flat.clone(), stripes.clone());
        let big = RestoredFace::new(placed(Similarity { a: 1.0, b: 0.0, tx: 0.0, ty: 0.0 }, [1400, 700]), flat, stripes);
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
        let at = |a: f32| detail_reach(&RestoredFace::new(placed(Similarity { a, b: 0.0, tx: 0.0, ty: 0.0 }, [1000, 1000]), flat.clone(), flat.clone()));
        assert_eq!(at(0.5), 0.5);
        assert_eq!(at(1.6), 1.0);
        assert_eq!(at(0.1), 0.25, "never below a quarter");
    }

    /// Onto a refined image (here warmer than the photo the face was found
    /// in) the blend starts from the refined pixels, not the stored ones.
    #[test]
    fn a_blend_onto_a_refined_image_keeps_the_refine() {
        let original = RgbImage::from_pixel(512, 512, Rgb([100, 100, 100]));
        let restored = RgbImage::from_pixel(512, 512, Rgb([100, 100, 100]));
        let face = face_of(original, restored);
        let refined = RgbaImage::from_pixel(600, 600, Rgba([160, 120, 100, 255]));
        let half = FaceBlend { strength: 0.5, ..FaceBlend::default() };
        let mut onto = refined.clone();
        compose_faces_onto(&mut onto, std::slice::from_ref(&face), half);
        let p = onto.get_pixel(10 + 256, 20 + 280);
        assert!(p[0].abs_diff(160) <= 2 && p[1].abs_diff(120) <= 2, "the refined colour stays: {p:?}");
        let mut into = refined.clone();
        compose_faces_into(&mut into, std::slice::from_ref(&face), half);
        assert!(into.get_pixel(10 + 256, 20 + 280)[0] < 140, "the stored photo would undo it");
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
