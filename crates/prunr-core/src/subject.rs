//! A coarse look at where the subject is, so the app can outline it while
//! the run works: taken the moment segmentation inference ends, and, for a
//! slow run, by a fast model before the slow one starts.

use std::time::Duration;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::engine::OrtEngine;
use crate::progress::{Progress, Step};
use crate::types::{CoreError, ModelKind};

/// The model that takes the first look: it finds the subject in about a
/// second on a CPU.
pub const FIRST_LOOK_MODEL: ModelKind = ModelKind::Silueta;

/// A run expected to take longer than this per image shows a first look;
/// a quicker one lands its own result about when a first look would.
pub const FIRST_LOOK_AFTER: Duration = Duration::from_secs(2);

/// Whether `model` can be slow enough for a first look: BiRefNet and
/// U2Net, never the first-look model itself.
pub fn may_take_first_look(model: ModelKind) -> bool {
    matches!(model, ModelKind::BiRefNetLite | ModelKind::U2net)
}

/// Whether a run of `model` expected to take `expected` per image
/// (`None`: not timed yet on this provider) shows a first look. A fast
/// accelerator's own result lands before a first look would; a weak one,
/// or the CPU, takes seconds. `expected` is asked only for a model that
/// may take one: the window asks every frame a batch runs.
pub fn first_look_pays(model: ModelKind, expected: impl FnOnce() -> Option<Duration>) -> bool {
    may_take_first_look(model) && expected().is_none_or(|t| t > FIRST_LOOK_AFTER)
}

/// The first-look model's engine, on the CPU beside a pool of `pool_size`
/// engines, with the share of the cores each of them has.
pub fn first_look_engine(pool_size: usize) -> Result<OrtEngine, CoreError> {
    OrtEngine::new_cpu_only(FIRST_LOOK_MODEL, crate::batch::ort_intra_threads(pool_size))
}

/// Report the subject `find_subject` sees in `img`, before the run's own
/// model starts. A cancel before or during it stops it unreported.
pub fn first_look(img: &DynamicImage, look: &OrtEngine, progress: &Progress) -> Result<(), CoreError> {
    progress.check_cancelled()?;
    let started = std::time::Instant::now();
    progress.step(Step::FindingSubject);
    let hint = find_subject(img, look)?;
    progress.check_cancelled()?;
    progress.subject(|| hint);
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "first look reported");
    Ok(())
}

/// Where `look` (the first-look model's engine) sees the subject in
/// `img`, for the first look's outline.
///
/// Peak beyond the session: the RGB copy the resize to the model's square
/// reads, gone before the model runs.
///
/// | image  | RGB copy |
/// |--------|----------|
/// | 12 MP  | 36 MB    |
/// | 24 MP  | 72 MB    |
pub fn find_subject(img: &DynamicImage, look: &OrtEngine) -> Result<SubjectHint, CoreError> {
    read_first_look(img, look, subject_hint)
}

/// The subject `look` is sure of in `photo` (`sure_subject_hint`), for
/// Auto focus; `None` when it is sure of none. The photo is shrunk to the
/// model's square first: no full-size copy, only the square (400 KB at
/// 320 px). The photos measured are opaque, so dropping alpha after the
/// shrink reads the same pixels as before it.
pub fn find_sure_subject(photo: &image::RgbaImage, look: &OrtEngine) -> Result<Option<SubjectHint>, CoreError> {
    let side = crate::preprocess::input_side(look.model_kind());
    let square = crate::formats::resize_rgba(photo, side, side, crate::formats::ResizeFilter::Lanczos3);
    find_sure_subject_decoded(&DynamicImage::ImageRgba8(square), look)
}

/// `find_sure_subject` for a decoded photo in any layout. Peak beyond the
/// session: the RGB copy the resize to the model's square reads (36 MB at
/// 12 MP), gone before the model runs.
pub fn find_sure_subject_decoded(img: &DynamicImage, look: &OrtEngine) -> Result<Option<SubjectHint>, CoreError> {
    read_first_look(img, look, sure_subject_hint)
}

fn read_first_look<T>(img: &DynamicImage, look: &OrtEngine, read: fn(&[f32], usize, usize, ModelKind) -> T) -> Result<T, CoreError> {
    let model = look.model_kind();
    look.infer_into(crate::preprocess::preprocess(img, model), |view| Ok(read_output(view, model, read)))
}

/// The hint for a segmentation model's raw output (`1 × 1 × h × w`).
pub(crate) fn hint_of_output(view: ndarray::ArrayView4<'_, f32>, model: ModelKind) -> SubjectHint {
    read_output(view, model, subject_hint)
}

fn read_output<T>(view: ndarray::ArrayView4<'_, f32>, model: ModelKind, read: fn(&[f32], usize, usize, ModelKind) -> T) -> T {
    let plane = view.slice(ndarray::s![0, 0, .., ..]);
    let (h, w) = (plane.nrows(), plane.ncols());
    let flat = plane.as_standard_layout();
    read(flat.as_slice().unwrap_or(&[]), h, w, model)
}

/// Longest side of a subject hint; plenty for a glow around the subject.
pub const SUBJECT_HINT_SIDE: usize = 128;

/// Subject coverage over the whole image (stretched to the model's square
/// input), `0` background to `255` subject, row-major.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubjectHint {
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

/// The hint for a raw `tensor_h × tensor_w` segmentation output, read the
/// way the mask is (`SegNormaliser`), over the sampled values.
pub fn subject_hint(tensor: &[f32], tensor_h: usize, tensor_w: usize, model: ModelKind) -> SubjectHint {
    let (w, h, samples) = sampled(tensor, tensor_h, tensor_w);
    let (lo, hi, _) = crate::math::slice_stats(&samples);
    let norm = crate::postprocess::SegNormaliser::new(lo, hi, model);
    let coverage = samples.iter().map(|&v| crate::math::round_u8(norm.apply(v) * 255.0)).collect();
    SubjectHint { width: w, height: h, coverage }
}

/// Fewest of a hint's cells a sure subject covers. Photos with a subject
/// give 16 to 43 % of cells past one half; photos without one give none.
const SURE_SUBJECT_MIN_SHARE: f32 = 0.005;

/// The cells the model is sure are subject (probability past one half)
/// at `255`, the rest at `0`; `None` when too few are for a subject.
/// Unlike `subject_hint` nothing is stretched, so faint output on a photo
/// without a subject stays background.
pub fn sure_subject_hint(tensor: &[f32], tensor_h: usize, tensor_w: usize, model: ModelKind) -> Option<SubjectHint> {
    let (w, h, samples) = sampled(tensor, tensor_h, tensor_w);
    let logits = crate::postprocess::outputs_logits(model);
    let coverage: Vec<u8> = samples
        .iter()
        .map(|&v| if logits { crate::math::sigmoid(v) } else { v })
        .map(|p| if p > 0.5 { 255 } else { 0 })
        .collect();
    let sure = coverage.iter().filter(|&&c| c == 255).count();
    (sure as f32 >= SURE_SUBJECT_MIN_SHARE * coverage.len() as f32).then_some(SubjectHint { width: w, height: h, coverage })
}

/// `tensor` sampled to at most `SUBJECT_HINT_SIDE` a side: width, height
/// and the values, row-major.
fn sampled(tensor: &[f32], tensor_h: usize, tensor_w: usize) -> (u32, u32, Vec<f32>) {
    let scale = (SUBJECT_HINT_SIDE as f32 / tensor_h.max(tensor_w).max(1) as f32).min(1.0);
    let (w, h) = (((tensor_w as f32 * scale) as usize).max(1), ((tensor_h as f32 * scale) as usize).max(1));
    let samples = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let ty = (y * tensor_h / h).min(tensor_h - 1);
            let tx = (x * tensor_w / w).min(tensor_w - 1);
            tensor.get(ty * tensor_w + tx).copied().unwrap_or(0.0)
        })
        .collect();
    (w as u32, h as u32, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hint_is_small_stretched_and_keeps_where_the_subject_is() {
        let (h, w) = (1024usize, 1024usize);
        let tensor: Vec<f32> = (0..h * w).map(|i| if (i % w) < w / 2 { 4.0 } else { -4.0 }).collect();
        let hint = subject_hint(&tensor, h, w, ModelKind::BiRefNetLite);
        assert_eq!((hint.width, hint.height), (128, 128));
        assert_eq!(hint.coverage[0], 255, "the left half is the subject");
        assert_eq!(hint.coverage[127], 0, "the right half is not");
    }

    /// Auto focus aims only at a subject the first look is sure of: the
    /// cells it gives past one half (Silueta's probabilities as they are,
    /// BiRefNet's logits through a sigmoid). Faint output, which the
    /// stretched hint would still call a subject, is none.
    #[test]
    fn a_sure_subject_is_what_the_first_look_gives_past_one_half() {
        let (h, w) = (64usize, 64usize);
        let half: Vec<f32> = (0..h * w).map(|i| if i % w < w / 2 { 0.9 } else { 0.05 }).collect();
        let hint = sure_subject_hint(&half, h, w, ModelKind::Silueta).expect("sure of the left half");
        assert_eq!((hint.coverage[0], hint.coverage[hint.width as usize - 1]), (255, 0));
        let faint: Vec<f32> = (0..h * w).map(|i| if i % w < w / 2 { 0.13 } else { 0.0 }).collect();
        assert!(subject_hint(&faint, h, w, ModelKind::Silueta).coverage[0] == 255, "stretched, faint output reads as subject");
        assert!(sure_subject_hint(&faint, h, w, ModelKind::Silueta).is_none(), "but no cell is past one half");
        let logits: Vec<f32> = half.iter().map(|&p| if p > 0.5 { 4.0 } else { -4.0 }).collect();
        assert!(sure_subject_hint(&logits, h, w, ModelKind::BiRefNetLite).is_some());
    }

    /// Which runs take a first look: BiRefNet and U2Net, untimed or timed
    /// over the threshold; never the first-look model itself.
    #[test]
    fn only_a_slow_run_pays_for_a_first_look() {
        let over = Some(FIRST_LOOK_AFTER + Duration::from_millis(1));
        let under = Some(Duration::from_millis(400));
        for (model, expected, pays) in [
            (ModelKind::BiRefNetLite, None, true),
            (ModelKind::BiRefNetLite, over, true),
            (ModelKind::BiRefNetLite, under, false),
            (ModelKind::BiRefNetLite, Some(FIRST_LOOK_AFTER), false),
            (ModelKind::U2net, None, true),
            (ModelKind::U2net, over, true),
            (ModelKind::U2net, under, false),
            (ModelKind::Silueta, None, false),
            (ModelKind::Silueta, over, false),
            (ModelKind::DdcolorTiny, None, false),
        ] {
            assert_eq!(first_look_pays(model, || expected), pays, "{model:?} expected {expected:?}");
        }
        assert!(!may_take_first_look(FIRST_LOOK_MODEL));
    }

    #[test]
    fn a_flat_tensor_reads_as_all_or_nothing() {
        let hint = subject_hint(&[0.9; 16], 4, 4, ModelKind::Silueta);
        assert!(hint.coverage.iter().all(|&c| c == 255));
        assert_eq!((hint.width, hint.height), (4, 4), "a small tensor is not enlarged");
    }
}
