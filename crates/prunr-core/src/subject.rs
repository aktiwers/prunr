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

/// Report where `look` (the first-look model's engine) sees the subject
/// in `img`, before the run's own model starts. A cancel before or during
/// it stops it unreported.
///
/// Peak beyond the session: the RGB copy the resize to the model's 320 px
/// square reads, gone before the model runs, as the run's own model
/// makes next.
///
/// | image  | RGB copy |
/// |--------|----------|
/// | 12 MP  | 36 MB    |
/// | 24 MP  | 72 MB    |
pub fn first_look(img: &DynamicImage, look: &OrtEngine, progress: &Progress) -> Result<(), CoreError> {
    progress.check_cancelled()?;
    let started = std::time::Instant::now();
    let model = look.model_kind();
    progress.step(Step::FindingSubject);
    let hint = look.infer_into(crate::preprocess::preprocess(img, model), |view| Ok(hint_of_output(view, model)))?;
    progress.check_cancelled()?;
    progress.subject(|| hint);
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "first look reported");
    Ok(())
}

/// The hint for a segmentation model's raw output (`1 × 1 × h × w`).
pub(crate) fn hint_of_output(view: ndarray::ArrayView4<'_, f32>, model: ModelKind) -> SubjectHint {
    let plane = view.slice(ndarray::s![0, 0, .., ..]);
    let (h, w) = (plane.nrows(), plane.ncols());
    let flat = plane.as_standard_layout();
    subject_hint(flat.as_slice().unwrap_or(&[]), h, w, model)
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
    let scale = (SUBJECT_HINT_SIDE as f32 / tensor_h.max(tensor_w).max(1) as f32).min(1.0);
    let (w, h) = (((tensor_w as f32 * scale) as usize).max(1), ((tensor_h as f32 * scale) as usize).max(1));
    let samples: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let ty = (y * tensor_h / h).min(tensor_h - 1);
            let tx = (x * tensor_w / w).min(tensor_w - 1);
            tensor.get(ty * tensor_w + tx).copied().unwrap_or(0.0)
        })
        .collect();
    let (lo, hi, _) = crate::math::slice_stats(&samples);
    let norm = crate::postprocess::SegNormaliser::new(lo, hi, model);
    let coverage = samples.iter().map(|&v| crate::math::round_u8(norm.apply(v) * 255.0)).collect();
    SubjectHint { width: w as u32, height: h as u32, coverage }
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
