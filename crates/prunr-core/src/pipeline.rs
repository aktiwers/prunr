use image::DynamicImage;
use ort::{inputs, value::Tensor};

use crate::{
    engine::OrtEngine,
    formats::load_image_from_bytes,
    postprocess::{postprocess, PostprocessOpts},
    preprocess::preprocess,
    progress::{Progress, Step},
    types::{CoreError, MaskSettings, ProcessResult},
};

/// Process a single image: remove background and return a transparent PNG.
///
/// # Arguments
/// - `img_bytes`: Raw image bytes (PNG, JPEG, WebP, or BMP)
/// - `engine`: OrtEngine with a loaded session. Create once, reuse across images.
/// - `progress`: the run's progress handle; reports Reading the image,
///   Finding the subject and Refining the edges, and carries cancel.
///
/// # Errors
/// - `CoreError::TooLarge` if the photo's full-size work does not fit the memory free now
/// - `CoreError::ImageFormat` if the bytes cannot be decoded as a supported format
/// - `CoreError::Inference` if ORT inference fails
pub fn process_image(img_bytes: &[u8], engine: &OrtEngine, progress: &Progress) -> Result<ProcessResult, CoreError> {
    process_image_with_mask(img_bytes, engine, &MaskSettings::default(), progress)
}

pub fn process_image_with_mask(
    img_bytes: &[u8],
    engine: &OrtEngine,
    mask: &MaskSettings,
    progress: &Progress,
) -> Result<ProcessResult, CoreError> {
    progress.step(Step::ReadingImage);
    if let Some((w, h)) = crate::formats::image_dimensions_from_bytes(img_bytes) {
        room_for_photo(w, h, false, mask.refine_edges)?;
    }
    let img = load_image_from_bytes(img_bytes)?;
    process_image_from_decoded(&img, engine, mask, progress)
}

/// A run's working memory at the photo's own size, per pixel, beyond the
/// model: the decoded photo, its RGBA copy, the mask scaled up to it and
/// the result. Measured on Silueta from 12 to 75 MP (RTX 3060 box): a
/// cut-out 7.7 bytes, with lines 11.5, with Refine edges 42 (its guided
/// filter holds float planes of the whole photo; lines then add nothing
/// to the peak). Rounded up for peaks the 20 ms samples missed.
const CUTOUT_BYTES_PER_PX: u64 = 10;
const LINES_BYTES_PER_PX: u64 = 14;
const REFINED_BYTES_PER_PX: u64 = 48;

/// What a run on a `w` × `h` photo holds at full size beyond the model,
/// with lines and with Refine edges:
///
/// | Photo | Cut-out | Lines | Refine edges |
/// |-------|---------|-------|--------------|
/// | 12 MP | 0.1 GB  | 0.2 GB | 0.6 GB      |
/// | 75 MP | 0.7 GB  | 1.0 GB | 3.4 GB      |
pub fn full_size_bytes(w: u32, h: u32, lines: bool, refine: bool) -> u64 {
    let per_px = if refine {
        REFINED_BYTES_PER_PX
    } else if lines {
        LINES_BYTES_PER_PX
    } else {
        CUTOUT_BYTES_PER_PX
    };
    u64::from(w) * u64::from(h) * per_px
}

/// Refuses a `w` × `h` photo whose full-size work does not fit the memory
/// and swap free now (`sysmem::room_for_buffers_bytes`); with no reading
/// of free memory it runs.
pub fn room_for_photo(w: u32, h: u32, lines: bool, refine: bool) -> Result<(), CoreError> {
    crate::sysmem::room_for_buffers_bytes().map_or(Ok(()), |free| fits_in(w, h, full_size_bytes(w, h, lines, refine), free))
}

fn fits_in(width: u32, height: u32, need: u64, free: u64) -> Result<(), CoreError> {
    if need <= free {
        return Ok(());
    }
    Err(CoreError::TooLarge { width, height, need_mb: need >> 20, free_mb: free >> 20 })
}

/// Run the full pipeline on an already-decoded image.
///
/// Use this when the image is already in memory (e.g., chain mode) to
/// avoid an unnecessary encode→decode round-trip through bytes.
pub fn process_image_from_decoded(
    img: &DynamicImage,
    engine: &OrtEngine,
    mask: &MaskSettings,
    progress: &Progress,
) -> Result<ProcessResult, CoreError> {
    if progress.is_cancelled() {
        return Err(CoreError::Cancelled);
    }
    let model = engine.model_kind();
    let input_array = preprocess(img, model);

    if progress.is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    // Postprocess reads ORT's bound output buffer in place, skipping the
    // `to_owned()` copy (4 MB at BiRefNet 1024², ~11.9% of CLI batch wall time).
    progress.step(Step::FindingSubject);

    let postprocess_opts = PostprocessOpts::new(mask, model);
    let rgba_image = engine.infer_into(input_array, |view| {
        if progress.is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        progress.subject(|| crate::subject::hint_of_output(view.view(), model));
        progress.step(Step::RefiningEdges);
        Ok(postprocess(view, img, &postprocess_opts))
    })?;

    Ok(ProcessResult {
        rgba_image,
        active_provider: engine.active_provider().to_string(),
    })
}

/// `infer_only`, after `look` (the first-look model's engine, when given)
/// has reported where it sees the subject. A first look that fails leaves
/// the run to go on without it; a cancel stops both.
pub fn infer_after_first_look(
    img: &DynamicImage,
    engine: &OrtEngine,
    look: Option<&OrtEngine>,
    progress: &Progress,
) -> Result<crate::types::InferenceResult, CoreError> {
    if let Some(look) = look {
        match crate::subject::first_look(img, look, progress) {
            Err(CoreError::Cancelled) => return Err(CoreError::Cancelled),
            Err(err) => tracing::warn!(%err, "first look failed; the run goes on without it"),
            Ok(()) => {}
        }
    }
    infer_only(img, engine, progress)
}

/// Run inference only (Tier 1). Returns the raw model tensor without postprocessing.
/// Reports Finding the subject; the caller reports the steps it runs next.
pub fn infer_only(img: &DynamicImage, engine: &OrtEngine, progress: &Progress) -> Result<crate::types::InferenceResult, CoreError> {
    if progress.is_cancelled() { return Err(CoreError::Cancelled); }
    let model = engine.model_kind();
    let input_array = preprocess(img, model);
    // Diagnostic stats walk the full preprocessed tensor (~16 MB at
    // BiRefNet 1024²); skip when DEBUG isn't enabled.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let shape = input_array.shape();
        let s = input_array.as_slice().unwrap_or(&[]);
        let (lo, hi, mean) = crate::math::slice_stats(s);
        tracing::debug!(
            ?model, input_shape = ?shape, input_len = s.len(),
            input_min = lo, input_max = hi, input_mean = mean,
            "infer_only preprocessed tensor ready",
        );
    }

    if progress.is_cancelled() { return Err(CoreError::Cancelled); }

    progress.step(Step::FindingSubject);
    let raw_output = engine.with_session(|session| {
        let input_name = session.inputs()[0].name().to_string();
        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| CoreError::Inference(format!("Failed to create input tensor: {e}")))?;
        let outputs = session
            .run(inputs![input_name.as_str() => &input_tensor])
            .map_err(|e| CoreError::Inference(format!("ORT inference failed: {e}")))?;
        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("Failed to extract output tensor: {e}")))?;
        tracing::debug!(
            shape = ?arr.shape(), strides = ?arr.strides(),
            is_standard = arr.is_standard_layout(),
            "infer_only ORT output extracted",
        );
        arr.into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("Output reshape error: {e}")))
            .map(|v| v.to_owned())
    })?;

    // Some EPs (DirectML, plausibly CoreML) return non-standard layouts;
    // reading the backing Vec as C-order then gives garbage.
    let is_std_owned = raw_output.is_standard_layout();
    let h = raw_output.shape()[2];
    let w = raw_output.shape()[3];
    // `from_shape_vec` on a nonzero-offset Vec includes the stale prefix
    // as fake tensor data — silent corruption. Slice past the offset
    // explicitly instead.
    let tensor_data: Vec<f32> = if is_std_owned {
        let (buf, offset) = raw_output.into_raw_vec_and_offset();
        match offset {
            Some(0) | None => buf,
            Some(off) => {
                debug_assert!(
                    off + h * w <= buf.len(),
                    "ndarray returned offset {off} + h*w {} > buf.len {}",
                    h * w, buf.len(),
                );
                buf[off..off + h * w].to_vec()
            }
        }
    } else {
        let standard = raw_output.as_standard_layout();
        // invariant: as_standard_layout always produces a contiguous view.
        standard.as_slice().unwrap().to_vec()
    };
    // Walks the full ~16 MB tensor; skip unless DEBUG is on.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let (lo, hi, mean) = crate::math::slice_stats(&tensor_data);
        let head: Vec<f32> = tensor_data.iter().take(6).copied().collect();
        tracing::debug!(
            ?model, h, w, tensor_len = tensor_data.len(),
            owned_was_standard = is_std_owned,
            tensor_min = lo, tensor_max = hi, tensor_mean = mean,
            ?head,
            "infer_only raw tensor flattened",
        );
    }
    progress.subject(|| crate::subject::subject_hint(&tensor_data, h, w, model));
    Ok(crate::types::InferenceResult {
        tensor_data,
        tensor_height: h,
        tensor_width: w,
        model,
        active_provider: engine.active_provider().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_photo_runs_when_its_full_size_work_fits_the_free_memory() {
        let gb = 1 << 30;
        assert!(fits_in(12000, 8000, 5 * gb, 6 * gb).is_ok());
        assert!(fits_in(12000, 8000, 6 * gb, 6 * gb).is_ok(), "exactly what is free");
        match fits_in(12000, 8000, 7 * gb, 6 * gb) {
            Err(CoreError::TooLarge { width: 12000, height: 8000, need_mb: 7168, free_mb: 6144 }) => {}
            other => panic!("expected a refusal naming the photo and both sizes, got {other:?}"),
        }
    }

    #[test]
    fn full_size_work_grows_with_the_pixels_and_most_with_refine_edges() {
        for (lines, refine) in [(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(full_size_bytes(2000, 1000, lines, refine) * 4, full_size_bytes(4000, 2000, lines, refine));
        }
        let (cut_out, lines, refined) = (full_size_bytes(4000, 2000, false, false), full_size_bytes(4000, 2000, true, false), full_size_bytes(4000, 2000, false, true));
        assert!(cut_out < lines && lines < refined);
        assert_eq!(full_size_bytes(4000, 2000, true, true), refined, "lines add nothing to a refined run's peak");
    }
    use image::{DynamicImage, Rgb, RgbImage};
    use std::io::Cursor;

    fn encode_as_png(img: DynamicImage) -> Vec<u8> {
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn make_png(width: u32, height: u32) -> Vec<u8> {
        encode_as_png(DynamicImage::ImageRgb8(RgbImage::from_pixel(
            width,
            height,
            Rgb([100, 150, 200]),
        )))
    }

    /// A dark disc on a light ground: something for a model to find.
    fn disc_photo(w: u32, h: u32) -> DynamicImage {
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 / 3.0);
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            let inside = (x as f32 - cx).hypot(y as f32 - cy) < r;
            if inside { image::Rgb([40, 30, 90]) } else { image::Rgb([235, 225, 210]) }
        }))
    }

    /// What kind of report each one is, for comparing the order of a run's.
    fn kinds(updates: &[crate::progress::ProgressUpdate]) -> Vec<&'static str> {
        use crate::progress::ProgressUpdate;
        updates.iter().map(|u| match u {
            ProgressUpdate::Step(_) => "step",
            ProgressUpdate::Subject(_) => "subject",
            _ => "count",
        }).collect()
    }

    /// The first look reports where the subject is before the run's own
    /// model starts, and the run then reports its own; together they are
    /// still a run the progress pill can follow.
    #[test]
    fn the_first_look_reports_the_subject_before_the_runs_own_model_starts() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        use crate::types::ModelKind;
        let look = crate::subject::first_look_engine(1).expect("the first-look model loads");
        assert_eq!(look.model_kind(), crate::subject::FIRST_LOOK_MODEL);
        let run = OrtEngine::new_cpu_only(ModelKind::Silueta, 2).expect("the run's model loads");
        let sink = std::sync::Arc::new(RecordingSink::default());
        let ir = infer_after_first_look(&disc_photo(240, 180), &run, Some(&look), &Progress::new(sink.clone()))
            .expect("the run");
        assert!(!ir.tensor_data.is_empty());
        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        assert_eq!(kinds(&updates), ["step", "subject", "step", "subject"], "{updates:?}");
        assert_eq!(updates[0], ProgressUpdate::Step(Step::FindingSubject));
        let ProgressUpdate::Subject(first) = &updates[1] else { unreachable!() };
        assert!(first.coverage.iter().any(|&c| c >= 128), "the first look found the disc");

        let alone = std::sync::Arc::new(RecordingSink::default());
        infer_after_first_look(&disc_photo(240, 180), &run, None, &Progress::new(alone.clone())).expect("the run");
        assert_eq!(kinds(&alone.updates()), ["step", "subject"], "without a look, the run reports as before");
    }

    /// Cancel stops the first look: before it starts, or while it runs,
    /// and then it reports no subject and the run's own model never starts.
    #[test]
    fn a_cancel_stops_the_first_look_unreported() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        use crate::progress::{ProgressSink, ProgressUpdate, RecordingSink};
        use crate::types::ModelKind;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        /// Cancels as soon as the look starts.
        struct CancelOnStart(Arc<AtomicBool>, RecordingSink);
        impl ProgressSink for CancelOnStart {
            fn report(&self, update: ProgressUpdate) {
                self.0.store(true, Ordering::Release);
                self.1.report(update);
            }
        }

        let look = crate::subject::first_look_engine(1).expect("the first-look model loads");
        let run = OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("the run's model loads");
        let photo = disc_photo(120, 90);

        let before = Arc::new(RecordingSink::default());
        let progress = Progress::new(before.clone()).with_cancel(Arc::new(AtomicBool::new(true)));
        assert!(matches!(infer_after_first_look(&photo, &run, Some(&look), &progress), Err(CoreError::Cancelled)));
        assert!(before.updates().is_empty(), "nothing reported: {:?}", before.updates());

        let flag = Arc::new(AtomicBool::new(false));
        let during = Arc::new(CancelOnStart(flag.clone(), RecordingSink::default()));
        let progress = Progress::new(during.clone()).with_cancel(flag);
        assert!(matches!(infer_after_first_look(&photo, &run, Some(&look), &progress), Err(CoreError::Cancelled)));
        assert_eq!(during.1.updates(), [ProgressUpdate::Step(Step::FindingSubject)], "the look ran but never showed");
    }

    /// Timings on this machine's CPU, for the first look's design: how
    /// soon it shows, and what it adds to the slow model's run, run
    /// before it or alongside it. `PRUNR_FIRST_LOOK_PHOTO` names a photo;
    /// a 12 MP drawing otherwise.
    #[test]
    #[ignore = "measurement"]
    fn first_look_timings() {
        use crate::progress::{ProgressSink, ProgressUpdate};
        use crate::types::ModelKind;
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        /// When the first subject was reported.
        struct FirstSubject(Instant, Mutex<Option<Duration>>);
        impl ProgressSink for FirstSubject {
            fn report(&self, update: ProgressUpdate) {
                if matches!(update, ProgressUpdate::Subject(_)) {
                    self.1.lock().unwrap().get_or_insert(self.0.elapsed());
                }
            }
        }
        // The best of three: other work on the machine only ever slows a run.
        let first_subject = |run: &dyn Fn(&Progress)| {
            (0..3).map(|_| {
                let sink = Arc::new(FirstSubject(Instant::now(), Mutex::new(None)));
                run(&Progress::new(sink.clone()));
                let total = sink.0.elapsed();
                let shown = sink.1.lock().unwrap().unwrap_or_default();
                (shown, total)
            }).fold((Duration::MAX, Duration::MAX), |(s, t), (s2, t2)| (s.min(s2), t.min(t2)))
        };
        let secs = |d: Duration| d.as_secs_f32();

        let photo = std::env::var_os("PRUNR_FIRST_LOOK_PHOTO")
            .map(|p| image::open(p).expect("photo"))
            .unwrap_or_else(|| disc_photo(4000, 3000));
        let started = Instant::now();
        let look = crate::subject::first_look_engine(1).expect("first-look model");
        eprintln!("first-look session built in {:.2}s", secs(started.elapsed()));
        crate::subject::first_look(&photo, &look, &Progress::none()).expect("warm-up");
        let (shown, _) = first_subject(&|p| crate::subject::first_look(&photo, &look, p).expect("look"));
        eprintln!("first look alone: {:.2}s", secs(shown));

        for model in [ModelKind::BiRefNetLite, ModelKind::U2net] {
            if !prunr_models::is_available(model.into()) {
                eprintln!("{model:?}: not installed, skipped");
                continue;
            }
            let engine = crate::batch::create_engine_pool(model, 1, true).expect("engine").remove(0);
            infer_only(&photo, &engine, &Progress::none()).expect("warm-up");
            let (own, alone) = first_subject(&|p| { infer_only(&photo, &engine, p).expect("run"); });
            let (look_shown, before) = first_subject(&|p| { infer_after_first_look(&photo, &engine, Some(&look), p).expect("run"); });
            let (side_shown, alongside) = first_subject(&|p| {
                std::thread::scope(|s| {
                    s.spawn(|| crate::subject::first_look(&photo, &look, p).expect("look"));
                    infer_only(&photo, &engine, &Progress::none()).expect("run");
                });
            });
            eprintln!(
                "{model:?}: alone {:.2}s (its own subject at {:.2}s) | look first: subject at {:.2}s, run {:.2}s | alongside: subject at {:.2}s, run {:.2}s",
                secs(alone), secs(own), secs(look_shown), secs(before), secs(side_shown), secs(alongside),
            );
        }
    }

    #[test]
    fn test_process_image_bad_bytes() {
        // load_image_from_bytes should return ImageFormat error for garbage bytes
        use crate::formats::load_image_from_bytes;
        let result = load_image_from_bytes(b"not_an_image");
        assert!(matches!(result, Err(CoreError::ImageFormat(_))));
    }

    #[test]
    fn test_process_image_progress_stages_called() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        use crate::{engine::OrtEngine, types::ModelKind};
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};

        let engine = OrtEngine::new(ModelKind::Silueta, 1).expect("Silueta is bundled");
        let sink = std::sync::Arc::new(RecordingSink::default());
        let png_bytes = make_png(64, 64);
        process_image(&png_bytes, &engine, &Progress::new(sink.clone())).expect("process_image");

        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        assert!(matches!(updates.as_slice(), [
            ProgressUpdate::Step(Step::ReadingImage),
            ProgressUpdate::Step(Step::FindingSubject),
            ProgressUpdate::Subject(_),
            ProgressUpdate::Step(Step::RefiningEdges),
        ]), "{:?}", kinds(&updates));
    }
}
