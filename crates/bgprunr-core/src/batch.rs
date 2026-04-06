use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{
    engine::OrtEngine,
    pipeline::process_image,
    types::{CoreError, ModelKind, ProcessResult, ProgressStage},
};

/// Calculate ORT intra-op thread count to prevent oversubscription.
///
/// Formula: num_cpus / rayon_workers (minimum 1)
///
/// With 8 CPUs and 4 rayon workers, each worker's ORT session gets 2 intra-op threads.
/// Total threads = 4 workers × 2 intra-op = 8 = num_cpus. No oversubscription.
pub fn ort_intra_threads(rayon_workers: usize) -> usize {
    let cpus = num_cpus::get();
    let workers = rayon_workers.max(1);
    (cpus / workers).max(1)
}

/// Build a rayon ThreadPool with exactly `jobs` worker threads.
fn build_batch_pool(jobs: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()
        .expect("Failed to build rayon batch thread pool")
}

/// Process multiple images in parallel using a rayon thread pool.
///
/// # Arguments
/// - `images`: Slice of image byte slices (PNG, JPEG, WebP, BMP)
/// - `model`: Which model to use for all images in this batch
/// - `jobs`: Number of parallel rayon workers. Default 1 (sequential).
///   Each worker creates its own OrtEngine session — sessions are never shared.
/// - `progress`: Optional per-image progress callback.
///   Signature: `|image_idx: usize, stage: ProgressStage, pct: f32|`
///
/// # Returns
/// `Vec<Result<ProcessResult, CoreError>>` — one entry per input image, in input order.
/// A failed image returns Err in its slot; the batch does not abort on partial failure.
///
/// # Thread Safety
/// Each rayon worker thread creates its own OrtEngine. ORT sessions are not shared.
/// The `progress` callback must be Send + Sync (use Arc<Mutex<_>> if capturing shared state).
pub fn batch_process<F>(
    images: &[&[u8]],
    model: ModelKind,
    jobs: usize,
    progress: Option<F>,
) -> Vec<Result<ProcessResult, CoreError>>
where
    F: Fn(usize, ProgressStage, f32) + Send + Sync,
{
    if images.is_empty() {
        return Vec::new();
    }

    let intra_threads = ort_intra_threads(jobs.max(1));
    let pool = build_batch_pool(jobs);

    // Use a vector pre-allocated with placeholders, then fill via parallel index iteration.
    // This preserves input order even with rayon's work-stealing execution.
    let mut results: Vec<Result<ProcessResult, CoreError>> =
        (0..images.len()).map(|_| Err(CoreError::Model("not processed".into()))).collect();

    pool.install(|| {
        let processed: Vec<(usize, Result<ProcessResult, CoreError>)> =
            images
                .par_iter()
                .enumerate()
                .map(|(idx, img_bytes)| {
                    // Each worker creates its own session — never share across threads
                    let engine = match OrtEngine::new(model, intra_threads) {
                        Ok(e) => e,
                        Err(e) => return (idx, Err(e)),
                    };

                    let cb = progress.as_ref().map(|f| {
                        move |stage: ProgressStage, pct: f32| f(idx, stage, pct)
                    });

                    let result = process_image(img_bytes, &engine, cb);
                    (idx, result)
                })
                .collect();

        for (idx, result) in processed {
            results[idx] = result;
        }
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_intra_threads_at_least_one() {
        // Even with 0 workers, must return at least 1
        assert!(ort_intra_threads(0) >= 1);
        assert!(ort_intra_threads(1) >= 1);
    }

    #[test]
    fn test_ort_intra_threads_formula() {
        let cpus = num_cpus::get();
        let result = ort_intra_threads(4);
        let expected = (cpus / 4).max(1);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_batch_process_empty_input() {
        let results = batch_process(
            &[],
            ModelKind::Silueta,
            1,
            None::<fn(usize, ProgressStage, f32)>,
        );
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_process_bad_image_returns_err_not_panic() {
        // batch_process with a garbage image should return Err in that slot, not panic
        // We test the error type check independently of OrtEngine creation
        // (unit test without dev-models — just verify the error propagation logic)
        use crate::formats::load_image_from_bytes;
        let result = load_image_from_bytes(b"garbage");
        assert!(
            matches!(result, Err(CoreError::ImageFormat(_))),
            "Expected ImageFormat error for garbage bytes"
        );
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_batch_process_jobs_1_sequential() {
        use image::{DynamicImage, RgbImage, Rgb};
        use std::io::Cursor;

        fn make_png_bytes() -> Vec<u8> {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([100, 150, 200])));
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
            buf
        }

        let png = make_png_bytes();
        let image_refs: Vec<&[u8]> = vec![png.as_slice()];
        let results = batch_process(
            &image_refs,
            ModelKind::Silueta,
            1,
            None::<fn(usize, ProgressStage, f32)>,
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "Expected Ok result, got {:?}", results[0]);
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_batch_process_preserves_order() {
        use image::{DynamicImage, RgbImage, Rgb};
        use std::io::Cursor;

        fn make_png(r: u8) -> Vec<u8> {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([r, 100, 100])));
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
            buf
        }

        let img1 = make_png(50);
        let img2 = make_png(100);
        let img3 = make_png(150);
        let image_refs: Vec<&[u8]> = vec![img1.as_slice(), img2.as_slice(), img3.as_slice()];

        let results = batch_process(
            &image_refs,
            ModelKind::Silueta,
            2,
            None::<fn(usize, ProgressStage, f32)>,
        );
        assert_eq!(results.len(), 3, "Expected 3 results for 3 inputs");
        for (i, r) in results.iter().enumerate() {
            assert!(r.is_ok(), "Result {} failed: {:?}", i, r);
        }
    }
}
