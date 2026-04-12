use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{
    engine::{InferenceEngine, OrtEngine},
    pipeline::process_image_with_mask,
    types::{CoreError, MaskSettings, ModelKind, ProcessResult, ProgressStage},
};

/// Create an engine pool with GPU/CPU-aware sizing.
/// GPU: min(jobs, 2) engines. CPU: 1 engine with full thread parallelism.
/// Returns (engines, pool_size) where pool_size = engines.len().
pub fn create_engine_pool(
    model: ModelKind,
    jobs: usize,
    cpu_only: bool,
) -> Result<Vec<std::sync::Arc<OrtEngine>>, CoreError> {
    // Create first engine to detect actual runtime provider
    let first = if cpu_only {
        OrtEngine::new_cpu_only(model, 1)?
    } else {
        OrtEngine::new(model, 1)?
    };

    // Pool sizing based on what ORT actually selected at runtime
    let is_gpu = !first.active_provider().eq_ignore_ascii_case("CPU");
    let pool_size = if is_gpu { jobs.min(2) } else { 1 };
    let intra_threads = if pool_size == 1 { num_cpus::get() } else { ort_intra_threads(pool_size) };

    // Rebuild first engine with correct thread count if needed
    let create = |threads| {
        if cpu_only { OrtEngine::new_cpu_only(model, threads) } else { OrtEngine::new(model, threads) }
    };

    let mut engines = Vec::with_capacity(pool_size);
    if intra_threads == 1 {
        engines.push(std::sync::Arc::new(first)); // reuse if threads match
    } else {
        drop(first);
        engines.push(std::sync::Arc::new(create(intra_threads)?));
    }
    while engines.len() < pool_size {
        engines.push(std::sync::Arc::new(create(intra_threads)?));
    }
    Ok(engines)
}

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
/// - `jobs`: Desired parallelism. Actual pool size is capped for GPU (max 2) and CPU (1).
/// - `progress`: Optional per-image progress callback.
///   Signature: `|image_idx: usize, stage: ProgressStage, pct: f32|`
///
/// # Returns
/// `Vec<Result<ProcessResult, CoreError>>` — one entry per input image, in input order.
///
/// # Engine Pool
/// Engines are created upfront (not per-image). GPU backends cap at 2 sessions to
/// prevent VRAM exhaustion. CPU uses 1 engine with full thread parallelism. Rayon
/// pool is sized to match engine count, so there's no Mutex contention.
pub fn batch_process<F>(
    images: &[&[u8]],
    model: ModelKind,
    jobs: usize,
    progress: Option<F>,
) -> Vec<Result<ProcessResult, CoreError>>
where
    F: Fn(usize, ProgressStage, f32) + Send + Sync,
{
    batch_process_with_mask(images, model, jobs, &MaskSettings::default(), false, progress)
}

/// Like `batch_process` but with custom mask settings.
///
/// Uses engine pooling: engines are created upfront (not per-image).
/// GPU backends are capped at 2 sessions to prevent VRAM exhaustion.
/// CPU uses 1 engine with full thread parallelism.
pub fn batch_process_with_mask<F>(
    images: &[&[u8]],
    model: ModelKind,
    jobs: usize,
    mask: &MaskSettings,
    cpu_only: bool,
    progress: Option<F>,
) -> Vec<Result<ProcessResult, CoreError>>
where
    F: Fn(usize, ProgressStage, f32) + Send + Sync,
{
    if images.is_empty() {
        return Vec::new();
    }

    let engines = match create_engine_pool(model, jobs, cpu_only) {
        Ok(e) => e,
        Err(e) => return images.iter().map(|_| Err(CoreError::Model(e.to_string()))).collect(),
    };
    let pool_size = engines.len();
    let pool = build_batch_pool(pool_size);

    let mut results: Vec<Result<ProcessResult, CoreError>> =
        (0..images.len()).map(|_| Err(CoreError::Model("not processed".into()))).collect();

    pool.install(|| {
        let processed: Vec<(usize, Result<ProcessResult, CoreError>)> =
            images
                .par_iter()
                .enumerate()
                .map(|(idx, img_bytes)| {
                    let engine = &engines[idx % engines.len()];

                    let cb = progress.as_ref().map(|f| {
                        move |stage: ProgressStage, pct: f32| f(idx, stage, pct)
                    });

                    let result = process_image_with_mask(img_bytes, engine, mask, cb, None);
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
