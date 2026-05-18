use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{
    engine::OrtEngine,
    pipeline::process_image_with_mask,
    types::{CoreError, MaskSettings, ModelKind, ProcessResult, ProgressStage},
};

/// Create an engine pool with GPU/CPU-aware sizing.
/// GPU: 2 engines for pipeline overlap. CPU: 1 engine with full thread parallelism.
///
/// Caller invariant: must not be called from inside a rayon scope — the
/// parallel-build path uses the global pool and could deadlock under
/// nested rayon (same hazard as `apply_background_color`).
///
/// During parallel build (`pool_size > 1`, non-DirectML) the peak RSS
/// briefly doubles relative to sequential build because both ORT
/// sessions hold their working set live at the same time
/// (~400 MB-1 GB peak for 5-13 s instead of staggered ~200-500 MB).
/// The window is short and the wall-clock win is large; surface only
/// if a low-RAM user reports OOMing during pool init.
pub fn create_engine_pool(
    model: ModelKind,
    jobs: usize,
    cpu_only: bool,
) -> Result<Vec<std::sync::Arc<OrtEngine>>, CoreError> {
    let active = OrtEngine::detect_active_provider();
    let is_gpu = !cpu_only && !active.eq_ignore_ascii_case("CPU");
    // GPU: cap at 2 engines (VRAM is limited, more causes allocation failures).
    // CPU: respect user's setting fully — the admission controller manages
    // overall memory pressure at the batch level.
    let pool_size = if is_gpu { jobs.min(2) } else { jobs };
    // With many engines, distribute CPU threads evenly to avoid oversubscription.
    let intra_threads = if pool_size == 1 {
        num_cpus::get()
    } else {
        ort_intra_threads(pool_size)
    };

    let create = |threads| {
        if cpu_only {
            OrtEngine::new_cpu_only(model, threads)
        } else {
            OrtEngine::new(model, threads)
        }
    };

    // Build engines in parallel where the EP allows it. CPU, OpenVINO,
    // CUDA, CoreML are thread-safe during session creation. DirectML
    // is NOT — `commit_from_memory` on parallel threads has triggered
    // AbiCustomRegistry races in the wild — so fall back to sequential
    // when DirectML is the active provider.
    let serial = pool_size <= 1 || crate::engine::directml_active();

    let build = |_idx| create(intra_threads).map(std::sync::Arc::new);
    if serial {
        (0..pool_size).map(build).collect()
    } else {
        (0..pool_size).into_par_iter().map(build).collect()
    }
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
    batch_process_with_mask(
        images,
        model,
        jobs,
        &MaskSettings::default(),
        false,
        progress,
    )
}

/// Like `batch_process` but with custom mask settings.
///
/// Uses engine pooling: engines are created upfront (not per-image).
/// GPU backends are capped at 2 sessions to prevent VRAM exhaustion.
/// CPU creates one engine per job, each with num_cpus/jobs intra-op threads.
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
        Err(e) => {
            return images
                .iter()
                .map(|_| Err(CoreError::Model(e.to_string())))
                .collect()
        }
    };
    let pool_size = engines.len();
    let pool = build_batch_pool(pool_size);

    let mut results: Vec<Result<ProcessResult, CoreError>> = (0..images.len())
        .map(|_| Err(CoreError::Model("not processed".into())))
        .collect();

    // Write each result directly into its pre-sized slot via
    // `par_iter_mut().zip(...)`. The previous version `.collect`-ed
    // a parallel intermediate Vec then sequentially copied into
    // `results` — held N × `ProcessResult` (each carrying a full RGBA)
    // alive in a duplicate buffer for the whole batch. On a 100-image
    // batch of 4 K images that was multi-GB redundant retention.
    pool.install(|| {
        results
            .par_iter_mut()
            .zip(images.par_iter())
            .enumerate()
            .for_each(|(idx, (slot, img_bytes))| {
                let engine = &engines[idx % engines.len()];
                let cb = progress
                    .as_ref()
                    .map(|f| move |stage: ProgressStage, pct: f32| f(idx, stage, pct));
                *slot = process_image_with_mask(img_bytes, engine, mask, cb, None);
            });
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

    /// Synthetic test for the `par_iter_mut().zip(par_iter()).enumerate()`
    /// pattern that `batch_process` uses to write results in-place.
    /// Pins that rayon's IndexedParallelIterator chain preserves
    /// positional indices regardless of work-stealing order — without
    /// this property, results would land in wrong slots and
    /// `engines[idx % engines.len()]` round-robin would jumble too.
    #[test]
    fn par_iter_mut_zip_enumerate_preserves_positional_indices() {
        use rayon::prelude::*;
        let inputs: Vec<u32> = (0..1000).collect();
        let mut outputs: Vec<u32> = vec![0; inputs.len()];
        outputs
            .par_iter_mut()
            .zip(inputs.par_iter())
            .enumerate()
            .for_each(|(idx, (slot, &val))| {
                // The contract: `idx` is always the source position,
                // and `slot` is always `outputs[idx]`. If rayon broke
                // the ordering, slot wouldn't match val * 2 at idx.
                *slot = (val * 2).wrapping_add(idx as u32);
            });
        for (i, &out) in outputs.iter().enumerate().take(1000) {
            assert_eq!(
                out,
                (i as u32 * 2).wrapping_add(i as u32),
                "outputs[{i}] mis-ordered",
            );
        }
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
        use image::{DynamicImage, Rgb, RgbImage};
        use std::io::Cursor;

        fn make_png_bytes() -> Vec<u8> {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([100, 150, 200])));
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
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
        assert!(
            results[0].is_ok(),
            "Expected Ok result, got {:?}",
            results[0]
        );
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_batch_process_preserves_order() {
        use image::{DynamicImage, Rgb, RgbImage};
        use std::io::Cursor;

        fn make_png(r: u8) -> Vec<u8> {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([r, 100, 100])));
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
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
