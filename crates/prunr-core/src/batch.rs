use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{
    engine::OrtEngine,
    pipeline::process_image_with_mask,
    progress::Progress,
    types::{CoreError, MaskSettings, ModelKind, ProcessResult},
};

/// How many engines a batch builds for `jobs` on `provider`: at most two
/// on a GPU (VRAM), one on the host CPU for a model whose single run
/// already fills every core, else one per job. OpenVINO runs on the host
/// CPU here, so the one-engine rule covers it; its pool keeps the cap of
/// two, which is what was measured.
pub fn engine_pool_size(model: ModelKind, jobs: usize, cpu_only: bool, provider: &str) -> usize {
    let jobs = jobs.max(1);
    let on_cpu = cpu_only || crate::engine::runs_on_host_cpu(provider);
    if on_cpu && !model.parallel_jobs_pay_on_cpu() {
        1
    } else if !cpu_only && !provider.eq_ignore_ascii_case(crate::engine::CPU_NAME) {
        jobs.min(2)
    } else {
        jobs
    }
}

/// Create an engine pool sized by `engine_pool_size`.
///
/// Caller invariant: must not be called from inside a rayon scope — the
/// parallel-build path uses the global pool and could deadlock under
/// nested rayon.
///
/// A parallel build briefly doubles peak RSS (~400 MB-1 GB for 5-13 s
/// instead of staggered ~200-500 MB), accepted for the wall-clock win.
pub fn create_engine_pool(
    model: ModelKind,
    jobs: usize,
    cpu_only: bool,
) -> Result<Vec<std::sync::Arc<OrtEngine>>, CoreError> {
    let pool_size = engine_pool_size(model, jobs, cpu_only, &OrtEngine::detect_active_provider());
    let intra_threads = if pool_size == 1 {
        num_cpus::get()
    } else {
        ort_intra_threads(pool_size)
    };

    // DirectML session creation is not thread-safe: parallel
    // `commit_from_memory` raced in AbiCustomRegistry.
    let serial = pool_size <= 1 || crate::engine::directml_active();

    // Held for the whole pool: a bundled model unpacks once for every
    // engine, built in parallel or one after another, and is freed after.
    let _weights = prunr_models::resolve_bytes(model.into());
    let build = |_idx| OrtEngine::with_threads(model, intra_threads, cpu_only).map(std::sync::Arc::new);
    if serial {
        (0..pool_size).map(build).collect()
    } else {
        (0..pool_size).into_par_iter().map(build).collect()
    }
}

/// ORT intra-op threads per rayon worker, so workers × threads stays at
/// num_cpus (minimum 1).
pub fn ort_intra_threads(rayon_workers: usize) -> usize {
    let cpus = num_cpus::get();
    let workers = rayon_workers.max(1);
    (cpus / workers).max(1)
}

fn build_batch_pool(jobs: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()
        .expect("Failed to build rayon batch thread pool")
}

/// Process multiple images in parallel using a rayon thread pool.
///
/// # Arguments
/// - `images`: image bytes (PNG, JPEG, WebP, BMP)
/// - `jobs`: Desired parallelism; the pool is sized by `engine_pool_size`.
/// - `progress_for`: the progress handle for image `idx`.
///
/// Returns one entry per input image, in input order. The rayon pool is
/// sized to the engine count, so engines see no Mutex contention.
pub fn batch_process<F>(
    images: &[&[u8]],
    model: ModelKind,
    jobs: usize,
    progress_for: F,
) -> Vec<Result<ProcessResult, CoreError>>
where
    F: Fn(usize) -> Progress + Send + Sync,
{
    batch_process_with_mask(images, model, jobs, &MaskSettings::default(), false, progress_for)
}

/// Like `batch_process` but with custom mask settings.
pub fn batch_process_with_mask<F>(
    images: &[&[u8]],
    model: ModelKind,
    jobs: usize,
    mask: &MaskSettings,
    cpu_only: bool,
    progress_for: F,
) -> Vec<Result<ProcessResult, CoreError>>
where
    F: Fn(usize) -> Progress + Send + Sync,
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

    // Results go straight into their slots: no second N × ProcessResult
    // buffer, each holding a full RGBA.
    pool.install(|| {
        results
            .par_iter_mut()
            .zip(images.par_iter())
            .enumerate()
            .for_each(|(idx, (slot, img_bytes))| {
                let engine = &engines[idx % engines.len()];
                *slot = process_image_with_mask(img_bytes, engine, mask, &progress_for(idx));
            });
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_heavy_model_gets_one_engine_on_the_host_cpu_and_a_gpu_at_most_two() {
        for (model, jobs, cpu_only, provider, engines) in [
            (ModelKind::BiRefNetLite, 4, false, "CPU", 1),
            (ModelKind::BiRefNetLite, 4, false, "OpenVINO", 1),
            (ModelKind::BiRefNetLite, 4, true, "CUDA", 1),
            (ModelKind::BiRefNetLite, 4, false, "CUDA", 2),
            (ModelKind::BiRefNetLite, 1, false, "CUDA", 1),
            (ModelKind::Silueta, 4, false, "CPU", 4),
            (ModelKind::Silueta, 4, true, "CUDA", 4),
            (ModelKind::Silueta, 4, false, "OpenVINO", 2),
            (ModelKind::Silueta, 4, false, "CUDA", 2),
            (ModelKind::U2net, 3, false, "CPU", 3),
            (ModelKind::U2net, 0, false, "CPU", 1),
        ] {
            assert_eq!(engine_pool_size(model, jobs, cpu_only, provider), engines, "{model:?} × {jobs} on {provider}, cpu_only {cpu_only}");
        }
    }

    #[test]
    fn test_ort_intra_threads_formula() {
        // Even with 0 workers, at least 1.
        assert!(ort_intra_threads(0) >= 1);
        assert!(ort_intra_threads(1) >= 1);
        assert_eq!(ort_intra_threads(4), (num_cpus::get() / 4).max(1));
    }

    #[test]
    fn test_batch_process_empty_input() {
        let results = batch_process(
            &[],
            ModelKind::Silueta,
            1,
            |_| Progress::none(),
        );
        assert!(results.is_empty());
    }
}
