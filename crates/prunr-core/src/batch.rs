use rayon::prelude::*;

use crate::{
    engine::OrtEngine,
    types::{CoreError, ModelKind},
};

/// How many engines a batch builds for `jobs` on `provider`, the one the
/// model runs on (the CPU under "CPU only"): at most two on a GPU (VRAM),
/// one on the host CPU for a model whose single run already fills every
/// core, else one per job. Where the model's sessions really run decides
/// (`model_on_host_cpu`): OpenVINO's processor, or a provider the table
/// keeps it off, is the host CPU; OpenVINO's pool keeps the cap of two,
/// which is what was measured.
pub fn engine_pool_size(model: ModelKind, jobs: usize, provider: &str) -> usize {
    let jobs = jobs.max(1);
    let on_cpu = crate::engine::model_on_host_cpu(model.model_id(), provider);
    if on_cpu && !model.parallel_jobs_pay_on_cpu() {
        1
    } else if !provider.eq_ignore_ascii_case(crate::engine::CPU_NAME) {
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
    let id: prunr_models::ModelId = model.into();
    let on = crate::memory::provider_for(id, cpu_only);
    let pool_size = engine_pool_size(model, jobs, on);
    let pool_size = crate::memory::engines_that_fit(id, on, pool_size, crate::memory::card_free_mb(on));
    let intra_threads = if pool_size == 1 {
        num_cpus::get()
    } else {
        ort_intra_threads(pool_size)
    };

    let serial = pool_size <= 1 || crate::engine::engines_build_one_at_a_time();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_heavy_model_gets_one_engine_on_the_host_cpu_and_a_gpu_at_most_two() {
        for (model, jobs, provider, engines) in [
            (ModelKind::BiRefNetLite, 4, "CPU", 1),
            (ModelKind::BiRefNetLite, 4, "OpenVINO", 1),
            (ModelKind::BiRefNetLite, 4, "CUDA", 2),
            (ModelKind::BiRefNetLite, 1, "CUDA", 1),
            (ModelKind::Silueta, 4, "CPU", 4),
            (ModelKind::Silueta, 4, "OpenVINO", 2),
            (ModelKind::Silueta, 4, "CUDA", 2),
            (ModelKind::U2net, 3, "CPU", 3),
            (ModelKind::U2net, 0, "CPU", 1),
        ] {
            assert_eq!(engine_pool_size(model, jobs, provider), engines, "{model:?} × {jobs} on {provider}");
        }
    }

    #[test]
    fn test_ort_intra_threads_formula() {
        // Even with 0 workers, at least 1.
        assert!(ort_intra_threads(0) >= 1);
        assert!(ort_intra_threads(1) >= 1);
        assert_eq!(ort_intra_threads(4), (num_cpus::get() / 4).max(1));
    }
}
