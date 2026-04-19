use std::sync::Mutex;

use crate::types::{CoreError, ModelKind};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
};

/// Trait for inference backends. Implemented by OrtEngine.
/// Send + Sync required because worker threads own the engine instance.
pub trait InferenceEngine: Send + Sync {
    /// Returns the name of the active execution provider (e.g., "CUDA", "CoreML", "CPU").
    fn active_provider(&self) -> &str;
}

/// ORT-backed inference engine. Holds one Session per model selection.
/// Create once per model; reuse across all images — never instantiate per-image.
///
/// The session is behind a Mutex because ort's `Session::run()` takes `&mut self`
/// internally (ORT updates state during inference). The Mutex enables shared `&OrtEngine`
/// references while still satisfying ort's mutability requirement.
pub struct OrtEngine {
    session: Mutex<Session>,
    provider_name: String,
    model_kind: ModelKind,
}

impl OrtEngine {
    /// Create a new OrtEngine for the given model.
    ///
    /// - `model`: Which ONNX model to load (Silueta = ~4MB fast, U2net = ~170MB quality)
    /// - `intra_threads`: ORT intra-op thread count. For batch use: num_cpus / rayon_workers.
    ///   Pass 1 for single-image use; the thread count is set at session creation time.
    ///
    /// Execution providers are registered in priority order: CUDA → CoreML → DirectML → CPU.
    /// ORT silently selects the first available EP. Call active_provider() after creation
    /// to confirm which EP was selected.
    pub fn new(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::new_with_fallback(model, intra_threads, false)
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Create a CPU-only engine. Instant — no GPU compilation.
    pub fn new_cpu_only(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::new_with_fallback(model, intra_threads, true)
    }

    /// Try optimized variant (FP16/INT8) first; fall back to FP32 if session creation fails.
    ///
    /// When `cpu_only=false`, we also fall back to CPU-only if the GPU EP's
    /// session creation crashes at init. Seen in the wild with DirectML on
    /// some Windows setups (AbiCustomRegistry exception during initialization);
    /// ORT bubbles the exception out of `commit_from_memory` rather than
    /// silently skipping the EP, so the CPU fallback in the EP list is never
    /// reached. Retrying with `cpu_only=true` gives us a working session.
    fn new_with_fallback(model: ModelKind, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        tracing::debug!(?model, intra_threads, cpu_only, "OrtEngine init");
        // Check if an optimized variant exists (loaded from filesystem, so Vec<u8>).
        if let Some(optimized) = Self::optimized_variant_bytes(model, cpu_only) {
            tracing::debug!(?model, variant_bytes = optimized.len(), "OrtEngine trying optimized variant");
            match Self::build_session(&optimized, intra_threads, model, cpu_only) {
                Ok(engine) => {
                    tracing::info!(?model, provider = %engine.provider_name, "OrtEngine ready (optimized variant)");
                    return Ok(engine);
                }
                Err(e) => {
                    tracing::warn!(?model, error = %e, "optimized variant failed — falling back to FP32");
                }
            }
        } else {
            tracing::debug!(?model, "no optimized variant on disk — using embedded FP32");
        }
        // Fall back to the embedded FP32 model (zero-copy &'static [u8]).
        let fp32 = Self::model_bytes(model);
        match Self::build_session(fp32, intra_threads, model, cpu_only) {
            Ok(engine) => {
                tracing::info!(?model, provider = %engine.provider_name, "OrtEngine ready (FP32)");
                Ok(engine)
            }
            Err(e) if !cpu_only => {
                tracing::warn!(?model, error = %e, "GPU session creation failed — retrying CPU-only");
                let engine = Self::build_session(fp32, intra_threads, model, true)?;
                tracing::info!(?model, provider = %engine.provider_name, "OrtEngine ready (FP32, CPU fallback)");
                Ok(engine)
            }
            Err(e) => Err(e),
        }
    }

    /// Return the embedded FP32 model bytes. Zero-copy — borrows from static cache.
    fn model_bytes(model: ModelKind) -> &'static [u8] {
        match model {
            ModelKind::Silueta => prunr_models::silueta_bytes(),
            ModelKind::U2net => prunr_models::u2net_bytes(),
            ModelKind::BiRefNetLite => prunr_models::birefnet_lite_bytes(),
        }
    }

    /// Try to load an optimized model variant from the filesystem.
    /// Returns None if no variant is available (falls back to embedded FP32).
    fn optimized_variant_bytes(model: ModelKind, cpu_only: bool) -> Option<Vec<u8>> {
        #[cfg(target_os = "macos")]
        { return None; } // macOS uses FP32 — CoreML does its own FP16 conversion

        #[cfg(not(target_os = "macos"))]
        {
            let pm = match model {
                ModelKind::Silueta => prunr_models::Model::Silueta,
                ModelKind::U2net => prunr_models::Model::U2net,
                ModelKind::BiRefNetLite => prunr_models::Model::BiRefNetLite,
            };
            if cpu_only {
                prunr_models::model_int8_bytes(pm)
            } else {
                prunr_models::model_fp16_bytes(pm)
            }
        }
    }

    fn build_session(model_bytes: &[u8], intra_threads: usize, model: ModelKind, cpu_only: bool) -> Result<Self, CoreError> {
        // CPU-only path: straight shot.
        if cpu_only {
            let session = Self::builder_with_base(intra_threads)?
                .with_execution_providers([
                    CPUExecutionProvider::default()
                        .with_arena_allocator(false) // lower memory baseline; subprocess handles OOM
                        .build(),
                ])
                .map_err(|e| CoreError::Inference(format!("ORT set CPU EP failed: {e}")))?
                .commit_from_memory(model_bytes)
                .map_err(|e| CoreError::Inference(format!("ORT session creation failed (CPU): {e}")))?;
            return Ok(Self {
                session: Mutex::new(session),
                provider_name: "CPU".to_string(),
                model_kind: model,
            });
        }

        // GPU path: try platform GPU EPs one by one so we know which succeeded
        // (registering them as a list hides which EP was actually selected and
        // lets a crashing EP abort session creation before the CPU fallback is
        // reached — exactly the DirectML AbiCustomRegistry failure seen in the
        // wild). Fall through on error; the caller retries with cpu_only=true
        // if all GPU EPs fail.
        let gpu_eps: &[&str] = {
            #[cfg(target_os = "macos")]
            { &["CoreML"] }
            #[cfg(target_os = "linux")]
            { &["CUDA"] }
            #[cfg(windows)]
            { &["CUDA", "DirectML"] }
            #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
            { &[] }
        };

        let mut last_err: Option<CoreError> = None;
        for ep_name in gpu_eps {
            let builder = Self::builder_with_base(intra_threads)?;
            let res = match *ep_name {
                #[cfg(not(target_os = "macos"))]
                "CUDA" => builder.with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .with_device_id(0)
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                        .with_cuda_graph(true)
                        .with_tf32(true)
                        .build(),
                ]),
                #[cfg(target_os = "macos")]
                "CoreML" => builder.with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_model_cache_dir(Self::coreml_cache_dir())
                        .build(),
                ]),
                #[cfg(windows)]
                "DirectML" => builder.with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default().build(),
                ]),
                _ => continue,
            };

            let mut built = match res {
                Ok(b) => b,
                Err(e) => {
                    let err = CoreError::Inference(format!("ORT register {ep_name} EP failed: {e}"));
                    tracing::warn!(?model, ep = %ep_name, error = %err, "GPU EP register failed");
                    last_err = Some(err);
                    continue;
                }
            };

            match built.commit_from_memory(model_bytes) {
                Ok(session) => {
                    tracing::debug!(?model, ep = %ep_name, "GPU session committed");
                    return Ok(Self {
                        session: Mutex::new(session),
                        provider_name: (*ep_name).to_string(),
                        model_kind: model,
                    });
                }
                Err(e) => {
                    let err = CoreError::Inference(format!("ORT session creation failed ({ep_name}): {e}"));
                    tracing::warn!(?model, ep = %ep_name, error = %err, "GPU session creation failed — trying next EP");
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            CoreError::Inference("No GPU EPs available on this platform".to_string())
        }))
    }

    fn builder_with_base(intra_threads: usize) -> Result<ort::session::builder::SessionBuilder, CoreError> {
        Session::builder()
            .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))
    }

    /// CoreML compiled model cache directory.
    /// Ensures the ~2-5 min compilation only happens once ever per model.
    #[cfg(target_os = "macos")]
    fn coreml_cache_dir() -> String {
        if let Some(cache) = dirs::cache_dir() {
            let path = cache.join("prunr").join("coreml");
            let _ = std::fs::create_dir_all(&path);
            path.to_string_lossy().into_owned()
        } else {
            "/tmp/prunr-coreml-cache".to_string()
        }
    }

    /// Detect the runtime provider (cached — runs once per process).
    pub fn detect_active_provider() -> String {
        static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        CACHED.get_or_init(|| {
            #[cfg(target_os = "macos")]
            { return "CoreML".to_string(); }

            #[cfg(all(not(target_os = "macos"), not(windows)))]
            {
                if std::process::Command::new("nvidia-smi")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map_or(false, |s| s.success())
                {
                    return "CUDA".to_string();
                }
                return "CPU".to_string();
            }

            #[cfg(windows)]
            { return "DirectML".to_string(); }

            #[allow(unreachable_code)]
            "CPU".to_string()
        }).clone()
    }

    /// Lock the underlying ORT Session for inference.
    /// Used by pipeline.rs — not part of the public trait API.
    /// ORT requires &mut Session for run(); Mutex provides interior mutability.
    pub(crate) fn with_session<T, F>(&self, f: F) -> Result<T, CoreError>
    where
        F: FnOnce(&mut Session) -> Result<T, CoreError>,
    {
        let mut session = self
            .session
            .lock()
            .map_err(|e| CoreError::Inference(format!("Session mutex poisoned: {e}")))?;
        f(&mut session)
    }
}

impl InferenceEngine for OrtEngine {
    fn active_provider(&self) -> &str {
        &self.provider_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEngine;
    impl InferenceEngine for MockEngine {
        fn active_provider(&self) -> &str { "CPU" }
    }

    #[test]
    fn test_inference_engine_trait_is_object_safe() {
        let engine: Box<dyn InferenceEngine> = Box::new(MockEngine);
        assert_eq!(engine.active_provider(), "CPU");
    }

    // Integration tests require dev-models feature and downloaded models.
    // Run with: cargo test -p prunr-core --features dev-models
    #[cfg(feature = "dev-models")]
    #[test]
    fn test_ort_engine_silueta_active_provider_non_empty() {
        let engine = OrtEngine::new(ModelKind::Silueta, 1)
            .expect("OrtEngine::new should succeed with dev-models and downloaded models");
        let provider = engine.active_provider();
        assert!(!provider.is_empty(), "active_provider() returned empty string");
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_ort_engine_u2net_creates_session() {
        let engine = OrtEngine::new(ModelKind::U2net, 1)
            .expect("OrtEngine::new(U2net) should succeed with dev-models");
        let provider = engine.active_provider();
        assert!(!provider.is_empty());
    }
}
