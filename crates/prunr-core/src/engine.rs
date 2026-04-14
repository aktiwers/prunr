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
    fn new_with_fallback(model: ModelKind, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        let optimized = Self::model_bytes_for_backend(model, cpu_only);
        let fp32 = Self::model_bytes(model);
        if optimized.len() != fp32.len() {
            if let Ok(engine) = Self::build_session(&optimized, intra_threads, model, cpu_only) {
                return Ok(engine);
            }
        }
        Self::build_session(&fp32, intra_threads, model, cpu_only)
    }

    fn model_bytes(model: ModelKind) -> Vec<u8> {
        match model {
            ModelKind::Silueta => prunr_models::silueta_bytes(),
            ModelKind::U2net => prunr_models::u2net_bytes(),
            ModelKind::BiRefNetLite => prunr_models::birefnet_lite_bytes(),
        }
    }

    /// Select the best model variant for the active backend.
    /// Prefers FP16 on CUDA/DirectML GPUs, INT8 on CPU, FP32 on macOS CoreML.
    ///
    /// macOS uses FP32 because CoreML silently converts to FP16 internally on Apple Silicon.
    /// Feeding it our FP16 variant stacks two conversions and the accumulated precision
    /// loss can collapse the segmentation mask to near-zero — producing a fully
    /// transparent output (image looks "completely removed").
    fn model_bytes_for_backend(model: ModelKind, cpu_only: bool) -> Vec<u8> {
        #[cfg(not(target_os = "macos"))]
        let pm = match model {
            ModelKind::Silueta => prunr_models::Model::Silueta,
            ModelKind::U2net => prunr_models::Model::U2net,
            ModelKind::BiRefNetLite => prunr_models::Model::BiRefNetLite,
        };
        if cpu_only {
            #[cfg(not(target_os = "macos"))]
            {
                prunr_models::model_int8_bytes(pm)
                    .unwrap_or_else(|| Self::model_bytes(model))
            }
            #[cfg(target_os = "macos")]
            {
                Self::model_bytes(model)
            }
        } else {
            #[cfg(not(target_os = "macos"))]
            {
                prunr_models::model_fp16_bytes(pm)
                    .unwrap_or_else(|| Self::model_bytes(model))
            }
            #[cfg(target_os = "macos")]
            {
                Self::model_bytes(model)
            }
        }
    }

    fn build_session(model_bytes: &[u8], intra_threads: usize, model: ModelKind, cpu_only: bool) -> Result<Self, CoreError> {
        let mut builder = Session::builder()
            .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))?;

        // Register all available EPs — ORT tries them in order and silently
        // skips any that aren't available at runtime (e.g. no CUDA drivers).
        builder = if cpu_only {
            builder.with_execution_providers([
                CPUExecutionProvider::default().build(),
            ])
        } else {
            builder.with_execution_providers([
                #[cfg(not(target_os = "macos"))]
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                    .with_cuda_graph(true)
                    .with_tf32(true)
                    .build(),
                #[cfg(target_os = "macos")]
                ort::execution_providers::CoreMLExecutionProvider::default()
                    .with_model_cache_dir(Self::coreml_cache_dir())
                    .build(),
                #[cfg(windows)]
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
        }.map_err(|e| CoreError::Inference(format!("ORT set execution providers failed: {e}")))?;

        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|e| CoreError::Inference(format!("ORT session creation failed: {e}")))?;

        let provider_name = if cpu_only {
            "CPU".to_string()
        } else {
            Self::detect_active_provider()
        };

        Ok(Self {
            session: Mutex::new(session),
            provider_name,
            model_kind: model,
        })
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
