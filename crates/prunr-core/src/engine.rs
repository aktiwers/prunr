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
        let bytes = match model {
            ModelKind::Silueta => prunr_models::silueta_bytes(),
            ModelKind::U2net => prunr_models::u2net_bytes(),
            ModelKind::BiRefNetLite => prunr_models::birefnet_lite_bytes(),
        };
        Self::new_from_bytes(&bytes, intra_threads, model)
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Create a CPU-only engine. Instant — no GPU compilation.
    /// Used as fallback while GPU engine compiles in the background.
    pub fn new_cpu_only(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        let bytes = match model {
            ModelKind::Silueta => prunr_models::silueta_bytes(),
            ModelKind::U2net => prunr_models::u2net_bytes(),
            ModelKind::BiRefNetLite => prunr_models::birefnet_lite_bytes(),
        };
        let mut builder = Session::builder()
            .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))?
            .with_execution_providers([
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| CoreError::Inference(format!("ORT set execution providers failed: {e}")))?;
        let session = builder
            .commit_from_memory(&bytes)
            .map_err(|e| CoreError::Inference(format!("ORT session creation failed: {e}")))?;
        Ok(Self {
            session: Mutex::new(session),
            provider_name: "CPU".to_string(),
            model_kind: model,
        })
    }

    fn new_from_bytes(model_bytes: &[u8], intra_threads: usize, model: ModelKind) -> Result<Self, CoreError> {
        let mut builder = Session::builder()
            .map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))?
            .with_execution_providers([
                #[cfg(all(feature = "cuda", not(target_os = "macos")))]
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                #[cfg(target_os = "macos")]
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                #[cfg(windows)]
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| CoreError::Inference(format!("ORT set execution providers failed: {e}")))?;

        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|e| CoreError::Inference(format!("ORT session creation failed: {e}")))?;

        // Determine which EP ORT selected.
        // ORT 2.0-rc.12 does not expose a direct "active EP" query API.
        // We infer it from compile-time feature flags. ORT logs the selected EP to stderr
        // at session init (visible in development). This matches rembg's approach.
        let provider_name = Self::detect_active_provider();

        Ok(Self {
            session: Mutex::new(session),
            provider_name,
            model_kind: model,
        })
    }

    /// Infer the active provider name from compile-time feature flags.
    /// On a standard Linux build without CUDA feature, returns "CPU".
    pub fn detect_active_provider() -> String {
        #[cfg(target_os = "macos")]
        { return "CoreML".to_string(); }

        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        { return "CUDA".to_string(); }

        #[cfg(windows)]
        { return "DirectML".to_string(); }

        "CPU".to_string()
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
