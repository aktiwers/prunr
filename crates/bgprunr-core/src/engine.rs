#[allow(unused_imports)]
use crate::types::CoreError;

/// Trait for inference backends. Implemented by the ORT backend in Phase 2.
/// Send + Sync required because the worker thread owns the engine.
pub trait InferenceEngine: Send + Sync {
    /// Returns the name of the active execution provider (e.g., "CUDA", "CoreML", "CPU").
    fn active_provider(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEngine;
    impl InferenceEngine for MockEngine {
        fn active_provider(&self) -> &str {
            "CPU"
        }
    }

    #[test]
    fn test_inference_engine_trait_is_object_safe() {
        let engine: Box<dyn InferenceEngine> = Box::new(MockEngine);
        assert_eq!(engine.active_provider(), "CPU");
    }
}
