pub mod engine;
pub mod types;
// Stub module declarations — implemented in later plans
pub mod pipeline;
pub mod preprocess;
pub mod postprocess;
pub mod batch;
pub mod formats;

pub use engine::InferenceEngine;
pub use types::{
    CoreError, ModelKind, ProgressStage, ProcessResult,
    LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET,
};
