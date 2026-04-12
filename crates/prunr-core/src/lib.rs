pub mod engine;
pub mod types;
pub mod pipeline;
pub mod preprocess;
pub mod postprocess;
pub mod guided_filter;
pub mod batch;
pub mod formats;
pub mod edge;

pub use engine::{InferenceEngine, OrtEngine};
pub use types::{
    CoreError, ModelKind, ProgressStage, ProcessResult, MaskSettings,
    LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET,
};
pub use pipeline::{process_image, process_image_unchecked, process_image_with_mask};
pub use batch::{batch_process, batch_process_with_mask, create_engine_pool};
pub use formats::{load_image_from_path, load_image_from_bytes, check_large_image, downscale_image, encode_rgba_png, apply_background_color};
pub use edge::EdgeEngine;
