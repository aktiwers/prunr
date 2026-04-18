pub mod engine;
pub mod types;
pub mod pipeline;
pub mod preprocess;
pub mod postprocess;
pub mod guided_filter;
pub mod batch;
pub mod formats;
pub mod edge;
pub mod recipe;

pub use engine::{InferenceEngine, OrtEngine};
pub use recipe::{ProcessingRecipe, InferenceRecipe, EdgeRecipe, MaskRecipe, CompositeRecipe, RequiredTier, resolve_tier};
pub use types::{
    CoreError, ModelKind, ProgressStage, ProcessResult, MaskSettings, EdgeSettings, InferenceResult,
    LineMode, LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET,
};
pub use pipeline::{process_image, process_image_from_decoded, process_image_unchecked, process_image_with_mask, infer_only};
pub use postprocess::{tensor_to_mask, apply_mask, postprocess_from_flat};
pub use batch::{batch_process, batch_process_with_mask, create_engine_pool};
pub use formats::{load_image_from_path, load_image_from_bytes, check_large_image, downscale_image, encode_rgba_png, apply_background_color};
pub use edge::{EdgeEngine, finalize_edges, tensor_to_edge_mask, compose_edges};
