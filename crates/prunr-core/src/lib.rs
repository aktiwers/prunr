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
pub mod brush;
pub mod math;

pub use engine::{InferenceEngine, OrtEngine};
pub use recipe::{ProcessingRecipe, InferenceRecipe, EdgeRecipe, MaskRecipe, CompositeRecipe, RequiredTier, resolve_tier};
pub use types::{
    CoreError, ModelKind, ProgressStage, ProcessResult, MaskSettings, EdgeSettings, EdgeScale,
    InferenceResult, LineMode, LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET,
};
pub use pipeline::{process_image, process_image_from_decoded, process_image_unchecked, process_image_with_mask, infer_only};
pub use postprocess::{tensor_to_mask, tensor_to_mask_from_flat, apply_mask, postprocess_from_flat};
pub use batch::{batch_process, batch_process_with_mask, create_engine_pool};
pub use formats::{load_image_from_path, load_image_from_bytes, check_large_image, downscale_image, encode_rgba_png, encode_gray_png, apply_background_color};
pub use edge::{EdgeEngine, EdgeInferenceResult, EDGE_SCALE_COUNT, finalize_edges, tensor_to_edge_mask, compose_edges, compose_edges_styled, compose_edges_dual_styled, compose_subject_outline};
pub use types::{ComposeMode, LineStyle, FillStyle, BgEffect, ChannelSwapVariant, InputTransform};
pub use edge::apply_input_transform;
pub use postprocess::{apply_fill_style, apply_bg_effect};
