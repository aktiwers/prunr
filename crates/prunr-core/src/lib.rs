pub mod blur;
pub mod cache;
pub mod color;
pub mod colorize;
pub mod depth;
pub mod ort_runtime;
pub mod engine;
pub mod env;
pub mod types;
pub mod pipeline;
pub mod preprocess;
pub mod postprocess;
pub mod guided_filter;
pub mod morphology;
pub mod decorate;
pub mod batch;
pub mod formats;
pub mod edge;
pub mod recipe;
pub mod brush;
pub mod math;
pub mod inpaint;
pub mod face;
pub mod inpaint_sd;
pub mod inpaint_blend;
pub mod ep_compat;
pub mod plain;
mod ep_coverage;
pub mod sysmem;
pub mod card_memory;
pub mod memory;
pub mod upscale;
pub mod denoise;
pub mod selection;
pub mod sam;
pub mod sam_session;
pub mod progress;
pub mod subject;

/// How long an unused model session stays loaded before its memory goes
/// back to the OS: "paint again" or "Process again" within it finds the
/// model ready; a user who moved on gets the RAM back.
pub const IDLE_RELEASE: std::time::Duration = std::time::Duration::from_secs(300);

pub use engine::OrtEngine;
pub use recipe::{ProcessingRecipe, InferenceRecipe, EdgeRecipe, MaskRecipe, CompositeRecipe, UpscaleRecipe, ColorizeRecipe, OutputScale, RequiredTier, resolve_tier, DEFAULT_SHARPEN};
pub use types::{
    CoreError, ModelKind, ProcessResult, MaskSettings, EdgeSettings, EdgeScale,
    InferenceResult, LineMode, LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET,
};
pub use pipeline::{process_image, process_image_from_decoded, process_image_with_mask, room_for_photo, infer_only, infer_after_first_look};
pub use postprocess::{tensor_to_mask, tensor_to_mask_from_flat, apply_mask, postprocess_from_flat, cut_for_lines, mask_from_flat, finish_with_mask, PostprocessOpts};
pub use batch::{batch_process, batch_process_with_mask, create_engine_pool};
pub use formats::{ExportFormat, load_image_from_path, load_image_from_bytes, load_image_to_fit_from_bytes, image_dimensions_from_bytes, image_dimensions_from_path, exceeds_large_image_limit, downscale_image, encode_rgba_png, encode_rgba_png_into, rgba_view, encode_gray_png_into, encode_gray_png_kept_into, apply_background_color, apply_background_image};
pub use edge::{EdgeEngine, EdgeInferenceResult, EdgeInput, scales_from_same_runs, EDGE_SCALE_COUNT, edge_scale_dims, tensor_to_edge_mask, edge_plane, thickened_edges, compose_edges_styled, compose_edges_dual_styled, compose_subject_outline, compose_whole_photo, outline_subject};
pub use types::{ComposeMode, LineStyle, SubjectEffect, BgEffect, BgImageFit, InputTransform, Decoration, ShadowKind, RefineGuide};
pub use edge::apply_input_transform;
pub use postprocess::{apply_bg_effect, apply_subject_effect, effect_backdrop};
pub use selection::{MaskArtifact, SelectionError};
pub use sam::{SamEmbedding, SamDecoderOutput, decode_to_mask_artifact, MaskReading, SAM_ENCODER_INPUT, SAM_MASK_RESOLUTION};
pub use progress::{Progress, ProgressSink, ProgressUpdate, Step, Unit, TilePreview, TileRect};
pub use subject::{subject_hint, SubjectHint};
pub use sam::prompt::{SamPrompt, PromptError, build_click_prompt, build_stroke_prompt, MAX_STROKE_POINTS};
