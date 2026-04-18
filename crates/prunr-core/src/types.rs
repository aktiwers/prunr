use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Image format error: {0}")]
    ImageFormat(String),
    #[error("Image too large: {width}x{height} exceeds {limit}px limit")]
    LargeImage { width: u32, height: u32, limit: u32 },
    #[error("Processing cancelled")]
    Cancelled,
}

impl From<image::ImageError> for CoreError {
    fn from(e: image::ImageError) -> Self {
        CoreError::ImageFormat(e.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelKind {
    Silueta,
    U2net,
    BiRefNetLite,
}

/// Line extraction mode. Determines which engines the pipeline uses.
///
/// `#[serde(alias = ...)]` preserves backward compatibility with v1 settings
/// files that used the older `LinesOnly` / `AfterBgRemoval` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LineMode {
    /// No line extraction
    Off,
    /// DexiNed on original image; segmentation skipped.
    #[serde(alias = "LinesOnly")]
    EdgesOnly,
    /// Segmentation first, then DexiNed — edges only within subject, body transparent.
    #[serde(alias = "AfterBgRemoval")]
    SubjectOutline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProgressStage {
    Decode,
    Resize,
    Normalize,
    Infer,
    Postprocess,
    Alpha,
    /// Loading and compiling the AI model (can be slow on first run with GPU backends)
    LoadingModel,
    /// GPU is still compiling; processing on CPU in the meantime
    LoadingModelCpuFallback,
}

#[derive(Debug)]
pub struct ProcessResult {
    /// Raw RGBA pixels of the output image with background removed
    pub rgba_image: image::RgbaImage,
    /// Name of the execution provider used (e.g., "CUDA", "CPU")
    pub active_provider: String,
}

/// Controls for post-processing the AI-generated mask before applying it as alpha.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaskSettings {
    /// Gamma curve applied to the mask. >1.0 = more aggressive removal, <1.0 = gentler.
    pub gamma: f32,
    /// Optional binary threshold (0.0–1.0). When set, alpha below this becomes 0, above becomes 255.
    pub threshold: Option<f32>,
    /// Edge hardness: >0 erodes (shrinks) the mask, <0 dilates (expands) it. In pixels.
    pub edge_shift: f32,
    /// Refine mask edges using guided filter (color-aware edge refinement).
    pub refine_edges: bool,
    /// Guided filter window radius in pixels. Smaller = crisper edges, larger = softer blend. Only used when refine_edges.
    pub guided_radius: u32,
    /// Guided filter regularization. Smaller = preserve edges from guide, larger = smoother. Only used when refine_edges.
    pub guided_epsilon: f32,
    /// Gaussian blur sigma (pixels). Color-agnostic edge softening.
    pub feather: f32,
}

impl Default for MaskSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            threshold: None,
            edge_shift: 0.0,
            refine_edges: false,
            guided_radius: 8,
            guided_epsilon: 1e-4,
            feather: 0.0,
        }
    }
}

/// Controls for the edge-detection postprocess (`finalize_edges`).
/// `line_mode` is kept separate — it picks whether edges run at all.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeSettings {
    /// DexiNed sensitivity, 0.0-1.0.
    pub line_strength: f32,
    /// Solid color override for edges. `None` = preserve original RGB.
    pub solid_line_color: Option<[u8; 3]>,
    /// Dilate edge mask by N pixels after threshold — thickens thin lines.
    pub edge_thickness: u32,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            line_strength: 0.5,
            solid_line_color: None,
            edge_thickness: 0,
        }
    }
}

/// Raw inference result (Tier 1 output) — the model tensor before postprocessing.
pub struct InferenceResult {
    /// Raw f32 tensor data in row-major order [1, 1, H, W].
    pub tensor_data: Vec<f32>,
    pub tensor_height: usize,
    pub tensor_width: usize,
    pub model: ModelKind,
    pub active_provider: String,
}

pub const LARGE_IMAGE_LIMIT: u32 = 8000;
pub const DOWNSCALE_TARGET: u32 = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_model_variant() {
        let err = CoreError::Model("test error".to_string());
        assert_eq!(err.to_string(), "Model error: test error");
    }

    #[test]
    fn test_inference_error_display() {
        let err = CoreError::Inference("ort failed".into());
        assert_eq!(err.to_string(), "Inference error: ort failed");
    }

    #[test]
    fn test_image_format_error_display() {
        let err = CoreError::ImageFormat("bad format".into());
        assert_eq!(err.to_string(), "Image format error: bad format");
    }

    #[test]
    fn test_large_image_error_display() {
        let err = CoreError::LargeImage {
            width: 9000,
            height: 5000,
            limit: 8000,
        };
        let msg = err.to_string();
        assert!(msg.contains("9000"), "Expected message to contain '9000', got: {}", msg);
        assert!(msg.contains("8000"), "Expected message to contain '8000', got: {}", msg);
    }

    #[test]
    fn test_model_kind_variants() {
        let silueta = ModelKind::Silueta;
        let u2net = ModelKind::U2net;
        // Both exist and implement Debug + Clone
        let _ = format!("{:?}", silueta);
        let _ = format!("{:?}", u2net);
        let _cloned = silueta.clone();
        let _cloned2 = u2net.clone();
        assert_ne!(silueta, u2net);
    }

    #[test]
    fn test_progress_stage_variants() {
        let stages = [
            ProgressStage::Decode,
            ProgressStage::Resize,
            ProgressStage::Normalize,
            ProgressStage::Infer,
            ProgressStage::Postprocess,
            ProgressStage::Alpha,
            ProgressStage::LoadingModel,
            ProgressStage::LoadingModelCpuFallback,
        ];
        // All variants compile and implement Debug + Clone + Copy
        for stage in &stages {
            let _ = format!("{:?}", stage);
            let _cloned = stage.clone();
            let _copied: ProgressStage = *stage;
        }
        assert_eq!(stages.len(), 8);
    }

    #[test]
    fn test_process_result_fields() {
        let result = ProcessResult {
            rgba_image: image::RgbaImage::new(1, 1),
            active_provider: "CPU".to_string(),
        };
        assert_eq!(result.rgba_image.width(), 1);
        assert_eq!(result.active_provider, "CPU");
    }
}
