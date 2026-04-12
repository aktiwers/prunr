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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Silueta,
    U2net,
    BiRefNetLite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy)]
pub struct MaskSettings {
    /// Gamma curve applied to the mask. >1.0 = more aggressive removal, <1.0 = gentler.
    pub gamma: f32,
    /// Optional binary threshold (0.0–1.0). When set, alpha below this becomes 0, above becomes 255.
    pub threshold: Option<f32>,
    /// Edge hardness: >0 erodes (shrinks) the mask, <0 dilates (expands) it. In pixels.
    pub edge_shift: f32,
    /// Refine mask edges using guided filter (color-aware edge refinement).
    pub refine_edges: bool,
}

impl Default for MaskSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            threshold: None,
            edge_shift: 0.0,
            refine_edges: false,
        }
    }
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
