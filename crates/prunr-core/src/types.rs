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
    /// How the subject RGB is transformed after masking and before compose.
    #[serde(default)]
    pub fill_style: FillStyle,
    /// Backdrop effect baked into the output where the mask was transparent.
    /// `None` preserves transparency (solid bg colour still renders via the
    /// GPU-rect fast path at display time).
    #[serde(default)]
    pub bg_effect: BgEffect,
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
            fill_style: FillStyle::default(),
            bg_effect: BgEffect::default(),
        }
    }
}

/// DexiNed output-scale selector. The model produces 6 side outputs
/// (`block0`..`block5`, fine → coarse) plus a fused `block_cat`. One inference
/// pass computes all 7; picking a scale just indexes into the outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EdgeScale {
    /// `block0` — finest, crispest micro-edges.
    Fine,
    /// `block3` — mid-scale; smoother transitions than Fine.
    Balanced,
    /// `block5` — coarsest side output; abstract outlines.
    Bold,
    /// `block_cat` — learned combination of all blocks; current default.
    Fused,
}

impl Default for EdgeScale {
    fn default() -> Self { Self::Fused }
}

impl std::fmt::Display for EdgeScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Fine => "fine",
            Self::Balanced => "balanced",
            Self::Bold => "bold",
            Self::Fused => "fused",
        })
    }
}

impl std::str::FromStr for EdgeScale {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fine" => Ok(Self::Fine),
            "balanced" => Ok(Self::Balanced),
            "bold" => Ok(Self::Bold),
            "fused" => Ok(Self::Fused),
            _ => Err(format!("unknown edge scale '{s}' (expected: fine, balanced, bold, fused)")),
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
    /// Which DexiNed output scale to read. Default `Fused` = current behaviour.
    pub edge_scale: EdgeScale,
    /// How the subject mask and edge mask combine in `LineMode::SubjectOutline`.
    /// Ignored in Off / EdgesOnly. Pure compose-time setting — changing it
    /// doesn't re-run inference.
    #[serde(default)]
    pub compose_mode: ComposeMode,
    /// How edge pixels are coloured (solid / gradient). Compose-time.
    #[serde(default)]
    pub line_style: LineStyle,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            line_strength: 0.5,
            solid_line_color: None,
            edge_thickness: 0,
            edge_scale: EdgeScale::Fused,
            compose_mode: ComposeMode::default(),
            line_style: LineStyle::default(),
        }
    }
}

/// How to combine the subject mask and edge mask when `LineMode::SubjectOutline`.
/// Each variant is a compose-time formula over the already-cached tensors, so
/// switching modes is instant in live preview — no re-inference.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum ComposeMode {
    /// Lines only, confined to the subject silhouette (`alpha = subject × edge`).
    /// Mask tweaks visibly adjust where lines fade. Transparent background.
    #[default]
    LinesOnly,
    /// Filled subject with lines drawn on top (`alpha = max(subject, edge)`).
    /// Edges outside the subject boundary show through; subject stays solid.
    SubjectFilled,
    /// Filled subject with lines CUT through it (`alpha = subject − edge`).
    /// Lines appear as transparent grooves — "engraved" look.
    Engraving,
    /// Faded subject with strong lines (`alpha ≈ 0.3 · subject + 0.8 · edge`).
    /// Ghostly see-through body with a sharp outline.
    Ghost,
    /// Lines in the background, subject stays transparent
    /// (`alpha = (255 − subject) × edge / 255`). Negative-space outlines.
    InverseMask,
}

impl std::fmt::Display for ComposeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ComposeMode::LinesOnly => "Lines only",
            ComposeMode::SubjectFilled => "Subject filled",
            ComposeMode::Engraving => "Engraving",
            ComposeMode::Ghost => "Ghost",
            ComposeMode::InverseMask => "Inverse mask",
        };
        f.write_str(s)
    }
}

impl ComposeMode {
    pub const ALL: &'static [Self] = &[
        Self::LinesOnly,
        Self::SubjectFilled,
        Self::Engraving,
        Self::Ghost,
        Self::InverseMask,
    ];
}

/// How line pixels are coloured in `LineMode::SubjectOutline` / `EdgesOnly`.
/// `Solid` reads the user's `solid_line_color` chip (or passes source RGB
/// through if that's unset). Gradient variants carry their own endpoints and
/// ignore `solid_line_color` — they paint at edge pixels only.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum LineStyle {
    /// Use `solid_line_color` (or source RGB if none).
    #[default]
    Solid,
    /// Vertical gradient — lerp between `top` (y=0) and `bottom` (y=H).
    GradientY { top: [u8; 3], bottom: [u8; 3] },
    /// Horizontal gradient — lerp between `left` (x=0) and `right` (x=W).
    GradientX { left: [u8; 3], right: [u8; 3] },
    /// Radial gradient — `center` (x, y) normalised 0..=255, `inner` at
    /// centre, `outer` at image-diagonal distance.
    RadialGradient { center: [u8; 2], inner: [u8; 3], outer: [u8; 3] },
    /// Hue rotates through the colour wheel along pixel index —
    /// `cycles` = how many full rotations across the image.
    Rainbow { cycles: u16 },
    /// Offset RGB channels of the edge — classic chromatic aberration.
    /// `offset` is pixels of R/B shift (G stays centred). 1..=8 feels good.
    Chromatic { offset: u32 },
    /// Per-edge-pixel hue noise. `amount` = jitter magnitude 0..=255 applied
    /// to a deterministic per-pixel hash — feels hand-drawn without dithering.
    Noise { amount: u8 },
    /// Two DexiNed scales layered — Fine at `fine_color` for micro-details,
    /// Bold at `bold_color` for structural edges. Both sets share
    /// `line_strength` and `edge_thickness` from the chip; colours are
    /// independent.
    DualScale { fine_color: [u8; 3], bold_color: [u8; 3] },
}

impl LineStyle {
    pub const ALL: &'static [Self] = &[
        LineStyle::Solid,
        LineStyle::GradientY { top: [255, 50, 50], bottom: [50, 50, 255] },
        LineStyle::GradientX { left: [255, 50, 50], right: [50, 50, 255] },
        LineStyle::RadialGradient { center: [128, 128], inner: [255, 200, 50], outer: [30, 20, 60] },
        LineStyle::Rainbow { cycles: 3 },
        LineStyle::Chromatic { offset: 3 },
        LineStyle::Noise { amount: 80 },
        LineStyle::DualScale { fine_color: [50, 180, 230], bold_color: [30, 20, 60] },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            LineStyle::Solid => "Solid",
            LineStyle::GradientY { .. } => "Vertical gradient",
            LineStyle::GradientX { .. } => "Horizontal gradient",
            LineStyle::RadialGradient { .. } => "Radial gradient",
            LineStyle::Rainbow { .. } => "Rainbow",
            LineStyle::Chromatic { .. } => "Chromatic",
            LineStyle::Noise { .. } => "Noise",
            LineStyle::DualScale { .. } => "Dual scale",
        }
    }
}

impl std::fmt::Display for LineStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// How the subject's RGB is transformed before compose. Applies to any mode
/// that produces a masked subject (`Off` or `SubjectOutline`). `EdgesOnly`
/// has no subject silhouette — this is a no-op there.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum FillStyle {
    /// Pass source RGB through unchanged.
    #[default]
    None,
    /// Luma grayscale.
    Desaturate,
    /// RGB negative.
    Invert,
    /// Remap luma: lerp between `dark` (black pixels) and `light` (white).
    Duotone { dark: [u8; 3], light: [u8; 3] },
    /// Warm brown monochrome — classic vintage photo.
    Sepia,
    /// Pure black/white at the luma threshold.
    Threshold { level: u8 },
    /// Quantise each RGB channel to N levels. N in 2..=8 gives cel-shaded flats.
    Posterize { levels: u8 },
    /// Invert channels above `pivot`. Partial negative / surreal look.
    Solarize { pivot: u8 },
    /// Rotate hue by `degrees` (-180..=180). Fast colour variation.
    HueShift { degrees: i16 },
    /// Saturation scale where `percent` = 100 is neutral. 0 = grayscale,
    /// >100 punchy, <100 muted. Clamped 0..=300.
    Saturate { percent: u16 },
    /// Desaturate everything OUTSIDE a hue range. `keep_hue` is 0..=359°,
    /// `tolerance` is the half-width (0..=180) — pixels within tolerance
    /// degrees of `keep_hue` keep their colour; everything else goes gray.
    ColorSplash { keep_hue: u16, tolerance: u16 },
    /// Nearest-neighbour block downscale. Each `block_size × block_size`
    /// region takes the colour of its top-left pixel.
    Pixelate { block_size: u32 },
    /// Split-tone: map shadow pixels toward `shadow`, highlight pixels toward
    /// `highlight`. Classic filmic cross-process look.
    CrossProcess { shadow: [u8; 3], highlight: [u8; 3] },
    /// Permute RGB channels. Cheap surreal palette flip.
    ChannelSwap { variant: ChannelSwapVariant },
    /// Newspaper halftone dots. `dot_spacing` is pixel pitch; dots shrink
    /// where luma is bright, grow where dark.
    Halftone { dot_spacing: u32 },
    /// Remap luma through a 4-stop gradient. More versatile than Duotone:
    /// mid-tones follow the middle stops so an orange→pink→purple→blue
    /// gradient reads as a continuous sky rather than a hard mix.
    GradientMap { stops: [[u8; 3]; 4] },
}

/// Channel permutation. Values are the new position for each input channel —
/// `Grb` means `output = (g, r, b)`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSwapVariant {
    #[default]
    Grb,
    Brg,
    Rbg,
    Bgr,
    Gbr,
}

impl ChannelSwapVariant {
    pub const ALL: &'static [Self] = &[Self::Grb, Self::Brg, Self::Rbg, Self::Bgr, Self::Gbr];
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grb => "GRB",
            Self::Brg => "BRG",
            Self::Rbg => "RBG",
            Self::Bgr => "BGR",
            Self::Gbr => "GBR",
        }
    }
}

/// Backdrop effect composited behind the masked subject. Unlike the solid
/// `bg` color (rendered as a GPU rect at display time — instant), effects
/// need pixels, so they're baked into the output RGBA by `postprocess`.
/// Changing an effect re-runs the postprocess tier.
///
/// `None` preserves the transparent-output + GPU-rect-bg-color render path;
/// any other variant hands the pipeline a derived backdrop built from the
/// source image.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BgEffect {
    /// Keep transparency. Solid `bg` colour (if set) paints at render/export time.
    #[default]
    None,
    /// Fill transparent areas with the source image, blurred. `radius` is the
    /// Gaussian sigma in pixels — higher = softer. Clamped 1..=64.
    BlurredSource { radius: u32 },
    /// Fill transparent areas with the negative of the source.
    InvertedSource,
    /// Fill transparent areas with the luma grayscale of the source.
    DesaturatedSource,
}

impl BgEffect {
    pub const ALL: &'static [Self] = &[
        BgEffect::None,
        BgEffect::BlurredSource { radius: 12 },
        BgEffect::InvertedSource,
        BgEffect::DesaturatedSource,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            BgEffect::None => "Transparent",
            BgEffect::BlurredSource { .. } => "Blurred source",
            BgEffect::InvertedSource => "Inverted source",
            BgEffect::DesaturatedSource => "Desaturated source",
        }
    }
}

impl std::fmt::Display for BgEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl FillStyle {
    pub const ALL: &'static [Self] = &[
        FillStyle::None,
        FillStyle::Desaturate,
        FillStyle::Invert,
        FillStyle::Sepia,
        FillStyle::Threshold { level: 128 },
        FillStyle::Posterize { levels: 4 },
        FillStyle::Solarize { pivot: 128 },
        FillStyle::HueShift { degrees: 90 },
        FillStyle::Saturate { percent: 180 },
        FillStyle::ColorSplash { keep_hue: 0, tolerance: 30 },
        FillStyle::Pixelate { block_size: 12 },
        FillStyle::Duotone { dark: [20, 20, 60], light: [240, 220, 180] },
        FillStyle::CrossProcess { shadow: [30, 60, 110], highlight: [245, 220, 180] },
        FillStyle::ChannelSwap { variant: ChannelSwapVariant::Grb },
        FillStyle::Halftone { dot_spacing: 8 },
        FillStyle::GradientMap { stops: [[20, 20, 60], [120, 60, 140], [240, 120, 80], [255, 230, 180]] },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            FillStyle::None => "None",
            FillStyle::Desaturate => "Desaturate",
            FillStyle::Invert => "Invert",
            FillStyle::Duotone { .. } => "Duotone",
            FillStyle::Sepia => "Sepia",
            FillStyle::Threshold { .. } => "Threshold",
            FillStyle::Posterize { .. } => "Posterize",
            FillStyle::Solarize { .. } => "Solarize",
            FillStyle::HueShift { .. } => "Hue shift",
            FillStyle::Saturate { .. } => "Saturate",
            FillStyle::ColorSplash { .. } => "Color splash",
            FillStyle::Pixelate { .. } => "Pixelate",
            FillStyle::CrossProcess { .. } => "Cross-process",
            FillStyle::ChannelSwap { .. } => "Channel swap",
            FillStyle::Halftone { .. } => "Halftone",
            FillStyle::GradientMap { .. } => "Gradient map",
        }
    }
}

impl std::fmt::Display for FillStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
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
    fn edge_scale_default_is_fused() {
        // Default must keep current behaviour (block_cat / fused output).
        assert_eq!(EdgeScale::default(), EdgeScale::Fused);
    }

    #[test]
    fn edge_scale_from_str_accepts_all_variants_case_insensitive() {
        use std::str::FromStr;
        assert_eq!(EdgeScale::from_str("fine").unwrap(), EdgeScale::Fine);
        assert_eq!(EdgeScale::from_str("Balanced").unwrap(), EdgeScale::Balanced);
        assert_eq!(EdgeScale::from_str("BOLD").unwrap(), EdgeScale::Bold);
        assert_eq!(EdgeScale::from_str("fused").unwrap(), EdgeScale::Fused);
        assert!(EdgeScale::from_str("ultra").is_err());
    }

    #[test]
    fn edge_settings_default_has_fused_scale() {
        // Backwards-compat invariant: loading old settings/preset files that
        // lack edge_scale must fall back to Fused, matching pre-Task-5 output.
        assert_eq!(EdgeSettings::default().edge_scale, EdgeScale::Fused);
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
