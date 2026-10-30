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
    #[error(
        "This photo is too large for the memory free now: {width}\u{00d7}{height} needs about {:.1} GB, {:.1} GB is free. Close other apps and try again.",
        *.need_mb as f64 / 1024.0, *.free_mb as f64 / 1024.0,
    )]
    TooLarge { width: u32, height: u32, need_mb: u64, free_mb: u64 },
    #[error("Processing cancelled")]
    Cancelled,
}

impl CoreError {
    /// A `width` × `height` photo whose work needs `need` bytes with only
    /// `free` free.
    pub fn too_large(width: u32, height: u32, need: u64, free: u64) -> Self {
        CoreError::TooLarge { width, height, need_mb: need >> 20, free_mb: free >> 20 }
    }

    /// The error for a person to read: the message alone, without the
    /// kind of failure in front of it ("Inference error: …").
    pub fn message(&self) -> String {
        match self {
            CoreError::Model(m) | CoreError::Inference(m) | CoreError::ImageFormat(m) => m.clone(),
            other => other.to_string(),
        }
    }

    /// A run or session that ran out of memory: a smaller piece of work,
    /// or the processor, may fit.
    /// Out of the card's memory, not the system's (`ep_compat::is_card_out_of_memory`).
    pub fn is_card_out_of_memory(&self) -> bool {
        match self {
            CoreError::Model(m) | CoreError::Inference(m) => crate::ep_compat::is_card_out_of_memory(m),
            _ => false,
        }
    }

    pub fn is_out_of_memory(&self) -> bool {
        match self {
            CoreError::Model(m) | CoreError::Inference(m) => crate::ep_compat::is_out_of_memory(m),
            _ => false,
        }
    }
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
    RealEsrganX4Plus,
    Nomos8kSchatL,
    RealEsrganX2Plus,
    FourXNmkdSiaxCx,
    FourXNmkdSuperscale,
    FourXNomos2RealPlksr,
    DdcolorTiny,
    DepthAnythingV2Small,
}

impl ModelKind {
    pub fn is_upscale(self) -> bool {
        match self {
            ModelKind::Silueta | ModelKind::U2net | ModelKind::BiRefNetLite | ModelKind::DdcolorTiny | ModelKind::DepthAnythingV2Small => false,
            ModelKind::RealEsrganX4Plus
            | ModelKind::Nomos8kSchatL
            | ModelKind::RealEsrganX2Plus
            | ModelKind::FourXNmkdSiaxCx
            | ModelKind::FourXNmkdSuperscale
            | ModelKind::FourXNomos2RealPlksr => true,
        }
    }

    /// Whether a batch on the host CPU runs faster with an engine per job:
    /// the registry's measured `parallel_jobs_pay_on_cpu`.
    pub fn parallel_jobs_pay_on_cpu(self) -> bool {
        prunr_models::descriptor(self.model_id()).is_some_and(|d| d.parallel_jobs_pay_on_cpu)
    }

    /// The registry id; `const` so a `ModelId` constant can name a kind.
    pub const fn model_id(self) -> prunr_models::ModelId {
        match self {
            ModelKind::Silueta => prunr_models::ModelId::Silueta,
            ModelKind::U2net => prunr_models::ModelId::U2net,
            ModelKind::BiRefNetLite => prunr_models::ModelId::BiRefNetLite,
            ModelKind::RealEsrganX4Plus => prunr_models::ModelId::RealEsrganX4Plus,
            ModelKind::Nomos8kSchatL => prunr_models::ModelId::Nomos8kSchatL,
            ModelKind::RealEsrganX2Plus => prunr_models::ModelId::RealEsrganX2Plus,
            ModelKind::FourXNmkdSiaxCx => prunr_models::ModelId::FourXNmkdSiaxCx,
            ModelKind::FourXNmkdSuperscale => prunr_models::ModelId::FourXNmkdSuperscale,
            ModelKind::FourXNomos2RealPlksr => prunr_models::ModelId::FourXNomos2RealPlksr,
            ModelKind::DdcolorTiny => prunr_models::ModelId::DdcolorTiny,
            ModelKind::DepthAnythingV2Small => prunr_models::ModelId::DepthAnythingV2Small,
        }
    }
}

impl From<ModelKind> for prunr_models::ModelId {
    fn from(kind: ModelKind) -> Self {
        kind.model_id()
    }
}

impl TryFrom<prunr_models::ModelId> for ModelKind {
    type Error = prunr_models::ModelId;

    /// Inpaint / SD / TAESD / edge ids have no `ModelKind` and return
    /// `Err(id)`; callers filter by category first.
    fn try_from(id: prunr_models::ModelId) -> Result<Self, Self::Error> {
        match id {
            prunr_models::ModelId::Silueta => Ok(ModelKind::Silueta),
            prunr_models::ModelId::U2net => Ok(ModelKind::U2net),
            prunr_models::ModelId::BiRefNetLite => Ok(ModelKind::BiRefNetLite),
            prunr_models::ModelId::RealEsrganX4Plus => Ok(ModelKind::RealEsrganX4Plus),
            prunr_models::ModelId::Nomos8kSchatL => Ok(ModelKind::Nomos8kSchatL),
            prunr_models::ModelId::RealEsrganX2Plus => Ok(ModelKind::RealEsrganX2Plus),
            prunr_models::ModelId::FourXNmkdSiaxCx => Ok(ModelKind::FourXNmkdSiaxCx),
            prunr_models::ModelId::FourXNmkdSuperscale => Ok(ModelKind::FourXNmkdSuperscale),
            prunr_models::ModelId::FourXNomos2RealPlksr => Ok(ModelKind::FourXNomos2RealPlksr),
            prunr_models::ModelId::DdcolorTiny => Ok(ModelKind::DdcolorTiny),
            prunr_models::ModelId::DepthAnythingV2Small => Ok(ModelKind::DepthAnythingV2Small),
            other => Err(other),
        }
    }
}

/// Line extraction mode. Determines which engines the pipeline uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LineMode {
    Off,
    /// Line detection on the original image; segmentation skipped.
    #[serde(alias = "LinesOnly")]
    EdgesOnly,
    /// Segmentation first, then line detection — edges only within subject, body transparent.
    #[serde(alias = "AfterBgRemoval")]
    SubjectOutline,
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
    /// Edge contrast, 0.0–1.0: 0 leaves the mask's soft edge, 1 cuts it
    /// hard at half coverage. Applied at full resolution after refine and
    /// feather.
    #[serde(default)]
    pub contrast: f32,
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
    #[serde(default, rename = "fill_style")]
    pub effect: SubjectEffect,
    /// Sticker outline and shadow around the cut-out.
    #[serde(default)]
    pub decoration: Decoration,
}

/// A sticker outline and a shadow drawn around the cut-out, under it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Decoration {
    /// Outline width in pixels; 0 draws none.
    pub outline_width: f32,
    pub outline_color: [u8; 3],
    pub shadow: ShadowKind,
    /// Shadow strength, 0–1.
    pub shadow_opacity: f32,
    /// Shadow softness in pixels.
    pub shadow_blur: f32,
    /// How far a Drop shadow falls below the subject, in pixels.
    pub shadow_distance: f32,
}

impl Default for Decoration {
    fn default() -> Self {
        Self {
            outline_width: 0.0,
            outline_color: [255, 255, 255],
            shadow: ShadowKind::Off,
            shadow_opacity: 0.55,
            shadow_blur: 24.0,
            shadow_distance: 20.0,
        }
    }
}

impl Decoration {
    pub fn has_outline(&self) -> bool {
        self.outline_width >= 0.5
    }

    /// The same decoration on the image at `scale` times its size: the
    /// outline, the shadow's softness and its fall scale with it.
    pub fn at_scale(&self, scale: f32) -> Self {
        Self {
            outline_width: self.outline_width * scale,
            shadow_blur: self.shadow_blur * scale,
            shadow_distance: self.shadow_distance * scale,
            ..*self
        }
    }

    pub fn has_shadow(&self) -> bool {
        self.shadow != ShadowKind::Off && self.shadow_opacity > 0.0
    }

    /// The widest outline the app and `--outline` offer, in pixels.
    pub const MAX_OUTLINE_PX: f32 = 96.0;

    /// How far below the subject the shadow falls, in whole pixels.
    pub fn shadow_fall_px(&self) -> u32 {
        if self.shadow == ShadowKind::Drop { self.shadow_distance.round().max(0.0) as u32 } else { 0 }
    }

    /// The radius of the shadow's three box passes.
    pub fn shadow_radius_px(&self) -> u32 {
        (self.shadow_blur / 2.0).round() as u32
    }

    /// How far past the subject the decoration's work reaches, in pixels:
    /// the shadow's blur with the margin that keeps its crop bit-exact.
    pub fn reach_px(&self) -> u32 {
        let outline = if self.has_outline() { self.outline_width.ceil() as u32 } else { 0 };
        let shadow = if self.has_shadow() {
            (crate::postprocess::BLUR_CROP_MARGIN.saturating_mul(self.shadow_radius_px())).saturating_add(self.shadow_fall_px())
        } else {
            0
        };
        outline.saturating_add(shadow).saturating_add(2)
    }
}

impl std::str::FromStr for ShadowKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "soft" => Ok(Self::Soft),
            "drop" => Ok(Self::Drop),
            _ => Err(format!("'{s}' is not one of off, soft, drop")),
        }
    }
}

/// The kind of shadow under a cut-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowKind {
    #[default]
    Off,
    /// A soft glow all around, as if lit from the front.
    Soft,
    /// A shadow falling below the subject.
    Drop,
}

impl MaskSettings {
    /// The pipeline's `edge_shift` (+ erodes) for a shift as people read
    /// it, + grows the subject: what the Shift edge slider, `--edge-shift`
    /// and `PRUNR_EDGE_SHIFT` take. Its own inverse.
    pub fn edge_shift_for_grow(grow: f32) -> f32 {
        0.0 - grow
    }
}

impl Default for MaskSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            contrast: 0.0,
            edge_shift: 0.0,
            refine_edges: false,
            guided_radius: 8,
            guided_epsilon: 1e-4,
            feather: 0.0,
            effect: SubjectEffect::default(),
            decoration: Decoration::default(),
        }
    }
}

/// Which TEED output the lines come from. Every scale is cheap to keep,
/// so a switch reads a cached plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub enum EdgeScale {
    /// The fused output of a run at twice the size: thin, detailed lines.
    Fine,
    /// The deepest side output: bold outlines.
    Bold,
    /// The fused output; the default. `Balanced` (a scale of the
    /// previous detector) loads as this.
    #[default]
    #[serde(alias = "Balanced")]
    Fused,
}

impl EdgeScale {
    /// Every scale, in tensor-slot (`as usize`) and display order.
    pub const ALL: [Self; 3] = [Self::Fine, Self::Bold, Self::Fused];
}

impl std::fmt::Display for EdgeScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Fine => "fine",
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
            "bold" => Ok(Self::Bold),
            "fused" | "balanced" => Ok(Self::Fused),
            _ => Err(format!("unknown edge scale '{s}' (expected: fine, bold, fused)")),
        }
    }
}

/// Controls for drawing the detected lines.
/// `line_mode` is kept separate — it picks whether edges run at all.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeSettings {
    /// Line detector sensitivity, 0.0-1.0.
    pub line_strength: f32,
    /// Solid color override for edges. `None` = preserve original RGB.
    pub solid_line_color: Option<[u8; 3]>,
    /// Dilate edge mask by N pixels after threshold — thickens thin lines.
    pub edge_thickness: u32,
    pub edge_scale: EdgeScale,
    /// How the subject mask and edge mask combine in `LineMode::SubjectOutline`.
    /// Ignored in Off / EdgesOnly. Pure compose-time setting — changing it
    /// doesn't re-run inference.
    #[serde(default)]
    pub compose_mode: ComposeMode,
    /// How edge pixels are coloured (solid / gradient). Compose-time.
    #[serde(default)]
    pub line_style: LineStyle,
    /// Transform applied to the image before line detection. Changing
    /// this invalidates the edge tensor cache.
    #[serde(default)]
    pub input_transform: InputTransform,
}

impl EdgeSettings {
    /// The scale the main line layer comes from: Dual scale layers Fine
    /// under Bold whatever the Scale chip says (presets or the CLI can
    /// still carry Bold there, which would collapse both layers into one).
    pub fn primary_scale(&self) -> EdgeScale {
        if matches!(self.line_style, LineStyle::DualScale { .. }) { EdgeScale::Fine } else { self.edge_scale }
    }

    /// Every scale a compose with these settings reads: the primary
    /// scale, plus Bold for Dual scale.
    pub fn drawn_scales(&self) -> &'static [EdgeScale] {
        match (self.line_style, self.edge_scale) {
            (LineStyle::DualScale { .. }, _) => &[EdgeScale::Fine, EdgeScale::Bold],
            (_, EdgeScale::Fine) => &[EdgeScale::Fine],
            (_, EdgeScale::Bold) => &[EdgeScale::Bold],
            (_, EdgeScale::Fused) => &[EdgeScale::Fused],
        }
    }
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
            input_transform: InputTransform::default(),
        }
    }
}

/// Transform applied to the image BEFORE it reaches the line detector.
/// Changes what the detector "sees", so switching `InputTransform` invalidates the edge tensor
/// cache and reruns the detector.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum InputTransform {
    /// Pass the image through unchanged.
    #[default]
    None,
    /// Convert to grayscale before edge detection — colour-edge artefacts
    /// collapse into pure luminance edges, producing ink-like lines.
    Grayscale,
    /// A light blur, so photo grain and noise don't become stray lines.
    Smooth,
    /// Scale contrast by `percent / 100` around luma 128. 100 = neutral,
    /// 150 = punchy, 50 = muted. Clamped 50..=300.
    ContrastBoost { percent: u16 },
    /// Quantise each channel to N levels before edge detection — the detector
    /// sees step transitions instead of gradients, producing cleaner lines.
    Posterize { levels: u8 },
}

impl InputTransform {
    pub const ALL: &'static [Self] = &[
        InputTransform::None,
        InputTransform::Grayscale,
        InputTransform::Smooth,
        InputTransform::ContrastBoost { percent: 150 },
        InputTransform::Posterize { levels: 4 },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            InputTransform::None => "None",
            InputTransform::Grayscale => "Grayscale",
            InputTransform::Smooth => "Smooth",
            InputTransform::ContrastBoost { .. } => "Contrast boost",
            InputTransform::Posterize { .. } => "Posterize",
        }
    }
}

impl std::fmt::Display for InputTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
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
}

impl std::fmt::Display for ComposeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A display name in CLI form: lowercase, words joined by `-`.
fn cli_name(display: &str) -> String {
    display.to_ascii_lowercase().replace(' ', "-")
}

/// The entry of `all` whose display name matches `s` in CLI form, so any
/// case and `-` or a space between words; else an error naming them all.
fn parse_cli_name<T: Copy>(s: &str, all: &[T], name: impl Fn(T) -> &'static str) -> Result<T, String> {
    let wanted = cli_name(s);
    all.iter().copied().find(|&t| cli_name(name(t)) == wanted).ok_or_else(|| {
        let names: Vec<String> = all.iter().map(|&t| cli_name(name(t))).collect();
        format!("'{s}' is not one of {}", names.join(", "))
    })
}

/// Accepts the display name in any case, with `-` or a space between
/// words: `lines-only`, `Subject filled`.
impl std::str::FromStr for ComposeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_cli_name(s, Self::ALL, Self::name)
    }
}

impl ComposeMode {
    pub fn name(self) -> &'static str {
        match self {
            ComposeMode::LinesOnly => "Lines only",
            ComposeMode::SubjectFilled => "Subject filled",
            ComposeMode::Engraving => "Engraving",
            ComposeMode::Ghost => "Ghost",
        }
    }

    /// The output alpha for a subject alpha `s` and an edge value `e`,
    /// both 0..=255.
    #[inline]
    pub fn alpha(self, s: i32, e: i32) -> u8 {
        match self {
            ComposeMode::LinesOnly => crate::math::div255_round((s * e) as u32) as u8,
            ComposeMode::SubjectFilled => s.max(e) as u8,
            ComposeMode::Engraving => (s - e).max(0) as u8,
            // 0.3 * subject + 0.8 * edge, clamped. Sums to > 1.0 on purpose —
            // saturates to fully opaque where subject AND edge both contribute.
            ComposeMode::Ghost => crate::math::div255_round((s * 77 + e * 204) as u32).min(255) as u8,
        }
    }

    pub const ALL: &'static [Self] = &[
        Self::LinesOnly,
        Self::SubjectFilled,
        Self::Engraving,
        Self::Ghost,
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
    /// Two detector scales layered — Fine at `fine_color` for micro-details,
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
        LineStyle::DualScale { fine_color: [50, 180, 230], bold_color: [30, 20, 60] },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            LineStyle::Solid => "Solid",
            LineStyle::GradientY { .. } => "Vertical gradient",
            LineStyle::GradientX { .. } => "Horizontal gradient",
            LineStyle::RadialGradient { .. } => "Radial gradient",
            LineStyle::Rainbow { .. } => "Rainbow",
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
pub enum SubjectEffect {
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
    /// Newspaper halftone dots. `dot_spacing` is pixel pitch; dots shrink
    /// where luma is bright, grow where dark.
    Halftone { dot_spacing: u32 },
    /// Remap luma through a 4-stop gradient. More versatile than Duotone:
    /// mid-tones follow the middle stops so an orange→pink→purple→blue
    /// gradient reads as a continuous sky rather than a hard mix.
    GradientMap { stops: [[u8; 3]; 4] },
}

/// A picture made from the photo, shown behind the result the way a
/// background image is: under whatever the result leaves transparent,
/// never baked into the result itself, so lines and compositions keep
/// their own transparency.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BgEffect {
    /// No picture. Settings saved with the retired Inverted source load
    /// as this.
    #[default]
    #[serde(alias = "inverted_source")]
    None,
    /// Fill transparent areas with the source image, blurred. `radius` is the
    /// Gaussian sigma in pixels — higher = softer. Clamped 1..=64.
    BlurredSource { radius: u32 },
    /// Fill transparent areas with the luma grayscale of the source.
    DesaturatedSource,
    /// The photo blurred like a wide-open lens focused on the subject:
    /// the farther from the subject's depth, the softer. `amount_pct`
    /// sets the strongest blur, 0-100.
    LensBlur { amount_pct: u8 },
}

impl BgEffect {
    /// Whether its picture is made with the photo's depth map.
    pub fn needs_depth(self) -> bool {
        matches!(self, BgEffect::LensBlur { .. })
    }

    /// Whether its picture depends on the cut-out: a blur is made of the
    /// photo with the subject filled from the background around it.
    pub fn takes_cutout(self) -> bool {
        matches!(self, BgEffect::BlurredSource { .. } | BgEffect::LensBlur { .. })
    }

    pub const ALL: &'static [Self] = &[
        BgEffect::None,
        BgEffect::BlurredSource { radius: 12 },
        BgEffect::DesaturatedSource,
        BgEffect::LensBlur { amount_pct: 50 },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            BgEffect::None => "Transparent",
            BgEffect::BlurredSource { .. } => "Blurred photo",
            BgEffect::DesaturatedSource => "Black & white photo",
            BgEffect::LensBlur { .. } => "Lens blur",
        }
    }
}

impl std::fmt::Display for BgEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// How a per-image background image is positioned and scaled inside the
/// result frame. CSS-style.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BgImageFit {
    /// Fills both target dimensions (uniform scale, may crop overhang).
    /// Settings saved with the retired Stretch load as Cover.
    #[default]
    #[serde(alias = "stretch")]
    Cover,
    /// Fits inside both target dimensions (uniform scale, may letterbox).
    /// Settings saved with the retired Center load as Contain.
    #[serde(alias = "center")]
    Contain,
    /// Repeated at native size.
    Tile,
}

impl BgImageFit {
    pub const ALL: &'static [Self] = &[
        BgImageFit::Cover,
        BgImageFit::Contain,
        BgImageFit::Tile,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BgImageFit::Cover => "Cover",
            BgImageFit::Contain => "Contain",
            BgImageFit::Tile => "Tile",
        }
    }
}

impl std::fmt::Display for BgImageFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Lower-case wire form for CLI (`--bg-image-fit cover`) — the
        // human-readable label uses `name()`.
        let s = match self {
            BgImageFit::Cover => "cover",
            BgImageFit::Contain => "contain",
            BgImageFit::Tile => "tile",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for BgImageFit {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cover" => Ok(BgImageFit::Cover),
            "contain" => Ok(BgImageFit::Contain),
            "tile" => Ok(BgImageFit::Tile),
            _ => Err(format!("'{s}' is not one of cover, contain, tile")),
        }
    }
}

impl SubjectEffect {
    /// The same style on the image at `scale` times its size: pixel
    /// blocks and halftone dots scale with it, at least a pixel each.
    pub fn at_scale(self, scale: f32) -> Self {
        let px = |v: u32| ((v as f32 * scale).round() as u32).max(1);
        match self {
            SubjectEffect::Pixelate { block_size } => SubjectEffect::Pixelate { block_size: px(block_size) },
            SubjectEffect::Halftone { dot_spacing } => SubjectEffect::Halftone { dot_spacing: px(dot_spacing) },
            other => other,
        }
    }

    pub const ALL: &'static [Self] = &[
        SubjectEffect::None,
        SubjectEffect::Desaturate,
        SubjectEffect::Invert,
        SubjectEffect::Sepia,
        SubjectEffect::Threshold { level: 128 },
        SubjectEffect::Posterize { levels: 4 },
        SubjectEffect::Solarize { pivot: 128 },
        SubjectEffect::HueShift { degrees: 90 },
        SubjectEffect::Saturate { percent: 180 },
        SubjectEffect::ColorSplash { keep_hue: 0, tolerance: 30 },
        SubjectEffect::Pixelate { block_size: 12 },
        SubjectEffect::Duotone { dark: [20, 20, 60], light: [240, 220, 180] },
        SubjectEffect::CrossProcess { shadow: [30, 60, 110], highlight: [245, 220, 180] },
        SubjectEffect::Halftone { dot_spacing: 8 },
        SubjectEffect::GradientMap { stops: [[20, 20, 60], [120, 60, 140], [240, 120, 80], [255, 230, 180]] },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            SubjectEffect::None => "None",
            SubjectEffect::Desaturate => "Desaturate",
            SubjectEffect::Invert => "Invert",
            SubjectEffect::Duotone { .. } => "Duotone",
            SubjectEffect::Sepia => "Sepia",
            SubjectEffect::Threshold { .. } => "Threshold",
            SubjectEffect::Posterize { .. } => "Posterize",
            SubjectEffect::Solarize { .. } => "Solarize",
            SubjectEffect::HueShift { .. } => "Hue shift",
            SubjectEffect::Saturate { .. } => "Saturate",
            SubjectEffect::ColorSplash { .. } => "Color splash",
            SubjectEffect::Pixelate { .. } => "Pixelate",
            SubjectEffect::CrossProcess { .. } => "Cross-process",
            SubjectEffect::Halftone { .. } => "Halftone",
            SubjectEffect::GradientMap { .. } => "Gradient map",
        }
    }
}

impl std::fmt::Display for SubjectEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Reads a display name in CLI form (`hue-shift`, `Color splash`), giving
/// that effect with its default settings.
impl std::str::FromStr for SubjectEffect {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_cli_name(s, Self::ALL, |f| f.name())
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
    #[test]
    fn pixel_sizes_scale_with_the_image() {
        use super::{Decoration, SubjectEffect, ShadowKind};
        let d = Decoration { outline_width: 12.0, shadow: ShadowKind::Drop, shadow_blur: 24.0, shadow_distance: 20.0, ..Decoration::default() };
        let half = d.at_scale(0.5);
        assert_eq!((half.outline_width, half.shadow_blur, half.shadow_distance), (6.0, 12.0, 10.0));
        assert_eq!((half.outline_color, half.shadow, half.shadow_opacity), (d.outline_color, d.shadow, d.shadow_opacity));
        assert_eq!(SubjectEffect::Pixelate { block_size: 12 }.at_scale(0.25), SubjectEffect::Pixelate { block_size: 3 });
        assert_eq!(SubjectEffect::Halftone { dot_spacing: 8 }.at_scale(0.01), SubjectEffect::Halftone { dot_spacing: 1 }, "at least a pixel");
        for style in SubjectEffect::ALL.iter().filter(|s| !matches!(s, SubjectEffect::Pixelate { .. } | SubjectEffect::Halftone { .. })) {
            assert_eq!(style.at_scale(0.3), *style, "{} has no pixel size", style.name());
        }
    }

    use super::*;

    #[test]
    fn an_errors_message_is_the_text_without_its_kind() {
        assert_eq!(CoreError::Inference("needs 15.6 GB".into()).message(), "needs 15.6 GB");
        assert_eq!(CoreError::Model("missing".into()).message(), "missing");
        assert_eq!(CoreError::Cancelled.message(), "Processing cancelled");
    }

    #[test]
    fn each_error_kind_names_itself_in_its_text() {
        assert_eq!(CoreError::Model("test error".to_string()).to_string(), "Model error: test error");
        assert_eq!(CoreError::Inference("ort failed".into()).to_string(), "Inference error: ort failed");
        assert_eq!(CoreError::ImageFormat("bad format".into()).to_string(), "Image format error: bad format");
        let msg = CoreError::TooLarge { width: 9000, height: 5000, need_mb: 5632, free_mb: 3072 }.to_string();
        assert!(msg.contains("9000\u{00d7}5000") && msg.contains("5.5 GB") && msg.contains("3.0 GB"), "{msg}");
    }

    #[test]
    fn model_kind_x2plus_roundtrips() {
        let model_id: prunr_models::ModelId = ModelKind::RealEsrganX2Plus.into();
        assert_eq!(model_id, prunr_models::ModelId::RealEsrganX2Plus);
        let roundtrip = ModelKind::try_from(prunr_models::ModelId::RealEsrganX2Plus).unwrap();
        assert_eq!(roundtrip, ModelKind::RealEsrganX2Plus);
    }

    #[test]
    fn edge_scale_default_is_fused() {
        assert_eq!(EdgeScale::default(), EdgeScale::Fused);
        // Settings and presets saved before edge_scale load as Fused.
        assert_eq!(EdgeSettings::default().edge_scale, EdgeScale::Fused);
    }

    #[test]
    fn compose_mode_parses_every_display_name_in_cli_form() {
        use std::str::FromStr;
        for mode in ComposeMode::ALL {
            assert_eq!(ComposeMode::from_str(&cli_name(mode.name())), Ok(*mode));
        }
        assert_eq!(ComposeMode::from_str("Subject filled"), Ok(ComposeMode::SubjectFilled));
        assert_eq!(ComposeMode::from_str("LINES only"), Ok(ComposeMode::LinesOnly));
        assert_eq!(
            ComposeMode::from_str("inverse-mask"),
            Err("'inverse-mask' is not one of lines-only, subject-filled, engraving, ghost".to_string()),
            "retired",
        );
    }

    /// Stretch and Center are retired: saved settings load as the nearest
    /// fit, and the CLI names the ones left.
    #[test]
    fn every_effect_parses_from_its_cli_name() {
        use std::str::FromStr;
        for style in SubjectEffect::ALL {
            let cli = cli_name(style.name());
            assert_eq!(SubjectEffect::from_str(&cli), Ok(*style), "{cli}");
        }
        assert_eq!(SubjectEffect::from_str("Color splash").map(|f| f.name()), Ok("Color splash"));
        assert!(SubjectEffect::from_str("channel-swap").is_err(), "retired");
    }

    #[test]
    fn the_retired_inverted_background_loads_as_none() {
        assert_eq!(serde_json::from_str::<BgEffect>("\"inverted_source\"").unwrap(), BgEffect::None);
    }

    #[test]
    fn retired_fits_load_as_their_nearest() {
        use std::str::FromStr;
        let load = |s: &str| serde_json::from_str::<BgImageFit>(&format!("\"{s}\"")).unwrap();
        assert_eq!(load("stretch"), BgImageFit::Cover);
        assert_eq!(load("center"), BgImageFit::Contain);
        assert_eq!(load("tile"), BgImageFit::Tile);
        assert!(BgImageFit::from_str("stretch").is_err());
        for fit in BgImageFit::ALL {
            assert_eq!(BgImageFit::from_str(&fit.to_string()), Ok(*fit));
        }
    }

    #[test]
    fn edge_scale_from_str_accepts_all_variants_case_insensitive() {
        use std::str::FromStr;
        assert_eq!(EdgeScale::from_str("fine").unwrap(), EdgeScale::Fine);
        assert_eq!(EdgeScale::from_str("Balanced").unwrap(), EdgeScale::Fused, "the retired Balanced scale reads as Fused");
        assert_eq!(EdgeScale::from_str("BOLD").unwrap(), EdgeScale::Bold);
        assert_eq!(EdgeScale::from_str("fused").unwrap(), EdgeScale::Fused);
        assert!(EdgeScale::from_str("ultra").is_err());
    }

    #[test]
    fn a_saved_balanced_scale_loads_as_fused() {
        let loaded: EdgeScale = serde_json::from_str("\"Balanced\"").expect("Balanced still loads");
        assert_eq!(loaded, EdgeScale::Fused);
    }

    #[test]
    fn drawn_scales_are_the_primary_plus_bold_for_dual_scale() {
        let dual = LineStyle::DualScale { fine_color: [0; 3], bold_color: [0; 3] };
        for scale in EdgeScale::ALL {
            let single = EdgeSettings { edge_scale: scale, ..EdgeSettings::default() };
            assert_eq!(single.drawn_scales(), &[scale]);
            let layered = EdgeSettings { edge_scale: scale, line_style: dual, ..EdgeSettings::default() };
            assert_eq!(layered.drawn_scales(), &[EdgeScale::Fine, EdgeScale::Bold]);
        }
    }
}
