//! Brush tool coordinator. Owns the toolbar toggle, current size /
//! hardness / mode, and the in-progress stroke buffer. Doesn't iterate
//! `BatchManager.items` — the caller hands it the active grid size
//! and writes the committed strokes back via `BatchItem`'s mutator.

use prunr_core::brush::{paint_circle, paint_line, paint_square, paint_square_line, BrushShape, Stamp};
use prunr_core::selection::{BrushMode, MaskArtifact};
use serde::{Deserialize, Serialize};

/// Brush radius in screen points, for the sliders and the size keys.
pub const BRUSH_RADIUS_RANGE: std::ops::RangeInclusive<f32> = 1.0..=200.0;

/// What the selection does on a cut-out: restored or erased live as it
/// is painted, or only selected (the menu's Restore and Delete act on it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CutoutAction {
    #[default]
    Restore,
    Erase,
    Select,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushSettings {
    pub radius: f32,
    /// Edge falloff: 0.0 = full smoothstep falloff, 1.0 = hard disc.
    pub hardness: f32,
    /// Stroke magnitude: 1.0 is a full gamma push, lower is gentler
    /// (`m → m * (1 - s)` for subtract).
    pub strength: f32,
    /// Add to or take from the selection; on a cut-out a stroke adds and
    /// Alt/Ctrl takes away, and `cutout_action` says what the selection does.
    pub mode: BrushMode,
    pub cutout_action: CutoutAction,
    pub shape: BrushShape,
    /// Eraser-only: unsharp-mask amount inside the filled region; 0.0 is the
    /// model's output as-is (slightly soft). Sweet spot 0.3-0.7.
    // Load-bearing: a file without the field loads as f32's 0.0, not
    // `Default`'s 0.6.
    #[serde(default)]
    pub inpaint_sharpen: f32,
    /// Eraser-only: pixels of soft blend at the painted-region boundary,
    /// hiding the seam between fill and source.
    pub inpaint_feather: f32,
    /// Eraser-only: dilate (>0) or erode (<0) the painted mask before
    /// inference. Small dilation gives LaMa more context for cleaner
    /// fills; erosion preserves more original detail at the boundary.
    pub inpaint_grow: f32,
    /// Eraser: how large a hole is filled (see `FillScale`).
    pub inpaint_fill_scale: prunr_core::inpaint::FillScale,
    /// Eraser: how the fill's colours meet the photo around it, when
    /// chosen; `None` is the eraser's own (`edge_colour`).
    #[serde(default, rename = "inpaint_edge_color")]
    pub inpaint_edge_colour: Option<prunr_core::inpaint_blend::EdgeColour>,
    /// SD and RORem: the Feather band is generated last, against the
    /// photo around it, and pasted back fading into it.
    #[serde(default = "default_soft_edge")]
    pub sd_soft_edge: bool,
    /// SD-only: the sampler, when chosen apart from the quality; `None`
    /// runs the quality's own.
    #[serde(default)]
    pub sd_sampler: Option<SdScheduler>,
    /// SD-only: the user's own prompt; empty runs `DEFAULT_SD_PROMPT`.
    #[serde(deserialize_with = "own_prompt")]
    pub sd_prompt: String,
    /// SD-only: negative prompt; defaults to the SD-1.5 anti-artifact list
    /// (text, watermark, blur, ...).
    pub sd_negative_prompt: String,
    /// SD-only: classifier-free guidance scale. 1.0 = no CFG (cond pass
    /// only). 7.5 is the typical SD-1.5 prompt strength. >1 doubles UNet
    /// cost per step (cond + uncond passes blended).
    pub sd_guidance_scale: f32,
    /// SD-only: the quality (sampler); Fast is the default.
    pub sd_scheduler: SdScheduler,
    /// SD-only: number of denoise steps, within `SdScheduler::steps_range`.
    pub sd_steps: u32,
    /// RORem only: its denoise steps, within `ROREM_STEPS_RANGE`.
    pub rorem_steps: u32,
    /// SD-only: pinned RNG seed; `None` = fresh random per stroke.
    pub sd_seed: Option<u64>,
    /// SD-only: TAESD fast VAE. `None` = auto (on when installed); `Some` is
    /// an explicit choice, still gated by install.
    pub sd_use_taesd: Option<bool>,
    /// Shared selection knob: edge feather in pixels, 0 = sharp. Applied to
    /// what the overlay shows and to what Delete / Copy / Cut act on.
    pub edge_feather: f32,
    /// Magic Brush only: the probability a pixel must reach to be
    /// selected (0.5 is the model's decision boundary).
    pub magic_confidence_threshold: f32,
    /// Magic Brush only: drop specks and fill pinholes in what the model
    /// returns.
    pub magic_remove_specks: bool,
    /// Magic Brush only: a click selects everything at about the clicked
    /// depth (the depth map) instead of the clicked object (SAM).
    pub magic_depth: bool,
    /// How far in nearness (0-1) from the clicked depth a Depth click
    /// reaches.
    pub magic_depth_range: f32,
}

/// The SD eraser's quality: which sampler runs the fill.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SdScheduler {
    /// Fast: the LCM-distilled model, a few steps.
    #[default]
    Lcm,
    /// Quality: DPM-Solver++ 2M with Karras spacing on the full model.
    /// Settings saved with the retired DDIM, UniPC or Euler-A load as this.
    #[serde(alias = "Ddim", alias = "UniPc", alias = "EulerA")]
    DpmPlusPlus2MKarras,
}

impl SdScheduler {
    pub const ALL: [SdScheduler; 2] = [SdScheduler::Lcm, SdScheduler::DpmPlusPlus2MKarras];

    /// The quality's name in the Quality chip.
    pub fn label(&self) -> &'static str {
        match self {
            SdScheduler::Lcm => "Fast",
            SdScheduler::DpmPlusPlus2MKarras => "Quality",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SdScheduler::Lcm => "A few seconds per fill",
            SdScheduler::DpmPlusPlus2MKarras => "The best fill, several times slower",
        }
    }

    /// Steps the sampler is made for.
    pub fn steps_range(&self) -> std::ops::RangeInclusive<u32> {
        match self {
            SdScheduler::Lcm => 2..=8,
            SdScheduler::DpmPlusPlus2MKarras => 10..=40,
        }
    }

    /// Guidance that suits the sampler: the distilled model breaks above 2.
    pub fn guidance_range(&self) -> std::ops::RangeInclusive<f32> {
        match self {
            SdScheduler::Lcm => 1.0..=2.0,
            SdScheduler::DpmPlusPlus2MKarras => 1.0..=10.0,
        }
    }
}

impl From<SdScheduler> for prunr_core::inpaint_sd::SchedulerKind {
    fn from(s: SdScheduler) -> Self {
        use prunr_core::inpaint_sd::SchedulerKind;
        match s {
            SdScheduler::Lcm => SchedulerKind::Lcm,
            SdScheduler::DpmPlusPlus2MKarras => SchedulerKind::DpmPp2MKarras,
        }
    }
}

fn default_feather() -> f32 { prunr_core::inpaint_blend::DEFAULT_EDGE_BLEND_PX }
fn default_soft_edge() -> bool { true }
pub(crate) fn default_edge_colour(generated: bool) -> prunr_core::inpaint_blend::EdgeColour {
    use prunr_core::inpaint_blend::EdgeColour;
    if generated { EdgeColour::Off } else { EdgeColour::AlongEdge }
}
fn default_grow() -> f32 { prunr_core::inpaint_blend::DEFAULT_GROW_PX }
fn default_magic_confidence_threshold() -> f32 { 0.5 }
fn default_magic_remove_specks() -> bool { true }
fn default_magic_depth_range() -> f32 { DEPTH_RANGE_DEFAULT }

/// The Range slider's span and starting point for a Depth click.
pub const DEPTH_RANGE: std::ops::RangeInclusive<f32> = 0.02..=0.3;
const DEPTH_RANGE_DEFAULT: f32 = 0.08;
fn default_cfg() -> f32 { crate::gui::presets::SdSchedulerBundle::default().guidance_scale }

/// Defaults for SD eraser prompts. Style-agnostic (no "photorealistic",
/// which biases non-photo sources); the negative blocks SD-1.5 failure
/// modes, above all text-like glyphs on weak conditioning.
pub const DEFAULT_SD_PROMPT: &str =
    "seamless continuation of the surroundings, matching style, high quality";
pub const DEFAULT_SD_NEGATIVE_PROMPT: &str =
    "text, letters, words, watermark, signature, logo, blurry, distorted, \
     low quality, oversaturated, jpeg artifacts";

/// A saved prompt, with a saved copy of the shipped one read as none of
/// the user's own.
pub(crate) fn own_prompt<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let prompt = String::deserialize(d)?;
    Ok(if prompt == DEFAULT_SD_PROMPT { String::new() } else { prompt })
}
pub(crate) fn default_sd_negative_prompt() -> String { DEFAULT_SD_NEGATIVE_PROMPT.to_string() }

fn default_sd_steps() -> u32 { crate::gui::presets::SdSchedulerBundle::default().steps }
fn default_rorem_steps() -> u32 { prunr_core::inpaint_sd::ROREM_STEPS }
/// The steps RORem's chip offers around the 4 it was distilled for.
pub const ROREM_STEPS_RANGE: std::ops::RangeInclusive<u32> = 2..=8;

impl BrushSettings {
    pub fn stamp(&self) -> Stamp {
        Stamp { hardness: self.hardness, strength: self.strength, mode: self.mode }
    }

    /// Stamp for the eraser's region, which is all or nothing: hard and
    /// full strength, so the region covers exactly the cursor, adding
    /// unless the subtract modifier is held.
    pub fn region_stamp(&self, subtract: bool) -> Stamp {
        let mode = if subtract { BrushMode::Subtract } else { BrushMode::Add };
        Stamp { hardness: 1.0, strength: 1.0, mode }
    }

    /// Whether the fast decoder runs, given whether it is installed. Left
    /// on auto it runs with Fast; Quality is for the best fill.
    pub fn sd_use_taesd_effective_with_avail(&self, taesd_available: bool) -> bool {
        self.sd_use_taesd.unwrap_or(self.sd_scheduler == SdScheduler::Lcm) && taesd_available
    }

    pub fn sd_use_taesd_effective(&self) -> bool {
        self.sd_use_taesd_effective_with_avail(prunr_models::is_available(prunr_models::ModelId::TaesdFp16))
    }

    /// The settings `model` starts from when no preset has an entry for it.
    pub fn floor_for(model: prunr_models::ModelId) -> Self {
        // SD draws its own detail; sharpening on top crunches it.
        let inpaint_sharpen = if model.is_sd_family() { 0.0 } else { Self::default().inpaint_sharpen };
        Self { inpaint_sharpen, ..Self::default() }
    }

    /// The mode a stroke paints with: Shift adds and Alt (or Ctrl)
    /// subtracts for the stroke's duration; otherwise it adds on a
    /// `cutout` (the switch there says what the selection does) and
    /// follows the chosen mode elsewhere.
    pub fn mode_under(&self, cutout: bool, shift: bool, subtract: bool) -> BrushMode {
        if shift {
            BrushMode::Add
        } else if subtract {
            BrushMode::Subtract
        } else if cutout {
            BrushMode::Add
        } else {
            self.mode
        }
    }

    /// One size step is a quarter of the radius, at least a pixel, so
    /// small brushes still move and large ones do not crawl.
    pub fn step_radius(&mut self, larger: bool) {
        let step = (self.radius * 0.25).max(1.0);
        let next = if larger { self.radius + step } else { self.radius - step };
        self.radius = next.round().clamp(*BRUSH_RADIUS_RANGE.start(), *BRUSH_RADIUS_RANGE.end());
    }

    /// The edge colour a fill runs with: the one chosen, else the eraser's
    /// own. A generated fill (SD, RORem) meets the photo through its soft
    /// edge, and a colour shift over it brings back a lighter or darker
    /// rim; the others match along the edge.
    pub fn edge_colour(&self, generated: bool) -> prunr_core::inpaint_blend::EdgeColour {
        self.inpaint_edge_colour.unwrap_or_else(|| default_edge_colour(generated))
    }

    /// How Magic Brush reads the model's output under these settings.
    pub fn magic_reading(&self) -> prunr_core::MaskReading {
        prunr_core::MaskReading { confidence: self.magic_confidence_threshold, remove_specks: self.magic_remove_specks }
    }

    /// The values a stroke is drawn with, which its brush's sliders set.
    pub(crate) fn stroke_values(&self) -> StrokeValues {
        StrokeValues { size: self.radius, hardness: self.hardness, opacity: self.strength, reading: self.magic_reading(), range: self.magic_depth_range }
    }

    /// The settings that belong to a model rather than to the brush: the
    /// fill's Sharpen and scale, and the Stable Diffusion quality, prompt
    /// and seed. Expand region and Feather sit on the brush strips and
    /// mean the same in every eraser, so they stay as the user set them.
    pub fn take_model_tuning_from(&mut self, source: &Self) {
        self.inpaint_sharpen = source.inpaint_sharpen;
        self.inpaint_fill_scale = source.inpaint_fill_scale;
        self.inpaint_edge_colour = source.inpaint_edge_colour;
        self.sd_soft_edge = source.sd_soft_edge;
        self.sd_sampler = source.sd_sampler;
        self.take_sd_quality_from(source);
        self.sd_prompt = source.sd_prompt.clone();
        self.sd_negative_prompt = source.sd_negative_prompt.clone();
        self.sd_seed = source.sd_seed;
        self.sd_use_taesd = source.sd_use_taesd;
    }

    /// The eraser's region knobs: Expand region, Feather, Sharpen.
    pub fn take_inpaint_tuning_from(&mut self, source: &Self) {
        self.inpaint_sharpen = source.inpaint_sharpen;
        self.inpaint_feather = source.inpaint_feather;
        self.inpaint_grow = source.inpaint_grow;
        self.inpaint_fill_scale = source.inpaint_fill_scale;
    }

    /// The prompt a fill runs with: the user's, or the shipped one when
    /// the field is empty. Unconditioned SD 1.5 draws text-like glyphs on
    /// texture, so a fill never runs without one.
    pub fn sd_prompt_for_run(&self) -> String {
        let own = self.sd_prompt.trim();
        if own.is_empty() { DEFAULT_SD_PROMPT } else { own }.to_string()
    }

    /// A quality's sampler settings only: picking Fast or Quality keeps
    /// the prompt and seed.
    pub fn take_sd_quality_from(&mut self, source: &Self) {
        self.sd_scheduler = source.sd_scheduler;
        self.sd_steps = source.sd_steps;
        self.sd_guidance_scale = source.sd_guidance_scale;
        self.fit_sd_to_quality();
    }

    /// Steps and guidance inside what the quality is made for; a value
    /// saved under the other quality or a retired sampler can sit outside.
    pub fn fit_sd_to_quality(&mut self) {
        let steps = self.sd_scheduler.steps_range();
        let guidance = self.sd_scheduler.guidance_range();
        self.sd_steps = self.sd_steps.clamp(*steps.start(), *steps.end());
        self.sd_guidance_scale = self.sd_guidance_scale.clamp(*guidance.start(), *guidance.end());
    }

    /// Reset the brush popover's sliders and shape to `source`. `strength`
    /// and `mode` carry user intent across a reset and stay; SD tuning
    /// belongs to the SD popover.
    pub fn reset_popover_fields_from(&mut self, source: &Self) {
        self.radius = source.radius;
        self.hardness = source.hardness;
        self.shape = source.shape;
        self.take_inpaint_tuning_from(source);
    }
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 24.0,
            hardness: 0.7,
            strength: 1.0,
            mode: BrushMode::Add,
            cutout_action: CutoutAction::default(),
            shape: BrushShape::Circle,
            inpaint_sharpen: 0.6,
            inpaint_feather: default_feather(),
            inpaint_grow: default_grow(),
            inpaint_fill_scale: Default::default(),
            inpaint_edge_colour: None,
            sd_soft_edge: default_soft_edge(),
            sd_sampler: None,
            sd_prompt: String::new(),
            sd_negative_prompt: default_sd_negative_prompt(),
            sd_guidance_scale: default_cfg(),
            sd_scheduler: SdScheduler::default(),
            sd_steps: default_sd_steps(),
            rorem_steps: default_rorem_steps(),
            sd_seed: None,
            sd_use_taesd: None,
            edge_feather: 0.0,
            magic_confidence_threshold: default_magic_confidence_threshold(),
            magic_remove_specks: default_magic_remove_specks(),
            magic_depth: false,
            magic_depth_range: default_magic_depth_range(),
        }
    }
}

/// Screen-space stamps `(x, y, radius)` of an in-progress stroke, drawn
/// each frame as the trail. Paint and Magic Brush share it so both tools
/// render the same trail.
#[derive(Default)]
pub(crate) struct Trail(Vec<(f32, f32, f32)>);

impl Trail {
    /// Skips a stamp within half a radius of the previous one: a slow drag
    /// at 60+ Hz stacks near-identical stamps that paint the same pixels.
    pub fn push_spaced(&mut self, sx: f32, sy: f32, screen_radius: f32) {
        if let Some(&(px, py, _)) = self.0.last() {
            let (dx, dy) = (sx - px, sy - py);
            if dx * dx + dy * dy < (screen_radius * 0.5).max(1.0).powi(2) {
                return;
            }
        }
        self.0.push((sx, sy, screen_radius));
    }

    pub fn stamps(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.0.iter().copied()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

}

/// Mid-drag stroke: its path, drawn once committed, off the UI thread.
struct ActiveStroke {
    trail: Trail,
    path: PaintPath,
}

/// A Paint Brush stroke as it was drawn: its shape (pinned when it
/// began), each pointer sample's position and radius in source pixels,
/// and the Size it was drawn at. The stroke and a redraw at another Size
/// are both drawn by `render`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PaintPath {
    shape: BrushShape,
    /// A Line keeps its press and its latest sample.
    samples: Vec<(f32, f32, f32)>,
    size: f32,
}

impl PaintPath {
    /// The stroke on a `width` × `height` grid at `size`, its radii scaled
    /// from the Size it was drawn at, with `stamp`. Each sample paints the
    /// segment from the one before, since a fast drag samples the pointer
    /// far apart.
    pub(crate) fn render(&self, width: u32, height: u32, size: f32, stamp: Stamp) -> MaskArtifact {
        let mut grid = MaskArtifact::new_empty(width, height);
        let scale = size / self.size.max(f32::EPSILON);
        let mut last = None;
        for &(x, y, radius) in &self.samples {
            let r = radius * scale;
            match (self.shape, last.replace((x, y))) {
                (BrushShape::Square, None) => paint_square(&mut grid, x, y, r, stamp),
                (BrushShape::Square, Some((lx, ly))) => paint_square_line(&mut grid, lx, ly, x, y, r, stamp),
                (_, None) => paint_circle(&mut grid, x, y, r, stamp),
                (_, Some((lx, ly))) => paint_line(&mut grid, lx, ly, x, y, r, stamp),
            }
        }
        grid
    }
}

/// The slider values strokes are drawn with: the Paint Brush's Size,
/// Hardness and Opacity, Magic Brush's reading and depth Range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StrokeValues {
    pub(crate) size: f32,
    pub(crate) hardness: f32,
    pub(crate) opacity: f32,
    pub(crate) reading: prunr_core::MaskReading,
    pub(crate) range: f32,
}

/// A stroke as it was drawn, kept with its undo step so its brush's
/// sliders can draw it again: a Magic Brush pick, a depth pick or a
/// Paint Brush stroke, each with the values it has now.
#[derive(Clone)]
pub(crate) enum StrokeDrawn {
    Magic { output: std::sync::Arc<prunr_core::sam::SamDecoderOutput>, mode: BrushMode, reading: prunr_core::MaskReading },
    Depth { depth: std::sync::Arc<prunr_core::depth::DepthMap>, around: f32, mode: BrushMode, range: f32 },
    Paint { path: std::sync::Arc<PaintPath>, size: f32, stamp: Stamp },
}

impl StrokeDrawn {
    pub(crate) fn painted(path: PaintPath, stamp: Stamp) -> Self {
        let size = path.size;
        Self::Paint { path: std::sync::Arc::new(path), size, stamp }
    }

    /// The values it is drawn with; those its brush has no slider for
    /// read as `UNUSED`.
    pub(crate) fn values(&self) -> StrokeValues {
        match self {
            Self::Magic { reading, .. } => StrokeValues { reading: *reading, ..UNUSED },
            Self::Depth { range, .. } => StrokeValues { range: *range, ..UNUSED },
            Self::Paint { size, stamp, .. } => StrokeValues { size: *size, hardness: stamp.hardness, opacity: stamp.strength, ..UNUSED },
        }
    }

    /// The stroke drawn at `values`, keeping what made it (its mode, its
    /// pick, its path).
    pub(crate) fn at(&self, values: StrokeValues) -> Self {
        use std::sync::Arc;
        match self {
            Self::Magic { output, mode, .. } => Self::Magic { output: Arc::clone(output), mode: *mode, reading: values.reading },
            Self::Depth { depth, around, mode, .. } => Self::Depth { depth: Arc::clone(depth), around: *around, mode: *mode, range: values.range },
            Self::Paint { path, stamp, .. } => Self::Paint {
                path: Arc::clone(path),
                size: values.size,
                stamp: Stamp { hardness: values.hardness, strength: values.opacity, ..*stamp },
            },
        }
    }

    pub(crate) fn mask(&self, width: u32, height: u32) -> MaskArtifact {
        match self {
            Self::Magic { output, mode, reading } => prunr_core::sam::decode_to_mask_artifact(output, width, height, *reading, *mode),
            Self::Depth { depth, around, mode, range } => prunr_core::depth::select_by_depth(depth, width, height, *around, *range, *mode),
            Self::Paint { path, size, stamp } => path.render(width, height, *size, *stamp),
        }
    }
}

/// The values of the sliders a stroke's brush does not have.
const UNUSED: StrokeValues = StrokeValues {
    size: 0.0,
    hardness: 0.0,
    opacity: 0.0,
    reading: prunr_core::MaskReading { confidence: 0.0, remove_specks: false },
    range: 0.0,
};

impl StrokeValues {
    /// Each value whose slider moved from `from` to `to` where the slider
    /// went; the others as they are.
    pub(crate) fn moved(self, from: Self, to: Self) -> Self {
        Self {
            size: moved_to(self.size, from.size, to.size),
            hardness: moved_to(self.hardness, from.hardness, to.hardness),
            opacity: moved_to(self.opacity, from.opacity, to.opacity),
            reading: prunr_core::MaskReading {
                confidence: moved_to(self.reading.confidence, from.reading.confidence, to.reading.confidence),
                remove_specks: moved_to(self.reading.remove_specks, from.reading.remove_specks, to.reading.remove_specks),
            },
            range: moved_to(self.range, from.range, to.range),
        }
    }

    /// Each value moved by as much as its slider moved from `from` to
    /// `to`: Size by the same factor, the others by the same amount within
    /// their ranges, so strokes keep their differences.
    pub(crate) fn adjusted(self, from: Self, to: Self) -> Self {
        let offset = |v: f32, a: f32, b: f32, range: std::ops::RangeInclusive<f32>| (v + b - a).clamp(*range.start(), *range.end());
        Self {
            size: (self.size * to.size / from.size.max(f32::EPSILON)).clamp(*BRUSH_RADIUS_RANGE.start(), *BRUSH_RADIUS_RANGE.end()),
            hardness: offset(self.hardness, from.hardness, to.hardness, 0.0..=1.0),
            opacity: offset(self.opacity, from.opacity, to.opacity, 0.0..=1.0),
            reading: prunr_core::MaskReading {
                confidence: offset(self.reading.confidence, from.reading.confidence, to.reading.confidence, 0.0..=1.0),
                remove_specks: moved_to(self.reading.remove_specks, from.reading.remove_specks, to.reading.remove_specks),
            },
            range: offset(self.range, from.range, to.range, DEPTH_RANGE),
        }
    }
}

fn moved_to<T: PartialEq>(value: T, from: T, to: T) -> T {
    if from == to { value } else { to }
}

/// Which strokes the brushes' sliders adjust besides the next ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StrokeReach {
    Next,
    #[default]
    Last,
    All,
}

impl StrokeReach {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Next => "Next stroke",
            Self::Last => "Last stroke",
            Self::All => "All strokes",
        }
    }

    pub(crate) fn following(self) -> Self {
        match self {
            Self::Next => Self::Last,
            Self::Last => Self::All,
            Self::All => Self::Next,
        }
    }
}

#[derive(Default)]
pub(crate) struct BrushState {
    enabled: bool,
    active: Option<ActiveStroke>,
    /// Not saved: each start adjusts the latest stroke.
    pub(crate) reach: StrokeReach,
    /// The selected image and the slider values last frame, so a move is
    /// seen with where it came from, on that image only.
    pub(crate) seen: Option<(u64, StrokeValues)>,
}

/// A brush tool; the app keeps at most one on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tool {
    Paint,
    Magic,
}

impl BrushState {
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Force the brush off, so an enabled state from a previous mode does
    /// not leak into one without a brush (upscale).
    pub fn disable(&mut self) {
        self.enabled = false;
        self.active = None;
    }

    pub fn has_active_stroke(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin_stroke(&mut self, shape: BrushShape, size: f32) {
        self.active = Some(ActiveStroke {
            trail: Trail::default(),
            path: PaintPath { shape, samples: Vec::new(), size },
        });
    }

    pub fn active_shape(&self) -> Option<BrushShape> {
        self.active.as_ref().map(|a| a.path.shape)
    }

    /// Record a screen-space stamp for the stroke trail.
    pub fn record_trail_stamp(&mut self, sx: f32, sy: f32, screen_radius: f32) {
        if let Some(active) = self.active.as_mut() {
            active.trail.push_spaced(sx, sy, screen_radius);
        }
    }

    /// Empty when no stroke is active.
    pub fn trail_stamps(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.active
            .as_ref()
            .into_iter()
            .flat_map(|a| a.trail.stamps())
    }

    /// Extend the active stroke at model-space coordinates. Caller
    /// converts screen→model so screen-radius confusion can't reach
    /// the grid.
    pub fn extend_stroke_with_radius(&mut self, x: f32, y: f32, radius: f32) {
        let Some(active) = self.active.as_mut() else { return };
        let samples = &mut active.path.samples;
        if active.path.shape == BrushShape::Line && samples.len() == 2 {
            samples[1] = (x, y, radius);
        } else if samples.last() != Some(&(x, y, radius)) {
            samples.push((x, y, radius));
        }
    }

    /// The finished stroke's path; `None` when it was never extended.
    pub fn commit_stroke(&mut self) -> Option<PaintPath> {
        self.active.take().map(|active| active.path).filter(|path| !path.samples.is_empty())
    }

}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    /// What a saved brush with no fields at all loads as. Every value is
    /// spelled out, so a change to `Default` or to a field's serde default
    /// shows up here: note `inpaint_sharpen` loads as 0.0, not Default's 0.6.
    fn loaded_from_nothing() -> super::BrushSettings {
        use super::*;
        BrushSettings {
            radius: 24.0,
            hardness: 0.7,
            strength: 1.0,
            mode: BrushMode::Add,
            cutout_action: CutoutAction::Restore,
            shape: BrushShape::Circle,
            inpaint_sharpen: 0.0,
            inpaint_feather: 4.0,
            inpaint_grow: 2.0,
            inpaint_fill_scale: prunr_core::inpaint::FillScale::Auto,
            inpaint_edge_colour: None,
            sd_soft_edge: true,
            sd_sampler: None,
            sd_prompt: String::new(),
            sd_negative_prompt: DEFAULT_SD_NEGATIVE_PROMPT.to_string(),
            sd_guidance_scale: 1.5,
            sd_scheduler: SdScheduler::Lcm,
            sd_steps: 8,
            rorem_steps: 4,
            sd_seed: None,
            sd_use_taesd: None,
            edge_feather: 0.0,
            magic_confidence_threshold: 0.5,
            magic_remove_specks: true,
            magic_depth: false,
            magic_depth_range: 0.08,
        }
    }

    /// A saved brush that never chose an edge colour (saved before it
    /// existed too) runs each eraser's own: off for a generated fill, along
    /// the edge for the others; a chosen one holds for every eraser.
    #[test]
    fn an_unchosen_edge_colour_is_the_erasers_own() {
        use super::*;
        use prunr_core::inpaint_blend::EdgeColour;
        let old: BrushSettings = serde_json::from_str(r#"{"inpaint_feather": 22.0}"#).unwrap();
        assert_eq!((old.edge_colour(true), old.edge_colour(false)), (EdgeColour::Off, EdgeColour::AlongEdge));
        let chosen = BrushSettings { inpaint_edge_colour: Some(EdgeColour::WholeArea), ..old };
        let reloaded: BrushSettings = serde_json::from_str(&serde_json::to_string(&chosen).unwrap()).unwrap();
        assert_eq!((reloaded.edge_colour(true), reloaded.edge_colour(false)), (EdgeColour::WholeArea, EdgeColour::WholeArea));
    }

    /// Saved brush settings load the same whatever they leave out: an
    /// empty object, and an object missing any one field, which takes its
    /// value from `loaded_from_nothing` while the rest keep theirs.
    #[test]
    fn every_missing_brush_field_loads_as_before() {
        use super::*;
        let empty: BrushSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, loaded_from_nothing());

        let probe = BrushSettings {
            radius: 50.0,
            hardness: 0.3,
            strength: 0.5,
            mode: BrushMode::Subtract,
            cutout_action: CutoutAction::Erase,
            shape: BrushShape::Square,
            inpaint_sharpen: 0.9,
            inpaint_feather: 7.0,
            inpaint_grow: -3.0,
            inpaint_fill_scale: prunr_core::inpaint::FillScale::Half,
            inpaint_edge_colour: Some(prunr_core::inpaint_blend::EdgeColour::WholeArea),
            sd_soft_edge: false,
            sd_sampler: Some(SdScheduler::Lcm),
            sd_prompt: "mine".into(),
            sd_negative_prompt: "neg".into(),
            sd_guidance_scale: 6.0,
            sd_scheduler: SdScheduler::DpmPlusPlus2MKarras,
            sd_steps: 20,
            rorem_steps: 6,
            sd_seed: Some(7),
            sd_use_taesd: Some(false),
            edge_feather: 3.0,
            magic_confidence_threshold: 0.7,
            magic_remove_specks: false,
            magic_depth: true,
            magic_depth_range: 0.2,
        };
        let full = serde_json::to_value(&probe).unwrap();
        let nothing = serde_json::to_value(loaded_from_nothing()).unwrap();
        let fields = full.as_object().unwrap();
        assert_eq!(fields.len(), 26, "a new field joins this test");
        for key in fields.keys() {
            assert_ne!(full[key], nothing[key], "{key}: the probe must differ from the default");
            let mut partial = full.clone();
            partial.as_object_mut().unwrap().remove(key);
            let loaded: BrushSettings = serde_json::from_value(partial).unwrap();
            let mut expected = full.clone();
            expected[key] = nothing[key].clone();
            assert_eq!(serde_json::to_value(&loaded).unwrap(), expected, "missing {key}");
        }
    }

    #[test]
    fn modifiers_override_the_chosen_mode_for_a_stroke() {
        let mut b = super::BrushSettings::default();
        b.mode = super::BrushMode::Subtract;
        assert_eq!(b.mode_under(false, false, false), super::BrushMode::Subtract);
        assert_eq!(b.mode_under(false, true, false), super::BrushMode::Add);
        assert_eq!(b.mode_under(false, true, true), super::BrushMode::Add, "Shift wins over Alt");
        b.mode = super::BrushMode::Add;
        assert_eq!(b.mode_under(false, false, true), super::BrushMode::Subtract);
        assert_eq!(b.mode_under(true, false, false), super::BrushMode::Add, "on a cut-out a plain stroke adds");
        assert_eq!(b.mode_under(true, false, true), super::BrushMode::Subtract);
    }

    #[test]
    fn step_radius_moves_at_least_a_pixel_and_stays_in_range() {
        let mut b = super::BrushSettings::default();
        b.radius = 1.0;
        b.step_radius(true);
        assert_eq!(b.radius, 2.0, "a pixel at the small end");
        b.step_radius(false);
        b.step_radius(false);
        assert_eq!(b.radius, 1.0, "clamped at the floor");
        b.radius = 100.0;
        b.step_radius(true);
        assert_eq!(b.radius, 125.0, "a quarter of the radius");
        b.radius = 190.0;
        b.step_radius(true);
        assert_eq!(b.radius, 200.0, "clamped at the ceiling");
    }

    use super::*;

    #[test]
    fn trail_keeps_only_stamps_half_a_radius_apart() {
        let mut t = Trail::default();
        t.push_spaced(10.0, 10.0, 8.0);
        t.push_spaced(12.0, 10.0, 8.0); // 2 px < 4 px: dropped
        t.push_spaced(15.0, 10.0, 8.0); // 5 px from the first: kept
        assert_eq!(t.stamps().collect::<Vec<_>>(), vec![(10.0, 10.0, 8.0), (15.0, 10.0, 8.0)]);
        t.clear();
        assert!(t.stamps().next().is_none());
    }

    #[test]
    fn brush_settings_default_edge_feather_is_zero() {
        let s = BrushSettings::default();
        assert!((s.edge_feather - 0.0).abs() < f32::EPSILON,
            "edge_feather default must be 0 (sharp edges)");
    }

    #[test]
    fn brush_settings_default_magic_confidence_is_0_5() {
        let s = BrushSettings::default();
        assert!((s.magic_confidence_threshold - 0.5).abs() < f32::EPSILON,
            "magic_confidence_threshold default must be 0.5");
    }

    #[test]
    fn default_disabled() {
        let s = BrushState::default();
        assert!(!s.is_enabled());
        assert!(!s.has_active_stroke());
    }

    /// An empty prompt runs the shipped one, so the field and the chip
    /// show only what the user wrote.
    #[test]
    fn an_empty_prompt_runs_the_shipped_prompt() {
        let mut s = BrushSettings::default();
        assert!(s.sd_prompt.is_empty(), "ships empty");
        assert_eq!(s.sd_prompt_for_run(), DEFAULT_SD_PROMPT);
        s.sd_prompt = "  ".into();
        assert_eq!(s.sd_prompt_for_run(), DEFAULT_SD_PROMPT);
        s.sd_prompt = "a red brick wall".into();
        assert_eq!(s.sd_prompt_for_run(), "a red brick wall");
    }

    /// A saved copy of the shipped prompt is no prompt of the user's own.
    #[test]
    fn a_saved_shipped_prompt_loads_as_empty() {
        let json = format!(r#"{{"sd_prompt":"{DEFAULT_SD_PROMPT}"}}"#);
        let s: BrushSettings = serde_json::from_str(&json).unwrap();
        assert!(s.sd_prompt.is_empty());
        let s: BrushSettings = serde_json::from_str(r#"{"sd_prompt":"mine"}"#).unwrap();
        assert_eq!(s.sd_prompt, "mine");
        let p: crate::gui::presets::SdPreset = serde_json::from_str(&json.replace("sd_prompt", "prompt")).unwrap();
        assert!(p.prompt.is_empty());
    }

    /// Settings saved with a retired sampler open on Quality.
    #[test]
    fn retired_samplers_load_as_quality() {
        for old in ["\"Ddim\"", "\"UniPc\"", "\"EulerA\""] {
            let s: SdScheduler = serde_json::from_str(old).expect("retired sampler still loads");
            assert_eq!(s, SdScheduler::DpmPlusPlus2MKarras, "{old}");
        }
    }

    /// Each quality's shipped steps and guidance sit inside the slider it shows.
    #[test]
    fn each_quality_ships_steps_and_guidance_inside_its_range() {
        for q in SdScheduler::ALL {
            let b = crate::gui::presets::SdSchedulerBundle::default_for(q);
            assert!(q.steps_range().contains(&b.steps), "{q:?} steps");
            assert!(q.guidance_range().contains(&b.guidance_scale), "{q:?} guidance");
        }
        let d = BrushSettings::default();
        assert!(d.sd_scheduler.steps_range().contains(&d.sd_steps));
        assert!(d.sd_scheduler.guidance_range().contains(&d.sd_guidance_scale));
    }

    #[test]
    fn sd_use_taesd_default_is_none_auto() {
        let s = BrushSettings::default();
        assert_eq!(s.sd_use_taesd, None,
            "default must be None (auto-on when TAESD bundle installed)");
    }

    /// The fast decoder runs only when installed; left on auto it runs with
    /// Fast and not with Quality, whose point is the best fill; the user's
    /// choice wins otherwise.
    #[test]
    fn the_fast_decoder_follows_the_install_the_quality_and_the_choice() {
        use SdScheduler::{DpmPlusPlus2MKarras as Quality, Lcm as Fast};
        for (scheduler, choice, installed, runs) in [
            (Fast, None, true, true),
            (Fast, None, false, false),
            (Quality, None, true, false),
            (Fast, Some(true), true, true),
            (Fast, Some(true), false, false),
            (Fast, Some(false), true, false),
            (Fast, Some(false), false, false),
        ] {
            let s = BrushSettings { sd_scheduler: scheduler, sd_use_taesd: choice, ..BrushSettings::default() };
            assert_eq!(s.sd_use_taesd_effective_with_avail(installed), runs, "{scheduler:?} {choice:?} installed={installed}");
        }
    }

    #[test]
    fn enable_and_disable_flip_enabled() {
        let mut s = BrushState::default();
        s.enable();
        assert!(s.is_enabled());
        s.disable();
        assert!(!s.is_enabled());
    }

    fn default_stamp() -> Stamp {
        BrushSettings::default().stamp()
    }

    #[test]
    fn disabling_drops_the_active_stroke() {
        let mut s = BrushState::default();
        s.enable();
        s.begin_stroke(BrushShape::Circle, 8.0);
        s.extend_stroke_with_radius(32.0, 32.0, 8.0);
        assert!(s.has_active_stroke());
        s.disable();
        assert!(!s.has_active_stroke());
    }

    /// A fast drag samples the pointer far apart; the stroke must still
    /// be continuous between samples, not a row of dots.
    #[test]
    fn a_fast_stroke_is_continuous_between_samples() {
        for shape in [BrushShape::Circle, BrushShape::Square] {
            let mut s = BrushState::default();
            s.begin_stroke(shape, 2.0);
            s.extend_stroke_with_radius(4.0, 30.0, 2.0);
            s.extend_stroke_with_radius(60.0, 30.0, 2.0);
            let c = StrokeDrawn::painted(s.commit_stroke().expect("populated stroke"), default_stamp()).mask(64, 64);
            let gaps = (4..=60).filter(|&x| c.cells()[30 * 64 + x] == 0).count();
            assert_eq!(gaps, 0, "{shape:?} left {gaps} unpainted cells between the samples");
        }
    }

    #[test]
    fn extend_without_begin_is_no_op() {
        let mut s = BrushState::default();
        s.extend_stroke_with_radius(10.0, 10.0, 8.0);
        assert!(!s.has_active_stroke());
        assert!(s.commit_stroke().is_none());
    }

    #[test]
    fn empty_stroke_commit_returns_none() {
        let mut s = BrushState::default();
        s.begin_stroke(BrushShape::Circle, 8.0);
        // No extend_stroke calls — buffer stays empty.
        assert!(s.commit_stroke().is_none());
        assert!(!s.has_active_stroke(), "commit should clear active stroke");
    }

    #[test]
    fn populated_stroke_commit_returns_correction() {
        let mut s = BrushState::default();
        s.begin_stroke(BrushShape::Circle, 8.0);
        s.extend_stroke_with_radius(32.0, 32.0, 8.0);
        let c = StrokeDrawn::painted(s.commit_stroke().expect("populated stroke"), default_stamp()).mask(64, 64);
        assert_eq!(c.width, 64);
        assert_eq!(c.height, 64);
        assert!(c.has_selected_region());
        assert!(!s.has_active_stroke());
    }


    /// The eraser fills what the cursor covers, whatever Hardness says:
    /// a soft stamp cut at half coverage used to shrink the region to
    /// half the cursor at 0 %.
    #[test]
    fn the_eraser_region_reaches_the_cursor_edge_at_any_hardness() {
        use prunr_core::selection::MaskArtifact;
        let s = BrushSettings { hardness: 0.0, ..BrushSettings::default() };
        let mut plane = MaskArtifact::new_empty(101, 101);
        prunr_core::brush::paint_circle(&mut plane, 50.0, 50.0, 40.0, s.region_stamp(false));
        let region = plane.region_mask(101, 101);
        assert_eq!(region.get_pixel(50 + 37, 50)[0], 255, "near the rim is filled");
        assert_eq!(region.get_pixel(50 + 44, 50)[0], 0, "outside the cursor is not");
    }

    #[test]
    fn reset_popover_fields_from_default_restores_visible_sliders_and_shape_only() {
        let mut s = BrushSettings {
            radius: 80.0,
            hardness: 0.0,
            strength: 0.42,
            mode: BrushMode::Add,
            shape: BrushShape::Square,
            inpaint_sharpen: 1.7,
            inpaint_feather: 24.0,
            inpaint_grow: -8.0,
            inpaint_fill_scale: prunr_core::inpaint::FillScale::Quarter,
            inpaint_edge_colour: Some(prunr_core::inpaint_blend::EdgeColour::Off),
            sd_soft_edge: false,
            sd_sampler: Some(SdScheduler::Lcm),
            sd_prompt: "custom prompt".into(),
            sd_negative_prompt: "custom negative".into(),
            sd_guidance_scale: 5.5,
            sd_scheduler: SdScheduler::DpmPlusPlus2MKarras,
            sd_steps: 30,
            rorem_steps: 6,
            sd_seed: Some(42),
            sd_use_taesd: Some(true),
            edge_feather: 5.0,
            magic_confidence_threshold: 0.7,
            magic_remove_specks: false,
            magic_depth: true,
            magic_depth_range: 0.2,
            cutout_action: CutoutAction::Erase,
        };
        s.reset_popover_fields_from(&BrushSettings::default());

        let d = BrushSettings::default();
        assert_eq!(s.radius, d.radius);
        assert_eq!(s.hardness, d.hardness);
        assert_eq!(s.shape, d.shape);
        assert_eq!(s.inpaint_sharpen, d.inpaint_sharpen);
        assert_eq!(s.inpaint_feather, d.inpaint_feather);
        assert_eq!(s.inpaint_grow, d.inpaint_grow);

        // strength + mode carry user intent (seg pipeline + Add/Subtract).
        assert!((s.strength - 0.42).abs() < f32::EPSILON, "strength must survive popover reset");
        assert_eq!(s.mode, BrushMode::Add, "mode must survive popover reset");
        assert_eq!(s.cutout_action, CutoutAction::Erase, "the cut-out action must survive popover reset");

        // SD-tuning fields are owned by the SD chip popover.
        assert_eq!(s.sd_prompt, "custom prompt");
        assert_eq!(s.sd_negative_prompt, "custom negative");
        assert!((s.sd_guidance_scale - 5.5).abs() < f32::EPSILON);
        assert_eq!(s.sd_scheduler, SdScheduler::DpmPlusPlus2MKarras);
        assert_eq!(s.sd_steps, 30);
        assert_eq!(s.sd_seed, Some(42));
        assert!(s.magic_depth && s.magic_depth_range == 0.2, "Depth is a strip knob, not the popover's");
    }

    #[test]
    fn reset_popover_fields_from_uses_source_not_factory() {
        let source = BrushSettings {
            radius: 99.0,
            hardness: 0.1,
            shape: BrushShape::Square,
            inpaint_sharpen: 1.3,
            inpaint_feather: 7.0,
            inpaint_grow: -3.0,
            ..BrushSettings::default()
        };
        let mut target = BrushSettings::default();
        target.radius = 10.0;
        target.strength = 0.42;
        target.mode = BrushMode::Add;

        target.reset_popover_fields_from(&source);

        assert!((target.radius - 99.0).abs() < f32::EPSILON);
        assert!((target.hardness - 0.1).abs() < f32::EPSILON);
        assert_eq!(target.shape, BrushShape::Square);
        assert!((target.inpaint_sharpen - 1.3).abs() < f32::EPSILON);
        assert!((target.inpaint_feather - 7.0).abs() < f32::EPSILON);
        assert!((target.inpaint_grow - (-3.0)).abs() < f32::EPSILON);

        // strength + mode carry user intent across reset — must not change.
        assert!((target.strength - 0.42).abs() < f32::EPSILON);
        assert_eq!(target.mode, BrushMode::Add);
    }

    #[test]
    fn brush_settings_stamp_reflects_fields() {
        let s = BrushSettings { hardness: 0.3, strength: 0.8, mode: BrushMode::Add, ..BrushSettings::default() };
        let stamp = s.stamp();
        assert!((stamp.hardness - 0.3).abs() < f32::EPSILON);
        assert!((stamp.strength - 0.8).abs() < f32::EPSILON);
        assert_eq!(stamp.mode, BrushMode::Add);
    }
}
