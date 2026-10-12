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
    /// Stroke magnitude. 1.0 = "neutral / equivalent to a gamma push";
    /// lower values give a gentler local effect (`m → m * (1 - s)` for
    /// subtract). Decoupled from hardness so users can fine-tune
    /// intensity without changing edge softness.
    pub strength: f32,
    /// Add to or take from the selection; on a cut-out a stroke adds and
    /// Alt/Ctrl takes away, and `cutout_action` says what the selection does.
    pub mode: BrushMode,
    pub cutout_action: CutoutAction,
    pub shape: BrushShape,
    /// Eraser-only: post-process unsharp-mask amount applied inside the
    /// inpainted region. 0.0 = LaMa output as-is (slightly soft, the
    /// model's bias). Higher values sharpen against the model's blur.
    /// Practical sweet spot: 0.3-0.7. Pinned to BrushSettings (global)
    /// rather than ItemSettings since it's a render preference, not
    /// per-image content.
    #[serde(default)]
    pub inpaint_sharpen: f32,
    /// Eraser-only: pixels of soft blend at the painted-region boundary.
    /// Hides the LaMa↔source seam by gradually mixing the model's output
    /// toward source within `feather_px` of the mask edge.
    #[serde(default = "default_feather")]
    pub inpaint_feather: f32,
    /// Eraser-only: dilate (>0) or erode (<0) the painted mask before
    /// inference. Small dilation gives LaMa more context for cleaner
    /// fills; erosion preserves more original detail at the boundary.
    #[serde(default = "default_grow")]
    pub inpaint_grow: f32,
    /// SD-only: the user's own prompt; empty runs `DEFAULT_SD_PROMPT`.
    #[serde(default, deserialize_with = "own_prompt")]
    pub sd_prompt: String,
    /// SD-only: negative prompt — content to push *away* from.
    /// Defaults to the standard SD-1.5 anti-failure-mode list (text,
    /// watermark, blur, etc.) which suppresses the most common
    /// artifacts on weakly-conditioned regions.
    #[serde(default = "default_sd_negative_prompt")]
    pub sd_negative_prompt: String,
    /// SD-only: classifier-free guidance scale. 1.0 = no CFG (cond pass
    /// only). 7.5 is the typical SD-1.5 prompt strength. >1 doubles UNet
    /// cost per step (cond + uncond passes blended).
    #[serde(default = "default_cfg")]
    pub sd_guidance_scale: f32,
    /// SD-only: the quality (sampler); Fast is the default.
    #[serde(default = "default_sd_scheduler")]
    pub sd_scheduler: SdScheduler,
    /// SD-only: number of denoise steps, within `SdScheduler::steps_range`.
    #[serde(default = "default_sd_steps")]
    pub sd_steps: u32,
    /// SD-only: pinned RNG seed for reproducibility. `None` = fresh
    /// random per stroke (the historical behavior).
    #[serde(default)]
    pub sd_seed: Option<u64>,
    /// SD-only: TAESD fast VAE preference. `None` = auto (on when
    /// the TAESD bundle is installed). `Some(false)` = explicit
    /// opt-out. `Some(true)` = explicit opt-in (still gated by
    /// install at dispatch time). Orthogonal to scheduler — works
    /// with both standard SD and LCM checkpoints.
    #[serde(default)]
    pub sd_use_taesd: Option<bool>,
    /// Shared selection knob: edge feather in pixels, 0 = sharp. Applied to
    /// what the overlay shows and to what Delete / Copy / Cut act on.
    #[serde(default)]
    pub edge_feather: f32,
    /// Magic Brush only: the probability a pixel must reach to be
    /// selected (0.5 is the model's decision boundary). Lives in shared
    /// BrushSettings like the other per-tool knobs.
    #[serde(default = "default_magic_confidence_threshold")]
    pub magic_confidence_threshold: f32,
    /// Magic Brush only: drop specks and fill pinholes in what the model
    /// returns.
    #[serde(default = "default_magic_remove_specks")]
    pub magic_remove_specks: bool,
}

/// The SD eraser's quality: which sampler runs the fill. Wired into
/// `SdInpaintRequest` at dispatch time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SdScheduler {
    /// Fast: the LCM-distilled model, a few steps.
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
            SdScheduler::Lcm => "a few seconds per fill",
            SdScheduler::DpmPlusPlus2MKarras => "the best fill, several times slower",
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
fn default_grow() -> f32 { prunr_core::inpaint_blend::DEFAULT_GROW_PX }
fn default_magic_confidence_threshold() -> f32 { 0.5 }
fn default_magic_remove_specks() -> bool { true }
fn default_cfg() -> f32 { crate::gui::presets::SdSchedulerBundle::default_for(default_sd_scheduler()).guidance_scale }

/// Defaults for SD eraser prompts. Style-agnostic — works whether the
/// source is a photo, drawing, anime, pixel art, render, etc. (avoiding
/// `"photorealistic"` which would bias non-photo sources). Conservative
/// on the positive (let surroundings dominate; reinforce eraser intent
/// via "seamless continuation"); aggressive on the negative (block
/// SD-1.5 failure modes, especially text-shape glyphs which fire on
/// weak conditioning).
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
fn default_sd_negative_prompt() -> String { DEFAULT_SD_NEGATIVE_PROMPT.to_string() }

fn default_sd_scheduler() -> SdScheduler { SdScheduler::Lcm }
fn default_sd_steps() -> u32 { crate::gui::presets::SdSchedulerBundle::default_for(default_sd_scheduler()).steps }

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

    /// Reset the brush popover's slider knobs (radius / hardness /
    /// inpaint_grow / inpaint_feather / inpaint_sharpen) and shape to
    /// the values carried by `source`. Leaves `strength` and `mode`
    /// alone — those carry user intent across reset (the Add/Subtract
    /// toggle and seg-mode strength) — and SD-tuning fields are owned
    /// by the SD chip popover.
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

    /// How Magic Brush reads the model's output under these settings.
    pub fn magic_reading(&self) -> prunr_core::MaskReading {
        prunr_core::MaskReading { confidence: self.magic_confidence_threshold, remove_specks: self.magic_remove_specks }
    }

    /// The settings that belong to a model rather than to the brush: the
    /// eraser's region knobs and the Stable Diffusion quality, prompt and seed.
    pub fn take_model_tuning_from(&mut self, source: &Self) {
        self.take_inpaint_tuning_from(source);
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
            sd_prompt: String::new(),
            sd_negative_prompt: default_sd_negative_prompt(),
            sd_guidance_scale: default_cfg(),
            sd_scheduler: default_sd_scheduler(),
            sd_steps: default_sd_steps(),
            sd_seed: None,
            sd_use_taesd: None,
            edge_feather: 0.0,
            magic_confidence_threshold: default_magic_confidence_threshold(),
            magic_remove_specks: default_magic_remove_specks(),
        }
    }
}

/// Screen-space stamps `(x, y, radius)` of an in-progress stroke, drawn
/// each frame as the trail. Paint and Magic Brush share it so both tools
/// render the same trail.
#[derive(Default)]
pub(crate) struct Trail(Vec<(f32, f32, f32)>);

impl Trail {
    /// Skips the stamp when it lies within half a radius of the previous
    /// one: pointer events at 60+ Hz over a slow drag stack near-identical
    /// stamps that paint the same pixels, and dropping them cuts the
    /// per-frame paint count without changing what the user sees.
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

/// Mid-drag stroke buffer at the active item's model resolution.
struct ActiveStroke {
    grid: MaskArtifact,
    /// Set the first time the stamp runs against `grid`. Lets
    /// `commit_stroke` skip an O(W·H) any-nonzero scan on click-without-drag.
    dirty: bool,
    trail: Trail,
    /// Stroke-time snapshot of the brush shape — pinned at begin_stroke
    /// so a mid-stroke shape switch doesn't desync the grid.
    shape: BrushShape,
    /// Line-tool state: present iff `shape == Line`. Tracks the press +
    /// most recent positions so `commit_stroke` can paint one segment.
    line: Option<LineState>,
    /// The last Circle / Square sample: the next one paints the segment
    /// from here, since a fast drag samples the pointer far apart.
    last: Option<(f32, f32)>,
}

#[derive(Clone, Copy, Debug)]
struct LineState {
    first: (f32, f32),
    last: (f32, f32),
    radius: f32,
}

#[derive(Default)]
pub(crate) struct BrushState {
    enabled: bool,
    active: Option<ActiveStroke>,
}

impl BrushState {
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if !self.enabled {
            self.active = None;
        }
    }

    /// Force the brush off. Used when the active model changes to a
    /// mode that has no brush surface (upscale) so a stray enabled
    /// state from a previous mode doesn't leak through.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.active = None;
    }

    /// True while the user is mid-drag.
    pub fn has_active_stroke(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin_stroke(&mut self, width: u32, height: u32, shape: BrushShape) {
        self.active = Some(ActiveStroke {
            grid: MaskArtifact::new_empty(width, height),
            dirty: false,
            trail: Trail::default(),
            shape,
            line: None,
            last: None,
        });
    }

    pub fn active_shape(&self) -> Option<BrushShape> {
        self.active.as_ref().map(|a| a.shape)
    }

    /// Record a screen-space stamp for the in-progress stroke trail.
    /// Caller is the canvas-side overlay; `BrushState` doesn't compute
    /// screen coords itself.
    pub fn record_trail_stamp(&mut self, sx: f32, sy: f32, screen_radius: f32) {
        if let Some(active) = self.active.as_mut() {
            active.trail.push_spaced(sx, sy, screen_radius);
        }
    }

    /// Iterator over `(sx, sy, screen_radius)` stamps in the active
    /// stroke's trail. Empty when no stroke is active.
    pub fn trail_stamps(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.active
            .as_ref()
            .into_iter()
            .flat_map(|a| a.trail.stamps())
    }

    /// Extend the active stroke at model-space coordinates. Caller
    /// converts screen→model so screen-radius confusion can't reach
    /// the grid. Line strokes wait for `commit_stroke` to paint.
    pub fn extend_stroke_with_radius(&mut self, x: f32, y: f32, radius: f32, stamp: Stamp) {
        let Some(active) = self.active.as_mut() else { return };
        match active.shape {
            BrushShape::Circle => {
                match active.last.replace((x, y)) {
                    Some((lx, ly)) => paint_line(&mut active.grid, lx, ly, x, y, radius, stamp),
                    None => paint_circle(&mut active.grid, x, y, radius, stamp),
                }
                active.dirty = true;
            }
            BrushShape::Square => {
                match active.last.replace((x, y)) {
                    Some((lx, ly)) => paint_square_line(&mut active.grid, lx, ly, x, y, radius, stamp),
                    None => paint_square(&mut active.grid, x, y, radius, stamp),
                }
                active.dirty = true;
            }
            BrushShape::Line => {
                let entry = active.line.get_or_insert(LineState {
                    first: (x, y),
                    last: (x, y),
                    radius,
                });
                entry.last = (x, y);
                entry.radius = radius;
            }
        }
    }

    pub fn commit_stroke(&mut self, stamp: Stamp) -> Option<MaskArtifact> {
        let mut active = self.active.take()?;
        if let Some(line) = active.line {
            paint_line(
                &mut active.grid,
                line.first.0, line.first.1,
                line.last.0, line.last.1,
                line.radius,
                stamp,
            );
            active.dirty = true;
        }
        if !active.dirty {
            return None;
        }
        Some(active.grid)
    }

}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
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

    /// Old persisted JSON without the new SD-tuning fields must
    /// deserialize cleanly into the new defaults — no panic. Pins
    /// serde back-compat for users carrying older settings.json files.
    #[test]
    fn brush_settings_serde_back_compat_old_json_without_new_fields() {
        let old_json = r#"{
            "radius": 24.0,
            "hardness": 0.7,
            "strength": 1.0,
            "mode": "Subtract",
            "shape": "Circle",
            "inpaint_sharpen": 0.6,
            "inpaint_feather": 4.0,
            "inpaint_grow": 2.0,
            "sd_prompt": "x",
            "sd_negative_prompt": "y",
            "sd_guidance_scale": 4.0
        }"#;
        let s: BrushSettings = serde_json::from_str(old_json).unwrap();
        // Missing fields fill with serde defaults:
        assert_eq!(s.sd_scheduler, SdScheduler::Lcm);
        assert_eq!(s.sd_steps, 8);
        assert_eq!(s.sd_seed, None);
        assert_eq!(s.sd_use_taesd, None);
        assert!((s.sd_guidance_scale - 4.0).abs() < f32::EPSILON);
    }

    /// Forward-compat tripwire: a JSON literal with only `{"radius":24}`
    /// must deserialize into a fully-defaulted `BrushSettings`. Preset
    /// files produced by older binaries that lack SD-tuning fields must
    /// keep loading.
    #[test]
    fn brush_settings_loads_with_only_radius() {
        let s: BrushSettings = serde_json::from_str(r#"{"radius":24}"#)
            .expect("BrushSettings must deserialize from a single-field JSON literal");
        assert!((s.radius - 24.0).abs() < f32::EPSILON);
        assert_eq!(s.sd_use_taesd, None);
        assert_eq!(s.sd_seed, None);
        assert_eq!(s.sd_steps, 8);
        assert_eq!(s.sd_scheduler, SdScheduler::Lcm);
        assert!((s.sd_guidance_scale - 1.5).abs() < f32::EPSILON);
        assert!(s.sd_prompt.is_empty());
        assert_eq!(s.sd_negative_prompt, DEFAULT_SD_NEGATIVE_PROMPT);
        assert!((s.inpaint_feather - 4.0).abs() < f32::EPSILON);
        assert!((s.inpaint_grow - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sd_use_taesd_default_is_none_auto() {
        let s = BrushSettings::default();
        assert_eq!(s.sd_use_taesd, None,
            "default must be None (auto-on when TAESD bundle installed)");
    }

    #[test]
    fn sd_use_taesd_serde_round_trip() {
        let mut s = BrushSettings::default();
        s.sd_use_taesd = Some(true);
        let json = serde_json::to_string(&s).unwrap();
        let restored: BrushSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.sd_use_taesd, Some(true));

        s.sd_use_taesd = Some(false);
        let json = serde_json::to_string(&s).unwrap();
        let restored: BrushSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.sd_use_taesd, Some(false));
    }

    #[test]
    fn sd_use_taesd_effective_explicit_off_beats_install() {
        let mut s = BrushSettings::default();
        s.sd_use_taesd = Some(false);
        assert!(!s.sd_use_taesd_effective_with_avail(true),
            "explicit opt-out must override installed bundle");
        assert!(!s.sd_use_taesd_effective_with_avail(false));
    }

    #[test]
    fn sd_use_taesd_effective_explicit_on_gated_by_install() {
        let mut s = BrushSettings::default();
        s.sd_use_taesd = Some(true);
        assert!(s.sd_use_taesd_effective_with_avail(true),
            "explicit opt-in + installed must dispatch TAESD");
        assert!(!s.sd_use_taesd_effective_with_avail(false),
            "explicit opt-in but not installed must fall back");
    }

    /// Left on auto, the fast decoder runs with Fast and not with Quality,
    /// whose point is the best fill.
    #[test]
    fn the_fast_decoder_follows_the_quality_until_set() {
        let mut s = BrushSettings { sd_scheduler: SdScheduler::DpmPlusPlus2MKarras, ..BrushSettings::default() };
        assert!(!s.sd_use_taesd_effective_with_avail(true), "Quality: off");
        s.sd_scheduler = SdScheduler::Lcm;
        assert!(s.sd_use_taesd_effective_with_avail(true), "Fast: on");
        s.sd_use_taesd = Some(false);
        assert!(!s.sd_use_taesd_effective_with_avail(true), "the user's choice wins");
    }

    #[test]
    fn sd_use_taesd_effective_auto_follows_install() {
        let mut s = BrushSettings::default();
        s.sd_use_taesd = None;
        assert!(s.sd_use_taesd_effective_with_avail(true),
            "auto mode + installed = on");
        assert!(!s.sd_use_taesd_effective_with_avail(false),
            "auto mode + not installed = off");
    }

    #[test]
    fn toggle_flips_enabled() {
        let mut s = BrushState::default();
        s.toggle();
        assert!(s.is_enabled());
        s.toggle();
        assert!(!s.is_enabled());
    }

    fn default_stamp() -> Stamp {
        BrushSettings::default().stamp()
    }

    #[test]
    fn toggle_off_drops_active_stroke() {
        let mut s = BrushState::default();
        s.toggle();
        s.begin_stroke(64, 64, BrushShape::Circle);
        s.extend_stroke_with_radius(32.0, 32.0, 8.0, default_stamp());
        assert!(s.has_active_stroke());
        s.toggle();
        assert!(!s.has_active_stroke());
    }

    /// A fast drag samples the pointer far apart; the stroke must still
    /// be continuous between samples, not a row of dots.
    #[test]
    fn a_fast_stroke_is_continuous_between_samples() {
        for shape in [BrushShape::Circle, BrushShape::Square] {
            let mut s = BrushState::default();
            s.begin_stroke(64, 64, shape);
            s.extend_stroke_with_radius(4.0, 30.0, 2.0, default_stamp());
            s.extend_stroke_with_radius(60.0, 30.0, 2.0, default_stamp());
            let c = s.commit_stroke(default_stamp()).expect("populated stroke");
            let gaps = (4..=60).filter(|&x| c.cells()[30 * 64 + x] == 0).count();
            assert_eq!(gaps, 0, "{shape:?} left {gaps} unpainted cells between the samples");
        }
    }

    #[test]
    fn extend_without_begin_is_no_op() {
        let mut s = BrushState::default();
        s.extend_stroke_with_radius(10.0, 10.0, 8.0, default_stamp());
        assert!(!s.has_active_stroke());
        assert!(s.commit_stroke(default_stamp()).is_none());
    }

    #[test]
    fn empty_stroke_commit_returns_none() {
        let mut s = BrushState::default();
        s.begin_stroke(64, 64, BrushShape::Circle);
        // No extend_stroke calls — buffer stays empty.
        assert!(s.commit_stroke(default_stamp()).is_none());
        assert!(!s.has_active_stroke(), "commit should clear active stroke");
    }

    #[test]
    fn populated_stroke_commit_returns_correction() {
        let mut s = BrushState::default();
        s.begin_stroke(64, 64, BrushShape::Circle);
        s.extend_stroke_with_radius(32.0, 32.0, 8.0, default_stamp());
        let c = s.commit_stroke(default_stamp()).expect("populated stroke");
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
            sd_prompt: "custom prompt".into(),
            sd_negative_prompt: "custom negative".into(),
            sd_guidance_scale: 5.5,
            sd_scheduler: SdScheduler::DpmPlusPlus2MKarras,
            sd_steps: 30,
            sd_seed: Some(42),
            sd_use_taesd: Some(true),
            edge_feather: 5.0,
            magic_confidence_threshold: 0.7,
            magic_remove_specks: false,
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
