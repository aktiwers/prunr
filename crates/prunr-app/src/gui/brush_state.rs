//! Brush tool coordinator. Owns the toolbar toggle, current size /
//! hardness / mode, and the in-progress stroke buffer. Doesn't iterate
//! `BatchManager.items` — the caller hands it the active grid size
//! and writes the committed strokes back via `BatchItem`'s mutator.

use prunr_core::brush::{paint_circle, paint_line, paint_square, BrushMode, BrushShape, MaskCorrection, Stamp};
use serde::{Deserialize, Serialize};

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
    pub mode: BrushMode,
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
    /// SD-only: text prompt for the inpainted region. Defaults to a
    /// generic "match surroundings + quality nudge" string suited to
    /// eraser use; user can rewrite in the brush popover. Empty =
    /// unconditional (high-variance output, often produces text-shape
    /// glyphs on textured regions — see `inpaint_sd` module docs).
    #[serde(default = "default_sd_prompt")]
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
    /// SD-only: which scheduler runs the denoise loop. LCM is the
    /// default — proven good after the LcmScheduler port; DDIM kept
    /// as a conservative baseline. Other variants gated by
    /// `is_available()` until they have a dispatch backend wired.
    #[serde(default = "default_sd_scheduler")]
    pub sd_scheduler: SdScheduler,
    /// SD-only: number of denoise steps. LCM ranges 1-8; standard SD
    /// 15-30. UI clamps the slider per scheduler.
    #[serde(default = "default_sd_steps")]
    pub sd_steps: u32,
    /// SD-only: use the Karras sigma noise schedule on top of the
    /// chosen scheduler. Disabled for LCM (which has its own fixed
    /// schedule).
    #[serde(default)]
    pub sd_use_karras_sigmas: bool,
    /// SD-only: pinned RNG seed for reproducibility. `None` = fresh
    /// random per stroke (the historical behavior).
    #[serde(default)]
    pub sd_seed: Option<u64>,
    /// SD-only: inpaint strength in [0, 1]. 1.0 = pure noise init,
    /// fully creative rewrite (default). Lower values preserve the
    /// original masked pixels proportionally — the dispatcher VAE-
    /// encodes the source, mixes with scaled noise, and skips the
    /// corresponding number of early denoise steps.
    #[serde(default = "default_strength")]
    pub sd_strength: f32,
}

/// SD eraser scheduler choice. Wired into `SdInpaintRequest` at
/// dispatch time so the worker picks the right denoise math.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SdScheduler {
    /// LCM-distilled multistep. 4-8 steps typical.
    Lcm,
    /// DDIM (Denoising Diffusion Implicit Models) — conservative
    /// baseline. 20-30 steps typical.
    Ddim,
    /// DPM-Solver++ 2M with Karras sigmas — modern Standard SD
    /// default in A1111 / ComfyUI / InvokeAI. 15-25 steps. Not yet
    /// wired to a dispatch backend; see `is_available()`.
    DpmPlusPlus2MKarras,
    /// UniPC predictor-corrector multistep — best quality at low step
    /// counts (8-12). Corrector solves a 2×2 system to compensate for
    /// the predictor's hardcoded rhos_p=0.5 approximation.
    UniPc,
    /// Euler-Ancestral — adds noise per step → creative variation per
    /// seed (non-deterministic).
    EulerA,
}

impl SdScheduler {
    /// Returns `false` for schedulers that don't have a dispatch
    /// backend wired yet. UI gates the dropdown on this so users
    /// can't pick something the worker can't run; dispatch should
    /// also gate as a defensive fallback.
    pub fn is_available(&self) -> bool {
        matches!(
            self,
            SdScheduler::Lcm
                | SdScheduler::Ddim
                | SdScheduler::DpmPlusPlus2MKarras
                | SdScheduler::EulerA
                | SdScheduler::UniPc,
        )
    }

    /// Short user-facing label for dropdowns. Matches A1111 conventions.
    pub fn label(&self) -> &'static str {
        match self {
            SdScheduler::Lcm => "LCM",
            SdScheduler::Ddim => "DDIM",
            SdScheduler::DpmPlusPlus2MKarras => "DPM++ 2M Karras",
            SdScheduler::UniPc => "UniPC",
            SdScheduler::EulerA => "Euler-A",
        }
    }
}

impl From<SdScheduler> for prunr_core::inpaint_sd::SchedulerKind {
    fn from(s: SdScheduler) -> Self {
        use prunr_core::inpaint_sd::SchedulerKind;
        match s {
            SdScheduler::Lcm => SchedulerKind::Lcm,
            SdScheduler::Ddim => SchedulerKind::Ddim,
            SdScheduler::DpmPlusPlus2MKarras => SchedulerKind::DpmPp2MKarras,
            SdScheduler::UniPc => SchedulerKind::UniPc,
            SdScheduler::EulerA => SchedulerKind::EulerA,
        }
    }
}

/// Built-in SD quality presets — bundles scheduler + steps + CFG +
/// Karras into one knob for users who don't want to tune individually.
/// Industry-standard pattern (A1111 / Lightroom / DaVinci): touching
/// any individual slider auto-switches the displayed preset to
/// `Custom`.
///
/// **Stateless.** The "active preset" is computed from the current
/// brush field values via `detect_from`. There's no persisted
/// `sd_quality_preset` — picking a preset writes the bundled values
/// into the individual fields and that's it. This eliminates the
/// "stored preset says Balanced but values say Custom" desync class
/// of bug.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SdQualityPreset {
    /// LCM @ 4 steps, CFG=1.0 — instant preview tier.
    Fast,
    /// LCM @ 8 steps, CFG=1.5 — current shipped default. Good balance
    /// of speed + quality on iGPU.
    Balanced,
    /// DPM++ 2M Karras @ 25 steps, CFG=4.0 — best quality. Requires
    /// the DPM++ scheduler to be available (`is_available()`); UI
    /// gates picking and dispatch falls back to LCM otherwise.
    Quality,
    /// Computed: current brush values don't match any built-in.
    /// Never an input to `apply_to` — it has no bundled config.
    Custom,
}

impl SdQualityPreset {
    /// Apply this preset's bundled scheduler + steps + CFG + Karras
    /// to a `BrushSettings`. `Custom` is a true no-op — it has no
    /// bundled config; callers shouldn't call `apply_to(Custom)` and
    /// the implementation does nothing if they do.
    pub fn apply_to(self, brush: &mut BrushSettings) {
        match self {
            SdQualityPreset::Fast => {
                brush.sd_scheduler = SdScheduler::Lcm;
                brush.sd_steps = 4;
                brush.sd_guidance_scale = 1.0;
                brush.sd_use_karras_sigmas = false;
            }
            SdQualityPreset::Balanced => {
                brush.sd_scheduler = SdScheduler::Lcm;
                brush.sd_steps = 8;
                brush.sd_guidance_scale = 1.5;
                brush.sd_use_karras_sigmas = false;
            }
            SdQualityPreset::Quality => {
                brush.sd_scheduler = SdScheduler::DpmPlusPlus2MKarras;
                brush.sd_steps = 25;
                brush.sd_guidance_scale = 4.0;
                brush.sd_use_karras_sigmas = true;
            }
            SdQualityPreset::Custom => {}
        }
    }

    /// Detect which preset (if any) the current `BrushSettings`
    /// match. Returns `Custom` when the values don't fit any
    /// built-in. Used after individual-slider edits to update the
    /// preset dropdown's displayed value.
    pub fn detect_from(brush: &BrushSettings) -> SdQualityPreset {
        const EPS: f32 = 1e-3;
        let cfg_eq = |a: f32, b: f32| (a - b).abs() < EPS;
        if brush.sd_scheduler == SdScheduler::Lcm
            && brush.sd_steps == 4
            && cfg_eq(brush.sd_guidance_scale, 1.0)
            && !brush.sd_use_karras_sigmas
        {
            SdQualityPreset::Fast
        } else if brush.sd_scheduler == SdScheduler::Lcm
            && brush.sd_steps == 8
            && cfg_eq(brush.sd_guidance_scale, 1.5)
            && !brush.sd_use_karras_sigmas
        {
            SdQualityPreset::Balanced
        } else if brush.sd_scheduler == SdScheduler::DpmPlusPlus2MKarras
            && brush.sd_steps == 25
            && cfg_eq(brush.sd_guidance_scale, 4.0)
            && brush.sd_use_karras_sigmas
        {
            SdQualityPreset::Quality
        } else {
            SdQualityPreset::Custom
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            SdQualityPreset::Fast => "Fast",
            SdQualityPreset::Balanced => "Balanced",
            SdQualityPreset::Quality => "Quality",
            SdQualityPreset::Custom => "Custom",
        }
    }
}

fn default_feather() -> f32 { 4.0 }
fn default_grow() -> f32 { 2.0 }
/// 1.5 matches the `Balanced` preset's CFG (LCM scheduler, CFG up to
/// 2.0 per Diffusers LCM guidance — community consensus is values
/// >2.0 degrade LCM output quality). For Standard SD via DDIM /
/// DPM++ the user can bump to 4.0–7.5 via the toolbar slider; the
/// `Quality` preset auto-fills 4.0 when picked.
pub(crate) fn default_cfg() -> f32 { 1.5 }

/// Defaults for SD eraser prompts. Style-agnostic — works whether the
/// source is a photo, drawing, anime, pixel art, render, etc. (avoiding
/// `"photorealistic"` which would bias non-photo sources). Conservative
/// on the positive (let surroundings dominate; reinforce eraser intent
/// via "seamless continuation"); aggressive on the negative (block
/// SD-1.5 failure modes, especially text-shape glyphs which fire on
/// weak conditioning).
pub(crate) const DEFAULT_SD_PROMPT: &str =
    "seamless continuation of the surroundings, matching style, high quality";
pub(crate) const DEFAULT_SD_NEGATIVE_PROMPT: &str =
    "text, letters, words, watermark, signature, logo, blurry, distorted, \
     low quality, oversaturated, jpeg artifacts";
fn default_sd_prompt() -> String { DEFAULT_SD_PROMPT.to_string() }
fn default_sd_negative_prompt() -> String { DEFAULT_SD_NEGATIVE_PROMPT.to_string() }

fn default_sd_scheduler() -> SdScheduler { SdScheduler::Lcm }
fn default_sd_steps() -> u32 { 8 }
fn default_strength() -> f32 { 1.0 }

impl BrushSettings {
    pub fn stamp(&self) -> Stamp {
        Stamp { hardness: self.hardness, strength: self.strength, mode: self.mode }
    }
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 24.0,
            hardness: 0.7,
            strength: 1.0,
            mode: BrushMode::Subtract,
            shape: BrushShape::Circle,
            inpaint_sharpen: 0.6,
            inpaint_feather: default_feather(),
            inpaint_grow: default_grow(),
            sd_prompt: default_sd_prompt(),
            sd_negative_prompt: default_sd_negative_prompt(),
            sd_guidance_scale: default_cfg(),
            sd_scheduler: default_sd_scheduler(),
            sd_steps: default_sd_steps(),
            sd_use_karras_sigmas: false,
            sd_seed: None,
            sd_strength: default_strength(),
        }
    }
}

/// Mid-drag stroke buffer at the active item's model resolution.
struct ActiveStroke {
    grid: MaskCorrection,
    /// Set the first time the stamp runs against `grid`. Lets
    /// `commit_stroke` skip an O(W·H) is_empty scan on click-without-drag.
    dirty: bool,
    /// Screen-space stamps painted so far. Drawn each frame as the
    /// in-progress trail until the stroke commits.
    trail: Vec<(f32, f32, f32)>,
    /// Stroke-time snapshot of the brush shape — pinned at begin_stroke
    /// so a mid-stroke shape switch doesn't desync the grid.
    shape: BrushShape,
    /// Line-tool state: present iff `shape == Line`. Tracks the press +
    /// most recent positions so `commit_stroke` can paint one segment.
    line: Option<LineState>,
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

    /// True while the user is mid-drag.
    pub fn has_active_stroke(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin_stroke(&mut self, width: u16, height: u16, shape: BrushShape) {
        self.active = Some(ActiveStroke {
            grid: MaskCorrection::empty(width, height),
            dirty: false,
            trail: Vec::new(),
            shape,
            line: None,
        });
    }

    pub fn active_shape(&self) -> Option<BrushShape> {
        self.active.as_ref().map(|a| a.shape)
    }

    /// Record a screen-space stamp for the in-progress stroke trail.
    /// Caller is the canvas-side overlay; `BrushState` doesn't compute
    /// screen coords itself.
    ///
    /// Spatial dedup: skip the push if the new stamp is within half-radius
    /// of the previous one. Pointer events at 60+ Hz over a slow drag
    /// stack near-identical stamps that paint to the same pixels — keeping
    /// only every-half-radius cuts the per-frame paint count without
    /// changing what the user sees.
    pub fn record_trail_stamp(&mut self, sx: f32, sy: f32, screen_radius: f32) {
        let Some(active) = self.active.as_mut() else { return };
        if let Some(&(px, py, _)) = active.trail.last() {
            let dx = sx - px;
            let dy = sy - py;
            let min_step_sq = (screen_radius * 0.5).max(1.0).powi(2);
            if dx * dx + dy * dy < min_step_sq {
                return;
            }
        }
        active.trail.push((sx, sy, screen_radius));
    }

    /// Iterator over `(sx, sy, screen_radius)` stamps in the active
    /// stroke's trail. Empty when no stroke is active.
    pub fn trail_stamps(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.active
            .as_ref()
            .into_iter()
            .flat_map(|a| a.trail.iter().copied())
    }

    /// Extend the active stroke at model-space coordinates. Caller
    /// converts screen→model so screen-radius confusion can't reach
    /// the grid. Line strokes wait for `commit_stroke` to paint.
    pub fn extend_stroke_with_radius(&mut self, x: f32, y: f32, radius: f32, stamp: Stamp) {
        let Some(active) = self.active.as_mut() else { return };
        match active.shape {
            BrushShape::Circle => {
                paint_circle(&mut active.grid, x, y, radius, stamp);
                active.dirty = true;
            }
            BrushShape::Square => {
                paint_square(&mut active.grid, x, y, radius, stamp);
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

    pub fn commit_stroke(&mut self, stamp: Stamp) -> Option<MaskCorrection> {
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
mod tests {
    use super::*;

    #[test]
    fn default_disabled() {
        let s = BrushState::default();
        assert!(!s.is_enabled());
        assert!(!s.has_active_stroke());
    }

    /// `BrushSettings::default()` must round-trip cleanly through
    /// `SdQualityPreset::detect_from` to `Balanced` — i.e. default
    /// values match the Balanced preset's bundled config. If they
    /// drift, new users see "Custom" in the dropdown on first launch
    /// which is misleading.
    #[test]
    fn default_brush_settings_match_balanced_preset() {
        let s = BrushSettings::default();
        assert_eq!(SdQualityPreset::detect_from(&s), SdQualityPreset::Balanced,
            "default BrushSettings must align with Balanced preset");
    }

    /// Each preset's `apply_to` must produce a `BrushSettings` that
    /// `detect_from` round-trips back to the same preset. Pins the
    /// auto-fill ↔ detection invariant so a future tweak to one side
    /// can't silently break the other.
    #[test]
    fn quality_presets_round_trip_apply_then_detect() {
        for preset in [SdQualityPreset::Fast, SdQualityPreset::Balanced, SdQualityPreset::Quality] {
            let mut s = BrushSettings::default();
            preset.apply_to(&mut s);
            assert_eq!(SdQualityPreset::detect_from(&s), preset,
                "apply→detect mismatch for {preset:?}");
        }
    }

    /// `apply_to(Custom)` is a true no-op — Custom has no bundled
    /// config, so no fields are mutated. Pinned because the
    /// alternative interpretation (silently set a "preset field") was
    /// flagged as ambiguous in review.
    #[test]
    fn apply_to_custom_is_noop() {
        let mut s = BrushSettings::default();
        let before = s.clone();
        SdQualityPreset::Custom.apply_to(&mut s);
        assert_eq!(s, before, "apply_to(Custom) must not mutate any field");
    }

    /// Touching any individual SD knob away from the active preset's
    /// bundled values must mark the preset as `Custom`. This is the
    /// industry-standard pattern (Lightroom, A1111) — the user's
    /// tweak is preserved without bouncing them between presets.
    #[test]
    fn individual_slider_edit_detects_as_custom() {
        let mut s = BrushSettings::default();
        // Default == Balanced (LCM, 8 steps, CFG=1.5, no Karras).
        // Bump CFG to a non-preset value:
        s.sd_guidance_scale = 3.0;
        assert_eq!(SdQualityPreset::detect_from(&s), SdQualityPreset::Custom,
            "off-preset CFG must detect as Custom");

        // Change scheduler away from the preset:
        let mut s = BrushSettings::default();
        s.sd_scheduler = SdScheduler::Ddim;
        assert_eq!(SdQualityPreset::detect_from(&s), SdQualityPreset::Custom,
            "off-preset scheduler must detect as Custom");

        // Toggle Karras (Balanced has it off; flipping on → Custom):
        let mut s = BrushSettings::default();
        s.sd_use_karras_sigmas = true;
        assert_eq!(SdQualityPreset::detect_from(&s), SdQualityPreset::Custom,
            "off-preset Karras toggle must detect as Custom");
    }

    /// `is_available()` reflects which schedulers have a dispatch
    /// backend wired today. UI gates the dropdown on this; dispatch
    /// should also gate as a defensive fallback.
    #[test]
    fn scheduler_availability_reflects_dispatch_readiness() {
        assert!(SdScheduler::Lcm.is_available());
        assert!(SdScheduler::Ddim.is_available());
        assert!(SdScheduler::DpmPlusPlus2MKarras.is_available());
        assert!(SdScheduler::EulerA.is_available());
        assert!(SdScheduler::UniPc.is_available());
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
        assert!(!s.sd_use_karras_sigmas);
        assert_eq!(s.sd_seed, None);
        // The persisted CFG=4.0 is kept; semantically that means the
        // user's effective preset is Custom (CFG mismatch with all
        // built-ins). The UI reads detect_from on every render so
        // this is reported correctly to the dropdown.
        assert_eq!(SdQualityPreset::detect_from(&s), SdQualityPreset::Custom);
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
        assert!(!c.is_empty());
        assert!(!s.has_active_stroke());
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
