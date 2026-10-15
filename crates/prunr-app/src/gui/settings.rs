use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use prunr_core::ModelKind;

use super::brush_state::BrushSettings;
use super::item_settings::ItemSettings;

/// Name of the built-in, non-deletable, non-overwritable preset representing
/// factory defaults. Always present in the preset dropdown; the synthetic
/// values are `ItemSettings::default()` regardless of what's in the map.
pub const PRUNR_PRESET: &str = "Prunr";

/// Global app config. Per-image knobs (gamma, contrast, line mode, bg, ...)
/// live on `BatchItem.settings: ItemSettings` instead. New images inherit
/// whichever preset `default_preset` points at; the adjustments toolbar
/// edits the current image's `ItemSettings` directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: SettingsModel,
    /// Start processing each image automatically on import. Field is
    /// `#[serde(alias = ...)]` for compatibility with pre-v2 settings
    /// files that used the old `auto_remove_on_import` name.
    #[serde(alias = "auto_remove_on_import")]
    pub auto_process_on_import: bool,
    pub parallel_jobs: usize,
    /// Canvas transparency checkerboard uses dark tones instead of light.
    #[serde(default)]
    pub dark_checker: bool,
    /// How strongly a selection is tinted on the image, 0..1.
    #[serde(default = "default_selection_fill")]
    pub selection_fill: f32,
    /// Whether a selection is outlined on the image.
    #[serde(default = "default_true")]
    pub selection_outline: bool,
    /// Reveals, glows and moving highlights; off shows each result at once.
    #[serde(default = "default_true")]
    pub animations: bool,
    /// Outline the faces found while the Faces strip is open.
    #[serde(default = "default_true")]
    pub face_outlines: bool,
    /// Auto-hide the adjustments toolbar when the cursor leaves it.
    #[serde(default)]
    pub auto_hide_adjustments: bool,
    /// Drag-out and Save emit subject/lines/mask PNGs instead of a single
    /// composite. Single toggle, both export paths.
    #[serde(default)]
    pub export_split_layers: bool,
    /// Save also writes each image's depth map as `<stem>-depth.png`.
    #[serde(default)]
    pub export_depth_map: bool,
    /// File format Save writes (the save dialog's extension can override it).
    #[serde(default)]
    pub export_format: prunr_core::ExportFormat,
    /// Named presets (excluding the synthetic "Prunr"). Populated from the
    /// filesystem store at load time; not written to settings.json on save —
    /// each preset lives as its own file in `~/.config/prunr/presets/` so
    /// users can share them by dropping JSON files into the folder.
    /// The `default` + `skip_serializing` pair lets older builds that
    /// embedded presets in settings.json still deserialize cleanly (the
    /// load path migrates them out on first run).
    #[serde(default, skip_serializing)]
    pub(crate) presets: HashMap<String, super::presets::PresetFile>,
    /// Which preset new imports inherit, and what Reset All Knobs restores.
    /// Always set; defaults to "Prunr". If the named preset goes missing
    /// (user deleted it out of serde state), resolution falls back to "Prunr".
    #[serde(default = "default_preset_name")]
    pub default_preset: String,

    /// `bg_image_hash` → on-disk path map. Lets a preset that captured a
    /// bg image reload the image when re-applied (or applied to another
    /// item). Sharing a preset to another user with no matching path is a
    /// graceful miss — the bg image clears and the user is toasted.
    /// Bytes themselves are never persisted; the path is the identity.
    #[serde(default)]
    pub bg_image_paths: HashMap<u64, PathBuf>,

    /// Persisted set of model ids whose restrictive licenses (CreativeML
    /// Open RAIL-M, NVIDIA SCL, …) the user has explicitly accepted.
    /// Stored as `ModelId` Debug names so prunr-models stays serde-free.
    /// Once accepted, the Model Store skips the license dialog on
    /// subsequent re-downloads of the same model.
    #[serde(default)]
    pub accepted_licenses: Vec<String>,

    /// Per-runtime snooze: unix-second timestamp before which we won't
    /// re-prompt the user to install. Set when the user clicks "Not now"
    /// on the first-launch runtime prompt. Cleared on a successful
    /// install or via "Reset all". Keyed by `RuntimeId` Debug name.
    #[serde(default)]
    pub runtime_prompt_snoozed_until: std::collections::HashMap<String, i64>,

    #[serde(default)]
    pub processor: ComputeMode,
    #[serde(skip)]
    pub active_backend: String,
    #[serde(default)]
    pub brush: BrushSettings,


    /// Hotkey overrides by action name ("Undo"), chords in the form
    /// "Mod+Shift+Z". Only the actions the user changed; an empty list
    /// unbinds the action.
    #[serde(default)]
    pub hotkeys: BTreeMap<String, Vec<String>>,

    /// Keep the Stable Diffusion eraser resident between strokes (about
    /// 16 GB) so each one after the first skips the session build.
    #[serde(default)]
    pub keep_sd_loaded: bool,

    /// How much free memory the SD eraser leaves for other apps.
    #[serde(default)]
    pub sd_memory: SdMemoryGuard,
}

/// The SD eraser's memory guard: the headroom its RAM gate adds on top
/// of the model's need.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SdMemoryGuard {
    /// About 1 GB: fits SD on a tight machine; other apps may be squeezed.
    Relaxed,
    #[default]
    Balanced,
    /// About 4 GB: for machines where other apps grab memory mid-run.
    Strict,
}

impl SdMemoryGuard {
    pub fn margin_mb(self) -> u64 {
        let balanced = prunr_core::inpaint_sd::SD_DEFAULT_MARGIN_MB;
        match self {
            Self::Relaxed => balanced / 2,
            Self::Balanced => balanced,
            Self::Strict => balanced * 2,
        }
    }

    /// The choice nearest a margin saved by builds that had a GB slider.
    fn from_margin_gb(gb: f64) -> Self {
        if gb < 1.5 { Self::Relaxed } else if gb < 3.0 { Self::Balanced } else { Self::Strict }
    }
}

impl Settings {
    /// Should the SD inpaint dispatcher route to the LCM checkpoint?
    /// True only when ALL of: user picked SD, scheduler is LCM, the
    /// LCM descriptor is in the registry (artifact published), and
    /// the bundle is downloaded. When any clause fails, raw_backend
    /// passes through and standard SD runs. Pinned in one place so
    /// adding a 5th clause (e.g. license-accepted) doesn't silently
    /// diverge between dispatch and UI gating.
    pub fn lcm_routing_active(&self, raw_backend: prunr_models::ModelId) -> bool {
        use crate::gui::brush_state::SdScheduler;
        raw_backend == prunr_models::ModelId::SdV15InpaintFp16
            && self.brush.sd_scheduler == SdScheduler::Lcm
            && Self::can_select_lcm_scheduler()
    }

    /// Can the user pick the LCM scheduler entry in the scheduler
    /// dropdown? True only when the LCM bundle is published in the
    /// registry AND downloaded. Single definition site so adding a
    /// future clause (e.g. license-accepted) lands in one place.
    pub fn can_select_lcm_scheduler() -> bool {
        prunr_models::descriptor(prunr_models::ModelId::SdV15LcmInpaintFp16).is_some()
            && prunr_models::is_available(prunr_models::ModelId::SdV15LcmInpaintFp16)
    }
}

fn default_true() -> bool { true }
fn default_selection_fill() -> f32 { 0.30 }
fn default_preset_name() -> String { PRUNR_PRESET.to_string() }

impl Settings {
    /// The brush erases the cut-out, so its strokes and marks are drawn
    /// in Erase's warm red.
    pub(crate) fn brush_erases(&self) -> bool {
        self.model.erases_with(self.brush.cutout_action)
    }

    /// Config file path: ~/.config/prunr/settings.json (Linux),
    /// ~/Library/Application Support/prunr/settings.json (macOS),
    /// %APPDATA%/prunr/settings.json (Windows).
    pub fn config_path() -> Option<PathBuf> {
        config_root().map(|d| d.join("prunr").join("settings.json"))
    }

    /// Load from disk, falling back to defaults if missing or corrupt.
    ///
    /// Presets come from the filesystem store (`~/.config/prunr/presets/`),
    /// one JSON file per preset — shareable by dropping a file into the
    /// folder. Anything still in settings.json from older builds gets
    /// migrated to files on this load and then cleared from the JSON.
    ///
    /// Migrates v1 per-image fields (mask_gamma, bg_color, line_mode, ...)
    /// into a "Previous defaults" preset file on first load.
    pub fn load() -> Self {
        // Seed curated built-in presets on first run (idempotent via marker
        // file). Runs before any load_all() so users see "Comic", "Neon"
        // etc. in the dropdown on first launch.
        super::presets_fs::seed_builtins_once();

        let Some(path) = Self::config_path() else { return Self::default() };
        let Ok(data) = std::fs::read_to_string(&path) else {
            // No settings.json, but the user may still have preset files
            // dropped into the folder. Pick those up.
            return Self {
                presets: super::presets_fs::load_all(),
                ..Self::default()
            };
        };

        // Parse to Value first so we can migrate v1 fields regardless of
        // whether strict struct parsing succeeds.
        let mut value: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(%e, "settings.json is corrupt; falling back to defaults");
                return Self::default();
            }
        };
        super::item_settings::migrate_old_item_settings(&mut value);
        migrate_sd_margin(&mut value);

        let mut settings: Self = match serde_json::from_value(value.clone()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "settings.json is corrupt; falling back to defaults");
                return Self::default();
            }
        };
        settings.brush.fit_sd_to_quality();

        // v1 migration: stash old per-image fields as a "Previous defaults"
        // preset. User can apply it explicitly if they want their old values
        // back; new imports still start with factory defaults.
        let is_v1 = value.get("mask_gamma").is_some()
            || value.get("apply_bg_color").is_some()
            || value.get("line_mode").is_some();
        const LEGACY_PRESET: &str = "Previous defaults";
        // Pre-existing presets (from this load path or the embedded-in-
        // settings.json migration of yore) are now PresetFile-shaped, so
        // skip_serializing drops them from this binary's settings.json
        // and this branch only fires for the first launch after a real
        // v1 → v2 settings.json upgrade.
        if is_v1 && !settings.presets.contains_key(LEGACY_PRESET) {
            let legacy_item = migrate_v1_item_settings(&value);
            if legacy_item != ItemSettings::default() {
                let file = super::presets::PresetFile::single(DEFAULT_PRESET_MODEL, super::presets::ModelPreset {
                    item_settings: legacy_item,
                    brush: Default::default(),
                    sd: None,
                });
                let _ = super::presets_fs::save(LEGACY_PRESET, &file);
                settings.presets.insert(LEGACY_PRESET.to_string(), file);
            }
        }

        // Embedded-presets-in-settings.json migration (from a pre-2025
        // build) is no longer reachable: the field is now
        // HashMap<String, PresetFile> and any old embedded payload
        // serde-defaults to empty. Users on that vintage hit `load_all`
        // below for their actual preset files.
        let mut loaded = super::presets_fs::load_all();
        // Preserve the freshly-migrated "Previous defaults" entry above
        // (load_all picks it up too if save succeeded, but be defensive).
        for (k, v) in settings.presets.drain() {
            loaded.entry(k).or_insert(v);
        }
        settings.presets = loaded;
        settings
    }

    /// Save to disk. Errors are silently ignored (best-effort).
    pub fn save(&self) {
        let Some(path) = Self::config_path() else { return };
        self.save_to_path(&path);
    }

    /// Path-injectable save (used by tests; production calls `save()`).
    /// Best-effort: any I/O error is swallowed so the GUI keeps running.
    fn save_to_path(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Whether an accelerator backend is active (OpenVINO counts, though it
    /// may run on the host CPU: `runs_on_host_cpu`). Settings offers the
    /// Processor choice only then.
    pub fn is_gpu(&self) -> bool {
        !self.active_backend.is_empty() && !self.active_backend.eq_ignore_ascii_case("CPU")
    }

    /// Record the provider a worker reports. Under "CPU only" the report
    /// says CPU by choice, so the detected accelerator is kept (Settings
    /// still offers Auto). A new provider re-derives the job count.
    pub fn note_backend(&mut self, provider: String) {
        if provider.is_empty() || self.cpu_only() || self.active_backend == provider {
            return;
        }
        self.active_backend = provider;
        self.parallel_jobs = self.default_jobs();
    }

    pub fn cpu_only(&self) -> bool {
        self.processor == ComputeMode::CpuOnly
    }

    /// The provider the models run on: the detected one, or the CPU
    /// under "CPU only".
    pub fn effective_backend(&self) -> &str {
        if self.cpu_only() { "CPU" } else { &self.active_backend }
    }

    /// Whether models run on the host CPU: "CPU only", no accelerator, or
    /// OpenVINO, which the app runs on the CPU.
    pub fn runs_on_host_cpu(&self) -> bool {
        prunr_core::engine::runs_on_host_cpu(self.effective_backend()) || self.active_backend.is_empty()
    }

    /// The provider a session of `model` lands on: the one models run on,
    /// or the CPU when that accelerator does not run this model.
    pub fn model_backend(&self, model: prunr_core::ModelKind) -> &str {
        let backend = self.effective_backend();
        let accelerated = !backend.is_empty() && !backend.eq_ignore_ascii_case("CPU");
        if accelerated && prunr_core::engine::model_runs_on(model.model_id(), backend) { backend } else { "CPU" }
    }

    /// Smart default for parallel jobs based on backend and model. In
    /// filter-only mode (`SettingsModel::None`) the pipeline runs pure CPU
    /// on already-decoded pixels, so the ORT-memory cap doesn't apply.
    pub fn default_jobs(&self) -> usize {
        let base = if self.runs_on_host_cpu() { (num_cpus::get() / 2).max(1) } else { 2 };
        match self.model.to_model_kind() {
            Some(model) => base.min(super::memory::safe_max_jobs(model)),
            None => base,
        }
    }

    /// Max recommended parallel jobs based on backend and model.
    pub fn max_jobs(&self) -> usize {
        let base = if self.runs_on_host_cpu() { num_cpus::get() } else { 4 };
        match self.model.to_model_kind() {
            Some(model) => base.min(super::memory::safe_max_jobs(model)),
            None => base,
        }
    }

    /// A preset's image settings for the active model; factory values for
    /// "Prunr", a missing preset or a missing entry. Cheap: no brush clone.
    pub fn preset_values(&self, name: &str) -> ItemSettings {
        self.preset_file(name)
            .and_then(|f| super::presets::entry_for(f, self.preset_model()))
            .map(|mp| mp.item_settings)
            .unwrap_or_default()
    }

    /// A preset's image and brush settings for the active model: what
    /// applying it, or Reset to it, sets. Falls back to factory values at
    /// every level.
    pub(crate) fn resolve_preset(
        &self,
        name: &str,
        scheduler: Option<super::brush_state::SdScheduler>,
    ) -> super::presets::ResolvedView {
        let empty = super::presets::PresetFile::default();
        let file = self.preset_file(name).unwrap_or(&empty);
        super::presets::resolve_preset_for_model(file, self.preset_model(), scheduler)
    }

    fn preset_file(&self, name: &str) -> Option<&super::presets::PresetFile> {
        (name != PRUNR_PRESET).then(|| self.presets.get(name)).flatten()
    }

    /// The model a preset entry is read for; No model reads the default
    /// model's entry (its filters and lines are the same knobs).
    pub(crate) fn preset_model(&self) -> prunr_models::ModelId {
        self.model.to_model_id().unwrap_or(DEFAULT_PRESET_MODEL)
    }

    /// Resolve the active preset to the live `ResolvedView` for the
    /// active model. `scheduler = None` uses the preset's stored
    /// `active_scheduler` for SD; `Some(_)` overrides — used by the
    /// scheduler-change-within-SD handler so a bundle swap doesn't
    /// rewrite the preset on disk.
    ///
    /// Falls back to Prunr (factory defaults) at every level: missing
    /// preset, missing model entry, missing scheduler bundle.
    pub(crate) fn resolve_active_preset(
        &self,
        scheduler: Option<super::brush_state::SdScheduler>,
    ) -> super::presets::ResolvedView {
        self.resolve_preset(&self.default_preset, scheduler)
    }

    /// Take the new model's tuning from the active preset after the model
    /// dropdown flips: the eraser's region knobs and the Stable Diffusion
    /// settings. What the user set for the brush itself (size, hardness,
    /// opacity, mode, feather, the selection look, Magic Brush) stays.
    pub(crate) fn on_model_change_resolve_brush(&mut self) {
        let resolved = self.resolve_active_preset(None).brush;
        self.brush.take_model_tuning_from(&resolved);
    }

    /// Switch the eraser to `scheduler`'s steps and guidance from the
    /// active preset. The prompt, seed and brush stay as the user set them.
    pub(crate) fn on_scheduler_change_resolve_sd(
        &mut self,
        scheduler: super::brush_state::SdScheduler,
    ) {
        let resolved = self.resolve_active_preset(Some(scheduler));
        self.brush.take_sd_quality_from(&resolved.brush);
    }

    /// Fast needs its own download; without it the eraser runs Quality.
    pub(crate) fn fit_sd_quality_to_install(&mut self, lcm_installed: bool) {
        use super::brush_state::SdScheduler;
        if self.brush.sd_scheduler == SdScheduler::Lcm && !lcm_installed {
            self.on_scheduler_change_resolve_sd(SdScheduler::DpmPlusPlus2MKarras);
        }
    }

    /// ItemSettings template for new batch items — the default_preset's
    /// values, with any `PRUNR_*` knob env-var overrides applied on top
    /// (test-harness only; production env is empty).
    pub fn item_defaults_for_new_item(&self) -> ItemSettings {
        let mut item = self.preset_values(&self.default_preset);
        super::env_overrides::apply_to_item_settings(&mut item);
        item
    }

    pub fn has_accepted_license(&self, id: prunr_models::ModelId) -> bool {
        let key = super::presets::model_id_key(id);
        self.accepted_licenses.iter().any(|s| s == &key)
    }

    /// Records license acceptance and persists settings. Idempotent — a
    /// repeated call is a no-op (no duplicate entries, no extra disk write
    /// when the entry already exists).
    pub fn accept_license(&mut self, id: prunr_models::ModelId) {
        let key = super::presets::model_id_key(id);
        if !self.accepted_licenses.iter().any(|s| s == &key) {
            self.accepted_licenses.push(key);
            self.save();
        }
    }

    pub fn is_runtime_prompt_snoozed(&self, runtime: crate::runtime_install::RuntimeId) -> bool {
        self.runtime_prompt_snoozed_until
            .get(runtime.settings_key())
            .copied().unwrap_or(0) > now_unix_secs()
    }

    pub fn snooze_runtime_prompt(&mut self, runtime: crate::runtime_install::RuntimeId, days: i64) {
        let until = now_unix_secs() + days * 24 * 3600;
        self.runtime_prompt_snoozed_until.insert(runtime.settings_key().to_string(), until);
        self.save();
    }

    /// Reset every preference to its default, keeping what the user owns
    /// or agreed to: presets and the default one, licence acceptances,
    /// background pictures, the model in use and the OpenVINO prompt
    /// snooze. `parallel_jobs` re-derives for the detected hardware.
    pub fn reset_preserving_identity(&mut self) {
        let kept = std::mem::take(self);
        *self = Self {
            active_backend: kept.active_backend,
            presets: kept.presets,
            default_preset: kept.default_preset,
            accepted_licenses: kept.accepted_licenses,
            bg_image_paths: kept.bg_image_paths,
            model: kept.model,
            runtime_prompt_snoozed_until: kept.runtime_prompt_snoozed_until,
            ..Self::default()
        };
        self.parallel_jobs = self.default_jobs();
    }
}

/// The folder the app's settings and presets live under: the platform
/// config dir, or under test a folder of this test process, so a test
/// that saves never overwrites the user's settings or presets.
pub(crate) fn config_root() -> Option<PathBuf> {
    #[cfg(test)]
    {
        Some(std::env::temp_dir().join(format!("prunr-test-config-{}", std::process::id())))
    }
    #[cfg(not(test))]
    {
        dirs::config_dir()
    }
}

/// A saved GB margin becomes the nearest memory guard.
fn migrate_sd_margin(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else { return };
    let Some(gb) = obj.remove("ram_safety_margin_gb").and_then(|v| v.as_f64()) else { return };
    obj.entry("sd_memory").or_insert_with(|| serde_json::to_value(SdMemoryGuard::from_margin_gb(gb)).unwrap_or_default());
}

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Extract v1 per-image settings from raw JSON into ItemSettings.
/// Each field is tolerant of missing/wrong-type values (falls back to default).
fn migrate_v1_item_settings(v: &serde_json::Value) -> ItemSettings {
    use prunr_core::LineMode;

    let mut s = ItemSettings::default();
    if let Some(x) = v.get("mask_gamma").and_then(|x| x.as_f64()) { s.gamma = x as f32; }
    if v.get("mask_threshold_enabled").and_then(|x| x.as_bool()) == Some(true) {
        s.contrast = ItemSettings::HARD_EDGE;
    }
    if let Some(x) = v.get("edge_shift").and_then(|x| x.as_f64()) { s.edge_shift = x as f32; }
    if let Some(x) = v.get("refine_edges").and_then(|x| x.as_bool()) { s.refine_edges = x; }
    if let Some(mode) = v.get("line_mode").and_then(|x| x.as_str()) {
        s.line_mode = match mode {
            "Off" => LineMode::Off,
            "LinesOnly" | "EdgesOnly" => LineMode::EdgesOnly,
            "AfterBgRemoval" | "SubjectOutline" => LineMode::SubjectOutline,
            _ => LineMode::Off,
        };
    }
    if let Some(x) = v.get("line_strength").and_then(|x| x.as_f64()) { s.line_strength = x as f32; }
    if v.get("solid_line_color").and_then(|x| x.as_bool()) == Some(true) {
        if let Some(arr) = v.get("line_color").and_then(|x| x.as_array()) {
            if let [r, g, b, ..] = arr.as_slice() {
                let to_u8 = |x: &serde_json::Value| x.as_u64().unwrap_or(0).min(255) as u8;
                s.solid_line_color = Some([to_u8(r), to_u8(g), to_u8(b)]);
            }
        }
    }
    if v.get("apply_bg_color").and_then(|x| x.as_bool()) == Some(true) {
        if let Some(arr) = v.get("bg_color").and_then(|x| x.as_array()) {
            if let [r, g, b, a, ..] = arr.as_slice() {
                let to_u8 = |x: &serde_json::Value| x.as_u64().unwrap_or(0).min(255) as u8;
                s.bg = Some([to_u8(r), to_u8(g), to_u8(b), to_u8(a)]);
            }
        }
    }
    s
}

/// Which hardware runs the models. Auto: a GPU provider where the model
/// supports one, else the CPU. CpuOnly: every model on the CPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComputeMode {
    #[default]
    Auto,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SettingsModel {
    Silueta,
    U2net,
    BiRefNetLite,
    /// "Filter-only" mode — no segmentation inference. The pipeline bypasses
    /// the seg stage entirely and just applies fill_style / bg_effect to the
    /// source. TEED still runs if `line_mode == EdgesOnly` (edges don't
    /// need the seg model); SubjectOutline is invalid without a seg model
    /// and gets greyed out in the UI.
    None,
    /// Object-removal / inpaint mode (LaMa-fp32 backend). Brush is the
    /// only input; stroke release runs the inpainter over the painted
    /// region. Bg-removal knobs are inert in this mode.
    Inpaint,
    /// Same mode as `Inpaint` but routed to the higher-quality Big-LaMa
    /// weights. Same architecture, same per-tile cost — different
    /// training data.
    BigInpaint,
    /// Lightweight GAN inpainter (~25 MB). Different architecture from
    /// LaMa: GAN-based, sharper on detail, less smooth on flat
    /// backgrounds.
    MiganInpaint,
    /// Stable Diffusion 1.5 Inpainting (FP16). GPU-required, ~2 GB
    /// multi-part bundle. Generative — produces plausible content
    /// rather than smooth fills.
    SdInpaint,
    /// RORem (SDXL tuned for object removal). Fills large holes with
    /// background instead of drawing something new; ~5.5 GB bundle.
    RoremInpaint,
    /// Real-ESRGAN x4plus upscaler. General-purpose 4× super-resolution.
    RealEsrganUpscale,
    /// 4xNomos8kSCHAT-L HAT-L upscaler (fp16). Photo-tuned; higher quality
    /// than Real-ESRGAN, slower on CPU EP.
    Nomos8kUpscale,
    /// 4x NMKD Siax-CX (RRDB-23, fp32). Community-trained universal
    /// upscaler — sharper than x4plus on clean / mildly-compressed photos.
    FourXNmkdSiaxCxUpscale,
    /// 4x NMKD Superscale (RRDB-23, fp32). Community-trained restoration
    /// upscaler — handles noise and JPEG artifacts on real photos.
    FourXNmkdSuperscaleUpscale,
    /// 4xNomos2 RealPLKSR. Fast, natural texture on skin and hair.
    Nomos2Upscale,
    /// DDColor: colours a black-and-white photo.
    ColorizeDdcolor,
}

impl SettingsModel {
    /// All variants in display order — source of truth for the model dropdown.
    pub const ALL: [Self; 15] = [
        Self::Silueta,
        Self::U2net,
        Self::BiRefNetLite,
        Self::None,
        Self::Inpaint,
        Self::BigInpaint,
        Self::MiganInpaint,
        Self::SdInpaint,
        Self::RoremInpaint,
        Self::RealEsrganUpscale,
        Self::Nomos8kUpscale,
        Self::FourXNmkdSiaxCxUpscale,
        Self::FourXNmkdSuperscaleUpscale,
        Self::Nomos2Upscale,
        Self::ColorizeDdcolor,
    ];

    /// This model's position in `ALL`.
    pub fn index(self) -> usize {
        // every variant is listed in ALL
        Self::ALL.iter().position(|m| *m == self).unwrap_or(0)
    }

    /// Parse a Debug-style name ("Silueta", "U2net", "BiRefNetLite", …) into
    /// a variant. Used by the harness escape hatch (PRUNR_OPEN_MODEL env
    /// var) to force the initial model without a full settings.json.
    /// `None` on unknown / typo input — caller leaves settings.model alone.
    pub fn from_debug_name(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|m| format!("{m:?}") == s)
    }

    /// Whether this variant resolves to an ORT seg model. Filter-only and
    /// Inpaint variants skip segmentation entirely.
    pub fn uses_segmentation(self) -> bool {
        matches!(self, Self::Silueta | Self::U2net | Self::BiRefNetLite)
    }

    /// True for any object-removal mode. Branches that drive the inpaint
    /// pipeline (brush auto-enable, popover simplification) check this.
    pub fn is_inpaint(self) -> bool {
        matches!(self, Self::Inpaint | Self::BigInpaint | Self::MiganInpaint | Self::SdInpaint | Self::RoremInpaint)
    }

    /// An eraser that generates its fill from a seed, so "Try another"
    /// can fill the same region again differently.
    pub fn generates(self) -> bool {
        matches!(self, Self::SdInpaint | Self::RoremInpaint)
    }

    /// True for any super-resolution upscale mode. Dispatch routes through
    /// `upscale_rgba`; these variants are not seg/inpaint capacity-priced.
    pub fn is_upscale(self) -> bool {
        matches!(
            self,
            Self::RealEsrganUpscale
                | Self::Nomos8kUpscale
                | Self::FourXNmkdSiaxCxUpscale
                | Self::FourXNmkdSuperscaleUpscale
                | Self::Nomos2Upscale
        )
    }

    pub fn is_colorize(self) -> bool {
        self == Self::ColorizeDdcolor
    }

    /// What this model does to the image, which decides what it shows.
    pub fn role(self) -> crate::gui::knob_catalog::ModelRole {
        use crate::gui::knob_catalog::ModelRole;
        if self.uses_segmentation() {
            ModelRole::Cutout
        } else if self.is_inpaint() {
            ModelRole::Erase
        } else if self.is_upscale() {
            ModelRole::Upscale
        } else if self.is_colorize() {
            ModelRole::Colorize
        } else {
            ModelRole::Filters
        }
    }

    /// The selection erases the cut-out under `action`.
    pub fn erases_with(self, action: super::brush_state::CutoutAction) -> bool {
        self.edits_cutout() && action == super::brush_state::CutoutAction::Erase
    }

    /// See `ModelRole::edits_cutout`.
    pub fn edits_cutout(self) -> bool {
        self.role().edits_cutout()
    }

    /// Convert to `ModelKind`, or `None` for variants outside the
    /// seg memory-cap model (Inpaint and Upscale both return `None`).
    pub fn to_model_kind(self) -> Option<ModelKind> {
        match self {
            Self::Silueta => Some(ModelKind::Silueta),
            Self::U2net => Some(ModelKind::U2net),
            Self::BiRefNetLite => Some(ModelKind::BiRefNetLite),
            Self::None
            | Self::Inpaint
            | Self::BigInpaint
            | Self::MiganInpaint
            | Self::SdInpaint
            | Self::RoremInpaint
            | Self::RealEsrganUpscale
            | Self::Nomos8kUpscale
            | Self::FourXNmkdSiaxCxUpscale
            | Self::FourXNmkdSuperscaleUpscale
            | Self::Nomos2Upscale
            | Self::ColorizeDdcolor => None,
        }
    }

    /// `ModelKind` for any registered model (seg, inpaint, upscale).
    /// Returns `None` only for `Self::None`. Use this at recipe-diff
    /// sites where the actually-dispatched model's kind matters; use
    /// `to_model_kind` only when you specifically need the
    /// seg-memory-cap-eligible subset.
    pub fn dispatch_model_kind(self) -> Option<ModelKind> {
        self.to_model_id().and_then(|id| ModelKind::try_from(id).ok())
    }

    /// Registry id, or `None` for the no-model variant.
    pub const fn to_model_id(self) -> Option<prunr_models::ModelId> {
        match self {
            Self::Silueta => Some(prunr_models::ModelId::Silueta),
            Self::U2net => Some(prunr_models::ModelId::U2net),
            Self::BiRefNetLite => Some(prunr_models::ModelId::BiRefNetLite),
            Self::Inpaint => Some(prunr_models::ModelId::LaMaFp32),
            Self::BigInpaint => Some(prunr_models::ModelId::BigLaMa),
            Self::MiganInpaint => Some(prunr_models::ModelId::Migan),
            Self::SdInpaint => Some(prunr_models::ModelId::SdV15InpaintFp16),
            Self::RoremInpaint => Some(prunr_models::ModelId::RoremFp16),
            Self::RealEsrganUpscale => Some(prunr_models::ModelId::RealEsrganX4Plus),
            Self::Nomos8kUpscale => Some(prunr_models::ModelId::Nomos8kSchatL),
            Self::FourXNmkdSiaxCxUpscale => Some(prunr_models::ModelId::FourXNmkdSiaxCx),
            Self::FourXNmkdSuperscaleUpscale => Some(prunr_models::ModelId::FourXNmkdSuperscale),
            Self::Nomos2Upscale => Some(prunr_models::ModelId::FourXNomos2RealPlksr),
            Self::ColorizeDdcolor => Some(prunr_models::ModelId::DdcolorTiny),
            Self::None => None,
        }
    }
}

/// The model a fresh install starts on.
pub const DEFAULT_MODEL: SettingsModel = SettingsModel::BiRefNetLite;

/// Where a preset is filed when no model is chosen (and where v1 files,
/// which predate per-model presets, land).
pub(crate) const DEFAULT_PRESET_MODEL: prunr_models::ModelId = match DEFAULT_MODEL.to_model_id() {
    Some(id) => id,
    None => panic!("DEFAULT_MODEL must be a registered model"),
};

/// The model every colorize run uses, in the app and the CLI.
pub const COLORIZER: ModelKind = ModelKind::DdcolorTiny;

pub use prunr_core::LineMode;

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL,
            auto_process_on_import: false,
            parallel_jobs: (num_cpus::get() / 2).max(1),
            dark_checker: false,
            selection_fill: default_selection_fill(),
            selection_outline: true,
            animations: true,
            face_outlines: true,
            auto_hide_adjustments: false,
            export_split_layers: false,
            export_depth_map: false,
            export_format: prunr_core::ExportFormat::Png,
            presets: HashMap::new(),
            default_preset: default_preset_name(),
            bg_image_paths: HashMap::new(),
            accepted_licenses: Vec::new(),
            runtime_prompt_snoozed_until: std::collections::HashMap::new(),
            processor: ComputeMode::Auto,
            active_backend: "CPU".to_string(),
            brush: BrushSettings::default(),
            sd_memory: SdMemoryGuard::Balanced,
            hotkeys: BTreeMap::new(),
            keep_sd_loaded: false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::gui::item_settings::item_with_gamma;
    use prunr_core::LineMode;

    #[test]
    fn openvino_and_cpu_only_count_as_the_host_cpu() {
        let mut s = Settings::default();
        for (backend, cpu_only, host) in [("OpenVINO", false, true), ("CPU", false, true), ("", false, true), ("CUDA", false, false), ("CUDA", true, true), ("DirectML", false, false)] {
            s.active_backend = backend.into();
            s.processor = if cpu_only { ComputeMode::CpuOnly } else { ComputeMode::Auto };
            assert_eq!(s.runs_on_host_cpu(), host, "{backend} cpu_only={cpu_only}");
        }
    }

    /// A model's runs are timed by the provider its session lands on: the
    /// CPU under CPU only, with no accelerator, or when the accelerator
    /// does not run the model (Silueta on OpenVINO).
    #[test]
    fn a_model_the_accelerator_does_not_run_lands_on_the_cpu() {
        use prunr_core::ModelKind;
        let mut s = Settings::default();
        for (backend, cpu_only, model) in [("CPU", false, ModelKind::BiRefNetLite), ("", false, ModelKind::U2net), ("CUDA", true, ModelKind::BiRefNetLite), ("OpenVINO", false, ModelKind::Silueta)] {
            s.active_backend = backend.into();
            s.processor = if cpu_only { ComputeMode::CpuOnly } else { ComputeMode::Auto };
            assert_eq!(s.model_backend(model), "CPU", "{model:?} on {backend} cpu_only={cpu_only}");
        }
    }

    #[test]
    fn v1_migration_parses_all_per_image_fields() {
        let v1_json = serde_json::json!({
            "model": "Silueta",
            "auto_remove_on_import": false,
            "parallel_jobs": 4,
            "mask_gamma": 1.5,
            "mask_threshold": 0.3,
            "mask_threshold_enabled": true,
            "edge_shift": 2.0,
            "refine_edges": true,
            "line_mode": "AfterBgRemoval",
            "line_strength": 0.8,
            "solid_line_color": true,
            "line_color": [10, 20, 30, 255],
            "apply_bg_color": true,
            "bg_color": [100, 150, 200, 255],
            "history_depth": 10,
            "chain_mode": false,
        });
        let migrated = migrate_v1_item_settings(&v1_json);
        assert_eq!(migrated.gamma, 1.5);
        assert_eq!(migrated.contrast, 1.0, "the old hard threshold is a hard edge");
        assert_eq!(migrated.edge_shift, 2.0);
        assert!(migrated.refine_edges);
        assert_eq!(migrated.line_mode, LineMode::SubjectOutline);
        assert_eq!(migrated.line_strength, 0.8);
        assert_eq!(migrated.solid_line_color, Some([10, 20, 30]));
        assert_eq!(migrated.bg, Some([100, 150, 200, 255]));
    }

    #[test]
    fn v1_migration_threshold_disabled() {
        let v1_json = serde_json::json!({
            "mask_threshold": 0.5,
            "mask_threshold_enabled": false,
        });
        let migrated = migrate_v1_item_settings(&v1_json);
        assert_eq!(migrated.contrast, 0.0);
    }

    #[test]
    fn v1_migration_bg_disabled() {
        let v1_json = serde_json::json!({
            "apply_bg_color": false,
            "bg_color": [100, 150, 200, 255],
        });
        let migrated = migrate_v1_item_settings(&v1_json);
        assert_eq!(migrated.bg, None);
    }

    #[test]
    fn default_preset_is_prunr() {
        let s = Settings::default();
        assert_eq!(s.default_preset, PRUNR_PRESET);
    }

    /// Whole-shape snapshot of `Settings::default()`. The individual
    /// helper tests (`progress_style_default_is_modal`,
    /// `default_preset_is_prunr`, …) each pin one field — a new
    /// field that lands without a default-aware test slips through.
    /// This pins the entire surface so a `parallel_jobs: 0` regression
    /// has to break this test on the way out.
    ///
    /// `parallel_jobs` is host-dependent (CPU count), so its row
    /// asserts a lower bound rather than an exact value.
    #[test]
    fn default_settings_snapshot() {
        let s = Settings::default();
        assert_eq!(s.model, SettingsModel::BiRefNetLite);
        assert!(!s.auto_process_on_import);
        assert!(s.parallel_jobs >= 1,
            "parallel_jobs must clamp to >= 1 even on a single-core host");
        assert!(!s.dark_checker);
        assert_eq!(s.selection_fill, 0.30);
        assert!(s.selection_outline);
        assert!(!s.auto_hide_adjustments);
        assert!(!s.export_split_layers);
        assert!(!s.export_depth_map);
        assert!(s.presets.is_empty());
        assert_eq!(s.default_preset, PRUNR_PRESET);
        assert!(s.bg_image_paths.is_empty());
        assert!(s.accepted_licenses.is_empty());
        assert!(s.runtime_prompt_snoozed_until.is_empty());
        assert_eq!(s.processor, ComputeMode::Auto);
        assert_eq!(s.active_backend, "CPU");
        assert_eq!(s.sd_memory, SdMemoryGuard::Balanced);
    }

    /// Wrap an `ItemSettings` into a single-entry v2 `PresetFile` keyed
    /// by the active model. Mirrors the runtime save flow's wrap step.
    fn wrap_for_model(s: &Settings, item: ItemSettings) -> super::super::presets::PresetFile {
        let mid = s.model.to_model_id().expect("test settings have a model_id");
        super::super::presets::PresetFile::single(mid, super::super::presets::ModelPreset {
            item_settings: item,
            brush: Default::default(),
            sd: None,
        })
    }

    #[test]
    fn preset_values_prunr_is_factory() {
        let s = Settings::default();
        assert_eq!(s.preset_values(PRUNR_PRESET), ItemSettings::default());
    }

    #[test]
    fn preset_values_returns_user_preset() {
        let mut s = Settings::default();
        let portrait = wrap_for_model(&s, item_with_gamma(2.0));
        s.presets.insert("Portrait".to_string(), portrait);
        assert_eq!(s.preset_values("Portrait").gamma, 2.0);
    }

    #[test]
    fn preset_values_missing_name_falls_back_to_factory() {
        let s = Settings::default();
        assert_eq!(s.preset_values("NonExistent"), ItemSettings::default());
    }

    #[test]
    fn item_defaults_for_new_item_uses_default_preset() {
        let mut s = Settings::default();
        let portrait = wrap_for_model(&s, item_with_gamma(2.0));
        s.presets.insert("Portrait".to_string(), portrait);
        s.default_preset = "Portrait".to_string();
        assert_eq!(s.item_defaults_for_new_item().gamma, 2.0);
    }

    #[test]
    fn item_defaults_for_new_item_factory_when_prunr_default() {
        let s = Settings::default();
        assert_eq!(s.item_defaults_for_new_item(), ItemSettings::default());
    }

    #[test]
    fn save_load_round_trip_via_tempdir() {
        // Verifies the JSON schema is stable across save → read-back, the
        // primary cross-platform contract for settings persistence.
        // Platform path resolution (config_dir) is delegated to the `dirs`
        // crate and tested separately.
        let dir = std::env::temp_dir().join(format!(
            "prunr-test-{}",
            std::process::id(),
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("settings.json");

        let original = Settings {
            parallel_jobs: 7,
            processor: ComputeMode::CpuOnly,
            auto_process_on_import: true,
            ..Settings::default()
        };
        original.save_to_path(&path);

        let json = std::fs::read_to_string(&path).expect("settings.json should exist");
        let restored: Settings = serde_json::from_str(&json).expect("schema round-trip");
        assert_eq!(restored.parallel_jobs, 7);
        assert!(restored.auto_process_on_import);
        assert_eq!(restored.processor, ComputeMode::CpuOnly, "the processor choice is saved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Reset keeps what the user owns or agreed to (presets, licence
    /// acceptances, background pictures, the model in use, the OpenVINO
    /// prompt snooze) and returns every preference to its default.
    #[test]
    fn reset_keeps_what_the_user_owns_and_resets_preferences() {
        let mut s = Settings::default();
        s.model = SettingsModel::SdInpaint;
        s.accepted_licenses.push("SdV15InpaintFp16".into());
        s.bg_image_paths.insert(7, PathBuf::from("/pics/beach.jpg"));
        s.runtime_prompt_snoozed_until.insert("OpenVino".into(), 99);
        s.hotkeys.insert("undo".into(), vec!["Ctrl+U".into()]);
        s.processor = ComputeMode::CpuOnly;
        s.keep_sd_loaded = true;
        s.reset_preserving_identity();
        assert_eq!(s.model, SettingsModel::SdInpaint);
        assert_eq!(s.accepted_licenses, vec!["SdV15InpaintFp16".to_string()]);
        assert_eq!(s.bg_image_paths.get(&7), Some(&PathBuf::from("/pics/beach.jpg")));
        assert_eq!(s.runtime_prompt_snoozed_until.get("OpenVino"), Some(&99));
        assert!(s.hotkeys.is_empty());
        assert_eq!(s.processor, ComputeMode::Auto);
        assert!(!s.keep_sd_loaded);
    }

    /// A margin saved by the GB slider loads as the nearest guard.
    #[test]
    fn a_saved_gb_margin_becomes_the_nearest_memory_guard() {
        for (gb, guard) in [(0.5, SdMemoryGuard::Relaxed), (2.0, SdMemoryGuard::Balanced), (6.0, SdMemoryGuard::Strict)] {
            let mut v = serde_json::to_value(Settings::default()).unwrap();
            let obj = v.as_object_mut().unwrap();
            obj.remove("sd_memory");
            obj.insert("ram_safety_margin_gb".into(), gb.into());
            migrate_sd_margin(&mut v);
            let s: Settings = serde_json::from_value(v).unwrap();
            assert_eq!(s.sd_memory, guard, "{gb} GB");
        }
        assert_eq!(SdMemoryGuard::Balanced.margin_mb(), prunr_core::inpaint_sd::SD_DEFAULT_MARGIN_MB);
    }

    /// Under "CPU only" every reader of the backend (status bar, the
    /// Model Store's GPU advisories) sees the CPU, not the detected GPU.
    #[test]
    fn cpu_only_reports_the_cpu_as_the_backend() {
        let mut s = Settings { active_backend: "OpenVINO".into(), ..Settings::default() };
        assert_eq!(s.effective_backend(), "OpenVINO");
        s.processor = ComputeMode::CpuOnly;
        assert!(s.cpu_only());
        assert_eq!(s.effective_backend(), "CPU");
        s.note_backend("CPU".into());
        assert_eq!(s.active_backend, "OpenVINO", "a CPU-only worker's report keeps the detected accelerator");
    }

    #[test]
    fn a_settings_file_from_before_the_progress_capsule_still_loads() {
        let mut json = serde_json::to_value(Settings { parallel_jobs: 5, ..Settings::default() }).unwrap();
        json["progress_style"] = serde_json::json!("banner");
        let restored: Settings = serde_json::from_value(json).expect("an old key is ignored");
        assert_eq!(restored.parallel_jobs, 5);
    }

    #[test]
    fn brush_settings_round_trip() {
        use prunr_core::brush::BrushShape;
        use prunr_core::selection::BrushMode;
        let dir = std::env::temp_dir().join(format!(
            "prunr-brush-test-{}",
            std::process::id(),
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("settings.json");

        let mut original = Settings::default();
        original.brush.radius = 87.5;
        original.brush.hardness = 0.42;
        original.brush.strength = 0.75;
        original.brush.mode = BrushMode::Add;
        original.brush.shape = BrushShape::Square;
        original.save_to_path(&path);

        let json = std::fs::read_to_string(&path).expect("settings.json exists");
        let restored: Settings = serde_json::from_str(&json).expect("schema round-trip");
        assert_eq!(restored.brush.radius, 87.5);
        assert_eq!(restored.brush.hardness, 0.42);
        assert_eq!(restored.brush.strength, 0.75);
        assert_eq!(restored.brush.mode, BrushMode::Add);
        assert_eq!(restored.brush.shape, BrushShape::Square);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_brush_field_falls_back_to_default() {
        let json = serde_json::json!({
            "model": "Silueta",
            "auto_process_on_import": false,
            "parallel_jobs": 4,
            "history_depth": 10,
            "chain_mode": false,
        });
        let restored: Settings = serde_json::from_value(json).expect("legacy json loads");
        assert_eq!(restored.brush, super::BrushSettings::default());
    }

    /// Settings live at <root>/prunr/settings.json, and under test the
    /// root is never the user's config dir: a test that saves must not
    /// overwrite the real settings or presets.
    #[test]
    fn tests_save_settings_away_from_the_user_s_config() {
        let path = Settings::config_path().expect("config root resolves");
        assert!(path.ends_with("prunr/settings.json") || path.ends_with("prunr\\settings.json"), "{path:?}");
        let user = dirs::config_dir().expect("config_dir resolves on test host");
        assert!(!path.starts_with(&user), "{path:?} is in the user's config");
        let presets = super::super::presets_fs::presets_dir().expect("presets dir");
        assert!(!presets.starts_with(&user), "{presets:?} is in the user's config");
    }

    /// Non-LCM scheduler short-circuits before the install gate —
    /// pins the AND ordering so a refactor can't silently turn the
    /// scheduler check into an OR.
    #[test]
    fn lcm_routing_inactive_for_non_lcm_scheduler() {
        use crate::gui::brush_state::SdScheduler;
        let mut s = Settings::default();
        s.brush.sd_scheduler = SdScheduler::DpmPlusPlus2MKarras;
        assert!(!s.lcm_routing_active(prunr_models::ModelId::SdV15InpaintFp16));
    }

    /// Non-SD backend short-circuits before the install gate.
    #[test]
    fn lcm_routing_inactive_for_non_sd_backend() {
        use crate::gui::brush_state::SdScheduler;
        let mut s = Settings::default();
        s.brush.sd_scheduler = SdScheduler::Lcm;
        assert!(!s.lcm_routing_active(prunr_models::ModelId::LaMaFp32));
    }

    #[test]
    fn resolve_active_preset_returns_prunr_floor_when_default_preset_is_prunr() {
        let s = Settings::default();
        let r = s.resolve_active_preset(None);
        assert_eq!(r.item_settings, ItemSettings::default());
        assert_eq!(r.brush, super::BrushSettings::default());
    }

    #[test]
    fn resolve_active_preset_returns_named_preset_values() {
        let mut s = Settings::default();
        s.model = SettingsModel::Silueta;
        let mut brush = super::BrushSettings::default();
        brush.radius = 80.0;
        let mp = super::super::presets::ModelPreset {
            item_settings: item_with_gamma(2.0),
            brush,
            sd: None,
        };
        let mut models = HashMap::new();
        models.insert(super::super::presets::model_id_key(prunr_models::ModelId::Silueta), mp);
        let file = super::super::presets::PresetFile {
            format_version: super::super::presets::PRESET_FORMAT_VERSION,
            models,
        };
        s.presets.insert("Foo".to_string(), file);
        s.default_preset = "Foo".to_string();

        let r = s.resolve_active_preset(None);
        assert!((r.item_settings.gamma - 2.0).abs() < f32::EPSILON);
        assert!((r.brush.radius - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_cutout_preset_is_shared_by_the_cutout_models_and_no_other() {
        let mut s = Settings::default();
        s.model = SettingsModel::U2net;
        let mut brush = super::BrushSettings::default();
        brush.radius = 80.0;
        let mp = super::super::presets::ModelPreset {
            item_settings: item_with_gamma(2.0),
            brush,
            sd: None,
        };
        let mut models = HashMap::new();
        // Foo only has a Silueta entry; active model is U2net.
        models.insert(super::super::presets::model_id_key(prunr_models::ModelId::Silueta), mp);
        let file = super::super::presets::PresetFile {
            format_version: super::super::presets::PRESET_FORMAT_VERSION,
            models,
        };
        s.presets.insert("Foo".to_string(), file);
        s.default_preset = "Foo".to_string();

        let r = s.resolve_active_preset(None);
        assert_eq!((r.item_settings.gamma, r.brush.radius), (2.0, 80.0), "U2net uses the Silueta entry");
        s.model = SettingsModel::BigInpaint;
        let r = s.resolve_active_preset(None);
        assert_eq!(r.item_settings, ItemSettings::default(), "an eraser has no entry: factory values");
        s.model = SettingsModel::None;
        assert_eq!(s.preset_values("Foo").gamma, 2.0, "No model reads the default model's entry, which Silueta's stands in for");
    }

    #[test]
    fn a_model_switch_takes_the_models_tuning_and_keeps_the_users_brush() {
        use crate::gui::brush_state::BrushSettings;
        use super::super::presets::{model_id_key, ModelPreset, PresetFile, PRESET_FORMAT_VERSION};

        let mut s = Settings::default();
        s.model = SettingsModel::Silueta;
        let mut silueta_brush = BrushSettings::default();
        silueta_brush.radius = 80.0;
        let mut u2net_brush = BrushSettings::default();
        u2net_brush.radius = 120.0;
        u2net_brush.inpaint_feather = 9.0;
        let mut models = HashMap::new();
        models.insert(model_id_key(prunr_models::ModelId::Silueta), ModelPreset {
            item_settings: ItemSettings::default(),
            brush: silueta_brush,
            sd: None,
        });
        models.insert(model_id_key(prunr_models::ModelId::U2net), ModelPreset {
            item_settings: ItemSettings::default(),
            brush: u2net_brush,
            sd: None,
        });
        let file = PresetFile { format_version: PRESET_FORMAT_VERSION, models };
        s.presets.insert("Foo".to_string(), file);
        s.default_preset = "Foo".to_string();
        s.brush.radius = 33.0;
        s.brush.hardness = 0.2;

        s.model = SettingsModel::U2net;
        s.on_model_change_resolve_brush();
        assert_eq!((s.brush.radius, s.brush.hardness), (33.0, 0.2), "what the user set for the brush survives a model switch");
        assert_eq!(s.brush.inpaint_feather, 9.0, "the new model's tuning comes from the preset");
    }

    /// Without the Fast download, Fast's few steps would run on the full
    /// model and leave a half-made fill; the eraser runs Quality instead.
    #[test]
    fn fast_without_its_download_runs_quality() {
        use crate::gui::brush_state::SdScheduler;
        let mut s = Settings::default();
        s.model = SettingsModel::SdInpaint;
        s.brush.sd_scheduler = SdScheduler::Lcm;
        s.brush.sd_prompt = "kept".into();

        s.fit_sd_quality_to_install(true);
        assert_eq!(s.brush.sd_scheduler, SdScheduler::Lcm, "installed: Fast stays");

        s.fit_sd_quality_to_install(false);
        assert_eq!(s.brush.sd_scheduler, SdScheduler::DpmPlusPlus2MKarras);
        assert!(SdScheduler::DpmPlusPlus2MKarras.steps_range().contains(&s.brush.sd_steps));
        assert_eq!(s.brush.sd_prompt, "kept");
    }

    #[test]
    fn on_scheduler_change_resolve_sd_swaps_bundle() {
        use crate::gui::brush_state::SdScheduler;
        use super::super::presets::{
            model_id_key, ModelPreset, PresetFile, SdPreset, SdSchedulerBundle,
            PRESET_FORMAT_VERSION,
        };

        let mut s = Settings::default();
        s.model = SettingsModel::SdInpaint;
        let mut schedulers = HashMap::new();
        schedulers.insert(SdScheduler::Lcm, SdSchedulerBundle {
            steps: 8, guidance_scale: 1.5,
        });
        schedulers.insert(SdScheduler::DpmPlusPlus2MKarras, SdSchedulerBundle {
            steps: 25, guidance_scale: 6.0,
        });
        let sd = SdPreset {
            active_scheduler: SdScheduler::Lcm,
            schedulers,
            ..SdPreset::default()
        };
        let mp = ModelPreset {
            item_settings: ItemSettings::default(),
            brush: super::BrushSettings::default(),
            sd: Some(sd),
        };
        let mut models = HashMap::new();
        models.insert(model_id_key(prunr_models::ModelId::SdV15InpaintFp16), mp);
        let file = PresetFile { format_version: PRESET_FORMAT_VERSION, models };
        s.presets.insert("Foo".to_string(), file);
        s.default_preset = "Foo".to_string();
        s.brush.sd_scheduler = SdScheduler::Lcm;
        s.brush.sd_steps = 8;
        s.brush.radius = 42.0;
        s.brush.sd_prompt = "a red brick wall".into();
        s.brush.sd_negative_prompt = "people".into();
        s.brush.sd_seed = Some(77);

        s.on_scheduler_change_resolve_sd(SdScheduler::DpmPlusPlus2MKarras);

        assert_eq!(s.brush.sd_scheduler, SdScheduler::DpmPlusPlus2MKarras);
        assert_eq!(s.brush.sd_steps, 25);
        assert!((s.brush.sd_guidance_scale - 6.0).abs() < f32::EPSILON);
        // Non-SD field stays put — owned by the brush popover.
        assert!((s.brush.radius - 42.0).abs() < f32::EPSILON);
        assert_eq!(s.brush.sd_prompt, "a red brick wall", "picking a quality keeps the prompt");
        assert_eq!(s.brush.sd_negative_prompt, "people");
        assert_eq!(s.brush.sd_seed, Some(77), "and the pinned seed");
    }

    #[test]
    fn save_creates_parent_dir() {
        // Regression: on a fresh install the parent dir doesn't exist yet;
        // save() must create it rather than silently failing.
        let dir = std::env::temp_dir().join(format!(
            "prunr-mkdir-test-{}",
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let nested = dir.join("nested").join("settings.json");
        Settings::default().save_to_path(&nested);
        assert!(nested.exists(), "save_to_path must mkdir -p the parent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cross-subsystem contract: a dozen call sites across canvas /
    /// toolbar / app branch on `is_inpaint()`, including the seg
    /// live-preview gate that prevents the no-tensor fallback from
    /// clobbering result_rgba. Any new variant added to `ALL` must be
    /// classified explicitly here.
    #[test]
    fn is_inpaint_classifies_every_variant() {
        for m in SettingsModel::ALL {
            let expected = matches!(
                m,
                SettingsModel::Inpaint
                    | SettingsModel::BigInpaint
                    | SettingsModel::MiganInpaint
                    | SettingsModel::SdInpaint
                    | SettingsModel::RoremInpaint,
            );
            assert_eq!(m.is_inpaint(), expected, "{m:?}");
        }
    }

    #[test]
    fn is_inpaint_and_uses_segmentation_are_disjoint() {
        for m in SettingsModel::ALL {
            assert!(
                !(m.is_inpaint() && m.uses_segmentation()),
                "{m:?} cannot be both inpaint and seg",
            );
        }
    }

    #[test]
    fn is_upscale_classifies_every_variant() {
        for m in SettingsModel::ALL {
            let expected = matches!(
                m,
                SettingsModel::RealEsrganUpscale
                    | SettingsModel::Nomos8kUpscale
                    | SettingsModel::FourXNmkdSiaxCxUpscale
                    | SettingsModel::FourXNmkdSuperscaleUpscale
                    | SettingsModel::Nomos2Upscale,
            );
            assert_eq!(m.is_upscale(), expected, "{m:?}");
        }
    }

    #[test]
    fn is_colorize_classifies_every_variant() {
        for m in SettingsModel::ALL {
            assert_eq!(m.is_colorize(), m == SettingsModel::ColorizeDdcolor, "{m:?}");
            assert_eq!(m.role() == crate::gui::knob_catalog::ModelRole::Colorize, m.is_colorize(), "{m:?}");
        }
    }

    #[test]
    fn upscale_inpaint_seg_mutually_exclusive() {
        for m in SettingsModel::ALL {
            let count = [m.is_upscale(), m.is_inpaint(), m.uses_segmentation(), m.is_colorize()]
                .iter()
                .filter(|&&b| b)
                .count();
            assert!(count <= 1, "{m:?} must belong to at most one category");
        }
    }

    #[test]
    fn to_model_id_returns_some_for_upscale_variants() {
        assert_eq!(
            SettingsModel::RealEsrganUpscale.to_model_id(),
            Some(prunr_models::ModelId::RealEsrganX4Plus),
        );
        assert_eq!(
            SettingsModel::Nomos8kUpscale.to_model_id(),
            Some(prunr_models::ModelId::Nomos8kSchatL),
        );
    }
}
