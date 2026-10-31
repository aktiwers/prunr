use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use prunr_core::ModelKind;

use super::brush_state::BrushSettings;
use super::item_settings::ItemSettings;

/// The built-in factory-defaults preset: never deleted or overwritten, and
/// always `ItemSettings::default()` whatever the map holds.
pub const PRUNR_PRESET: &str = "Prunr";

const SETTINGS_UNREADABLE: &str = "Your settings could not be read, so Prunr started with its defaults. The old file is kept as settings.json.bad.";
const SOME_SETTINGS_RESET: &str = "Some settings could not be read and are back to their defaults; the rest are kept. The old file is kept as settings.json.bad.";

/// Keep a settings file that did not read beside it, as `<name>.bad`.
fn keep_unreadable(path: &std::path::Path) {
    let mut bad = path.as_os_str().to_owned();
    bad.push(".bad");
    if let Err(err) = std::fs::copy(path, &bad) {
        tracing::warn!(%err, ?path, "could not keep the unreadable settings file");
    }
}

/// Global app config; per-image knobs live on `BatchItem.settings`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: SettingsModel,
    /// The alias reads pre-v2 settings files.
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
    /// composite.
    #[serde(default)]
    pub export_split_layers: bool,
    /// Save also writes each image's depth map as `<stem>-depth.png`.
    #[serde(default)]
    pub export_depth_map: bool,
    /// File format Save writes (the save dialog's extension can override it).
    #[serde(default)]
    pub export_format: prunr_core::ExportFormat,
    /// Named presets (not "Prunr"), loaded from the presets folder where
    /// each is its own shareable file; never written to settings.json.
    /// `default` lets old files that embedded presets still load.
    #[serde(default, skip_serializing)]
    pub(crate) presets: HashMap<String, super::presets::Preset>,
    /// Names (lowercased) of presets a newer Prunr saved: left out of
    /// `presets`, and never saved over.
    #[serde(skip)]
    pub(crate) newer_presets: std::collections::HashSet<String>,
    /// Which preset new imports inherit, and what Reset All Knobs restores;
    /// a missing one falls back to "Prunr".
    #[serde(default = "default_preset_name")]
    pub default_preset: String,

    /// `bg_image_hash` → path, so a preset can reload its bg image; a
    /// shared preset with no matching path clears the bg with a toast.
    #[serde(default)]
    pub bg_image_paths: HashMap<u64, PathBuf>,

    /// Models whose restrictive licenses the user accepted, as `ModelId`
    /// Debug names so prunr-models stays serde-free.
    #[serde(default)]
    pub accepted_licenses: Vec<String>,

    /// Unix seconds before which the runtime install prompt stays quiet
    /// ("Not now"), keyed by `RuntimeId` Debug name.
    #[serde(default)]
    pub runtime_prompt_snoozed_until: std::collections::HashMap<String, i64>,

    #[serde(default)]
    pub processor: ComputeMode,
    #[serde(skip)]
    pub active_backend: String,
    #[serde(default)]
    pub brush: BrushSettings,
    /// What was changed in `brush` since a preset set it, per model; kept,
    /// so a restart does not count those changes as untouched.
    #[serde(default)]
    pub(crate) brush_changes: super::preset_changes::Changes<BrushSettings>,

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
    /// Route the SD eraser to the LCM checkpoint: SD picked, LCM
    /// scheduler, and the LCM bundle published and downloaded. Dispatch
    /// and UI gating both read this.
    pub fn lcm_routing_active(&self, picked_model: prunr_models::ModelId) -> bool {
        use crate::gui::brush_state::SdScheduler;
        picked_model == prunr_models::ModelId::SdV15InpaintFp16
            && self.brush.sd_scheduler == SdScheduler::Lcm
            && Self::can_select_lcm_scheduler()
    }

    /// The LCM bundle is published in the registry AND downloaded.
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

    /// The edge a new stroke is drawn with, as its brush's strip sets it: a
    /// selection's Feather, or under an eraser its Expand region and the
    /// fill's Feather.
    pub(crate) fn stroke_edge(&self) -> super::brush_state::StrokeEdge {
        use super::brush_state::StrokeEdge;
        use crate::gui::knob_catalog::{shows, Surface};
        let b = &self.brush;
        if shows(self.model, Surface::SelectionStrokes) {
            StrokeEdge::Selection { feather: b.edge_feather }
        } else {
            StrokeEdge::Eraser { expand: b.inpaint_grow, feather: b.inpaint_feather }
        }
    }

    /// The values a stroke is drawn with, which its brush's sliders set.
    pub(crate) fn stroke_values(&self) -> super::brush_state::StrokeValues {
        let b = &self.brush;
        super::brush_state::StrokeValues {
            size: b.radius,
            hardness: b.hardness,
            opacity: b.strength,
            reading: b.magic_reading(),
            range: b.magic_depth_range,
            ..self.stroke_edge().values()
        }
    }

    /// Config file path: ~/.config/prunr/settings.json (Linux),
    /// ~/Library/Application Support/prunr/settings.json (macOS),
    /// %APPDATA%/prunr/settings.json (Windows).
    pub fn config_path() -> Option<PathBuf> {
        config_root().map(|d| d.join("prunr").join("settings.json"))
    }

    /// The saved settings and presets, and a note for the first frame when
    /// some did not read. Missing or corrupt falls back to defaults.
    pub fn load() -> (Self, Option<&'static str>) {
        // Before any load_all(), so the built-ins show on first launch.
        super::presets_fs::seed_builtins_once();

        let (mut settings, note) = Self::config_path().map_or_else(|| (Self::default(), None), |path| Self::read(&path));
        let (mut loaded, newer) = super::presets_fs::load_all();
        settings.newer_presets = newer;
        // The migrated "Previous defaults" survives a failed save.
        for (k, v) in settings.presets.drain() {
            loaded.entry(k).or_insert(v);
        }
        settings.presets = loaded;
        (settings, note)
    }

    /// The settings saved at `path`, kept as far as they read, and a note
    /// for the person when some could not be: a file that is no JSON gives
    /// the defaults, a setting this version cannot read (one a newer Prunr
    /// saved) gives that setting's default and keeps the rest. Either way
    /// the file is kept beside it as `.bad`, as the next save replaces it.
    fn read(path: &std::path::Path) -> (Self, Option<&'static str>) {
        let Ok(data) = std::fs::read_to_string(path) else { return (Self::default(), None) };

        // Via Value, so v1 fields migrate even when strict parsing fails.
        let mut value: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(%e, "settings.json is not JSON; starting from defaults");
                keep_unreadable(path);
                return (Self::default(), Some(SETTINGS_UNREADABLE));
            }
        };
        super::item_settings::migrate_old_item_settings(&mut value);
        migrate_sd_margin(&mut value);

        let (mut settings, reset, note) = match serde_json::from_value(value.clone()) {
            Ok(s) => (s, Vec::new(), None),
            Err(e) => {
                let (s, reset) = Self::merge_readable(&value);
                tracing::warn!(%e, ?reset, "settings.json has settings this version cannot read; they start from defaults");
                keep_unreadable(path);
                (s, reset, Some(SOME_SETTINGS_RESET))
            }
        };
        settings.brush.fit_sd_to_quality();
        // Without changes kept over the brush as read (a file from before
        // they were kept, or a brush or changes that did not read), they
        // start from it, or every setting apart from the code's default
        // would count as one.
        let changes_kept = value.get("brush_changes").is_some() && !reset.iter().any(|key| key == "brush" || key == "brush_changes");
        if !changes_kept {
            settings.restart_brush_changes();
        }

        // v1 per-image fields become a "Previous defaults" preset; new
        // imports still start with factory defaults.
        let is_v1 = value.get("mask_gamma").is_some()
            || value.get("apply_bg_color").is_some()
            || value.get("line_mode").is_some();
        const LEGACY_PRESET: &str = "Previous defaults";
        if is_v1 && !settings.presets.contains_key(LEGACY_PRESET) {
            let legacy_item = migrate_v1_item_settings(&value);
            if legacy_item != ItemSettings::default() {
                let file = super::presets::PresetFile::of_item_settings(legacy_item);
                let _ = super::presets_fs::save(LEGACY_PRESET, &file);
                settings.presets.insert(LEGACY_PRESET.to_string(), file.into());
            }
        }
        (settings, note)
    }

    /// The defaults with every saved setting in `value` that reads laid
    /// over them, and the names of those that did not.
    fn merge_readable(value: &serde_json::Value) -> (Self, Vec<String>) {
        value.as_object().map_or_else(|| (Self::default(), Vec::new()), |saved| super::presets::lay_readable(Self::default(), saved))
    }

    /// Nothing changed on the brush from here: a preset was applied, or
    /// the brush was read without changes kept over it.
    pub(crate) fn restart_brush_changes(&mut self) {
        self.brush_changes = super::preset_changes::Changes::new(self.brush.clone());
    }

    /// Best-effort, off the UI thread, after every earlier save.
    pub fn save(&self) {
        let Some(path) = Self::config_path() else { return };
        match serde_json::to_string_pretty(self) {
            Ok(json) => super::disk_queue::write(path, json.into_bytes()),
            Err(err) => tracing::warn!(%err, "settings not saved"),
        }
    }

    /// Whether an accelerator backend is active (OpenVINO counts, though it
    /// may run on the host CPU: `runs_on_host_cpu`). Settings offers the
    /// Processor choice only then.
    pub fn is_gpu(&self) -> bool {
        is_accelerator(&self.active_backend)
    }

    /// Record the provider a worker reports, or one that started in this
    /// process (an add-on installed while Prunr runs). Under "CPU only" a
    /// worker reports the CPU by choice, so the detected accelerator is
    /// kept (Settings still offers Auto). A new provider re-derives the
    /// job count.
    pub fn note_backend(&mut self, backend: String) {
        if backend.is_empty() || (self.cpu_only() && !is_accelerator(&backend)) || self.active_backend == backend {
            return;
        }
        self.active_backend = backend;
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
    /// OpenVINO without Intel graphics.
    pub fn runs_on_host_cpu(&self) -> bool {
        prunr_core::engine::runs_on_host_cpu(self.effective_backend()) || self.active_backend.is_empty()
    }

    /// The provider a session of `model` lands on: its fastest accelerator
    /// (two can run, CUDA beside DirectML), or the CPU when none runs this
    /// model, under "CPU only", or until the probe has answered.
    pub fn model_backend(&self, model: prunr_core::ModelKind) -> &'static str {
        self.engine_of(model.model_id())
    }

    /// `model_backend` for any model.
    pub fn engine_of(&self, model: prunr_models::ModelId) -> &'static str {
        if is_accelerator(self.effective_backend()) { prunr_core::memory::provider_now(model, false) } else { prunr_core::engine::CPU_NAME }
    }

    /// With no model the pipeline is pure CPU on decoded pixels, so the
    /// ORT-memory cap doesn't apply.
    pub fn default_jobs(&self) -> usize {
        super::memory::default_jobs(self.model.to_model_kind(), self.runs_on_host_cpu(), self.cpu_only())
    }

    pub fn max_jobs(&self) -> usize {
        let base = if self.runs_on_host_cpu() { num_cpus::get() } else { 4 };
        super::memory::cap_jobs(self.model.to_model_kind(), base, self.cpu_only())
    }

    /// A preset's image settings for the active model. Missing presets
    /// and models fall back to Prunr's, here and below.
    pub fn preset_values(&self, name: &str) -> ItemSettings {
        self.preset(name).item_settings(self.model)
    }

    /// A preset's image and brush settings for the active model: what
    /// applying it, or Reset to it, sets. `scheduler`: `None` uses the
    /// preset's own quality.
    pub(crate) fn resolve_preset(
        &self,
        name: &str,
        scheduler: Option<super::brush_state::SdScheduler>,
    ) -> super::presets::ResolvedView {
        self.preset(name).resolve(self.model, scheduler)
    }

    /// The named preset; Prunr's for "Prunr" and a missing one.
    pub(crate) fn preset(&self, name: &str) -> &super::presets::Preset {
        super::presets::named(&self.presets, name)
    }

    /// `resolve_preset` for the default preset.
    pub(crate) fn resolve_active_preset(
        &self,
        scheduler: Option<super::brush_state::SdScheduler>,
    ) -> super::presets::ResolvedView {
        self.resolve_preset(&self.default_preset, scheduler)
    }

    /// Apply preset `name` to `item` and to the brush.
    pub(crate) fn apply_preset(&mut self, name: &str, item: &mut ItemSettings) {
        let resolved = self.resolve_preset(name, None);
        *item = resolved.item_settings;
        self.brush = resolved.brush;
    }

    /// After a switch from `from`: the brush settings changed since a
    /// preset set them stay, the rest take the new model's from preset
    /// `name`.
    pub(crate) fn switch_brush_model(&mut self, from: SettingsModel, name: &str) {
        self.brush_changes.note(&self.brush, &super::preset_fields::brush_entries_of(from));
        let brush = super::presets::take_untouched_brush(self.preset(name), self.model, self.brush.clone(), &self.brush_changes);
        self.brush_changes.rebase(&brush);
        self.brush = brush;
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

    /// The default preset's values, with `PRUNR_*` knob env overrides on
    /// top (test harness only).
    pub fn item_defaults_for_new_item(&self) -> ItemSettings {
        let mut item = self.preset_values(&self.default_preset);
        super::env_overrides::apply_to_item_settings(&mut item);
        item
    }

    pub fn has_accepted_license(&self, id: prunr_models::ModelId) -> bool {
        self.has_accepted(&super::presets::model_id_key(id))
    }

    /// Persists; idempotent (no duplicate, no extra write).
    pub fn accept_license(&mut self, id: prunr_models::ModelId) {
        self.accept(&super::presets::model_id_key(id));
    }

    /// Whether the license kept under `key` was accepted.
    pub fn has_accepted(&self, key: &str) -> bool {
        self.accepted_licenses.iter().any(|s| s == key)
    }

    /// `accept_license` for a license kept under `key`.
    pub fn accept(&mut self, key: &str) {
        if !self.has_accepted(key) {
            self.accepted_licenses.push(key.to_string());
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
    /// snooze. The brush takes the default preset's settings for the
    /// model; `parallel_jobs` re-derives for the detected hardware.
    pub fn reset_preserving_identity(&mut self) {
        let kept = std::mem::take(self);
        *self = Self {
            active_backend: kept.active_backend,
            presets: kept.presets,
            newer_presets: kept.newer_presets,
            default_preset: kept.default_preset,
            accepted_licenses: kept.accepted_licenses,
            bg_image_paths: kept.bg_image_paths,
            model: kept.model,
            runtime_prompt_snoozed_until: kept.runtime_prompt_snoozed_until,
            ..Self::default()
        };
        self.brush = self.resolve_active_preset(None).brush;
        self.restart_brush_changes();
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

pub(crate) fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Each field falls back to default on a missing or wrong-type value.
fn migrate_v1_item_settings(v: &serde_json::Value) -> ItemSettings {
    use prunr_core::LineMode;

    let to_u8 = |x: &serde_json::Value| x.as_u64().unwrap_or(0).min(255) as u8;
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
                s.solid_line_color = Some([to_u8(r), to_u8(g), to_u8(b)]);
            }
        }
    }
    if v.get("apply_bg_color").and_then(|x| x.as_bool()) == Some(true) {
        if let Some(arr) = v.get("bg_color").and_then(|x| x.as_array()) {
            if let [r, g, b, a, ..] = arr.as_slice() {
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
    /// Filter-only: no segmentation. Whole-photo lines still run;
    /// SubjectOutline needs a seg model and is greyed out.
    None,
    /// LaMa eraser; bg-removal knobs are inert.
    Inpaint,
    /// Big-LaMa: same architecture and cost as LaMa, better weights.
    BigInpaint,
    /// MI-GAN (~25 MB): sharper on detail than LaMa, less smooth on flat
    /// backgrounds.
    MiganInpaint,
    /// SD 1.5 Inpainting (FP16), ~2 GB bundle: generates content rather
    /// than smooth fills.
    SdInpaint,
    /// RORem (SDXL tuned for object removal). Fills large holes with
    /// background instead of drawing something new; ~5.5 GB bundle.
    RoremInpaint,
    RealEsrganUpscale,
    /// HAT-L, photo-tuned: better than Real-ESRGAN, slower on CPU.
    Nomos8kUpscale,
    /// Sharper than x4plus on clean / mildly-compressed photos.
    FourXNmkdSiaxCxUpscale,
    /// Handles noise and JPEG artifacts on real photos.
    FourXNmkdSuperscaleUpscale,
    /// 4xNomos2 RealPLKSR. Fast, natural texture on skin and hair.
    Nomos2Upscale,
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

    pub fn index(self) -> usize {
        // every variant is listed in ALL
        Self::ALL.iter().position(|m| *m == self).unwrap_or(0)
    }

    /// Parse a Debug-style name ("BiRefNetLite"); `None` on unknown input.
    pub fn from_debug_name(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|m| format!("{m:?}") == s)
    }

    /// Whether this variant resolves to an ORT seg model. Filter-only and
    /// Inpaint variants skip segmentation entirely.
    pub fn uses_segmentation(self) -> bool {
        matches!(self, Self::Silueta | Self::U2net | Self::BiRefNetLite)
    }

    pub fn is_inpaint(self) -> bool {
        matches!(self, Self::Inpaint | Self::BigInpaint | Self::MiganInpaint | Self::SdInpaint | Self::RoremInpaint)
    }

    /// An eraser that generates its fill from a seed, so "Try another"
    /// can fill the same region again differently.
    pub fn generates(self) -> bool {
        matches!(self, Self::SdInpaint | Self::RoremInpaint)
    }

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

    /// `None` outside the seg memory-cap models (erasers, upscalers).
    pub fn to_model_kind(self) -> Option<ModelKind> {
        self.uses_segmentation().then(|| self.dispatch_model_kind()).flatten()
    }

    /// The cut-out model that runs `kind`, the inverse of `to_model_kind`.
    pub fn of_kind(kind: ModelKind) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.to_model_kind() == Some(kind))
    }

    /// Any registered model's kind; `None` only for `Self::None`.
    pub fn dispatch_model_kind(self) -> Option<ModelKind> {
        self.to_model_id().and_then(|id| ModelKind::try_from(id).ok())
    }

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

/// A provider other than the CPU (OpenVINO counts).
fn is_accelerator(backend: &str) -> bool {
    !backend.is_empty() && !backend.eq_ignore_ascii_case("CPU")
}

/// The model a fresh install starts on.
pub const DEFAULT_MODEL: SettingsModel = SettingsModel::BiRefNetLite;

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
            newer_presets: Default::default(),
            default_preset: default_preset_name(),
            bg_image_paths: HashMap::new(),
            accepted_licenses: Vec::new(),
            runtime_prompt_snoozed_until: std::collections::HashMap::new(),
            processor: ComputeMode::Auto,
            active_backend: "CPU".to_string(),
            brush: BrushSettings::default(),
            brush_changes: Default::default(),
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

    /// A selection stroke is drawn with the brush's Feather; an eraser
    /// stroke with the eraser's Expand region and Feather, which its strip
    /// shows in their place.
    #[test]
    fn a_stroke_takes_the_edge_its_strip_shows() {
        use crate::gui::brush_state::StrokeEdge;
        let mut s = Settings::default();
        (s.brush.edge_feather, s.brush.inpaint_grow, s.brush.inpaint_feather) = (6.0, -3.0, 12.0);
        for model in SettingsModel::ALL {
            s.model = model;
            let want = if crate::gui::knob_catalog::shows(model, crate::gui::knob_catalog::Surface::SelectionStrokes) {
                StrokeEdge::Selection { feather: 6.0 }
            } else {
                StrokeEdge::Eraser { expand: -3.0, feather: 12.0 }
            };
            assert_eq!(s.stroke_edge(), want, "{model:?}");
        }
        s.model = SettingsModel::Inpaint;
        let v = s.stroke_values();
        assert_eq!((v.feather, v.expand, v.fill_feather), (0.0, -3.0, 12.0), "an eraser");
    }

    /// A settings file that is no JSON starts from the defaults and is
    /// kept beside it; the person is told.
    #[test]
    fn unreadable_settings_start_from_defaults_and_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, b"{ not json").unwrap();
        let (settings, note) = Settings::read(&path);
        assert_eq!(settings.parallel_jobs, Settings::default().parallel_jobs);
        assert_eq!(note, Some(SETTINGS_UNREADABLE));
        assert_eq!(std::fs::read(dir.path().join("settings.json.bad")).unwrap(), b"{ not json");
    }

    /// A setting this version cannot read (a newer Prunr's model) starts
    /// from its default; every other setting is kept, and the file too.
    #[test]
    fn a_setting_this_version_cannot_read_resets_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut saved = serde_json::to_value(Settings { parallel_jobs: 3, ..Settings::default() }).unwrap();
        saved["model"] = serde_json::json!("ModelFromTheFuture");
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        let (settings, note) = Settings::read(&path);
        assert_eq!(settings.parallel_jobs, 3, "the rest are kept");
        assert_eq!(settings.model, Settings::default().model);
        assert_eq!(note, Some(SOME_SETTINGS_RESET));
        assert!(dir.path().join("settings.json.bad").exists());
        let (_, note) = Settings::read(&{ let p = dir.path().join("fine.json"); std::fs::write(&p, serde_json::to_vec(&serde_json::to_value(Settings::default()).unwrap()).unwrap()).unwrap(); p });
        assert_eq!(note, None, "settings that read say nothing");
    }

    #[test]
    fn openvino_and_cpu_only_count_as_the_host_cpu() {
        let mut s = Settings::default();
        let openvino_host = !prunr_core::engine::openvino_on_gpu();
        for (backend, cpu_only, host) in [("OpenVINO", false, openvino_host), ("CPU", false, true), ("", false, true), ("CUDA", false, false), ("CUDA", true, true), ("DirectML", false, false)] {
            s.active_backend = backend.into();
            s.processor = if cpu_only { ComputeMode::CpuOnly } else { ComputeMode::Auto };
            assert_eq!(s.runs_on_host_cpu(), host, "{backend} cpu_only={cpu_only}");
        }
    }

    /// A model's runs are timed by the provider its session lands on: the
    /// CPU under CPU only or with no accelerator, else the model's own
    /// (the CPU where no accelerator runs it, as Silueta on OpenVINO), on
    /// whatever this machine offers.
    #[test]
    fn a_model_lands_on_its_own_provider_or_the_cpu() {
        use prunr_core::ModelKind;
        let mut s = Settings::default();
        for (backend, cpu_only, model) in [("CPU", false, ModelKind::BiRefNetLite), ("", false, ModelKind::U2net), ("CUDA", true, ModelKind::BiRefNetLite)] {
            s.active_backend = backend.into();
            s.processor = if cpu_only { ComputeMode::CpuOnly } else { ComputeMode::Auto };
            assert_eq!(s.model_backend(model), "CPU", "{model:?} on {backend} cpu_only={cpu_only}");
        }
        s.active_backend = "OpenVINO".into();
        s.processor = ComputeMode::Auto;
        assert_eq!(s.model_backend(ModelKind::Silueta), prunr_core::memory::provider_now(prunr_models::ModelId::Silueta, false));
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

    /// Whole-shape snapshot of `Settings::default()`, so a
    /// `parallel_jobs: 0` regression has to break this test on the way out.
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

    /// A preset of `item` for the active model, saved as the app saves.
    fn preset_of_item(s: &Settings, item: ItemSettings) -> super::super::presets::Preset {
        let brush = super::super::presets::prunr_view(s.model, None).brush;
        super::super::presets::PresetFile::default().with_settings(s.model, &item, &brush).into()
    }

    /// A preset read from a file's JSON.
    fn preset(json: serde_json::Value) -> super::super::presets::Preset {
        serde_json::from_value::<super::super::presets::PresetFile>(json).expect("a preset file").into()
    }

    /// "Prunr", a missing preset, and a new item under "Prunr" all read
    /// factory values.
    #[test]
    fn prunr_and_missing_presets_read_factory_values() {
        let s = Settings::default();
        assert_eq!(s.preset_values(PRUNR_PRESET), ItemSettings::default());
        assert_eq!(s.preset_values("NonExistent"), ItemSettings::default());
        assert_eq!(s.item_defaults_for_new_item(), ItemSettings::default());
        let r = s.resolve_active_preset(None);
        assert_eq!((r.item_settings, r.brush), (ItemSettings::default(), super::BrushSettings::default()));
    }

    #[test]
    fn item_defaults_for_new_item_uses_default_preset() {
        let mut s = Settings::default();
        let portrait = preset_of_item(&s, item_with_gamma(2.0));
        s.presets.insert("Portrait".to_string(), portrait);
        s.default_preset = "Portrait".to_string();
        assert_eq!(s.item_defaults_for_new_item().gamma, 2.0);
    }

    /// What `save()` writes reads back the same, brush included.
    #[test]
    fn saved_settings_read_back_the_same() {
        use prunr_core::brush::BrushShape;
        use prunr_core::selection::BrushMode;
        let mut original = Settings {
            parallel_jobs: 7,
            processor: ComputeMode::CpuOnly,
            auto_process_on_import: true,
            ..Settings::default()
        };
        original.brush.radius = 87.5;
        original.brush.hardness = 0.42;
        original.brush.strength = 0.75;
        original.brush.mode = BrushMode::Add;
        original.brush.shape = BrushShape::Square;
        let json = serde_json::to_string_pretty(&original).unwrap();
        let restored: Settings = serde_json::from_str(&json).expect("schema round-trip");
        assert_eq!(restored.parallel_jobs, 7);
        assert!(restored.auto_process_on_import);
        assert_eq!(restored.processor, ComputeMode::CpuOnly, "the processor choice is saved");
        assert_eq!(restored.brush, original.brush);
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
        s.brush.inpaint_sharpen = 0.9;
        s.reset_preserving_identity();
        assert_eq!(s.model, SettingsModel::SdInpaint);
        assert_eq!(s.brush, s.resolve_active_preset(None).brush, "the brush is the default preset's for the model");
        assert_eq!(s.accepted_licenses, vec!["SdV15InpaintFp16".to_string()]);
        assert_eq!(s.bg_image_paths.get(&7), Some(&PathBuf::from("/pics/beach.jpg")));
        assert_eq!(s.runtime_prompt_snoozed_until.get("OpenVino"), Some(&99));
        assert!(s.hotkeys.is_empty());
        assert_eq!(s.processor, ComputeMode::Auto);
        assert!(!s.keep_sd_loaded);
    }

    /// A settings file from before the brush's changes were kept starts
    /// them from the brush it saved: nothing counts as changed.
    #[test]
    fn an_older_settings_file_starts_the_brush_changes_from_its_brush() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut v = serde_json::to_value(Settings::default()).unwrap();
        v["brush"]["inpaint_sharpen"] = serde_json::json!(0.0);
        v.as_object_mut().unwrap().remove("brush_changes");
        std::fs::write(&path, v.to_string()).unwrap();
        let (s, _) = Settings::read(&path);
        assert_eq!(s.brush_changes, super::super::preset_changes::Changes::new(s.brush.clone()));
    }

    /// A brush that does not read starts from the defaults, and its changes
    /// start from them too, not from the brush the file kept them over.
    #[test]
    fn a_brush_that_does_not_read_starts_its_changes_afresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut kept = Settings::default();
        kept.brush.inpaint_sharpen = 0.0;
        kept.restart_brush_changes();
        let mut v = serde_json::to_value(&kept).unwrap();
        v["brush"]["radius"] = serde_json::json!("wide");
        std::fs::write(&path, v.to_string()).unwrap();
        let (s, _) = Settings::read(&path);
        assert_eq!(s.brush, BrushSettings::default());
        assert_eq!(s.brush_changes, super::super::preset_changes::Changes::new(s.brush.clone()));
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

    /// CUDA installed on a computer the probe found no accelerator on is
    /// the backend from then on, under "CPU only" too (Settings offers
    /// Auto again).
    #[test]
    fn an_add_on_that_starts_is_the_backend_when_there_was_none() {
        for processor in [ComputeMode::Auto, ComputeMode::CpuOnly] {
            let mut s = Settings { active_backend: "CPU".into(), processor, ..Settings::default() };
            s.note_backend("CUDA".into());
            assert_eq!(s.active_backend, "CUDA");
            assert!(s.is_gpu());
        }
    }

    #[test]
    fn a_settings_file_from_before_the_progress_capsule_still_loads() {
        let mut json = serde_json::to_value(Settings { parallel_jobs: 5, ..Settings::default() }).unwrap();
        json["progress_style"] = serde_json::json!("banner");
        let restored: Settings = serde_json::from_value(json).expect("an old key is ignored");
        assert_eq!(restored.parallel_jobs, 5);
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

    /// Non-SD model short-circuits before the install gate.
    #[test]
    fn lcm_routing_inactive_for_non_sd_model() {
        use crate::gui::brush_state::SdScheduler;
        let mut s = Settings::default();
        s.brush.sd_scheduler = SdScheduler::Lcm;
        assert!(!s.lcm_routing_active(prunr_models::ModelId::LaMaFp32));
    }

    /// A preset applies to the models it has entries for; every other
    /// model, another cut-out model too, takes Prunr's settings.
    #[test]
    fn a_preset_applies_only_to_its_own_models() {
        use super::super::presets::prunr_view;
        let mut s = Settings::default();
        s.presets.insert("Foo".to_string(), preset(serde_json::json!({ "format_version": 2, "models": {
            "Silueta": { "item_settings": { "gamma": 2.0 }, "brush": { "radius": 80.0 } }
        } })));
        s.default_preset = "Foo".to_string();
        s.model = SettingsModel::Silueta;
        let r = s.resolve_active_preset(None);
        assert_eq!((r.item_settings.gamma, r.brush.radius), (2.0, 80.0), "Silueta reads its own entry");
        for other in [SettingsModel::U2net, SettingsModel::BiRefNetLite, SettingsModel::None, SettingsModel::BigInpaint] {
            s.model = other;
            assert_eq!(s.resolve_active_preset(None), prunr_view(other, None), "{other:?} takes Prunr's");
        }
    }

    /// A switch keeps the brush the user changed and takes the rest of the
    /// new model's from the preset; an eraser's fill tuning is always its
    /// own, as changed on it or as the preset has it.
    #[test]
    fn a_model_switch_keeps_the_brush_changes_and_takes_the_rest() {
        use super::super::presets::prunr_view;
        let mut s = Settings::default();
        s.model = SettingsModel::BigInpaint;
        s.presets.insert("Foo".to_string(), preset(serde_json::json!({ "format_version": 2, "models": {
            "LaMaFp32": { "brush": { "inpaint_sharpen": 0.3, "inpaint_fill_scale": "Full", "inpaint_grow": 20.0 } }
        } })));
        let mut item = ItemSettings::default();
        s.apply_preset("Foo", &mut item);
        s.brush_changes = super::super::preset_changes::Changes::new(s.brush.clone());
        s.brush.radius = 33.0;
        s.brush.inpaint_feather = 30.0;
        s.brush.inpaint_sharpen = 0.9;

        s.model = SettingsModel::Inpaint;
        s.switch_brush_model(SettingsModel::BigInpaint, "Foo");
        assert_eq!((s.brush.radius, s.brush.inpaint_feather), (33.0, 30.0), "what the user changed stays");
        assert_eq!(s.brush.inpaint_sharpen, 0.3, "the fill's Sharpen is LaMa's own, not Big-LaMa's");
        assert_eq!(s.brush.inpaint_fill_scale, prunr_core::inpaint::FillScale::Full, "the rest is the new model's from the preset");
        assert_eq!(s.brush.inpaint_grow, 20.0);

        s.model = SettingsModel::BigInpaint;
        s.switch_brush_model(SettingsModel::Inpaint, "Foo");
        assert_eq!(s.brush.inpaint_grow, prunr_view(SettingsModel::BigInpaint, None).brush.inpaint_grow, "a model the preset has no entry for takes Prunr's");
        assert_eq!(s.brush.inpaint_sharpen, 0.9, "back on Big-LaMa, the Sharpen changed there");

        s.model = SettingsModel::SdInpaint;
        s.switch_brush_model(SettingsModel::BigInpaint, "Foo");
        assert_eq!(s.brush.inpaint_sharpen, 0.0, "a generated fill keeps its own Sharpen 0");
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

        let mut s = Settings::default();
        s.model = SettingsModel::SdInpaint;
        s.presets.insert("Foo".to_string(), preset(serde_json::json!({ "format_version": 2, "models": {
            "SdV15InpaintFp16": { "sd": { "active_scheduler": "Lcm", "schedulers": {
                "Lcm": { "steps": 8, "guidance_scale": 1.5 },
                "DpmPlusPlus2MKarras": { "steps": 25, "guidance_scale": 6.0 }
            } } }
        } })));
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
    fn only_the_cut_out_models_have_a_seg_model_kind() {
        for m in SettingsModel::ALL {
            let expected = match m {
                SettingsModel::Silueta => Some(ModelKind::Silueta),
                SettingsModel::U2net => Some(ModelKind::U2net),
                SettingsModel::BiRefNetLite => Some(ModelKind::BiRefNetLite),
                _ => None,
            };
            assert_eq!(m.to_model_kind(), expected, "{m:?}");
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
