use std::collections::HashMap;
use std::path::{Path, PathBuf};
use prunr_core::ModelKind;

use super::brush_state::BrushSettings;
use super::item_settings::ItemSettings;

/// Name of the built-in, non-deletable, non-overwritable preset representing
/// factory defaults. Always present in the preset dropdown; the synthetic
/// values are `ItemSettings::default()` regardless of what's in the map.
pub const PRUNR_PRESET: &str = "Prunr";

/// Global app config. Per-image knobs (gamma, threshold, line mode, bg, ...)
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
    /// Maximum number of undo steps per image.
    pub history_depth: usize,
    /// When true, Process uses the current result as input instead of the original.
    pub chain_mode: bool,
    /// Canvas transparency checkerboard uses dark tones instead of light.
    #[serde(default)]
    pub dark_checker: bool,
    /// Auto-rerun Tier 2 on knob tweaks (live preview). Default ON.
    #[serde(default = "default_live_preview")]
    pub live_preview: bool,
    /// Auto-hide the adjustments toolbar when the cursor leaves it.
    #[serde(default)]
    pub auto_hide_adjustments: bool,
    /// Drag-out and Save emit subject/lines/mask PNGs instead of a single
    /// composite. Single toggle, both export paths.
    #[serde(default)]
    pub export_split_layers: bool,
    /// User-configurable keyboard shortcuts. Rebinding UI not yet wired.
    #[serde(default)]
    pub shortcuts: HashMap<String, String>,
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

    /// Force CPU inference even when GPU is available (not persisted — resets each launch).
    #[serde(skip)]
    pub force_cpu: bool,
    #[serde(skip)]
    pub active_backend: String,
    #[serde(default)]
    pub brush: BrushSettings,

    /// Free RAM the SD pre-flight gate requires *on top of* the model's
    /// declared `working_set_mb`. Default 2 GB matches the historical
    /// hardcoded `SAFETY_MARGIN_MB`. Lower → SD runs in tighter
    /// memory situations (riskier, may swap-thrash). Higher → more
    /// conservative on systems where other apps spike during inference.
    #[serde(default = "default_ram_safety_margin_gb")]
    pub ram_safety_margin_gb: f32,
}

fn default_ram_safety_margin_gb() -> f32 { 2.0 }

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

fn default_live_preview() -> bool { true }
fn default_preset_name() -> String { PRUNR_PRESET.to_string() }

impl Settings {
    /// Config file path: ~/.config/prunr/settings.json (Linux),
    /// ~/Library/Application Support/prunr/settings.json (macOS),
    /// %APPDATA%/prunr/settings.json (Windows).
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("prunr").join("settings.json"))
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
        let value: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(%e, "settings.json is corrupt; falling back to defaults");
                return Self::default();
            }
        };

        let mut settings: Self = match serde_json::from_value(value.clone()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "settings.json is corrupt; falling back to defaults");
                return Self::default();
            }
        };

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
                let wrap_key = Settings::default()
                    .model
                    .to_model_id()
                    .expect("Settings::default().model always has a model_id");
                let mut models = HashMap::new();
                models.insert(super::presets::model_id_key(wrap_key), super::presets::ModelPreset {
                    item_settings: legacy_item,
                    brush: Default::default(),
                    sd: None,
                });
                let file = super::presets::PresetFile {
                    format_version: super::presets::PRESET_FORMAT_VERSION,
                    models,
                };
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

    /// Whether the active backend is a GPU (not CPU).
    pub fn is_gpu(&self) -> bool {
        !self.active_backend.is_empty() && !self.active_backend.eq_ignore_ascii_case("CPU")
    }

    /// Smart default for parallel jobs based on backend and model. In
    /// filter-only mode (`SettingsModel::None`) the pipeline runs pure CPU
    /// on already-decoded pixels, so the ORT-memory cap doesn't apply.
    pub fn default_jobs(&self) -> usize {
        let base = if self.is_gpu() { 2 } else { (num_cpus::get() / 2).max(1) };
        match self.model.to_model_kind() {
            Some(model) => base.min(super::memory::safe_max_jobs(model)),
            None => base,
        }
    }

    /// Max recommended parallel jobs based on backend and model.
    pub fn max_jobs(&self) -> usize {
        let base = if self.is_gpu() { 4 } else { num_cpus::get() };
        match self.model.to_model_kind() {
            Some(model) => base.min(super::memory::safe_max_jobs(model)),
            None => base,
        }
    }

    /// Resolve a preset name to its `ItemSettings` for the active model.
    /// "Prunr" is always `ItemSettings::default()`; user presets come from
    /// the per-model entry keyed by `Settings.model.to_model_id()`. Missing
    /// preset, missing model entry, or filter-only mode (no model id) all
    /// fall back to factory defaults.
    pub fn preset_values(&self, name: &str) -> ItemSettings {
        if name == PRUNR_PRESET {
            return ItemSettings::default();
        }
        let Some(file) = self.presets.get(name) else { return ItemSettings::default() };
        let Some(model_id) = self.model.to_model_id() else {
            // Filter-only mode has no model id. The active item still has
            // an ItemSettings — fall back to whichever model entry the
            // preset stored, or to factory defaults if none.
            return file.models.values().next()
                .map(|mp| mp.item_settings)
                .unwrap_or_default();
        };
        file.models.get(&super::presets::model_id_key(model_id))
            .map(|mp| mp.item_settings)
            .unwrap_or_default()
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

    /// Reset to `Default` while preserving identity-bearing fields the user
    /// expects to survive — the active backend probe and the user-authored
    /// preset library + chosen default. `parallel_jobs` re-derives so it
    /// snaps to a sane value for whatever GPU/CPU is detected.
    pub fn reset_preserving_identity(&mut self) {
        let backend = self.active_backend.clone();
        let presets = std::mem::take(&mut self.presets);
        let default_preset = self.default_preset.clone();
        *self = Self::default();
        self.active_backend = backend;
        self.parallel_jobs = self.default_jobs();
        self.presets = presets;
        self.default_preset = default_preset;
    }
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
        if let Some(x) = v.get("mask_threshold").and_then(|x| x.as_f64()) {
            s.threshold = Some(x as f32);
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SettingsModel {
    Silueta,
    U2net,
    BiRefNetLite,
    /// "Filter-only" mode — no segmentation inference. The pipeline bypasses
    /// the seg stage entirely and just applies fill_style / bg_effect to the
    /// source. DexiNed still runs if `line_mode == EdgesOnly` (edges don't
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
}

impl SettingsModel {
    /// All variants in display order — source of truth for the model dropdown.
    pub const ALL: [Self; 8] = [
        Self::Silueta,
        Self::U2net,
        Self::BiRefNetLite,
        Self::None,
        Self::Inpaint,
        Self::BigInpaint,
        Self::MiganInpaint,
        Self::SdInpaint,
    ];

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
        matches!(self, Self::Inpaint | Self::BigInpaint | Self::MiganInpaint | Self::SdInpaint)
    }

    /// Convert to `ModelKind`, or `None` for non-seg variants.
    pub fn to_model_kind(self) -> Option<ModelKind> {
        match self {
            Self::Silueta => Some(ModelKind::Silueta),
            Self::U2net => Some(ModelKind::U2net),
            Self::BiRefNetLite => Some(ModelKind::BiRefNetLite),
            Self::None | Self::Inpaint | Self::BigInpaint | Self::MiganInpaint | Self::SdInpaint => None,
        }
    }

    /// Registry id, or `None` for the no-model variant.
    pub fn to_model_id(self) -> Option<prunr_models::ModelId> {
        match self {
            Self::Silueta => Some(prunr_models::ModelId::Silueta),
            Self::U2net => Some(prunr_models::ModelId::U2net),
            Self::BiRefNetLite => Some(prunr_models::ModelId::BiRefNetLite),
            Self::Inpaint => Some(prunr_models::ModelId::LaMaFp32),
            Self::BigInpaint => Some(prunr_models::ModelId::BigLaMa),
            Self::MiganInpaint => Some(prunr_models::ModelId::Migan),
            Self::SdInpaint => Some(prunr_models::ModelId::SdV15InpaintFp16),
            Self::None => None,
        }
    }
}

impl From<ModelKind> for SettingsModel {
    fn from(m: ModelKind) -> Self {
        match m {
            ModelKind::Silueta => SettingsModel::Silueta,
            ModelKind::U2net => SettingsModel::U2net,
            ModelKind::BiRefNetLite => SettingsModel::BiRefNetLite,
        }
    }
}

pub use prunr_core::LineMode;

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: SettingsModel::BiRefNetLite,
            auto_process_on_import: false,
            parallel_jobs: (num_cpus::get() / 2).max(1),
            history_depth: 10,
            chain_mode: false,
            dark_checker: false,
            live_preview: true,
            auto_hide_adjustments: false,
            export_split_layers: false,
            shortcuts: HashMap::new(),
            presets: HashMap::new(),
            default_preset: default_preset_name(),
            bg_image_paths: HashMap::new(),
            accepted_licenses: Vec::new(),
            runtime_prompt_snoozed_until: std::collections::HashMap::new(),
            force_cpu: false,
            active_backend: "CPU".to_string(),
            brush: BrushSettings::default(),
            ram_safety_margin_gb: default_ram_safety_margin_gb(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item_settings::item_with_gamma;
    use prunr_core::LineMode;

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
        assert_eq!(migrated.threshold, Some(0.3));
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
        assert_eq!(migrated.threshold, None);
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

    /// Wrap an `ItemSettings` into a single-entry v2 `PresetFile` keyed
    /// by the active model. Mirrors the runtime save flow's wrap step.
    fn wrap_for_model(s: &Settings, item: ItemSettings) -> super::super::presets::PresetFile {
        let mid = s.model.to_model_id().expect("test settings have a model_id");
        let mut models = HashMap::new();
        models.insert(super::super::presets::model_id_key(mid), super::super::presets::ModelPreset {
            item_settings: item,
            brush: Default::default(),
            sd: None,
        });
        super::super::presets::PresetFile {
            format_version: super::super::presets::PRESET_FORMAT_VERSION,
            models,
        }
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
            auto_process_on_import: true,
            history_depth: 25,
            chain_mode: true,
            ..Settings::default()
        };
        original.save_to_path(&path);

        let json = std::fs::read_to_string(&path).expect("settings.json should exist");
        let restored: Settings = serde_json::from_str(&json).expect("schema round-trip");
        assert_eq!(restored.parallel_jobs, 7);
        assert!(restored.auto_process_on_import);
        assert_eq!(restored.history_depth, 25);
        assert!(restored.chain_mode);
        // `force_cpu` and `active_backend` are #[serde(skip)] by design
        // (machine-specific runtime state, reset on every launch). The
        // round-trip must not preserve them.
        assert!(!restored.force_cpu);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn brush_settings_round_trip() {
        use prunr_core::brush::{BrushMode, BrushShape};
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

    #[test]
    fn config_path_is_under_prunr_subfolder() {
        // Cross-platform contract: settings always live at
        // <config_dir>/prunr/settings.json on Linux/macOS/Windows.
        // `dirs::config_dir` returns None only in genuinely broken envs.
        let path = Settings::config_path().expect("config_dir resolves on test host");
        assert!(
            path.ends_with("prunr/settings.json") || path.ends_with("prunr\\settings.json"),
            "expected .../prunr/settings.json, got {path:?}",
        );
    }

    /// Non-LCM scheduler short-circuits before the install gate —
    /// pins the AND ordering so a refactor can't silently turn the
    /// scheduler check into an OR.
    #[test]
    fn lcm_routing_inactive_for_non_lcm_scheduler() {
        use crate::gui::brush_state::SdScheduler;
        let mut s = Settings::default();
        s.brush.sd_scheduler = SdScheduler::Ddim;
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
}
