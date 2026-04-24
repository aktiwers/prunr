use std::collections::HashMap;
use std::path::{Path, PathBuf};
use prunr_core::ModelKind;

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
    pub presets: HashMap<String, ItemSettings>,
    /// Which preset new imports inherit, and what Reset All Knobs restores.
    /// Always set; defaults to "Prunr". If the named preset goes missing
    /// (user deleted it out of serde state), resolution falls back to "Prunr".
    #[serde(default = "default_preset_name")]
    pub default_preset: String,

    /// Force CPU inference even when GPU is available (not persisted — resets each launch).
    #[serde(skip)]
    pub force_cpu: bool,
    #[serde(skip)]
    pub active_backend: String,
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
            let mut settings = Self::default();
            settings.presets = super::presets_fs::load_all();
            return settings;
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
        if is_v1 && !settings.presets.contains_key(LEGACY_PRESET) {
            let legacy = migrate_v1_item_settings(&value);
            if legacy != ItemSettings::default() {
                settings.presets.insert(LEGACY_PRESET.to_string(), legacy);
            }
        }

        // One-shot migration: if settings.json still embeds presets from an
        // older build, write each to the filesystem store and persist the
        // cleared settings (skip_serializing drops them from settings.json).
        // The embedded copies stay in memory for this session — no need to
        // re-parse what we just wrote.
        if !settings.presets.is_empty() {
            for (name, values) in &settings.presets {
                let _ = super::presets_fs::save(name, values);
            }
            settings.save();
            return settings;
        }

        // Steady state: no embedded presets. Load from the filesystem store.
        settings.presets = super::presets_fs::load_all();
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

    /// Resolve a preset name to its values. "Prunr" is always
    /// `ItemSettings::default()`; user presets come from the map. A missing
    /// user preset falls back to factory defaults.
    pub fn preset_values(&self, name: &str) -> ItemSettings {
        if name == PRUNR_PRESET {
            return ItemSettings::default();
        }
        self.presets.get(name).copied().unwrap_or_default()
    }

    /// ItemSettings template for new batch items — the default_preset's values.
    pub fn item_defaults_for_new_item(&self) -> ItemSettings {
        self.preset_values(&self.default_preset)
    }
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
}

impl SettingsModel {
    /// All variants in display order — source of truth for the model dropdown.
    /// `None` is listed last so the dropdown visually separates it from the
    /// real models (UI draws a separator before the last entry).
    pub const ALL: [Self; 4] = [
        Self::Silueta,
        Self::U2net,
        Self::BiRefNetLite,
        Self::None,
    ];

    /// Whether this variant resolves to an ORT seg model. `None` skips
    /// segmentation inference entirely — callers branch on this rather
    /// than trying to convert `None` to a `ModelKind`.
    pub fn uses_segmentation(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Convert to `ModelKind`, or `None` for the filter-only variant.
    pub fn to_model_kind(self) -> Option<ModelKind> {
        match self {
            Self::Silueta => Some(ModelKind::Silueta),
            Self::U2net => Some(ModelKind::U2net),
            Self::BiRefNetLite => Some(ModelKind::BiRefNetLite),
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
            force_cpu: false,
            active_backend: "CPU".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn preset_values_prunr_is_factory() {
        let s = Settings::default();
        assert_eq!(s.preset_values(PRUNR_PRESET), ItemSettings::default());
    }

    #[test]
    fn preset_values_returns_user_preset() {
        let mut s = Settings::default();
        let mut portrait = ItemSettings::default();
        portrait.gamma = 2.0;
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
        let mut portrait = ItemSettings::default();
        portrait.gamma = 2.0;
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

        let mut original = Settings::default();
        original.parallel_jobs = 7;
        original.auto_process_on_import = true;
        original.history_depth = 25;
        original.chain_mode = true;
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
