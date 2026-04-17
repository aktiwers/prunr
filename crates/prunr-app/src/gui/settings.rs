use std::collections::HashMap;
use std::path::PathBuf;
use prunr_core::ModelKind;

use super::item_settings::ItemSettings;

/// Global app config. Per-image knobs (gamma, threshold, line mode, bg, ...)
/// live on `BatchItem.settings: ItemSettings` instead. The settings modal
/// edits `item_defaults` (the template for new images); the adjustments
/// toolbar edits the current image's `ItemSettings` directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: SettingsModel,
    pub auto_remove_on_import: bool,
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
    /// User-configurable keyboard shortcuts (Phase 5 wires up the UI).
    #[serde(default)]
    pub shortcuts: HashMap<String, String>,
    /// Named presets: `HashMap<name, snapshot>`.
    #[serde(default)]
    pub presets: HashMap<String, ItemSettings>,
    /// Preset to apply to new items on import. `None` uses `item_defaults`.
    #[serde(default)]
    pub default_preset: Option<String>,
    /// Template applied to new items when `default_preset` is None.
    /// v1 settings migrate their per-image fields into this on first load.
    #[serde(default)]
    pub item_defaults: ItemSettings,

    /// Force CPU inference even when GPU is available (not persisted — resets each launch).
    #[serde(skip)]
    pub force_cpu: bool,
    #[serde(skip)]
    pub active_backend: String,
}

fn default_live_preview() -> bool { true }

impl Settings {
    /// Config file path: ~/.config/prunr/settings.json (Linux),
    /// ~/Library/Application Support/prunr/settings.json (macOS),
    /// %APPDATA%/prunr/settings.json (Windows).
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("prunr").join("settings.json"))
    }

    /// Load from disk, falling back to defaults if missing or corrupt.
    /// Migrates v1 per-image fields (mask_gamma, bg_color, line_mode, ...)
    /// into `item_defaults`.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else { return Self::default() };
        let Ok(data) = std::fs::read_to_string(&path) else { return Self::default() };

        // Parse to Value first so we can migrate v1 fields regardless of
        // whether strict struct parsing succeeds.
        let value: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("warning: settings.json is corrupt ({e}), using defaults");
                return Self::default();
            }
        };

        let mut settings: Self = match serde_json::from_value(value.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: settings.json is corrupt ({e}), using defaults");
                return Self::default();
            }
        };

        // Detect v1 format (presence of old per-image keys) and migrate.
        // Only migrate if item_defaults is still at ItemSettings::default(),
        // so we don't clobber an already-migrated v2 file on subsequent loads.
        let is_v1 = value.get("mask_gamma").is_some()
            || value.get("apply_bg_color").is_some()
            || value.get("line_mode").is_some();
        if is_v1 && settings.item_defaults == ItemSettings::default() {
            settings.item_defaults = migrate_v1_item_settings(&value);
        }

        settings
    }

    /// Save to disk. Errors are silently ignored (best-effort).
    pub fn save(&self) {
        let Some(path) = Self::config_path() else { return };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Whether the active backend is a GPU (not CPU).
    pub fn is_gpu(&self) -> bool {
        !self.active_backend.is_empty() && !self.active_backend.eq_ignore_ascii_case("CPU")
    }

    /// Smart default for parallel jobs based on backend and model.
    pub fn default_jobs(&self) -> usize {
        let model: prunr_core::ModelKind = self.model.into();
        let safe = super::memory::safe_max_jobs(model);
        let base = if self.is_gpu() { 2 } else { (num_cpus::get() / 2).max(1) };
        base.min(safe)
    }

    /// Max recommended parallel jobs based on backend and model.
    pub fn max_jobs(&self) -> usize {
        let model: prunr_core::ModelKind = self.model.into();
        let safe = super::memory::safe_max_jobs(model);
        let base = if self.is_gpu() { 4 } else { num_cpus::get() };
        base.min(safe)
    }

    /// ItemSettings template for new batch items. Resolves `default_preset`
    /// against the preset map, falling back to `item_defaults` if the named
    /// preset is missing.
    pub fn item_defaults_for_new_item(&self) -> ItemSettings {
        if let Some(ref name) = self.default_preset {
            if let Some(preset) = self.presets.get(name) {
                return *preset;
            }
        }
        self.item_defaults
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
}

impl From<SettingsModel> for ModelKind {
    fn from(m: SettingsModel) -> Self {
        match m {
            SettingsModel::Silueta => ModelKind::Silueta,
            SettingsModel::U2net => ModelKind::U2net,
            SettingsModel::BiRefNetLite => ModelKind::BiRefNetLite,
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
            model: SettingsModel::Silueta,
            auto_remove_on_import: false,
            parallel_jobs: (num_cpus::get() / 2).max(1),
            history_depth: 10,
            chain_mode: false,
            dark_checker: false,
            live_preview: true,
            auto_hide_adjustments: false,
            shortcuts: HashMap::new(),
            presets: HashMap::new(),
            default_preset: None,
            item_defaults: ItemSettings::default(),
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
    fn v1_migration_populates_item_defaults() {
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
    fn item_defaults_for_new_item_uses_preset() {
        let mut s = Settings::default();
        let mut preset_values = ItemSettings::default();
        preset_values.gamma = 2.0;
        s.presets.insert("Portrait".to_string(), preset_values);
        s.default_preset = Some("Portrait".to_string());

        let new_item = s.item_defaults_for_new_item();
        assert_eq!(new_item.gamma, 2.0);
    }

    #[test]
    fn item_defaults_for_new_item_falls_back_if_preset_missing() {
        let mut s = Settings::default();
        s.default_preset = Some("NonExistent".to_string());
        s.item_defaults.gamma = 1.2;

        let new_item = s.item_defaults_for_new_item();
        assert_eq!(new_item.gamma, 1.2);
    }
}
