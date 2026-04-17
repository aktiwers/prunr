use std::path::PathBuf;
use prunr_core::{MaskSettings, ModelKind};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: SettingsModel,
    pub auto_remove_on_import: bool,
    pub parallel_jobs: usize,
    /// Mask gamma: >1 = more aggressive removal, <1 = gentler. Default 1.0.
    pub mask_gamma: f32,
    /// Optional binary threshold (0.0–1.0). 0.0 means disabled (soft mask).
    pub mask_threshold: f32,
    /// Whether binary threshold is enabled.
    pub mask_threshold_enabled: bool,
    /// Edge shift in pixels: >0 erodes (shrinks), <0 dilates (expands). Default 0.
    pub edge_shift: f32,
    /// Refine mask edges using guided filter for better detail.
    pub refine_edges: bool,
    /// Line extraction mode.
    pub line_mode: LineMode,
    /// Line detection sensitivity: 0.0 = minimal lines, 1.0 = maximum detail. Default 0.5.
    pub line_strength: f32,
    /// Override line color instead of keeping original colors.
    pub solid_line_color: bool,
    /// The solid color for lines when solid_line_color is enabled. [R, G, B, A]
    pub line_color: [u8; 4],
    /// Fill transparent areas with a solid background color.
    pub apply_bg_color: bool,
    /// The background fill color. [R, G, B, A]
    pub bg_color: [u8; 4],
    /// Maximum number of undo steps per image. Default 10.
    pub history_depth: usize,
    /// When true, Process uses the current result as input instead of the original image.
    pub chain_mode: bool,
    /// When true, canvas transparency checkerboard uses dark tones instead of light.
    /// `#[serde(default)]` so this new field doesn't reset older settings files.
    #[serde(default)]
    pub dark_checker: bool,
    /// Force CPU inference even when GPU is available (not persisted — resets each launch).
    #[serde(skip)]
    pub force_cpu: bool,
    #[serde(skip)]
    pub active_backend: String,
}

impl Settings {
    /// Config file path: ~/.config/prunr/settings.json (Linux),
    /// ~/Library/Application Support/prunr/settings.json (macOS),
    /// %APPDATA%/prunr/settings.json (Windows).
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("prunr").join("settings.json"))
    }

    /// Load from disk, falling back to defaults if missing or corrupt.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else { return Self::default() };
        let Ok(data) = std::fs::read_to_string(&path) else { return Self::default() };
        match serde_json::from_str(&data) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: settings.json is corrupt ({e}), using defaults");
                Self::default()
            }
        }
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
    /// Capped at what the system can safely handle.
    pub fn default_jobs(&self) -> usize {
        let model: prunr_core::ModelKind = self.model.into();
        let safe = super::memory::safe_max_jobs(model);
        let base = if self.is_gpu() { 2 } else { (num_cpus::get() / 2).max(1) };
        base.min(safe)
    }

    /// Max recommended parallel jobs based on backend and model.
    /// Limited by available system RAM to prevent OOM.
    pub fn max_jobs(&self) -> usize {
        let model: prunr_core::ModelKind = self.model.into();
        let safe = super::memory::safe_max_jobs(model);
        let base = if self.is_gpu() { 4 } else { num_cpus::get() };
        base.min(safe)
    }

    /// Build a ProcessingRecipe from the current settings state.
    /// Used by tier routing to compare against each item's applied_recipe.
    pub fn current_recipe(&self) -> prunr_core::ProcessingRecipe {
        let model: ModelKind = self.model.into();
        let solid_line = self.solid_line_color_rgb();
        // Mask settings are irrelevant for EdgesOnly — fix to defaults
        // so changing gamma/threshold doesn't trigger unnecessary reprocessing.
        let mask = if self.line_mode == LineMode::EdgesOnly {
            prunr_core::MaskRecipe::new(1.0, None, 0.0, false)
        } else {
            prunr_core::MaskRecipe::new(
                self.mask_gamma,
                self.threshold_value(),
                self.edge_shift,
                self.refine_edges,
            )
        };
        prunr_core::ProcessingRecipe {
            inference: prunr_core::InferenceRecipe {
                model,
                uses_segmentation: self.line_mode != LineMode::EdgesOnly,
                uses_edge_detection: self.line_mode != LineMode::Off,
                line_strength_bits: self.line_strength.to_bits(),
                solid_line_color: solid_line,
            },
            mask,
            composite: prunr_core::CompositeRecipe {
                bg_color: if self.apply_bg_color {
                    Some([self.bg_color[0], self.bg_color[1], self.bg_color[2]])
                } else {
                    None
                },
                solid_line_color: solid_line,
            },
            was_chain: self.chain_mode,
        }
    }

    /// Solid line color as RGB triple, or None if disabled.
    pub fn solid_line_color_rgb(&self) -> Option<[u8; 3]> {
        if self.solid_line_color {
            Some([self.line_color[0], self.line_color[1], self.line_color[2]])
        } else {
            None
        }
    }

    /// Binary threshold value if enabled, None if soft mask.
    fn threshold_value(&self) -> Option<f32> {
        if self.mask_threshold_enabled { Some(self.mask_threshold) } else { None }
    }

    pub fn mask_settings(&self) -> MaskSettings {
        MaskSettings {
            gamma: self.mask_gamma,
            threshold: self.threshold_value(),
            edge_shift: self.edge_shift,
            refine_edges: self.refine_edges,
        }
    }
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
            mask_gamma: 1.0,
            mask_threshold: 0.5,
            mask_threshold_enabled: false,
            edge_shift: 0.0,
            refine_edges: false,
            line_mode: LineMode::Off,
            line_strength: 0.5,
            solid_line_color: false,
            line_color: [0, 0, 0, 255],
            apply_bg_color: false,
            bg_color: [255, 255, 255, 255],
            history_depth: 10,
            chain_mode: false,
            dark_checker: false,
            force_cpu: false,
            active_backend: "CPU".to_string(),
        }
    }
}
