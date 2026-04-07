use bgprunr_core::ModelKind;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: SettingsModel,
    pub auto_remove_on_import: bool,
    pub parallel_jobs: usize,
    pub reveal_animation_enabled: bool,
    #[serde(skip)]
    pub active_backend: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SettingsModel {
    Silueta,
    U2net,
}

impl From<SettingsModel> for ModelKind {
    fn from(m: SettingsModel) -> Self {
        match m {
            SettingsModel::Silueta => ModelKind::Silueta,
            SettingsModel::U2net => ModelKind::U2net,
        }
    }
}

impl From<ModelKind> for SettingsModel {
    fn from(m: ModelKind) -> Self {
        match m {
            ModelKind::Silueta => SettingsModel::Silueta,
            ModelKind::U2net => SettingsModel::U2net,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: SettingsModel::Silueta,
            auto_remove_on_import: false,
            parallel_jobs: (num_cpus::get() / 2).max(1),
            reveal_animation_enabled: true,
            active_backend: "CPU".to_string(),
        }
    }
}
