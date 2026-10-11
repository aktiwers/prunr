use crate::runtime_install::RuntimeId;

/// What is installed on disk, read once at launch and after every
/// install, download or delete, so the toolbar never stats the
/// filesystem per frame.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HardwareInstallCache {
    pub openvino: bool,
    /// The TAESD fast decoder bundle.
    pub taesd: bool,
    /// The LCM eraser bundle, as `Settings::can_select_lcm_scheduler` sees it.
    pub lcm_bundle: bool,
    pub x2plus: bool,
    /// Bit `i` set when `SettingsModel::ALL[i]` is installed.
    models: u16,
}

impl HardwareInstallCache {
    pub fn refresh() -> Self {
        use prunr_models::ModelId;
        Self {
            openvino: RuntimeId::OpenVino.is_installed(),
            taesd: prunr_models::is_available(ModelId::TaesdFp16),
            lcm_bundle: crate::gui::settings::Settings::can_select_lcm_scheduler(),
            x2plus: prunr_models::is_available(ModelId::RealEsrganX2Plus),
            models: crate::gui::settings::SettingsModel::ALL.iter().enumerate()
                .filter(|(_, m)| m.to_model_id().is_none_or(prunr_models::is_available))
                .fold(0, |bits, (i, _)| bits | 1 << i),
        }
    }

    /// Whether `model` can be picked: bundled, downloaded, or No model.
    pub fn has_model(&self, model: crate::gui::settings::SettingsModel) -> bool {
        crate::gui::settings::SettingsModel::ALL.iter().position(|m| *m == model)
            .is_some_and(|i| self.models & (1 << i) != 0)
    }
}
