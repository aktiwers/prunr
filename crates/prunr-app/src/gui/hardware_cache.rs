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
}

impl HardwareInstallCache {
    pub fn refresh() -> Self {
        use prunr_models::ModelId;
        Self {
            openvino: RuntimeId::OpenVino.is_installed(),
            taesd: prunr_models::is_available(ModelId::TaesdFp16),
            lcm_bundle: crate::gui::settings::Settings::can_select_lcm_scheduler(),
            x2plus: prunr_models::is_available(ModelId::RealEsrganX2Plus),
        }
    }
}
