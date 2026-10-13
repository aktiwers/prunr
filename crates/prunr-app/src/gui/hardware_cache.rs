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
    /// The face models, in `FaceModel::ALL` order.
    pub faces: [bool; 2],
    /// The depth model, for every depth effect.
    pub depth: bool,
    /// Bit `SettingsModel::index()` set when that model is installed.
    models: u32,
}

impl HardwareInstallCache {
    pub fn refresh() -> Self {
        use prunr_models::ModelId;
        Self {
            openvino: RuntimeId::OpenVino.is_installed(),
            taesd: prunr_models::is_available(ModelId::TaesdFp16),
            lcm_bundle: crate::gui::settings::Settings::can_select_lcm_scheduler(),
            x2plus: prunr_models::is_available(ModelId::RealEsrganX2Plus),
            faces: crate::gui::item_settings::FaceModel::ALL.map(|m| prunr_models::is_available(m.id())),
            depth: prunr_models::is_available(prunr_core::depth::MODEL.into()),
            models: crate::gui::settings::SettingsModel::ALL.iter()
                .filter(|m| m.to_model_id().is_none_or(prunr_models::is_available))
                .fold(0, |bits, m| bits | 1 << m.index()),
        }
    }

    /// Every model and runtime present, for tests that drive runs; a
    /// test that wants something missing starts from `Default`.
    #[cfg(test)]
    pub(crate) fn all_installed() -> Self {
        Self {
            openvino: true,
            taesd: true,
            lcm_bundle: true,
            x2plus: true,
            faces: [true; 2],
            depth: true,
            models: crate::gui::settings::SettingsModel::ALL.iter().fold(0, |bits, m| bits | 1 << m.index()),
        }
    }

    pub fn has_face_model(&self, model: crate::gui::item_settings::FaceModel) -> bool {
        self.faces[model as usize]
    }

    /// Whether `model` can be picked: bundled, downloaded, or No model.
    pub fn has_model(&self, model: crate::gui::settings::SettingsModel) -> bool {
        self.models & (1 << model.index()) != 0
    }
}

const _: () = assert!(crate::gui::settings::SettingsModel::ALL.len() <= 32, "one bit per model in HardwareInstallCache::models");
