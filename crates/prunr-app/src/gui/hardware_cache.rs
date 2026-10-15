use crate::runtime_install::{RuntimeChoice, RuntimeId};

/// What is installed on disk, read once at launch and after every
/// install, download or delete, so the toolbar never stats the
/// filesystem per frame.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct HardwareInstallCache {
    pub openvino: bool,
    /// The app ships a GPU runtime (DirectML on Windows release builds).
    pub built_in_gpu: bool,
    /// The runtime the store's preference names, `RuntimeChoice::preferred`.
    pub preferred_runtime: Option<RuntimeChoice>,
    /// The runtime this process loaded, `RuntimeChoice::loaded`.
    pub loaded_runtime: Option<RuntimeChoice>,
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
            built_in_gpu: prunr_core::ort_runtime::BUILT_IN_HAS_GPU && prunr_core::ort_runtime::has_built_in(),
            preferred_runtime: RuntimeChoice::preferred(),
            loaded_runtime: RuntimeChoice::loaded(),
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
            built_in_gpu: true,
            preferred_runtime: None,
            loaded_runtime: None,
            taesd: true,
            lcm_bundle: true,
            x2plus: true,
            faces: [true; 2],
            depth: true,
            models: crate::gui::settings::SettingsModel::ALL.iter().fold(0, |bits, m| bits | 1 << m.index()),
        }
    }

    pub fn has_runtime(&self, runtime: RuntimeId) -> bool {
        match runtime {
            RuntimeId::OpenVino => self.openvino,
        }
    }

    /// Whether `choice` can load: the built-in GPU runtime is there, or
    /// the store install is.
    pub fn can_load(&self, choice: RuntimeChoice) -> bool {
        match choice {
            RuntimeChoice::BuiltIn => self.built_in_gpu,
            RuntimeChoice::Store(rt) => self.has_runtime(rt),
        }
    }

    pub fn has_face_model(&self, model: crate::gui::item_settings::FaceModel) -> bool {
        self.faces[model as usize]
    }

    /// Whether `model` can be picked: bundled, downloaded, or No model.
    pub fn has_model(&self, model: crate::gui::settings::SettingsModel) -> bool {
        self.models & (1 << model.index()) != 0
    }

    /// A Stable Diffusion eraser is installed: SD 1.5 (either quality) or RORem.
    pub fn has_sd_eraser(&self) -> bool {
        use crate::gui::settings::SettingsModel;
        self.has_model(SettingsModel::SdInpaint) || self.has_model(SettingsModel::RoremInpaint) || self.lcm_bundle
    }
}

const _: () = assert!(crate::gui::settings::SettingsModel::ALL.len() <= 32, "one bit per model in HardwareInstallCache::models");
