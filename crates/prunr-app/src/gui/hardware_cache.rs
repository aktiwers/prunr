use crate::runtime_install::{RuntimeChoice, RuntimeId};

/// What is installed on disk, read once at launch and after every
/// install, download or delete, so the toolbar never stats the
/// filesystem per frame.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct HardwareInstallCache {
    pub openvino: bool,
    /// The app ships a GPU runtime (DirectML on Windows release builds).
    pub built_in_gpu: bool,
    /// The runtime the next start loads.
    pub next_runtime: Option<RuntimeChoice>,
    /// The runtime this process loaded.
    pub loaded_runtime: Option<RuntimeChoice>,
    /// The TAESD fast decoder bundle.
    pub taesd: bool,
    /// The LCM eraser bundle, as `Settings::can_select_lcm_scheduler` sees it.
    pub lcm_bundle: bool,
    pub x2plus: bool,
    /// The face models, in `FaceModel::ALL` order.
    pub faces: [bool; crate::gui::item_settings::FaceModel::ALL.len()],
    /// The depth model, for every depth effect.
    pub depth: bool,
    /// Bit `SettingsModel::index()` set when that model is installed.
    models: u32,
    /// Bit `i` set when `REGISTRY[i]` is on disk (or bundled).
    registry: u128,
}

fn registry_bits() -> u128 {
    debug_assert!(prunr_models::REGISTRY.len() <= 128, "the registry outgrew its bit set");
    prunr_models::REGISTRY.iter().enumerate()
        .filter(|(_, d)| prunr_models::is_available(d.id))
        .fold(0, |bits, (i, _)| bits | 1 << i)
}

impl HardwareInstallCache {
    pub fn refresh() -> Self {
        use prunr_models::ModelId;
        Self {
            openvino: RuntimeId::OpenVino.is_installed(),
            built_in_gpu: prunr_core::ort_runtime::BUILT_IN_HAS_GPU && prunr_core::ort_runtime::has_built_in(),
            next_runtime: RuntimeChoice::next(),
            loaded_runtime: RuntimeChoice::loaded(),
            taesd: prunr_models::is_available(ModelId::TaesdFp16),
            lcm_bundle: crate::gui::settings::Settings::can_select_lcm_scheduler(),
            x2plus: prunr_models::is_available(ModelId::RealEsrganX2Plus),
            faces: crate::gui::item_settings::FaceModel::ALL.map(|m| prunr_models::is_available(m.id())),
            depth: prunr_models::is_available(prunr_core::depth::MODEL.into()),
            models: model_bits(crate::gui::settings::SettingsModel::ALL.iter()
                .filter(|m| m.to_model_id().is_none_or(prunr_models::is_available))),
            registry: registry_bits(),
        }
    }

    /// Whether `id`'s weights are installed, as read at the last refresh.
    pub fn has_installed(&self, id: prunr_models::ModelId) -> bool {
        prunr_models::REGISTRY.iter().position(|d| d.id == id).is_some_and(|i| self.registry & (1 << i) != 0)
    }

    /// Every model and runtime present, for tests that drive runs; a
    /// test that wants something missing starts from `Default`.
    #[cfg(test)]
    pub(crate) fn all_installed() -> Self {
        Self {
            openvino: true,
            built_in_gpu: true,
            next_runtime: None,
            loaded_runtime: None,
            taesd: true,
            lcm_bundle: true,
            x2plus: true,
            faces: [true; crate::gui::item_settings::FaceModel::ALL.len()],
            depth: true,
            models: model_bits(&crate::gui::settings::SettingsModel::ALL),
            registry: u128::MAX,
        }
    }

    /// Only `models` installed, for tests that want a partial install.
    #[cfg(test)]
    pub(crate) fn with_models(models: &[crate::gui::settings::SettingsModel]) -> Self {
        Self { models: model_bits(models), ..Self::default() }
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
        self.models & model_bit(model) != 0
    }

    /// The Stable Diffusion erasers installed, each by the name the user
    /// knows it by and the model its memory need is read from: SD 1.5
    /// (either quality) and RORem.
    pub fn sd_erasers(&self) -> impl Iterator<Item = (&'static str, prunr_models::ModelId)> {
        use crate::gui::settings::SettingsModel;
        use prunr_models::ModelId;
        let sd15 = if self.has_model(SettingsModel::SdInpaint) {
            Some(ModelId::SdV15InpaintFp16)
        } else {
            self.lcm_bundle.then_some(ModelId::SdV15LcmInpaintFp16)
        };
        let rorem = self.has_model(SettingsModel::RoremInpaint).then_some(ModelId::RoremFp16);
        [sd15.map(|id| ("SD 1.5", id)), rorem.map(|id| ("RORem", id))].into_iter().flatten()
    }

    pub fn has_sd_eraser(&self) -> bool {
        self.sd_erasers().next().is_some()
    }
}

const _: () = assert!(crate::gui::settings::SettingsModel::ALL.len() <= 32, "one bit per model in HardwareInstallCache::models");

fn model_bit(model: crate::gui::settings::SettingsModel) -> u32 {
    1 << model.index()
}

fn model_bits<'a>(models: impl IntoIterator<Item = &'a crate::gui::settings::SettingsModel>) -> u32 {
    models.into_iter().fold(0, |bits, m| bits | model_bit(*m))
}

#[cfg(test)]
mod tests {
    /// A card reads installed weights from the cache, not the disk:
    /// nothing counts as installed until a refresh has read it.
    #[test]
    fn installed_weights_come_from_the_last_refresh() {
        let id = prunr_models::REGISTRY[0].id;
        assert!(!super::HardwareInstallCache::default().has_installed(id));
        assert!(super::HardwareInstallCache::all_installed().has_installed(id));
    }
}
