use crate::runtime_install::{RuntimeChoice, RuntimeId};

/// What is installed on disk, read once at launch and after every
/// install, download or delete, so the toolbar never stats the
/// filesystem per frame.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct HardwareInstallCache {
    pub openvino: bool,
    pub cuda: bool,
    pub migraphx: bool,
    /// NVIDIA's libraries beside the CUDA runtime.
    pub nvidia_libraries: bool,
    /// OpenVINO may run models on the Intel GPU from the next start.
    pub intel_gpu: bool,
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
    /// Bytes the downloaded models take, multi-part bundles included.
    pub models_on_disk: u64,
    /// What the last test of this computer found (Settings › Hardware).
    pub test: TestSummary,
}

/// What the last test of this computer found, for the places this
/// process reads (`engine::column_for`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TestSummary {
    /// When it finished, in seconds since 1970; `None`: never, or reset.
    pub tested: Option<u64>,
    /// Models it placed on the accelerator.
    pub there: usize,
    /// Models it left on the CPU.
    pub off: usize,
}

impl TestSummary {
    /// Some place is measured here: a test ran, or one was cancelled after
    /// its first models.
    pub fn any(&self) -> bool {
        self.tested.is_some() || self.there + self.off > 0
    }

    pub fn read() -> Self {
        prunr_core::ep_compat::refresh();
        let mut summary = Self { tested: prunr_core::ep_compat::tested(), ..Self::default() };
        for model in prunr_models::REGISTRY.iter().map(|d| d.id) {
            let measured = prunr_core::ep_compat::measured_for(model);
            for (column, place) in prunr_models::Column::ALL.into_iter().zip(measured) {
                let Some(place) = place.filter(|_| prunr_core::engine::column_for(model, column.provider()) == Some(column)) else { continue };
                match place {
                    prunr_models::Place::There(_) => summary.there += 1,
                    _ => summary.off += 1,
                }
            }
        }
        summary
    }

    /// "9 models with CUDA, 3 models without it."
    pub fn found(&self, accelerator: &str) -> String {
        use crate::gui::views::fmt::count;
        format!("{} with {accelerator}, {} without it.", count(self.there, "model", "models"), count(self.off, "model", "models"))
    }
}

/// Bytes the downloaded models take in `dir`, multi-part bundles included.
fn disk_usage_in(dir: &std::path::Path) -> u64 {
    use prunr_models::ModelSource;
    let size = |p: std::path::PathBuf| std::fs::metadata(p).map_or(0, |m| m.len());
    prunr_models::REGISTRY.iter().map(|d| match d.source {
        ModelSource::OnDemand { filename, .. } => size(dir.join(filename)),
        ModelSource::MultiPartOnDemand { subdir, parts, .. } => {
            parts.iter().map(|p| size(dir.join(subdir).join(p.filename))).sum()
        }
        ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => 0,
    }).sum()
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
            cuda: RuntimeId::Cuda.is_installed(),
            migraphx: RuntimeId::MiGraphX.is_installed(),
            nvidia_libraries: crate::runtime_install::nvidia_libraries_installed(),
            intel_gpu: prunr_core::ort_runtime::intel_gpu_chosen(),
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
            models_on_disk: prunr_models::on_demand_dir().map_or(0, |dir| disk_usage_in(&dir)),
            test: TestSummary::read(),
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
            cuda: true,
            migraphx: true,
            nvidia_libraries: false,
            intel_gpu: false,
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
            models_on_disk: 0,
            test: TestSummary::default(),
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
            RuntimeId::Cuda => self.cuda,
            RuntimeId::MiGraphX => self.migraphx,
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
        [sd15.map(|id| ("Stable Diffusion 1.5", id)), rorem.map(|id| ("RORem", id))].into_iter().flatten()
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
    /// Disk usage counts the multi-part bundles (Stable Diffusion) too.
    #[test]
    fn disk_usage_counts_single_files_and_multi_part_bundles() {
        let dir = tempfile::tempdir().unwrap();
        let single = prunr_models::REGISTRY.iter().find_map(|d| match d.source {
            prunr_models::ModelSource::OnDemand { filename, .. } => Some(filename), _ => None,
        }).unwrap();
        let (subdir, part) = prunr_models::REGISTRY.iter().find_map(|d| match d.source {
            prunr_models::ModelSource::MultiPartOnDemand { subdir, parts, .. } => Some((subdir, parts[0].filename)), _ => None,
        }).unwrap();
        std::fs::write(dir.path().join(single), [0u8; 10]).unwrap();
        std::fs::create_dir_all(dir.path().join(subdir)).unwrap();
        std::fs::write(dir.path().join(subdir).join(part), [0u8; 32]).unwrap();
        assert_eq!(super::disk_usage_in(dir.path()), 42);
    }

    /// A card reads installed weights from the cache, not the disk:
    /// nothing counts as installed until a refresh has read it.
    #[test]
    fn installed_weights_come_from_the_last_refresh() {
        let id = prunr_models::REGISTRY[0].id;
        assert!(!super::HardwareInstallCache::default().has_installed(id));
        assert!(super::HardwareInstallCache::all_installed().has_installed(id));
    }
}
