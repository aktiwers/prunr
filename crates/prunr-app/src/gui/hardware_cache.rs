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
    /// The add-on is installed but this process loaded an older runtime
    /// from the Runtime Store, which cannot take it.
    pub addon_waits: bool,
    /// OpenVINO may run models on the Intel GPU from the next start.
    pub intel_gpu: bool,
    /// The app ships a GPU runtime (DirectML on Windows release builds).
    pub built_in_gpu: bool,
    /// The app ships its own runtime beside it.
    pub built_in: bool,
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
}

/// What the tests of this computer measured, as Settings › Hardware shows
/// it: a row per installed model measured here, a cell per engine. Read
/// with the install cache, never per frame.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct TestResults {
    /// When a whole test finished, in seconds since 1970; `None`: never,
    /// or forgotten.
    pub tested: Option<u64>,
    /// The engines beside the processor, in the order sessions try them.
    pub engines: Vec<Engine>,
    pub rows: Vec<TestRow>,
    /// What changed since this computer was tested and dropped what it
    /// measured; empty when nothing did.
    pub changed: Vec<prunr_core::ep_compat::ScopePart>,
    /// The installed models a test would time here (`bench::here::times`).
    pub testable: Vec<prunr_models::ModelId>,
    /// "the processor, DirectML and CUDA": what a test times each model on.
    pub engines_said: String,
    /// Read once the accelerator probe had answered: before, the engines
    /// and what a test would time are not known, and asking would wait
    /// for a graphics driver.
    pub after_probe: bool,
}

/// An engine of this computer's: one this process runs models on, or
/// CUDA where an NVIDIA card could use it and it is not installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Engine {
    /// Its provider's name.
    pub name: &'static str,
    pub installed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TestRow {
    pub model: prunr_models::ModelId,
    /// A run's time on the processor.
    pub cpu: Option<Time>,
    /// One per engine of `TestResults::engines`.
    pub cells: Vec<Cell>,
    /// The provider it runs on while the accelerators are on.
    pub runs_on: &'static str,
    /// Measured on every accelerator a test tries it on; a cancelled or
    /// failed run leaves some untimed.
    pub complete: bool,
}

impl TestRow {
    /// The provider it runs on: the processor under "CPU only".
    pub fn engine(&self, cpu_only: bool) -> &'static str {
        if cpu_only { prunr_core::engine::CPU_NAME } else { self.runs_on }
    }

    /// Its time on `provider`, one of `engines` or the processor.
    pub fn time_on(&self, engines: &[Engine], provider: &str) -> Option<&Time> {
        if provider == prunr_core::engine::CPU_NAME {
            return self.cpu.as_ref();
        }
        engines.iter().zip(&self.cells).find(|(e, _)| e.name == provider).and_then(|(_, cell)| match cell {
            Cell::Time(time) => Some(time),
            _ => None,
        })
    }
}

/// A run's time, and how the window says it, said once when it is read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Time {
    pub ms: f32,
    pub text: String,
}

impl Time {
    pub fn of(ms: f32) -> Self {
        Self { ms, text: crate::gui::views::fmt::seconds(ms) }
    }
}

/// What a test found for one model on one engine; an engine not installed
/// has `Untested`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Cell {
    Time(Time),
    Fails,
    /// A different picture from the processor's.
    Differs,
    Untested,
}

impl Cell {
    pub fn of(measured: Option<&prunr_core::ep_compat::Measured>) -> Self {
        use prunr_core::ep_compat::PlaceKind;
        match measured {
            None => Self::Untested,
            Some(m) => match m.kind {
                PlaceKind::Fails => Self::Fails,
                PlaceKind::Differs => Self::Differs,
                PlaceKind::There | PlaceKind::Slower | PlaceKind::NoGain => m.ms.map_or(Self::Untested, |ms| Self::Time(Time::of(ms))),
            },
        }
    }
}

impl TestResults {
    /// "the graphics driver", "CUDA and the graphics driver": what changed
    /// since the last test, as people say it; `None` when nothing did.
    pub fn what_changed(&self) -> Option<String> {
        use prunr_core::ep_compat::ScopePart;
        let said = |part| match part {
            ScopePart::Drivers => "the graphics driver",
            ScopePart::Addons => "an accelerator add-on",
            ScopePart::Ort => "Prunr's runtime",
            ScopePart::Setup => "how Prunr sets models up",
        };
        (!self.changed.is_empty()).then(|| crate::gui::views::fmt::and_list(self.changed.iter().map(|&part| said(part))))
    }

    /// The engines this process runs models on.
    pub fn installed(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.engines.iter().filter(|e| e.installed).map(|e| e.name)
    }

    /// Some place is measured here: a test ran, or one was cancelled after
    /// its first models.
    pub fn any(&self) -> bool {
        self.tested.is_some() || !self.rows.is_empty()
    }

    pub fn row(&self, model: prunr_models::ModelId) -> Option<&TestRow> {
        self.rows.iter().find(|r| r.model == model)
    }

    /// A test timed `model` on every accelerator it tries.
    pub fn tested_everywhere(&self, model: prunr_models::ModelId) -> bool {
        self.row(model).is_some_and(|row| row.complete)
    }

    /// Whether the Model Store's Test may time `id` here.
    pub fn testable(&self, id: prunr_models::ModelId) -> bool {
        self.testable.contains(&id)
    }

    /// The store's measurements for the models `installs` holds, on the
    /// engines this process runs (none until its probe has answered); CUDA
    /// where it could run on this computer's card and is not installed,
    /// so its column says so.
    pub fn read(installs: &HardwareInstallCache) -> Self {
        prunr_core::ep_compat::refresh();
        // Before the engines: a probe that answers in between leaves these
        // results to be read again, not marked as read after it.
        let after_probe = prunr_core::OrtEngine::probed();
        let mut engines: Vec<Engine> = prunr_core::engine::gpu_providers_now()
            .unwrap_or_default()
            .into_iter()
            .map(|name| Engine { name, installed: true })
            .collect();
        if installs.cuda_offered() && !engines.iter().any(|e| e.name == prunr_core::engine::CUDA_NAME) {
            engines.push(Engine { name: prunr_core::engine::CUDA_NAME, installed: false });
        }
        let installed = || prunr_models::REGISTRY.iter().map(|d| d.id).filter(|&id| installs.has_installed(id));
        let rows = installed()
            .filter_map(|id| {
                let measured = prunr_core::ep_compat::measurements(id);
                // By provider, not column: OpenVINO's Intel graphics a test
                // found failing reads as its processor column afterwards.
                let timed = |provider: &str| prunr_models::Column::ALL.iter().any(|c| c.provider() == provider && measured[c.index()].is_some());
                let complete = after_probe && prunr_core::engine::tries_in_a_test(id).iter().all(|&(provider, _)| timed(provider));
                test_row(id, &engines, &measured, |p| prunr_core::engine::column_for(id, p), prunr_core::memory::provider_now(id, false), complete)
            })
            .collect();
        let mut results = Self {
            tested: prunr_core::ep_compat::tested(),
            engines,
            rows,
            changed: prunr_core::ep_compat::changed_since_test(),
            testable: if after_probe { installed().filter(|&id| crate::bench::here::times(id)).collect() } else { Vec::new() },
            engines_said: String::new(),
            after_probe,
        };
        results.engines_said = crate::gui::views::fmt::and_list(std::iter::once(crate::gui::views::fmt::engine(prunr_core::engine::CPU_NAME)).chain(results.installed()));
        results
    }
}

/// `model`'s row from what was measured in each column; `column_of` names
/// the column a provider's session of it reads. A column the test does
/// not try, as Prunr's own tests found it fails there, says so. `None`
/// when no engine measured it.
fn test_row(
    model: prunr_models::ModelId,
    engines: &[Engine],
    measured: &[Option<prunr_core::ep_compat::Measured>],
    column_of: impl Fn(&str) -> Option<prunr_models::Column>,
    runs_on: &'static str,
    complete: bool,
) -> Option<TestRow> {
    let fails_by_registry = |column: prunr_models::Column| {
        prunr_models::registry_places(model).get(column).fails()
    };
    if measured.iter().all(Option::is_none) {
        return None;
    }
    let cells: Vec<Cell> = engines
        .iter()
        .map(|e| match (e.installed, column_of(e.name)) {
            (false, _) => Cell::Untested,
            (true, Some(column)) if measured[column.index()].is_none() && fails_by_registry(column) => Cell::Fails,
            (true, column) => Cell::of(column.and_then(|c| measured[c.index()].as_ref())),
        })
        .collect();
    let cpu = measured.iter().flatten().find_map(|m| m.cpu_ms).map(Time::of);
    Some(TestRow { model, cpu, cells, runs_on, complete })
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
            addon_waits: RuntimeId::Cuda.is_addon() && RuntimeId::Cuda.is_installed() && !prunr_core::ort_runtime::hosts_addons(),
            intel_gpu: prunr_core::ort_runtime::intel_gpu_chosen(),
            built_in_gpu: prunr_core::ort_runtime::BUILT_IN_HAS_GPU && prunr_core::ort_runtime::has_built_in(),
            built_in: prunr_core::ort_runtime::has_built_in(),
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
            addon_waits: false,
            intel_gpu: false,
            built_in_gpu: true,
            built_in: true,
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
        }
    }

    /// Only `models` installed, for tests that want a partial install.
    #[cfg(test)]
    pub(crate) fn with_models(models: &[crate::gui::settings::SettingsModel]) -> Self {
        Self { models: model_bits(models), ..Self::default() }
    }

    /// CUDA could run on this computer's NVIDIA card beside the built-in
    /// runtime, and is not installed.
    pub fn cuda_offered(&self) -> bool {
        !self.cuda && RuntimeId::Cuda.is_addon() && crate::hardware::profile().runs(RuntimeId::Cuda)
    }

    pub fn has_runtime(&self, runtime: RuntimeId) -> bool {
        match runtime {
            RuntimeId::OpenVino => self.openvino,
            RuntimeId::Cuda => self.cuda,
            RuntimeId::MiGraphX => self.migraphx,
        }
    }

    /// Whether `choice` can load: the app's own runtime is beside it, or
    /// the store install is there. Prunr's own can be chosen again over
    /// one from the store.
    pub fn can_load(&self, choice: RuntimeChoice) -> bool {
        match choice {
            RuntimeChoice::BuiltIn => self.built_in,
            RuntimeChoice::Store(rt) => !rt.is_addon() && self.has_runtime(rt),
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

    /// A row holds what each installed engine measured and the processor's
    /// time from any of them; a model no engine measured has no row.
    #[test]
    fn a_row_shows_each_engine_and_the_processor() {
        use super::{Cell, Engine};
        use prunr_core::ep_compat::{Measured, PlaceKind};
        use prunr_models::{Column, ModelId};
        let engines = [Engine { name: "DirectML", installed: true }, Engine { name: "CUDA", installed: false }];
        let column_of = |p: &str| Column::of(p);
        let mut measured: [Option<Measured>; Column::ALL.len()] = Default::default();
        assert_eq!(super::test_row(ModelId::Silueta, &engines, &measured, column_of, "CPU", true), None, "nothing measured");
        measured[Column::Cuda.index()] = Some(Measured::timed(PlaceKind::There, 440.0, 9_000.0, ""));
        let all = [Engine { name: "DirectML", installed: true }, Engine { name: "CUDA", installed: true }];
        let row = super::test_row(ModelId::BiRefNetLite, &all, &measured, column_of, "CUDA", true).expect("a row");
        assert_eq!(row.cells, [Cell::Fails, Cell::Time(super::Time::of(440.0))], "DirectML, never tried as Prunr's tests found it fails, says so");
        measured[Column::Cuda.index()] = None;
        measured[Column::DirectMl.index()] = Some(Measured::new(PlaceKind::Fails, "E_OUTOFMEMORY").with_times(None, Some(10_100.0)));
        let row = super::test_row(ModelId::BiRefNetLite, &engines, &measured, column_of, "CPU", true).expect("a row");
        assert_eq!(row.cells, [Cell::Fails, Cell::Untested]);
        assert_eq!(row.cpu.map(|t| t.ms), Some(10_100.0));
        measured[Column::DirectMl.index()] = Some(Measured::timed(PlaceKind::Slower, 900.0, 670.0, ""));
        let row = super::test_row(ModelId::Silueta, &engines, &measured, column_of, "CPU", true).expect("a row");
        assert_eq!((&row.cells[0], row.cpu.map(|t| t.ms)), (&Cell::Time(super::Time::of(900.0)), Some(670.0)), "a slower engine still shows its time");
        measured[Column::DirectMl.index()] = Some(Measured::new(PlaceKind::Differs, "mean 3/255"));
        assert_eq!(super::test_row(ModelId::Silueta, &engines, &measured, column_of, "CPU", true).map(|r| r.cells[0].clone()), Some(Cell::Differs));
    }

    /// Prunr's own runtime can be chosen whenever it is beside the app,
    /// over one from the store too; a store install only once installed.
    #[test]
    fn the_built_in_runtime_is_a_choice_wherever_it_ships() {
        use crate::runtime_install::{RuntimeChoice, RuntimeId};
        let mut installs = super::HardwareInstallCache { built_in: true, ..Default::default() };
        assert!(installs.can_load(RuntimeChoice::BuiltIn));
        assert!(!installs.can_load(RuntimeChoice::Store(RuntimeId::OpenVino)));
        installs.openvino = true;
        assert!(installs.can_load(RuntimeChoice::Store(RuntimeId::OpenVino)) && installs.can_load(RuntimeChoice::BuiltIn));
        assert!(!super::HardwareInstallCache::default().can_load(RuntimeChoice::BuiltIn), "none beside the app");
    }

    /// What changed since the test is said in people's words.
    #[test]
    fn what_changed_is_said_in_words() {
        use prunr_core::ep_compat::ScopePart;
        let results = |changed: &[ScopePart]| super::TestResults { changed: changed.to_vec(), ..Default::default() };
        assert_eq!(results(&[]).what_changed(), None);
        assert_eq!(results(&[ScopePart::Drivers]).what_changed().as_deref(), Some("the graphics driver"));
        assert_eq!(results(&[ScopePart::Addons, ScopePart::Drivers]).what_changed().as_deref(), Some("an accelerator add-on and the graphics driver"));
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
