//! Hardware profile detection — CPU vendor/brand + dGPU/iGPU vendors.
//! PCI vendor IDs come from sysfs on Linux and DXGI on Windows: neither
//! needs a graphics context, which `wgpu`/`ash` would fail to get in
//! headless containers.

#[cfg_attr(not(windows), allow(dead_code))] // only Windows lists adapters through DXGI; the tests run the rules everywhere
mod adapters;
#[cfg(windows)]
mod dxgi;

use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Apple,
    Other,
}

impl std::fmt::Display for CpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Intel => "Intel",
            Self::Amd => "AMD",
            Self::Apple => "Apple",
            Self::Other => "Other",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Other => "Other",
        })
    }
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_vendor: CpuVendor,
    pub cpu_brand: String,
    pub dgpu: Option<GpuVendor>,
    /// The graphics card's name, where the system tells it: DXGI on
    /// Windows, NVIDIA's driver on Linux.
    pub dgpu_name: Option<String>,
    /// The graphics card's PCI device id, which names its chip.
    pub dgpu_device: Option<u32>,
    pub igpu: Option<GpuVendor>,
    /// A graphics card DirectML runs well on (Windows only).
    pub suits_directml: bool,
    pub os: &'static str,
    pub arch: &'static str,
}

impl HardwareProfile {
    /// Whether `rt` has a device to run on here: CUDA an NVIDIA card from
    /// Turing on, MIGraphX an AMD one. OpenVINO's add-on lists Intel
    /// processors and graphics only; its runtime ran on any processor.
    pub fn runs(&self, rt: crate::runtime_install::RuntimeId) -> bool {
        use crate::runtime_install::RuntimeId;
        match rt {
            RuntimeId::OpenVino => !rt.is_addon() || self.recommends_openvino(),
            RuntimeId::Cuda => self.nvidia_runs_cuda(),
            RuntimeId::MiGraphX => self.recommends_rocm(),
        }
    }

    /// An NVIDIA card the CUDA add-on starts on: its cuDNN dropped
    /// Maxwell, Pascal and Volta. A card whose id went unread is given
    /// the benefit of the doubt.
    fn nvidia_runs_cuda(&self) -> bool {
        self.dgpu == Some(GpuVendor::Nvidia) && self.dgpu_device.is_none_or(|id| id >= crate::runtime_install::CUDA_FIRST_DEVICE_ID)
    }

    /// Intel graphics, built in or a card, which OpenVINO can run on.
    pub fn intel_graphics(&self) -> bool {
        self.igpu == Some(GpuVendor::Intel) || self.dgpu == Some(GpuVendor::Intel)
    }

    /// OpenVINO suits Intel graphics, and its add-on an Intel processor
    /// too: it ran the models 1.1 to 1.8 times as fast as Prunr's own
    /// processor engine on an i7-6700.
    pub fn recommends_openvino(&self) -> bool {
        self.intel_graphics() || (crate::runtime_install::RuntimeId::OpenVino.is_addon() && self.cpu_vendor == CpuVendor::Intel)
    }

    /// The runtime download the first-launch prompt and Settings
    /// recommend, among those offered here. An NVIDIA card CUDA starts on
    /// gets it where it is offered, Windows included: it runs models
    /// DirectML refuses or breaks. Any other card that suits DirectML
    /// needs none, since DirectML is built in, even with an Intel iGPU
    /// beside it. A graphics card wins over an Intel iGPU beside it (the
    /// usual Linux laptop): CUDA or MIGraphX runs the card, OpenVINO only
    /// the iGPU or the processor.
    pub fn recommended_runtime(&self) -> Option<crate::runtime_install::RuntimeId> {
        use crate::runtime_install::RuntimeId;
        let cuda = self.nvidia_runs_cuda().then_some(RuntimeId::Cuda).filter(|rt| RuntimeId::OFFERED.contains(rt));
        if cuda.is_some() || self.suits_directml {
            return cuda;
        }
        [
            self.recommends_rocm().then_some(RuntimeId::MiGraphX),
            self.recommends_openvino().then_some(RuntimeId::OpenVino),
        ]
        .into_iter()
        .flatten()
        .find(|rt| RuntimeId::OFFERED.contains(rt))
    }

    /// AMD discrete GPU only. APU iGPUs excluded — ROCm/MIGraphX
    /// coverage on AMD APUs is unreliable.
    pub fn recommends_rocm(&self) -> bool {
        self.dgpu == Some(GpuVendor::Amd)
    }
}

/// The graphics drivers a GPU provider runs on, as one string that a
/// driver update changes: each DXGI adapter on Windows; on Linux the
/// NVIDIA driver and ROCm; the macOS version, which CoreML comes with.
/// Empty where none is read. Every process reads the same, whatever its
/// settings: one that read another would drop what the others learned.
pub fn driver_identity() -> String {
    #[cfg(target_os = "windows")]
    {
        dxgi::driver_identity()
    }
    #[cfg(target_os = "linux")]
    {
        linux_driver_identity(|path| std::fs::read_to_string(path).ok())
    }
    #[cfg(target_os = "macos")]
    {
        sysinfo::System::os_version().map(|v| format!("macos {v}")).unwrap_or_default()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        String::new()
    }
}

/// The NVIDIA card's name as its driver lists it: the first card by bus,
/// read from a file the driver keeps, which wakes no card on a laptop the
/// way NVML can.
#[cfg(target_os = "linux")]
fn nvidia_card_name() -> Option<String> {
    let mut cards: Vec<std::path::PathBuf> = std::fs::read_dir("/proc/driver/nvidia/gpus").ok()?.flatten().map(|e| e.path().join("information")).collect();
    cards.sort();
    cards.iter().find_map(|card| model_named_in(&std::fs::read_to_string(card).ok()?))
}

/// The card's name in the driver's information file, its `Model:` line.
#[cfg(any(target_os = "linux", test))]
fn model_named_in(information: &str) -> Option<String> {
    information.lines().find_map(|line| line.strip_prefix("Model:")).map(|name| name.trim().to_string()).filter(|name| !name.is_empty())
}

/// Intel's graphics driver is left out: a record of OpenVINO on the
/// Intel graphics holds for it alone (`ep_compat`), so its update leaves
/// every other engine's records.
#[cfg(any(target_os = "linux", test))]
fn linux_driver_identity(read: impl Fn(&str) -> Option<String>) -> String {
    let read = |path: &str| read(path).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    [
        read("/sys/module/nvidia/version").map(|v| format!("nvidia {v}")),
        read("/opt/rocm/.info/version").map(|v| format!("rocm {v}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(", ")
}

pub fn profile() -> &'static HardwareProfile {
    static CACHE: OnceLock<HardwareProfile> = OnceLock::new();
    CACHE.get_or_init(detect_now)
}

/// Total system RAM (bytes), read once.
pub fn total_ram_bytes() -> u64 {
    static CACHE: OnceLock<u64> = OnceLock::new();
    *CACHE.get_or_init(|| prunr_core::sysmem::total_ram_bytes().unwrap_or(0))
}

/// Currently-available RAM (bytes). Re-reads on every call — fresh value
/// for UIs that show live headroom. ~1 ms on Linux. Don't call inside a
/// per-frame loop; prefer `available_ram_bytes_throttled` for any caller
/// that fires more than once per second. 0 when nothing is free, and when
/// the system reports no memory at all (`total_ram_bytes` 0 too), so the
/// gates refuse rather than guess.
pub fn available_ram_bytes_now() -> u64 {
    prunr_core::sysmem::available_ram_bytes().unwrap_or(0)
}

/// `available_ram_bytes_throttled` in MiB, the unit the admission gates
/// compare against a model's `needs`.
pub fn available_ram_mb_throttled() -> u32 {
    (available_ram_bytes_throttled() / (1024 * 1024)) as u32
}

/// Cached available-RAM read with a 1-second TTL; cheap enough for
/// per-frame UI code (~10 ns on a cache hit).
pub fn available_ram_bytes_throttled() -> u64 {
    static CACHE: Reading<u64> = Mutex::new(None);
    throttled(&CACHE, available_ram_bytes_now)
}

/// Free memory on CUDA's card in MB, read at most once a second;
/// `None` with no NVIDIA driver.
pub fn cuda_card_free_mb_throttled() -> Option<u64> {
    static CACHE: Reading<Option<u64>> = Mutex::new(None);
    throttled(&CACHE, || prunr_core::card_memory::cuda_card_memory().map(|c| c.free_mb))
}

/// `prunr_core::memory::card_free_mb`, read at most once a second.
pub fn card_free_mb_throttled(provider: &str) -> Option<u64> {
    (provider == prunr_core::engine::CUDA_NAME).then(cuda_card_free_mb_throttled).flatten()
}

/// What `id` takes where it will run, and what is free there now
/// (readings at most a second old); until the GPU probe answers, the
/// CPU's charge.
pub fn charge_and_room(id: prunr_models::ModelId, cpu_only: bool) -> Option<(prunr_core::memory::Charge, prunr_core::memory::Room)> {
    let provider = prunr_core::memory::provider_now(id, cpu_only);
    let card_free = card_free_mb_throttled(provider);
    let charge = prunr_core::memory::charge_with(id, provider, card_free)?;
    let card_mb = card_free.filter(|_| charge.on_card());
    Some((charge, prunr_core::memory::Room { ram_mb: available_ram_mb_throttled().into(), card_mb }))
}

/// Whether a run of `id` goes to the CPU: when `cpu_only`, when its GPU
/// provider's charge does not fit what is free and the CPU's does, or
/// when the card ran out of memory for it and has gained no room since.
/// `Err`: it fits neither.
pub fn runs_on_cpu(id: prunr_models::ModelId, cpu_only: bool) -> Result<bool, prunr_core::memory::Short> {
    let cpu = charge_and_room(id, true);
    let short = !cpu_only && prunr_core::memory::ran_short(id, prunr_core::memory::provider_now(id, false));
    let asked = if cpu_only || short { cpu } else { charge_and_room(id, false) };
    prunr_core::memory::falls_to_cpu(asked, cpu).map(|fell| fell || cpu_only || short)
}

/// A reading and when it was taken.
type Reading<T> = Mutex<Option<(std::time::Instant, T)>>;

/// `cache`'s reading while under a second old, else `read` afresh.
fn throttled<T: Copy>(cache: &Reading<T>, read: impl FnOnce() -> T) -> T {
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = std::time::Instant::now();
    match *guard {
        Some((at, value)) if now.duration_since(at) < std::time::Duration::from_secs(1) => value,
        _ => {
            let value = read();
            *guard = Some((now, value));
            value
        }
    }
}

fn detect_now() -> HardwareProfile {
    let (cpu_vendor, cpu_brand) = detect_cpu();
    let GpuSlots { dgpu, dgpu_name, dgpu_device, igpu, suits_directml } = detect_gpus(cpu_vendor);
    HardwareProfile {
        cpu_vendor,
        cpu_brand,
        dgpu,
        dgpu_name,
        dgpu_device,
        igpu,
        suits_directml,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    }
}

fn detect_cpu() -> (CpuVendor, String) {
    // Vendor + brand are static across the process; skip the per-core
    // frequency / usage sampling that `refresh_cpu_all` would do.
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_specifics(sysinfo::CpuRefreshKind::nothing());
    let cpu = sys.cpus().first();
    let brand = cpu.map(|c| c.brand().to_string()).unwrap_or_default();
    let vendor_id = cpu.map(|c| c.vendor_id().to_string()).unwrap_or_default();
    (classify_cpu_vendor(&vendor_id, &brand), brand)
}

/// The processor as Settings shows it: the brand string without its
/// trademark marks, led by the vendor only when the brand does not name it
/// already ("Intel (Intel(R) Core(TM) i7 …)" said Intel twice).
pub fn cpu_label(vendor: CpuVendor, brand: &str) -> String {
    let brand = brand.replace("(R)", "").replace("(TM)", "").replace("(tm)", "");
    let brand = brand.split_whitespace().collect::<Vec<_>>().join(" ");
    let vendor = vendor.to_string();
    if brand.is_empty() {
        vendor
    } else if vendor == "Other" || brand.to_ascii_lowercase().contains(&vendor.to_ascii_lowercase()) {
        brand
    } else {
        format!("{vendor} {brand}")
    }
}

fn classify_cpu_vendor(vendor_id: &str, brand: &str) -> CpuVendor {
    let vid = vendor_id.to_ascii_lowercase();
    if vid.contains("genuineintel") { return CpuVendor::Intel; }
    if vid.contains("authenticamd") { return CpuVendor::Amd; }
    if vid.contains("apple") { return CpuVendor::Apple; }
    // sysinfo on macOS sometimes returns empty vendor_id; fall back
    // to brand inspection since "Apple M1/M2/M3..." is unambiguous.
    let brand_lower = brand.to_ascii_lowercase();
    if brand_lower.starts_with("apple ") { return CpuVendor::Apple; }
    if brand_lower.contains("intel") { return CpuVendor::Intel; }
    if brand_lower.contains("amd") || brand_lower.contains("ryzen") { return CpuVendor::Amd; }
    CpuVendor::Other
}

/// The GPUs found, by slot.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct GpuSlots {
    dgpu: Option<GpuVendor>,
    /// The card's name, where the system tells it.
    dgpu_name: Option<String>,
    dgpu_device: Option<u32>,
    igpu: Option<GpuVendor>,
    suits_directml: bool,
}

fn detect_gpus(cpu_vendor: CpuVendor) -> GpuSlots {
    #[cfg(target_os = "linux")]
    {
        let mut slots = detect_gpus_linux(cpu_vendor);
        slots.dgpu_name = (slots.dgpu == Some(GpuVendor::Nvidia)).then(nvidia_card_name).flatten();
        slots
    }
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon: GPU is part of the SoC. Intel Macs return None
        // (we don't ship OpenVINO on macOS, so detail isn't needed).
        let _ = cpu_vendor;
        let igpu = (std::env::consts::ARCH == "aarch64").then_some(GpuVendor::Apple);
        GpuSlots { igpu, ..GpuSlots::default() }
    }
    #[cfg(target_os = "windows")]
    {
        // DXGI says which card is which; the CPU vendor guess is not needed.
        let _ = cpu_vendor;
        let (adapters, os_choice) = dxgi::adapters();
        adapters::sort_adapters(&adapters, os_choice)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = cpu_vendor;
        GpuSlots::default()
    }
}

#[cfg(target_os = "linux")]
fn detect_gpus_linux(cpu_vendor: CpuVendor) -> GpuSlots {
    let mut slots = GpuSlots::default();
    let Ok(read) = std::fs::read_dir("/sys/class/drm") else {
        return slots;
    };
    for entry in read.flatten() {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if !is_card_dir(&name_s) { continue; }
        let device = entry.path().join("device");
        let vendor_path = device.join("vendor");
        let Ok(vendor_raw) = std::fs::read_to_string(&vendor_path) else { continue };
        let gpu_vendor = parse_pci_id(&vendor_raw).map_or(GpuVendor::Other, gpu_vendor);
        // Heuristic: GPU vendor matching CPU vendor is the iGPU. Known
        // limitation: AMD CPU + AMD dGPU (Ryzen + Radeon RX) reads the
        // dGPU as an iGPU.
        let is_integrated = matches_cpu(gpu_vendor, cpu_vendor);
        if is_integrated {
            slots.igpu.get_or_insert(gpu_vendor);
        } else if slots.dgpu.is_none() {
            slots.dgpu = Some(gpu_vendor);
            slots.dgpu_device = std::fs::read_to_string(device.join("device")).ok().and_then(|raw| parse_pci_id(&raw));
        }
    }
    slots
}

#[cfg(target_os = "linux")]
fn is_card_dir(name: &str) -> bool {
    // Match `card0`, `card1`, ... but not `card0-DP-1` (those are
    // connector subnodes, not GPUs).
    name.starts_with("card")
        && name.len() >= 5
        && name[4..].chars().all(|c| c.is_ascii_digit())
}

/// A sysfs PCI id (`0x10de\n`).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // sysfs is Linux's; the tests run it everywhere
fn parse_pci_id(raw: &str) -> Option<u32> {
    u32::from_str_radix(raw.trim().trim_start_matches("0x"), 16).ok()
}

/// A PCI vendor id, as sysfs and DXGI both report it.
fn gpu_vendor(id: u32) -> GpuVendor {
    match id {
        0x10de => GpuVendor::Nvidia,
        0x1002 | 0x1022 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        0x106b => GpuVendor::Apple,
        _ => GpuVendor::Other,
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // only sysfs needs the CPU to tell the iGPU apart; the tests run it everywhere
fn matches_cpu(gpu: GpuVendor, cpu: CpuVendor) -> bool {
    matches!(
        (gpu, cpu),
        (GpuVendor::Intel, CpuVendor::Intel)
            | (GpuVendor::Amd, CpuVendor::Amd)
            | (GpuVendor::Apple, CpuVendor::Apple)
    )
}

/// The GPUs the system names, for a report a user sends back: Windows'
/// video controllers with their driver versions, macOS's chipsets, Linux's
/// PCI display devices. Runs a system tool (about a second on Windows), so
/// never on a frame; empty when the tool is missing or names none.
pub fn gpu_names() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut cmd = std::process::Command::new("powershell");
        cmd.args([
            "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name) (driver $($_.DriverVersion))\" }",
        ]).creation_flags(CREATE_NO_WINDOW);
        command_stdout(cmd).map_or_else(Vec::new, |out| non_empty_lines(&out))
    }
    #[cfg(target_os = "macos")]
    {
        let mut cmd = std::process::Command::new("system_profiler");
        cmd.arg("SPDisplaysDataType");
        command_stdout(cmd).map_or_else(Vec::new, |out| chipsets_from_system_profiler(&out))
    }
    #[cfg(target_os = "linux")]
    {
        command_stdout(std::process::Command::new("lspci")).map_or_else(Vec::new, |out| displays_from_lspci(&out))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Vec::new()
    }
}

fn command_stdout(mut cmd: std::process::Command) -> Option<String> {
    let out = cmd.stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(any(target_os = "windows", test))]
fn non_empty_lines(text: &str) -> Vec<String> {
    text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect()
}

/// `Chipset Model: Apple M2 Pro` lines of `system_profiler SPDisplaysDataType`.
#[cfg(any(target_os = "macos", test))]
fn chipsets_from_system_profiler(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("Chipset Model:"))
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect()
}

/// The device names of `lspci`'s VGA, 3D and display controllers.
#[cfg(any(target_os = "linux", test))]
fn displays_from_lspci(text: &str) -> Vec<String> {
    const CLASSES: [&str; 3] = ["VGA compatible controller: ", "3D controller: ", "Display controller: "];
    text.lines()
        .filter_map(|l| CLASSES.iter().find_map(|class| l.split_once(class).map(|(_, name)| name.trim().to_string())))
        .collect()
}

#[cfg(test)]
mod tests {
    /// The card's name is the driver's `Model:` line, trimmed.
    #[test]
    fn the_nvidia_card_is_named_as_its_driver_lists_it() {
        let information = "Model: \t\t NVIDIA GeForce RTX 3060\nIRQ:   \t\t 161\nGPU UUID: \t GPU-4c1c\n";
        assert_eq!(super::model_named_in(information).as_deref(), Some("NVIDIA GeForce RTX 3060"));
        assert_eq!(super::model_named_in("IRQ: 161\nModel:   \n"), None);
    }

    /// The Linux identity is the NVIDIA driver's and ROCm's versions,
    /// trimmed, in that order; no Intel graphics driver is in it.
    #[test]
    fn the_linux_driver_identity_names_nvidia_and_rocm_only() {
        let files = |present: &'static [(&'static str, &'static str)]| move |path: &str| present.iter().find(|(p, _)| *p == path).map(|(_, v)| v.to_string());
        const ALL: [(&str, &str); 4] = [("/sys/module/nvidia/version", "580.95.05\n"), ("/opt/rocm/.info/version", "7.0.1"), ("/sys/module/i915/srcversion", "ABCDEF"), ("/sys/module/xe/srcversion", "123")];
        let all: &'static [(&str, &str)] = &ALL;
        assert_eq!(super::linux_driver_identity(files(all)), "nvidia 580.95.05, rocm 7.0.1");
        assert_eq!(super::linux_driver_identity(files(&all[2..])), "", "an Intel module alone is no identity");
        assert_eq!(super::linux_driver_identity(files(&[("/sys/module/nvidia/version", "  ")])), "");
    }

    /// Settings offers a runtime only where it has a device: an Intel-only
    /// computer sees OpenVINO alone, CUDA needs an NVIDIA card, MIGraphX an
    /// AMD one.
    #[test]
    fn a_runtime_is_offered_only_for_hardware_it_runs_on() {
        use super::{CpuVendor, GpuVendor, HardwareProfile};
        use crate::runtime_install::RuntimeId::{Cuda, MiGraphX, OpenVino};
        let machine = |dgpu, igpu| HardwareProfile {
            cpu_vendor: CpuVendor::Intel, cpu_brand: String::new(), dgpu, dgpu_name: None, dgpu_device: None, igpu,
            suits_directml: false, os: "linux", arch: "x86_64",
        };
        let runs = |p: &HardwareProfile| [OpenVino, Cuda, MiGraphX].map(|rt| p.runs(rt));
        assert_eq!(runs(&machine(None, Some(GpuVendor::Intel))), [true, false, false], "Intel only");
        assert_eq!(runs(&machine(Some(GpuVendor::Nvidia), Some(GpuVendor::Intel))), [true, true, false], "NVIDIA card");
        assert_eq!(runs(&machine(Some(GpuVendor::Amd), None)), [true, false, true], "AMD card");
        assert_eq!(runs(&machine(None, Some(GpuVendor::Amd))), [true, false, false], "an AMD APU is not offered ROCm");
        let amd = |igpu| HardwareProfile { cpu_vendor: CpuVendor::Amd, ..machine(None, igpu) };
        assert_eq!(amd(None).runs(OpenVino), !OpenVino.is_addon(), "OpenVINO's add-on lists no AMD processor");
        assert!(amd(Some(GpuVendor::Intel)).runs(OpenVino), "but Intel graphics beside one");
    }

    /// The CUDA add-on's cuDNN runs Turing and later: a GTX 1080, a GT
    /// 1030 or a Tesla V100 would download it and never start it, so a
    /// Windows one keeps DirectML. A card whose id went unread is offered it.
    #[test]
    fn cuda_is_offered_only_to_turing_cards_and_later() {
        use crate::runtime_install::RuntimeId;
        let card = |dgpu_device| HardwareProfile {
            cpu_vendor: CpuVendor::Amd, cpu_brand: String::new(), dgpu: Some(GpuVendor::Nvidia), dgpu_name: None,
            dgpu_device, igpu: None, suits_directml: true, os: "windows", arch: "x86_64",
        };
        let offered = RuntimeId::OFFERED.contains(&RuntimeId::Cuda);
        for (device, runs) in [
            (Some(0x1b80), false),
            (Some(0x1d01), false),
            (Some(0x1db4), false),
            (Some(0x1e04), true),
            (Some(0x1f82), true),
            (Some(0x2182), true),
            (Some(0x2504), true),
            (Some(0x2d04), true),
            (None, true),
        ] {
            let p = card(device);
            assert_eq!(p.runs(RuntimeId::Cuda), runs, "{device:#x?}");
            assert_eq!(p.recommended_runtime() == Some(RuntimeId::Cuda), runs && offered, "{device:#x?}");
        }
    }

    #[test]
    fn a_cards_pci_id_is_read_as_sysfs_writes_it() {
        assert_eq!(parse_pci_id("0x2504\n"), Some(0x2504));
        assert_eq!(parse_pci_id("garbage"), None);
    }

    #[test]
    fn the_cpu_label_names_the_vendor_once() {
        assert_eq!(cpu_label(CpuVendor::Intel, "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz"), "Intel Core i7-6700 CPU @ 3.40GHz");
        assert_eq!(cpu_label(CpuVendor::Apple, "Apple M2"), "Apple M2");
        assert_eq!(cpu_label(CpuVendor::Amd, "Ryzen 7 5800X"), "AMD Ryzen 7 5800X");
        assert_eq!(cpu_label(CpuVendor::Intel, ""), "Intel");
    }

    use super::*;

    #[test]
    fn gpu_names_come_from_each_systems_tool_output() {
        let lspci = "00:00.0 Host bridge: Intel Corporation Xeon E3-1200 v5\n\
                     00:02.0 VGA compatible controller: Intel Corporation HD Graphics 530 (rev 06)\n\
                     01:00.0 3D controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)\n";
        assert_eq!(displays_from_lspci(lspci), ["Intel Corporation HD Graphics 530 (rev 06)", "NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)"]);
        let profiler = "Graphics/Displays:\n\n    Apple M2 Pro:\n\n      Chipset Model: Apple M2 Pro\n      Type: GPU\n      Total Number of Cores: 19\n";
        assert_eq!(chipsets_from_system_profiler(profiler), ["Apple M2 Pro"]);
        assert_eq!(non_empty_lines("NVIDIA GeForce RTX 4090 (driver 32.0.15.6094)\r\n\r\n"), ["NVIDIA GeForce RTX 4090 (driver 32.0.15.6094)"]);
    }

    /// The vendor id decides; without one, the brand string does.
    #[test]
    fn the_cpu_vendor_comes_from_the_vendor_id_or_the_brand() {
        for (vendor_id, brand, want) in [
            ("GenuineIntel", "", CpuVendor::Intel),
            ("AuthenticAMD", "", CpuVendor::Amd),
            ("", "Apple M3 Pro", CpuVendor::Apple),
            ("", "AMD Ryzen 9 7950X", CpuVendor::Amd),
        ] {
            assert_eq!(classify_cpu_vendor(vendor_id, brand), want, "{vendor_id:?} {brand:?}");
        }
    }

    #[test]
    fn parse_pci_vendor_known_ids() {
        let vendor = |raw: &str| parse_pci_id(raw).map_or(GpuVendor::Other, gpu_vendor);
        assert_eq!(vendor("0x10de\n"), GpuVendor::Nvidia);
        assert_eq!(vendor("0x1002\n"), GpuVendor::Amd);
        assert_eq!(vendor("0x8086\n"), GpuVendor::Intel);
        assert_eq!(vendor("0x1234\n"), GpuVendor::Other);
    }

    #[test]
    fn matches_cpu_pairs() {
        assert!(matches_cpu(GpuVendor::Intel, CpuVendor::Intel));
        assert!(matches_cpu(GpuVendor::Amd, CpuVendor::Amd));
        assert!(!matches_cpu(GpuVendor::Nvidia, CpuVendor::Intel));
        assert!(!matches_cpu(GpuVendor::Intel, CpuVendor::Amd));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn is_card_dir_filters_correctly() {
        assert!(is_card_dir("card0"));
        assert!(is_card_dir("card1"));
        assert!(is_card_dir("card12"));
        assert!(!is_card_dir("card"));
        assert!(!is_card_dir("card0-DP-1"));
        assert!(!is_card_dir("controlD64"));
    }

    /// OpenVINO is recommended for Intel graphics, built in or a card, and
    /// where it comes as an add-on for an Intel processor too.
    #[test]
    fn recommends_openvino_for_intel_hardware() {
        let addon = crate::runtime_install::RuntimeId::OpenVino.is_addon();
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: None,
            dgpu_name: None,
            dgpu_device: None,
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "linux", arch: "x86_64",
        };
        assert!(p.recommends_openvino() && p.intel_graphics());
        p.igpu = None;
        assert_eq!(p.recommends_openvino(), addon, "an Intel processor alone");
        assert!(!p.intel_graphics());
        p.cpu_vendor = CpuVendor::Amd;
        assert!(!p.recommends_openvino());
        p.igpu = Some(GpuVendor::Apple);
        assert!(!p.recommends_openvino());
        p.dgpu = Some(GpuVendor::Intel);
        assert!(p.recommends_openvino() && p.intel_graphics(), "an Intel card");
    }

    #[test]
    fn recommends_rocm_only_for_amd_dgpu() {
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: Some(GpuVendor::Amd),
            dgpu_name: None,
            dgpu_device: None,
            igpu: None,
            suits_directml: false,
            os: "linux", arch: "x86_64",
        };
        assert!(p.recommends_rocm());
        // APU iGPU isn't dGPU — should be excluded.
        p.dgpu = None;
        p.igpu = Some(GpuVendor::Amd);
        assert!(!p.recommends_rocm());
    }

    #[test]
    fn a_graphics_card_needs_no_download_and_an_intel_igpu_alone_gets_openvino_where_offered() {
        use crate::runtime_install::RuntimeId;
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: None,
            dgpu_name: None,
            dgpu_device: None,
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "windows", arch: "x86_64",
        };
        let openvino_if_offered = RuntimeId::OFFERED.contains(&RuntimeId::OpenVino).then_some(RuntimeId::OpenVino);
        assert_eq!(p.recommended_runtime(), openvino_if_offered);
        p.dgpu = Some(GpuVendor::Amd);
        p.suits_directml = true;
        assert_eq!(p.recommended_runtime(), None, "DirectML is built in");
        p.dgpu = Some(GpuVendor::Nvidia);
        let cuda_if_offered = RuntimeId::OFFERED.contains(&RuntimeId::Cuda).then_some(RuntimeId::Cuda);
        assert_eq!(p.recommended_runtime(), cuda_if_offered, "an NVIDIA card gets CUDA where offered, DirectML or not");
        p.dgpu = None;
        p.igpu = None;
        p.suits_directml = false;
        let addon = RuntimeId::OpenVino.is_addon().then_some(RuntimeId::OpenVino);
        assert_eq!(p.recommended_runtime(), addon, "an Intel processor alone, where OpenVINO is an add-on");
    }

    /// An NVIDIA card gets CUDA where it is offered, even beside an Intel
    /// iGPU; where it is not, the iGPU still gets OpenVINO.
    #[test]
    fn an_nvidia_card_gets_cuda_over_openvino_where_offered() {
        use crate::runtime_install::RuntimeId;
        let p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: Some(GpuVendor::Nvidia),
            dgpu_name: None,
            dgpu_device: None,
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "linux", arch: "x86_64",
        };
        let offered = |rt| RuntimeId::OFFERED.contains(&rt).then_some(rt);
        assert_eq!(p.recommended_runtime(), offered(RuntimeId::Cuda).or(offered(RuntimeId::OpenVino)));
        let no_igpu = HardwareProfile { igpu: None, ..p.clone() };
        assert_eq!(no_igpu.recommended_runtime(), offered(RuntimeId::Cuda));
        let amd = HardwareProfile { dgpu: Some(GpuVendor::Amd), ..p };
        assert_eq!(amd.recommended_runtime(), offered(RuntimeId::MiGraphX).or(offered(RuntimeId::OpenVino)));
    }

    #[test]
    fn detect_runs_without_panic() {
        // Sanity: profile() shouldn't panic on the host machine,
        // regardless of what hardware it's running on.
        let p = profile();
        assert!(!p.os.is_empty());
        assert!(!p.arch.is_empty());
        eprintln!("HardwareProfile detected on this machine: {p:#?}");
        eprintln!("  recommended_runtime: {:?}", p.recommended_runtime());
        eprintln!("  recommends_rocm:     {}", p.recommends_rocm());
    }
}
