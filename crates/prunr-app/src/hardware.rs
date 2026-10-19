//! Hardware profile detection — CPU vendor/brand + dGPU/iGPU vendors.
//! Cached via OnceLock; first call does the work, subsequent ones are
//! zero-cost. PCI vendor IDs are read from `/sys/class/drm/.../device/vendor`
//! on Linux and from DXGI's adapter list on Windows, because neither
//! needs a graphics context — `wgpu`/`ash` would fail in headless
//! containers.

#[cfg_attr(not(windows), allow(dead_code))] // only Windows lists adapters through DXGI; the tests run the rules everywhere
mod adapters;
#[cfg(windows)]
mod dxgi;

use std::sync::OnceLock;

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
    pub igpu: Option<GpuVendor>,
    /// A graphics card DirectML runs well on (Windows only).
    pub suits_directml: bool,
    pub os: &'static str,
    pub arch: &'static str,
}

impl HardwareProfile {
    pub fn recommends_openvino(&self) -> bool {
        self.igpu == Some(GpuVendor::Intel)
    }

    /// The runtime download the first-launch prompt and Settings
    /// recommend, among those offered here. A graphics card that suits
    /// DirectML needs none, since DirectML is built in, even with an
    /// Intel iGPU beside it. A graphics card wins over an Intel iGPU beside
    /// it (the usual Linux laptop): CUDA or MIGraphX runs the card,
    /// OpenVINO only the iGPU.
    pub fn recommended_runtime(&self) -> Option<crate::runtime_install::RuntimeId> {
        use crate::runtime_install::RuntimeId;
        if self.suits_directml {
            return None;
        }
        [
            (self.dgpu == Some(GpuVendor::Nvidia)).then_some(RuntimeId::Cuda),
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
/// driver update changes: each DXGI adapter on Windows, the NVIDIA
/// kernel driver on Linux. Empty where none is read.
pub fn driver_identity() -> String {
    #[cfg(target_os = "windows")]
    {
        dxgi::driver_identity()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/sys/module/nvidia/version")
            .map(|v| format!("nvidia {}", v.trim()))
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        String::new()
    }
}

pub fn profile() -> &'static HardwareProfile {
    static CACHE: OnceLock<HardwareProfile> = OnceLock::new();
    CACHE.get_or_init(detect_now)
}

/// Total system RAM (bytes). Cached — total memory doesn't change at
/// runtime so we read it once.
pub fn total_ram_bytes() -> u64 {
    static CACHE: OnceLock<u64> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.total_memory()
    })
}

/// Free bytes on the disk holding `path` (or its nearest existing
/// parent): the mount with the longest prefix of it. `None` when no
/// mount matches.
pub fn free_space_at(path: &std::path::Path) -> Option<u64> {
    let existing = path.ancestors().find(|p| p.exists())?;
    // std's canonicalize gives `\\?\C:\…` on Windows, which no mount
    // point (`C:\`) prefixes; dunce keeps the plain form.
    let path = dunce::canonicalize(existing).unwrap_or_else(|_| existing.to_path_buf());
    sysinfo::Disks::new_with_refreshed_list()
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(sysinfo::Disk::available_space)
}

/// Currently-available RAM (bytes). Re-reads on every call — fresh value
/// for UIs that show live headroom. ~1 ms on Linux. Don't call inside a
/// per-frame loop; prefer `available_ram_bytes_throttled` for any caller
/// that fires more than once per second.
pub fn available_ram_bytes_now() -> u64 {
    use std::sync::{Mutex, PoisonError};
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let sys = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    // Poison recovery: the System is only a reused buffer, refreshed below.
    let mut sys = sys.lock().unwrap_or_else(PoisonError::into_inner);
    sys.refresh_memory();
    sys.available_memory()
}

/// `available_ram_bytes_throttled` in MiB, the unit the admission gates
/// compare against `working_set_mb`.
pub fn available_ram_mb_throttled() -> u32 {
    (available_ram_bytes_throttled() / (1024 * 1024)) as u32
}

/// Cached available-RAM read with a 1-second TTL. Cheap to call from
/// per-frame UI code (~10 ns on cache hit). The 1 s freshness is fine
/// for a settings panel — RAM displays are read at human speed, not
/// frame speed. First call on cold start does the full ~1 ms read.
pub fn available_ram_bytes_throttled() -> u64 {
    use std::sync::Mutex;
    use std::time::Instant;
    static CACHE: OnceLock<Mutex<Option<(Instant, u64)>>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    if let Some((at, bytes)) = *guard {
        if now.duration_since(at).as_secs_f32() < 1.0 {
            return bytes;
        }
    }
    let bytes = available_ram_bytes_now();
    *guard = Some((now, bytes));
    bytes
}

fn detect_now() -> HardwareProfile {
    let (cpu_vendor, cpu_brand) = detect_cpu();
    let GpuSlots { dgpu, igpu, suits_directml } = detect_gpus(cpu_vendor);
    HardwareProfile {
        cpu_vendor,
        cpu_brand,
        dgpu,
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct GpuSlots {
    dgpu: Option<GpuVendor>,
    igpu: Option<GpuVendor>,
    suits_directml: bool,
}

fn detect_gpus(cpu_vendor: CpuVendor) -> GpuSlots {
    #[cfg(target_os = "linux")]
    {
        let (dgpu, igpu) = detect_gpus_linux(cpu_vendor);
        GpuSlots { dgpu, igpu, suits_directml: false }
    }
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon: GPU is part of the SoC. Intel Macs return None
        // (we don't ship OpenVINO on macOS, so detail isn't needed).
        let _ = cpu_vendor;
        let igpu = (std::env::consts::ARCH == "aarch64").then_some(GpuVendor::Apple);
        GpuSlots { dgpu: None, igpu, suits_directml: false }
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
fn detect_gpus_linux(cpu_vendor: CpuVendor) -> (Option<GpuVendor>, Option<GpuVendor>) {
    let Ok(read) = std::fs::read_dir("/sys/class/drm") else {
        return (None, None);
    };
    let mut dgpu = None;
    let mut igpu = None;
    for entry in read.flatten() {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if !is_card_dir(&name_s) { continue; }
        let vendor_path = entry.path().join("device").join("vendor");
        let Ok(vendor_raw) = std::fs::read_to_string(&vendor_path) else { continue };
        let gpu_vendor = parse_pci_vendor(&vendor_raw);
        // Heuristic: GPU vendor matching CPU vendor is the iGPU; distinct
        // vendor is dGPU. Known limitation — AMD CPU + AMD dGPU (Ryzen +
        // Radeon RX) misclassifies the dGPU as iGPU and `recommends_rocm`
        // returns false. Reading `/sys/class/drm/.../device/class` (PCI
        // class byte 0x030000 = VGA) would disambiguate; deferred until
        // we have a real-hardware tester for that combo.
        let is_integrated = matches_cpu(gpu_vendor, cpu_vendor);
        if is_integrated {
            igpu.get_or_insert(gpu_vendor);
        } else {
            dgpu.get_or_insert(gpu_vendor);
        }
    }
    (dgpu, igpu)
}

#[cfg(target_os = "linux")]
fn is_card_dir(name: &str) -> bool {
    // Match `card0`, `card1`, ... but not `card0-DP-1` (those are
    // connector subnodes, not GPUs).
    name.starts_with("card")
        && name.len() >= 5
        && name[4..].chars().all(|c| c.is_ascii_digit())
}

/// Parse a `/sys/class/drm/.../device/vendor` value (e.g. `0x10de\n`).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // sysfs is Linux's; the tests run it everywhere
fn parse_pci_vendor(raw: &str) -> GpuVendor {
    u32::from_str_radix(raw.trim().trim_start_matches("0x"), 16).map_or(GpuVendor::Other, gpu_vendor)
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
        assert_eq!(parse_pci_vendor("0x10de\n"), GpuVendor::Nvidia);
        assert_eq!(parse_pci_vendor("0x1002\n"), GpuVendor::Amd);
        assert_eq!(parse_pci_vendor("0x8086\n"), GpuVendor::Intel);
        assert_eq!(parse_pci_vendor("0x1234\n"), GpuVendor::Other);
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

    #[test]
    fn recommends_openvino_only_for_intel_igpu() {
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: None,
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "linux", arch: "x86_64",
        };
        assert!(p.recommends_openvino());
        p.igpu = None;
        assert!(!p.recommends_openvino());
        p.igpu = Some(GpuVendor::Apple);
        assert!(!p.recommends_openvino());
    }

    #[test]
    fn recommends_rocm_only_for_amd_dgpu() {
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: Some(GpuVendor::Amd),
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
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "windows", arch: "x86_64",
        };
        let openvino_if_offered = RuntimeId::OFFERED.contains(&RuntimeId::OpenVino).then_some(RuntimeId::OpenVino);
        assert_eq!(p.recommended_runtime(), openvino_if_offered);
        p.dgpu = Some(GpuVendor::Nvidia);
        p.suits_directml = true;
        assert_eq!(p.recommended_runtime(), None, "DirectML is built in");
        p.dgpu = None;
        p.igpu = None;
        p.suits_directml = false;
        assert_eq!(p.recommended_runtime(), None);
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
