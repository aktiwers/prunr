//! Hardware profile detection — CPU vendor/brand + dGPU/iGPU vendors.
//! Cached via OnceLock; first call does the work, subsequent ones are
//! zero-cost. PCI vendor IDs are read from `/sys/class/drm/.../device/vendor`
//! on Linux and from DXGI's adapter list on Windows, because neither
//! needs a graphics context — `wgpu`/`ash` would fail in headless
//! containers.

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

    pub fn recommends_directml(&self) -> bool {
        self.suits_directml
    }

    /// The one runtime the first-launch prompt and Settings recommend.
    /// A graphics card that suits DirectML outranks an Intel iGPU beside it.
    pub fn recommended_runtime(&self) -> Option<crate::runtime_install::RuntimeId> {
        use crate::runtime_install::RuntimeId;
        if self.recommends_directml() {
            Some(RuntimeId::DirectMl)
        } else if self.recommends_openvino() {
            Some(RuntimeId::OpenVino)
        } else {
            None
        }
    }

    /// A hardware GPU was found: a software adapter (a VM's or remote
    /// desktop's) counts as none.
    pub fn has_hardware_gpu(&self) -> bool {
        self.dgpu.is_some() || self.igpu.is_some()
    }

    /// The accelerator to name before any session reports one. A DirectML
    /// runtime says it has DirectML on any Windows machine, a VM's
    /// included; without a hardware GPU its sessions run on the CPU.
    pub fn startup_provider(&self, detected: String) -> String {
        if !self.has_hardware_gpu() && detected.eq_ignore_ascii_case(prunr_core::engine::DIRECTML_NAME) {
            "CPU".to_string()
        } else {
            detected
        }
    }

    /// AMD discrete GPU only. APU iGPUs excluded — ROCm/MIGraphX
    /// coverage on AMD APUs is unreliable.
    pub fn recommends_rocm(&self) -> bool {
        self.dgpu == Some(GpuVendor::Amd)
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

/// Currently-available RAM (bytes). Re-reads on every call — fresh value
/// for UIs that show live headroom. ~1 ms on Linux. Don't call inside a
/// per-frame loop; prefer `available_ram_bytes_throttled` for any caller
/// that fires more than once per second.
pub fn available_ram_bytes_now() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Cached available-RAM read with a 1-second TTL. Cheap to call from
/// per-frame UI code (~10 ns on cache hit). The 1 s freshness is fine
/// for a settings panel — RAM displays are read at human speed, not
/// frame speed. First call on cold start does the full ~1 ms read.
/// `available_ram_bytes_throttled` in MiB, the unit the admission gates
/// compare against `working_set_mb`.
pub fn available_ram_mb_throttled() -> u32 {
    (available_ram_bytes_throttled() / (1024 * 1024)) as u32
}

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
        sort_adapters(&adapters, os_choice)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = cpu_vendor;
        GpuSlots::default()
    }
}

/// One display adapter as DXGI reports it, read without creating a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterFacts {
    pub vendor_id: u32,
    pub dedicated_mb: u64,
    pub is_software: bool,
}

const MICROSOFT_VENDOR_ID: u32 = 0x1414;

/// Dedicated memory that only a graphics card has: Intel and AMD iGPUs
/// reserve 128 MB to 1 GB, Intel Arc and every recent card have 4 GB or more.
const CARD_MIN_DEDICATED_MB: u64 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuKind {
    Discrete,
    Integrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterClass {
    pub kind: GpuKind,
    pub vendor: GpuVendor,
    pub suits_directml: bool,
}

/// What a display adapter is. `None` for a software adapter (WARP, the
/// Basic Render Driver a VM or remote desktop gets). `is_high_perf`: the
/// strong card beside another hardware adapter (see `strong_card`).
/// An Intel iGPU without a card beside it is left to OpenVINO.
pub(crate) fn classify_adapter(vendor_id: u32, dedicated_mb: u64, is_software: bool, is_high_perf: bool) -> Option<AdapterClass> {
    if is_software || vendor_id == MICROSOFT_VENDOR_ID {
        return None;
    }
    let vendor = gpu_vendor(vendor_id);
    let has_card_memory = dedicated_mb >= CARD_MIN_DEDICATED_MB;
    let kind = if is_high_perf || has_card_memory || vendor == GpuVendor::Nvidia {
        GpuKind::Discrete
    } else {
        GpuKind::Integrated
    };
    let suits_directml = has_card_memory || matches!(vendor, GpuVendor::Nvidia | GpuVendor::Amd);
    Some(AdapterClass { kind, vendor, suits_directml })
}

/// The strong card among two or more hardware adapters: the one the OS
/// ranks first for high performance (`os_choice`), else the one with the
/// most dedicated memory. A lone adapter has nothing to outrank.
pub(crate) fn strong_card(adapters: &[AdapterFacts], os_choice: Option<usize>) -> Option<usize> {
    let hardware = |a: &AdapterFacts| !a.is_software && a.vendor_id != MICROSOFT_VENDOR_ID;
    if adapters.iter().filter(|a| hardware(a)).count() < 2 {
        return None;
    }
    os_choice.filter(|&i| adapters.get(i).is_some_and(hardware)).or_else(|| {
        adapters.iter().enumerate()
            .filter(|(_, a)| hardware(a))
            .max_by_key(|(_, a)| a.dedicated_mb)
            .map(|(i, _)| i)
    })
}

#[cfg_attr(not(windows), allow(dead_code))] // only Windows lists adapters through DXGI; the tests run it everywhere
fn sort_adapters(adapters: &[AdapterFacts], os_choice: Option<usize>) -> GpuSlots {
    let strong = strong_card(adapters, os_choice);
    let mut slots = GpuSlots::default();
    for (i, a) in adapters.iter().enumerate() {
        let Some(class) = classify_adapter(a.vendor_id, a.dedicated_mb, a.is_software, strong == Some(i)) else { continue };
        let slot = match class.kind {
            GpuKind::Discrete => &mut slots.dgpu,
            GpuKind::Integrated => &mut slots.igpu,
        };
        slot.get_or_insert(class.vendor);
        slots.suits_directml |= class.suits_directml;
    }
    slots
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
    fn classify_intel_via_vendor_id() {
        assert_eq!(classify_cpu_vendor("GenuineIntel", ""), CpuVendor::Intel);
    }

    #[test]
    fn classify_amd_via_vendor_id() {
        assert_eq!(classify_cpu_vendor("AuthenticAMD", ""), CpuVendor::Amd);
    }

    #[test]
    fn classify_apple_falls_back_to_brand() {
        assert_eq!(
            classify_cpu_vendor("", "Apple M3 Pro"),
            CpuVendor::Apple,
        );
    }

    #[test]
    fn classify_amd_via_ryzen_brand_when_vendor_id_missing() {
        assert_eq!(
            classify_cpu_vendor("", "AMD Ryzen 9 7950X"),
            CpuVendor::Amd,
        );
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

    fn adapter(vendor_id: u32, dedicated_mb: u64) -> AdapterFacts {
        AdapterFacts { vendor_id, dedicated_mb, is_software: false }
    }

    /// Microsoft Basic Render Driver.
    fn basic_render() -> AdapterFacts {
        AdapterFacts { vendor_id: 0x1414, dedicated_mb: 0, is_software: true }
    }

    /// One row per machine shape the Windows detection meets.
    #[test]
    fn dxgi_adapters_sort_into_slots_and_the_directml_verdict() {
        let none = GpuSlots::default();
        let rows: [(&str, Vec<AdapterFacts>, Option<usize>, GpuSlots); 9] = [
            ("VM or remote desktop", vec![basic_render()], Some(0), none),
            ("Intel iGPU alone", vec![adapter(0x8086, 128), basic_render()], Some(0),
                GpuSlots { igpu: Some(GpuVendor::Intel), ..none }),
            ("Intel Arc iGPU alone", vec![adapter(0x8086, 128)], None,
                GpuSlots { igpu: Some(GpuVendor::Intel), ..none }),
            ("Intel Arc card", vec![adapter(0x8086, 16_000)], Some(0),
                GpuSlots { dgpu: Some(GpuVendor::Intel), suits_directml: true, ..none }),
            ("hybrid laptop, OS ranks the card", vec![adapter(0x8086, 128), adapter(0x10de, 4096)], Some(1),
                GpuSlots { dgpu: Some(GpuVendor::Nvidia), igpu: Some(GpuVendor::Intel), suits_directml: true }),
            ("Ryzen APU alone", vec![adapter(0x1002, 512)], Some(0),
                GpuSlots { igpu: Some(GpuVendor::Amd), suits_directml: true, ..none }),
            ("Ryzen APU + small Radeon card, OS ranks the card", vec![adapter(0x1002, 512), adapter(0x1002, 1024)], Some(1),
                GpuSlots { dgpu: Some(GpuVendor::Amd), igpu: Some(GpuVendor::Amd), suits_directml: true }),
            ("Ryzen APU + small Radeon card, no OS ranking", vec![adapter(0x1002, 1024), adapter(0x1002, 512)], None,
                GpuSlots { dgpu: Some(GpuVendor::Amd), igpu: Some(GpuVendor::Amd), suits_directml: true }),
            ("NVIDIA card alone", vec![adapter(0x10de, 2048), basic_render()], Some(0),
                GpuSlots { dgpu: Some(GpuVendor::Nvidia), suits_directml: true, ..none }),
        ];
        for (shape, adapters, os_choice, want) in rows {
            assert_eq!(sort_adapters(&adapters, os_choice), want, "{shape}");
        }
    }

    #[test]
    fn the_strong_card_needs_another_hardware_adapter_to_outrank() {
        assert_eq!(strong_card(&[adapter(0x1002, 512), basic_render()], Some(0)), None);
        let pair = [adapter(0x8086, 128), adapter(0x10de, 8192)];
        assert_eq!(strong_card(&pair, Some(1)), Some(1));
        assert_eq!(strong_card(&pair, None), Some(1), "most dedicated memory without the OS's ranking");
        assert_eq!(strong_card(&pair, Some(7)), Some(1), "an index DXGI did not list");
    }

    #[test]
    fn the_runtime_recommendation_prefers_directml_over_an_intel_igpu() {
        use crate::runtime_install::RuntimeId;
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Intel,
            cpu_brand: "Intel Core i7".into(),
            dgpu: None,
            igpu: Some(GpuVendor::Intel),
            suits_directml: false,
            os: "windows", arch: "x86_64",
        };
        assert_eq!(p.recommended_runtime(), Some(RuntimeId::OpenVino));
        p.dgpu = Some(GpuVendor::Nvidia);
        p.suits_directml = true;
        assert_eq!(p.recommended_runtime(), Some(RuntimeId::DirectMl));
        p.dgpu = None;
        p.igpu = None;
        p.suits_directml = false;
        assert_eq!(p.recommended_runtime(), None);
        assert!(!p.has_hardware_gpu());
    }

    #[test]
    fn directml_is_named_at_startup_only_with_a_hardware_gpu() {
        let mut p = HardwareProfile {
            cpu_vendor: CpuVendor::Amd,
            cpu_brand: "AMD EPYC".into(),
            dgpu: None,
            igpu: None,
            suits_directml: false,
            os: "windows", arch: "x86_64",
        };
        assert_eq!(p.startup_provider("DirectML".into()), "CPU", "a VM's software adapter");
        assert_eq!(p.startup_provider("OpenVINO".into()), "OpenVINO");
        p.dgpu = Some(GpuVendor::Amd);
        assert_eq!(p.startup_provider("DirectML".into()), "DirectML");
    }

    #[test]
    fn detect_runs_without_panic() {
        // Sanity: profile() shouldn't panic on the host machine,
        // regardless of what hardware it's running on.
        let p = profile();
        assert!(!p.os.is_empty());
        assert!(!p.arch.is_empty());
        eprintln!("HardwareProfile detected on this machine: {p:#?}");
        eprintln!("  recommends_openvino: {}", p.recommends_openvino());
        eprintln!("  recommends_directml: {}", p.recommends_directml());
        eprintln!("  recommends_rocm:     {}", p.recommends_rocm());
    }
}
