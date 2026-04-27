//! Hardware profile detection — CPU vendor/brand + dGPU/iGPU vendors.
//! Cached via OnceLock; first call does the work, subsequent ones are
//! zero-cost. PCI vendor IDs are read from `/sys/class/drm/.../device/vendor`
//! on Linux because that path doesn't require initializing a graphics
//! context — `wgpu`/`ash` would fail in headless containers.

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
    pub os: &'static str,
    pub arch: &'static str,
}

impl HardwareProfile {
    pub fn recommends_openvino(&self) -> bool {
        self.igpu == Some(GpuVendor::Intel)
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
/// per-frame loop; use a cache + manual invalidate (modal-open / event)
/// if needed.
pub fn available_ram_bytes_now() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Coarse "can the user actually run this model?" verdict per model
/// working set. Compared against currently-available RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamVerdict {
    /// `available_ram > 1.5 × working_set`. Comfortable headroom.
    Comfortable,
    /// `working_set ≤ available_ram ≤ 1.5 × working_set`. Likely OK
    /// but other apps competing for RAM may push it over the edge.
    Tight,
    /// `available_ram < working_set`. Refusing to load is the right
    /// move; the model would swap-thrash or OOM.
    Insufficient,
}

pub fn ram_verdict(working_set_bytes: u64, available_bytes: u64) -> RamVerdict {
    if available_bytes < working_set_bytes {
        RamVerdict::Insufficient
    } else if available_bytes < working_set_bytes * 3 / 2 {
        RamVerdict::Tight
    } else {
        RamVerdict::Comfortable
    }
}

/// Auto-detect default for SD inpaint's "Fast mode" — ON when no real
/// GPU is detected (CPU or Intel-iGPU-only), OFF otherwise. Real GPUs
/// run standard SD fast enough that the LCM/TAESD quality trade-off
/// isn't worth it. Pure function of the profile so callers can mock.
pub fn sd_fast_mode_auto_default(profile: &HardwareProfile) -> bool {
    let real_dgpu = matches!(profile.dgpu, Some(GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Apple));
    let apple_soc = profile.igpu == Some(GpuVendor::Apple);
    !(real_dgpu || apple_soc)
}

/// Effective SD fast-mode flag: `user_override` wins when set, otherwise
/// `sd_fast_mode_auto_default(profile)`. Single source of truth — the
/// dispatch path and UI both call this so they can't disagree.
pub fn sd_fast_mode_active(user_override: Option<bool>, profile: &HardwareProfile) -> bool {
    user_override.unwrap_or_else(|| sd_fast_mode_auto_default(profile))
}

fn detect_now() -> HardwareProfile {
    let (cpu_vendor, cpu_brand) = detect_cpu();
    let (dgpu, igpu) = detect_gpus(cpu_vendor);
    HardwareProfile {
        cpu_vendor,
        cpu_brand,
        dgpu,
        igpu,
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

fn detect_gpus(cpu_vendor: CpuVendor) -> (Option<GpuVendor>, Option<GpuVendor>) {
    #[cfg(target_os = "linux")]
    { detect_gpus_linux(cpu_vendor) }
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon: GPU is part of the SoC. Intel Macs return None
        // (we don't ship OpenVINO on macOS, so detail isn't needed).
        let _ = cpu_vendor;
        if std::env::consts::ARCH == "aarch64" {
            (None, Some(GpuVendor::Apple))
        } else {
            (None, None)
        }
    }
    #[cfg(target_os = "windows")]
    {
        // DXGI enumeration is the right approach but adds the `windows`
        // crate's GPU surface. Stubbed until that lands; first-launch
        // prompt silently skips on Windows in the interim.
        let _ = cpu_vendor;
        (None, None)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = cpu_vendor;
        (None, None)
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
fn parse_pci_vendor(raw: &str) -> GpuVendor {
    let trimmed = raw.trim().trim_start_matches("0x").to_ascii_lowercase();
    match trimmed.as_str() {
        "10de" => GpuVendor::Nvidia,
        "1002" | "1022" => GpuVendor::Amd,
        "8086" => GpuVendor::Intel,
        "106b" => GpuVendor::Apple,
        _ => GpuVendor::Other,
    }
}

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
            os: "linux", arch: "x86_64",
        };
        assert!(p.recommends_rocm());
        // APU iGPU isn't dGPU — should be excluded.
        p.dgpu = None;
        p.igpu = Some(GpuVendor::Amd);
        assert!(!p.recommends_rocm());
    }

    #[test]
    fn ram_verdict_thresholds() {
        const WS: u64 = 6 * 1024 * 1024 * 1024; // 6 GB working set
        // Below working set → insufficient
        assert_eq!(ram_verdict(WS, WS - 1), RamVerdict::Insufficient);
        // Exactly working set → tight (we want >1.5× for comfortable)
        assert_eq!(ram_verdict(WS, WS), RamVerdict::Tight);
        // Just below 1.5× → tight
        assert_eq!(ram_verdict(WS, WS * 3 / 2 - 1), RamVerdict::Tight);
        // At 1.5× exactly → comfortable
        assert_eq!(ram_verdict(WS, WS * 3 / 2), RamVerdict::Comfortable);
        // Way above → comfortable
        assert_eq!(ram_verdict(WS, WS * 4), RamVerdict::Comfortable);
    }

    #[test]
    fn sd_fast_mode_default_on_for_cpu_or_intel_igpu() {
        let cpu_only = HardwareProfile {
            cpu_vendor: CpuVendor::Intel, cpu_brand: "i7".into(),
            dgpu: None, igpu: None, os: "linux", arch: "x86_64",
        };
        assert!(sd_fast_mode_auto_default(&cpu_only));

        let intel_igpu = HardwareProfile {
            cpu_vendor: CpuVendor::Intel, cpu_brand: "i7".into(),
            dgpu: None, igpu: Some(GpuVendor::Intel),
            os: "linux", arch: "x86_64",
        };
        assert!(sd_fast_mode_auto_default(&intel_igpu));
    }

    #[test]
    fn sd_fast_mode_default_off_for_real_gpus() {
        let nvidia = HardwareProfile {
            cpu_vendor: CpuVendor::Intel, cpu_brand: "i7".into(),
            dgpu: Some(GpuVendor::Nvidia), igpu: None,
            os: "linux", arch: "x86_64",
        };
        assert!(!sd_fast_mode_auto_default(&nvidia));

        let amd_dgpu = HardwareProfile {
            cpu_vendor: CpuVendor::Amd, cpu_brand: "Ryzen".into(),
            dgpu: Some(GpuVendor::Amd), igpu: None,
            os: "linux", arch: "x86_64",
        };
        assert!(!sd_fast_mode_auto_default(&amd_dgpu));

        let apple = HardwareProfile {
            cpu_vendor: CpuVendor::Apple, cpu_brand: "M2".into(),
            dgpu: None, igpu: Some(GpuVendor::Apple),
            os: "macos", arch: "aarch64",
        };
        assert!(!sd_fast_mode_auto_default(&apple));
    }

    #[test]
    fn user_override_beats_auto_detect() {
        let cpu_only = HardwareProfile {
            cpu_vendor: CpuVendor::Intel, cpu_brand: "i7".into(),
            dgpu: None, igpu: None, os: "linux", arch: "x86_64",
        };
        // Auto would say ON; explicit OFF wins.
        assert!(!sd_fast_mode_active(Some(false), &cpu_only));
        // Auto would say ON; explicit ON also ON.
        assert!(sd_fast_mode_active(Some(true), &cpu_only));
        // Auto-default fallback.
        assert!(sd_fast_mode_active(None, &cpu_only));
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
        eprintln!("  recommends_rocm:     {}", p.recommends_rocm());
    }
}
