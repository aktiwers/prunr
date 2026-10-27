//! Free memory on the NVIDIA card CUDA runs on, read through NVML, the
//! monitoring library every NVIDIA driver installs. NVML answers without
//! starting CUDA, so a check before a model loads costs nothing on the
//! card.
//!
//! NVML numbers the cards in PCI order and CUDA puts the fastest first,
//! so with several cards NVML's first need not be CUDA's device 0: the
//! card is then found by the PCI bus id CUDA's driver gives its device 0
//! (the driver library, installed with every driver, not a CUDA runtime,
//! whose version depends on the pack in use).

#[cfg(any(target_os = "linux", windows, test))]
use std::ffi::{c_char, c_uint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardMemory {
    pub free_mb: u64,
    pub total_mb: u64,
}

/// The memory of the card CUDA's device 0 is; `None` with no NVIDIA
/// driver, or when that card cannot be told apart from the others.
pub fn cuda_card_memory() -> Option<CardMemory> {
    #[cfg(any(target_os = "linux", windows))]
    {
        nvml::card_memory()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// The name of the card CUDA's device 0 is ("NVIDIA GeForce RTX 3060");
/// `None` with no NVIDIA driver.
pub fn cuda_card_name() -> Option<String> {
    #[cfg(any(target_os = "linux", windows))]
    {
        nvml::card_name()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// What process `pid` holds on CUDA's card, MB; `None` with no NVIDIA
/// driver, or when the driver does not say (Windows' display model).
pub fn process_card_mb(pid: u32) -> Option<u64> {
    #[cfg(any(target_os = "linux", windows))]
    {
        nvml::process_card_mb(pid)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = pid;
        None
    }
}

/// Which NVML card to read: the only one, or the one with CUDA's PCI id.
#[cfg(any(target_os = "linux", windows, test))]
#[derive(Debug, PartialEq, Eq)]
enum Pick<'a> {
    Index(c_uint),
    PciBusId(&'a [c_char]),
}

#[cfg(any(target_os = "linux", windows, test))]
fn pick(cards: c_uint, cuda_device0: impl FnOnce() -> Option<&'static [c_char]>) -> Option<Pick<'static>> {
    match cards {
        0 => None,
        1 => Some(Pick::Index(0)),
        _ => cuda_device0().map(Pick::PciBusId),
    }
}

#[cfg(any(target_os = "linux", windows))]
mod nvml {
    use super::*;
    use std::ffi::{c_int, c_void};
    use std::sync::OnceLock;

    #[cfg(target_os = "linux")]
    const NVML: &str = "libnvidia-ml.so.1";
    #[cfg(windows)]
    const NVML: &str = "nvml.dll";
    #[cfg(target_os = "linux")]
    const CUDA_DRIVER: &str = "libcuda.so.1";
    #[cfg(windows)]
    const CUDA_DRIVER: &str = "nvcuda.dll";

    const SUCCESS: c_int = 0;

    type Device = *mut c_void;

    #[repr(C)]
    #[derive(Default)]
    struct Memory {
        total: u64,
        free: u64,
        _used: u64,
    }

    /// NVML's `nvmlProcessInfo_t`.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct ProcessInfo {
        pid: c_uint,
        used: u64,
        _gpu_instance: c_uint,
        _compute_instance: c_uint,
    }

    /// NVML's answer for a figure the driver does not keep.
    const NOT_AVAILABLE: u64 = u64::MAX;

    struct Nvml {
        _lib: libloading::Library,
        count: unsafe extern "C" fn(*mut c_uint) -> c_int,
        by_index: unsafe extern "C" fn(c_uint, *mut Device) -> c_int,
        by_pci_bus_id: unsafe extern "C" fn(*const c_char, *mut Device) -> c_int,
        memory: unsafe extern "C" fn(Device, *mut Memory) -> c_int,
        processes: unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> c_int,
        /// `None` on a driver without it: the rest still works.
        name: Option<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int>,
    }

    // SAFETY: the handles are NVML's, which is thread-safe once initialised.
    unsafe impl Send for Nvml {}
    unsafe impl Sync for Nvml {}

    fn nvml() -> Option<&'static Nvml> {
        static NVML_LIB: OnceLock<Option<Nvml>> = OnceLock::new();
        NVML_LIB
            .get_or_init(|| {
                // SAFETY: NVML's documented entry points with their C
                // signatures; the library stays loaded for the process.
                unsafe {
                    let lib = libloading::Library::new(NVML).ok()?;
                    let init: unsafe extern "C" fn() -> c_int = *lib.get(b"nvmlInit_v2").ok()?;
                    if init() != SUCCESS {
                        return None;
                    }
                    let count = *lib.get(b"nvmlDeviceGetCount_v2").ok()?;
                    let by_index = *lib.get(b"nvmlDeviceGetHandleByIndex_v2").ok()?;
                    let by_pci_bus_id = *lib.get(b"nvmlDeviceGetHandleByPciBusId_v2").ok()?;
                    let memory = *lib.get(b"nvmlDeviceGetMemoryInfo").ok()?;
                    let processes = *lib.get(b"nvmlDeviceGetComputeRunningProcesses_v3").ok()?;
                    let name = lib.get(b"nvmlDeviceGetName").ok().map(|f| *f);
                    Some(Nvml { _lib: lib, count, by_index, by_pci_bus_id, memory, processes, name })
                }
            })
            .as_ref()
    }

    /// CUDA's device 0 as a PCI bus id ("0000:01:00.0"), read once.
    /// `cuInit` starts the driver, not a context: nothing on the card.
    fn cuda_device0_pci_bus_id() -> Option<&'static [c_char]> {
        static ID: OnceLock<Option<[c_char; 32]>> = OnceLock::new();
        ID.get_or_init(|| {
            // SAFETY: the CUDA driver API's C signatures: cuInit(flags),
            // cuDeviceGet(CUdevice*, ordinal), cuDeviceGetPCIBusId(char*,
            // len, CUdevice), the last writing a NUL-terminated id.
            unsafe {
                let lib = libloading::Library::new(CUDA_DRIVER).ok()?;
                let init: unsafe extern "C" fn(c_uint) -> c_int = *lib.get(b"cuInit").ok()?;
                let device: unsafe extern "C" fn(*mut c_int, c_int) -> c_int = *lib.get(b"cuDeviceGet").ok()?;
                let bus_id: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int = *lib.get(b"cuDeviceGetPCIBusId").ok()?;
                let mut dev: c_int = 0;
                let mut id = [0 as c_char; 32];
                let ok = init(0) == SUCCESS && device(&mut dev, 0) == SUCCESS && bus_id(id.as_mut_ptr(), id.len() as c_int, dev) == SUCCESS;
                std::mem::forget(lib);
                ok.then_some(id)
            }
        })
        .as_ref()
        .map(|id| id.as_slice())
    }

    const MB: u64 = 1024 * 1024;

    /// CUDA's device 0 as NVML's handle.
    fn device(nvml: &Nvml) -> Option<Device> {
        let mut cards: c_uint = 0;
        let mut device: Device = std::ptr::null_mut();
        // SAFETY: NVML initialised; each call writes only through the
        // pointer it is given.
        unsafe {
            if (nvml.count)(&mut cards) != SUCCESS {
                return None;
            }
            let found = match pick(cards, cuda_device0_pci_bus_id)? {
                Pick::Index(i) => (nvml.by_index)(i, &mut device),
                Pick::PciBusId(id) => (nvml.by_pci_bus_id)(id.as_ptr(), &mut device),
            };
            (found == SUCCESS).then_some(device)
        }
    }

    pub(super) fn card_memory() -> Option<CardMemory> {
        let nvml = nvml()?;
        let device = device(nvml)?;
        let mut memory = Memory::default();
        // SAFETY: a handle NVML gave; the call writes only `memory`.
        if unsafe { (nvml.memory)(device, &mut memory) } != SUCCESS {
            return None;
        }
        Some(CardMemory { free_mb: memory.free / MB, total_mb: memory.total / MB })
    }

    pub(super) fn card_name() -> Option<String> {
        let nvml = nvml()?;
        let device = device(nvml)?;
        let mut name = [0 as c_char; 96];
        // SAFETY: a handle NVML gave; it writes a NUL-terminated name of
        // at most `name.len()` bytes.
        if unsafe { (nvml.name?)(device, name.as_mut_ptr(), name.len() as c_uint) } != SUCCESS {
            return None;
        }
        let bytes: Vec<u8> = name.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        Some(String::from_utf8_lossy(&bytes).into_owned()).filter(|n| !n.is_empty())
    }

    pub(super) fn process_card_mb(pid: u32) -> Option<u64> {
        let nvml = nvml()?;
        let device = device(nvml)?;
        let mut infos = [ProcessInfo::default(); 64];
        let mut count = infos.len() as c_uint;
        // SAFETY: a handle NVML gave; it writes at most `count` entries.
        if unsafe { (nvml.processes)(device, &mut count, infos.as_mut_ptr()) } != SUCCESS {
            return None;
        }
        infos[..count as usize].iter().find(|p| p.pid == pid).filter(|p| p.used != NOT_AVAILABLE).map(|p| p.used / MB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One card is read by its index; with several, only CUDA's own, by
    /// its PCI id, and none when CUDA cannot say which it is.
    #[test]
    fn the_card_read_is_cudas_device_0() {
        static ID: [c_char; 2] = [b'1' as c_char, 0];
        assert_eq!(pick(0, || Some(&ID[..])), None);
        assert_eq!(pick(1, || panic!("one card needs no CUDA")), Some(Pick::Index(0)));
        assert_eq!(pick(2, || Some(&ID[..])), Some(Pick::PciBusId(&ID[..])));
        assert_eq!(pick(2, || None), None);
    }
}
