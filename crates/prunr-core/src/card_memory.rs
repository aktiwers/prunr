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

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

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

/// Which NVML card to read: the only one, or the one with CUDA's PCI id.
#[derive(Debug, PartialEq, Eq)]
enum Pick<'a> {
    Index(c_uint),
    PciBusId(&'a [c_char]),
}

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

    struct Nvml {
        _lib: libloading::Library,
        count: unsafe extern "C" fn(*mut c_uint) -> c_int,
        by_index: unsafe extern "C" fn(c_uint, *mut Device) -> c_int,
        by_pci_bus_id: unsafe extern "C" fn(*const c_char, *mut Device) -> c_int,
        memory: unsafe extern "C" fn(Device, *mut Memory) -> c_int,
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
                    Some(Nvml { _lib: lib, count, by_index, by_pci_bus_id, memory })
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

    pub(super) fn card_memory() -> Option<CardMemory> {
        let nvml = nvml()?;
        let mut cards: c_uint = 0;
        // SAFETY: NVML initialised; each call writes only through the
        // pointer it is given.
        unsafe {
            if (nvml.count)(&mut cards) != SUCCESS {
                return None;
            }
            let mut device: Device = std::ptr::null_mut();
            let found = match pick(cards, cuda_device0_pci_bus_id)? {
                Pick::Index(i) => (nvml.by_index)(i, &mut device),
                Pick::PciBusId(id) => (nvml.by_pci_bus_id)(id.as_ptr(), &mut device),
            };
            let mut memory = Memory::default();
            if found != SUCCESS || (nvml.memory)(device, &mut memory) != SUCCESS {
                return None;
            }
            const MB: u64 = 1024 * 1024;
            Some(CardMemory { free_mb: memory.free / MB, total_mb: memory.total / MB })
        }
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
