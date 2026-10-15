//! Display adapters through DXGI. Listing them creates no D3D device, so
//! it works on a machine whose GPU driver is missing or broken.

use windows::core::Interface;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIFactory6, DXGI_ADAPTER_FLAG_SOFTWARE,
    DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
};

use super::AdapterFacts;

/// Every adapter DXGI lists, and the index of the one Windows ranks first
/// for high performance (Windows 10 1803 and later).
pub(super) fn adapters() -> (Vec<AdapterFacts>, Option<usize>) {
    // SAFETY (every unsafe block here): DXGI factory and adapter calls
    // whose out-values the `windows` wrappers own and release.
    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(factory) => factory,
        Err(e) => {
            tracing::warn!(%e, "DXGI factory unavailable; no GPU detected");
            return (Vec::new(), None);
        }
    };
    let mut adapters = Vec::new();
    let mut luids = Vec::new();
    for index in 0u32.. {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => {
                tracing::warn!(index, %e, "DXGI adapter listing stopped early");
                break;
            }
        };
        let desc = match unsafe { adapter.GetDesc1() } {
            Ok(desc) => desc,
            Err(e) => {
                tracing::warn!(index, %e, "DXGI adapter description unavailable");
                continue;
            }
        };
        let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let facts = AdapterFacts {
            vendor_id: desc.VendorId,
            dedicated_mb: (desc.DedicatedVideoMemory / (1024 * 1024)) as u64,
            is_software: (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
        };
        tracing::info!(
            index,
            name = %String::from_utf16_lossy(&desc.Description[..name_len]),
            vendor_id = format_args!("{:#06x}", facts.vendor_id),
            device_id = format_args!("{:#06x}", desc.DeviceId),
            dedicated_mb = facts.dedicated_mb,
            software = facts.is_software,
            "DXGI adapter",
        );
        adapters.push(facts);
        luids.push(desc.AdapterLuid);
    }
    let os_choice = factory
        .cast::<IDXGIFactory6>()
        .ok()
        .and_then(|factory6| unsafe { factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) }.ok())
        .and_then(|adapter| unsafe { adapter.GetDesc1() }.ok())
        .and_then(|desc| luids.iter().position(|&luid| luid == desc.AdapterLuid));
    (adapters, os_choice)
}
