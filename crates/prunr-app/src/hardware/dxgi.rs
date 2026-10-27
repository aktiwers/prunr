//! Display adapters through DXGI. Listing them creates no D3D device, so
//! it works on a machine whose GPU driver is missing or broken.

use windows::core::Interface;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIDevice, IDXGIFactory1, IDXGIFactory6, DXGI_ADAPTER_DESC1,
    DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
};

use super::adapters::AdapterFacts;

// SAFETY (every unsafe block here): DXGI factory and adapter calls whose
// out-values the `windows` wrappers own and release.

fn factory() -> Option<IDXGIFactory1> {
    unsafe { CreateDXGIFactory1() }
        .inspect_err(|e| tracing::warn!(%e, "DXGI factory unavailable; no GPU detected"))
        .ok()
}

/// Each adapter the factory lists, with its description.
fn listed(factory: &IDXGIFactory1) -> Vec<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    let mut listed = Vec::new();
    for index in 0u32.. {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => {
                tracing::warn!(index, %e, "DXGI adapter listing stopped early");
                break;
            }
        };
        match unsafe { adapter.GetDesc1() } {
            Ok(desc) => listed.push((adapter, desc)),
            Err(e) => tracing::warn!(index, %e, "DXGI adapter description unavailable"),
        }
    }
    listed
}

fn is_software(desc: &DXGI_ADAPTER_DESC1) -> bool {
    (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
}

/// Every adapter DXGI lists, and the index of the one Windows ranks first
/// for high performance (Windows 10 1803 and later).
pub(super) fn adapters() -> (Vec<AdapterFacts>, Option<usize>) {
    let Some(factory) = factory() else { return (Vec::new(), None) };
    let listed = listed(&factory);
    let mut adapters = Vec::new();
    for (index, (_, desc)) in listed.iter().enumerate() {
        let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let facts = AdapterFacts {
            name: String::from_utf16_lossy(&desc.Description[..name_len]),
            vendor_id: desc.VendorId,
            dedicated_mb: (desc.DedicatedVideoMemory / (1024 * 1024)) as u64,
            is_software: is_software(desc),
        };
        tracing::info!(
            index,
            name = %facts.name,
            vendor_id = format_args!("{:#06x}", facts.vendor_id),
            device_id = format_args!("{:#06x}", desc.DeviceId),
            dedicated_mb = facts.dedicated_mb,
            software = facts.is_software,
            "DXGI adapter",
        );
        adapters.push(facts);
    }
    let os_choice = factory
        .cast::<IDXGIFactory6>()
        .ok()
        .and_then(|factory6| unsafe { factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) }.ok())
        .and_then(|adapter| unsafe { adapter.GetDesc1() }.ok())
        .and_then(|desc| listed.iter().position(|(_, d)| d.AdapterLuid == desc.AdapterLuid));
    (adapters, os_choice)
}

/// Each hardware adapter as `vendor:device:driver version`, joined; a
/// new driver or another card changes it. The driver version is the
/// user-mode driver's, which DXGI reports for `IDXGIDevice` without
/// creating one.
pub(super) fn driver_identity() -> String {
    let Some(factory) = factory() else { return String::new() };
    listed(&factory)
        .iter()
        .filter(|(_, desc)| !is_software(desc))
        .map(|(adapter, desc)| {
            let driver = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0);
            format!("{:04x}:{:04x}:{driver:x}", desc.VendorId, desc.DeviceId)
        })
        .collect::<Vec<_>>()
        .join(",")
}
