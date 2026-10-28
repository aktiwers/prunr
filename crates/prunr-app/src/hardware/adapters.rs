//! Which display adapter is the graphics card, which the integrated GPU,
//! and whether one suits DirectML, from what DXGI reports of each. Pure,
//! so the rules are tested on every platform.

use super::{gpu_vendor, GpuSlots, GpuVendor};

/// One display adapter as DXGI reports it, read without creating a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AdapterFacts {
    pub(super) name: String,
    pub(super) vendor_id: u32,
    pub(super) dedicated_mb: u64,
    pub(super) is_software: bool,
}

const MICROSOFT_VENDOR_ID: u32 = 0x1414;

/// Dedicated memory that only a graphics card has: Intel and AMD iGPUs
/// reserve 128 MB to 1 GB, Intel Arc and every recent card have 4 GB or more.
const CARD_MIN_DEDICATED_MB: u64 = 2048;

impl AdapterFacts {
    /// A GPU, not a software adapter (WARP, the Basic Render Driver a VM
    /// or remote desktop gets).
    fn is_hardware(&self) -> bool {
        !self.is_software && self.vendor_id != MICROSOFT_VENDOR_ID
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuKind {
    Discrete,
    Integrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdapterClass {
    kind: GpuKind,
    vendor: GpuVendor,
    suits_directml: bool,
}

/// What a display adapter is; `None` for a software one. `strong`: it
/// is the strong card beside another hardware adapter (`Some(true)`), or
/// another adapter beside one (`Some(false)`), where an AMD or Intel
/// adapter is the integrated GPU however much memory it reserves (a
/// Ryzen APU can hold 2 GB or more). An Intel iGPU without a card beside
/// it does not suit DirectML.
fn classify_adapter(adapter: &AdapterFacts, strong: Option<bool>) -> Option<AdapterClass> {
    if !adapter.is_hardware() {
        return None;
    }
    let vendor = gpu_vendor(adapter.vendor_id);
    let has_card_memory = adapter.dedicated_mb >= CARD_MIN_DEDICATED_MB;
    let kind = if strong == Some(true) || vendor == GpuVendor::Nvidia || (has_card_memory && strong != Some(false)) {
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
fn strong_card(adapters: &[AdapterFacts], os_choice: Option<usize>) -> Option<usize> {
    if adapters.iter().filter(|a| a.is_hardware()).count() < 2 {
        return None;
    }
    os_choice.filter(|&i| adapters.get(i).is_some_and(AdapterFacts::is_hardware)).or_else(|| {
        adapters.iter().enumerate()
            .filter(|(_, a)| a.is_hardware())
            .max_by_key(|(_, a)| a.dedicated_mb)
            .map(|(i, _)| i)
    })
}

pub(super) fn sort_adapters(adapters: &[AdapterFacts], os_choice: Option<usize>) -> GpuSlots {
    let strong = strong_card(adapters, os_choice);
    let mut slots = GpuSlots::default();
    // The strong card first: it fills the card's slot, and names it.
    let order = strong.into_iter().chain((0..adapters.len()).filter(|&i| Some(i) != strong));
    for i in order {
        let a = &adapters[i];
        let Some(class) = classify_adapter(a, strong.map(|s| s == i)) else { continue };
        let slot = match class.kind {
            GpuKind::Discrete if slots.dgpu.is_none() => {
                slots.dgpu_name = Some(a.name.clone()).filter(|name| !name.is_empty());
                &mut slots.dgpu
            }
            GpuKind::Discrete => &mut slots.dgpu,
            GpuKind::Integrated => &mut slots.igpu,
        };
        slot.get_or_insert(class.vendor);
        slots.suits_directml |= class.suits_directml;
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(vendor_id: u32, dedicated_mb: u64) -> AdapterFacts {
        AdapterFacts { name: String::new(), vendor_id, dedicated_mb, is_software: false }
    }

    /// Microsoft Basic Render Driver.
    fn basic_render() -> AdapterFacts {
        AdapterFacts { name: "Microsoft Basic Render Driver".into(), vendor_id: 0x1414, dedicated_mb: 0, is_software: true }
    }

    /// The card's name is the adapter in the card's slot: not the iGPU
    /// beside it, even one that reserves card-sized memory, nor the
    /// software adapter a VM shows.
    #[test]
    fn the_card_named_is_the_one_in_the_cards_slot() {
        let named = |name: &str, vendor_id, dedicated_mb| AdapterFacts { name: name.into(), ..adapter(vendor_id, dedicated_mb) };
        let hybrid = [named("Intel(R) UHD Graphics 630", 0x8086, 128), named("NVIDIA GeForce RTX 3060 Laptop GPU", 0x10de, 6144)];
        let apu = [named("AMD Radeon 780M Graphics", 0x1002, 4096), named("NVIDIA GeForce RTX 4060 Laptop GPU", 0x10de, 8192)];
        let card_name = |adapters: &[AdapterFacts]| sort_adapters(adapters, Some(adapters.len() - 1)).dgpu_name;
        assert_eq!(card_name(&apu).as_deref(), Some("NVIDIA GeForce RTX 4060 Laptop GPU"), "not the APU before it");
        assert_eq!(card_name(&hybrid).as_deref(), Some("NVIDIA GeForce RTX 3060 Laptop GPU"));
        assert_eq!(card_name(&hybrid[..1]), None, "an iGPU alone is no card");
        assert_eq!(card_name(&[basic_render()]), None);
    }

    /// One row per machine shape the Windows detection meets.
    #[test]
    fn dxgi_adapters_sort_into_slots_and_the_directml_verdict() {
        let none = GpuSlots::default;
        let rows: [(&str, Vec<AdapterFacts>, Option<usize>, GpuSlots); 10] = [
            ("VM or remote desktop", vec![basic_render()], Some(0), none()),
            ("Intel iGPU alone", vec![adapter(0x8086, 128), basic_render()], Some(0),
                GpuSlots { igpu: Some(GpuVendor::Intel), ..none() }),
            ("Intel Arc iGPU alone", vec![adapter(0x8086, 128)], None,
                GpuSlots { igpu: Some(GpuVendor::Intel), ..none() }),
            ("Intel Arc card", vec![adapter(0x8086, 16_000)], Some(0),
                GpuSlots { dgpu: Some(GpuVendor::Intel), suits_directml: true, ..none() }),
            ("hybrid laptop, OS ranks the card", vec![adapter(0x8086, 128), adapter(0x10de, 4096)], Some(1),
                GpuSlots { dgpu: Some(GpuVendor::Nvidia), igpu: Some(GpuVendor::Intel), suits_directml: true, ..none() }),
            ("Ryzen APU alone", vec![adapter(0x1002, 512)], Some(0),
                GpuSlots { igpu: Some(GpuVendor::Amd), suits_directml: true, ..none() }),
            ("Ryzen APU + small Radeon card, OS ranks the card", vec![adapter(0x1002, 512), adapter(0x1002, 1024)], Some(1),
                GpuSlots { dgpu: Some(GpuVendor::Amd), igpu: Some(GpuVendor::Amd), suits_directml: true, ..none() }),
            ("Ryzen APU + small Radeon card, no OS ranking", vec![adapter(0x1002, 1024), adapter(0x1002, 512)], None,
                GpuSlots { dgpu: Some(GpuVendor::Amd), igpu: Some(GpuVendor::Amd), suits_directml: true, ..none() }),
            ("Ryzen APU with 4 GB reserved + NVIDIA card, OS ranks the card", vec![adapter(0x1002, 4096), adapter(0x10de, 8192)], Some(1),
                GpuSlots { dgpu: Some(GpuVendor::Nvidia), igpu: Some(GpuVendor::Amd), suits_directml: true, ..none() }),
            ("NVIDIA card alone", vec![adapter(0x10de, 2048), basic_render()], Some(0),
                GpuSlots { dgpu: Some(GpuVendor::Nvidia), suits_directml: true, ..none() }),
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
}
