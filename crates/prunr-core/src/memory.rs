//! One memory check for every model: what it needs where it will run
//! (`prunr_models::Needs`), against what is free now. A provider whose
//! needs are not measured yet is charged the CPU figure in system RAM,
//! the rule every model ran under before card memory was counted.

use prunr_models::{ModelId, Needs, Priced};

/// What a model takes where it runs; `card_mb` is 0 off a card and on
/// unified memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Charge {
    pub ram_mb: u64,
    pub card_mb: u64,
}

impl Charge {
    /// `needs` on `provider`; `None` when the model is never run there.
    pub fn of(needs: &Needs, provider: &str) -> Option<Self> {
        match needs.on(provider) {
            Priced::Same(cost) => Some(Self { ram_mb: cost.host_mb.into(), card_mb: cost.card_mb.into() }),
            Priced::Unmeasured => Some(Self { ram_mb: needs.host_cpu().host_mb.into(), card_mb: 0 }),
            Priced::Never => None,
        }
    }

    pub fn on_card(self) -> bool {
        self.card_mb > 0
    }
}

/// What is free: system RAM, and the card's memory (`None`: no reading,
/// and a model placed on a card is trusted there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    pub ram_mb: u64,
    pub card_mb: Option<u64>,
}

/// What a refused run was short of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Short {
    Ram { need_mb: u64, free_mb: u64 },
    Card { need_mb: u64, free_mb: u64 },
}

/// The card's memory kept free for the desktop and the driver, as
/// ComfyUI keeps (more on Windows, where the compositor holds more).
pub const CARD_RESERVE_MB: u64 = if cfg!(windows) { 600 } else { 400 };

/// Whether `charge` fits `room`: the card first (it is the smaller), then
/// system RAM.
pub fn fit(charge: Charge, room: Room) -> Result<(), Short> {
    if let Some(free_mb) = room.card_mb.filter(|_| charge.on_card()) {
        let need_mb = charge.card_mb + CARD_RESERVE_MB;
        if free_mb < need_mb {
            return Err(Short::Card { need_mb, free_mb });
        }
    }
    if room.ram_mb < charge.ram_mb {
        return Err(Short::Ram { need_mb: charge.ram_mb, free_mb: room.ram_mb });
    }
    Ok(())
}

/// Whether a run that asked for a GPU goes to the CPU: its GPU charge
/// (`asked`, with the room for it) runs where it fits, else the CPU's
/// (`cpu`, `None` when it asked for the CPU) when that fits. `None` for a
/// side's charge or room: no reading, and the side is trusted. `Err`: it
/// fits neither, short of what the CPU would need.
pub fn falls_to_cpu(asked: Option<(Charge, Room)>, cpu: Option<(Charge, Room)>) -> Result<bool, Short> {
    let Some((charge, room)) = asked else { return Ok(false) };
    let Err(short) = fit(charge, room) else { return Ok(false) };
    match cpu {
        Some((charge, room)) => fit(charge, room).map(|()| true),
        None => Err(short),
    }
}

/// What `id` takes on `provider`; the CPU's figure where it never runs
/// there (the ladder then falls to the CPU).
pub fn charge_on(id: ModelId, provider: &str) -> Option<Charge> {
    let needs = prunr_models::descriptor(id)?.needs;
    Charge::of(&needs, provider).or_else(|| Charge::of(&needs, crate::engine::CPU_NAME))
}

/// How many of `wanted` engines of `id` fit the card `provider` runs it
/// on, at least one: each engine holds its own pool there (BiRefNet takes
/// 7.7 GB of an RTX 3060, so a second would run out). Off a card, or with
/// no reading of it, `wanted`.
pub fn engines_that_fit(id: ModelId, provider: &str, wanted: usize) -> usize {
    let Some(charge) = charge_on(id, provider).filter(|c| c.on_card()) else { return wanted };
    match room_now(charge, provider).and_then(|room| room.card_mb) {
        Some(free_mb) => fitting(charge.card_mb, free_mb, wanted),
        None => wanted,
    }
}

fn fitting(card_mb: u64, free_mb: u64, wanted: usize) -> usize {
    let fits = free_mb.saturating_sub(CARD_RESERVE_MB) / card_mb.max(1);
    usize::try_from(fits).unwrap_or(usize::MAX).clamp(1, wanted.max(1))
}

/// The provider a session of `id` starts on: the first GPU provider the
/// engine ladder would try for it, or the CPU.
pub fn provider_for(id: ModelId, cpu_only: bool) -> &'static str {
    crate::engine::first_gpu_provider(id, cpu_only).unwrap_or(crate::engine::CPU_NAME)
}

/// `provider_for` without waiting for the GPU probe, for the UI thread:
/// `None` until the probe has answered.
pub fn provider_once_probed(id: ModelId, cpu_only: bool) -> Option<&'static str> {
    crate::engine::OrtEngine::detected_provider().map(|_| provider_for(id, cpu_only))
}

/// What is free now for a run charged `charge`: the card is read only
/// when the run goes to one and it is CUDA's. `None` when free RAM
/// cannot be read.
pub fn room_now(charge: Charge, provider: &str) -> Option<Room> {
    let ram_mb = crate::sysmem::available_ram_bytes()? / (1024 * 1024);
    let card_mb = (charge.on_card() && provider == crate::engine::CUDA_NAME)
        .then(crate::card_memory::cuda_card_memory)
        .flatten()
        .map(|card| card.free_mb);
    Some(Room { ram_mb, card_mb })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::{Cost, Priced};

    fn needs(cuda: Priced) -> Needs {
        Needs {
            cpu: Cost::estimate(9_000, 0, "test"),
            mac_cpu: Priced::Unmeasured,
            cuda,
            directml: Priced::Unmeasured,
            coreml: Priced::Same(Cost::estimate(3_000, 0, "test")),
            openvino: Priced::Never,
            migraphx: Priced::Unmeasured,
        }
    }

    /// A GPU run that does not fit goes to the CPU when the CPU's figure
    /// does, and is refused only when neither fits.
    #[test]
    fn a_gpu_run_that_does_not_fit_goes_to_the_cpu() {
        let coreml = Charge { ram_mb: 6_200, card_mb: 0 };
        let cuda = Charge { ram_mb: 1_000, card_mb: 2_200 };
        let cpu = Charge { ram_mb: 2_500, card_mb: 0 };
        let ram = |ram_mb| Room { ram_mb, card_mb: None };
        assert_eq!(falls_to_cpu(Some((coreml, ram(8_000))), Some((cpu, ram(8_000)))), Ok(false));
        assert_eq!(falls_to_cpu(Some((coreml, ram(4_000))), Some((cpu, ram(4_000)))), Ok(true));
        assert_eq!(falls_to_cpu(Some((coreml, ram(2_000))), Some((cpu, ram(2_000)))), Err(Short::Ram { need_mb: 2_500, free_mb: 2_000 }));
        let card = Room { ram_mb: 16_000, card_mb: Some(1_000) };
        assert_eq!(falls_to_cpu(Some((cuda, card)), Some((cpu, ram(16_000)))), Ok(true), "the card is short, the RAM is not");
        assert_eq!(falls_to_cpu(Some((cpu, ram(2_000))), None), Err(Short::Ram { need_mb: 2_500, free_mb: 2_000 }), "asked for the CPU");
        assert_eq!(falls_to_cpu(None, Some((cpu, ram(0)))), Ok(false), "no reading: trusted");
    }

    /// Measured on a provider: its own figures. Unmeasured: the CPU's, in
    /// system RAM. Never: no charge, the model does not run there.
    #[test]
    fn a_model_is_charged_what_it_needs_where_it_runs() {
        let measured = needs(Priced::Same(Cost::measured(2_000, 6_000, "test")));
        assert_eq!(Charge::of(&measured, "CUDA"), Some(Charge { ram_mb: 2_000, card_mb: 6_000 }));
        assert_eq!(Charge::of(&measured, "CPU"), Some(Charge { ram_mb: 9_000, card_mb: 0 }));
        assert_eq!(Charge::of(&measured, "DirectML"), Some(Charge { ram_mb: 9_000, card_mb: 0 }), "unmeasured: the CPU's figure");
        assert_eq!(Charge::of(&measured, "CoreML"), Some(Charge { ram_mb: 3_000, card_mb: 0 }), "unified memory");
        assert_eq!(Charge::of(&measured, "OpenVINO"), None);
        assert_eq!(Charge::of(&needs(Priced::Unmeasured), "CUDA"), Some(Charge { ram_mb: 9_000, card_mb: 0 }));
    }

    /// As many engines as the card holds with its reserve free, never
    /// more than asked for and never none.
    #[test]
    fn engines_on_a_card_are_as_many_as_fit() {
        let free = 7_700 * 2 + CARD_RESERVE_MB;
        assert_eq!(fitting(7_700, free, 2), 2);
        assert_eq!(fitting(7_700, free - 1, 2), 1);
        assert_eq!(fitting(7_700, 0, 2), 1, "one engine even when none fits: it fails or falls to the CPU as before");
        assert_eq!(fitting(650, 11_000, 2), 2, "never more than asked");
    }

    /// The card counts only for a run on it, with the reserve kept free;
    /// an unread card is trusted; system RAM always counts.
    #[test]
    fn the_fit_counts_the_card_only_for_a_run_on_it() {
        let card = Charge { ram_mb: 2_000, card_mb: 6_000 };
        let room = |ram_mb, card_mb| Room { ram_mb, card_mb };
        assert_eq!(fit(card, room(2_000, Some(6_000 + CARD_RESERVE_MB))), Ok(()));
        assert_eq!(
            fit(card, room(2_000, Some(6_000 + CARD_RESERVE_MB - 1))),
            Err(Short::Card { need_mb: 6_000 + CARD_RESERVE_MB, free_mb: 6_000 + CARD_RESERVE_MB - 1 })
        );
        assert_eq!(fit(card, room(1_999, Some(u64::MAX))), Err(Short::Ram { need_mb: 2_000, free_mb: 1_999 }));
        assert_eq!(fit(card, room(2_000, None)), Ok(()), "no card reading: trusted");
        let host = Charge { ram_mb: 2_000, card_mb: 0 };
        assert_eq!(fit(host, room(2_000, Some(0))), Ok(()), "a run on the host ignores the card");
    }
}
