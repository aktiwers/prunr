//! `prunr bench --needs`: what each model takes of system RAM and of the
//! graphics card's memory, on the CPU and on the GPU path the app takes,
//! for the registry's `needs`. Each figure is measured in a fresh process
//! of its own (`--needs-one`), so no earlier model's pools or caches count,
//! and is the peak over building the session and two runs, above the
//! process before the model was touched. An upscaler runs one tile of its
//! default size, the input the memory gate's figure stands for.

use std::process::Command;

use prunr_models::ModelId;
use serde::{Deserialize, Serialize};

use super::{input::Inputs, Family};

/// One model on one side, as the child process measured it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub name: String,
    pub gpu_side: bool,
    pub provider: String,
    pub weights: Option<String>,
    pub input: (u32, u32),
    /// Peak system RAM above the process before the model, MB.
    pub host_mb: Option<u64>,
    /// Peak memory this process held on the card, MB; `None` off a card
    /// or where the driver does not say.
    pub card_mb: Option<u64>,
    pub error: Option<String>,
}

impl Point {
    fn new(id: ModelId, gpu_side: bool, input: (u32, u32)) -> Self {
        Self { name: id.stable_name().into(), gpu_side, provider: String::new(), weights: None, input, host_mb: None, card_mb: None, error: None }
    }

    pub fn side_label(&self) -> &'static str {
        if self.gpu_side { "GPU path" } else { "CPU" }
    }
}

/// The line a child prints its point on, apart from any log output.
const MARK: &str = "PRUNR-NEEDS ";

/// Measure `id` on one side here, in this process, and print the point;
/// an upscaler on `tile` px tiles when given, else its own.
pub fn measure_here(id: ModelId, gpu_side: bool, image: Option<image::DynamicImage>, tile: Option<u32>) {
    let point = measure(id, gpu_side, image, tile);
    println!("{MARK}{}", serde_json::to_string(&point).unwrap_or_default());
}

fn measure(id: ModelId, gpu_side: bool, image: Option<image::DynamicImage>, tile: Option<u32>) -> Point {
    let runtime = prunr_core::ort_runtime::ensure_initialized();
    // The tile the upscaler takes on the provider it will run on.
    let provider = if gpu_side { prunr_core::memory::provider_for(id, false) } else { prunr_core::engine::CPU_NAME };
    let tile = tile.unwrap_or_else(|| {
        prunr_models::descriptor(id)
            .and_then(|d| d.recommended_tile)
            .map_or(super::input::UPSCALE_SIDE, |tile| prunr_core::upscale::tile_for_provider(tile, provider))
    });
    let photo = image.unwrap_or_else(|| image::DynamicImage::ImageRgba8(super::input::synthetic_photo()));
    let inputs = Inputs::from_photo_with_tile(photo, String::new(), tile);
    let family = Family::of(id);
    let input = match family {
        Some(Family::Upscale) => inputs.tile.dimensions(),
        _ => inputs.photo.dimensions(),
    };
    let mut point = Point::new(id, gpu_side, input);
    let Some(family) = family else {
        point.error = Some("not measured here: the bench does not run this kind of model".into());
        return point;
    };
    if let Err(e) = runtime {
        point.error = Some(e);
        return point;
    }
    let base = peak::current_mb();
    peak::reset();
    let card = gpu_side.then(card_sampler::start);
    let run = super::open(family, id, &inputs, !gpu_side).and_then(|mut session| {
        let pick = session.pick().clone();
        session.run()?;
        session.run()?;
        Ok((pick, session.weights()))
    });
    point.host_mb = peak::peak_mb().zip(base).map(|(peak, base)| peak.saturating_sub(base));
    point.card_mb = card.and_then(card_sampler::Sampler::stop);
    match run {
        Ok((pick, weights)) => {
            point.provider = pick.provider;
            point.weights = weights.map(String::from);
        }
        Err(e) => point.error = Some(e),
    }
    point
}

/// Each of `models` on the CPU and on its GPU path, each in a child
/// process, an upscaler on `tile` px tiles when given; `on` hears each
/// point as it lands.
pub fn measure_all(models: &[ModelId], image: Option<&std::path::Path>, tile: Option<u32>, on: &mut dyn FnMut(&Point)) -> Vec<Point> {
    let exe = std::env::current_exe().ok();
    let mut points = Vec::new();
    for &id in models {
        let gpu_ladder = super::gpu_passed_over(&prunr_core::engine::gpu_ladder(id)).is_none();
        for gpu_side in [false, true].into_iter().filter(|&gpu| !gpu || gpu_ladder) {
            let point = match &exe {
                Some(exe) => in_child(exe, id, gpu_side, image, tile),
                None => Err("cannot find this program to run a measurement".into()),
            }
            .unwrap_or_else(|error| Point { error: Some(error), ..Point::new(id, gpu_side, (0, 0)) });
            on(&point);
            points.push(point);
        }
    }
    points
}

fn in_child(exe: &std::path::Path, id: ModelId, gpu_side: bool, image: Option<&std::path::Path>, tile: Option<u32>) -> Result<Point, String> {
    let side = if gpu_side { "gpu" } else { "cpu" };
    let mut cmd = Command::new(exe);
    cmd.args(["bench", "--needs-one", &format!("{}:{side}", id.stable_name())]);
    if let Some(image) = image {
        cmd.arg("--image").arg(image);
    }
    if let Some(tile) = tile {
        cmd.arg("--tile").arg(tile.to_string());
    }
    let out = cmd.output().map_err(|e| format!("could not start the measurement: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(MARK))
        .ok_or_else(|| format!("the measurement ended without a figure ({})", out.status))
        .and_then(|json| serde_json::from_str(json).map_err(|e| e.to_string()))
}

/// MB rounded up to the next 50, as the registry carries them.
pub fn rounded(mb: u64) -> u64 {
    mb.div_ceil(50) * 50
}

/// A Markdown report: one row per point, then each GPU point as the
/// registry line it measures.
pub fn report(points: &[Point], machine: &str) -> String {
    let mut out = format!("# Prunr memory needs\n\n{machine}\n\n| Model | Side | Provider | Weights | Input | System RAM | Card |\n|---|---|---|---|---|---:|---:|\n");
    let mb = |v: Option<u64>| v.map_or("—".to_string(), |v| format!("{v} MB"));
    for p in points {
        let side = p.side_label();
        let (provider, weights) = match &p.error {
            Some(e) => (format!("failed: {e}"), String::new()),
            None => (p.provider.clone(), p.weights.clone().unwrap_or_default()),
        };
        out += &format!("| {} | {side} | {provider} | {weights} | {}×{} | {} | {} |\n", p.name, p.input.0, p.input.1, mb(p.host_mb), mb(p.card_mb));
    }
    out += "\n## Registry lines (rounded up to 50 MB)\n\n```rust\n";
    for p in points.iter().filter(|p| p.error.is_none()) {
        let host = rounded(p.host_mb.unwrap_or(0));
        let card = rounded(p.card_mb.unwrap_or(0));
        let field = match p.provider.as_str() {
            "CPU" if !p.gpu_side => "cpu".to_string(),
            "CPU" => continue,
            provider => provider.to_ascii_lowercase(),
        };
        let cost = format!("Cost::measured({}, {}, \"{machine}\")", grouped(host), grouped(card));
        let line = if field == "cpu" { format!("cpu: {cost},") } else { format!("{field}: Priced::Same({cost}),") };
        out += &format!("// {}\n{line}\n", p.name);
    }
    out += "```\n";
    out
}

/// `12_345` as the registry writes it.
fn grouped(v: u64) -> String {
    if v >= 1000 {
        format!("{}_{:03}", v / 1000, v % 1000)
    } else {
        v.to_string()
    }
}

/// The process's peak RAM, resettable where the OS allows it.
mod peak {
    /// Linux: the resident set; macOS: the footprint, Metal's buffers in it.
    pub fn current_mb() -> Option<u64> {
        imp::current_mb()
    }

    pub fn reset() {
        imp::reset()
    }

    pub fn peak_mb() -> Option<u64> {
        imp::peak_mb()
    }

    #[cfg(target_os = "linux")]
    mod imp {
        fn status_mb(key: &str) -> Option<u64> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            let kb: u64 = status.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()?;
            Some(kb / 1024)
        }
        pub fn current_mb() -> Option<u64> {
            status_mb("VmRSS:")
        }
        /// Writing 5 to `clear_refs` sets the peak back to the current size.
        pub fn reset() {
            let _ = std::fs::write("/proc/self/clear_refs", "5");
        }
        pub fn peak_mb() -> Option<u64> {
            status_mb("VmHWM:")
        }
    }

    #[cfg(target_os = "macos")]
    mod imp {
        fn usage() -> Option<libc::rusage_info_v4> {
            let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
            // SAFETY: proc_pid_rusage writes one rusage_info_v4 for the
            // flavour asked, for this process.
            let ok = unsafe { libc::proc_pid_rusage(std::process::id() as i32, libc::RUSAGE_INFO_V4, info.as_mut_ptr().cast()) } == 0;
            // SAFETY: written in full when the call succeeds.
            ok.then(|| unsafe { info.assume_init() })
        }
        pub fn current_mb() -> Option<u64> {
            usage().map(|u| u.ri_phys_footprint / (1024 * 1024))
        }
        /// The lifetime peak cannot be set back; a fresh process starts low.
        pub fn reset() {}
        pub fn peak_mb() -> Option<u64> {
            usage().map(|u| u.ri_lifetime_max_phys_footprint / (1024 * 1024))
        }
    }

    /// Windows: the working set, the resident set's counterpart.
    #[cfg(windows)]
    mod imp {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const MB: usize = 1024 * 1024;

        fn counters() -> Option<PROCESS_MEMORY_COUNTERS> {
            // SAFETY: plain data; zero is a valid value of every field.
            let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
            counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            // SAFETY: writes one PROCESS_MEMORY_COUNTERS of the size given,
            // for this process's pseudo-handle.
            (unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) } != 0).then_some(counters)
        }
        pub fn current_mb() -> Option<u64> {
            counters().map(|c| (c.WorkingSetSize / MB) as u64)
        }
        /// The peak since the process started: each measurement has a
        /// process of its own.
        pub fn reset() {}
        pub fn peak_mb() -> Option<u64> {
            counters().map(|c| (c.PeakWorkingSetSize / MB) as u64)
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    mod imp {
        pub fn current_mb() -> Option<u64> {
            None
        }
        pub fn reset() {}
        pub fn peak_mb() -> Option<u64> {
            None
        }
    }
}

/// The most memory this process holds on its graphics card while a model runs,
/// read every 20 ms.
mod card_sampler {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    pub struct Sampler {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicU64>,
        thread: std::thread::JoinHandle<()>,
    }

    pub fn start() -> Sampler {
        let (stop, peak) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
        let thread = std::thread::spawn({
            let (stop, peak) = (Arc::clone(&stop), Arc::clone(&peak));
            move || {
                let pid = std::process::id();
                while !stop.load(Ordering::Relaxed) {
                    if let Some(mb) = prunr_core::card_memory::process_card_mb(pid) {
                        peak.fetch_max(mb, Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        });
        Sampler { stop, peak, thread }
    }

    impl Sampler {
        /// The peak seen, `None` when the card never showed this process.
        pub fn stop(self) -> Option<u64> {
            self.stop.store(true, Ordering::Relaxed);
            let _ = self.thread.join();
            Some(self.peak.load(Ordering::Relaxed)).filter(|&mb| mb > 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figures_round_up_to_fifty_mb_and_print_as_the_registry_writes_them() {
        assert_eq!((rounded(0), rounded(1), rounded(50), rounded(51), rounded(4_866)), (0, 50, 50, 100, 4_900));
        assert_eq!((grouped(950), grouped(1_000), grouped(12_345)), ("950".to_string(), "1_000".to_string(), "12_345".to_string()));
    }

    /// A GPU point prints as its provider's registry field, a CPU point as
    /// `cpu`, and a GPU side that fell back to the CPU prints nothing.
    #[test]
    fn the_report_prints_each_measured_side_as_its_registry_line() {
        let point = |gpu_side, provider: &str, host, card| Point {
            name: "u2net".into(),
            gpu_side,
            provider: provider.into(),
            weights: Some("FP32".into()),
            input: (512, 512),
            host_mb: Some(host),
            card_mb: card,
            error: None,
        };
        let report = report(&[point(false, "CPU", 760, None), point(true, "CUDA", 1_210, Some(1_480)), point(true, "CPU", 760, None)], "test machine");
        assert!(report.contains("cpu: Cost::measured(800, 0, \"test machine\"),"), "{report}");
        assert!(report.contains("cuda: Priced::Same(Cost::measured(1_250, 1_500, \"test machine\")),"), "{report}");
        assert_eq!(report.matches("Cost::measured").count(), 2, "{report}");
    }
}
