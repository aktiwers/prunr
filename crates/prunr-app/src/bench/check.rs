//! The checker behind `prunr_core::provider_check`: a model's bench, the
//! CPU against the GPU path, run once in a process of its own (`prunr
//! bench --check`), so a provider that hangs or runs out of memory takes
//! only that process with it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use prunr_core::engine::{Weights, CPU_NAME};
use prunr_core::memory;
use prunr_core::provider_check::Check;
use prunr_models::ModelId;

use super::{Family, Gpu, Inputs, Row, Verdict};

/// The GPU path is used when it runs at least this much of the CPU's
/// time: a margin over one timed run's noise.
const FASTER_BY: f64 = 0.9;

/// Past this the provider is passed over: a compile that never ends
/// (CoreML's Neural Engine compiler has hung for an hour).
pub(crate) const LIMIT: Duration = Duration::from_secs(300);

/// The line the check's process prints its verdict on.
pub const LINE: &str = "PRUNR-CHECK ";

/// The verdict a bench row gives its GPU path: passed when its output is
/// the CPU's and it runs in `FASTER_BY` of the CPU's time. `None` when
/// the row says nothing about a provider (none ran the model, or the CPU
/// gave no reference).
pub fn judge(row: &Row) -> Option<Check> {
    let Gpu::Ran(gpu) = &row.gpu else { return None };
    if gpu.on_cpu {
        return None;
    }
    let weights = gpu.weights?.to_string();
    let verdict = |rejected: Option<String>| Some(Check { provider: gpu.provider.clone(), weights: weights.clone(), rejected });
    match row.verdict {
        Verdict::GpuFailed => verdict(Some(format!("it failed to run on {}: {}", gpu.provider, gpu.error.as_deref().unwrap_or("no reason given")))),
        Verdict::Mismatch => {
            let off = row.agreement.as_ref().map_or_else(
                || "a different size".to_string(),
                |a| format!("mean {:.2}/255, {:.1}% of values more than 2/255 apart", a.mean_abs, a.over_2 * 100.0),
            );
            verdict(Some(format!("its output on {} differs from the CPU's ({off})", gpu.provider)))
        }
        Verdict::Match => {
            let cpu_ms = row.cpu.as_ref()?.timing.as_ref()?.typical_ms();
            let gpu_ms = gpu.timing.as_ref()?.typical_ms();
            if gpu_ms <= cpu_ms * FASTER_BY {
                verdict(None)
            } else {
                verdict(Some(format!("it runs slower on {} than on the CPU ({gpu_ms:.0} ms against {cpu_ms:.0} ms)", gpu.provider)))
            }
        }
        _ => None,
    }
}

/// The bench of `id` a check runs: the built-in picture, the CPU's first
/// run as its time (no timed runs: minutes on the CPU for little), the
/// GPU path's one timed run after its first.
pub fn bench_once(id: ModelId) -> Row {
    super::bench_model(id, &Inputs::built_in(), super::Runs { cpu: 0, gpu: 1 }, None, &mut |_| {})
}

/// `prunr_core::provider_check::Checker`. `None` for a model the bench
/// does not cover (and Stable Diffusion, wrong on the CPU by design),
/// and when free memory cannot hold the larger of its two sides.
pub fn run_in_child(model: ModelId, provider: &str, weights: Weights) -> Option<Check> {
    Family::of(model).filter(|&family| family != Family::Diffusion)?;
    if !either_side_fits(model, provider) {
        tracing::info!(?model, provider, "not enough memory free to check the provider; using it unchecked");
        return None;
    }
    let exe = crate::subprocess::manager::worker_exe().ok()?;
    let mut child = Command::new(exe)
        .args(["bench", "--check", model.stable_name()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .inspect_err(|e| tracing::warn!(?model, %e, "could not start the provider check"))
        .ok()?;
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > LIMIT => {
                let _ = child.kill();
                let _ = child.wait();
                tracing::warn!(?model, provider, "the provider check ran past its limit");
                let rejected = Some(format!("its check on {provider} ran past {} minutes", LIMIT.as_secs() / 60));
                return Some(Check { provider: provider.to_string(), weights: weights.label().to_string(), rejected });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => {
                tracing::warn!(?model, %e, "lost the provider check");
                return None;
            }
        }
    };
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    let check = ended(status, &out, provider, weights);
    tracing::info!(?model, provider, ?check, %status, elapsed_s = started.elapsed().as_secs(), "provider check done");
    check
}

/// What a finished check says: its verdict line; or, stopped without one
/// on a crash, a rejection, since the app's own session would crash the
/// same way. A clean exit without a verdict, and a kill (the system's
/// out-of-memory killer), say nothing about the provider.
fn ended(status: std::process::ExitStatus, out: &str, provider: &str, weights: Weights) -> Option<Check> {
    if let Some(check) = parse(out) {
        return Some(check);
    }
    if status.success() {
        return None;
    }
    match crate::subprocess::manager::Crash::of(status) {
        crate::subprocess::manager::Crash::OutOfMemory => None,
        crash => Some(Check {
            provider: provider.to_string(),
            weights: weights.label().to_string(),
            rejected: Some(format!("its check on {provider} {crash}")),
        }),
    }
}

fn parse(out: &str) -> Option<Check> {
    out.lines().find_map(|line| line.strip_prefix(LINE)).and_then(|json| serde_json::from_str(json).ok())
}

/// The bench runs one side, then the other: the check needs room for
/// each.
fn either_side_fits(model: ModelId, provider: &str) -> bool {
    [CPU_NAME, provider].iter().all(|side| memory::fits_now(model, side).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::stats::{Agreement, Timing};
    use crate::bench::Side;

    fn side(provider: &str, ms: f64) -> Side {
        Side {
            provider: provider.into(),
            on_cpu: provider == CPU_NAME,
            weights: Some("FP32"),
            timing: Some(Timing { session_ms: 1.0, first_ms: ms, runs_ms: vec![ms] }),
            failed: Vec::new(),
            error: None,
            score: None,
        }
    }

    fn row(gpu_ms: f64, verdict: Verdict) -> Row {
        Row {
            model: "DDColor",
            name: "ddcolor_tiny",
            family: Some(Family::Colorize),
            cpu: Some(side(CPU_NAME, 1_000.0)),
            gpu: Gpu::Ran(side("CoreML", gpu_ms)),
            agreement: Some(Agreement { max_abs: 29, mean_abs: 1.24, over_2: 0.145, samples: 10 }),
            note: None,
            verdict,
            leaning: None,
            saved: Vec::new(),
        }
    }

    /// Faster at the CPU's output passes; slower, or off the CPU's
    /// output, or failing to run is passed over with the reason.
    #[test]
    fn a_provider_passes_only_faster_at_the_cpus_output() {
        let passed = judge(&row(363.0, Verdict::Match)).expect("a verdict");
        assert_eq!((passed.provider.as_str(), passed.weights.as_str(), passed.rejected), ("CoreML", "FP32", None));
        let slower = judge(&row(950.0, Verdict::Match)).and_then(|c| c.rejected).expect("rejected");
        assert!(slower.contains("950 ms against 1000 ms"), "{slower}");
        let drifts = judge(&row(363.0, Verdict::Mismatch)).and_then(|c| c.rejected).expect("rejected");
        assert!(drifts.contains("14.5%"), "{drifts}");
        let mut failed = row(363.0, Verdict::GpuFailed);
        if let Gpu::Ran(side) = &mut failed.gpu {
            side.error = Some("Error in building plan".into());
        }
        assert!(judge(&failed).and_then(|c| c.rejected).is_some_and(|r| r.contains("Error in building plan")));
    }

    #[test]
    fn nothing_is_judged_without_a_gpu_run_or_a_reference() {
        let mut fell_back = row(363.0, Verdict::FellBack);
        fell_back.gpu = Gpu::Ran(side(CPU_NAME, 900.0));
        assert_eq!(judge(&fell_back), None);
        assert_eq!(judge(&row(363.0, Verdict::NoReference)), None);
        let mut none = row(363.0, Verdict::NoGpuProvider);
        none.gpu = Gpu::NoProvider;
        assert_eq!(judge(&none), None);
        let mut no_room = row(363.0, Verdict::NoRoom);
        no_room.gpu = Gpu::NoRoom { short: "needs about 9000 MB of free memory, 4000 MB is free".into() };
        assert_eq!(judge(&no_room), None, "a shortage of memory says nothing about the provider");
    }

    /// A provider that crashes the check is passed over, so it never
    /// crashes the app; an out-of-memory kill or a clean exit is no verdict.
    #[cfg(unix)]
    #[test]
    fn a_check_that_crashes_passes_the_provider_over() {
        use std::os::unix::process::ExitStatusExt;
        let status = |raw| std::process::ExitStatus::from_raw(raw);
        let crashed = ended(status(11), "", "CoreML", Weights::Fp32).and_then(|c| c.rejected).expect("rejected");
        assert_eq!(crashed, "its check on CoreML crashed (segmentation fault)");
        assert!(ended(status(256), "", "CUDA", Weights::Fp32).is_some(), "an error exit");
        assert_eq!(ended(status(9), "", "CUDA", Weights::Fp32), None, "killed for memory");
        assert_eq!(ended(status(0), "", "CUDA", Weights::Fp32), None, "nothing to judge");
        let passed = Check { provider: "CUDA".into(), weights: "FP32".into(), rejected: None };
        let line = format!("{LINE}{}", serde_json::to_string(&passed).unwrap());
        assert_eq!(ended(status(0), &line, "CUDA", Weights::Fp32), Some(passed));
    }

    #[test]
    fn the_verdict_line_reads_back() {
        let check = Check { provider: "CUDA".into(), weights: "FP16".into(), rejected: None };
        let out = format!("noise\n{LINE}{}\n", serde_json::to_string(&check).unwrap());
        assert_eq!(parse(&out), Some(check));
        assert_eq!(parse("no verdict here"), None);
    }
}
