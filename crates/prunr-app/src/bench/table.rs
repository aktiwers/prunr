//! `prunr bench --table`: where each model runs on this machine's
//! accelerator, as the registry's `places` lines. The rule: the CPU's
//! result first, then the faster, and within `EVEN` of each other the one
//! taking less system memory.

use prunr_core::memory::Charge;
use prunr_models::Needs;

use super::report::fmt_ms;
use super::stats::steady_pair;
use super::{Gpu, Row, Verdict};

/// Two speeds this close are one run's noise apart: memory decides.
const EVEN: f64 = 1.1;

/// Where a bench row places its model, as the registry writes it.
#[derive(Debug, Clone, PartialEq)]
pub enum Found {
    There(String),
    /// On the CPU: the `Why` variant and its text.
    Cpu(&'static str, String),
}

impl Found {
    fn rust(&self) -> String {
        match self {
            Found::There(text) => format!("Place::There({text:?})"),
            Found::Cpu(why, text) => format!("Place::Cpu(Why::{why}({text:?}))"),
        }
    }
}

/// `row`'s place on `provider`, measured where `label` says; `None` when
/// the row measured nothing there (not run, no room, no CPU reference).
/// `needs` prices the memory tie-break.
pub fn found(row: &Row, needs: &Needs, provider: &str, label: &str) -> Option<Found> {
    let gpu = row.gpu.on_gpu();
    match row.verdict {
        Verdict::GpuFailed => {
            let error = gpu.and_then(|s| s.error.as_deref()).unwrap_or("no reason given");
            Some(Found::Cpu("Fails", format!("{} ({label})", first_line(error))))
        }
        Verdict::FellBack => {
            let Gpu::Ran(side) = &row.gpu else { return None };
            let failed = side.failed.first()?;
            Some(Found::Cpu("Fails", format!("{} ({label})", first_line(failed))))
        }
        Verdict::Mismatch => {
            let off = row.agreement.as_ref().map_or_else(
                || "a different size".to_string(),
                |a| format!("mean {:.2}/255, {:.1}% of values more than 2/255 apart", a.mean_abs, a.over_2 * 100.0),
            );
            Some(Found::Cpu("Differs", format!("{off} ({label})")))
        }
        Verdict::Match => {
            let (cpu_ms, gpu_ms) = steady_pair(row.cpu.as_ref()?.timing.as_ref()?, gpu?.timing.as_ref()?);
            let ratio = cpu_ms / gpu_ms;
            let times = format!("{ratio:.2}x: {} against the CPU's {}", fmt_ms(gpu_ms), fmt_ms(cpu_ms));
            if ratio >= EVEN {
                return Some(Found::There(format!("{times} ({label})")));
            }
            if ratio <= 1.0 / EVEN {
                return Some(Found::Cpu("Slower", format!("{times} ({label})")));
            }
            let cpu_mb = Charge::of(needs, prunr_core::engine::CPU_NAME).ram_mb;
            let there_mb = Charge::of(needs, provider).ram_mb;
            let memory = format!("{there_mb} MB of system memory against the CPU's {cpu_mb} MB");
            let even = format!("{times}, within 10%; {memory} ({label})");
            Some(if there_mb < cpu_mb || (there_mb == cpu_mb && ratio >= 1.0) { Found::There(even) } else { Found::Cpu("NoGain", even) })
        }
        _ => None,
    }
}

/// The registry field of the column `Places::columns` names `column`.
pub fn field(column: &str) -> Option<&'static str> {
    Some(match column {
        "CUDA" => "cuda",
        "DirectML" => "directml",
        "CoreML" => "coreml",
        "MIGraphX" => "migraphx",
        "OpenVINO" => "openvino",
        "OpenVINO (Intel graphics on)" => "openvino_gpu",
        _ => return None,
    })
}

/// A Markdown section: each row's place on `column`, as registry lines.
pub fn section(rows: &[Row], column: &str, label: &str) -> String {
    let Some(field) = field(column) else {
        return format!("\n## Places\n\nNo registry column for {column}.\n");
    };
    let provider = column.split(" (").next().unwrap_or(column);
    let mut out = format!("\n## Places on {column}\n\n```rust\n");
    for row in rows {
        let Some(needs) = super::model_named(row.name).and_then(prunr_models::descriptor).map(|d| d.needs) else { continue };
        match found(row, &needs, provider, label) {
            Some(found) => out += &format!("// {}\n{field}: {},\n", row.name, found.rust()),
            None => out += &format!("// {}: not measured ({})\n", row.name, row.verdict.label()),
        }
    }
    out += "```\n";
    out
}

fn first_line(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or(text).trim();
    const MAX: usize = 160;
    match line.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::stats::{Agreement, Timing};
    use crate::bench::Side;
    use prunr_models::{Cost, Priced};

    fn side(provider: &str, ms: f64) -> Side {
        Side {
            provider: provider.into(),
            on_cpu: provider == "CPU",
            weights: Some("FP32"),
            timing: Some(Timing { session_ms: 1.0, first_ms: ms * 2.0, runs_ms: vec![ms * 3.0, ms, ms] }),
            failed: Vec::new(),
            error: None,
            score: Default::default(),
        }
    }

    fn row(gpu_ms: f64, verdict: Verdict) -> Row {
        Row {
            model: "U2Net",
            name: "u2net",
            family: None,
            cpu: Some(side("CPU", 400.0)),
            gpu: Gpu::Ran(side("OpenVINO", gpu_ms)),
            agreement: Some(Agreement { max_abs: 29, mean_abs: 1.24, over_2: 0.145, samples: 10 }),
            note: None,
            verdict,
            saved: Vec::new(),
        }
    }

    fn needs(there_mb: u32) -> Needs {
        let cost = |mb| Cost::measured(mb, 0, "test");
        Needs {
            cpu: cost(700),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Same(cost(there_mb)),
            migraphx: Priced::Unmeasured,
        }
    }

    fn kind(found: Option<Found>) -> Option<&'static str> {
        found.map(|f| match f {
            Found::There(_) => "There",
            Found::Cpu(why, _) => why,
        })
    }

    /// The CPU's result first, then the faster by more than 10%; within
    /// 10% the one taking less system memory, the middle run of each side
    /// weighed so one slow run does not decide.
    #[test]
    fn a_model_goes_where_it_is_right_then_faster_then_lighter() {
        let at = |gpu_ms, verdict, there_mb| kind(found(&row(gpu_ms, verdict), &needs(there_mb), "OpenVINO", "test, 2026-10-25"));
        assert_eq!(at(300.0, Verdict::Match, 900), Some("There"), "1.33x faster, more memory");
        assert_eq!(at(500.0, Verdict::Match, 500), Some("Slower"), "0.8x");
        assert_eq!(at(380.0, Verdict::Match, 600), Some("There"), "1.05x and lighter");
        assert_eq!(at(380.0, Verdict::Match, 800), Some("NoGain"), "1.05x and heavier");
        assert_eq!(at(420.0, Verdict::Match, 600), Some("There"), "0.95x and lighter");
        assert_eq!(at(300.0, Verdict::Mismatch, 600), Some("Differs"), "faster but not the CPU's result");
        assert_eq!(at(300.0, Verdict::NoRoom, 600), None, "nothing measured");

        let faster = found(&row(300.0, Verdict::Match), &needs(900), "OpenVINO", "test, 2026-10-25").expect("placed");
        assert_eq!(faster, Found::There("1.33x: 300 ms against the CPU's 400 ms (test, 2026-10-25)".into()));
        assert_eq!(faster.rust(), r#"Place::There("1.33x: 300 ms against the CPU's 400 ms (test, 2026-10-25)")"#);

        let mut failed = row(300.0, Verdict::GpuFailed);
        if let Gpu::Ran(side) = &mut failed.gpu {
            side.error = Some("\nOutput names mismatch between OpenVINO and ONNX\nmore".into());
        }
        assert_eq!(
            found(&failed, &needs(600), "OpenVINO", "i7-6700").map(|f| f.rust()),
            Some(r#"Place::Cpu(Why::Fails("Output names mismatch between OpenVINO and ONNX (i7-6700)"))"#.to_string()),
        );
    }

    #[test]
    fn every_column_has_a_registry_field() {
        for (column, _) in prunr_models::Places::UNMEASURED.columns() {
            assert!(field(column).is_some(), "{column}");
        }
    }
}
