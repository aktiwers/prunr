//! `prunr bench --table`: where each model runs on this machine's
//! accelerator, as the registry's `places` lines. The rule: the CPU's
//! result first, then the faster, and within `EVEN` of each other the one
//! taking less system memory.

use prunr_core::memory::Charge;
use prunr_models::{Column, ModelId, Needs, Priced};

use super::report::fmt_ms;
use super::stats::steady_pair;
use super::{Gpu, Row, Verdict};

/// Two speeds this close are one run's noise apart: memory decides.
const EVEN: f64 = 1.1;

/// The registry's `Place::There` or the `Why` of `Place::Cpu`, by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    There,
    Fails,
    Differs,
    Slower,
    NoGain,
}

/// Where a bench row places its model, and the measurement that says so.
#[derive(Debug, Clone, PartialEq)]
struct Found {
    kind: Kind,
    text: String,
}

impl Found {
    fn new(kind: Kind, text: impl Into<String>) -> Self {
        Self { kind, text: text.into() }
    }

    /// As the registry writes it, measured where `label` says.
    fn rust(&self, label: &str) -> String {
        let text = format!("{} ({label})", self.text);
        match self.kind {
            Kind::There => format!("Place::There({text:?})"),
            why => format!("Place::Cpu(Why::{why:?}({text:?}))"),
        }
    }
}

/// `row`'s place in `column`; `None` when the row measured nothing there
/// (not run, no room, no CPU reference, or a failure that says nothing
/// lasting about the model there). `needs` prices the memory tie-break.
fn found(row: &Row, needs: &Needs, column: Column) -> Option<Found> {
    let gpu = row.gpu.on_gpu();
    match row.verdict {
        Verdict::GpuFailed => {
            let error = gpu.and_then(|s| s.error.as_deref()).filter(|e| lasting(e))?;
            Some(Found::new(Kind::Fails, first_line(error)))
        }
        Verdict::FellBack => {
            let Gpu::Ran(side) = &row.gpu else { return None };
            let error = side.failed.iter().find(|e| lasting(e))?;
            Some(Found::new(Kind::Fails, first_line(error)))
        }
        Verdict::Mismatch => {
            let off = row.agreement.as_ref().map_or_else(
                || "a different size".to_string(),
                |a| format!("mean {:.2}/255, {:.1}% of values more than 2/255 apart", a.mean_abs, a.over_2 * 100.0),
            );
            Some(Found::new(Kind::Differs, off))
        }
        Verdict::Match => {
            let (cpu_ms, gpu_ms) = steady_pair(row.cpu.as_ref()?.timing.as_ref()?, gpu?.timing.as_ref()?);
            let ratio = cpu_ms / gpu_ms;
            let times = format!("{ratio:.2}x: {} against the CPU's {}", fmt_ms(gpu_ms), fmt_ms(cpu_ms));
            if ratio >= EVEN {
                return Some(Found::new(Kind::There, times));
            }
            if ratio <= 1.0 / EVEN {
                return Some(Found::new(Kind::Slower, times));
            }
            let within = format!("{times}, within {:.0}%", (EVEN - 1.0) * 100.0);
            let faster = if ratio >= 1.0 { Kind::There } else { Kind::NoGain };
            // `Needs` has one OpenVINO entry, measured on the processor.
            let measured = column != Column::OpenVinoGpu && matches!(needs.on(column.provider()), Priced::Same(_));
            if !measured {
                return Some(Found::new(faster, format!("{within}; its memory there not measured")));
            }
            let cpu_mb = Charge::of(needs, prunr_core::engine::CPU_NAME).ram_mb;
            let there_mb = Charge::of(needs, column.provider()).ram_mb;
            let kind = match there_mb.cmp(&cpu_mb) {
                std::cmp::Ordering::Less => Kind::There,
                std::cmp::Ordering::Equal => faster,
                std::cmp::Ordering::Greater => Kind::NoGain,
            };
            Some(Found::new(kind, format!("{within}, in {there_mb} MB of system memory against the CPU's {cpu_mb} MB")))
        }
        _ => None,
    }
}

/// A Markdown section: each row's place in the column called `column`,
/// as registry lines measured where `label` says. A row is written only
/// where it ran: `column_of` names the column a model's session on the
/// provider reads (the Intel graphics leave fills, and models slower
/// there, on OpenVINO's processor).
pub fn section(rows: &[Row], column: &str, label: &str, column_of: impl Fn(ModelId, &str) -> Option<Column>) -> String {
    let Some(column) = Column::named(column) else {
        return format!("\n## Places\n\nNo registry column for {column}.\n");
    };
    let mut out = format!("\n## Places on {}\n\n```rust\n", column.name());
    for row in rows {
        let Some(id) = super::model_named(row.name) else { continue };
        let Some(needs) = prunr_models::descriptor(id).map(|d| d.needs) else { continue };
        let ran_on = column_of(id, column.provider());
        if ran_on != Some(column) {
            let elsewhere = ran_on.map_or("the CPU", Column::name);
            out += &format!("// {}: ran on {elsewhere}, not measured here\n", row.name);
            continue;
        }
        match found(row, &needs, column) {
            Some(found) => out += &format!("// {}\n{}: {},\n", row.name, column.field(), found.rust(label)),
            None => out += &format!("// {}: not measured ({})\n", row.name, row.verdict.label()),
        }
    }
    out += "```\n";
    out
}

/// Whether a failure says something lasting about the model there: not
/// the device's own trouble (`ep_compat::is_transient`), not a provider
/// this machine could not set up, not a provider passed over untried.
fn lasting(error: &str) -> bool {
    !prunr_core::ep_compat::is_transient(error) && !error.contains("ORT register") && !error.contains("no GPU provider to try")
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


    /// The CPU's result first, then the faster by more than 10%; within
    /// 10% the one taking less system memory, the middle run of each side
    /// weighed so one slow run does not decide.
    #[test]
    fn a_model_goes_where_it_is_right_then_faster_then_lighter() {
        let at = |gpu_ms, verdict, there_mb| found(&row(gpu_ms, verdict), &needs(there_mb), Column::OpenVino).map(|f| f.kind);
        assert_eq!(at(300.0, Verdict::Match, 900), Some(Kind::There), "1.33x faster, more memory");
        assert_eq!(at(500.0, Verdict::Match, 500), Some(Kind::Slower), "0.8x");
        assert_eq!(at(380.0, Verdict::Match, 600), Some(Kind::There), "1.05x and lighter");
        assert_eq!(at(380.0, Verdict::Match, 800), Some(Kind::NoGain), "1.05x and heavier");
        assert_eq!(at(420.0, Verdict::Match, 600), Some(Kind::There), "0.95x and lighter");
        assert_eq!(at(300.0, Verdict::Mismatch, 600), Some(Kind::Differs), "faster but not the CPU's result");
        assert_eq!(at(300.0, Verdict::NoRoom, 600), None, "nothing measured");
        let unpriced = found(&row(420.0, Verdict::Match), &needs(600), Column::Cuda).expect("placed");
        assert_eq!((unpriced.kind, unpriced.text.ends_with("its memory there not measured")), (Kind::NoGain, true), "0.95x, its memory unknown: the faster");
        let graphics = found(&row(380.0, Verdict::Match), &needs(600), Column::OpenVinoGpu).expect("placed");
        assert!(graphics.text.ends_with("its memory there not measured"), "the OpenVINO figure is the processor's: {}", graphics.text);

        let faster = found(&row(300.0, Verdict::Match), &needs(900), Column::OpenVino).expect("placed");
        assert_eq!(faster, Found::new(Kind::There, "1.33x: 300 ms against the CPU's 400 ms"));
        assert_eq!(faster.rust("test, 2026-10-25"), r#"Place::There("1.33x: 300 ms against the CPU's 400 ms (test, 2026-10-25)")"#);

        let mut failed = row(300.0, Verdict::GpuFailed);
        if let Gpu::Ran(side) = &mut failed.gpu {
            side.error = Some("\nOutput names mismatch between OpenVINO and ONNX\nmore".into());
        }
        assert_eq!(
            found(&failed, &needs(600), Column::OpenVino).map(|f| f.rust("i7-6700")),
            Some(r#"Place::Cpu(Why::Fails("Output names mismatch between OpenVINO and ONNX (i7-6700)"))"#.to_string()),
        );
    }

    /// Only a failure that says something lasting about the model makes it
    /// a permanent failure there: not running out of memory, not a skipped
    /// copy of its weights.
    #[test]
    fn a_passing_failure_is_not_written_as_failing_there() {
        let mut out_of_memory = row(300.0, Verdict::GpuFailed);
        if let Gpu::Ran(side) = &mut out_of_memory.gpu {
            side.error = Some("CUDA failure 2: out of memory".into());
        }
        assert_eq!(found(&out_of_memory, &needs(600), Column::Cuda), None);
        let mut fell_back = row(300.0, Verdict::FellBack);
        if let Gpu::Ran(side) = &mut fell_back.gpu {
            side.failed = vec![
                "FP16 weights: Inference error: no GPU provider to try: OpenVINO is kept off: slower".into(),
                "FP32 weights: Output names mismatch".into(),
            ];
        }
        assert_eq!(found(&fell_back, &needs(600), Column::OpenVino), Some(Found::new(Kind::Fails, "FP32 weights: Output names mismatch")));
    }

    /// A row goes into the column it ran on: with the Intel graphics on, a
    /// fill ran on OpenVINO's processor and is not written as a graphics run.
    #[test]
    fn a_row_is_written_only_into_the_column_it_ran_on() {
        let rows = [
            Row { name: "real_esrgan_x4plus", ..row(300.0, Verdict::Match) },
            Row { name: "lama_fp32", ..row(300.0, Verdict::Match) },
        ];
        let column_of = |id: ModelId, _: &str| Some(if id == ModelId::LaMaFp32 { Column::OpenVino } else { Column::OpenVinoGpu });
        let out = section(&rows, Column::OpenVinoGpu.name(), "test", column_of);
        assert!(out.contains("// real_esrgan_x4plus\nopenvino_gpu: Place::There("), "{out}");
        assert!(out.contains("// lama_fp32: ran on OpenVINO, not measured here"), "{out}");
        assert_eq!(out.matches("openvino_gpu:").count(), 1, "only the run on the graphics: {out}");
    }
}
