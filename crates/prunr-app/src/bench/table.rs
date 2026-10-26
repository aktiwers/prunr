//! `prunr bench --table`: where each model runs on this machine's
//! accelerator, as the registry's `places` lines. The rule: the CPU's
//! result first, then the faster, and within `EVEN` of each other the one
//! taking less system memory.

use prunr_core::ep_compat::{Measured, PlaceKind};
use prunr_core::memory::Charge;
use prunr_models::{Column, ModelId, Needs, Priced};

use super::report::fmt_ms;
use super::stats::steady_pair;
use super::{Family, Gpu, Row, Side, Verdict};

/// Two speeds this close are one run's noise apart: memory decides.
const EVEN: f64 = 1.1;

/// An upscale, shrunk back, this far from its input (0-255 levels) is
/// wrong: right ones land within 4 (1.3-3.8 for every upscaler on CUDA
/// and OpenVINO), broken ones near 123 (Nomos8k black on CUDA, Nomos2 on
/// an HD 530).
const FAR_FROM_INPUT: f64 = 16.0;

/// As the registry writes `place`, measured where `label` says.
fn rust(place: &Measured, label: &str) -> String {
    let text = format!("{} ({label})", place.text);
    match place.kind {
        PlaceKind::There => format!("Place::There({text:?})"),
        why => format!("Place::Cpu(Why::{why:?}({text:?}))"),
    }
}

/// `row`'s place in `column`; `None` when the row measured nothing there
/// (not run, no room, no CPU reference, or a failure that says nothing
/// lasting about the model there). `needs` prices the memory tie-break.
fn found(row: &Row, needs: &Needs, column: Column) -> Option<Measured> {
    let gpu = row.gpu.on_gpu();
    match row.verdict {
        Verdict::GpuFailed | Verdict::FellBack => match &row.gpu {
            Gpu::Ran(side) => lasting_failure(side),
            _ => None,
        },
        Verdict::Mismatch => {
            let off = row.agreement.as_ref().map_or_else(
                || "a different size".to_string(),
                |a| format!("mean {:.2}/255, {:.1}% of values more than 2/255 apart", a.mean_abs, a.over_2 * 100.0),
            );
            Some(Measured::new(PlaceKind::Differs, off))
        }
        Verdict::Match => {
            let (cpu_ms, gpu_ms) = steady_pair(row.cpu.as_ref()?.timing.as_ref()?, gpu?.timing.as_ref()?);
            let ratio = cpu_ms / gpu_ms;
            let times = format!("{ratio:.2}x: {} against the CPU's {}", fmt_ms(gpu_ms), fmt_ms(cpu_ms));
            if ratio >= EVEN {
                return Some(Measured::new(PlaceKind::There, times));
            }
            if ratio <= 1.0 / EVEN {
                return Some(Measured::new(PlaceKind::Slower, times));
            }
            let within = format!("{times}, within {:.0}%", (EVEN - 1.0) * 100.0);
            let faster = if ratio >= 1.0 { PlaceKind::There } else { PlaceKind::NoGain };
            // `Needs` has one OpenVINO entry, measured on the processor.
            let measured = column != Column::OpenVinoGpu && matches!(needs.on(column.provider()), Priced::Same(_));
            if !measured {
                return Some(Measured::new(faster, format!("{within}; its memory there not measured")));
            }
            let cpu_mb = Charge::of(needs, prunr_core::engine::CPU_NAME).ram_mb;
            let there_mb = Charge::of(needs, column.provider()).ram_mb;
            let kind = match there_mb.cmp(&cpu_mb) {
                std::cmp::Ordering::Less => PlaceKind::There,
                std::cmp::Ordering::Equal => faster,
                std::cmp::Ordering::Greater => PlaceKind::NoGain,
            };
            Some(Measured::new(kind, format!("{within}, in {there_mb} MB of system memory against the CPU's {cpu_mb} MB")))
        }
        _ => None,
    }
}

/// The place a GPU run judged alone gives, with no CPU run to compare: a
/// lasting failure, a blank result, or an upscale far from its input keep
/// the model off. `None` when nothing is wrong: it stays unmeasured, and
/// runs there.
pub fn found_alone(side: &Side, family: Family) -> Option<Measured> {
    if side.error.is_some() || side.on_cpu {
        return lasting_failure(side);
    }
    if let (Some(level), Some(blank)) = (side.score.flat, family.blank_in()) {
        return Some(Measured::new(PlaceKind::Differs, format!("its result here is blank: {}", blank.describe(level))));
    }
    side.score.off_input.filter(|&off| off > FAR_FROM_INPUT).map(|off| {
        Measured::new(PlaceKind::Differs, format!("its upscale here, shrunk back, lands {off:.0}/255 from its input; a right one within 4"))
    })
}

/// A GPU side's failure that says something lasting about the model
/// there: its run's error, or for a session that fell back to the CPU the
/// failures on the way. `None` for one that passes (out of memory, a lost
/// device).
fn lasting_failure(side: &Side) -> Option<Measured> {
    let error = match &side.error {
        Some(error) => Some(error.as_str()).filter(|e| lasting(e)),
        None => side.failed.iter().map(String::as_str).find(|e| lasting(e)),
    };
    error.map(|e| Measured::new(PlaceKind::Fails, first_line(e)))
}

/// `row`'s place for `id` in `column`, the column its GPU side ran on.
pub fn found_here(row: &Row, id: ModelId, column: Column) -> Option<Measured> {
    found(row, &prunr_models::descriptor(id)?.needs, column)
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
            Some(found) => out += &format!("// {}\n{}: {},\n", row.name, column.field(), rust(&found, label)),
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
        assert_eq!(at(300.0, Verdict::Match, 900), Some(PlaceKind::There), "1.33x faster, more memory");
        assert_eq!(at(500.0, Verdict::Match, 500), Some(PlaceKind::Slower), "0.8x");
        assert_eq!(at(380.0, Verdict::Match, 600), Some(PlaceKind::There), "1.05x and lighter");
        assert_eq!(at(380.0, Verdict::Match, 800), Some(PlaceKind::NoGain), "1.05x and heavier");
        assert_eq!(at(420.0, Verdict::Match, 600), Some(PlaceKind::There), "0.95x and lighter");
        assert_eq!(at(300.0, Verdict::Mismatch, 600), Some(PlaceKind::Differs), "faster but not the CPU's result");
        assert_eq!(at(300.0, Verdict::NoRoom, 600), None, "nothing measured");
        let unpriced = found(&row(420.0, Verdict::Match), &needs(600), Column::Cuda).expect("placed");
        assert_eq!((unpriced.kind, unpriced.text.ends_with("its memory there not measured")), (PlaceKind::NoGain, true), "0.95x, its memory unknown: the faster");
        let graphics = found(&row(380.0, Verdict::Match), &needs(600), Column::OpenVinoGpu).expect("placed");
        assert!(graphics.text.ends_with("its memory there not measured"), "the OpenVINO figure is the processor's: {}", graphics.text);

        let faster = found(&row(300.0, Verdict::Match), &needs(900), Column::OpenVino).expect("placed");
        assert_eq!(faster, Measured::new(PlaceKind::There, "1.33x: 300 ms against the CPU's 400 ms"));
        assert_eq!(rust(&faster, "test, 2026-10-25"), r#"Place::There("1.33x: 300 ms against the CPU's 400 ms (test, 2026-10-25)")"#);

        let mut failed = row(300.0, Verdict::GpuFailed);
        if let Gpu::Ran(side) = &mut failed.gpu {
            side.error = Some("\nOutput names mismatch between OpenVINO and ONNX\nmore".into());
        }
        assert_eq!(
            found(&failed, &needs(600), Column::OpenVino).map(|f| rust(&f, "i7-6700")),
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
        assert_eq!(found(&fell_back, &needs(600), Column::OpenVino), Some(Measured::new(PlaceKind::Fails, "FP32 weights: Output names mismatch")));
    }

    /// A GPU run judged alone keeps the model off only when it failed for
    /// good, came out blank, or upscaled far from its input.
    #[test]
    fn a_run_judged_alone_keeps_the_model_off_only_when_it_is_wrong() {
        use crate::bench::triage::Score;
        let ran = |score: Score| Side { score, ..side("DirectML", 100.0) };
        let kind = |side: &Side, family| found_alone(side, family).map(|m| m.kind);
        assert_eq!(kind(&ran(Score::default()), Family::Segmentation), None, "right: it stays unmeasured");
        assert_eq!(kind(&ran(Score { flat: Some(0), off_input: None }), Family::Segmentation), Some(PlaceKind::Differs));
        assert_eq!(kind(&ran(Score { flat: Some(0), off_input: None }), Family::FaceFinder), None, "no faces is an answer");
        assert_eq!(kind(&ran(Score { flat: None, off_input: Some(3.8) }), Family::Upscale), None, "a right upscale");
        assert_eq!(kind(&ran(Score { flat: None, off_input: Some(123.5) }), Family::Upscale), Some(PlaceKind::Differs));
        let failed = |error: &str| Side { error: Some(error.into()), ..side("DirectML", 100.0) };
        assert_eq!(kind(&failed("887A0005 device removed"), Family::Upscale), None, "the device's trouble, not the model's");
        assert_eq!(kind(&failed("Output names mismatch"), Family::Upscale), Some(PlaceKind::Fails));
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
