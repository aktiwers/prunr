//! What the bench found, as a table for the terminal and as Markdown and
//! JSON for the user to send back.

use serde::Serialize;

use super::stats::{speedup, Timing};
use super::{Family, Gpu, Row, Side, Verdict};

/// The machine and runtime a report was made on.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Header {
    pub prunr: String,
    pub os: String,
    pub cpu: String,
    pub ram: String,
    pub gpus: Vec<String>,
    pub runtime: String,
    pub gpu_providers: Vec<String>,
    pub input: String,
    pub runs: u32,
}

impl Header {
    /// This machine and the loaded runtime; asks the system for its GPU
    /// names, about a second on Windows.
    pub fn detect(input: &str, runs: u32) -> Self {
        let hw = crate::hardware::profile();
        let threads = std::thread::available_parallelism().map_or(0, |n| n.get());
        let os = sysinfo::System::long_os_version().unwrap_or_else(|| hw.os.to_string());
        let mut gpus = crate::hardware::gpu_names();
        if gpus.is_empty() {
            gpus.extend(hw.dgpu.map(|v| format!("{v} (discrete)")));
            gpus.extend(hw.igpu.map(|v| format!("{v} (integrated)")));
        }
        let diag = prunr_core::ort_runtime::diagnose();
        let tag = prunr_core::ort_runtime::runtime_tag().unwrap_or("version unknown");
        let runtime = match diag.resolved {
            Some((path, source)) => format!("{tag}, {source}: {}", path.display()),
            None => format!("{tag}, not found"),
        };
        Self {
            prunr: format!("{} ({})", env!("CARGO_PKG_VERSION"), if cfg!(debug_assertions) { "debug build" } else { "release build" }),
            os: format!("{os}, {}", hw.arch),
            cpu: format!("{}, {threads} threads", crate::hardware::cpu_label(hw.cpu_vendor, &hw.cpu_brand)),
            ram: format!("{:.0} GB", crate::hardware::total_ram_bytes() as f64 / (1u64 << 30) as f64),
            gpus,
            runtime,
            gpu_providers: prunr_core::engine::gpu_provider_names().into_iter().map(str::to_string).collect(),
            input: input.to_string(),
            runs,
        }
    }

    pub fn lines(&self) -> Vec<(&'static str, String)> {
        let or = |list: &[String], none: &str| if list.is_empty() { none.to_string() } else { list.join("; ") };
        vec![
            ("Prunr", self.prunr.clone()),
            ("OS", self.os.clone()),
            ("CPU", self.cpu.clone()),
            ("RAM", self.ram.clone()),
            ("GPUs", or(&self.gpus, "none detected")),
            ("ONNX Runtime", self.runtime.clone()),
            ("GPU providers", or(&self.gpu_providers, "none in this runtime")),
            ("Input", self.input.clone()),
            ("Runs", format!("each side: 1 first run, then {} timed; the CPU runs a heavy model once", self.runs)),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub header: Header,
    pub rows: Vec<Row>,
    /// The step under way. The report is rewritten as the bench goes, so
    /// one a crash left behind names the step it crashed in.
    pub running: Option<String>,
}

pub const COLUMNS: [&str; 11] = [
    "Model", "GPU path", "Session", "First run", "Steady", "CPU steady", "Speed-up", "Max Δ", "Mean Δ", ">2/255", "Result",
];
const RIGHT_ALIGNED: [bool; 11] = [false, false, true, true, true, true, true, true, true, true, false];

/// A row's cells under `COLUMNS`. The times are the GPU path's; a dash
/// where a value does not apply.
pub fn cells(row: &Row) -> [String; 11] {
    let dash = || "—".to_string();
    let gpu_side = match &row.gpu {
        Gpu::Ran(side) => Some(side),
        _ => None,
    };
    let gpu_timing = gpu_side.and_then(|s| s.timing.as_ref());
    let cpu_timing = row.cpu.as_ref().and_then(|s| s.timing.as_ref());
    let on_gpu = gpu_side.is_some_and(|s| !s.on_cpu && s.error.is_none());
    let speed = match (on_gpu, cpu_timing, gpu_timing) {
        (true, Some(cpu), Some(gpu)) => speedup(cpu, gpu).map(|x| format!("{x:.1}×")),
        _ => None,
    };
    let a = row.agreement.as_ref();
    [
        row.model.to_string(),
        gpu_path(&row.gpu),
        gpu_timing.map_or_else(dash, |t| fmt_ms(t.session_ms)),
        gpu_timing.map_or_else(dash, |t| fmt_ms(t.first_ms)),
        gpu_timing.map_or_else(dash, |t| fmt_ms(t.typical_ms())),
        cpu_timing.map_or_else(dash, |t| fmt_ms(t.typical_ms())),
        speed.unwrap_or_else(dash),
        a.map_or_else(dash, |a| a.max_abs.to_string()),
        a.map_or_else(dash, |a| format!("{:.2}", a.mean_abs)),
        a.map_or_else(dash, |a| format!("{:.1}%", a.over_2 * 100.0)),
        row.verdict.label().to_string(),
    ]
}

fn gpu_path(gpu: &Gpu) -> String {
    match gpu {
        Gpu::Ran(side) => provider_and_weights(side),
        Gpu::NoProvider => "none".into(),
        Gpu::ByDesign { .. } => "CPU by design".into(),
        Gpu::FailedBefore { providers } => format!("{} skipped", providers.join(", ")),
        Gpu::NotInstalled => "—".into(),
    }
}

fn provider_and_weights(side: &Side) -> String {
    match side.weights {
        Some(weights) => format!("{} · {weights}", side.provider),
        None => side.provider.clone(),
    }
}

/// A duration as people read it: "4.2 ms", "380 ms", "2.41 s", "3m 05s".
pub fn fmt_ms(ms: f64) -> String {
    if ms < 10.0 {
        format!("{ms:.1} ms")
    } else if ms < 1000.0 {
        format!("{ms:.0} ms")
    } else if ms < 60_000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else {
        let secs = (ms / 1000.0).round() as u64;
        format!("{}m {:02}s", secs / 60, secs % 60)
    }
}

/// "3 match, 1 MISMATCH, 2 CPU by design", in `Verdict::ALL` order.
pub fn summary(rows: &[Row]) -> String {
    Verdict::ALL.iter()
        .filter_map(|&v| {
            let n = rows.iter().filter(|r| r.verdict == v).count();
            (n > 0).then(|| format!("{n} {}", v.label()))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The header as aligned "Label  value" lines.
pub fn header_text(header: &Header) -> String {
    let lines = header.lines();
    let width = lines.iter().map(|(label, _)| label.chars().count()).max().unwrap_or(0);
    lines.iter().map(|(label, value)| format!("{label:<width$}  {value}\n")).collect()
}

/// The rows as a padded table, numbers right-aligned.
pub fn terminal_table(rows: &[Row]) -> String {
    let body: Vec<[String; 11]> = rows.iter().map(cells).collect();
    let widths: [usize; 11] = std::array::from_fn(|c| {
        body.iter().map(|r| r[c].chars().count()).chain([COLUMNS[c].chars().count()]).max().unwrap_or(0)
    });
    let line = |cells: &[String]| {
        let padded: Vec<String> = cells.iter().enumerate().map(|(c, cell)| {
            let pad = " ".repeat(widths[c] - cell.chars().count());
            if RIGHT_ALIGNED[c] { format!("{pad}{cell}") } else { format!("{cell}{pad}") }
        }).collect();
        padded.join("  ").trim_end().to_string() + "\n"
    };
    let headings = COLUMNS.map(str::to_string);
    let rule: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
    let mut out = line(&headings);
    out.push_str(&line(&rule));
    for row in &body {
        out.push_str(&line(row));
    }
    out
}

pub fn markdown(report: &Report) -> String {
    let mut md = String::from("# Prunr GPU bench\n\n| | |\n|---|---|\n");
    for (label, value) in report.header.lines() {
        md.push_str(&format!("| {label} | {} |\n", escape(&value)));
    }
    if let Some(step) = &report.running {
        md.push_str(&format!("\n> **Running:** {step}. If the bench stopped, it stopped in this step.\n"));
    }
    md.push_str("\n## Results\n\n");
    md.push_str(&format!("| {} |\n", COLUMNS.join(" | ")));
    let align: Vec<&str> = RIGHT_ALIGNED.iter().map(|&r| if r { "---:" } else { "---" }).collect();
    md.push_str(&format!("|{}|\n", align.join("|")));
    for row in &report.rows {
        let cells = cells(row).map(|c| escape(&c));
        md.push_str(&format!("| {} |\n", cells.join(" | ")));
    }
    if !report.rows.is_empty() {
        md.push_str(&format!("\n{}.\n", summary(&report.rows)));
    }
    md.push_str("\n## Details\n");
    for row in &report.rows {
        md.push_str(&details(row));
    }
    md.push_str(&legend());
    md
}

fn details(row: &Row) -> String {
    let mut md = format!("\n### {} · `{}`\n\n", row.model, row.name);
    if let Some(family) = row.family {
        let t = family.tolerance();
        md.push_str(&format!(
            "- Compared on {}; a mismatch above mean Δ {:.1} or {:.0}% of values over 2/255.\n",
            family.compared_on(), t.mean_abs, t.over_2 * 100.0,
        ));
    }
    if let Some(cpu) = &row.cpu {
        md.push_str(&side_lines("CPU", cpu));
    }
    match &row.gpu {
        Gpu::Ran(side) => md.push_str(&side_lines("GPU path", side)),
        Gpu::NoProvider => md.push_str("- GPU path: the loaded ONNX Runtime has no GPU provider.\n"),
        Gpu::ByDesign { reason } => md.push_str(&format!("- GPU path: CPU by design ({reason}).\n")),
        Gpu::FailedBefore { providers } => md.push_str(&format!(
            "- GPU path: skipped. {} failed this model on an earlier run, and the app skips it until `prunr --clear-ep-cache`.\n",
            providers.join(", "),
        )),
        Gpu::NotInstalled => md.push_str("- Not installed: download it in the app's Model Store.\n"),
    }
    if let Some(note) = &row.note {
        md.push_str(&format!("- Note: {note}.\n"));
    }
    md
}

fn side_lines(label: &str, side: &Side) -> String {
    let mut md = format!("- {label}: {}", provider_and_weights(side));
    if let Some(t) = &side.timing {
        md.push_str(&timing_text(t));
    }
    md.push('\n');
    if !side.failed.is_empty() {
        md.push_str(&format!("  - Failed before landing on {}:\n", side.provider));
        md.push_str(&fenced(&side.failed.join("\n")));
    }
    if let Some(error) = &side.error {
        md.push_str("  - Error:\n");
        md.push_str(&fenced(error));
    }
    md
}

fn timing_text(t: &Timing) -> String {
    let mut text = format!("; session {}, first run {}", fmt_ms(t.session_ms), fmt_ms(t.first_ms));
    if !t.runs_ms.is_empty() {
        let list: Vec<String> = t.runs_ms.iter().map(|&ms| fmt_ms(ms)).collect();
        text.push_str(&format!(", steady {} ({})", fmt_ms(t.typical_ms()), list.join(", ")));
    }
    text
}

fn fenced(text: &str) -> String {
    let body: String = text.lines().map(|l| format!("    {l}\n")).collect();
    format!("\n    ```text\n{body}    ```\n\n")
}

fn legend() -> String {
    let tolerances: Vec<String> = Family::ALL.iter().map(|f| {
        let t = f.tolerance();
        format!("{f:?} {:.1} / {:.0}%", t.mean_abs, t.over_2 * 100.0)
    }).collect();
    format!(
        "\n## Reading it\n\n\
         - **GPU path**: where the app's provider ladder put the model's session, and the weights it loaded. \
         \"fell back to CPU\" means no GPU provider took it, or one part of a Stable Diffusion eraser; the details say why.\n\
         - **Session**: building the session, cold. **First run**: the first inference, which includes any compile a provider defers. \
         **Steady**: the mean of the timed runs. **CPU steady**: the same on the CPU provider. **Speed-up**: CPU steady over Steady.\n\
         - **Max Δ, Mean Δ, >2/255**: how far the GPU path's output is from the CPU's, in 0-255 levels, \
         and the share of values more than 2 levels apart. Tolerance (mean Δ / share): {}.\n",
        tolerances.join(", "),
    )
}

/// A table cell's text with its pipes escaped.
fn escape(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

pub fn json(report: &Report) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::stats::Agreement;

    fn header() -> Header {
        Header {
            prunr: "0.5.0 (release build)".into(),
            os: "Windows 11 Pro, x86_64".into(),
            cpu: "AMD Ryzen 9 7950X, 32 threads".into(),
            ram: "64 GB".into(),
            gpus: vec!["NVIDIA GeForce RTX 4090 (driver 32.0.15.6094)".into()],
            runtime: "1.24.1-3fa2c901, bundled: C:\\Program Files\\Prunr\\runtime\\onnxruntime.dll".into(),
            gpu_providers: vec!["DirectML".into()],
            input: "built-in 512×512 picture".into(),
            runs: 3,
        }
    }

    fn timing(session: f64, first: f64, runs: &[f64]) -> Timing {
        Timing { session_ms: session, first_ms: first, runs_ms: runs.to_vec() }
    }

    fn ran(provider: &str, weights: &'static str, t: Timing) -> Side {
        Side { provider: provider.into(), on_cpu: provider == "CPU", weights: Some(weights), timing: Some(t), failed: Vec::new(), error: None }
    }

    fn matched() -> Row {
        Row {
            model: "Silueta",
            name: "silueta",
            family: Some(Family::Segmentation),
            cpu: Some(ran("CPU", "FP32", timing(120.0, 160.0, &[140.0, 140.0]))),
            gpu: Gpu::Ran(ran("DirectML", "FP16", timing(812.0, 95.0, &[12.0, 16.0]))),
            agreement: Some(Agreement { max_abs: 37, mean_abs: 0.41, over_2: 0.012, samples: 262_144 }),
            note: None,
            verdict: Verdict::Match,
        }
    }

    fn skipped() -> Row {
        Row {
            model: "TEED",
            name: "teed",
            family: None,
            cpu: None,
            gpu: Gpu::ByDesign { reason: "its sessions run on the CPU provider only".into() },
            agreement: None,
            note: None,
            verdict: Verdict::CpuByDesign,
        }
    }

    #[test]
    fn durations_read_in_the_unit_that_fits() {
        assert_eq!(fmt_ms(4.23), "4.2 ms");
        assert_eq!(fmt_ms(380.4), "380 ms");
        assert_eq!(fmt_ms(2414.0), "2.41 s");
        assert_eq!(fmt_ms(185_000.0), "3m 05s");
    }

    #[test]
    fn a_row_shows_the_gpu_paths_times_the_cpus_steady_time_and_the_difference() {
        assert_eq!(
            cells(&matched()),
            ["Silueta", "DirectML · FP16", "812 ms", "95 ms", "14 ms", "140 ms", "10.0×", "37", "0.41", "1.2%", "match"].map(str::to_string),
        );
        let dashes = cells(&skipped());
        assert_eq!(dashes[1], "CPU by design");
        assert!(dashes[2..10].iter().all(|c| c == "—"), "{dashes:?}");
        let fell_back = Row {
            gpu: Gpu::Ran(Side { failed: vec!["FP16 weights: ORT session creation failed (DirectML): E_INVALIDARG".into()], ..ran("CPU", "FP32", timing(100.0, 150.0, &[140.0])) }),
            agreement: None,
            verdict: Verdict::FellBack,
            ..matched()
        };
        assert_eq!(cells(&fell_back)[6], "—", "no speed-up for a session that fell back to the CPU");
    }

    #[test]
    fn the_terminal_table_lines_up_its_columns() {
        let table = terminal_table(&[matched(), skipped()]);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("Model    GPU path"), "{}", lines[0]);
        let result_col = lines[0].find("Result").unwrap();
        assert_eq!(lines[2].chars().count(), lines[0][..result_col].chars().count() + "match".len());
        assert!(lines[3].ends_with("CPU by design"));
    }

    #[test]
    fn the_markdown_report_has_the_header_the_table_and_each_models_details() {
        let failed = Row {
            model: "U2Net",
            name: "u2net",
            gpu: Gpu::Ran(Side {
                error: Some("ORT run failed:\nDML device removed | 0x887A0005".into()),
                ..ran("DirectML", "FP16", timing(900.0, 50.0, &[]))
            }),
            agreement: None,
            verdict: Verdict::GpuFailed,
            ..matched()
        };
        let report = Report { header: header(), rows: vec![matched(), failed, skipped()], running: Some("U2Net: GPU timed runs".into()) };
        let md = markdown(&report);
        assert!(md.contains("| GPUs | NVIDIA GeForce RTX 4090 (driver 32.0.15.6094) |"), "{md}");
        assert!(md.contains("> **Running:** U2Net: GPU timed runs."));
        assert!(md.contains("| Silueta | DirectML · FP16 | 812 ms | 95 ms | 14 ms | 140 ms | 10.0× | 37 | 0.41 | 1.2% | match |"), "{md}");
        assert!(md.contains("1 match, 1 GPU run failed, 1 CPU by design."), "{md}");
        assert!(md.contains("### U2Net · `u2net`"));
        assert!(md.contains("    DML device removed | 0x887A0005\n"), "errors keep their text in a block: {md}");
        assert!(md.contains("- GPU path: CPU by design (its sessions run on the CPU provider only)."));
        assert_eq!(md.matches("\n|---|").count(), 2, "one header table, one results table");
    }

    #[test]
    fn the_json_report_carries_every_row_and_the_header() {
        let report = Report { header: header(), rows: vec![matched(), skipped()], running: None };
        let value: serde_json::Value = serde_json::from_str(&json(&report)).unwrap();
        assert_eq!(value["header"]["gpu_providers"][0], "DirectML");
        assert_eq!(value["rows"][0]["gpu"]["kind"], "ran");
        assert_eq!(value["rows"][0]["gpu"]["provider"], "DirectML");
        assert_eq!(value["rows"][0]["verdict"], "match");
        assert_eq!(value["rows"][1]["gpu"]["kind"], "by_design");
        assert!(value["running"].is_null());
    }
}
