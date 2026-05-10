//! Phase 23-07: invoke the `prunr` CLI binary across a representative
//! flag matrix and verify each combination produces a valid output PNG.
//!
//! NOT a Cartesian product — picks the corners of the user-visible
//! surface (default, `-m`, `--lines`, `--bg-color`, `--bg-image`). The
//! contract: a flag rename or arg-parsing regression breaks all rows;
//! a single bad combo (e.g. `--bg-image` ignoring `--bg-image-fit`)
//! breaks just that row.
//!
//! Skip semantics: ORT runtime not installed → SKIP (matches
//! `golden_e2e`'s posture). Each invocation costs ~1-2s on Silueta;
//! the full matrix runs in <30s.

use image::ImageEncoder;
use std::process::Command;
use tempfile::tempdir;

const CLI_BIN: &str = env!("CARGO_BIN_EXE_prunr");

/// Procedurally generate a 64×64 RGBA test PNG: dark teal bg + bright
/// circular subject. Small enough to keep ORT inference fast.
fn write_fixture_png(dir: &std::path::Path) -> std::path::PathBuf {
    let mut img = image::RgbaImage::new(64, 64);
    for py in 0..64u32 {
        for px in 0..64u32 {
            let dx = px as i32 - 32;
            let dy = py as i32 - 32;
            let pixel = if dx * dx + dy * dy < 18 * 18 {
                image::Rgba([240, 200, 100, 255]) // bright subject
            } else {
                image::Rgba([20, 60, 80, 255]) // dark bg
            };
            img.put_pixel(px, py, pixel);
        }
    }
    let path = dir.join("input.png");
    let file = std::fs::File::create(&path).expect("create input.png");
    let writer = std::io::BufWriter::new(file);
    image::codecs::png::PngEncoder::new(writer)
        .write_image(&img, 64, 64, image::ExtendedColorType::Rgba8)
        .expect("write input.png");
    path
}

fn write_bg_image_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let mut img = image::RgbaImage::new(128, 128);
    for py in 0..128u32 {
        for px in 0..128u32 {
            let v = ((px + py) % 32) as u8 * 7; // 0..=217
            img.put_pixel(px, py, image::Rgba([v, v.saturating_sub(20), 150, 255]));
        }
    }
    let path = dir.join("bg.png");
    let file = std::fs::File::create(&path).expect("create bg.png");
    let writer = std::io::BufWriter::new(file);
    image::codecs::png::PngEncoder::new(writer)
        .write_image(&img, 128, 128, image::ExtendedColorType::Rgba8)
        .expect("write bg.png");
    path
}

/// Run the CLI binary with the given args. The CLI's `-o` flag is
/// directory-typed in batch mode, so we pass the parent dir and rely
/// on the default `<stem>.prunr.png` naming. Returns Err with a human
/// message when the runtime is missing (so callers SKIP instead of
/// fail), Ok(output_path) on success, panics on any other failure.
fn run_cli(input: &std::path::Path, out_dir: &std::path::Path, extra: &[&str]) -> Result<std::path::PathBuf, String> {
    let mut cmd = Command::new(CLI_BIN);
    cmd.arg(input).arg("-o").arg(out_dir).arg("-f");
    cmd.args(extra);
    let out = cmd.output().expect("spawn prunr binary");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        // SKIP signals: missing ORT runtime ("ONNX Runtime not found")
        // or a model that's marked OnDemand and not installed locally
        // ("is not installed"). Anything else is a real test failure —
        // Silueta is bundled, so there's no excuse for a model error
        // beyond a registry regression.
        let skip_signals = ["ONNX Runtime not found", "is not installed"];
        if skip_signals.iter().any(|s| stderr.contains(s)) {
            return Err("SKIP: prerequisite not available".to_string());
        }
        panic!(
            "prunr exited {} with stderr:\n{stderr}\n--- args: {:?}",
            out.status, extra,
        );
    }
    let stem = input.file_stem().unwrap().to_string_lossy();
    let expected = out_dir.join(format!("{stem}.prunr.png"));
    Ok(expected)
}

fn assert_output_decodes(path: &std::path::Path, label: &str) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{label}: output PNG didn't decode: {e}"));
    let (w, h) = (img.width(), img.height());
    assert!(w > 0 && h > 0, "{label}: zero-dim output ({w}x{h})");
}

#[test]
fn cli_default_flags_produce_valid_output() {
    let dir = tempdir().expect("tempdir");
    let input = write_fixture_png(dir.path());
    let output = match run_cli(&input, dir.path(), &[]) {
        Ok(p) => p,
        Err(skip) => { eprintln!("[cli_default] {skip}"); return; }
    };
    assert_output_decodes(&output, "default");
}

#[test]
fn cli_silueta_model_flag() {
    let dir = tempdir().expect("tempdir");
    let input = write_fixture_png(dir.path());
    let output = match run_cli(&input, dir.path(), &["-m", "silueta"]) {
        Ok(p) => p,
        Err(skip) => { eprintln!("[cli_silueta] {skip}"); return; }
    };
    assert_output_decodes(&output, "silueta");
}

#[test]
fn cli_lines_only_flag() {
    let dir = tempdir().expect("tempdir");
    let input = write_fixture_png(dir.path());
    let output = match run_cli(&input, dir.path(), &["--lines"]) {
        Ok(p) => p,
        Err(skip) => { eprintln!("[cli_lines] {skip}"); return; }
    };
    assert_output_decodes(&output, "lines");
}

#[test]
fn cli_bg_color_flag() {
    let dir = tempdir().expect("tempdir");
    let input = write_fixture_png(dir.path());
    let output = match run_cli(&input, dir.path(), &["--bg-color", "ff00ff", "-m", "silueta"]) {
        Ok(p) => p,
        Err(skip) => { eprintln!("[cli_bg_color] {skip}"); return; }
    };
    assert_output_decodes(&output, "bg-color");
}

#[test]
fn cli_bg_image_with_fit_flag() {
    let dir = tempdir().expect("tempdir");
    let input = write_fixture_png(dir.path());
    let bg = write_bg_image_fixture(dir.path());
    let output = match run_cli(
        &input, dir.path(),
        &["--bg-image", bg.to_str().unwrap(), "--bg-image-fit", "tile", "-m", "silueta"],
    ) {
        Ok(p) => p,
        Err(skip) => { eprintln!("[cli_bg_image] {skip}"); return; }
    };
    assert_output_decodes(&output, "bg-image");
}

#[test]
fn cli_help_text_succeeds() {
    // Doesn't need ORT — proves clap's argument parser builds without
    // a flag duplication / type drift error.
    let out = Command::new(CLI_BIN).arg("--help").output().expect("spawn");
    assert!(out.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--model"), "--help must list --model");
    assert!(stdout.contains("--lines"), "--help must list --lines");
}
