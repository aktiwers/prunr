//! Invoke the `prunr` CLI binary across a representative
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

/// Run the CLI on several inputs, killed if it outlives `limit`: the exit
/// code (`None`: it hung) and stderr.
fn run_cli_batch(inputs: &[&std::path::Path], out_dir: &std::path::Path, extra: &[&str], limit: std::time::Duration) -> (Option<i32>, String) {
    let mut child = Command::new(CLI_BIN)
        .args(inputs)
        .arg("-o").arg(out_dir).arg("-f")
        .args(extra)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn prunr binary");
    let mut stderr_pipe = child.stderr.take().expect("piped stderr");
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut stderr_pipe, &mut text);
        text
    });
    let deadline = std::time::Instant::now() + limit;
    let code = loop {
        if let Some(status) = child.try_wait().expect("wait on prunr") {
            break status.code();
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    (code, reader.join().unwrap_or_default())
}

/// A batch goes on past an image that fails: the next one is processed
/// and the run ends with a partial-failure exit.
#[test]
fn cli_batch_goes_on_after_an_image_fails() {
    let dir = tempdir().expect("tempdir");
    let good = write_fixture_png(dir.path());
    let bytes = std::fs::read(&good).expect("read fixture");
    // Its header reads, so it reaches the worker; its pixels do not decode.
    let broken = dir.path().join("broken.png");
    std::fs::write(&broken, &bytes[..bytes.len() / 2]).expect("write broken.png");
    let (code, stderr) = run_cli_batch(&[&broken, &good], dir.path(), &["-m", "silueta", "-j", "1"], std::time::Duration::from_secs(120));
    if ["ONNX Runtime not found", "is not installed"].iter().any(|s| stderr.contains(s)) {
        eprintln!("[cli_batch_error] SKIP: prerequisite not available");
        return;
    }
    assert_eq!(code, Some(2), "a partial failure, not a hang; stderr:\n{stderr}");
    assert_output_decodes(&dir.path().join("input.prunr.png"), "the image after the failure");
}

/// An input that is not an image says so on stderr when no spinner shows
/// it (output piped, as from a script), and the batch goes on.
#[test]
fn an_input_that_is_not_an_image_says_so() {
    let dir = tempdir().expect("tempdir");
    let good = write_fixture_png(dir.path());
    let notes = dir.path().join("notes.png");
    std::fs::write(&notes, b"not an image").expect("write notes.png");
    let (code, stderr) = run_cli_batch(&[&notes, &good], dir.path(), &["-m", "silueta"], std::time::Duration::from_secs(120));
    if ["ONNX Runtime not found", "is not installed"].iter().any(|s| stderr.contains(s)) {
        eprintln!("[cli_not_an_image] SKIP: prerequisite not available");
        return;
    }
    assert_eq!(code, Some(2), "stderr:\n{stderr}");
    assert!(stderr.lines().any(|l| l.contains("notes.png") && l.contains("not a valid image")), "stderr:\n{stderr}");
}

/// A file name that is not UTF-8 (Linux allows any bytes) reaches the
/// worker and is processed like any other.
#[cfg(target_os = "linux")]
#[test]
fn cli_processes_a_file_whose_name_is_not_utf8() {
    use std::os::unix::ffi::OsStrExt;
    let dir = tempdir().expect("tempdir");
    let good = write_fixture_png(dir.path());
    let odd = dir.path().join(std::ffi::OsStr::from_bytes(b"caf\xe9.png"));
    std::fs::copy(&good, &odd).expect("copy to a non-UTF-8 name");
    let (code, stderr) = run_cli_batch(&[&odd], dir.path(), &["-m", "silueta"], std::time::Duration::from_secs(120));
    if ["ONNX Runtime not found", "is not installed"].iter().any(|s| stderr.contains(s)) {
        eprintln!("[cli_non_utf8] SKIP: prerequisite not available");
        return;
    }
    assert_eq!(code, Some(0), "stderr:\n{stderr}");
    assert_output_decodes(&dir.path().join(std::ffi::OsStr::from_bytes(b"caf\xe9.prunr.png")), "non-UTF-8 name");
}

/// A batch leaves nothing in its IPC folder, which on Linux is RAM
/// (`/dev/shm`): the worker writes each run's tensors for a parent that
/// re-runs from them, and the CLI, which never does, deletes them.
#[cfg(target_os = "linux")]
#[test]
fn cli_batch_leaves_no_ipc_files() {
    let dir = tempdir().expect("tempdir");
    let good = write_fixture_png(dir.path());
    let second = dir.path().join("second.png");
    std::fs::copy(&good, &second).expect("copy fixture");
    for extra in [&["-m", "silueta"][..], &["-m", "silueta", "--lines"]] {
        let child = Command::new(CLI_BIN)
            .args([&good, &second]).arg("-o").arg(dir.path()).arg("-f").args(extra)
            .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped())
            .spawn().expect("spawn prunr binary");
        let pid = child.id();
        let out = child.wait_with_output().expect("wait on prunr");
        let stderr = String::from_utf8_lossy(&out.stderr);
        if ["ONNX Runtime not found", "is not installed"].iter().any(|s| stderr.contains(s)) {
            eprintln!("[cli_ipc_files] SKIP: prerequisite not available");
            return;
        }
        assert!(out.status.success(), "{extra:?}: stderr:\n{stderr}");
        let ipc = std::path::PathBuf::from(format!("/dev/shm/prunr-ipc-{pid}"));
        let left: Vec<_> = std::fs::read_dir(&ipc).map(|d| d.flatten().map(|e| e.file_name()).collect()).unwrap_or_default();
        assert!(left.is_empty(), "{extra:?}: left in {}: {left:?}", ipc.display());
    }
}

/// A worker started by another version of the app (an update replaced
/// the binary while the app ran) stops at once with its own exit code,
/// before it reads a message it might not understand.
#[test]
fn a_worker_of_another_version_stops_at_once() {
    for args in [&["--worker", "--worker-of", "0.0.0-another"][..], &["--worker"]] {
        let out = Command::new(CLI_BIN).args(args).stdin(std::process::Stdio::null()).output().expect("spawn prunr binary");
        assert_eq!(out.status.code(), Some(75), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
}
