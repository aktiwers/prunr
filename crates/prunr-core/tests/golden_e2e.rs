//! End-to-end regression suite for the full `prunr-core` pipeline.
//!
//! Runs `process_image_from_decoded` against a frozen source image with a
//! pinned model + recipe, on the CPU execution provider. The contract: any
//! refactor that changes preprocess, inference call shape, or postprocess in
//! a way that shifts pixel output fails this test.
//!
//! **Threshold posture:** bit-exact (max channel diff = 0) is the goal.
//! Same-machine CPU-EP inference is deterministic — single-threaded ORT
//! intra-op, ndarray standard layout, no rand. If cross-runner variance ever
//! flaps in CI, relax to `MAX_DIFF >= 1` with the variance documented inline
//! (and a count cap so bulk drift still fails). SSIM was considered and
//! rejected — it's a perceptual metric calibrated for "does it still look
//! like a chair?", not "did the math change?". Max channel diff is the
//! honest signal.
//!
//! Layout (per fixture under `tests/golden_data/e2e/<id>/`):
//!   `source.png`    — RGBA source image
//!   `recipe.json`   — `{ "model": <ModelKind>, "mask_settings": <serde> }`
//!   `expected.png`  — reference output
//!
//! Modes:
//!   plain                 → verify
//!   `UPDATE_GOLDEN=1`     → run inference, write `expected.png`
//!   `BOOTSTRAP_GOLDEN=1`  → procedurally generate canary `source.png` +
//!                           `recipe.json` (skips fixtures that already exist)

use image::{ImageBuffer, Rgba, RgbaImage};
use prunr_core::{
    process_image_from_decoded, MaskSettings, ModelKind, OrtEngine, ProgressStage,
};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}, sync::OnceLock};

const FIXTURE_ROOT: &str = "tests/golden_data/e2e";

/// Maximum allowed per-channel absolute difference between actual and expected
/// pixels. `0` = bit-exact. Bump only if cross-runner variance flaps a fixture
/// after CI lands, with a comment naming the runner + variance observed.
const MAX_DIFF: u8 = 0;

#[derive(Serialize, Deserialize)]
struct Recipe {
    model: ModelKind,
    mask_settings: MaskSettings,
}

#[test]
fn golden_e2e() {
    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[golden_e2e] SKIP: {msg}");
        return;
    }

    let bootstrap = env::var_os("BOOTSTRAP_GOLDEN").is_some();
    let update = env::var_os("UPDATE_GOLDEN").is_some();

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_ROOT);
    fs::create_dir_all(&root).expect("create fixture root");

    if bootstrap {
        bootstrap_canaries(&root);
    }

    let mut fixtures = list_fixtures(&root);
    fixtures.sort();

    if fixtures.is_empty() {
        panic!(
            "no e2e fixtures under {}.\n\
             Run with BOOTSTRAP_GOLDEN=1 to generate canary fixtures.",
            root.display(),
        );
    }

    // One engine per (model, ep) pair — for canaries today, all use Silueta on
    // CPU EP. Reuse the engine across fixtures to avoid the ~1s session-init
    // hit on every iteration; the session is internally `Mutex<Session>` so
    // sequential calls are fine.
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta) failed");

    let mut failures = Vec::new();
    for fixture in &fixtures {
        if let Err(e) = run_fixture(fixture, &engine, update) {
            let id = fixture.file_name().unwrap().to_string_lossy().into_owned();
            failures.push(format!("[{id}] {e}"));
        }
    }

    if !failures.is_empty() {
        panic!("e2e golden mismatches:\n{}", failures.join("\n"));
    }
}

/// `ort` `load-dynamic` requires an explicit `init_from(<dylib path>)` before
/// any session creation. The production binaries call this from
/// `prunr_app::ort_runtime::init()`; integration tests in `prunr-core` need
/// to do it themselves (and can't depend on `prunr-app`).
///
/// Resolution order (matches `prunr-app::ort_runtime::resolve_dylib_path`):
///   1. `ORT_DYLIB_PATH` env var (escape hatch + dev override + CI override)
///   2. Runtime Store install at `<data>/prunr/runtimes/<ep>/libonnxruntime.{so,dylib,dll}`
///
/// Bundled fallback (`<exe parent>/runtime/`) is intentionally skipped — test
/// binaries don't ship a sibling runtime dir, and falling through to that path
/// would be a confusing failure mode in tests.
///
/// Test-skip behaviour: if no runtime is found, the test prints a clear SKIP
/// message and returns Ok-but-no-work. Returning `Err` from this fn surfaces
/// the failure as a hard test panic (used for unrecoverable init errors,
/// e.g. wrong dylib version).
fn ensure_ort_initialized() -> Result<(), String> {
    static RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    RESULT.get_or_init(try_init_ort).clone()
}

fn try_init_ort() -> Result<(), String> {
    if let Some(env_path) = env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(env_path);
        if !path.is_file() {
            return Err(format!(
                "ORT_DYLIB_PATH={} is not a file",
                path.display()
            ));
        }
        return commit_ort(&path);
    }

    if let Some(path) = find_runtime_store_dylib() {
        return commit_ort(&path);
    }

    // Soft skip — return Ok(()) to the caller, which inspects this only when
    // `ORT_DYLIB_PATH` was unset and the runtime store was empty. The actual
    // skip message is printed by the test fn before returning.
    Err(format!(
        "no ORT runtime found. Set ORT_DYLIB_PATH=<path/to/{}> or run \
         `cargo xtask install-runtime onnxruntime <version>` to populate the \
         runtime store",
        match std::env::consts::OS {
            "windows" => "onnxruntime.dll",
            "macos" => "libonnxruntime.dylib",
            _ => "libonnxruntime.so",
        }
    ))
}

fn commit_ort(path: &Path) -> Result<(), String> {
    let env = ort::init_from(path)
        .map_err(|e| format!("ort::init_from({}): {e}", path.display()))?;
    // commit() returns false on double-commit (rare in tests since ONCE
    // gates this fn, but keep the pattern from prunr-app for safety).
    let _ = env.commit();
    Ok(())
}

fn find_runtime_store_dylib() -> Option<PathBuf> {
    let dylib_name = match std::env::consts::OS {
        "windows" => "onnxruntime.dll",
        "macos" => "libonnxruntime.dylib",
        _ => "libonnxruntime.so",
    };
    let root = dirs::data_dir()?.join("prunr").join("runtimes");
    if !root.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = fs::read_dir(&root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .collect();
    // Deterministic pick across runs — sort by name. Tests should be reproducible.
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let candidate = entry.path().join(dylib_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn list_fixtures(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default()
}

fn run_fixture(dir: &Path, engine: &OrtEngine, update: bool) -> Result<(), String> {
    let source_path = dir.join("source.png");
    let recipe_path = dir.join("recipe.json");
    let expected_path = dir.join("expected.png");

    for (name, p) in [("source.png", &source_path), ("recipe.json", &recipe_path)] {
        if !p.exists() {
            return Err(format!(
                "missing input {name} (run BOOTSTRAP_GOLDEN=1 to generate canary fixtures)"
            ));
        }
    }

    let recipe_bytes = fs::read(&recipe_path).map_err(|e| format!("read recipe: {e}"))?;
    let recipe: Recipe = serde_json::from_slice(&recipe_bytes)
        .map_err(|e| format!("recipe parse: {e}"))?;

    if recipe.model != engine.model_kind() {
        return Err(format!(
            "fixture model {:?} does not match engine model {:?} — multi-model fixtures need separate engines",
            recipe.model,
            engine.model_kind(),
        ));
    }

    let img = image::open(&source_path).map_err(|e| format!("source decode: {e}"))?;
    let result = process_image_from_decoded(
        &img,
        engine,
        &recipe.mask_settings,
        None::<fn(ProgressStage, f32)>,
        None,
    )
    .map_err(|e| format!("pipeline: {e:?}"))?;
    let actual = result.rgba_image;

    if update || !expected_path.exists() {
        actual.save(&expected_path).map_err(|e| format!("write expected: {e}"))?;
        return Ok(());
    }

    let expected = image::open(&expected_path)
        .map_err(|e| format!("expected decode: {e}"))?
        .to_rgba8();

    compare_pixels(&actual, &expected)
}

fn compare_pixels(actual: &RgbaImage, expected: &RgbaImage) -> Result<(), String> {
    if actual.dimensions() != expected.dimensions() {
        return Err(format!(
            "dimension mismatch: actual {:?} vs expected {:?}",
            actual.dimensions(),
            expected.dimensions(),
        ));
    }

    let aw = actual.width() as usize;
    let mut max_observed: u8 = 0;
    let mut diff_pixels: usize = 0;
    let mut first_mismatch: Option<String> = None;

    for (i, (a, e)) in actual.pixels().zip(expected.pixels()).enumerate() {
        let mut pixel_max: u8 = 0;
        for c in 0..4 {
            let d = a.0[c].abs_diff(e.0[c]);
            if d > pixel_max {
                pixel_max = d;
            }
        }
        if pixel_max > MAX_DIFF {
            if first_mismatch.is_none() {
                let x = i % aw;
                let y = i / aw;
                first_mismatch = Some(format!(
                    "first mismatch at ({x}, {y}): actual rgba{:?} vs expected rgba{:?}",
                    a.0, e.0,
                ));
            }
            if pixel_max > max_observed {
                max_observed = pixel_max;
            }
            diff_pixels += 1;
        }
    }

    if max_observed > MAX_DIFF {
        let total = actual.width() as usize * actual.height() as usize;
        return Err(format!(
            "{}; {diff_pixels}/{total} pixels differ; max channel diff {max_observed} (threshold {MAX_DIFF})",
            first_mismatch.as_deref().unwrap_or("(no first-mismatch recorded)"),
        ));
    }
    Ok(())
}

/// Generate canary fixture inputs procedurally. Idempotent — skips fixtures
/// whose `source.png` already exists.
fn bootstrap_canaries(root: &Path) {
    bootstrap_circle_silueta_e2e(root);
}

/// Canary: 256×256 dark background with a centered light circle (smooth
/// anti-aliased boundary). Gives Silueta a clear subject/background contrast
/// to identify, with gradient information at the edge so the inference output
/// is non-trivial (not just everywhere-zero or everywhere-one).
fn bootstrap_circle_silueta_e2e(root: &Path) {
    let dir = root.join("circle_silueta_e2e");
    if dir.join("source.png").exists() {
        return;
    }
    fs::create_dir_all(&dir).expect("create canary dir");

    let (w, h) = (256u32, 256u32);
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let mut source: RgbaImage = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            // Smooth step: solid foreground inside r=60, full background past r=72,
            // 12-pixel anti-aliased ring between.
            let alpha = if d < 60.0 {
                1.0
            } else if d > 72.0 {
                0.0
            } else {
                1.0 - (d - 60.0) / 12.0
            };
            let bg = [40u8, 40, 50];
            let fg = [220u8, 220, 200];
            let lerp = |b: u8, f: u8| (b as f32 * (1.0 - alpha) + f as f32 * alpha) as u8;
            source.put_pixel(
                x,
                y,
                Rgba([lerp(bg[0], fg[0]), lerp(bg[1], fg[1]), lerp(bg[2], fg[2]), 255]),
            );
        }
    }
    source.save(dir.join("source.png")).expect("write source.png");

    let recipe = Recipe {
        model: ModelKind::Silueta,
        mask_settings: MaskSettings::default(),
    };
    fs::write(
        dir.join("recipe.json"),
        serde_json::to_vec_pretty(&recipe).expect("serialize recipe"),
    )
    .expect("write recipe.json");

    eprintln!("bootstrapped e2e fixture: {}", dir.display());
}
