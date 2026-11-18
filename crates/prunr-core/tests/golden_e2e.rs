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
//! rejected — perceptual metric calibrated for "does it still look like a
//! chair?", not for "did the math change?". Max channel diff is the honest
//! signal.
//!
//! ## Fixture roots
//!
//! Two roots are scanned in order, both with the same per-fixture layout:
//!
//! - `tests/golden_data/e2e/<id>/` — committed synthetic fixtures, small,
//!   procedurally generated for past-bug categories. Public-repo safe.
//! - `tests/golden_data_local/e2e/<id>/` — gitignored, dev-machine-only.
//!   Drop a `source.png` here and `BOOTSTRAP_GOLDEN` populates `recipe.json`
//!   while `UPDATE_GOLDEN` populates `expected.png`. Personal photos and
//!   licensed stock — anything that can't ship to the public repo — lives here.
//!
//! ## Per-fixture layout
//!
//!   `source.png`    — RGBA source image
//!   `recipe.json`   — `{ "model": <ModelKind>, "mask_settings": <serde> }`
//!   `expected.png`  — reference output
//!
//! ## Modes
//!
//!   plain                 → verify
//!   `UPDATE_GOLDEN=1`     → run inference, write `expected.png`
//!   `BOOTSTRAP_GOLDEN=1`  → procedurally generate canary `source.png` +
//!                           `recipe.json` (skips fixtures that already
//!                           exist); also fills `recipe.json` defaults for
//!                           any local fixture that has `source.png` but
//!                           lacks `recipe.json`.

mod test_common;

use image::{ImageBuffer, Rgba, RgbaImage};
use prunr_core::{
    process_image_from_decoded, MaskSettings, ModelKind, OrtEngine,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use test_common::{
    assert_golden_fixtures, golden_fixtures, list_fixture_dirs, match_or_update,
    render_synthetic_source, skip_if_no_ort, synthetic_specs, GoldenRun,
};

const FIXTURE_ROOT_COMMITTED: &str = "tests/golden_data/e2e";
const FIXTURE_ROOT_LOCAL: &str = "tests/golden_data_local/e2e";

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
    if skip_if_no_ort("golden_e2e") {
        return;
    }

    let run = GoldenRun::new(FIXTURE_ROOT_COMMITTED, FIXTURE_ROOT_LOCAL);
    if run.bootstrap {
        bootstrap_committed(&run.committed);
        bootstrap_recipes_for_local(&run.local);
    }

    let fixtures = golden_fixtures("e2e", &run.committed, &run.local);

    // One engine for all Silueta fixtures; reuse to amortize the ~1s
    // session-init across iterations. Fixtures with a different model in
    // their recipe will fail the per-fixture model check below.
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta) failed");

    assert_golden_fixtures("e2e", &fixtures, |f| run_fixture(f, &engine, run.update));
}

fn run_fixture(dir: &Path, engine: &OrtEngine, update: bool) -> Result<(), String> {
    let source_path = dir.join("source.png");
    let recipe_path = dir.join("recipe.json");
    let expected_path = dir.join("expected.png");

    for (name, p) in [("source.png", &source_path), ("recipe.json", &recipe_path)] {
        if !p.exists() {
            return Err(format!(
                "missing input {name} (run BOOTSTRAP_GOLDEN=1 to generate canary fixtures \
                 or to fill defaults for a local-fixture source)"
            ));
        }
    }

    let recipe_bytes = fs::read(&recipe_path).map_err(|e| format!("read recipe: {e}"))?;
    let recipe: Recipe = serde_json::from_slice(&recipe_bytes)
        .map_err(|e| format!("recipe parse: {e}"))?;

    if recipe.model != engine.model_kind() {
        return Err(format!(
            "fixture model {:?} does not match engine model {:?} — multi-model fixtures need \
             separate engines",
            recipe.model,
            engine.model_kind(),
        ));
    }

    let img = image::open(&source_path).map_err(|e| format!("source decode: {e}"))?;
    let result = process_image_from_decoded(
        &img,
        engine,
        &recipe.mask_settings,
        &prunr_core::Progress::none(),
    )
    .map_err(|e| format!("pipeline: {e:?}"))?;

    match_or_update(&result.rgba_image, &expected_path, update, MAX_DIFF)
}

// ---------- bootstrap: committed synthetic fixtures ----------

/// Generate committed fixture inputs procedurally. Idempotent — skips fixtures
/// whose `source.png` already exists. Each fixture targets a specific category
/// of past bug.
fn bootstrap_committed(root: &Path) {
    bootstrap_circle_silueta_e2e(root);
    bootstrap_synthetic_e2e_fixtures(root);
}

/// Existing canary: 256×256 dark background with a centered light circle
/// (anti-aliased boundary). Default `MaskSettings`.
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
    write_default_recipe(&dir.join("recipe.json"));
    eprintln!("bootstrapped e2e fixture: {}", dir.display());
}

/// Procedural synthetic fixtures targeting past-bug categories. Each fixture:
/// - Generates a small RGBA `source.png` from `test_common::synthetic_specs`
/// - Writes a default `recipe.json` (Silueta + default `MaskSettings`)
/// - Lets `UPDATE_GOLDEN=1` then run real Silueta to write `expected.png`
fn bootstrap_synthetic_e2e_fixtures(root: &Path) {
    for spec in synthetic_specs() {
        let dir = root.join(spec.id);
        if dir.join("source.png").exists() {
            continue;
        }
        fs::create_dir_all(&dir).expect("create synthetic fixture dir");
        let img = render_synthetic_source(spec);
        img.save(dir.join("source.png")).expect("write source.png");
        write_default_recipe(&dir.join("recipe.json"));
        eprintln!("bootstrapped synthetic e2e fixture: {}", dir.display());
    }
}

fn write_default_recipe(path: &Path) {
    let recipe = Recipe {
        model: ModelKind::Silueta,
        mask_settings: MaskSettings::default(),
    };
    fs::write(
        path,
        serde_json::to_vec_pretty(&recipe).expect("serialize recipe"),
    )
    .expect("write recipe.json");
}

// ---------- bootstrap: local fixtures ----------

/// For any local fixture dir that has `source.png` but is missing `recipe.json`,
/// write a default recipe (Silueta + default `MaskSettings`). Lets the user
/// drop a source.png into `golden_data_local/e2e/<id>/` and just run
/// BOOTSTRAP_GOLDEN to fill in defaults.
fn bootstrap_recipes_for_local(root: &Path) {
    for dir in list_fixture_dirs(root) {
        let source = dir.join("source.png");
        let recipe = dir.join("recipe.json");
        if source.is_file() && !recipe.exists() {
            write_default_recipe(&recipe);
            eprintln!("bootstrapped recipe.json for local fixture: {}", dir.display());
        }
    }
}

