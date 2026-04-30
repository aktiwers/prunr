//! Bit-exact regression suite for the postprocess pipeline.
//!
//! The contract: refactors of `postprocess.rs` / `guided_filter.rs` /
//! `inpaint_blend.rs` / `apply_mask` / `apply_fill_style` / `apply_bg_effect`
//! that are claimed math-neutral must produce byte-identical output for a
//! frozen tensor input. Any deviation fails the test with the first mismatching
//! pixel coordinate. Math-changing refactors regenerate `expected.png` via
//! `UPDATE_GOLDEN=1`; visually inspect the resulting binary diff (e.g. via
//! `cargo xtask render-golden-diff`) before committing.
//!
//! Layout (per fixture under `tests/golden_data/postprocess/<id>/`):
//!   `source.png`    — RGBA source image
//!   `tensor.bin`    — raw f32 LE bytes of the model output (`H*W*4` bytes)
//!   `sidecar.json`  — `{ "shape": [N,C,H,W], "model": <ModelKind>,
//!                        "mask_settings": <MaskSettings serde> }`
//!   `expected.png`  — bit-exact reference output
//!
//! Modes:
//!   plain                 → verify (panics on first pixel mismatch)
//!   `UPDATE_GOLDEN=1`     → regenerate `expected.png` from current code
//!   `BOOTSTRAP_GOLDEN=1`  → procedurally generate canary fixture inputs

use image::{ImageBuffer, Rgba, RgbaImage};
use prunr_core::{postprocess_from_flat, MaskSettings, ModelKind, PostprocessOpts};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}};

const FIXTURE_ROOT: &str = "tests/golden_data/postprocess";

#[derive(Serialize, Deserialize)]
struct Sidecar {
    /// `[N, C, H, W]` of the `tensor.bin` payload.
    shape: [usize; 4],
    model: ModelKind,
    mask_settings: MaskSettings,
}

#[test]
fn golden_postprocess() {
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
            "no postprocess fixtures under {}.\n\
             Run with BOOTSTRAP_GOLDEN=1 to generate canary fixtures.",
            root.display(),
        );
    }

    let mut failures = Vec::new();
    for fixture in &fixtures {
        if let Err(e) = run_fixture(fixture, update) {
            let id = fixture.file_name().unwrap().to_string_lossy().into_owned();
            failures.push(format!("[{id}] {e}"));
        }
    }

    if !failures.is_empty() {
        panic!("postprocess golden mismatches:\n{}", failures.join("\n"));
    }
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

fn run_fixture(dir: &Path, update: bool) -> Result<(), String> {
    let source_path = dir.join("source.png");
    let tensor_path = dir.join("tensor.bin");
    let sidecar_path = dir.join("sidecar.json");
    let expected_path = dir.join("expected.png");

    for (name, p) in [
        ("source.png", &source_path),
        ("tensor.bin", &tensor_path),
        ("sidecar.json", &sidecar_path),
    ] {
        if !p.exists() {
            return Err(format!(
                "missing input {name} (run BOOTSTRAP_GOLDEN=1 to generate canary fixtures)"
            ));
        }
    }

    let sidecar_bytes = fs::read(&sidecar_path).map_err(|e| format!("read sidecar: {e}"))?;
    let sidecar: Sidecar = serde_json::from_slice(&sidecar_bytes)
        .map_err(|e| format!("sidecar parse: {e}"))?;

    let source = image::open(&source_path).map_err(|e| format!("source decode: {e}"))?;
    let tensor_bytes = fs::read(&tensor_path).map_err(|e| format!("read tensor: {e}"))?;
    let tensor = bytes_to_f32s_le(&tensor_bytes);

    let [_, _, h, w] = sidecar.shape;
    if tensor.len() != h * w {
        return Err(format!(
            "tensor.bin length {} != H*W ({h} * {w} = {})",
            tensor.len(),
            h * w,
        ));
    }

    let opts = PostprocessOpts::new(&sidecar.mask_settings, sidecar.model);
    let actual = postprocess_from_flat(&tensor, h, w, &source, &opts)
        .map_err(|e| format!("postprocess: {e:?}"))?;

    if update || !expected_path.exists() {
        actual.save(&expected_path).map_err(|e| format!("write expected: {e}"))?;
        // When update is set, treat as pass; the reviewer compares the diff.
        // When expected.png was missing without update, this is the first run
        // — pass once so the file lands; reruns enforce.
        return Ok(());
    }

    let expected = image::open(&expected_path)
        .map_err(|e| format!("expected decode: {e}"))?
        .to_rgba8();

    if actual.dimensions() != expected.dimensions() {
        return Err(format!(
            "dimension mismatch: actual {:?} vs expected {:?}",
            actual.dimensions(),
            expected.dimensions(),
        ));
    }

    let aw = actual.width() as usize;
    for (i, (a, e)) in actual.pixels().zip(expected.pixels()).enumerate() {
        if a != e {
            let x = i % aw;
            let y = i / aw;
            return Err(format!(
                "pixel mismatch at ({x}, {y}): actual rgba{:?} vs expected rgba{:?}",
                a.0, e.0,
            ));
        }
    }

    Ok(())
}

fn bytes_to_f32s_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32s_to_bytes_le(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Generate canary fixture inputs procedurally. Idempotent — skips fixtures
/// whose source already exists. Each canary is small (256² → ~256 KB tensor +
/// ~50 KB PNGs), checked into git as fixture data.
fn bootstrap_canaries(root: &Path) {
    bootstrap_circle_silueta_basic(root);
}

/// Canary: 256×256 RGBA gradient + a centered circular mask tensor.
///
/// Stresses three arms of the postprocess math at once:
/// - Inner saturation (r < 60: tensor = 1.0 → opaque alpha)
/// - Linear falloff (60 ≤ r ≤ 120: tensor = 1 - (r-60)/60 → smooth alpha gradient)
/// - Outer saturation (r > 120: tensor = 0 → fully transparent)
///
/// Default `MaskSettings` (no threshold, gamma 1.0, no edge refinement) keeps
/// the postprocess path on the bit-exact-deterministic branch.
fn bootstrap_circle_silueta_basic(root: &Path) {
    let dir = root.join("circle_silueta_basic");
    if dir.join("source.png").exists() {
        return;
    }
    fs::create_dir_all(&dir).expect("create canary dir");

    let (w, h) = (256u32, 256u32);

    let mut source: RgbaImage = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / (w - 1)) as u8;
            let g = (y * 255 / (h - 1)) as u8;
            let b = ((x + y) * 255 / (w + h - 2)) as u8;
            source.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    source.save(dir.join("source.png")).expect("write source.png");

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let mut tensor = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let v = if dist < 60.0 {
                1.0
            } else if dist > 120.0 {
                0.0
            } else {
                1.0 - (dist - 60.0) / 60.0
            };
            tensor[(y * w + x) as usize] = v;
        }
    }
    fs::write(dir.join("tensor.bin"), f32s_to_bytes_le(&tensor))
        .expect("write tensor.bin");

    let sidecar = Sidecar {
        shape: [1, 1, h as usize, w as usize],
        model: ModelKind::Silueta,
        mask_settings: MaskSettings::default(),
    };
    fs::write(
        dir.join("sidecar.json"),
        serde_json::to_vec_pretty(&sidecar).expect("serialize sidecar"),
    )
    .expect("write sidecar.json");

    eprintln!("bootstrapped fixture: {}", dir.display());
}
