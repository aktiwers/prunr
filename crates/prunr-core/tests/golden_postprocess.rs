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
//! ## Fixture roots
//!
//! Two roots are scanned in order, both with the same per-fixture layout:
//!
//! - `tests/golden_data/postprocess/<id>/` — committed synthetic fixtures.
//!   Each pairs a procedurally-generated source with a procedurally-generated
//!   tensor (a centered circle mask at 320×320 = Silueta output shape).
//!   No ORT needed at verify time. Public-repo safe.
//! - `tests/golden_data_local/postprocess/<id>/` — gitignored, dev-machine
//!   only. Drop a `source.png` here; BOOTSTRAP_GOLDEN runs Silueta to
//!   produce a real `tensor.bin` + `sidecar.json`. Personal photos /
//!   licensed stock / anything that can't ship to the public repo.
//!
//! ## Per-fixture layout
//!
//!   `source.png`    — RGBA source image
//!   `tensor.bin`    — raw f32 LE bytes of the model output (`H*W*4` bytes)
//!   `sidecar.json`  — `{ "shape": [N,C,H,W], "model": <ModelKind>,
//!                        "mask_settings": <MaskSettings serde> }`
//!   `expected.png`  — bit-exact reference output
//!
//! ## Modes
//!
//!   plain                 → verify (panics on first pixel mismatch)
//!   `UPDATE_GOLDEN=1`     → regenerate `expected.png` from current code
//!   `BOOTSTRAP_GOLDEN=1`  → procedurally generate committed fixture inputs
//!                           (synthetic source + synthetic tensor) AND run
//!                           Silueta on local fixtures missing tensor.bin

mod test_common;

use image::RgbaImage;
use prunr_core::{
    infer_only, postprocess_from_flat, MaskSettings, ModelKind, OrtEngine, PostprocessOpts,
    ProgressStage,
};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}};
use test_common::{ensure_ort_initialized, render_synthetic_source, synthetic_specs};

const FIXTURE_ROOT_COMMITTED: &str = "tests/golden_data/postprocess";
const FIXTURE_ROOT_LOCAL: &str = "tests/golden_data_local/postprocess";

/// Fixed tensor dimensions for synthetic fixtures — matches Silueta's output
/// shape so the postprocess upscaling code runs the same way it does on real
/// inference outputs.
const SYNTH_TENSOR_W: usize = 320;
const SYNTH_TENSOR_H: usize = 320;

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

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let committed_root = manifest_dir.join(FIXTURE_ROOT_COMMITTED);
    let local_root = manifest_dir.join(FIXTURE_ROOT_LOCAL);
    fs::create_dir_all(&committed_root).expect("create committed fixture root");

    if bootstrap {
        bootstrap_committed(&committed_root);
        bootstrap_local_real_fixtures(&local_root);
    }

    let mut fixtures = Vec::new();
    fixtures.extend(list_fixtures(&committed_root));
    if local_root.is_dir() {
        fixtures.extend(list_fixtures(&local_root));
    }
    fixtures.sort();

    if fixtures.is_empty() {
        panic!(
            "no postprocess fixtures under {} or {}.\n\
             Run with BOOTSTRAP_GOLDEN=1 to generate canary fixtures.",
            committed_root.display(),
            local_root.display(),
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
                "missing input {name} (run BOOTSTRAP_GOLDEN=1 to generate canary fixtures \
                 or to populate a local-fixture tensor)"
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

// ---------- bootstrap: committed synthetic fixtures ----------

/// Generate committed fixture inputs: synthetic source + synthetic tensor +
/// sidecar. Idempotent — skips fixtures whose source.png already exists.
/// No ORT needed (synthetic tensor).
fn bootstrap_committed(root: &Path) {
    bootstrap_circle_silueta_basic(root);
    bootstrap_synthetic_postprocess_fixtures(root);
}

/// Existing canary: 256² gradient source + centered circle tensor.
fn bootstrap_circle_silueta_basic(root: &Path) {
    let dir = root.join("circle_silueta_basic");
    if dir.join("source.png").exists() {
        return;
    }
    fs::create_dir_all(&dir).expect("create canary dir");

    use image::{ImageBuffer, Rgba};
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

/// Procedural synthetic fixtures using shared `synthetic_specs()` for source
/// generation, paired with a `synthesize_circle_tensor` (Silueta-shape, 320²
/// centered circle). Each fixture catches a different combination of
/// source-color × source-shape × postprocess-math-arms.
fn bootstrap_synthetic_postprocess_fixtures(root: &Path) {
    for spec in synthetic_specs() {
        let dir = root.join(spec.id);
        if dir.join("source.png").exists() {
            continue;
        }
        fs::create_dir_all(&dir).expect("create synthetic fixture dir");

        let img = render_synthetic_source(spec);
        img.save(dir.join("source.png")).expect("write source.png");

        let tensor = synthesize_circle_tensor(SYNTH_TENSOR_W, SYNTH_TENSOR_H);
        fs::write(dir.join("tensor.bin"), f32s_to_bytes_le(&tensor))
            .expect("write tensor.bin");

        let sidecar = Sidecar {
            shape: [1, 1, SYNTH_TENSOR_H, SYNTH_TENSOR_W],
            model: ModelKind::Silueta,
            mask_settings: MaskSettings::default(),
        };
        fs::write(
            dir.join("sidecar.json"),
            serde_json::to_vec_pretty(&sidecar).expect("serialize sidecar"),
        )
        .expect("write sidecar.json");

        eprintln!("bootstrapped synthetic postprocess fixture: {}", dir.display());
    }
}

/// Centered circle mask at the given tensor resolution. Three arms:
/// - inner (r < 0.4 * min_dim / 2): saturate to 1.0
/// - falloff (0.4..0.5 * min_dim / 2): linear taper
/// - outer (r > 0.5 * min_dim / 2): saturate to 0.0
fn synthesize_circle_tensor(w: usize, h: usize) -> Vec<f32> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let min_dim = w.min(h) as f32;
    let r_inner = 0.4 * min_dim / 2.0;
    let r_outer = 0.5 * min_dim / 2.0;
    let mut tensor = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let v = if d < r_inner {
                1.0
            } else if d > r_outer {
                0.0
            } else {
                1.0 - (d - r_inner) / (r_outer - r_inner)
            };
            tensor[y * w + x] = v;
        }
    }
    tensor
}

// ---------- bootstrap: local real fixtures (Silueta inference) ----------

/// For each local fixture dir with `source.png` but missing `tensor.bin` /
/// `sidecar.json`, run Silueta CPU inference and write both. Idempotent.
/// Local fixtures are never committed — see `.gitignore`.
fn bootstrap_local_real_fixtures(root: &Path) {
    if !root.is_dir() {
        return;
    }
    let dirs: Vec<PathBuf> = fs::read_dir(root)
        .ok()
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .filter(|p| p.join("source.png").is_file())
                .filter(|p| !p.join("tensor.bin").is_file() || !p.join("sidecar.json").is_file())
                .collect()
        })
        .unwrap_or_default();

    if dirs.is_empty() {
        return;
    }

    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[golden_postprocess::bootstrap_local] SKIP: {msg}");
        return;
    }

    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta) failed in postprocess bootstrap");

    for dir in dirs {
        let source_path = dir.join("source.png");
        let img = match image::open(&source_path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("[bootstrap-local] {}: source decode failed: {e}", dir.display());
                continue;
            }
        };

        let result = match infer_only(&img, &engine, None::<fn(ProgressStage, f32)>, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[bootstrap-local] {}: inference failed: {e:?}", dir.display());
                continue;
            }
        };

        fs::write(dir.join("tensor.bin"), f32s_to_bytes_le(&result.tensor_data))
            .expect("write tensor.bin");

        let sidecar = Sidecar {
            shape: [1, 1, result.tensor_height, result.tensor_width],
            model: ModelKind::Silueta,
            mask_settings: MaskSettings::default(),
        };
        fs::write(
            dir.join("sidecar.json"),
            serde_json::to_vec_pretty(&sidecar).expect("serialize sidecar"),
        )
        .expect("write sidecar.json");

        eprintln!("bootstrapped local real fixture: {}", dir.display());
    }
}
