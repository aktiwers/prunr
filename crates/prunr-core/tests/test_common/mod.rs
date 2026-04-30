//! Shared test utilities for prunr-core integration tests.
//!
//! `ensure_ort_initialized` — `ort` `load-dynamic` requires an explicit
//! `init_from(<dylib path>)` before any session creation. Production binaries
//! call this from `prunr_app::ort_runtime::init()`; integration tests in
//! `prunr-core` need to do it themselves (and can't depend on `prunr-app`).
//!
//! Resolution order (matches `prunr-app::ort_runtime::resolve_dylib_path`):
//!   1. `ORT_DYLIB_PATH` env var (escape hatch + dev override + CI override)
//!   2. Runtime Store install at `<data>/prunr/runtimes/<ep>/libonnxruntime.{so,dylib,dll}`
//!
//! Bundled fallback (`<exe parent>/runtime/`) is intentionally skipped — test
//! binaries don't ship a sibling runtime dir, and falling through to that path
//! would be a confusing failure mode in tests.

#![allow(dead_code)] // Each test binary uses a subset; suppress per-binary warnings.

use image::{ImageBuffer, Rgba, RgbaImage};
use std::{env, fs, path::{Path, PathBuf}, sync::OnceLock};

/// Returns Ok(()) when ORT is committed and ready for `OrtEngine::new_*`.
/// Returns Err with a clear human message when no runtime can be found —
/// caller decides whether to skip the test or panic.
pub fn ensure_ort_initialized() -> Result<(), String> {
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

    Err(format!(
        "no ORT runtime found. Set ORT_DYLIB_PATH=<path/to/{}> or run \
         `cargo xtask install-runtime onnxruntime <version>` to populate the \
         runtime store",
        match env::consts::OS {
            "windows" => "onnxruntime.dll",
            "macos" => "libonnxruntime.dylib",
            _ => "libonnxruntime.so",
        }
    ))
}

fn commit_ort(path: &Path) -> Result<(), String> {
    let env = ort::init_from(path)
        .map_err(|e| format!("ort::init_from({}): {e}", path.display()))?;
    let _ = env.commit();
    Ok(())
}

fn find_runtime_store_dylib() -> Option<PathBuf> {
    let dylib_name = match env::consts::OS {
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
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let candidate = entry.path().join(dylib_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ---------- synthetic fixture catalog (shared by both golden harnesses) ----------

/// Spec for one synthetic fixture. Each id maps to a past-bug category from
/// `.planning/phases/20-golden-image-suite/PLAN.md`. The draw fn paints into
/// a pre-allocated RGBA buffer (zero-init = transparent black).
pub struct SyntheticSpec {
    pub id: &'static str,
    pub width: u32,
    pub height: u32,
    pub draw_source: fn(&mut RgbaImage),
}

/// Catalog of synthetic fixtures shared by both tiers. Adding an entry here
/// adds one fixture to BOTH the postprocess tier (with a synthetic tensor)
/// and the e2e tier (with real Silueta inference at update time).
pub fn synthetic_specs() -> &'static [SyntheticSpec] {
    &[
        SyntheticSpec { id: "block_color_synth",       width: 256, height: 256, draw_source: draw_block_color },
        SyntheticSpec { id: "thin_features_synth",     width: 256, height: 256, draw_source: draw_thin_features },
        SyntheticSpec { id: "subject_to_border_synth", width: 256, height: 256, draw_source: draw_subject_to_border },
        SyntheticSpec { id: "alpha_input_synth",       width: 256, height: 256, draw_source: draw_alpha_input },
        SyntheticSpec { id: "multi_subject_synth",     width: 256, height: 256, draw_source: draw_multi_subject },
        SyntheticSpec { id: "tiny_mask_huge_synth",    width: 512, height: 512, draw_source: draw_tiny_mask_huge },
        SyntheticSpec { id: "hard_edge_synth",         width: 256, height: 256, draw_source: draw_hard_edge },
    ]
}

/// Block-color subject: white bg + 2 solid-color rectangles + thin grid lines.
/// Mimics a UI screenshot — flat fills, sharp edges, thin row separators.
pub fn draw_block_color(img: &mut RgbaImage) {
    fill(img, [245, 245, 248, 255]);
    fill_rect(img, 30, 50, 90, 30, [80, 130, 200, 255]);
    fill_rect(img, 140, 50, 90, 30, [200, 80, 100, 255]);
    for y in (90..=200).step_by(20) {
        fill_rect(img, 20, y, 216, 1, [180, 180, 185, 255]);
    }
}

/// Thin features: white bg + 2-pixel-wide cross + small filled circle.
/// Tests bbox-crop and edge-feathering on 1-2 px features.
pub fn draw_thin_features(img: &mut RgbaImage) {
    fill(img, [255, 255, 255, 255]);
    let cx = (img.width() / 2) as i32;
    let cy = (img.height() / 2) as i32;
    fill_rect(img, cx as u32 - 1, 40, 2, 176, [20, 20, 20, 255]);
    fill_rect(img, 40, cy as u32 - 1, 176, 2, [20, 20, 20, 255]);
    draw_circle_filled(img, cx, cy, 12, [20, 20, 20, 255]);
}

/// Subject touching border: solid triangle clipping the right + bottom edges.
/// Tests boundary handling in mask refinement / bbox-clamp.
pub fn draw_subject_to_border(img: &mut RgbaImage) {
    fill(img, [60, 70, 90, 255]);
    let (w, h) = (img.width() as i32, img.height() as i32);
    for y in 0..h {
        for x in 0..w {
            if x + y > w {
                img.put_pixel(x as u32, y as u32, Rgba([220, 200, 160, 255]));
            }
        }
    }
}

/// Pre-existing alpha: solid shape on a fully transparent background.
/// Source has alpha=0 outside the shape; postprocess must handle this without
/// alpha-bleeding into the mask result.
pub fn draw_alpha_input(img: &mut RgbaImage) {
    let cx = (img.width() / 2) as i32;
    let cy = (img.height() / 2) as i32;
    draw_circle_filled(img, cx, cy, 60, [180, 90, 220, 255]);
}

/// Multi-subject: 2 separate shapes at distinct positions on a bg. Tests
/// composite math + multi-component mask handling.
pub fn draw_multi_subject(img: &mut RgbaImage) {
    fill(img, [50, 60, 70, 255]);
    draw_circle_filled(img, 80, 128, 40, [255, 200, 80, 255]);
    draw_circle_filled(img, 180, 128, 40, [80, 200, 255, 255]);
}

/// Tiny mask on huge image: 512² mostly-empty with a small (~60px) shape in
/// one corner. Tests bbox-crop's RAM-win path on small ROIs.
pub fn draw_tiny_mask_huge(img: &mut RgbaImage) {
    fill(img, [240, 240, 245, 255]);
    draw_circle_filled(img, 60, 60, 30, [40, 80, 120, 255]);
}

/// Hard edge: solid rectangle with no anti-aliasing. Tests hard-edge mask math.
pub fn draw_hard_edge(img: &mut RgbaImage) {
    fill(img, [100, 110, 120, 255]);
    fill_rect(img, 60, 60, 136, 136, [220, 80, 80, 255]);
}

// ---------- drawing primitives ----------

pub fn fill(img: &mut RgbaImage, color: [u8; 4]) {
    for p in img.pixels_mut() {
        *p = Rgba(color);
    }
}

pub fn fill_rect(img: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    let (img_w, img_h) = (img.width(), img.height());
    let xe = (x + w).min(img_w);
    let ye = (y + h).min(img_h);
    for py in y..ye {
        for px in x..xe {
            img.put_pixel(px, py, Rgba(color));
        }
    }
}

pub fn draw_circle_filled(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, color: [u8; 4]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let r2 = (r * r) as f32;
    for y in (cy - r).max(0)..(cy + r).min(h - 1) + 1 {
        for x in (cx - r).max(0)..(cx + r).min(w - 1) + 1 {
            let dx = (x - cx) as f32;
            let dy = (y - cy) as f32;
            if dx * dx + dy * dy <= r2 {
                img.put_pixel(x as u32, y as u32, Rgba(color));
            }
        }
    }
}

/// Helper for synthetic source generation: bootstrap an `RgbaImage` of the
/// given dimensions, paint it via the spec's `draw_source`, and return.
pub fn render_synthetic_source(spec: &SyntheticSpec) -> RgbaImage {
    let mut img: RgbaImage = ImageBuffer::new(spec.width, spec.height);
    (spec.draw_source)(&mut img);
    img
}
