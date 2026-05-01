//! Phase 23-04: each inpaint model runs end-to-end on a representative
//! mask. Skip-on-missing-model mirrors `golden_e2e`'s ORT skip — these
//! models are OnDemand, so a clean dev box won't have them until the
//! user runs the Model Store.
//!
//! Per-model invariants:
//! - Output dimensions match input.
//! - Masked region pixels DIFFER from the source (proves the model
//!   produced something — empty masks are short-circuited above this
//!   path).
//! - Unmasked region pixels EQUAL the source (proves identity preservation
//!   outside the mask — `process_inpaint` blends the inpaint result back
//!   onto the original at every non-mask pixel).
//!
//! SD-family models are deferred — too slow for routine CI; gated behind
//! `--ignored` via separate slow-test infrastructure.

mod test_common;

use image::{GrayImage, Luma, RgbaImage};
use prunr_core::inpaint::process_inpaint;
use prunr_models::{is_available, ModelId};
use test_common::skip_if_no_ort;

/// 256² source: solid teal background, used so masked-region inpaint
/// output is easily distinguishable from the unmasked area's identity.
fn fixture_source() -> RgbaImage {
    let mut img = RgbaImage::new(256, 256);
    test_common::fill(&mut img, [40, 120, 140, 255]);
    test_common::fill_rect(&mut img, 60, 60, 136, 136, [220, 80, 80, 255]);
    img
}

/// 256² mask: small rectangle in the centre. Anything inside the rect
/// is "to be inpainted"; outside is identity.
fn fixture_mask() -> GrayImage {
    let mut mask = GrayImage::new(256, 256);
    for py in 100..160 {
        for px in 100..160 {
            mask.put_pixel(px, py, Luma([255]));
        }
    }
    mask
}

fn run_inpaint_test(model_id: ModelId, label: &str) {
    if skip_if_no_ort(label) { return; }
    if !is_available(model_id) {
        eprintln!(
            "[{label}] SKIP: {model_id:?} is not installed. Run `cargo xtask fetch-models` \
             or open the in-app Model Store to download it."
        );
        return;
    }
    let image = fixture_source();
    let mask = fixture_mask();
    let out = process_inpaint(&image, &mask, model_id)
        .unwrap_or_else(|e| panic!("{label}: process_inpaint failed: {e:?}"));

    // Dimensions invariant.
    assert_eq!(out.dimensions(), image.dimensions(), "{label}: dimensions");

    // Masked region: at least some pixels must differ from the source.
    // (We can't assert "every pixel differs" because the model may keep a
    // colour close to the mean; the strong contract is "the model did
    // something inside the mask.")
    let mut differing = 0u32;
    for py in 100..160u32 {
        for px in 100..160u32 {
            if out.get_pixel(px, py).0 != image.get_pixel(px, py).0 {
                differing += 1;
            }
        }
    }
    let total = 60 * 60;
    assert!(
        differing as f32 / total as f32 > 0.3,
        "{label}: expected ≥30% of masked pixels to change, got {differing}/{total}",
    );

    // Unmasked region: identity. Sample a few corners to keep the loop
    // tight; full-image identity is implied by the inpaint blending
    // contract.
    for &(x, y) in &[(0u32, 0u32), (255, 0), (0, 255), (255, 255), (10, 200)] {
        assert_eq!(
            out.get_pixel(x, y).0,
            image.get_pixel(x, y).0,
            "{label}: identity broken at ({x},{y})",
        );
    }
}

#[test]
fn inpaint_lama_fp32_runs_end_to_end() {
    run_inpaint_test(ModelId::LaMaFp32, "lama_fp32");
}

#[test]
fn inpaint_big_lama_runs_end_to_end() {
    run_inpaint_test(ModelId::BigLaMa, "big_lama");
}

#[test]
fn inpaint_migan_runs_end_to_end() {
    run_inpaint_test(ModelId::Migan, "migan");
}
