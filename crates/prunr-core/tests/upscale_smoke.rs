//! End-to-end smoke tests for the upscale module.
//!
//! These tests require the OnDemand ONNX files to be installed in the
//! user data dir. On a fresh machine the tests skip-with-message rather
//! than fail — install via Model Store to exercise them.
//!
//! Run with:
//!   cargo test -p prunr-core --test upscale_smoke -- --nocapture

mod test_common;

use image::{Rgba, RgbaImage};
use prunr_core::upscale::upscale_rgba;
use prunr_models::{is_available, ModelId};
use test_common::skip_if_no_ort;

fn gradient_test_image(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let r = ((x * 255) / w.max(1)) as u8;
        let g = ((y * 255) / h.max(1)) as u8;
        Rgba([r, g, 128, 255])
    })
}

fn skip_if_unavailable(id: ModelId) -> bool {
    if !is_available(id) {
        eprintln!("SKIP: {id:?} not installed — run the Model Store download");
        return true;
    }
    false
}

#[test]
fn real_esrgan_and_nomos8k_upscale_128x128_to_512x512() {
    if skip_if_no_ort("upscale_128x128") {
        return;
    }
    for id in [ModelId::RealEsrganX4Plus, ModelId::Nomos8kSchatL] {
        if skip_if_unavailable(id) {
            continue;
        }
        let input = gradient_test_image(128, 128);
        let out = upscale_rgba(&input, id, 4, 2, |_, _| {}, None)
            .expect("upscale_rgba should succeed");
        assert_eq!(out.width(), 512, "{id:?} width");
        assert_eq!(out.height(), 512, "{id:?} height");
    }
}

#[test]
fn nomos_8k_schatl_upscales_non_multiple_of_16() {
    if skip_if_no_ort("upscale_non_multiple_of_16") {
        return;
    }
    if skip_if_unavailable(ModelId::Nomos8kSchatL) {
        return;
    }
    let input = gradient_test_image(120, 120); // NOT a multiple of 16
    let out = upscale_rgba(&input, ModelId::Nomos8kSchatL, 4, 2, |_, _| {}, None)
        .expect("upscale_rgba should pad+trim for non-multiple-of-16 input");
    assert_eq!(out.width(), 480); // 120 * 4
    assert_eq!(out.height(), 480);
}

#[test]
fn scale_2_downscales_to_half_of_native() {
    if skip_if_no_ort("upscale_scale_2") {
        return;
    }
    if skip_if_unavailable(ModelId::RealEsrganX4Plus) {
        return;
    }
    let input = gradient_test_image(256, 256);
    let out = upscale_rgba(&input, ModelId::RealEsrganX4Plus, 2, 2, |_, _| {}, None)
        .expect("upscale_rgba scale=2 should succeed");
    // upscale_rgba always runs the model at 4×, then halves: 256 → 1024 → 512.
    assert_eq!(out.width(), 256 * 4 / 2);
    assert_eq!(out.height(), 256 * 4 / 2);

    // Alpha must propagate through the scale=2 path; a fully-opaque input
    // must produce a fully-opaque output. Regression guard for the alpha
    // companion (`upscale_alpha_lanczos3`) used in the scale=2 branch.
    assert!(
        out.pixels().all(|p| p.0[3] == 255),
        "scale=2 alpha channel must remain fully opaque for opaque input"
    );
}
