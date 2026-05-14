//! End-to-end smoke tests for the upscale module.
//!
//! These tests REQUIRE the OnDemand ONNX files to be installed in the
//! user data dir (see PRECONDITIONS.md). On a fresh machine the tests
//! skip-with-message rather than fail — install via Model Store to
//! exercise them.
//!
//! Run with:
//!   cargo test -p prunr-core --test upscale_smoke -- --nocapture

use image::{Rgba, RgbaImage};
use prunr_core::upscale::upscale_rgba;
use prunr_models::{is_available, ModelId};

fn gradient_test_image(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let r = ((x * 255) / w.max(1)) as u8;
        let g = ((y * 255) / h.max(1)) as u8;
        let b = 128;
        let a = 255;
        Rgba([r, g, b, a])
    })
}

fn skip_if_unavailable(id: ModelId) -> bool {
    if !is_available(id) {
        eprintln!(
            "SKIP: {id:?} not installed — run Model Store download or see PRECONDITIONS.md"
        );
        return true;
    }
    false
}

#[test]
fn real_esrgan_x4plus_upscales_128x128_to_512x512() {
    if skip_if_unavailable(ModelId::RealEsrganX4Plus) {
        return;
    }
    let input = gradient_test_image(128, 128);
    let out = upscale_rgba(&input, ModelId::RealEsrganX4Plus, 4, 2, |_, _| {}, None)
        .expect("upscale_rgba should succeed");
    assert_eq!(out.width(), 512);
    assert_eq!(out.height(), 512);
}

#[test]
fn nomos_8k_schatl_upscales_128x128_to_512x512() {
    if skip_if_unavailable(ModelId::Nomos8kSchatL) {
        return;
    }
    let input = gradient_test_image(128, 128);
    let out = upscale_rgba(&input, ModelId::Nomos8kSchatL, 4, 2, |_, _| {}, None)
        .expect("upscale_rgba should succeed");
    assert_eq!(out.width(), 512);
    assert_eq!(out.height(), 512);
}

#[test]
fn nomos_8k_schatl_upscales_non_multiple_of_16() {
    if skip_if_unavailable(ModelId::Nomos8kSchatL) {
        return;
    }
    let input = gradient_test_image(120, 120); // NOT a multiple of 16
    let out = upscale_rgba(&input, ModelId::Nomos8kSchatL, 4, 2, |_, _| {}, None)
        .expect("upscale_rgba should pad+trim for non-multiple-of-16 input");
    assert_eq!(out.width(), 480); // 120 * 4
    assert_eq!(out.height(), 480); // 120 * 4
}

#[test]
fn scale_2_downscales_to_half_of_native() {
    if skip_if_unavailable(ModelId::RealEsrganX4Plus) {
        return;
    }
    let input = gradient_test_image(256, 256);
    let out = upscale_rgba(&input, ModelId::RealEsrganX4Plus, 2, 2, |_, _| {}, None)
        .expect("upscale_rgba scale=2 should succeed");
    assert_eq!(out.width(), 512); // 256 * 2
    assert_eq!(out.height(), 512);
}
