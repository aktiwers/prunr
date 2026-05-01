//! Phase 23-02: every `BgEffect` variant runs end-to-end through
//! `process_image_from_decoded` and produces a structurally-valid result.
//!
//! NOT a pixel-bit-exact gate (that's `golden_e2e`). The contract is "did
//! the feature even run?" — a refactor that breaks one variant's wiring
//! without breaking the others lights up here.
//!
//! Per-variant invariants are commented inline next to the assertion.
//! Skip-on-missing-ORT mirrors `golden_e2e`'s pattern.

mod test_common;

use image::DynamicImage;
use prunr_core::{
    process_image_from_decoded, BgEffect, MaskSettings, ModelKind, OrtEngine, ProgressStage,
};
use test_common::{ensure_ort_initialized, render_synthetic_source, SyntheticSpec};

fn fixture_source() -> DynamicImage {
    // Reuse the multi_subject canary: two distinct circles on a flat bg.
    // Silueta has a stable mask response on this and the bg fill exercises
    // each BgEffect variant on a non-trivial transparent region.
    let spec = SyntheticSpec {
        id: "feature_bg_effect_src",
        width: 256,
        height: 256,
        draw_source: test_common::draw_multi_subject,
    };
    DynamicImage::ImageRgba8(render_synthetic_source(&spec))
}

fn run_with_bg_effect(engine: &OrtEngine, bg_effect: BgEffect) -> image::RgbaImage {
    let img = fixture_source();
    let mask = MaskSettings { bg_effect, ..MaskSettings::default() };
    let result = process_image_from_decoded(
        &img, engine, &mask, None::<fn(ProgressStage, f32)>, None,
    ).expect("pipeline should succeed");
    result.rgba_image
}

/// `BgEffect::None` keeps transparency outside the subject — alpha must
/// vary across the image.
#[test]
fn bg_effect_none_preserves_transparency() {
    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[bg_effect_none] SKIP: {msg}");
        return;
    }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let out = run_with_bg_effect(&engine, BgEffect::None);
    assert_eq!(out.dimensions(), (256, 256));

    // Alpha varies: at least one fully-opaque pixel (subject) AND at least
    // one fully-transparent pixel (bg). Without this, the pipeline would
    // be silently ignoring the BgEffect::None contract.
    let (mut min_a, mut max_a) = (255u8, 0u8);
    for p in out.pixels() {
        let a = p.0[3];
        if a < min_a { min_a = a; }
        if a > max_a { max_a = a; }
    }
    assert!(max_a > 200, "expected at least one near-opaque subject pixel, got max α={max_a}");
    assert!(min_a < 50, "expected at least one near-transparent bg pixel, got min α={min_a}");
}

/// `BgEffect::BlurredSource` fills transparent areas with a blurred copy
/// of the source — the result must be fully opaque everywhere.
#[test]
fn bg_effect_blurred_source_produces_opaque_result() {
    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[bg_effect_blurred_source] SKIP: {msg}");
        return;
    }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let out = run_with_bg_effect(&engine, BgEffect::BlurredSource { radius: 8 });
    assert_eq!(out.dimensions(), (256, 256));
    assert_all_alpha_opaque(&out, "BlurredSource");
}

/// `BgEffect::InvertedSource` fills transparent areas with the negative
/// of the source — fully opaque everywhere.
#[test]
fn bg_effect_inverted_source_produces_opaque_result() {
    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[bg_effect_inverted_source] SKIP: {msg}");
        return;
    }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let out = run_with_bg_effect(&engine, BgEffect::InvertedSource);
    assert_eq!(out.dimensions(), (256, 256));
    assert_all_alpha_opaque(&out, "InvertedSource");
}

/// `BgEffect::DesaturatedSource` fills transparent areas with luma —
/// fully opaque everywhere; bg pixels should be near-grey.
#[test]
fn bg_effect_desaturated_source_produces_opaque_result() {
    if let Err(msg) = ensure_ort_initialized() {
        eprintln!("[bg_effect_desaturated_source] SKIP: {msg}");
        return;
    }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let out = run_with_bg_effect(&engine, BgEffect::DesaturatedSource);
    assert_eq!(out.dimensions(), (256, 256));
    assert_all_alpha_opaque(&out, "DesaturatedSource");
}

/// Pins the contract: `BgEffect::ALL` enumerates every variant, so adding
/// a new variant without a feature test here lights up as an "expected
/// 5 variants, got 4" mismatch.
#[test]
fn bg_effect_all_variants_have_a_feature_test() {
    // 4 variants today. If this assertion fails, the fix is one new
    // `bg_effect_<name>_*` test above + bumping the expected count.
    assert_eq!(
        BgEffect::ALL.len(),
        4,
        "BgEffect grew a new variant — add a feature test in feature_bg_effect.rs",
    );
}

fn assert_all_alpha_opaque(img: &image::RgbaImage, label: &str) {
    let mut min_a: u8 = 255;
    for p in img.pixels() {
        if p.0[3] < min_a { min_a = p.0[3]; }
    }
    assert_eq!(min_a, 255, "{label}: expected all alpha=255, found min α={min_a}");
}
