//! Phase 23-03: every `LineMode` variant runs the DexiNed edge-detection
//! path end-to-end and produces a structurally-valid result. Plus
//! representative samples across `EdgeScale`, `ComposeMode`, and
//! `LineStyle` so a refactor that breaks one combo without breaking the
//! others lights up here.
//!
//! Each test asserts: output dimensions match source; output is a valid
//! RGBA image. NOT pixel-bit-exact (that's `golden_e2e`'s job).

mod test_common;

use image::DynamicImage;
use prunr_core::{
    compose_subject_outline, process_image_from_decoded, ComposeMode, EdgeEngine,
    EdgeScale, EdgeSettings, InputTransform, LineStyle, MaskSettings, ModelKind,
    OrtEngine, ProgressStage,
};
use test_common::{render_synthetic_source, skip_if_no_ort, SyntheticSpec};

fn fixture_source() -> DynamicImage {
    let spec = SyntheticSpec {
        id: "feature_line_mode_src",
        width: 256,
        height: 256,
        draw_source: test_common::draw_multi_subject,
    };
    DynamicImage::ImageRgba8(render_synthetic_source(&spec))
}

fn default_edge_settings(scale: EdgeScale, compose: ComposeMode, style: LineStyle) -> EdgeSettings {
    EdgeSettings {
        line_strength: 0.5,
        solid_line_color: Some([255, 0, 0]),
        edge_thickness: 1,
        edge_scale: scale,
        compose_mode: compose,
        line_style: style,
        input_transform: InputTransform::None,
    }
}

/// `LineMode::Off` runs the segmentation pipeline only — no DexiNed
/// inference. Already covered by golden_e2e + bg_effect; this test pins
/// "no error, dimensions match" for the LineMode::Off branch directly.
#[test]
fn line_mode_off_runs_seg_only() {
    if skip_if_no_ort("line_mode_off") { return; }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let img = fixture_source();
    let result = process_image_from_decoded(
        &img, &engine, &MaskSettings::default(),
        None::<fn(ProgressStage, f32)>, None,
    ).expect("Off pipeline should succeed");
    assert_eq!(result.rgba_image.dimensions(), (256, 256));
}

/// `LineMode::EdgesOnly` runs DexiNed on the source and finalizes lines
/// without segmentation. Tests the standalone edge path.
#[test]
fn line_mode_edges_only_produces_lines_from_source() {
    if skip_if_no_ort("line_mode_edges_only") { return; }
    let edge_engine = EdgeEngine::new().expect("EdgeEngine::new");
    let img = fixture_source();
    let edge = default_edge_settings(EdgeScale::Fused, ComposeMode::LinesOnly, LineStyle::Solid);
    let out = edge_engine.detect(&img, &edge).expect("EdgeEngine::detect");
    assert_eq!(out.dimensions(), (256, 256));
}

/// `LineMode::SubjectOutline` runs segmentation + DexiNed and composites
/// edges within the subject only. Exercises the full two-engine pipeline.
#[test]
fn line_mode_subject_outline_combines_seg_and_edge() {
    if skip_if_no_ort("line_mode_subject_outline") { return; }
    let seg_engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let edge_engine = EdgeEngine::new().expect("EdgeEngine::new");
    let img = fixture_source();

    // Step 1: segmentation produces masked RGBA.
    let masked = process_image_from_decoded(
        &img, &seg_engine, &MaskSettings::default(),
        None::<fn(ProgressStage, f32)>, None,
    ).expect("seg pipeline").rgba_image;

    // Step 2: edge inference + compose_subject_outline overlays lines.
    let edge_res = edge_engine.infer_all_tensors(&img).expect("infer_all_tensors");
    let edge = default_edge_settings(EdgeScale::Fused, ComposeMode::SubjectFilled, LineStyle::Solid);
    let out = compose_subject_outline(&edge_res, &masked, &edge);

    assert_eq!(out.dimensions(), (256, 256));
    // SubjectFilled keeps the subject fill — at least one near-opaque
    // pixel should exist where the subject sits.
    let max_a = out.pixels().map(|p| p.0[3]).max().unwrap_or(0);
    assert!(max_a > 200, "SubjectOutline must keep some opaque subject pixels, got max α={max_a}");
}

/// Sample the `EdgeScale` corners (Fine + Bold) plus the default (Fused)
/// to catch a refactor that mis-indexes the per-scale tensor lookup.
#[test]
fn edge_scale_fine_bold_fused_all_run() {
    if skip_if_no_ort("edge_scale_*") { return; }
    let edge_engine = EdgeEngine::new().expect("EdgeEngine::new");
    let img = fixture_source();
    for scale in [EdgeScale::Fine, EdgeScale::Bold, EdgeScale::Fused] {
        let edge = default_edge_settings(scale, ComposeMode::LinesOnly, LineStyle::Solid);
        let out = edge_engine.detect(&img, &edge).expect("detect should succeed");
        assert_eq!(out.dimensions(), (256, 256), "{scale:?}: dimension mismatch");
    }
}

/// Sample two `LineStyle` variants — Solid and DualScale — to exercise
/// the divergent compose paths in `compose_subject_outline`.
#[test]
fn line_style_solid_and_dual_scale_compose() {
    if skip_if_no_ort("line_style_*") { return; }
    let seg_engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let edge_engine = EdgeEngine::new().expect("EdgeEngine::new");
    let img = fixture_source();
    let masked = process_image_from_decoded(
        &img, &seg_engine, &MaskSettings::default(),
        None::<fn(ProgressStage, f32)>, None,
    ).expect("seg").rgba_image;
    let edge_res = edge_engine.infer_all_tensors(&img).expect("infer_all_tensors");

    for style in [
        LineStyle::Solid,
        LineStyle::DualScale { fine_color: [255, 0, 0], bold_color: [0, 0, 255] },
    ] {
        let edge = default_edge_settings(EdgeScale::Fused, ComposeMode::SubjectFilled, style);
        let out = compose_subject_outline(&edge_res, &masked, &edge);
        assert_eq!(out.dimensions(), (256, 256), "{style:?}: dimension mismatch");
    }
}
