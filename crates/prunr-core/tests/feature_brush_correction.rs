//! Phase 23-05: brush correction strokes affect the postprocess result.
//!
//! Drives the full Tier-1 + Tier-2 path: `infer_only` → tensor → two
//! `postprocess_from_flat` runs (without and with a `MaskCorrection`).
//! Asserts the corrected result differs in the painted region and
//! matches outside it.

mod test_common;

use prunr_core::{
    brush::{paint_circle, BrushMode, MaskCorrection, Stamp},
    infer_only, postprocess_from_flat, MaskSettings, ModelKind, OrtEngine, PostprocessOpts,
    ProgressStage,
};
use test_common::{multi_subject_canary, skip_if_no_ort};

const SIZE: u32 = 256;

#[test]
fn brush_correction_stroke_alters_the_painted_region_only() {
    if skip_if_no_ort("brush_correction") {
        return;
    }

    let engine =
        OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("OrtEngine::new_cpu_only(Silueta)");
    let img = multi_subject_canary();
    let mask_settings = MaskSettings::default();

    // Step 1: run inference once; reuse the tensor for both postprocess
    // passes (changing only the correction between them).
    let ir = infer_only(&img, &engine, None::<fn(ProgressStage, f32)>, None).expect("infer_only");
    let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);

    // Step 2a: baseline result, no correction.
    let baseline = postprocess_from_flat(
        &ir.tensor_data,
        ir.tensor_height,
        ir.tensor_width,
        &img,
        &opts,
    )
    .expect("postprocess baseline");

    // Step 2b: apply a `Subtract` brush stroke covering a small disk.
    // Subtract drives mask alpha → 0 in the painted region, so the
    // result's alpha there should drop noticeably.
    let mut correction = MaskCorrection::empty(SIZE as u16, SIZE as u16);
    let stamp = Stamp {
        hardness: 1.0,
        strength: 1.0,
        mode: BrushMode::Subtract,
    };
    paint_circle(&mut correction, 80.0, 128.0, 25.0, stamp);
    let opts_with_corr = opts.with_correction(Some(&correction));
    let corrected = postprocess_from_flat(
        &ir.tensor_data,
        ir.tensor_height,
        ir.tensor_width,
        &img,
        &opts_with_corr,
    )
    .expect("postprocess corrected");

    assert_eq!(baseline.dimensions(), corrected.dimensions());

    // The painted region must show a noticeable alpha drop versus the
    // baseline. We measure mean alpha inside the disk.
    let mean_a = |img: &image::RgbaImage, cx: u32, cy: u32, r: u32| -> f32 {
        let mut sum = 0u64;
        let mut count = 0u64;
        for py in cy.saturating_sub(r)..=cy + r {
            for px in cx.saturating_sub(r)..=cx + r {
                let dx = px as i32 - cx as i32;
                let dy = py as i32 - cy as i32;
                if dx * dx + dy * dy <= (r * r) as i32 && px < img.width() && py < img.height() {
                    sum += img.get_pixel(px, py).0[3] as u64;
                    count += 1;
                }
            }
        }
        sum as f32 / count.max(1) as f32
    };

    let baseline_inside = mean_a(&baseline, 80, 128, 20);
    let corrected_inside = mean_a(&corrected, 80, 128, 20);
    assert!(
        corrected_inside + 50.0 < baseline_inside,
        "subtract stroke should drop mean alpha inside disk: baseline={baseline_inside}, corrected={corrected_inside}"
    );

    // Outside the disk the two results must match — corrections are
    // localized. Sample the right-half subject (other circle, untouched).
    for &(x, y) in &[(180u32, 128u32), (200, 100), (160, 160), (240, 240)] {
        assert_eq!(
            baseline.get_pixel(x, y).0,
            corrected.get_pixel(x, y).0,
            "untouched pixel ({x},{y}) must match baseline",
        );
    }
}

/// Pins the cache-key contract: identical correction → identical result.
/// `correction_hash` on `MaskSettings` keys the cache; if hashing drifts,
/// the cache misses or stales — both failure modes caught by byte equality.
#[test]
fn brush_correction_is_deterministic_across_runs() {
    if skip_if_no_ort("brush_correction_deterministic") {
        return;
    }
    let engine =
        OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("OrtEngine::new_cpu_only(Silueta)");
    let img = multi_subject_canary();
    let mask_settings = MaskSettings::default();
    let ir = infer_only(&img, &engine, None::<fn(ProgressStage, f32)>, None).expect("infer_only");
    let opts = PostprocessOpts::new(&mask_settings, ModelKind::Silueta);

    let mut correction = MaskCorrection::empty(SIZE as u16, SIZE as u16);
    let stamp = Stamp {
        hardness: 1.0,
        strength: 1.0,
        mode: BrushMode::Subtract,
    };
    paint_circle(&mut correction, 80.0, 128.0, 25.0, stamp);

    let opts_with_corr = opts.with_correction(Some(&correction));
    let run1 = postprocess_from_flat(
        &ir.tensor_data,
        ir.tensor_height,
        ir.tensor_width,
        &img,
        &opts_with_corr,
    )
    .expect("run 1");
    let run2 = postprocess_from_flat(
        &ir.tensor_data,
        ir.tensor_height,
        ir.tensor_width,
        &img,
        &opts_with_corr,
    )
    .expect("run 2");
    assert_eq!(
        run1.as_raw(),
        run2.as_raw(),
        "same correction + same inputs must produce identical bytes"
    );
}
