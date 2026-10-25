//! End-to-end smoke test for depth. Needs the OnDemand Depth Anything
//! model installed; skips with a message otherwise.
//!
//! Run with:
//!   cargo test -p prunr-core --test depth_smoke -- --nocapture

mod test_common;

use std::sync::Arc;

use image::{Rgba, RgbaImage};
use prunr_core::depth::{estimate_depth, estimate_depth_decoded, DepthLook, LensBlurMode, MODEL};
use prunr_core::progress::{check_contract, RecordingSink};
use prunr_core::{OrtEngine, Progress};
use test_common::{skip_if_model_missing, skip_if_no_ort};

/// A floor that brightens toward the bottom under a flat sky: the model
/// should read the bottom as nearer than the top.
fn ground_plane(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        if y < h / 3 {
            Rgba([150, 190, 235, 255])
        } else {
            let checker = ((x / (4 + y / 6)) + (y / (4 + y / 6))) % 2 == 0;
            let v = if checker { 200 } else { 60 };
            Rgba([v, v - 20, v - 40, 255])
        }
    })
}

#[test]
fn depth_maps_a_photo_and_reports_its_step() {
    if skip_if_no_ort("depth") {
        return;
    }
    if skip_if_model_missing("depth", MODEL.into()) {
        return;
    }
    let src = ground_plane(320, 240);
    let engine = OrtEngine::new_cpu_only(MODEL, 2).expect("depth session");
    let sink = Arc::new(RecordingSink::default());
    let depth = estimate_depth(&engine, &src, &Progress::new(sink.clone())).expect("estimate_depth");
    assert_eq!(check_contract(&sink.updates()), Ok(()));
    assert_eq!(depth.dimensions(), (518, 392));
    assert!(depth.at(0.5, 0.95) > depth.at(0.5, 0.05), "the floor's near edge is nearer than the sky");
    let rgb = image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(src.clone()).to_rgb8());
    let decoded = estimate_depth_decoded(&engine, &rgb, &Progress::none()).expect("estimate_depth_decoded");
    assert_eq!(decoded, depth, "an opaque photo measures the same whatever its layout");
    let look = DepthLook { blur: LensBlurMode::Point, blur_pct: 60, fog_pct: 40, ..DepthLook::default() };
    let out = look.apply(&src, &depth).expect("an active look");
    assert_eq!(out.dimensions(), src.dimensions());
}

#[test]
fn a_cancelled_depth_run_returns_cancelled() {
    if skip_if_no_ort("depth_cancel") || skip_if_model_missing("depth_cancel", MODEL.into()) {
        return;
    }
    let engine = OrtEngine::new_cpu_only(MODEL, 2).expect("depth session");
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let got = estimate_depth(&engine, &ground_plane(32, 32), &Progress::none().with_cancel(flag));
    assert!(matches!(got, Err(prunr_core::CoreError::Cancelled)));
}

/// Depth on the GPU path the app takes gives the CPU's map at each photo
/// shape, also where the engine compiles a session per input size
/// (CoreML), and going back to a size it ran before.
#[test]
fn the_gpu_path_maps_each_photo_shape_as_the_cpu_does() {
    if skip_if_no_ort("depth") || skip_if_model_missing("depth", MODEL.into()) {
        return;
    }
    let gpu = OrtEngine::for_model(MODEL.into(), 4, false).expect("depth on the GPU path");
    let cpu = OrtEngine::new_cpu_only(MODEL, 4).expect("depth on the CPU");
    for (w, h) in [(640, 480), (480, 640), (640, 480), (500, 500)] {
        let photo = ground_plane(w, h);
        let [a, b] = [&gpu, &cpu].map(|engine| estimate_depth(engine, &photo, &Progress::none()).expect("depth runs").to_gray16(w, h));
        let (worst, sum) = a.as_raw().iter().zip(b.as_raw()).fold((0u16, 0u64), |(worst, sum), (x, y)| {
            let d = x.abs_diff(*y);
            (worst.max(d), sum + u64::from(d))
        });
        let mean = sum as f64 / a.as_raw().len() as f64 / 257.0;
        eprintln!("[depth] {w}x{h} on {}: mean Δ {mean:.2}/255, max {}/255", gpu.active_provider(), worst / 257);
        assert!(mean <= 3.0, "{w}x{h} on {}: mean Δ {mean:.2}/255 from the CPU's map", gpu.active_provider());
    }
}
