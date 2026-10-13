//! End-to-end smoke test for the colorizer. Needs the OnDemand DDColor
//! model installed; skips with a message otherwise.
//!
//! Run with:
//!   cargo test -p prunr-core --test colorize_smoke -- --nocapture

mod test_common;

use std::sync::Arc;

use image::{Rgba, RgbaImage};
use prunr_core::colorize::{colorize_plane, colorized, COLORIZE_INPUT};
use prunr_core::progress::{check_contract, RecordingSink};
use prunr_core::{ModelKind, OrtEngine, Progress};
use prunr_models::{is_available, ModelId};
use test_common::skip_if_no_ort;

#[test]
fn ddcolor_colours_a_grey_photo_and_reports_its_step() {
    if skip_if_no_ort("ddcolor") {
        return;
    }
    if !is_available(ModelId::DdcolorTiny) {
        eprintln!("SKIP: DdcolorTiny not installed — run the Model Store download");
        return;
    }
    let src = RgbaImage::from_fn(96, 64, |x, y| {
        let v = ((x + y) * 255 / 160) as u8;
        Rgba([v, v, v, 255])
    });
    let engine = OrtEngine::new_cpu_only(ModelKind::DdcolorTiny, 2).expect("DDColor session");
    let sink = Arc::new(RecordingSink::default());
    let plane = colorize_plane(&engine, &src, &Progress::new(sink.clone())).expect("colorize_plane");
    assert_eq!(plane.len(), 2 * (COLORIZE_INPUT * COLORIZE_INPUT) as usize);
    assert_eq!(check_contract(&sink.updates()), Ok(()));
    assert!(plane.iter().all(|v| v.is_finite()));
    let out = colorized(&src, &plane, 1.0);
    assert_eq!(out.dimensions(), src.dimensions());
}

#[test]
fn a_cancelled_run_returns_cancelled() {
    if skip_if_no_ort("ddcolor_cancel") || !is_available(ModelId::DdcolorTiny) {
        return;
    }
    let engine = OrtEngine::new_cpu_only(ModelKind::DdcolorTiny, 2).expect("DDColor session");
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let src = RgbaImage::from_pixel(16, 16, Rgba([100, 100, 100, 255]));
    let got = colorize_plane(&engine, &src, &Progress::none().with_cancel(flag));
    assert!(matches!(got, Err(prunr_core::CoreError::Cancelled)));
}
