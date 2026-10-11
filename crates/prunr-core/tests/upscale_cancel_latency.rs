//! Measures how fast a running upscale tile aborts after
//! `RunOptions::terminate()`. Ignored by default: it needs an installed
//! upscale model and takes several seconds. Run with
//!   cargo test -p prunr-core --test upscale_cancel_latency -- --ignored --nocapture

mod test_common;

/// Prints each finished tile's time, for reading the cancel latency.
struct TilePrinter(Instant);

impl prunr_core::ProgressSink for TilePrinter {
    fn report(&self, update: prunr_core::ProgressUpdate) {
        if let prunr_core::ProgressUpdate::Inner { done, total, .. } = update {
            eprintln!("tile {done}/{total} done at {:?}", self.0.elapsed());
        }
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use prunr_core::upscale::{pick_optimization_level, upscale_rgba_with_engine, UpscaleRunOptions};
use prunr_core::{CoreError, InferenceEngine, ModelKind, OrtEngine};
use prunr_models::{is_available, ModelId};
use test_common::skip_if_no_ort;

#[test]
#[ignore = "needs an installed upscale model and measures wall-clock"]
fn terminate_aborts_a_running_tile() {
    if skip_if_no_ort("upscale_cancel_latency") {
        return;
    }
    let (id, kind) = (ModelId::FourXNmkdSiaxCx, ModelKind::FourXNmkdSiaxCx);
    if !is_available(id) {
        eprintln!("SKIP: {id:?} not installed");
        return;
    }
    let descriptor = prunr_models::descriptor(id).expect("registry entry");
    let level = pick_optimization_level(descriptor);
    let t0 = Instant::now();
    let engine = Arc::new(
        OrtEngine::new_with_optimization_level(kind, 2, level).expect("engine"),
    );
    eprintln!("engine ready in {:?} on {}", t0.elapsed(), engine.active_provider());

    let input = image::RgbaImage::from_fn(1024, 1024, |x, y| {
        image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
    });
    let cancel = Arc::new(AtomicBool::new(false));
    let run_options = Arc::new(UpscaleRunOptions::new().expect("RunOptions"));

    let worker = {
        let engine = Arc::clone(&engine);
        let cancel = Arc::clone(&cancel);
        let run_options = Arc::clone(&run_options);
        std::thread::spawn(move || {
            let started = Instant::now();
            let result = upscale_rgba_with_engine(
                &input, &engine, id, 4, &prunr_core::Progress::new(Arc::new(TilePrinter(started))).with_cancel(cancel), Some(&run_options),
            );
            (result.map(|_| ()), started.elapsed())
        })
    };

    std::thread::sleep(Duration::from_secs(3));
    let cancel_at = Instant::now();
    let terminate = run_options.terminate();
    cancel.store(true, Ordering::Release);
    let (result, total) = worker.join().expect("worker thread");
    eprintln!(
        "terminate() -> {terminate:?}; worker returned {:?} {:?} after cancel (total run {total:?})",
        result.as_ref().err(),
        cancel_at.elapsed()
    );
    assert!(matches!(result, Err(CoreError::Cancelled)), "expected Cancelled, got {result:?}");
}

/// Wall-clock of a whole 1024² upscale, the other side of the tile-size
/// trade: smaller tiles cancel sooner and cost more overlap. Run with
///   cargo test -p prunr-core --test upscale_cancel_latency full_run -- --ignored --nocapture
#[test]
#[ignore = "needs an installed upscale model and measures wall-clock"]
fn full_run_time() {
    if skip_if_no_ort("upscale_full_run") {
        return;
    }
    let (id, kind) = (ModelId::FourXNmkdSiaxCx, ModelKind::FourXNmkdSiaxCx);
    if !is_available(id) {
        eprintln!("SKIP: {id:?} not installed");
        return;
    }
    let descriptor = prunr_models::descriptor(id).expect("registry entry");
    let engine = OrtEngine::new_with_optimization_level(kind, 2, pick_optimization_level(descriptor)).expect("engine");
    let input = image::RgbaImage::from_fn(1024, 1024, |x, y| {
        image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
    });
    let started = Instant::now();
    let sink = Arc::new(prunr_core::progress::RecordingSink::default());
    let out = upscale_rgba_with_engine(&input, &engine, id, 4, &prunr_core::Progress::new(sink.clone()), None)
        .expect("upscale");
    let tiles = sink.updates().iter().filter(|u| matches!(u, prunr_core::ProgressUpdate::Tile { done: true, .. })).count();
    eprintln!("{} on {}: {tiles} tiles in {:?}", out.width(), engine.active_provider(), started.elapsed());
}
