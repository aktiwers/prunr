//! Microbench for the edge-shift kernel (erode / dilate by N pixels)
//! behind the Edge shift knob and the Lines thickness. Runs on a 4K mask at
//! the shifts the knob reaches.
//!
//! Run: `cargo bench -p prunr-core --bench edge_shift`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::GrayImage;
use prunr_core::morphology::shift_mask;

fn make_mask(w: u32, h: u32) -> GrayImage {
    // A soft disc: interior 255, a ~40 px ramp, exterior 0, so every
    // shift moves real edges rather than flat fill.
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let radius = h as f32 * 0.4;
    GrayImage::from_fn(w, h, |x, y| {
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        let v = ((radius - d) / 40.0 + 0.5).clamp(0.0, 1.0);
        image::Luma([(v * 255.0) as u8])
    })
}

pub fn bench(c: &mut Criterion) {
    let (w, h) = (4096, 3072);
    let mask = make_mask(w, h);
    let mut group = c.benchmark_group("edge_shift_4K");
    group.throughput(Throughput::Elements((w * h) as u64));
    group.sample_size(10);
    for shift in [1.0f32, 2.5, 10.0, 50.0, -50.0] {
        group.bench_function(format!("shift_{shift}"), |b| {
            b.iter_batched(
                || mask.clone(),
                |mut m| {
                    shift_mask(&mut m, black_box(shift));
                    black_box(m);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
