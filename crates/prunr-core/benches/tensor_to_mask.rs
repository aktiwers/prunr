//! Microbench for `tensor_to_mask` — Tier 2 hot path. Refine-edges-on
//! exercises `guided_filter_alpha` so this bench partly overlaps with
//! `guided_filter` but with the upstream resize-and-threshold steps
//! attached. Refine-edges-off measures the resize+threshold path
//! alone, which the live-preview slider drag sits on.
//!
//! Run: `cargo bench -p prunr-core --bench tensor_to_mask`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array4;
use prunr_core::{postprocess::PostprocessOpts, tensor_to_mask, MaskSettings, ModelKind};

fn make_original(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, 128]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn make_tensor(tensor_h: usize, tensor_w: usize) -> Array4<f32> {
    // Smoothly varying [0,1] values so threshold + guided filter both
    // see real signal — uniform input would short-circuit the variance
    // term inside the guided filter.
    Array4::from_shape_fn((1, 1, tensor_h, tensor_w), |(_, _, y, x)| {
        let dx = (x as f32 / tensor_w as f32) - 0.5;
        let dy = (y as f32 / tensor_h as f32) - 0.5;
        (1.0 - 2.0 * (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0)
    })
}

pub fn bench(c: &mut Criterion) {
    // Original = 4K, tensor = 1024² (BiRefNet-class output). Two
    // configurations: refine_edges on and off; the on-case dominates
    // the postprocess wall-clock, the off-case is the hot path during
    // live-preview drag.
    let original = make_original(4096, 3072);
    let tensor = make_tensor(1024, 1024);
    let mask_no_refine = MaskSettings {
        refine_edges: false,
        ..Default::default()
    };
    let mask_refine = MaskSettings {
        refine_edges: true,
        ..Default::default()
    };

    let mut group = c.benchmark_group("tensor_to_mask");
    group.throughput(Throughput::Elements((4096 * 3072) as u64));
    group.bench_function("4K_no_refine", |b| {
        let opts = PostprocessOpts::new(&mask_no_refine, ModelKind::BiRefNetLite);
        b.iter(|| {
            let out = tensor_to_mask(black_box(tensor.view()), black_box(&original), &opts);
            black_box(out);
        });
    });
    group.bench_function("4K_refine", |b| {
        let opts = PostprocessOpts::new(&mask_refine, ModelKind::BiRefNetLite);
        b.iter(|| {
            let out = tensor_to_mask(black_box(tensor.view()), black_box(&original), &opts);
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
