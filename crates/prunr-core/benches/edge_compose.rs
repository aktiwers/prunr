//! Microbench for the Lines live-preview path at 4K: edge mask from the
//! DexiNed tensor, then the plain, styled and dual-scale compositions.
//!
//! Run: `cargo bench -p prunr-core --bench edge_compose`.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use image::{DynamicImage, Rgba, RgbaImage};
use prunr_core::{compose_edges, compose_edges_dual_styled, compose_edges_styled, tensor_to_edge_mask, ComposeMode, LineStyle};

const W: u32 = 4096;
const H: u32 = 3072;

fn tensor(seed: f32) -> Vec<f32> {
    // Ridges every 17 columns, so ~6 % of the plane is edge.
    (0..640 * 480).map(|i| if (i % 17) as u32 % 17 < 1 { 6.0 + seed } else { -6.0 }).collect()
}

fn base() -> RgbaImage {
    RgbaImage::from_fn(W, H, |x, y| {
        let inside = (x as i64 - W as i64 / 2).pow(2) + (y as i64 - H as i64 / 2).pow(2) < (H as i64 / 3).pow(2);
        Rgba([(x & 255) as u8, (y & 255) as u8, 128, if inside { 255 } else { 0 }])
    })
}

pub fn bench(c: &mut Criterion) {
    let fine = tensor(0.0);
    let bold = tensor(1.0);
    let base = base();
    let original = DynamicImage::ImageRgba8(base.clone());
    let mask = tensor_to_edge_mask(&fine, 480, 640, W, H, 0.5);
    let bold_mask = tensor_to_edge_mask(&bold, 480, 640, W, H, 0.5);

    let mut group = c.benchmark_group("edge_4K");
    group.throughput(Throughput::Elements((W * H) as u64));
    group.sample_size(10);
    group.bench_function("tensor_to_edge_mask", |b| {
        b.iter(|| black_box(tensor_to_edge_mask(black_box(&fine), 480, 640, W, H, 0.5)))
    });
    group.bench_function("compose_edges_plain", |b| {
        b.iter(|| black_box(compose_edges(black_box(&mask), &original, None, 2)))
    });
    group.bench_function("compose_edges_solid_color", |b| {
        b.iter(|| black_box(compose_edges(black_box(&mask), &original, Some([0, 0, 0]), 2)))
    });
    for (name, style) in [
        ("styled_solid", LineStyle::Solid),
        ("styled_gradient_y", LineStyle::GradientY { top: [255, 0, 0], bottom: [0, 0, 255] }),
        ("styled_radial", LineStyle::RadialGradient { center: [128, 128], inner: [255, 255, 0], outer: [0, 255, 255] }),
        ("styled_rainbow", LineStyle::Rainbow { cycles: 3 }),
        ("styled_chromatic", LineStyle::Chromatic { offset: 4 }),
        ("styled_noise", LineStyle::Noise { amount: 60 }),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || (),
                |_| black_box(compose_edges_styled(black_box(&mask), &base, ComposeMode::SubjectFilled, style, Some([0, 0, 0]), 2)),
                BatchSize::LargeInput,
            )
        });
    }
    group.bench_function("dual_styled", |b| {
        b.iter(|| black_box(compose_edges_dual_styled(black_box(&mask), &bold_mask, &base, ComposeMode::Ghost, [255, 0, 0], [0, 0, 255], 2)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
