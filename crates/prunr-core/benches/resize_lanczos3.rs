//! Microbench for `resize_rgb_lanczos3` + `resize_gray_lanczos3`. Both
//! sit on every full-pipeline image (encode → resize-to-model →
//! infer → resize-back). Down-resize and up-resize are separate cases
//! because fast_image_resize's SIMD path differs.
//!
//! Run: `cargo bench -p prunr-core --bench resize_lanczos3`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use prunr_core::formats::{resize_gray_lanczos3, resize_rgb_lanczos3};

fn make_rgb(w: u32, h: u32) -> DynamicImage {
    // Same gradient pattern as the guided-filter bench — non-trivial
    // signal so SIMD doesn't short-circuit on uniform inputs.
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((x ^ y) & 0xff) as u8;
            let b = (((x + y) * 5) & 0xff) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn make_gray(w: u32, h: u32) -> GrayImage {
    let mut g = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            g.put_pixel(x, y, Luma([((x ^ y) & 0xff) as u8]));
        }
    }
    g
}

pub fn bench(c: &mut Criterion) {
    // Down-resize (4K → 1K — typical model input).
    let big_rgb = make_rgb(4096, 3072);
    {
        let mut group = c.benchmark_group("resize_rgb_lanczos3_down");
        group.throughput(Throughput::Elements(4096 * 3072));
        group.bench_function("4096x3072_to_1024x768", |b| {
            b.iter(|| {
                let out = resize_rgb_lanczos3(black_box(&big_rgb), 1024, 768);
                black_box(out);
            });
        });
        group.finish();
    }
    // Up-resize (1K → 4K — mask back to image resolution).
    let small_gray = make_gray(1024, 768);
    {
        let mut group = c.benchmark_group("resize_gray_lanczos3_up");
        group.throughput(Throughput::Elements(1024 * 768));
        group.bench_function("1024x768_to_4096x3072", |b| {
            b.iter(|| {
                let out = resize_gray_lanczos3(black_box(&small_gray), 4096, 3072);
                black_box(out);
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
