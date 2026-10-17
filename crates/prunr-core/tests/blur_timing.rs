//! Timings behind the blur choices, on a 24 MP photo. Ignored: run with
//! `cargo test -p prunr-core --release --test blur_timing -- --ignored --nocapture`.

use std::time::Instant;

use image::{Rgba, RgbaImage};

fn photo(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let v = ((x * 7 + y * 13) % 251) as u8;
        Rgba([v, v.wrapping_mul(3), v.wrapping_add(90), 255])
    })
}

/// Largest and mean per-byte difference, as "max/255, mean x.xx".
fn diff(a: &RgbaImage, b: &RgbaImage) -> String {
    let (mut max, mut sum) = (0u8, 0u64);
    for (&x, &y) in a.as_raw().iter().zip(b.as_raw()) {
        let d = x.abs_diff(y);
        max = max.max(d);
        sum += u64::from(d);
    }
    format!("max {max}/255, mean {:.2}", sum as f64 / a.as_raw().len() as f64)
}

/// `blur_rgba` against `image::imageops::fast_blur` at the two radii the
/// app uses them with: the Blurred background's and Sharpen's.
#[test]
#[ignore]
fn blur_rgba_against_fast_blur() {
    let src = photo(6000, 4000);
    for sigma in [12.0f32, 0.5] {
        let t = Instant::now();
        let a = image::imageops::fast_blur(&src, sigma);
        let fast = t.elapsed();
        let t = Instant::now();
        let b = prunr_core::blur::blur_rgba(&src, sigma);
        let ours = t.elapsed();
        println!("sigma {sigma}: fast_blur {fast:?}, blur_rgba {ours:?}, {}", diff(&a, &b));
    }
}
