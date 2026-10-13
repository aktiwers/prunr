//! A Gaussian-like RGBA blur that uses every core: three box passes per
//! axis, with the box sizes `image::imageops::fast_blur` picks for the
//! same sigma, so it looks the same at a fraction of the wall time.

use image::RgbaImage;
use rayon::prelude::*;

const PASSES: usize = 3;
/// Columns moved together by a transpose, so each read is one cache line.
const TRANSPOSE_BLOCK: usize = 16;

/// Box radii whose three passes approximate a Gaussian of `sigma`
/// (Kutskir's method, as in `image`'s `fast_blur`).
fn box_radii(sigma: f32) -> [usize; PASSES] {
    let w_ideal = (12.0 * sigma * sigma / PASSES as f32 + 1.0).sqrt();
    let mut w_l = w_ideal.floor();
    if w_l % 2.0 == 0.0 {
        w_l -= 1.0;
    }
    let w_u = w_l + 2.0;
    let m_ideal = 0.25 * PASSES as f32 * (w_l + 3.0) - 3.0 * sigma * sigma / (w_l + 1.0);
    let m = m_ideal.round().max(0.0) as usize;
    let ceil_to_odd = |x: usize| if x.is_multiple_of(2) { x + 1 } else { x };
    std::array::from_fn(|i| {
        let size = if i < m { w_l as usize } else { w_u as usize };
        ceil_to_odd(size.saturating_sub(1) / 2)
    })
}

/// One box pass along each row of `src` (rows of `width` RGBA pixels),
/// edges replicated. Integer sums for all four channels in one sweep,
/// rounded once per pixel; reads are clamped only near the ends.
fn box_rows(src: &[u8], dst: &mut [u8], width: usize, radius: usize) {
    let row_len = width * 4;
    let kernel = (2 * radius + 1) as u32;
    let half = kernel / 2;
    let last = width - 1;
    // Between `head_end` and `tail_start` the window lies inside the row.
    let head_end = radius.min(width);
    let tail_start = width.saturating_sub(radius + 1).max(head_end);
    dst.par_chunks_exact_mut(row_len).zip(src.par_chunks_exact(row_len)).for_each(|(out, row)| {
        let px = |x: usize| {
            let i = x * 4;
            [u32::from(row[i]), u32::from(row[i + 1]), u32::from(row[i + 2]), u32::from(row[i + 3])]
        };
        let clamped = |x: usize| px(x.min(last));
        let mut sum = px(0).map(|v| v * (radius as u32 + 1));
        for x in 1..=radius {
            let p = clamped(x);
            for c in 0..4 {
                sum[c] += p[c];
            }
        }
        let mut slide = |x: usize, add: [u32; 4], sub: [u32; 4]| {
            let o = &mut out[x * 4..x * 4 + 4];
            for c in 0..4 {
                o[c] = ((sum[c] + half) / kernel) as u8;
                sum[c] = sum[c] + add[c] - sub[c];
            }
        };
        for x in 0..head_end {
            slide(x, clamped(x + radius + 1), px(0));
        }
        for x in head_end..tail_start {
            slide(x, px(x + radius + 1), px(x - radius));
        }
        for x in tail_start..width {
            slide(x, px(last), px(x - radius));
        }
    });
}

/// `src` (`width × height` RGBA) transposed into `dst` (`height × width`).
fn transpose(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let dst_row = height * 4;
    dst.par_chunks_mut(TRANSPOSE_BLOCK * dst_row).enumerate().for_each(|(block, chunk)| {
        let x0 = block * TRANSPOSE_BLOCK;
        let columns = chunk.len() / dst_row;
        for y in 0..height {
            let from = (y * width + x0) * 4;
            for dx in 0..columns {
                let to = (dx * height + y) * 4;
                chunk[to..to + 4].copy_from_slice(&src[from + dx * 4..from + dx * 4 + 4]);
            }
        }
    });
}

/// `src` blurred by roughly a Gaussian of `sigma` pixels; alpha too.
///
/// Peak working set: two buffers the size of `src` (w·h·4 bytes each),
/// one of which is returned.
pub fn blur_rgba(src: &RgbaImage, sigma: f32) -> RgbaImage {
    let (w, h) = src.dimensions();
    if sigma <= 0.0 || w == 0 || h == 0 {
        return src.clone();
    }
    let radii = box_radii(sigma);
    let [first, rest @ ..] = radii;
    let (w, h) = (w as usize, h as usize);
    let mut a = vec![0u8; src.as_raw().len()];
    let mut b = vec![0u8; a.len()];
    box_rows(src.as_raw(), &mut a, w, first);
    for &r in &rest {
        box_rows(&a, &mut b, w, r);
        std::mem::swap(&mut a, &mut b);
    }
    transpose(&a, &mut b, w, h);
    std::mem::swap(&mut a, &mut b);
    for &r in &radii {
        box_rows(&a, &mut b, h, r);
        std::mem::swap(&mut a, &mut b);
    }
    transpose(&a, &mut b, h, w);
    drop(a);
    // `b` holds w·h·4 bytes after the second transpose
    RgbaImage::from_raw(w as u32, h as u32, b).expect("buffer sized from src")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x ^ y) % 256) as u8, (200 + x % 50) as u8]))
    }

    #[test]
    fn it_matches_image_fast_blur_within_rounding() {
        for (w, h, sigma) in [(97, 61, 3.0), (200, 150, 12.5), (40, 300, 30.0), (5, 5, 2.0)] {
            let src = pattern(w, h);
            let ours = blur_rgba(&src, sigma);
            let theirs = image::imageops::fast_blur(&src, sigma);
            let worst = ours.as_raw().iter().zip(theirs.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
            assert!(worst <= 2, "{w}x{h} sigma {sigma}: off by {worst}");
        }
    }

    #[test]
    fn a_flat_image_stays_flat_and_zero_sigma_is_a_copy() {
        let flat = RgbaImage::from_pixel(33, 21, image::Rgba([10, 120, 230, 77]));
        assert_eq!(blur_rgba(&flat, 9.0), flat);
        let src = pattern(16, 9);
        assert_eq!(blur_rgba(&src, 0.0), src);
    }

    #[test]
    fn transposing_twice_gives_the_original() {
        let src = pattern(37, 19);
        let mut t = vec![0u8; src.as_raw().len()];
        let mut back = vec![0u8; src.as_raw().len()];
        transpose(src.as_raw(), &mut t, 37, 19);
        transpose(&t, &mut back, 19, 37);
        assert_eq!(&back, src.as_raw());
    }
}
