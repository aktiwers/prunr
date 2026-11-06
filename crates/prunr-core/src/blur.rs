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

/// One box pass along `row` (`width` RGBA pixels) into `out`, edges
/// replicated. Integer sums for all four channels in one sweep, rounded
/// once per pixel; reads are clamped only near the ends.
fn box_row(row: &[u8], out: &mut [u8], width: usize, radius: usize) {
    let kernel = (2 * radius + 1) as u32;
    let half = kernel / 2;
    let last = width - 1;
    // Between `head_end` and `tail_start` the window lies inside the row.
    let head_end = radius.min(width);
    let tail_start = width.saturating_sub(radius + 1).max(head_end);
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
}

/// Every box pass of `radii` over `line` in place, `scratch` as the
/// other buffer (both `width` RGBA pixels).
fn box_line(line: &mut [u8], scratch: &mut [u8], width: usize, radii: &[usize; PASSES]) {
    for (i, &r) in radii.iter().enumerate() {
        if i % 2 == 0 {
            box_row(line, scratch, width, r);
        } else {
            box_row(scratch, line, width, r);
        }
    }
    if PASSES % 2 == 1 {
        line.copy_from_slice(scratch);
    }
}

/// `src` blurred by roughly a Gaussian of `sigma` pixels; alpha too.
pub fn blur_rgba(src: &RgbaImage, sigma: f32) -> RgbaImage {
    let mut out = src.clone();
    blur_rgba_in_place(&mut out, sigma);
    out
}

/// `src` blurred with each pixel's colour weighted by its alpha, back in
/// straight alpha: the colour under a transparent pixel (black after an
/// alpha-weighted enlargement, or whatever another editor left there)
/// does not bleed into its visible neighbours. An opaque image blurs as
/// `blur_rgba` does; a pixel whose blurred alpha is 0 comes back black.
///
/// rationale: premultiplied only between the two passes here, so callers
/// see straight sRGB like every other `RgbaImage` in core.
///
/// Peak working set: as `blur_rgba_in_place`, plus the returned image.
pub fn blur_rgba_alpha_weighted(src: &RgbaImage, sigma: f32) -> RgbaImage {
    if crate::formats::is_opaque(src.as_raw()) {
        return blur_rgba(src, sigma);
    }
    let (w, h) = src.dimensions();
    let mut out = RgbaImage::new(w, h);
    out.as_mut().par_chunks_exact_mut(4).zip(src.as_raw().par_chunks_exact(4)).for_each(|(o, p)| {
        let a = f32::from(p[3]) / 255.0;
        for (o, &v) in o[..3].iter_mut().zip(&p[..3]) {
            *o = crate::math::round_u8(f32::from(v) * a);
        }
        o[3] = p[3];
    });
    blur_rgba_in_place(&mut out, sigma);
    out.as_mut().par_chunks_exact_mut(4).filter(|o| o[3] > 0).for_each(|o| {
        let unweigh = 255.0 / f32::from(o[3]);
        for v in &mut o[..3] {
            *v = crate::math::round_u8(f32::from(*v) * unweigh);
        }
    });
    out
}

/// `blur_rgba` over `img` itself.
///
/// Peak working set: `img`, blurred in place, plus per thread a row's
/// scratch and a band of `TRANSPOSE_BLOCK` columns (h·64 bytes, 256 KB
/// for a 4000 px tall photo).
pub fn blur_rgba_in_place(img: &mut RgbaImage, sigma: f32) {
    let (w, h) = img.dimensions();
    if sigma <= 0.0 || w == 0 || h == 0 {
        return;
    }
    let radii = box_radii(sigma);
    let (w, h) = (w as usize, h as usize);
    img.as_mut().par_chunks_exact_mut(w * 4).for_each_init(|| vec![0u8; w * 4], |scratch, row| box_line(row, scratch, w, &radii));
    // Columns a band at a time: each band is copied out as rows of `h`
    // pixels, boxed like rows, and copied back.
    let dp = crate::math::SendMutPtr(img.as_mut_ptr());
    (0..w.div_ceil(TRANSPOSE_BLOCK)).into_par_iter().for_each_init(
        || (vec![0u8; TRANSPOSE_BLOCK * h * 4], vec![0u8; h * 4]),
        |(band, scratch), block| {
            let x0 = block * TRANSPOSE_BLOCK;
            let columns = TRANSPOSE_BLOCK.min(w - x0);
            // SAFETY: band `block` reads and writes columns x0..x0 +
            // columns only; bands are column-disjoint, so no two threads
            // touch the same byte.
            let at = |y: usize, dx: usize| unsafe { dp.0.add((y * w + x0 + dx) * 4) };
            for y in 0..h {
                for dx in 0..columns {
                    unsafe { std::ptr::copy_nonoverlapping(at(y, dx), band[(dx * h + y) * 4..].as_mut_ptr(), 4) };
                }
            }
            for column in band.chunks_exact_mut(h * 4).take(columns) {
                box_line(column, scratch, h, &radii);
            }
            for y in 0..h {
                for dx in 0..columns {
                    unsafe { std::ptr::copy_nonoverlapping(band[(dx * h + y) * 4..].as_ptr(), at(y, dx), 4) };
                }
            }
            // Capture `dp` whole: its raw-pointer field alone is not Sync.
            let _ = &dp;
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x ^ y) % 256) as u8, (200 + x % 50) as u8]))
    }

    /// The colour under a transparent pixel does not bleed into what
    /// shows, and an opaque image blurs as the plain blur does.
    #[test]
    fn the_alpha_weighted_blur_reads_only_what_shows() {
        let cut = RgbaImage::from_fn(40, 20, |x, _| if x < 20 { image::Rgba([200, 60, 30, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let out = blur_rgba_alpha_weighted(&cut, 3.0);
        // Unweighted, the black would pull a half-shown pixel down to ~100.
        let worst = out.pixels().filter(|p| p[3] >= 128).map(|p| p[0].abs_diff(200).max(p[1].abs_diff(60))).max();
        assert!(worst <= Some(2), "the visible colour moved by {worst:?}");
        let mut photo = pattern(64, 48);
        photo.pixels_mut().for_each(|p| p[3] = 255);
        assert_eq!(blur_rgba_alpha_weighted(&photo, 2.0), blur_rgba(&photo, 2.0));
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

    /// The plain two-buffer blur, whole-image passes with a transpose
    /// between the axes: the oracle.
    fn two_buffer_reference(src: &RgbaImage, sigma: f32) -> Vec<u8> {
        let (w, h) = (src.width() as usize, src.height() as usize);
        let rows = |from: &[u8], to: &mut [u8], width: usize, r: usize| {
            for (o, row) in to.chunks_exact_mut(width * 4).zip(from.chunks_exact(width * 4)) {
                box_row(row, o, width, r);
            }
        };
        let transpose = |from: &[u8], to: &mut [u8], width: usize, height: usize| {
            for y in 0..height {
                for x in 0..width {
                    to[(x * height + y) * 4..][..4].copy_from_slice(&from[(y * width + x) * 4..][..4]);
                }
            }
        };
        let radii = box_radii(sigma);
        let (mut a, mut b) = (src.as_raw().clone(), vec![0u8; w * h * 4]);
        for &r in &radii {
            rows(&a, &mut b, w, r);
            std::mem::swap(&mut a, &mut b);
        }
        transpose(&a, &mut b, w, h);
        std::mem::swap(&mut a, &mut b);
        for &r in &radii {
            rows(&a, &mut b, h, r);
            std::mem::swap(&mut a, &mut b);
        }
        transpose(&a, &mut b, h, w);
        b
    }

    /// In place, a row and a band of columns at a time, the blur gives
    /// the very bytes the two-buffer blur gave, a partial last band
    /// included.
    #[test]
    fn the_in_place_blur_matches_the_two_buffer_one() {
        for (w, h, sigma) in [(97, 61, 3.0), (40, 300, 30.0), (5, 5, 2.0), (33, 17, 0.8)] {
            let src = pattern(w, h);
            assert!(blur_rgba(&src, sigma).as_raw() == &two_buffer_reference(&src, sigma), "{w}x{h} sigma {sigma}");
        }
    }
}
