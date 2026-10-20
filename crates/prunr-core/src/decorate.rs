//! A sticker outline and a shadow around a cut-out, drawn under it.
//!
//! Works on straight-alpha sRGB like the rest of the pipeline. The outline
//! covers everything within its width of the subject by Euclidean distance
//! (round corners, an anti-aliased rim) in the outline colour. The shadow
//! is the alpha of subject and outline blurred by three box passes (close
//! to a Gaussian), moved down for a Drop shadow, in black at its opacity.
//!
//! Only the subject's bbox plus the decoration's reach is touched, in
//! place; pixels farther out cannot change. Scans, transforms and blends
//! run row-parallel: on a 20 MP subject an outline costs ~80–100 ms at any
//! width, a shadow ~90 ms.
//!
//! Peak working set for a region of n pixels: the outline's alpha copy (n)
//! and two f32 distance planes (8n); the shadow's alpha copy (n) and two
//! f32 planes (8n) plus the box filter's row sums (4n). Each set is freed
//! before the next starts.

use image::RgbaImage;
use rayon::prelude::*;

use crate::types::Decoration;

pub fn apply_decoration(rgba: &mut RgbaImage, d: &Decoration) {
    if !d.has_outline() && !d.has_shadow() {
        return;
    }
    let (w, h) = rgba.dimensions();
    let Some((x0, y0, x1, y1)) = alpha_bbox(rgba) else { return };
    let reach = d.reach_px();
    let region = Region {
        x: x0.saturating_sub(reach) as usize,
        y: y0.saturating_sub(reach) as usize,
        w: ((x1 + reach + 1).min(w) - x0.saturating_sub(reach)) as usize,
        h: ((y1 + reach + 1).min(h) - y0.saturating_sub(reach)) as usize,
        stride: w as usize,
    };
    decorate_region(rgba, d, &region);
}

fn decorate_region(rgba: &mut RgbaImage, d: &Decoration, region: &Region) {
    if d.has_outline() {
        let cover = outline_cover(&region.alpha(rgba), region.w, region.h, d.outline_width);
        region.composite_under(rgba, d.outline_color, |i| cover[i]);
    }
    if d.has_shadow() {
        let cast = shadow_cover(&region.alpha(rgba), region.w, region.h, d);
        region.composite_under(rgba, [0, 0, 0], |i| cast[i]);
    }
}

/// The part of the image the decoration can touch: the subject's bbox
/// plus the decoration's reach. Pixels outside it never change.
struct Region {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    /// Image width, for row offsets.
    stride: usize,
}

impl Region {
    fn rows<'a>(&self, rgba: &'a mut RgbaImage) -> impl IndexedParallelIterator<Item = &'a mut [u8]> {
        let (x, w, stride) = (self.x, self.w, self.stride);
        rgba.as_mut()
            .par_chunks_exact_mut(stride * 4)
            .skip(self.y)
            .take(self.h)
            .map(move |row| &mut row[x * 4..(x + w) * 4])
    }

    /// The region's alpha, one byte per pixel.
    fn alpha(&self, rgba: &RgbaImage) -> Vec<u8> {
        let raw = rgba.as_raw();
        let mut out = vec![0u8; self.w * self.h];
        out.par_chunks_exact_mut(self.w).enumerate().for_each(|(ry, row)| {
            let start = ((self.y + ry) * self.stride + self.x) * 4;
            for (o, px) in row.iter_mut().zip(raw[start..start + self.w * 4].chunks_exact(4)) {
                *o = px[3];
            }
        });
        out
    }

    /// The region composited over a layer of `color` at coverage `under`
    /// (0–1 for each region pixel, row-major), straight alpha.
    fn composite_under(&self, rgba: &mut RgbaImage, color: [u8; 3], under: impl Fn(usize) -> f32 + Sync) {
        let w = self.w;
        self.rows(rgba).enumerate().for_each(|(ry, row)| {
            for (rx, px) in row.chunks_exact_mut(4).enumerate() {
                if px[3] == 255 {
                    continue;
                }
                let top = px[3] as f32 / 255.0;
                let below = under(ry * w + rx) * (1.0 - top);
                let out = top + below;
                if out <= 0.0 {
                    continue;
                }
                let inv = 1.0 / out;
                for c in 0..3 {
                    px[c] = ((px[c] as f32 * top + color[c] as f32 * below) * inv).round().clamp(0.0, 255.0) as u8;
                }
                px[3] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        });
    }
}

/// Inclusive bounds of the pixels with any alpha.
fn alpha_bbox(rgba: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let w = rgba.width() as usize;
    rgba.as_raw()
        .par_chunks_exact(w * 4)
        .enumerate()
        .filter_map(|(y, row)| {
            let first = row.chunks_exact(4).position(|px| px[3] != 0)?;
            let last = row.chunks_exact(4).rposition(|px| px[3] != 0)?;
            Some((first, y, last, y))
        })
        .reduce_with(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
        .map(|(a, b, c, e)| (a as u32, b as u32, c as u32, e as u32))
}

/// Outline coverage: 1 inside the subject (alpha ≥ half), falling from 1
/// to 0 across the pixel at `width` from it, by Euclidean distance (round
/// corners, anti-aliased rim). O(n) whatever the width.
fn outline_cover(alpha: &[u8], w: usize, h: usize, width: f32) -> Vec<f32> {
    let d2 = squared_distance_to(alpha, w, h, |a| a >= 128);
    d2.par_iter().map(|&d| (width + 0.5 - d.sqrt()).clamp(0.0, 1.0)).collect()
}

/// Shadow coverage: the alpha moved down for a Drop shadow, blurred by
/// three box passes (close to a Gaussian), at the shadow's opacity.
fn shadow_cover(alpha: &[u8], w: usize, h: usize, d: &Decoration) -> Vec<f32> {
    let fall = d.shadow_fall_px() as usize;
    let mut cast = vec![0.0f32; w * h];
    // Rows above `fall` start empty; each row below takes the one `fall` up.
    cast.par_chunks_exact_mut(w).enumerate().skip(fall).for_each(|(y, row)| {
        for (o, &a) in row.iter_mut().zip(&alpha[(y - fall) * w..(y - fall + 1) * w]) {
            *o = a as f32 / 255.0;
        }
    });
    let radius = d.shadow_radius_px();
    if radius > 0 {
        crate::guided_filter::box_blur3_into(&mut cast, w as u32, h as u32, radius, &mut Vec::new());
    }
    let opacity = d.shadow_opacity.clamp(0.0, 1.0);
    cast.par_iter_mut().for_each(|v| *v *= opacity);
    cast
}

/// Squared Euclidean distance from each pixel to the nearest pixel where
/// `inside` holds (0 there), by Felzenszwalb–Huttenlocher's separable
/// lower-envelope transform: columns, then rows, each in parallel.
fn squared_distance_to(alpha: &[u8], w: usize, h: usize, inside: impl Fn(u8) -> bool + Sync) -> Vec<f32> {
    const FAR: f32 = 1e20;
    let mut grid: Vec<f32> = alpha.par_iter().map(|&a| if inside(a) { 0.0 } else { FAR }).collect();
    // Columns: transpose so each column is a contiguous line.
    let mut cols = vec![0.0f32; w * h];
    cols.par_chunks_exact_mut(h).enumerate().for_each(|(x, col)| {
        for (y, c) in col.iter_mut().enumerate() {
            *c = grid[y * w + x];
        }
        edt_line(col);
    });
    grid.par_chunks_exact_mut(w).enumerate().for_each(|(y, row)| {
        for (x, r) in row.iter_mut().enumerate() {
            *r = cols[x * h + y];
        }
        edt_line(row);
    });
    grid
}

/// One line of the lower-envelope transform, in place: `f` holds 0 or
/// FAR in, squared distance to the nearest 0 out.
/// In f64: with FAR near 1e20, f32 would lose the `q²` terms.
fn edt_line(f: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let src: Vec<f64> = f.iter().map(|&x| x as f64).collect();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let at = |q: usize| src[q] + (q * q) as f64;
    for q in 1..n {
        let mut s = (at(q) - at(v[k])) / (2.0 * (q - v[k]) as f64);
        // z[0] is -inf, so this stops at k = 0.
        while s <= z[k] {
            k -= 1;
            s = (at(q) - at(v[k])) / (2.0 * (q - v[k]) as f64);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out) in f.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dq = (q - v[k].min(q)) as f64 + (v[k].saturating_sub(q)) as f64;
        *out = (dq * dq + src[v[k]]) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ShadowKind;

    /// An opaque red square in the middle of a transparent canvas.
    fn square() -> RgbaImage {
        let mut img = RgbaImage::new(64, 64);
        for y in 24..40 {
            for x in 24..40 {
                img.put_pixel(x, y, image::Rgba([200, 0, 0, 255]));
            }
        }
        img
    }

    #[test]
    fn no_decoration_leaves_the_image_untouched() {
        let mut img = square();
        apply_decoration(&mut img, &Decoration::default());
        assert_eq!(img, square());
    }

    /// The outline is an opaque ring of its width and colour around the
    /// subject, rounded at the corners, and the subject stays as it was.
    #[test]
    fn the_outline_rings_the_subject_in_its_color() {
        let mut img = square();
        let d = Decoration { outline_width: 4.0, outline_color: [255, 255, 255], ..Decoration::default() };
        apply_decoration(&mut img, &d);
        assert_eq!(img.get_pixel(30, 30).0, [200, 0, 0, 255], "inside is the subject");
        assert_eq!(img.get_pixel(21, 30).0, [255, 255, 255, 255], "3 px out is outline");
        assert_eq!(img.get_pixel(18, 30).0[3], 0, "past the width is clear");
        assert_eq!(img.get_pixel(20, 20).0[3], 0, "the corner is round, not square");
    }

    /// A Drop shadow darkens below the subject and not above it; a Soft
    /// one reaches every side.
    #[test]
    fn a_drop_shadow_falls_below_and_a_soft_one_all_around() {
        let drop = Decoration { shadow: ShadowKind::Drop, shadow_opacity: 0.5, shadow_blur: 4.0, shadow_distance: 8.0, ..Decoration::default() };
        let mut img = square();
        apply_decoration(&mut img, &drop);
        assert!(img.get_pixel(32, 45).0[3] > 60, "below the subject is shadow");
        assert_eq!(img.get_pixel(32, 20).0[3], 0, "above it is clear");
        assert_eq!(img.get_pixel(32, 45).0[..3], [0, 0, 0], "the shadow is black");
        assert_eq!(img.get_pixel(30, 30).0, [200, 0, 0, 255], "the subject covers its shadow");

        let soft = Decoration { shadow: ShadowKind::Soft, ..drop };
        let mut img = square();
        apply_decoration(&mut img, &soft);
        for (x, y) in [(32, 21), (32, 42), (21, 32), (42, 32)] {
            assert!(img.get_pixel(x, y).0[3] > 0, "soft shadow at ({x}, {y})");
        }
    }

    /// The bbox crop changes nothing: the same subject shifted within a
    /// larger canvas gets the same decoration.
    #[test]
    fn the_result_does_not_depend_on_where_the_subject_sits() {
        let d = Decoration { outline_width: 3.0, shadow: ShadowKind::Drop, shadow_opacity: 0.6, shadow_blur: 6.0, shadow_distance: 5.0, ..Decoration::default() };
        let mut small = square();
        apply_decoration(&mut small, &d);
        let mut big = RgbaImage::new(128, 128);
        image::imageops::replace(&mut big, &square(), 40, 40);
        apply_decoration(&mut big, &d);
        let back = image::imageops::crop_imm(&big, 40, 40, 64, 64).to_image();
        assert_eq!(back, small);
    }

    /// Cost of each stage on a 20 MP subject; run with
    /// `cargo test -p prunr-core --profile profiling --lib decorate_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement"]
    fn decorate_cost() {
        let (w, h) = (5472u32, 3648u32);
        let mut base = RgbaImage::new(w, h);
        for y in 400..3300 { for x in 1200..4200 {
            let dx = x as f32 - 2700.0; let dy = y as f32 - 1850.0;
            if dx * dx / 1500.0f32.powi(2) + dy * dy / 1450.0f32.powi(2) < 1.0 { base.put_pixel(x, y, image::Rgba([90, 60, 40, 255])); }
        }}
        for (name, d) in [
            ("outline 16", Decoration { outline_width: 16.0, ..Decoration::default() }),
            ("outline 96", Decoration { outline_width: 96.0, ..Decoration::default() }),
            ("drop shadow 24", Decoration { shadow: ShadowKind::Drop, ..Decoration::default() }),
            ("both, 16 + drop 64", Decoration { outline_width: 16.0, shadow: ShadowKind::Drop, shadow_blur: 64.0, ..Decoration::default() }),
        ] {
            let mut img = base.clone();
            let t = std::time::Instant::now();
            apply_decoration(&mut img, &d);
            eprintln!("{name}: {} ms", t.elapsed().as_millis());
        }
    }


    /// Decorating only the subject's reach changes no pixel against
    /// decorating the whole image: the shadow's crop leaves its blur room.
    #[test]
    fn the_decorated_region_equals_the_whole_image() {
        let subject = RgbaImage::from_fn(160, 150, |x, y| {
            let (dx, dy) = (x as f32 - 70.0, y as f32 - 60.0);
            image::Rgba([90, 60, 40, if dx * dx + dy * dy < 225.0 { 255 } else if dx * dx + dy * dy < 256.0 { 120 } else { 0 }])
        });
        for d in [
            Decoration { shadow: ShadowKind::Drop, shadow_opacity: 0.6, shadow_blur: 6.0, shadow_distance: 5.0, ..Decoration::default() },
            Decoration { shadow: ShadowKind::Drop, shadow_opacity: 0.8, shadow_blur: 3.0, shadow_distance: 0.0, ..Decoration::default() },
            Decoration { outline_width: 3.0, shadow: ShadowKind::Drop, shadow_opacity: 0.5, shadow_blur: 9.0, shadow_distance: 4.0, ..Decoration::default() },
        ] {
            let mut cropped = subject.clone();
            apply_decoration(&mut cropped, &d);
            let mut whole = subject.clone();
            let (w, h) = whole.dimensions();
            decorate_region(&mut whole, &d, &Region { x: 0, y: 0, w: w as usize, h: h as usize, stride: w as usize });
            let off: Vec<u8> = cropped.as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).filter(|&v| v > 0).collect();
            assert!(off.is_empty(), "{} channels off, by up to {}: {d:?}", off.len(), off.iter().max().unwrap_or(&0));
        }
    }

    /// The distance transform matches brute force on a scattered mask,
    /// edges and corners included.
    #[test]
    fn squared_distance_matches_brute_force() {
        let (w, h) = (23usize, 17usize);
        let alpha: Vec<u8> = (0..w * h).map(|i| if (i * 7919) % 31 == 0 || i == w * h - 1 { 255 } else { 0 }).collect();
        let got = squared_distance_to(&alpha, w, h, |a| a >= 128);
        for y in 0..h { for x in 0..w {
            let want = (0..w * h).filter(|&j| alpha[j] >= 128)
                .map(|j| { let (jx, jy) = (j % w, j / w); ((jx as i64 - x as i64).pow(2) + (jy as i64 - y as i64).pow(2)) as f32 })
                .fold(f32::INFINITY, f32::min);
            assert_eq!(got[y * w + x], want, "({x}, {y})");
        }}
    }

}
