//! Grayscale erosion / dilation of a mask by N pixels, with replicated
//! edges, in constant time per pixel.
//!
//! N iterated 3×3 min/max passes equal one (2N+1)² window, and a square
//! window is separable: a horizontal 1-D window followed by a vertical
//! one. Each 1-D window is the van Herk / Gil-Werman pass: within blocks
//! of `2N+1` keep running prefix and suffix extrema, then every window
//! is one `pick` of a suffix and a prefix. Output is bit-identical to the
//! iterated form (`tests::brute_force_shift` is the oracle).

use image::GrayImage;
use rayon::prelude::*;

/// Masks with fewer rows run the passes on the calling thread.
const PAR_MIN_ROWS: usize = 512;

/// Erode (`shift > 0`) or dilate (`shift < 0`) a mask by `|shift|`
/// pixels; a fractional part blends between the two nearest integer
/// shifts.
///
/// Peak working set: two mask-sized planes (24 MB at 4K).
pub fn shift_mask(mask: &mut GrayImage, shift: f32) {
    let Some(plan) = Plan::new(mask.width(), mask.height(), shift) else { return };
    let (mut hz, mut scratch) = plan.scratch();
    if plan.full > 0 {
        plan.rows(mask.as_raw(), &mut hz);
        plan.cols_into(&mut hz, &mut scratch, mask.as_mut());
    }
    plan.blend_partner(mask.as_mut(), &mut hz, &mut scratch);
}

/// `shift_mask` into a new image, leaving `mask` untouched: the live
/// preview dilates a cached mask on every tick and must keep the cached
/// one. Saves the clone an in-place call would need.
pub fn shifted(mask: &GrayImage, shift: f32) -> GrayImage {
    let Some(plan) = Plan::new(mask.width(), mask.height(), shift) else { return mask.clone() };
    let (mut hz, mut scratch) = plan.scratch();
    let mut out = GrayImage::new(mask.width(), mask.height());
    if plan.full > 0 {
        plan.rows(mask.as_raw(), &mut hz);
        plan.cols_into(&mut hz, &mut scratch, out.as_mut());
    } else {
        out.as_mut().copy_from_slice(mask.as_raw());
    }
    plan.blend_partner(out.as_mut(), &mut hz, &mut scratch);
    out
}

/// One shift: its integer part, the fractional blend, and the direction.
/// Each pass is monomorphised per extremum so the elementwise loops
/// vectorise.
struct Plan {
    w: usize,
    h: usize,
    full: usize,
    frac: f32,
    erode: bool,
}

impl Plan {
    fn new(w: u32, h: u32, shift: f32) -> Option<Self> {
        let abs = shift.abs();
        if abs < 0.01 || w == 0 || h == 0 {
            return None;
        }
        let full = abs.floor() as usize;
        Some(Self { w: w as usize, h: h as usize, full, frac: abs - full as f32, erode: shift > 0.0 })
    }

    /// The two mask-sized planes every pass works in.
    fn scratch(&self) -> (Vec<u8>, Vec<u8>) {
        (vec![0u8; self.w * self.h], vec![0u8; self.w * self.h])
    }

    fn with_window<R>(&self, f: impl FnOnce(&dyn WindowPass) -> R) -> R {
        let par = self.h >= PAR_MIN_ROWS;
        if self.erode {
            f(&Window { w: self.w, h: self.h, pick: u8::min, par })
        } else {
            f(&Window { w: self.w, h: self.h, pick: u8::max, par })
        }
    }

    fn rows(&self, src: &[u8], dst: &mut [u8]) {
        self.with_window(|win| win.rows_pass(src, dst, self.full));
    }

    fn cols_into(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8]) {
        self.with_window(|win| win.cols_copy(plane, suffix, dst, self.full));
    }

    /// Partner = the integer result shifted one more pixel, blended into
    /// `mask` as it is produced.
    fn blend_partner(&self, mask: &mut [u8], hz: &mut [u8], scratch: &mut [u8]) {
        if self.frac < 0.01 {
            return;
        }
        let (frac, inv) = (self.frac, 1.0 - self.frac);
        self.with_window(|win| {
            win.rows_pass(mask, hz, 1);
            win.cols_blend(hz, scratch, mask, frac, inv);
        });
    }
}

/// The passes of one extremum, behind one vtable call per pass (not
/// per pixel), so the two directions share the plan code.
trait WindowPass: Sync {
    fn rows_pass(&self, src: &[u8], dst: &mut [u8], r: usize);
    fn cols_copy(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8], r: usize);
    fn cols_blend(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8], frac: f32, inv: f32);
}

impl<F: Fn(u8, u8) -> u8 + Copy + Sync> WindowPass for Window<F> {
    fn rows_pass(&self, src: &[u8], dst: &mut [u8], r: usize) {
        self.rows(src, dst, r);
    }

    fn cols_copy(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8], r: usize) {
        self.cols(plane, suffix, dst, r, |o, v| *o = v);
    }

    fn cols_blend(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8], frac: f32, inv: f32) {
        self.cols(plane, suffix, dst, 1, move |a, b| *a = (*a as f32 * inv + b as f32 * frac + 0.5) as u8);
    }
}

/// The suffix and prefix taps whose `pick` is the clamped window
/// `[i - r, i + r]` on a line of `n`. A clipped border window lies in
/// one block and is a prefix alone (left edge) or a suffix alone
/// (right edge); every other window spans two adjacent blocks, or is a
/// whole block, where suffix and prefix agree.
fn taps(i: usize, n: usize, r: usize) -> (Option<usize>, Option<usize>) {
    let k = 2 * r + 1;
    let last = n - 1;
    if i < r {
        (None, Some((i + r).min(last)))
    } else if i + r <= last {
        (Some(i - r), Some(i + r))
    } else if (i - r) / k == last / k {
        (Some(i - r), None)
    } else {
        (Some(i - r), Some(last))
    }
}

fn window_line<F: Fn(u8, u8) -> u8 + Copy>(src: &[u8], out: &mut [u8], p: &mut [u8], s: &mut [u8], r: usize, pick: F) {
    let n = src.len();
    let k = 2 * r + 1;
    for (bi, block) in src.chunks(k).enumerate() {
        let base = bi * k;
        let last = block.len() - 1;
        let mut acc = block[0];
        p[base] = acc;
        for (i, &v) in block.iter().enumerate().skip(1) {
            acc = pick(acc, v);
            p[base + i] = acc;
        }
        let mut acc = block[last];
        s[base + last] = acc;
        for i in (0..last).rev() {
            acc = pick(acc, block[i]);
            s[base + i] = acc;
        }
    }
    // Interior windows are one pick of two shifted slices; only the
    // 2r border pixels take the general taps.
    let a_end = r.min(n);
    let b_end = n.saturating_sub(r).max(a_end);
    if b_end > a_end {
        // Here a_end == r, so suffix taps start at 0 and prefix taps at 2r.
        for ((o, &sv), &pv) in out[a_end..b_end].iter_mut().zip(&s[..b_end - r]).zip(&p[2 * r..]) {
            *o = pick(sv, pv);
        }
    }
    for x in (0..a_end).chain(b_end..n) {
        out[x] = match taps(x, n, r) {
            (Some(a), Some(b)) => pick(s[a], p[b]),
            (None, Some(b)) => p[b],
            (Some(a), None) => s[a],
            (None, None) => unreachable!("a window always has a tap"),
        };
    }
}

/// One mask's geometry and extremum for the two separable passes.
struct Window<F> {
    w: usize,
    h: usize,
    pick: F,
    par: bool,
}

impl<F: Fn(u8, u8) -> u8 + Copy + Sync> Window<F> {
    fn rows(&self, src: &[u8], dst: &mut [u8], r: usize) {
        let (w, pick) = (self.w, self.pick);
        if self.par {
            dst.par_chunks_mut(w)
                .zip(src.par_chunks(w))
                .for_each_init(|| (vec![0u8; w], vec![0u8; w]), |(p, s), (out, row)| {
                    window_line(row, out, p, s, r, pick);
                });
        } else {
            let (mut p, mut s) = (vec![0u8; w], vec![0u8; w]);
            for (out, row) in dst.chunks_mut(w).zip(src.chunks(w)) {
                window_line(row, out, &mut p, &mut s, r, pick);
            }
        }
    }

    /// The vertical window, row-wise so every operation streams whole
    /// rows. Within blocks of `2r + 1` rows, `suffix` receives the
    /// suffix rows and `plane` becomes the prefix rows in place; then
    /// each output row is one elementwise pick, handed to `merge` with
    /// the current `dst` byte.
    fn cols<M: Fn(&mut u8, u8) + Sync>(&self, plane: &mut [u8], suffix: &mut [u8], dst: &mut [u8], r: usize, merge: M) {
        let (w, h, pick, par) = (self.w, self.h, self.pick, self.par);
        let block = |(plane, suffix): (&mut [u8], &mut [u8])| {
            let rows = plane.len() / w;
            suffix[(rows - 1) * w..].copy_from_slice(&plane[(rows - 1) * w..]);
            for y in (0..rows - 1).rev() {
                let (cur, next) = suffix[y * w..].split_at_mut(w);
                for ((c, &a), &b) in cur.iter_mut().zip(&plane[y * w..(y + 1) * w]).zip(&next[..w]) {
                    *c = pick(a, b);
                }
            }
            for y in 1..rows {
                let (prev, cur) = plane[(y - 1) * w..].split_at_mut(w);
                for (c, &p) in cur[..w].iter_mut().zip(prev.iter()) {
                    *c = pick(*c, p);
                }
            }
        };
        let block_bytes = (2 * r + 1) * w;
        if par {
            plane.par_chunks_mut(block_bytes).zip(suffix.par_chunks_mut(block_bytes)).for_each(block);
        } else {
            plane.chunks_mut(block_bytes).zip(suffix.chunks_mut(block_bytes)).for_each(block);
        }

        let (prefix, suffix): (&[u8], &[u8]) = (plane, suffix);
        fn row(buf: &[u8], w: usize, y: usize) -> &[u8] {
            &buf[y * w..(y + 1) * w]
        }
        let out_row = |(y, out): (usize, &mut [u8])| match taps(y, h, r) {
            (Some(a), Some(b)) => {
                for ((o, &sv), &pv) in out.iter_mut().zip(row(suffix, w, a)).zip(row(prefix, w, b)) {
                    merge(o, pick(sv, pv));
                }
            }
            (None, Some(b)) => out.iter_mut().zip(row(prefix, w, b)).for_each(|(o, &v)| merge(o, v)),
            (Some(a), None) => out.iter_mut().zip(row(suffix, w, a)).for_each(|(o, &v)| merge(o, v)),
            (None, None) => unreachable!("a window always has a tap"),
        };
        if par {
            dst.par_chunks_mut(w).enumerate().for_each(out_row);
        } else {
            dst.chunks_mut(w).enumerate().for_each(out_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec: `pick` over the clamped (2N+1)² neighbourhood, then the
    /// same blend as production between N and N+1.
    fn brute_force_shift(mask: &GrayImage, shift: f32) -> Vec<u8> {
        let abs = shift.abs();
        let pick: fn(u8, u8) -> u8 = if shift > 0.0 { u8::min } else { u8::max };
        let full = abs.floor() as i64;
        let frac = abs - full as f32;
        let (w, h) = (mask.width() as i64, mask.height() as i64);
        let window = |n: i64| -> Vec<u8> {
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    let mut acc = mask.get_pixel(x as u32, y as u32)[0];
                    for ny in (y - n).max(0)..=(y + n).min(h - 1) {
                        for nx in (x - n).max(0)..=(x + n).min(w - 1) {
                            acc = pick(acc, mask.get_pixel(nx as u32, ny as u32)[0]);
                        }
                    }
                    out.push(acc);
                }
            }
            out
        };
        let mut a = window(full);
        if frac >= 0.01 {
            let inv = 1.0 - frac;
            for (a, b) in a.iter_mut().zip(window(full + 1)) {
                *a = (*a as f32 * inv + b as f32 * frac + 0.5) as u8;
            }
        }
        a
    }

    fn noise_mask(w: u32, h: u32, seed: u64) -> GrayImage {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        GrayImage::from_fn(w, h, |_, _| image::Luma([rng.random()]))
    }

    #[test]
    fn shift_matches_the_brute_force_window() {
        let shifts = [1.0f32, 2.0, 3.0, 2.5, 0.4, 7.0, 7.75, 50.0, -1.0, -3.0, -4.5, -50.0];
        // The last size takes the rayon branches.
        let sizes = [(1, 1, 1), (1, 9, 2), (9, 1, 3), (3, 3, 4), (5, 7, 5), (37, 23, 6), (64, 64, 7), (130, 9, 8), (70, 520, 9)];
        for (w, h, seed) in sizes {
            let source = noise_mask(w, h, seed);
            // The oracle is O(N²) per pixel; big shifts only on small masks.
            for &shift in shifts.iter().filter(|s| s.abs() <= 10.0 || w * h <= 2_000) {
                let mut fast = source.clone();
                shift_mask(&mut fast, shift);
                assert!(fast.as_raw() == &brute_force_shift(&source, shift), "{w}x{h} shift {shift}");
            }
        }
    }

    #[test]
    fn shifted_matches_shift_mask() {
        let source = noise_mask(37, 23, 11);
        for &shift in &[1.0f32, 2.5, -3.0, 0.0] {
            let mut in_place = source.clone();
            shift_mask(&mut in_place, shift);
            assert_eq!(shifted(&source, shift).as_raw(), in_place.as_raw(), "shift {shift}");
        }
    }

    #[test]
    fn shift_zero_is_a_no_op() {
        let source = noise_mask(16, 16, 10);
        let mut m = source.clone();
        shift_mask(&mut m, 0.0);
        assert_eq!(m.as_raw(), source.as_raw());
    }
}
