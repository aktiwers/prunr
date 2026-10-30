//! Grayscale erosion / dilation of a mask by N pixels, with replicated
//! edges.
//!
//! The window is a regular octagon standing in for a disc of radius N: a
//! (2p+1)² square, itself a horizontal and a vertical 1-D window,
//! followed by the two diagonal lines of 2m+1 pixels. The vertical and
//! diagonal windows are van Herk / Gil-Werman passes over whole rows,
//! constant time per pixel: within blocks of `2r+1` rows keep running
//! prefix and suffix extrema, then every window is one `pick` of a suffix
//! and a prefix. The horizontal window stays inside one row in cache and
//! is built by doubling instead, `log₂(2p+1)` vectorised picks per pixel.
//!
//! The mask is padded by N with its nearest edge pixel and each pass
//! keeps only the windows that lie wholly inside what the last one left,
//! so the planes shrink back to the mask and every pixel gets the
//! extremum over the octagon clipped to the image
//! (`tests::brute_force_shift` is the oracle).

use image::GrayImage;
use rayon::prelude::*;

/// Masks with fewer rows run the passes on the calling thread.
const PAR_MIN_ROWS: usize = 512;

/// Erode (`shift > 0`) or dilate (`shift < 0`) a mask by `|shift|`
/// pixels, so edges move the same distance in every direction; a
/// fractional part blends between the two nearest integer shifts.
///
/// Peak working set: two planes of the mask padded by `|shift| + 1` on
/// each side, plus two rows per thread.
///
/// | mask      | shift 20 | shift 50 |
/// |-----------|----------|----------|
/// | 1024²     | 2.3 MB   | 2.5 MB   |
/// | 4096×3072 | 25.8 MB  | 26.6 MB  |
/// | 6000×4000 | 48.8 MB  | 50.1 MB  |
pub fn shift_mask(mask: &mut GrayImage, shift: f32) {
    let abs = shift.abs();
    if abs.is_nan() || abs < 0.01 || mask.width() == 0 || mask.height() == 0 {
        return;
    }
    // Past w + h the window around every pixel holds the whole image.
    let abs = abs.min(mask.width() as f32 + mask.height() as f32);
    let full = abs.floor() as usize;
    let frac = abs - full as f32;
    // Monomorphised per extremum so the elementwise loops vectorise.
    if shift > 0.0 {
        shift_by(mask, full, frac, u8::min);
    } else {
        shift_by(mask, full, frac, u8::max);
    }
}

fn shift_by<F: Fn(u8, u8) -> u8 + Copy + Sync>(mask: &mut GrayImage, full: usize, frac: f32, pick: F) {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let own = Octagon::of(full);
    if own.covers(w, h) {
        // Padding by `full` would only spend RAM: every window, and the
        // larger partner's, is the whole mask.
        let all = mask.as_raw().iter().copied().reduce(pick).unwrap_or_default();
        mask.as_mut().fill(all);
        return;
    }
    let partner = (frac >= 0.01).then(|| Octagon::of(full + 1));
    // The passes both shapes share run once on the planes; what each has
    // beyond them is at most a 3×3 square or a radius-2 diamond of taps.
    let base = partner.map_or(own, |p| Octagon { square: own.square.min(p.square), diag: own.diag.min(p.diag) });
    let margin = partner.unwrap_or(own).reach();
    let mut planes = Planes::padded(mask, margin, base.square, pick);
    planes.down(base.square, 0);
    planes.down(base.diag, 1);
    planes.down(base.diag, -1);
    planes.finish(mask, own.beyond(base), partner.map(|p| (p.beyond(base), frac)));
}

/// A square of half-width `square` followed by two diagonal lines of
/// half-length `diag`: the offsets with `|dx|, |dy| ≤ square + 2·diag`
/// and `|dx| + |dy| ≤ 2·(square + diag)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Octagon {
    square: usize,
    diag: usize,
}

impl Octagon {
    /// The octagon of radius `n`: `n` along the axes and, to the nearest
    /// pixel, `n/√2` per axis along the diagonals, which takes
    /// `square + 2·diag = n` and `diag = round(n·(1 − 1/√2))`.
    fn of(n: usize) -> Self {
        let diag = (n as f64 * (1.0 - std::f64::consts::FRAC_1_SQRT_2)).round() as usize;
        match n - 2 * diag {
            // The two lines alone reach only even dx + dy; it takes a
            // square at least 3 wide to fill the checkerboard they leave.
            0 if diag > 0 => Self { square: n, diag: 0 },
            square => Self { square, diag },
        }
    }

    fn reach(self) -> usize {
        self.square + 2 * self.diag
    }

    fn contains(self, dx: isize, dy: isize) -> bool {
        let (dx, dy) = (dx.unsigned_abs(), dy.unsigned_abs());
        dx.max(dy) <= self.reach()
            && dx + dy <= 2 * (self.square + self.diag)
            && (self.square > 0 || (dx + dy) % 2 == 0)
    }

    fn offsets(self) -> impl Iterator<Item = (isize, isize)> {
        let r = self.reach() as isize;
        (-r..=r).flat_map(move |dy| (-r..=r).map(move |dx| (dx, dy))).filter(move |&(dx, dy)| self.contains(dx, dy))
    }

    /// Whether the window around every pixel of a `w`×`h` mask holds all
    /// of it.
    fn covers(self, w: usize, h: usize) -> bool {
        self.reach() >= w.max(h) - 1 && self.square + self.diag >= (w + h - 1) / 2
    }

    /// What is left of `self` once `base` has run.
    fn beyond(self, base: Self) -> Self {
        Self { square: self.square - base.square, diag: self.diag - base.diag }
    }
}

/// The padded mask part-way through the passes, in two planes of one
/// stride: `cur` holds the result so far at rows `0..h`, columns
/// `x0..x0 + w`, and `spare` is scratch.
struct Planes<F> {
    cur: Vec<u8>,
    spare: Vec<u8>,
    stride: usize,
    x0: usize,
    w: usize,
    h: usize,
    pick: F,
    par: bool,
}

impl<F: Fn(u8, u8) -> u8 + Copy + Sync> Planes<F> {
    /// `mask` padded by `margin` on every side with its nearest edge
    /// pixel, through the horizontal window of half-width `r` on the way
    /// in.
    fn padded(mask: &GrayImage, margin: usize, r: usize, pick: F) -> Self {
        let (mw, mh) = (mask.width() as usize, mask.height() as usize);
        let (stride, rows) = (mw + 2 * margin, mh + 2 * margin);
        let par = mh >= PAR_MIN_ROWS;
        let w = stride - 2 * r;
        let mut cur = vec![0u8; stride * rows];
        let fill = |(line, scratch): &mut (Vec<u8>, Vec<u8>), (out, src): (&mut [u8], &[u8])| {
            line[..margin].fill(src[0]);
            line[margin..margin + mw].copy_from_slice(src);
            line[margin + mw..].fill(src[mw - 1]);
            valid_line(line, scratch, &mut out[..w], r, pick);
        };
        let buffers = || (vec![0u8; stride], vec![0u8; stride]);
        let body = &mut cur[margin * stride..(margin + mh) * stride];
        if par {
            body.par_chunks_mut(stride).zip(mask.as_raw().par_chunks(mw)).for_each_init(buffers, fill);
        } else {
            let mut b = buffers();
            body.chunks_mut(stride).zip(mask.as_raw().chunks(mw)).for_each(|rows| fill(&mut b, rows));
        }
        // A padding row repeats the nearest mask row, so it filters the same.
        let (top, rest) = cur.split_at_mut(margin * stride);
        top.chunks_mut(stride).for_each(|row| row.copy_from_slice(&rest[..stride]));
        let (rest, bottom) = cur.split_at_mut((margin + mh) * stride);
        let last = &rest[(margin + mh - 1) * stride..];
        bottom.chunks_mut(stride).for_each(|row| row.copy_from_slice(last));
        Self { cur, spare: vec![0u8; stride * rows], stride, x0: 0, w, h: rows, pick, par }
    }

    /// The window of `2r + 1` pixels along the line that moves `dx`
    /// columns per row down: vertical at 0, the diagonals at ±1. Within
    /// blocks of `2r + 1` rows the suffixes go to `spare` and `cur`
    /// becomes the prefixes in place; then each window is one pick of a
    /// suffix row and the prefix row `2r` below, written over the suffix.
    /// Every step is elementwise over whole rows.
    fn down(&mut self, r: usize, dx: isize) {
        if r == 0 {
            return;
        }
        let Self { stride, x0, w, h, pick, par, .. } = *self;
        let span = |y: usize| y * stride + x0..y * stride + x0 + w;
        let block = |(pre, suf): (&mut [u8], &mut [u8])| {
            let rows = pre.len() / stride;
            suf[span(rows - 1)].copy_from_slice(&pre[span(rows - 1)]);
            for y in (0..rows - 1).rev() {
                let (this, below) = suf.split_at_mut((y + 1) * stride);
                let this = &mut this[span(y)];
                this.copy_from_slice(&pre[span(y)]);
                step(this, &below[span(0)], dx, pick);
            }
            for y in 1..rows {
                let (above, this) = pre.split_at_mut(y * stride);
                step(&mut this[span(0)], &above[span(y - 1)], -dx, pick);
            }
        };
        let block_len = (2 * r + 1) * stride;
        let (cur, spare) = (&mut self.cur[..h * stride], &mut self.spare[..h * stride]);
        if par {
            cur.par_chunks_mut(block_len).zip(spare.par_chunks_mut(block_len)).for_each(block);
        } else {
            cur.chunks_mut(block_len).zip(spare.chunks_mut(block_len)).for_each(block);
        }

        let prefix: &[u8] = cur;
        let combine = |(y, out): (usize, &mut [u8])| {
            step(&mut out[span(0)], &prefix[span(y + 2 * r)], 2 * r as isize * dx, pick);
        };
        let out = &mut spare[..(h - 2 * r) * stride];
        if par {
            out.par_chunks_mut(stride).enumerate().for_each(combine);
        } else {
            out.chunks_mut(stride).enumerate().for_each(combine);
        }
        std::mem::swap(&mut self.cur, &mut self.spare);
        self.h = h - 2 * r;
        self.w = w - 2 * r * dx.unsigned_abs();
        if dx < 0 {
            self.x0 = x0 + 2 * r;
        }
    }

    /// Writes into `mask` the pick over `own` around each pixel, blended
    /// by `frac` toward the pick over the partner shape when there is one.
    fn finish(&self, mask: &mut GrayImage, own: Octagon, partner: Option<(Octagon, f32)>) {
        let (mw, mh) = (mask.width() as usize, mask.height() as usize);
        let (stride, pick, src) = (self.stride, self.pick, &self.cur);
        let edge = (self.h - mh) / 2;
        let taps = |shape: Octagon| -> Vec<usize> {
            let centre = (edge * stride + self.x0 + edge) as isize;
            shape.offsets().map(|(dx, dy)| (centre + dy * stride as isize + dx) as usize).collect()
        };
        let gather = |out: &mut [u8], taps: &[usize], y: usize| {
            let at = |t: usize| &src[y * stride + t..][..mw];
            out.copy_from_slice(at(taps[0]));
            for &t in &taps[1..] {
                for (o, &v) in out.iter_mut().zip(at(t)) {
                    *o = pick(*o, v);
                }
            }
        };
        let own = taps(own);
        let partner = partner.map(|(shape, frac)| (taps(shape), frac));
        let row = |other: &mut Vec<u8>, (y, out): (usize, &mut [u8])| {
            gather(out, &own, y);
            if let Some((taps, frac)) = &partner {
                gather(other, taps, y);
                let inv = 1.0 - frac;
                for (a, &b) in out.iter_mut().zip(other.iter()) {
                    *a = crate::math::round_u8(*a as f32 * inv + b as f32 * frac);
                }
            }
        };
        let rows = mask.as_mut();
        if self.par {
            rows.par_chunks_mut(mw).enumerate().for_each_init(|| vec![0u8; mw], row);
        } else {
            let mut b = vec![0u8; mw];
            rows.chunks_mut(mw).enumerate().for_each(|yr| row(&mut b, yr));
        }
    }
}

/// `a[i] = pick(a[i], b[i + d])` wherever `i + d` falls inside `b`.
fn step<F: Fn(u8, u8) -> u8>(a: &mut [u8], b: &[u8], d: isize, pick: F) {
    let (n, k) = (a.len(), d.unsigned_abs());
    let (a, b) = if d >= 0 { (&mut a[..n - k], &b[k..]) } else { (&mut a[k..], &b[..n - k]) };
    for (x, &y) in a.iter_mut().zip(b) {
        *x = pick(*x, y);
    }
}

/// `mask` grown by a disc of `radius` pixels (the max over a round
/// window), so thick lines get round ends and corners instead of the
/// faceted ones `shift_mask`'s octagon gives. Each row of the disc is a
/// 1-D window pass over one source row, so the cost is `2·radius + 1`
/// passes per output row (about 40 ms at 4K for radius 20).
///
/// Peak working set: one output plane plus three row buffers per thread.
pub fn dilate_round(mask: &GrayImage, radius: u32) -> GrayImage {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let r = radius as usize;
    let mut out = GrayImage::new(mask.width(), mask.height());
    if w == 0 || h == 0 {
        return out;
    }
    // Half-width of the disc at each row offset; `r + 0.5` keeps small
    // discs from turning into crosses.
    let reach = (r as f32 + 0.5).powi(2);
    let halves: Vec<usize> = (0..=r).map(|dy| (reach - (dy * dy) as f32).sqrt().floor() as usize).collect();
    let src = mask.as_raw();
    let buffers = || (vec![0u8; w], vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    let row_of = |sy: usize| &src[sy * w..(sy + 1) * w];
    let fill_row = |(pair, line, p, s): &mut (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), (y, row): (usize, &mut [u8])| {
        for (dy, &half) in halves.iter().enumerate() {
            // The rows `dy` above and below share a span: merge them first,
            // since the window max of a max is the max of the windows.
            let above = y.checked_sub(dy);
            let below = (dy > 0 && y + dy < h).then_some(y + dy);
            let src_row: &[u8] = match (above, below) {
                (Some(a), Some(b)) => {
                    for ((o, &u), &v) in pair.iter_mut().zip(row_of(a)).zip(row_of(b)) {
                        *o = u.max(v);
                    }
                    pair
                }
                (Some(a), None) => row_of(a),
                (None, Some(b)) => row_of(b),
                (None, None) => continue,
            };
            window_line(src_row, line, p, s, half, u8::max);
            for (o, &v) in row.iter_mut().zip(line.iter()) {
                *o = (*o).max(v);
            }
        }
    };
    if h >= PAR_MIN_ROWS {
        out.as_mut().par_chunks_mut(w).enumerate().for_each_init(buffers, fill_row);
    } else {
        let mut b = buffers();
        out.as_mut().chunks_mut(w).enumerate().for_each(|yr| fill_row(&mut b, yr));
    }
    out
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

/// `out[i]` = pick over `line[i..=i + 2r]`, for the windows wholly
/// inside `line`, which it overwrites. Windows of 2, 4, 8 … pixels are
/// each one pick of two halves, and the `2r + 1` window one pick of two
/// overlapping ones: a few elementwise passes over a row in cache, which
/// vectorise where van Herk's running extrema cannot.
fn valid_line<F: Fn(u8, u8) -> u8 + Copy>(line: &mut [u8], scratch: &mut [u8], out: &mut [u8], r: usize, pick: F) {
    let (n, k) = (line.len(), 2 * r + 1);
    let (mut cur, mut next) = (line, scratch);
    let mut len = 1;
    while 2 * len <= k {
        let m = n + 1 - 2 * len;
        for ((o, &a), &b) in next[..m].iter_mut().zip(&cur[..m]).zip(&cur[len..]) {
            *o = pick(a, b);
        }
        std::mem::swap(&mut cur, &mut next);
        len *= 2;
    }
    for ((o, &a), &b) in out.iter_mut().zip(&cur[..n + 1 - k]).zip(&cur[k - len..]) {
        *o = pick(a, b);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    /// A shift that is not a number leaves the mask; one past the image's
    /// size works like one of its size: everything eroded or grown.
    #[test]
    fn a_shift_not_a_number_or_past_the_image_is_sound() {
        let disc = GrayImage::from_fn(9, 7, |x, y| image::Luma([if (x as i32 - 4).pow(2) + (y as i32 - 3).pow(2) < 5 { 255 } else { 0 }]));
        let mut m = disc.clone();
        shift_mask(&mut m, f32::NAN);
        assert_eq!(m, disc, "NaN shifts nothing");
        for (shift, want) in [(1e30, 0u8), (f32::INFINITY, 0), (-1e30, 255), (f32::NEG_INFINITY, 255)] {
            let mut m = disc.clone();
            shift_mask(&mut m, shift);
            assert!(m.pixels().all(|p| p[0] == want), "{shift}");
        }
    }

    /// A dot grows into a disc: round, not square, and grey levels keep
    /// their maximum.
    #[test]
    fn a_round_dilation_grows_a_disc() {
        let mut m = GrayImage::new(15, 15);
        m.put_pixel(7, 7, image::Luma([200]));
        let d = dilate_round(&m, 4);
        assert_eq!(d.get_pixel(7 + 4, 7)[0], 200, "the radius along an axis");
        assert_eq!(d.get_pixel(7 + 2, 7 + 3)[0], 200, "inside the disc");
        assert_eq!(d.get_pixel(7 + 4, 7 + 4)[0], 0, "the square's corner is outside the disc");
        assert_eq!(d.get_pixel(7 + 5, 7)[0], 0);
        assert_eq!(dilate_round(&m, 0).as_raw(), m.as_raw(), "radius 0 changes nothing");
    }

    /// The octagon reaches N along the axes and N/√2 per axis along the
    /// diagonals, to half a pixel; its square fills the checkerboard the
    /// diagonal lines leave; and each radius holds the one below, so a
    /// fractional shift blends between nested shapes.
    #[test]
    fn the_octagon_reaches_n_along_axes_and_diagonals() {
        for n in 0..=500 {
            let o = Octagon::of(n);
            assert_eq!(o.reach(), n, "{o:?}");
            assert!(o.diag == 0 || o.square >= 1, "{n}: {o:?}");
            if n >= 3 {
                let corner = (o.square + o.diag) as f64;
                assert!((corner - n as f64 * FRAC_1_SQRT_2).abs() <= 0.5, "{n}: {o:?}");
            }
            let next = Octagon::of(n + 1);
            assert!(next.square + next.diag >= o.square + o.diag, "{n}: {o:?} then {next:?}");
        }
    }

    /// A dot dilated by N reaches N along the axes and N/√2 along the
    /// diagonals, where a square window would reach (N, N), and covers an
    /// octagon's area.
    #[test]
    fn a_dot_grows_into_an_octagon() {
        for n in [4i64, 7, 12, 20] {
            let c = n + 3;
            let size = (2 * c + 1) as u32;
            let mut m = GrayImage::new(size, size);
            m.put_pixel(c as u32, c as u32, image::Luma([255]));
            shift_mask(&mut m, -(n as f32));
            let at = |dx: i64, dy: i64| m.get_pixel((c + dx) as u32, (c + dy) as u32)[0];
            for (dx, dy) in [(n, 0), (-n, 0), (0, n), (0, -n)] {
                assert_eq!(at(dx, dy), 255, "N = {n}: axis ({dx}, {dy})");
            }
            assert_eq!(at(n + 1, 0), 0, "N = {n}: past the axis");
            assert_eq!(at(n, n), 0, "N = {n}: the square's corner");
            let d = (n as f64 * FRAC_1_SQRT_2).round() as i64;
            for (sx, sy) in [(1, 1), (-1, 1), (1, -1), (-1, -1)] {
                assert_eq!(at(sx * (d - 1), sy * (d - 1)), 255, "N = {n}: inside the diagonal");
                assert_eq!(at(sx * (d + 1), sy * (d + 1)), 0, "N = {n}: past the diagonal");
            }
            // Pixel centres within the sides at N + 0.5 and, diagonally,
            // at |dx| + |dy| = 2·(square + diag) + 0.5.
            let o = Octagon::of(n as usize);
            let (side, cut) = (n as f64 + 0.5, 2.0 * (o.square + o.diag) as f64 + 0.5);
            let octagon = 4.0 * side * side - 2.0 * (2.0 * side - cut).powi(2);
            let area = m.pixels().filter(|p| p[0] == 255).count() as f64;
            assert!((area / octagon - 1.0).abs() < 0.02, "N = {n}: area {area} against {octagon}");
        }
    }

    /// A subject that runs off the image keeps its edge there: eroding a
    /// corner quadrant moves only its inner edges, by N.
    #[test]
    fn an_erode_keeps_the_subject_at_the_image_border() {
        let (w, h, n) = (40u32, 30u32, 6u32);
        let quadrant = GrayImage::from_fn(w, h, |x, y| image::Luma([if x >= 20 && y >= 15 { 255 } else { 0 }]));
        let mut m = quadrant.clone();
        shift_mask(&mut m, n as f32);
        for (x, y, p) in m.enumerate_pixels() {
            let want = if x >= 20 + n && y >= 15 + n { 255 } else { 0 };
            assert_eq!(p[0], want, "({x}, {y})");
        }
    }

    /// The spec: `pick` over the octagon of radius N clipped to the mask,
    /// then the same blend as production between N and N + 1.
    fn brute_force_shift(mask: &GrayImage, shift: f32) -> Vec<u8> {
        let abs = shift.abs();
        let pick: fn(u8, u8) -> u8 = if shift > 0.0 { u8::min } else { u8::max };
        let full = abs.floor() as i64;
        let frac = abs - full as f32;
        let (w, h) = (mask.width() as i64, mask.height() as i64);
        let src = mask.as_raw();
        let window = |n: i64| -> Vec<u8> {
            let shape = Octagon::of(n as usize);
            let l1 = 2 * (shape.square + shape.diag) as i64;
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    let mut acc = src[(y * w + x) as usize];
                    for ny in (y - n).max(0)..=(y + n).min(h - 1) {
                        let half = n.min(l1 - (ny - y).abs());
                        let (lo, hi) = ((x - half).max(0), (x + half).min(w - 1));
                        if lo <= hi {
                            let row = &src[(ny * w) as usize..][lo as usize..=hi as usize];
                            acc = row.iter().copied().fold(acc, pick);
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
    fn shift_matches_the_brute_force_octagon() {
        let shifts = [
            0.4f32, 1.0, 2.0, 3.0, 7.0, 12.0, 40.0, -1.0, -2.0, -3.0, -7.0, -12.0, -40.0, 2.5, 5.5, 7.75, -4.5, -12.3,
            40.5,
        ];
        // 40 outgrows most of these; the last size takes the rayon branches.
        let sizes = [
            (1, 1, 1),
            (1, 9, 2),
            (9, 1, 3),
            (2, 2, 4),
            (3, 3, 5),
            (4, 6, 6),
            (5, 7, 7),
            (37, 23, 8),
            (64, 64, 9),
            (130, 9, 10),
            (200, 150, 11),
            (70, 520, 12),
        ];
        for (w, h, seed) in sizes {
            let source = noise_mask(w, h, seed);
            // The oracle costs N² per pixel, twice with a fraction.
            let affordable = |s: &&f32| (w * h) as f32 * s.powi(2) * if s.fract() == 0.0 { 1.0 } else { 2.0 } <= 5e7;
            for &shift in shifts.iter().filter(affordable) {
                let mut fast = source.clone();
                shift_mask(&mut fast, shift);
                assert!(fast.as_raw() == &brute_force_shift(&source, shift), "{w}x{h} shift {shift}");
            }
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
