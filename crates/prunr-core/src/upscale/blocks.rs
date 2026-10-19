//! Upscaling with a network that normalizes over its whole input
//! (RealPLKSR's GroupNorm, in Nomos2). Tiled the usual way, each tile is
//! normalized on its own statistics and neighbours come out toned apart:
//! seams that no overlap hides, since every pixel depends on the whole
//! tile (Buglakova et al., ICCV 2025). So the model is cut at its norms
//! (a head, each block up to its norm, a tail) and run part by part: each
//! part crop by crop, with a border wide enough that a crop's middle is
//! exact, and each norm on the whole image's statistics, as a one-piece
//! run has them.
//!
//! Peak working set, per input pixel with 64 f32 feature channels: the
//! blocks hold the image (12 B), the features and one block's output
//! before its norm (256 B each), 524 B; the tail the image, the features
//! and the 4× output (460 B); the pack the output and its bytes (240 B).
//! The parts and one crop through one add ~0.73 GB (`working_set_mb`).
//! Measured on the CPU provider (i7-6700):
//!
//! | Input  | Planes  | Peak                     |
//! |--------|---------|--------------------------|
//! | 0.6 MP | 0.32 GB | 0.99 GB                  |
//! | 1.4 MP | 0.72 GB | 1.32 GB                  |
//! | 8 MP   | 4.2 GB  | ~4.3 GB (from the above) |

use std::sync::Mutex;

use image::{RgbImage, RgbaImage};
use ort::session::Session;
use prunr_models::{PartLayout, PART_HEAD, PART_TAIL};
use rayon::prelude::*;

use crate::progress::{Progress, Step, TileRect, Unit};
use crate::CoreError;

/// A crop's exact middle, at most, in input pixels. Smaller crops ran
/// faster on the CPU provider (67 s at 256 against 76 s in one crop) and
/// keep one part's scratch small.
const CROP: usize = 256;

/// What a part-by-part upscale of `w × h` by `scale` holds in its planes
/// at its peak (see the module doc); the parts and a crop's scratch are
/// the model's `working_set_mb`.
pub(crate) fn peak_bytes(layout: PartLayout, w: u32, h: u32, scale: u32) -> u64 {
    let px = u64::from(w) * u64::from(h);
    let (image, features, output) = (3 * 4, layout.features as u64 * 4, u64::from(scale).pow(2) * 3 * 4);
    let blocks = image + 2 * features;
    let tail = image + features + output;
    let pack = output + output / 4;
    px * blocks.max(tail).max(pack)
}

/// What the runner knows of a parted network besides its files.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NetShape {
    pub layout: PartLayout,
    /// The image's input name, at the head and again at the tail.
    pub image_input: &'static str,
}

/// What the runner needs of a parted network; `BlockNet` runs the real
/// parts, tests a stand-in. Planes are channel-major (C × h × w).
pub(crate) trait Stages {
    fn layout(&self) -> PartLayout;
    fn blocks(&self) -> usize;
    fn groups(&self) -> usize;
    fn eps(&self) -> f32;
    /// Block `k`'s norm weight and bias, one per channel.
    fn norm(&self, k: usize) -> (&[f32], &[f32]);
    /// The image (3 planes, 0-1) to the features.
    fn head(&self, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError>;
    /// Block `k`'s output before its norm.
    fn block(&self, k: usize, features: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError>;
    /// The last features and the image to the output (3 planes at `scale`×).
    fn tail(&self, features: Vec<f32>, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError>;
}

/// A span of one axis a part runs on, and the part of it that comes out
/// exact: all of it at an image edge, `border` in from a side with a
/// neighbour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    keep_start: usize,
    keep_end: usize,
}

/// Crops along an axis of `len`: the axis cut into equal kept parts of
/// at most `crop`, each run with `border` around it, every crop one size
/// (at an image edge it reaches further in instead) so the model compiles
/// one shape. Equal parts keep the work run twice to the borders.
fn spans(len: usize, crop: usize, border: usize) -> Vec<Span> {
    let n = len.div_ceil(crop.max(1));
    let kept = len.div_ceil(n);
    let side = (kept + 2 * border).min(len);
    (0..n).map(|i| {
        let keep_start = i * kept;
        let keep_end = (keep_start + kept).min(len);
        let start = keep_start.saturating_sub(border).min(len - side);
        Span { start, end: start + side, keep_start, keep_end }
    }).collect()
}

/// Every crop of a `w × h` image for a part reading `border` around a
/// pixel, row by row. Where they are kept does not depend on `border`, so
/// every stage lays the same cells.
fn crops(w: usize, h: usize, border: usize) -> Vec<(Span, Span)> {
    let xs = spans(w, CROP, border);
    spans(h, CROP, border).into_iter().flat_map(|y| xs.iter().map(move |&x| (x, y))).collect()
}

/// The kept part of a crop, as x, y, w, h of the image.
fn kept_rect((x, y): (Span, Span)) -> (u32, u32, u32, u32) {
    (x.keep_start as u32, y.keep_start as u32, (x.keep_end - x.keep_start) as u32, (y.keep_end - y.keep_start) as u32)
}

/// The crop `(x, y)` of each of `planes.len() / (w × h)` planes of a
/// `w`-wide image.
fn cut(planes: &[f32], w: usize, h: usize, (x, y): (Span, Span)) -> Vec<f32> {
    let channels = planes.len() / (w * h);
    let (cw, ch) = (x.end - x.start, y.end - y.start);
    let mut out = Vec::with_capacity(channels * cw * ch);
    for c in 0..channels {
        for row in y.start..y.end {
            let at = (c * h + row) * w;
            out.extend_from_slice(&planes[at + x.start..at + x.end]);
        }
    }
    out
}

/// Write the kept part of `result` (a crop's output at `scale`×) into
/// `out` (`channels` planes of the `w × h` image at `scale`×).
fn keep(out: &mut [f32], result: &[f32], w: usize, h: usize, scale: usize, (x, y): (Span, Span)) {
    let (ow, oh) = (w * scale, h * scale);
    let channels = out.len() / (ow * oh);
    let (cw, ch) = ((x.end - x.start) * scale, (y.end - y.start) * scale);
    let (kx0, kx1) = ((x.keep_start - x.start) * scale, (x.keep_end - x.start) * scale);
    for c in 0..channels {
        for row in (y.keep_start - y.start) * scale..(y.keep_end - y.start) * scale {
            let src = (c * ch + row) * cw;
            let dst = (c * oh + y.start * scale + row) * ow + x.start * scale;
            out[dst + kx0..dst + kx1].copy_from_slice(&result[src + kx0..src + kx1]);
        }
    }
}

/// A part's output when it is `len` values long; a part of another shape
/// errs here instead of writing past its crop.
fn sized(out: Vec<f32>, len: usize) -> Result<Vec<f32>, CoreError> {
    if out.len() == len {
        Ok(out)
    } else {
        Err(CoreError::Inference(format!("a part gave {} values where its crop takes {len}", out.len())))
    }
}

/// Pixel `i` of 3 planes of `plane` values each (0-1), as bytes.
fn rgb_at(planes: &[f32], plane: usize, i: usize) -> image::Rgb<u8> {
    image::Rgb(std::array::from_fn(|c| crate::math::round_u8(planes[c * plane + i] * 255.0)))
}

/// The kept part of a `cw × ch` crop's output at `scale`×, as an image.
fn kept_rgb(out: &[f32], (cw, ch): (usize, usize), scale: usize, (x, y): (Span, Span)) -> RgbImage {
    let (ow, plane) = (cw * scale, cw * ch * scale * scale);
    let (x0, y0) = ((x.keep_start - x.start) * scale, (y.keep_start - y.start) * scale);
    let (kw, kh) = ((x.keep_end - x.keep_start) * scale, (y.keep_end - y.keep_start) * scale);
    RgbImage::from_fn(kw as u32, kh as u32, |px, py| rgb_at(out, plane, (y0 + py as usize) * ow + x0 + px as usize))
}

/// `features += γ ⊙ (pre − μ) / √(σ² + ε) + β`, per channel, with μ and
/// σ² over the whole image for each of `groups` runs of channels: what
/// GroupNorm and the block's skip compute on a one-piece run.
fn add_normalized(features: &mut [f32], pre: &[f32], plane: usize, groups: usize, eps: f32, (gamma, beta): (&[f32], &[f32])) {
    let per_group = pre.len() / groups;
    let stats: Vec<(f64, f64)> = pre.par_chunks(per_group).map(|g| {
        let (sum, sq) = g.iter().fold((0.0f64, 0.0f64), |(s, q), &v| (s + f64::from(v), q + f64::from(v) * f64::from(v)));
        let n = g.len() as f64;
        let mean = sum / n;
        (mean, (sq / n - mean * mean).max(0.0))
    }).collect();
    let channels_per_group = per_group / plane;
    features.par_chunks_mut(plane).zip(pre.par_chunks(plane)).enumerate().for_each(|(c, (f, p))| {
        let (mean, var) = stats[c / channels_per_group];
        let scale = gamma[c] / (var + f64::from(eps)).sqrt() as f32;
        let shift = beta[c] - mean as f32 * scale;
        f.iter_mut().zip(p).for_each(|(f, &p)| *f += p * scale + shift);
    });
}

/// Run `net` over `input` part by part (see the module doc); the output
/// is `scale`× the input, RGB, alpha left to the caller. The tail's crops
/// are its finished pixels, so they go out as previews.
pub(crate) fn upscale_by_parts(net: &impl Stages, input: &RgbaImage, scale: u32, progress: &Progress) -> Result<RgbImage, CoreError> {
    let (w, h) = (input.width() as usize, input.height() as usize);
    let plane = w * h;
    let mut image = vec![0.0f32; 3 * plane];
    for (i, p) in input.pixels().enumerate() {
        for c in 0..3 {
            image[c * plane + i] = f32::from(p.0[c]) / 255.0;
        }
    }
    let layout = net.layout();
    let channels = layout.features;
    let stages = Stage { total: net.blocks() as u32 + 2, w, h };
    progress.step(Step::Upscaling);

    let mut features = vec![0.0f32; channels * plane];
    stages.each(progress, 0, layout.head, |crop, (cw, ch)| {
        let out = sized(net.head(cut(&image, w, h, crop), cw, ch)?, channels * cw * ch)?;
        keep(&mut features, &out, w, h, 1, crop);
        Ok(())
    })?;
    let mut pre = vec![0.0f32; channels * plane];
    for k in 0..net.blocks() {
        stages.each(progress, k as u32 + 1, layout.block, |crop, (cw, ch)| {
            let out = sized(net.block(k, cut(&features, w, h, crop), cw, ch)?, channels * cw * ch)?;
            keep(&mut pre, &out, w, h, 1, crop);
            Ok(())
        })?;
        add_normalized(&mut features, &pre, plane, net.groups(), net.eps(), net.norm(k));
    }
    drop(pre);
    let s = scale as usize;
    let mut rgb = vec![0.0f32; 3 * plane * s * s];
    stages.each(progress, stages.total - 1, layout.tail, |crop, (cw, ch)| {
        let out = sized(net.tail(cut(&features, w, h, crop), cut(&image, w, h, crop), cw, ch)?, 3 * cw * ch * s * s)?;
        keep(&mut rgb, &out, w, h, s, crop);
        let rect = kept_rect(crop);
        progress.tile_preview(rect, (w as u32, h as u32), |pw, ph| super::tiling::preview_with_alpha(input, &kept_rgb(&out, (cw, ch), s, crop), rect, pw, ph));
        Ok(())
    })?;
    progress.outer(stages.total, stages.total, Unit::Step);
    drop((features, image));
    progress.step(Step::Finishing);
    let (ow, oh) = (w * s, h * s);
    Ok(RgbImage::from_fn(ow as u32, oh as u32, |x, y| rgb_at(&rgb, ow * oh, y as usize * ow + x as usize)))
}

/// The run's stages (head, blocks, tail) over a `w × h` image.
struct Stage {
    total: u32,
    w: usize,
    h: usize,
}

impl Stage {
    /// Stage `stage`, crop by crop for a part reading `border` around a
    /// pixel: `run` gets each crop and its size. Cancel stops it between crops.
    fn each(&self, progress: &Progress, stage: u32, border: usize, mut run: impl FnMut((Span, Span), (usize, usize)) -> Result<(), CoreError>) -> Result<(), CoreError> {
        let (w, h) = (self.w, self.h);
        let crops = crops(w, h, border);
        let n = crops.len() as u32;
        progress.outer(stage, self.total, Unit::Step);
        progress.tiles(crops.iter().map(|&crop| {
            let (x, y, cw, ch) = kept_rect(crop);
            TileRect::of_pixels(x, y, cw, ch, w as u32, h as u32)
        }).collect());
        progress.inner(0, n, Unit::Tile);
        for (i, crop) in crops.into_iter().enumerate() {
            progress.check_cancelled()?;
            progress.tile(i as u32, false);
            run(crop, (crop.0.end - crop.0.start, crop.1.end - crop.1.start))?;
            progress.tile(i as u32, true);
            progress.inner(i as u32 + 1, n, Unit::Tile);
        }
        Ok(())
    }
}

/// One part's session and the names of what it takes and gives.
struct Part {
    session: Mutex<Session>,
    inputs: Vec<String>,
    output: String,
}

impl Part {
    fn of(key: &str, session: Session) -> Result<Self, CoreError> {
        let inputs: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let output = session.outputs().first().map(|o| o.name().to_string())
            .ok_or_else(|| CoreError::Model(format!("part {key} gives nothing")))?;
        if inputs.is_empty() {
            return Err(CoreError::Model(format!("part {key} takes nothing")));
        }
        Ok(Self { session: Mutex::new(session), inputs, output })
    }

    /// Run on `feeds` (input name, channels, data), each `w × h`;
    /// `terminate` aborts it mid-run.
    fn run(&self, feeds: Vec<(&str, usize, Vec<f32>)>, w: usize, h: usize, terminate: Option<&super::UpscaleRunOptions>) -> Result<Vec<f32>, CoreError> {
        let inputs = feeds.into_iter().map(|(name, c, data)| {
            let array = ndarray::Array4::from_shape_vec([1, c, h, w], data)
                .map_err(|e| CoreError::Inference(format!("part input shape: {e}")))?;
            let tensor = ort::value::Tensor::from_array(array).map_err(|e| CoreError::Inference(format!("part input: {e}")))?;
            Ok((name, tensor))
        }).collect::<Result<Vec<_>, CoreError>>()?;
        let mut session = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = match terminate {
            Some(opts) => session.run_with_options(inputs, opts),
            None => session.run(inputs),
        }
        .map_err(super::classify_run_error)?;
        let output = outputs.get(self.output.as_str())
            .ok_or_else(|| CoreError::Inference(format!("part gave no {}", self.output)))?;
        let (_, values) = output.try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(format!("part {}: output: {e}", self.output)))?;
        Ok(values.to_vec())
    }
}

/// Each block's norm: its groups and epsilon, and per block a weight and
/// a bias per channel.
#[derive(serde::Deserialize)]
pub(crate) struct Norms {
    groups: usize,
    eps: f32,
    gamma: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
}

impl Norms {
    /// The norms' JSON, checked against the network's `features` channels.
    pub(crate) fn parse(bytes: &[u8], features: usize) -> Result<Self, CoreError> {
        let norms: Self = serde_json::from_slice(bytes).map_err(|e| CoreError::Model(format!("norms: {e}")))?;
        let rows_fit = norms.gamma.len() == norms.beta.len() && norms.gamma.iter().chain(&norms.beta).all(|row| row.len() == features);
        if norms.groups == 0 || !features.is_multiple_of(norms.groups) || !rows_fit {
            return Err(CoreError::Model(format!(
                "norms: {} groups over {features} channels, {} weight and {} bias rows",
                norms.groups, norms.gamma.len(), norms.beta.len(),
            )));
        }
        Ok(norms)
    }
}

/// The real parts: a head, its blocks in key order, a tail, the norms.
pub(crate) struct BlockNet {
    head: Part,
    blocks: Vec<Part>,
    tail: Part,
    norms: Norms,
    shape: NetShape,
}

impl BlockNet {
    pub(crate) fn new(parts: Vec<(&'static str, Session)>, norms: Norms, shape: NetShape) -> Result<Self, CoreError> {
        let (mut head, mut tail, mut blocks) = (None, None, Vec::new());
        for (key, session) in parts {
            let part = Part::of(key, session)?;
            match key {
                PART_HEAD => head = Some(part),
                PART_TAIL => tail = Some(part),
                block => blocks.push((block, part)),
            }
        }
        blocks.sort_by_key(|(key, _)| *key);
        let blocks: Vec<Part> = blocks.into_iter().map(|(_, p)| p).collect();
        let missing = |what: &str| CoreError::Model(format!("parted model has no {what}"));
        let (head, tail) = (head.ok_or_else(|| missing(PART_HEAD))?, tail.ok_or_else(|| missing(PART_TAIL))?);
        let image = shape.image_input;
        let fits = head.inputs == [image]
            && blocks.iter().all(|b| b.inputs.len() == 1)
            && tail.inputs.len() == 2
            && tail.inputs.iter().filter(|name| *name == image).count() == 1
            && norms.gamma.len() == blocks.len();
        if !fits {
            return Err(CoreError::Model(format!(
                "parts do not fit: a head taking {:?}, a tail taking {:?}, {} blocks for {} norms, the image as {image:?}",
                head.inputs, tail.inputs, blocks.len(), norms.gamma.len(),
            )));
        }
        Ok(Self { head, blocks, tail, norms, shape })
    }
}

/// A run of `net`'s parts, each abortable through `terminate`.
pub(crate) struct Running<'a> {
    pub net: &'a BlockNet,
    pub terminate: Option<&'a super::UpscaleRunOptions>,
}

impl Stages for Running<'_> {
    fn layout(&self) -> PartLayout { self.net.shape.layout }
    fn blocks(&self) -> usize { self.net.blocks.len() }
    fn groups(&self) -> usize { self.net.norms.groups }
    fn eps(&self) -> f32 { self.net.norms.eps }
    // `k` < blocks, as many as the norms' rows (`BlockNet::new`).
    fn norm(&self, k: usize) -> (&[f32], &[f32]) { (&self.net.norms.gamma[k], &self.net.norms.beta[k]) }

    fn head(&self, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        self.net.head.run(vec![(self.net.shape.image_input, 3, image)], w, h, self.terminate)
    }

    fn block(&self, k: usize, features: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        let part = &self.net.blocks[k];
        // One input each (`BlockNet::new`).
        part.run(vec![(part.inputs[0].as_str(), self.layout().features, features)], w, h, self.terminate)
    }

    fn tail(&self, features: Vec<f32>, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        let tail = &self.net.tail;
        let image_input = self.net.shape.image_input;
        let other = tail.inputs.iter().find(|name| *name != image_input).map_or("", String::as_str);
        tail.run(vec![(other, self.layout().features, features), (image_input, 3, image)], w, h, self.terminate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crops' kept parts cover each axis, every crop one size, with
    /// a border where a neighbour reads across.
    #[test]
    fn crops_cover_the_image_at_one_size() {
        for (len, border) in [(960usize, 12usize), (635, 12), (280, 12), (100, 12), (1000, 2), (5000, 16)] {
            let s = spans(len, CROP, border);
            let side = s[0].end - s[0].start;
            assert!(s.iter().all(|sp| sp.end - sp.start == side), "{len}: one size");
            assert!(s.iter().all(|sp| sp.keep_end - sp.keep_start <= CROP), "{len}: no kept part past a crop");
            let mut covered = vec![false; len];
            for sp in &s {
                assert!(sp.keep_start >= sp.start && sp.keep_end <= sp.end && sp.keep_start < sp.keep_end);
                assert!(sp.start == 0 || sp.keep_start - sp.start >= border, "{len}: a border before an inner edge");
                assert!(sp.end == len || sp.end - sp.keep_end >= border, "{len}: a border after an inner edge");
                covered[sp.keep_start..sp.keep_end].iter_mut().for_each(|c| *c = true);
            }
            assert!(covered.iter().all(|&c| c), "{len}: every pixel kept once at least");
        }
    }

    /// GroupNorm and the skip, against a direct computation.
    #[test]
    fn the_norm_uses_each_groups_whole_image_statistics() {
        let plane = 3;
        let pre: Vec<f32> = vec![1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 0.0, 4.0, 8.0, 1.0, 1.0, 1.0];
        let mut features = vec![10.0f32; 12];
        let (gamma, beta) = (vec![1.0, 2.0, 1.0, 1.0], vec![0.0, 0.0, 0.5, 0.0]);
        add_normalized(&mut features, &pre, plane, 2, 0.0, (&gamma, &beta));
        // Group 0: 1,2,3,5,5,5 → mean 3.5, var 2.5833…
        let (m, sd) = (3.5f32, (2.583_333_3f32).sqrt());
        assert!((features[0] - (10.0 + (1.0 - m) / sd)).abs() < 1e-5);
        assert!((features[3] - (10.0 + 2.0 * (5.0 - m) / sd)).abs() < 1e-5);
        // Group 1: 0,4,8,1,1,1 → mean 2.5
        let g1: [f32; 6] = [0.0, 4.0, 8.0, 1.0, 1.0, 1.0];
        let mean = g1.iter().sum::<f32>() / 6.0;
        let sd1 = (g1.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 6.0).sqrt();
        assert!((features[6] - (10.0 + (0.0 - mean) / sd1 + 0.5)).abs() < 1e-5);
    }

    /// A stand-in shaped like the real one: local filters of the parts'
    /// reach and GroupNorm between blocks. Run part by part in crops, it
    /// must give what one pass over the whole image gives.
    struct Toy {
        gamma: Vec<f32>,
        beta: Vec<f32>,
        /// Channels the head gives; 4 fits the network.
        head_out: usize,
    }

    impl Toy {
        fn new(gamma: Vec<f32>, beta: Vec<f32>) -> Self {
            Self { gamma, beta, head_out: 4 }
        }
    }

    /// A 3 × 3 box mean, zero outside, `times` over.
    fn blur(plane: &[f32], w: usize, h: usize, times: usize) -> Vec<f32> {
        let mut cur = plane.to_vec();
        for _ in 0..times {
            let mut next = vec![0.0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let (yy, xx) = (y as i32 + dy, x as i32 + dx);
                            if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                                s += cur[yy as usize * w + xx as usize];
                            }
                        }
                    }
                    next[y * w + x] = s / 9.0 + 0.1;
                }
            }
            cur = next;
        }
        cur
    }

    impl Stages for Toy {
        fn layout(&self) -> PartLayout { PartLayout { features: 4, head: 1, block: 3, tail: 2 } }
        fn blocks(&self) -> usize { 3 }
        fn groups(&self) -> usize { 2 }
        fn eps(&self) -> f32 { 1e-5 }
        fn norm(&self, _: usize) -> (&[f32], &[f32]) { (&self.gamma, &self.beta) }
        fn head(&self, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
            let p = w * h;
            Ok((0..self.head_out).flat_map(|c| blur(&image[(c % 3) * p..(c % 3 + 1) * p], w, h, 1)).collect())
        }
        fn block(&self, k: usize, f: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
            let p = w * h;
            Ok((0..4).flat_map(|c| blur(&f[c * p..(c + 1) * p], w, h, 3).into_iter().map(move |v| v * (1.0 + k as f32 + c as f32))).collect())
        }
        fn tail(&self, f: Vec<f32>, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
            // 2× nearest of a blurred mix, plus the image.
            let p = w * h;
            let mut out = vec![0.0; 3 * 4 * p];
            for c in 0..3 {
                let mixed: Vec<f32> = f[c * p..(c + 1) * p].iter().zip(&image[c * p..(c + 1) * p]).map(|(a, b)| a * 0.01 + b).collect();
                let b = blur(&mixed, w, h, 2);
                for y in 0..2 * h {
                    for x in 0..2 * w {
                        out[c * 4 * p + y * 2 * w + x] = b[(y / 2) * w + x / 2].clamp(0.0, 1.0);
                    }
                }
            }
            Ok(out)
        }
    }

    fn toy_input(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 7 % 255) as u8, (y * 3 % 255) as u8, ((x * y) % 251) as u8, 255]))
    }

    /// What the stand-in gives in one pass: the same runner with crops as
    /// large as the image.
    fn one_pass(net: &Toy, input: &RgbaImage) -> RgbImage {
        let (w, h) = (input.width() as usize, input.height() as usize);
        let p = w * h;
        let image: Vec<f32> = (0..3).flat_map(|c| input.pixels().map(move |px| f32::from(px.0[c]) / 255.0)).collect();
        let mut f = net.head(image.clone(), w, h).unwrap();
        for k in 0..3 {
            let pre = net.block(k, f.clone(), w, h).unwrap();
            add_normalized(&mut f, &pre, p, 2, 1e-5, net.norm(k));
        }
        let rgb = net.tail(f, image, w, h).unwrap();
        RgbImage::from_fn(2 * w as u32, 2 * h as u32, |x, y| image::Rgb(std::array::from_fn(|c| crate::math::round_u8(rgb[c * 4 * p + y as usize * 2 * w + x as usize] * 255.0))))
    }

    /// Part by part in crops matches one pass over the whole image: no
    /// seam where crops meet, whatever the image's size against a crop.
    #[test]
    fn part_by_part_matches_one_pass() {
        let net = Toy::new(vec![0.5, 1.5, 1.0, 0.8], vec![0.1, -0.2, 0.0, 0.3]);
        for (w, h) in [(300u32, 270u32), (600, 90), (100, 100)] {
            let input = toy_input(w, h);
            let parts = upscale_by_parts(&net, &input, 2, &Progress::none()).unwrap();
            let whole = one_pass(&net, &input);
            let worst = parts.as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(worst <= 1, "{w}×{h}: off by {worst}");
        }
    }

    /// The run reports a step, each stage and each crop, within their totals.
    #[test]
    fn part_by_part_keeps_the_progress_contract() {
        use crate::progress::{check_contract, RecordingSink};
        let net = Toy::new(vec![1.0; 4], vec![0.0; 4]);
        let sink = std::sync::Arc::new(RecordingSink::default());
        upscale_by_parts(&net, &toy_input(300, 270), 2, &Progress::new(sink.clone())).unwrap();
        check_contract(&sink.updates()).unwrap();
    }

    /// Cancel stops the run at the next crop.
    #[test]
    fn a_cancelled_run_stops() {
        let net = Toy::new(vec![1.0; 4], vec![0.0; 4]);
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let r = upscale_by_parts(&net, &toy_input(300, 270), 2, &Progress::none().with_cancel(flag));
        assert!(matches!(r, Err(CoreError::Cancelled)));
    }

    /// A part that gives another shape than its crop fails the run
    /// instead of writing past the crop.
    #[test]
    fn a_part_of_the_wrong_shape_fails_the_run() {
        let net = Toy { head_out: 3, ..Toy::new(vec![1.0; 4], vec![0.0; 4]) };
        let r = upscale_by_parts(&net, &toy_input(100, 100), 2, &Progress::none());
        assert!(matches!(r, Err(CoreError::Inference(_))), "{:?}", r.err());
    }

    /// Asked for previews, the tail sends each finished crop once, in its
    /// place, with the input's alpha; together they cover the image.
    #[test]
    fn the_tail_shows_each_finished_crop() {
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        let net = Toy::new(vec![1.0; 4], vec![0.0; 4]);
        let mut input = toy_input(600, 300);
        input.pixels_mut().for_each(|p| p.0[3] = 128);
        let sink = std::sync::Arc::new(RecordingSink::default());
        upscale_by_parts(&net, &input, 2, &Progress::new(sink.clone()).with_tile_previews(300)).unwrap();
        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        let previews: Vec<_> = updates.iter().filter_map(|u| match u { ProgressUpdate::TilePreview(p) => Some(p), _ => None }).collect();
        assert_eq!(previews.len(), crops(600, 300, net.layout().tail).len(), "one per crop of the tail");
        let mut covered = vec![false; 300 * 150];
        for p in &previews {
            assert_eq!((p.full_width, p.full_height), (300, 150), "the longer side fits the asked size");
            for (i, px) in p.rgba.chunks(4).enumerate() {
                assert!(px[3].abs_diff(128) <= 1, "{px:?}");
                covered[((p.y + i as u32 / p.width) * 300 + p.x + i as u32 % p.width) as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "the crops cover the preview");
        let quiet = std::sync::Arc::new(RecordingSink::default());
        upscale_by_parts(&net, &input, 2, &Progress::new(quiet.clone())).unwrap();
        assert!(!quiet.updates().iter().any(|u| matches!(u, ProgressUpdate::TilePreview(_))), "no preview unless asked");
    }

    /// The peak is the largest stage: the blocks with many features, the
    /// tail or the pack with few.
    #[test]
    fn the_peak_is_the_largest_stage() {
        let layout = PartLayout { features: 64, head: 2, block: 12, tail: 16 };
        assert_eq!(peak_bytes(layout, 1000, 1000, 4), 1_000_000 * (12 + 2 * 256));
        assert_eq!(peak_bytes(PartLayout { features: 16, ..layout }, 10, 10, 4), 100 * (12 + 64 + 192), "the tail");
        assert_eq!(peak_bytes(PartLayout { features: 8, ..layout }, 10, 10, 4), 100 * (192 + 48), "the pack");
    }

    /// Norms that do not fit the network are refused before any part runs.
    #[test]
    fn norms_that_do_not_fit_the_network_are_refused() {
        let json = |groups: usize, row: usize| {
            let row = format!("[{}]", vec!["1.0"; row].join(","));
            format!(r#"{{"groups":{groups},"eps":1e-5,"gamma":[{row}],"beta":[{row}]}}"#)
        };
        assert!(Norms::parse(json(2, 4).as_bytes(), 4).is_ok());
        assert!(Norms::parse(json(0, 4).as_bytes(), 4).is_err(), "no groups");
        assert!(Norms::parse(json(3, 4).as_bytes(), 4).is_err(), "groups that do not divide the channels");
        assert!(Norms::parse(json(2, 3).as_bytes(), 4).is_err(), "a row short of the channels");
        assert!(Norms::parse(b"{}", 4).is_err(), "no weights");
    }
}
