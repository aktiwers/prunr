//! Upscaling with a network that normalizes over its whole input
//! (RealPLKSR's GroupNorm, in Nomos2). Tiled the usual way, each tile is
//! normalized on its own statistics and neighbours come out toned apart:
//! seams that no overlap hides (Buglakova et al., ICCV 2025). So the model
//! is cut at its norms and run part by part: each part crop by crop, with
//! a border wide enough that a crop's middle is exact, and each norm on
//! the whole image's statistics, as a one-piece run has them.
//!
//! Peak working set, per input pixel with 64 f32 feature channels: the
//! blocks hold the image (12 B), the features (256 B) and each block's
//! output before its norm, held the first way that fits free memory; the
//! tail holds the image, the features and the output's bytes (316 B at
//! 4×). The parts and one crop through one add ~0.73 GB (`working_set_mb`).
//!
//! | A block's output                     | Blocks             | Cost                           |
//! |--------------------------------------|--------------------|--------------------------------|
//! | held, f32                            | 524 B              | none                           |
//! | held, f16                            | 396 B              | within 1/255 of f32 on a photo |
//! | not held: run for its statistics, then again to apply them | 268 B and two bands of rows | each block runs twice, ~1.9× the time |
//!
//! A run whose blocks hold nothing peaks in its tail instead: 316 B at 4×.
//! Held f32 on the CPU provider (i7-6700): 0.99 GB peak at 0.6 MP,
//! 1.32 GB at 1.4 MP.

use std::sync::{Arc, Mutex};

use half::f16;
use image::{RgbImage, RgbaImage};
use ort::session::Session;
use prunr_models::{PartLayout, PART_HEAD, PART_NORMS, PART_TAIL};
use rayon::prelude::*;

use crate::progress::{Progress, Step, TileRect, Unit};
use crate::CoreError;

/// A crop's exact middle, at most, in input pixels. Smaller crops ran
/// faster on the CPU provider (67 s at 256 against 76 s in one crop) and
/// keep one part's scratch small.
const CROP: usize = 256;

/// How the blocks hold each one's output before its norm, from the most
/// memory to the least (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreNorm {
    Whole,
    Half,
    /// Not held: each block runs once for its statistics and once to
    /// apply them.
    Twice,
}

impl PreNorm {
    const ALL: [PreNorm; 3] = [PreNorm::Whole, PreNorm::Half, PreNorm::Twice];
    pub(crate) const LEANEST: PreNorm = PreNorm::Twice;
}

/// The first `PreNorm` whose planes fit in `room` bytes, else the leanest.
pub(crate) fn pre_norm_within(layout: PartLayout, w: u32, h: u32, scale: u32, room: u64) -> PreNorm {
    PreNorm::ALL.into_iter().find(|&pre| peak_bytes(layout, w, h, scale, pre) <= room).unwrap_or(PreNorm::LEANEST)
}

/// What a part-by-part upscale of `w × h` by `scale` holds in its planes
/// at its peak, its blocks' outputs held as `pre` (see the module doc);
/// the parts and a crop's scratch are the model's `working_set_mb`.
pub(crate) fn peak_bytes(layout: PartLayout, w: u32, h: u32, scale: u32, pre: PreNorm) -> u64 {
    let (w, h) = (u64::from(w), u64::from(h));
    let features = layout.features as u64 * 4;
    let held = match pre {
        PreNorm::Whole => w * h * features,
        PreNorm::Half => w * h * features / 2,
        PreNorm::Twice => 2 * band_rows(h as usize) as u64 * w * features,
    };
    let blocks = w * h * (12 + features) + held;
    let tail = w * h * (12 + features + u64::from(scale).pow(2) * 3);
    blocks.max(tail)
}

/// The rows a crop row keeps of an image `h` tall.
fn band_rows(h: usize) -> usize {
    spans(h, CROP, 0).first().map_or(0, |span| span.kept_len())
}

/// What the runner knows of a parted network besides its files.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NetShape {
    pub layout: PartLayout,
    /// The image's input name, at the head and again at the tail.
    pub image_input: &'static str,
}

/// What the runner needs of a parted network; `Running` runs the real
/// parts, tests a stand-in. Planes are channel-major (C × h × w).
pub(crate) trait Stages {
    fn layout(&self) -> PartLayout;
    fn norms(&self) -> &Norms;
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

impl Span {
    fn len(self) -> usize {
        self.end - self.start
    }

    fn kept_len(self) -> usize {
        self.keep_end - self.keep_start
    }

    /// Where the kept part starts, from the span's own start.
    fn kept_from(self) -> usize {
        self.keep_start - self.start
    }
}

/// Spans along an axis of `len`: the axis cut into equal kept parts of
/// at most `crop`, each run with `border` around it, every span one size
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

/// One crop of the image: a span of each axis.
#[derive(Clone, Copy, Debug)]
struct Crop {
    x: Span,
    y: Span,
}

impl Crop {
    fn size(self) -> (usize, usize) {
        (self.x.len(), self.y.len())
    }

    /// The kept part, as x, y, w, h of the image.
    fn kept_rect(self) -> (u32, u32, u32, u32) {
        (self.x.keep_start as u32, self.y.keep_start as u32, self.x.kept_len() as u32, self.y.kept_len() as u32)
    }
}

/// Every crop of a `w × h` image for a part reading `border` around a
/// pixel, row by row. Where they are kept does not depend on `border`, so
/// every stage lays the same cells.
fn crops(w: usize, h: usize, border: usize) -> Vec<Crop> {
    let xs = spans(w, CROP, border);
    spans(h, CROP, border).into_iter().flat_map(|y| xs.iter().map(move |&x| Crop { x, y })).collect()
}

/// `crop` of each of `planes.len() / (w × h)` planes of a `w`-wide image.
fn cut(planes: &[f32], w: usize, h: usize, crop: Crop) -> Vec<f32> {
    let channels = planes.len() / (w * h);
    let (cw, ch) = crop.size();
    let mut out = Vec::with_capacity(channels * cw * ch);
    for c in 0..channels {
        for row in crop.y.start..crop.y.end {
            let at = (c * h + row) * w;
            out.extend_from_slice(&planes[at + crop.x.start..at + crop.x.end]);
        }
    }
    out
}

/// Write the kept part of `result` (`crop`'s output) into `out` (planes
/// of the `w × h` image), each value through `to`.
fn keep<T: Copy>(out: &mut [T], result: &[f32], w: usize, h: usize, crop: Crop, to: impl Fn(f32) -> T) {
    let channels = out.len() / (w * h);
    let (cw, ch) = crop.size();
    let (x0, x1) = (crop.x.kept_from(), crop.x.kept_from() + crop.x.kept_len());
    for c in 0..channels {
        for row in crop.y.kept_from()..crop.y.kept_from() + crop.y.kept_len() {
            let src = (c * ch + row) * cw;
            let dst = (c * h + crop.y.start + row) * w + crop.x.start;
            out[dst + x0..dst + x1].iter_mut().zip(&result[src + x0..src + x1]).for_each(|(o, &r)| *o = to(r));
        }
    }
}

/// Write the kept part of the tail's `result` (3 planes, 0-1, at `scale`×
/// the crop) into `out` as bytes.
fn keep_rgb(out: &mut RgbImage, result: &[f32], scale: usize, crop: Crop) {
    let (cw, ch) = crop.size();
    let (ow, plane) = (cw * scale, cw * ch * scale * scale);
    let (x0, y0) = (crop.x.kept_from() * scale, crop.y.kept_from() * scale);
    for py in 0..crop.y.kept_len() * scale {
        for px in 0..crop.x.kept_len() * scale {
            let i = (y0 + py) * ow + x0 + px;
            let rgb = std::array::from_fn(|c| crate::math::round_u8(result[c * plane + i] * 255.0));
            out.put_pixel(((crop.x.keep_start * scale) + px) as u32, ((crop.y.keep_start * scale) + py) as u32, image::Rgb(rgb));
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

/// Values' sum and sum of squares, f64.
fn sums_of(values: impl Iterator<Item = f32>) -> (f64, f64) {
    values.fold((0.0, 0.0), |(s, q), v| (s + f64::from(v), q + f64::from(v) * f64::from(v)))
}

/// Each channel's `(scale, shift)`, so `p·scale + shift` is GroupNorm
/// (μ and σ² over each of `groups` runs of channels, `count` pixels a
/// channel) then the norm's weight and bias. `sums` holds each channel's
/// sum and sum of squares.
fn affine(sums: &[(f64, f64)], count: usize, groups: usize, eps: f32, (gamma, beta): (&[f32], &[f32])) -> Vec<(f32, f32)> {
    let per_group = sums.len() / groups;
    let stats: Vec<(f64, f64)> = sums.chunks(per_group).map(|g| {
        let (sum, sq) = g.iter().fold((0.0, 0.0), |(s, q), &(cs, cq)| (s + cs, q + cq));
        let n = (count * g.len()) as f64;
        let mean = sum / n;
        (mean, (sq / n - mean * mean).max(0.0))
    }).collect();
    gamma.iter().zip(beta).enumerate().map(|(c, (&gamma, &beta))| {
        let (mean, var) = stats[c / per_group];
        let scale = gamma / (var + f64::from(eps)).sqrt() as f32;
        (scale, beta - mean as f32 * scale)
    }).collect()
}

/// `features += γ ⊙ (pre − μ) / √(σ² + ε) + β`, per channel, with μ and
/// σ² over the whole image for each of `groups` runs of channels: what
/// GroupNorm and the block's skip compute on a one-piece run. One plane
/// per weight in the norm; `to` reads a held value.
fn add_normalized<T: Copy + Send + Sync>(features: &mut [f32], pre: &[T], groups: usize, eps: f32, norm: (&[f32], &[f32]), to: impl Fn(T) -> f32 + Sync) {
    let plane = pre.len() / norm.0.len();
    let sums: Vec<(f64, f64)> = pre.par_chunks(plane).map(|p| sums_of(p.iter().map(|&v| to(v)))).collect();
    let affine = affine(&sums, plane, groups, eps, norm);
    features.par_chunks_mut(plane).zip(pre.par_chunks(plane)).zip(affine.par_iter()).for_each(|((f, p), &(scale, shift))| {
        f.iter_mut().zip(p).for_each(|(f, &p)| *f += to(p) * scale + shift);
    });
}

/// Add each channel's sum and sum of squares over `crop`'s kept part of
/// `result` to `sums`.
fn add_kept_sums(sums: &mut [(f64, f64)], result: &[f32], crop: Crop) {
    let (cw, ch) = crop.size();
    let (from, len) = (crop.x.kept_from(), crop.x.kept_len());
    for (c, (sum, sq)) in sums.iter_mut().enumerate() {
        for row in crop.y.kept_from()..crop.y.kept_from() + crop.y.kept_len() {
            let at = (c * ch + row) * cw + from;
            let (s, q) = sums_of(result[at..at + len].iter().copied());
            *sum += s;
            *sq += q;
        }
    }
}

/// A twice-run block's new features for the crop rows it has finished,
/// held until the next row has read the old ones around them. A row reads
/// `border` rows into a neighbour, never past the next row, so two are
/// held at most.
#[derive(Default)]
struct Bands {
    before: Option<Band>,
    current: Option<Band>,
}

/// New features for the image's rows `top..top + rows`, every channel.
struct Band {
    top: usize,
    rows: usize,
    values: Vec<f32>,
}

impl Bands {
    /// Put `crop`'s new features, the old ones plus `result·scale + shift`
    /// over its kept part, in its row's band. Reaching a new row writes
    /// the one two back into `features`.
    fn put(&mut self, features: &mut [f32], result: &[f32], (w, h): (usize, usize), crop: Crop, affine: &[(f32, f32)]) {
        if self.current.as_ref().is_some_and(|band| band.top != crop.y.keep_start) {
            if let Some(done) = self.before.take() {
                done.write(features, (w, h));
            }
            self.before = self.current.take();
        }
        let rows = crop.y.kept_len();
        let band = self.current.get_or_insert_with(|| Band { top: crop.y.keep_start, rows, values: vec![0.0; affine.len() * rows * w] });
        let (cw, ch) = crop.size();
        let (from, len) = (crop.x.kept_from(), crop.x.kept_len());
        for (c, &(scale, shift)) in affine.iter().enumerate() {
            for r in 0..rows {
                let src = (c * ch + crop.y.kept_from() + r) * cw + from;
                let old = (c * h + crop.y.keep_start + r) * w + crop.x.keep_start;
                let dst = (c * band.rows + r) * w + crop.x.keep_start;
                band.values[dst..dst + len].iter_mut().zip(&features[old..old + len]).zip(&result[src..src + len])
                    .for_each(|((new, &f), &p)| *new = f + (p * scale + shift));
            }
        }
    }

    /// Write what is still held into `features`.
    fn finish(self, features: &mut [f32], size: (usize, usize)) {
        self.before.into_iter().chain(self.current).for_each(|band| band.write(features, size));
    }
}

impl Band {
    fn write(self, features: &mut [f32], (w, h): (usize, usize)) {
        let len = self.rows * w;
        for (c, values) in self.values.chunks(len).enumerate() {
            let at = (c * h + self.top) * w;
            features[at..at + len].copy_from_slice(values);
        }
    }
}

/// Run `net` over `input` part by part (see the module doc), its blocks'
/// outputs held as `pre`; the output is `scale`× the input, RGB, alpha
/// left to the caller. The tail's crops are finished pixels, so they go
/// out as previews.
pub(crate) fn upscale_by_parts(net: &impl Stages, input: &RgbaImage, scale: u32, pre: PreNorm, progress: &Progress) -> Result<RgbImage, CoreError> {
    let (w, h) = (input.width() as usize, input.height() as usize);
    let plane = w * h;
    let mut image = vec![0.0f32; 3 * plane];
    for (i, p) in input.pixels().enumerate() {
        for c in 0..3 {
            image[c * plane + i] = f32::from(p.0[c]) / 255.0;
        }
    }
    let (layout, norms) = (net.layout(), net.norms());
    let channels = layout.features;
    let runs_per_block = if pre == PreNorm::Twice { 2 } else { 1 };
    let stages = (norms.blocks() * runs_per_block + 2) as u32;
    progress.step(if pre == PreNorm::Twice { Step::UpscalingInLessMemory } else { Step::Upscaling });

    let mut features = vec![0.0f32; channels * plane];
    run_stage(progress, (w, h), (0, stages), layout.head, |crop| {
        let (cw, ch) = crop.size();
        let out = sized(net.head(cut(&image, w, h, crop), cw, ch)?, channels * cw * ch)?;
        keep(&mut features, &out, w, h, crop, |v| v);
        Ok(())
    })?;
    let block_out = |k: usize, features: &[f32], crop: Crop| {
        let (cw, ch) = crop.size();
        sized(net.block(k, cut(features, w, h, crop), cw, ch)?, channels * cw * ch)
    };
    let mut stage = 0;
    let mut next = || {
        stage += 1;
        (stage, stages)
    };
    match pre {
        PreNorm::Whole => {
            let mut held = vec![0.0f32; channels * plane];
            for k in 0..norms.blocks() {
                run_stage(progress, (w, h), next(), layout.block, |crop| {
                    keep(&mut held, &block_out(k, &features, crop)?, w, h, crop, |v| v);
                    Ok(())
                })?;
                add_normalized(&mut features, &held, norms.groups, norms.eps, norms.norm(k), |v| v);
            }
        }
        PreNorm::Half => {
            let mut held = vec![f16::ZERO; channels * plane];
            for k in 0..norms.blocks() {
                run_stage(progress, (w, h), next(), layout.block, |crop| {
                    keep(&mut held, &block_out(k, &features, crop)?, w, h, crop, f16::from_f32);
                    Ok(())
                })?;
                add_normalized(&mut features, &held, norms.groups, norms.eps, norms.norm(k), f16::to_f32);
            }
        }
        PreNorm::Twice => {
            for k in 0..norms.blocks() {
                let mut sums = vec![(0.0, 0.0); channels];
                run_stage(progress, (w, h), next(), layout.block, |crop| {
                    add_kept_sums(&mut sums, &block_out(k, &features, crop)?, crop);
                    Ok(())
                })?;
                let affine = affine(&sums, plane, norms.groups, norms.eps, norms.norm(k));
                let mut bands = Bands::default();
                run_stage(progress, (w, h), next(), layout.block, |crop| {
                    let out = block_out(k, &features, crop)?;
                    bands.put(&mut features, &out, (w, h), crop, &affine);
                    Ok(())
                })?;
                bands.finish(&mut features, (w, h));
            }
        }
    }
    let s = scale as usize;
    let mut out = RgbImage::new((w * s) as u32, (h * s) as u32);
    run_stage(progress, (w, h), (stages - 1, stages), layout.tail, |crop| {
        let (cw, ch) = crop.size();
        let result = sized(net.tail(cut(&features, w, h, crop), cut(&image, w, h, crop), cw, ch)?, 3 * cw * ch * s * s)?;
        keep_rgb(&mut out, &result, s, crop);
        let rect = crop.kept_rect();
        progress.tile_preview(rect, (w as u32, h as u32), |pw, ph| {
            let kept = image::imageops::crop_imm(&out, rect.0 * scale, rect.1 * scale, rect.2 * scale, rect.3 * scale).to_image();
            super::tiling::preview_with_alpha(input, &kept, rect, pw, ph)
        });
        Ok(())
    })?;
    progress.outer(stages, stages, Unit::Step);
    progress.step(Step::Finishing);
    Ok(out)
}

/// Stage `stage` of `total`, crop by crop of the `w × h` image for a part
/// reading `border` around a pixel. Cancel stops it between crops.
fn run_stage(progress: &Progress, (w, h): (usize, usize), (stage, total): (u32, u32), border: usize, mut run: impl FnMut(Crop) -> Result<(), CoreError>) -> Result<(), CoreError> {
    let crops = crops(w, h, border);
    let n = crops.len() as u32;
    progress.outer(stage, total, Unit::Step);
    progress.tiles(crops.iter().map(|crop| {
        let (x, y, cw, ch) = crop.kept_rect();
        TileRect::of_pixels(x, y, cw, ch, w as u32, h as u32)
    }).collect());
    progress.inner(0, n, Unit::Tile);
    for (i, crop) in crops.into_iter().enumerate() {
        progress.check_cancelled()?;
        progress.tile(i as u32, false);
        run(crop)?;
        progress.tile(i as u32, true);
        progress.inner(i as u32 + 1, n, Unit::Tile);
    }
    Ok(())
}

/// One part's session and the names of what it takes and gives.
struct Part {
    session: Mutex<Session>,
    inputs: Vec<String>,
    output: String,
}

impl Part {
    fn of(key: &str, session: Session) -> Result<Self, CoreError> {
        let inputs = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let output = session.outputs().first().map(|o| o.name().to_string())
            .ok_or_else(|| CoreError::Model(format!("part {key} gives nothing")))?;
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

    /// One norm per block.
    fn blocks(&self) -> usize {
        self.gamma.len()
    }

    /// Block `k`'s weight and bias, one per channel.
    fn norm(&self, k: usize) -> (&[f32], &[f32]) {
        (&self.gamma[k], &self.beta[k])
    }
}

/// A parted model's ONNX parts as bytes, by key.
pub(crate) type PartBytes = Vec<(&'static str, Arc<Vec<u8>>)>;

/// `id`'s parts, and its norms checked against `features` channels.
pub(crate) fn read_parts(id: prunr_models::ModelId, features: usize) -> Result<(PartBytes, Norms), CoreError> {
    let parts = match prunr_models::descriptor(id).map(|d| d.source) {
        Some(prunr_models::ModelSource::MultiPartOnDemand { parts, .. }) if prunr_models::is_available(id) => parts,
        _ => return Err(CoreError::Inference(prunr_models::not_installed_error(id))),
    };
    let bytes = |key: &str| prunr_models::resolve_part_bytes(id, key).map_err(|e| CoreError::Inference(format!("{id:?} {key}: {e:?}")));
    let norms = Norms::parse(&bytes(PART_NORMS)?, features)?;
    let onnx = parts.iter().filter(|p| p.key != PART_NORMS).map(|p| Ok((p.key, bytes(p.key)?))).collect::<Result<_, CoreError>>()?;
    Ok((onnx, norms))
}

/// The real parts: a head, its blocks in key order, a tail, the norms.
pub(crate) struct BlockNet {
    head: Part,
    blocks: Vec<Part>,
    tail: Part,
    /// The tail's other input, beside the image.
    tail_features: String,
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
        let tail_features = match tail.inputs.as_slice() {
            [a, b] if a == image && b != image => Some(b.clone()),
            [a, b] if b == image && a != image => Some(a.clone()),
            _ => None,
        };
        let fits = head.inputs == [image] && blocks.iter().all(|b| b.inputs.len() == 1) && norms.blocks() == blocks.len();
        match tail_features {
            Some(tail_features) if fits => Ok(Self { head, blocks, tail, tail_features, norms, shape }),
            _ => Err(CoreError::Model(format!(
                "parts do not fit: a head taking {:?}, a tail taking {:?}, {} blocks for {} norms, the image as {image:?}",
                head.inputs, tail.inputs, blocks.len(), norms.blocks(),
            ))),
        }
    }
}

/// A run of `net`'s parts, each abortable through `terminate`.
pub(crate) struct Running<'a> {
    pub net: &'a BlockNet,
    pub terminate: Option<&'a super::UpscaleRunOptions>,
}

impl BlockNet {
    pub(crate) fn layout(&self) -> PartLayout {
        self.shape.layout
    }
}

impl Stages for Running<'_> {
    fn layout(&self) -> PartLayout { self.net.layout() }
    fn norms(&self) -> &Norms { &self.net.norms }

    fn head(&self, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        self.net.head.run(vec![(self.net.shape.image_input, 3, image)], w, h, self.terminate)
    }

    fn block(&self, k: usize, features: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        let part = &self.net.blocks[k];
        // One input each (`BlockNet::new`).
        part.run(vec![(part.inputs[0].as_str(), self.layout().features, features)], w, h, self.terminate)
    }

    fn tail(&self, features: Vec<f32>, image: Vec<f32>, w: usize, h: usize) -> Result<Vec<f32>, CoreError> {
        let feeds = vec![(self.net.tail_features.as_str(), self.layout().features, features), (self.net.shape.image_input, 3, image)];
        self.net.tail.run(feeds, w, h, self.terminate)
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
        let pre: Vec<f32> = vec![1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 0.0, 4.0, 8.0, 1.0, 1.0, 1.0];
        let mut features = vec![10.0f32; 12];
        let (gamma, beta) = (vec![1.0, 2.0, 1.0, 1.0], vec![0.0, 0.0, 0.5, 0.0]);
        add_normalized(&mut features, &pre, 2, 0.0, (&gamma, &beta), |v| v);
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
        norms: Norms,
        /// Channels the head gives; 4 fits the network.
        head_out: usize,
    }

    impl Toy {
        /// Three blocks, each norm `gamma` and `beta` scaled a little
        /// further, so a block read with another's norm would show.
        fn new(gamma: Vec<f32>, beta: Vec<f32>) -> Self {
            let row = |v: &[f32], k: usize| v.iter().map(|x| x * (1.0 + 0.1 * k as f32)).collect();
            let norms = Norms { groups: 2, eps: 1e-5, gamma: (0..3).map(|k| row(&gamma, k)).collect(), beta: (0..3).map(|k| row(&beta, k)).collect() };
            Self { norms, head_out: 4 }
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
        fn norms(&self) -> &Norms { &self.norms }
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

    /// The stand-in over the whole image in one piece, no crops: the
    /// reference a part-by-part run must match.
    fn one_pass(net: &Toy, input: &RgbaImage) -> RgbImage {
        let (w, h) = (input.width() as usize, input.height() as usize);
        let p = w * h;
        let image: Vec<f32> = (0..3).flat_map(|c| input.pixels().map(move |px| f32::from(px.0[c]) / 255.0)).collect();
        let mut f = net.head(image.clone(), w, h).unwrap();
        for k in 0..net.norms.blocks() {
            let pre = net.block(k, f.clone(), w, h).unwrap();
            add_normalized(&mut f, &pre, net.norms.groups, net.norms.eps, net.norms.norm(k), |v| v);
        }
        let rgb = net.tail(f, image, w, h).unwrap();
        RgbImage::from_fn(2 * w as u32, 2 * h as u32, |x, y| image::Rgb(std::array::from_fn(|c| crate::math::round_u8(rgb[c * 4 * p + y as usize * 2 * w + x as usize] * 255.0))))
    }

    /// Part by part in crops matches one pass over the whole image however
    /// the blocks' outputs are held: no seam where crops meet, whatever the
    /// image's size against a crop. 300 × 600 has three rows of crops, so
    /// a twice-run block hands a band on across two rows.
    #[test]
    fn part_by_part_matches_one_pass() {
        let net = Toy::new(vec![0.5, 1.5, 1.0, 0.8], vec![0.1, -0.2, 0.0, 0.3]);
        for (w, h) in [(300u32, 270u32), (300, 600), (600, 90), (100, 100)] {
            let input = toy_input(w, h);
            let whole = one_pass(&net, &input);
            for pre in PreNorm::ALL {
                let parts = upscale_by_parts(&net, &input, 2, pre, &Progress::none()).unwrap();
                let worst = parts.as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                assert!(worst <= 1, "{w}×{h} {pre:?}: off by {worst}");
            }
        }
    }

    /// The first way to hold the blocks' outputs that fits the room, from
    /// the most memory to the least; the least when none does.
    #[test]
    fn a_run_holds_the_most_that_fits() {
        let layout = PartLayout { features: 64, head: 2, block: 12, tail: 16 };
        let peak = |pre| peak_bytes(layout, 2000, 4000, 4, pre);
        assert!(peak(PreNorm::Whole) > peak(PreNorm::Half) && peak(PreNorm::Half) > peak(PreNorm::Twice));
        assert_eq!(pre_norm_within(layout, 2000, 4000, 4, peak(PreNorm::Whole)), PreNorm::Whole);
        assert_eq!(pre_norm_within(layout, 2000, 4000, 4, peak(PreNorm::Whole) - 1), PreNorm::Half);
        assert_eq!(pre_norm_within(layout, 2000, 4000, 4, peak(PreNorm::Half) - 1), PreNorm::Twice);
        assert_eq!(pre_norm_within(layout, 2000, 4000, 4, 0), PreNorm::Twice);
    }

    /// The run reports a step, each stage and each crop, within their
    /// totals, however the blocks' outputs are held.
    #[test]
    fn part_by_part_keeps_the_progress_contract() {
        use crate::progress::{check_contract, RecordingSink};
        let net = Toy::new(vec![1.0; 4], vec![0.0; 4]);
        for pre in PreNorm::ALL {
            let sink = std::sync::Arc::new(RecordingSink::default());
            upscale_by_parts(&net, &toy_input(300, 600), 2, pre, &Progress::new(sink.clone())).unwrap();
            check_contract(&sink.updates()).unwrap_or_else(|e| panic!("{pre:?}: {e}"));
        }
    }

    /// Cancel stops the run at the next crop.
    #[test]
    fn a_cancelled_run_stops() {
        let net = Toy::new(vec![1.0; 4], vec![0.0; 4]);
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let r = upscale_by_parts(&net, &toy_input(300, 270), 2, PreNorm::Whole, &Progress::none().with_cancel(flag));
        assert!(matches!(r, Err(CoreError::Cancelled)));
    }

    /// A part that gives another shape than its crop fails the run
    /// instead of writing past the crop.
    #[test]
    fn a_part_of_the_wrong_shape_fails_the_run() {
        let net = Toy { head_out: 3, ..Toy::new(vec![1.0; 4], vec![0.0; 4]) };
        let r = upscale_by_parts(&net, &toy_input(100, 100), 2, PreNorm::Whole, &Progress::none());
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
        upscale_by_parts(&net, &input, 2, PreNorm::Whole, &Progress::new(sink.clone()).with_tile_previews(300)).unwrap();
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
        upscale_by_parts(&net, &input, 2, PreNorm::Whole, &Progress::new(quiet.clone())).unwrap();
        assert!(!quiet.updates().iter().any(|u| matches!(u, ProgressUpdate::TilePreview(_))), "no preview unless asked");
    }

    /// The peak is the larger stage: the blocks with many features, the
    /// tail with few; the blocks hold their outputs whole in f32, half that
    /// in f16, or two bands of a crop row's rows.
    #[test]
    fn the_peak_is_the_larger_stage() {
        let layout = PartLayout { features: 64, head: 2, block: 12, tail: 16 };
        assert_eq!(peak_bytes(layout, 1000, 1000, 4, PreNorm::Whole), 1_000_000 * (12 + 2 * 256), "the blocks");
        assert_eq!(peak_bytes(layout, 1000, 1000, 4, PreNorm::Half), 1_000_000 * (12 + 256 + 128));
        assert_eq!(peak_bytes(layout, 1000, 4000, 1, PreNorm::Twice), 4_000_000 * (12 + 256) + 2 * 250 * 1000 * 256);
        assert_eq!(peak_bytes(layout, 1000, 4000, 4, PreNorm::Twice), 4_000_000 * (12 + 256 + 48), "the tail outweighs blocks that hold nothing");
        assert_eq!(peak_bytes(PartLayout { features: 8, ..layout }, 10, 10, 4, PreNorm::Whole), 100 * (12 + 32 + 48), "the tail");
    }

    /// The real Nomos2 gives the same picture however its blocks' outputs
    /// are held: f16 within a level of f32, twice-run alike. Needs the
    /// model installed; `PRUNR_TEST_IMAGE` names a photo to crop, else a
    /// made-up one. Three rows of crops, so the band hand-over runs.
    #[test]
    #[ignore = "real model"]
    fn the_real_model_holds_three_ways_alike() {
        let id = prunr_models::ModelId::FourXNomos2RealPlksr;
        if !prunr_models::is_available(id) {
            return;
        }
        let input = match std::env::var("PRUNR_TEST_IMAGE") {
            Ok(path) => image::imageops::crop_imm(&image::open(path).expect("PRUNR_TEST_IMAGE opens").to_rgba8(), 0, 0, 300, 600).to_image(),
            Err(_) => RgbaImage::from_fn(300, 600, |x, y| {
                let ripple = (((x as f32) * 0.21).sin() * ((y as f32) * 0.13).cos() * 60.0) as i32;
                image::Rgba([(80 + x as i32 / 3 + ripple).clamp(0, 255) as u8, (40 + y as i32 / 4).clamp(0, 255) as u8, (200 - ripple).clamp(0, 255) as u8, 255])
            }),
        };
        let engine = crate::OrtEngine::for_model(id, 4, true).expect("Nomos2 builds");
        let net = engine.parts().expect("Nomos2 runs in parts");
        let run = |pre| {
            let started = std::time::Instant::now();
            let out = upscale_by_parts(&Running { net, terminate: None }, &input, 4, pre, &Progress::none()).expect("runs");
            eprintln!("{pre:?}: {:?}", started.elapsed());
            out
        };
        let whole = run(PreNorm::Whole);
        for pre in [PreNorm::Half, PreNorm::Twice] {
            let worst = run(pre).as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
            assert!(worst <= 1, "{pre:?}: off by {worst}");
        }
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
