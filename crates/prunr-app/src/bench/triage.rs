//! Which of two outputs is right. That they differ says nothing on its
//! own, so each side's output is also judged alone: for a blank output
//! (one value throughout, or a colorizer's no colour), which no model
//! makes of the bench photo; and an upscale, shrunk back to its input's
//! size, against its input, where it should land.
//!
//! A colorized photo is not held against the photo's own colours: the
//! mean distance favours no colour at all. On the built-in picture a
//! colourless output lands 31/255 off, nearer than DDColor's own guess
//! at 35.

use std::path::Path;

use image::{DynamicImage, RgbaImage};
use prunr_core::formats::{self, ResizeFilter};
use serde::Serialize;

use super::stats::{self, Agreement};
use super::Family;

/// The most a blank output's values spread: the bench's own "apart".
const FLAT_SPREAD: u8 = 2;

/// Two sides whose distances from their input differ by this share of
/// the larger are as near as each other: an upscaler's FP16 and FP32 runs
/// land within 2% (2.25 and 2.29/255 for Siax on an RTX 3060).
const EVEN_SHARE: f64 = 0.05;

/// What a run makes.
pub enum Output {
    Picture(DynamicImage),
    /// The face finder's count and boxes.
    Values(Vec<u8>),
}

/// What a blank output of a kind is flat in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlankIn {
    /// The values compared.
    Values,
    /// The colour of the RGB values compared: a colorizer keeps the
    /// photo's lightness whatever it predicts.
    Colour,
}

impl BlankIn {
    pub fn describe(self, level: u8) -> String {
        match self {
            Self::Values => format!("every value is {level}/255"),
            Self::Colour => "it adds no colour".into(),
        }
    }
}

/// One side's output on its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Score {
    /// It is this level throughout, give or take `FLAT_SPREAD`.
    pub flat: Option<u8>,
    /// An upscale shrunk back to its input's size: its mean distance from
    /// the input in 0-255 levels; lower is nearer.
    pub off_input: Option<f64>,
}

/// `output` of `family` judged alone; `samples` are its compared values,
/// `tile` the upscalers' input.
pub fn score(family: Family, samples: &[u8], output: &Output, tile: &RgbaImage) -> Score {
    let flat = family.blank_in().and_then(|basis| match basis {
        BlankIn::Values => flat(samples.iter().copied()),
        BlankIn::Colour => flat(samples.chunks_exact(3).map(|p| p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]))),
    });
    let off_input = match output {
        Output::Picture(picture) if family.shrinks_back() => off_input(picture, tile),
        _ => None,
    };
    Score { flat, off_input }
}

fn flat(mut values: impl Iterator<Item = u8>) -> Option<u8> {
    let first = values.next()?;
    let (mut lo, mut hi) = (first, first);
    for v in values {
        (lo, hi) = (lo.min(v), hi.max(v));
        if hi - lo > FLAT_SPREAD {
            return None;
        }
    }
    Some(lo)
}

fn off_input(picture: &DynamicImage, input: &RgbaImage) -> Option<f64> {
    let output = stats::rgba(picture);
    let (w, h) = input.dimensions();
    // Box: the plain average over each enlarged pixel, favouring no
    // upscaler's sharpening.
    let shrunk = (output.dimensions() != (w, h)).then(|| formats::resize_rgba(&output, w, h, ResizeFilter::Box));
    let output = shrunk.as_ref().unwrap_or(&output);
    Agreement::of(&stats::rgb_samples(output), &stats::rgb_samples(input)).map(|a| a.mean_abs)
}

/// Which output looks right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Leaning {
    GpuBlank,
    CpuBlank,
    BothBlank,
    CpuNearer,
    GpuNearer,
    /// Within `EVEN_SHARE` of each other.
    Even,
}

impl Leaning {
    pub fn label(self) -> &'static str {
        match self {
            Self::GpuBlank => "GPU blank",
            Self::CpuBlank => "CPU blank",
            Self::BothBlank => "both blank",
            Self::CpuNearer => "CPU nearer",
            Self::GpuNearer => "GPU nearer",
            Self::Even => "as near",
        }
    }
}

/// A blank side, whatever the verdict; on a `mismatch`, the side nearer
/// its input. `gpu` is the GPU path's score only when it ran on a GPU.
pub fn leaning(cpu: Option<&Score>, gpu: Option<&Score>, mismatch: bool) -> Option<Leaning> {
    let blank = |s: Option<&Score>| s.is_some_and(|s| s.flat.is_some());
    match (blank(cpu), blank(gpu)) {
        (true, true) => Some(Leaning::BothBlank),
        (true, false) => Some(Leaning::CpuBlank),
        (false, true) => Some(Leaning::GpuBlank),
        (false, false) if mismatch => {
            let (c, g) = (cpu?.off_input?, gpu?.off_input?);
            Some(if (c - g).abs() <= EVEN_SHARE * c.max(g) {
                Leaning::Even
            } else if c < g {
                Leaning::CpuNearer
            } else {
                Leaning::GpuNearer
            })
        }
        (false, false) => None,
    }
}

/// The pictures of a model worth a look, held only until they are saved.
#[derive(Default)]
pub struct Pictures {
    pub cpu: Option<DynamicImage>,
    pub gpu: Option<DynamicImage>,
}

/// Each side's picture as `dir/<name>-<side>.png`, for a person to look
/// at; the paths written.
pub fn save(dir: &Path, name: &str, pictures: &Pictures) -> Result<Vec<String>, String> {
    let sides = [("cpu", &pictures.cpu), ("gpu", &pictures.gpu)];
    if sides.iter().all(|(_, picture)| picture.is_none()) {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    sides.into_iter()
        .filter_map(|(side, picture)| picture.as_ref().map(|p| (side, p)))
        .map(|(side, picture)| {
            let path = dir.join(format!("{name}-{side}.png"));
            picture.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(path.display().to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::Inputs;

    fn picture(img: RgbaImage) -> Output {
        Output::Picture(DynamicImage::ImageRgba8(img))
    }

    #[test]
    fn a_blank_output_is_one_level_give_or_take_two() {
        assert_eq!(flat([0, 0, 1, 2].into_iter()), Some(0));
        assert_eq!(flat([255, 254, 253].into_iter()), Some(253));
        assert_eq!(flat([0, 3].into_iter()), None);
        assert_eq!(flat(std::iter::empty()), None);
        let tile = RgbaImage::new(1, 1);
        let none_found = Output::Values(vec![0]);
        assert_eq!(score(Family::FaceFinder, &[0], &none_found, &tile).flat, None, "no faces is a list, not a blank picture");
        assert_eq!(score(Family::Depth, &[0, 0], &picture(tile.clone()), &tile).flat, Some(0));
    }

    /// An upscale that is its input enlarged lands back on it; a black
    /// one lands far off; other kinds are not shrunk back.
    #[test]
    fn an_upscale_is_scored_shrunk_back_onto_its_input() {
        let tile = Inputs::built_in().tile;
        let (w, h) = tile.dimensions();
        let enlarged = image::imageops::resize(&tile, w * 4, h * 4, image::imageops::FilterType::Nearest);
        let faithful = score(Family::Upscale, &[1, 9], &picture(enlarged), &tile).off_input.expect("scored");
        assert!(faithful < 0.01, "{faithful}");
        let black = score(Family::Upscale, &[1, 9], &picture(RgbaImage::new(w * 4, h * 4)), &tile).off_input.expect("scored");
        assert!(black > 100.0, "{black}");
        let rgb = Output::Picture(DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(tile.clone()).into_rgb8()));
        assert!(score(Family::Upscale, &[1, 9], &rgb, &tile).off_input.is_some_and(|off| off < 0.01), "any pixel layout is scored");
        assert_eq!(score(Family::Depth, &[1, 9], &picture(tile.clone()), &tile).off_input, None);
    }

    /// A colorizer that adds no colour is blank though its lightness
    /// varies; one that adds colour is not.
    #[test]
    fn a_colorizer_is_blank_when_it_adds_no_colour() {
        let photo = Inputs::built_in().photo;
        let grey = DynamicImage::ImageRgba8(photo.clone()).grayscale().into_rgba8();
        let colourless = score(Family::Colorize, &stats::rgb_samples(&grey), &picture(grey.clone()), &photo);
        assert_eq!(colourless, Score { flat: Some(0), off_input: None });
        assert_eq!(score(Family::Colorize, &stats::rgb_samples(&photo), &picture(photo.clone()), &photo), Score::default());
        assert_eq!(score(Family::Segmentation, &stats::rgb_samples(&grey), &picture(grey), &photo), Score::default(), "varying values");
    }

    #[test]
    fn the_leaning_names_a_blank_side_or_on_a_mismatch_the_one_nearer_its_input() {
        let near = |off| Score { flat: None, off_input: Some(off) };
        let blank = Score { flat: Some(0), off_input: Some(120.0) };
        let unscored = Score::default();
        let cases = [
            (Some(near(3.0)), Some(blank), false, Some(Leaning::GpuBlank)),
            (Some(blank), Some(near(3.0)), true, Some(Leaning::CpuBlank)),
            (Some(blank), Some(blank), false, Some(Leaning::BothBlank)),
            (Some(blank), None, false, Some(Leaning::CpuBlank)),
            (None, Some(blank), false, Some(Leaning::GpuBlank)),
            (Some(near(3.0)), Some(near(4.0)), true, Some(Leaning::CpuNearer)),
            (Some(near(4.0)), Some(near(3.0)), true, Some(Leaning::GpuNearer)),
            (Some(near(2.29)), Some(near(2.25)), true, Some(Leaning::Even)),
            (Some(near(3.0)), Some(near(4.0)), false, None),
            (Some(unscored), Some(unscored), true, None),
            (Some(near(3.0)), None, true, None),
        ];
        for (cpu, gpu, mismatch, expected) in cases {
            assert_eq!(leaning(cpu.as_ref(), gpu.as_ref(), mismatch), expected, "{cpu:?} {gpu:?} {mismatch}");
        }
    }

    #[test]
    fn each_sides_picture_is_saved_by_model_and_side() {
        let dir = tempfile::tempdir().unwrap();
        let outputs = dir.path().join("prunr-bench-outputs");
        let depth = DynamicImage::ImageLuma16(image::ImageBuffer::from_pixel(3, 2, image::Luma([40_000u16])));
        let saved = save(&outputs, "depth_anything", &Pictures { cpu: Some(depth.clone()), gpu: None }).unwrap();
        assert_eq!(saved, [outputs.join("depth_anything-cpu.png").display().to_string()]);
        assert_eq!(image::open(&saved[0]).unwrap().into_luma16(), depth.into_luma16(), "16-bit depth reads back as written");
        let elsewhere = dir.path().join("never-made");
        assert_eq!(save(&elsewhere, "x", &Pictures::default()).unwrap(), Vec::<String>::new());
        assert!(!elsewhere.exists(), "nothing to save makes no folder");
    }
}
