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
use super::{Family, Inputs};

/// The most a blank output's values spread: the bench's own "apart".
const FLAT_SPREAD: u8 = 2;

/// Two sides whose distances from the truth differ by this share of the
/// larger are as near as each other: an upscaler's FP16 and FP32 runs
/// land within 2% (2.25 and 2.29/255 for Siax on an RTX 3060).
const EVEN_SHARE: f64 = 0.05;

/// One side's output on its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Score {
    /// Every compared value is this level, give or take `FLAT_SPREAD`.
    pub flat: Option<u8>,
    /// Mean distance from the truth in 0-255 levels; lower is nearer.
    pub off_truth: Option<f64>,
}

/// What a side ran into: the values compared, and the picture they came
/// from to score and save (none for the face finder's list of boxes).
pub struct Output {
    pub samples: Vec<u8>,
    pub picture: Option<DynamicImage>,
}

pub fn score(family: Family, inputs: &Inputs, output: &Output) -> Score {
    let picture = output.picture.as_ref();
    let flat = match family {
        Family::FaceFinder => None,
        // A colorizer keeps the photo's lightness whatever it predicts.
        Family::Colorize => picture.and_then(DynamicImage::as_rgba8).and_then(|p| flat(&colour(p))),
        _ => flat(&output.samples),
    };
    Score { flat, off_truth: picture.and_then(|p| off_truth(family, inputs, p)) }
}

fn flat(samples: &[u8]) -> Option<u8> {
    let min = *samples.iter().min()?;
    let max = *samples.iter().max()?;
    (max - min <= FLAT_SPREAD).then_some(min)
}

/// Each pixel's colour: how far its channels spread.
fn colour(picture: &RgbaImage) -> Vec<u8> {
    picture.pixels().map(|p| {
        let [r, g, b, _] = p.0;
        r.max(g).max(b) - r.min(g).min(b)
    }).collect()
}

fn off_truth(family: Family, inputs: &Inputs, picture: &DynamicImage) -> Option<f64> {
    if family != Family::Upscale {
        return None;
    }
    let truth = &inputs.tile;
    let output = picture.as_rgba8()?;
    let (w, h) = truth.dimensions();
    // Box: the plain average over each enlarged pixel, favouring no
    // upscaler's sharpening.
    let shrunk = (output.dimensions() != (w, h)).then(|| formats::resize_rgba(output, w, h, ResizeFilter::Box));
    let output = shrunk.as_ref().unwrap_or(output);
    Agreement::of(&stats::rgb_samples(output), &stats::rgb_samples(truth)).map(|a| a.mean_abs)
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
/// the truth. `gpu` is the GPU path's score only when it ran on a GPU.
pub fn leaning(cpu: Option<&Score>, gpu: Option<&Score>, mismatch: bool) -> Option<Leaning> {
    let blank = |s: Option<&Score>| s.is_some_and(|s| s.flat.is_some());
    match (blank(cpu), blank(gpu)) {
        (true, true) => Some(Leaning::BothBlank),
        (true, false) => Some(Leaning::CpuBlank),
        (false, true) => Some(Leaning::GpuBlank),
        (false, false) if mismatch => {
            let (c, g) = (cpu?.off_truth?, gpu?.off_truth?);
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

/// How a side's blank output reads in the report.
pub fn blank_text(family: Family, level: u8) -> String {
    match family {
        Family::Colorize => "it adds no colour".into(),
        _ => format!("every value is {level}/255"),
    }
}

/// Each side's picture as `dir/<name>-<side>.png`, for a person to look
/// at; the paths written.
pub fn save(dir: &Path, name: &str, sides: [(&str, Option<&DynamicImage>); 2]) -> Result<Vec<String>, String> {
    let mut saved = Vec::new();
    for (side, picture) in sides {
        let Some(picture) = picture else { continue };
        if saved.is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let path = dir.join(format!("{name}-{side}.png"));
        picture.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        saved.push(path.display().to_string());
    }
    Ok(saved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn output(picture: RgbaImage, samples: Vec<u8>) -> Output {
        Output { samples, picture: Some(DynamicImage::ImageRgba8(picture)) }
    }

    #[test]
    fn a_blank_output_is_one_level_give_or_take_two() {
        assert_eq!(flat(&[0, 0, 1, 2]), Some(0));
        assert_eq!(flat(&[255, 254, 253]), Some(253));
        assert_eq!(flat(&[0, 3]), None);
        assert_eq!(flat(&[]), None);
        let inputs = Inputs::built_in();
        let none_found = Output { samples: vec![0], picture: None };
        assert_eq!(score(Family::FaceFinder, &inputs, &none_found).flat, None, "no faces is a list, not a blank picture");
        assert_eq!(score(Family::Depth, &inputs, &none_found).flat, Some(0));
    }

    /// An upscale that is its input enlarged lands back on it; a black
    /// one lands far off.
    #[test]
    fn an_upscale_is_scored_shrunk_back_onto_its_input() {
        let inputs = Inputs::built_in();
        let (w, h) = inputs.tile.dimensions();
        let enlarged = image::imageops::resize(&inputs.tile, w * 4, h * 4, image::imageops::FilterType::Nearest);
        let faithful = score(Family::Upscale, &inputs, &output(enlarged, vec![1, 9])).off_truth.expect("scored");
        assert!(faithful < 0.01, "{faithful}");
        let black = score(Family::Upscale, &inputs, &output(RgbaImage::new(w * 4, h * 4), vec![0, 0])).off_truth.expect("scored");
        assert!(black > 100.0, "{black}");
        assert_eq!(score(Family::Depth, &inputs, &output(inputs.tile.clone(), vec![1, 9])).off_truth, None, "depth knows no truth");
    }

    /// A colorizer that adds no colour is blank though its lightness
    /// varies; one that adds colour is not, and neither is scored against
    /// the photo's own colours.
    #[test]
    fn a_colorizer_is_blank_when_it_adds_no_colour() {
        let inputs = Inputs::built_in();
        let photo = inputs.photo.clone();
        let grey = DynamicImage::ImageRgba8(photo.clone()).grayscale().into_rgba8();
        let samples = stats::rgb_samples(&grey);
        let colourless = score(Family::Colorize, &inputs, &output(grey, samples));
        assert_eq!(colourless, Score { flat: Some(0), off_truth: None });
        let samples = stats::rgb_samples(&photo);
        assert_eq!(score(Family::Colorize, &inputs, &output(photo, samples)), Score::default());
        assert_eq!(colour(&RgbaImage::from_pixel(1, 1, Rgba([200, 90, 60, 255]))), [140]);
    }

    #[test]
    fn the_leaning_names_a_blank_side_or_on_a_mismatch_the_one_nearer_the_truth() {
        let near = |off| Score { flat: None, off_truth: Some(off) };
        let blank = Score { flat: Some(0), off_truth: Some(120.0) };
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
        let picture = DynamicImage::ImageLuma16(image::ImageBuffer::from_pixel(3, 2, image::Luma([40_000u16])));
        let saved = save(&outputs, "depth_anything", [("cpu", Some(&picture)), ("gpu", None)]).unwrap();
        assert_eq!(saved, [outputs.join("depth_anything-cpu.png").display().to_string()]);
        assert_eq!(image::open(&saved[0]).unwrap().into_luma16(), picture.into_luma16(), "16-bit depth reads back as written");
        assert_eq!(save(&outputs, "x", [("cpu", None), ("gpu", None)]).unwrap(), Vec::<String>::new());
    }
}
