//! One progress handle for every long-running pipeline.
//!
//! A pipeline names its steps (`Step`), counts its units (tiles, denoising
//! steps, crops) and lays out the tiles it works on; the handle passes each
//! report to whoever listens (`ProgressSink`): the app's progress pill, the
//! subprocess's IPC writer, the CLI, a test. Cancel travels on the same
//! handle, so a pipeline that reports also stops. With no listener every
//! call is a branch on `None`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// A named step, with the words the app shows for it. One table, so the
/// wording stays the same wherever a step is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Step {
    LoadingModel,
    ReadingImage,
    FindingSubject,
    RefiningEdges,
    DrawingLines,
    Upscaling,
    Finishing,
    Erasing,
    ReadingPrompt,
    Denoising,
    DecodingResult,
    Blending,
    LoadingMagicBrush,
}

impl Step {
    pub fn label(self) -> &'static str {
        match self {
            Step::LoadingModel => "Loading the model",
            Step::ReadingImage => "Reading the image",
            Step::FindingSubject => "Finding the subject",
            Step::RefiningEdges => "Refining the edges",
            Step::DrawingLines => "Drawing the lines",
            Step::Upscaling => "Upscaling",
            Step::Finishing => "Finishing",
            Step::Erasing => "Erasing",
            Step::ReadingPrompt => "Reading the prompt",
            Step::Denoising => "Denoising",
            Step::DecodingResult => "Decoding the result",
            Step::Blending => "Blending into the photo",
            Step::LoadingMagicBrush => "Loading Magic Brush",
        }
    }
}

/// What a count counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Unit {
    Tile,
    Step,
    Crop,
    Pass,
    Image,
}

impl Unit {
    /// "tile" / "tiles", for "Tile 3 of 48" and "48 tiles".
    pub fn noun(self, n: u32) -> &'static str {
        let (one, many) = match self {
            Unit::Tile => ("tile", "tiles"),
            Unit::Step => ("step", "steps"),
            Unit::Crop => ("crop", "crops"),
            Unit::Pass => ("pass", "passes"),
            Unit::Image => ("image", "images"),
        };
        if n == 1 { one } else { many }
    }
}

/// A rectangle as fractions of the image the pipeline works on, so the
/// tile map draws over the displayed image at any size or zoom.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TileRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl TileRect {
    /// A pixel rectangle inside a `width × height` image.
    pub fn of_pixels(x: u32, y: u32, w: u32, h: u32, width: u32, height: u32) -> Self {
        let (fw, fh) = (width.max(1) as f32, height.max(1) as f32);
        Self { x: x as f32 / fw, y: y as f32 / fh, w: w as f32 / fw, h: h as f32 / fh }
    }
}

/// One report from a pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProgressUpdate {
    Step(Step),
    /// The outer count: crops of a stroke, passes of a two-pass upscale.
    Outer { done: u32, total: u32, unit: Unit },
    /// The count inside the current outer unit: tiles, denoising steps.
    Inner { done: u32, total: u32, unit: Unit },
    /// Where the work happens, once per layout; replaces any earlier one.
    Tiles(Vec<TileRect>),
    /// Tile `index` of the last layout started (`done: false`) or finished.
    Tile { index: u32, done: bool },
}

/// Whoever listens to a pipeline's reports.
pub trait ProgressSink: Send + Sync {
    fn report(&self, update: ProgressUpdate);
}

/// The handle a pipeline reports through and checks for cancel.
#[derive(Clone, Default)]
pub struct Progress {
    sink: Option<Arc<dyn ProgressSink>>,
    cancel: Option<Arc<AtomicBool>>,
}

impl Progress {
    /// Nobody listens and nothing cancels.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn new(sink: Arc<dyn ProgressSink>) -> Self {
        Self { sink: Some(sink), cancel: None }
    }

    pub fn with_cancel(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = Some(flag);
        self
    }

    pub fn cancel_flag(&self) -> Option<&Arc<AtomicBool>> {
        self.cancel.as_ref()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|f| f.load(Ordering::Acquire))
    }

    pub fn report(&self, update: ProgressUpdate) {
        if let Some(sink) = &self.sink {
            sink.report(update);
        }
    }

    pub fn step(&self, step: Step) {
        self.report(ProgressUpdate::Step(step));
    }

    pub fn outer(&self, done: u32, total: u32, unit: Unit) {
        self.report(ProgressUpdate::Outer { done, total, unit });
    }

    pub fn inner(&self, done: u32, total: u32, unit: Unit) {
        self.report(ProgressUpdate::Inner { done, total, unit });
    }

    pub fn tiles(&self, rects: Vec<TileRect>) {
        if self.sink.is_some() {
            self.report(ProgressUpdate::Tiles(rects));
        }
    }

    pub fn tile(&self, index: u32, done: bool) {
        self.report(ProgressUpdate::Tile { index, done });
    }
}

/// Keeps every report, for tests and for `check_contract`.
#[derive(Default)]
pub struct RecordingSink {
    updates: Mutex<Vec<ProgressUpdate>>,
}

impl RecordingSink {
    pub fn updates(&self) -> Vec<ProgressUpdate> {
        self.updates.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

impl ProgressSink for RecordingSink {
    fn report(&self, update: ProgressUpdate) {
        self.updates.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(update);
    }
}

/// What every pipeline's reports for a finished run must satisfy, so the
/// pill can follow any pipeline without knowing it: at least one named
/// step; counts within their totals, never going backwards while the total
/// stands (an outer report starts the inner count afresh), and ending
/// complete; tile updates naming a tile of the last
/// layout, inside the image.
pub fn check_contract(updates: &[ProgressUpdate]) -> Result<(), String> {
    if !updates.iter().any(|u| matches!(u, ProgressUpdate::Step(_))) {
        return Err("no step was named".into());
    }
    let mut last: [Option<(u32, u32, Unit)>; 2] = [None, None];
    let mut tiles = 0usize;
    for u in updates {
        match *u {
            ProgressUpdate::Outer { done, total, unit } | ProgressUpdate::Inner { done, total, unit } => {
                let slot = usize::from(matches!(u, ProgressUpdate::Inner { .. }));
                if total == 0 || done > total {
                    return Err(format!("{u:?}: count outside its total"));
                }
                if let Some((prev, prev_total, prev_unit)) = last[slot] {
                    if prev_total == total && prev_unit == unit && done < prev {
                        return Err(format!("{u:?}: went back from {prev}"));
                    }
                }
                last[slot] = Some((done, total, unit));
                if slot == 0 {
                    // The next outer unit (crop, pass) counts its inner units afresh.
                    last[1] = None;
                }
            }
            ProgressUpdate::Tiles(ref rects) => {
                let inside = |r: &TileRect| r.x >= 0.0 && r.y >= 0.0 && r.x + r.w <= 1.0 + 1e-4 && r.y + r.h <= 1.0 + 1e-4;
                if let Some(r) = rects.iter().find(|r| !inside(r)) {
                    return Err(format!("tile {r:?} outside the image"));
                }
                tiles = rects.len();
            }
            ProgressUpdate::Tile { index, .. } => {
                if index as usize >= tiles {
                    return Err(format!("tile {index} of a layout of {tiles}"));
                }
            }
            ProgressUpdate::Step(_) => {}
        }
    }
    for (done, total, unit) in last.into_iter().flatten() {
        if done != total {
            return Err(format!("ended at {done} of {total} {}", unit.noun(total)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_reach_the_sink_and_cancel_rides_along() {
        let sink = Arc::new(RecordingSink::default());
        let flag = Arc::new(AtomicBool::new(false));
        let p = Progress::new(sink.clone()).with_cancel(flag.clone());
        p.step(Step::Upscaling);
        p.inner(1, 2, Unit::Tile);
        assert_eq!(sink.updates(), vec![
            ProgressUpdate::Step(Step::Upscaling),
            ProgressUpdate::Inner { done: 1, total: 2, unit: Unit::Tile },
        ]);
        assert!(!p.is_cancelled());
        flag.store(true, Ordering::Release);
        assert!(p.is_cancelled());
        assert!(!Progress::none().is_cancelled());
        Progress::none().step(Step::Erasing);
    }

    #[test]
    fn the_contract_accepts_a_finished_run_and_names_what_breaks_it() {
        use ProgressUpdate::*;
        let ok = [
            Step(super::Step::Upscaling),
            Tiles(vec![TileRect::of_pixels(0, 0, 50, 100, 100, 100), TileRect::of_pixels(50, 0, 50, 100, 100, 100)]),
            Tile { index: 0, done: false }, Inner { done: 1, total: 2, unit: Unit::Tile }, Tile { index: 0, done: true },
            Tile { index: 1, done: false }, Inner { done: 2, total: 2, unit: Unit::Tile }, Tile { index: 1, done: true },
        ];
        assert_eq!(check_contract(&ok), Ok(()));
        assert!(check_contract(&ok[1..]).unwrap_err().contains("no step"));
        assert!(check_contract(&ok[..4]).unwrap_err().contains("ended at 1 of 2"));
        let back = [Step(super::Step::Denoising), Inner { done: 3, total: 9, unit: Unit::Step }, Inner { done: 2, total: 9, unit: Unit::Step }];
        assert!(check_contract(&back).unwrap_err().contains("went back"));
        let stray = [Step(super::Step::Erasing), Tile { index: 0, done: true }];
        assert!(check_contract(&stray).unwrap_err().contains("layout of 0"));
        // A new total (the next crop's steps) may start again from zero.
        let next_crop = [Step(super::Step::Denoising), Inner { done: 4, total: 4, unit: Unit::Step }, Inner { done: 0, total: 6, unit: Unit::Step }, Inner { done: 6, total: 6, unit: Unit::Step }];
        assert_eq!(check_contract(&next_crop), Ok(()));
        let two_crops = [
            Step(super::Step::Denoising),
            Outer { done: 0, total: 2, unit: Unit::Crop }, Inner { done: 20, total: 20, unit: Unit::Step },
            Outer { done: 1, total: 2, unit: Unit::Crop }, Inner { done: 0, total: 20, unit: Unit::Step }, Inner { done: 20, total: 20, unit: Unit::Step },
            Outer { done: 2, total: 2, unit: Unit::Crop },
        ];
        assert_eq!(check_contract(&two_crops), Ok(()), "each crop counts its steps afresh");
    }

    #[test]
    fn every_step_has_words_and_units_count() {
        assert_eq!(Step::LoadingMagicBrush.label(), "Loading Magic Brush");
        assert_eq!((Unit::Tile.noun(1), Unit::Tile.noun(48)), ("tile", "tiles"));
        assert_eq!(TileRect::of_pixels(256, 0, 256, 512, 1024, 512), TileRect { x: 0.25, y: 0.0, w: 0.25, h: 1.0 });
    }
}
