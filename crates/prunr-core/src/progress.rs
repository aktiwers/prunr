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
    FindingFaces,
    RestoringFaces,
    Colorizing,
    EstimatingDepth,
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
            Step::FindingFaces => "Finding faces",
            Step::RestoringFaces => "Restoring faces",
            Step::Colorizing => "Adding color",
            Step::EstimatingDepth => "Measuring depth",
        }
    }
}

/// What a count counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Unit {
    Tile,
    Step,
    Crop,
    Image,
    Face,
}

impl Unit {
    /// "tile" / "tiles", for "Tile 3 of 48" and "48 tiles".
    pub fn noun(self, n: u32) -> &'static str {
        let (one, many) = match self {
            Unit::Tile => ("tile", "tiles"),
            Unit::Step => ("step", "steps"),
            Unit::Crop => ("crop", "crops"),
            Unit::Image => ("image", "images"),
            Unit::Face => ("face", "faces"),
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
    /// The outer count: crops of a stroke, images of a batch.
    Outer { done: u32, total: u32, unit: Unit },
    /// The count inside the current outer unit: tiles, denoising steps.
    Inner { done: u32, total: u32, unit: Unit },
    /// Where the work happens, once per layout; replaces any earlier one.
    Tiles(Vec<TileRect>),
    /// Tile `index` of the last layout started (`done: false`) or finished.
    Tile { index: u32, done: bool },
    /// Where the subject is, as soon as segmentation has found it.
    Subject(crate::subject::SubjectHint),
    /// The faces a face search found, as soon as it has, before any is
    /// redrawn.
    Faces(Vec<crate::face::FacePlace>),
    /// A finished tile's result, small, for the listener to show in place.
    TilePreview(TilePreview),
}

/// A finished tile at preview size: straight-alpha RGBA, `width × height`,
/// placed at (`x`, `y`) in a `full_width × full_height` preview of the
/// whole image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TilePreview {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub rgba: Vec<u8>,
}

/// Whoever listens to a pipeline's reports.
pub trait ProgressSink: Send + Sync {
    fn report(&self, update: ProgressUpdate);
}

/// Where the image a pipeline works on sits in the photo the listener
/// shows: a rectangle in the photo's pixels, and the photo's size.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frame {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    photo: (u32, u32),
}

impl Frame {
    /// `r`, given as fractions of the framed image, as fractions of the photo.
    fn map(&self, r: TileRect) -> TileRect {
        let (pw, ph) = (f64::from(self.photo.0.max(1)), f64::from(self.photo.1.max(1)));
        TileRect {
            x: ((self.x + f64::from(r.x) * self.w) / pw) as f32,
            y: ((self.y + f64::from(r.y) * self.h) / ph) as f32,
            w: (f64::from(r.w) * self.w / pw) as f32,
            h: (f64::from(r.h) * self.h / ph) as f32,
        }
    }
}

/// The handle a pipeline reports through and checks for cancel.
#[derive(Clone, Default)]
pub struct Progress {
    sink: Option<Arc<dyn ProgressSink>>,
    cancel: Option<Arc<AtomicBool>>,
    preview_edge: Option<u32>,
    /// `None`: the pipeline works on the photo itself.
    frame: Option<Frame>,
}

impl Progress {
    /// Nobody listens and nothing cancels.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn new(sink: Arc<dyn ProgressSink>) -> Self {
        Self { sink: Some(sink), ..Self::default() }
    }

    /// The handle for work on a copy, at any size, of `rect` (x, y, w, h)
    /// of the `size` image this handle's pipeline works on: tiles and
    /// previews reported through it land where `rect` sits in the photo.
    pub fn within(&self, rect: (u32, u32, u32, u32), size: (u32, u32)) -> Self {
        Self { frame: Some(self.place(rect, size)), ..self.clone() }
    }

    /// `rect` of the `size` image the pipeline works on, in the photo.
    fn place(&self, (x, y, w, h): (u32, u32, u32, u32), (width, height): (u32, u32)) -> Frame {
        let (x, y, w, h) = (f64::from(x), f64::from(y), f64::from(w), f64::from(h));
        match self.frame {
            None => Frame { x, y, w, h, photo: (width, height) },
            Some(f) => {
                let (sx, sy) = (f.w / f64::from(width.max(1)), f.h / f64::from(height.max(1)));
                Frame { x: f.x + x * sx, y: f.y + y * sy, w: w * sx, h: h * sy, photo: f.photo }
            }
        }
    }

    pub fn with_cancel(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = Some(flag);
        self
    }

    /// Ask for each finished tile as a preview whose whole image is at most
    /// `longest_edge` pixels on its longer side.
    pub fn with_tile_previews(mut self, longest_edge: u32) -> Self {
        self.preview_edge = Some(longest_edge);
        self
    }

    /// The preview size asked for, when someone listens for previews.
    pub fn preview_edge(&self) -> Option<u32> {
        self.sink.as_ref().and(self.preview_edge)
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

    /// The layout as the pipeline plans it, as fractions of the image it
    /// works on; the listener gets them as fractions of the photo, with
    /// overlapping tiles as cells that meet edge to edge.
    pub fn tiles(&self, rects: Vec<TileRect>) {
        if self.sink.is_none() {
            return;
        }
        let rects = match self.frame {
            None => rects,
            Some(f) => rects.into_iter().map(|r| f.map(r)).collect(),
        };
        self.report(ProgressUpdate::Tiles(split_overlaps(&rects)));
    }

    /// Report the finished `rect` (x, y, w, h) of the `size` image the
    /// pipeline works on, in its place on a preview of the whole photo
    /// whose longer side is at most the asked edge. `pixels(width,
    /// height)` gives the tile at that size as straight-alpha RGBA; it
    /// runs only when previews were asked for.
    pub fn tile_preview(&self, rect: (u32, u32, u32, u32), size: (u32, u32), pixels: impl FnOnce(u32, u32) -> Vec<u8>) {
        let Some(edge) = self.preview_edge() else { return };
        let f = self.place(rect, size);
        let (photo_w, photo_h) = f.photo;
        let s = (f64::from(edge) / f64::from(photo_w.max(photo_h).max(1))).min(1.0);
        let at = |v: f64| (v * s).round() as u32;
        let (full_width, full_height) = (at(f64::from(photo_w)).max(1), at(f64::from(photo_h)).max(1));
        let (x, y) = (at(f.x).min(full_width - 1), at(f.y).min(full_height - 1));
        let width = at(f.x + f.w).clamp(x + 1, full_width) - x;
        let height = at(f.y + f.h).clamp(y + 1, full_height) - y;
        let rgba = pixels(width, height);
        debug_assert_eq!(rgba.len(), (width * height * 4) as usize);
        self.report(ProgressUpdate::TilePreview(TilePreview { x, y, width, height, full_width, full_height, rgba }));
    }

    pub fn tile(&self, index: u32, done: bool) {
        self.report(ProgressUpdate::Tile { index, done });
    }

    /// Report where the subject is; `hint` runs only when someone listens.
    pub fn subject(&self, hint: impl FnOnce() -> crate::subject::SubjectHint) {
        if self.sink.is_some() {
            self.report(ProgressUpdate::Subject(hint()));
        }
    }
}

/// Trim each tile where it overlaps a neighbour, at the middle of the
/// overlap, so overlapping tiles paint as cells that meet edge to edge.
fn split_overlaps(rects: &[TileRect]) -> Vec<TileRect> {
    rects.iter().map(|t| {
        let mut cell = *t;
        for o in rects {
            let (ox1, oy1) = (o.x + o.w, o.y + o.h);
            if std::ptr::eq(o, t) || o.x >= t.x + t.w || ox1 <= t.x || o.y >= t.y + t.h || oy1 <= t.y {
                continue;
            }
            if o.x < t.x {
                let mid = (t.x + ox1) / 2.0;
                if mid > cell.x { cell.w -= mid - cell.x; cell.x = mid; }
            }
            if o.x > t.x { cell.w = cell.w.min((o.x + t.x + t.w) / 2.0 - cell.x); }
            if o.y < t.y {
                let mid = (t.y + oy1) / 2.0;
                if mid > cell.y { cell.h -= mid - cell.y; cell.y = mid; }
            }
            if o.y > t.y { cell.h = cell.h.min((o.y + t.y + t.h) / 2.0 - cell.y); }
        }
        cell
    }).collect()
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
            ProgressUpdate::TilePreview(ref t) => {
                if t.x + t.width > t.full_width || t.y + t.height > t.full_height || t.rgba.len() != (t.width * t.height * 4) as usize {
                    return Err(format!("preview tile {}×{} at {},{} outside its {}×{} image", t.width, t.height, t.x, t.y, t.full_width, t.full_height));
                }
            }
            ProgressUpdate::Step(_) | ProgressUpdate::Subject(_) | ProgressUpdate::Faces(_) => {}
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
    fn overlapping_tiles_meet_in_the_middle_of_the_overlap() {
        // Two columns overlapping by 0.1, two rows overlapping by 0.2.
        let t = |x, y| TileRect { x, y, w: 0.55, h: 0.6 };
        let cells = split_overlaps(&[t(0.0, 0.0), t(0.45, 0.0), t(0.0, 0.4), t(0.45, 0.4)]);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(cells[0].w, 0.5) && close(cells[0].h, 0.5), "{:?}", cells[0]);
        assert!(close(cells[1].x, 0.5) && close(cells[1].x + cells[1].w, 1.0), "{:?}", cells[1]);
        assert!(close(cells[3].y, 0.5) && close(cells[3].y + cells[3].h, 1.0), "{:?}", cells[3]);
        let apart = [TileRect { x: 0.0, y: 0.0, w: 0.2, h: 0.2 }, TileRect { x: 0.5, y: 0.5, w: 0.2, h: 0.2 }];
        assert_eq!(split_overlaps(&apart), apart, "tiles that do not touch stay whole");
    }

    /// Work on a quarter-size copy of a crop reports where the crop sits
    /// in the photo: its layout as fractions of the photo, its previews
    /// at the photo's preview scale.
    #[test]
    fn a_framed_handle_reports_where_its_crop_sits() {
        let sink = Arc::new(RecordingSink::default());
        let photo = Progress::new(sink.clone()).with_tile_previews(500);
        let crop = photo.within((200, 100, 400, 200), (1000, 500));
        crop.tiles(vec![TileRect::of_pixels(0, 0, 100, 50, 100, 50)]);
        crop.within((0, 0, 50, 25), (100, 50)).tiles(vec![TileRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 }]);
        crop.tile_preview((50, 0, 50, 50), (100, 50), |w, h| vec![7; (w * h * 4) as usize]);
        photo.tile_preview((10, 0, 20, 500), (1000, 500), |w, h| vec![7; (w * h * 4) as usize]);
        let close = |a: &TileRect, b: TileRect| [a.x - b.x, a.y - b.y, a.w - b.w, a.h - b.h].iter().all(|d| d.abs() < 1e-6);
        let updates = sink.updates();
        let ProgressUpdate::Tiles(t) = &updates[0] else { panic!("{:?}", updates[0]) };
        assert!(close(&t[0], TileRect { x: 0.2, y: 0.2, w: 0.4, h: 0.4 }), "{t:?}");
        let ProgressUpdate::Tiles(t) = &updates[1] else { panic!("{:?}", updates[1]) };
        assert!(close(&t[0], TileRect { x: 0.2, y: 0.2, w: 0.2, h: 0.2 }), "a frame within a frame: {t:?}");
        let place = |u: &ProgressUpdate| match u {
            ProgressUpdate::TilePreview(p) => (p.x, p.y, p.width, p.height, p.full_width, p.full_height),
            u => panic!("{u:?}"),
        };
        assert_eq!(place(&updates[2]), (200, 50, 100, 100, 500, 250), "the crop's right half, at half the photo's size");
        assert_eq!(place(&updates[3]), (5, 0, 10, 250, 500, 250), "unframed: the photo's own pixels, halved");
        assert_eq!(check_contract(&[ProgressUpdate::Step(Step::Erasing), updates[2].clone()]), Ok(()));

        let ran = std::cell::Cell::new(false);
        Progress::new(sink.clone()).tile_preview((0, 0, 1, 1), (1, 1), |_, _| { ran.set(true); vec![0; 4] });
        Progress::none().with_tile_previews(500).tile_preview((0, 0, 1, 1), (1, 1), |_, _| { ran.set(true); vec![0; 4] });
        assert!(!ran.get(), "no preview is made unless asked for and listened to");
    }

    #[test]
    fn every_step_has_words_and_units_count() {
        assert_eq!(Step::LoadingMagicBrush.label(), "Loading Magic Brush");
        assert_eq!((Unit::Tile.noun(1), Unit::Tile.noun(48)), ("tile", "tiles"));
        assert_eq!(TileRect::of_pixels(256, 0, 256, 512, 1024, 512), TileRect { x: 0.25, y: 0.0, w: 0.25, h: 1.0 });
    }
}
