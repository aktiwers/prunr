//! The sequence explorer: random sequences of what a person does (press a
//! control, drag a slider, paint a stroke, press a key, wait for work) on
//! the real app through the test harness, a stand-in answering for the
//! cut-out worker, and rules checked after every step. Ordinary tests
//! start from a tidy window and do one thing; the bugs this looks for
//! need a history (pick, process, undo, then a slider). A sequence that
//! breaks a rule is cut down to the shortest that still does and printed
//! as steps to paste into a test.
//!
//! Not picked yet, each for want of a stand-in for its engine: the Stable
//! Diffusion erasers (their own process), Faces and the depth looks.

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;

use egui_kittest::Harness;

use super::harness::{harness, settle, tree};
use crate::gui::app::PrunrApp;
use crate::gui::item_settings::ItemSettings;
use crate::gui::settings::SettingsModel;
use crate::gui::views::shortcuts::Chord;
use crate::gui::worker::{CompressedTensor, RawSegTensor, WorkerMessage, WorkerResult};

/// xorshift64*: a sequence is its seed, so a failure replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn percent(&mut self) -> u8 {
        self.below(101) as u8
    }
}

/// One thing a person does. Replayed on a window where it does not apply
/// (no such control, it is greyed or covered, no brush in hand), it is
/// skipped, so a shortened sequence stays a valid one.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    Click(String),
    /// Press the slider named so at this percent of its width.
    Slide(String, u8),
    /// Drag over the image with the brush in hand, between two points
    /// given in percent of the image's width and height.
    Stroke([u8; 4]),
    Key(&'static str),
    /// The worker answers every run asked for, and the window settles.
    Work,
}

const KEYS: [&str; 3] = ["Mod+Z", "Mod+Shift+Z", "Escape"];

/// Controls the explorer leaves alone, by the start of their name, each
/// for its reason.
const LEFT_ALONE: &[(&str, &str)] = &[
    ("Open", "asks the system for files"),
    ("Save", "asks the system for a file; Save options and Save current as… with it"),
    ("Import", "asks the system for files"),
    ("Export preset", "asks the system for a file"),
    ("Show folder", "opens the file manager"),
    ("Model Store", "downloads and removals change the machine"),
    ("Get more models", "opens the Model Store"),
    ("Settings", "changes the machine's settings file"),
    ("Help", "its pages change nothing"),
    ("Delete preset", "removes a preset file"),
    ("Set as default", "writes the settings file"),
    ("Eraser (Stable", "runs in a process of its own, which a test window does not start"),
    ("Eraser (RORem", "runs in a process of its own, which a test window does not start"),
    ("Faces", "no stand-in for the face models yet"),
    ("Depth", "no stand-in for the depth model yet"),
    ("Lens", "a depth look"),
    ("Golden Bokeh", "a depth look"),
    ("Teal and Orange", "a depth look on No model"),
    ("Pick image", "asks the system for a file"),
];

fn left_alone(name: &str) -> bool {
    LEFT_ALONE.iter().any(|(start, _)| name.starts_with(start))
}

/// Controls that end a scenario early (an image gone, a run stopped):
/// picked one time in ten, so sequences grow a history first.
fn ends_things(name: &str) -> bool {
    ["Delete photo", "Remove", "Cancel"].iter().any(|start| name.starts_with(start))
}

/// The cut-out worker's stand-in: it holds the other ends of the
/// processor's two channels and answers each run with a cut made by the
/// real postprocess from a tensor of its own, so the app's own landing
/// code runs and the pictures are real maths.
struct StandIn {
    asked: mpsc::Receiver<WorkerMessage>,
    answers: mpsc::Sender<WorkerResult>,
    answered: std::cell::Cell<usize>,
}

impl StandIn {
    /// A soft ellipse, the same for every run.
    fn tensor(model: prunr_core::ModelKind) -> RawSegTensor {
        const SIDE: u32 = 32;
        let data = (0..SIDE * SIDE)
            .map(|i| {
                let (x, y) = ((i % SIDE) as f32 / (SIDE - 1) as f32 - 0.5, (i / SIDE) as f32 / (SIDE - 1) as f32 - 0.5);
                (1.4 - 3.2 * (x * x + y * y).sqrt()).clamp(0.0, 1.0)
            })
            .collect();
        RawSegTensor { data, height: SIDE, width: SIDE, model }
    }

    fn cut(photo: &image::RgbaImage, raw: &RawSegTensor, mask: &prunr_core::MaskSettings) -> Result<prunr_core::ProcessResult, String> {
        let opts = prunr_core::PostprocessOpts::new(mask, raw.model);
        prunr_core::postprocess_from_flat(&raw.data, raw.height as usize, raw.width as usize, photo, &opts)
            .map(|rgba_image| prunr_core::ProcessResult { rgba_image, active_provider: "CPU".into() })
            .map_err(|e| e.to_string())
    }

    /// The lines model's stand-in: at half the picture's size (the Fine
    /// scale at twice that), how much each point's lightness differs from
    /// its right and lower neighbours.
    fn trace(image: &image::DynamicImage) -> Result<prunr_core::EdgeInferenceResult, prunr_core::CoreError> {
        let grey = image.to_luma8();
        let (height, width) = ((grey.height() / 2).max(1), (grey.width() / 2).max(1));
        let tensors = prunr_core::EdgeScale::ALL.map(|scale| {
            let (h, w) = prunr_core::edge_scale_dims(scale, height, width);
            let small = image::imageops::resize(&grey, w, h, image::imageops::FilterType::Triangle);
            let at = |x: u32, y: u32| f32::from(small.get_pixel(x.min(w - 1), y.min(h - 1))[0]);
            (0..h * w).map(|i| {
                let (x, y) = (i % w, i / w);
                (((at(x + 1, y) - at(x, y)).abs() + (at(x, y + 1) - at(x, y)).abs()) / 64.0).min(1.0)
            }).collect()
        });
        Ok(prunr_core::EdgeInferenceResult { tensors, height, width })
    }

    fn lines(traced: prunr_core::EdgeInferenceResult) -> Option<crate::gui::worker::CompressedEdgeTensors> {
        crate::gui::worker::CompressedEdgeTensors::from_raw(crate::gui::worker::RawEdgeTensors { tensors: traced.tensors, height: traced.height, width: traced.width, spare: None })
    }

    /// Lines over the subject: the plain cut traced, its effect put on, the
    /// lines drawn over it, as the worker composes them.
    fn outline(subject: image::RgbaImage, effect: prunr_core::SubjectEffect, edge: &prunr_core::EdgeSettings) -> Result<(prunr_core::ProcessResult, Option<crate::gui::worker::CompressedEdgeTensors>), String> {
        prunr_core::outline_subject(subject, effect, edge, None, Self::trace)
            .map(|(rgba_image, traced)| (prunr_core::ProcessResult { rgba_image, active_provider: "CPU".into() }, Self::lines(traced)))
            .map_err(|e| e.to_string())
    }

    /// A full run of `picture` (the photo, or the start it runs from when
    /// `chained`) under `run`'s Sketch mode, as the worker makes it: the
    /// result, the segmentation kept and the lines traced.
    fn full_run(picture: image::RgbaImage, chained: bool, model: prunr_core::ModelKind, run: &crate::gui::worker::RunSettings) -> (Result<prunr_core::ProcessResult, String>, Option<CompressedTensor>, Option<crate::gui::worker::CompressedEdgeTensors>) {
        use prunr_core::LineMode;
        match run.line_mode {
            LineMode::Off => {
                let raw = Self::tensor(model);
                (Self::cut(&picture, &raw, &run.mask), CompressedTensor::from_raw(raw), None)
            }
            LineMode::EdgesOnly => {
                let picture = image::DynamicImage::ImageRgba8(picture);
                match Self::trace(&prunr_core::apply_input_transform(&picture, run.edge.input_transform)) {
                    Ok(traced) => {
                        let rgba_image = prunr_core::compose_whole_photo(&traced, &picture, &run.edge, None);
                        (Ok(prunr_core::ProcessResult { rgba_image, active_provider: "CPU".into() }), None, Self::lines(traced))
                    }
                    Err(e) => (Err(e.to_string()), None, None),
                }
            }
            // A run on an earlier result: that image is the cut-out already.
            LineMode::SubjectOutline if chained => match Self::outline(picture, prunr_core::SubjectEffect::None, &run.edge) {
                Ok((result, lines)) => (Ok(result), None, lines),
                Err(e) => (Err(e), None, None),
            },
            LineMode::SubjectOutline => {
                let raw = Self::tensor(model);
                let opts = prunr_core::PostprocessOpts::new(&run.mask, raw.model);
                let made = prunr_core::cut_for_lines(&raw.data, raw.height as usize, raw.width as usize, &picture, &opts)
                    .map_err(|e| e.to_string())
                    .and_then(|subject| Self::outline(subject, run.mask.effect, &run.edge));
                match made {
                    Ok((result, lines)) => (Ok(result), CompressedTensor::from_raw(raw), lines),
                    Err(e) => (Err(e), None, None),
                }
            }
        }
    }

    /// Answer every run asked for so far; how many images that was.
    fn answer(&self, app: &PrunrApp) -> usize {
        let mut answered = 0;
        while let Ok(msg) = self.asked.try_recv() {
            let WorkerMessage::BatchProcess { batch, items, tier2_items, add_edge_items, config, .. } = msg else { continue };
            let model = config.engines.model;
            let photo_of = |id: u64| app.batch.find_by_id(id).and_then(photo);
            for item in &items {
                // What the bridge reads and sends: the start the run cuts
                // from, and a chained cut's start, which it reports back.
                let (result, seg, edges) = match item.inputs_for_test() {
                    Err(err) => (Err(err), None, None),
                    Ok((start, built)) => {
                        if let Some(image) = built {
                            let _ = self.answers.send(WorkerResult::ChainStart { item_id: item.id, run: item.run, image });
                        }
                        let chained = start.is_some();
                        match start.map(|s| (*s).clone()).or_else(|| photo_of(item.id)) {
                            Some(picture) => Self::full_run(picture, chained, model, &config.run),
                            None => (Err("the photo could not be read".to_string()), None, None),
                        }
                    }
                };
                let _ = self.answers.send(WorkerResult::BatchItemDone { batch, item_id: item.id, result, seg, edges });
                answered += 1;
            }
            // A mask re-run is cut from its kept tensor; turning the
            // outline on cuts it and traces the cut.
            for (work, traces) in tier2_items.iter().map(|w| (w, false)).chain(add_edge_items.iter().map(|w| (w, true))) {
                let made = work
                    .tensor
                    .bundle()
                    .ok_or_else(|| "the kept tensor could not be read".to_string())
                    .and_then(|raw| {
                        let photo = work.photo.as_deref().cloned().or_else(|| photo_of(work.item_id)).ok_or_else(|| "the photo could not be read".to_string())?;
                        if !traces {
                            return Self::cut(&photo, &raw, &work.mask).map(|result| (result, None));
                        }
                        let opts = prunr_core::PostprocessOpts::new(&work.mask, raw.model);
                        let subject = prunr_core::cut_for_lines(&raw.data, raw.height as usize, raw.width as usize, &photo, &opts).map_err(|e| e.to_string())?;
                        Self::outline(subject, work.mask.effect, &config.run.edge)
                    });
                let (result, edges) = match made {
                    Ok((result, edges)) => (Ok(result), edges),
                    Err(e) => (Err(e), None),
                };
                let _ = self.answers.send(WorkerResult::BatchItemDone { batch, item_id: work.item_id, result, seg: None, edges });
                answered += 1;
            }
            let _ = self.answers.send(WorkerResult::BatchComplete { batch });
        }
        self.answered.set(self.answered.get() + answered);
        answered
    }
}

/// How many erases and Magic Brush picks the stand-in answered.
#[derive(Default, Clone, Copy)]
struct ModelsAnswered {
    erases: usize,
    picks: usize,
    upscales: usize,
    colorizes: usize,
}

/// The eraser's, the Magic Brush's, the upscalers' and Colorize's
/// stand-in: what the test app was asked to run, answered as the models
/// would: a fill that greys the region, an encode, a pick of a soft disc
/// round the click (whose edge moves with the Confidence slider), the
/// picture enlarged, and a warm colour.
fn answer_models(app: &PrunrApp, answered: &std::cell::Cell<ModelsAnswered>) {
    use crate::gui::processor::ModelAsk;
    for ask in app.processor.take_asks_for_test() {
        let mut count = answered.get();
        match ask {
            ModelAsk::Erase { item_id, generation, image, region } => {
                let fill = erase_fill(&image, &region);
                app.processor.answer_erase_for_test(item_id, generation, fill);
                count.erases += 1;
            }
            ModelAsk::Encode { item_id, saw } => app.processor.answer_encode_for_test(item_id, saw),
            ModelAsk::Decode { item_id, at, mode, reading, edge, side } => {
                let output = std::sync::Arc::new(disc_pick(at));
                let drawn = crate::gui::brush_state::StrokeDrawn::Magic { output, mode, reading, edge, side };
                app.processor.answer_decode_for_test(item_id, drawn);
                count.picks += 1;
            }
            ModelAsk::Upscale { item_id, run, input, recipe } => {
                app.processor.answer_upscale_for_test(item_id, run, &input, recipe);
                count.upscales += 1;
            }
            ModelAsk::Colorize { item_id, run, input, recipe } => {
                app.processor.answer_colorize_for_test(item_id, run, &input, recipe);
                count.colorizes += 1;
            }
        }
        answered.set(count);
    }
}

/// `image` with the cells `region` selects greyed out: a fill.
fn erase_fill(image: &crate::gui::item::Pixels, region: &crate::gui::item::Region) -> image::RgbaImage {
    let Ok(picture) = image.read() else { return image::RgbaImage::new(1, 1) };
    let mut fill = (*picture).clone();
    if let Some(mask) = region.load().filter(|m| (m.width, m.height) == fill.dimensions()) {
        for (pixel, &cell) in fill.pixels_mut().zip(mask.cells()) {
            if prunr_core::selection::MaskArtifact::is_selected(cell) {
                *pixel = image::Rgba([128, 128, 128, 255]);
            }
        }
    }
    fill
}

/// A pick's decoder output: a disc round `at` (the decoder's 1024-wide
/// space), sure at its middle and less so outward.
fn disc_pick(at: (f32, f32)) -> prunr_core::sam::SamDecoderOutput {
    let side = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
    let (cx, cy) = (at.0 / 4.0, at.1 / 4.0);
    let mut masks = vec![-9.0f32; 3 * side * side];
    for (i, logit) in masks.iter_mut().take(side * side).enumerate() {
        let (x, y) = ((i % side) as f32 - cx, (i / side) as f32 - cy);
        *logit = 6.0 - (x * x + y * y).sqrt() / 6.0;
    }
    prunr_core::sam::SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] }.best_only()
}

/// The image's photo: decoded already, or from its bytes.
fn photo(item: &crate::gui::item::BatchItem) -> Option<image::RgbaImage> {
    if let Some(decoded) = &item.source_rgba {
        return Some((**decoded).clone());
    }
    let bytes = item.source.load_bytes().ok()?;
    image::load_from_memory(&bytes).ok().map(|img| img.into_rgba8())
}

/// A small photo with something in it: a gradient under a bright disc.
fn photo_png(seed: u8) -> Vec<u8> {
    let (w, h) = (240u32, 160u32);
    let img = image::RgbaImage::from_fn(w, h, |x, y| {
        let (dx, dy) = (x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0);
        if dx * dx + dy * dy < 2200.0 {
            image::Rgba([230, 60u8.wrapping_add(seed.wrapping_mul(40)), 50, 255])
        } else {
            image::Rgba([x as u8, (y as u8).wrapping_mul(3) / 2, 90u8.wrapping_add(seed.wrapping_mul(30)), 255])
        }
    });
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).expect("a PNG in memory");
    bytes
}

struct World {
    h: Harness<'static, PrunrApp>,
    worker: StandIn,
    models: std::cell::Cell<ModelsAnswered>,
}

impl World {
    /// The window with two photos opened, the first on the canvas, and
    /// the showcases in the preset list.
    fn new() -> Self {
        let mut h = harness();
        let (to_worker, asked) = mpsc::channel();
        let (answers, from_worker) = mpsc::channel();
        {
            let app = h.state_mut();
            app.processor.worker_tx = to_worker;
            app.processor.worker_rx = from_worker;
            for (name, preset) in crate::gui::presets_fs::showcases() {
                app.settings.presets.insert((*name).to_string(), preset.clone());
            }
            for id in [1u8, 2] {
                app.add_to_batch(photo_png(id), format!("photo{id}.png"));
            }
            app.batch.select_item(0);
        }
        let mut world = Self { h, worker: StandIn { asked, answers, answered: std::cell::Cell::new(0) }, models: Default::default() };
        world.quiet();
        world
    }

    fn app(&self) -> &PrunrApp {
        self.h.state()
    }

    fn busy(&self) -> bool {
        let app = self.app();
        crate::gui::automation::state::busy(app)
            || app.batch.selection_queue.any_pending()
            || app.magic_brush_state.has_pending_encoder()
            || app.processor.any_inpaint_in_flight()
            || app.batch.items.iter().any(|item| app.processor.selecting(item.id))
    }

    /// Frames, the worker answering between them, until nothing is owed
    /// or two minutes pass. Whether it went quiet. Time, not a count of
    /// frames: a picture enlarged three times over (157 MP from these
    /// photos) takes the debug build over a minute to make its texture
    /// and thumbnail, which is slow, not stuck.
    fn quiet(&mut self) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while std::time::Instant::now() < deadline {
            self.worker.answer(self.h.state());
            answer_models(self.h.state(), &self.models);
            self.h.step();
            if !self.busy() {
                settle(&mut self.h);
                self.worker.answer(self.h.state());
                answer_models(self.h.state(), &self.models);
                self.h.step();
                if !self.busy() {
                    return true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        false
    }

    /// The controls a person could press now, by name, with whether each
    /// is a slider: enabled, on top (nothing covers them), not left alone.
    fn offered(&self) -> Vec<(String, bool)> {
        let offered: BTreeSet<(String, bool)> = tree(&self.h)
            .iter()
            .filter(|n| n.is_control() && !n.disabled && crate::gui::automation::reachable_at(&self.h.ctx, n.id).is_some())
            .filter_map(|n| Some((n.name.clone()?, n.role == accesskit::Role::Slider)))
            .filter(|(name, _)| !left_alone(name))
            .collect();
        offered.into_iter().collect()
    }

    /// The control named `name` a person could press, the last of them:
    /// its place, where a press reaches it, and how it reads (its role
    /// and whether it is on).
    fn pressable(&self, name: &str) -> Option<Pressable> {
        tree(&self.h)
            .iter()
            .filter(|n| n.is_control() && !n.disabled && n.name.as_deref() == Some(name))
            .filter_map(|n| {
                let at = crate::gui::automation::reachable_at(&self.h.ctx, n.id)?;
                Some(Pressable { at, rect: n.rect?, role: format!("{:?}", n.role), on: n.toggled })
            })
            .last()
    }

    /// A pointer press at `from`, a drag to `to` when it differs, and the
    /// release, a frame each, as a mouse makes.
    fn drag(&mut self, from: egui::Pos2, to: egui::Pos2) {
        let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::default() };
        let mut events = vec![egui::Event::PointerMoved(from), button(from, true)];
        if to != from {
            events.extend((1..=4).map(|i| egui::Event::PointerMoved(from + (to - from) * (i as f32 / 4.0))));
        }
        events.push(button(to, false));
        for event in events {
            self.h.input_mut().events.push(event);
            self.h.step();
        }
        settle(&mut self.h);
    }

    /// Do `action`; false when it did not apply here.
    fn act(&mut self, action: &Action) -> bool {
        match action {
            Action::Click(name) => {
                let Some(control) = self.pressable(name) else { return false };
                self.drag(control.at, control.at);
            }
            Action::Slide(name, percent) => {
                let Some(control) = self.pressable(name).filter(|c| c.role == "Slider") else { return false };
                let [left, top, right, bottom] = control.rect;
                let at = egui::pos2(left + (right - left) * f32::from(*percent) / 100.0, (top + bottom) / 2.0);
                self.drag(at, at);
            }
            Action::Stroke([ax, ay, bx, by]) => {
                let Some(image) = self.app().zoom_state.shown_at.filter(|_| self.app().active_tool().is_some()) else { return false };
                let point = |x: u8, y: u8| image.min + egui::vec2(image.width() * f32::from(x) / 100.0, image.height() * f32::from(y) / 100.0);
                let (from, to) = (point(*ax, *ay), point(*bx, *by));
                // A press a panel covers lands on the panel, not the image.
                if self.h.ctx.layer_id_at(from).is_some_and(|layer| layer.order != egui::Order::Background) {
                    return false;
                }
                self.drag(from, to);
            }
            Action::Key(chord) => {
                let chord = Chord::parse(chord).expect("a chord the app knows");
                self.h.key_press_modifiers(chord.modifiers(), chord.key);
                settle(&mut self.h);
            }
            Action::Work => {
                self.quiet();
            }
        }
        true
    }

    /// The brush in hand has strokes its sliders draw again, with
    /// something to show for it: Last or All is on, it drew a stroke since
    /// the picture last changed, and a selection covering part of the
    /// image is on screen (one cleared since, or one a huge brush filled
    /// the image with, shows no change of edge or size).
    fn brush_reaches_strokes(&self) -> bool {
        use crate::gui::brush_state::StrokeReach;
        let app = self.app();
        let (Some(tool), Some(item)) = (app.active_tool(), app.batch.selected_item()) else { return false };
        // Selected at least half (a faint stroke at low Opacity leaves no
        // edge for Feather to soften), and not all of it.
        let partly_selected = item.selection_mask.as_ref().is_some_and(|mask| {
            let cells = mask.cells();
            cells.contains(&0) && cells.iter().any(|&c| prunr_core::selection::MaskArtifact::is_selected(c))
        });
        if !partly_selected {
            return false;
        }
        // A stroke keeps the kind of edge it was drawn with: under a
        // cut-out model the selection's Feather does not shape a stroke
        // painted under an eraser (Expand and a fill band), nor the
        // reverse. The sliders reach the strokes whose edge they shape.
        use crate::gui::selection_queue::SelectionMade;
        let kind = std::mem::discriminant(&app.settings.stroke_edge());
        let shaped = |history: crate::gui::timeline::StrokeHistory, latest_only: bool| {
            let mut own = history.ops.iter().filter(|op| op.drawn_by(tool)).filter_map(|op| match op {
                SelectionMade::Stroke(drawn) => Some(std::mem::discriminant(&drawn.edge()) == kind),
                _ => None,
            });
            if latest_only { own.next_back().unwrap_or(false) } else { own.any(|same| same) }
        };
        match app.brush_state.reach {
            StrokeReach::Next => false,
            StrokeReach::Last => item.timeline.latest_stroke(tool).is_some_and(|h| shaped(h, true)),
            StrokeReach::All => item.timeline.strokes_of(tool).is_some_and(|h| shaped(h, false)),
        }
    }

    /// The values the strokes a brush slider draws again were drawn with:
    /// the latest of the brush in hand under Last, all of them under All.
    fn drawn_values(&self) -> Vec<String> {
        use crate::gui::brush_state::StrokeReach;
        use crate::gui::selection_queue::SelectionMade;
        let app = self.app();
        let (Some(tool), Some(item)) = (app.active_tool(), app.batch.selected_item()) else { return Vec::new() };
        let strokes = match app.brush_state.reach {
            StrokeReach::Next => None,
            StrokeReach::Last => item.timeline.latest_stroke(tool),
            StrokeReach::All => item.timeline.strokes_of(tool),
        };
        strokes.map_or_else(Vec::new, |history| {
            history.ops.iter().filter(|op| op.drawn_by(tool)).filter_map(|op| match op {
                SelectionMade::Stroke(drawn) => Some(format!("{:?}", drawn.values())),
                _ => None,
            }).collect()
        })
    }

    /// The value the slider named `name` shows, the last of that name.
    fn slider_value(&self, name: &str) -> Option<String> {
        tree(&self.h).iter().filter(|n| n.role == accesskit::Role::Slider && n.name.as_deref() == Some(name)).filter_map(|n| n.number.map(|v| format!("{v:.4}"))).last()
    }

    /// What the window shows and holds, as far as a person could tell:
    /// what a rule compares before and after.
    fn seen(&self) -> Seen {
        let app = self.app();
        let style = crate::gui::background_io::SelectionStyle::from_settings(&app.settings);
        let images = app
            .batch
            .items
            .iter()
            .map(|item| ImageSeen {
                id: item.id,
                name: item.filename.clone(),
                settings: format!("{:?}", item.settings),
                preset: item.applied_preset.clone(),
                selection: item.selection_hash,
                overlay: item.selection_mask.as_ref().map(|mask| style.for_plane(mask).shown_region(mask).map_or_else(|| mask.content_hash(), |shown| shown.content_hash())),
                result: shown_picture(item),
                undo_steps: item.timeline.undo_steps(),
                can_redo: item.timeline.can_redo(),
                status: format!("{:?}", item.status()),
                busy: item.is_busy(),
            })
            .collect();
        let controls = tree(&self.h)
            .iter()
            .filter(|n| n.is_control())
            .map(|n| format!("{:?}|{}|{:?}|{:?}|{:?}", n.name, n.disabled, n.toggled, n.value, n.number))
            .collect();
        Seen {
            model: format!("{:?}", app.settings.model),
            selected: app.batch.selected_id(),
            tool: format!("{:?}", app.active_tool()),
            brush: format!("{:?}", app.settings.brush),
            modals: format!("{:?}", app.open_modals()),
            comparing: app.show_original,
            toasts_asked: app.toasts.asked_for_test(),
            images,
            controls,
        }
    }
}

struct Pressable {
    at: egui::Pos2,
    rect: [f32; 4],
    role: String,
    on: Option<bool>,
}

/// The picture of the step `item` shows, unpacked or stored: a result
/// not unpacked yet is the same picture.
fn shown_picture(item: &crate::gui::item::BatchItem) -> Option<u64> {
    item.result_pixels().and_then(|pixels| pixels.load()).map(|img| pixels_hash(&img))
}

fn pixels_hash(img: &image::RgbaImage) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    img.dimensions().hash(&mut hasher);
    img.as_raw().hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ImageSeen {
    id: u64,
    name: String,
    settings: String,
    preset: String,
    selection: Option<u64>,
    /// What the selection overlay shows: under an eraser, the area the
    /// erase will touch, which a selection no eraser stroke drew takes
    /// from the Expand and Feather set now.
    overlay: Option<u64>,
    result: Option<u64>,
    undo_steps: usize,
    can_redo: bool,
    status: String,
    busy: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    model: String,
    selected: Option<u64>,
    tool: String,
    brush: String,
    modals: String,
    /// Compare is on: the canvas shows the photo as opened.
    comparing: bool,
    toasts_asked: usize,
    images: Vec<ImageSeen>,
    controls: Vec<String>,
}

/// One image of the document: what undo and redo give back.
#[derive(Debug, PartialEq, Eq)]
struct Page<'a> {
    id: u64,
    settings: &'a str,
    preset: &'a str,
    selection: Option<u64>,
    result: Option<u64>,
}

impl Seen {
    fn document(&self) -> Vec<Page<'_>> {
        self.images.iter().map(|i| Page { id: i.id, settings: &i.settings, preset: &i.preset, selection: i.selection, result: i.result }).collect()
    }
}

/// The sliders of a brush's strip: with Last or All on they draw the
/// brush's strokes again.
const BRUSH_SLIDERS: [&str; 6] = ["Size", "Hardness", "Opacity", "Feather", "Confidence", "Range"];

/// A slider went from `before` to `after` by a tenth or more of the
/// larger value: a move a stroke shows (Hardness 0.98 to 1.0 moves no
/// cell of a small stroke).
fn moved_enough(before: Option<&str>, after: Option<&str>) -> bool {
    let (Some(before), Some(after)) = (before.and_then(|v| v.parse::<f64>().ok()), after.and_then(|v| v.parse::<f64>().ok())) else { return false };
    (after - before).abs() >= 0.1 * before.abs().max(after.abs()).max(f64::EPSILON)
}

/// Why a press on `name` may change nothing here, when that is by design.
/// `role` and `on`: how the control read before the press.
fn quiet_by_design(name: &str, before: &Seen, role: &str, on: Option<bool>) -> Option<&'static str> {
    if before.images.iter().any(|i| before.selected == Some(i.id) && i.name == name) {
        return Some("the image is on the canvas already");
    }
    if role == "RadioButton" && on == Some(true) {
        return Some("it is the choice already picked");
    }
    if role == "SpinButton" {
        return Some("a click on a number box starts typing into it, which the snapshot does not see");
    }
    if name == "Reset" || name == "Reset the brush" {
        return Some("nothing differs from the preset");
    }
    None
}

/// A rule a sequence broke.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Broken {
    rule: &'static str,
    /// What marks this failure among others of its rule: shortening keeps
    /// a sequence only while it breaks the same rule the same way.
    what: String,
    detail: String,
}

/// Run `actions` on a fresh window, checking the rules; the first broken.
fn run(actions: &[Action]) -> Option<Broken> {
    let mut world = World::new();
    for action in actions {
        let before = world.seen();
        let busy_before = busy_images(&world);
        let pressed = match action {
            Action::Click(name) => world.pressable(name),
            _ => None,
        };
        let (slider_before, world_before_reached, drawn_before) = match action {
            Action::Slide(name, _) => (world.slider_value(name), world.brush_reaches_strokes(), world.drawn_values()),
            _ => (None, false, Vec::new()),
        };
        if !world.act(action) {
            continue;
        }
        if world.app().brush_state.has_active_stroke() || !world.app().magic_brush_state.active_stroke.is_empty() {
            return Some(Broken {
                rule: "a stroke ends when the button is let go",
                what: "in hand".into(),
                detail: format!("after {action:?}, with the button up, a stroke is still in hand"),
            });
        }
        if let (Action::Click(name), Some(control)) = (action, pressed) {
            let after = world.seen();
            let nothing = after == before && busy_images(&world) == busy_before && !world.busy();
            if nothing && quiet_by_design(name, &before, &control.role, control.on).is_none() {
                return Some(Broken {
                    rule: "an enabled control does something",
                    what: name.clone(),
                    detail: format!("pressing {name:?} (a {}, on: {:?}) changed nothing a person could see, started no work and said nothing", control.role, control.on),
                });
            }
        }
        if let Action::Slide(name, _) = action {
            let reaches = BRUSH_SLIDERS.contains(&name.as_str()) && world_before_reached;
            let moved = moved_enough(slider_before.as_deref(), world.slider_value(name).as_deref());
            if reaches && moved {
                world.quiet();
                let after = world.seen();
                let shown = |seen: &Seen| seen.images.iter().find(|i| seen.selected == Some(i.id)).map(|i| (i.selection, i.overlay, i.result));
                // A stroke under a wider one (a small stroke inside a large
                // one's band) is drawn again with its new value and shows
                // no change: its own record still has to take the value.
                if shown(&after) == shown(&before) && world.drawn_values() == drawn_before {
                    return Some(Broken {
                        rule: "a brush slider reaches its strokes",
                        what: name.clone(),
                        detail: format!("{name:?} moved from {slider_before:?} to {:?} with strokes of the brush in hand to draw again, and neither the selection, its overlay, the picture nor the strokes' values changed", world.slider_value(name)),
                    });
                }
            }
        }
        if matches!(action, Action::Work) && world.busy() {
            let stuck: Vec<String> = world.app().batch.items.iter().filter(|i| i.is_busy()).map(|i| format!("image {} ({:?})", i.id, i.status())).collect();
            return Some(Broken { rule: "nothing stays stuck", what: "work".into(), detail: format!("still busy after the worker answered everything: {stuck:?}") });
        }
    }
    if !world.quiet() {
        return Some(Broken { rule: "nothing stays stuck", what: "end".into(), detail: "the window never went quiet at the end".into() });
    }
    the_route_does_not_matter(&world).or_else(|| undo_then_redo_gives_it_back(&mut world))
}

fn busy_images(world: &World) -> usize {
    world.app().batch.items.iter().filter(|i| i.is_busy()).count()
}

/// The model whose cut the image shows, if it shows one.
fn cut_by(item: &crate::gui::item::BatchItem) -> Option<SettingsModel> {
    match item.timeline.current().origin.as_ref()?.cutter? {
        crate::gui::item::Cutter::Model(kind) => SettingsModel::of_kind(kind),
        crate::gui::item::Cutter::NoModel => None,
    }
}

/// A cut-out's picture comes from its photo, the model that cut it and
/// its settings: the same three on a fresh window, processed once, give
/// the same picture, whatever route led here. (A model picked since
/// leaves the cut as it is until Process.) So does an upscale's or a
/// colorize's made from the photo. Images with strokes are left out:
/// their route is part of what they are.
fn the_route_does_not_matter(world: &World) -> Option<Broken> {
    let app = world.app();
    for item in &app.batch.items {
        use crate::gui::timeline::StepKind;
        // A stroke or a selection anywhere before the shown step may live
        // on in its picture (a restore painted live, its selection since
        // cleared under the eraser): part of its route.
        let touched = item.timeline.kinds_to_shown().any(|kind| matches!(kind, StepKind::Stroke | StepKind::Strokes | StepKind::Selection | StepKind::Edit));
        if touched || item.selection_mask.is_some() || item.edits.is_some() {
            continue;
        }
        let Some(model) = (if item.shows_cutout() { cut_by(item) } else { kept_run_by(item) }) else { continue };
        let Some(here) = shown_picture(item) else { continue };
        let straight = straight_run(model, item.settings, &item.filename)?;
        if straight != here {
            return Some(Broken {
                rule: "the route does not matter",
                what: "picture".into(),
                detail: format!("{} under {model:?} shows another picture than the same settings give on a fresh window\n  settings: {:?}", item.filename, item.settings),
            });
        }
    }
    None
}

/// The upscaler or Colorize whose run the image shows, when that run
/// read the photo itself (an upscale of a cut-out or of an erase is part
/// of its route).
fn kept_run_by(item: &crate::gui::item::BatchItem) -> Option<SettingsModel> {
    use crate::gui::item::Engine;
    let origin = item.timeline.current().origin.as_ref().filter(|o| o.start.is_none())?;
    match origin.engine {
        Engine::Upscale => {
            let id = item.applied_recipe.as_ref()?.upscale.model?;
            SettingsModel::ALL.into_iter().find(|m| m.to_model_id() == Some(id))
        }
        Engine::Colorize => Some(SettingsModel::ColorizeDdcolor),
        _ => None,
    }
}

/// The picture `settings` give `name`'s photo under `model` on a fresh
/// window, processed once.
fn straight_run(model: SettingsModel, settings: ItemSettings, name: &str) -> Option<u64> {
    let mut fresh = World::new();
    let id = {
        let app = fresh.h.state_mut();
        app.settings.model = model;
        let idx = app.batch.items.iter().position(|i| i.filename == name)?;
        app.batch.select_item(idx);
        app.batch.items[idx].settings = settings;
        app.batch.items[idx].id
    };
    fresh.quiet();
    fresh.act(&Action::Click("Process".into()));
    fresh.quiet();
    shown_picture(fresh.app().batch.find_by_id(id)?)
}

/// Every step of the image on the canvas undone, then redone: that image
/// is as it was. (Undo and redo reach every ticked image, each by its own
/// history, so the others may differ.)
fn undo_then_redo_gives_it_back(world: &mut World) -> Option<Broken> {
    let shown = world.app().batch.selected_id()?;
    let before = world.seen();
    let steps = world.app().batch.selected_item().map_or(0, |i| i.timeline.undo_steps());
    for chord in ["Mod+Z", "Mod+Shift+Z"] {
        for _ in 0..steps {
            world.act(&Action::Key(chord));
            world.quiet();
        }
    }
    let after = world.seen();
    let page = |seen: &Seen| seen.document().into_iter().find(|p| p.id == shown).map(|p| format!("{p:?}"));
    (page(&after) != page(&before)).then(|| Broken {
        rule: "undo then redo gives it back",
        what: "document".into(),
        detail: format!("after {steps} undos and {steps} redos:\n  before: {:?}\n  after:  {:?}", page(&before), page(&after)),
    })
}

/// What the sequences reached, so a clean run can be told from an idle one.
#[derive(Default)]
struct Reached {
    pressed: BTreeSet<String>,
    presses: usize,
    slides: usize,
    strokes: usize,
    runs_answered: usize,
    erases: usize,
    picks: usize,
    upscales: usize,
    colorizes: usize,
    most_undo_steps: usize,
    cut_outs: usize,
}

/// A sequence of `len` actions a person could take from a fresh window,
/// each picked from what the window offers at that point.
fn random_sequence(rng: &mut Rng, len: usize, reached: &mut Reached) -> Vec<Action> {
    let mut world = World::new();
    let mut actions = Vec::with_capacity(len);
    // Most sequences open with a run a random walk seldom puts in order:
    // an erase (the eraser picked, a region painted, Process), an upscale,
    // a colorize, or lines (Sketch on, Process). An empty step is a stroke;
    // a step names the start of a control's name, one picked at random
    // when several fit (the upscalers).
    let opener: &[&str] = match rng.below(10) {
        0 | 1 => &["Model", "Eraser (MI-GAN)", "Paint Brush", "", "Process"],
        2 | 3 => &["Model", "Upscale (", "Process"],
        4 => &["Model", "Colorize (", "Process"],
        5 => &["Sketch", "Subject", "Process"],
        6 => &["Sketch", "Whole photo", "Process"],
        _ => &[],
    };
    for &step in opener {
        // An eraser may come with the Paint Brush in hand: a click would put it down.
        if step == "Paint Brush" && world.app().active_tool() == Some(crate::gui::brush_state::Tool::Paint) {
            continue;
        }
        let action = if step.is_empty() {
            Action::Stroke([rng.percent() / 2 + 20, rng.percent() / 2 + 20, rng.percent() / 2 + 20, rng.percent() / 2 + 20])
        } else {
            let fits: Vec<String> = world.offered().into_iter().map(|(name, _)| name).filter(|name| name.starts_with(step)).collect();
            Action::Click(if fits.is_empty() { step.to_string() } else { fits[rng.below(fits.len())].clone() })
        };
        world.act(&action);
        actions.push(action);
    }
    while actions.len() < len {
        let offered = world.offered();
        let sliders: Vec<&String> = offered.iter().filter(|(_, slider)| *slider).map(|(name, _)| name).collect();
        let buttons: Vec<&String> = offered.iter().filter(|(name, slider)| !slider && (!ends_things(name) || rng.below(10) == 0)).map(|(name, _)| name).collect();
        let brush_in_hand = world.app().active_tool().is_some();
        let action = match rng.below(20) {
            0 | 1 => Action::Key(KEYS[rng.below(KEYS.len())]),
            2..=5 => Action::Work,
            6..=8 if brush_in_hand => Action::Stroke([rng.percent(), rng.percent(), rng.percent(), rng.percent()]),
            9..=11 if !sliders.is_empty() => Action::Slide(sliders[rng.below(sliders.len())].clone(), rng.percent()),
            _ if !buttons.is_empty() => Action::Click(buttons[rng.below(buttons.len())].clone()),
            _ => Action::Work,
        };
        if world.act(&action) {
            match &action {
                Action::Click(name) => {
                    reached.pressed.insert(name.clone());
                    reached.presses += 1;
                }
                Action::Slide(..) => reached.slides += 1,
                Action::Stroke(_) => reached.strokes += 1,
                _ => {}
            }
        }
        actions.push(action);
    }
    world.quiet();
    reached.runs_answered += world.worker.answered.get();
    reached.erases += world.models.get().erases;
    reached.picks += world.models.get().picks;
    reached.upscales += world.models.get().upscales;
    reached.colorizes += world.models.get().colorizes;
    for item in &world.app().batch.items {
        reached.most_undo_steps = reached.most_undo_steps.max(item.timeline.undo_steps());
        reached.cut_outs += usize::from(item.has_result());
    }
    actions
}

/// The shortest part of `actions` that still breaks the same rule the
/// same way: halves dropped first, then single actions.
fn shortest(actions: Vec<Action>, broken: &Broken) -> Vec<Action> {
    let same = |candidate: &[Action]| run(candidate).is_some_and(|b| b.rule == broken.rule && b.what == broken.what);
    let mut kept = actions;
    let mut chunk = kept.len() / 2;
    while chunk >= 1 {
        let mut at = 0;
        while at < kept.len() {
            let mut shorter = kept.clone();
            shorter.drain(at..(at + chunk).min(shorter.len()));
            if same(&shorter) {
                kept = shorter;
            } else {
                at += chunk;
            }
        }
        chunk /= 2;
    }
    kept
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Explore `sequences` sequences from `seed`; every distinct failure,
/// shortened.
fn explore(seed: u64, sequences: u64, len: usize) -> Vec<(Vec<Action>, Broken)> {
    let mut found: Vec<(Vec<Action>, Broken)> = Vec::new();
    let mut reached = Reached::default();
    for n in 0..sequences {
        // The seed in the high half: `seed + n` made sequence n of seed s
        // the same as n - 1 of seed s + 1, so seeds mostly replayed each other.
        let mut rng = Rng(((seed << 32) | n).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let actions = random_sequence(&mut rng, len, &mut reached);
        let Some(broken) = run(&actions) else { continue };
        if found.iter().any(|(_, known)| known.rule == broken.rule && known.what == broken.what) {
            continue;
        }
        let short = shortest(actions, &broken);
        let broken = run(&short).unwrap_or(broken);
        eprintln!("\n[explorer] sequence {n} of seed {seed} breaks: {}\n  {}\n  steps: {short:?}", broken.rule, broken.detail);
        found.push((short, broken));
    }
    eprintln!(
        "\n[explorer] reached: {} presses on {} different controls, {} slides, {} strokes, {} cut-out runs, {} erases, {} Magic Brush picks, {} upscales and {} colorizes answered, {} cut-outs at the ends, up to {} undo steps\n  controls: {:?}",
        reached.presses,
        reached.pressed.len(),
        reached.slides,
        reached.strokes,
        reached.runs_answered,
        reached.erases,
        reached.picks,
        reached.upscales,
        reached.colorizes,
        reached.cut_outs,
        reached.most_undo_steps,
        reached.pressed,
    );
    found
}

/// A few fixed sequences run with the suite: the explorer itself works,
/// and the app holds its rules on them.
#[test]
fn a_few_sequences_hold_the_rules() {
    let found = explore(1, 4, 12);
    assert!(found.is_empty(), "{} rule(s) broken; see the steps above", found.len());
}

/// The real run: `PRUNR_EXPLORE_SEQS` sequences (200) of
/// `PRUNR_EXPLORE_LEN` actions (30) from `PRUNR_EXPLORE_SEED` (1).
/// `cargo test -p prunr-app --lib explorer -- --ignored --nocapture`
#[test]
#[ignore = "long: run by hand or nightly"]
fn explore_many_sequences() {
    let (seed, sequences, len) = (env_number("PRUNR_EXPLORE_SEED", 1), env_number("PRUNR_EXPLORE_SEQS", 200), env_number("PRUNR_EXPLORE_LEN", 30) as usize);
    let found = explore(seed, sequences, len);
    eprintln!("\n[explorer] {sequences} sequences of {len} from seed {seed}: {} distinct failure(s)", found.len());
    assert!(found.is_empty(), "{} rule(s) broken; see the steps above", found.len());
}
