//! The sequence explorer: random sequences of what a person does (click a
//! control, press a key, wait for work) on the real app through the test
//! harness, a stand-in answering for the cut-out worker, and rules checked
//! after every step. Ordinary tests start from a tidy window and do one
//! thing; the bugs this looks for need a history (pick, process, undo,
//! then a slider). A sequence that breaks a rule is cut down to the
//! shortest that still does and printed as steps to paste into a test.
//!
//! Not picked yet, each for want of a stand-in for its engine: erasers,
//! upscalers, Colorize, the brushes, Sketch, Faces and the depth looks.

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;

use egui_kittest::Harness;

use super::harness::{harness, settle, tree};
use crate::gui::app::PrunrApp;
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
}

/// One thing a person does. Replayed on a window where it does not apply
/// (no such control, or it is greyed), it is skipped, so a shortened
/// sequence stays a valid one.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    Click(String),
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
    ("Eraser (", "no stand-in for the eraser engines yet"),
    ("Upscale (", "no stand-in for the upscale engines yet"),
    ("Colorize (", "no stand-in for Colorize yet"),
    ("Magic Brush", "no stand-in for the Magic Brush's model yet"),
    ("Paint Brush", "strokes are not drawn yet"),
    ("Sketch", "no stand-in for the lines model yet"),
    ("Faces", "no stand-in for the face models yet"),
    ("Depth", "no stand-in for the depth model yet"),
    ("Lens", "a depth look"),
    ("Golden Bokeh", "a depth look"),
    ("Teal and Orange", "a depth look on No model"),
    ("Blueprint", "a lines look"),
    ("Pencil Sketch", "a lines look"),
    ("Sunset Lines", "a lines look"),
    ("X-Ray", "a lines look"),
    ("Pick image", "asks the system for a file"),
];

fn left_alone(name: &str) -> bool {
    LEFT_ALONE.iter().any(|(start, _)| name.starts_with(start))
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
    /// A soft ellipse, the same for every run of an image this size.
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

    /// Answer every run asked for so far; how many images that was.
    fn answer(&self, app: &PrunrApp) -> usize {
        let mut answered = 0;
        while let Ok(msg) = self.asked.try_recv() {
            let WorkerMessage::BatchProcess { batch, items, tier2_items, add_edge_items, config, .. } = msg else { continue };
            let model = config.engines.model;
            let photo_of = |id: u64| app.batch.find_by_id(id).and_then(photo);
            for item in &items {
                let raw = Self::tensor(model);
                let result = photo_of(item.id).ok_or_else(|| "the photo could not be read".to_string()).and_then(|p| Self::cut(&p, &raw, &config.run.mask));
                let _ = self.answers.send(WorkerResult::BatchItemDone { batch, item_id: item.id, result, seg: CompressedTensor::from_raw(raw), edges: None });
                answered += 1;
            }
            for work in tier2_items.iter().chain(&add_edge_items) {
                let result = work
                    .tensor
                    .bundle()
                    .ok_or_else(|| "the kept tensor could not be read".to_string())
                    .and_then(|raw| {
                        let photo = work.photo.as_deref().cloned().or_else(|| photo_of(work.item_id)).ok_or_else(|| "the photo could not be read".to_string())?;
                        Self::cut(&photo, &raw, &work.mask)
                    });
                let _ = self.answers.send(WorkerResult::BatchItemDone { batch, item_id: work.item_id, result, seg: None, edges: None });
                answered += 1;
            }
            let _ = self.answers.send(WorkerResult::BatchComplete { batch });
        }
        self.answered.set(self.answered.get() + answered);
        answered
    }
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
    let (w, h) = (48u32, 32u32);
    let img = image::RgbaImage::from_fn(w, h, |x, y| {
        let (dx, dy) = (x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0);
        if dx * dx + dy * dy < 90.0 {
            image::Rgba([230, 60u8.wrapping_add(seed.wrapping_mul(40)), 50, 255])
        } else {
            image::Rgba([(x * 5) as u8, (y * 7) as u8, 90u8.wrapping_add(seed.wrapping_mul(30)), 255])
        }
    });
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).expect("a PNG in memory");
    bytes
}

struct World {
    h: Harness<'static, PrunrApp>,
    worker: StandIn,
}

impl World {
    /// The window with two photos opened, the first on the canvas.
    fn new() -> Self {
        let mut h = harness();
        let (to_worker, asked) = mpsc::channel();
        let (answers, from_worker) = mpsc::channel();
        {
            let app = h.state_mut();
            app.processor.worker_tx = to_worker;
            app.processor.worker_rx = from_worker;
            for id in [1u8, 2] {
                app.add_to_batch(photo_png(id), format!("photo{id}.png"));
            }
            app.batch.select_item(0);
        }
        let mut world = Self { h, worker: StandIn { asked, answers, answered: std::cell::Cell::new(0) } };
        world.quiet();
        world
    }

    fn app(&self) -> &PrunrApp {
        self.h.state()
    }

    /// Frames, the worker answering between them, until nothing is owed
    /// or the rounds run out. Whether it went quiet.
    fn quiet(&mut self) -> bool {
        for _ in 0..200 {
            self.worker.answer(self.h.state());
            self.h.step();
            if !crate::gui::automation::state::busy(self.h.state()) && !self.h.state().batch.selection_queue.any_pending() {
                settle(&mut self.h);
                self.worker.answer(self.h.state());
                self.h.step();
                if !crate::gui::automation::state::busy(self.h.state()) {
                    return true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        false
    }

    /// The controls a person could click now: enabled, on top (nothing
    /// covers them), and not left alone.
    fn clickable(&self) -> Vec<String> {
        let names: BTreeSet<String> = tree(&self.h)
            .iter()
            .filter(|n| n.is_control() && !n.disabled && crate::gui::automation::reachable_at(&self.h.ctx, n.id).is_some())
            .filter_map(|n| n.name.clone())
            .filter(|name| !left_alone(name))
            .collect();
        names.into_iter().collect()
    }

    /// The control named `name` a person could press, the last of them:
    /// where to press it, and how it reads (its role and whether it is on).
    fn pressable(&self, name: &str) -> Option<(egui::Pos2, String, Option<bool>)> {
        tree(&self.h)
            .iter()
            .filter(|n| n.is_control() && !n.disabled && n.name.as_deref() == Some(name))
            .filter_map(|n| crate::gui::automation::reachable_at(&self.h.ctx, n.id).map(|at| (at, format!("{:?}", n.role), n.toggled)))
            .last()
    }

    /// A pointer press and release at `at`, a frame each, as a mouse makes.
    fn press(&mut self, at: egui::Pos2) {
        let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::default() };
        for event in [egui::Event::PointerMoved(at), button(true), button(false)] {
            self.h.input_mut().events.push(event);
            self.h.step();
        }
        settle(&mut self.h);
    }

    /// Do `action`; false when it did not apply here.
    fn act(&mut self, action: &Action) -> bool {
        match action {
            Action::Click(name) => {
                let Some((at, ..)) = self.pressable(name) else { return false };
                self.press(at);
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

    /// What the window shows and holds, as far as a person could tell:
    /// what a rule compares before and after.
    fn seen(&self) -> Seen {
        let app = self.app();
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
                result: item.result_rgba.as_deref().map(pixels_hash),
                undo_steps: item.timeline.undo_steps(),
                can_redo: item.timeline.can_redo(),
                status: format!("{:?}", item.status()),
                busy: item.is_busy(),
            })
            .collect();
        let controls = tree(&self.h)
            .iter()
            .filter(|n| n.is_control())
            .map(|n| format!("{:?}|{}|{:?}|{:?}", n.name, n.disabled, n.toggled, n.value))
            .collect();
        Seen {
            model: format!("{:?}", app.settings.model),
            selected: app.batch.selected_id(),
            tool: format!("{:?}", app.active_tool()),
            modals: format!("{:?}", app.open_modals()),
            comparing: app.show_original,
            toasts_asked: app.toasts.asked_for_test(),
            images,
            controls,
        }
    }
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

/// Why a click on `name` may change nothing here, when that is by design.
/// `role` and `on`: how the control read before the click.
fn quiet_by_design(name: &str, before: &Seen, role: &str, on: Option<bool>) -> Option<&'static str> {
    if before.images.iter().any(|i| before.selected == Some(i.id) && i.name == name) {
        return Some("the image is on the canvas already");
    }
    if role == "RadioButton" && on == Some(true) {
        return Some("it is the choice already picked");
    }
    if name == "Reset" {
        return Some("nothing differs from the default preset");
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
        let asked_before = pending_work(&world);
        let pressed = match action {
            Action::Click(name) => world.pressable(name),
            _ => None,
        };
        if !world.act(action) {
            continue;
        }
        if let (Action::Click(name), Some((_, role, on))) = (action, pressed) {
            let after = world.seen();
            let nothing = after == before && pending_work(&world) == asked_before && !crate::gui::automation::state::busy(world.app());
            if nothing && quiet_by_design(name, &before, &role, on).is_none() {
                return Some(Broken {
                    rule: "an enabled control does something",
                    what: name.clone(),
                    detail: format!("clicking {name:?} (a {role}, on: {on:?}) changed nothing a person could see, started no work and said nothing"),
                });
            }
        }
        if matches!(action, Action::Work) && crate::gui::automation::state::busy(world.app()) {
            let stuck: Vec<String> = world.app().batch.items.iter().filter(|i| i.is_busy()).map(|i| format!("image {} ({:?})", i.id, i.status())).collect();
            return Some(Broken { rule: "nothing stays stuck", what: "work".into(), detail: format!("still busy after the worker answered everything: {stuck:?}") });
        }
    }
    if !world.quiet() {
        return Some(Broken { rule: "nothing stays stuck", what: "end".into(), detail: "the window never went quiet at the end".into() });
    }
    undo_then_redo_gives_it_back(&mut world)
}

/// Runs the worker has been asked for and not answered.
fn pending_work(world: &World) -> usize {
    world.app().batch.items.iter().filter(|i| i.is_busy()).count()
}

/// Every step of the image on the canvas undone, then redone: the
/// document is as it was.
fn undo_then_redo_gives_it_back(world: &mut World) -> Option<Broken> {
    let before = world.seen();
    let steps = world.app().batch.selected_item().map_or(0, |i| i.timeline.undo_steps());
    for chord in ["Mod+Z", "Mod+Shift+Z"] {
        for _ in 0..steps {
            world.act(&Action::Key(chord));
            world.quiet();
        }
    }
    let after = world.seen();
    (after.document() != before.document()).then(|| Broken {
        rule: "undo then redo gives it back",
        what: "document".into(),
        detail: format!("after {steps} undos and {steps} redos:\n  before: {:?}\n  after:  {:?}", before.document(), after.document()),
    })
}

/// What the sequences reached, so a clean run can be told from an idle one.
#[derive(Default)]
struct Reached {
    pressed: BTreeSet<String>,
    presses: usize,
    runs_answered: usize,
    most_undo_steps: usize,
    cut_outs: usize,
}

/// A sequence of `len` actions a person could take from a fresh window,
/// each picked from what the window offers at that point.
fn random_sequence(rng: &mut Rng, len: usize, reached: &mut Reached) -> Vec<Action> {
    let mut world = World::new();
    let mut actions = Vec::with_capacity(len);
    for _ in 0..len {
        let action = match rng.below(10) {
            0 => Action::Key(KEYS[rng.below(KEYS.len())]),
            1 | 2 => Action::Work,
            _ => {
                let offered = world.clickable();
                if offered.is_empty() {
                    Action::Work
                } else {
                    Action::Click(offered[rng.below(offered.len())].clone())
                }
            }
        };
        if world.act(&action) {
            if let Action::Click(name) = &action {
                reached.pressed.insert(name.clone());
                reached.presses += 1;
            }
        }
        actions.push(action);
    }
    world.quiet();
    reached.runs_answered += world.worker.answered.get();
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
        let mut rng = Rng(seed.wrapping_add(n).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
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
        "\n[explorer] reached: {} presses on {} different controls, {} runs answered, {} cut-outs at the ends, up to {} undo steps\n  controls: {:?}",
        reached.presses,
        reached.pressed.len(),
        reached.runs_answered,
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
/// `PRUNR_EXPLORE_LEN` actions (20) from `PRUNR_EXPLORE_SEED` (1).
/// `cargo test -p prunr-app --lib explorer -- --ignored --nocapture`
#[test]
#[ignore = "long: run by hand or nightly"]
fn explore_many_sequences() {
    let (seed, sequences, len) = (env_number("PRUNR_EXPLORE_SEED", 1), env_number("PRUNR_EXPLORE_SEQS", 200), env_number("PRUNR_EXPLORE_LEN", 20) as usize);
    let found = explore(seed, sequences, len);
    eprintln!("\n[explorer] {sequences} sequences of {len} from seed {seed}: {} distinct failure(s)", found.len());
    assert!(found.is_empty(), "{} rule(s) broken; see the steps above", found.len());
}
