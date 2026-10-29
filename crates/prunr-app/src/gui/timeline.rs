//! One undo timeline per image: the states it has been in, and which one
//! is shown. Undo and redo move the cursor; any other edit cuts the redo tail and
//! appends. Each state is complete (result, recipe, selection, settings),
//! so stepping back shows a stored image instead of re-running a model.
//! The version, bumped on every move, lets a background job land only on
//! the state it started from.
//!
//! Payloads are shared: a stroke step points at the same stored result as
//! the step before it. A stored payload stays as-is while anything else
//! holds it (the canvas, a job), is compressed off-thread once nothing
//! does, and moves to disk under memory pressure.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use prunr_core::selection::MaskArtifact;

use super::item::Engine;
use super::item_settings::ItemSettings;

/// Steps kept per image; the oldest go first.
pub(crate) const TIMELINE_DEPTH: usize = 64;

/// Knob changes closer together than this are one step: a slider drag,
/// or a few clicks on a stepper.
const ADJUST_MERGE_WINDOW: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepKind {
    Original,
    Process,
    Stroke,
    Selection,
    Edit,
    Erase,
    Upscale,
    Colorize,
    Preset,
    Adjust,
    /// Every stroke since the picture changed, drawn again.
    Strokes,
}

impl StepKind {
    /// What the step did, for "Stroke undone" / "Erase redone".
    pub(crate) fn noun(self) -> &'static str {
        match self {
            StepKind::Original => "Original",
            StepKind::Process => "Processing",
            StepKind::Stroke => "Stroke",
            StepKind::Selection => "Selection change",
            StepKind::Edit => "Edit",
            StepKind::Erase => "Erase",
            StepKind::Upscale => "Upscale",
            StepKind::Colorize => "Colorize",
            StepKind::Preset => "Preset",
            StepKind::Adjust => "Adjustment",
            StepKind::Strokes => "Stroke adjustment",
        }
    }
}

/// A value a timeline stores and can compress.
pub(crate) trait Payload: Send + Sync + Sized + 'static {
    fn encode(&self) -> std::io::Result<Vec<u8>>;
    fn decode(bytes: &[u8]) -> std::io::Result<Self>;
    /// Width and height, kept beside the stored form so a reader needs
    /// no unpack to know them.
    fn dims(&self) -> (u32, u32);
}

impl Payload for image::RgbaImage {
    fn dims(&self) -> (u32, u32) {
        self.dimensions()
    }

    fn encode(&self) -> std::io::Result<Vec<u8>> {
        super::history_disk::compress_to_ram(self)
    }

    fn decode(bytes: &[u8]) -> std::io::Result<Self> {
        super::history_disk::decompress_from_bytes(bytes)
    }
}

impl Payload for MaskArtifact {
    fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn encode(&self) -> std::io::Result<Vec<u8>> {
        let mut raw = Vec::with_capacity(8 + self.cells().len());
        raw.extend_from_slice(&self.width.to_le_bytes());
        raw.extend_from_slice(&self.height.to_le_bytes());
        raw.extend(self.cells().iter().map(|&c| c as u8));
        zstd::encode_all(raw.as_slice(), super::history_disk::ZSTD_LEVEL)
    }

    fn decode(bytes: &[u8]) -> std::io::Result<Self> {
        let raw = zstd::decode_all(bytes)?;
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "selection plane does not match its size");
        let (head, cells) = raw.split_first_chunk::<8>().ok_or_else(bad)?;
        let (w, h) = super::history_disk::read_dims(head);
        if cells.len() != (w as usize) * (h as usize) {
            return Err(bad());
        }
        Ok(MaskArtifact::from_cells(w, h, cells.iter().map(|&c| c as i8).collect()))
    }
}

enum Tier<T> {
    Plain(Arc<T>),
    /// Being compressed off-thread; still usable meanwhile.
    Compressing(Arc<T>),
    Compressed(Arc<Vec<u8>>),
    OnDisk(PathBuf),
    /// Read back for use; every reader gets this one value, so an
    /// unchanged payload is recognised and shared, and once nothing uses
    /// it the stored form is kept again without re-encoding.
    Loaded(Arc<T>, Encoded),
}

/// One stored payload, shared by every state that has it.
pub(crate) struct Stored<T: Payload> {
    tier: Mutex<Tier<T>>,
    dims: (u32, u32),
}

#[derive(Clone, PartialEq)]
enum Encoded {
    Bytes(Arc<Vec<u8>>),
    File(PathBuf),
}

impl Encoded {
    /// The stored bytes: borrowed when they are in memory, never copied.
    fn read(&self) -> std::io::Result<std::borrow::Cow<'_, [u8]>> {
        match self {
            Encoded::Bytes(b) => Ok(std::borrow::Cow::Borrowed(b.as_slice())),
            Encoded::File(p) => std::fs::read(p).map(std::borrow::Cow::Owned),
        }
    }

    fn into_tier<T>(self) -> Tier<T> {
        match self {
            Encoded::Bytes(b) => Tier::Compressed(b),
            Encoded::File(p) => Tier::OnDisk(p),
        }
    }
}

impl<T: Payload> Stored<T> {
    pub(crate) fn new(value: Arc<T>) -> Arc<Self> {
        let dims = value.dims();
        Arc::new(Self { tier: Mutex::new(Tier::Plain(value)), dims })
    }

    fn lock(&self) -> MutexGuard<'_, Tier<T>> {
        self.tier.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The payload, decoded when stored compressed. `None` only when a
    /// decode fails (a disk file removed under us).
    pub(crate) fn load(&self) -> Option<Arc<T>> {
        let encoded = match &*self.lock() {
            Tier::Plain(v) | Tier::Compressing(v) | Tier::Loaded(v, _) => return Some(Arc::clone(v)),
            Tier::Compressed(b) => Encoded::Bytes(Arc::clone(b)),
            Tier::OnDisk(p) => Encoded::File(p.clone()),
        };
        let value = encoded.read().and_then(|b| T::decode(&b))
            .map(Arc::new)
            .map_err(|err| tracing::warn!(%err, "a stored undo state could not be read"))
            .ok()?;
        let mut tier = self.lock();
        let unchanged = match &*tier {
            Tier::Compressed(b) => encoded == Encoded::Bytes(Arc::clone(b)),
            Tier::OnDisk(p) => encoded == Encoded::File(p.clone()),
            _ => false,
        };
        if unchanged {
            *tier = Tier::Loaded(Arc::clone(&value), encoded);
        }
        Some(value)
    }

    /// The payload's width and height, packed or not.
    pub(crate) fn dims(&self) -> (u32, u32) {
        self.dims
    }

    /// The payload when it is already decoded; `None` while it is stored
    /// compressed or on disk. Never decodes, so the UI thread may ask.
    pub(crate) fn loaded(&self) -> Option<Arc<T>> {
        match &*self.lock() {
            Tier::Plain(v) | Tier::Compressing(v) | Tier::Loaded(v, _) => Some(Arc::clone(v)),
            Tier::Compressed(_) | Tier::OnDisk(_) => None,
        }
    }

    /// True when `value` is the payload this cell hands out.
    pub(crate) fn holds(&self, value: &Arc<T>) -> bool {
        matches!(&*self.lock(), Tier::Plain(v) | Tier::Compressing(v) | Tier::Loaded(v, _) if Arc::ptr_eq(v, value))
    }

    /// The payload to compress, when nothing besides this cell uses it. A
    /// read-back payload nobody uses returns to its stored form at once.
    fn begin_compress(&self) -> Option<Arc<T>> {
        let mut tier = self.lock();
        match &*tier {
            Tier::Loaded(v, encoded) if Arc::strong_count(v) == 1 => {
                let stored = encoded.clone().into_tier();
                // The read-back value frees off the UI thread the sweep runs on.
                super::reclaim::later(std::mem::replace(&mut *tier, stored));
                None
            }
            Tier::Plain(v) if Arc::strong_count(v) == 1 => {
                let v = Arc::clone(v);
                *tier = Tier::Compressing(Arc::clone(&v));
                Some(v)
            }
            _ => None,
        }
    }

    fn finish_compress(&self, encoded: std::io::Result<Vec<u8>>) {
        let mut tier = self.lock();
        let Tier::Compressing(v) = &*tier else { return };
        *tier = match encoded {
            Ok(bytes) => Tier::Compressed(Arc::new(bytes)),
            // Kept as-is; the next sweep tries again.
            Err(_) => Tier::Plain(Arc::clone(v)),
        };
    }

    /// The compression to run on a worker, when nothing else uses the
    /// payload.
    pub(crate) fn compress_job(self: &Arc<Self>) -> Option<CompressJob> {
        let value = self.begin_compress()?;
        let cell = Arc::clone(self);
        Some(Box::new(move || cell.finish_compress(value.encode())))
    }

    /// Move a compressed payload to disk; true when it moved.
    pub(crate) fn demote_to_disk(&self) -> bool {
        let bytes = match &*self.lock() {
            Tier::Compressed(b) => Arc::clone(b),
            _ => return false,
        };
        let Ok(path) = super::history_disk::write_blob(&bytes) else { return false };
        let mut tier = self.lock();
        match &*tier {
            Tier::Compressed(b) if Arc::ptr_eq(b, &bytes) => {
                *tier = Tier::OnDisk(path);
                true
            }
            _ => {
                let _ = std::fs::remove_file(&path);
                false
            }
        }
    }

    /// It can be read back: all but one kept on disk whose file was
    /// removed under the app (a cache cleaner). No decode.
    pub(crate) fn is_readable(&self) -> bool {
        match &*self.lock() {
            Tier::OnDisk(p) => p.is_file(),
            _ => true,
        }
    }

    #[cfg(test)]
    pub(crate) fn disk_path(&self) -> Option<PathBuf> {
        match &*self.lock() {
            Tier::OnDisk(p) => Some(p.clone()),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn is_on_disk(&self) -> bool {
        self.disk_path().is_some()
    }

    #[cfg(test)]
    pub(crate) fn is_compressed(&self) -> bool {
        matches!(&*self.lock(), Tier::Compressed(_) | Tier::OnDisk(_))
    }
}

impl<T: Payload> Drop for Stored<T> {
    fn drop(&mut self) {
        if let Tier::OnDisk(path) | Tier::Loaded(_, Encoded::File(path)) = &*self.lock() {
            super::disk_queue::remove(path.clone());
        }
    }
}

/// One payload's compression, run off the UI thread.
pub(crate) type CompressJob = Box<dyn FnOnce() + Send>;

pub(crate) type StoredImage = Arc<Stored<image::RgbaImage>>;
pub(crate) type StoredSelection = Arc<Stored<MaskArtifact>>;

/// Everything undo restores about an image.
#[derive(Clone)]
pub(crate) struct DocState {
    pub(crate) kind: StepKind,
    /// `None`: the unprocessed image.
    pub(crate) result: Option<StoredImage>,
    pub(crate) recipe: Option<prunr_core::ProcessingRecipe>,
    pub(crate) selection: Option<StoredSelection>,
    /// The selection's content hash, so restoring it costs no rehash.
    pub(crate) selection_hash: Option<u64>,
    /// On a cut-out, the restores and erases already made (signed: plus
    /// restores, minus erases); the selection is applied on top.
    pub(crate) edits: Option<StoredSelection>,
    pub(crate) settings: ItemSettings,
    pub(crate) applied_preset: String,
    /// The run that made this step's image; `None` for the original and
    /// for edits made by hand.
    pub(crate) origin: Option<Origin>,
    /// The image has restored faces in it, blended by its own run or
    /// carried over from the image it was made from.
    pub(crate) faced: bool,
    pub(crate) at: Instant,
    /// Which step this is (`next_step_id`): two can share an `at`.
    pub(crate) id: u64,
    /// What the step did to the selection, for a replay of the strokes.
    pub(crate) made: Option<super::selection_queue::Made>,
}

pub(crate) fn next_step_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// What made a step's image: the engine, and where its run started
/// (`None`: the original). Running the same engine again redoes the step
/// from there instead of stacking on its own output.
#[derive(Clone)]
pub(crate) struct Origin {
    pub(crate) engine: Engine,
    pub(crate) start: Option<RunStart>,
    /// The kept run (an upscale for Refine, a colorize for Strength) that
    /// made this step, so another step never re-renders it. Run ids are
    /// unique across both kinds.
    pub(crate) kept_run: Option<u64>,
    /// The seed a generated fill ran with, for Pin seed.
    pub(crate) seed: Option<u64>,
    /// The region an erase filled, so "Try another" can fill it again
    /// (shared with the stroke step that painted it).
    pub(crate) region: Option<StoredSelection>,
    /// How large an erase filled its region: 1 is full size.
    pub(crate) fill_scale: Option<f32>,
    /// Who cut this step out (`None`: another engine, or not known), so a
    /// Process by a different one builds on it.
    pub(crate) cutter: Option<super::item::Cutter>,
    /// The image a live render on this step started from, when the step
    /// is another run's (an erase, an upscale): the run stays the step's
    /// own, and the next render redoes from here, not on its own output.
    pub(crate) rendered_from: Option<RunStart>,
}

impl Origin {
    /// A run of `engine` from `start` (`None`: the original), with no seed,
    /// kept run or region.
    pub(crate) fn new(engine: Engine, start: Option<RunStart>) -> Self {
        Self { engine, start, kept_run: None, seed: None, region: None, fill_scale: None, cutter: None, rendered_from: None }
    }
}

/// An image a run starts from: a stored timeline image, loaded only when
/// the run reads it; its size is known without loading it.
pub(crate) type RunStart = StoredImage;

impl DocState {
    /// Every stored part of the step can be read back.
    pub(crate) fn readable(&self) -> bool {
        let starts = self.origin.iter().flat_map(|o| o.start.iter().chain(&o.rendered_from));
        self.result.iter().chain(starts).all(|s| s.is_readable())
            && self.selection.iter().chain(self.edits.as_ref()).all(|s| s.is_readable())
    }

    /// The image as imported: no result, no selection.
    pub(crate) fn original(settings: ItemSettings, applied_preset: String) -> Self {
        Self {
            kind: StepKind::Original,
            result: None,
            recipe: None,
            selection: None,
            selection_hash: None,
            edits: None,
            settings,
            applied_preset,
            origin: None,
            faced: false,
            at: Instant::now(),
            id: next_step_id(),
            made: None,
        }
    }
}

/// Strokes to draw again: the step they start from, with its selection
/// and edits, and what each step after it did.
#[derive(Clone)]
pub(crate) struct StrokeHistory {
    pub(crate) from: u64,
    pub(crate) selection: Option<StoredSelection>,
    pub(crate) edits: Option<StoredSelection>,
    pub(crate) ops: Arc<Vec<super::selection_queue::SelectionMade>>,
}

impl StrokeHistory {
    fn from_step(start: &DocState, ops: Vec<super::selection_queue::SelectionMade>) -> Self {
        Self { from: start.id, selection: start.selection.clone(), edits: start.edits.clone(), ops: Arc::new(ops) }
    }

    /// The same strokes, each as `change` makes it.
    pub(crate) fn changed(&self, change: impl Fn(&super::brush_state::StrokeDrawn) -> super::brush_state::StrokeDrawn) -> Self {
        Self { ops: Arc::new(self.ops.iter().map(|op| op.changed(&change)).collect()), ..self.clone() }
    }
}

pub(crate) struct Timeline {
    states: VecDeque<DocState>,
    cursor: usize,
    version: u64,
    revision: u64,
    /// The photo as opened fell off the end.
    opened_dropped: bool,
}

impl Timeline {
    pub(crate) fn new(original: DocState) -> Self {
        Self { states: VecDeque::from([original]), cursor: 0, version: 0, revision: 0, opened_dropped: false }
    }

    pub(crate) fn current(&self) -> &DocState {
        &self.states[self.cursor]
    }

    /// The latest run at or before the shown step: hand edits after a
    /// run keep building on what that run made.
    pub(crate) fn latest_run(&self) -> Option<&Origin> {
        self.states.range(..=self.cursor).rev().find_map(|s| s.origin.as_ref())
    }

    /// Whether a step holding `image` has restored faces in it.
    pub(crate) fn image_faced(&self, image: &StoredImage) -> bool {
        self.states.iter().any(|s| s.faced && s.result.as_ref().is_some_and(|r| Arc::ptr_eq(r, image)))
    }

    /// Whether step `id` is the shown one or before it: not undone, nor
    /// dropped by a step made after an undo.
    pub(crate) fn in_effect(&self, id: u64) -> bool {
        self.states.range(..=self.cursor).any(|s| s.id == id)
    }

    /// The state before the shown one.
    pub(crate) fn previous(&self) -> Option<&DocState> {
        self.cursor.checked_sub(1).map(|i| &self.states[i])
    }

    /// The state a redo goes to.
    pub(crate) fn next(&self) -> Option<&DocState> {
        self.states.get(self.cursor + 1)
    }

    /// The strokes since the picture last changed, up to the shown step,
    /// to draw again. Steps that change only the settings pass through;
    /// any other step without a record, or one that changed the picture,
    /// is where they start. A redraw of earlier strokes stands in for the
    /// steps it redrew. `None` without a stroke to draw.
    pub(crate) fn strokes_since_picture(&self) -> Option<StrokeHistory> {
        use super::selection_queue::{Made, SelectionMade};
        let mut ops = Vec::new();
        for i in (0..=self.cursor).rev() {
            let step = &self.states[i];
            let start = match (&step.made, step.kind) {
                (Some(Made::Redrawn { from, ops: redrawn }), _) => {
                    ops.extend(redrawn.iter().rev().cloned());
                    self.states.range(..i).find(|s| s.id == *from)?
                }
                (Some(Made::One(made)), _) => {
                    ops.push(made.clone());
                    continue;
                }
                (None, StepKind::Adjust | StepKind::Preset) if i > 0 => continue,
                (None, _) => step,
            };
            ops.reverse();
            return ops.iter().any(|op| matches!(op, SelectionMade::Stroke(_))).then(|| StrokeHistory::from_step(start, ops));
        }
        None
    }

    /// The shown step's stroke, while it is the newest step, to draw
    /// again from the step before it.
    pub(crate) fn latest_stroke(&self) -> Option<StrokeHistory> {
        use super::selection_queue::{Made, SelectionMade};
        let shown = self.current();
        let stroke = match &shown.made {
            Some(Made::One(stroke @ SelectionMade::Stroke(_))) if !self.can_redo() => stroke.clone(),
            _ => return None,
        };
        Some(StrokeHistory::from_step(self.previous()?, vec![stroke]))
    }

    /// Whether step `id` is the shown one and nothing is ahead of it.
    pub(crate) fn is_newest_shown(&self, id: u64) -> bool {
        self.current().id == id && !self.can_redo()
    }

    /// Fill in the shown state as a job it started lands: no new step.
    pub(crate) fn current_mut(&mut self) -> &mut DocState {
        self.revision += 1;
        &mut self.states[self.cursor]
    }

    /// Changes on every step, undo and redo; a job compares it with the
    /// value it started at before it lands.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// Changes whenever the shown state may have: every move, and every
    /// change in place (a merged knob change, a fill-in), which `version`
    /// lets pass so jobs stay valid. What a save is checked against.
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// A new step after the shown one; any redo tail goes. A knob change
    /// right after another knob change joins it instead.
    pub(crate) fn push(&mut self, state: DocState) {
        self.revision += 1;
        let at_end = self.cursor + 1 == self.states.len();
        let top = &self.states[self.cursor];
        if state.kind == StepKind::Adjust
            && top.kind == StepKind::Adjust
            && at_end
            && state.at.saturating_duration_since(top.at) < ADJUST_MERGE_WINDOW
        {
            self.states[self.cursor] = state;
            return;
        }
        // Dropped states (the redo tail, the oldest past the depth) free
        // off the UI thread.
        let redo_tail: Vec<DocState> = self.states.drain(self.cursor + 1..).collect();
        super::reclaim::later(redo_tail);
        self.states.push_back(state);
        while self.states.len() > TIMELINE_DEPTH {
            super::reclaim::later(self.states.pop_front());
            self.opened_dropped = true;
        }
        self.cursor = self.states.len() - 1;
        self.version += 1;
    }

    /// Step back; returns what the undone step did.
    pub(crate) fn undo(&mut self) -> Option<StepKind> {
        let undone = self.can_undo().then(|| self.states[self.cursor].kind)?;
        self.cursor -= 1;
        self.version += 1;
        self.revision += 1;
        Some(undone)
    }

    /// Step forward; returns what the redone step did.
    pub(crate) fn redo(&mut self) -> Option<StepKind> {
        if !self.can_redo() {
            return None;
        }
        self.cursor += 1;
        self.version += 1;
        self.revision += 1;
        Some(self.states[self.cursor].kind)
    }

    pub(crate) fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub(crate) fn can_redo(&self) -> bool {
        self.cursor + 1 < self.states.len()
    }

    /// Steps an undo can walk back.
    pub(crate) fn undo_steps(&self) -> usize {
        self.cursor
    }

    /// Whether the photo as it was opened is the state shown.
    pub(crate) fn shows_opened(&self) -> bool {
        self.cursor == 0 && !self.opened_dropped
    }

    /// Every stored image: each step's result, and each step's start
    /// that is no longer any step's result (its step fell off the end).
    pub(crate) fn stored_images(&self) -> impl Iterator<Item = &StoredImage> {
        let is_a_result = |b: &StoredImage| self.states.iter().any(|s| s.result.as_ref().is_some_and(|r| Arc::ptr_eq(r, b)));
        self.states.iter().filter_map(|s| s.result.as_ref())
            .chain(self.states.iter().filter_map(|s| s.origin.as_ref()).flat_map(|o| o.start.iter().chain(&o.rendered_from)).filter(move |b| !is_a_result(b)))
    }

    /// Every stored selection and cut-out edits plane.
    pub(crate) fn stored_selections(&self) -> impl Iterator<Item = &StoredSelection> {
        self.states.iter().flat_map(|s| s.selection.iter().chain(s.edits.iter()))
    }
}

/// `value` stored, reusing `previous` when it already holds the same
/// payload, so an unchanged result or selection costs nothing per step.
pub(crate) fn store<T: Payload>(value: Option<&Arc<T>>, previous: Option<&Arc<Stored<T>>>) -> Option<Arc<Stored<T>>> {
    let value = value?;
    Some(match previous {
        Some(p) if p.holds(value) => Arc::clone(p),
        _ => Stored::new(Arc::clone(value)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(kind: StepKind, gamma: f32) -> DocState {
        DocState {
            kind,
            result: None,
            recipe: None,
            selection: None,
            selection_hash: None,
            edits: None,
            settings: super::super::item_settings::item_with_gamma(gamma),
            applied_preset: String::new(),
            origin: None,
            faced: false,
            at: Instant::now(),
            id: next_step_id(),
            made: None,
        }
    }

    fn gamma(t: &Timeline) -> f32 {
        t.current().settings.gamma
    }

    #[test]
    fn undo_and_redo_move_the_cursor_and_a_new_step_cuts_the_redo_tail() {
        let mut t = Timeline::new(state(StepKind::Original, 1.0));
        t.push(state(StepKind::Process, 2.0));
        t.push(state(StepKind::Stroke, 3.0));
        assert_eq!((t.undo(), gamma(&t)), (Some(StepKind::Stroke), 2.0));
        assert_eq!((t.undo(), gamma(&t)), (Some(StepKind::Process), 1.0));
        assert_eq!(t.undo(), None, "the original is the floor");
        assert_eq!((t.redo(), gamma(&t)), (Some(StepKind::Process), 2.0));
        t.push(state(StepKind::Erase, 4.0));
        assert_eq!(t.redo(), None, "a new step drops the undone ones");
        assert_eq!(t.undo_steps(), 2);
    }

    #[test]
    fn every_move_changes_the_version_and_a_fill_in_does_not() {
        let mut t = Timeline::new(state(StepKind::Original, 1.0));
        let v0 = t.version();
        t.push(state(StepKind::Process, 2.0));
        let v1 = t.version();
        t.current_mut().recipe = None;
        assert_eq!(t.version(), v1, "filling in the shown state keeps jobs valid");
        t.undo();
        let v2 = t.version();
        t.redo();
        assert!(v0 < v1 && v1 < v2 && v2 < t.version());
    }

    /// The revision moves on anything that changes what the image shows:
    /// a step, a knob change merged into the last one, a fill-in, an undo
    /// or a redo. A save is current while the revision it wrote is.
    #[test]
    fn every_change_moves_the_revision() {
        let mut t = Timeline::new(state(StepKind::Original, 1.0));
        let mut last = t.revision();
        let mut moved = |t: &Timeline, what: &str| {
            assert!(t.revision() > last, "{what}");
            last = t.revision();
        };
        t.push(state(StepKind::Adjust, 2.0));
        moved(&t, "a step");
        t.push(state(StepKind::Adjust, 3.0));
        moved(&t, "a knob change merged into the last one");
        t.current_mut().recipe = None;
        moved(&t, "a fill-in");
        t.undo();
        moved(&t, "an undo");
        t.redo();
        moved(&t, "a redo");
        let _ = (t.current(), t.version(), t.undo_steps());
        assert_eq!(t.revision(), last, "reading changes nothing");
    }

    #[test]
    fn knob_changes_in_quick_succession_are_one_step() {
        let mut t = Timeline::new(state(StepKind::Original, 1.0));
        let start = Instant::now();
        for (i, g) in [1.1, 1.2, 1.3].into_iter().enumerate() {
            t.push(DocState { at: start + Duration::from_millis(100 * i as u64), ..state(StepKind::Adjust, g) });
        }
        assert_eq!((t.undo_steps(), gamma(&t)), (1, 1.3));
        t.push(DocState { at: start + Duration::from_millis(800), ..state(StepKind::Adjust, 1.4) });
        assert_eq!(t.undo_steps(), 2, "two knobs set 0.6 s apart are two steps");
        t.push(DocState { at: start + Duration::from_secs(5), ..state(StepKind::Adjust, 1.5) });
        assert_eq!(t.undo_steps(), 3, "a pause starts a new step");
        t.undo();
        t.push(DocState { at: start + Duration::from_secs(5), ..state(StepKind::Adjust, 1.6) });
        assert_eq!(t.undo_steps(), 3, "after an undo a knob change is its own step");
    }

    #[test]
    fn the_oldest_steps_go_first_and_the_cursor_follows() {
        let mut t = Timeline::new(state(StepKind::Original, 0.0));
        for i in 1..=(TIMELINE_DEPTH + 5) {
            t.push(state(StepKind::Process, i as f32));
        }
        assert_eq!(t.undo_steps(), TIMELINE_DEPTH - 1);
        assert_eq!(gamma(&t), (TIMELINE_DEPTH + 5) as f32);
        while t.undo().is_some() {}
        assert_eq!(gamma(&t), 6.0);
    }

    #[test]
    fn a_payload_compresses_only_once_nothing_else_holds_it() {
        let img = Arc::new(image::RgbaImage::from_pixel(32, 32, image::Rgba([1, 2, 3, 255])));
        let cell = Stored::new(Arc::clone(&img));
        assert!(cell.begin_compress().is_none(), "the canvas still holds it");
        drop(img);
        let value = cell.begin_compress().expect("only the cell holds it");
        assert!(cell.load().is_some(), "usable while compressing");
        cell.finish_compress(value.encode());
        assert!(cell.is_compressed());
        assert_eq!(*cell.load().unwrap(), image::RgbaImage::from_pixel(32, 32, image::Rgba([1, 2, 3, 255])));
    }

    /// A payload read back is the one value every reader gets, so the next
    /// step recognises it as unchanged; unused again, it returns to its
    /// stored bytes without re-encoding.
    #[test]
    fn a_read_back_payload_is_shared_and_returns_to_its_stored_form() {
        let cell = Stored::new(Arc::new(image::RgbaImage::from_pixel(8, 8, image::Rgba([4, 4, 4, 255]))));
        let value = cell.begin_compress().unwrap();
        cell.finish_compress(value.encode());
        drop(value);
        let shown = cell.load().unwrap();
        assert!(cell.holds(&shown));
        assert!(Arc::ptr_eq(&store(Some(&shown), Some(&cell)).unwrap(), &cell), "the next step shares the cell");
        assert!(cell.begin_compress().is_none() && !cell.is_compressed(), "still in use");
        drop(shown);
        assert!(cell.begin_compress().is_none(), "nothing to encode");
        assert!(cell.is_compressed());
    }

    #[test]
    fn a_disk_payload_reads_back_and_its_file_goes_with_it() {
        let cell = Stored::new(Arc::new(image::RgbaImage::from_pixel(8, 8, image::Rgba([9, 9, 9, 255]))));
        let value = cell.begin_compress().unwrap();
        cell.finish_compress(value.encode());
        drop(value);
        assert!(cell.demote_to_disk());
        let path = match &*cell.lock() {
            Tier::OnDisk(p) => p.clone(),
            _ => unreachable!("just demoted"),
        };
        assert_eq!(cell.load().unwrap().get_pixel(3, 3), &image::Rgba([9, 9, 9, 255]));
        drop(cell);
        super::super::disk_queue::flush();
        assert!(!path.exists(), "the file is removed with its last state");
    }

    /// Masks and images stored for undo come back as they went in, and
    /// a payload's size header reads back as written.
    #[test]
    fn payloads_round_trip() {
        let mut seed = 0x9E37_79B9u32;
        let mut next = move || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed };
        for (w, h) in [(1, 1), (3, 7), (64, 48), (257, 129)] {
            let plane = MaskArtifact::from_cells(w, h, (0..w * h).map(|_| next() as i8).collect());
            assert_eq!(MaskArtifact::decode(&plane.encode().unwrap()).unwrap(), plane, "{w}x{h} mask");
            let img = image::RgbaImage::from_fn(w, h, |_, _| image::Rgba(next().to_le_bytes()));
            assert_eq!(image::RgbaImage::decode(&img.encode().unwrap()).unwrap(), img, "{w}x{h} image");
            let (wb, hb) = (w.to_le_bytes(), h.to_le_bytes());
            let header = [wb[0], wb[1], wb[2], wb[3], hb[0], hb[1], hb[2], hb[3]];
            assert_eq!(super::super::history_disk::read_dims(&header), (w, h));
        }
    }

    #[test]
    fn an_unchanged_payload_is_shared_between_steps() {
        let img = Arc::new(image::RgbaImage::new(2, 2));
        let first = store(Some(&img), None).unwrap();
        let again = store(Some(&img), Some(&first)).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        let other = store(Some(&Arc::new(image::RgbaImage::new(2, 2))), Some(&first)).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        assert!(store::<image::RgbaImage>(None, Some(&first)).is_none());
    }
}
