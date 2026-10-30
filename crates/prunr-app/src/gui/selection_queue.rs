//! Selection edits (a stroke merged in, Invert, a cut-out's selection
//! folded into its edits) are full-resolution passes: 10-25 ms each at
//! 12 MP. They run in order on one worker, each image's chained on the
//! previous one's output so quick strokes build on each other, as a
//! paint program's strokes queue does. A result lands only on the
//! selection and edits it was built from; on any other it goes back in
//! the queue against what the image shows then, so no stroke is lost.

use std::collections::HashMap;
use std::sync::{mpsc, Arc};

use prunr_core::selection::MaskArtifact;

use super::brush_state::{CutoutAction, StrokeDrawn};
use super::timeline::StrokeHistory;

/// An image's selection and its cut-out edits: what an edit starts from
/// and makes.
#[derive(Clone, Default)]
pub(crate) struct SelState {
    pub(crate) selection: Option<Arc<MaskArtifact>>,
    pub(crate) edits: Option<Arc<MaskArtifact>>,
}

impl SelState {
    /// Whether both hold the very same planes.
    pub(crate) fn same(&self, other: &Self) -> bool {
        use super::item::same_arc;
        same_arc(self.selection.as_ref(), other.selection.as_ref()) && same_arc(self.edits.as_ref(), other.edits.as_ref())
    }
}

#[derive(Clone)]
pub(crate) enum SelectionOp {
    /// A stroke drawn here on a `dims` image with its edge (a selection's
    /// feathered against the photo `source`, an eraser's grown with its
    /// fill's band), and merged into the selection: Add grows it,
    /// Subtract carves it away. Its step records it, for its brush's
    /// sliders to draw it again.
    Stroke { drawn: StrokeDrawn, dims: (u32, u32), source: Option<Arc<image::RgbaImage>> },
    Invert,
    /// The selection folded into the cut-out's edits as `action` says,
    /// and cleared.
    Bake { action: CutoutAction },
    /// Strokes drawn again from the step they start from.
    Replay(Arc<Replay>),
}

impl SelectionOp {
    /// The step the edit makes once landed, and what a replay of the
    /// strokes records of it.
    pub(crate) fn step(&self) -> (super::timeline::StepKind, Option<Made>) {
        use super::timeline::StepKind;
        match self {
            Self::Stroke { drawn, .. } => (StepKind::Stroke, Some(Made::One(SelectionMade::Stroke(drawn.clone())))),
            Self::Invert => (StepKind::Selection, Some(Made::One(SelectionMade::Invert))),
            Self::Bake { action } => (StepKind::Selection, Some(Made::One(SelectionMade::Fold { action: *action }))),
            Self::Replay(replay) => (StepKind::Strokes, Some(replay.record())),
        }
    }
}

/// One thing a step did to the selection, as a replay of the strokes
/// redoes it.
#[derive(Clone)]
pub(crate) enum SelectionMade {
    Stroke(StrokeDrawn),
    Invert,
    /// Folded into the cut-out's edits.
    Fold { action: CutoutAction },
    Clear,
}

impl SelectionMade {
    /// A stroke as `change` makes it; anything else as it was.
    pub(crate) fn changed(&self, change: impl Fn(&StrokeDrawn) -> StrokeDrawn) -> Self {
        match self {
            Self::Stroke(drawn) => Self::Stroke(change(drawn)),
            other => other.clone(),
        }
    }

    /// A stroke `tool` drew.
    pub(crate) fn drawn_by(&self, tool: super::brush_state::Tool) -> bool {
        matches!(self, Self::Stroke(drawn) if drawn.tool() == tool)
    }

    /// Whether a redraw of one in place of the other would change it.
    fn differs(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Stroke(a), Self::Stroke(b)) => a.values() != b.values(),
            _ => false,
        }
    }
}

/// What a step did to the selection: the record a replay of the image's
/// strokes redoes. A selection step without one stops a replay there.
#[derive(Clone)]
pub(crate) enum Made {
    One(SelectionMade),
    /// Every stroke since step `from` drawn again: `ops` is that history
    /// as it stands now.
    Redrawn { from: u64, ops: Arc<Vec<SelectionMade>> },
}

/// Whether any of `ops` would draw differently from `shown`.
pub(crate) fn any_differ(ops: &[SelectionMade], shown: &[SelectionMade]) -> bool {
    ops.iter().zip(shown).any(|(a, b)| a.differs(b))
}

/// Strokes to draw again on a `dims` image: from where they started, each
/// as it is to be drawn now.
pub(crate) struct Replay {
    pub(crate) strokes: StrokeHistory,
    pub(crate) dims: (u32, u32),
    /// The photo each stroke's feather follows.
    pub(crate) source: Option<Arc<image::RgbaImage>>,
}

impl Replay {
    /// What the step it lands on records.
    fn record(&self) -> Made {
        match (self.strokes.own_step, self.strokes.ops.first()) {
            (true, Some(op)) => Made::One(op.clone()),
            _ => Made::Redrawn { from: self.strokes.from, ops: Arc::clone(&self.strokes.ops) },
        }
    }

    /// The planes the steps make, from the ones they started from; the
    /// stored planes are unpacked here, on the edit thread.
    fn make(&self) -> SelState {
        let (w, h) = self.dims;
        let start = SelState {
            selection: self.strokes.selection.as_ref().and_then(|s| s.load()),
            edits: self.strokes.edits.as_ref().and_then(|e| e.load()),
        };
        self.strokes.ops.iter().fold(start, |state, op| match op {
            SelectionMade::Stroke(drawn) => merged(&state, drawn.mask(w, h, self.source.as_deref())),
            SelectionMade::Invert => apply(&SelectionOp::Invert, &state),
            SelectionMade::Fold { action } => apply(&SelectionOp::Bake { action: *action }, &state),
            SelectionMade::Clear => SelState { selection: None, edits: state.edits },
        })
    }
}

/// What the worker is sent: an edit, or word that an image's chain is
/// done (nothing pending), so its last output is let go.
enum Msg {
    Job(Job),
    Forget(u64),
}

struct Job {
    item_id: u64,
    op: SelectionOp,
    /// `None`: the image's previous edit's output.
    base: Option<SelState>,
}

/// An edit made: what it was built from and what it makes, with the new
/// selection's content hash and, for a stroke, the mask it drew.
pub(crate) struct Done {
    pub(crate) item_id: u64,
    pub(crate) op: SelectionOp,
    pub(crate) base: SelState,
    pub(crate) made: SelState,
    pub(crate) hash: Option<u64>,
    pub(crate) stroke: Option<super::item::StrokePatch>,
}

pub(crate) struct SelectionQueue {
    tx: mpsc::Sender<Msg>,
    rx: mpsc::Receiver<Done>,
    /// Edits sent per image and not landed yet.
    pending: HashMap<u64, u32>,
    /// Undo / redo asked for while the images it acts on had edits
    /// pending, with those images, replayed in order once they have none:
    /// an undo then undoes the stroke just made.
    deferred: Vec<(super::views::shortcuts::Action, Vec<u64>)>,
    /// Woken when an edit is made; set once the window exists.
    repaint: Arc<std::sync::OnceLock<egui::Context>>,
}

impl SelectionQueue {
    pub(crate) fn new() -> Self {
        let (tx, jobs) = mpsc::channel::<Msg>();
        let (done_tx, rx) = mpsc::channel();
        let repaint: Arc<std::sync::OnceLock<egui::Context>> = Arc::default();
        let wake = Arc::clone(&repaint);
        std::thread::Builder::new()
            .name("selection-edits".into())
            .spawn(move || {
                // Each image's last output while its chain runs (12 MB a
                // plane at 12 MP): dropped when the chain is done.
                let mut last: HashMap<u64, SelState> = HashMap::new();
                for msg in jobs {
                    let job = match msg {
                        Msg::Job(job) => job,
                        Msg::Forget(item_id) => {
                            last.remove(&item_id);
                            continue;
                        }
                    };
                    let base = job.base.unwrap_or_else(|| last.get(&job.item_id).cloned().unwrap_or_default());
                    // A panic makes nothing, so the image is not left waiting on it.
                    let (made, stroke) = crate::panics::caught("a selection edit", || made_by(&job.op, &base))
                        .unwrap_or_else(|| (base.clone(), None));
                    // Hashed here, and the overlay's question answered here,
                    // so the UI thread reads both without scanning the plane.
                    let hash = made.selection.as_ref().map(|s| {
                        s.has_unlayered_selection();
                        s.content_hash()
                    });
                    last.insert(job.item_id, made.clone());
                    let _ = done_tx.send(Done { item_id: job.item_id, op: job.op, base, made, hash, stroke });
                    if let Some(ctx) = wake.get() {
                        ctx.request_repaint();
                    }
                }
            })
            .map_err(|err| tracing::error!(%err, "selection edit thread did not start"))
            .ok();
        Self { tx, rx, pending: HashMap::new(), deferred: Vec::new(), repaint }
    }

    pub(crate) fn wake(&self, ctx: &egui::Context) {
        let _ = self.repaint.set(ctx.clone());
    }

    /// Queue `op` on the image showing `current`; it builds on the image's
    /// edits still pending, if any.
    pub(crate) fn push(&mut self, item_id: u64, op: SelectionOp, current: SelState) {
        let chained = self.pending.get(&item_id).is_some_and(|&n| n > 0);
        self.send(item_id, op, (!chained).then_some(current));
    }

    /// Queue `op` again, on `current`, the state the image shows now: an
    /// edit made on a state the image has left. The edits still pending
    /// for it chain on this one in turn, so the queue settles.
    pub(crate) fn rebase(&mut self, item_id: u64, op: SelectionOp, current: SelState) {
        self.send(item_id, op, Some(current));
    }

    fn send(&mut self, item_id: u64, op: SelectionOp, base: Option<SelState>) {
        *self.pending.entry(item_id).or_default() += 1;
        let _ = self.tx.send(Msg::Job(Job { item_id, op, base }));
    }

    /// The edits made since the last frame, in order; each is no longer
    /// pending (one that goes back in the queue is pushed again). An
    /// image left with none pending has its chain let go.
    pub(crate) fn take_done(&mut self) -> Vec<Done> {
        let done: Vec<Done> = self.rx.try_iter().collect();
        for d in &done {
            if let Some(count) = self.pending.get_mut(&d.item_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.pending.remove(&d.item_id);
                    let _ = self.tx.send(Msg::Forget(d.item_id));
                }
            }
        }
        done
    }

    pub(crate) fn is_pending(&self, item_id: u64) -> bool {
        self.pending.contains_key(&item_id)
    }

    pub(crate) fn any_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Hold `action` on the images `ids` until they have no edits pending.
    pub(crate) fn defer(&mut self, action: super::views::shortcuts::Action, ids: Vec<u64>) {
        self.deferred.push((action, ids));
    }

    /// The held actions whose images are done, here and by `elsewhere`, in
    /// the order held; one waiting holds the ones after it too.
    pub(crate) fn take_deferred(&mut self, elsewhere: impl Fn(u64) -> bool) -> Vec<(super::views::shortcuts::Action, Vec<u64>)> {
        let ready = self.deferred.iter().take_while(|(_, ids)| !ids.iter().any(|&id| self.is_pending(id) || elsewhere(id))).count();
        self.deferred.drain(..ready).collect()
    }
}

/// `op` on `base`, and where a stroke painted, for an erase filling
/// meanwhile.
fn made_by(op: &SelectionOp, base: &SelState) -> (SelState, Option<super::item::StrokePatch>) {
    match op {
        SelectionOp::Stroke { drawn, dims: (w, h), source } => {
            let stroke = drawn.mask(*w, *h, source.as_deref());
            let patch = super::item::StrokePatch::of(&stroke);
            (merged(base, stroke), patch)
        }
        op => (apply(op, base), None),
    }
}

/// `op` on `base`. Peak working set: the new plane (1 B/px, 12 MB at
/// 12 MP) beside the inputs it reads, and a stroke's own mask; Bake's
/// correction build documents its own.
pub(crate) fn apply(op: &SelectionOp, base: &SelState) -> SelState {
    match op {
        SelectionOp::Stroke { drawn, dims: (w, h), source } => merged(base, drawn.mask(*w, *h, source.as_deref())),
        SelectionOp::Invert => SelState {
            selection: base.selection.as_ref().map(|s| Arc::new(s.invert())),
            edits: base.edits.clone(),
        },
        SelectionOp::Bake { action } => {
            let Some(selection) = base.selection.clone() else { return base.clone() };
            // A blank selection folds in nothing: the edits stay as they are.
            if selection.is_blank() {
                return SelState { selection: None, edits: base.edits.clone() };
            }
            let edits = match action {
                CutoutAction::Select => base.edits.clone(),
                action => Some(Arc::new(super::item::apply_selection(base.edits.as_deref(), &selection, *action))),
            };
            SelState { selection: None, edits }
        }
        SelectionOp::Replay(replay) => replay.make(),
    }
}

/// `stroke` merged into `base`'s selection.
fn merged(base: &SelState, stroke: MaskArtifact) -> SelState {
    SelState {
        selection: Some(Arc::new(super::processor::merge_stroke(base.selection.clone(), stroke))),
        edits: base.edits.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(cells: &[i8]) -> Arc<MaskArtifact> {
        Arc::new(MaskArtifact::from_cells(cells.len() as u32, 1, cells.to_vec()))
    }

    fn wait(queue: &mut SelectionQueue, n: usize) -> Vec<Done> {
        let mut done = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while done.len() < n && std::time::Instant::now() < deadline {
            done.extend(queue.take_done());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        done
    }

    /// Two strokes queued before either lands: the second builds on the
    /// first's output, in order; the image is pending until both land.
    #[test]
    fn quick_strokes_build_on_each_other_in_order() {
        let mut queue = SelectionQueue::new();
        let at = |x: f32| {
            let drawn = super::super::tests::fixtures::painted(prunr_core::brush::BrushShape::Square, 1.0, &[(x, 0.5, 0.5)], super::super::tests::fixtures::HARD);
            SelectionOp::Stroke { drawn, dims: (3, 1), source: None }
        };
        queue.push(1, at(0.5), SelState::default());
        queue.push(1, at(2.5), SelState::default());
        assert!(queue.is_pending(1));
        let done = wait(&mut queue, 2);
        assert_eq!(done.len(), 2);
        assert!(done[1].base.same(&done[0].made), "the second starts from the first's output");
        let both = done[1].made.selection.as_ref().expect("a selection");
        assert!(both.cells()[0] != 0 && both.cells()[2] != 0, "both strokes are in it");
        assert_eq!(done[1].hash, Some(both.content_hash()));
        assert!(!queue.is_pending(1) && !queue.any_pending());
    }

    /// Once an image has nothing pending, the worker lets its chain go:
    /// it holds none of the image's planes.
    #[test]
    fn a_done_chain_holds_no_planes() {
        let mut queue = SelectionQueue::new();
        queue.push(1, SelectionOp::Invert, SelState { selection: Some(plane(&[0, 0])), edits: None });
        let done = wait(&mut queue, 1);
        let made = done[0].made.selection.clone().expect("inverted");
        drop(done);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while Arc::strong_count(&made) > 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(Arc::strong_count(&made), 1, "the worker let it go");
    }

    /// Bake folds the selection into the edits and clears it; Select
    /// keeps the edits as they were.
    #[test]
    fn a_bake_folds_the_selection_into_the_edits() {
        use prunr_core::selection::FULL;
        let base = SelState { selection: Some(plane(&[FULL, 0])), edits: None };
        let erased = apply(&SelectionOp::Bake { action: CutoutAction::Erase }, &base);
        assert!(erased.selection.is_none() && erased.edits.is_some());
        let kept = apply(&SelectionOp::Bake { action: CutoutAction::Select }, &base);
        assert!(kept.selection.is_none() && kept.edits.is_none());
        let edits = plane(&[-FULL, 0]);
        let blank = SelState { selection: Some(plane(&[0, 0])), edits: Some(Arc::clone(&edits)) };
        let folded = apply(&SelectionOp::Bake { action: CutoutAction::Erase }, &blank);
        assert!(folded.selection.is_none() && folded.edits.is_some_and(|e| Arc::ptr_eq(&e, &edits)), "a blank selection folds in nothing");
    }

    /// Undo asked for while a stroke is pending waits for the queue, and
    /// for an edit still being made elsewhere (a Magic Brush pick).
    #[test]
    fn an_action_waits_for_the_queue() {
        use super::super::views::shortcuts::Action;
        let mut queue = SelectionQueue::new();
        queue.push(1, SelectionOp::Invert, SelState { selection: Some(plane(&[0])), edits: None });
        queue.defer(Action::Undo, vec![1]);
        queue.defer(Action::Redo, vec![2]);
        assert!(queue.take_deferred(|_| false).is_empty(), "held while its image is pending, and the one after it too");
        wait(&mut queue, 1);
        assert!(queue.take_deferred(|id| id == 1).is_empty(), "or while its image is still being edited elsewhere");
        assert_eq!(queue.take_deferred(|_| false), vec![(Action::Undo, vec![1]), (Action::Redo, vec![2])]);
    }
}
