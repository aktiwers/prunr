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

use super::brush_state::CutoutAction;

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
        fn eq(a: &Option<Arc<MaskArtifact>>, b: &Option<Arc<MaskArtifact>>) -> bool {
            match (a, b) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
        }
        eq(&self.selection, &other.selection) && eq(&self.edits, &other.edits)
    }
}

/// One selection edit.
#[derive(Clone)]
pub(crate) enum SelectionOp {
    /// A stroke, painted or decoded, merged into the selection: Add grows
    /// it, Subtract carves it away.
    Merge(Arc<MaskArtifact>),
    Invert,
    /// The selection folded into the cut-out's edits as `action` says,
    /// and cleared; `feathered` is the plane the overlay shows, when built.
    Bake { action: CutoutAction, feathered: Option<Arc<MaskArtifact>> },
}

/// What happens once an edit has landed, besides the step it records.
#[derive(Clone)]
pub(crate) enum AfterLanding {
    Nothing,
    /// A Magic Brush click: its decode is kept for a Confidence retune.
    MagicDecode {
        output: Arc<prunr_core::sam::SamDecoderOutput>,
        mode: prunr_core::selection::BrushMode,
        reading: prunr_core::MaskReading,
    },
    /// A Magic Brush "By depth" stroke: no decode to retune.
    ByDepth,
}

/// An edit for the worker.
struct Job {
    item_id: u64,
    op: SelectionOp,
    after: AfterLanding,
    /// `None`: the image's previous edit's output.
    base: Option<SelState>,
}

/// An edit made: what it was built from and what it makes, with the new
/// selection's content hash.
pub(crate) struct Done {
    pub(crate) item_id: u64,
    pub(crate) op: SelectionOp,
    pub(crate) after: AfterLanding,
    pub(crate) base: SelState,
    pub(crate) made: SelState,
    pub(crate) hash: Option<u64>,
}

pub(crate) struct SelectionQueue {
    tx: mpsc::Sender<Job>,
    rx: mpsc::Receiver<Done>,
    /// Edits sent per image and not landed yet.
    pending: HashMap<u64, u32>,
    /// Undo / redo asked for while edits were pending, replayed in order
    /// once the queue is empty: an undo then undoes the stroke just made.
    deferred: Vec<super::views::shortcuts::Action>,
}

impl SelectionQueue {
    pub(crate) fn new(ctx: Option<egui::Context>) -> Self {
        let (tx, jobs) = mpsc::channel::<Job>();
        let (done_tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("selection-edits".into())
            .spawn(move || {
                let mut last: HashMap<u64, SelState> = HashMap::new();
                for job in jobs {
                    let base = job.base.unwrap_or_else(|| last.get(&job.item_id).cloned().unwrap_or_default());
                    let made = apply(&job.op, &base);
                    let hash = made.selection.as_ref().map(|s| s.content_hash());
                    last.insert(job.item_id, made.clone());
                    let _ = done_tx.send(Done { item_id: job.item_id, op: job.op, after: job.after, base, made, hash });
                    if let Some(ctx) = &ctx {
                        ctx.request_repaint();
                    }
                }
            })
            .map_err(|err| tracing::error!(%err, "selection edit thread did not start"))
            .ok();
        Self { tx, rx, pending: HashMap::new(), deferred: Vec::new() }
    }

    /// Queue `op` on the image showing `current`; it builds on the image's
    /// edits still pending, if any.
    pub(crate) fn push(&mut self, item_id: u64, op: SelectionOp, after: AfterLanding, current: SelState) {
        let count = self.pending.entry(item_id).or_default();
        let base = (*count == 0).then_some(current);
        *count += 1;
        let _ = self.tx.send(Job { item_id, op, after, base });
    }

    /// The edits made since the last frame, in order; each is no longer
    /// pending (one that goes back in the queue is pushed again).
    pub(crate) fn take_done(&mut self) -> Vec<Done> {
        let done: Vec<Done> = self.rx.try_iter().collect();
        for d in &done {
            if let Some(count) = self.pending.get_mut(&d.item_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.pending.remove(&d.item_id);
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

    /// Hold `action` until the queue is empty.
    pub(crate) fn defer(&mut self, action: super::views::shortcuts::Action) {
        self.deferred.push(action);
    }

    /// The actions held, once nothing is pending.
    pub(crate) fn take_deferred(&mut self) -> Vec<super::views::shortcuts::Action> {
        if self.any_pending() { Vec::new() } else { std::mem::take(&mut self.deferred) }
    }
}

/// `op` on `base`. Peak working set: the new plane (1 B/px, 12 MB at
/// 12 MP) beside the inputs it reads; Bake's correction build documents
/// its own.
pub(crate) fn apply(op: &SelectionOp, base: &SelState) -> SelState {
    match op {
        SelectionOp::Merge(stroke) => SelState {
            selection: Some(Arc::new(super::processor::merge_stroke(base.selection.clone(), (**stroke).clone()))),
            edits: base.edits.clone(),
        },
        SelectionOp::Invert => SelState {
            selection: base.selection.as_ref().map(|s| Arc::new(s.invert())),
            edits: base.edits.clone(),
        },
        SelectionOp::Bake { action, feathered } => {
            let Some(selection) = base.selection.clone() else { return base.clone() };
            let edits = match action {
                CutoutAction::Select => base.edits.clone(),
                action => {
                    let shown = feathered.clone().unwrap_or(selection);
                    Some(Arc::new(super::item::apply_selection(base.edits.as_deref(), &shown, *action)))
                }
            };
            SelState { selection: None, edits }
        }
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
        use prunr_core::selection::FULL;
        let mut queue = SelectionQueue::new(None);
        let first = plane(&[FULL, 0, 0]);
        let second = plane(&[0, 0, FULL]);
        queue.push(1, SelectionOp::Merge(Arc::clone(&first)), AfterLanding::Nothing, SelState::default());
        queue.push(1, SelectionOp::Merge(Arc::clone(&second)), AfterLanding::Nothing, SelState::default());
        assert!(queue.is_pending(1));
        let done = wait(&mut queue, 2);
        assert_eq!(done.len(), 2);
        assert!(done[1].base.same(&done[0].made), "the second starts from the first's output");
        let both = done[1].made.selection.as_ref().expect("a selection");
        assert!(both.cells()[0] != 0 && both.cells()[2] != 0, "both strokes are in it");
        assert_eq!(done[1].hash, Some(both.content_hash()));
        assert!(!queue.is_pending(1) && !queue.any_pending());
    }

    /// Bake folds the selection into the edits and clears it; Select
    /// keeps the edits as they were.
    #[test]
    fn a_bake_folds_the_selection_into_the_edits() {
        use prunr_core::selection::FULL;
        let base = SelState { selection: Some(plane(&[FULL, 0])), edits: None };
        let erased = apply(&SelectionOp::Bake { action: CutoutAction::Erase, feathered: None }, &base);
        assert!(erased.selection.is_none() && erased.edits.is_some());
        let kept = apply(&SelectionOp::Bake { action: CutoutAction::Select, feathered: None }, &base);
        assert!(kept.selection.is_none() && kept.edits.is_none());
    }

    /// Undo asked for while a stroke is pending waits for the queue.
    #[test]
    fn an_action_waits_for_the_queue() {
        use super::super::views::shortcuts::Action;
        let mut queue = SelectionQueue::new(None);
        queue.push(1, SelectionOp::Invert, AfterLanding::Nothing, SelState { selection: Some(plane(&[0])), edits: None });
        queue.defer(Action::Undo);
        assert!(queue.take_deferred().is_empty(), "held while pending");
        wait(&mut queue, 1);
        assert_eq!(queue.take_deferred(), vec![Action::Undo]);
    }
}
