//! The brushes and the selection: tool switching, selection actions, Magic Brush encode and decode.

use std::sync::Arc;

use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::timeline::StepKind;
use crate::gui::views::selection_action_bar::SelectionAction;
use crate::gui::brush_state::Tool;
use crate::gui::item::BatchItem;
use crate::gui::selection_queue::{AfterLanding, SelectionOp};
use super::PrunrApp;

impl PrunrApp {
    /// What the tool strip or a brush panel changed: persist on a
    /// settle, reset from the active preset, flip Auto-apply.
    pub(crate) fn apply_brush_change(&mut self, change: crate::gui::views::brush_chip::BrushChipOutcome) {
        if change.reset_brush_requested {
            let resolved = self.settings.resolve_active_preset(None);
            self.settings.brush.reset_popover_fields_from(&resolved.brush);
        }
        if change.committed {
            self.settings.save();
        }
        if change.action_changed {
            let model = self.settings.model;
            let follow: Vec<usize> = self.batch.items.iter().enumerate()
                .filter(|(_, i)| i.selection_mask.is_some() && i.corrects_cutout(model))
                .map(|(idx, _)| idx)
                .collect();
            for idx in follow {
                self.dispatch_brush_rerun(idx);
            }
        }
    }

    pub(super) fn dispatch_brush_rerun(&mut self, idx: usize) {
        use crate::gui::live_preview::PreviewKind;
        let item_id = self.batch.items[idx].id;
        self.processor.live_preview.render_now(item_id, PreviewKind::Mask);
    }

    /// Per-model interpretation rule for a newly-committed selection.
    /// Called after every commit, and again when a result lands, so the BG-removal
    /// continuous-auto-apply UX is preserved while SD / LaMa wait for
    /// explicit Process.
    ///
    /// IMPORTANT: every call site that calls `BatchManager::commit_selection`
    /// MUST follow up with this method. Pairing the two prevents per-model
    /// dispatch rules from drifting between input surfaces.
    pub(crate) fn apply_selection_to_active_model(&mut self, item_id: u64) {
        let Some(idx) = self.batch.index_of(item_id) else { return };
        // A cut-out re-cuts at once; the eraser's region waits for Process;
        // elsewhere the selection only feeds Delete / Copy / Cut.
        if !self.settings.model.edits_cutout() {
            return;
        }
        // With Feather on, the feathered plane is built off the UI thread
        // and its landing re-cuts; cutting now would flash the hard edge.
        let waits_for_feather = self.settings.brush.edge_feather >= 0.5 && self.batch.items[idx].selection_mask.is_some();
        if !waits_for_feather {
            self.rerun_with_strokes(idx);
        }
    }

    /// Re-cut a shown cut-out with its selection: when a result lands with a
    /// selection kept across Process (part of that run, so no step of its
    /// own), and when the selection's feathered plane lands.
    pub(crate) fn recut_with_selection(&mut self, item_id: u64) {
        let Some(idx) = self.batch.index_of(item_id) else { return };
        let item = &self.batch.items[idx];
        if (item.selection_mask.is_some() || item.edits.is_some()) && item.shows_cutout() && self.settings.model.edits_cutout() {
            self.dispatch_brush_rerun(idx);
        }
    }

    /// Re-cuts the image with the painted selection: Restore and Erase
    /// strokes take effect at once, from the cached tensor. Without a
    /// cut-out on screen there is nothing to correct yet; the selection
    /// waits and `on_batch_item_done` applies it when a result lands.
    fn rerun_with_strokes(&mut self, idx: usize) {
        if !self.batch.items[idx].shows_cutout() {
            return;
        }
        self.dispatch_brush_rerun(idx);
    }

    /// Single-source-of-truth call for writing a new selection: persist the
    /// mask and fire the per-model dispatch rule. The commit drops the old
    /// texture, so `ensure_selection_texture` rebuilds it next frame.
    ///
    /// Tests set a selection this way; the app's authors go through
    /// `queue_selection_op`.
    #[cfg(test)]
    pub(crate) fn commit_selection_and_dispatch(
        &mut self,
        item_id: u64,
        mask: prunr_core::selection::MaskArtifact,
        kind: StepKind,
    ) {
        if !self.batch.commit_selection(item_id, mask, kind) {
            return;
        }
        self.apply_selection_to_active_model(item_id);
    }

    /// Make a selection edit off the UI thread, in order with the image's
    /// other edits; it lands through `land_selection_edits`. Every author
    /// (Paint Brush, Magic Brush, Invert, a fold into the cut-out) goes
    /// through here.
    pub(crate) fn queue_selection_op(&mut self, item_id: u64, op: SelectionOp, after: AfterLanding) {
        let Some(current) = self.batch.find_by_id(item_id).map(BatchItem::selection_state) else { return };
        self.batch.selection_queue.push(item_id, op, after, current);
    }

    /// The selection edits made since the last frame land, in order: each
    /// on the selection it was built from, as one step; one whose base
    /// the image has left goes back in the queue against what it shows
    /// now. Undo / redo held for the queue run once it is empty.
    pub(crate) fn land_selection_edits(&mut self, ctx: &egui::Context) {
        use crate::gui::selection_queue::SelectionOp;
        let style = self.selection_style();
        let mut landed = false;
        for done in self.batch.selection_queue.take_done() {
            landed = true;
            let item_id = done.item_id;
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if !item.selection_state().same(&done.base) {
                tracing::info!(item_id, "selection edit made on a selection since changed; made again");
                let current = item.selection_state();
                // A fold reads the feathered plane of the selection it folds.
                let op = match done.op {
                    SelectionOp::Bake { action, .. } => SelectionOp::Bake { action, feathered: item.feathered_selection(style) },
                    op => op,
                };
                self.batch.selection_queue.rebase(item_id, op, done.after, current);
                continue;
            }
            let kind = match done.op {
                SelectionOp::Merge(_) => StepKind::Stroke,
                SelectionOp::Invert | SelectionOp::Bake { .. } => StepKind::Selection,
            };
            let changed = item.land_selection_state(done.made, done.hash, kind);
            // A click that added nothing leaves the last retunable one as it was.
            match done.after {
                AfterLanding::MagicDecode { output, mode, reading } if changed => {
                    item.last_decode = item.selection_hash.map(|committed_hash| crate::gui::item::LastDecode { output, mode, committed_hash, reading });
                }
                AfterLanding::ByDepth if changed => item.last_decode = None,
                _ => {}
            }
            if changed {
                self.apply_selection_to_active_model(item_id);
            }
            ctx.request_repaint();
        }
        for (action, ids) in self.batch.selection_queue.take_deferred() {
            self.step_items(ctx, action, &ids);
        }
        // A queued run that waited for its image's selection goes now.
        if landed {
            self.dispatch_next_queued_run();
        }
    }

    /// Dispatch a selection action (Delete / Copy / Cut / Invert / Clear).
    /// Called from `apply_toolbar_change` and from keyboard shortcuts that
    /// have a live selection.
    ///
    /// - Delete: scales alpha by coverage inside the selection on
    ///   `result_rgba`, as one undo step so Cmd+Z can
    ///   restore it.
    /// - Copy: copies selection-masked pixels to the system clipboard. No
    ///   undo step — non-destructive read.
    /// - Cut: Copy + Delete in one undo step, not two.
    /// - Invert: replaces the selection mask with its complement and
    ///   commits it like any other author.
    /// - Clear: removes the selection mask entirely.
    pub(crate) fn handle_selection_action(
        &mut self,
        idx: usize,
        action: SelectionAction,
        ctx: &egui::Context,
    ) {
        let item_id = self.batch.items[idx].id;

        // On a cut-out the selection is folded into the cut-out's edits:
        // Delete erases it, Restore brings it back, Clear keeps what the
        // switch is doing to it and starts a new area. Before the cut-out
        // exists, Delete cuts the photo itself.
        if self.batch.items[idx].corrects_cutout(self.settings.model) {
            use crate::gui::brush_state::CutoutAction;
            let fold = match action {
                SelectionAction::Restore => Some(CutoutAction::Restore),
                SelectionAction::Delete => Some(CutoutAction::Erase),
                SelectionAction::Clear => Some(self.settings.brush.cutout_action),
                SelectionAction::Cut => {
                    // The copy is made from the selection before it folds in.
                    self.start_selection_edit(idx, SelectionEdit::Copy(crate::gui::system_bridge::CopyNote::SelectionCut));
                    Some(CutoutAction::Erase)
                }
                SelectionAction::Copy | SelectionAction::Invert => None,
            };
            if let Some(fold) = fold {
                let feathered = self.batch.items[idx].feathered_selection(self.selection_style());
                self.queue_selection_op(item_id, SelectionOp::Bake { action: fold, feathered }, AfterLanding::Nothing);
                ctx.request_repaint();
                return;
            }
        }

        match action {
            // Only a cut-out offers it; handled above.
            SelectionAction::Restore => {}
            SelectionAction::Delete => self.start_selection_edit(idx, SelectionEdit::Cut),
            SelectionAction::Copy => self.start_selection_edit(idx, SelectionEdit::Copy(crate::gui::system_bridge::CopyNote::Selection)),
            SelectionAction::Cut => self.start_selection_edit(idx, SelectionEdit::CopyAndCut),
            SelectionAction::Invert => {
                if self.batch.items[idx].selection_mask.is_some() {
                    self.queue_selection_op(item_id, SelectionOp::Invert, AfterLanding::Nothing);
                }
                ctx.request_repaint();
            }
            SelectionAction::Clear => {
                if self.batch.clear_selection(item_id) {
                    self.apply_selection_to_active_model(item_id);
                }
                ctx.request_repaint();
            }
        }
    }

    /// Whether `action` can act on the shown image's selection: Copy
    /// needs a painted region on pixels that can be read; Clear needs a
    /// selection; every other action also changes the image, so it waits
    /// for a run under way, which it would make stale. Clear is a step,
    /// so it waits too. Shared by the action bar, the shortcuts, Escape
    /// and the actions themselves.
    pub(crate) fn can_selection_action(&self, action: SelectionAction) -> bool {
        self.selection_action_refusal(action).is_none()
    }

    /// Why `action` can't act now, if it can't: the action bar's tooltip.
    pub(crate) fn selection_action_refusal(&self, action: SelectionAction) -> Option<&'static str> {
        use crate::gui::capability::{Need, Reason};
        if !shows(self.settings.model, Surface::Brushes) {
            return Some(Reason::NotForModel.text());
        }
        let role = self.settings.model.role();
        let item = self.batch.selected_item()?;
        // Every action reads the selection the edits under way will change.
        if self.batch.selection_queue.is_pending(item.id) {
            return Some(Reason::FinishingSelection.text());
        }
        if item.selection_mask.is_none() {
            return Some(Reason::PaintFirst.text());
        }
        if action != SelectionAction::Copy && !item.is_idle() {
            return Some(Reason::Running.text());
        }
        if action == SelectionAction::Clear {
            return None;
        }
        item.verdict(Need::PaintedRegion, role).reason()
            .or_else(|| item.verdict(Need::ShownImage, role).reason())
            .map(Reason::text)
    }

    /// Make a selection Delete / Cut / Copy off the UI thread: the
    /// handles are taken here, the image is unpacked, feathered, copied
    /// and cut on a worker, and `land_selection_edit` applies it.
    fn start_selection_edit(&mut self, idx: usize, edit: SelectionEdit) {
        let item = &self.batch.items[idx];
        let Some(mask) = item.selection_mask.clone() else { return };
        // What gets cut is what the overlay shows: the feathered mask, which
        // the texture build already computed when it is current.
        let style = self.selection_style();
        let feathered = item.current_selection_texture(style).and_then(|t| t.feathered.clone());
        let (base, photo) = (item.edit_pixels(), item.photo_pixels());
        let (item_id, version) = (item.id, item.timeline.version());
        let tx = self.batch.bg_io.selection_edit_tx.clone();
        std::thread::spawn(move || {
            let Some(base) = base.load() else {
                tracing::warn!(item_id, "selection edit skipped: the image could not be read");
                return;
            };
            let shown = feathered
                .or_else(|| style.feather(&mask, photo.load().as_deref()))
                .unwrap_or(mask);
            let copy = edit.copy_note().and_then(|note| shown.copy_to_rgba(&base).map(|rgba| (rgba, note)));
            let cut = edit.cuts().then(|| {
                let mut edited = (*base).clone();
                shown.alpha_cut(&mut edited);
                edited
            });
            let _ = tx.send(crate::gui::background_io::SelectionEditDone { item_id, version, copy, cut });
        });
    }

    /// A selection edit made off the UI thread lands: its copy goes to
    /// the clipboard, and its cut becomes the result, one undo step, when
    /// the image is still in the state the edit started from.
    pub(super) fn land_selection_edit(&mut self, done: crate::gui::background_io::SelectionEditDone, ctx: &egui::Context) {
        let crate::gui::background_io::SelectionEditDone { item_id, version, copy, cut } = done;
        if let Some((rgba, note)) = copy {
            self.system.copy_image(Arc::new(rgba), note);
        }
        let Some(cut) = cut else { return };
        let Some(idx) = self.batch.index_of(item_id) else { return };
        if self.batch.items[idx].timeline.version() != version {
            tracing::info!(item_id, "selection cut dropped: the image changed while it was made");
            self.toasts.warning("The image changed while the area was being cut, so it was not cut. Try again.");
            return;
        }
        self.publish_result(item_id, Arc::new(cut), "edit", ctx);
        self.batch.items[idx].commit(StepKind::Edit);
    }

    /// Eager encoder: once Magic Brush has been used, the selected image
    /// gets an embedding as soon as it has a decoded source, so the tool
    /// is ready the moment it is turned on again. Runs from
    /// `reconcile_selected` every frame. With the tool off it waits for
    /// the warm sessions and an idle item, and a refusal is remembered
    /// instead of retried; with the tool on, admission or encoder failure
    /// turns the tool off, so this cannot retry in a loop.
    pub(super) fn ensure_magic_embedding_for_selected(&mut self, ctx: &egui::Context) {
        if !self.magic_brush_ready_ahead() || self.magic_brush_state.has_pending_encoder() {
            return;
        }
        if let Some(wait) = self.magic_brush_state.settle_remaining(std::time::Instant::now()) {
            ctx.request_repaint_after(wait);
            return;
        }
        let Some(item) = self.batch.selected_item() else { return };
        if item.magic_brush_embedding.is_some() {
            return;
        }
        let active = self.magic_brush_state.is_active();
        if !active
            && (item.magic_preencode_refused
                || !self.processor.sam_sessions_ready()
                || !item.is_idle())
        {
            return;
        }
        // Decode still pending: try again next frame.
        let Some(source) = item.source_rgba.clone() else { return };
        let item_id = item.id;
        let avail_ram = crate::hardware::available_ram_mb_throttled();
        match self.processor.dispatch_sam_encoder(item_id, source, avail_ram) {
            Ok(()) => self.magic_brush_state.set_encoder_pending(true),
            Err(err) => self.on_encoder_failure(item_id, &err),
        }
    }

    /// A Magic Brush click with By depth on: everything near that depth.
    /// Before the image's map is there the click waits for it; the map
    /// is on its way, since the brush asks for the shown image's map,
    /// and `pump_depth` runs the click when it lands.
    pub(crate) fn magic_depth_click(&mut self, item_id: u64, at: (f32, f32), dims: (u32, u32), mode: prunr_core::selection::BrushMode) {
        let Some(depth) = self.batch.find_by_id(item_id).and_then(|i| i.depth.clone()) else {
            self.magic_brush_state.pending_depth_click = Some(super::super::magic_brush_state::PendingDepthClick { item_id, at, dims, mode });
            return;
        };
        let (w, h) = dims;
        self.processor.dispatch_depth_select(crate::gui::processor::DepthSelectRequest {
            item_id,
            depth,
            at: ((at.0 + 0.5) / w as f32, (at.1 + 0.5) / h as f32),
            range: self.settings.brush.magic_depth_range,
            mode,
            source_dims: dims,
            reading: self.settings.brush.magic_reading(),
        });
    }

    /// Whether the selected model has brushes, so Magic Brush's models
    /// may be in memory.
    pub(crate) fn magic_brush_wanted(&self) -> bool {
        shows(self.settings.model, Surface::Brushes)
    }

    /// Whether Magic Brush's models load and encode ahead of a click: the
    /// model has brushes and the tool has been on this session. The
    /// encoder costs seconds of every core for each image looked at,
    /// which someone who never uses the brush should not pay.
    pub(crate) fn magic_brush_ready_ahead(&self) -> bool {
        self.magic_brush_wanted() && self.magic_brush_state.used()
    }

    /// After a model switch: a model without brushes turns them off and
    /// frees Magic Brush (its sessions and each image's 16 MB embedding),
    /// so nothing encodes behind an upscale; one with brushes warms it.
    pub(super) fn fit_brush_models_to_model(&mut self) {
        if self.magic_brush_wanted() {
            if self.magic_brush_ready_ahead() {
                self.processor.warm_sam_sessions();
            }
            // An eraser's only input is a painted region: the brush comes
            // on by itself, unless a tool is in use already.
            if self.settings.model.is_inpaint() && self.active_tool().is_none() {
                self.set_tool(Some(Tool::Paint));
            }
            return;
        }
        self.set_tool(None);
        self.processor.release_sam_sessions();
        self.batch.clear_magic_embeddings();
    }

    /// The selection overlay's style at the current zoom. The overlay and
    /// Delete / Copy / Cut read this one value, so the actions find the
    /// feathered plane the overlay built.
    pub(crate) fn selection_style(&self) -> crate::gui::background_io::SelectionStyle {
        crate::gui::background_io::SelectionStyle::from_settings(&self.settings).at_zoom(self.zoom_state.zoom)
    }

    pub(crate) fn magic_brush_preparing(&self) -> bool {
        let has_embedding = self.batch.selected_item().is_none_or(|i| i.magic_brush_embedding.is_some());
        self.magic_brush_state.preparing(has_embedding)
    }

    /// With the tool on, the user asked for it: say so and turn it off.
    /// With it off, the pre-encode was speculative: remember the refusal
    /// on the item so it is not retried every frame.
    fn on_encoder_failure(&mut self, item_id: u64, err: &str) {
        if self.magic_brush_state.is_active() {
            tracing::error!(item_id, %err, "SAM encoder failed");
            self.magic_brush_unavailable(err);
        } else {
            tracing::debug!(item_id, %err, "Magic Brush pre-encode skipped");
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.magic_preencode_refused = true;
            }
        }
    }

    /// Admission or encoder failure: tell the user and turn the tool off,
    /// which also stops the per-frame encoder check from retrying.
    fn magic_brush_unavailable(&mut self, err: &str) {
        self.toasts.error(format!("Magic Brush unavailable: {err}"));
        // The sessions stay warm and the selected image keeps its
        // embedding, so turning it back on is instant; background images
        // drop theirs on the next eviction pass.
        self.magic_brush_state.deactivate();
    }

    /// The tool the canvas's left button belongs to.
    pub(crate) fn active_tool(&self) -> Option<Tool> {
        if self.magic_brush_state.is_active() {
            Some(Tool::Magic)
        } else if self.brush_state.is_enabled() {
            Some(Tool::Paint)
        } else {
            None
        }
    }

    /// The one way a tool comes on or goes off, from any surface: the
    /// Paint and Magic brushes never share the canvas, so turning one on
    /// turns the other off. Returns whether anything changed.
    pub(crate) fn set_tool(&mut self, tool: Option<Tool>) -> bool {
        if self.active_tool() == tool {
            return false;
        }
        match tool {
            None => {
                self.brush_state.disable();
                self.magic_brush_state.deactivate();
            }
            Some(Tool::Paint) => {
                self.picking_depth_focus = false;
                self.magic_brush_state.deactivate();
                self.brush_state.enable();
            }
            Some(Tool::Magic) => {
                self.picking_depth_focus = false;
                self.brush_state.disable();
                self.magic_brush_state.activate();
            }
        }
        true
    }

    /// The Depth chip's "Set focus": the next click on the image focuses
    /// there, so a brush that would take that click goes off.
    pub(crate) fn begin_focus_pick(&mut self) {
        self.set_tool(None);
        self.picking_depth_focus = true;
    }

    /// Keep the selected image's embedding current, then drain SAM encoder
    /// and decoder results from the background rayon threads.
    /// Encoder results: write embedding to BatchItem, clear encoder_pending.
    /// Decoder results: convert to MaskArtifact, apply modifier, commit.
    pub(super) fn pump_sam_results(&mut self, ctx: &egui::Context) {
        use crate::gui::processor::DecodedSelection;
        let encoder_results = self.processor.pump_sam_encoder_results();
        for result in encoder_results {
            self.magic_brush_state.set_encoder_pending(false);
            match result.result {
                Ok(embedding) => {
                    if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                        item.magic_brush_embedding = Some(std::sync::Arc::new(embedding));
                        tracing::info!(item_id = result.item_id, "SAM encoder embedding cached");
                    }
                    ctx.request_repaint();
                }
                Err(err) => self.on_encoder_failure(result.item_id, &err),
            }
        }

        self.maybe_rethreshold_last_stroke();
        let decoder_results = self.processor.pump_sam_decoder_results();
        for result in decoder_results {
            // A selection is a step: one landing after a run started on the
            // image would make that run's result stale, so it is dropped.
            if result.result.is_ok() && self.batch.find_by_id(result.item_id).is_some_and(|i| !i.is_idle()) {
                tracing::info!(item_id = result.item_id, "Magic Brush selection dropped: a run started on the image");
                continue;
            }
            match result.result {
                Ok(DecodedSelection::Fresh { mask, output }) => {
                    tracing::info!(item_id = result.item_id, mode = ?result.mode, "SAM decoder mask queued");
                    let after = AfterLanding::MagicDecode { output, mode: result.mode, reading: result.reading };
                    self.queue_selection_op(result.item_id, SelectionOp::Merge(Arc::new(mask)), after);
                }
                Ok(DecodedSelection::ByDepth(mask)) => {
                    self.queue_selection_op(result.item_id, SelectionOp::Merge(Arc::new(mask)), AfterLanding::ByDepth);
                }
                Ok(DecodedSelection::Retuned { merged, hash, for_hash }) => {
                    self.magic_brush_state.rethreshold_in_flight = false;
                    self.apply_rethreshold(result.item_id, merged, hash, result.reading, for_hash);
                }
                Err(err) => {
                    tracing::error!(item_id = result.item_id, %err, "SAM decoder failed");
                    self.toasts.error(format!("Magic Brush decoder failed: {err}"));
                }
            }
        }
    }

    /// The selected image's last stroke still owns the selection and the
    /// Confidence knob moved: re-threshold its logits at the new value.
    pub(crate) fn maybe_rethreshold_last_stroke(&mut self) {
        if self.magic_brush_state.rethreshold_in_flight {
            return;
        }
        let want = self.settings.brush.magic_reading();
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &mut self.batch.items[idx];
        let Some((output, mode, reading, for_hash)) = item.retunable_decode().map(|ld| (Arc::clone(&ld.output), ld.mode, ld.reading, ld.committed_hash)) else { return };
        if reading == want {
            return;
        }
        // Asked for only once a retune runs: a handle the job unpacks, not
        // a decode on every frame.
        let (item_id, source_dims, base) = (item.id, item.dimensions, item.pre_stroke_selection());
        self.processor.dispatch_sam_rethreshold(crate::gui::processor::SamRethresholdRequest {
            item_id, output, mode,
            source_dims, reading: want, base, for_hash,
        });
        self.magic_brush_state.rethreshold_in_flight = true;
    }

    /// Swap the last stroke's selection for its re-thresholded version,
    /// replaced in place so the stroke's undo entry stays the one entry.
    /// Only while the stroke it retunes is still the last one: a retune
    /// for an older click lands on nothing.
    pub(crate) fn apply_rethreshold(&mut self, item_id: u64, merged: Arc<prunr_core::selection::MaskArtifact>, hash: u64, reading: prunr_core::MaskReading, for_hash: u64) {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        if !item.retunable_decode().is_some_and(|ld| ld.committed_hash == for_hash) {
            return;
        }
        let changed = item.replace_selection_in_place(merged, hash);
        if let Some(ld) = item.last_decode.as_mut() {
            ld.reading = reading;
            ld.committed_hash = hash;
        }
        if changed {
            self.apply_selection_to_active_model(item_id);
        }
    }
}


/// What a selection action makes of the selected pixels.
#[derive(Clone)]
enum SelectionEdit {
    Copy(crate::gui::system_bridge::CopyNote),
    Cut,
    CopyAndCut,
}

impl SelectionEdit {
    fn copy_note(&self) -> Option<crate::gui::system_bridge::CopyNote> {
        match self {
            Self::Copy(note) => Some(note.clone()),
            Self::CopyAndCut => Some(crate::gui::system_bridge::CopyNote::SelectionCut),
            Self::Cut => None,
        }
    }

    fn cuts(&self) -> bool {
        matches!(self, Self::Cut | Self::CopyAndCut)
    }
}

#[cfg(test)]
mod selection_action_tests {
    use std::sync::Arc;
    use prunr_core::selection::MaskArtifact;
    use crate::gui::item::{BatchItem, BatchStatus, ImageSource};

    fn make_mask(w: u32, h: u32, selected: bool) -> MaskArtifact {
        let v = if selected { prunr_core::selection::FULL } else { 0 };
        MaskArtifact::from_cells(w, h, vec![v; (w * h) as usize])
    }

    /// Let the selection edit started by the last action land, as the
    /// frame loop does: the job, then its copy's outcome.
    fn land_edit(app: &mut super::PrunrApp) {
        let ctx = egui::Context::default();
        let done = app.batch.bg_io.selection_edit_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("the edit lands");
        app.land_selection_edit(done, &ctx);
        for (note, took) in app.system.take_copy_outcomes() {
            app.on_copy_done(note, took);
        }
    }

    fn make_item_done() -> BatchItem {
        let mut item = BatchItem::new(
            42,
            "test.png".to_string(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (4, 4),
            Default::default(),
            String::new(),
        );
        // Give it a result image (4×4 all-white fully-opaque).
        let mut rgba = image::RgbaImage::new(4, 4);
        for px in rgba.pixels_mut() {
            *px = image::Rgba([255, 255, 255, 255]);
        }
        item.result_rgba = Some(Arc::new(rgba));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item
    }

    /// Delete on an image that has not been processed cuts the source and
    /// becomes the item's result; the stale result texture is dropped so
    /// the canvas rebuilds it.
    #[test]
    fn delete_on_unprocessed_item_cuts_the_source_into_a_result() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        let source = item.result_rgba.take().unwrap();
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Pending);
        item.source_rgba = Some(source);
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Delete, &egui::Context::default());
        land_edit(&mut app);

        let item = &app.batch.items[0];
        let result = item.result_rgba.as_ref().expect("Delete must produce a result from the source");
        assert!(result.pixels().all(|px| px.0[3] == 0), "selected region must be cut");
        assert_eq!(item.status(), BatchStatus::Done);
        assert!(item.result_tex_pending, "a texture build must be in flight for the new result");
        assert_eq!(item.timeline.undo_steps(), 1, "one Edit step, back to the unprocessed source");
    }

    /// Cut through the real action path: the result is cut like Delete and
    /// copy + cut is one undo step.
    #[test]
    fn cut_through_the_real_path_is_one_step() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Cut, &egui::Context::default());
        land_edit(&mut app);

        let item = &app.batch.items[0];
        assert!(item.result_rgba.as_ref().unwrap().pixels().all(|px| px.0[3] == 0));
        assert_eq!(item.timeline.undo_steps(), 1, "Cut is exactly one step");
    }

    /// Copy and Cut report with a success toast, on a plain result and on
    /// a cut-out.
    #[test]
    fn copy_and_cut_report_success() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        for cutout in [false, true] {
            for (action, text) in [
                (SelectionAction::Copy, "Selection copied to clipboard"),
                (SelectionAction::Cut, "Selection cut to clipboard"),
            ] {
                let mut app = super::PrunrApp::new_for_test();
                let mut item = make_item_done();
                item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
                if cutout {
                    crate::gui::tests::fixtures::cut_out(&mut item, prunr_core::ModelKind::Silueta);
                }
                app.batch.items.push(item);
                assert_eq!(app.batch.items[0].corrects_cutout(app.settings.model), cutout);
                app.system.accept_copies_for_test();
                app.handle_selection_action(0, action, &egui::Context::default());
                land_edit(&mut app);
                assert_eq!(app.toasts.last_for_test(), Some(("success", text)), "cut-out {cutout}");
            }
        }
    }

    /// A selection cut made off the UI thread lands only on the state it
    /// started from: a step taken meanwhile (an undo, a new stroke) keeps
    /// the image as it now is.
    #[test]
    fn a_selection_cut_for_a_state_left_behind_is_dropped() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        use crate::gui::timeline::StepKind;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        item.commit(StepKind::Process);
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);
        let before = app.batch.items[0].result_rgba.clone();
        app.handle_selection_action(0, SelectionAction::Delete, &egui::Context::default());
        app.batch.items[0].commit(StepKind::Adjust);
        land_edit(&mut app);
        let item = &app.batch.items[0];
        assert!(item.result_rgba.as_ref().zip(before.as_ref()).is_some_and(|(a, b)| Arc::ptr_eq(a, b)), "the image stays as it is");
        assert_eq!(item.timeline.undo_steps(), 2, "no Edit step from the stale cut");
    }

    /// Invert: coverage complement, signed by the active brush mode.
    #[test]
    fn invert_action_replaces_mask_with_inverse() {
        use prunr_core::selection::FULL;
        let original = make_mask(4, 4, true);
        let inverted = original.invert();
        assert!(inverted.cells().iter().all(|&v| v == 0), "inverted all-selected mask must be all-zero");
        let partial = MaskArtifact::from_cells(2, 2, vec![0, FULL, 0, FULL]);
        assert_eq!(partial.invert().cells(), &[-FULL, 0, -FULL, 0]);
    }
}

#[cfg(test)]
mod tool_mutual_exclusion_tests {
    use super::PrunrApp;
    use crate::gui::brush_state::Tool;
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;
    use crate::gui::views::adjustments_toolbar::ToolbarChange;

    fn app() -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        push_test_item(&mut app, 1);
        app.batch.select_item(0);
        app
    }

    fn on(app: &PrunrApp) -> (bool, bool) {
        (app.brush_state.is_enabled(), app.magic_brush_state.is_active())
    }

    /// Set focus takes the next click like a tool does: starting it turns
    /// a brush off, and a brush coming on ends the pick.
    #[test]
    fn set_focus_and_the_brushes_take_turns() {
        let mut app = app();
        app.set_tool(Some(Tool::Paint));
        app.begin_focus_pick();
        assert_eq!(on(&app), (false, false));
        assert!(app.picking_depth_focus);
        assert_eq!(crate::gui::automation::state::dump(&app).tool, "set_focus");
        app.set_tool(Some(Tool::Magic));
        assert!(!app.picking_depth_focus && on(&app) == (false, true));
        assert_eq!(crate::gui::automation::state::dump(&app).tool, "magic_brush");
    }

    /// Whichever way a tool comes on, the other goes off, and the on
    /// tool's own button turns it off.
    #[test]
    fn one_tool_at_a_time_whichever_way_it_comes_on() {
        let ctx = egui::Context::default();
        let mut app = app();
        let settings = app.batch.items[0].settings;
        app.set_tool(Some(Tool::Magic));
        assert_eq!((on(&app), app.active_tool()), ((false, true), Some(Tool::Magic)));

        app.apply_toolbar_change(&ctx, ToolbarChange { toggle_paint: true, ..Default::default() }, settings);
        assert_eq!((on(&app), app.active_tool()), ((true, false), Some(Tool::Paint)), "Paint over Magic");
        app.apply_toolbar_change(&ctx, ToolbarChange { toggle_magic: true, ..Default::default() }, settings);
        assert_eq!((on(&app), app.active_tool()), ((false, true), Some(Tool::Magic)), "Magic over Paint");
        app.apply_toolbar_change(&ctx, ToolbarChange { toggle_magic: true, ..Default::default() }, settings);
        assert_eq!((on(&app), app.active_tool()), ((false, false), None), "its own button turns it off");
        app.apply_toolbar_change(&ctx, ToolbarChange { toggle_paint: true, ..Default::default() }, settings);
        app.apply_toolbar_change(&ctx, ToolbarChange { toggle_paint: true, ..Default::default() }, settings);
        assert_eq!(app.active_tool(), None);
    }

    /// Switching to an eraser brings the brush on by itself, but never
    /// over a tool in use: that left both brushes on at once.
    #[test]
    fn an_eraser_brings_the_brush_on_unless_a_tool_is_in_use() {
        let mut app = app();
        app.settings.model = SettingsModel::Inpaint;
        app.fit_brush_models_to_model();
        assert_eq!(on(&app), (true, false));

        app.set_tool(Some(Tool::Magic));
        app.fit_brush_models_to_model();
        assert_eq!(on(&app), (false, true), "Magic Brush stays the tool");
    }

    /// Magic Brush loads and encodes ahead only once it has been used in
    /// the session: the encoder costs seconds of every core per image
    /// looked at, which someone who never uses the brush should not pay.
    #[test]
    fn magic_brush_works_ahead_only_after_its_first_use() {
        let mut app = app();
        app.settings.model = SettingsModel::Silueta;
        app.fit_brush_models_to_model();
        assert!(!app.magic_brush_ready_ahead(), "never used: nothing ahead");
        app.set_tool(Some(Tool::Magic));
        app.set_tool(None);
        assert!(app.magic_brush_ready_ahead(), "used once: each image shown is encoded ahead");
        app.settings.model = SettingsModel::RealEsrganUpscale;
        assert!(!app.magic_brush_ready_ahead(), "never under a model without brushes");
    }

    /// A model without brushes turns both off.
    #[test]
    fn a_model_without_brushes_turns_the_tools_off() {
        let mut app = app();
        app.set_tool(Some(Tool::Paint));
        app.settings.model = SettingsModel::RealEsrganUpscale;
        app.fit_brush_models_to_model();
        assert_eq!((on(&app), app.active_tool()), ((false, false), None));
    }
}
