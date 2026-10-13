//! The brushes and the selection: tool switching, selection actions, Magic Brush encode and decode.

use std::sync::Arc;

use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::timeline::StepKind;
use crate::gui::state::AppState;
use crate::gui::views::selection_action_bar::SelectionAction;
use crate::gui::brush_state::Tool;
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
        // Unconditional — see canvas::handle_brush_input.
        self.processor.live_preview.mark_tweak(item_id, PreviewKind::Mask);
        self.processor.live_preview.flush(item_id);
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
    /// Every call site that authors a selection (Paint Brush, Magic Brush,
    /// Invert, any future tool) calls THIS method.
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
                    if let Some((mask, base)) = self.selection_edit_inputs(idx) {
                        self.system.copy_image(&mask.copy_to_rgba(&base));
                        self.set_temporary_status("Selection cut to clipboard");
                    }
                    Some(CutoutAction::Erase)
                }
                SelectionAction::Copy | SelectionAction::Invert => None,
            };
            if let Some(fold) = fold {
                let style = self.selection_style();
                if self.batch.bake_selection(item_id, style, fold) {
                    self.apply_selection_to_active_model(item_id);
                }
                ctx.request_repaint();
                return;
            }
        }

        match action {
            // Only a cut-out offers it; handled above.
            SelectionAction::Restore => {}
            SelectionAction::Delete => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.apply_selection_cut(idx, &mask, &base, ctx);
            }
            SelectionAction::Copy => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.system.copy_image(&mask.copy_to_rgba(&base));
                self.set_temporary_status("Selection copied to clipboard");
            }
            SelectionAction::Cut => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.system.copy_image(&mask.copy_to_rgba(&base));
                self.apply_selection_cut(idx, &mask, &base, ctx);
                self.set_temporary_status("Selection cut to clipboard");
            }
            SelectionAction::Invert => {
                let Some(mask) = self.batch.items[idx].selection_mask.clone() else { return };
                let inverted = mask.invert();
                self.commit_selection_and_dispatch(item_id, inverted, StepKind::Selection);
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

    /// Selection actions act on the latest result or, before processing,
    /// on the source. Shared by the action bar, the keyboard shortcuts and
    /// the actions themselves.
    pub(crate) fn can_selection_action(&self) -> bool {
        shows(self.settings.model, Surface::Brushes)
            && self.batch.selected_item().is_some_and(|i| {
            i.selection_mask.is_some() && (i.result_rgba.is_some() || i.source_rgba.is_some())
        })
    }

    /// The selection plus the image a Delete / Copy / Cut acts on.
    fn selection_edit_inputs(
        &self,
        idx: usize,
    ) -> Option<(Arc<prunr_core::selection::MaskArtifact>, Arc<image::RgbaImage>)> {
        let item = &self.batch.items[idx];
        let mask = item.selection_mask.clone()?;
        let base = item.source_for_inpaint()?;
        // What gets cut is what the overlay shows: the feathered mask, which
        // the texture build already computed when it is current.
        let style = self.selection_style();
        let shown = item.current_selection_texture(style)
            .and_then(|t| t.feathered.clone())
            .or_else(|| style.feather(&mask, item.source_rgba.as_deref()))
            .unwrap_or(mask);
        Some((shown, base))
    }

    /// Cut the selection out of `base` and publish it as the item's result,
    /// one undo step.
    fn apply_selection_cut(
        &mut self,
        idx: usize,
        mask: &prunr_core::selection::MaskArtifact,
        base: &image::RgbaImage,
        ctx: &egui::Context,
    ) {
        let item_id = self.batch.items[idx].id;
        let mut edited = base.clone();
        mask.alpha_cut(&mut edited);
        self.publish_result(item_id, Arc::new(edited), "edit", ctx);
        self.batch.items[idx].commit(StepKind::Edit);
    }

    /// Eager encoder: the selected image gets an embedding as soon as it
    /// has a decoded source, so Magic Brush is ready the moment it is
    /// turned on. Runs from `reconcile_selected` every frame. With the tool
    /// off it waits for the warm sessions and an idle item, and a refusal
    /// is remembered instead of retried; with the tool on, admission or
    /// encoder failure turns the tool off, so this cannot retry in a loop.
    pub(super) fn ensure_magic_embedding_for_selected(&mut self, ctx: &egui::Context) {
        if !self.magic_brush_wanted() || self.magic_brush_state.has_pending_encoder() {
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
                || self.batch.app_state() == AppState::Processing)
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

    /// Whether the selected model has brushes, so Magic Brush's models
    /// belong in memory.
    pub(crate) fn magic_brush_wanted(&self) -> bool {
        shows(self.settings.model, Surface::Brushes)
    }

    /// After a model switch: a model without brushes turns them off and
    /// frees Magic Brush (its sessions and each image's 16 MB embedding),
    /// so nothing encodes behind an upscale; one with brushes warms it.
    pub(super) fn fit_brush_models_to_model(&mut self) {
        if self.magic_brush_wanted() {
            self.processor.warm_sam_sessions();
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
        self.deactivate_magic_brush();
    }

    /// Turn Magic Brush off. The sessions stay warm and the selected
    /// image keeps its embedding, so turning it back on is instant;
    /// background images drop theirs on the next eviction pass.
    pub(crate) fn deactivate_magic_brush(&mut self) {
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
        use crate::gui::processor::{merge_stroke, DecodedSelection};
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
            match result.result {
                Ok(DecodedSelection::Fresh { mask, output }) => {
                    let existing = self.batch
                        .find_by_id(result.item_id)
                        .and_then(|i| i.selection_mask.clone());
                    let final_mask = merge_stroke(existing, mask);
                    tracing::info!(item_id = result.item_id, mode = ?result.mode, "SAM decoder mask committed");
                    self.commit_selection_and_dispatch(result.item_id, final_mask, StepKind::Stroke);
                    if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                        item.last_decode = item.selection_hash.map(|committed_hash| crate::gui::item::LastDecode {
                            output, mode: result.mode, committed_hash,
                            reading: result.reading,
                        });
                    }
                }
                Ok(DecodedSelection::ByDepth(mask)) => {
                    let existing = self.batch
                        .find_by_id(result.item_id)
                        .and_then(|i| i.selection_mask.clone());
                    let final_mask = merge_stroke(existing, mask);
                    self.commit_selection_and_dispatch(result.item_id, final_mask, StepKind::Stroke);
                    if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                        item.last_decode = None;
                    }
                }
                Ok(DecodedSelection::Retuned { merged, hash }) => {
                    self.magic_brush_state.rethreshold_in_flight = false;
                    self.apply_rethreshold(result.item_id, merged, hash, result.reading);
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
        let (item_id, source_dims, base) = (item.id, item.dimensions, item.pre_stroke_selection());
        let Some(ld) = item.retunable_decode() else { return };
        if ld.reading == want {
            return;
        }
        self.processor.dispatch_sam_rethreshold(crate::gui::processor::SamRethresholdRequest {
            item_id, output: Arc::clone(&ld.output), mode: ld.mode,
            source_dims, reading: want, base,
        });
        self.magic_brush_state.rethreshold_in_flight = true;
    }

    /// Swap the last stroke's selection for its re-thresholded version,
    /// replaced in place so the stroke's undo entry stays the one entry.
    pub(crate) fn apply_rethreshold(&mut self, item_id: u64, merged: Arc<prunr_core::selection::MaskArtifact>, hash: u64, reading: prunr_core::MaskReading) {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        if item.retunable_decode().is_none() {
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

#[cfg(test)]
mod selection_action_tests {
    use std::sync::Arc;
    use prunr_core::selection::MaskArtifact;
    use crate::gui::item::{BatchItem, BatchStatus, ImageSource};

    fn make_mask(w: u32, h: u32, selected: bool) -> MaskArtifact {
        let v = if selected { prunr_core::selection::FULL } else { 0 };
        MaskArtifact::from_cells(w, h, vec![v; (w * h) as usize])
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
        item.status = BatchStatus::Done;
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
        item.status = BatchStatus::Pending;
        item.source_rgba = Some(source);
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Delete, &egui::Context::default());

        let item = &app.batch.items[0];
        let result = item.result_rgba.as_ref().expect("Delete must produce a result from the source");
        assert!(result.pixels().all(|px| px.0[3] == 0), "selected region must be cut");
        assert_eq!(item.status, BatchStatus::Done);
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

        let item = &app.batch.items[0];
        assert!(item.result_rgba.as_ref().unwrap().pixels().all(|px| px.0[3] == 0));
        assert_eq!(item.timeline.undo_steps(), 1, "Cut is exactly one step");
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
        app.set_tool(Some(Tool::Magic));
        assert!(!app.picking_depth_focus && on(&app) == (false, true));
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
