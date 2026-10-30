//! The adjustments toolbar: when it shows, rendering it, and routing its changes.

use crate::gui::item::Engine;
use crate::gui::timeline::StepKind;
use crate::gui::theme;
use crate::gui::views::adjustments_toolbar;
use crate::gui::brush_state::Tool;
use super::PrunrApp;

impl PrunrApp {
    /// Shift+H and an empty batch always hide it. `auto_hide_adjustments`
    /// hides it unless the cursor is in a peek zone near the top of the
    /// window or a popup (combo / color picker) is open.
    pub(super) fn adjustments_should_show(&self, ctx: &egui::Context) -> bool {
        if self.adjustments_hidden || self.batch.items.is_empty() {
            return false;
        }
        if !self.settings.auto_hide_adjustments {
            return true;
        }
        let screen_rect = ctx.content_rect();
        // Main toolbar + ~half the adjustments toolbar: catches the cursor
        // heading for the chips, not ordinary canvas work.
        let peek_zone = egui::Rect::from_min_size(
            screen_rect.min,
            egui::vec2(screen_rect.width(), theme::TOOLBAR_HEIGHT + 32.0),
        );
        let hover_in_peek = ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| peek_zone.contains(p)));
        hover_in_peek || egui::Popup::is_any_open(ctx)
    }

    pub(super) fn render_empty_model_bar(&mut self, ui: &mut egui::Ui, panel_frame: egui::Frame) {
        let installed = self.hardware_install_cache;
        let change = adjustments_panel(panel_frame)
            .show_inside(ui, |ui| adjustments_toolbar::render_model_only(ui, &mut self.settings, installed))
            .inner;
        if change.model_changed {
            self.on_model_changed();
        }
        if let Some(req) = change.open_model_store {
            self.open_model_store(req, ui.ctx());
        }
    }

    /// Open the Model Store; the flyout or popover that asked for it
    /// closes, or it stays drawn over the store.
    pub(crate) fn open_model_store(&mut self, req: adjustments_toolbar::ModelStoreRequest, ctx: &egui::Context) {
        egui::Popup::close_all(ctx);
        self.model_store = Some(req);
    }

    pub(super) fn render_adjustments_toolbar(&mut self, ui: &mut egui::Ui, panel_frame: egui::Frame) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let mut toolbar_change = adjustments_toolbar::ToolbarChange::default();
        // A run a model switch or a knob it reads would break: a cut queued
        // or under way on any image (the batch runs on the model), or the
        // upscale or colorize in flight. An erase took its model and
        // settings when it started.
        let is_processing = self.batch.any_awaiting_cut() || self.processor.is_run_in_flight();
        // Compared after the toolbar ran: a changed knob or preset is a step.
        let settings_before = self.batch.items[idx].settings;
        adjustments_panel(panel_frame)
            .show_inside(ui, |ui| {
                let can_compare = self.can_compare_intent();
                let selection_gates = crate::gui::views::selection_action_bar::SelectionGates::of(|a| self.selection_action_refusal(a));
                // Split borrow: disjoint fields, so the toolbar edits the
                // preset string in place without a clone + writeback.
                let settings_ref = &mut self.settings;
                let brush_state_ref = &mut self.brush_state;
                let item = &mut self.batch.items[idx];
                // Both brushes author the selection against the image shown,
                // so they need only its pixels; a cut-out correction waits
                // for the cut-out's own need (`apply_selection_to_active_model`).
                let brush_gate = item.verdict(crate::gui::capability::Need::ShownImage, settings_ref.model.role())
                    .reason().map(crate::gui::capability::Reason::text);
                let has_bg_image = item.bg_image.is_some();
                let bg_image_label = item.bg_image.as_deref()
                    .and_then(|bg| bg.source_path.as_deref())
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str());
                // Upscale row shows projected output size: the size of the
                // image the next upscale starts from.
                let source_dims = settings_ref.model.dispatch_model_kind()
                    .filter(|m| m.is_upscale())
                    .map_or_else(|| item.run_input_dims(Engine::Upscale), |m| item.next_upscale_dims(m));
                let state = adjustments_toolbar::ToolbarState {
                    magic_brush_active: self.magic_brush_state.is_active(),
                    brush_gate,
                    processing: is_processing,
                    has_bg_image,
                    bg_image_label,
                    source_dims,
                    has_selection: item.selection_mask.is_some(),
                    corrects_cutout: item.corrects_cutout(settings_ref.model),
                    selection_gates,
                    installed: self.hardware_install_cache,
                    show_original: self.show_original,
                    can_compare,
                    shown_seed: item.shown_seed(),
                    facts: crate::gui::capability::ItemFacts::of(item),
                    faces_found: Self::face_search(item, crate::gui::item::FaceStage::of(settings_ref.model.role())),
                    picking_depth_focus: self.picking_depth_focus,
                };
                toolbar_change = adjustments_toolbar::render(
                    ui,
                    &mut item.settings,
                    settings_ref,
                    crate::gui::views::preset_dropdown::PresetUi { applied: &mut item.applied_preset, save_dialog: &mut self.save_preset },
                    brush_state_ref,
                    state,
                );
            });
        self.apply_toolbar_change(ui.ctx(), toolbar_change, settings_before);
    }

    pub(super) fn apply_toolbar_change(
        &mut self,
        ctx: &egui::Context,
        toolbar_change: adjustments_toolbar::ToolbarChange,
        settings_before: crate::gui::item_settings::ItemSettings,
    ) {
        use crate::gui::knob_catalog::DispatchKind;
        use crate::gui::live_preview::PreviewKind;

        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        if toolbar_change.preset_applied {
            let item = &mut self.batch.items[idx];
            let shown = item.timeline.current();
            if item.settings != shown.settings || item.applied_preset != shown.applied_preset {
                item.commit(StepKind::Preset);
            }
            // The preset may carry a bg_image_hash: reload its bytes from the
            // persisted path map (or clear) so the recipe diff holds.
            self.reconcile_bg_image_after_preset(idx);
        }
        if toolbar_change.model_changed {
            self.on_model_changed();
        }
        if toolbar_change.brush_settings_committed || toolbar_change.faces.outline {
            self.settings.save();
        }
        if let Some(req) = toolbar_change.open_model_store {
            self.open_model_store(req, ctx);
        }
        if toolbar_change.pick_bg_image {
            self.handle_pick_bg_image(idx);
        } else if toolbar_change.clear_bg_image {
            self.batch.items[idx].clear_bg_image();
            ctx.request_repaint();
        }
        if let Some(action) = toolbar_change.selection_action.filter(|&a| self.can_selection_action(a)) {
            self.handle_selection_action(idx, action, ctx);
        }
        if toolbar_change.toggle_compare {
            self.show_original = !self.show_original;
        }
        if toolbar_change.toggle_paint {
            let next = (self.active_tool() != Some(Tool::Paint)).then_some(Tool::Paint);
            self.set_tool(next);
        }
        if toolbar_change.toggle_magic {
            let next = (self.active_tool() != Some(Tool::Magic)).then_some(Tool::Magic);
            if self.set_tool(next) && next.is_some() {
                self.ensure_magic_embedding_for_selected(ctx);
            }
        }

        self.batch.items[idx].apply_cache_impact(toolbar_change.cache_impact);
        if !toolbar_change.preset_applied && self.batch.items[idx].settings != settings_before {
            self.batch.items[idx].commit(StepKind::Adjust);
        }
        self.on_faces_change(idx, toolbar_change.faces);
        let role = self.settings.model.role();
        if self.batch.items[idx].settings.wants_depth(role) && !settings_before.wants_depth(role) {
            self.offer_depth_model(ctx);
        }
        if toolbar_change.pick_depth_focus {
            self.begin_focus_pick();
            // The Depth panel covers the middle of the image; the click
            // must land on the image, wherever the user wants the focus.
            crate::gui::theme::close_flyouts(ctx);
        }

        let dispatch = self.resolve_auto_dispatch(idx, &toolbar_change);
        let item_id = self.batch.items[idx].id;

        let mut queued = false;
        match dispatch {
            DispatchKind::None => {}
            DispatchKind::Render => {}
            // The queue's door refuses a render the model has none of.
            DispatchKind::LivePreviewMask => {
                let kind = PreviewKind::photo_rerun(self.batch.items[idx].settings.line_mode);
                self.processor.live_preview.mark_tweak(item_id, kind);
                queued = true;
            }
            DispatchKind::LivePreviewEdge => {
                self.processor.live_preview.mark_tweak(item_id, PreviewKind::Edge);
                queued = true;
            }
            DispatchKind::SubprocessAddEdge | DispatchKind::SubprocessFullPipeline => {
                let role = self.settings.model.role();
                if self.batch.items[idx].verdict(crate::gui::capability::Need::Processed, role).allows_request() {
                    self.process_items(|item| item.id == item_id, None);
                }
            }
        }

        // Refine's after-the-model knobs need no model, so they preview
        // whatever the Live preview setting. Checked only on a change (or
        // a slider's release), not every frame.
        let changed = self.batch.items[idx].settings != settings_before;
        if let Some(kind) = self.wants_kept_run_refinish(idx).filter(|_| changed || toolbar_change.commit) {
            self.processor.live_preview.mark_tweak(item_id, kind);
            queued = true;
        }
        // A release settles the drag's render at once, also when the
        // release frame left the value as it was and queued nothing.
        if toolbar_change.commit {
            self.processor.live_preview.flush(item_id);
            ctx.request_repaint();
        } else if queued {
            ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
        }

        if toolbar_change.render_repaint {
            // bg paints as a GPU rect behind the result texture; also repaints
            // a no-op preset apply for the commit toast.
            ctx.request_repaint();
        }
    }

    /// Starts from the catalog dispatch (only `auto_trigger_on_commit`
    /// knobs), refined by item state (cached tensors). A preset apply
    /// reduces to the recipe diff, so a no-op pick returns `None`.
    fn resolve_auto_dispatch(
        &self,
        idx: usize,
        tc: &adjustments_toolbar::ToolbarChange,
    ) -> crate::gui::knob_catalog::DispatchKind {
        use crate::gui::knob_catalog::{self, LineModeChange};
        use prunr_core::RequiredTier;
        let item = &self.batch.items[idx];
        let cached_seg = item.shows_cutout();
        let cached_edge = item.cached_edge_tensors.is_some();

        let mut dispatch = tc.auto_dispatch;

        if let Some(from) = tc.line_mode_from {
            let change = LineModeChange { from, to: item.settings.line_mode };
            dispatch = dispatch.max(knob_catalog::line_mode_spec(change, cached_edge).dispatch);
        }
        if tc.input_transform_changed {
            dispatch = dispatch.max(
                knob_catalog::input_transform_spec(cached_seg).dispatch,
            );
        }

        if tc.preset_applied {
            // Upscale and inpaint must diff against their own ModelKind, not
            // the seg-only `to_model_kind` fallback.
            let model = self
                .settings
                .model
                .dispatch_model_kind()
                .unwrap_or(super::NO_MODEL_RECIPE_KIND);
            let preset_dispatch = match &item.applied_recipe {
                None => knob_catalog::DispatchKind::SubprocessFullPipeline,
                Some(old) => {
                    let new = item.settings.current_recipe(model);
                    match prunr_core::resolve_tier(old, &new) {
                        // UpscaleRerun (Tier-1): user must re-click Process.
                        RequiredTier::Skip
                        | RequiredTier::CompositeOnly
                        | RequiredTier::UpscaleRerun => knob_catalog::DispatchKind::None,
                        // No DispatchKind for these: the caller routes them
                        // to the upscale live preview.
                        RequiredTier::UpscaleTier2 | RequiredTier::ColorizeTier2 => knob_catalog::DispatchKind::None,
                        RequiredTier::EdgeRerun => knob_catalog::DispatchKind::LivePreviewEdge,
                        RequiredTier::MaskRerun => knob_catalog::DispatchKind::LivePreviewMask,
                        RequiredTier::AddEdgeInference => {
                            knob_catalog::DispatchKind::SubprocessAddEdge
                        }
                        RequiredTier::FullPipeline => {
                            knob_catalog::DispatchKind::SubprocessFullPipeline
                        }
                    }
                }
            };
            dispatch = dispatch.max(preset_dispatch);
        }
        let previews_lines = matches!(dispatch, knob_catalog::DispatchKind::LivePreviewEdge | knob_catalog::DispatchKind::LivePreviewMask);
        if previews_lines && item.lines_need_a_run() {
            dispatch = dispatch.max(knob_catalog::DispatchKind::SubprocessAddEdge);
        }
        // A drag shows the lines where they were; its release traces them
        // on the new cut-out.
        if dispatch == knob_catalog::DispatchKind::LivePreviewMask && tc.commit && item.lines_traced_on_another_cut() {
            dispatch = knob_catalog::DispatchKind::SubprocessAddEdge;
        }
        dispatch
    }
}

/// The second toolbar row's panel, with or without an image loaded.
fn adjustments_panel(frame: egui::Frame) -> egui::Panel {
    let margin = egui::Margin::symmetric(theme::SPACE_SM as i8, theme::SPACE_SM as i8);
    egui::Panel::top("adjustments_toolbar")
        .exact_size(theme::CHIP_HEIGHT + theme::SPACE_SM * 2.0)
        .frame(egui::Frame { inner_margin: margin, ..frame })
}

#[cfg(test)]
mod release_tests {
    use super::PrunrApp;
    use crate::gui::live_preview::PreviewKind;
    use crate::gui::tests::fixtures::push_test_item;

    /// A slider's release settles the drag's owed render at once, also
    /// when the release frame left the value as it was.
    #[test]
    fn a_release_settles_the_drag_when_the_value_did_not_change() {
        let mut app = PrunrApp::new_for_test();
        push_test_item(&mut app, 1);
        app.batch.select_item(0);
        app.processor.live_preview.owe_for_test(1, PreviewKind::Mask);
        let before = app.batch.items[0].settings;
        let release = super::adjustments_toolbar::ToolbarChange { commit: true, ..Default::default() };
        app.apply_toolbar_change(&egui::Context::default(), release, before);
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(true));
    }

    /// A mask drag under lines on the cut-out previews them where they
    /// were; its release traces them on the new cut-out.
    #[test]
    fn a_mask_drags_release_traces_the_lines_again() {
        use crate::gui::knob_catalog::DispatchKind;
        use crate::gui::worker::{CompressedEdgeTensors, CompressedTensor, RawEdgeTensors, RawSegTensor};
        use prunr_core::{EdgeScale, LineMode, ModelKind::Silueta};
        let mut app = PrunrApp::new_for_test();
        let item = push_test_item(&mut app, 1);
        item.settings.line_mode = LineMode::SubjectOutline;
        item.start_cut(Some(crate::gui::timeline::StepKind::Process), crate::gui::item::Cutter::Model(Silueta));
        let tensors = EdgeScale::ALL.map(|s| {
            let (h, w) = prunr_core::edge_scale_dims(s, 2, 2);
            vec![0.5; (h * w) as usize]
        });
        let edges = CompressedEdgeTensors::from_raw(RawEdgeTensors { tensors, height: 2, width: 2, spare: None });
        let seg = CompressedTensor::from_raw(RawSegTensor { data: vec![0.5; 4], height: 2, width: 2, model: Silueta });
        let result = prunr_core::ProcessResult { rgba_image: image::RgbaImage::new(2, 2), active_provider: "CPU".into() };
        let recipe = item.settings.current_recipe(Silueta);
        item.apply_tier_result(Ok(result), seg, edges, recipe);
        item.finish_job();
        item.settings.feather = 3.0;
        let drag = super::adjustments_toolbar::ToolbarChange { auto_dispatch: DispatchKind::LivePreviewMask, ..Default::default() };
        assert_eq!(app.resolve_auto_dispatch(0, &drag), DispatchKind::LivePreviewMask, "a drag frame");
        let release = super::adjustments_toolbar::ToolbarChange { commit: true, ..drag };
        assert_eq!(app.resolve_auto_dispatch(0, &release), DispatchKind::SubprocessAddEdge, "the release");

        app.batch.select_item(0);
        app.processor.live_preview.owe_for_test(1, PreviewKind::Mask);
        let before = app.batch.items[0].settings;
        app.apply_toolbar_change(&egui::Context::default(), release, before);
        assert!(app.batch.items[0].job_run().is_some(), "the lines are traced again");
        assert!(!app.processor.live_preview.works_on(1), "no render of the drag lands over them");
    }

    /// A scale switch previews from the kept lines when they hold the
    /// scale or the worker is still making it, and runs the line detector
    /// again when they lack it.
    #[test]
    fn a_scale_the_kept_lines_lack_runs_the_lines_again() {
        use crate::gui::knob_catalog::DispatchKind;
        use crate::gui::worker::{CompressedEdgeTensors, RawEdgeTensors};
        use prunr_core::{EdgeScale, LineMode};
        let kept = |scales: &[EdgeScale], spare: Option<u64>| {
            let tensors = EdgeScale::ALL.map(|s| {
                let (h, w) = prunr_core::edge_scale_dims(s, 2, 2);
                if scales.contains(&s) { vec![0.5; (h * w) as usize] } else { Vec::new() }
            });
            CompressedEdgeTensors::from_raw(RawEdgeTensors { tensors, height: 2, width: 2, spare })
        };
        let switch = DispatchKind::LivePreviewEdge;
        let change = super::adjustments_toolbar::ToolbarChange { auto_dispatch: switch, ..Default::default() };
        let drawn = &[EdgeScale::Bold, EdgeScale::Fused][..];
        for (line_mode, held, spare, want) in [
            (LineMode::SubjectOutline, &EdgeScale::ALL[..], None, switch),
            (LineMode::SubjectOutline, drawn, None, DispatchKind::SubprocessAddEdge),
            (LineMode::SubjectOutline, drawn, Some(9), switch),
            (LineMode::EdgesOnly, drawn, None, DispatchKind::SubprocessAddEdge),
            (LineMode::EdgesOnly, &[EdgeScale::Fine][..], None, switch),
        ] {
            let mut app = PrunrApp::new_for_test();
            push_test_item(&mut app, 1);
            let item = &mut app.batch.items[0];
            item.settings.line_mode = line_mode;
            item.settings.edge_scale = EdgeScale::Fine;
            item.cached_edge_tensors = kept(held, spare);
            assert_eq!(app.resolve_auto_dispatch(0, &change), want, "{line_mode:?} holding {held:?}, spare {spare:?}");
        }
    }
}
