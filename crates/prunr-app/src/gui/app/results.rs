//! Where results land: worker events, finished runs, faces, depth maps and the background-channel drain.

use std::sync::Arc;

use crate::gui::item::{BatchItem, BatchStatus, Engine};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::background_io::TexTarget;
use crate::gui::timeline::StepKind;
use crate::gui::worker::WorkerResult;
use super::PrunrApp;

/// Why a run gives its image nothing.
#[derive(Clone, Copy)]
enum RunLost {
    /// A knob change, a stroke or an undo while the run was under way.
    ImageChanged,
    /// The batch ended and the worker never answered for the image.
    Unanswered,
}

/// Cap on `WorkerResult` messages drained per frame from `processor.worker_rx`.
/// Keeps the UI responsive during heavy batch processing — remaining messages
/// are picked up on the next frame (request_repaint_after ensures continuity).
const WORKER_POLL_PER_FRAME: usize = 8;

/// Cap on file-load receipts drained per frame from `batch.bg_io.file_load_rx`.
/// Same rationale as `WORKER_POLL_PER_FRAME`.
const FILE_LOAD_DRAIN_PER_FRAME: usize = 5;

impl PrunrApp {
    pub(super) fn pump_inpaint_results(&mut self, ctx: &egui::Context) {
        // Bridge → inpaint_rx fan-in must run before the drain.
        self.processor.pump_inpaint_subprocess();
        let (results, cancelled, errors) = self.processor.drain_inpaint_results();
        // A cancelled or failed erase records nothing; the painted region
        // stays for another try.
        for &item_id in cancelled.iter().chain(errors.iter().map(|(id, _)| id)) {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.abandon_job();
            }
        }
        if !cancelled.is_empty() || !errors.is_empty() {
            ctx.request_repaint();
        }
        for _ in &cancelled {
            self.toasts.info("Erase cancelled");
        }
        for (item_id, msg) in &errors {
            tracing::warn!(item_id, %msg, "erase failed");
            // The worker's message is user-facing already (the SD RAM
            // gate names the free and needed memory).
            self.toasts.error(msg.clone());
        }
        for crate::gui::processor::LandedFill { item_id, rgba, fill_scale } in results {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if !item.job_is_current() {
                self.end_stale_run(item_id, "erase");
                continue;
            }
            self.publish_result(item_id, Arc::new(rgba), "inpaint", ctx);
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                // The region is filled: the step shows the result without
                // its brush marks; undoing it brings both back.
                item.invalidate_selection();
                item.finish_job();
                item.note_fill_scale(fill_scale);
            }
        }
        // The slot is free for the next ticked image.
        self.dispatch_next_queued_run();
    }

    /// Install a new result the way every result producer must: the old
    /// texture stays on screen until the new one lands (a cleared texture
    /// paints nothing for a frame), the thumbnail is refreshed, and a
    /// Pending item becomes Done.
    pub(super) fn publish_result(
        &mut self,
        item_id: u64,
        rgba: Arc<image::RgbaImage>,
        tag: &str,
        ctx: &egui::Context,
    ) {
        let handles = self.batch.bg_io.tex_prep_handles();
        let switch = self.result_switch_id;
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        item.show_result(Arc::clone(&rgba));
        let source = item.source.clone();
        Self::spawn_tex_prep(
            rgba.clone(), item_id, Self::tex_name(tag, item_id, Some(switch)), TexTarget::Result, handles, ctx.clone(),
        );
        self.batch.request_thumbnail(item_id, &source, Some(&rgba));
        ctx.request_repaint();
    }

    /// Paint each finished upscale tile into its square of the item's
    /// preview texture, made on the run's first tile. Only the tile is
    /// uploaded (`set_partial`).
    pub(super) fn pump_upscale_previews(&mut self, ctx: &egui::Context) {
        let tiles = self.processor.pump_upscale_previews();
        if tiles.is_empty() {
            return;
        }
        for (item_id, tile) in tiles {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if !item.job_is_current() {
                continue;
            }
            let full = [tile.full_width as usize, tile.full_height as usize];
            let texture = match &mut item.tile_preview {
                Some(t) if t.size() == full => t,
                slot => slot.insert(ctx.load_texture(
                    format!("tile-preview-{item_id}"),
                    egui::ColorImage::new(full, vec![egui::Color32::TRANSPARENT; full[0] * full[1]]),
                    egui::TextureOptions::LINEAR,
                )),
            };
            let image = egui::ColorImage::from_rgba_unmultiplied([tile.width as usize, tile.height as usize], &tile.rgba);
            texture.set_partial([tile.x as usize, tile.y as usize], image, egui::TextureOptions::LINEAR);
        }
        ctx.request_repaint();
    }

    /// Drain completed upscale results and apply them to the matching item.
    /// A result for a state the item has left is dropped. `.try_recv()`
    /// keeps this non-blocking.
    pub(super) fn pump_upscale_results(&mut self, ctx: &egui::Context) {
        let results = self.processor.pump_upscale_results();
        if results.is_empty() {
            return;
        }
        for r in results {
            let item_id = r.item_id;
            match r.result {
                Ok(output) => {
                    if self.land_run(ctx, item_id, output.shown, r.recipe, "upscale", |item| item.land_upscale(output.run)) {
                        self.after_result_landed(item_id);
                        tracing::info!(item_id, "upscale result applied");
                    }
                }
                Err(err) => self.end_unlanded_run(item_id, "Upscale", err),
            }
        }
        self.dispatch_next_queued_run();
        ctx.request_repaint();
    }

    /// Drain finished colorize runs and land each on its item. Nothing
    /// goes on top: a colorize has no selection re-cut and no Restore faces.
    pub(super) fn pump_colorize_results(&mut self, ctx: &egui::Context) {
        let results = self.processor.pump_colorize_results();
        if results.is_empty() {
            return;
        }
        for r in results {
            let item_id = r.item_id;
            match r.result {
                Ok(output) => {
                    if self.land_run(ctx, item_id, output.shown, r.recipe, "colorize", |item| item.land_colorize(output.run)) {
                        tracing::info!(item_id, "colorize result applied");
                    }
                }
                Err(err) => self.end_unlanded_run(item_id, "Colorize", err),
            }
        }
        self.dispatch_next_queued_run();
        ctx.request_repaint();
    }

    /// Show a finished upscale or colorize on its item, unless the item
    /// has left the state the run started from; `keep` records the step
    /// and keeps the run. Returns whether it landed.
    fn land_run(
        &mut self,
        ctx: &egui::Context,
        item_id: u64,
        shown: Arc<image::RgbaImage>,
        recipe: prunr_core::ProcessingRecipe,
        tex_kind: &str,
        keep: impl FnOnce(&mut BatchItem),
    ) -> bool {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return false };
        if !item.job_is_current() {
            self.end_stale_run(item_id, tex_kind);
            return false;
        }
        item.show_run_result(Arc::clone(&shown), recipe);
        keep(item);
        let source = item.source.clone();
        Self::spawn_tex_prep(
            Arc::clone(&shown), item_id,
            Self::tex_name(tex_kind, item_id, Some(self.result_switch_id)),
            TexTarget::Result, self.batch.bg_io.tex_prep_handles(), ctx.clone(),
        );
        self.batch.request_thumbnail(item_id, &source, Some(&shown));
        true
    }

    /// A result arrived for a state the image has since left (a knob
    /// change, a stroke or an undo while the run was under way), so it
    /// cannot land: the image shows its current state again instead of
    /// waiting on, and the user hears why. Silent when nothing was awaited
    /// any more (an undo already showed the earlier state).
    fn end_stale_run(&mut self, item_id: u64, run: &str) {
        self.end_run_without_result(item_id, run, RunLost::ImageChanged);
    }

    /// The image stops waiting on a run that will give it nothing, and
    /// the user hears why.
    fn end_run_without_result(&mut self, item_id: u64, run: &str, why: RunLost) {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        if item.status != BatchStatus::Processing && item.job.is_none() {
            return;
        }
        let name = item.filename.clone();
        item.abandon_job();
        match why {
            RunLost::ImageChanged => {
                tracing::warn!(item_id, run, "result dropped: the image changed while the run was under way");
                self.toasts.warning(format!("{name} changed while it was processing, so that result was dropped. Run it again."));
            }
            RunLost::Unanswered => {
                tracing::warn!(item_id, run, "the batch ended without an answer for this image");
                self.toasts.warning(format!("{name} was not finished: the processing worker lost it. Run it again."));
            }
        }
        self.refresh_batch_progress_status();
        self.admission_release_and_admit(item_id);
    }

    /// An upscale or colorize ended without a result: its item stops
    /// waiting, and a toast says whether it was cancelled or failed.
    fn end_unlanded_run(&mut self, item_id: u64, run: &str, err: prunr_core::CoreError) {
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.job = None;
            item.tile_preview = None;
        }
        if matches!(err, prunr_core::CoreError::Cancelled) {
            tracing::info!(item_id, run, "run cancelled by user");
            self.toasts.info(format!("{run} cancelled"));
        } else {
            tracing::error!(item_id, run, %err, "run failed");
            self.toasts.error(format!("{run} failed: {err}"));
        }
    }

    pub(crate) fn pump_runtime_install(&mut self, ctx: &egui::Context) {
        use crate::runtime_install::InstallEvent;
        let Some(progress) = self.runtime_install.as_mut() else { return };
        while let Ok(event) = progress.rx.try_recv() {
            progress.last_event = event.clone();
            match event {
                InstallEvent::Done { .. } => {
                    let name = progress.runtime.display_name();
                    self.toasts.success(format!("{name} installed"));
                    self.runtime_install = None;
                    self.refresh_install_cache();
                    return;
                }
                InstallEvent::Failed { error } => {
                    let name = progress.runtime.display_name();
                    self.toasts.error(format!("{name} install failed: {error}"));
                    self.runtime_install = None;
                    self.refresh_install_cache();
                    return;
                }
                _ => {}
            }
        }
        ctx.request_repaint();
    }

    pub(super) fn pump_download_manager(&mut self, ctx: &egui::Context) {
        let events = self.download_manager.pump();
        if events.is_empty() {
            return;
        }
        for event in events {
            use crate::gui::download_manager::DownloadEvent;
            match event {
                DownloadEvent::Complete { id } => {
                    let name = prunr_models::descriptor(id)
                        .map_or("Model", |d| d.display_name);
                    self.toasts.success(format!("{name} ready"));
                    self.refresh_install_cache();
                }
                DownloadEvent::Failed { id, error } => {
                    let name = prunr_models::descriptor(id)
                        .map_or("Model", |d| d.display_name);
                    self.toasts.error(format!("{name} download failed: {error}"));
                    self.refresh_install_cache();
                }
                DownloadEvent::Progress { .. } | DownloadEvent::Verifying { .. } | DownloadEvent::Cancelled { .. } => {}
            }
        }
        ctx.request_repaint();
    }

    /// Keep faces found for the photo and model the image still wants, and
    /// blend them in; a search for anything else is dropped.
    fn land_faces(&mut self, landed: crate::gui::processor::FacesResult) {
        let Some(idx) = self.batch.index_of(landed.item_id) else { return };
        let item_id = landed.item_id;
        let stage = self.face_stage();
        match self.batch.items[idx].keep_faces(landed, stage) {
            None => return,
            Some(Ok(0)) => self.set_temporary_status("No faces found"),
            Some(Ok(_)) => {}
            Some(Err(err)) => {
                tracing::warn!(item_id, %err, "face restore failed");
                self.toasts.error(format!("Restore faces failed: {err}"));
            }
        }
        self.redo_faces_blend(idx);
    }

    /// The Faces chip changed: a blend change re-blends what was found; a
    /// search change re-blends now (or takes the faces out) and
    /// `pump_faces` finds them again where needed.
    pub(super) fn on_faces_change(&mut self, idx: usize, change: crate::gui::views::faces_chip::FacesChange) {
        let item = &mut self.batch.items[idx];
        if change.search && item.found_faces.as_ref().is_some_and(|f| f.failed) {
            item.found_faces = None;
        }
        if change.search || change.blend {
            self.batch.items[idx].faces_changed();
            // Before a result exists there is nothing to blend into, and a
            // re-cut would publish the plain photo as one.
            if self.shows_face_result(idx) {
                self.redo_faces_blend(idx);
            }
        }
    }

    /// What the Faces panel says about the search of `item`'s photo.
    pub(super) fn face_search(item: &BatchItem, stage: crate::gui::item::FaceStage) -> crate::gui::views::faces_chip::FoundCount {
        use crate::gui::views::faces_chip::FaceSearch;
        if item.faces_already_restored(stage) {
            return Some(FaceSearch::RestoredEarlier);
        }
        let found = item.found_for(&item.face_photo(stage)?, item.settings.faces.wanted()?)?;
        Some(match (found.stopped, found.failed) {
            (true, _) => FaceSearch::Stopped,
            (false, true) => FaceSearch::Failed,
            (false, false) => FaceSearch::Found(found.faces.len()),
        })
    }

    /// Restore faces works on the active model's stage.
    fn face_stage(&self) -> crate::gui::item::FaceStage {
        crate::gui::item::FaceStage::of(self.settings.model.role())
    }

    /// The image shows what Restore faces blends into: a finished result,
    /// or No model's photo, which is its own result.
    fn shows_face_result(&self, idx: usize) -> bool {
        let item = &self.batch.items[idx];
        let photo_is_result = self.settings.model.role() == crate::gui::knob_catalog::ModelRole::Filters;
        shows(self.settings.model, Surface::Faces) && (item.status == BatchStatus::Done || photo_is_result)
    }

    /// Blend the faces again: a Refine re-finish for a shown upscale, a
    /// re-cut otherwise.
    fn redo_faces_blend(&mut self, idx: usize) {
        if self.batch.items[idx].kept_upscale().is_some() {
            let id = self.batch.items[idx].id;
            self.processor.live_preview.mark_tweak(id, crate::gui::live_preview::PreviewKind::UpscaleTier2);
            self.processor.live_preview.flush(id);
        } else {
            self.dispatch_brush_rerun(idx);
        }
    }

    /// A run's result landed (a cut-out, No model's photo, an upscale):
    /// what the model can't know goes on top. The selection re-cuts it,
    /// and the faces found for its photo blend in.
    fn after_result_landed(&mut self, item_id: u64) {
        self.recut_with_selection(item_id);
        let Some(idx) = self.batch.index_of(item_id) else { return };
        if self.batch.items[idx].faces_input(self.face_stage()).is_some() {
            self.redo_faces_blend(idx);
        }
        // The run drew the lines unfaded and the photo without its look:
        // re-render them over its depth map, measured first when needed.
        if self.batch.items[idx].settings.wants_depth(self.settings.model.role()) {
            if self.batch.items[idx].depth.is_some() {
                self.after_depth_landed(idx);
            } else {
                self.request_depth(idx);
            }
        }
    }

    /// Whether the image at `idx` needs its depth map now: a depth knob on
    /// its settings, or Magic Brush selecting by depth on the shown image.
    fn wants_depth_now(&self, idx: usize) -> bool {
        let item = &self.batch.items[idx];
        let magic = self.magic_brush_state.is_active() && self.settings.brush.magic_depth && self.batch.is_selected(item.id);
        item.settings.wants_depth(self.settings.model.role()) || magic
    }

    /// Queue a depth run for the image at `idx` when it has no map, none
    /// is under way and the model is installed. Off the UI thread: the
    /// depth thread decodes the source itself when it is not decoded.
    fn request_depth(&mut self, idx: usize) {
        if !self.hardware_install_cache.depth {
            return;
        }
        let cpu_only = self.settings.cpu_only();
        let item = &mut self.batch.items[idx];
        if !item.needs_depth_run() {
            return;
        }
        let input = item.original_run_input();
        item.depth_run.pending = true;
        self.processor.dispatch_depth(item.id, item.depth_run.epoch, input, cpu_only);
    }

    /// Depth maps: keep what lands and re-render what reads it; measure
    /// the shown image's photo when something wants its map.
    pub(super) fn pump_depth(&mut self, ctx: &egui::Context) {
        for landed in self.processor.pump_depth_results() {
            let Some(idx) = self.batch.index_of(landed.item_id) else { continue };
            let error = landed.result.as_ref().err().cloned();
            let item = &mut self.batch.items[idx];
            let current = landed.epoch == item.depth_run.epoch;
            if item.land_depth(landed.epoch, landed.result) {
                tracing::info!(item_id = landed.item_id, "depth map landed");
                self.on_depth_landed(idx);
                ctx.request_repaint();
            } else if let Some(err) = error.filter(|_| current) {
                self.toasts.error(format!("Depth failed: {err}"));
            }
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        if self.wants_depth_now(idx) {
            self.request_depth(idx);
        }
    }

    /// The image's depth map landed: what reads it re-renders, and a By
    /// depth click that waited for it runs (turning the brush off drops one).
    pub(crate) fn on_depth_landed(&mut self, idx: usize) {
        self.after_depth_landed(idx);
        let item_id = self.batch.items[idx].id;
        if let Some(click) = self.magic_brush_state.pending_depth_click.take_if(|c| c.item_id == item_id) {
            self.magic_depth_click(click.item_id, click.at, click.dims, click.mode);
        }
    }

    /// The image's depth map is there: re-render the photo look and the
    /// faded lines that read it. A lens-blur background rebuilds itself,
    /// its texture key holding the map.
    fn after_depth_landed(&mut self, idx: usize) {
        use crate::gui::knob_catalog::ModelRole;
        let role = self.settings.model.role();
        let item = &self.batch.items[idx];
        let reads = (role == ModelRole::Filters && item.settings.depth.is_active()) || item.settings.line_fade() > 0.0;
        let has_result = item.status == BatchStatus::Done || role == ModelRole::Filters;
        if reads && has_result && shows(self.settings.model, Surface::Effect) {
            let (id, kind) = (item.id, crate::gui::live_preview::PreviewKind::photo_rerun(item.settings.line_mode));
            self.processor.live_preview.mark_tweak(id, kind);
            self.processor.live_preview.flush(id);
        }
    }

    /// A depth feature was turned on without the depth model: say so and
    /// open the Model Store on Depth, as other on-demand features do.
    pub(crate) fn offer_depth_model(&mut self, ctx: &egui::Context) {
        if self.hardware_install_cache.depth {
            return;
        }
        use crate::gui::views::get_model;
        let model = prunr_core::depth::MODEL.into();
        self.toasts.info(get_model::needs("Depth", model));
        self.open_model_store(get_model::store_request(model), ctx);
    }

    /// The next click on the image picks the Depth chip's focus.
    pub(crate) fn set_depth_focus(&mut self, fx: f32, fy: f32) {
        self.picking_depth_focus = false;
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &mut self.batch.items[idx];
        item.settings.depth.set_focus(fx, fy);
        item.commit(StepKind::Adjust);
        let (id, kind) = (item.id, crate::gui::live_preview::PreviewKind::photo_rerun(item.settings.line_mode));
        self.processor.live_preview.mark_tweak(id, kind);
        self.processor.live_preview.flush(id);
    }

    /// Restore faces: keep the shown image's faces found for its photo,
    /// and blend in the ones that land.
    pub(super) fn pump_faces(&mut self) {
        for landed in self.processor.pump_faces_results() {
            self.land_faces(landed);
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let installed = self.hardware_install_cache.has_face_model(item.settings.faces.model);
        if !item.settings.faces.on || !installed || !self.shows_face_result(idx) {
            return;
        }
        let Some(photo) = item.face_photo(self.face_stage()) else { return };
        if let Some(model) = item.faces_to_find(&photo) {
            let id = item.id;
            self.batch.items[idx].faces_pending = Some((photo.clone(), model));
            self.processor.dispatch_find_faces(id, photo, model);
        }
    }

    pub(super) fn poll_worker_results(&mut self, ctx: &egui::Context) {
        for _ in 0..WORKER_POLL_PER_FRAME {
            let Ok(msg) = self.processor.worker_rx.try_recv() else { break };
            match msg {
                WorkerResult::Report { item_id, update } => self.on_seg_report(item_id, update, ctx),
                WorkerResult::BatchItemDone { item_id, result, tensor_cache, edge_cache } => {
                    self.on_batch_item_done(ctx, item_id, result, tensor_cache, edge_cache);
                }
                WorkerResult::ChainStart { item_id, image } => {
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.chain_start_built(image);
                    }
                }
                WorkerResult::BatchComplete { batch } => self.on_batch_complete(batch),
                WorkerResult::Cancelled => self.on_cancelled(),
                WorkerResult::SubprocessRetry { reduced_jobs, re_queued_count } => {
                    self.on_subprocess_retry(reduced_jobs, re_queued_count);
                }
                WorkerResult::BackendReady(provider) => self.settings.note_backend(provider),
            }
        }
    }

    /// A background-removal step: the shared step words, except Loading
    /// the model, which names the models being loaded.
    fn on_seg_report(&mut self, item_id: u64, update: prunr_core::ProgressUpdate, ctx: &egui::Context) {
        if let prunr_core::ProgressUpdate::Subject(hint) = &update {
            self.show_subject(item_id, hint, ctx);
        } else if update == prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel) {
            let text = self.loading_model_status_text(item_id);
            self.processor.set_seg_step_text(text);
        } else {
            self.processor.apply_seg_report(update);
        }
    }

    /// Outline the subject a run has just found on the shown image, while
    /// its edges are refined. A 128 px hint costs well under a millisecond.
    fn show_subject(&mut self, item_id: u64, hint: &prunr_core::SubjectHint, ctx: &egui::Context) {
        if !self.settings.animations {
            return;
        }
        let awaited = self.batch.find_by_id(item_id).is_some_and(|i| i.status == BatchStatus::Processing && i.job_is_current());
        if !self.batch.is_selected(item_id) || !awaited {
            return;
        }
        let Some((image, order)) = crate::gui::views::effects::subject_glow(hint) else { return };
        let texture = ctx.load_texture(format!("subject-{item_id}"), image, egui::TextureOptions::LINEAR);
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.subject = Some(crate::gui::item::SubjectOutline { texture, shown_at: std::time::Instant::now(), order: order.into() });
        }
        ctx.request_repaint();
    }

    /// Include which models are loading so a TEED-only reload is
    /// distinguishable from a segmentation model load.
    fn loading_model_status_text(&self, item_id: u64) -> String {
        let line_mode = self.batch.find_by_id(item_id)
            .map(|b| b.settings.line_mode)
            .unwrap_or(prunr_core::LineMode::Off);
        let seg_name = crate::gui::views::model_name(self.settings.model);
        let models = match line_mode {
            prunr_core::LineMode::Off => seg_name.to_string(),
            prunr_core::LineMode::EdgesOnly => "TEED".to_string(),
            prunr_core::LineMode::SubjectOutline => format!("{seg_name} + TEED"),
        };
        if cfg!(target_os = "macos") {
            format!("Loading {models} (first run may take a few minutes)...")
        } else {
            format!("Loading {models}...")
        }
    }

    fn on_batch_item_done(
        &mut self,
        ctx: &egui::Context,
        item_id: u64,
        result: Result<prunr_core::ProcessResult, String>,
        tensor_cache: Option<crate::gui::worker::TensorCache>,
        edge_cache: Option<crate::gui::worker::EdgeTensorCache>,
    ) {
        let is_selected = self.batch.is_selected(item_id);
        let Some(recipe_snapshot) = self.take_dispatch_recipe(item_id) else { return };

        // A cancelled run leaves the item on the state it shows.
        if matches!(result.as_ref(), Err(e) if e == crate::subprocess::protocol::CANCELLED_ERR_MSG) {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                if item.status == BatchStatus::Processing {
                    item.abandon_job();
                }
            }
            self.admission_release_and_admit(item_id);
            self.refresh_batch_progress_status();
            return;
        }

        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        // A cancelled run's item already shows its state again.
        if item.status != BatchStatus::Processing {
            return;
        }
        if !item.job_is_current() {
            self.end_stale_run(item_id, "processing");
            return;
        }
        let succeeded = result.is_ok();
        if succeeded {
            item.hold_for_reveal();
        }
        let backend_update = item.apply_tier_result(result, tensor_cache, edge_cache, recipe_snapshot, is_selected);
        if succeeded {
            item.finish_job();
        } else {
            item.job = None;
            item.reveal = None;
            item.subject = None;
        }
        // Single completion event — useful for `RUST_LOG=prunr=debug` bug
        // reports (when did inference actually finish?) and as a stable
        // signal for any external observer (e.g. a smoke driver) that a
        // dispatch round-trip closed for this item.
        tracing::info!(item_id, status = ?item.status, "item processing complete");

        // A model the accelerator does not run used the CPU by design; the
        // app's accelerator, and the job count derived from it, stay.
        if let Some(provider) = backend_update.filter(|p| !p.is_empty() && *p != self.settings.active_backend) {
            let runs_on_accelerator = self.settings.model.to_model_id()
                .is_none_or(|m| prunr_core::engine::model_runs_on(m, &self.settings.active_backend));
            if runs_on_accelerator {
                self.settings.note_backend(provider);
            }
        }

        self.refresh_batch_progress_status();

        self.after_result_landed(item_id);

        if is_selected {
            self.result_switch_id += 1;
            self.sync_selected_batch_textures(ctx);
        }

        self.admission_release_and_admit(item_id);
        self.batch.enforce_tensor_budget();

        if crate::gui::memory::under_memory_pressure() {
            self.batch.demote_history_to_disk();
            self.batch.evict_all_tensors();
        }
    }

    /// Recipe to stamp on a finished item. Pulled from the in-flight slot
    /// owned by `Processor` — name is `take_*` because each call removes
    /// the entry. Late deliveries after `drain_recipes` fall back to
    /// reconstructing from current item state; returns `None` when the item
    /// is no longer in the batch (already removed by the user).
    fn take_dispatch_recipe(&mut self, item_id: u64) -> Option<prunr_core::ProcessingRecipe> {
        if let Some(recipe) = self.processor.take_recipe(item_id) {
            return Some(recipe);
        }
        let model: prunr_core::ModelKind = self.settings.model
            .to_model_kind()
            .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
        self.batch.find_by_id(item_id)
            .map(|b| b.settings.current_recipe(model))
    }

    fn refresh_batch_progress_status(&mut self) {
        let report = self.batch.progress();
        self.status.stage = report.stage;
        self.status.pct = report.pct;
    }

    /// `items` are the images the finished batch owned. One of them still
    /// marked as processing was never answered and must not wait on, with
    /// nothing under way; an image of a batch queued behind this one is
    /// left to its own batch.
    fn on_batch_complete(&mut self, batch: u64) {
        let never_answered = self.processor.finish_batch(batch);
        let unanswered: Vec<u64> = self.batch.items.iter()
            .filter(|i| never_answered.contains(&i.id) && i.status == BatchStatus::Processing && i.job.as_ref().is_some_and(|j| j.engine == Engine::Segment))
            .map(|i| i.id)
            .collect();
        for item_id in unanswered {
            self.end_run_without_result(item_id, "processing", RunLost::Unanswered);
        }
        let counts = self.batch.status_counts();
        let still_processing = counts.processing > 0;
        if counts.errored > 0 {
            let first_err = self.batch.first_error_message().unwrap_or("unknown error");
            let msg = if counts.errored == 1 {
                format!("Image failed: {first_err}")
            } else {
                format!("{} failed — first: {first_err}", crate::gui::views::fmt::count(counts.errored, "image", "images"))
            };
            self.status.text = msg.clone();
            self.toasts.warning(msg);
        } else if !still_processing {
            let msg = format!("All done \u{2014} {} processed", crate::gui::views::fmt::count(counts.done, "image", "images"));
            self.status.text = msg.clone();
            self.toasts.success(msg);
        }
    }

    fn on_cancelled(&mut self) {
        self.processor.drain_recipes();
        self.status.text = "Cancelled".to_string();
        self.processor.clear_admission();
    }

    fn on_subprocess_retry(&mut self, reduced_jobs: usize, re_queued_count: usize) {
        let msg = format!(
            "Memory pressure \u{2014} retrying {re_queued_count} images with {reduced_jobs} parallel jobs"
        );
        self.toasts.warning(msg.clone());
        self.status.text = msg;
    }

    /// Drain completed thumbnails from the background decoder into GPU
    /// textures. Throttles uploads so a 50-image import doesn't burst all
    /// `ctx.load_texture` calls into one frame and overwhelm the egui
    /// command buffer. Requests a repaint while the queue is still saturated.
    fn pump_thumbnail_results(&mut self, ctx: &egui::Context) -> bool {
        const THUMB_UPLOADS_PER_FRAME: usize = 3;
        let mut uploaded = 0;
        while uploaded < THUMB_UPLOADS_PER_FRAME {
            let Ok((item_id, tw, th, pixels)) = self.batch.bg_io.thumb_rx.try_recv() else { break };
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                let ci = egui::ColorImage::from_rgba_unmultiplied(
                    [tw as usize, th as usize],
                    &pixels,
                );
                item.thumb_texture = Some(ctx.load_texture(
                    format!("thumb_{item_id}"),
                    ci,
                    egui::TextureOptions::LINEAR,
                ));
                item.thumb_pending = false;
                uploaded += 1;
            }
        }
        if uploaded == THUMB_UPLOADS_PER_FRAME {
            ctx.request_repaint();
        }
        uploaded > 0
    }

    /// Returns whether anything landed on an image this frame.
    pub(super) fn drain_background_channels(&mut self, ctx: &egui::Context) -> bool {
        let mut landed = false;
        let mut decode_arrived = false;
        while let Ok((item_id, result)) = self.batch.bg_io.decode_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if item.land_decode(result) {
                decode_arrived = true;
            } else {
                // A render waiting on this photo would ask for it again
                // every frame, and fail again.
                self.processor.live_preview.cancel(item_id);
            }
        }
        landed |= decode_arrived;
        if decode_arrived {
            // Freshly-decoded RGBA for the viewed item: clear zoom state
            // entirely (not just re-arm the flag). A bare flag keeps
            // previous_zoom / pan_offset around, which lets `canvas.rs`'s
            // toggle-back branch fire against the old image's state.
            self.zoom_state.reset();
            self.show_original = false;
        }

        while let Ok((item_id, name, color_image, target)) = self.batch.bg_io.tex_prep_rx.try_recv() {
            let tex = ctx.load_texture(name, color_image, egui::TextureOptions::default());
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.land_texture(tex, target);
                landed = true;
            }
        }

        // Drain filter-only Process results (model=None path).
        let mut filter_only_arrived = false;
        let mut filtered = Vec::new();
        while let Ok((item_id, result, chain_start)) = self.batch.bg_io.filter_only_rx.try_recv() {
            filtered.push(item_id);
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if let Some(start) = chain_start {
                item.chain_start_built(start);
            }
            if !item.job_is_current() {
                self.end_stale_run(item_id, "filters");
                continue;
            }
            match result {
                Ok(rgba) => {
                    self.publish_result(item_id, rgba, "filter", ctx);
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.status = BatchStatus::Done;
                        item.finish_job();
                    }
                }
                Err(msg) => {
                    item.job = None;
                    item.status = BatchStatus::Error(msg);
                }
            }
            filter_only_arrived = true;
        }
        if filter_only_arrived {
            self.result_switch_id += 1;
            landed = true;
        }
        for item_id in filtered {
            self.after_result_landed(item_id);
        }

        // Drain files loaded by background thread (max 5 per frame to stay responsive)
        while let Ok((rgba, hint)) = self.batch.bg_io.copy_ready_rx.try_recv() {
            self.finish_copy(&rgba, hint);
        }
        // Drain save completion notifications
        while let Ok(msg) = self.batch.bg_io.save_done_rx.try_recv() {
            if msg.contains("fail") {
                self.toasts.error(msg);
            } else {
                self.toasts.success(msg);
            }
        }

        landed |= self.pump_thumbnail_results(ctx);
        self.batch.compress_history_off_thread();

        // Drain off-thread selection textures. `ctx.load_texture` is
        // allowed here — this drain runs from `logic()`, not a render
        // closure. A result for a mask that has since changed is dropped;
        // `ensure_selection_texture` already requested the current one.
        let mut feathered_landed = Vec::new();
        while let Ok(result) = self.batch.bg_io.selection_texture_rx.try_recv() {
            let item_id = result.item_id;
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            let load = |image| ctx.load_texture(format!("selection_{item_id}"), image, egui::TextureOptions::LINEAR);
            if item.land_selection_texture(result, load) {
                feathered_landed.push(item_id);
            }
        }

        // A feathered selection that just landed changes what the cut-out
        // applies; re-cut with it.
        for id in feathered_landed {
            self.recut_with_selection(id);
        }

        while let Ok(load) = self.batch.bg_io.bg_image_rx.try_recv() {
            self.land_bg_image(load);
            landed = true;
            ctx.request_repaint();
        }

        // A picture that is no longer the item's backdrop (changed while it
        // was built) is dropped; the next kick builds the current one.
        while let Ok((item_id, key, ci)) = self.batch.bg_io.bg_tex_prep_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if item.land_backdrop(key, ci, ctx) {
                landed = true;
            } else {
                ctx.request_repaint();
            }
        }

        let id_floor = self.batch.next_id;
        let mut loaded_count = 0u32;
        let mut channel_drained = false;
        for _ in 0..FILE_LOAD_DRAIN_PER_FRAME {
            match self.batch.bg_io.file_load_rx.try_recv() {
                Ok((path, name)) => {
                    self.add_to_batch_path(path, name);
                    loaded_count += 1;
                }
                Err(_) => { channel_drained = true; break; }
            }
        }
        if loaded_count > 0 {
            ctx.request_repaint();
            // Select the new image if only one was loaded and no more are pending
            if loaded_count == 1 && channel_drained {
                self.batch.selected_index = self.batch.items.len() - 1;
                self.zoom_state.reset();
                self.sync_selected_batch_textures(ctx);
            }
            self.auto_process_imported(id_floor);
        }
        landed
    }
}

#[cfg(test)]
mod faces_tests {
    use std::sync::Arc;
    use crate::gui::item::{FacePhoto, FoundFaces};
    use crate::gui::item_settings::FaceModel;
    use crate::gui::processor::FacesResult;
    use crate::gui::tests::fixtures::push_test_item;

    /// An item showing a cut-out of its source, Restore faces on.
    fn app_with_faces_on() -> (super::PrunrApp, Arc<image::RgbaImage>) {
        let mut app = super::PrunrApp::new_for_test();
        let source = Arc::new(image::RgbaImage::new(8, 8));
        let item = push_test_item(&mut app, 1);
        item.source_rgba = Some(Arc::clone(&source));
        item.settings.faces.on = true;
        (app, source)
    }

    fn landed(photo: &Arc<image::RgbaImage>, model: FaceModel) -> FacesResult {
        FacesResult { item_id: 1, photo: FacePhoto::Source(Arc::clone(photo)), model: model.id(), result: Ok(Vec::new()), stopped: false }
    }

    /// Faces count only for the photo and model they were found with:
    /// another model or photo needs a search, and off needs nothing.
    #[test]
    fn found_faces_belong_to_one_photo_and_model() {
        let (mut app, source) = app_with_faces_on();
        let item = &mut app.batch.items[0];
        let photo = item.face_photo(crate::gui::item::FaceStage::Photo).expect("the source");
        assert_eq!(item.faces_to_find(&photo), Some(FaceModel::Gfpgan.id()));
        item.found_faces = Some(FoundFaces { photo: FacePhoto::Source(Arc::clone(&source)), found_at: std::time::Instant::now(), model: FaceModel::Gfpgan.id(), faces: Arc::default(), failed: false, stopped: false });
        assert!(item.faces_for(&photo).is_some() && item.faces_to_find(&photo).is_none());
        item.settings.faces.model = FaceModel::RestoreFormerPp;
        assert!(item.faces_for(&photo).is_none());
        assert_eq!(item.faces_to_find(&photo), Some(FaceModel::RestoreFormerPp.id()));
        let other = FacePhoto::Source(Arc::new(image::RgbaImage::new(8, 8)));
        assert!(item.faces_for(&other).is_none(), "another photo");
        item.settings.faces.on = false;
        assert!(item.faces_to_find(&photo).is_none() && item.faces_for(&photo).is_none());
    }

    /// Faces that land for the photo shown are kept and blended in; a
    /// search for a model no longer chosen is dropped.
    #[test]
    fn landed_faces_are_kept_only_when_still_wanted() {
        let (mut app, source) = app_with_faces_on();
        app.land_faces(landed(&source, FaceModel::RestoreFormerPp));
        assert!(app.batch.items[0].found_faces.is_none(), "not the chosen model");
        app.land_faces(landed(&source, FaceModel::Gfpgan));
        assert!(app.batch.items[0].found_faces.is_some());
        assert!(app.processor.live_preview.is_pending_for(1), "the re-blend is queued");
    }

    /// Faces are restored once per chain: a run starting from an image
    /// that already has them (a faced cut-out, an erase on one) leaves
    /// them alone; redoing the faced step itself restores them again.
    #[test]
    fn faces_are_not_restored_over_restored_faces() {
        use crate::gui::item::{Engine, FaceStage};
        use crate::gui::timeline::StepKind;
        let (mut app, _) = app_with_faces_on();
        let item = &mut app.batch.items[0];
        item.status = crate::gui::item::BatchStatus::Done;
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        item.commit(StepKind::Process);
        item.record_origin(Engine::Segment, None, None);
        item.settle_faced(Engine::Segment, true);
        assert!(item.face_photo(FaceStage::Photo).is_some(), "redoing the faced step starts from the photo");
        assert!(!item.faces_already_restored(FaceStage::Photo));

        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        item.commit(StepKind::Edit);
        assert!(item.faces_already_restored(FaceStage::Photo), "the next cut-out starts from restored faces");
        assert!(item.face_photo(FaceStage::Photo).is_none());
        assert!(item.faces_input(FaceStage::Photo).is_none());

        item.undo_step();
        assert!(!item.faces_already_restored(FaceStage::Photo), "back on the faced step: its redo restores again");
    }

    /// A failed search is not repeated on its own (it would fail again
    /// every frame); changing On / Off or the model tries again.
    #[test]
    fn a_failed_search_is_retried_on_a_search_change() {
        let (mut app, source) = app_with_faces_on();
        let failed = FacesResult { result: Err("no memory".into()), ..landed(&source, FaceModel::Gfpgan) };
        app.land_faces(failed);
        let photo = app.batch.items[0].face_photo(crate::gui::item::FaceStage::Photo).expect("the source");
        assert!(app.batch.items[0].faces_to_find(&photo).is_none(), "not searched again every frame");
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { blend: true, ..Default::default() });
        assert!(app.batch.items[0].faces_to_find(&photo).is_none(), "a blend change does not search");
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { search: true, ..Default::default() });
        assert_eq!(app.batch.items[0].faces_to_find(&photo), Some(FaceModel::Gfpgan.id()), "tried again");
    }

    /// A stopped search says so and is not started again on its own;
    /// turning Faces on again (a search change) searches.
    #[test]
    fn a_stopped_search_waits_to_be_asked_again() {
        use crate::gui::item::FaceStage;
        use crate::gui::views::faces_chip::{FaceSearch, FacesChange};
        let (mut app, source) = app_with_faces_on();
        app.land_faces(FacesResult { result: Err("cancelled".into()), stopped: true, ..landed(&source, FaceModel::Gfpgan) });
        let photo = app.batch.items[0].face_photo(FaceStage::Photo).expect("the source");
        assert!(app.batch.items[0].faces_to_find(&photo).is_none(), "not restarted on the next frame");
        assert!(matches!(super::PrunrApp::face_search(&app.batch.items[0], FaceStage::Photo), Some(FaceSearch::Stopped)));
        app.on_faces_change(0, FacesChange { search: true, ..Default::default() });
        assert_eq!(app.batch.items[0].faces_to_find(&photo), Some(FaceModel::Gfpgan.id()), "searched again when asked");
    }

    /// A blend change drops the faced photo built for the old blend and
    /// re-blends.
    #[test]
    fn a_blend_change_reblends() {
        let (mut app, _) = app_with_faces_on();
        app.batch.items[0].status = crate::gui::item::BatchStatus::Done;
        let faced = Arc::new(image::DynamicImage::new_rgba8(8, 8));
        let key = crate::gui::live_preview::FacedKey { faces: Arc::default(), blend: Default::default() };
        app.batch.items[0].faced_photo = Some((faced, key));
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { blend: true, ..Default::default() });
        assert!(app.batch.items[0].faced_photo.is_none());
        assert!(app.processor.live_preview.is_pending_for(1));
    }

    /// Before an image is processed a Faces change has nothing to blend
    /// into; a re-cut would publish the plain photo as the result.
    #[test]
    fn a_faces_change_before_processing_publishes_nothing() {
        let (mut app, _) = app_with_faces_on();
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { search: true, ..Default::default() });
        assert!(!app.processor.live_preview.is_pending_for(1));
    }

    /// Undo / redo can bring back other face settings: the faced photo and
    /// the cut-out over it are dropped, not reused.
    #[test]
    fn undo_drops_what_was_built_with_other_face_settings() {
        let (mut app, _) = app_with_faces_on();
        let item = &mut app.batch.items[0];
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        item.commit(crate::gui::timeline::StepKind::Process);
        item.settings.faces.on = false;
        item.commit(crate::gui::timeline::StepKind::Adjust);
        item.cached_masked_base = Some((Arc::new(image::RgbaImage::new(8, 8)), prunr_core::MaskRecipe::from(&item.settings.mask_settings()), prunr_core::ModelKind::Silueta));
        item.undo_step();
        assert!(item.cached_masked_base.is_none());
    }

    /// An image not on screen drops its faced photo; its faces stay.
    #[test]
    fn images_not_on_screen_drop_the_faced_photo() {
        let (mut app, source) = app_with_faces_on();
        push_test_item(&mut app, 2);
        let found = FoundFaces { photo: FacePhoto::Source(source), found_at: std::time::Instant::now(), model: FaceModel::Gfpgan.id(), faces: Arc::default(), failed: false, stopped: false };
        let key = crate::gui::live_preview::FacedKey { faces: Arc::default(), blend: Default::default() };
        let item = &mut app.batch.items[0];
        item.found_faces = Some(found);
        item.faced_photo = Some((Arc::new(image::DynamicImage::new_rgba8(8, 8)), key));
        app.batch.select_item(1);
        app.evict_background_item_caches(1);
        assert!(app.batch.items[0].faced_photo.is_none());
        assert!(app.batch.items[0].found_faces.is_some());
    }

    /// An image not on screen lets go of its unpacked undo / redo steps
    /// (up to two images each across a whole batch).
    #[test]
    fn images_not_on_screen_release_their_warm_steps() {
        let mut app = super::PrunrApp::new_for_test();
        for id in [1, 2] {
            let item = push_test_item(&mut app, id);
            item.on_screen = true;
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            item.commit(crate::gui::timeline::StepKind::Process);
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            item.commit(crate::gui::timeline::StepKind::Adjust);
        }
        let held = |app: &super::PrunrApp, i: usize| app.batch.items[i].warm_steps.lock().unwrap().iter().any(Option::is_some);
        assert!(held(&app, 0) && held(&app, 1));
        app.evict_background_item_caches(1);
        assert!(!held(&app, 0), "off screen: released");
        assert!(held(&app, 1), "the shown image keeps its own");
    }
}

/// Every kind of step on the one timeline: what it records, what undo
/// brings back, and that a result computed for a state left behind never
/// lands.
#[cfg(test)]
mod timeline_contract_tests {
    use std::sync::Arc;
    use super::PrunrApp;
    use crate::gui::item::{BatchStatus, Engine};
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;
    use crate::gui::timeline::StepKind;

    fn px(v: u8) -> Arc<image::RgbaImage> {
        Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([v, v, v, 255])))
    }

    fn shown(app: &PrunrApp) -> Option<u8> {
        app.batch.items[0].result_rgba.as_ref().map(|r| r.get_pixel(0, 0)[0])
    }

    fn mask() -> prunr_core::selection::MaskArtifact {
        let mut cells = vec![0i8; 16];
        cells[5] = prunr_core::selection::FULL;
        prunr_core::selection::MaskArtifact::from_cells(4, 4, cells)
    }

    fn app_with_processed_item(model: SettingsModel) -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        app.settings.model = model;
        let item = push_test_item(&mut app, 1);
        item.dimensions = (4, 4);
        item.source_rgba = Some(px(0));
        item.result_rgba = Some(px(1));
        item.status = BatchStatus::Done;
        item.commit(StepKind::Process);
        app.batch.select_item(0);
        app
    }

    /// A saved result is named after its source, with the format's
    /// extension.
    #[test]
    fn a_saved_result_is_named_after_its_source_and_format() {
        use prunr_core::ExportFormat;
        assert_eq!(crate::gui::drag_export::export_name("beach.jpeg", ExportFormat::Png), "beach.prunr.png");
        assert_eq!(crate::gui::drag_export::export_name("beach.jpeg", ExportFormat::Jpeg), "beach.prunr.jpg");
        assert_eq!(crate::gui::drag_export::export_name("noext", ExportFormat::Webp), "noext.prunr.webp");
    }

    /// Cmd+S works whenever the Save button shows: a ticked finished
    /// image is enough even while the shown one is still unprocessed.
    #[test]
    fn save_is_possible_with_a_ticked_finished_image() {
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        push_test_item(&mut app, 2);
        app.batch.select_item(1);
        app.batch.items[0].selected = true;
        assert!(app.can_save_intent(), "the ticked image is done");
        app.batch.items[0].selected = false;
        assert!(!app.can_save_intent(), "nothing finished to save");
    }

    /// Auto-process on import runs background removal (or the No-model
    /// filter) on new images; an eraser needs a painted region and an
    /// upscale runs on the image you pick, so those wait for Process.
    #[test]
    fn auto_process_on_import_runs_only_for_cut_out_and_filter_models() {
        for model in [SettingsModel::SdInpaint, SettingsModel::Inpaint, SettingsModel::RealEsrganUpscale, SettingsModel::ColorizeDdcolor, SettingsModel::BiRefNetLite, SettingsModel::None] {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = model;
            app.settings.auto_process_on_import = true;
            push_test_item(&mut app, 5);
            app.auto_process_imported(5);
            let ran = matches!(app.batch.items[0].status, BatchStatus::Processing);
            assert_eq!(ran, matches!(model, SettingsModel::BiRefNetLite | SettingsModel::None), "{model:?}");
        }
    }

    /// A batch runs with the selected image's settings, so the No-model
    /// filter path is chosen from those: another image's lines must not
    /// send the whole batch to a cut-out model.
    #[test]
    fn no_model_with_no_lines_selected_runs_the_filter_for_the_batch() {
        let mut app = app_with_processed_item(SettingsModel::None);
        push_test_item(&mut app, 2).settings.line_mode = prunr_core::LineMode::EdgesOnly;
        app.batch.select_item(0);
        app.process_items(|_| true, Some(StepKind::Process));
        assert!(app.batch.items.iter().all(|i| matches!(i.status, BatchStatus::Processing)), "both run as filters");
        assert_eq!(app.batch.items[1].settings.line_mode, prunr_core::LineMode::Off, "and show the settings they ran with");
    }

    /// A picked background picture replaces the background as one step;
    /// a pick that fails to load leaves the old one; a preset's picture
    /// lands only while the settings still name it.
    #[test]
    fn a_background_picture_lands_as_one_step_or_not_at_all() {
        use crate::gui::background_io::BgImageLoad;
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        app.batch.items[0].settings.bg = Some([9, 9, 9, 255]);
        let load = |picked, result| BgImageLoad { item_id: 1, path: "bg.png".into(), picked, result };

        let steps = app.batch.items[0].timeline.undo_steps();
        app.land_bg_image(load(true, Err("unreadable".into())));
        assert_eq!(app.batch.items[0].settings.bg, Some([9, 9, 9, 255]), "a failed pick keeps the colour");
        assert_eq!(app.batch.items[0].timeline.undo_steps(), steps);

        app.land_bg_image(load(true, Ok((image::DynamicImage::new_rgba8(2, 2), 42))));
        let item = &app.batch.items[0];
        assert_eq!(item.settings.backdrop(), crate::gui::item_settings::Backdrop::Image(std::num::NonZeroU64::new(42).unwrap()));
        assert_eq!(item.settings.bg, None, "the picture replaces the colour");
        assert_eq!(item.timeline.undo_steps(), steps + 1, "as one step");

        app.land_bg_image(load(false, Ok((image::DynamicImage::new_rgba8(3, 3), 77))));
        assert_eq!(app.batch.items[0].bg_image.as_ref().map(|b| b.hash), Some(42), "a picture no setting names is dropped");
    }

    /// Refine's after-the-model knobs re-finish the kept upscale; a Scale
    /// change needs a new run.
    #[test]
    fn refine_knobs_refinish_the_upscale() {
        use crate::gui::item::UpscaleCache;
        let mut app = app_with_processed_item(SettingsModel::RealEsrganUpscale);
        let item = &mut app.batch.items[0];
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus));
        item.settings.sharpen = 0.4;
        assert!(!app.wants_upscale_tier2(0), "no kept run yet");

        let img = Arc::new(image::RgbaImage::new(4, 4));
        app.batch.items[0].start_job(Some(StepKind::Upscale), Engine::Upscale);
        app.batch.items[0].land_upscale(UpscaleCache::new(img.clone(), img, None));
        assert!(app.wants_upscale_tier2(0), "a Sharpen change re-finishes the kept run");
        app.batch.items[0].settings.output_scale = prunr_core::OutputScale::X2;
        assert!(!app.wants_upscale_tier2(0), "a Scale change needs a new run");
    }

    /// A Refine preview re-finishes the kept upscale run: it needs neither
    /// the decoded photo nor a copy of it.
    #[test]
    fn a_refine_preview_needs_only_the_kept_upscale() {
        use crate::gui::item::UpscaleCache;
        use crate::gui::live_preview::PreviewKind;
        let mut app = app_with_processed_item(SettingsModel::RealEsrganUpscale);
        let item = &mut app.batch.items[0];
        item.source_rgba = None;
        let img = Arc::new(image::RgbaImage::new(4, 4));
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.land_upscale(UpscaleCache::new(img.clone(), img, None));
        let style = app.selection_style();
        let inputs = PrunrApp::build_preview_inputs(&mut app.batch.items, 1, PreviewKind::UpscaleTier2, false, style, app.settings.brush.cutout_action);
        assert!(inputs.is_some_and(|i| i.upscale.is_some()), "dispatched from the kept run");
        assert!(app.batch.items[0].source_dyn.is_none() && app.batch.items[0].start_dyn_cache.is_none(), "without decoding or copying the photo");
    }

    /// A model without brushes loads none of Magic Brush: switching to an
    /// upscaler drops every image's embedding and the encoder stays off.
    #[test]
    fn an_upscale_model_keeps_magic_brush_unloaded() {
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        app.batch.items[0].magic_brush_embedding = Some(Arc::new(prunr_core::sam::SamEmbedding {
            image_embed: vec![0.0; 4],
            high_res_feats_0: vec![0.0; 4],
            high_res_feats_1: vec![0.0; 4],
        }));
        app.settings.model = SettingsModel::RealEsrganUpscale;
        app.fit_brush_models_to_model();
        assert!(app.batch.items[0].magic_brush_embedding.is_none(), "its embedding is freed");
        assert!(!app.magic_brush_wanted(), "and the encoder stays off");
        app.settings.model = SettingsModel::Silueta;
        assert!(app.magic_brush_wanted());
    }

    /// A Full-lines run draws on the whole photo: a Pre-filter change must
    /// run the lines again, never compose them onto a cut-out left in the
    /// cache by an earlier Off run.
    #[test]
    fn a_full_lines_rerun_never_composes_onto_an_old_cutout() {
        use crate::gui::worker::{CompressedTensor, TensorCache};
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        let item = &mut app.batch.items[0];
        item.record_origin(Engine::Segment, None, None);
        item.set_cached_tensor(CompressedTensor::from_raw(TensorCache { data: vec![0.5; 16], height: 4, width: 4, model: prunr_core::ModelKind::Silueta }));
        item.settings.line_mode = prunr_core::LineMode::EdgesOnly;
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
        item.settings.input_transform = prunr_core::InputTransform::Grayscale;
        let ids = std::collections::HashSet::from([1]);
        let tiers = app.batch.classify(&ids, prunr_core::ModelKind::Silueta, Some(StepKind::Process), crate::gui::item::Cutter::Model(prunr_core::ModelKind::Silueta));
        assert!(tiers.tier_add_edge.is_empty(), "no lines composed onto the cached cut-out");
        assert!(tiers.tier1.contains(&1), "the lines run again on the whole photo");
    }

    /// A filters-only result lands the way every result does: the texture
    /// on screen stays until the new one is uploaded, and the step records.
    #[test]
    fn a_filters_result_keeps_the_old_texture_until_its_own_lands() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::None);
        let old = ctx.load_texture("old", egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let item = &mut app.batch.items[0];
        item.result_texture = Some(old.clone());
        item.status = BatchStatus::Processing;
        item.start_job(Some(StepKind::Process), Engine::Segment);
        app.batch.bg_io.filter_only_tx.send((1, Ok(px(5)), None)).unwrap();
        app.drain_background_channels(&ctx);
        let item = &app.batch.items[0];
        assert_eq!(shown(&app), Some(5));
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(old.id()), "the old texture stays until the new one lands");
        assert!(item.result_tex_pending);
        assert_eq!(item.status, BatchStatus::Done);
        assert!(item.job.is_none(), "the step is recorded");
    }

    /// An erase lands as one step that shows the filled image without its
    /// brush marks; undoing it brings back both the image and the marks.
    #[test]
    fn an_erase_lands_as_one_step_and_undo_brings_the_strokes_back() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);

        let item = &app.batch.items[0];
        assert_eq!(shown(&app), Some(7));
        assert!(item.selection_mask.is_none(), "the filled region's marks are gone");
        assert_eq!(item.timeline.undo_steps(), 3);

        app.handle_undo(&ctx);
        assert_eq!(shown(&app), Some(1));
        assert!(app.batch.items[0].selection_mask.is_some(), "undo brings the marks back with the image");
    }

    /// An erase records how large it filled its region on its step, so
    /// the footer can show what Auto picked; undo and redo keep it with
    /// the step.
    #[test]
    fn an_erase_records_how_large_it_filled() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.processor.land_inpaint_scaled_for_test(1, (*px(7)).clone(), 0.5);
        app.pump_inpaint_results(&ctx);
        let fill_scale = |app: &PrunrApp| app.batch.items[0].timeline.current().origin.as_ref().and_then(|o| o.fill_scale);
        assert_eq!(fill_scale(&app), Some(0.5));
        app.handle_undo(&ctx);
        assert_eq!(fill_scale(&app), None, "the stroke step has no fill");
        app.handle_redo(&ctx);
        assert_eq!(fill_scale(&app), Some(0.5));
    }

    /// The control socket reports the app busy while an erase runs, so a
    /// driver waits for the fill instead of reading the working overlay.
    #[test]
    fn the_app_is_busy_while_an_erase_runs() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        assert!(!crate::gui::automation::state::dump(&app).busy);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        assert!(crate::gui::automation::state::dump(&app).busy, "an erase in flight");
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);
        assert!(app.batch.items[0].job.is_none(), "landed: only the texture upload is left");
    }

    /// "Try another" reruns a generated fill from the image it started
    /// on, with the region painted for it and a new seed, as a new step:
    /// undo shows the fill before. It keeps working after a re-roll.
    #[test]
    fn try_another_reruns_the_fill_from_its_start_as_a_new_step() {
        let mut app = app_with_processed_item(SettingsModel::SdInpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        let painted = app.batch.items[0].timeline.current().selection.clone().expect("the stroke's region");
        let item = &mut app.batch.items[0];
        assert!(item.reroll().is_none(), "nothing generated yet");
        let first_start = item.run_start(Engine::Erase).map(|s| s.dims);
        item.start_erase(Some(1));
        item.result_rgba = Some(px(10));
        item.invalidate_selection();
        item.finish_job();
        let r = item.reroll().expect("a generated fill can be re-rolled");
        assert!(Arc::ptr_eq(&r.region, &painted), "the region painted for it");
        assert_eq!(r.start.as_ref().map(|s| s.dims), first_start);
        item.start_reroll(r, 2);
        item.result_rgba = Some(px(20));
        item.finish_job();
        assert_eq!(item.shown_seed(), Some(2));
        let again = item.reroll().expect("re-rolls chain");
        assert!(Arc::ptr_eq(&again.region, &painted));
        assert_eq!(again.start.as_ref().map(|s| s.dims), first_start, "still from the pre-erase image");
        item.undo_step();
        assert_eq!(item.shown_seed(), Some(1), "undo shows the fill before");
    }

    /// "Try another" is offered exactly when the SD eraser is chosen and
    /// the shown step is a generated fill with nothing running on it.
    #[test]
    fn try_another_is_offered_only_for_a_shown_generated_fill() {
        let mut app = app_with_processed_item(SettingsModel::SdInpaint);
        assert!(!app.can_reroll_intent(), "nothing generated yet");
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        let item = &mut app.batch.items[0];
        item.start_erase(Some(1));
        assert!(!app.can_reroll_intent(), "not while a fill runs");
        let item = &mut app.batch.items[0];
        item.result_rgba = Some(px(10));
        item.invalidate_selection();
        item.finish_job();
        assert!(app.can_reroll_intent());
        app.settings.model = SettingsModel::Inpaint;
        assert!(!app.can_reroll_intent(), "LaMa fills are not generated");
        app.settings.model = SettingsModel::SdInpaint;
        app.batch.items[0].undo_step();
        assert!(!app.can_reroll_intent(), "the shown step is the stroke");
    }

    /// A refused or failed erase ends its run: the item stops waiting,
    /// so it is not shown working forever.
    #[test]
    fn a_failed_erase_ends_its_run() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.processor.fail_inpaint_for_test(1, "refused to load: only 6 GB RAM free");
        app.pump_inpaint_results(&ctx);
        assert!(app.batch.items[0].job.is_none(), "the run is over");
        assert!(app.batch.items[0].selection_mask.is_some(), "the painted region stays for another try");
    }

    /// An upscale or colorize that fails or is cancelled ends its run:
    /// the item stops waiting, so it is not shown working forever.
    #[test]
    fn an_upscale_or_colorize_that_does_not_land_ends_its_run() {
        let ctx = egui::Context::default();
        let runs = [
            (SettingsModel::RealEsrganUpscale, Engine::Upscale, StepKind::Upscale),
            (SettingsModel::ColorizeDdcolor, Engine::Colorize, StepKind::Colorize),
        ];
        let errors: [fn() -> prunr_core::CoreError; 2] = [
            || prunr_core::CoreError::Cancelled,
            || prunr_core::CoreError::Inference("out of memory".into()),
        ];
        for (model, engine, step) in runs {
            for error in errors {
                let mut app = app_with_processed_item(model);
                app.batch.items[0].start_job(Some(step), engine);
                let recipe = app.batch.items[0].settings.current_recipe(prunr_core::ModelKind::Silueta);
                app.processor.end_run_for_test(engine, 1, error(), recipe);
                app.pump_upscale_results(&ctx);
                app.pump_colorize_results(&ctx);
                let item = &app.batch.items[0];
                assert!(item.job.is_none(), "{engine:?} {}: the run is over", error());
                assert_eq!(shown(&app), Some(1), "{engine:?}: the shown result stays");
            }
        }
    }

    /// An erase result that lands after an undo is dropped.
    #[test]
    fn an_erase_after_an_undo_does_not_land() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.handle_undo(&ctx);
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);
        assert_eq!(shown(&app), Some(1));
        assert!(app.batch.items[0].timeline.can_redo(), "the undone stroke is still there to redo");
    }

    /// A Process result that lands after an undo is dropped, and the undo
    /// asked the worker to stop.
    #[test]
    fn a_process_result_after_an_undo_does_not_land() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        app.handle_undo(&ctx);
        assert!(app.processor.cancels.is_cancelled(1), "the run is asked to stop");

        let late = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: String::new() };
        app.on_batch_item_done(&ctx, 1, Ok(late), None, None);
        assert_eq!(shown(&app), None, "undo went back to the original and stays there");
        assert_eq!(app.batch.items[0].status, BatchStatus::Pending);
    }

    /// A result for a state the image has since left (a knob change while
    /// it ran) cannot land; the image shows its current state again
    /// instead of staying "Processing" with nothing under way.
    #[test]
    fn a_result_for_a_state_the_image_has_left_ends_the_run() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        item.settings.gamma = 1.8;
        item.commit(StepKind::Adjust);

        let late = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: String::new() };
        app.on_batch_item_done(&ctx, 1, Ok(late), None, None);
        let item = &app.batch.items[0];
        assert_eq!(item.status, BatchStatus::Done, "shows its current state again");
        assert!(item.job.is_none(), "nothing is awaited any more");
        assert_eq!(shown(&app), Some(1), "the late result did not land");
        assert!(!crate::gui::automation::state::dump(&app).busy);
    }

    /// The same for an erase, an upscale and a colorize: a result for a
    /// state the image has left ends the run instead of hanging it.
    #[test]
    fn a_stale_erase_upscale_or_colorize_result_ends_its_run() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.batch.items[0].settings.gamma = 1.8;
        app.batch.items[0].commit(StepKind::Adjust);
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);
        assert!(app.batch.items[0].job.is_none(), "erase: the run is over");
        assert_eq!(shown(&app), Some(1), "erase: the late result did not land");

        let runs = [
            (SettingsModel::RealEsrganUpscale, Engine::Upscale, StepKind::Upscale),
            (SettingsModel::ColorizeDdcolor, Engine::Colorize, StepKind::Colorize),
        ];
        for (model, engine, step) in runs {
            let mut app = app_with_processed_item(model);
            app.batch.items[0].start_job(Some(step), engine);
            app.batch.items[0].status = BatchStatus::Processing;
            app.batch.items[0].settings.gamma = 1.8;
            app.batch.items[0].commit(StepKind::Adjust);
            let recipe = app.batch.items[0].settings.current_recipe(prunr_core::ModelKind::Silueta);
            app.processor.land_run_for_test(engine, 1, px(7), recipe);
            app.pump_upscale_results(&ctx);
            app.pump_colorize_results(&ctx);
            let item = &app.batch.items[0];
            assert_eq!(item.status, BatchStatus::Done, "{engine:?}: shows its current state again");
            assert!(item.job.is_none(), "{engine:?}: the run is over");
            assert_eq!(shown(&app), Some(1), "{engine:?}: the late result did not land");
        }
    }

    /// A batch that ends with an image still marked as processing (its
    /// result never came) frees that image.
    #[test]
    fn a_batch_ending_with_an_image_still_processing_frees_it() {
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        let recipe = item.settings.current_recipe(prunr_core::ModelKind::BiRefNetLite);
        let batch = app.processor.track_dispatch(recipe, [1]);
        app.on_batch_complete(batch);
        assert_eq!(app.batch.items[0].status, BatchStatus::Done);
        assert!(app.batch.items[0].job.is_none());
        assert!(!crate::gui::automation::state::dump(&app).busy);
    }

    /// Process pressed while a batch runs queues a second batch behind it;
    /// the first batch's end frees only the images it owned.
    #[test]
    fn a_batch_ending_leaves_the_queued_batch_processing() {
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let queued = push_test_item(&mut app, 2);
        queued.dimensions = (4, 4);
        for item in &mut app.batch.items {
            item.start_job(Some(StepKind::Process), Engine::Segment);
            item.status = BatchStatus::Processing;
        }
        let recipe = app.batch.items[0].settings.current_recipe(prunr_core::ModelKind::BiRefNetLite);
        let first = app.processor.track_dispatch(recipe.clone(), [1]);
        let queued = app.processor.track_dispatch(recipe, [2]);
        app.on_batch_complete(first);
        assert_eq!(app.batch.items[0].status, BatchStatus::Done, "its own unanswered image is freed");
        assert_eq!(app.batch.items[1].status, BatchStatus::Processing, "the queued batch's image waits for its own batch");
        assert!(app.batch.items[1].job.is_some());
        assert_eq!(app.processor.current_dispatch_progress(), Some((0, 1)), "the queued batch is still tracked");
        app.on_batch_complete(queued);
        assert_eq!(app.batch.items[1].status, BatchStatus::Pending, "never answered and never had a result: back to Pending");
        assert!(app.batch.items[1].job.is_none());
    }

    /// A Process result that lands on the state it started from is a step.
    /// Silueta cannot run on OpenVINO and reports the CPU by design; its
    /// result used to switch the app's accelerator to the CPU (Settings
    /// lost its Processor row) and reset the Parallel jobs you set.
    #[test]
    fn a_model_the_accelerator_does_not_run_leaves_the_backend_and_jobs_alone() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        app.settings.active_backend = "OpenVINO".into();
        app.settings.parallel_jobs = 3;
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        let cpu = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: "CPU".into() };
        app.on_batch_item_done(&ctx, 1, Ok(cpu), None, None);
        assert_eq!(app.settings.active_backend, "OpenVINO");
        assert_eq!(app.settings.parallel_jobs, 3);
    }

    #[test]
    fn a_process_result_lands_as_its_own_step() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        let fresh = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: String::new() };
        app.on_batch_item_done(&ctx, 1, Ok(fresh), None, None);
        assert_eq!(shown(&app), Some(9));
        app.handle_undo(&ctx);
        assert_eq!(shown(&app), Some(1), "undo shows the result it replaced");
    }

    /// A knob change is an Adjust step; undo brings the old value back,
    /// with the image that went with it.
    #[test]
    fn a_knob_change_is_a_step_and_undo_restores_it() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let before = app.batch.items[0].settings;
        app.batch.items[0].settings.gamma = 1.8;
        app.apply_toolbar_change(&ctx, Default::default(), before);
        assert_eq!(app.batch.items[0].timeline.undo_steps(), 2);

        app.handle_undo(&ctx);
        assert_eq!(app.batch.items[0].settings.gamma, before.gamma);
        app.handle_redo(&ctx);
        assert_eq!(app.batch.items[0].settings.gamma, 1.8);
    }

    /// Clearing the selection is a step: undo brings it back.
    #[test]
    fn clearing_the_selection_is_a_step() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        assert!(app.batch.clear_selection(1));
        assert!(app.batch.items[0].selection_mask.is_none());
        app.handle_undo(&ctx);
        assert!(app.batch.items[0].selection_mask.is_some());
    }
}

#[cfg(test)]
mod depth_tests {
    use std::sync::Arc;
    use super::PrunrApp;
    use crate::gui::item::BatchStatus;
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;

    fn map() -> Arc<prunr_core::depth::DepthMap> {
        Arc::new(prunr_core::depth::DepthMap::from_raw(1, 2, vec![0.0, 1.0]))
    }

    /// A landing depth map re-renders what reads it, and only that: the
    /// Depth chip's look on No model, faded lines on a finished cut-out;
    /// nothing for an image that reads no depth or has no result yet.
    #[test]
    fn a_landed_depth_map_re_renders_only_what_reads_it() {
        use prunr_core::depth::DepthLook;
        type Case = (&'static str, SettingsModel, BatchStatus, fn(&mut crate::gui::item_settings::ItemSettings), bool);
        let cases: [Case; 5] = [
            ("look on No model", SettingsModel::None, BatchStatus::Pending, |s| s.depth = DepthLook { fog_pct: 40, ..DepthLook::default() }, true),
            ("look on a cut-out model", SettingsModel::BiRefNetLite, BatchStatus::Done, |s| s.depth = DepthLook { fog_pct: 40, ..DepthLook::default() }, false),
            ("faded lines, done", SettingsModel::BiRefNetLite, BatchStatus::Done, |s| { s.line_mode = prunr_core::LineMode::EdgesOnly; s.line_fade_pct = 50; }, true),
            ("faded lines, not processed", SettingsModel::BiRefNetLite, BatchStatus::Pending, |s| { s.line_mode = prunr_core::LineMode::EdgesOnly; s.line_fade_pct = 50; }, false),
            ("nothing reads depth", SettingsModel::None, BatchStatus::Done, |_| {}, false),
        ];
        for (name, model, status, set, reruns) in cases {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = model;
            let item = push_test_item(&mut app, 1);
            item.status = status;
            set(&mut item.settings);
            let epoch = item.depth_run.epoch;
            assert!(item.land_depth(epoch, Ok(map())));
            app.after_depth_landed(0);
            assert_eq!(app.processor.live_preview.is_pending_for(1), reruns, "{name}");
        }
    }

    /// A lens-blur background waits for the depth map: no texture build
    /// starts without it, and the texture key changes when it lands.
    #[test]
    fn a_lens_blur_background_waits_for_the_depth_map() {
        let mut app = PrunrApp::new_for_test();
        let ctx = egui::Context::default();
        let item = push_test_item(&mut app, 1);
        item.settings.bg_effect = prunr_core::BgEffect::LensBlur { amount_pct: 50 };
        item.source_rgba = Some(Arc::new(image::RgbaImage::new(1, 1)));
        let before = item.backdrop_key();
        app.kick_backdrop_tex_prep(1, &ctx);
        assert!(app.batch.items[0].backdrop_tex_pending.is_none(), "no build without the map");
        app.batch.items[0].depth = Some(map());
        assert_ne!(app.batch.items[0].backdrop_key(), before, "the map is part of the key");
        app.kick_backdrop_tex_prep(1, &ctx);
        assert!(app.batch.items[0].backdrop_tex_pending.is_some(), "it builds once the map is there");
    }

    /// A By depth click on an image whose map is not there yet waits for
    /// the map and selects when it lands; the brush turned off meanwhile
    /// drops it.
    #[test]
    fn a_by_depth_click_before_the_map_runs_when_it_lands() {
        use crate::gui::processor::DecodedSelection;
        use prunr_core::selection::BrushMode;
        let mut app = PrunrApp::new_for_test();
        app.settings.model = SettingsModel::Silueta;
        push_test_item(&mut app, 1);
        app.batch.items[0].dimensions = (8, 8);
        app.batch.select_item(0);
        app.magic_brush_state.activate();
        app.magic_depth_click(1, (3.0, 3.0), (8, 8), BrushMode::Add);
        assert!(app.magic_brush_state.pending_depth_click.is_some(), "parked until the map is there");
        assert!(app.processor.pump_sam_decoder_results().is_empty());
        let epoch = app.batch.items[0].depth_run.epoch;
        assert!(app.batch.items[0].land_depth(epoch, Ok(map())));
        app.on_depth_landed(0);
        assert!(app.magic_brush_state.pending_depth_click.is_none());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let result = loop {
            if let Some(r) = app.processor.pump_sam_decoder_results().pop() { break r; }
            assert!(std::time::Instant::now() < deadline, "the click never ran");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert!(matches!(result.result, Ok(DecodedSelection::ByDepth(_))), "selected by depth");

        app.batch.items[0].depth = None;
        app.magic_depth_click(1, (3.0, 3.0), (8, 8), BrushMode::Add);
        app.magic_brush_state.deactivate();
        assert!(app.magic_brush_state.pending_depth_click.is_none(), "the brush turned off drops the waiting click");
    }

    /// Set focus stores the click as fractions, makes it an undo step and
    /// re-renders the photo.
    #[test]
    fn a_focus_click_is_a_step_that_re_renders() {
        let mut app = PrunrApp::new_for_test();
        app.settings.model = SettingsModel::None;
        push_test_item(&mut app, 1);
        app.batch.select_item(0);
        app.picking_depth_focus = true;
        let version = app.batch.items[0].timeline.version();
        app.set_depth_focus(0.25, 0.75);
        let item = &app.batch.items[0];
        assert_eq!(item.settings.depth.focus_point(), Some((0.25, 0.75)));
        assert!(item.timeline.version() != version, "an undo step");
        assert!(!app.picking_depth_focus);
        assert!(app.processor.live_preview.is_pending_for(1));
    }
}
