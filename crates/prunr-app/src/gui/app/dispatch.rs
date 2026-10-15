//! Sending work out: Process runs, tier routing, admission, and the upscale / colorize / erase dispatches.

use std::collections::HashSet;
use std::sync::mpsc;

use crate::gui::item::{BatchItem, BatchStatus, Cutter, Engine, ImageSource, KeptRunTier};
use crate::gui::live_preview::PreviewKind;
use crate::gui::processor::RunInput;
use crate::gui::timeline::StepKind;
use crate::gui::worker::{CachedTensorWork, WorkerMessage};
use super::PrunrApp;

/// Why an upscale to `out` cannot run in `free` bytes, with the model's
/// own `working_set_mb` beside it; `None` when it fits.
fn too_large_to_upscale(out: (u32, u32), working_set_mb: u32, free: u64) -> Option<String> {
    let needed = prunr_core::upscale::peak_bytes(out.0, out.1) + u64::from(working_set_mb) * 1024 * 1024;
    (needed > free).then(|| {
        use crate::gui::views::format_byte_size;
        format!(
            "Too large to upscale: {}\u{00d7}{} needs about {} of free memory, {} is free. Undo to a smaller image or pick a lower Scale.",
            out.0, out.1, format_byte_size(needed), format_byte_size(free),
        )
    })
}

impl PrunrApp {
    /// The target images an erase runs on (those with a painted region),
    /// and how many targets have none.
    pub(super) fn erase_targets(&self) -> (Vec<usize>, usize) {
        let (with_region, without): (Vec<usize>, Vec<usize>) = self.batch.targeted_indices().into_iter()
            .partition(|&idx| self.batch.items[idx].has_painted_region());
        (with_region, without.len())
    }

    /// Start the next queued upscale, colorize or erase when the slot is free,
    /// with the shown image's settings, as a batch of cut-outs runs.
    pub(super) fn dispatch_next_queued_run(&mut self) {
        // An image gone from the batch, or one without a run, passes its
        // turn on: nothing else would call this again for the rest.
        while let Some(item_id) = self.processor.next_queued_run() {
            if let Some(shown) = self.batch.selected_idx_clamped() {
                self.batch.share_settings(shown, &HashSet::from([item_id]));
            }
            let started = if self.settings.model.is_inpaint() {
                self.batch.index_of(item_id).is_some_and(|idx| self.dispatch_inpaint_for_item(idx))
            } else if self.settings.model.is_colorize() {
                self.dispatch_colorize_for(item_id)
            } else {
                self.dispatch_upscale_for(item_id)
            };
            if started {
                return;
            }
        }
    }

    /// Process on an upscaler or the colorizer. Every target takes the
    /// shown image's settings first, as a cut-out run does. One that took
    /// only a Refine or Strength change finishes its kept run again; one
    /// already showing this model's run as asked is enlarged again by an
    /// upscaler and left alone by the colorizer (its run would give the
    /// same picture); the rest queue for a run.
    pub(super) fn process_kept_run_items(&mut self) {
        let Some(model) = self.settings.model.dispatch_model_kind() else { return };
        let ids = self.batch.items_to_process();
        let shown = self.batch.selected_idx_clamped();
        let shown_settings = shown.map(|i| self.batch.items[i].settings);
        let adopting: HashSet<u64> = ids.iter().copied()
            .filter(|&id| self.batch.find_by_id(id).is_some_and(|i| Some(i.settings) != shown_settings))
            .collect();
        if let Some(shown) = shown {
            self.batch.share_settings(shown, &ids.iter().copied().collect());
        }
        let refinish = if model.is_upscale() { PreviewKind::UpscaleTier2 } else { PreviewKind::ColorizeTier2 };
        let (mut runs, mut up_to_date, mut refinished) = (Vec::new(), 0, 0);
        for id in ids {
            let Some(item) = self.batch.find_by_id_mut(id) else { continue };
            match item.kept_run_tier(model) {
                KeptRunTier::UpToDate if model.is_upscale() => runs.push(id),
                KeptRunTier::UpToDate => up_to_date += 1,
                KeptRunTier::Refinish => {
                    // The shown image's own knob change is a step already.
                    if adopting.contains(&id) {
                        item.commit(StepKind::Adjust);
                    }
                    self.processor.live_preview.render_now(id, refinish);
                    refinished += 1;
                }
                KeptRunTier::Full => runs.push(id),
            }
        }
        self.notify_skip(up_to_date, runs.len() + refinished);
        // One image at a time through the run queue: an upscale's working
        // set can be most of the machine's RAM.
        self.processor.queue_runs(runs);
        self.dispatch_next_queued_run();
    }

    /// Start an upscale of `item_id` from where its timeline says a run
    /// starts, refusing an output too large for free memory. Must only be
    /// called after `can_process_intent()` confirms the model is installed
    /// and idle.
    fn dispatch_upscale_for(&mut self, item_id: u64) -> bool {
        let Some(item) = self.batch.find_by_id(item_id) else { return false };
        let Some(model_id) = self.settings.model.to_model_id() else { return false };
        let output_scale = item.settings.output_scale;
        // Capture the recipe at dispatch time so the pump can stamp
        // `item.applied_recipe` against what actually ran (not what the
        // user has tweaked to since).
        let Some(model_kind) = self.settings.model.dispatch_model_kind() else {
            tracing::error!(?model_id, "upscale dispatch skipped: ModelId has no ModelKind mapping");
            return false;
        };
        let recipe = item.settings.current_recipe(model_kind);
        let start = item.next_upscale_start(model_kind);
        let (w, h) = start.as_ref().map_or(item.dimensions, |s| s.dims);
        let out = (w.saturating_mul(output_scale.factor()), h.saturating_mul(output_scale.factor()));
        let free = crate::hardware::available_ram_bytes_throttled();
        if let Some(why) = too_large_to_upscale(out, crate::gui::memory::working_set_mb(model_kind), free) {
            tracing::warn!(item_id, out_w = out.0, out_h = out.1, free, "upscale refused: too large for free memory");
            self.toasts.error(why);
            return false;
        }
        let input = start.as_ref().and_then(|s| s.image.load()).map_or_else(|| item.original_run_input(), RunInput::Decoded);
        let run_model = prunr_core::upscale::model_for_scale(model_id, output_scale, self.hardware_install_cache.x2plus);
        tracing::info!(item_id, ?model_id, ?run_model, ?output_scale, "upscale dispatched");
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.start_job_at(Some(StepKind::Upscale), Engine::Upscale, start);
        }
        self.processor.dispatch_upscale(item_id, input, run_model, output_scale, self.settings.cpu_only(), recipe);
        true
    }

    /// Start a colorize of `item_id` from where its timeline says a run
    /// starts.
    fn dispatch_colorize_for(&mut self, item_id: u64) -> bool {
        let Some(item) = self.batch.find_by_id(item_id) else { return false };
        let input = item.run_input_or_source(Engine::Colorize);
        let recipe = item.settings.current_recipe(crate::gui::settings::COLORIZER);
        tracing::info!(item_id, "colorize dispatched");
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.start_job(Some(StepKind::Colorize), Engine::Colorize);
        }
        self.processor.dispatch_colorize(item_id, input, self.settings.cpu_only(), recipe);
        true
    }

    /// Start the erase of the image at `idx`; false when it has nothing
    /// to fill or no pixels to fill it on.
    pub(crate) fn dispatch_inpaint_for_item(&mut self, idx: usize) -> bool {
        let item = &self.batch.items[idx];
        let item_id = item.id;
        // selection_mask is the single source of truth for the inpaint
        // region. It persists across Process, so Reprocess is available
        // whenever any selection exists — no separate "last_inpaint" cache.
        let Some(selection) = item.selection_mask.as_ref().cloned() else {
            tracing::debug!(item_id, "inpaint dispatch skipped: no selection_mask");
            return false;
        };
        // Stack-based inpaint: each stroke runs against the previous
        // result so earlier strokes stay intact. source_for_inpaint
        // walks result_rgba → source_rgba → source_dyn, the last arm
        // handling memory-pressure eviction.
        let Some(source) = item.source_for_inpaint() else {
            tracing::warn!(item_id, "inpaint dispatch skipped: source RGBA unavailable");
            return false;
        };
        let picked_model = self.eraser_model();
        let seed = picked_model.is_sd_family()
            .then(|| self.settings.brush.sd_seed.unwrap_or_else(prunr_core::inpaint_sd::fresh_seed));
        tracing::info!(item_id, ?picked_model, ?seed, "inpaint stroke committed; dispatching");
        let tuning = self.erase_tuning(seed);
        self.processor.dispatch_inpaint(item_id, source, selection, tuning);
        self.batch.items[idx].start_erase(seed);
        true
    }

    /// The eraser model the Model picker names (LaMa without one).
    fn eraser_model(&self) -> prunr_models::ModelId {
        self.settings.model.to_model_id().unwrap_or(prunr_models::ModelId::LaMaFp32)
    }

    /// How a fill runs under the current settings, with `seed` for a
    /// generated one.
    pub(super) fn erase_tuning(&mut self, seed: Option<u64>) -> crate::gui::processor::InpaintTuning {
        self.settings.fit_sd_quality_to_install(self.hardware_install_cache.lcm_bundle);
        let picked_model = self.eraser_model();
        let model = if self.settings.lcm_routing_active(picked_model) {
            prunr_models::ModelId::SdV15LcmInpaintFp16
        } else {
            picked_model
        };
        let bs = &self.settings.brush;
        crate::gui::processor::InpaintTuning {
            sharpen: bs.inpaint_sharpen,
            feather_px: bs.inpaint_feather,
            grow_px: bs.inpaint_grow,
            fill_scale: bs.inpaint_fill_scale,
            model,
            sd_prompt: bs.sd_prompt_for_run(),
            sd_negative_prompt: bs.sd_negative_prompt.clone(),
            sd_guidance_scale: bs.sd_guidance_scale,
            sd_scheduler: bs.sd_scheduler,
            sd_steps: bs.sd_steps,
            rorem_steps: bs.rorem_steps,
            sd_seed: seed,
            use_taesd: bs.sd_use_taesd_effective(),
            sd_tuning: prunr_core::inpaint_sd::plan_tuning(
                self.settings.keep_sd_loaded,
                self.settings.sd_memory.margin_mb(),
            )
            .with_env_overrides(),
            cpu_only: self.settings.cpu_only(),
        }
    }

    /// Run background removal (or the No-model filter) on images imported
    /// from `id_floor` on, when the setting is on. Erasers wait for a
    /// painted region and upscales for the image to enlarge, so they
    /// never start on import.
    pub(super) fn auto_process_imported(&mut self, id_floor: u64) {
        let model = self.settings.model;
        if !self.settings.auto_process_on_import || model.role().runs_on_demand() {
            return;
        }
        if self.batch.items.iter().any(|i| i.id >= id_floor) {
            self.process_items(|item| item.id >= id_floor, Some(StepKind::Process));
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    /// Uses tier routing: compares each item's applied_recipe against current
    /// settings to determine the minimum work needed (skip / mask rerun / full).
    /// `step`: the undo step the results record, or `None` to fill in the
    /// shown one (a re-run a knob change asked for).
    pub(super) fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool, step: Option<StepKind>) {
        self.batch.fit_line_modes_to(self.settings.model);
        // Pure filter mode: No model AND line_mode = Off. Skip inference +
        // subprocess entirely. (No model + EdgesOnly still needs TEED,
        // so falls through to the normal path with a dummy ModelKind — the
        // seg engine won't spawn because needs_segmentation is false.)
        let is_pure_filter = self.settings.model.to_model_kind().is_none()
            && self.run_settings().line_mode == prunr_core::LineMode::Off;
        if is_pure_filter {
            self.process_filter_only(filter, step);
            return;
        }
        // No-model-but-EdgesOnly falls through with the placeholder. The
        // subprocess Init still receives this field; `needs_segmentation`
        // is false for EdgesOnly so the seg model never loads.
        let model: prunr_core::ModelKind = self.settings.model
            .to_model_kind()
            .unwrap_or(super::NO_MODEL_RECIPE_KIND);

        let candidate_ids: HashSet<u64> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }
        // What you see is what you process: every target takes the shown
        // image's settings first, so one that was up to date under its own
        // reruns under these. A background image's result is released; a
        // run may start from it.
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &candidate_ids);
        }
        for item in self.batch.in_set(&candidate_ids) {
            item.unpark_result();
        }

        let cutter = Cutter::of(self.settings.model.to_model_kind());
        let tiers = self.batch.classify(&candidate_ids, model, step, cutter);
        self.notify_skip(tiers.up_to_date_count, tiers.process_count());
        if tiers.process_count() == 0 { return; }

        for id in tiers.all_process_ids() {
            if let Some(item) = self.batch.find_by_id_mut(id) {
                item.start_cut(step, cutter);
            }
        }

        let mask_work = self.batch.cached_tensor_work(&tiers.mask);
        let add_edge_work = self.batch.cached_tensor_work(&tiers.add_edge);

        let jobs = self.settings.parallel_jobs.min(crate::gui::memory::safe_max_jobs(model));
        self.update_on_screen();
        if tiers.full.len() > 1 {
            self.dispatch_with_admission(&tiers.full, mask_work, add_edge_work, model, jobs);
        } else {
            self.dispatch_small_batch(&tiers.full, mask_work, add_edge_work, model, jobs);
        }
    }

    /// Filter-only Process path (model=`None`). Dispatches each target item
    /// to a background thread via `BatchManager::request_filter_only` so the
    /// UI stays responsive on large batches. Results land on
    /// `bg_io.filter_only_rx`, drained in `drain_background_channels`.
    fn process_filter_only(&mut self, filter: impl Fn(&BatchItem) -> bool, step: Option<StepKind>) {
        let settings = self.run_settings();
        let dispatches: Vec<(u64, ImageSource)> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| (i.id, i.source.clone()))
            .collect();
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &dispatches.iter().map(|(id, _)| *id).collect());
        }
        for (id, source) in dispatches {
            let Some(item) = self.batch.find_by_id_mut(id) else { continue };
            item.status = BatchStatus::Processing;
            item.start_cut(step, Cutter::NoModel);
            let input = item.job_run_image();
            self.batch.request_filter_only(id, &source, input, settings.effect);
        }
    }

    /// Tell the user when some items were skipped. Three shapes:
    /// nothing-skipped (silent), all-skipped, partially-skipped.
    fn notify_skip(&mut self, skipped: usize, processing: usize) {
        if skipped == 0 { return; }
        let msg = if processing == 0 {
            if skipped == 1 {
                "Already up to date".to_string()
            } else {
                format!("{skipped} images already up to date")
            }
        } else {
            format!("{skipped} up to date, processing {processing}")
        };
        self.toasts.info(msg);
    }

    /// Dispatch with streaming admission — used when >1 Tier 1 items so the
    /// batch doesn't blow past memory limits. Admits what fits now; the
    /// worker bridge receives more items via `admission_tx` as earlier items
    /// complete and free memory.
    fn dispatch_with_admission(
        &mut self,
        tier1_ids: &HashSet<u64>,
        mask_work: Vec<CachedTensorWork>,
        add_edge_work: Vec<CachedTensorWork>,
        model: prunr_core::ModelKind,
        jobs: usize,
    ) {
        let first_look = self.processor.first_look.get().is_some_and(|id| tier1_ids.contains(&id));
        let mut ctrl = crate::gui::memory::AdmissionController::new(model, jobs, self.settings.cpu_only(), first_look);
        ctrl.enqueue(self.batch.full_run_costs(tier1_ids));
        let batch = &mut self.batch;
        let (initial_items, additional_items_rx) =
            self.processor.open_admission(ctrl, |id| batch.find_by_id_mut(id).and_then(BatchItem::full_run_work));
        self.batch.mark_waiting(tier1_ids);

        // All Tier 1 items failed load_bytes AND no Tier 2 / add-edge work —
        // skip dispatch.
        if initial_items.is_empty() && mask_work.is_empty() && add_edge_work.is_empty() {
            self.processor.clear_admission();
            return;
        }
        self.dispatch_batch(initial_items, mask_work, add_edge_work, model, jobs, Some(additional_items_rx));
    }

    /// Single-item or small-batch fast path — skip admission entirely.
    fn dispatch_small_batch(
        &mut self,
        tier1_ids: &HashSet<u64>,
        mask_work: Vec<CachedTensorWork>,
        add_edge_work: Vec<CachedTensorWork>,
        model: prunr_core::ModelKind,
        jobs: usize,
    ) {
        let items = self.batch.full_run_work(tier1_ids);

        // If all prep failed, don't dispatch an empty batch.
        if items.is_empty() && mask_work.is_empty() && add_edge_work.is_empty() { return; }

        self.dispatch_batch(items, mask_work, add_edge_work, model, jobs, None);
    }

    /// The settings a batch runs with: the shown image's — what you see
    /// is what you process. Factory defaults on an empty batch.
    fn run_settings(&self) -> crate::gui::item_settings::ItemSettings {
        self.batch.selected_idx_clamped().map(|i| self.batch.items[i].settings).unwrap_or_default()
    }

    /// Build and send a WorkerMessage::BatchProcess with current settings.
    fn dispatch_batch(
        &mut self,
        items: Vec<crate::gui::worker::WorkItem>,
        tier2_items: Vec<CachedTensorWork>,
        add_edge_items: Vec<CachedTensorWork>,
        model: prunr_core::ModelKind,
        jobs: usize,
        additional_items_rx: Option<mpsc::Receiver<crate::gui::worker::WorkItem>>,
    ) {
        self.processor.cancels.reset();

        let current_settings = self.run_settings();
        let process_ids: std::collections::HashSet<u64> = items.iter()
            .map(|wi| wi.0)
            .chain(tier2_items.iter().map(|ti| ti.item_id))
            .chain(add_edge_items.iter().map(|ai| ai.item_id))
            .collect();

        let recipe = current_settings.current_recipe(model);
        let batch = self.processor.track_dispatch(recipe, process_ids.iter().copied());

        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.processor.worker_tx.send(WorkerMessage::BatchProcess {
            batch,
            items,
            tier2_items,
            add_edge_items,
            config: Box::new(crate::gui::worker::ProcessingConfig {
                engines: crate::gui::worker::EngineSpec { model, jobs, cpu_only: self.settings.cpu_only() },
                run: crate::gui::worker::RunSettings {
                    mask: current_settings.mask_settings(),
                    line_mode: current_settings.line_mode,
                    edge: current_settings.edge_settings(),
                },
                first_look: self.processor.first_look.clone(),
                on_screen: self.processor.on_screen.clone(),
            }),
            cancels: self.processor.cancels.clone(),
            additional_items_rx,
        });
    }

    /// Tell the worker which image is on screen, and whether its run
    /// shows a first look at its subject, as the view and the settings
    /// say now.
    pub(super) fn update_on_screen(&mut self) {
        let shown = self.batch.selected_item();
        self.processor.on_screen.set(shown.map(|i| i.id));
        let (settings, times) = (&self.settings, &self.processor.cut_times);
        let item = crate::gui::first_look::first_look_for(
            shown.map(|i| i.id),
            settings.model.to_model_kind(),
            shown.map_or(prunr_core::LineMode::Off, |i| i.settings.line_mode),
            settings.animations,
            |model| times.expected(model, settings.model_backend(model)),
        );
        self.processor.show_first_look(item);
    }

    /// Release an item's memory budget and greedily admit the next fitting items.
    pub(super) fn admission_release_and_admit(&mut self, completed_id: u64) {
        let batch = &mut self.batch;
        self.processor.release_and_admit(completed_id, |id| batch.find_by_id_mut(id).and_then(BatchItem::full_run_work));
    }
}

/// Process runs any model on the ticked images, the way the cut-out
/// models always did.
#[cfg(test)]
mod process_targets_tests {
    use super::PrunrApp;
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;
    use crate::gui::timeline::StepKind;
    use crate::gui::item::Engine;

    fn mask() -> std::sync::Arc<prunr_core::selection::MaskArtifact> {
        let mut cells = vec![0i8; 16];
        cells[5] = prunr_core::selection::FULL;
        std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(4, 4, cells))
    }

    fn two_images(model: SettingsModel) -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        app.settings.model = model;
        for id in [1, 2] {
            let item = push_test_item(&mut app, id);
            item.dimensions = (4, 4);
        }
        app.batch.select_item(0);
        app
    }

    #[test]
    fn an_eraser_runs_on_every_ticked_image_with_a_region() {
        let mut app = two_images(SettingsModel::Inpaint);
        assert!(!app.can_process_intent(), "nothing painted yet");
        app.batch.items[1].commit_selection_mask(mask(), StepKind::Stroke);
        assert!(!app.can_process_intent(), "the viewed image has no region");
        app.batch.items[1].ticked = true;
        assert!(app.can_process_intent(), "a ticked image has one");
        assert_eq!(app.erase_targets(), (vec![1], 0));
        app.batch.items[0].ticked = true;
        assert_eq!(app.erase_targets(), (vec![1], 1), "the ticked one without a region is counted");

        // Erases go one at a time through the run queue, like upscales:
        // a fill's working set can be most of the machine's RAM.
        app.processor.queue_runs(vec![1, 2]);
        app.processor.set_inpaint_in_flight_for_test(1, true);
        assert!(!app.can_process_intent(), "one erase at a time");
        assert_eq!(app.processor.next_queued_run(), None, "the next waits for the slot");
        app.processor.set_inpaint_in_flight_for_test(1, false);
        assert_eq!(app.processor.next_queued_run(), Some(1));
    }

    /// A finished (or failed) erase frees the slot and the next ticked
    /// image starts; one with nothing to fill passes its turn on.
    #[test]
    fn a_finished_erase_starts_the_next_ticked_one() {
        let ctx = egui::Context::default();
        let mut app = two_images(SettingsModel::Inpaint);
        app.processor.queue_runs(vec![2, 1]);
        app.pump_inpaint_results(&ctx);
        assert_eq!(app.processor.next_queued_run(), None, "neither image had a region: both turns passed");
    }

    /// A batch tells the worker which image on screen shows a first look;
    /// with animations off none does, the running batch's worker hears so
    /// at once and the warm worker lets the first-look model go.
    #[test]
    fn animations_off_loads_no_first_look() {
        use crate::gui::worker::WorkerMessage;
        let mut app = two_images(SettingsModel::BiRefNetLite);
        let (tx, rx) = std::sync::mpsc::channel();
        app.processor.worker_tx = tx;
        // One image ticked: a batch of several sizes its admission by the
        // provider, which loads the runtime a test app never loads.
        let process = |app: &mut PrunrApp| {
            let item = &mut app.batch.items[1];
            item.ticked = true;
            item.status = crate::gui::item::BatchStatus::Pending;
            app.process_items(|i| i.ticked, Some(StepKind::Process));
            match rx.try_recv() {
                Ok(WorkerMessage::BatchProcess { config, .. }) => config.first_look,
                _ => panic!("a batch is sent"),
            }
        };
        let shown = process(&mut app);
        assert_eq!(shown.get(), Some(1), "the image on screen: the batch's own image 2 never looks");

        app.settings.animations = false;
        app.update_on_screen();
        assert_eq!(shown.get(), None, "the running batch's worker hears first looks are off");
        assert!(matches!(rx.try_recv(), Ok(WorkerMessage::FirstLookOff)), "the warm worker lets the model go");
        assert_eq!(process(&mut app).get(), None, "a batch with animations off loads no first-look model");
    }

    /// Process gives every ticked image the shown image's settings before
    /// deciding what to run, so an image that was up to date under its own
    /// settings reruns under the shown ones, as the status bar promises.
    #[test]
    fn process_applies_the_shown_settings_to_every_ticked_image() {
        use crate::gui::worker::{CompressedTensor, RawSegTensor};
        let mut app = two_images(SettingsModel::Silueta);
        for item in &mut app.batch.items {
            // Both are done under their own settings, with the tensor a mask rerun needs.
            item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
            item.set_cached_tensor(CompressedTensor::from_raw(RawSegTensor { data: vec![0.5; 16], height: 4, width: 4, model: prunr_core::ModelKind::Silueta }));
            item.status = crate::gui::item::BatchStatus::Done;
            item.ticked = true;
        }
        app.batch.items[0].settings.feather = 3.0;
        app.process_items(|i| i.ticked, Some(StepKind::Process));
        let other = &app.batch.items[1];
        assert_eq!(other.settings.feather, 3.0, "the ticked image took the shown settings");
        assert_eq!(other.status, crate::gui::item::BatchStatus::Processing, "and reruns under them");
    }

    /// Process by another model than the one that cut the image on screen
    /// builds on that cut-out, with No model too; the same model, or a
    /// knob's re-run, redoes the step from its own start.
    #[test]
    fn process_by_another_model_builds_on_the_cutout_on_screen() {
        use crate::gui::item::Cutter;
        use prunr_core::ModelKind;
        let mut app = two_images(SettingsModel::Silueta);
        for item in &mut app.batch.items {
            item.applied_recipe = Some(item.settings.current_recipe(ModelKind::Silueta));
            item.land_cut_for_test(Cutter::Model(ModelKind::Silueta), std::sync::Arc::new(image::RgbaImage::new(4, 4)));
        }
        let chained = |app: &PrunrApp, i: usize| app.batch.items[i].job.as_ref().map(|j| j.keep.is_some());

        app.process_items(|i| i.id == 1, Some(StepKind::Process));
        assert_eq!(chained(&app, 0), None, "the same model: up to date, nothing runs");

        app.settings.model = SettingsModel::BiRefNetLite;
        app.process_items(|i| i.id == 1, None);
        assert_eq!(chained(&app, 0), Some(false), "a knob's re-run redoes the Silueta step");
        app.batch.items[0].abandon_job();

        app.process_items(|i| i.id == 1, Some(StepKind::Process));
        assert_eq!(chained(&app, 0), Some(true), "Process by BiRefNet builds on the Silueta cut-out");

        app.settings.model = SettingsModel::None;
        app.process_items(|i| i.id == 2, Some(StepKind::Process));
        assert_eq!(chained(&app, 1), Some(true), "No model's pass builds on it too");
    }

    /// The picker lists a model that is not on disk with "Get"; Process
    /// stays off until it is.
    #[test]
    fn process_is_off_while_the_saved_model_is_not_installed() {
        let mut app = two_images(SettingsModel::Silueta);
        assert!(app.can_process_intent());
        app.hardware_install_cache = Default::default();
        assert!(!app.can_process_intent());
    }

    /// An image gone from the batch by the time its turn comes passes it
    /// on, so the rest of the queue still runs.
    #[test]
    fn a_queued_image_that_is_gone_passes_its_turn_on() {
        let mut app = two_images(SettingsModel::RealEsrganUpscale);
        app.processor.queue_runs(vec![99, 98, 97]);
        app.processor.set_upscale_in_flight_for_test(true);
        app.dispatch_next_queued_run();
        assert_eq!(app.processor.next_queued_run(), None, "the slot is taken: nothing moved");
        app.processor.set_upscale_in_flight_for_test(false);
        app.dispatch_next_queued_run();
        assert_eq!(app.processor.next_queued_run(), None, "every image that could not start passed its turn on; none is left waiting");
    }

    /// Process on an upscaler enlarges an image whose upscale is what
    /// the settings ask again, and re-finishes the kept upscale of one
    /// that took only a Refine change from the shown image.
    #[test]
    fn process_on_an_upscaler_stacks_or_refinishes() {
        use crate::gui::item::{BatchStatus, UpscaleCache};
        let mut app = two_images(SettingsModel::RealEsrganUpscale);
        let model = app.settings.model.dispatch_model_kind().expect("an upscaler");
        let img = std::sync::Arc::new(image::RgbaImage::new(4, 4));
        for item in &mut app.batch.items {
            item.start_job(Some(StepKind::Upscale), Engine::Upscale);
            item.applied_recipe = Some(item.settings.current_recipe(model));
            item.result_rgba = Some(img.clone());
            item.land_upscale(UpscaleCache::new(img.clone(), img.clone(), None));
            item.status = BatchStatus::Done;
            item.ticked = true;
        }
        // The slot is taken while Process runs, so the queue holds what it
        // sent; freed to read the queue back.
        let queued = |app: &PrunrApp| {
            app.processor.set_upscale_in_flight_for_test(false);
            let ids = std::iter::from_fn(|| app.processor.next_queued_run()).collect::<Vec<_>>();
            app.processor.set_upscale_in_flight_for_test(true);
            ids
        };
        app.processor.set_upscale_in_flight_for_test(true);
        app.process_kept_run_items();
        assert_eq!(queued(&app), vec![1, 2], "both up to date: Process enlarges each again");

        app.batch.items[0].settings.sharpen = 0.4;
        let steps = app.batch.items[1].timeline.undo_steps();
        app.process_kept_run_items();
        assert!(queued(&app).is_empty(), "a Sharpen change needs no new upscale");
        assert!(app.processor.live_preview.is_pending_for(2), "the ticked image re-finishes its kept upscale");
        assert_eq!(app.batch.items[1].settings.sharpen, 0.4, "with the shown image's Sharpen");
        assert_eq!(app.batch.items[1].timeline.undo_steps(), steps + 1, "as a step of its own");
    }

    /// Process on Colorize leaves an image already colorized as asked
    /// alone: its run would give the same picture.
    #[test]
    fn process_on_colorize_skips_what_is_up_to_date() {
        use crate::gui::item::{BatchStatus, ColorizeCache};
        let mut app = two_images(SettingsModel::ColorizeDdcolor);
        let model = crate::gui::settings::COLORIZER;
        let img = std::sync::Arc::new(image::RgbaImage::new(4, 4));
        let plane = std::sync::Arc::new(vec![0.0; 2 * (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize]);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Colorize), Engine::Colorize);
        item.applied_recipe = Some(item.settings.current_recipe(model));
        item.result_rgba = Some(img.clone());
        item.land_colorize(ColorizeCache::new(plane, img));
        item.status = BatchStatus::Done;
        app.process_kept_run_items();
        assert!(app.batch.items[0].job.is_none() && !app.processor.is_colorize_in_flight(), "nothing runs");
    }

    /// An upscale that would not fit in free memory is refused, saying so.
    #[test]
    fn an_upscale_too_large_for_free_memory_is_refused() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(super::too_large_to_upscale((4000, 3000), 500, 8 * gb), None, "12 MP: about 0.7 GB");
        let why = super::too_large_to_upscale((16000, 12000), 500, 2 * gb).expect("192 MP: about 4 GB");
        assert!(why.contains("16000\u{00d7}12000") && why.contains("free"), "{why}");
    }

    #[test]
    fn an_upscale_can_start_for_ticked_images_and_waits_for_the_slot() {
        let mut app = two_images(SettingsModel::RealEsrganUpscale);
        assert!(app.can_process_intent(), "the viewed image");
        app.batch.items[0].ticked = true;
        app.batch.items[1].ticked = true;
        assert!(app.can_process_intent());
        app.processor.set_upscale_in_flight_for_test(true);
        assert!(!app.can_process_intent(), "one at a time");
        app.processor.queue_runs(app.batch.items_to_process());
        app.dispatch_next_queued_run();
        app.processor.set_upscale_in_flight_for_test(false);
        assert_eq!(app.processor.next_queued_run(), Some(1), "the queue waited for the slot");
    }
}
