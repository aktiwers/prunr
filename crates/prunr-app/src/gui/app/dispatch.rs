//! Sending work out: Process runs, tier routing, admission, and the upscale / colorize / erase dispatches.

use std::collections::HashSet;
use std::sync::mpsc;

use crate::gui::item::{BatchItem, BatchStatus, Engine, ImageSource};
use crate::gui::timeline::StepKind;
use crate::gui::worker::WorkerMessage;
use super::PrunrApp;

impl PrunrApp {
    /// The target images an erase runs on (those with a painted region),
    /// and how many targets have none.
    pub(super) fn erase_targets(&self) -> (Vec<usize>, usize) {
        let mut without = 0;
        let with_region = self.batch.targeted_indices().into_iter().filter(|&idx| {
            let has = self.batch.items[idx].selection_mask.as_ref().is_some_and(|m| m.has_selected_region());
            if !has { without += 1; }
            has
        }).collect();
        (with_region, without)
    }

    /// Start the next queued upscale or colorize when the slot is free,
    /// with the shown image's settings, as a batch of cut-outs runs.
    pub(super) fn dispatch_next_queued_run(&mut self) {
        let Some(item_id) = self.processor.next_queued_run() else { return };
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &HashSet::from([item_id]));
        }
        if self.settings.model.is_colorize() {
            self.dispatch_colorize_for(item_id);
        } else {
            self.dispatch_upscale_for(item_id);
        }
    }

    /// Upscale branch of `handle_process_intent`. Resolves the input image
    /// from the timeline (`BatchItem::run_input_or_source`), then hands off
    /// to `Processor::dispatch_upscale`. Must only be called after
    /// `can_process_intent()` confirms the model is installed and idle.
    fn dispatch_upscale_for(&mut self, item_id: u64) {
        let Some(item) = self.batch.find_by_id(item_id) else { return };
        let model_id = match self.settings.model.to_model_id() {
            Some(id) => id,
            None => return,
        };
        let input = item.run_input_or_source(Engine::Upscale);
        let output_scale = item.settings.output_scale;
        // Capture the recipe at dispatch time so the pump can stamp
        // `item.applied_recipe` against what actually ran (not what the
        // user has tweaked to since).
        let model_kind = match prunr_core::ModelKind::try_from(model_id) {
            Ok(k) => k,
            Err(id) => {
                tracing::error!(?id, "upscale dispatch skipped: ModelId has no ModelKind mapping");
                return;
            }
        };
        let recipe = item.settings.current_recipe(model_kind);
        let run_model = prunr_core::upscale::model_for_scale(model_id, output_scale, self.hardware_install_cache.x2plus);
        tracing::info!(item_id, ?model_id, ?run_model, ?output_scale, "upscale dispatched");
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        }
        self.processor.dispatch_upscale(item_id, input, run_model, output_scale, self.settings.cpu_only(), recipe);
    }

    /// Colorize branch of `handle_process_intent`: the selected image, from
    /// where its timeline says a run starts, to `Processor::dispatch_colorize`.
    fn dispatch_colorize_for(&mut self, item_id: u64) {
        let Some(item) = self.batch.find_by_id(item_id) else { return };
        let input = item.run_input_or_source(Engine::Colorize);
        let recipe = item.settings.current_recipe(crate::gui::settings::COLORIZER);
        tracing::info!(item_id, "colorize dispatched");
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.start_job(Some(StepKind::Colorize), Engine::Colorize);
        }
        self.processor.dispatch_colorize(item_id, input, self.settings.cpu_only(), recipe);
    }

    pub(crate) fn dispatch_inpaint_for_item(&mut self, idx: usize) {
        let item = &self.batch.items[idx];
        let item_id = item.id;
        // selection_mask is the single source of truth for the inpaint
        // region. It persists across Process, so Reprocess is available
        // whenever any selection exists — no separate "last_inpaint" cache.
        let Some(selection) = item.selection_mask.as_ref().cloned() else {
            tracing::debug!(item_id, "inpaint dispatch skipped: no selection_mask");
            return;
        };
        // Stack-based inpaint: each stroke runs against the previous
        // result so earlier strokes stay intact. source_for_inpaint
        // walks result_rgba → source_rgba → source_dyn, the last arm
        // handling memory-pressure eviction.
        let Some(source) = item.source_for_inpaint() else {
            tracing::warn!(item_id, "inpaint dispatch skipped: source RGBA unavailable");
            return;
        };
        let raw_backend = self.erase_backend();
        let seed = raw_backend.is_sd_family()
            .then(|| self.settings.brush.sd_seed.unwrap_or_else(prunr_core::inpaint_sd::fresh_seed));
        tracing::info!(item_id, ?raw_backend, ?seed, "inpaint stroke committed; dispatching");
        let tuning = self.erase_tuning(seed);
        self.processor.dispatch_inpaint(item_id, source, selection, tuning);
        self.batch.items[idx].start_erase(seed);
    }

    /// The eraser model the Model picker names (LaMa without one).
    fn erase_backend(&self) -> prunr_models::ModelId {
        self.settings.model.to_model_id().unwrap_or(prunr_models::ModelId::LaMaFp32)
    }

    /// How a fill runs under the current settings, with `seed` for a
    /// generated one.
    pub(super) fn erase_tuning(&mut self, seed: Option<u64>) -> crate::gui::processor::InpaintTuning {
        self.settings.fit_sd_quality_to_install(self.hardware_install_cache.lcm_bundle);
        let raw_backend = self.erase_backend();
        let backend = if self.settings.lcm_routing_active(raw_backend) {
            prunr_models::ModelId::SdV15LcmInpaintFp16
        } else {
            raw_backend
        };
        let bs = &self.settings.brush;
        crate::gui::processor::InpaintTuning {
            sharpen: bs.inpaint_sharpen,
            feather_px: bs.inpaint_feather,
            grow_px: bs.inpaint_grow,
            fill_scale: bs.inpaint_fill_scale,
            backend,
            sd_prompt: bs.sd_prompt_for_run(),
            sd_negative_prompt: bs.sd_negative_prompt.clone(),
            sd_guidance_scale: bs.sd_guidance_scale,
            sd_scheduler: bs.sd_scheduler,
            sd_steps: bs.sd_steps,
            sd_seed: seed,
            use_taesd: bs.sd_use_taesd_effective(),
            sd_tuning: crate::gui::env_overrides::sd_tuning(prunr_core::inpaint_sd::plan_tuning(
                self.settings.keep_sd_loaded,
                self.settings.sd_memory.margin_mb(),
            )),
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
        // No-model-but-EdgesOnly falls through with a placeholder. The
        // subprocess Init still receives this field; `needs_segmentation`
        // is false for EdgesOnly so the seg model never loads.
        let model: prunr_core::ModelKind = self.settings.model
            .to_model_kind()
            .unwrap_or(prunr_core::ModelKind::BiRefNetLite);

        let candidate_ids: HashSet<u64> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }
        // A background image's result is released; a run may start from it.
        for item in self.batch.items.iter_mut().filter(|i| candidate_ids.contains(&i.id)) {
            item.unpark_result();
        }

        let tiers = self.classify_candidates(&candidate_ids, model);
        let process_count = tiers.tier1.len() + tiers.tier2.len() + tiers.tier_add_edge.len();
        self.notify_skip(tiers.skip_count, process_count);
        if process_count == 0 { return; }

        for id in tiers.all_process_ids() {
            if let Some(item) = self.batch.find_by_id_mut(id) {
                item.start_job(step, Engine::Segment);
            }
        }

        let tier2_work = self.build_tier2_work(&tiers.tier2);
        let add_edge_work = self.build_add_edge_work(&tiers.tier_add_edge);

        let jobs = self.settings.parallel_jobs.min(crate::gui::memory::safe_max_jobs(model));
        if tiers.tier1.len() > 1 {
            self.dispatch_with_admission(&tiers.tier1, tier2_work, add_edge_work, model, jobs);
        } else {
            self.dispatch_small_batch(&tiers.tier1, tier2_work, add_edge_work, model, jobs);
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
            item.start_job(step, Engine::Segment);
            let input = item.job_input();
            self.batch.request_filter_only(id, &source, input, settings.fill_style);
        }
    }

    fn build_add_edge_work(&mut self, ids: &HashSet<u64>) -> Vec<crate::gui::worker::AddEdgeWorkItem> {
        let mut out = Vec::new();
        for item in &mut self.batch.items {
            if !ids.contains(&item.id) { continue; }
            let Some(ref ct) = item.cached_tensor else { continue };
            let tensor_data = ct.decompress();
            let mask = item.settings.mask_settings();
            match (tensor_data, item.source.load_bytes()) {
                (Some(data), Ok(bytes)) => {
                    out.push(crate::gui::worker::AddEdgeWorkItem {
                        item_id: item.id,
                        tensor_data: data,
                        tensor_height: ct.height,
                        tensor_width: ct.width,
                        model: ct.model,
                        original_bytes: bytes,
                        mask,
                    });
                    item.status = BatchStatus::Processing;
                }
                (None, _) => {
                    item.status = BatchStatus::Error("Seg tensor cache corrupt".into());
                    item.set_cached_tensor(None);
                    item.applied_recipe = None;
                }
                (_, Err(e)) => {
                    item.status = BatchStatus::Error(format!("Failed to load: {e}"));
                }
            }
        }
        out
    }

    /// Classify each candidate into Tier 1 (full pipeline), Tier 2 (mask
    /// rerun from cached tensor), or Skip (already up to date).
    /// Mutates items in-place: invalidates stale caches via the catalog,
    /// syncs composite-only recipe changes, and never downgrades Tier 2
    /// items without a tensor.
    pub(super) fn classify_candidates(
        &mut self,
        candidate_ids: &HashSet<u64>,
        model: prunr_core::ModelKind,
    ) -> ClassifiedTiers {
        use crate::gui::knob_catalog;
        use prunr_core::RequiredTier;
        let mut tiers = ClassifiedTiers::default();

        for item in &mut self.batch.items {
            if !candidate_ids.contains(&item.id) { continue; }

            // Never-processed items always need the full pipeline.
            let Some(ref old_recipe) = item.applied_recipe else {
                tiers.tier1.insert(item.id);
                continue;
            };

            // A run that starts from an earlier result works on pixels the
            // cached tensor never saw, unless it redoes the shown cut-out
            // from that cut-out's own starting image.
            if item.starts_from_result(Engine::Segment) {
                if !item.redoes_shown_step(Engine::Segment) {
                    item.set_cached_tensor(None);
                }
                tiers.tier1.insert(item.id);
                continue;
            }

            let current_recipe = item.settings.current_recipe(model);
            let tier = prunr_core::resolve_tier(old_recipe, &current_recipe);
            let impact = knob_catalog::cache_impact_for_recipe_diff(old_recipe, &current_recipe);
            item.apply_cache_impact(impact);

            match tier {
                RequiredTier::Skip
                | RequiredTier::CompositeOnly
                | RequiredTier::UpscaleRerun => {
                    // CompositeOnly (bg_color) and UpscaleRerun re-render at
                    // display/export time, not via the seg/edge dispatcher;
                    // sync the stored composite so status reads stay accurate.
                    if let Some(ref mut recipe) = item.applied_recipe {
                        recipe.composite = current_recipe.composite.clone();
                    }
                    tiers.skip_count += 1;
                }
                RequiredTier::UpscaleTier2 | RequiredTier::ColorizeTier2 => {
                    // Tier-2 upscale (sharpen/ai_blend/color_match)
                    // dispatches the cached-buffer postprocess via live preview
                    // — no subprocess admission, no re-inference.
                    if let Some(ref mut recipe) = item.applied_recipe {
                        recipe.composite = current_recipe.composite.clone();
                    }
                    tiers.skip_count += 1;
                }
                RequiredTier::MaskRerun => {
                    if item.cached_tensor.is_some() {
                        // SubjectOutline mask rerun needs TEED re-composite
                        // on the new masked base; Tier 2 RePostProcess only
                        // runs the mask side, which would drop the outline.
                        if current_recipe.inference.uses_edge_detection {
                            tiers.tier_add_edge.insert(item.id);
                        } else {
                            tiers.tier2.insert(item.id);
                        }
                    } else {
                        tiers.tier1.insert(item.id);
                    }
                }
                RequiredTier::EdgeRerun => {
                    // Edge rerun via the subprocess isn't wired for batch
                    // dispatch — fall through to a full pipeline run (live
                    // preview handles the in-process path).
                    tiers.tier1.insert(item.id);
                }
                RequiredTier::AddEdgeInference => {
                    if item.cached_tensor.is_some() {
                        tiers.tier_add_edge.insert(item.id);
                    } else {
                        tiers.tier1.insert(item.id);
                    }
                }
                RequiredTier::FullPipeline => {
                    tiers.tier1.insert(item.id);
                }
            }
        }
        tiers
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

    fn build_tier2_work(&mut self, tier2_ids: &HashSet<u64>) -> Vec<crate::gui::worker::Tier2WorkItem> {
        let mut out = Vec::new();
        for item in &mut self.batch.items {
            if !tier2_ids.contains(&item.id) { continue; }
            let Some(ref ct) = item.cached_tensor else { continue };
            let tensor_data = ct.decompress();
            let mask = item.settings.mask_settings();
            match (tensor_data, item.source.load_bytes()) {
                (Some(data), Ok(bytes)) => {
                    out.push(crate::gui::worker::Tier2WorkItem {
                        item_id: item.id,
                        tensor_data: data,
                        tensor_height: ct.height,
                        tensor_width: ct.width,
                        model: ct.model,
                        original_bytes: bytes,
                        mask,
                    });
                    item.status = BatchStatus::Processing;
                }
                (None, _) => {
                    item.status = BatchStatus::Error("Tensor cache corrupt".into());
                    item.set_cached_tensor(None);
                    item.applied_recipe = None;
                }
                (_, Err(e)) => {
                    item.status = BatchStatus::Error(format!("Failed to load: {e}"));
                    item.set_cached_tensor(None);
                    item.applied_recipe = None;
                }
            }
        }
        out
    }

    /// Dispatch with streaming admission — used when >1 Tier 1 items so the
    /// batch doesn't blow past memory limits. Admits what fits now; the
    /// worker bridge receives more items via `admission_tx` as earlier items
    /// complete and free memory.
    fn dispatch_with_admission(
        &mut self,
        tier1_ids: &HashSet<u64>,
        tier2_work: Vec<crate::gui::worker::Tier2WorkItem>,
        add_edge_work: Vec<crate::gui::worker::AddEdgeWorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
    ) {
        use crate::gui::memory::{AdmissionController, ImageMemCost};

        let mut ctrl = AdmissionController::new(model, jobs);
        let costs: Vec<ImageMemCost> = self.batch.items.iter()
            .filter(|i| tier1_ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(
                i.id, i.dimensions, i.source.estimated_size(), crate::gui::memory::RESIDENT_HISTORY_STEPS,
            ))
            .collect();
        ctrl.enqueue(costs);

        let mut initial_items = Vec::new();
        while let Some(admitted_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(admitted_id) {
                if let Ok(bytes) = item.source.load_bytes() {
                    let input = item.job_input();
                    initial_items.push((item.id, bytes, input));
                    item.status = BatchStatus::Processing;
                }
            }
        }

        for item in &mut self.batch.items {
            if tier1_ids.contains(&item.id) && item.status != BatchStatus::Processing {
                item.status = BatchStatus::Pending;
            }
        }

        // All Tier 1 items failed load_bytes AND no Tier 2 / add-edge work —
        // skip dispatch.
        if initial_items.is_empty() && tier2_work.is_empty() && add_edge_work.is_empty() { return; }

        let (atx, arx) = mpsc::channel();
        self.processor.admission = Some(ctrl);
        self.processor.admission_tx = Some(atx);
        self.dispatch_batch(initial_items, tier2_work, add_edge_work, model, jobs, Some(arx));
    }

    /// Single-item or small-batch fast path — skip admission entirely.
    fn dispatch_small_batch(
        &mut self,
        tier1_ids: &HashSet<u64>,
        tier2_work: Vec<crate::gui::worker::Tier2WorkItem>,
        add_edge_work: Vec<crate::gui::worker::AddEdgeWorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
    ) {
        let items: Vec<_> = self.batch.items.iter_mut()
            .filter(|i| tier1_ids.contains(&i.id))
            .filter_map(|i| {
                let bytes = i.source.load_bytes().ok()?;
                let input = i.job_input();
                i.status = BatchStatus::Processing;
                Some((i.id, bytes, input))
            })
            .collect();

        // If all prep failed, don't dispatch an empty batch.
        if items.is_empty() && tier2_work.is_empty() && add_edge_work.is_empty() { return; }

        self.dispatch_batch(items, tier2_work, add_edge_work, model, jobs, None);
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
        tier2_items: Vec<crate::gui::worker::Tier2WorkItem>,
        add_edge_items: Vec<crate::gui::worker::AddEdgeWorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        additional_items_rx: Option<mpsc::Receiver<crate::gui::worker::WorkItem>>,
    ) {
        self.processor.cancels.reset();

        let current_settings = self.run_settings();

        // Broadcast: every item about to be processed inherits current.settings
        // so their `applied_recipe` ends up consistent with what ran.
        let process_ids: std::collections::HashSet<u64> = items.iter()
            .map(|wi| wi.0)
            .chain(tier2_items.iter().map(|ti| ti.item_id))
            .chain(add_edge_items.iter().map(|ai| ai.item_id))
            .collect();
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &process_ids);
        }

        let recipe = current_settings.current_recipe(model);
        self.processor.track_dispatch(recipe, process_ids.iter().copied());

        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.processor.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            tier2_items,
            add_edge_items,
            config: Box::new(crate::gui::worker::ProcessingConfig {
                model,
                jobs,
                mask: current_settings.mask_settings(),
                force_cpu: self.settings.cpu_only(),
                line_mode: current_settings.line_mode,
                edge: current_settings.edge_settings(),
            }),
            cancels: self.processor.cancels.clone(),
            additional_items_rx,
        });
    }

    /// Release an item's memory budget and greedily admit the next fitting items.
    pub(super) fn admission_release_and_admit(&mut self, completed_id: u64) {
        let Some(ref mut ctrl) = self.processor.admission else { return; };
        ctrl.release(completed_id);

        let mut streamed_ids = Vec::new();
        while let Some(next_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(next_id) {
                let Ok(bytes) = item.source.load_bytes() else { continue; };
                let input = item.job_input();
                let tuple = (next_id, bytes, input);
                item.status = BatchStatus::Processing;

                if let Some(ref tx) = self.processor.admission_tx {
                    if tx.send(tuple).is_err() {
                        break; // worker gone
                    }
                    streamed_ids.push(next_id);
                }
            }
        }

        let admission_complete = ctrl.is_complete();
        // ctrl borrow ends here so we can re-borrow processor.
        for id in streamed_ids {
            self.processor.track_streamed(id);
        }
        if admission_complete {
            self.processor.clear_admission();
        }
    }
}

/// Classification output from `classify_candidates`: which items go to the
/// full subprocess pipeline (tier1), which can do an in-subprocess mask
/// rerun from their cached tensor (tier2), and how many were already
/// up-to-date (skip_count, used only for the user-facing toast).
#[derive(Default)]
pub(super) struct ClassifiedTiers {
    pub(super) tier1: HashSet<u64>,
    tier2: HashSet<u64>,
    /// AddEdgeInference items: seg tensor cached, run TEED only.
    pub(super) tier_add_edge: HashSet<u64>,
    skip_count: usize,
}

impl ClassifiedTiers {
    /// Union of tier1 + tier2 + add-edge — the set of items that will actually
    /// be reprocessed this batch (history seeding, progress counting).
    fn all_process_ids(&self) -> HashSet<u64> {
        self.tier1.iter()
            .chain(self.tier2.iter())
            .chain(self.tier_add_edge.iter())
            .copied().collect()
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
        app.batch.items[1].selected = true;
        assert!(app.can_process_intent(), "a ticked image has one");
        assert_eq!(app.erase_targets(), (vec![1], 0));
        app.batch.items[0].selected = true;
        assert_eq!(app.erase_targets(), (vec![1], 1), "the ticked one without a region is counted");
    }

    #[test]
    fn an_upscale_can_start_for_ticked_images_and_waits_for_the_slot() {
        let mut app = two_images(SettingsModel::RealEsrganUpscale);
        assert!(app.can_process_intent(), "the viewed image");
        app.batch.items[0].selected = true;
        app.batch.items[1].selected = true;
        assert!(app.can_process_intent());
        app.processor.set_upscale_in_flight_for_test(true);
        assert!(!app.can_process_intent(), "one at a time");
        app.processor.queue_runs(app.batch.items_to_process());
        app.dispatch_next_queued_run();
        app.processor.set_upscale_in_flight_for_test(false);
        assert_eq!(app.processor.next_queued_run(), Some(1), "the queue waited for the slot");
    }
}
