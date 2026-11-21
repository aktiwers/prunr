//! Live preview: building each preview's inputs, landing its results, and the quiet-work pill.

use std::sync::Arc;

use crate::gui::item::{BatchItem, Engine, KeptRunTier, Pixels};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::background_io::TexTarget;
use crate::gui::live_preview::PreviewKind;
use super::{PrunrApp, BUSY_PILL_AFTER};

/// What brings a picture that drifted from its settings back in line.
enum Drift {
    /// A live render.
    Render(PreviewKind),
    /// A run of the model, as a knob change that needs it starts one.
    Run,
}

impl PrunrApp {
    /// Background work with nothing on the canvas to show for it: a live
    /// render, or a texture, thumbnail, decode or backdrop a thread owes
    /// an image. Model runs show their own capsule.
    pub(crate) fn quiet_background_work(&self) -> Option<&'static str> {
        if self.processor.live_preview.is_busy() {
            Some("Rendering\u{2026}")
        } else if self.batch.has_pending_work() {
            Some("Preparing\u{2026}")
        } else {
            None
        }
    }

    /// Each frame: when quiet work started, if any is under way. A result
    /// landing restarts the clock, so a live drag whose renders keep up
    /// never shows the pill.
    pub(crate) fn note_quiet_work(&mut self, now: std::time::Instant) {
        if self.quiet_background_work().is_some() {
            self.busy_since.get_or_insert(now);
        } else {
            self.busy_since = None;
        }
    }

    /// The pill's words once quiet work has gone on for `BUSY_PILL_AFTER`;
    /// none while a run's capsule or Magic Brush's own pill shows.
    pub(crate) fn busy_pill(&self, now: std::time::Instant) -> Option<&'static str> {
        let long_enough = self.busy_since.is_some_and(|since| now.duration_since(since) >= BUSY_PILL_AFTER);
        let shown_otherwise = self.processor.dispatch_progress().is_some()
            || (self.magic_brush_state.is_active() && self.magic_brush_preparing());
        if !long_enough || shown_otherwise {
            return None;
        }
        self.quiet_background_work()
    }

    /// What brings the picture of the image at `idx` back in line with
    /// its settings, if it drifted: a live render for a cut-out's applied
    /// mask against the mask its settings make, or for a kept upscale's or
    /// colorize's Refine or Strength moved past its picture; a run of the
    /// model for a look that needs it (another Sketch mode), as a knob
    /// change on a processed cut-out runs it. `None` when in line, or when
    /// nothing could land or is wanted (a re-cut the render door refuses; a
    /// cut another model made, which stays as it is until Process; a look
    /// whose run was cancelled, which waits for Process).
    fn drift(&self, idx: usize) -> Option<Drift> {
        use prunr_core::RequiredTier;
        let item = &self.batch.items[idx];
        // Only a cut-out or the filters apply the mask settings; firing a
        // mask preview over an erase or an upscale would clobber it with
        // the source through `run_preview`'s no-tensor fallback.
        if !shows(self.settings.model, Surface::Effect) {
            return self.wants_kept_run_refinish(idx).map(Drift::Render);
        }
        let applied = item.applied_recipe.as_ref()?;
        let kind = self.settings.model.recipe_kind();
        // The mask as the run applies it: pinned to the defaults where the
        // run uses none (Sketch on Full), so its settings never drift.
        let current = item.settings.current_recipe(kind);
        let role = self.settings.model.role();
        if applied.inference.model == kind && matches!(prunr_core::resolve_tier(applied, &current), RequiredTier::FullPipeline | RequiredTier::AddEdgeInference) {
            let may_run = item.is_idle() && !item.timeline.current().look_cancelled && item.verdict(crate::gui::capability::Need::Processed, role).allows_request();
            return may_run.then_some(Drift::Run);
        }
        let recut = item.verdict(crate::gui::capability::Need::Cutout, role);
        (applied.mask != current.mask && recut.allows_request()).then_some(Drift::Render(PreviewKind::Mask))
    }

    /// Re-render the image at `idx` live when its picture drifted from its
    /// settings and no render for it is under way: a run landed, on screen
    /// or off, after its knobs moved (the move's queued render goes when
    /// the image leaves the screen).
    pub(crate) fn render_drift(&mut self, idx: usize) {
        let id = self.batch.items[idx].id;
        if let Some(Drift::Render(kind)) = self.drift(idx).filter(|_| !self.processor.live_preview.works_on(id)) {
            self.processor.live_preview.render_now(id, kind);
        }
    }

    /// An undo or redo moved the image at `idx` to a step whose own render
    /// or run was dropped as the image moved on: it renders again, and a
    /// look that needs the model runs again for the image on screen, whose
    /// look a run takes.
    pub(crate) fn catch_up_step(&mut self, idx: usize) {
        let id = self.batch.items[idx].id;
        match self.drift(idx) {
            Some(Drift::Render(_)) => self.render_drift(idx),
            Some(Drift::Run) if self.batch.is_selected(id) => self.process_items(|item| item.id == id, None),
            _ => {}
        }
    }

    /// The drift check on the image on screen, every frame: a safety net
    /// for edits no knob dispatched (brush commits, hotkeys). Asked once
    /// per step and settings: a render refused for memory never lands. It
    /// renders only; a run waits for an undo or redo to the step, or for
    /// Process, so a Cancel stays a Cancel.
    pub(crate) fn recipe_drift_tripwire(&mut self) {
        if self.processor.live_preview.has_in_flight() {
            return;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let id = item.id;
        let checked_as = (id, item.timeline.revision(), item.settings);
        if self.last_drift_check == Some(checked_as) {
            return;
        }
        let mut checked = true;
        match self.drift(idx) {
            Some(Drift::Render(kind)) => {
                let live = &mut self.processor.live_preview;
                if live.is_pending_for(id) {
                    // A drag under way: its queued frame goes now instead of
                    // at the throttle, and the drag settles on release or
                    // stillness, not here.
                    live.hurry(id);
                    checked = false;
                } else if !live.works_on(id) {
                    tracing::debug!(item_id = id, ?kind, "recipe-drift tripwire fired — dispatching Tier-2");
                    live.render_now(id, kind);
                } else {
                    checked = false;
                }
            }
            // Only an undo or redo that arrives on the step runs it again: a
            // look whose run was cancelled keeps waiting for Process.
            Some(Drift::Run) | None => {}
        }
        self.last_drift_check = checked.then_some(checked_as);
    }

    /// Live-preview pump: dispatch debounced Tier 2 reruns + apply completed ones.
    pub(super) fn pump_live_preview(&mut self, ctx: &egui::Context) {
        self.recipe_drift_tripwire();

        let results = self.processor.live_preview.drain_results();
        if !results.is_empty() {
            self.apply_completed_previews(ctx, results);
        }
        // Named: off-screen images refuse too, each its own toast.
        for (id, why) in self.processor.live_preview.take_refusals() {
            if let Some(item) = self.batch.find_by_id(id) {
                self.toasts.warning(format!("{}: {}", item.filename, prunr_core::plain::reason(why)));
            }
        }
        // Ticked after the landing: a landed frame frees its image for the
        // next, which builds on the caches it left. The closure borrows
        // `batch.items` mutably so `build_preview_inputs` can hold the
        // run's start on the item for the drag session.
        let role = self.settings.model.role();
        let action = self.settings.brush.cutout_action;
        let batch = &mut self.batch;
        let shown = batch.selected_id();
        let wait = self.processor.live_preview.tick(shown, |id, kind| {
            let item = batch.find_by_id_mut(id).ok_or(crate::gui::capability::Verdict::Never(crate::gui::capability::Reason::Removed))?;
            Self::build_preview_inputs(item, kind, role, action)
        });
        // If a future dispatch is waiting, schedule a repaint when the
        // debounce elapses so tick() can fire on its own.
        if let Some(w) = wait {
            ctx.request_repaint_after(w);
        }

        // A render due for an image whose photo is not decoded (a ticked
        // image off screen, its map just measured) waits for the decode;
        // nothing else asks for it, and the render would wait for good with
        // the busy pill up.
        for id in self.processor.live_preview.take_awaiting_decode() {
            if let Some(idx) = self.batch.index_of(id) {
                self.ensure_source_decoded(idx);
            }
        }

        let live = &self.processor.live_preview;
        for item in &mut self.batch.items {
            item.quick.settle(live.works_on(item.id));
        }

        // Covers the worker-running / not-yet-drained window.
        if self.processor.live_preview.has_in_flight() {
            ctx.request_repaint_after(super::ASYNC_POLL);
        }
    }

    /// Snapshot the dispatch inputs for one preview job: original RGBA +
    /// decompressed tensors + settings + reusable edge mask if its
    /// line_strength still matches. Returns `None` to abort the dispatch
    /// when a required tensor cache isn't available (user must Process first).
    /// The photo goes over as its `Arc`: no copy.
    pub(crate) fn build_preview_inputs(
        item: &mut BatchItem,
        kind: PreviewKind,
        role: crate::gui::knob_catalog::ModelRole,
        action: crate::gui::brush_state::CutoutAction,
    ) -> Result<crate::gui::live_preview::DispatchInputs, crate::gui::capability::Verdict> {
        use crate::gui::capability::{check, ItemFacts, Need, Reason, Verdict};
        use crate::gui::knob_catalog::ModelRole;
        use crate::gui::live_preview::{DispatchInputs, PreviewKind};
        let facts = ItemFacts::of(item);
        let need = |need| match check(&facts, need, role) {
            Verdict::Ready => Ok(()),
            refused => Err(refused),
        };
        let version = item.timeline.version();
        // A Refine render needs only the kept upscale run and the knobs.
        if matches!(kind, PreviewKind::UpscaleTier2) {
            need(Need::KeptUpscale)?;
            let upscale = item.kept_upscale().ok_or(Verdict::Never(Reason::UpscaleFirst))?.clone();
            return Ok(DispatchInputs {
                upscale: Some(upscale),
                faces: item.faces_input(crate::gui::item::FaceStage::Upscale),
                // `original` is unread by a Refine preview, which re-finishes the kept run.
                ..DispatchInputs::new(kind, version, Arc::new(image::RgbaImage::new(0, 0)), item.settings)
            });
        }
        if matches!(kind, PreviewKind::ColorizeTier2) {
            need(Need::KeptColorize)?;
            let colorize = item.kept_colorize().ok_or(Verdict::Never(Reason::ColorizeFirst))?.clone();
            // `original` is unread: Strength re-renders the kept run.
            return Ok(DispatchInputs {
                colorize: Some(colorize),
                ..DispatchInputs::new(kind, version, Arc::new(image::RgbaImage::new(0, 0)), item.settings)
            });
        }
        need(Need::PhotoRender)?;
        need(Need::ShownImage)?;
        // A photo render without a scale the lines draw would show the
        // image without those lines: it waits for the worker's spare
        // scales, or for a new run when none are coming.
        if kind != PreviewKind::Photo && item.lacks_drawn_scales() {
            need(Need::DrawnLines)?;
        }
        // A Mask render re-cuts the cut-out on screen; without one it would
        // publish the plain photo as a result. No model's photo is its own.
        let is_filter_only = role == ModelRole::Filters;
        if matches!(kind, PreviewKind::Mask) {
            need(Need::Cutout)?;
        }
        let (seg_tensor, packed_seg) = item.render_seg();
        let seg_model = seg_tensor.as_ref().map(|s| s.model).or_else(|| packed_seg.as_ref().map(|c| c.model));
        if matches!(kind, PreviewKind::Mask) && !is_filter_only && seg_model.is_none() {
            return Err(Verdict::Never(Reason::ProcessAgain));
        }
        // The preview starts where Process would, keeping an earlier erase or
        // hand edit; the start is held on the item while previews read it, so
        // it is not packed away between ticks.
        let start = item.run_start(Engine::Segment);
        let (original, original_from) = match start.as_ref() {
            Some(s) => match s.loaded() {
                Some(rgba) => {
                    item.start_photo = Some(Arc::clone(&rgba));
                    (rgba, None)
                }
                None => (Arc::new(image::RgbaImage::new(0, 0)), Some(Arc::clone(s))),
            },
            None => {
                item.start_photo = None;
                let undecoded = match facts.photo {
                    crate::gui::capability::Photo::Unreadable => Verdict::Never(Reason::Unreadable),
                    _ => Verdict::Wait(Reason::Decoding),
                };
                (Arc::clone(item.source_rgba.as_ref().ok_or(undecoded)?), None)
            }
        };
        // The line scales the render reads: the active one (hot from the
        // drag's last tick, else unpacked on the render's thread) and,
        // for a dual-scale style, Bold.
        let primary = item.settings.edge_settings().primary_scale();
        let edges = item.cached_edge_tensors.clone();
        let has_scale = |scale| edges.as_ref().is_some_and(|e| e.has_all(&[scale]));
        let edge_tensor = item.volatile_edge_tensor.as_ref()
            .filter(|(scale, _)| *scale == primary)
            .zip(edges.as_ref())
            .map(|((_, data), e)| {
                let (height, width) = e.dims(primary);
                crate::gui::live_preview::EdgeTensor { data: Arc::clone(data), height, width }
            });
        let packed_primary = (edge_tensor.is_none() && has_scale(primary)).then_some(primary);
        let bold = item.settings.draws_bold_lines() && has_scale(prunr_core::EdgeScale::Bold);
        match kind {
            // Mask without a seg tensor is the filter-only path (effects on
            // the raw source); Edge has nothing to preview without a tensor.
            PreviewKind::Edge if edge_tensor.is_none() && packed_primary.is_none() => {
                return Err(match need(Need::Idle) {
                    Ok(()) => Verdict::Never(Reason::LinesNotRun),
                    Err(wait) => wait,
                });
            }
            _ => {}
        }
        let has_depth = item.depth.is_some();
        let plane_key = |scale: prunr_core::EdgeScale| crate::gui::live_preview::EdgePlaneKey::new(&item.settings, scale, has_depth);
        let cached_edge = item.edge_planes.lookup(plane_key(item.settings.edge_settings().primary_scale()));
        let cached_bold = item.edge_planes.lookup(plane_key(prunr_core::EdgeScale::Bold));
        let cached_masked_base = item.cached_masked_base.as_ref().and_then(|(base, recipe, model)| {
            let current_recipe = prunr_core::MaskRecipe::from(&item.settings.mask_settings());
            let seg_model_match = seg_model == Some(*model);
            (*recipe == current_recipe && seg_model_match).then(|| base.clone())
        });
        // Without a seg tensor the correction has nothing to apply to
        // (filter-only path).
        let correction = seg_model.and_then(|_| item.correction(action));
        let faces = item.faces_input(crate::gui::item::FaceStage::Photo);
        let depth = Self::depth_input(item, &original, faces.as_ref(), role);
        Ok(DispatchInputs {
            start,
            original_from,
            seg_tensor, edge_tensor,
            packed: crate::gui::live_preview::Packed {
                seg: packed_seg,
                edges: edges.filter(|_| packed_primary.is_some() || bold),
                primary: packed_primary,
                bold,
            },
            cached_edge, cached_bold, cached_masked_base,
            kept_alpha: item.kept_alpha.clone(),
            correction,
            faces,
            depth,
            quick_photo: item.quick_photo.clone(),
            ..DispatchInputs::new(kind, version, original, item.settings)
        })
    }

    /// The depth map a photo preview reads, with the Depth chip's look on
    /// No model and the looked photo kept from last time when it was made
    /// from the same photo (the faced one, when faces go in first).
    fn depth_input(
        item: &BatchItem,
        original: &Arc<image::RgbaImage>,
        faces: Option<&crate::gui::live_preview::FacesInput>,
        role: crate::gui::knob_catalog::ModelRole,
    ) -> Option<crate::gui::live_preview::DepthInput> {
        let map = item.depth.clone()?;
        let look = item.settings.photo_depth_look(role);
        let base = match faces {
            None => Some(Arc::clone(original)),
            Some(f) => f.faced.clone(),
        };
        let photo = look.zip(base).and_then(|(look, base)| {
            let want = crate::gui::item::DepthPhotoKey { base, depth: Arc::clone(&map), look };
            item.depth_photo.as_ref().filter(|(_, k)| k.matches(&want)).map(|(p, _)| Arc::clone(p))
        });
        Some(crate::gui::live_preview::DepthInput { map, look, photo })
    }

    /// Do NOT null `result_texture` here: the old texture stays until the new
    /// one lands via `drain_background_channels`, or the canvas flashes black
    /// for a frame.
    pub(crate) fn apply_completed_previews(
        &mut self,
        ctx: &egui::Context,
        results: Vec<crate::gui::live_preview::PreviewResult>,
    ) {
        let handles = self.batch.bg_io.tex_prep_handles();
        // A result the image still wants restarts the quiet-work clock; a
        // stale one changes nothing on screen.
        if results.iter().any(|r| self.batch.find_by_id(r.item_id).is_some_and(|i| r.version == i.timeline.version())) {
            self.busy_since = None;
        }
        let mut left_behind = Vec::new();
        for r in results {
            let (item_id, source, is_final, landed) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else {
                    continue;
                };
                if r.version != item.timeline.version() {
                    // Computed for a state the item has since left. Undo
                    // and redo cancel a render before it comes back, so
                    // this one was left for a new step, which may draw
                    // nothing of its own (a background pick after a
                    // stroke): it is made again for that step, unless a run
                    // under way will finish the look.
                    if item.job.is_none() {
                        let kind = match r.kind {
                            PreviewKind::Mask | PreviewKind::Edge => PreviewKind::photo_rerun(item.settings.line_mode),
                            other => other,
                        };
                        left_behind.push((item.id, kind));
                    }
                    continue;
                }
                if let Some(start) = r.new_start.clone() {
                    item.keep_loaded_start(start);
                }
                if let Some(built) = r.new_correction.clone() {
                    item.keep_built_correction(built);
                }
                item.keep_unpacked(r.unpacked.clone());
                if r.quick {
                    // Shown only: the result, its step, its recipe and its
                    // thumbnail wait for the full-size render. The faced
                    // photo is made at full size before the frame shrinks.
                    if let Some(faced) = r.new_faced {
                        item.faced_photo = Some(faced);
                    }
                    if let Some(small) = r.new_quick_photo {
                        item.quick_photo = Some(small);
                    }
                    if item.on_screen {
                        item.quick.expect(r.generation);
                        Self::spawn_tex_prep(
                            Pixels::Resident(r.rgba), item.id, Self::tex_name("quick", item.id, None),
                            TexTarget::QuickFrame(r.generation), handles.clone(), ctx.clone(),
                        );
                    }
                    continue;
                }
                if r.kind == PreviewKind::Photo {
                    if let Some(faced) = r.new_faced {
                        item.faced_photo = Some(faced);
                    }
                    if item.on_screen {
                        Self::spawn_tex_prep(
                            Pixels::Resident(r.rgba), item.id, Self::tex_name("photo-look", item.id, None),
                            TexTarget::PhotoLook, handles.clone(), ctx.clone(),
                        );
                    }
                    continue;
                }
                let new_rgba = r.rgba;
                item.result_rgba = Some(new_rgba.clone());
                // Mark pending so reconcile_selected doesn't also
                // spawn its own prep on this same frame.
                item.result_tex_pending = true;
                for (planes, key) in r.new_edge.into_iter().chain(r.new_bold) {
                    item.edge_planes.store(planes, key);
                }
                // The images these replace free off the UI thread.
                let replaced = (
                    r.new_faced.and_then(|faced| item.faced_photo.replace(faced)),
                    r.new_depth_photo.and_then(|looked| item.depth_photo.replace(looked)),
                    r.new_masked_base.and_then(|base| item.cached_masked_base.replace(base)),
                    r.new_alpha.and_then(|alpha| item.kept_alpha.replace(alpha)),
                );
                crate::gui::reclaim::later(replaced);
                if let (Some(reference), Some(run)) = (r.new_reference, item.upscale.as_mut()) {
                    run.reference = Some(reference);
                }
                // Mark the recipe applied so the drift tripwire doesn't re-fire on it.
                if let Some(applied) = item.applied_recipe.as_mut() {
                    applied.mask = r.applied_mask;
                    applied.depth = r.applied_depth;
                    if let Some(knobs) = r.applied_tier2_knobs {
                        applied.upscale.sharpen_bits = knobs.sharpen.to_bits();
                        applied.upscale.ai_blend_bits = knobs.ai_blend.to_bits();
                        applied.upscale.color_match = knobs.color_match;
                    }
                    if let Some(strength) = r.applied_color_strength {
                        applied.colorize.strength_bits = strength.to_bits();
                    }
                }
                item.amend();
                // A Refine or Strength re-render keeps its run's origin; a
                // colorize has no faces.
                let face_stage = match r.kind {
                    PreviewKind::ColorizeTier2 => None,
                    PreviewKind::UpscaleTier2 => Some(crate::gui::item::FaceStage::Upscale),
                    _ => {
                        item.record_photo_pass(r.start, r.cutter);
                        Some(crate::gui::item::FaceStage::Photo)
                    }
                };
                if let Some(stage) = face_stage {
                    let blended = item.faces_input(stage).is_some_and(|f| !f.key.faces.is_empty());
                    item.settle_faced(stage.engine(), blended);
                }
                tracing::debug!(
                    event = "preview_applied",
                    item_id = item.id,
                    kind = ?r.kind,
                    is_final = r.is_final,
                    "apply_completed_previews: result drained"
                );
                if item.on_screen {
                    let switch = self.result_switch_id;
                    let target = item.request_result_texture(true);
                    Self::spawn_tex_prep(
                        Pixels::Resident(Arc::clone(&new_rgba)), item.id, Self::tex_name("result", item.id, Some(switch)),
                        target, handles.clone(), ctx.clone(),
                    );
                    item.quick.supersede();
                } else {
                    item.quick.clear();
                    item.result_tex_pending = false;
                    item.drop_hot_tensors();
                    item.park_result();
                }
                (item.id, item.source.clone(), r.is_final, new_rgba)
            };
            // Only the drag-settled result regenerates the sidebar thumbnail,
            // not every mid-drag render.
            if is_final {
                if let Some(request) = self.batch.find_by_id_mut(item_id).map(|item| item.thumb.asked()) {
                    self.batch.request_thumbnail(item_id, request, &source, Some(crate::gui::item::Pixels::Resident(Arc::clone(&landed))));
                }
            }
        }
        let live = &mut self.processor.live_preview;
        for (item_id, kind) in left_behind {
            if !live.works_on(item_id) {
                live.render_now(item_id, kind);
            }
        }
        ctx.request_repaint();
    }

    /// How the image at `idx` re-finishes its kept run, when it changed
    /// only in what follows the model since that run: Refine's knobs
    /// after an upscale, Strength after a colorize.
    pub(super) fn wants_kept_run_refinish(&self, idx: usize) -> Option<PreviewKind> {
        let model = self.settings.model.dispatch_model_kind()?;
        let kind = if model.is_upscale() {
            PreviewKind::UpscaleTier2
        } else if self.settings.model.is_colorize() {
            PreviewKind::ColorizeTier2
        } else {
            return None;
        };
        let item = self.batch.items.get(idx)?;
        (item.kept_run_tier(model) == KeptRunTier::Refinish).then_some(kind)
    }
}

#[cfg(test)]
mod upscale_tier2_routing_tests {
    use prunr_core::{RequiredTier, resolve_tier};
    use crate::gui::item_settings::ItemSettings;
    use prunr_core::OutputScale;

    fn recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::BiRefNetLite)
    }

    fn upscale_recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus)
    }

    #[test]
    fn app_tier_upscaletier2_routes_to_mark_tweak() {
        // A diff where only sharpen changed must resolve to UpscaleTier2 so the
        // apply_toolbar_change block fires mark_tweak(PreviewKind::UpscaleTier2).
        let old_s = ItemSettings {
            output_scale: OutputScale::X4,
            sharpen: 0.0,
            ..ItemSettings::default()
        };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "sharpen-only diff must resolve to UpscaleTier2 so the live-preview gate opens",
        );
    }

    #[test]
    fn app_tier_upscale_rerun_still_gates_out_of_live_preview() {
        // A diff where output_scale changed must resolve to UpscaleRerun (Tier-1),
        // NOT UpscaleTier2 — so mark_tweak(UpscaleTier2) is NOT called.
        let old_s = ItemSettings { output_scale: OutputScale::X4, ..ItemSettings::default() };
        let new_s = ItemSettings { output_scale: OutputScale::X2, ..old_s };

        let old_recipe = recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleRerun,
            "output_scale diff must remain UpscaleRerun — Tier-1 stays gated out of live preview",
        );
    }

    /// A recipe-diff built with a model-kind
    /// mismatch (BiRefNetLite vs RealEsrganX4Plus) must NOT resolve to
    /// UpscaleTier2 — the bug was caused by the broken fallback masking
    /// upscale variants as seg models.
    #[test]
    fn tier2_chain_model_kind_mismatch_never_routes_to_upscale_tier2() {
        let old_s = ItemSettings { sharpen: 0.0, ..ItemSettings::default() };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = upscale_recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_ne!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "model-kind mismatch between applied_recipe and new_recipe must NOT \
             resolve to UpscaleTier2 — this pins that the broken path never fires mark_tweak",
        );
    }

    /// The fixed path: both recipes use RealEsrganX4Plus, and each Tier-2
    /// knob alone resolves to UpscaleTier2, the live-preview dispatch.
    #[test]
    fn tier2_chain_each_knob_individually_routes_to_upscale_tier2() {
        let base = ItemSettings {
            sharpen: 0.0,
            ai_blend: 1.0,
            color_match: false,
            ..ItemSettings::default()
        };

        let cases: &[(&str, ItemSettings)] = &[
            ("sharpen",     ItemSettings { sharpen: 0.3, ..base }),
            ("ai_blend",    ItemSettings { ai_blend: 0.7, ..base }),
            ("color_match", ItemSettings { color_match: true, ..base }),
        ];

        let old_recipe = upscale_recipe_for(&base);
        for (knob_name, new_settings) in cases {
            let new_recipe = upscale_recipe_for(new_settings);
            assert_eq!(
                resolve_tier(&old_recipe, &new_recipe),
                RequiredTier::UpscaleTier2,
                "knob '{knob_name}' diff must resolve to UpscaleTier2 \
                 so the live-preview gate fires for that knob",
            );
        }
    }
}

/// How a live render of each kind lands: a quick frame is shown and kept
/// nowhere; a settled render is the result, as before quick frames.
#[cfg(test)]
mod landing_tests {
    use std::sync::Arc;
    use std::time::Duration;
    use image::RgbaImage;
    use super::PrunrApp;
    use crate::gui::background_io::TexTarget;
    use crate::gui::item::BatchStatus;
    use crate::gui::live_preview::{PreviewKind, PreviewResult};
    use crate::gui::timeline::StepKind;
    use crate::gui::tests::fixtures::push_test_item;

    /// Every kind a live render can be.
    fn every_kind() -> [PreviewKind; 4] {
        let all = [PreviewKind::Mask, PreviewKind::Edge, PreviewKind::UpscaleTier2, PreviewKind::ColorizeTier2];
        for kind in all {
            // A new kind fails to compile here until it joins `all`.
            match kind {
                PreviewKind::Mask | PreviewKind::Edge | PreviewKind::UpscaleTier2 | PreviewKind::ColorizeTier2 => {}
                // Shown over the photo, never a result: tested on its own.
                PreviewKind::Photo => {}
            }
        }
        all
    }

    /// A result on screen, its step recorded.
    fn shown() -> (PrunrApp, Arc<RgbaImage>) {
        let mut app = PrunrApp::new_for_test();
        let item = push_test_item(&mut app, 1);
        let result = Arc::new(RgbaImage::new(2, 2));
        item.result_rgba = Some(Arc::clone(&result));
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
        item.commit(StepKind::Process);
        item.set_on_screen(true);
        app.batch.select_item(0);
        (app, result)
    }

    /// A render of `kind` for the state shown, made with settings other
    /// than the result's.
    fn render(app: &PrunrApp, kind: PreviewKind, quick: bool, rgba: Arc<RgbaImage>) -> PreviewResult {
        let mask = prunr_core::MaskSettings { gamma: 2.0, ..prunr_core::MaskSettings::default() };
        PreviewResult {
            kind,
            quick,
            generation: 3,
            version: app.batch.items[0].timeline.version(),
            applied_mask: prunr_core::MaskRecipe::from(&mask),
            applied_tier2_knobs: (kind == PreviewKind::UpscaleTier2)
                .then_some(prunr_core::upscale::Refine { sharpen: 0.7, sharpen_sigma: 2.0, ai_blend: 0.5, color_match: true }),
            applied_color_strength: (kind == PreviewKind::ColorizeTier2).then_some(0.4),
            ..PreviewResult::mask_for_test(1, rgba)
        }
    }

    /// Land the next prepared texture on the item, as the frame loop does.
    fn land_next_texture(app: &mut PrunrApp, ctx: &egui::Context) -> TexTarget {
        let prep = app.batch.bg_io.tex_prep.rx.recv_timeout(Duration::from_secs(10)).expect("a texture is prepared");
        let mut uploads = crate::gui::image_texture::Uploads::default();
        uploads.push(prep.name, prep.image.expect("prepared"), ());
        let (texture, ()) = uploads.pump(ctx, |_| true).landed.pop().expect("a small image uploads in one frame");
        app.batch.find_by_id_mut(prep.item_id).expect("the item").land_texture(texture, prep.target);
        prep.target
    }

    #[test]
    fn a_quick_frame_is_shown_never_kept_and_a_settled_render_is_the_result() {
        let ctx = egui::Context::default();
        for kind in every_kind() {
            let (mut app, result) = shown();
            let recipe = app.batch.items[0].applied_recipe.clone();
            let version = app.batch.items[0].timeline.version();

            let frame = render(&app, kind, true, Arc::new(RgbaImage::new(1, 1)));
            app.apply_completed_previews(&ctx, vec![frame]);
            let item = &app.batch.items[0];
            assert!(item.result_rgba.as_ref().is_some_and(|r| Arc::ptr_eq(r, &result)), "{kind:?}: the result stays");
            assert!(item.timeline.current().result.as_ref().is_some_and(|r| r.holds(&result)), "{kind:?}: so does its step");
            assert_eq!(item.applied_recipe, recipe, "{kind:?}: and the recipe that made it");
            assert_eq!(item.timeline.version(), version, "{kind:?}");
            assert!(!item.thumb.is_pending() && !item.result_tex_pending, "{kind:?}: no thumbnail or result texture");
            assert_eq!(land_next_texture(&mut app, &ctx), TexTarget::QuickFrame(3), "{kind:?}: the frame's own texture");
            assert!(app.batch.items[0].quick.texture().is_some(), "{kind:?}: shown");

            let settled = Arc::new(RgbaImage::new(2, 2));
            let full = render(&app, kind, false, Arc::clone(&settled));
            app.apply_completed_previews(&ctx, vec![full]);
            let item = &app.batch.items[0];
            assert!(item.result_rgba.as_ref().is_some_and(|r| Arc::ptr_eq(r, &settled)), "{kind:?}: a settled render is the result");
            assert!(item.timeline.current().result.as_ref().is_some_and(|r| r.holds(&settled)), "{kind:?}: of the shown step");
            assert_ne!(item.applied_recipe, recipe, "{kind:?}: made with the settings it rendered");
            assert!(item.thumb.is_pending() && item.result_tex_pending, "{kind:?}: its thumbnail and texture follow");
            assert!(item.quick.texture().is_some(), "{kind:?}: the frame stays until that texture lands");
            assert!(matches!(land_next_texture(&mut app, &ctx), TexTarget::LiveResult(_)), "{kind:?}");
            assert!(app.batch.items[0].quick.texture().is_none(), "{kind:?}: then goes");
        }
    }

    /// A render the image left for a new step that draws nothing of its
    /// own (a background pick right after a stroke) is made again for that
    /// step: nothing else would draw what it was asked for. A render
    /// already queued for the new step stands, and a run under way finishes
    /// the look itself.
    #[test]
    fn a_render_left_for_a_new_step_is_made_again_on_it() {
        let ctx = egui::Context::default();
        let bg_picked = |app: &mut PrunrApp| {
            let item = &mut app.batch.items[0];
            item.settings.bg = Some([123, 45, 142, 255]);
            item.commit(StepKind::Adjust);
        };
        for kind in [PreviewKind::Mask, PreviewKind::UpscaleTier2, PreviewKind::ColorizeTier2, PreviewKind::Photo] {
            let (mut app, result) = shown();
            let left = render(&app, kind, false, Arc::new(RgbaImage::new(2, 2)));
            bg_picked(&mut app);
            app.apply_completed_previews(&ctx, vec![left]);
            let item = &app.batch.items[0];
            assert!(item.result_rgba.as_ref().is_some_and(|r| Arc::ptr_eq(r, &result)), "{kind:?}: it does not land on the new step");
            assert_eq!(app.processor.live_preview.pending_kind_for_test(1), Some(kind), "{kind:?}: it is asked for again");
            assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(true), "{kind:?}: at once");
        }

        // Lines over the whole photo turned on since: an Edge render draws them.
        let (mut app, _) = shown();
        let left = render(&app, PreviewKind::Mask, false, Arc::new(RgbaImage::new(2, 2)));
        app.batch.items[0].settings.line_mode = prunr_core::LineMode::EdgesOnly;
        bg_picked(&mut app);
        app.apply_completed_previews(&ctx, vec![left]);
        assert_eq!(app.processor.live_preview.pending_kind_for_test(1), Some(PreviewKind::Edge), "the render for the state as it stands");

        let (mut app, _) = shown();
        let left = render(&app, PreviewKind::Mask, false, Arc::new(RgbaImage::new(2, 2)));
        bg_picked(&mut app);
        app.processor.live_preview.mark_tweak(1, PreviewKind::UpscaleTier2);
        app.apply_completed_previews(&ctx, vec![left]);
        assert_eq!(app.processor.live_preview.pending_kind_for_test(1), Some(PreviewKind::UpscaleTier2), "the newer render stands");

        let (mut app, _) = shown();
        let left = render(&app, PreviewKind::Mask, false, Arc::new(RgbaImage::new(2, 2)));
        bg_picked(&mut app);
        app.batch.items[0].start_job(Some(StepKind::Process), crate::gui::item::Engine::Segment);
        app.apply_completed_previews(&ctx, vec![left]);
        assert_eq!(app.processor.live_preview.pending_kind_for_test(1), None, "the run under way finishes the look");
    }
}

/// The drift check against a drag: the drag settles on its own, so the
/// check never re-marks the image every frame nor settles it early.
#[cfg(test)]
mod drift_tests {
    use std::sync::Arc;
    use super::PrunrApp;
    use crate::gui::item::BatchStatus;
    use crate::gui::live_preview::PreviewKind;
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;

    /// A cut-out whose mask settings have moved past the mask it applied.
    fn drifted() -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        app.settings.model = SettingsModel::Silueta;
        let item = push_test_item(&mut app, 1);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(1, 1)));
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
        item.settings.gamma = 1.5;
        app.batch.select_item(0);
        app
    }

    #[test]
    fn a_drag_owing_its_settle_is_not_marked_again_every_frame() {
        let mut app = drifted();
        app.processor.live_preview.owe_for_test(1, PreviewKind::Mask);
        for _ in 0..3 {
            app.recipe_drift_tripwire();
            assert!(!app.processor.live_preview.is_pending_for(1), "the held-still settle is the drag's own");
        }
    }

    #[test]
    fn a_drags_queued_frame_is_hurried_not_settled() {
        let mut app = drifted();
        app.processor.live_preview.mark_tweak(1, PreviewKind::Mask);
        app.recipe_drift_tripwire();
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(false));
    }

    #[test]
    fn with_no_render_under_way_the_drift_is_cut_again() {
        let mut app = drifted();
        app.recipe_drift_tripwire();
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(true));
    }

    /// A render for the drift that ends without landing (refused for
    /// memory) is not asked for again every frame; new settings ask again.
    #[test]
    fn a_drift_render_that_never_lands_is_asked_for_once() {
        let mut app = drifted();
        app.recipe_drift_tripwire();
        app.processor.live_preview.cancel(1);
        app.recipe_drift_tripwire();
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), None, "once for these settings");
        app.batch.items[0].settings.gamma = 1.8;
        app.recipe_drift_tripwire();
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(true), "again for new ones");
    }

    /// An upscale's or a colorize's step whose Refine or Strength picture
    /// never landed (Reset made a new step while it rendered, and Undo came
    /// back to it) is finished again when shown, as a cut-out's drifted
    /// mask is cut again. It kept the picture before the change, under
    /// settings that said otherwise.
    #[test]
    fn a_kept_runs_drift_is_finished_again() {
        use crate::gui::item::{ColorizeCache, Engine, UpscaleCache};
        use crate::gui::timeline::StepKind;
        type Land = fn(&mut crate::gui::item::BatchItem, Arc<image::RgbaImage>);
        type Change = fn(&mut crate::gui::item_settings::ItemSettings);
        let rows: [(SettingsModel, prunr_core::ModelKind, Engine, Land, Change, PreviewKind); 2] = [
            (SettingsModel::RealEsrganUpscale, prunr_core::ModelKind::RealEsrganX4Plus, Engine::Upscale,
                |item, img| item.land_upscale(UpscaleCache::new(Arc::clone(&img), img, None)), |s| s.sharpen = -0.5, PreviewKind::UpscaleTier2),
            (SettingsModel::ColorizeDdcolor, crate::gui::settings::COLORIZER, Engine::Colorize,
                |item, img| {
                    let n = (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize;
                    item.land_colorize(ColorizeCache::new(Arc::new(vec![0.0; 2 * n]), img));
                }, |s| s.color_strength_pct = 40, PreviewKind::ColorizeTier2),
        ];
        for (model, kind, engine, land, change, preview) in rows {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = model;
            let item = push_test_item(&mut app, 1);
            item.dimensions = (4, 4);
            item.applied_recipe = Some(item.settings.current_recipe(kind));
            let img = Arc::new(image::RgbaImage::new(16, 16));
            item.start_job(Some(StepKind::Upscale), engine);
            item.result_rgba = Some(Arc::clone(&img));
            land(item, img);
            let before = item.settings;
            change(&mut item.settings);
            item.commit(StepKind::Adjust);
            item.settings = before;
            item.commit(StepKind::Preset);
            item.undo_step().expect("back to the changed step");
            app.batch.select_item(0);
            app.recipe_drift_tripwire();
            assert_eq!(app.processor.live_preview.pending_kind_for_test(1), Some(preview), "{model:?}");
        }
    }

    /// Undo reaches every ticked image: one off screen that comes back to
    /// a step whose Refine render was dropped is finished again too. The
    /// drift check read only the image on screen.
    #[test]
    fn an_undo_finishes_an_off_screen_images_drift() {
        use crate::gui::item::{Engine, UpscaleCache};
        use crate::gui::timeline::StepKind;
        let mut app = PrunrApp::new_for_test();
        app.settings.model = SettingsModel::RealEsrganUpscale;
        for id in [1, 2] {
            let item = push_test_item(&mut app, id);
            item.dimensions = (4, 4);
            item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus));
            let img = Arc::new(image::RgbaImage::new(16, 16));
            item.start_job(Some(StepKind::Upscale), Engine::Upscale);
            item.result_rgba = Some(Arc::clone(&img));
            item.land_upscale(UpscaleCache::new(Arc::clone(&img), img, None));
            item.settings.sharpen = -0.5;
            item.commit(StepKind::Adjust);
            item.settings.sharpen = 0.0;
            item.commit(StepKind::Preset);
            item.ticked = true;
        }
        app.batch.select_item(1);
        app.handle_undo(&egui::Context::default());
        assert_eq!(app.processor.live_preview.pending_kind_for_test(1), Some(PreviewKind::UpscaleTier2), "the image off screen");
    }

    /// A cut-out step whose look needs the model again, and whose run was
    /// dropped as the image moved on (Pencil Sketch picked, Reset before
    /// its lines landed), runs again when an undo comes back to it, as the
    /// pick did: on a processed cut-out a knob change runs what it needs
    /// by itself. Under another model it stays as it is until Process,
    /// and the per-frame check never runs it, so a look whose run was
    /// cancelled waits for Process (and a failing run cannot repeat).
    #[test]
    fn an_undo_to_a_look_whose_run_was_dropped_runs_it_again() {
        use crate::gui::timeline::StepKind;
        use crate::gui::worker::WorkerMessage;
        let ctx = egui::Context::default();
        for (model, runs) in [(SettingsModel::Silueta, true), (SettingsModel::BiRefNetLite, false)] {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = model;
            let item = push_test_item(&mut app, 1);
            item.dimensions = (4, 4);
            crate::gui::tests::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
            item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
            item.settings.line_mode = prunr_core::LineMode::SubjectOutline;
            item.commit(StepKind::Preset);
            app.batch.select_item(0);
            let (tx, rx) = std::sync::mpsc::channel();
            app.processor.worker_tx = tx;
            app.recipe_drift_tripwire();
            assert!(rx.try_iter().all(|m| !matches!(m, WorkerMessage::BatchProcess { .. })), "{model:?}: the per-frame check runs nothing");

            let item = &mut app.batch.items[0];
            item.settings.line_mode = prunr_core::LineMode::Off;
            item.commit(StepKind::Preset);
            app.handle_undo(&ctx);
            let ran = rx.try_iter().any(|m| matches!(m, WorkerMessage::BatchProcess { .. }));
            assert_eq!(ran, runs, "{model:?}");
        }
    }

    /// A look whose run was cancelled stays a cancelled look: an undo
    /// and a redo back to it give it back as it was, waiting for Process,
    /// and do not run what Cancel stopped. Also a look that became a step
    /// only as its run was cancelled (the shown image's look, carried to a
    /// ticked one by Process).
    #[test]
    fn a_cancelled_look_is_not_run_again_by_a_redo() {
        use crate::gui::timeline::StepKind;
        use crate::gui::worker::WorkerMessage;
        let ctx = egui::Context::default();
        for look_was_a_step in [true, false] {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = SettingsModel::Silueta;
            let item = push_test_item(&mut app, 1);
            item.dimensions = (4, 4);
            crate::gui::tests::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
            item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
            app.batch.select_item(0);
            let (tx, rx) = std::sync::mpsc::channel();
            app.processor.worker_tx = tx;
            app.batch.items[0].settings.line_mode = prunr_core::LineMode::SubjectOutline;
            if look_was_a_step {
                app.batch.items[0].commit(StepKind::Preset);
            }
            app.process_items(|item| item.id == 1, None);
            assert!(rx.try_iter().any(|m| matches!(m, WorkerMessage::BatchProcess { .. })), "{look_was_a_step}: the look's run");
            app.apply_cancel_shortcut(&ctx);
            let item = &app.batch.items[0];
            assert!(item.job.is_none(), "{look_was_a_step}: Escape cancelled it");
            assert_eq!(item.timeline.current().settings.line_mode, prunr_core::LineMode::SubjectOutline, "{look_was_a_step}: the look is a step");

            app.handle_undo(&ctx);
            assert!(!app.batch.items[0].timeline.current().look_cancelled, "{look_was_a_step}: the step before it ran as it is");
            app.handle_redo(&ctx);
            assert!(rx.try_iter().all(|m| !matches!(m, WorkerMessage::BatchProcess { .. })), "{look_was_a_step}: the redo runs nothing Cancel stopped");
        }
    }

    /// A re-cut that draws lines it has none of waits for them. Undone
    /// to the photo (which has no lines) the kept lines go; a redo to a
    /// stroke on the lined cut-out re-cut it with no lines, and wrote a
    /// cut-out without its outline into the step. The check that a render
    /// has every scale it draws let it through when none was kept at all.
    #[test]
    fn a_recut_without_the_lines_it_draws_waits() {
        use crate::gui::capability::{Reason, Verdict};
        use crate::gui::knob_catalog::ModelRole;
        let mut app = PrunrApp::new_for_test();
        app.settings.model = SettingsModel::Silueta;
        let item = push_test_item(&mut app, 1);
        item.dimensions = (2, 2);
        item.source_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        item.settings.line_mode = prunr_core::LineMode::SubjectOutline;
        crate::gui::tests::fixtures::cut_with_lines(item);
        assert!(!item.lacks_drawn_scales());
        item.invalidate_edge_cache();
        assert!(item.lacks_drawn_scales(), "no lines kept lacks every scale drawn");
        let refused = PrunrApp::build_preview_inputs(item, PreviewKind::Mask, ModelRole::Cutout, crate::gui::brush_state::CutoutAction::Restore);
        assert_eq!(refused.err(), Some(Verdict::Never(Reason::LinesNotRun)));
    }

    /// A drift render dropped because the image moved on (a stroke made a
    /// new step under it) is asked for again on the new step.
    #[test]
    fn a_drift_render_dropped_for_a_new_step_is_asked_again() {
        let mut app = drifted();
        app.recipe_drift_tripwire();
        app.processor.live_preview.cancel(1);
        app.batch.items[0].commit(crate::gui::timeline::StepKind::Selection);
        app.recipe_drift_tripwire();
        assert_eq!(app.processor.live_preview.pending_settled_for_test(1), Some(true));
    }
}

/// The pill for quiet background work: it waits a second, so a drag whose
/// renders keep up never shows it, and never doubles another indicator.
#[cfg(test)]
mod busy_pill_tests {
    use std::time::{Duration, Instant};
    use super::PrunrApp;
    use crate::gui::tests::fixtures::push_test_item;

    fn app() -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        push_test_item(&mut app, 1);
        app.batch.select_item(0);
        app
    }

    /// A photo decoded for an image off screen leaves the shown image's
    /// zoom and Compare as they are.
    #[test]
    fn an_off_screen_decode_leaves_the_shown_zoom() {
        let mut app = app();
        push_test_item(&mut app, 2);
        app.zoom_state.zoom = 3.0;
        app.batch.bg_io.decode.tx.send((2, Ok(std::sync::Arc::new(image::RgbaImage::new(1, 1))))).unwrap();
        app.drain_background_channels(&egui::Context::default());
        assert_eq!(app.zoom_state.zoom, 3.0);
    }

    /// A render due for an image off screen whose photo is not decoded
    /// asks for the decode instead of waiting for good with the pill up.
    #[test]
    fn a_queued_render_of_an_undecoded_image_asks_for_its_decode() {
        let mut app = app();
        push_test_item(&mut app, 2);
        assert!(app.batch.items[1].source_rgba.is_none() && !app.batch.items[1].decode_pending);
        app.processor.live_preview.render_now(2, crate::gui::live_preview::PreviewKind::Mask);
        app.pump_live_preview(&egui::Context::default());
        assert!(app.batch.items[1].decode_pending, "the decode the render needs is under way");
        assert!(!app.batch.items[0].decode_pending, "an image with nothing queued is left alone");
    }

    /// A render that waits on a decode that fails is dropped: asking for
    /// the decode again every frame would only fail again.
    #[test]
    fn a_failed_decode_drops_the_render_waiting_for_it() {
        let mut app = app();
        push_test_item(&mut app, 2);
        app.processor.live_preview.render_now(2, crate::gui::live_preview::PreviewKind::Mask);
        app.pump_live_preview(&egui::Context::default());
        app.batch.bg_io.decode.tx.send((2, Err("bad file".into()))).unwrap();
        app.drain_background_channels(&egui::Context::default());
        app.pump_live_preview(&egui::Context::default());
        assert!(!app.batch.items[1].decode_pending, "no second decode");
        assert!(!app.processor.live_preview.is_pending_for(2), "the render is gone");
    }

    #[test]
    fn the_pill_waits_a_second_and_leaves_with_the_work() {
        let mut app = app();
        let t0 = Instant::now();
        app.note_quiet_work(t0);
        assert_eq!(app.busy_pill(t0 + Duration::from_secs(5)), None, "nothing under way");

        app.batch.items[0].result_tex_pending = true;
        app.note_quiet_work(t0);
        assert_eq!(app.busy_pill(t0 + Duration::from_millis(500)), None, "too soon to worry");
        app.note_quiet_work(t0 + Duration::from_millis(1500));
        assert_eq!(app.busy_pill(t0 + Duration::from_millis(1500)), Some("Preparing\u{2026}"));

        app.batch.items[0].result_tex_pending = false;
        app.note_quiet_work(t0 + Duration::from_secs(2));
        assert_eq!(app.busy_pill(t0 + Duration::from_secs(9)), None, "the work landed");
        app.batch.items[0].result_tex_pending = true;
        app.note_quiet_work(t0 + Duration::from_secs(10));
        assert_eq!(app.busy_pill(t0 + Duration::from_millis(10_500)), None, "the clock restarted");
    }

    #[test]
    fn a_runs_capsule_outranks_the_pill() {
        let mut app = app();
        let t0 = Instant::now();
        app.batch.items[0].result_tex_pending = true;
        app.note_quiet_work(t0);
        app.processor.show_progress_for_test(crate::gui::dispatch_progress::DispatchProgress::new(
            crate::gui::dispatch_progress::ProgressKind::Seg, "Starting",
        ));
        assert_eq!(app.busy_pill(t0 + Duration::from_secs(3)), None);
    }
}
