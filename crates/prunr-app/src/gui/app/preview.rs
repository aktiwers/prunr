//! Live preview: building each preview's inputs, landing its results, and the quiet-work pill.

use std::sync::Arc;

use crate::gui::item::{BatchItem, BatchStatus, Engine, KeptRunTier};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::background_io::TexTarget;
use super::{PrunrApp, BUSY_PILL_AFTER};

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

    /// Catch any drift between the active item's `applied_recipe.mask`
    /// and the recipe derived from current settings. Safety net for
    /// non-toolbar state mutations (brush commits, hotkeys) so they
    /// never silently fail to update the result.
    ///
    /// Active-item only by design — the only drift surfaces today
    /// (brush, undo/redo) act on the selected item; widening to the
    /// full batch would do per-frame work for items the user isn't
    /// editing for no current benefit.
    pub(crate) fn recipe_drift_tripwire(&mut self) {
        use crate::gui::live_preview::PreviewKind;
        // Only a cut-out or the filters apply the mask settings; firing a
        // preview over an erase or an upscale would clobber it with the
        // source through `run_preview`'s no-tensor fallback.
        if !shows(self.settings.model, Surface::Effect) {
            return;
        }
        if self.processor.live_preview.has_in_flight() {
            return;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let Some(applied) = item.applied_recipe.as_ref() else { return };
        let id = item.id;
        let current_settings = item.settings.mask_settings();
        if self.last_drift_check == Some((id, current_settings)) {
            return;
        }
        // The mask as the run applies it: pinned to the defaults where the
        // run uses none (Sketch on Full), so its settings never drift.
        let model = self.settings.model.to_model_kind().unwrap_or(prunr_core::ModelKind::BiRefNetLite);
        let current = item.settings.current_recipe(model).mask;
        if applied.mask == current {
            self.last_drift_check = Some((id, current_settings));
            return;
        }
        self.last_drift_check = None;
        let live = &mut self.processor.live_preview;
        if live.is_pending_for(id) {
            // A drag under way: its queued frame goes now instead of at the
            // throttle, and the drag settles on release or stillness, not here.
            live.hurry(id);
        } else if !live.works_on(id) {
            tracing::debug!(item_id = id, "recipe-drift tripwire fired — dispatching Tier-2");
            live.mark_tweak(id, PreviewKind::Mask);
            live.flush(id);
        }
    }

    /// Live-preview pump: dispatch debounced Tier 2 reruns + apply completed ones.
    /// Called once per frame at the start of `ui()` so the current frame renders
    /// with the newest available results.
    pub(super) fn pump_live_preview(&mut self, ctx: &egui::Context) {
        if let Some(msg) = self.pending_onboarding_toast.take() {
            self.toasts.info(msg);
        }
        // The SD eraser's reports and result come through its bridge. A
        // run sends its pictures before its result, and they land first:
        // the result reveals over them.
        self.processor.pump_inpaint_subprocess();
        self.pump_run_pictures(ctx);
        self.pump_inpaint_results(ctx);
        self.pump_upscale_results(ctx);
        self.pump_colorize_results(ctx);
        self.pump_sam_results(ctx);
        self.pump_faces();
        self.pump_depth(ctx);
        self.pump_download_manager(ctx);
        self.pump_runtime_install(ctx);
        self.recipe_drift_tripwire();

        // Dispatch phase: tick() invokes the closure for each item whose
        // debounce expired this frame. The closure borrows `batch.items`
        // mutably so `build_preview_inputs` can lazily cache an
        // `Arc<DynamicImage>` on the item (built once, reused across every
        // subsequent dispatch of the drag session — see `source_dyn`).
        let is_filter_only = self.settings.model == crate::gui::settings::SettingsModel::None;
        let style = self.selection_style();
        let action = self.settings.brush.cutout_action;
        let batch_items = &mut self.batch.items;
        let wait = self.processor.live_preview.tick(|id, kind| {
            Self::build_preview_inputs(batch_items, id, kind, is_filter_only, style, action)
        });
        // If a future dispatch is waiting, schedule a repaint when the
        // debounce elapses so tick() can fire on its own.
        if let Some(w) = wait {
            ctx.request_repaint_after(w);
        }

        // A render queued for an image whose photo is not decoded (a
        // ticked image off screen, its map just measured) waits for the
        // decode; nothing else asks for it, and the render would wait for
        // good with the busy pill up.
        let waiting: Vec<usize> = self.processor.live_preview.pending_photo_renders()
            .filter_map(|id| self.batch.index_of(id))
            .collect();
        for idx in waiting {
            self.ensure_source_decoded(idx);
        }

        let results = self.processor.live_preview.drain_results();
        if !results.is_empty() {
            self.apply_completed_previews(ctx, results);
        }
        let live = &self.processor.live_preview;
        for item in &mut self.batch.items {
            item.quick.settle(live.works_on(item.id));
        }

        // Covers the worker-running / not-yet-drained window. Same
        // self-extinguishing 50ms poll as the tex_pending loop in `logic()`.
        if self.processor.live_preview.has_in_flight() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    /// Snapshot the dispatch inputs for one preview job: original RGBA +
    /// decompressed tensors + settings + reusable edge mask if its
    /// line_strength still matches. Returns `None` to abort the dispatch
    /// when a required tensor cache isn't available (user must Process first).
    ///
    /// Lazily builds (and then reuses) `item.source_dyn` — the first dispatch
    /// of a drag session pays the ~15ms / 48 MB clone to wrap `source_rgba`
    /// in a `DynamicImage`, every subsequent dispatch is just an `Arc::clone`.
    pub(crate) fn build_preview_inputs(
        items: &mut [BatchItem],
        id: u64,
        kind: crate::gui::live_preview::PreviewKind,
        is_filter_only: bool,
        style: crate::gui::background_io::SelectionStyle,
        action: crate::gui::brush_state::CutoutAction,
    ) -> Option<crate::gui::live_preview::DispatchInputs> {
        use crate::gui::live_preview::{DispatchInputs, PreviewKind};
        let item = items.iter_mut().find(|b| b.id == id)?;
        let version = item.timeline.version();
        // UpscaleTier2: short-circuit the seg/edge path entirely. We need only
        // the kept upscale run + knob snapshot. Return None if there is no
        // upscale run yet (Tier-1 must run first).
        if matches!(kind, PreviewKind::UpscaleTier2) {
            let upscale = item.kept_upscale()?.clone();
            let knobs = item.settings.upscale_refine();
            return Some(DispatchInputs {
                upscale: Some(upscale),
                upscale_tier2_knobs: Some(knobs),
                faces: item.faces_input(crate::gui::item::FaceStage::Upscale),
                // `original` is unread by a Refine preview, which re-finishes the kept run.
                ..DispatchInputs::new(kind, version, Arc::new(image::DynamicImage::new_rgba8(0, 0)), item.settings)
            });
        }
        if matches!(kind, PreviewKind::ColorizeTier2) {
            let colorize = item.kept_colorize()?.clone();
            // `original` is unread: Strength re-renders the kept run.
            return Some(DispatchInputs {
                colorize: Some(colorize),
                ..DispatchInputs::new(kind, version, Arc::new(image::DynamicImage::new_rgba8(0, 0)), item.settings)
            });
        }
        // A photo render without a scale the lines draw would show the
        // image without those lines; the tick retries once the worker's
        // spare scales (or a new run) bring it.
        if item.lacks_drawn_scales() {
            return None;
        }
        let seg_tensor = item.hot_seg_tensor();
        // A Mask dispatch in a seg model with no cached tensor falls
        // through to `run_preview`'s source+fill_style fallback and
        // overwrites any existing `result_rgba`. Filter-only mode is
        // the legitimate user of that fallback (the filter IS the
        // pipeline). Anywhere else, refuse — better to skip the
        // dispatch than discard a prior result.
        if matches!(kind, PreviewKind::Mask)
            && !is_filter_only
            && seg_tensor.is_none()
            && item.result_rgba.is_some()
        {
            return None;
        }
        // A run on top of an earlier result (an erase, a hand edit) keeps
        // it: the preview starts where Process would, the shown step's own
        // start when it redoes that step.
        let start = item.run_start(Engine::Segment);
        let original = if let Some(rgba) = start.as_ref().and_then(|s| s.image.load()) {
            let cached = item.start_dyn_cache.as_ref()
                .and_then(|(src, dyn_img)| Arc::ptr_eq(&rgba, src).then(|| Arc::clone(dyn_img)));
            match cached {
                Some(dyn_img) => dyn_img,
                None => {
                    let dyn_img = Arc::new(image::DynamicImage::ImageRgba8((*rgba).clone()));
                    item.start_dyn_cache = Some((rgba, Arc::clone(&dyn_img)));
                    dyn_img
                }
            }
        } else {
            item.start_dyn_cache = None;
            if item.source_dyn.is_none() {
                let rgba = item.source_rgba.as_ref()?;
                item.source_dyn = Some(Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone())));
            }
            // invariant: source_dyn was filled just above when it was None.
            Arc::clone(item.source_dyn.as_ref().unwrap())
        };
        let edge_tensor = Self::edge_tensor_for_active_scale(item);
        // DualScale needs TWO edge tensors — the primary (active scale, used
        // as the Fine layer) and a secondary Bold layer. Only decompress the
        // Bold tensor when the style actually uses it; skipping this keeps
        // single-scale dispatches fast.
        let secondary_edge_tensor = matches!(item.settings.line_style, prunr_core::LineStyle::DualScale { .. })
            .then(|| Self::edge_tensor_for_scale(item, prunr_core::EdgeScale::Bold))
            .flatten();
        match kind {
            // Mask kind without a seg tensor is the filter-only path:
            // `run_preview` applies fill_style to the raw source. No tensor
            // required. Edge kind still needs an edge tensor (nothing to
            // preview without one).
            PreviewKind::Edge if edge_tensor.is_none() => return None,
            _ => {}
        }
        let has_depth = item.depth.is_some();
        let plane_key = |scale: prunr_core::EdgeScale| crate::gui::live_preview::EdgePlaneKey::new(&item.settings, scale, has_depth);
        let cached_edge = item.edge_planes.lookup(plane_key(item.settings.edge_settings().primary_scale()));
        let cached_bold = item.edge_planes.lookup(plane_key(prunr_core::EdgeScale::Bold));
        let cached_masked_base = item.cached_masked_base.as_ref().and_then(|(base, recipe, model)| {
            let current_recipe = prunr_core::MaskRecipe::from(&item.settings.mask_settings());
            let seg_model_match = seg_tensor.as_ref().is_some_and(|s| s.model == *model);
            (*recipe == current_recipe && seg_model_match).then(|| base.clone())
        });
        // Without a seg tensor the correction has nothing to apply to
        // (filter-only path).
        let correction = seg_tensor.as_ref().and_then(|_| item.correction_plane(style, action));
        let faces = item.faces_input(crate::gui::item::FaceStage::Photo);
        let depth = Self::depth_input(item, &original, faces.as_ref(), is_filter_only);
        Some(DispatchInputs {
            start,
            seg_tensor, edge_tensor, secondary_edge_tensor,
            cached_edge, cached_bold, cached_masked_base,
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
        original: &Arc<image::DynamicImage>,
        faces: Option<&crate::gui::live_preview::FacesInput>,
        is_filter_only: bool,
    ) -> Option<crate::gui::live_preview::DepthInput> {
        let map = item.depth.clone()?;
        let look = is_filter_only.then_some(item.settings.depth).filter(prunr_core::depth::DepthLook::is_active);
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

    /// Decompress the edge tensor for the item's currently-selected scale,
    /// using the `volatile_edge_tensor` hot cache to skip zstd work during a
    /// drag. The tensor rides through as `Arc<Vec<f32>>` so hot hits are a
    /// pointer bump, not a 1.2 MB memcpy per dispatch.
    fn edge_tensor_for_active_scale(item: &mut BatchItem) -> Option<crate::gui::live_preview::EdgeTensor> {
        Self::edge_tensor_for_scale(item, item.settings.edge_settings().primary_scale())
    }

    /// Decompress the edge tensor for a specific scale. Uses the
    /// `volatile_edge_tensor` hot cache when the requested scale matches
    /// the cached one; otherwise pays the zstd decompress. Only updates the
    /// hot cache when `scale` matches the item's active scale — otherwise a
    /// DualScale dispatch that asks for Bold would evict the active-scale
    /// entry and make the next Edge tweak miss. Returns `None` when the
    /// multi-scale cache isn't populated (user must Process first).
    fn edge_tensor_for_scale(
        item: &mut BatchItem,
        scale: prunr_core::EdgeScale,
    ) -> Option<crate::gui::live_preview::EdgeTensor> {
        let hot_hit = item.volatile_edge_tensor.as_ref()
            .filter(|(s, _)| *s == scale)
            .map(|(_, arc)| arc.clone());
        if let Some(arc) = hot_hit {
            let (height, width) = item.cached_edge_tensors.as_ref()?.dims(scale);
            return Some(crate::gui::live_preview::EdgeTensor { data: arc, height, width });
        }

        let (data, (height, width)) = {
            let cache = item.cached_edge_tensors.as_ref()?;
            (Arc::new(cache.decompress(scale)?), cache.dims(scale))
        };
        if scale == item.settings.edge_settings().primary_scale() {
            item.volatile_edge_tensor = Some((scale, data.clone()));
        }
        Some(crate::gui::live_preview::EdgeTensor { data, height, width })
    }

    /// Critical: do NOT null `result_texture` here — the old texture must stay
    /// visible until the newly-built one lands via `drain_background_channels`.
    /// Clearing it causes the canvas to flash black for a frame (no texture
    /// to draw → BG_PRIMARY shows). Instead we spawn a tex prep for the new
    /// RGBA directly and let drain swap it in atomically when ready.
    pub(crate) fn apply_completed_previews(
        &mut self,
        ctx: &egui::Context,
        results: Vec<crate::gui::live_preview::PreviewResult>,
    ) {
        use crate::gui::live_preview::PreviewKind;
        let handles = self.batch.bg_io.tex_prep_handles();
        // A result the image still wants restarts the quiet-work clock; a
        // stale one changes nothing on screen.
        if results.iter().any(|r| self.batch.find_by_id(r.item_id).is_some_and(|i| r.version == i.timeline.version())) {
            self.busy_since = None;
        }
        for r in results {
            let (item_id, source, is_final, landed) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else {
                    continue;
                };
                if r.version != item.timeline.version() {
                    // Computed for a state the item has since left.
                    continue;
                }
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
                            r.rgba, item.id, Self::tex_name("quick", item.id, None),
                            TexTarget::QuickFrame(r.generation), handles.clone(), ctx.clone(),
                        );
                    }
                    continue;
                }
                let is_upscale_tier2 = matches!(r.kind, PreviewKind::UpscaleTier2);
                let new_rgba = r.rgba;
                item.unpark_result();
                item.result_rgba = Some(new_rgba.clone());
                // Filter-only mode (model=None) never clicks Process — the
                // first live preview result IS the processed result. Promote
                // a Pending item to Done so the canvas flips from source
                // view to result view. Processing / Done items untouched.
                if item.status == BatchStatus::Pending {
                    item.status = BatchStatus::Done;
                }
                // Mark pending so reconcile_selected doesn't also
                // spawn its own prep on this same frame.
                item.result_tex_pending = true;
                for (planes, key) in r.new_edge.into_iter().chain(r.new_bold) {
                    item.edge_planes.store(planes, key);
                }
                if let Some(faced) = r.new_faced {
                    item.faced_photo = Some(faced);
                }
                if let Some(looked) = r.new_depth_photo {
                    item.depth_photo = Some(looked);
                }
                if let Some((base, recipe, model)) = r.new_masked_base {
                    item.cached_masked_base = Some((base, recipe, model));
                }
                if let (Some(reference), Some(run)) = (r.new_reference, item.upscale.as_mut()) {
                    run.reference = Some(reference);
                }
                // Mark the dispatched MaskRecipe as applied so the
                // recipe-drift tripwire doesn't immediately re-fire on
                // a result it has already consumed.
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
                        // A pass that cut nothing keeps who cut the step.
                        let cutter = r.cutter.or_else(|| item.shown_cutter());
                        item.record_origin(Engine::Segment, r.start, cutter);
                        Some(crate::gui::item::FaceStage::Photo)
                    }
                };
                if let Some(stage) = face_stage {
                    let blended = item.faces_input(stage).is_some_and(|f| !f.key.faces.is_empty());
                    item.settle_faced(stage.engine(), blended);
                }
                tracing::debug!(
                    event = "upscale_tier2_apply",
                    item_id = item.id,
                    is_upscale_tier2,
                    is_final = r.is_final,
                    "apply_completed_previews: result drained"
                );
                if item.on_screen {
                    let switch = self.result_switch_id;
                    let target = item.request_result_texture(true);
                    Self::spawn_tex_prep(
                        Arc::clone(&new_rgba), item.id, Self::tex_name("result", item.id, Some(switch)),
                        target, handles.clone(), ctx.clone(),
                    );
                    item.quick.supersede();
                } else {
                    item.quick.clear();
                    item.result_tex_pending = false;
                    item.drop_hot_tensors();
                    item.source_dyn = None;
                    item.start_dyn_cache = None;
                    item.park_result();
                }
                (item.id, item.source.clone(), r.is_final, new_rgba)
            };
            // On the drag-settled result only, regenerate the sidebar
            // thumbnail. Unlike nulling `thumb_texture` + letting the
            // sidebar re-request, we call request_thumbnail directly while
            // the old texture remains displayed — `pump_thumbnail_results`
            // swaps it atomically when the new thumb arrives, so the user
            // never sees a spinner-gap. Mid-drag results skip this entirely
            // (is_final false) to avoid regenerating thumbs every debounce.
            if is_final {
                if let Some(item) = self.batch.find_by_id_mut(item_id) {
                    item.thumb_pending = true;
                }
                self.batch.request_thumbnail(item_id, &source, Some(&landed));
            }
        }
        ctx.request_repaint();
    }

    /// Whether the image at `idx` changed only in Refine's after-the-model
    /// knobs since its upscale, so its kept run can be re-finished.
    pub(super) fn wants_upscale_tier2(&self, idx: usize) -> bool {
        let model = self.settings.model.dispatch_model_kind();
        self.batch.items.get(idx).zip(model).is_some_and(|(item, model)| model.is_upscale() && item.kept_run_tier(model) == KeptRunTier::Refinish)
    }

    /// Whether the image at `idx` changed only in Strength since its
    /// colorize, so its kept run can be re-rendered.
    pub(super) fn wants_colorize_tier2(&self, idx: usize) -> bool {
        self.settings.model.is_colorize()
            && self.batch.items.get(idx).is_some_and(|item| item.kept_run_tier(crate::gui::settings::COLORIZER) == KeptRunTier::Refinish)
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

    /// Regression guard for Bug #3: a recipe-diff built with a model-kind
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

    /// The fixed path: both recipes use RealEsrganX4Plus. Sharpen-only
    /// diff must resolve to UpscaleTier2.
    #[test]
    fn tier2_chain_correct_model_kind_routes_sharpen_to_upscale_tier2() {
        let old_s = ItemSettings { sharpen: 0.0, ..ItemSettings::default() };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = upscale_recipe_for(&old_s); // applied_recipe at Tier-1 result time
        let new_recipe = upscale_recipe_for(&new_s); // fixed path: same model kind used
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "sharpen-only diff with matching RealEsrganX4Plus model in both recipes \
             must resolve to UpscaleTier2 — this is the contract that enables live preview",
        );
    }

    /// Same as above but for all four Tier-2 knobs, ensuring each independently
    /// triggers the live-preview dispatch.
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
        item.status = BatchStatus::Done;
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
        let (id, name, image, target) = app.batch.bg_io.tex_prep_rx.recv_timeout(Duration::from_secs(10)).expect("a texture is prepared");
        let texture = ctx.load_texture(name, image, Default::default());
        app.batch.find_by_id_mut(id).expect("the item").land_texture(texture, target);
        target
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
            assert!(!item.thumb_pending && !item.result_tex_pending, "{kind:?}: no thumbnail or result texture");
            assert_eq!(land_next_texture(&mut app, &ctx), TexTarget::QuickFrame(3), "{kind:?}: the frame's own texture");
            assert!(app.batch.items[0].quick.texture().is_some(), "{kind:?}: shown");

            let settled = Arc::new(RgbaImage::new(2, 2));
            let full = render(&app, kind, false, Arc::clone(&settled));
            app.apply_completed_previews(&ctx, vec![full]);
            let item = &app.batch.items[0];
            assert!(item.result_rgba.as_ref().is_some_and(|r| Arc::ptr_eq(r, &settled)), "{kind:?}: a settled render is the result");
            assert!(item.timeline.current().result.as_ref().is_some_and(|r| r.holds(&settled)), "{kind:?}: of the shown step");
            assert_ne!(item.applied_recipe, recipe, "{kind:?}: made with the settings it rendered");
            assert!(item.thumb_pending && item.result_tex_pending, "{kind:?}: its thumbnail and texture follow");
            assert!(item.quick.texture().is_some(), "{kind:?}: the frame stays until that texture lands");
            assert!(matches!(land_next_texture(&mut app, &ctx), TexTarget::LiveResult(_)), "{kind:?}");
            assert!(app.batch.items[0].quick.texture().is_none(), "{kind:?}: then goes");
        }
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
        item.status = BatchStatus::Done;
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

    /// A render queued for an image off screen whose photo is not decoded
    /// asks for the decode instead of waiting for good with the pill up.
    #[test]
    fn a_queued_render_of_an_undecoded_image_asks_for_its_decode() {
        let mut app = app();
        push_test_item(&mut app, 2);
        assert!(app.batch.items[1].source_rgba.is_none() && !app.batch.items[1].decode_pending);
        app.processor.live_preview.mark_tweak(2, crate::gui::live_preview::PreviewKind::Mask);
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
        app.processor.live_preview.mark_tweak(2, crate::gui::live_preview::PreviewKind::Mask);
        app.pump_live_preview(&egui::Context::default());
        app.batch.bg_io.decode_tx.send((2, Err("bad file".into()))).unwrap();
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
