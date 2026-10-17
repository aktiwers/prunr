//! The shown image's textures: the selection-change reconcile, background eviction and texture prep.


use crate::gui::background_io::TexTarget;
use crate::gui::item::Pixels;
use super::PrunrApp;

impl PrunrApp {
    /// Run the selection-change work even though the selected id is
    /// unchanged: the result or history behind it was replaced.
    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        self.synced_selected = None;
        self.reconcile_selected(ctx);
    }

    /// Once per frame. On a selection change (detected here, so no
    /// selection path can forget it): free the background items' results,
    /// restore the selected one from history, leave Compare. Every frame:
    /// the decoded source, the canvas textures, the bg-image texture, the
    /// selection texture and the Magic Brush embedding the selected item
    /// lacks. Every arm is idempotent behind its pending flag.
    pub(super) fn reconcile_selected(&mut self, ctx: &egui::Context) {
        let selected = self.batch.selected_item().map(|i| i.id);
        let idx = self.batch.selected_idx_clamped();
        self.magic_brush_state.note_shown(selected, std::time::Instant::now());
        if selected != self.synced_selected {
            self.synced_selected = selected;
            if let Some(idx) = idx {
                self.evict_background_item_caches(idx);
                self.batch.items[idx].set_on_screen(true);
            }
            self.show_original = false;
        }
        if let (Some(idx), Some(id)) = (idx, selected) {
            self.batch.items[idx].settle_reveal();
            self.ensure_source_decoded(idx);
            self.request_selected_textures(idx, ctx);
            self.kick_backdrop_tex_prep(id, ctx);
        }
        let style = self.selection_style();
        self.batch.ensure_selection_texture(style, ctx);
        self.ensure_magic_embedding_for_selected(ctx);
    }

    /// For every item that is not selected: drop the decoded tensors kept
    /// for slider drags, and release the full-resolution result of Done
    /// items (`BatchItem::park_result`); the timeline keeps it compressed.
    pub(super) fn evict_background_item_caches(&mut self, selected_idx: usize) {
        for (i, item) in self.batch.items.iter_mut().enumerate() {
            if i == selected_idx {
                continue;
            }
            item.drop_hot_tensors();
            item.drop_upscale_reference();
            item.reveal = None;
            item.subject = None;
            // The photo with faces blended in, one image of RAM; the faces
            // themselves (a few MB) stay, so it is rebuilt without a search.
            // Likewise the photo with the Depth chip's look; the map stays.
            // Freed off the UI thread.
            item.face_preview = None;
            crate::gui::reclaim::later((item.faced_photo.take(), item.depth_photo.take(), item.quick_photo.take()));
            // Up to two images; warmed again when the image is shown.
            item.set_on_screen(false);
            // A full-size picture made from the photo; rebuilt when shown.
            if matches!(item.settings.backdrop(), crate::gui::item_settings::Backdrop::Effect(_)) {
                item.backdrop_texture = None;
            }
            // 16 MB each: the shown image keeps its embedding, and with the
            // tool on so does the one shown before it.
            if !self.magic_brush_state.keeps_embedding(item.id) {
                item.magic_brush_embedding = None;
            }
            if item.park_result() {
                item.result_texture = None;
                item.result_tex_pending = false;
            }
        }
    }

    /// Build the picture behind the result off the UI thread: the
    /// background image, or the effect made from the photo (once it is
    /// decoded). `drain_background_channels` uploads it. One build at a
    /// time: a blur-radius drag builds the latest value when the one in
    /// flight lands, not a blur per frame.
    pub(crate) fn kick_backdrop_tex_prep(&mut self, item_id: u64, ctx: &egui::Context) {
        use crate::gui::item_settings::Backdrop;
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        let backdrop = item.settings.backdrop();
        item.refresh_lens_focus();
        let Some(key) = item.backdrop_key() else {
            item.backdrop_texture = None;
            return;
        };
        if item.backdrop_tex_pending.is_some() || item.backdrop_texture.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let build: Box<dyn FnOnce() -> Option<image::RgbaImage> + Send> = match backdrop {
            Backdrop::Image(_) => {
                let Some(bg) = item.bg_image.clone() else { return };
                Box::new(move || Some(bg.image.to_rgba8()))
            }
            Backdrop::Effect(effect) => {
                let Some(photo) = item.backdrop_photo() else { return };
                let depth = item.depth.clone();
                if effect.needs_depth() && depth.is_none() {
                    // Built when the depth map lands.
                    item.backdrop_texture = None;
                    return;
                }
                // A parked cut-out is unpacked on the build's thread.
                let cutout = item.result_pixels().filter(|_| effect.needs_depth());
                Box::new(move || {
                    let cutout = cutout.and_then(|c| c.load());
                    photo.load().and_then(|p| prunr_core::effect_backdrop(&p, effect, depth.as_deref(), cutout.as_deref()))
                })
            }
            Backdrop::Transparent | Backdrop::Color(_) => return,
        };
        item.backdrop_tex_pending = Some(key);
        let tx = self.batch.bg_io.bg_tex_prep_tx.clone();
        let slots = self.batch.bg_io.decode_slots.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            let ci = build().map(|rgba| crate::gui::image_texture::PreparedImage::from_rgba(&rgba));
            let _ = tx.send((item_id, key, ci));
            ctx.request_repaint();
        });
    }

    /// Kick off a background decode when the image has no decoded source:
    /// the selected one after sidebar navigation, or one a queued render
    /// reads.
    pub(super) fn ensure_source_decoded(&mut self, idx: usize) {
        let item = &self.batch.items[idx];
        // A source that failed to decode is not asked for again every frame.
        if item.source_rgba.is_some() || item.decode_pending || item.decode_failed {
            return;
        }
        self.batch.items[idx].decode_pending = true;
        let id = self.batch.items[idx].id;
        self.batch.request_decode_source(id, &self.batch.items[idx].source);
    }

    /// Spawn ColorImage preparation threads for whichever textures are missing.
    /// The actual `ctx.load_texture()` call happens in `drain_background_channels`
    /// after the prepared ColorImage arrives on `tex_prep_rx`.
    fn request_selected_textures(&mut self, idx: usize, ctx: &egui::Context) {
        let item_id = self.batch.items[idx].id;

        if self.batch.items[idx].source_texture.is_none()
            && !self.batch.items[idx].source_tex_pending
        {
            if let Some(rgba) = self.batch.items[idx].source_rgba.clone() {
                self.batch.items[idx].source_tex_pending = true;
                Self::spawn_tex_prep(
                    Pixels::Resident(rgba), item_id, Self::tex_name("source", item_id, None), TexTarget::Source,
                    self.batch.bg_io.tex_prep_handles(), ctx.clone(),
                );
            }
        }

        if self.batch.items[idx].result_texture.is_none()
            && !self.batch.items[idx].result_tex_pending
        {
            // A parked result is unpacked by the build, not here.
            if let Some(pixels) = self.batch.items[idx].result_pixels().filter(|_| !self.batch.items[idx].result_unreadable()) {
                let switch = self.result_switch_id;
                self.batch.items[idx].result_tex_pending = true;
                let target = self.batch.items[idx].request_result_texture(false);
                Self::spawn_tex_prep(
                    pixels, item_id, Self::tex_name("result", item_id, Some(switch)), target,
                    self.batch.bg_io.tex_prep_handles(), ctx.clone(),
                );
            }
        }
    }

    pub(crate) fn tex_name(kind: &str, item_id: u64, switch: Option<u64>) -> String {
        match switch {
            Some(sw) => format!("{kind}_{item_id}_{sw}"),
            None => format!("{kind}_{item_id}"),
        }
    }

    pub(super) fn spawn_tex_prep(
        pixels: Pixels,
        item_id: u64,
        name: String,
        target: TexTarget,
        handles: crate::gui::background_io::TexPrepHandles,
        ctx: egui::Context,
    ) {
        std::thread::spawn(move || {
            // Park here past `available_parallelism()` so a Process All
            // batch-completion burst doesn't hold N × ~50 MB ColorImage
            // peaks simultaneously. Same slot pool as the bg-io decode /
            // thumbnail / filter spawns.
            let queued = std::time::Instant::now();
            let _slot = handles.slots.acquire();
            let waited_ms = queued.elapsed().as_millis() as u64;
            let parked = matches!(pixels, Pixels::Parked(_));
            let rgba = match pixels.read() {
                Ok(rgba) => rgba,
                Err(err) => {
                    tracing::warn!(item_id, %err, "texture not prepared");
                    let _ = handles.tx.send(crate::gui::background_io::TexPrepMsg { item_id, name, image: None, target, unpacked: None });
                    return ctx.request_repaint();
                }
            };
            let ci = crate::gui::image_texture::PreparedImage::from_rgba(&rgba);
            tracing::debug!(item_id, waited_ms, convert_ms = queued.elapsed().as_millis() as u64 - waited_ms, "texture prepared");
            let unpacked = parked.then_some(rgba);
            let _ = handles.tx.send(crate::gui::background_io::TexPrepMsg { item_id, name, image: Some(ci), target, unpacked });
            ctx.request_repaint();
        });
    }
}

#[cfg(test)]
mod background_item_tests {
    use std::sync::Arc;
    use crate::gui::item::UpscaleCache;
    use crate::gui::tests::fixtures::push_test_item;

    /// An image not on screen drops its upscale's enlarged reference (the
    /// size of the output); the model's output and input stay, so Refine
    /// rebuilds it on return. The shown image keeps its own.
    #[test]
    fn images_not_on_screen_drop_the_upscale_reference() {
        let mut app = super::PrunrApp::new_for_test();
        let image = || Arc::new(image::RgbaImage::new(8, 8));
        for id in [1, 2] {
            push_test_item(&mut app, id).upscale = Some(UpscaleCache::new(image(), image(), Some(image())));
        }
        app.batch.select_item(0);
        app.evict_background_item_caches(0);
        let kept = |i: usize| app.batch.items[i].upscale.as_ref().map(|u| u.reference.is_some());
        assert_eq!(kept(0), Some(true), "the shown image keeps its reference");
        assert_eq!(kept(1), Some(false), "the other drops it but keeps its run");
    }
}
