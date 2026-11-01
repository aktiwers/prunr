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

    /// Another image is shown (picked by key or in the list, or just
    /// loaded): the canvas fits it afresh and leaves Compare.
    pub(crate) fn on_selected_changed(&mut self, ctx: &egui::Context) {
        self.zoom_state.reset();
        self.show_original = false;
        self.sync_selected_batch_textures(ctx);
    }

    /// Once per frame. On a selection change (detected here, so no
    /// selection path can forget it): free the background items' results,
    /// restore the selected one from history, leave Compare. Every frame:
    /// the background picture every processed image shows (one texture
    /// a picture, so thumbnails draw it), and the decoded source, the
    /// canvas textures, the backdrop texture, the selection texture and
    /// the Magic Brush embedding the selected item lacks. Every arm is
    /// idempotent behind its pending flag.
    pub(super) fn reconcile_selected(&mut self, ctx: &egui::Context) {
        let selected = self.batch.selected_id();
        let idx = self.batch.selected_idx_clamped();
        self.magic_brush_state.note_shown(selected, std::time::Instant::now());
        if selected != self.synced_selected {
            self.synced_selected = selected;
            if let Some(idx) = idx {
                self.evict_background_item_caches(idx);
                self.batch.items[idx].set_on_screen(true);
                // A failed decode is tried again on each visit: memory may
                // have been short, or the file has come back.
                self.batch.items[idx].decode_failed = false;
            }
            self.show_original = false;
        }
        for id in self.batch.lacking_picture_backdrops() {
            self.kick_backdrop_tex_prep(id, ctx);
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
                item.release_result_texture();
            }
        }
        if crate::gui::memory::under_memory_pressure() {
            self.free_off_screen_photos();
        }
    }

    /// Images off screen let go of their photos, but not one a queued
    /// preview reads.
    pub(super) fn free_off_screen_photos(&mut self) {
        self.batch.free_off_screen_photos(|id| self.processor.live_preview.works_on(id));
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
        item.refresh_backdrop_cut();
        let Some(key) = item.backdrop_key() else {
            item.backdrop_texture = None;
            return;
        };
        if item.backdrop_tex_pending.is_some() || item.backdrop_texture_for(key).is_some() {
            return;
        }
        if matches!(backdrop, Backdrop::Image(_)) {
            use crate::gui::batch_manager::PictureBackdrop;
            match self.batch.picture_backdrop(key) {
                Some(PictureBackdrop::Held(texture)) => {
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.install_backdrop(key, texture);
                    }
                    return;
                }
                Some(PictureBackdrop::Building) => return,
                None => {}
            }
        }
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        use crate::gui::background_io::BackdropBuilt;
        use crate::gui::image_texture::PreparedImage;
        let build: Box<dyn FnOnce() -> BackdropBuilt + Send> = match backdrop {
            Backdrop::Image(_) => {
                let Some(bg) = item.named_bg_image().cloned() else { return };
                Box::new(move || BackdropBuilt::Picture(Some(PreparedImage::from_rgba(&bg.image.to_rgba8())), None, None))
            }
            Backdrop::Effect(effect) => {
                let Some(photo) = item.backdrop_photo() else { return };
                if item.awaits_depth(effect) {
                    // Built when the depth map lands.
                    item.backdrop_texture = None;
                    return;
                }
                let depth = item.depth.clone();
                // A parked cut-out is unpacked on the build's thread, where a
                // new result with the shown backdrop's alpha keeps it.
                let cutout = item.result_pixels().filter(|_| effect.takes_cutout());
                let (shown, apart_from_cut) = (item.shown_backdrop_made(), item.backdrop_key_apart_from_cut());
                let thumb_key = item.thumb_key_to_bring();
                Box::new(move || {
                    let cutout = cutout.and_then(|c| c.load());
                    let alpha = cutout.as_deref().map(prunr_core::formats::alpha_fingerprint);
                    if alpha.is_some() && shown.map(|made| made.alpha) == alpha {
                        tracing::debug!(item_id, "backdrop kept: the cut-out's alpha is the same");
                        return BackdropBuilt::Same;
                    }
                    let started = std::time::Instant::now();
                    let made = apart_from_cut.zip(alpha).map(|(apart_from_cut, alpha)| crate::gui::item::BackdropMade { apart_from_cut, alpha });
                    let rgba = photo.load().and_then(|p| prunr_core::effect_backdrop(&p, effect, depth.as_deref(), cutout.as_deref()));
                    let thumb = thumb_key.zip(rgba.as_ref()).map(|(key, rgba)| (key, thumbnail_copy(rgba)));
                    let picture = rgba.map(|rgba| PreparedImage::from_rgba(&rgba));
                    tracing::debug!(item_id, elapsed_ms = started.elapsed().as_millis() as u64, "backdrop built");
                    BackdropBuilt::Picture(picture, made, thumb)
                })
            }
            Backdrop::Transparent | Backdrop::Color(_) => return,
        };
        item.backdrop_tex_pending = Some(key);
        let tx = self.batch.bg_io.bg_tex_prep_tx.clone();
        let ctx = ctx.clone();
        self.batch.bg_io.decode_slots.spawn(move || {
            let _ = tx.send((item_id, key, build()));
            ctx.request_repaint();
        });
    }

    /// Make the thumbnail copy of an image's effect backdrop, off screen
    /// where its full-size picture is not kept: from the photo and cut-out
    /// shrunk to `BACKDROP_THUMB_WORK_PX`, the effect's pixel sizes with
    /// them. A blur made small is not the full-size one shrunk, but at a
    /// thumbnail's size the two look alike; Save makes its own at full
    /// size. One at a time, and not while memory is short; the image's
    /// sidebar row asks while it is in view.
    ///
    /// Peak: one full-size image unpacked at a time to be shrunk (96 MB at
    /// 24 MP; a JPEG photo file decodes at the small size instead), then a
    /// few MB.
    pub(crate) fn kick_backdrop_thumb(&mut self, item_id: u64, ctx: &egui::Context) {
        if self.batch.building_backdrop_thumb.is_some() || crate::gui::memory::under_memory_pressure() {
            return;
        }
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        let Some((effect, key)) = item.wanted_backdrop_thumb() else { return };
        item.backdrop_thumb_asked(key);
        let photo = item.backdrop_photo().unwrap_or_else(|| item.photo_pixels());
        let photo_width = photo.decoded_dims().map_or(item.dimensions.0, |(w, _)| w).max(1);
        let cutout = item.result_pixels().filter(|_| effect.takes_cutout());
        let depth = item.depth.clone();
        self.batch.building_backdrop_thumb = Some(item_id);
        let tx = self.batch.bg_io.bg_tex_prep_tx.clone();
        let ctx = ctx.clone();
        self.batch.bg_io.decode_slots.spawn(move || {
            // Answered even on a panic, or the one-at-a-time slot stays taken.
            let thumb = crate::panics::caught("a thumbnail backdrop", || {
                let photo = photo.load_to_fit(BACKDROP_THUMB_WORK_PX)?;
                let cutout = cutout.and_then(|c| c.load_to_fit(BACKDROP_THUMB_WORK_PX));
                let effect = effect.at_scale(photo.width() as f32 / photo_width as f32);
                prunr_core::effect_backdrop(&photo, effect, depth.as_deref(), cutout.as_ref()).map(|rgba| thumbnail_copy(&rgba))
            }).flatten();
            // The full backdrop's key has no part in a copy alone.
            let _ = tx.send((item_id, 0, crate::gui::background_io::BackdropBuilt::Thumb(key, thumb)));
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

    /// Prepares the missing textures off-thread; `drain_background_channels` uploads them.
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
            // Park past `available_parallelism()` so a Process All burst doesn't
            // hold N × ~50 MB ColorImages at once.
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

    /// A photo whose decode failed is tried again when its image is
    /// shown again: the failure may have been memory short at the time.
    #[test]
    fn a_failed_decode_is_tried_again_when_the_image_is_shown() {
        let mut app = super::PrunrApp::new_for_test();
        push_test_item(&mut app, 1);
        push_test_item(&mut app, 2).decode_failed = true;
        app.batch.select_item(0);
        let ctx = egui::Context::default();
        app.reconcile_selected(&ctx);
        app.batch.select_item(1);
        app.reconcile_selected(&ctx);
        assert!(app.batch.items[1].decode_pending);
    }

    /// An image not on screen drops its upscale's enlarged reference (the
    /// size of the output); the model's output and input stay, so Refine
    /// rebuilds it on return. The shown image keeps its own.
    #[test]
    fn images_not_on_screen_drop_the_upscale_reference() {
        let mut app = super::PrunrApp::new_for_test();
        let image = || Arc::new(image::RgbaImage::new(8, 8));
        for id in [1, 2] {
            push_test_item(&mut app, id).upscale = Some(UpscaleCache::new(image(), image(), Some(Arc::new(prunr_core::upscale::Reference::new((*image()).clone())))));
        }
        app.batch.select_item(0);
        app.evict_background_item_caches(0);
        let kept = |i: usize| app.batch.items[i].upscale.as_ref().map(|u| u.reference.is_some());
        assert_eq!(kept(0), Some(true), "the shown image keeps its reference");
        assert_eq!(kept(1), Some(false), "the other drops it but keeps its run");
    }
}

/// The size a never-shown image's backdrop copy is made at, four times a
/// thumbnail's: a blur's radius stays over a pixel there, and the shrink
/// to the thumbnail hides what making it small changes.
const BACKDROP_THUMB_WORK_PX: u32 = 640;

fn thumbnail_copy(rgba: &image::RgbaImage) -> crate::gui::image_texture::PreparedImage {
    crate::gui::image_texture::PreparedImage::from_rgba(&crate::gui::batch_manager::thumbnail_rgba(rgba))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use super::PrunrApp;
    use crate::gui::tests::fixtures::push_test_item;

    use crate::gui::tests::fixtures::bg_picture as picture;

    /// A newly named picture loads while the one held before stays: the
    /// backdrop is built from the named one only, never the old one under
    /// the new picture's key, which would stay on screen.
    #[test]
    fn a_backdrop_is_built_only_from_the_picture_the_settings_name() {
        let mut app = PrunrApp::new_for_test();
        let ctx = egui::Context::default();
        let item = push_test_item(&mut app, 1);
        item.hold_bg_image(picture(7));
        item.settings.bg_image_hash = NonZeroU64::new(8);
        app.kick_backdrop_tex_prep(1, &ctx);
        assert!(app.batch.items[0].backdrop_tex_pending.is_none(), "the old picture is not built as the new one");
        app.batch.items[0].hold_bg_image(picture(8));
        app.kick_backdrop_tex_prep(1, &ctx);
        assert!(app.batch.items[0].backdrop_tex_pending.is_some(), "it builds once the named one is held");
    }

    /// An image processed off screen gets its background picture built
    /// without being shown, so its thumbnail draws it.
    #[test]
    fn an_image_off_screen_gets_its_picture_for_its_thumbnail() {
        let mut app = PrunrApp::new_for_test();
        let ctx = egui::Context::default();
        push_test_item(&mut app, 1);
        let item = push_test_item(&mut app, 2);
        item.settings.bg_image_hash = NonZeroU64::new(7);
        item.hold_bg_image(picture(7));
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        app.batch.select_item(0);
        app.reconcile_selected(&ctx);
        assert!(app.batch.items[1].backdrop_tex_pending.is_some(), "its picture builds while it is off screen");
    }

    /// Off screen, effect thumbnail copies are made one image at a time,
    /// and one made is kept when the full-size pictures are let go.
    #[test]
    fn off_screen_effect_thumbnails_are_made_one_at_a_time_and_kept() {
        let mut app = PrunrApp::new_for_test();
        let ctx = egui::Context::default();
        for id in [1, 2, 3] {
            let item = push_test_item(&mut app, id);
            crate::gui::tests::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
            item.settings.bg_effect = prunr_core::BgEffect::BlurredSource { radius: 12 };
        }
        app.kick_backdrop_thumb(2, &ctx);
        app.kick_backdrop_thumb(3, &ctx);
        assert_eq!(app.batch.building_backdrop_thumb, Some(2), "one at a time");
        let started = std::time::Instant::now();
        while app.batch.building_backdrop_thumb.is_some() && started.elapsed() < std::time::Duration::from_secs(20) {
            app.drain_background_channels(&ctx);
        }
        assert_eq!(app.batch.building_backdrop_thumb, None, "the build answered");
        assert!(!app.batch.items[1].wants_backdrop_thumb(), "answered, it is not asked for again");

        let key = app.batch.items[2].thumb_backdrop_key().expect("an effect");
        let texture = crate::gui::tests::fixtures::red_texture(&ctx);
        app.batch.items[2].install_backdrop_thumb(key, texture.clone());
        app.evict_background_item_caches(0);
        assert_eq!(app.batch.items[2].thumbnail_backdrop().map(|t| t.id()), Some(texture.id()), "kept off screen");
    }

    /// One picture builds once, the image on screen included: the others
    /// wait for it, then take that texture.
    #[test]
    fn a_picture_builds_once_for_every_image_showing_it() {
        let mut app = PrunrApp::new_for_test();
        let ctx = egui::Context::default();
        for id in [1, 2, 3] {
            let item = push_test_item(&mut app, id);
            item.settings.bg_image_hash = NonZeroU64::new(7);
            item.hold_bg_image(picture(7));
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        }
        app.batch.select_item(2);
        app.reconcile_selected(&ctx);
        let building: Vec<u64> = app.batch.items.iter().filter(|i| i.backdrop_tex_pending.is_some()).map(|i| i.id).collect();
        assert_eq!(building.len(), 1, "one build: {building:?}");
        let key = app.batch.items[0].backdrop_key().expect("a picture backdrop");
        let texture = crate::gui::tests::fixtures::red_texture(&ctx);
        let builder = app.batch.find_by_id_mut(building[0]).expect("pushed");
        assert!(builder.install_backdrop(key, texture.clone()));
        app.reconcile_selected(&ctx);
        for item in &app.batch.items {
            assert_eq!(item.backdrop_texture_for(key).map(|t| t.id()), Some(texture.id()), "image {} takes the one texture", item.id);
        }
    }
}
