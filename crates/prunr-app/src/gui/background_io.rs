use std::sync::{mpsc, Arc, Condvar, Mutex};

/// Filter-only Process result: `Ok(rgba)` on success, `Err(msg)` so load /
/// decode failures take the same channel path as successful results
/// (drained in `drain_background_channels`, mapped to `BatchStatus::Error`).
pub type FilterOnlyResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// Pre-decode result: `Ok(rgba)` on success, `Err(msg)` so a malformed
/// image clears `decode_pending` instead of leaving the item stuck.
pub type DecodeResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// History-eviction → off-thread compress → drain payload. The bg
/// thread sends `(item_id, compressed_entry, recipe)` once
/// `compress_to_ram` returns; the drain swaps the in-memory placeholder
/// at `history.back()` for the compressed slot. Recipe is matched at
/// drain time so a freshly-processed result can't be clobbered by a
/// stale eviction.
pub type HistoryDemoteResult = (
    u64,
    super::history_disk::CompressedEntry,
    Option<prunr_core::ProcessingRecipe>,
);

/// What a selection texture was built from: the mask's content hash and
/// the style baked into it. The item compares this against the current
/// pair to decide whether a rebuild is due.
pub(crate) type SelectionTextureKey = (u64, SelectionStyle);

/// Selection texture (fill + outline, ACCENT-tinted ColorImage) built
/// off-thread. A full image is 4 B/px (~48 MB at 4K) briefly on the
/// rayon worker; a patch is the changed region only. Either drops
/// after the upload in `drain_background_channels`.
pub(crate) struct SelectionTextureResult {
    pub(crate) item_id: u64,
    pub(crate) image: SelectionImage,
    pub(crate) key: SelectionTextureKey,
}

pub(crate) enum SelectionImage {
    /// The whole overlay, and the feathered plane it was built from when
    /// the style feathers (so actions can reuse it).
    Full { image: egui::ColorImage, feathered: Option<Arc<prunr_core::selection::MaskArtifact>> },
    /// The pixels that differ from the texture showing `base_hash`,
    /// to upload at `pos` into that texture.
    Patch { base_hash: u64, pos: [usize; 2], image: egui::ColorImage },
}

/// Counting semaphore used to bound the number of simultaneously-decoding
/// background threads. Without this, a 50-image Process All fans out 50
/// threads each holding `compressed bytes + DynamicImage + RgbaImage`
/// (~50–80 MB at 4 K) → multi-GB transient before any thread releases.
/// Threads still spawn immediately; they park on `acquire` until a slot
/// opens. Cap is `available_parallelism()` so cold-cache disk paths
/// still saturate cores.
pub struct DecodeSlots {
    state: Mutex<usize>,
    cv: Condvar,
}

impl DecodeSlots {
    pub fn new(slots: usize) -> Self {
        Self { state: Mutex::new(slots.max(1)), cv: Condvar::new() }
    }
    pub fn acquire(self: &Arc<Self>) -> DecodeSlotGuard {
        let mut s = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while *s == 0 {
            s = self.cv.wait(s).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *s -= 1;
        DecodeSlotGuard { sem: self.clone() }
    }
}

pub struct DecodeSlotGuard {
    sem: Arc<DecodeSlots>,
}

impl Drop for DecodeSlotGuard {
    fn drop(&mut self) {
        let mut s = self.sem.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *s += 1;
        self.sem.cv.notify_one();
    }
}

/// Owned handles to the texture-prep channel + decode slots, bundled so
/// `PrunrApp::spawn_tex_prep` takes one parameter instead of two. The
/// `Clone` derive is shallow (Sender is Clone, Arc is Clone) and cheap.
#[derive(Clone)]
pub struct TexPrepHandles {
    pub tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
    pub slots: Arc<DecodeSlots>,
}

/// Bundles all background thread communication channels.
pub struct BackgroundIO {
    /// File paths from file dialog / drag-and-drop (loaded lazily on demand)
    pub file_load_tx: mpsc::Sender<(std::path::PathBuf, String)>,
    pub file_load_rx: mpsc::Receiver<(std::path::PathBuf, String)>,
    /// Thumbnail generation results
    pub thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
    pub thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    /// Pre-decoded source images for instant canvas switching
    pub decode_tx: mpsc::Sender<DecodeResult>,
    pub decode_rx: mpsc::Receiver<DecodeResult>,
    /// Save completion notifications
    pub save_done_tx: mpsc::Sender<String>,
    pub save_done_rx: mpsc::Receiver<String>,
    /// Pre-built ColorImages ready for GPU upload (item_id, texture_name, image, is_result)
    pub tex_prep_tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
    pub tex_prep_rx: mpsc::Receiver<(u64, String, egui::ColorImage, bool)>,
    /// Filter-only (model=None) Process results — keeps the UI thread free
    /// while load + decode + `apply_fill_style` runs per item.
    pub filter_only_tx: mpsc::Sender<FilterOnlyResult>,
    pub filter_only_rx: mpsc::Receiver<FilterOnlyResult>,
    /// Off-thread zstd compression of evicted background-item RGBAs.
    /// Eviction places an in-memory placeholder at `history.back()` and
    /// kicks a worker that compresses + posts here; drain swaps the
    /// placeholder for `HistorySlot::Compressed`. Without this the
    /// zstd encode (~10–50 ms per 4K image × N items) ran inline on
    /// every selection change.
    pub history_demote_tx: mpsc::Sender<HistoryDemoteResult>,
    pub history_demote_rx: mpsc::Receiver<HistoryDemoteResult>,
    /// Off-thread RGBA decode + ColorImage build for a per-item bg
    /// image. Drain calls `ctx.load_texture` with the bg-specific
    /// options (Linear filter + Repeat wrap for `BgImageFit::Tile`).
    /// Without this the `to_rgba8()` clone (~32 MB on a 4K bg) +
    /// `ctx.load_texture` ran inline in `views/canvas.rs::render` on
    /// the first frame after every bg-image pick.
    pub bg_tex_prep_tx: mpsc::Sender<(u64, u64, egui::ColorImage)>,
    pub bg_tex_prep_rx: mpsc::Receiver<(u64, u64, egui::ColorImage)>,
    /// Bounds simultaneous decode/thumbnail/filter threads to
    /// `available_parallelism()`. Threads spawn immediately but park here
    /// until a slot opens, capping transient RAM at N × per-thread peak.
    pub decode_slots: Arc<DecodeSlots>,
    /// Selection texture built off-thread whenever the selected item's
    /// texture is missing. Drained via ctx.load_texture (allowed —
    /// drain_background_channels runs in logic(), not a render closure).
    pub(crate) selection_texture_tx: mpsc::Sender<SelectionTextureResult>,
    pub(crate) selection_texture_rx: mpsc::Receiver<SelectionTextureResult>,
}

impl BackgroundIO {
    /// Clone the (tx, slots) pair as one handle bundle for spawn paths.
    pub fn tex_prep_handles(&self) -> TexPrepHandles {
        TexPrepHandles {
            tx: self.tex_prep_tx.clone(),
            slots: self.decode_slots.clone(),
        }
    }
}

impl Default for BackgroundIO {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundIO {
    pub fn new() -> Self {
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();
        let (decode_tx, decode_rx) = mpsc::channel();
        let (save_done_tx, save_done_rx) = mpsc::channel();
        let (tex_prep_tx, tex_prep_rx) = mpsc::channel();
        let (filter_only_tx, filter_only_rx) = mpsc::channel();
        let (history_demote_tx, history_demote_rx) = mpsc::channel();
        let (bg_tex_prep_tx, bg_tex_prep_rx) = mpsc::channel();
        let (selection_texture_tx, selection_texture_rx) = mpsc::channel();
        let cap = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            file_load_tx, file_load_rx,
            thumb_tx, thumb_rx,
            decode_tx, decode_rx,
            save_done_tx, save_done_rx,
            tex_prep_tx, tex_prep_rx,
            filter_only_tx, filter_only_rx,
            history_demote_tx, history_demote_rx,
            bg_tex_prep_tx, bg_tex_prep_rx,
            decode_slots: Arc::new(DecodeSlots::new(cap)),
            selection_texture_tx, selection_texture_rx,
        }
    }

    /// Spawn an off-thread job to build the selection texture for a mask.
    ///
    /// Peak RAM on the worker: the ColorImage (4 B/px, ~33 MB at 4K) plus,
    /// with feather > 0, the bbox-sized guided-filter scratch documented on
    /// `feather_edges`. The result carries its key; the drain installs it
    /// only while the item's mask still matches.
    ///
    /// With `base` (the plane the item's texture shows, and its hash)
    /// the result is a patch of the changed region instead of a full
    /// image.
    pub(crate) fn request_selection_visualization(
        &self,
        item_id: u64,
        mask: Arc<prunr_core::selection::MaskArtifact>,
        key: SelectionTextureKey,
        source: Option<Arc<image::RgbaImage>>,
        base: Option<(u64, Arc<prunr_core::selection::MaskArtifact>)>,
        ctx: egui::Context,
    ) {
        let texture_tx = self.selection_texture_tx.clone();
        rayon::spawn(move || {
            let style = key.1;
            let patch = base.and_then(|(base_hash, base)| {
                let (pos, image) = build_selection_patch(&base, &mask, style)?;
                Some(SelectionImage::Patch { base_hash, pos, image })
            });
            let image = patch.unwrap_or_else(|| {
                let feathered = style.feather(&mask, source.as_deref());
                let shown = feathered.as_ref().unwrap_or(&mask);
                SelectionImage::Full { image: build_selection_image(shown, style), feathered }
            });
            let _ = texture_tx.send(SelectionTextureResult { item_id, image, key });
            // Some compositors drop thread-initiated wake-ups while the
            // window is idle; the pending key's poll in `logic()` is the
            // fallback.
            ctx.request_repaint();
        });
    }
}

/// The style knobs a selection texture bakes in. All four apply on
/// commit: the overlay is a single textured quad drawn untinted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SelectionStyle {
    pub(crate) fill_opacity: f32,
    pub(crate) outline_opacity: f32,
    /// Outline width in image pixels.
    pub(crate) outline_thickness: f32,
    pub(crate) edge_feather_px: u32,
}

impl SelectionStyle {
    pub(crate) fn from_brush(b: &super::brush_state::BrushSettings) -> Self {
        Self {
            fill_opacity: b.fill_opacity,
            outline_opacity: b.outline_opacity,
            outline_thickness: b.outline_thickness,
            edge_feather_px: b.edge_feather.round().max(0.0) as u32,
        }
    }

    /// The mask as the user sees it: feathered against `source` when the
    /// feather knob is on and a source is available. Both the texture
    /// build and Delete / Copy / Cut go through here.
    /// The feathered selection, or `None` when this style shows `mask`
    /// as it is (no feather, or no photo to feather against).
    pub(crate) fn feather(
        &self,
        mask: &prunr_core::selection::MaskArtifact,
        source: Option<&image::RgbaImage>,
    ) -> Option<std::sync::Arc<prunr_core::selection::MaskArtifact>> {
        match (self.edge_feather_px, source) {
            (0, _) | (_, None) => None,
            (px, Some(src)) => Some(std::sync::Arc::new(
                prunr_core::selection::refine::feather_edges(mask, src, px),
            )),
        }
    }

    fn alpha(opacity: f32) -> u8 {
        (opacity * 255.0).round().clamp(0.0, 255.0) as u8
    }

    /// Dilation radius for the outline band: thickness 1 → the boundary
    /// pixels only, 10 → an 11 px band. `None` when no outline is drawn.
    fn band_radius(&self) -> Option<u32> {
        if self.outline_thickness < 0.5 || Self::alpha(self.outline_opacity) == 0 {
            return None;
        }
        Some(((self.outline_thickness - 1.0) / 2.0).round().max(0.0) as u32)
    }
}

/// ACCENT-tinted image: the selected interior at the fill alpha, the
/// outline band stamped over it at the outline alpha, transparent
/// elsewhere.
pub(crate) fn build_selection_image(
    mask: &prunr_core::selection::MaskArtifact,
    style: SelectionStyle,
) -> egui::ColorImage {
    use prunr_core::selection::MaskArtifact;
    let (w, h) = (mask.width as usize, mask.height as usize);
    let accent = super::theme::ACCENT;
    let paint = |a: u8| egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a);
    let fill = paint(SelectionStyle::alpha(style.fill_opacity));
    let mut pixels: Vec<egui::Color32> = mask
        .cells()
        .iter()
        .map(|&v| if MaskArtifact::is_selected(v) { fill } else { egui::Color32::TRANSPARENT })
        .collect();
    if let Some(r) = style.band_radius() {
        let outline = paint(SelectionStyle::alpha(style.outline_opacity));
        let r = r as usize;
        for (x, y) in prunr_core::selection::refine::outline_polyline(mask) {
            let (x, y) = (x as usize, y as usize);
            let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
            for row in y.saturating_sub(r)..=(y + r).min(h - 1) {
                pixels[row * w + x0..=row * w + x1].fill(outline);
            }
        }
    }
    egui::ColorImage::new([w, h], pixels)
}

/// The region of the rendered selection that differs between `base`
/// and `mask`, as `(top-left, image)`, or `None` when the planes are
/// identical or differ in size. A changed cell can change the rendering
/// of the pixels within the outline radius plus one (boundary status),
/// so that margin is kept; to render those correctly the crop needs the
/// same margin again, because `outline_polyline` treats the crop edge
/// as unselected.
pub(crate) fn build_selection_patch(
    base: &prunr_core::selection::MaskArtifact,
    mask: &prunr_core::selection::MaskArtifact,
    style: SelectionStyle,
) -> Option<([usize; 2], egui::ColorImage)> {
    let (w, h) = (mask.width, mask.height);
    let margin = style.band_radius().unwrap_or(0) + 1;
    let keep = base.diff_bbox(mask)?.grown(margin, w, h);
    let crop = keep.grown(margin, w, h);
    let rendered = build_selection_image(&mask.crop(crop), style);
    let pixels = sub_rect(
        &rendered.pixels,
        crop.width() as usize,
        (keep.x0 - crop.x0) as usize,
        (keep.y0 - crop.y0) as usize,
        keep.width() as usize,
        keep.height() as usize,
    );
    Some(([keep.x0 as usize, keep.y0 as usize], egui::ColorImage::new([keep.width() as usize, keep.height() as usize], pixels)))
}

/// The `w × h` block at `(x, y)` of a row-major buffer with `stride`.
fn sub_rect<T: Copy>(src: &[T], stride: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<T> {
    src.chunks(stride).skip(y).take(h).flat_map(|row| row[x..x + w].iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_core::selection::{MaskArtifact, FULL};

    fn centre_block() -> MaskArtifact {
        let mut data = vec![0i8; 64];
        for y in 2..6usize {
            for x in 2..6usize {
                data[y * 8 + x] = FULL;
            }
        }
        MaskArtifact::from_cells(8, 8, data)
    }

    fn style(fill: f32, outline: f32, thickness: f32) -> SelectionStyle {
        SelectionStyle { fill_opacity: fill, outline_opacity: outline, outline_thickness: thickness, edge_feather_px: 0 }
    }

    #[test]
    fn texture_bakes_fill_and_outline_alpha() {
        let img = build_selection_image(&centre_block(), style(0.15, 1.0, 1.0));
        assert_eq!(img.pixels[2 * 8 + 2].a(), 255, "boundary pixel carries the outline alpha");
        assert_eq!(img.pixels[3 * 8 + 3].a(), 38, "interior carries the fill alpha (0.15)");
        assert_eq!(img.pixels[0].a(), 0, "outside is transparent");

        let img = build_selection_image(&centre_block(), style(0.5, 0.25, 1.0));
        assert_eq!(img.pixels[3 * 8 + 3].a(), 128);
        assert_eq!(img.pixels[2 * 8 + 2].a(), 64);
    }

    #[test]
    fn zero_thickness_or_invisible_outline_bakes_no_band() {
        let img = build_selection_image(&centre_block(), style(0.15, 1.0, 0.0));
        assert_eq!(img.pixels[2 * 8 + 2].a(), 38, "no band: boundary pixel is plain fill");
        let img = build_selection_image(&centre_block(), style(0.15, 0.0, 4.0));
        assert_eq!(img.pixels[2 * 8 + 2].a(), 38, "invisible outline: boundary pixel is plain fill");
    }

    /// The patch must reproduce the full build inside its region, and
    /// the full build must be unchanged outside it.
    #[test]
    fn patch_equals_the_full_build_where_it_matters() {
        use prunr_core::brush::{paint_circle, Stamp};
        use prunr_core::selection::BrushMode;
        let mut base = MaskArtifact::new_empty(40, 30);
        paint_circle(&mut base, 12.0, 15.0, 7.0, Stamp { hardness: 1.0, strength: 1.0, mode: BrushMode::Add });
        let mut mask = base.clone();
        paint_circle(&mut mask, 20.0, 14.0, 5.0, Stamp { hardness: 0.5, strength: 1.0, mode: BrushMode::Add });
        for thickness in [0.0, 1.0, 3.0, 7.0] {
            let style = style(0.3, 0.8, thickness);
            let (pos, patch) = build_selection_patch(&base, &mask, style).expect("masks differ");
            let before = build_selection_image(&base, style);
            let after = build_selection_image(&mask, style);
            for y in 0..30 {
                for x in 0..40 {
                    let inside = (pos[0]..pos[0] + patch.size[0]).contains(&x) && (pos[1]..pos[1] + patch.size[1]).contains(&y);
                    let expected = after.pixels[y * 40 + x];
                    if inside {
                        assert_eq!(patch.pixels[(y - pos[1]) * patch.size[0] + (x - pos[0])], expected, "patch ({x},{y}) t={thickness}");
                    } else {
                        assert_eq!(before.pixels[y * 40 + x], expected, "outside ({x},{y}) t={thickness}");
                    }
                }
            }
        }
        assert!(build_selection_patch(&base, &base, style(0.3, 0.8, 1.0)).is_none(), "identical planes: nothing to patch");
    }

    #[test]
    fn thickness_widens_the_band_outward_in_image_pixels() {
        let img = build_selection_image(&centre_block(), style(0.15, 1.0, 3.0));
        // radius 1: the band reaches one pixel outside the block.
        assert_eq!(img.pixels[9].a(), 255); // (1, 1)
        assert_eq!(img.pixels[0].a(), 0);
    }
}
