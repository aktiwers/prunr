use std::sync::{mpsc, Arc, Condvar, Mutex};

/// An image's No-model result, and the start a chained pass built. A load
/// or decode failure is an `Err` on the same channel (mapped to
/// `BatchStatus::Error` in `drain_background_channels`).
/// A filter-only pass: the item, its run, the result, and the chained
/// start built for it.
pub type FilterOnlyResult = (u64, u64, Result<std::sync::Arc<image::RgbaImage>, String>, Option<std::sync::Arc<image::RgbaImage>>);

/// A background picture loaded off the UI thread: the item, the file,
/// whether the user picked it (it then becomes the background) or a preset
/// names it (it fills in the picture the preset's settings refer to), and
/// the decoded image with its content hash.
pub struct BgImageLoad {
    pub item_id: u64,
    pub path: std::path::PathBuf,
    pub picked: bool,
    pub result: Result<(image::DynamicImage, u64), String>,
}

/// Pre-decode result: `Ok(rgba)` on success, `Err(msg)` so a malformed
/// image clears `decode_pending` instead of leaving the item stuck.
pub type DecodeResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// An import, read off the UI thread: an image (a file, or bytes dropped
/// in) with its batch name, size in bytes and pixel size (`None`: its
/// header could not be read), or word that the folders dropped held no
/// images.
pub(crate) enum FileLoad {
    Image { source: crate::gui::item::ImageSource, name: String, len: usize, dims: Option<(u32, u32)> },
    NoImagesInFolder,
}

/// Where an uploaded texture lands.
pub(crate) enum TexLanding {
    /// An image texture, with the parked result its build unpacked.
    Image { item_id: u64, target: TexTarget, unpacked: Option<std::sync::Arc<image::RgbaImage>> },
    /// A whole selection overlay: the mask it shows and the feathered
    /// plane it was built from.
    Selection {
        item_id: u64,
        key: SelectionTextureKey,
        shown: Arc<prunr_core::selection::MaskArtifact>,
        feathered: Option<Arc<prunr_core::selection::MaskArtifact>>,
    },
    /// The picture behind the result, for `key`.
    Backdrop { item_id: u64, key: u64 },
}

impl TexLanding {
    pub(crate) fn item_id(&self) -> u64 {
        match *self {
            Self::Image { item_id, .. } | Self::Selection { item_id, .. } | Self::Backdrop { item_id, .. } => item_id,
        }
    }
}

/// A texture built off-thread for upload.
pub(crate) struct TexPrepMsg {
    pub item_id: u64,
    pub name: String,
    /// Its tiles; `None`: the pixels could not be read (a packed step
    /// whose file is gone).
    pub image: Option<crate::gui::image_texture::PreparedImage>,
    pub target: TexTarget,
    /// A parked result the build unpacked, for the item to hold again.
    pub unpacked: Option<std::sync::Arc<image::RgbaImage>>,
}

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
    /// The plane the texture now shows: the painted mask, or under an
    /// eraser its reach, which the next stroke patches from.
    pub(crate) shown: Arc<prunr_core::selection::MaskArtifact>,
}

pub(crate) enum SelectionImage {
    /// The whole overlay, and the feathered plane it was built from when
    /// the style feathers (so actions can reuse it).
    Full { image: crate::gui::image_texture::PreparedImage, feathered: Option<Arc<prunr_core::selection::MaskArtifact>> },
    /// The pixels that differ from the texture showing `base_hash`,
    /// to upload at `pos` into that texture.
    Patch { base_hash: u64, pos: [usize; 2], image: egui::ColorImage },
}

/// Counting semaphore bounding how many background jobs (decode, thumbnail,
/// filter-only pass, background picture, texture prep, history compression)
/// run at once. Without it, a 50-image Process All fans out 50
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
    pub(crate) tx: mpsc::Sender<TexPrepMsg>,
    pub slots: Arc<DecodeSlots>,
}

/// Bundles all background thread communication channels.
pub struct BackgroundIO {
    /// File paths from file dialog / drag-and-drop (loaded lazily on demand)
    pub(crate) file_load_tx: mpsc::Sender<FileLoad>,
    pub(crate) file_load_rx: mpsc::Receiver<FileLoad>,
    /// Thumbnail generation results
    pub thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
    pub thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    /// Pre-decoded source images for instant canvas switching
    pub decode_tx: mpsc::Sender<DecodeResult>,
    pub decode_rx: mpsc::Receiver<DecodeResult>,
    /// A finished save's toast text: `Ok` when everything was written,
    /// `Err` when anything failed.
    pub save_done_tx: mpsc::Sender<Result<String, String>>,
    pub save_done_rx: mpsc::Receiver<Result<String, String>>,
    /// A model or runtime removed off the UI thread: its toast text. The
    /// installed set is read again when it lands.
    pub(crate) removed_tx: mpsc::Sender<Result<String, String>>,
    pub(crate) removed_rx: mpsc::Receiver<Result<String, String>>,
    /// A copy whose background was baked off the UI thread, for the
    /// clipboard, with the note to show once copied.
    pub copy_ready_tx: mpsc::Sender<(Arc<image::RgbaImage>, Option<String>)>,
    pub copy_ready_rx: mpsc::Receiver<(Arc<image::RgbaImage>, Option<String>)>,
    /// Selection Delete / Cut / Copy made off the UI thread.
    pub(crate) selection_edit_tx: mpsc::Sender<SelectionEditDone>,
    pub(crate) selection_edit_rx: mpsc::Receiver<SelectionEditDone>,
    /// Pre-built ColorImages ready for GPU upload.
    pub(crate) tex_prep_tx: mpsc::Sender<TexPrepMsg>,
    pub(crate) tex_prep_rx: mpsc::Receiver<TexPrepMsg>,
    /// Prepared textures uploading a few tiles per frame, with what each
    /// lands on.
    pub(crate) uploads: crate::gui::image_texture::Uploads<TexLanding>,
    /// Filter-only (model=None) Process results — keeps the UI thread free
    /// while load + decode + `apply_subject_effect` runs per item.
    pub filter_only_tx: mpsc::Sender<FilterOnlyResult>,
    pub filter_only_rx: mpsc::Receiver<FilterOnlyResult>,
    /// Off-thread RGBA decode + ColorImage build for a per-item bg
    /// image. Drain calls `ctx.load_texture` with the bg-specific
    /// options (Linear filter + Repeat wrap for `BgImageFit::Tile`).
    /// The `to_rgba8()` copy (~32 MB on a 4K bg) stays off the UI thread.
    pub(crate) bg_tex_prep_tx: mpsc::Sender<(u64, u64, Option<crate::gui::image_texture::PreparedImage>)>,
    pub(crate) bg_tex_prep_rx: mpsc::Receiver<(u64, u64, Option<crate::gui::image_texture::PreparedImage>)>,
    pub bg_image_tx: mpsc::Sender<BgImageLoad>,
    pub bg_image_rx: mpsc::Receiver<BgImageLoad>,
    /// Bounds simultaneous background jobs to `available_parallelism()`.
    /// Threads spawn immediately but park here until a slot opens,
    /// capping transient RAM at N × per-thread peak.
    pub decode_slots: Arc<DecodeSlots>,
    /// Selection texture built off-thread whenever the selected item's
    /// texture is missing. Drained via ctx.load_texture (allowed —
    /// drain_background_channels runs in logic(), not a render closure).
    pub(crate) selection_texture_tx: mpsc::Sender<SelectionTextureResult>,
    pub(crate) selection_texture_rx: mpsc::Receiver<SelectionTextureResult>,
}

/// What a prepared texture is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexTarget {
    Source,
    /// A result from a discrete change: it reveals over the held image.
    /// Each result texture carries the item's request number: only the
    /// newest asked for lands, as preps finish in any order.
    Result(u64),
    /// A live re-cut during a slider drag: swaps in at once.
    LiveResult(u64),
    /// A drag's quick frame of this generation: shown over the image, not
    /// the result.
    QuickFrame(u64),
    /// The unprocessed photo's look (restored faces), shown in its place.
    PhotoLook,
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
        let (removed_tx, removed_rx) = mpsc::channel();
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();
        let (decode_tx, decode_rx) = mpsc::channel();
        let (save_done_tx, save_done_rx) = mpsc::channel();
        let (copy_ready_tx, copy_ready_rx) = mpsc::channel();
        let (selection_edit_tx, selection_edit_rx) = mpsc::channel();
        let (tex_prep_tx, tex_prep_rx) = mpsc::channel();
        let (filter_only_tx, filter_only_rx) = mpsc::channel();
        let (bg_tex_prep_tx, bg_tex_prep_rx) = mpsc::channel();
        let (bg_image_tx, bg_image_rx) = mpsc::channel();
        let (selection_texture_tx, selection_texture_rx) = mpsc::channel();
        let cap = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            removed_tx, removed_rx,
            file_load_tx, file_load_rx,
            thumb_tx, thumb_rx,
            decode_tx, decode_rx,
            save_done_tx, save_done_rx,
            copy_ready_tx, copy_ready_rx,
            selection_edit_tx, selection_edit_rx,
            tex_prep_tx, tex_prep_rx,
            uploads: Default::default(),
            filter_only_tx, filter_only_rx,
            bg_tex_prep_tx, bg_tex_prep_rx,
            bg_image_tx, bg_image_rx,
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
            let patch_from = |base: Option<(u64, Arc<prunr_core::selection::MaskArtifact>)>, plane: &prunr_core::selection::MaskArtifact| {
                base.and_then(|(base_hash, base)| {
                    let (pos, image) = build_selection_patch(&base, plane, style)?;
                    Some(SelectionImage::Patch { base_hash, pos, image })
                })
            };
            // An eraser's region reaches beyond the painted cells: its
            // overlay is drawn from the reach, and patched from the last
            // reach rather than from the paint.
            let (image, shown) = match style.shown_region(&mask) {
                Some(reach) => {
                    let image = patch_from(base, &reach)
                        .unwrap_or_else(|| SelectionImage::Full { image: prepared(build_selection_image(&reach, style)), feathered: None });
                    (image, reach)
                }
                None => {
                    let image = patch_from(base, &mask).unwrap_or_else(|| {
                        let feathered = style.feather(&mask, source.as_deref());
                        let shown = feathered.as_ref().unwrap_or(&mask);
                        SelectionImage::Full { image: prepared(build_selection_image(shown, style)), feathered }
                    });
                    (image, mask)
                }
            };
            let _ = texture_tx.send(SelectionTextureResult { item_id, image, key, shown });
            // Some compositors drop thread-initiated wake-ups while the
            // window is idle; the pending key's poll in `logic()` is the
            // fallback.
            ctx.request_repaint();
        });
    }
}

/// Narrowest outline, in screen points, the overlay draws at any zoom.
const MIN_OUTLINE_SCREEN_PX: f32 = 1.5;

/// A selection Delete / Cut / Copy made off the UI thread: the copy for
/// the clipboard and the cut image, for the item at the timeline version
/// it started from.
pub(crate) struct SelectionEditDone {
    pub(crate) item_id: u64,
    pub(crate) version: u64,
    pub(crate) copy: Option<(image::RgbaImage, crate::gui::system_bridge::CopyNote)>,
    pub(crate) cut: Option<image::RgbaImage>,
}

/// What a selection texture bakes in; the overlay is a single textured
/// quad drawn untinted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SelectionStyle {
    pub(crate) fill_opacity: f32,
    /// Drawn in Erase's warm red: on a cut-out, the brush erases.
    pub(crate) erase: bool,
    /// Outline width in image pixels; 0 draws no outline.
    pub(crate) outline_thickness: f32,
    pub(crate) edge_feather_px: u32,
    /// Under an eraser, the overlay shows the region the erase will
    /// touch: the painted one grown, plus its blend band.
    pub(crate) fill: Option<FillRegionPx>,
}

/// An eraser's Expand region and Feather, in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FillRegionPx {
    pub(crate) grow: f32,
    pub(crate) blend: f32,
}

impl SelectionStyle {
    /// Outline width in image pixels at 1:1; `at_zoom` widens it.
    const OUTLINE_PX: f32 = 2.0;

    pub(crate) fn from_settings(s: &super::settings::Settings) -> Self {
        Self {
            fill_opacity: s.selection_fill,
            erase: s.brush_erases(),
            outline_thickness: if s.selection_outline { Self::OUTLINE_PX } else { 0.0 },
            edge_feather_px: s.brush.edge_feather.round().max(0.0) as u32,
            fill: s.model.is_inpaint().then_some(FillRegionPx { grow: s.brush.inpaint_grow, blend: s.brush.inpaint_feather }),
        }
    }

    /// Under an eraser, the plane the overlay draws for `mask`: what the
    /// erase will touch. `None` shows `mask` as it is.
    pub(crate) fn shown_region(&self, mask: &prunr_core::selection::MaskArtifact) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        self.fill.map(|f| Arc::new(prunr_core::inpaint_blend::fill_preview(mask, f.grow, f.blend)))
    }

    /// The default outline and no feather, with `fill_opacity`.
    #[cfg(test)]
    pub(crate) fn with_fill(fill_opacity: f32) -> Self {
        Self { fill_opacity, erase: false, outline_thickness: Self::OUTLINE_PX, edge_feather_px: 0, fill: None }
    }

    /// The feathered selection, or `None` when this style shows `mask` as
    /// it is (no feather, or no photo to feather against). The texture
    /// build, the re-cut and Delete / Copy / Cut all go through here.
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

    /// The style as drawn at `zoom` (screen points per image pixel): an
    /// outline that is on gets at least `MIN_OUTLINE_SCREEN_PX` on screen,
    /// widened in odd whole-pixel bands (the dilation's steps) so the
    /// texture is rebuilt only when the band changes.
    pub(crate) fn at_zoom(self, zoom: f32) -> Self {
        if self.band_radius().is_none() || zoom <= 0.0 {
            return self;
        }
        let needed = MIN_OUTLINE_SCREEN_PX / zoom;
        if needed <= self.outline_thickness {
            return self;
        }
        let band = 2.0 * ((needed - 1.0) / 2.0).ceil() + 1.0;
        Self { outline_thickness: band, ..self }
    }

    fn alpha(opacity: f32) -> u8 {
        (opacity * 255.0).round().clamp(0.0, 255.0) as u8
    }

    /// Dilation radius for the outline band: thickness 1 → the boundary
    /// pixels only, 10 → an 11 px band. `None` when no outline is drawn.
    fn band_radius(&self) -> Option<u32> {
        if self.outline_thickness < 0.5 {
            return None;
        }
        Some(((self.outline_thickness - 1.0) / 2.0).round().max(0.0) as u32)
    }
}

/// ACCENT-tinted image (Erase's red when `style.erase`): each cell at the fill alpha scaled by its
/// coverage, so a feathered edge or a faint stroke shows as such, and
/// An overlay cut into tiles on the thread that built it.
fn prepared(image: egui::ColorImage) -> crate::gui::image_texture::PreparedImage {
    crate::gui::image_texture::PreparedImage::from_color_image(image)
}

/// the outline band stamped over it.
pub(crate) fn build_selection_image(
    mask: &prunr_core::selection::MaskArtifact,
    style: SelectionStyle,
) -> egui::ColorImage {
    use prunr_core::selection::MaskArtifact;
    let (w, h) = (mask.width as usize, mask.height as usize);
    let accent = super::theme::mark_color(style.erase);
    let paint = |a: u8| egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a);
    let fill_alpha = SelectionStyle::alpha(style.fill_opacity) as f32;
    // A cell's colour depends only on its value: one per value.
    let lut: [egui::Color32; 256] = std::array::from_fn(|i| paint((fill_alpha * MaskArtifact::coverage(i as u8 as i8)).round() as u8));
    let mut pixels: Vec<egui::Color32> = mask.cells().iter().map(|&v| lut[v as u8 as usize]).collect();
    if let Some(r) = style.band_radius() {
        let outline = paint(255);
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

    /// Under an eraser the overlay shows the erase's reach, and the
    /// region knobs are part of its key, so a change redraws it.
    #[test]
    fn an_erasers_region_knobs_shape_the_overlay() {
        use crate::gui::settings::{Settings, SettingsModel};
        let mut s = Settings { model: SettingsModel::BiRefNetLite, ..Default::default() };
        assert_eq!(SelectionStyle::from_settings(&s).fill, None, "a cut-out shows the paint as it is");
        s.model = SettingsModel::Inpaint;
        let style = SelectionStyle::from_settings(&s);
        assert_eq!(style.fill, Some(FillRegionPx { grow: s.brush.inpaint_grow, blend: s.brush.inpaint_feather }));
        s.brush.inpaint_grow += 3.0;
        assert_ne!(SelectionStyle::from_settings(&s), style, "a knob change is a new key");

        let mut cells = vec![0i8; 9 * 9];
        cells[4 * 9 + 4] = FULL;
        let painted = MaskArtifact::from_cells(9, 9, cells);
        let reach = SelectionStyle { fill: Some(FillRegionPx { grow: 1.0, blend: 0.0 }), ..style }.shown_region(&painted).expect("an eraser shows the reach");
        assert_eq!(reach.cells()[4 * 9 + 3], FULL, "the overlay shows the grown region");
        assert!(SelectionStyle::with_fill(0.5).shown_region(&painted).is_none());
    }

    /// An eraser's overlay is drawn from the reach and a stroke patches it
    /// from the last reach: the patch over the first build equals a fresh
    /// build of the new reach.
    #[test]
    fn an_erasers_overlay_patches_from_its_last_reach() {
        let style = SelectionStyle { fill: Some(FillRegionPx { grow: 1.0, blend: 0.0 }), ..SelectionStyle::with_fill(0.5) };
        let painted = |cells: &[usize]| {
            let mut v = vec![0i8; 16 * 16];
            for &c in cells { v[c] = FULL; }
            Arc::new(MaskArtifact::from_cells(16, 16, v))
        };
        let (a, b) = (painted(&[5 * 16 + 5]), painted(&[5 * 16 + 5, 9 * 16 + 9]));
        let io = BackgroundIO::new();
        let ctx = egui::Context::default();
        let wait = std::time::Duration::from_secs(10);
        io.request_selection_visualization(1, Arc::clone(&a), (1, style), None, None, ctx.clone());
        let first = io.selection_texture_rx.recv_timeout(wait).expect("the first build");
        let SelectionImage::Full { .. } = first.image else { panic!("a first build is whole") };
        let full_a = build_selection_image(&first.shown, style);
        assert_eq!(first.shown.cells(), style.shown_region(&a).expect("an eraser").cells(), "the texture remembers the reach it drew");
        io.request_selection_visualization(1, Arc::clone(&b), (2, style), None, Some((1, Arc::clone(&first.shown))), ctx);
        let second = io.selection_texture_rx.recv_timeout(wait).expect("the second build");
        let SelectionImage::Patch { base_hash: 1, pos, image: patch } = second.image else { panic!("a stroke patches the last reach") };
        let mut pasted = full_a;
        for (row, chunk) in patch.pixels.chunks(patch.width()).enumerate() {
            let start = (pos[1] + row) * 16 + pos[0];
            pasted.pixels[start..start + patch.width()].copy_from_slice(chunk);
        }
        let fresh = build_selection_image(&style.shown_region(&b).expect("an eraser"), style);
        assert_eq!(pasted.pixels, fresh.pixels);
        assert_eq!(second.shown.cells(), style.shown_region(&b).expect("an eraser").cells());
    }

    fn centre_block() -> MaskArtifact {
        let mut data = vec![0i8; 64];
        for y in 2..6usize {
            for x in 2..6usize {
                data[y * 8 + x] = FULL;
            }
        }
        MaskArtifact::from_cells(8, 8, data)
    }

    fn style(fill: f32, thickness: f32) -> SelectionStyle {
        SelectionStyle { outline_thickness: thickness, ..SelectionStyle::with_fill(fill) }
    }

    /// A faint or feathered cell shows faintly: the fill is scaled by
    /// coverage, so strokes under half coverage are visible too.
    #[test]
    fn the_overlay_fill_follows_coverage() {
        use prunr_core::selection::{MaskArtifact, FULL};
        let mask = MaskArtifact::from_cells(3, 1, vec![FULL, -(FULL / 4), 0]);
        let no_outline = SelectionStyle { outline_thickness: 0.0, ..SelectionStyle::with_fill(0.8) };
        let img = build_selection_image(&mask, no_outline);
        let (full, faint, none) = (img.pixels[0].a(), img.pixels[1].a(), img.pixels[2].a());
        assert!(faint > 0 && faint < full, "a quarter-coverage stroke shows, fainter ({faint} vs {full})");
        assert_eq!(none, 0);
    }

    /// On a cut-out, a selection while the switch is on Erase is drawn in
    /// Erase's warm red; otherwise in the accent.
    #[test]
    fn an_erasing_brush_marks_in_warm_red() {
        use crate::gui::brush_state::CutoutAction;
        use crate::gui::settings::{Settings, SettingsModel};
        use prunr_core::selection::{MaskArtifact, FULL};
        let mask = MaskArtifact::from_cells(1, 1, vec![FULL]);
        let tint = |model, action| {
            let mut s = Settings { model, ..Settings::default() };
            s.brush.cutout_action = action;
            let style = SelectionStyle { outline_thickness: 0.0, ..SelectionStyle::from_settings(&s) };
            build_selection_image(&mask, style).pixels[0].to_srgba_unmultiplied()
        };
        // Premultiplied storage rounds each channel by a step or two.
        let near = |got: [u8; 4], want: egui::Color32| {
            got[..3].iter().zip([want.r(), want.g(), want.b()]).all(|(&g, w)| g.abs_diff(w) <= 3)
        };
        assert!(near(tint(SettingsModel::BiRefNetLite, CutoutAction::Erase), crate::gui::theme::ERASE));
        assert!(near(tint(SettingsModel::BiRefNetLite, CutoutAction::Restore), crate::gui::theme::ACCENT));
        assert!(near(tint(SettingsModel::BiRefNetLite, CutoutAction::Select), crate::gui::theme::ACCENT));
        assert!(near(tint(SettingsModel::Inpaint, CutoutAction::Erase), crate::gui::theme::ACCENT), "a region for the eraser is not an erase");
    }

    /// At fit zoom on a big image a 2 px outline is half a screen pixel and
    /// fades out under bilinear sampling; it widens to stay visible, in
    /// odd whole-pixel steps so a zoom drag rebuilds the overlay only a
    /// few times.
    #[test]
    fn the_outline_stays_visible_when_zoomed_out() {
        let s = style(0.15, 2.0);
        assert_eq!(s.at_zoom(1.0).outline_thickness, 2.0, "at 1:1 the setting holds");
        assert_eq!(s.at_zoom(0.75).outline_thickness, 2.0);
        assert_eq!(s.at_zoom(0.25).outline_thickness, 7.0, "6 px needed, next odd band");
        assert!(s.at_zoom(0.25).outline_thickness * 0.25 >= MIN_OUTLINE_SCREEN_PX);
        assert_eq!(s.at_zoom(0.26), s.at_zoom(0.27), "nearby zooms share one texture");
        assert_eq!(style(0.15, 0.0).at_zoom(0.25).outline_thickness, 0.0, "an outline turned off stays off");
        assert_eq!(style(0.15, 9.0).at_zoom(0.25).outline_thickness, 9.0, "a wider setting wins");
    }

    #[test]
    fn texture_bakes_fill_and_outline_alpha() {
        let img = build_selection_image(&centre_block(), style(0.15, 1.0));
        assert_eq!(img.pixels[2 * 8 + 2].a(), 255, "boundary pixel carries the outline alpha");
        assert_eq!(img.pixels[3 * 8 + 3].a(), 38, "interior carries the fill alpha (0.15)");
        assert_eq!(img.pixels[0].a(), 0, "outside is transparent");

        let img = build_selection_image(&centre_block(), style(0.5, 1.0));
        assert_eq!(img.pixels[3 * 8 + 3].a(), 128);
    }

    #[test]
    fn an_outline_turned_off_bakes_no_band() {
        let img = build_selection_image(&centre_block(), style(0.15, 0.0));
        assert_eq!(img.pixels[2 * 8 + 2].a(), 38, "no band: boundary pixel is plain fill");
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
            let style = style(0.3, thickness);
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
        assert!(build_selection_patch(&base, &base, style(0.3, 1.0)).is_none(), "identical planes: nothing to patch");
    }

    #[test]
    fn thickness_widens_the_band_outward_in_image_pixels() {
        let img = build_selection_image(&centre_block(), style(0.15, 3.0));
        // radius 1: the band reaches one pixel outside the block.
        assert_eq!(img.pixels[9].a(), 255); // (1, 1)
        assert_eq!(img.pixels[0].a(), 0);
    }
}
