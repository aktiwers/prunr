//! Getting results out: Save, Save layers, copy to the clipboard and drag-out.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

#[cfg(any(target_os = "windows", target_os = "macos"))]
use crate::gui::drag_export_state::DragExportState;
use crate::gui::item::{BatchItem, ExportDepth};
use crate::gui::system_bridge::DialogRequest;
use super::PrunrApp;

impl PrunrApp {
    /// The Save button's and Cmd+S's one gate: an image is shown or
    /// ticked. Save writes what the canvas shows; an untouched photo is
    /// written as it is, in the chosen format.
    pub fn can_save_intent(&self) -> bool {
        self.batch.selected_item().is_some() || self.batch.items.iter().any(|i| i.ticked)
    }

    /// "Separate layers" applies: the switch is on and the model cuts out a
    /// subject (layers come from its mask and lines).
    pub(crate) fn saves_layers(&self) -> bool {
        self.settings.export_split_layers && self.offers_layers()
    }

    /// "Separate layers" applies to this model: it cuts out a subject.
    pub(crate) fn offers_layers(&self) -> bool {
        self.settings.model.role() == crate::gui::knob_catalog::ModelRole::Cutout
    }

    pub fn handle_save_selected(&mut self) {
        let has_selection = self.batch.items.iter().any(|i| i.ticked);
        // Layers mode: always folder-picker, regardless of selection count.
        // The filenames are derived from each item's source stem + layer suffix.
        if self.saves_layers() {
            self.save_layers_to_folder(has_selection);
            return;
        }
        if has_selection {
            self.save_selected_to_folder();
        } else {
            self.save_current_to_file();
        }
    }

    /// Layers mode: ask for a folder for the ticked images, or the shown
    /// one; `on_save_layers_picked` writes into it.
    fn save_layers_to_folder(&mut self, has_selection: bool) {
        let targets: Vec<u64> = if has_selection {
            self.batch.ticked_ids()
        } else {
            self.batch.selected_item().map(|i| vec![i.id]).unwrap_or_default()
        };
        if !targets.is_empty() {
            self.ask_dialog(DialogRequest::SaveLayers { targets });
        }
    }

    /// Multi-file save for layers mode: up to 3 PNGs per target (subject /
    /// lines / mask), rendered and written on a background thread from a
    /// `LayerJob` snapshot. Per-item fallback to the composite when the
    /// layers' data is gone, so every target lands at least one file.
    pub(super) fn on_save_layers_picked(&mut self, targets: &[u64], folder: PathBuf) {
        let style = self.selection_style();
        let format = self.settings.export_format;
        let jobs: Vec<LayerSave> = targets.iter()
            .filter_map(|id| self.batch.find_by_id(*id))
            .map(|item| LayerSave {
                layers: crate::gui::drag_export::LayerJob::of(item, style, self.settings.brush.cutout_action),
                composite: (item.export_image(), item.export_bg()),
                depth: self.settings.export_depth_map.then(|| item.depth_export()),
            })
            .collect();
        let without_layers: Vec<&str> = targets.iter()
            .filter_map(|id| self.batch.find_by_id(*id))
            .zip(&jobs)
            .filter(|(_, job)| !job.layers.has_layers())
            .map(|(item, _)| item.filename.as_str())
            .collect();
        if !without_layers.is_empty() {
            self.toasts.info(format!(
                "Saved as one image, without layers: {}. Process it again for layers.",
                without_layers.join(", "),
            ));
        }
        self.toasts.info("Saving layers\u{2026}");
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_layers(folder, jobs, format, tx);
    }

    /// No checkboxes selected — save just the currently-viewed result.
    fn save_current_to_file(&mut self) {
        self.save_item_to_file(self.batch.selected_index);
    }

    /// Ask where to save one image's result, suggesting `<stem>.prunr.<ext>`
    /// from its source; `on_save_image_picked` writes it. The sidebar's
    /// per-row save and Save with nothing ticked.
    pub(crate) fn save_item_to_file(&mut self, idx: usize) {
        let Some(item) = self.batch.items.get(idx) else { return };
        let preferred = self.settings.export_format;
        let request = DialogRequest::SaveImage {
            item_id: item.id,
            default_name: crate::gui::drag_export::export_name(&item.filename, preferred),
            preferred,
        };
        self.ask_dialog(request);
    }

    /// Bake the background, encode in the format the chosen name asks for
    /// and write on a background thread. The image goes by id: one removed
    /// while the dialog was up is not saved, and no other takes its place.
    pub(super) fn on_save_image_picked(&mut self, item_id: u64, preferred: prunr_core::ExportFormat, path: PathBuf) {
        let Some(item) = self.batch.find_by_id(item_id) else {
            self.toasts.error("Not saved — the image was removed");
            return;
        };
        let depth = self.settings.export_depth_map.then(|| item.depth_export());
        let one = SaveOne { image: item.export_image(), bg: item.export_bg(), depth };
        let (path, format) = match prunr_core::ExportFormat::from_path(&path) {
            Some(f) => (path, f),
            None => (path.with_extension(preferred.extension()), preferred),
        };
        let tx = self.batch.bg_io.save_done_tx.clone();
        self.toasts.info("Saving\u{2026}");
        spawn_save_single(path, one, format, tx);
    }

    /// One or more sidebar checkboxes selected — ask for a folder;
    /// `on_save_selected_picked` writes into it.
    fn save_selected_to_folder(&mut self) {
        let targets = self.batch.ticked_ids();
        self.ask_dialog(DialogRequest::SaveSelected { targets });
    }

    /// Write each image as `<source-stem>.prunr.<ext>` into `folder`.
    /// Encode + write run on a background thread.
    pub(super) fn on_save_selected_picked(&mut self, targets: &[u64], folder: PathBuf) {
        let with_depth = self.settings.export_depth_map;
        let items: Vec<BatchSave> = targets.iter()
            .filter_map(|id| self.batch.find_by_id(*id))
            .map(|item| BatchSave {
                filename: item.filename.clone(),
                result: item.export_image(),
                bg: item.export_bg(),
                depth: with_depth.then(|| item.depth_export()),
            })
            .collect();
        if items.is_empty() {
            self.toasts.error("Not saved — the images were removed");
            return;
        }
        let count = items.len();
        self.toasts.info(format!("Saving {}\u{2026}", crate::gui::views::fmt::count(count, "image", "images")));
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_batch(folder, items, self.settings.export_format, tx);
    }

    /// Initiate an OS drag-out for the given batch item IDs.
    /// On Windows/macOS: calls the `drag` crate with PNG temp files.
    /// On Linux: a one-time fallback toast, no files written (winit and the
    /// drag crate conflict on Linux).
    pub fn initiate_drag_out(&mut self, ids: Vec<u64>, frame: &eframe::Frame, ctx: &egui::Context) {
        if !crate::gui::drag_export::DRAG_OUT_SUPPORTED {
            if !self.drag_export.linux_notified {
                self.drag_export.linux_notified = true;
                self.toasts.info(
                    "Drag to external apps isn't supported on Linux yet.\n\
                     Use Ctrl+C to copy to clipboard, or the Save button to export."
                );
            }
            return;
        }
        // The drag waits for files not written ahead yet, on workers, and
        // starts once they are in: writing them here held the frame.
        self.drag_export.begin(ids);
        self.continue_drag_out(frame, ctx);
    }

    /// A drag waiting for its files: starts once they are all written,
    /// while the button is still held; until then the cursor says it is
    /// working. Let go first, and there is no drag.
    pub(super) fn continue_drag_out(&mut self, frame: &eframe::Frame, ctx: &egui::Context) {
        use crate::gui::drag_export::{DragFiles, DragKey};
        use crate::gui::drag_export_state::DragReady;
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let _ = frame;
        let spec = self.drag_spec();
        let batch = &self.batch;
        let ready = self.drag_export.poll_start(
            |id| batch.find_by_id(id).map(|item| DragKey::of(item, spec)),
            |id| batch.find_by_id(id).map(|item| DragFiles::of(item, spec)),
        );
        let held = ctx.input(|i| i.pointer.any_down());
        let (ids, paths) = match ready {
            DragReady::Idle => return,
            DragReady::Waiting if held => {
                ctx.set_cursor_icon(egui::CursorIcon::Progress);
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
                return;
            }
            DragReady::Waiting => {
                self.drag_export.cancel_start();
                return;
            }
            DragReady::Ready { ids, paths, failed } => {
                for id in failed {
                    let name = self.batch.find_by_id(id).map_or("an image", |i| i.filename.as_str()).to_string();
                    self.toasts.error(format!("Could not drag {name} out. Save it instead."));
                }
                (ids, paths)
            }
        };
        if paths.is_empty() || !held {
            return;
        }

        // Publish dragged IDs so sidebar can dim those thumbnails.
        if let Ok(mut set) = self.drag_export.items.lock() {
            set.clear();
            set.extend(ids.iter().copied());
        }
        self.drag_export.active.store(true, Ordering::Release);

        #[cfg(any(target_os = "windows", target_os = "macos"))]
        {
            let active_flag = self.drag_export.active.clone();
            let items_set = self.drag_export.items.clone();
            let preview_path = paths[0].clone();

            let result = drag::start_drag(
                frame,
                drag::DragItem::Files(paths),
                drag::Image::File(preview_path),
                move |_result, _cursor| {
                    DragExportState::reset(&active_flag, &items_set);
                },
                drag::Options::default(),
            );
            if let Err(e) = result {
                DragExportState::reset(&self.drag_export.active, &self.drag_export.items);
                self.toasts.error(format!("Could not drag out: {e}"));
            }
        }
    }

    /// Write the drag files of the thumbnail under the pointer (and the
    /// other ticked images a drag of it takes) ahead, on workers, so a
    /// drag out of the window starts at once. True while more are due.
    pub(super) fn prewrite_drag_files(&mut self) -> bool {
        if !crate::gui::drag_export::DRAG_OUT_SUPPORTED {
            return false;
        }
        let Some(id) = self.drag_export.due(std::time::Instant::now()) else { return false };
        let Some(idx) = self.batch.index_of(id) else { return false };
        let targets = self.batch.drag_targets(idx);
        let spec = self.drag_spec();
        let batch = &self.batch;
        self.drag_export.queue_writes(
            id,
            &targets,
            |t| batch.find_by_id(t).map(|item| crate::gui::drag_export::DragKey::of(item, spec)),
            |t| batch.find_by_id(t).map(|item| crate::gui::drag_export::DragFiles::of(item, spec)),
        )
    }

    /// How a drag writes its files under the current settings.
    fn drag_spec(&self) -> crate::gui::drag_export::DragSpec {
        crate::gui::drag_export::DragSpec { split: self.saves_layers(), style: self.selection_style(), action: self.settings.brush.cutout_action }
    }

    pub fn handle_copy(&mut self) {
        // Selection rules:
        // - 0 ticked → copy the currently-viewed result (original behavior).
        // - 1 ticked → copy that one (even if not the currently-viewed item).
        // - 2+ ticked → copy the first, show a hint toast about drag-out.
        //   (System clipboards can't hold multiple images as bitmaps; drag-out
        //    is the native multi-image export path.)
        let shown = |b: &BatchItem| (b.export_image(), b.export_bg());
        let ticked: Vec<_> = self.batch.items.iter().filter(|b| b.ticked).map(shown).collect();
        let (to_copy, multi_hint) = match ticked.len() {
            0 => (self.batch.selected_item().map(shown), None),
            1 => (ticked.into_iter().next(), None),
            n => (
                ticked.into_iter().next(),
                Some(format!(
                    "Copied 1 of {n} selected. Drag thumbnails out of the window to export multiple.")
                ),
            ),
        };

        let Some((result, bg)) = to_copy else { return };
        // The background goes into the clipboard image so it matches the
        // canvas; a blur of the full photo, or a parked result to unpack,
        // is made off the UI thread.
        if let (true, Some(rgba)) = (bg.is_empty(), result.ready()) {
            let rgba = Arc::clone(rgba);
            self.system.copy_image(rgba, crate::gui::system_bridge::CopyNote::Image { hint: multi_hint });
            return;
        }
        let tx = self.batch.bg_io.copy_ready_tx.clone();
        std::thread::spawn(move || {
            let image = result.load().map(|rgba| bg.bake(&rgba, &ExportDepth::default()));
            let _ = tx.send((image, multi_hint));
        });
    }

    /// Say how a copy went once the clipboard has it.
    pub(super) fn on_copy_done(&mut self, note: crate::gui::system_bridge::CopyNote, took: bool) {
        use crate::gui::system_bridge::CopyNote;
        if !took {
            self.status_error("Could not copy to clipboard. Try saving instead.");
            return;
        }
        match note {
            CopyNote::Image { hint: Some(msg) } => {
                self.toasts.info(msg);
            }
            CopyNote::Image { hint: None } => self.status_success("Copied to clipboard"),
            CopyNote::Selection => self.status_success("Selection copied to clipboard"),
            CopyNote::SelectionCut => self.status_success("Selection cut to clipboard"),
        }
    }
}

/// One image to save: its result, what goes behind it, and its depth map
/// when Save options asks for it.
struct SaveOne {
    image: crate::gui::item::ExportImage,
    bg: crate::gui::item::ExportBg,
    depth: Option<crate::gui::item::DepthExport>,
}

/// Encode + write one image on a background thread, then send the result
/// toast text on `tx`. Stays at module scope so neither save method needs
/// `&mut self` once they've kicked off the work.
fn spawn_save_single(
    path: PathBuf,
    one: SaveOne,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<Result<String, String>>,
) {
    std::thread::spawn(move || {
        let depth = ExportDepth::default();
        let mut outcome = match one.image.load().map(|rgba| write_composite(&path, &rgba, &one.bg, format, &depth)) {
            Some(Ok(())) => Ok("Saved".to_string()),
            Some(Err(e)) => Err(format!("Could not save: {e}")),
            None => Err("Could not save: the image could not be read".to_string()),
        };
        let folder = path.parent().unwrap_or(Path::new("."));
        if let Some(Err(e)) = one.depth.map(|d| d.write(folder, &depth)) {
            tracing::warn!(%e, "depth map save failed");
            let msg = outcome.unwrap_or_else(|failed| failed);
            outcome = Err(format!("{msg}; could not save the depth map: {e}"));
        }
        let _ = tx.send(outcome);
    });
}

/// Bake `bg` behind `rgba`, encode as `format` and write it to `path`.
fn write_composite(
    path: &Path,
    rgba: &Arc<image::RgbaImage>,
    bg: &crate::gui::item::ExportBg,
    format: prunr_core::ExportFormat,
    measure: &ExportDepth,
) -> Result<(), String> {
    let bytes = format.encode(&bg.bake(rgba, measure)).map_err(|e| e.to_string())?;
    crate::fs_util::write_bytes_replacing(path, &bytes).map_err(|e| e.to_string())
}

/// What a multi-file save did: files saved, files that failed, and why
/// the first one failed.
#[derive(Default)]
struct SaveTally {
    saved: usize,
    failed: usize,
    first_error: Option<String>,
}

impl SaveTally {
    fn count(&mut self, written: Result<(), String>) {
        match written {
            Ok(()) => self.saved += 1,
            Err(why) => {
                tracing::warn!(%why, "a file of a save failed");
                self.failed += 1;
                self.first_error.get_or_insert(why);
            }
        }
    }

    /// The toast: an error naming the first failure when any file failed.
    fn summary(self, one: &str, many: &str) -> Result<String, String> {
        match self.first_error {
            Some(why) => Err(format!("Saved {}, failed {}: {why}", self.saved, self.failed)),
            None => Ok(format!("Saved {}", crate::gui::views::fmt::count(self.saved, one, many))),
        }
    }
}

/// Write `depth` into `folder` when asked for.
fn save_depth_beside(depth: Option<&crate::gui::item::DepthExport>, folder: &Path, measure: &ExportDepth, tally: &mut SaveTally) {
    if let Some(depth) = depth {
        tally.count(depth.write(folder, measure));
    }
}

/// One image of a layers save: its layers, and the composite to save
/// instead when it has none.
struct LayerSave {
    layers: crate::gui::drag_export::LayerJob,
    composite: (crate::gui::item::ExportImage, crate::gui::item::ExportBg),
    depth: Option<crate::gui::item::DepthExport>,
}

/// Render and write each image's layers into `folder` on a background
/// thread (the composite where an image has none); report the counts.
fn spawn_save_layers(folder: PathBuf, jobs: Vec<LayerSave>, format: prunr_core::ExportFormat, tx: mpsc::Sender<Result<String, String>>) {
    std::thread::spawn(move || {
        let mut tally = SaveTally::default();
        let depth = ExportDepth::default();
        for job in jobs {
            let mut files = job.layers.render_bytes(&depth);
            if files.is_empty() {
                let (image, bg) = &job.composite;
                let composite = image.load()
                    .ok_or_else(|| format!("{} could not be read", job.layers.filename()))
                    .and_then(|rgba| format.encode(&bg.bake(&rgba, &depth)).map_err(|e| e.to_string()));
                match composite {
                    Ok(bytes) => files.push((crate::gui::drag_export::export_name(job.layers.filename(), format), bytes)),
                    Err(why) => tally.count(Err(why)),
                }
            }
            for (name, bytes) in files {
                tally.count(crate::fs_util::write_bytes_replacing(&folder.join(&name), &bytes).map_err(|e| format!("{name}: {e}")));
            }
            save_depth_beside(job.depth.as_ref(), &folder, &depth, &mut tally);
        }
        let _ = tx.send(tally.summary("file", "files"));
    });
}

/// One image of a batch save.
struct BatchSave {
    filename: String,
    result: crate::gui::item::ExportImage,
    bg: crate::gui::item::ExportBg,
    depth: Option<crate::gui::item::DepthExport>,
}

/// Encode + write N images into `folder` (named by `drag_export::export_name`) on a
/// background thread; report aggregate counts when done.
fn spawn_save_batch(
    folder: PathBuf,
    items: Vec<BatchSave>,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<Result<String, String>>,
) {
    std::thread::spawn(move || {
        let mut tally = SaveTally::default();
        let measure = ExportDepth::default();
        for BatchSave { filename, result, bg, depth } in items {
            save_depth_beside(depth.as_ref(), &folder, &measure, &mut tally);
            let out_path = folder.join(crate::gui::drag_export::export_name(&filename, format));
            let written = result.load()
                .ok_or_else(|| format!("{filename} could not be read"))
                .and_then(|rgba| write_composite(&out_path, &rgba, &bg, format, &measure).map_err(|e| format!("{filename}: {e}")));
            tally.count(written);
        }
        let _ = tx.send(tally.summary("image", "images"));
    });
}

#[cfg(test)]
mod save_tally_tests {
    /// A save that wrote nothing because everything failed is an error,
    /// never "Saved 0", and it says why the first file failed.
    #[test]
    fn a_save_with_failures_says_why() {
        let mut all_failed = super::SaveTally::default();
        all_failed.count(Err("photo.png could not be read".into()));
        all_failed.count(Err("second reason".into()));
        assert_eq!(all_failed.summary("file", "files"), Err("Saved 0, failed 2: photo.png could not be read".into()));
        let mut fine = super::SaveTally::default();
        fine.count(Ok(()));
        assert_eq!(fine.summary("file", "files"), Ok("Saved 1 file".into()));
    }
}

/// Each status and save message keeps its toast kind: success for what
/// worked, error for what did not.
#[cfg(test)]
mod toast_kind_tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::gui::item::BatchStatus;
    use crate::gui::tests::fixtures::push_test_item;
    use super::PrunrApp;

    fn done_item(app: &mut PrunrApp, id: u64) {
        let item = push_test_item(app, id);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(3, 2, image::Rgba([10, 20, 30, 255]))));
    }

    /// The toast a save thread's answer turns into, once it lands.
    fn save_toast(app: &mut PrunrApp) -> (&'static str, String) {
        let before = app.toasts.len_for_test();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            app.drain_background_channels(&egui::Context::default());
            if app.toasts.len_for_test() > before {
                let (level, caption) = app.toasts.last_for_test().expect("a toast");
                return (level, caption.to_string());
            }
            assert!(Instant::now() < deadline, "the save never answered");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn an_image_that_will_not_load_is_an_error() {
        let mut app = PrunrApp::new_for_test();
        app.handle_open_path("/nonexistent/prunr-test/missing.png".into());
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not open missing.png: not an image Prunr can read")));
        app.handle_open_bytes(vec![1, 2, 3], "junk".into());
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not open junk: not an image Prunr can read")));
    }

    #[test]
    fn a_copy_says_whether_the_clipboard_took_it() {
        use crate::gui::system_bridge::CopyNote;
        let mut app = PrunrApp::new_for_test();
        let rgba = Arc::new(image::RgbaImage::new(2, 2));
        let land = |app: &mut PrunrApp| {
            for (note, took) in app.system.take_copy_outcomes() {
                app.on_copy_done(note, took);
            }
        };
        app.system.copy_image(Arc::clone(&rgba), CopyNote::Image { hint: None });
        land(&mut app);
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not copy to clipboard. Try saving instead.")));
        app.system.accept_copies_for_test();
        app.system.copy_image(Arc::clone(&rgba), CopyNote::Image { hint: None });
        land(&mut app);
        assert_eq!(app.toasts.last_for_test(), Some(("success", "Copied to clipboard")));
        app.system.copy_image(rgba, CopyNote::SelectionCut);
        land(&mut app);
        assert_eq!(app.toasts.last_for_test(), Some(("success", "Selection cut to clipboard")));
    }

    #[test]
    fn a_single_save_is_a_success_or_an_error() {
        let mut app = PrunrApp::new_for_test();
        done_item(&mut app, 1);
        let dir = tempfile::tempdir().unwrap();
        app.on_save_image_picked(1, prunr_core::ExportFormat::Png, dir.path().join("out.png"));
        assert_eq!(save_toast(&mut app), ("success", "Saved".to_string()));

        app.on_save_image_picked(1, prunr_core::ExportFormat::Png, dir.path().join("missing/out.png"));
        let (level, caption) = save_toast(&mut app);
        assert_eq!(level, "error");
        assert!(caption.starts_with("Could not save: "), "{caption}");
    }

    #[test]
    fn a_save_of_several_is_an_error_when_any_failed() {
        let mut app = PrunrApp::new_for_test();
        done_item(&mut app, 1);
        done_item(&mut app, 2);
        let dir = tempfile::tempdir().unwrap();
        app.on_save_selected_picked(&[1, 2], dir.path().to_path_buf());
        assert_eq!(save_toast(&mut app), ("success", "Saved 2 images".to_string()));
        app.on_save_selected_picked(&[1, 2], dir.path().join("missing"));
        let (level, caption) = save_toast(&mut app);
        assert_eq!(level, "error");
        assert!(caption.starts_with("Saved 0, failed 2: "), "{caption}: says why");

        app.on_save_layers_picked(&[1], dir.path().to_path_buf());
        assert_eq!(save_toast(&mut app), ("success", "Saved 1 file".to_string()));
        app.on_save_layers_picked(&[1], dir.path().join("missing"));
        let (level, caption) = save_toast(&mut app);
        assert_eq!(level, "error");
        assert!(caption.starts_with("Saved 0, failed 1: "), "{caption}: says why");
    }
}
