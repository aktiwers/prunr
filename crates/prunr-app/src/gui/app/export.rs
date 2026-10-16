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
        self.settings.export_split_layers
            && self.settings.model.role() == crate::gui::knob_catalog::ModelRole::Cutout
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
                composite: Some((item.export_image(), item.export_bg())),
                depth: self.settings.export_depth_map.then(|| item.depth_export()),
            })
            .collect();
        let without_layers: Vec<&str> = targets.iter()
            .filter_map(|id| self.batch.find_by_id(*id))
            .zip(&jobs)
            .filter(|(_, job)| !job.layers.has_layers() && job.composite.is_some())
            .map(|(item, _)| item.filename.as_str())
            .collect();
        if jobs.iter().all(|j| !j.layers.has_layers() && j.composite.is_none()) {
            self.toasts.error("Nothing to save — process the image first");
            return;
        }
        if !without_layers.is_empty() {
            self.toasts.info(format!(
                "Saved as one image, without layers: {}. Process it again for layers.",
                without_layers.join(", "),
            ));
        }
        self.toasts.info("Saving layers...");
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
        self.toasts.info("Saving...");
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
        self.toasts.info(format!("Saving {count} image(s)..."));
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_batch(folder, items, self.settings.export_format, tx);
    }

    /// Initiate an OS drag-out for the given batch item IDs.
    /// On Windows/macOS: calls the `drag` crate with PNG temp files.
    /// On Linux: a one-time fallback toast, no files written (winit and the
    /// drag crate conflict on Linux).
    pub fn initiate_drag_out(&mut self, ids: Vec<u64>, frame: &eframe::Frame) {
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let _ = frame;
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
        use crate::gui::drag_export::{DragFiles, DragKey};
        let spec = self.drag_spec();
        // Files not written ahead are written now, in parallel, while the
        // writes ahead still running finish.
        let mut ahead: Vec<(usize, u64, DragKey)> = Vec::new();
        let mut fresh: Vec<(usize, DragFiles)> = Vec::new();
        for (slot, id) in ids.iter().enumerate() {
            let Some(item) = self.batch.find_by_id(*id) else { continue };
            let key = DragKey::of(item, spec);
            if self.drag_export.has_current(*id, &key) {
                ahead.push((slot, *id, key));
            } else {
                fresh.push((slot, DragFiles::of(item, spec)));
            }
        }
        let mut written: Vec<Option<std::io::Result<Vec<PathBuf>>>> = ids.iter().map(|_| None).collect();
        let (slots, jobs): (Vec<usize>, Vec<DragFiles>) = fresh.into_iter().unzip();
        for (slot, result) in slots.into_iter().zip(crate::gui::drag_export::write_all(jobs)) {
            written[slot] = Some(result);
        }
        // A write ahead that failed or went stale meanwhile is written now,
        // with the rest of the misses.
        let mut missed: Vec<(usize, DragFiles)> = Vec::new();
        for (slot, id, key) in ahead {
            match self.drag_export.take_written(id, &key) {
                Some(paths) => written[slot] = Some(Ok(paths)),
                None => missed.extend(self.batch.find_by_id(id).map(|item| (slot, DragFiles::of(item, spec)))),
            }
        }
        let (slots, jobs): (Vec<usize>, Vec<DragFiles>) = missed.into_iter().unzip();
        for (slot, result) in slots.into_iter().zip(crate::gui::drag_export::write_all(jobs)) {
            written[slot] = Some(result);
        }
        let mut paths: Vec<PathBuf> = Vec::with_capacity(ids.len());
        for result in written.into_iter().flatten() {
            match result {
                Ok(mut ps) => paths.append(&mut ps),
                Err(e) => { self.toasts.error(format!("Drag export failed: {e}")); }
            }
        }
        if paths.is_empty() {
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
                self.toasts.error(format!("Drag failed: {e}"));
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
        if let (true, crate::gui::item::ExportImage::Shown(rgba)) = (bg.is_empty(), &result) {
            let rgba = Arc::clone(rgba);
            self.finish_copy(&rgba, multi_hint);
            return;
        }
        let tx = self.batch.bg_io.copy_ready_tx.clone();
        std::thread::spawn(move || {
            if let Some(rgba) = result.load() {
                let _ = tx.send((bg.bake(&rgba, &ExportDepth::default()), multi_hint));
            }
        });
    }

    /// Hand a copy to the clipboard and say how it went.
    pub(super) fn finish_copy(&mut self, rgba: &image::RgbaImage, multi_hint: Option<String>) {
        if self.system.copy_image(rgba) {
            if let Some(msg) = multi_hint {
                self.toasts.info(msg);
            } else {
                self.status_success("Copied to clipboard");
            }
        } else {
            self.status_error("Could not copy to clipboard. Try saving instead.");
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
            Some(Err(e)) => Err(format!("Save failed: {e}")),
            None => Err("Save failed: the image could not be read".to_string()),
        };
        let folder = path.parent().unwrap_or(Path::new("."));
        if let Some(Err(e)) = one.depth.map(|d| d.write(folder, &depth)) {
            tracing::warn!(%e, "depth map save failed");
            let msg = outcome.unwrap_or_else(|failed| failed);
            outcome = Err(format!("{msg}; depth map failed: {e}"));
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
    std::fs::write(path, &bytes).map_err(|e| e.to_string())
}

/// The toast after a multi-file save; an error when any file failed.
fn save_summary(saved: usize, failed: usize, noun: &str) -> Result<String, String> {
    if failed > 0 {
        Err(format!("Saved {saved}, failed {failed}"))
    } else {
        Ok(format!("Saved {saved} {noun}"))
    }
}

/// Write `depth` into `folder` when asked for; counts it as saved or failed.
fn save_depth_beside(depth: Option<&crate::gui::item::DepthExport>, folder: &Path, measure: &ExportDepth, saved: &mut usize, failed: &mut usize) {
    match depth.map(|d| d.write(folder, measure)) {
        Some(Ok(())) => *saved += 1,
        Some(Err(e)) => {
            tracing::warn!(%e, "depth map save failed");
            *failed += 1;
        }
        None => {}
    }
}

/// One image of a layers save: its layers, and the composite to save
/// instead when it has none.
struct LayerSave {
    layers: crate::gui::drag_export::LayerJob,
    composite: Option<(crate::gui::item::ExportImage, crate::gui::item::ExportBg)>,
    depth: Option<crate::gui::item::DepthExport>,
}

/// Render and write each image's layers into `folder` on a background
/// thread (the composite where an image has none); report the counts.
fn spawn_save_layers(folder: PathBuf, jobs: Vec<LayerSave>, format: prunr_core::ExportFormat, tx: mpsc::Sender<Result<String, String>>) {
    std::thread::spawn(move || {
        let mut saved = 0usize;
        let mut failed = 0usize;
        let depth = ExportDepth::default();
        for job in jobs {
            let mut files = job.layers.render_bytes(&depth);
            if files.is_empty() {
                if let Some((rgba, bg)) = job.composite.as_ref().and_then(|(r, bg)| Some((r.load()?, bg))) {
                    if let Ok(bytes) = format.encode(&bg.bake(&rgba, &depth)) {
                        files.push((crate::gui::drag_export::export_name(job.layers.filename(), format), bytes));
                    }
                }
            }
            for (name, bytes) in files {
                match std::fs::write(folder.join(name), &bytes) {
                    Ok(()) => saved += 1,
                    Err(_) => failed += 1,
                }
            }
            save_depth_beside(job.depth.as_ref(), &folder, &depth, &mut saved, &mut failed);
        }
        let _ = tx.send(save_summary(saved, failed, "file(s)"));
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
        let mut saved = 0usize;
        let mut failed = 0usize;
        let measure = ExportDepth::default();
        for BatchSave { filename, result, bg, depth } in items {
            save_depth_beside(depth.as_ref(), &folder, &measure, &mut saved, &mut failed);
            let out_path = folder.join(crate::gui::drag_export::export_name(&filename, format));
            let Some(rgba) = result.load() else {
                failed += 1;
                continue;
            };
            match write_composite(&out_path, &rgba, &bg, format, &measure) {
                Ok(()) => saved += 1,
                Err(_) => failed += 1,
            }
        }
        let _ = tx.send(save_summary(saved, failed, "image(s)"));
    });
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
        item.set_status_for_test(BatchStatus::Done);
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
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not load image")));
        app.handle_open_bytes(vec![1, 2, 3], "junk".into());
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not load image")));
    }

    #[test]
    fn a_copy_says_whether_the_clipboard_took_it() {
        let mut app = PrunrApp::new_for_test();
        let rgba = image::RgbaImage::new(2, 2);
        let took = app.system.copy_image(&rgba);
        app.finish_copy(&rgba, None);
        let expected = if took {
            ("success", "Copied to clipboard")
        } else {
            ("error", "Could not copy to clipboard. Try saving instead.")
        };
        assert_eq!(app.toasts.last_for_test(), Some(expected));

        app.system.drop_clipboard_for_test();
        app.finish_copy(&rgba, None);
        assert_eq!(app.toasts.last_for_test(), Some(("error", "Could not copy to clipboard. Try saving instead.")));
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
        assert!(caption.starts_with("Save failed: "), "{caption}");
    }

    #[test]
    fn a_save_of_several_is_an_error_when_any_failed() {
        let mut app = PrunrApp::new_for_test();
        done_item(&mut app, 1);
        done_item(&mut app, 2);
        let dir = tempfile::tempdir().unwrap();
        app.on_save_selected_picked(&[1, 2], dir.path().to_path_buf());
        assert_eq!(save_toast(&mut app), ("success", "Saved 2 image(s)".to_string()));
        app.on_save_selected_picked(&[1, 2], dir.path().join("missing"));
        assert_eq!(save_toast(&mut app), ("error", "Saved 0, failed 2".to_string()));

        app.on_save_layers_picked(&[1], dir.path().to_path_buf());
        assert_eq!(save_toast(&mut app), ("success", "Saved 1 file(s)".to_string()));
        app.on_save_layers_picked(&[1], dir.path().join("missing"));
        assert_eq!(save_toast(&mut app), ("error", "Saved 0, failed 1".to_string()));
    }
}
