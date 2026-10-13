//! Getting results out: Save, Save layers, copy to the clipboard and drag-out.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

#[cfg(any(target_os = "windows", target_os = "macos"))]
use crate::gui::drag_export_state::DragExportState;
use crate::gui::item::{BatchItem, BatchStatus, ExportDepth};
use crate::gui::state::AppState;
use super::PrunrApp;

impl PrunrApp {
    /// The Save button's and Cmd+S's one gate: the shown image has a
    /// result, or a ticked image does.
    pub fn can_save_intent(&self) -> bool {
        self.batch.app_state() == AppState::Done
            || self.batch.items.iter().any(|i| i.selected && i.status == BatchStatus::Done)
    }

    /// "Separate layers" applies: the switch is on and the model cuts out a
    /// subject (layers come from its mask and lines).
    pub(crate) fn saves_layers(&self) -> bool {
        self.settings.export_split_layers
            && self.settings.model.role() == crate::gui::knob_catalog::ModelRole::Cutout
    }

    pub fn handle_save_selected(&mut self) {
        let has_selection = self.batch.items.iter()
            .any(|i| i.selected && i.status == BatchStatus::Done && i.result_rgba.is_some());
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

    /// Folder-picker + multi-file save for layers mode: up to 3 PNGs per
    /// target (subject / lines / mask), rendered and written on a background
    /// thread from a `LayerJob` snapshot. Per-item fallback to the composite
    /// when the layers' data is gone, so every target lands at least one file.
    fn save_layers_to_folder(&mut self, has_selection: bool) {
        let Some(folder) = self.system.pick_folder_dialog(
            self.last_open_dir.as_deref(),
            "Save layers \u{2014} Choose folder",
        ) else { return };

        let targets: Vec<u64> = if has_selection {
            self.batch.items.iter()
                .filter(|i| i.selected && i.status == BatchStatus::Done)
                .map(|i| i.id)
                .collect()
        } else {
            self.batch.selected_item().map(|i| vec![i.id]).unwrap_or_default()
        };
        if targets.is_empty() { return; }

        let style = self.selection_style();
        let format = self.settings.export_format;
        let jobs: Vec<LayerSave> = targets.iter()
            .filter_map(|id| self.batch.find_by_id(*id))
            .map(|item| LayerSave {
                layers: crate::gui::drag_export::LayerJob::of(item, style, self.settings.brush.cutout_action),
                composite: item.result_ref().map(|r| (r, item.export_bg())),
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

    /// No checkboxes selected — save just the currently-viewed result via a
    /// save-as dialog. Suggests a `<stem>.prunr.png` name based on the source.
    fn save_current_to_file(&mut self) {
        let Some(item) = self.batch.selected_item() else { return };
        let Some(rgba) = item.result_rgba.clone() else { return };
        let default_name = crate::gui::drag_export::export_name(&item.filename, self.settings.export_format);
        let bg = item.export_bg();
        let depth = self.settings.export_depth_map.then(|| item.depth_export());
        self.save_rgba_with_dialog(SaveOne { rgba, bg, depth }, &default_name);
    }

    /// Save one specific batch item (by index) via a save-as dialog. Used by
    /// the sidebar's per-row save button, which needs a non-selection-based
    /// entry point into the same encode-on-background pipeline as
    /// `save_current_to_file`.
    pub(crate) fn save_item_to_file(&mut self, idx: usize) {
        let (one, default_name) = {
            let Some(item) = self.batch.items.get(idx) else { return };
            let Some(rgba) = item.result_rgba.clone() else { return };
            let depth = self.settings.export_depth_map.then(|| item.depth_export());
            (SaveOne { rgba, bg: item.export_bg(), depth }, crate::gui::drag_export::export_name(&item.filename, self.settings.export_format))
        };
        self.save_rgba_with_dialog(one, &default_name);
    }

    /// Shared tail for the two single-item save paths: open the save-as
    /// dialog, then bake the background, encode in the format the chosen
    /// name asks for and write on a background thread.
    fn save_rgba_with_dialog(&mut self, one: SaveOne, default_name: &str) {
        let preferred = self.settings.export_format;
        let Some(path) = self.system.save_image_dialog(
            self.last_open_dir.as_deref(),
            default_name,
            preferred,
        ) else { return };
        let (path, format) = match prunr_core::ExportFormat::from_path(&path) {
            Some(f) => (path, f),
            None => (path.with_extension(preferred.extension()), preferred),
        };
        let tx = self.batch.bg_io.save_done_tx.clone();
        self.toasts.info("Saving...");
        spawn_save_single(path, one, format, tx);
    }

    /// One or more sidebar checkboxes selected — pick a folder, write each
    /// as `<source-stem>.prunr.png`. Encode + write run on a background thread.
    fn save_selected_to_folder(&mut self) {
        let with_depth = self.settings.export_depth_map;
        let items: Vec<BatchSave> = self.batch.items.iter()
            .filter(|i| i.selected && i.status == BatchStatus::Done)
            .filter_map(|item| Some(BatchSave {
                filename: item.filename.clone(),
                result: item.result_ref()?,
                bg: item.export_bg(),
                depth: with_depth.then(|| item.depth_export()),
            }))
            .collect();
        let Some(folder) = self.system.pick_folder_dialog(
            self.last_open_dir.as_deref(),
            "Save Selected \u{2014} Choose Folder",
        ) else { return };
        let count = items.len();
        self.toasts.info(format!("Saving {count} image(s)..."));
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_batch(folder, items, self.settings.export_format, tx);
    }

    /// Initiate an OS drag-out for the given batch item IDs.
    /// On Windows/macOS: calls the `drag` crate with PNG temp files.
    /// On Linux: a one-time fallback toast, no files written (winit + drag
    /// crate incompatibility; see Cargo.toml comment).
    #[allow(unused_variables)]
    pub fn initiate_drag_out(&mut self, ids: Vec<u64>, frame: &eframe::Frame) {
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
        // - 0 checkbox-selected → copy the currently-viewed result (original behavior).
        // - 1 checkbox-selected → copy that one (even if not the currently-viewed item).
        // - 2+ checkbox-selected → copy the first, show a hint toast about drag-out.
        //   (System clipboards can't hold multiple images as bitmaps; drag-out
        //    is the native multi-image export path.)
        let with_result = |b: &BatchItem| Some((b.result_ref()?, b.export_bg()));
        let ticked: Vec<_> = self.batch.items.iter().filter(|b| b.selected).filter_map(with_result).collect();
        let (to_copy, multi_hint) = match ticked.len() {
            0 => (self.batch.selected_item().and_then(with_result), None),
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
        if let (true, crate::gui::item::ResultRef::Shown(rgba)) = (bg.is_empty(), &result) {
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
                self.set_temporary_status("Copied to clipboard");
            }
        } else {
            self.set_temporary_status("Could not copy to clipboard. Try saving instead.");
        }
    }
}

/// Encode + write one image on a background thread, then send the result
/// toast text on `tx`. Stays at module scope so neither save method needs
/// `&mut self` once they've kicked off the work.
/// One image to save: its result, what goes behind it, and its depth map
/// when Save options asks for it.
struct SaveOne {
    rgba: Arc<image::RgbaImage>,
    bg: crate::gui::item::ExportBg,
    depth: Option<crate::gui::item::DepthExport>,
}

fn spawn_save_single(
    path: PathBuf,
    one: SaveOne,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<String>,
) {
    std::thread::spawn(move || {
        let depth = ExportDepth::default();
        let mut msg = match format.encode(&one.bg.bake(&one.rgba, &depth)) {
            Ok(bytes) => match std::fs::write(&path, &bytes) {
                Ok(()) => "Saved".into(),
                Err(e) => format!("Save failed: {e}"),
            },
            Err(e) => format!("Save failed: {e}"),
        };
        let folder = path.parent().unwrap_or(Path::new("."));
        if let Some(Err(e)) = one.depth.map(|d| d.write(folder, &depth)) {
            tracing::warn!(%e, "depth map save failed");
            msg = format!("{msg}; depth map failed: {e}");
        }
        let _ = tx.send(msg);
    });
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
    composite: Option<(crate::gui::item::ResultRef, crate::gui::item::ExportBg)>,
    depth: Option<crate::gui::item::DepthExport>,
}

/// Render and write each image's layers into `folder` on a background
/// thread (the composite where an image has none); report the counts.
fn spawn_save_layers(folder: PathBuf, jobs: Vec<LayerSave>, format: prunr_core::ExportFormat, tx: mpsc::Sender<String>) {
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
        let msg = if failed > 0 {
            format!("Saved {saved}, failed {failed}")
        } else {
            format!("Saved {saved} file(s)")
        };
        let _ = tx.send(msg);
    });
}

/// Encode + write N images into `folder` (named by `drag_export::export_name`) on a
/// background thread; report aggregate counts when done.
/// One image of a batch save.
struct BatchSave {
    filename: String,
    result: crate::gui::item::ResultRef,
    bg: crate::gui::item::ExportBg,
    depth: Option<crate::gui::item::DepthExport>,
}

fn spawn_save_batch(
    folder: PathBuf,
    items: Vec<BatchSave>,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<String>,
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
            match format.encode(&bg.bake(&rgba, &measure)) {
                Ok(bytes) => match std::fs::write(&out_path, &bytes) {
                    Ok(()) => saved += 1,
                    Err(_) => failed += 1,
                },
                Err(_) => failed += 1,
            }
        }
        let msg = if failed > 0 {
            format!("Saved {saved}, failed {failed}")
        } else {
            format!("Saved {saved} image(s)")
        };
        let _ = tx.send(msg);
    });
}
