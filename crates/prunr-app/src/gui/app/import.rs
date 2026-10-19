//! Images into and out of the batch: open dialogs, pasted bytes, drag-and-drop, removal.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gui::drag_export_state::DragExportState;
use crate::gui::item::{BatchItem, ImageSource};
use crate::gui::background_io::FileLoad;
use crate::gui::system_bridge::{DialogRequest, OPEN_EXTENSIONS};
use super::PrunrApp;

impl PrunrApp {
    fn sync_after_batch_change(&mut self) {
        self.batch.clamp_selected_index();
        if !self.batch.items.is_empty() {
            self.pending_batch_sync = true;
        }
    }

    fn load_image_source(&mut self, source: ImageSource, dims: (u32, u32), name: String, ctx: &egui::Context) {
        let do_decode = matches!(&source, ImageSource::Bytes(_)); // decode eagerly for in-memory
        let id = self.push_new_item(source, dims, name);
        self.batch.selected_index = self.batch.items.len() - 1;
        if do_decode {
            // invariant: push occurred above, so batch.items is non-empty.
            if let Ok(bytes) = self.batch.items.last().unwrap().source.load_bytes() {
                self.batch.request_decode_bytes(id, bytes);
            }
        }

        self.status.text = crate::gui::status_state::READY.to_string();
        self.on_selected_changed(ctx);
    }

    pub fn handle_open_path(&mut self, path: PathBuf, ctx: &egui::Context) {
        let Some(dims) = prunr_core::image_dimensions_from_path(&path) else {
            self.status_error(&format!("Could not open {}: not an image Prunr can read", display_name(&path, "the file")));
            return;
        };
        let filename = display_name(&path, "image");
        self.load_image_source(ImageSource::file(path), dims, filename, ctx);
    }

    pub fn handle_open_dialog(&mut self) {
        self.ask_dialog(DialogRequest::OpenImages);
    }

    pub(super) fn on_open_picked(&mut self, paths: Vec<PathBuf>) {
        if let Some(first) = paths.first() {
            self.last_open_dir = first.parent().map(|p| p.to_path_buf());
        }
        queue_file_loads(&self.batch.bg_io.file_load_tx, paths);
    }

    pub fn remove_selected(&mut self) {
        let count = self.batch.ticked_count();
        // A run on a removed image would go on for an image that is gone.
        let removed: Vec<u64> = self.batch.items.iter().filter(|i| i.ticked).map(|i| i.id).collect();
        for id in removed {
            self.cancel_runs_on(id);
        }
        let (removed, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.batch.items).into_iter().partition(|item| item.ticked);
        self.batch.items = kept;
        crate::gui::reclaim::later(removed);
        self.sync_after_batch_change();
        if count > 0 {
            self.toasts.info(format!("Removed {}", crate::gui::views::fmt::count(count, "image", "images")));
        }
    }

    /// Add an image whose header was read where it was imported (a file's
    /// bytes are read when used).
    pub(super) fn add_imported(&mut self, source: ImageSource, filename: String, len: usize, dims: (u32, u32)) {
        self.add_to_batch_source(source, dims, filename);
        if let Some(item) = self.batch.items.last_mut() {
            item.source_len = len;
        }
    }

    pub fn add_to_batch(&mut self, bytes: Vec<u8>, filename: String) {
        let Some(dims) = prunr_core::image_dimensions_from_bytes(&bytes) else { return };
        self.add_to_batch_source(ImageSource::Bytes(Arc::new(bytes)), dims, filename);
    }

    fn add_to_batch_source(&mut self, source: ImageSource, dims: (u32, u32), filename: String) {
        self.push_new_item(source, dims, filename);

        // Do NOT touch `zoom_state` here: the callers that select the new item
        // own the full reset; setting only `pending_fit_zoom` would leak the
        // previous image's `previous_zoom` into the fit logic.
        self.pending_batch_sync = true;
    }

    fn push_new_item(&mut self, source: ImageSource, dims: (u32, u32), name: String) -> u64 {
        let id = self.batch.next_id;
        self.batch.next_id += 1;
        self.batch.items.push(BatchItem::new(
            id,
            name,
            source,
            dims,
            self.settings.item_defaults_for_new_item(),
            self.settings.default_preset.clone(),
        ));
        id
    }

    pub fn remove_batch_item(&mut self, idx: usize) {
        if let Some(id) = self.batch.items.get(idx).map(|i| i.id) {
            self.cancel_runs_on(id);
        }
        if self.batch.remove(idx) {
            self.sync_after_batch_change();
        }
    }

    pub(super) fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() { return; }
        tracing::debug!(
            count = dropped.len(),
            existing_items = self.batch.items.len(),
            "drag-and-drop received",
        );

        let mut paths: Vec<PathBuf> = Vec::new();
        let mut inline_items: Vec<(Arc<[u8]>, String)> = Vec::new();
        let mut saw_self_drop = false;

        for file in dropped {
            if let Some(path) = file.path {
                // Our own drag-out's temp files would reopen as new images.
                if crate::gui::drag_export::is_self_drop(&path) {
                    saw_self_drop = true;
                    continue;
                }
                self.last_open_dir = path.parent().map(|p| p.to_path_buf());
                paths.push(path);
            } else if let Some(bytes) = file.bytes {
                inline_items.push((bytes, file.name.clone()));
            }
        }

        // Pure self-drop (our own drag landed back on the canvas): clear any
        // lingering drag state in case the drag crate's completion callback
        // didn't fire (observed on Windows).
        if saw_self_drop && paths.is_empty() && inline_items.is_empty() {
            DragExportState::reset(&self.drag_export.active, &self.drag_export.items);
            ctx.stop_dragging();
            return;
        }

        if !paths.is_empty() {
            queue_file_loads(&self.batch.bg_io.file_load_tx, paths);
        }

        // Bytes dropped in (Wayland) import like files, off the UI thread.
        if !inline_items.is_empty() {
            queue_bytes_loads(&self.batch.bg_io.file_load_tx, inline_items);
        }
    }
}

/// The batch name of an image loaded by path without being opened on its own.
pub(super) const UNTITLED: &str = "untitled";

/// The name `path`'s image goes by in the batch: its file name, else
/// `fallback`.
pub(super) fn display_name(path: &Path, fallback: &str) -> String {
    path.file_name().and_then(|n| n.to_str()).unwrap_or(fallback).to_string()
}

/// Resolve dropped paths: directories expand to their immediate
/// supported-image children (no recursion). The flag is set when a
/// dropped directory held no images.
fn expand_dropped_paths(input: Vec<PathBuf>) -> (Vec<PathBuf>, bool) {
    let mut out = Vec::with_capacity(input.len());
    let mut any_empty_dir = false;
    for path in input {
        if path.is_dir() {
            let before = out.len();
            for entry in std::fs::read_dir(&path).into_iter().flatten().flatten() {
                let p = entry.path();
                if p.is_file() && is_supported_image_ext(&p) {
                    out.push(p);
                }
            }
            if out.len() == before { any_empty_dir = true; }
        } else {
            out.push(path);
        }
    }
    (out, any_empty_dir)
}

/// Import `paths` on a worker: folders expand to their images (one level:
/// deeper would surprise), and each file's header is read for its size.
/// Bytes are read when used, not all into RAM up front.
pub(crate) fn queue_file_loads(tx: &std::sync::mpsc::Sender<FileLoad>, paths: Vec<PathBuf>) {
    let tx = tx.clone();
    std::thread::spawn(move || {
        let (resolved, dropped_empty_dir) = expand_dropped_paths(paths);
        if dropped_empty_dir && resolved.is_empty() {
            let _ = tx.send(FileLoad::NoImagesInFolder);
        }
        for path in resolved {
            let name = display_name(&path, UNTITLED);
            let stamp = crate::gui::item::FileStamp::at(&path);
            let len = stamp.map_or(0, |s| s.len as usize);
            let dims = prunr_core::image_dimensions_from_path(&path);
            // Once the receiver is gone every later send fails too.
            if tx.send(FileLoad::Image { source: ImageSource::Path(path, stamp), name, len, dims }).is_err() {
                break;
            }
        }
    });
}

/// Import images dropped in as bytes (Wayland) on a worker: each is
/// copied out of the drop and its header read there.
fn queue_bytes_loads(tx: &std::sync::mpsc::Sender<FileLoad>, dropped: Vec<(Arc<[u8]>, String)>) {
    let tx = tx.clone();
    std::thread::spawn(move || {
        for (bytes, name) in dropped {
            let bytes = Arc::new(bytes.to_vec());
            let dims = prunr_core::image_dimensions_from_bytes(&bytes);
            let name = if name.is_empty() { "image".into() } else { name };
            let load = FileLoad::Image { len: bytes.len(), source: ImageSource::Bytes(bytes), name, dims };
            if tx.send(load).is_err() {
                break;
            }
        }
    });
}

fn is_supported_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| OPEN_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    /// Bytes dropped in are copied and read off the UI thread, and land
    /// as an import with their size; bytes that are no image say so.
    #[test]
    fn dropped_bytes_import_off_the_ui_thread() {
        let mut png = Vec::new();
        image::RgbaImage::new(3, 2).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        super::queue_bytes_loads(&tx, vec![(png.clone().into(), String::new()), (vec![1u8, 2, 3].into(), "junk".into())]);
        let mut landed: Vec<_> = (0..2).map(|_| rx.recv_timeout(std::time::Duration::from_secs(5)).expect("lands")).collect();
        let super::FileLoad::Image { name, len, dims, .. } = landed.remove(0) else { panic!("an image") };
        assert_eq!((name.as_str(), len, dims), ("image", png.len(), Some((3, 2))));
        assert!(matches!(landed.remove(0), super::FileLoad::Image { dims: None, .. }), "junk has no header");
    }
}
