//! Images into and out of the batch: open dialogs, pasted bytes, drag-and-drop, removal.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gui::drag_export_state::DragExportState;
use crate::gui::item::{BatchItem, ImageSource};
use crate::gui::system_bridge::DialogRequest;
use super::PrunrApp;

impl PrunrApp {
    /// Sync after batch modification — clamp index and refresh canvas.
    fn sync_after_batch_change(&mut self) {
        self.batch.clamp_selected_index();
        if !self.batch.items.is_empty() {
            self.pending_batch_sync = true;
        }
    }

    /// Core image loading: creates a BatchItem from a source + dimensions.
    fn load_image_source(&mut self, source: ImageSource, dims: (u32, u32), name: String) {
        let do_decode = matches!(&source, ImageSource::Bytes(_)); // decode eagerly for in-memory
        let id = self.push_new_item(source, dims, name);
        self.batch.selected_index = self.batch.items.len() - 1;
        if do_decode {
            // invariant: push occurred above, so batch.items is non-empty.
            if let Ok(bytes) = self.batch.items.last().unwrap().source.load_bytes() {
                self.batch.request_decode_bytes(id, bytes);
            }
        }

        self.status.text = "Ready".to_string();
        self.canvas_switch_id += 1;
        self.zoom_state.reset();
        self.show_original = false;
    }

    /// Load an image from raw bytes (clipboard paste, CLI pipe).
    fn load_image(&mut self, bytes: Vec<u8>, filename: Option<String>) {
        let Some(dims) = probe_dims_bytes(&bytes) else {
            self.set_temporary_status("Could not load image");
            return;
        };
        let name = filename.unwrap_or_else(|| "image".into());
        self.load_image_source(ImageSource::Bytes(Arc::new(bytes)), dims, name);
    }

    pub fn handle_open_path(&mut self, path: PathBuf) {
        let Some(dims) = probe_dims_path(&path) else {
            self.set_temporary_status("Could not load image");
            return;
        };
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image")
            .to_string();
        self.load_image_source(ImageSource::Path(path), dims, filename);
    }

    pub fn handle_open_bytes(&mut self, bytes: Vec<u8>, name: String) {
        let filename = if name.is_empty() { None } else { Some(name) };
        self.load_image(bytes, filename);
    }

    /// Open: ask for images; `on_open_picked` imports them.
    pub fn handle_open_dialog(&mut self) {
        self.ask_dialog(DialogRequest::OpenImages);
    }

    pub(super) fn on_open_picked(&mut self, paths: Vec<PathBuf>) {
        if let Some(first) = paths.first() {
            self.last_open_dir = first.parent().map(|p| p.to_path_buf());
        }
        if paths.len() == 1 && self.batch.items.is_empty() {
            // invariant: paths.len() == 1 checked in the guard above.
            self.handle_open_path(paths.into_iter().next().unwrap());
        } else {
            // Send file paths for lazy loading — bytes read on demand.
            let tx = self.batch.bg_io.file_load_tx.clone();
            std::thread::spawn(move || {
                for path in paths {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("untitled")
                        .to_string();
                    if tx.send((path, name)).is_err() {
                        break;
                    }
                }
            });
        }
    }

    pub fn remove_selected(&mut self) {
        let count = self.batch.selected_count();
        self.batch.items.retain(|item| !item.selected);
        self.sync_after_batch_change();
        if count > 0 {
            self.toasts.info(format!("Removed {count} image(s)"));
        }
    }

    /// Add an image to the batch from a file path (lazy — bytes not loaded yet).
    pub fn add_to_batch_path(&mut self, path: PathBuf, filename: String) {
        let Some(dims) = probe_dims_path(&path) else { return };
        self.add_to_batch_source(ImageSource::Path(path), dims, filename);
    }

    /// Add an image to the batch from raw bytes (clipboard/paste).
    pub fn add_to_batch(&mut self, bytes: Vec<u8>, filename: String) {
        let Some(dims) = probe_dims_bytes(&bytes) else { return };
        self.add_to_batch_source(ImageSource::Bytes(Arc::new(bytes)), dims, filename);
    }

    fn add_to_batch_source(&mut self, source: ImageSource, dims: (u32, u32), filename: String) {
        self.push_new_item(source, dims, filename);

        // NOTE: do NOT touch `zoom_state` here. The callers that actually
        // change selection to the newly-added item (DnD inline single-file,
        // force-select-after-drain) own the full `zoom_state.reset()`, so
        // `previous_zoom` and `pan_offset` are cleared along with the flag.
        // Setting only `pending_fit_zoom` here would leak stale toggle-state
        // (previous_zoom) from the prior image into the fit logic, which can
        // mis-fire `canvas.rs`'s "toggle-back to previous_zoom" branch.
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

        // Collect paths (need background I/O) and inline bytes (already in memory)
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut inline_items: Vec<(Vec<u8>, String)> = Vec::new();
        let mut saw_self_drop = false;

        for file in dropped {
            if let Some(path) = file.path {
                // Reject self-originated drops: our own drag-out writes temp files
                // under prunr-drag/. Dropping those back onto the canvas would
                // reopen them as new images.
                if crate::gui::drag_export::is_self_drop(&path) {
                    saw_self_drop = true;
                    continue;
                }
                self.last_open_dir = path.parent().map(|p| p.to_path_buf());
                paths.push(path);
            } else if let Some(bytes) = file.bytes {
                inline_items.push((bytes.to_vec(), file.name.clone()));
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

        // Expand directories to their immediate image children (one-level —
        // deeper recursion would surprise users). Cheap I/O; doing it on the
        // UI thread keeps the toast channel available without a cross-thread
        // bridge.
        let (resolved, dropped_empty_dir) = expand_dropped_paths(paths);
        if dropped_empty_dir && resolved.is_empty() {
            self.toasts.warning(
                "Dropped folder had no supported images (PNG / JPG / WebP / BMP / SVG)."
                    .to_string(),
            );
        }

        // Send file paths for lazy loading (avoids reading all into RAM upfront).
        if !resolved.is_empty() {
            let tx = self.batch.bg_io.file_load_tx.clone();
            std::thread::spawn(move || {
                for path in resolved {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("untitled")
                        .to_string();
                    let _ = tx.send((path, name));
                }
            });
        }

        // Handle inline bytes immediately (Wayland — already in memory, no I/O)
        if !inline_items.is_empty() {
            if inline_items.len() == 1 && self.batch.items.is_empty() {
                // invariant: inline_items.len() == 1 checked in the guard above.
                let (bytes, name) = inline_items.into_iter().next().unwrap();
                self.handle_open_bytes(bytes, name);
            } else {
                let id_floor = self.batch.next_id;
                let count = inline_items.len();
                for (bytes, name) in inline_items {
                    self.add_to_batch(bytes, name);
                }
                if count == 1 {
                    self.batch.selected_index = self.batch.items.len() - 1;
                    self.zoom_state.reset();
                    self.pending_batch_sync = true;
                }
                self.auto_process_imported(id_floor);
            }
        }
    }
}

/// Reads only the header, never the pixels.
fn probe_dims_bytes(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Reads only the header, so the file is never loaded into RAM.
fn probe_dims_path(path: &Path) -> Option<(u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    image::ImageReader::new(std::io::BufReader::new(file))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Resolve a list of dropped paths: directories expand to their immediate
/// supported-image children (no recursion). Returns `(resolved_paths,
/// any_dir_was_empty)` — the second flag drives a "no images in folder"
/// toast at the call site.
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

/// Lower-case extension match for the file types `prunr_core::load_image_*`
/// can decode (raster) plus SVG (rasterized via resvg).
fn is_supported_image_ext(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "svg")
    )
}
