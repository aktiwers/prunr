use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use egui::{Key, ViewportCommand};
use image::GenericImageView;

use bgprunr_core::ProgressStage;

use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, settings, shortcuts, sidebar, statusbar, toolbar};

pub(crate) struct BatchItem {
    pub id: u64,
    pub filename: String,
    pub source_bytes: Vec<u8>,
    pub dimensions: (u32, u32),
    pub source_texture: Option<egui::TextureHandle>,
    pub thumb_texture: Option<egui::TextureHandle>,
    pub thumb_pending: bool,
    pub result_rgba: Option<image::RgbaImage>,
    pub result_texture: Option<egui::TextureHandle>,
    pub status: BatchStatus,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BatchStatus {
    Pending,
    Processing,
    Done,
    Error(String),
}

pub struct BgPrunrApp {
    // State
    pub(crate) state: AppState,
    pub(crate) loaded_filename: Option<String>,
    pub(crate) source_bytes: Option<Vec<u8>>,
    pub(crate) image_dimensions: Option<(u32, u32)>,

    // Worker thread communication
    worker_tx: mpsc::Sender<WorkerMessage>,
    worker_rx: mpsc::Receiver<WorkerResult>,
    pub(crate) cancel_flag: Arc<AtomicBool>,

    // Progress tracking
    pub(crate) progress_stage: String,
    pub(crate) progress_pct: f32,
    pub(crate) status_text: String,
    pub(crate) status_is_temporary: bool,
    status_set_at: Option<std::time::Instant>,

    // Textures
    pub(crate) source_texture: Option<egui::TextureHandle>,
    pub(crate) result_texture: Option<egui::TextureHandle>,

    // Result image for save/copy
    pub(crate) result_rgba: Option<image::RgbaImage>,

    // Clipboard (MUST live for app lifetime -- Wayland ownership requirement)
    clipboard: Option<arboard::Clipboard>,

    // UI state
    pub(crate) show_shortcuts: bool,

    // Set by raw_input_hook — egui converts Ctrl+C to Event::Copy before we see it
    pending_copy: bool,

    // Zoom/Pan state
    pub(crate) zoom: f32,
    pub(crate) pan_offset: egui::Vec2,
    pub(crate) previous_zoom: f32,
    pub(crate) is_panning: bool,

    // Before/After toggle
    pub(crate) show_original: bool,

    // Animation state
    pub(crate) anim_progress: f32,
    pub(crate) anim_mask: Option<Vec<u8>>,
    /// Decoded source RGBA cached for animation (avoids per-frame decode)
    pub(crate) source_rgba_cache: Option<image::RgbaImage>,

    // Window title change detection
    prev_title: String,

    // Batch items
    pub(crate) batch_items: Vec<BatchItem>,
    pub(crate) selected_batch_index: usize,
    pub(crate) show_sidebar: bool,
    pub(crate) next_batch_id: u64,

    // Settings
    pub(crate) show_settings: bool,
    pub(crate) settings: Settings,

    // Pending zoom actions (flags consumed by canvas.rs)
    pub(crate) pending_fit_zoom: bool,
    pub(crate) pending_actual_size: bool,
    /// Set by add_to_batch — triggers sync_selected_batch_textures in next logic()
    pending_batch_sync: bool,
    /// Set by toolbar Open button — processed in logic() where ctx is available
    pub(crate) pending_open_dialog: bool,
    /// Channel for receiving files loaded by background thread
    file_load_rx: mpsc::Receiver<(Vec<u8>, String)>,
    file_load_tx: mpsc::Sender<(Vec<u8>, String)>,
    /// Channel for receiving decoded thumbnails from background thread
    /// (batch_item_id, width, height, rgba_pixels)
    pub(crate) thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
}

/// Convert an RgbaImage to an egui TextureHandle.
pub(crate) fn rgba_to_texture(
    rgba: &image::RgbaImage,
    name: &str,
    ctx: &egui::Context,
) -> egui::TextureHandle {
    let (w, h) = (rgba.width(), rgba.height());
    let ci = egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        rgba.as_flat_samples().as_slice(),
    );
    ctx.load_texture(name, ci, egui::TextureOptions::default())
}


impl BgPrunrApp {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let (worker_tx, worker_rx) = spawn_worker(cc.egui_ctx.clone());

        // Set dark visuals
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        // Customize visuals — suppress all bright/red borders
        let mut visuals = cc.egui_ctx.global_style().visuals.clone();
        visuals.window_fill = theme::BG_PRIMARY;
        visuals.panel_fill = theme::BG_SECONDARY;
        let subtle = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x3a, 0x3a));
        visuals.widgets.noninteractive.bg_stroke = subtle;
        visuals.widgets.inactive.bg_stroke = subtle;
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, theme::ACCENT);
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x50, 0x50, 0x50));
        visuals.widgets.open.bg_stroke = subtle; // ComboBox "open" state
        visuals.window_stroke = subtle;
        visuals.error_fg_color = theme::DESTRUCTIVE; // keep red for actual errors only
        cc.egui_ctx.set_visuals(visuals);

        // Override font sizes and suppress debug red-border warnings
        let mut style = (*cc.egui_ctx.global_style()).clone();
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::proportional(theme::FONT_SIZE_BODY),
        );
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::proportional(theme::FONT_SIZE_HEADING),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::monospace(theme::FONT_SIZE_MONO),
        );
        // Disable debug red-border warnings that fire on layout shifts
        // (e.g. sidebar appearing after file-open changes widget rects between frames).
        #[cfg(debug_assertions)]
        {
            style.debug.warn_if_rect_changes_id = false;
        }
        cc.egui_ctx.set_global_style(style);

        let clipboard = arboard::Clipboard::new().ok();
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();

        Self {
            state: AppState::Empty,
            loaded_filename: None,
            source_bytes: None,
            image_dimensions: None,
            worker_tx,
            worker_rx,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            progress_stage: String::new(),
            progress_pct: 0.0,
            status_text: "Ready".to_string(),
            status_is_temporary: false,
            status_set_at: None,
            source_texture: None,
            result_texture: None,
            result_rgba: None,
            clipboard,
            show_shortcuts: false,
            pending_copy: false,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            previous_zoom: 1.0,
            is_panning: false,
            show_original: false,
            anim_progress: 0.0,
            anim_mask: None,
            source_rgba_cache: None,
            prev_title: String::new(),
            batch_items: Vec::new(),
            selected_batch_index: 0,
            show_sidebar: false,
            next_batch_id: 0,
            show_settings: false,
            settings: Settings::default(),
            pending_fit_zoom: false,
            pending_actual_size: false,
            pending_batch_sync: false,
            pending_open_dialog: false,
            file_load_tx,
            file_load_rx,
            thumb_tx,
            thumb_rx,
        }
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();
        Self {
            state: AppState::Empty,
            loaded_filename: None,
            source_bytes: None,
            image_dimensions: None,
            worker_tx,
            worker_rx,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            progress_stage: String::new(),
            progress_pct: 0.0,
            status_text: "Ready".to_string(),
            status_is_temporary: false,
            status_set_at: None,
            source_texture: None,
            result_texture: None,
            result_rgba: None,
            clipboard: None,
            show_shortcuts: false,
            pending_copy: false,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            previous_zoom: 1.0,
            is_panning: false,
            show_original: false,
            anim_progress: 0.0,
            anim_mask: None,
            source_rgba_cache: None,
            prev_title: String::new(),
            batch_items: Vec::new(),
            selected_batch_index: 0,
            show_sidebar: false,
            next_batch_id: 0,
            show_settings: false,
            settings: Settings::default(),
            pending_fit_zoom: false,
            pending_actual_size: false,
            pending_batch_sync: false,
            pending_open_dialog: false,
            file_load_tx,
            file_load_rx,
            thumb_tx,
            thumb_rx,
        }
    }

    fn set_temporary_status(&mut self, text: impl Into<String>) {
        self.status_text = text.into();
        self.status_is_temporary = true;
        self.status_set_at = Some(std::time::Instant::now());
    }

    /// Reset app state after all batch items are removed.
    fn clear_to_empty(&mut self) {
        self.source_bytes = None;
        self.source_texture = None;
        self.result_texture = None;
        self.result_rgba = None;
        self.loaded_filename = None;
        self.image_dimensions = None;
        self.state = AppState::Empty;
        self.selected_batch_index = 0;
    }

    /// Sync after batch modification — clamp index and refresh canvas.
    fn sync_after_batch_change(&mut self) {
        if self.batch_items.is_empty() {
            self.clear_to_empty();
        } else {
            self.selected_batch_index = self.selected_batch_index.min(self.batch_items.len() - 1);
            self.pending_batch_sync = true;
        }
    }

    fn load_image(&mut self, bytes: Vec<u8>, filename: Option<String>) {
        let dims = match image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
        {
            Some(d) => d,
            None => {
                self.set_temporary_status("Could not load image");
                return;
            }
        };

        let name = filename.unwrap_or_else(|| "image".into());

        // Add to batch so it appears in the sidebar
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        self.batch_items.push(BatchItem {
            id,
            filename: name.clone(),
            source_bytes: bytes.clone(),
            dimensions: dims,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            status: BatchStatus::Pending,
            selected: false,
        });
        self.selected_batch_index = self.batch_items.len() - 1;

        // Set app-level state for canvas
        self.image_dimensions = Some(dims);
        self.source_bytes = Some(bytes);
        self.loaded_filename = Some(name);
        self.source_texture = None;
        self.result_texture = None;
        self.result_rgba = None;
        self.source_rgba_cache = None;
        self.state = AppState::Loaded;
        self.status_text = "Ready".to_string();
        self.pan_offset = egui::Vec2::ZERO;
        self.previous_zoom = 1.0;
        self.pending_fit_zoom = true;
        self.show_original = false;
    }

    pub fn handle_open_path(&mut self, path: PathBuf) {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string());
                self.load_image(bytes, filename);
            }
            Err(e) => {
                self.set_temporary_status(format!("Could not read file: {e}"));
            }
        }
    }

    pub fn handle_open_bytes(&mut self, bytes: Vec<u8>, name: String) {
        let filename = if name.is_empty() { None } else { Some(name) };
        self.load_image(bytes, filename);
    }

    pub fn handle_open_dialog(&mut self) {
        let paths = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .set_title("Open Image(s)")
            .pick_files();
        if let Some(paths) = paths {
            if paths.len() == 1 && self.batch_items.is_empty() {
                self.handle_open_path(paths.into_iter().next().unwrap());
            } else {
                // Read files on a background thread to avoid blocking the UI
                let tx = self.file_load_tx.clone();
                std::thread::spawn(move || {
                    for path in paths {
                        if let Ok(bytes) = std::fs::read(&path) {
                            let name = path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("untitled")
                                .to_string();
                            if tx.send((bytes, name)).is_err() {
                                break;
                            }
                        }
                    }
                });
            }
        }
    }

    pub fn handle_remove_bg(&mut self) {
        let has_selected = self.batch_items.iter().any(|i| i.selected);
        if has_selected {
            // Process all checked items via batch
            self.process_items(|item| item.selected);
        } else {
            // No checkboxes — process current image only
            if let Some(ref bytes) = self.source_bytes {
                self.cancel_flag.store(false, Ordering::Relaxed);
                self.state = AppState::Processing;
                self.progress_pct = 0.0;
                self.progress_stage = "Starting".to_string();
                if !self.batch_items.is_empty() {
                    let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                    self.batch_items[idx].status = BatchStatus::Processing;
                }
                let _ = self.worker_tx.send(WorkerMessage::ProcessImage {
                    img_bytes: bytes.clone(),
                    model: self.settings.model.into(),
                    cancel: self.cancel_flag.clone(),
                });
            }
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        let items: Vec<(u64, Vec<u8>)> = self.batch_items.iter()
            .filter(|i| filter(i) && matches!(i.status, BatchStatus::Pending | BatchStatus::Error(_)))
            .map(|i| (i.id, i.source_bytes.clone()))
            .collect();
        if items.is_empty() { return; }
        for item in &mut self.batch_items {
            if filter(item) && matches!(item.status, BatchStatus::Pending | BatchStatus::Error(_)) {
                item.status = BatchStatus::Processing;
            }
        }
        self.cancel_flag.store(false, Ordering::Relaxed);
        self.state = AppState::Processing;
        let _ = self.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            model: self.settings.model.into(),
            jobs: self.settings.parallel_jobs,
            cancel: self.cancel_flag.clone(),
        });
    }

    /// Save selected images (or current image if none selected).
    /// Single selection → save-as dialog; multiple → folder picker.
    pub fn handle_save_selected(&mut self) {
        let selected: Vec<_> = self.batch_items.iter()
            .filter(|i| i.selected && i.status == BatchStatus::Done && i.result_rgba.is_some())
            .collect();

        if selected.is_empty() {
            // No checkboxes selected — save current image via save-as dialog
            if let Some(ref rgba) = self.result_rgba {
                let default_name = self
                    .loaded_filename
                    .as_deref()
                    .and_then(|name| Path::new(name).file_stem()?.to_str())
                    .map(|stem| format!("{stem}-nobg.png"))
                    .unwrap_or_else(|| "result-nobg.png".to_string());
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PNG Image", &["png"])
                    .set_file_name(&default_name)
                    .set_title("Save PNG")
                    .save_file()
                {
                    match bgprunr_core::encode_rgba_png(rgba) {
                        Ok(png_bytes) => match std::fs::write(&path, &png_bytes) {
                            Ok(()) => self.set_temporary_status("Saved"),
                            Err(_) => self.set_temporary_status("Could not save file"),
                        },
                        Err(_) => self.set_temporary_status("Could not save file"),
                    }
                }
            }
            return;
        }

        // Multiple selected — folder picker
        if let Some(folder) = rfd::FileDialog::new()
            .set_title("Save Selected — Choose Folder")
            .pick_folder()
        {
            let mut saved = 0usize;
            let mut failed = 0usize;
            for item in &selected {
                if let Some(ref rgba) = item.result_rgba {
                    let stem = Path::new(&item.filename)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("image");
                    let out_path = folder.join(format!("{stem}-nobg.png"));
                    match bgprunr_core::encode_rgba_png(rgba) {
                        Ok(png_bytes) => match std::fs::write(&out_path, &png_bytes) {
                            Ok(()) => saved += 1,
                            Err(_) => failed += 1,
                        },
                        Err(_) => failed += 1,
                    }
                }
            }
            if failed > 0 {
                self.set_temporary_status(format!("Saved {saved}, failed {failed}"));
            } else {
                self.set_temporary_status(format!("Saved {saved} image(s)"));
            }
        }
    }

    pub fn handle_save_all(&mut self) {
        // Select all done items, then delegate to save_selected
        for item in &mut self.batch_items {
            if item.status == BatchStatus::Done && item.result_rgba.is_some() {
                item.selected = true;
            }
        }
        self.handle_save_selected();
        // Deselect after save
        for item in &mut self.batch_items {
            item.selected = false;
        }
    }

    pub fn remove_selected(&mut self) {
        self.batch_items.retain(|item| !item.selected);
        self.sync_after_batch_change();
    }

    pub fn handle_copy(&mut self) {
        if let Some(ref mut clipboard) = self.clipboard {
            if let Some(ref rgba) = self.result_rgba {
                let width = rgba.width() as usize;
                let height = rgba.height() as usize;
                let bytes: Vec<u8> = rgba.as_flat_samples().as_slice().to_vec();
                let image_data = arboard::ImageData {
                    width,
                    height,
                    bytes: Cow::Owned(bytes),
                };
                match clipboard.set_image(image_data) {
                    Ok(()) => self.set_temporary_status("Copied to clipboard"),
                    Err(_) => self.set_temporary_status(
                        "Could not copy to clipboard. Try saving instead.",
                    ),
                }
            }
        } else {
            self.set_temporary_status("Could not copy to clipboard. Try saving instead.");
        }
    }

    pub fn handle_paste(&mut self) {
        if let Some(ref mut clipboard) = self.clipboard {
            match clipboard.get_image() {
                Ok(img_data) => {
                    let rgba = image::RgbaImage::from_raw(
                        img_data.width as u32,
                        img_data.height as u32,
                        img_data.bytes.into_owned(),
                    );
                    if let Some(rgba) = rgba {
                        let mut png_bytes = Vec::new();
                        if image::DynamicImage::ImageRgba8(rgba)
                            .write_to(
                                &mut std::io::Cursor::new(&mut png_bytes),
                                image::ImageFormat::Png,
                            )
                            .is_ok()
                        {
                            self.load_image(png_bytes, Some("pasted-image.png".to_string()));
                            return;
                        }
                    }
                    self.set_temporary_status("Could not decode clipboard image");
                }
                Err(_) => {
                    self.set_temporary_status("No image in clipboard");
                }
            }
        } else {
            self.set_temporary_status("Clipboard not available");
        }
    }

    pub fn handle_cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    pub fn add_to_batch(&mut self, bytes: Vec<u8>, filename: String) {
        // Use image reader to get dimensions without full decode
        let dims = match image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
        {
            Some(d) => d,
            None => return, // not a valid image
        };

        // Migrate existing single image to batch as item 0
        if self.batch_items.is_empty() {
            if let Some(existing_bytes) = self.source_bytes.take() {
                let existing_name = self.loaded_filename.clone().unwrap_or_else(|| "image".into());
                let existing_dims = self.image_dimensions.unwrap_or((0, 0));
                let eid = self.next_batch_id;
                self.next_batch_id += 1;
                let existing_item = BatchItem {
                    id: eid,
                    filename: existing_name,
                    source_bytes: existing_bytes,
                    dimensions: existing_dims,
                    source_texture: self.source_texture.take(),
                    thumb_texture: None,
                    thumb_pending: false,
                    result_rgba: self.result_rgba.take(),
                    result_texture: self.result_texture.take(),
                    status: if self.state == AppState::Done { BatchStatus::Done } else { BatchStatus::Pending },
                    selected: false,
                };
                self.batch_items.insert(0, existing_item);
                self.selected_batch_index = 0;
            }
        }

        let id = self.next_batch_id;
        self.next_batch_id += 1;

        self.batch_items.push(BatchItem {
            id,
            filename,
            source_bytes: bytes,
            dimensions: dims,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            status: BatchStatus::Pending,
            selected: false,
        });

        if self.state == AppState::Empty {
            self.state = AppState::Loaded;
        }
        self.pending_batch_sync = true;
    }

    /// Request thumbnail generation on a background thread for a batch item.
    /// If result_rgba is Some, thumbnails from result; otherwise decodes source bytes.
    pub(crate) fn request_thumbnail(&self, item_id: u64, source_bytes: &[u8], result_rgba: Option<&image::RgbaImage>) {
        let tx = self.thumb_tx.clone();
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone();
            std::thread::spawn(move || {
                let thumb = image::imageops::thumbnail(&rgba, 160, 160);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let bytes = source_bytes.to_vec();
            std::thread::spawn(move || {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    let thumb = image::imageops::thumbnail(&img.to_rgba8(), 160, 160);
                    let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
                }
            });
        }
    }

    pub fn remove_batch_item(&mut self, idx: usize) {
        if idx >= self.batch_items.len() { return; }
        self.batch_items.remove(idx);
        self.sync_after_batch_change();
    }

    pub fn handle_process_all(&mut self) {
        self.process_items(|_| true);
    }

    /// Set up reveal animation state from a completed result image.
    fn setup_reveal_animation(&mut self, rgba: &image::RgbaImage, ctx: &egui::Context) {
        let alpha_mask: Vec<u8> = rgba.pixels().map(|p| p[3]).collect();
        self.anim_mask = Some(alpha_mask);
        self.result_rgba = Some(rgba.clone());
        self.result_texture = Some(rgba_to_texture(rgba, "result", ctx));
        if self.source_rgba_cache.is_none() {
            if let Some(ref bytes) = self.source_bytes {
                self.source_rgba_cache = image::load_from_memory(bytes)
                    .ok()
                    .map(|img| img.to_rgba8());
            }
        }
        self.state = AppState::Animating;
        self.anim_progress = 0.0;
        self.show_original = false;
    }

    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        if self.batch_items.is_empty() { return; }
        let idx = self.selected_batch_index.min(self.batch_items.len() - 1);

        // Load source texture lazily
        if self.batch_items[idx].source_texture.is_none() {
            if let Ok(img) = image::load_from_memory(&self.batch_items[idx].source_bytes) {
                let rgba = img.to_rgba8();
                let item_id = self.batch_items[idx].id;
                self.batch_items[idx].source_texture = Some(
                    rgba_to_texture(&rgba, &format!("source_{item_id}"), ctx),
                );
            }
        }

        // Load result texture lazily
        if self.batch_items[idx].result_texture.is_none() {
            if let Some(ref rgba) = self.batch_items[idx].result_rgba {
                let item_id = self.batch_items[idx].id;
                self.batch_items[idx].result_texture = Some(
                    rgba_to_texture(rgba, &format!("result_{item_id}"), ctx),
                );
            }
        }

        // Sync app-level state for canvas rendering
        // Use references/clones only for textures (cheap Arc clones), avoid cloning raw bytes
        let item = &self.batch_items[idx];
        self.source_texture = item.source_texture.clone();
        self.loaded_filename = Some(item.filename.clone());
        self.image_dimensions = Some(item.dimensions);
        self.source_rgba_cache = None;
        self.show_original = false;
        // Cancel any in-progress animation (data belongs to previous image)
        self.anim_mask = None;
        self.anim_progress = 0.0;

        // Reset zoom/pan — canvas will apply fit-to-window on same frame
        self.pan_offset = egui::Vec2::ZERO;
        self.previous_zoom = 1.0;
        self.zoom = 1.0;
        self.pending_fit_zoom = true;

        // Set state based on this item's status, not global processing state
        let item_status = item.status.clone();
        let result_texture = item.result_texture.clone();
        let result_rgba = item.result_rgba.clone();
        match item_status {
            BatchStatus::Done => {
                self.result_texture = result_texture;
                self.result_rgba = result_rgba;
                self.state = AppState::Done;
            }
            BatchStatus::Processing => {
                self.result_texture = None;
                self.result_rgba = None;
                self.state = AppState::Processing;
            }
            _ => {
                self.result_texture = None;
                self.result_rgba = None;
                self.state = AppState::Loaded;
            }
        }
        self.source_bytes = Some(self.batch_items[idx].source_bytes.clone());
    }
}

impl eframe::App for BgPrunrApp {
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        // egui_winit converts Ctrl+C to Event::Copy. Intercept it so we can
        // use it for image clipboard copy (egui's Copy is for text widgets).
        raw_input.events.retain(|event| {
            if matches!(event, egui::Event::Copy) {
                self.pending_copy = true;
                false
            } else {
                true
            }
        });
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // a. Poll worker channel
        while let Ok(msg) = self.worker_rx.try_recv() {
            match msg {
                WorkerResult::Progress(stage, pct) => {
                    self.progress_stage = match stage {
                        ProgressStage::Decode | ProgressStage::Resize | ProgressStage::Normalize => {
                            "Preprocessing".to_string()
                        }
                        ProgressStage::Infer => "Inferring".to_string(),
                        ProgressStage::Postprocess | ProgressStage::Alpha => {
                            "Applying alpha".to_string()
                        }
                    };
                    self.progress_pct = pct;
                }
                WorkerResult::Done(result) => {
                    if let Ok(img) = image::load_from_memory(&result.rgba_bytes) {
                        let rgba_img = img.to_rgba8();
                        if self.settings.reveal_animation_enabled {
                            self.setup_reveal_animation(&rgba_img, ctx);
                        } else {
                            self.result_texture = Some(rgba_to_texture(&rgba_img, "result", ctx));
                            self.result_rgba = Some(rgba_img.clone());
                            self.state = AppState::Done;
                            self.status_text = "Done".to_string();
                        }
                        // Sync result back to batch item so it persists when switching
                        if !self.batch_items.is_empty() {
                            let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                            self.batch_items[idx].result_rgba = Some(rgba_img);
                            self.batch_items[idx].result_texture = self.result_texture.clone();
                            self.batch_items[idx].status = BatchStatus::Done;
                            self.batch_items[idx].thumb_texture = None;
                            self.batch_items[idx].thumb_pending = false;
                        }
                    }
                    self.settings.active_backend = result.active_provider;
                }
                WorkerResult::Cancelled => {
                    if self.state == AppState::Processing {
                        self.state = AppState::Loaded;
                        self.status_text = "Cancelled".to_string();
                    }
                }
                WorkerResult::Error(msg) => {
                    self.state = AppState::Loaded;
                    self.status_text =
                        "Processing failed. Try a different image or restart the app.".to_string();
                    eprintln!("Worker error: {msg}");
                }
                WorkerResult::BatchItemDone { item_id, result } => {
                    let is_selected = self.batch_items.get(self.selected_batch_index)
                        .map_or(false, |b| b.id == item_id);
                    // Update batch item first (release borrow before animation setup)
                    let mut animate_rgba: Option<image::RgbaImage> = None;
                    if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                        match result {
                            Ok(pr) => {
                                if let Ok(img) = image::load_from_memory(&pr.rgba_bytes) {
                                    let rgba = img.to_rgba8();
                                    if is_selected && self.settings.reveal_animation_enabled {
                                        animate_rgba = Some(rgba.clone());
                                    }
                                    item.result_rgba = Some(rgba);
                                    item.status = BatchStatus::Done;
                                    item.thumb_texture = None;
                                    item.thumb_pending = false;
                                }
                                self.settings.active_backend = pr.active_provider;
                            }
                            Err(e) => {
                                item.status = BatchStatus::Error(e);
                            }
                        }
                    }
                    if let Some(rgba) = animate_rgba {
                        self.setup_reveal_animation(&rgba, ctx);
                    } else if is_selected {
                        self.sync_selected_batch_textures(ctx);
                    }
                }
                WorkerResult::BatchComplete => {
                    let done = self.batch_items.iter().filter(|i| i.status == BatchStatus::Done).count();
                    let failed = self.batch_items.iter().filter(|i| matches!(i.status, BatchStatus::Error(_))).count();
                    if failed > 0 {
                        self.status_text = format!("{failed} image(s) failed to process. Check the status icons in the sidebar.");
                    } else {
                        self.status_text = format!("All done \u{2014} {done} images processed");
                    }
                    self.state = AppState::Done;
                }
            }
        }

        // b. Handle drag-and-drop (works on X11; no-op on native Wayland — winit#1881)
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let mut new_items: Vec<(Vec<u8>, String)> = Vec::new();
            for file in dropped {
                if let Some(path) = file.path {
                    if let Ok(bytes) = std::fs::read(&path) {
                        let name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("untitled")
                            .to_string();
                        new_items.push((bytes, name));
                    }
                } else if let Some(bytes) = file.bytes {
                    new_items.push((bytes.to_vec(), file.name.clone()));
                }
            }

            if new_items.len() == 1 && self.batch_items.is_empty() {
                // Single image, no existing batch — use single-image flow
                let (bytes, name) = new_items.into_iter().next().unwrap();
                self.handle_open_bytes(bytes, name);
            } else {
                // Multiple images OR existing batch — add to batch queue
                for (bytes, name) in new_items {
                    self.add_to_batch(bytes, name);
                }
                // Auto-remove on import
                if self.settings.auto_remove_on_import && self.batch_items.iter().any(|i| i.status == BatchStatus::Pending) {
                    self.handle_process_all();
                }
            }
        }

        // b2. Advance animation
        if self.state == AppState::Animating {
            let dt = ctx.input(|i| i.stable_dt);
            self.anim_progress = (self.anim_progress + dt / theme::ANIM_DURATION_SECS).min(1.0);

            // Check for skip (any key or mouse click)
            let skip = ctx.input(|i| {
                let any_key = i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }));
                i.pointer.any_pressed() || any_key
            });

            if skip || self.anim_progress >= 1.0 {
                self.state = AppState::Done;
                self.anim_progress = 0.0;
                self.status_text = "Done".to_string();
            } else {
                ctx.request_repaint(); // keep animation loop running
            }
        }

        // c. Keyboard shortcuts
        // Ctrl+C is intercepted via raw_input_hook (egui converts it to Event::Copy).
        let (mut open_requested, mut remove_requested, mut save_requested) =
            (false, false, false);
        let (mut cancel_requested, mut toggle_shortcuts) = (false, false);
        let (mut toggle_before_after, mut fit_to_window, mut actual_size) = (false, false, false);
        let mut toggle_settings = false;
        let (mut nav_prev, mut nav_next, mut toggle_sidebar) = (false, false, false);

        ctx.input(|i| {
            if i.modifiers.command && i.key_pressed(Key::O) {
                open_requested = true;
            }
            if i.modifiers.command && i.key_pressed(Key::R) {
                remove_requested = true;
            }
            if i.modifiers.command && i.key_pressed(Key::S) {
                save_requested = true;
            }
            if i.key_pressed(Key::Escape) {
                cancel_requested = true;
            }
            if i.key_pressed(Key::F1) {
                toggle_shortcuts = true;
            }
            if i.key_pressed(Key::B) {
                toggle_before_after = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Num0) {
                fit_to_window = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Num1) {
                actual_size = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Comma) {
                toggle_settings = true;
            }
            if i.key_pressed(Key::OpenBracket) {
                nav_prev = true;
            }
            if i.key_pressed(Key::CloseBracket) {
                nav_next = true;
            }
            if i.key_pressed(Key::Tab) {
                toggle_sidebar = true;
            }
        });

        let copy_requested = std::mem::take(&mut self.pending_copy);

        if open_requested || std::mem::take(&mut self.pending_open_dialog) {
            self.handle_open_dialog();
        }
        if remove_requested && matches!(self.state, AppState::Loaded | AppState::Done) {
            self.handle_remove_bg();
        }
        if save_requested && self.state == AppState::Done {
            self.handle_save_selected();
        }
        if copy_requested && self.state == AppState::Done {
            self.handle_copy();
        }
        if toggle_before_after && self.state == AppState::Done {
            self.show_original = !self.show_original;
        }
        if fit_to_window {
            self.pending_fit_zoom = true;
        }
        if actual_size {
            self.pending_actual_size = true;
        }
        // Cancel batch processing
        let batch_processing = self.batch_items.iter().any(|i| i.status == BatchStatus::Processing);
        if cancel_requested && batch_processing {
            self.handle_cancel();
            for item in &mut self.batch_items {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
        } else if cancel_requested && self.state == AppState::Animating {
            self.state = AppState::Done;
            self.anim_progress = 0.0;
        } else if cancel_requested && self.state == AppState::Processing {
            self.handle_cancel();
            self.state = AppState::Loaded;
            self.status_text = "Cancelled".to_string();
        } else if cancel_requested && self.show_settings {
            self.show_settings = false;
        } else if cancel_requested && self.show_shortcuts {
            self.show_shortcuts = false;
        }
        if toggle_shortcuts {
            self.show_shortcuts = !self.show_shortcuts;
        }
        if toggle_settings {
            self.show_settings = !self.show_settings;
        }
        if nav_prev && !self.batch_items.is_empty() {
            if self.selected_batch_index == 0 {
                self.selected_batch_index = self.batch_items.len() - 1;
            } else {
                self.selected_batch_index -= 1;
            }
            self.sync_selected_batch_textures(ctx);
            self.show_original = false;
        }
        if nav_next && !self.batch_items.is_empty() {
            self.selected_batch_index = (self.selected_batch_index + 1) % self.batch_items.len();
            self.sync_selected_batch_textures(ctx);
            self.show_original = false;
        }
        if toggle_sidebar {
            self.show_sidebar = !self.show_sidebar;
        }
        // Deferred batch sync from sidebar click
        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }

        // Drain files loaded by background thread (max 5 per frame to stay responsive)
        let mut loaded_any = false;
        for _ in 0..5 {
            match self.file_load_rx.try_recv() {
                Ok((bytes, name)) => {
                    self.add_to_batch(bytes, name);
                    loaded_any = true;
                }
                Err(_) => break,
            }
        }
        if loaded_any {
            ctx.request_repaint(); // more may be pending
            if self.settings.auto_remove_on_import
                && self.batch_items.iter().any(|i| i.status == BatchStatus::Pending)
            {
                self.handle_process_all();
            }
        }

        // d. Update window title (only when changed)
        let title = if self.batch_items.len() >= 2 {
            format!("BgPrunR \u{2014} {} images", self.batch_items.len())
        } else {
            match &self.loaded_filename {
                Some(name) => format!("BgPrunR \u{2014} {name}"),
                None => "BgPrunR".to_string(),
            }
        };
        if title != self.prev_title {
            self.prev_title = title.clone();
            ctx.send_viewport_cmd(ViewportCommand::Title(title));
        }

        // e. Clear temporary status text after ~2 seconds
        if self.status_is_temporary {
            if let Some(set_at) = self.status_set_at {
                if set_at.elapsed() > std::time::Duration::from_secs(2) {
                    self.status_text = "Ready".into();
                    self.status_is_temporary = false;
                    self.status_set_at = None;
                }
            }
        }

        // Load source texture if we have bytes but no texture yet
        if self.source_texture.is_none() {
            if let Some(ref bytes) = self.source_bytes {
                if let Ok(img) = image::load_from_memory(bytes) {
                    let rgba = img.to_rgba8();
                    self.source_texture = Some(rgba_to_texture(&rgba, "source", ctx));
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let panel_frame = egui::Frame {
            fill: theme::BG_SECONDARY,
            stroke: egui::Stroke::new(1.0, egui::Color32::from_rgb(0x2a, 0x2a, 0x2a)),
            inner_margin: egui::Margin::symmetric(theme::SPACE_SM as i8, 0),
            ..Default::default()
        };
        egui::Panel::top("toolbar")
            .exact_size(theme::TOOLBAR_HEIGHT)
            .frame(panel_frame)
            .show_inside(ui, |ui| toolbar::render(ui, self));

        egui::Panel::bottom("statusbar")
            .exact_size(theme::STATUS_BAR_HEIGHT)
            .frame(panel_frame)
            .show_inside(ui, |ui| statusbar::render(ui, self));

        let sidebar_visible = self.show_sidebar || !self.batch_items.is_empty();
        if sidebar_visible {
            egui::Panel::right("sidebar")
                .exact_size(theme::SIDEBAR_WIDTH)
                .resizable(false)
                .show_inside(ui, |ui| sidebar::render(ui, self));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| canvas::render(ui, self));

        if self.show_shortcuts {
            if shortcuts::render(ui.ctx()) {
                self.show_shortcuts = false;
            }
        }
        if self.show_settings {
            settings::render(ui.ctx(), self);
        }
    }
}
