use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use egui::{Key, ViewportCommand};

use prunr_core::ProgressStage;
use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, cli_help, settings, shortcuts, sidebar, statusbar, toolbar};

pub(crate) struct BatchItem {
    pub id: u64,
    pub filename: String,
    pub source_bytes: Arc<Vec<u8>>,
    pub dimensions: (u32, u32),
    /// Pre-decoded source RGBA (decoded on background thread for instant switching)
    pub source_rgba: Option<image::RgbaImage>,
    pub source_texture: Option<egui::TextureHandle>,
    pub thumb_texture: Option<egui::TextureHandle>,
    pub thumb_pending: bool,
    pub result_rgba: Option<Arc<image::RgbaImage>>,
    pub result_texture: Option<egui::TextureHandle>,
    /// History stack for undo: previous results, newest last.
    pub history: Vec<Arc<image::RgbaImage>>,
    /// Redo stack: results undone, newest last. Cleared on new processing.
    pub redo_stack: Vec<Arc<image::RgbaImage>>,
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

pub struct PrunrApp {
    // State
    pub(crate) state: AppState,
    pub(crate) loaded_filename: Option<String>,
    /// Directory of the most recently opened file (for save dialog default)
    pub(crate) last_open_dir: Option<std::path::PathBuf>,
    pub(crate) source_bytes: Option<Arc<Vec<u8>>>,
    pub(crate) image_dimensions: Option<(u32, u32)>,

    // Worker thread communication
    worker_tx: mpsc::Sender<WorkerMessage>,
    worker_rx: mpsc::Receiver<WorkerResult>,
    pub(crate) cancel_flag: Arc<AtomicBool>,

    pub(crate) status: super::status_state::StatusState,

    // Textures
    pub(crate) source_texture: Option<egui::TextureHandle>,
    pub(crate) result_texture: Option<egui::TextureHandle>,

    // Result image for save/copy
    pub(crate) result_rgba: Option<Arc<image::RgbaImage>>,

    // Clipboard (MUST live for app lifetime -- Wayland ownership requirement)
    clipboard: Option<arboard::Clipboard>,

    // UI state
    pub(crate) show_shortcuts: bool,
    pub(crate) show_cli_help: bool,

    // Set by raw_input_hook — egui converts Ctrl+C to Event::Copy before we see it
    pending_copy: bool,

    pub(crate) zoom_state: super::zoom_state::ZoomState,

    // Before/After toggle
    pub(crate) show_original: bool,

    // Window title change detection
    prev_title: String,

    // Batch items
    pub(crate) batch_items: Vec<BatchItem>,
    pub(crate) selected_batch_index: usize,
    /// User explicitly hid the sidebar via Tab
    pub(crate) sidebar_hidden: bool,
    pub(crate) next_batch_id: u64,

    // Settings
    pub(crate) show_settings: bool,
    /// Timestamp when settings was last opened (for click-outside debounce)
    pub(crate) settings_opened_at: f64,
    pub(crate) settings: Settings,

    // Canvas fade-in: incremented on every image switch
    pub(crate) canvas_switch_id: u64,
    /// Incremented when a result completes, drives crossfade in render_done
    pub(crate) result_switch_id: u64,

    /// Set by add_to_batch — triggers sync_selected_batch_textures in next logic()
    pending_batch_sync: bool,
    /// Set by toolbar Open button — processed in logic() where ctx is available
    pub(crate) pending_open_dialog: bool,
    pub(crate) bg_io: super::background_io::BackgroundIO,
    /// Toast notification system
    pub(crate) toasts: egui_notify::Toasts,
}

/// Compute dimensions that fit within max_w x max_h preserving aspect ratio.
fn fit_dimensions(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32).min(1.0);
    ((src_w as f32 * scale).round().max(1.0) as u32,
     (src_h as f32 * scale).round().max(1.0) as u32)
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


impl PrunrApp {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        // Worker is spawned below after prewarm_engine is created
        let worker_ctx = cc.egui_ctx.clone();

        // Initialize material icons font
        egui_material_icons::initialize(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);

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
        visuals.error_fg_color = theme::DESTRUCTIVE;
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
        let bg_io = super::background_io::BackgroundIO::new();
        let mut settings = Settings::load();
        settings.active_backend = prunr_core::OrtEngine::detect_active_provider();

        // Pre-warm the default model on a background thread.
        // On macOS, CoreML compiles the ONNX model on first use (can take minutes).
        // By starting this at launch, the model is ready by the time the user needs it.
        // The warmed engine is stored and passed to the worker to avoid re-creation.
        let prewarm_engine: Arc<std::sync::OnceLock<prunr_core::OrtEngine>> = Arc::new(std::sync::OnceLock::new());
        {
            let model: prunr_core::ModelKind = settings.model.into();
            let lock = prewarm_engine.clone();
            std::thread::Builder::new()
                .name("model-prewarm".into())
                .spawn(move || {
                    if let Ok(engine) = prunr_core::OrtEngine::new(model, 1) {
                        let _ = lock.set(engine);
                    }
                })
                .ok();
        }
        let (worker_tx, worker_rx) = spawn_worker(worker_ctx, prewarm_engine);

        Self {
            state: AppState::Empty,
            loaded_filename: None,
            last_open_dir: None,
            source_bytes: None,
            image_dimensions: None,
            worker_tx,
            worker_rx,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            status: Default::default(),
            source_texture: None,
            result_texture: None,
            result_rgba: None,
            clipboard,
            show_shortcuts: false,
            show_cli_help: false,
            pending_copy: false,
            zoom_state: Default::default(),
            show_original: false,
            prev_title: String::new(),
            batch_items: Vec::new(),
            selected_batch_index: 0,
            sidebar_hidden: false,
            next_batch_id: 0,
            show_settings: false,
            settings_opened_at: 0.0,
            settings,
            canvas_switch_id: 0,
            result_switch_id: 0,
            pending_batch_sync: false,
            pending_open_dialog: false,
            bg_io,
            toasts: egui_notify::Toasts::default()
                    .with_anchor(egui_notify::Anchor::TopLeft)
                    .with_margin(egui::vec2(theme::SPACE_SM, theme::TOOLBAR_HEIGHT + theme::SPACE_SM)),
        }
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        let bg_io = super::background_io::BackgroundIO::new();
        let settings = Settings::default();
        Self {
            state: AppState::Empty,
            loaded_filename: None,
            last_open_dir: None,
            source_bytes: None,
            image_dimensions: None,
            worker_tx,
            worker_rx,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            status: Default::default(),
            source_texture: None,
            result_texture: None,
            result_rgba: None,
            clipboard: None,
            show_shortcuts: false,
            show_cli_help: false,
            pending_copy: false,
            zoom_state: Default::default(),
            show_original: false,
            prev_title: String::new(),
            batch_items: Vec::new(),
            selected_batch_index: 0,
            sidebar_hidden: false,
            next_batch_id: 0,
            show_settings: false,
            settings_opened_at: 0.0,
            settings,
            canvas_switch_id: 0,
            result_switch_id: 0,
            pending_batch_sync: false,
            pending_open_dialog: false,
            bg_io,
            toasts: egui_notify::Toasts::default()
                    .with_anchor(egui_notify::Anchor::TopLeft)
                    .with_margin(egui::vec2(theme::SPACE_SM, theme::TOOLBAR_HEIGHT + theme::SPACE_SM)),
        }
    }

    fn set_temporary_status(&mut self, text: impl Into<String>) {
        let msg: String = text.into();
        if msg.contains("fail") || msg.contains("Could not") || msg.contains("not available") {
            self.toasts.error(msg.clone());
        } else {
            self.toasts.success(msg.clone());
        }
        self.status.set_temporary(&msg);
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
        let bytes = Arc::new(bytes);
        self.batch_items.push(BatchItem {
            id,
            filename: name.clone(),
            source_bytes: bytes.clone(),
            dimensions: dims,
            source_rgba: None,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            history: Vec::new(),
            redo_stack: Vec::new(),
            status: BatchStatus::Pending,
            selected: false,
        });
        self.selected_batch_index = self.batch_items.len() - 1;
        self.request_decode(id, &self.batch_items.last().unwrap().source_bytes);

        // Set app-level state for canvas
        self.image_dimensions = Some(dims);
        self.source_bytes = Some(bytes);
        self.loaded_filename = Some(name);
        self.source_texture = None;
        self.result_texture = None;
        self.result_rgba = None;

        self.state = AppState::Loaded;
        self.status.text = "Ready".to_string();
        self.canvas_switch_id += 1;
        self.zoom_state.reset();
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
            if let Some(first) = paths.first() {
                self.last_open_dir = first.parent().map(|p| p.to_path_buf());
            }
            if paths.len() == 1 && self.batch_items.is_empty() {
                self.handle_open_path(paths.into_iter().next().unwrap());
            } else {
                // Read files on a background thread to avoid blocking the UI
                let tx = self.bg_io.file_load_tx.clone();
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
            self.process_items(|item| item.selected);
        } else {
            // Process current image via batch path (item_id tracking ensures
            // result goes to correct image even if user switches during processing)
            let idx = self.selected_batch_index.min(self.batch_items.len().saturating_sub(1));
            if let Some(target_id) = self.batch_items.get(idx).map(|b| b.id) {
                self.process_items(|item| item.id == target_id);
            }
        }
    }

    pub(crate) fn close_settings(&mut self) {
        self.show_settings = false;
        self.settings.save();
        self.toasts.info("Settings saved");
    }

    pub(crate) fn selected_item(&self) -> Option<&BatchItem> {
        self.batch_items.get(self.selected_batch_index)
    }

    pub(crate) fn any_modal_open(&self) -> bool {
        self.show_settings || self.show_shortcuts || self.show_cli_help
    }

    /// Undo background removal on selected items (or current item if none selected).
    /// Reverts Done/Error items back to Pending, clearing their results.
    fn handle_undo(&mut self, ctx: &egui::Context) {
        let has_selected = self.batch_items.iter().any(|i| i.selected);
        let current_id = self.selected_item().map(|b| b.id);
        let mut undone = 0u32;
        for item in &mut self.batch_items {
            let target = if has_selected { item.selected } else { Some(item.id) == current_id };
            if target && item.status == BatchStatus::Done && !item.history.is_empty() {
                // Push current result to redo stack
                if let Some(current) = item.result_rgba.take() {
                    item.redo_stack.push(current);
                }
                // Pop from history
                item.result_rgba = item.history.pop();
                if item.result_rgba.is_none() {
                    item.status = BatchStatus::Pending;
                }
                item.result_texture = None;
                item.thumb_texture = None;
                item.thumb_pending = false;
                undone += 1;
            }
        }
        if undone > 0 {
            self.sync_selected_batch_textures(ctx);
            if undone == 1 {
                self.toasts.info("Undone");
            } else {
                self.toasts.info(format!("Undone {undone} images"));
            }
        }
    }

    fn handle_redo(&mut self, ctx: &egui::Context) {
        let has_selected = self.batch_items.iter().any(|i| i.selected);
        let current_id = self.selected_item().map(|b| b.id);
        let mut redone = 0u32;
        for item in &mut self.batch_items {
            let target = if has_selected { item.selected } else { Some(item.id) == current_id };
            if target && !item.redo_stack.is_empty() {
                // Push current result to history
                if let Some(current) = item.result_rgba.take() {
                    item.history.push(current);
                }
                // Pop from redo
                item.result_rgba = item.redo_stack.pop();
                item.status = BatchStatus::Done;
                item.result_texture = None;
                item.thumb_texture = None;
                item.thumb_pending = false;
                redone += 1;
            }
        }
        if redone > 0 {
            self.sync_selected_batch_textures(ctx);
            if redone == 1 {
                self.toasts.info("Result restored");
            } else {
                self.toasts.info(format!("Restored {redone} images"));
            }
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        let chain = self.settings.chain_mode;
        let items: Vec<(u64, Arc<Vec<u8>>, Option<Arc<image::RgbaImage>>)> = self.batch_items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| {
                let chain_input = if chain { i.result_rgba.clone() } else { None };
                (i.id, i.source_bytes.clone(), chain_input)
            })
            .collect();
        if items.is_empty() { return; }
        for item in &mut self.batch_items {
            if filter(item) && !matches!(item.status, BatchStatus::Processing) {
                // Save current result to history before reprocessing
                if item.status == BatchStatus::Done {
                    if let Some(current) = item.result_rgba.take() {
                        item.history.push(current);
                        // Enforce depth limit
                        let max = self.settings.history_depth;
                        while item.history.len() > max {
                            item.history.remove(0);
                        }
                    }
                    // Clear redo stack on new processing (standard behavior)
                    item.redo_stack.clear();
                    item.result_texture = None;
                    item.thumb_texture = None;
                    item.thumb_pending = false;
                }
                item.status = BatchStatus::Processing;
            }
        }
        self.cancel_flag.store(false, Ordering::Relaxed);
        self.state = AppState::Processing;
        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            model: self.settings.model.into(),
            jobs: self.settings.parallel_jobs,
            cancel: self.cancel_flag.clone(),
            mask: self.settings.mask_settings(),
            force_cpu: self.settings.force_cpu,
            line_mode: self.settings.line_mode,
            line_strength: self.settings.line_strength,
            solid_line_color: if self.settings.solid_line_color {
                let c = self.settings.line_color;
                Some([c[0], c[1], c[2]])
            } else { None },
            bg_color: if self.settings.apply_bg_color {
                let c = self.settings.bg_color;
                Some([c[0], c[1], c[2]])
            } else { None },
        });
    }

    /// Save selected images (or current image if none selected).
    /// Single selection → save-as dialog; multiple → folder picker.
    pub(crate) fn save_dialog(&self) -> rfd::FileDialog {
        let mut dlg = rfd::FileDialog::new();
        if let Some(ref dir) = self.last_open_dir {
            dlg = dlg.set_directory(dir);
        }
        dlg
    }

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
                if let Some(path) = self.save_dialog()
                    .add_filter("PNG Image", &["png"])
                    .set_file_name(&default_name)
                    .set_title("Save PNG")
                    .save_file()
                {
                    let rgba = rgba.clone();
                    let tx = self.bg_io.save_done_tx.clone();
                    self.toasts.info("Saving...");
                    std::thread::spawn(move || {
                        let msg = match prunr_core::encode_rgba_png(&rgba) {
                            Ok(png_bytes) => match std::fs::write(&path, &png_bytes) {
                                Ok(()) => "Saved".into(),
                                Err(e) => format!("Save failed: {e}"),
                            },
                            Err(e) => format!("Save failed: {e}"),
                        };
                        let _ = tx.send(msg);
                    });
                }
            }
            return;
        }

        // Multiple selected — folder picker, encode+write on background thread
        if let Some(folder) = self.save_dialog()
            .set_title("Save Selected — Choose Folder")
            .pick_folder()
        {
            let items: Vec<(String, Arc<image::RgbaImage>)> = selected.iter()
                .filter_map(|item| {
                    let rgba = item.result_rgba.clone()?;
                    Some((item.filename.clone(), rgba))
                })
                .collect();
            let count = items.len();
            self.toasts.info(format!("Saving {count} image(s)..."));
            let tx = self.bg_io.save_done_tx.clone();
            std::thread::spawn(move || {
                let mut saved = 0usize;
                let mut failed = 0usize;
                for (filename, rgba) in items {
                    let stem = Path::new(&filename)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("image");
                    let out_path = folder.join(format!("{stem}-nobg.png"));
                    match prunr_core::encode_rgba_png(&rgba) {
                        Ok(png_bytes) => match std::fs::write(&out_path, &png_bytes) {
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
    }

    pub fn remove_selected(&mut self) {
        let count = self.batch_items.iter().filter(|i| i.selected).count();
        self.batch_items.retain(|item| !item.selected);
        self.sync_after_batch_change();
        if count > 0 {
            self.toasts.info(format!("Removed {count} image(s)"));
        }
    }

    pub fn handle_copy(&mut self) {
        if let Some(ref mut clipboard) = self.clipboard {
            if let Some(ref rgba) = self.result_rgba {
                let width = rgba.width() as usize;
                let height = rgba.height() as usize;
                let samples = rgba.as_flat_samples();
                let image_data = arboard::ImageData {
                    width,
                    height,
                    bytes: Cow::Borrowed(samples.as_slice()),
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

        let bytes = Arc::new(bytes);

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
                    source_rgba: None,
                    source_texture: self.source_texture.take(),
                    thumb_texture: None,
                    thumb_pending: false,
                    result_rgba: self.result_rgba.take(),
                    result_texture: self.result_texture.take(),
                    history: Vec::new(),
                    redo_stack: Vec::new(),
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
            source_rgba: None,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            history: Vec::new(),
            redo_stack: Vec::new(),
            status: BatchStatus::Pending,
            selected: false,
        });
        self.request_decode(id, &self.batch_items.last().unwrap().source_bytes);

        if self.state == AppState::Empty {
            self.state = AppState::Loaded;
        }
        self.pending_batch_sync = true;
    }

    /// Pre-decode source image on a background thread for instant canvas switching.
    fn request_decode(&self, item_id: u64, bytes: &Arc<Vec<u8>>) {
        let tx = self.bg_io.decode_tx.clone();
        let bytes = bytes.clone(); // Arc clone — cheap pointer copy
        std::thread::spawn(move || {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let _ = tx.send((item_id, img.to_rgba8()));
            }
        });
    }

    /// Request thumbnail generation on a background thread for a batch item.
    /// If result_rgba is Some, thumbnails from result; otherwise decodes source bytes.
    pub(crate) fn request_thumbnail(&self, item_id: u64, source_bytes: &[u8], result_rgba: Option<&Arc<image::RgbaImage>>) {
        let tx = self.bg_io.thumb_tx.clone();
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone(); // Arc clone = cheap pointer copy
            std::thread::spawn(move || {
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                let thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let bytes = source_bytes.to_vec();
            std::thread::spawn(move || {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    let rgba = img.to_rgba8();
                    let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                    let thumb = image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Triangle);
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

    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        if self.batch_items.is_empty() { return; }
        let idx = self.selected_batch_index.min(self.batch_items.len() - 1);

        // Create source texture from pre-decoded RGBA (fast GPU upload, no decode)
        if self.batch_items[idx].source_texture.is_none() {
            if let Some(ref rgba) = self.batch_items[idx].source_rgba {
                let item_id = self.batch_items[idx].id;
                self.batch_items[idx].source_texture = Some(
                    rgba_to_texture(rgba, &format!("source_{item_id}"), ctx),
                );
            }
            // source_rgba not ready yet — background decode thread will deliver it
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
        self.show_original = false;

        self.zoom_state.reset();

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

impl PrunrApp {
    fn poll_worker_results(&mut self, ctx: &egui::Context) {
        // Cap messages per frame to keep the UI responsive during heavy batch processing.
        // Remaining messages are picked up next frame (request_repaint_after ensures continuity).
        for _ in 0..8 {
            let Ok(msg) = self.worker_rx.try_recv() else { break };
            match msg {
                WorkerResult::BatchProgress { item_id, stage, pct } => {
                    // Update progress if this is the currently viewed item
                    let is_selected = self.selected_item()
                        .map_or(false, |b| b.id == item_id);
                    if is_selected {
                        self.status.stage = match stage {
                            ProgressStage::LoadingModel => "Loading model...".into(),
                            ProgressStage::LoadingModelCpuFallback => "GPU warming up \u{2014} using CPU".into(),
                            ProgressStage::Decode => "Decoding image".into(),
                            ProgressStage::Resize => "Resizing".into(),
                            ProgressStage::Normalize => "Normalizing pixels".into(),
                            ProgressStage::Infer => "Running AI model".into(),
                            ProgressStage::Postprocess => "Building mask".into(),
                            ProgressStage::Alpha => "Applying transparency".into(),
                        };
                        self.status.pct = pct;
                    }
                }
                WorkerResult::BatchItemDone { item_id, result } => {
                    let is_selected = self.selected_item()
                        .map_or(false, |b| b.id == item_id);
                    if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                        match result {
                            Ok(pr) => {
                                item.result_rgba = Some(Arc::new(pr.rgba_image));
                                item.status = BatchStatus::Done;
                                item.thumb_texture = None;
                                item.thumb_pending = false;
                                let backend_changed = self.settings.active_backend != pr.active_provider;
                                self.settings.active_backend = pr.active_provider;
                                if backend_changed {
                                    // Adjust parallel jobs to smart default for detected backend
                                    self.settings.parallel_jobs = self.settings.default_jobs();
                                }
                            }
                            Err(e) => {
                                item.status = BatchStatus::Error(e);
                            }
                        }
                    }
                    // Update progress info
                    let done = self.batch_items.iter().filter(|i| i.status == BatchStatus::Done).count();
                    let total = self.batch_items.len();
                    let processing = self.batch_items.iter().filter(|i| i.status == BatchStatus::Processing).count();
                    if processing > 0 {
                        self.status.stage = format!("Processing {done}/{total}");
                    } else {
                        self.status.stage = "Finishing up".to_string();
                    }
                    self.status.pct = done as f32 / total.max(1) as f32;

                    if is_selected {
                        self.result_switch_id += 1;
                        self.sync_selected_batch_textures(ctx);
                    }
                }
                WorkerResult::BatchComplete => {
                    let done = self.batch_items.iter().filter(|i| i.status == BatchStatus::Done).count();
                    let failed = self.batch_items.iter().filter(|i| matches!(i.status, BatchStatus::Error(_))).count();
                    let still_processing = self.batch_items.iter().any(|i| i.status == BatchStatus::Processing);
                    if failed > 0 {
                        let msg = format!("{failed} image(s) failed to process");
                        self.status.text = msg.clone();
                        self.toasts.warning(msg);
                    } else if !still_processing {
                        let msg = format!("All done \u{2014} {done} images processed");
                        self.status.text = msg.clone();
                        self.toasts.success(msg);
                    }
                    // Update app state to match viewed item (textures already synced by BatchItemDone)
                    if !still_processing {
                        let idx = self.selected_batch_index.min(self.batch_items.len().saturating_sub(1));
                        if let Some(item) = self.batch_items.get(idx) {
                            match item.status {
                                BatchStatus::Done => self.state = AppState::Done,
                                BatchStatus::Processing => self.state = AppState::Processing,
                                _ => self.state = AppState::Loaded,
                            }
                        }
                    }
                }
                WorkerResult::Cancelled => {
                    if self.state == AppState::Processing {
                        self.state = AppState::Loaded;
                        self.status.text = "Cancelled".to_string();
                    }
                }
            }
        }
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() { return; }

        // Collect paths (need background I/O) and inline bytes (already in memory)
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut inline_items: Vec<(Vec<u8>, String)> = Vec::new();

        for file in dropped {
            if let Some(path) = file.path {
                self.last_open_dir = path.parent().map(|p| p.to_path_buf());
                paths.push(path);
            } else if let Some(bytes) = file.bytes {
                inline_items.push((bytes.to_vec(), file.name.clone()));
            }
        }

        // Offload path-based file reads to background thread (avoids GUI freeze)
        if !paths.is_empty() {
            let tx = self.bg_io.file_load_tx.clone();
            std::thread::spawn(move || {
                for path in paths {
                    if let Ok(bytes) = std::fs::read(&path) {
                        let name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("untitled")
                            .to_string();
                        let _ = tx.send((bytes, name));
                    }
                }
            });
        }

        // Handle inline bytes immediately (Wayland — already in memory, no I/O)
        if !inline_items.is_empty() {
            if inline_items.len() == 1 && self.batch_items.is_empty() {
                let (bytes, name) = inline_items.into_iter().next().unwrap();
                self.handle_open_bytes(bytes, name);
            } else {
                let id_floor = self.next_batch_id;
                let count = inline_items.len();
                for (bytes, name) in inline_items {
                    self.add_to_batch(bytes, name);
                }
                if count == 1 {
                    self.selected_batch_index = self.batch_items.len() - 1;
                    self.pending_batch_sync = true;
                }
                if self.settings.auto_remove_on_import && self.next_batch_id > id_floor {
                    self.process_items(|item| item.id >= id_floor);
                }
            }
        }
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        let (mut open_requested, mut remove_requested, mut save_requested) =
            (false, false, false);
        let (mut cancel_requested, mut toggle_shortcuts) = (false, false);
        let (mut toggle_before_after, mut fit_to_window, mut actual_size) = (false, false, false);
        let mut toggle_settings = false;
        let (mut nav_prev, mut nav_next, mut toggle_sidebar) = (false, false, false);
        let mut undo_requested = false;
        let mut redo_requested = false;

        // Suppress bare-key shortcuts when any widget has focus (e.g., hex color input)
        let text_focused = ctx.memory(|m| m.focused().is_some());

        ctx.input(|i| {
            // Modifier shortcuts always work
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
            if i.key_pressed(Key::F2) {
                self.show_cli_help = !self.show_cli_help;
            }
            if i.modifiers.command && i.key_pressed(Key::Num0) {
                fit_to_window = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Num1) {
                actual_size = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Space) {
                toggle_settings = true;
            }
            // Bare-key shortcuts — only when no text field is focused
            if !text_focused {
                if i.key_pressed(Key::B) {
                    toggle_before_after = true;
                }
                if i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::A) {
                    nav_prev = true;
                }
                if i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::D) {
                    nav_next = true;
                }
                if i.key_pressed(Key::Tab) {
                    toggle_sidebar = true;
                }
            }
            if i.modifiers.command && i.key_pressed(Key::Z) {
                undo_requested = true;
            }
            if i.modifiers.command && i.key_pressed(Key::Y) {
                redo_requested = true;
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
            self.zoom_state.pending_fit_zoom = true;
        }
        if actual_size {
            self.zoom_state.pending_actual_size = true;
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
        } else if cancel_requested && self.state == AppState::Processing {
            self.handle_cancel();
            self.state = AppState::Loaded;
            self.status.text = "Cancelled".to_string();
        } else if cancel_requested && self.show_settings {
            self.close_settings();
        } else if cancel_requested && self.show_shortcuts {
            self.show_shortcuts = false;
        } else if cancel_requested && self.show_cli_help {
            self.show_cli_help = false;
        }
        if toggle_shortcuts {
            self.show_shortcuts = !self.show_shortcuts;
        }
        if toggle_settings {
            if self.show_settings {
                self.close_settings();
            } else {
                self.show_settings = true;
                self.settings_opened_at = ctx.input(|i| i.time);
            }
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
            self.sidebar_hidden = !self.sidebar_hidden;
        }
        if undo_requested {
            self.handle_undo(ctx);
        }
        if redo_requested {
            self.handle_redo(ctx);
        }
        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }
    }

    fn drain_background_channels(&mut self, ctx: &egui::Context) {
        while let Ok((item_id, rgba)) = self.bg_io.decode_rx.try_recv() {
            if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                item.source_rgba = Some(rgba);
            }
        }

        // Drain files loaded by background thread (max 5 per frame to stay responsive)
        // Drain save completion notifications
        while let Ok(msg) = self.bg_io.save_done_rx.try_recv() {
            if msg.contains("fail") {
                self.toasts.error(msg);
            } else {
                self.toasts.success(msg);
            }
        }

        let id_floor = self.next_batch_id;
        let mut loaded_count = 0u32;
        let mut channel_drained = false;
        for _ in 0..5 {
            match self.bg_io.file_load_rx.try_recv() {
                Ok((bytes, name)) => {
                    self.add_to_batch(bytes, name);
                    loaded_count += 1;
                }
                Err(_) => { channel_drained = true; break; }
            }
        }
        if loaded_count > 0 {
            ctx.request_repaint();
            // Select the new image if only one was loaded and no more are pending
            if loaded_count == 1 && channel_drained {
                self.selected_batch_index = self.batch_items.len() - 1;
                self.sync_selected_batch_textures(ctx);
            }
            if self.settings.auto_remove_on_import && self.next_batch_id > id_floor {
                self.process_items(|item| item.id >= id_floor);
            }
        }
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let title = if self.batch_items.len() >= 2 {
            format!("Prunr \u{2014} {} images", self.batch_items.len())
        } else {
            match &self.loaded_filename {
                Some(name) => format!("Prunr \u{2014} {name}"),
                None => "Prunr".to_string(),
            }
        };
        if title != self.prev_title {
            self.prev_title = title.clone();
            ctx.send_viewport_cmd(ViewportCommand::Title(title));
        }
    }
}

impl eframe::App for PrunrApp {
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
        self.poll_worker_results(ctx);
        self.handle_drag_and_drop(ctx);
        self.handle_keyboard_shortcuts(ctx);
        self.drain_background_channels(ctx);
        self.update_window_title(ctx);
        self.status.tick();
        if self.source_texture.is_none() && !self.batch_items.is_empty() {
            self.sync_selected_batch_textures(ctx);
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

        let sidebar_visible = !self.batch_items.is_empty() && !self.sidebar_hidden;
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
        if self.show_cli_help {
            if cli_help::render(ui.ctx(), &mut self.toasts) {
                self.show_cli_help = false;
            }
        }
        if self.show_settings {
            settings::render(ui.ctx(), self);
        }

        // Toast notifications — rendered last as foreground overlay
        self.toasts.show(ui.ctx());
    }
}
