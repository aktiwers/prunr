use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use egui::{Key, ViewportCommand};

use prunr_core::ProgressStage;
use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, cli_help, settings, shortcuts, sidebar, statusbar, toolbar};

/// Three-tiered history entry:
/// - Tier 1 (Hot): Raw `Arc<RgbaImage>` — instant access, full RAM cost.
/// - Tier 2 (Warm): Zstd-compressed bytes in RAM — ~3-4x smaller, ~8ms decompress.
/// - Tier 3 (Cold): Zstd file on disk — zero RAM cost, ~50-100ms read.
pub(crate) enum HistorySlot {
    /// Tier 1: uncompressed RGBA in RAM.
    InMemory(Arc<image::RgbaImage>),
    /// Tier 2: zstd-compressed in RAM (~3-4x smaller).
    Compressed(super::history_disk::CompressedEntry),
    /// Tier 3: zstd file on disk.
    OnDisk(super::history_disk::DiskHistoryEntry),
}

impl HistorySlot {
    /// Compress an RGBA image to RAM (Tier 2), falling back to uncompressed (Tier 1).
    pub fn compress(rgba: Arc<image::RgbaImage>) -> Self {
        super::history_disk::compress_to_ram(&rgba)
            .map(Self::Compressed)
            .unwrap_or(Self::InMemory(rgba))
    }

    /// Demote this slot to disk (Tier 3). Only affects Tier 1/2; Tier 3 is a no-op.
    pub fn demote_to_disk(self, item_id: u64, seq: usize) -> Self {
        match self {
            Self::InMemory(rgba) => {
                super::history_disk::write_history(item_id, seq, &rgba)
                    .map(Self::OnDisk)
                    .unwrap_or(Self::InMemory(rgba))
            }
            Self::Compressed(entry) => {
                super::history_disk::demote_to_disk(&entry, item_id, seq)
                    .map(Self::OnDisk)
                    .unwrap_or(Self::Compressed(entry))
            }
            Self::OnDisk(_) => self,
        }
    }

    /// Materialise the RGBA image from any tier.
    /// Deletes the backing file only on successful disk read.
    pub fn into_rgba(self) -> Option<Arc<image::RgbaImage>> {
        match self {
            Self::InMemory(rgba) => Some(rgba),
            Self::Compressed(entry) => {
                super::history_disk::decompress_from_ram(&entry)
                    .ok()
                    .map(|img| Arc::new(img))
            }
            Self::OnDisk(entry) => match super::history_disk::read_history(&entry) {
                Ok(img) => {
                    super::history_disk::delete_entry(&entry);
                    Some(Arc::new(img))
                }
                Err(_) => None,
            },
        }
    }

    /// Delete the backing disk file if Tier 3 (no-op for Tier 1/2).
    pub fn cleanup(&self) {
        if let Self::OnDisk(entry) = self {
            super::history_disk::delete_entry(entry);
        }
    }
}

impl Default for HistorySlot {
    fn default() -> Self {
        Self::InMemory(Arc::new(image::RgbaImage::new(1, 1)))
    }
}

/// Where an image's raw bytes live — file path (lazy) or in-memory (clipboard/paste).
#[derive(Clone)]
pub(crate) enum ImageSource {
    /// Loaded from a file. Bytes read on demand and dropped after use.
    Path(PathBuf),
    /// From clipboard, drag-drop, or CLI pipe. Bytes kept in memory.
    Bytes(Arc<Vec<u8>>),
}

impl ImageSource {
    /// Read the image bytes. For Path, reads from disk. For Bytes, clones the Arc.
    pub fn load_bytes(&self) -> std::io::Result<Arc<Vec<u8>>> {
        match self {
            Self::Path(path) => Ok(Arc::new(std::fs::read(path)?)),
            Self::Bytes(bytes) => Ok(bytes.clone()),
        }
    }

    /// Estimated compressed file size (for admission cost estimation).
    pub fn estimated_size(&self) -> usize {
        match self {
            Self::Path(path) => std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0),
            Self::Bytes(bytes) => bytes.len(),
        }
    }
}

pub(crate) struct BatchItem {
    pub id: u64,
    pub filename: String,
    pub source: ImageSource,
    pub dimensions: (u32, u32),
    /// Pre-decoded source RGBA (decoded on background thread for instant switching)
    pub source_rgba: Option<Arc<image::RgbaImage>>,
    pub source_texture: Option<egui::TextureHandle>,
    pub thumb_texture: Option<egui::TextureHandle>,
    pub thumb_pending: bool,
    pub result_rgba: Option<Arc<image::RgbaImage>>,
    pub result_texture: Option<egui::TextureHandle>,
    /// True while a background thread is building the source ColorImage.
    pub source_tex_pending: bool,
    /// True while a background thread is building the result ColorImage.
    pub result_tex_pending: bool,
    /// True while a background thread is decoding source bytes to RGBA.
    pub decode_pending: bool,
    /// History stack for undo: previous results, newest last.
    pub history: VecDeque<HistorySlot>,
    /// Redo stack: results undone, newest last. Cleared on new processing.
    pub redo_stack: VecDeque<HistorySlot>,
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

    // ── Drag-out (OS drag to external apps) ────────────────────────────────
    /// One-shot: set by sidebar when a drag escapes the sidebar rect.
    /// Consumed by `ui()` which then invokes the platform drag crate.
    pub(crate) drag_out_pending: Option<Vec<u64>>,
    /// True while an OS drag session is in progress. Flipped false by the
    /// drag crate's completion callback. Read by sidebar to dim thumbnails.
    pub(crate) drag_out_active: Arc<AtomicBool>,
    /// Item IDs currently being dragged — sidebar reads this to know which
    /// thumbnails to dim. Shared with the drag callback thread.
    pub(crate) drag_out_items: Arc<Mutex<HashSet<u64>>>,
    /// One-time flag: true if we've already shown the "Linux not supported"
    /// toast this session. Prevents repeat spam on every drag attempt.
    pub(crate) drag_out_linux_notified: bool,

    // ── Memory-aware batch admission ──────────────────────────────────────
    /// Active admission controller (present only during streaming batch processing).
    admission: Option<super::memory::AdmissionController>,
    /// Sender for streaming additional items to the worker.
    admission_tx: Option<mpsc::Sender<super::worker::WorkItem>>,
    /// Last time periodic history cleanup ran.
    last_history_cleanup: std::time::Instant,
}

/// Compute dimensions that fit within max_w x max_h preserving aspect ratio.
fn fit_dimensions(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32).min(1.0);
    ((src_w as f32 * scale).round().max(1.0) as u32,
     (src_h as f32 * scale).round().max(1.0) as u32)
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

        // Housekeeping: clean up stale temp files from prior sessions.
        super::drag_export::cleanup_stale();
        super::history_disk::cleanup_stale();

        let mut settings = Settings::load();
        settings.active_backend = prunr_core::OrtEngine::detect_active_provider();

        // Subprocess worker: inference runs in a child process for OOM isolation.
        // No prewarm needed — the subprocess creates its own engine pool.
        let (worker_tx, worker_rx) = spawn_worker(worker_ctx);

        Self {
            state: AppState::Empty,
            loaded_filename: None,
            last_open_dir: None,
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
            drag_out_pending: None,
            drag_out_active: Arc::new(AtomicBool::new(false)),
            drag_out_items: Arc::new(Mutex::new(HashSet::new())),
            drag_out_linux_notified: false,
            admission: None,
            admission_tx: None,
            last_history_cleanup: std::time::Instant::now(),
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
            drag_out_pending: None,
            drag_out_active: Arc::new(AtomicBool::new(false)),
            drag_out_items: Arc::new(Mutex::new(HashSet::new())),
            drag_out_linux_notified: false,
            admission: None,
            admission_tx: None,
            last_history_cleanup: std::time::Instant::now(),
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

    /// Core image loading: creates a BatchItem from a source + dimensions.
    fn load_image_source(&mut self, source: ImageSource, dims: (u32, u32), name: String) {
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        let do_decode = matches!(&source, ImageSource::Bytes(_)); // decode eagerly for in-memory
        self.batch_items.push(BatchItem {
            id,
            filename: name.clone(),
            source: source,
            dimensions: dims,
            source_rgba: None,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            source_tex_pending: false,
            result_tex_pending: false,
            decode_pending: false,
            history: VecDeque::new(),
            redo_stack: VecDeque::new(),
            status: BatchStatus::Pending,
            selected: false,
        });
        self.selected_batch_index = self.batch_items.len() - 1;
        if do_decode {
            if let Ok(bytes) = self.batch_items.last().unwrap().source.load_bytes() {
                self.request_decode_bytes(id, bytes);
            }
        }

        // Set app-level state for canvas
        self.image_dimensions = Some(dims);
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

    /// Load an image from raw bytes (clipboard paste, CLI pipe).
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
        self.load_image_source(ImageSource::Bytes(Arc::new(bytes)), dims, name);
    }

    pub fn handle_open_path(&mut self, path: PathBuf) {
        // Read dimensions from header only — don't load the full file into RAM.
        let dims = match std::fs::File::open(&path)
            .ok()
            .and_then(|f| {
                image::ImageReader::new(std::io::BufReader::new(f))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
            })
        {
            Some(d) => d,
            None => {
                self.set_temporary_status("Could not load image");
                return;
            }
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
                // Send file paths for lazy loading — bytes read on demand.
                let tx = self.bg_io.file_load_tx.clone();
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
                // Push current result to redo stack (compress to RAM)
                if let Some(current) = item.result_rgba.take() {
                    item.redo_stack.push_back(HistorySlot::compress(current));
                }
                // Pop most recent from history (materialise from disk if needed)
                item.result_rgba = item.history.pop_back().and_then(|slot| slot.into_rgba());
                if item.result_rgba.is_none() || item.history.is_empty() {
                    // History exhausted — back to unprocessed state
                    item.status = BatchStatus::Pending;
                    item.result_rgba = None;
                }
                item.result_texture = None;
                item.thumb_texture = None;
                item.thumb_pending = false;
                item.source_tex_pending = false;
                item.result_tex_pending = false;
                // Reset decode state so lazy decode re-triggers for canvas display
                item.source_texture = None;
                item.decode_pending = false;
                undone += 1;
            }
        }
        if undone > 0 {
            self.result_switch_id += 1;
            self.canvas_switch_id += 1;
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
                // Push current result to history (compress to RAM)
                if let Some(current) = item.result_rgba.take() {
                    item.history.push_back(HistorySlot::compress(current));
                }
                // Pop most recent from redo stack (materialise from disk if needed)
                item.result_rgba = item.redo_stack.pop_back().and_then(|slot| slot.into_rgba());
                item.status = BatchStatus::Done;
                item.result_texture = None;
                item.thumb_texture = None;
                item.thumb_pending = false;
                item.source_tex_pending = false;
                item.result_tex_pending = false;
                item.decode_pending = false;
                redone += 1;
            }
        }
        if redone > 0 {
            self.result_switch_id += 1;
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
        use super::memory::{AdmissionController, ImageMemCost};

        let chain = self.settings.chain_mode;

        // Identify candidate items
        let candidate_ids: HashSet<u64> = self.batch_items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }

        // Save history for candidates before processing
        let chain_mode = self.settings.chain_mode;
        let max_depth = self.settings.history_depth;
        for item in &mut self.batch_items {
            if !candidate_ids.contains(&item.id) { continue; }
            // Seed history with the original image on the very first process.
            // This lets undo walk all the way back to "no processing applied".
            if item.history.is_empty() && item.redo_stack.is_empty() {
                if let Some(ref src_rgba) = item.source_rgba {
                    // Source already decoded — compress it to RAM
                    item.history.push_back(HistorySlot::compress(src_rgba.clone()));
                } else if let Ok(bytes) = item.source.load_bytes() {
                    // Lazy-loaded: decode source from disk for history seed
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        item.history.push_back(HistorySlot::compress(Arc::new(img.to_rgba8())));
                    }
                }
            }
            // Save current result to history before reprocessing
            if item.status == BatchStatus::Done {
                if let Some(current) = item.result_rgba.take() {
                    if chain_mode {
                        item.result_rgba = Some(current.clone());
                    }
                    item.history.push_back(HistorySlot::compress(current));
                    while item.history.len() > max_depth {
                        if let Some(old) = item.history.pop_front() {
                            old.cleanup();
                        }
                    }
                }
                for slot in item.redo_stack.drain(..) {
                    slot.cleanup();
                }
                if !chain_mode {
                    item.result_texture = None;
                }
                item.thumb_texture = None;
                item.thumb_pending = false;
                item.source_tex_pending = false;
                item.result_tex_pending = false;
            }
        }

        // Build admission controller with per-image cost estimates.
        // Clamp parallel_jobs to what the system can safely handle for this model.
        let model: prunr_core::ModelKind = self.settings.model.into();
        let safe_jobs = super::memory::safe_max_jobs(model);
        let jobs = self.settings.parallel_jobs.min(safe_jobs);
        let use_admission = candidate_ids.len() > 1;
        if use_admission {
            let mut ctrl = AdmissionController::new(model, jobs);
            let costs: Vec<ImageMemCost> = self.batch_items.iter()
                .filter(|i| candidate_ids.contains(&i.id))
                .map(|i| AdmissionController::estimate_cost(i.id, i.dimensions, i.source.estimated_size()))
                .collect();
            ctrl.enqueue(costs);

            // Admit initial window
            let mut initial_items = Vec::new();
            while let Some(admitted_id) = ctrl.try_admit_next() {
                if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == admitted_id) {
                    if let Ok(bytes) = item.source.load_bytes() {
                        let chain_input = if chain { item.result_rgba.clone() } else { None };
                        initial_items.push((item.id, bytes, chain_input));
                        item.status = BatchStatus::Processing;
                    }
                }
            }

            // Mark non-admitted items as Pending (they wait for admission)
            for item in &mut self.batch_items {
                if candidate_ids.contains(&item.id) && item.status != BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }

            // Create streaming channel for additional items
            let (atx, arx) = mpsc::channel();
            self.admission = Some(ctrl);
            self.admission_tx = Some(atx);

            self.dispatch_batch(initial_items, model, jobs, Some(arx));
            return;
        }

        // Single item: no admission needed, process directly
        let items: Vec<_> = self.batch_items.iter_mut()
            .filter(|i| candidate_ids.contains(&i.id))
            .filter_map(|i| {
                let bytes = i.source.load_bytes().ok()?;
                let chain_input = if chain { i.result_rgba.clone() } else { None };
                i.status = BatchStatus::Processing;
                Some((i.id, bytes, chain_input))
            })
            .collect();

        self.dispatch_batch(items, model, jobs, None);
    }

    /// Build and send a WorkerMessage::BatchProcess with current settings.
    fn dispatch_batch(
        &mut self,
        items: Vec<super::worker::WorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        additional_items_rx: Option<mpsc::Receiver<super::worker::WorkItem>>,
    ) {
        self.cancel_flag.store(false, Ordering::Release);
        self.state = AppState::Processing;
        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            model,
            jobs,
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
            additional_items_rx,
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

    /// Clear active drag-out state (used on drag end, error, and Linux fallback).
    fn reset_drag_out_state(
        active: &AtomicBool,
        items: &Mutex<HashSet<u64>>,
    ) {
        active.store(false, Ordering::Release);
        if let Ok(mut set) = items.lock() {
            set.clear();
        }
    }

    /// Initiate an OS drag-out for the given batch item IDs.
    /// On Windows/macOS: calls the `drag` crate with PNG temp files.
    /// On Linux: clears drag state and shows a one-time fallback toast
    /// (winit + drag crate incompatibility; see Cargo.toml comment).
    #[allow(unused_variables)]
    pub fn initiate_drag_out(&mut self, ids: Vec<u64>, frame: &eframe::Frame) {
        let line_mode = self.settings.line_mode;

        let mut paths: Vec<PathBuf> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(item) = self.batch_items.iter().find(|b| b.id == *id) {
                match super::drag_export::prepare(item, line_mode) {
                    Ok(path) => paths.push(path),
                    Err(e) => {
                        self.toasts.error(format!("Drag export failed: {e}"));
                    }
                }
            }
        }
        if paths.is_empty() {
            return;
        }

        // Publish dragged IDs so sidebar can dim those thumbnails.
        if let Ok(mut set) = self.drag_out_items.lock() {
            set.clear();
            set.extend(ids.iter().copied());
        }
        self.drag_out_active.store(true, Ordering::Release);

        #[cfg(any(target_os = "windows", target_os = "macos"))]
        {
            let active_flag = self.drag_out_active.clone();
            let items_set = self.drag_out_items.clone();
            let preview_path = paths[0].clone();

            let result = drag::start_drag(
                frame,
                drag::DragItem::Files(paths),
                drag::Image::File(preview_path),
                move |_result, _cursor| {
                    Self::reset_drag_out_state(&active_flag, &items_set);
                },
                drag::Options::default(),
            );
            if let Err(e) = result {
                Self::reset_drag_out_state(&self.drag_out_active, &self.drag_out_items);
                self.toasts.error(format!("Drag failed: {e}"));
            }
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            Self::reset_drag_out_state(&self.drag_out_active, &self.drag_out_items);
            if !self.drag_out_linux_notified {
                self.drag_out_linux_notified = true;
                self.toasts.info(
                    "Drag to external apps isn't supported on Linux yet.\n\
                     Use Ctrl+C to copy to clipboard, or the Save button to export."
                );
            }
        }
    }

    pub fn handle_copy(&mut self) {
        // Selection rules:
        // - 0 checkbox-selected → copy the currently-viewed result (original behavior).
        // - 1 checkbox-selected → copy that one (even if not the currently-viewed item).
        // - 2+ checkbox-selected → copy the first, show a hint toast about drag-out.
        //   (System clipboards can't hold multiple images as bitmaps; drag-out
        //    is the native multi-image export path.)
        let selected_with_result: Vec<Arc<image::RgbaImage>> = self
            .batch_items
            .iter()
            .filter(|b| b.selected)
            .filter_map(|b| b.result_rgba.clone())
            .collect();

        let (rgba_to_copy, multi_hint) = match selected_with_result.len() {
            0 => (self.result_rgba.clone(), None),
            1 => (Some(selected_with_result[0].clone()), None),
            n => (
                Some(selected_with_result[0].clone()),
                Some(format!(
                    "Copied 1 of {n} selected. Drag thumbnails out of the window to export multiple.")
                ),
            ),
        };

        let Some(clipboard) = self.clipboard.as_mut() else {
            self.set_temporary_status("Could not copy to clipboard. Try saving instead.");
            return;
        };
        let Some(rgba) = rgba_to_copy else {
            return;
        };

        let width = rgba.width() as usize;
        let height = rgba.height() as usize;
        let samples = rgba.as_flat_samples();
        let image_data = arboard::ImageData {
            width,
            height,
            bytes: Cow::Borrowed(samples.as_slice()),
        };
        match clipboard.set_image(image_data) {
            Ok(()) => {
                if let Some(msg) = multi_hint {
                    self.toasts.info(msg);
                } else {
                    self.set_temporary_status("Copied to clipboard");
                }
            }
            Err(_) => self.set_temporary_status(
                "Could not copy to clipboard. Try saving instead.",
            ),
        }
    }


    pub fn handle_cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::Release);
    }

    /// Add an image to the batch from a file path (lazy — bytes not loaded yet).
    pub fn add_to_batch_path(&mut self, path: PathBuf, filename: String) {
        // Read dimensions from header only
        let dims = match std::fs::File::open(&path)
            .ok()
            .and_then(|f| {
                image::ImageReader::new(std::io::BufReader::new(f))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
            })
        {
            Some(d) => d,
            None => return, // not a valid image
        };
        self.add_to_batch_source(ImageSource::Path(path), dims, filename);
    }

    /// Add an image to the batch from raw bytes (clipboard/paste).
    pub fn add_to_batch(&mut self, bytes: Vec<u8>, filename: String) {
        let dims = match image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
        {
            Some(d) => d,
            None => return,
        };
        self.add_to_batch_source(ImageSource::Bytes(Arc::new(bytes)), dims, filename);
    }

    fn add_to_batch_source(&mut self, source: ImageSource, dims: (u32, u32), filename: String) {
        let id = self.next_batch_id;
        self.next_batch_id += 1;

        self.batch_items.push(BatchItem {
            id,
            filename,
            source: source,
            dimensions: dims,
            source_rgba: None,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            result_texture: None,
            source_tex_pending: false,
            result_tex_pending: false,
            decode_pending: false,
            history: VecDeque::new(),
            redo_stack: VecDeque::new(),
            status: BatchStatus::Pending,
            selected: false,
        });

        if self.state == AppState::Empty {
            self.state = AppState::Loaded;
        }
        self.pending_batch_sync = true;
    }

    /// Pre-decode source image on a background thread (from Arc bytes).
    fn request_decode_bytes(&self, item_id: u64, bytes: Arc<Vec<u8>>) {
        let tx = self.bg_io.decode_tx.clone();
        std::thread::spawn(move || {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let _ = tx.send((item_id, Arc::new(img.to_rgba8())));
            }
        });
    }

    /// Pre-decode from an ImageSource (reads file if needed).
    fn request_decode_source(&self, item_id: u64, source: &ImageSource) {
        if let Ok(bytes) = source.load_bytes() {
            self.request_decode_bytes(item_id, bytes);
        }
    }

    /// Request thumbnail generation on a background thread for a batch item.
    /// If result_rgba is Some, thumbnails from result; otherwise decodes source bytes.
    pub(crate) fn request_thumbnail(&self, item_id: u64, source: &ImageSource, result_rgba: Option<&Arc<image::RgbaImage>>) {
        let tx = self.bg_io.thumb_tx.clone();
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone();
            std::thread::spawn(move || {
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                let thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let source = source.clone();
            std::thread::spawn(move || {
                if let Ok(bytes) = source.load_bytes() {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let rgba = img.to_rgba8();
                        let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                        let thumb = image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Triangle);
                        let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
                    }
                }
            });
        }
    }

    pub fn remove_batch_item(&mut self, idx: usize) {
        if idx >= self.batch_items.len() { return; }
        let item = self.batch_items.remove(idx);
        // Clean up any on-disk history/redo entries for this item
        for slot in item.history {
            slot.cleanup();
        }
        for slot in item.redo_stack {
            slot.cleanup();
        }
        self.sync_after_batch_change();
    }

    pub fn handle_process_all(&mut self) {
        self.process_items(|_| true);
    }

    /// Release an item's memory budget and greedily admit the next fitting items.
    fn admission_release_and_admit(&mut self, completed_id: u64) {
        let Some(ref mut ctrl) = self.admission else { return; };
        ctrl.release(completed_id);

        let chain = self.settings.chain_mode;
        while let Some(next_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == next_id) {
                let Ok(bytes) = item.source.load_bytes() else { continue; };
                let chain_input = if chain { item.result_rgba.clone() } else { None };
                let tuple = (next_id, bytes, chain_input);
                item.status = BatchStatus::Processing;

                if let Some(ref tx) = self.admission_tx {
                    if tx.send(tuple).is_err() {
                        break; // worker gone
                    }
                }
            }
        }

        // If all items admitted and released, drop the sender to signal worker
        if ctrl.is_complete() {
            self.admission_tx = None;
            self.admission = None;
        }
    }

    /// Demote all Tier 2 (compressed RAM) history/redo entries to Tier 3 (disk).
    /// Called when system memory pressure is detected.
    fn demote_history_to_disk(&mut self) {
        for item in &mut self.batch_items {
            let mut seq = 0usize;
            for slot in item.history.iter_mut().chain(item.redo_stack.iter_mut()) {
                if matches!(slot, HistorySlot::Compressed(_)) {
                    *slot = std::mem::take(slot).demote_to_disk(item.id, seq);
                }
                seq += 1;
            }
        }
    }

    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        if self.batch_items.is_empty() { return; }
        let idx = self.selected_batch_index.min(self.batch_items.len() - 1);

        // Evict full-res result_rgba for non-selected Done items to save RAM.
        // Compressed copy is saved in history; restored on demand when re-selected.
        for (i, item) in self.batch_items.iter_mut().enumerate() {
            if i != idx && item.result_rgba.is_some() && item.status == BatchStatus::Done {
                if let Some(rgba) = item.result_rgba.take() {
                    // Replace the latest history entry with the current result
                    // (don't push — the result IS the latest state)
                    if let Some(back) = item.history.back_mut() {
                        back.cleanup();
                        *back = HistorySlot::compress(rgba);
                    } else {
                        item.history.push_back(HistorySlot::compress(rgba));
                    }
                }
                item.result_texture = None;
                item.result_tex_pending = false;
            }
        }

        // Restore result for the selected item if it was evicted
        if self.batch_items[idx].status == BatchStatus::Done
            && self.batch_items[idx].result_rgba.is_none()
        {
            if let Some(slot) = self.batch_items[idx].history.back_mut() {
                // Peek at the latest history entry — decompress without removing
                let restored = match slot {
                    HistorySlot::InMemory(rgba) => Some(rgba.clone()),
                    HistorySlot::Compressed(entry) => {
                        super::history_disk::decompress_from_ram(entry)
                            .ok()
                            .map(|img| Arc::new(img))
                    }
                    HistorySlot::OnDisk(entry) => {
                        super::history_disk::read_history(entry)
                            .ok()
                            .map(|img| Arc::new(img))
                    }
                };
                self.batch_items[idx].result_rgba = restored;
            }
        }

        // Lazy decode: if the selected item has no decoded source RGBA, decode on demand.
        if self.batch_items[idx].source_rgba.is_none() && !self.batch_items[idx].decode_pending {
            self.batch_items[idx].decode_pending = true;
            self.request_decode_source(self.batch_items[idx].id, &self.batch_items[idx].source);
        }

        // Dispatch ColorImage preparation to background threads if needed.
        // The actual ctx.load_texture() happens in drain_background_channels.
        let item_id = self.batch_items[idx].id;

        if self.batch_items[idx].source_texture.is_none()
            && !self.batch_items[idx].source_tex_pending
        {
            if let Some(rgba) = self.batch_items[idx].source_rgba.clone() {
                self.batch_items[idx].source_tex_pending = true;
                Self::spawn_tex_prep(
                    rgba, item_id, format!("source_{item_id}"), false,
                    self.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }

        if self.batch_items[idx].result_texture.is_none()
            && !self.batch_items[idx].result_tex_pending
        {
            if let Some(rgba) = self.batch_items[idx].result_rgba.clone() {
                let switch = self.result_switch_id;
                self.batch_items[idx].result_tex_pending = true;
                Self::spawn_tex_prep(
                    rgba, item_id, format!("result_{item_id}_{switch}"), true,
                    self.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }

        // Sync app-level state for canvas rendering.
        // Keep the previous texture visible while the new one is being prepared
        // (avoids a spinner flash on every sidebar click).
        let item = &self.batch_items[idx];
        if item.source_texture.is_some() || item.source_rgba.is_none() {
            self.source_texture = item.source_texture.clone();
        }
        self.loaded_filename = Some(item.filename.clone());
        self.image_dimensions = Some(item.dimensions);
        self.show_original = false;

        self.zoom_state.reset();

        let item_status = item.status.clone();
        let result_texture = item.result_texture.clone();
        let result_rgba = item.result_rgba.clone();
        match item_status {
            BatchStatus::Done => {
                if item.result_texture.is_some() || item.result_rgba.is_none() {
                    self.result_texture = result_texture;
                }
                self.result_rgba = result_rgba;
                self.state = AppState::Done;
            }
            BatchStatus::Processing => {
                self.result_texture = result_texture;
                self.result_rgba = result_rgba;
                self.state = AppState::Processing;
            }
            _ => {
                self.result_texture = None;
                self.result_rgba = None;
                self.state = AppState::Loaded;
            }
        }
    }

    fn spawn_tex_prep(
        rgba: Arc<image::RgbaImage>,
        item_id: u64,
        name: String,
        is_result: bool,
        tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
        ctx: egui::Context,
    ) {
        std::thread::spawn(move || {
            let (w, h) = (rgba.width(), rgba.height());
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_flat_samples().as_slice(),
            );
            let _ = tx.send((item_id, name, ci, is_result));
            ctx.request_repaint();
        });
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
                        // Skip results for items that were already cancelled (reset to Pending)
                        if item.status != BatchStatus::Processing {
                            continue;
                        }
                        match result {
                            Ok(pr) => {
                                item.result_rgba = Some(Arc::new(pr.rgba_image));
                                item.status = BatchStatus::Done;
                                // Evict decoded source to free RAM — lazy decode
                                // will re-decode on demand if the user navigates back.
                                if !is_selected {
                                    item.source_rgba = None;
                                    item.source_texture = None;
                                }
                                item.result_texture = None;
                                item.thumb_texture = None;
                                item.thumb_pending = false;
                                item.source_tex_pending = false;
                                item.result_tex_pending = false;
                                let backend_changed = self.settings.active_backend != pr.active_provider;
                                self.settings.active_backend = pr.active_provider;
                                if backend_changed {
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

                    // Memory admission: release budget and admit next items
                    self.admission_release_and_admit(item_id);

                    // Under memory pressure: demote Tier 2 (compressed RAM) history to Tier 3 (disk)
                    if super::memory::under_memory_pressure() {
                        self.demote_history_to_disk();
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
                    self.admission = None;
                    self.admission_tx = None;
                }
                WorkerResult::SubprocessRetry { reduced_jobs, re_queued_count } => {
                    let msg = format!(
                        "Memory pressure \u{2014} retrying {re_queued_count} images with {reduced_jobs} parallel jobs"
                    );
                    self.toasts.warning(msg.clone());
                    self.status.text = msg;
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
        let mut saw_self_drop = false;

        for file in dropped {
            if let Some(path) = file.path {
                // Reject self-originated drops: our own drag-out writes temp files
                // under prunr-drag/. Dropping those back onto the canvas would
                // reopen them as new images.
                if super::drag_export::is_self_drop(&path) {
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
            self.drag_out_active.store(false, Ordering::Release);
            if let Ok(mut set) = self.drag_out_items.lock() {
                set.clear();
            }
            ctx.stop_dragging();
            return;
        }

        // Send file paths for lazy loading (avoids reading all into RAM upfront)
        if !paths.is_empty() {
            let tx = self.bg_io.file_load_tx.clone();
            std::thread::spawn(move || {
                for path in paths {
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
            // Immediately drop admission state so no more items are admitted
            self.admission = None;
            self.admission_tx = None;
            for item in &mut self.batch_items {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
        } else if cancel_requested && self.state == AppState::Processing {
            self.handle_cancel();
            self.admission = None;
            self.admission_tx = None;
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
        let mut decode_arrived = false;
        while let Ok((item_id, rgba)) = self.bg_io.decode_rx.try_recv() {
            if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                item.source_rgba = Some(rgba);
                item.decode_pending = false;
                decode_arrived = true;
            }
        }
        if decode_arrived {
            self.sync_selected_batch_textures(ctx);
        }

        // Receive pre-built ColorImages and upload to GPU (lightweight — just queues the upload)
        let mut tex_arrived = false;
        while let Ok((item_id, name, color_image, is_result)) = self.bg_io.tex_prep_rx.try_recv() {
            let tex = ctx.load_texture(name, color_image, egui::TextureOptions::default());
            if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                if is_result {
                    item.result_texture = Some(tex);
                    item.result_tex_pending = false;
                } else {
                    item.source_texture = Some(tex);
                    item.source_tex_pending = false;
                }
                tex_arrived = true;
            }
        }
        if tex_arrived {
            self.sync_selected_batch_textures(ctx);
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
                Ok((path, name)) => {
                    self.add_to_batch_path(path, name);
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

impl Drop for PrunrApp {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::Release);
        super::drag_export::cleanup_all();
        super::history_disk::cleanup_all();
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
        // Periodic cleanup of stale history files (every 10 minutes)
        if self.last_history_cleanup.elapsed().as_secs() >= 600 {
            self.last_history_cleanup = std::time::Instant::now();
            super::history_disk::cleanup_stale();
        }
    }


    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
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

        // Consume pending drag-out (sidebar set this when a drag escaped the sidebar).
        // Must run after sidebar renders so the user sees the drag cursor leave the area.
        if let Some(ids) = self.drag_out_pending.take() {
            self.initiate_drag_out(ids, frame);
            // Clear egui's internal drag state — the OS drag session has taken over.
            // Without this, egui keeps showing the DnD crosshair cursor because it
            // still thinks an internal drag is in progress.
            ui.ctx().stop_dragging();
        }
    }
}
