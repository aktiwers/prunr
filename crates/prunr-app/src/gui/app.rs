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
use super::views::{adjustments_toolbar, canvas, chip, cli_help, settings, shortcuts, sidebar, statusbar, toolbar};

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

/// A history entry: image data + the recipe that produced it.
pub(crate) struct HistoryEntry {
    pub(crate) slot: HistorySlot,
    pub(crate) recipe: Option<prunr_core::ProcessingRecipe>,
}

impl HistoryEntry {
    fn new(rgba: Arc<image::RgbaImage>, recipe: Option<prunr_core::ProcessingRecipe>) -> Self {
        Self { slot: HistorySlot::compress(rgba), recipe }
    }

    fn cleanup(&self) {
        self.slot.cleanup();
    }

    fn demote_to_disk(self, item_id: u64, seq: usize) -> Self {
        Self { slot: self.slot.demote_to_disk(item_id, seq), recipe: self.recipe }
    }

    fn into_parts(self) -> (HistorySlot, Option<prunr_core::ProcessingRecipe>) {
        (self.slot, self.recipe)
    }
}

impl Default for HistoryEntry {
    fn default() -> Self {
        Self { slot: HistorySlot::default(), recipe: None }
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
    /// History stack for undo: previous results + their recipes, newest last.
    pub history: VecDeque<HistoryEntry>,
    /// Redo stack: results undone, newest last. Cleared on new processing.
    pub redo_stack: VecDeque<HistoryEntry>,
    pub status: BatchStatus,
    pub selected: bool,
    /// Per-image processing settings. Edited via the adjustments toolbar (Phase 2).
    /// In Phase 1 the settings modal edits `AppSettings.item_defaults` and close_settings
    /// propagates the template to all items to preserve v1 UX until the toolbar lands.
    pub settings: super::item_settings::ItemSettings,
    /// Settings snapshot taken when checkbox was checked; uncheck restores it.
    /// `None` once committed via Process or when the item was never checked.
    /// Phase 4 wires up the read path (selection-as-apply semantics).
    #[allow(dead_code)]
    pub pre_check_settings: Option<super::item_settings::ItemSettings>,
    /// The recipe that produced the current result_rgba. None if never processed.
    pub applied_recipe: Option<prunr_core::ProcessingRecipe>,
    /// Compressed cached tensor from Tier 1 inference (for Tier 2 mask reruns).
    pub cached_tensor: Option<super::worker::CompressedTensor>,
    /// Compressed cached DexiNed output (for Tier 2 edge reruns on line_strength tweaks).
    pub cached_edge_tensor: Option<super::worker::CompressedTensor>,
    /// Which preset was last APPLIED to this image (via the dropdown's row
    /// click or via Reset All). The preset button compares current `settings`
    /// against this preset's values to show a modified/clean icon. Stays set
    /// across unrelated tweaks — so "Portrait ✎" keeps saying Portrait even
    /// after the user modifies something.
    pub applied_preset: String,
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
    /// User explicitly hid the sidebar via Tab / configured hotkey.
    pub(crate) sidebar_hidden: bool,
    /// User explicitly hid the adjustments toolbar (rows 2 + 3) via Shift+H.
    pub(crate) adjustments_hidden: bool,
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
    /// Recipe snapshot taken at dispatch time — stored on completed items.
    dispatch_recipe: Option<prunr_core::ProcessingRecipe>,
    /// Last time periodic history cleanup ran.
    last_history_cleanup: std::time::Instant,
    /// Tier 2 live preview dispatcher. Debounces chip tweaks and runs
    /// postprocess_from_flat / finalize_edges on rayon threads.
    pub(crate) live_preview: super::live_preview::LivePreview,
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
            adjustments_hidden: false,
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
                    .with_anchor(egui_notify::Anchor::BottomLeft)
                    .with_margin(egui::vec2(theme::SPACE_SM, theme::STATUS_BAR_HEIGHT + theme::SPACE_SM)),
            drag_out_pending: None,
            drag_out_active: Arc::new(AtomicBool::new(false)),
            drag_out_items: Arc::new(Mutex::new(HashSet::new())),
            drag_out_linux_notified: false,
            admission: None,
            admission_tx: None,
            dispatch_recipe: None,
            last_history_cleanup: std::time::Instant::now(),
            live_preview: super::live_preview::LivePreview::default(),
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
            adjustments_hidden: false,
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
                    .with_anchor(egui_notify::Anchor::BottomLeft)
                    .with_margin(egui::vec2(theme::SPACE_SM, theme::STATUS_BAR_HEIGHT + theme::SPACE_SM)),
            drag_out_pending: None,
            drag_out_active: Arc::new(AtomicBool::new(false)),
            drag_out_items: Arc::new(Mutex::new(HashSet::new())),
            drag_out_linux_notified: false,
            admission: None,
            admission_tx: None,
            dispatch_recipe: None,
            last_history_cleanup: std::time::Instant::now(),
            live_preview: super::live_preview::LivePreview::default(),
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
        let new_settings = self.settings.item_defaults_for_new_item();
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
            settings: new_settings,
            pre_check_settings: None,
            applied_recipe: None,
            cached_tensor: None,
            cached_edge_tensor: None,
            applied_preset: self.settings.default_preset.clone(),
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

    pub(crate) fn close_settings(&mut self, _ctx: &egui::Context) {
        self.show_settings = false;
        self.settings.save();
        // Phase 2: modal no longer edits per-image knobs, so there's nothing
        // to propagate to batch_items and no texture invalidation needed.
        // The toolbar edits item.settings directly and handles its own
        // texture invalidation (P2.9).
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
            if target && item.status == BatchStatus::Done {
                let current_recipe = item.applied_recipe.take();
                if item.history.is_empty() {
                    if let Some(current) = item.result_rgba.take() {
                        item.redo_stack.push_back(HistoryEntry::new(current, current_recipe));
                    }
                    item.status = BatchStatus::Pending;
                    item.result_rgba = None;
                } else {
                    if let Some(current) = item.result_rgba.take() {
                        item.redo_stack.push_back(HistoryEntry::new(current, current_recipe));
                    }
                    if let Some(entry) = item.history.pop_back() {
                        let (slot, recipe) = entry.into_parts();
                        item.applied_recipe = recipe;
                        item.result_rgba = slot.into_rgba();
                    }
                    if item.result_rgba.is_none() || item.history.is_empty() {
                        item.status = BatchStatus::Pending;
                        item.result_rgba = None;
                        item.applied_recipe = None;
                    }
                }
                item.cached_tensor = None;
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
                let current_recipe = item.applied_recipe.take();
                if let Some(current) = item.result_rgba.take() {
                    item.history.push_back(HistoryEntry::new(current, current_recipe));
                }
                if let Some(entry) = item.redo_stack.pop_back() {
                    let (slot, recipe) = entry.into_parts();
                    item.applied_recipe = recipe;
                    item.result_rgba = slot.into_rgba();
                }
                item.cached_tensor = None;
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
    /// Uses tier routing: compares each item's applied_recipe against current
    /// settings to determine the minimum work needed (skip / mask rerun / full).
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        use super::memory::{AdmissionController, ImageMemCost};
        use prunr_core::RequiredTier;

        let chain = self.settings.chain_mode;
        let model: prunr_core::ModelKind = self.settings.model.into();

        // Identify candidate items (not already processing)
        let candidate_ids: HashSet<u64> = self.batch_items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }

        // ── Tier classification ──────────────────────────────────────────
        let mut tier1_ids: HashSet<u64> = HashSet::new(); // Full pipeline
        let mut tier2_ids: HashSet<u64> = HashSet::new(); // Mask rerun
        let mut skip_count = 0usize;

        for item in &mut self.batch_items {
            if !candidate_ids.contains(&item.id) { continue; }

            // Never-processed items always need full pipeline
            let Some(ref old_recipe) = item.applied_recipe else {
                tier1_ids.insert(item.id);
                continue;
            };

            // Chain mode with existing result: input changes each time → always full
            if chain && item.result_rgba.is_some() {
                item.cached_tensor = None;
                tier1_ids.insert(item.id);
                continue;
            }

            let current_recipe = item.settings.current_recipe(model, chain);
            match prunr_core::resolve_tier(old_recipe, &current_recipe) {
                RequiredTier::Skip | RequiredTier::CompositeOnly => {
                    // CompositeOnly (bg_color) is handled at display/export time,
                    // so it's effectively a skip. Update composite to stay in sync.
                    if let Some(ref mut recipe) = item.applied_recipe {
                        recipe.composite = current_recipe.composite.clone();
                    }
                    skip_count += 1;
                }
                RequiredTier::MaskRerun => {
                    if item.cached_tensor.is_some() {
                        tier2_ids.insert(item.id);
                    } else {
                        tier1_ids.insert(item.id);
                    }
                }
                RequiredTier::EdgeRerun => {
                    // Phase 1: edge rerun dispatch lands in Phase 3. For now,
                    // if we have the edge cache we still route to Tier 1 because
                    // the subprocess path isn't wired yet. When Phase 3 lands,
                    // this becomes: `if cached_edge_tensor.is_some() { tier2_edge }`.
                    item.cached_edge_tensor = None;
                    tier1_ids.insert(item.id);
                }
                RequiredTier::FullPipeline => {
                    // Model / chain / mode changed — both caches invalid.
                    item.cached_tensor = None;
                    item.cached_edge_tensor = None;
                    tier1_ids.insert(item.id);
                }
            }
        }

        let process_count = tier1_ids.len() + tier2_ids.len();
        if process_count == 0 {
            if skip_count > 0 {
                let msg = if skip_count == 1 {
                    "Already up to date".to_string()
                } else {
                    format!("{skip_count} images already up to date")
                };
                self.toasts.info(msg);
            }
            return;
        }

        if skip_count > 0 {
            let msg = format!("{skip_count} up to date, processing {process_count}");
            self.toasts.info(msg);
        }

        // ── Save history for items being reprocessed ──────────────────
        let process_ids: HashSet<u64> = tier1_ids.iter().chain(tier2_ids.iter()).copied().collect();
        let chain_mode = self.settings.chain_mode;
        let max_depth = self.settings.history_depth;
        for item in &mut self.batch_items {
            if !process_ids.contains(&item.id) { continue; }
            if item.history.is_empty() && item.redo_stack.is_empty() {
                if let Some(ref src_rgba) = item.source_rgba {
                    // Seed with original — no recipe (it's the unprocessed source)
                    item.history.push_back(HistoryEntry::new(src_rgba.clone(), None));
                }
            }
            if item.status == BatchStatus::Done {
                if let Some(current) = item.result_rgba.take() {
                    if chain_mode {
                        item.result_rgba = Some(current.clone());
                    }
                    item.history.push_back(HistoryEntry::new(current, item.applied_recipe.clone()));
                    while item.history.len() > max_depth {
                        if let Some(old) = item.history.pop_front() {
                            old.cleanup();
                        }
                    }
                }
                for entry in item.redo_stack.drain(..) {
                    entry.cleanup();
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

        // ── Build Tier 2 work items ──────────────────────────────────
        let mut tier2_work: Vec<super::worker::Tier2WorkItem> = Vec::new();
        for item in &mut self.batch_items {
            if !tier2_ids.contains(&item.id) { continue; }
            if let Some(ref ct) = item.cached_tensor {
                let tensor_data = ct.decompress();
                let mask = item.settings.mask_settings();
                match (tensor_data, item.source.load_bytes()) {
                    (Some(data), Ok(bytes)) => {
                        tier2_work.push(super::worker::Tier2WorkItem {
                            item_id: item.id,
                            tensor_data: data,
                            tensor_height: ct.height,
                            tensor_width: ct.width,
                            model: ct.model,
                            original_bytes: bytes,
                            mask,
                        });
                        item.status = BatchStatus::Processing;
                    }
                    (None, _) => {
                        item.status = BatchStatus::Error("Tensor cache corrupt".into());
                        item.cached_tensor = None;
                        item.applied_recipe = None;
                    }
                    (_, Err(e)) => {
                        item.status = BatchStatus::Error(format!("Failed to load: {e}"));
                        item.cached_tensor = None;
                        item.applied_recipe = None;
                    }
                }
            }
        }

        // ── Build Tier 1 work items + admission ──────────────────────
        let model: prunr_core::ModelKind = self.settings.model.into();
        let safe_jobs = super::memory::safe_max_jobs(model);
        let jobs = self.settings.parallel_jobs.min(safe_jobs);
        let use_admission = tier1_ids.len() > 1;

        if use_admission {
            let mut ctrl = AdmissionController::new(model, jobs);
            let costs: Vec<ImageMemCost> = self.batch_items.iter()
                .filter(|i| tier1_ids.contains(&i.id))
                .map(|i| AdmissionController::estimate_cost(i.id, i.dimensions, i.source.estimated_size()))
                .collect();
            ctrl.enqueue(costs);

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

            for item in &mut self.batch_items {
                if tier1_ids.contains(&item.id) && item.status != BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }

            // All Tier 1 items failed load_bytes AND no Tier 2 work — skip dispatch.
            if initial_items.is_empty() && tier2_work.is_empty() { return; }

            let (atx, arx) = mpsc::channel();
            self.admission = Some(ctrl);
            self.admission_tx = Some(atx);

            self.dispatch_batch(initial_items, tier2_work, model, jobs, Some(arx));
            return;
        }

        // Small batch / single item: no admission needed
        let items: Vec<_> = self.batch_items.iter_mut()
            .filter(|i| tier1_ids.contains(&i.id))
            .filter_map(|i| {
                let bytes = i.source.load_bytes().ok()?;
                let chain_input = if chain { i.result_rgba.clone() } else { None };
                i.status = BatchStatus::Processing;
                Some((i.id, bytes, chain_input))
            })
            .collect();

        // If all prep failed (both Tier 2 decompress + Tier 1 load_bytes), don't
        // dispatch an empty batch — the subprocess would spin up for nothing
        // and items stay in Error status set above.
        if items.is_empty() && tier2_work.is_empty() { return; }

        self.dispatch_batch(items, tier2_work, model, jobs, None);
    }

    /// Build and send a WorkerMessage::BatchProcess with current settings.
    fn dispatch_batch(
        &mut self,
        items: Vec<super::worker::WorkItem>,
        tier2_items: Vec<super::worker::Tier2WorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        additional_items_rx: Option<mpsc::Receiver<super::worker::WorkItem>>,
    ) {
        self.cancel_flag.store(false, Ordering::Release);
        self.state = AppState::Processing;

        // Use the currently-viewed item's settings for the batch. The toolbar
        // always binds to the current item, so this matches "what you see is
        // what you process." Phase 5's explicit Process Selected broadcast
        // will copy current.settings to each scoped item first, keeping their
        // stored settings in sync with what's actually dispatched.
        //
        // IMPORTANT: this is NOT `self.settings.item_defaults` — that's the
        // template for NEW imports and is stale relative to toolbar tweaks.
        let idx = self.selected_batch_index.min(self.batch_items.len().saturating_sub(1));
        let current_settings = self.batch_items.get(idx)
            .map(|b| b.settings)
            .unwrap_or(self.settings.item_defaults);

        // Broadcast: every item about to be processed inherits current.settings
        // so their `applied_recipe` ends up consistent with what ran. This is
        // the Phase 3 stand-in for Phase 5's explicit selection-apply.
        let process_ids: std::collections::HashSet<u64> = items.iter()
            .map(|wi| wi.0)
            .chain(tier2_items.iter().map(|ti| ti.item_id))
            .collect();
        for item in &mut self.batch_items {
            if process_ids.contains(&item.id) {
                item.settings = current_settings;
            }
        }

        self.dispatch_recipe = Some(current_settings.current_recipe(model, self.settings.chain_mode));

        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            tier2_items,
            config: super::worker::ProcessingConfig {
                model,
                jobs,
                mask: current_settings.mask_settings(),
                force_cpu: self.settings.force_cpu,
                line_mode: current_settings.line_mode,
                line_strength: current_settings.line_strength,
                solid_line_color: current_settings.solid_line_color,
            },
            cancel: self.cancel_flag.clone(),
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

    /// Apply background color to a result image for export/save.
    /// Returns a new image when `bg` is `Some`; otherwise clones the Arc.
    pub(crate) fn apply_bg_for_export(
        rgba: &Arc<image::RgbaImage>,
        bg: Option<[u8; 3]>,
    ) -> Arc<image::RgbaImage> {
        if let Some(c) = bg {
            let mut copy = (**rgba).clone();
            prunr_core::apply_background_color(&mut copy, c);
            Arc::new(copy)
        } else {
            rgba.clone()
        }
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
                    let bg = self.selected_item().and_then(|i| i.settings.bg_rgb());
                    let rgba = Self::apply_bg_for_export(rgba, bg);
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
                    let rgba = item.result_rgba.as_ref()?;
                    Some((item.filename.clone(), Self::apply_bg_for_export(rgba, item.settings.bg_rgb())))
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
        let mut paths: Vec<PathBuf> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(item) = self.batch_items.iter().find(|b| b.id == *id) {
                match super::drag_export::prepare(item) {
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

        // Compute bg before borrowing clipboard mutably (avoids aliasing self).
        let bg = self.selected_item().and_then(|i| i.settings.bg_rgb());
        let Some(clipboard) = self.clipboard.as_mut() else {
            self.set_temporary_status("Could not copy to clipboard. Try saving instead.");
            return;
        };
        let Some(rgba) = rgba_to_copy else {
            return;
        };
        // Apply bg_color for clipboard (matches display).
        let rgba = Self::apply_bg_for_export(&rgba, bg);

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

        let new_settings = self.settings.item_defaults_for_new_item();
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
            settings: new_settings,
            pre_check_settings: None,
            applied_recipe: None,
            cached_tensor: None,
            cached_edge_tensor: None,
            applied_preset: self.settings.default_preset.clone(),
        });

        if self.state == AppState::Empty {
            self.state = AppState::Loaded;
        }
        // Fit-to-window on first import so images open at a sensible size
        // (matching Ctrl+0). Any subsequent image change also fits — the reset
        // happens via zoom_state.reset() when the user navigates away.
        self.zoom_state.pending_fit_zoom = true;
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
    ///
    /// `bg` applies a background color to the thumbnail (matches display).
    /// Result images stay transparent in storage; the thumb texture needs
    /// its own composite so the sidebar matches what's drawn on the canvas.
    /// Solid line color is already baked into `result_rgba` by the pipeline,
    /// so we don't need a separate parameter for it.
    pub(crate) fn request_thumbnail(
        &self,
        item_id: u64,
        source: &ImageSource,
        result_rgba: Option<&Arc<image::RgbaImage>>,
        bg: Option<[u8; 3]>,
    ) {
        let tx = self.bg_io.thumb_tx.clone();
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone();
            std::thread::spawn(move || {
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                let mut thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
                if let Some(bg) = bg {
                    prunr_core::apply_background_color(&mut thumb, bg);
                }
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            // Source-only thumbnails have no transparency, so `bg` is a no-op.
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
        for entry in item.history {
            entry.cleanup();
        }
        for entry in item.redo_stack {
            entry.cleanup();
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
            for entry in item.history.iter_mut().chain(item.redo_stack.iter_mut()) {
                if matches!(entry.slot, HistorySlot::Compressed(_)) {
                    *entry = std::mem::take(entry).demote_to_disk(item.id, seq);
                }
                seq += 1;
            }
        }
    }

    /// Live-preview pump: dispatch debounced Tier 2 reruns + apply completed ones.
    /// Called once per frame at the start of `ui()` so the current frame renders
    /// with the newest available results.
    fn pump_live_preview(&mut self, ctx: &egui::Context) {
        // Dispatch any pending previews whose debounce has expired. The closure
        // snapshots the inputs (original image + cached tensors + settings) for
        // each dispatch — run on the UI thread, but only for items that are
        // actually dispatched this frame.
        let batch_items = &self.batch_items;
        let wait = self.live_preview.tick(|id, kind| {
            use crate::gui::live_preview::{DispatchInputs, PreviewKind, decompress_edge, decompress_seg};
            let item = batch_items.iter().find(|b| b.id == id)?;
            let rgba = item.source_rgba.as_ref()?;
            // One clone here — live preview workers need an owned DynamicImage.
            // Clones the ~48 MB RGBA once per dispatch (not per frame); the user
            // has paused typing for 300ms so a single clone is acceptable.
            let original = std::sync::Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone()));
            let seg_tensor = item.cached_tensor.as_ref().and_then(decompress_seg);
            let edge_tensor = item.cached_edge_tensor.as_ref().and_then(decompress_edge);
            // Abort dispatch if the cache needed for this kind isn't available.
            // (User must Process first to populate the tensor cache.)
            match kind {
                PreviewKind::Mask if seg_tensor.is_none() => return None,
                PreviewKind::Edge if edge_tensor.is_none() => return None,
                _ => {}
            }
            Some(DispatchInputs {
                kind,
                original,
                settings: item.settings,
                seg_tensor,
                edge_tensor,
            })
        });

        // If a future dispatch is waiting, schedule a repaint when the debounce
        // elapses so tick() can fire.
        if let Some(w) = wait {
            ctx.request_repaint_after(w);
        }

        // Apply any completed previews. Critical: do NOT null `result_texture`
        // here — the old texture must stay visible until the newly-built one
        // lands via drain_background_channels. Clearing it causes the canvas
        // to flash black for a frame (no texture to draw → BG_PRIMARY shows).
        //
        // Instead we spawn a tex prep for the new RGBA directly and let
        // drain_background_channels swap it in atomically when ready.
        let results = self.live_preview.drain_results();
        if !results.is_empty() {
            let tex_prep_tx = self.bg_io.tex_prep_tx.clone();
            for r in results {
                let Some(item) = self.batch_items.iter_mut().find(|b| b.id == r.item_id) else {
                    continue;
                };
                let new_rgba = std::sync::Arc::new(r.rgba);
                item.result_rgba = Some(new_rgba.clone());
                // Mark pending so sync_selected_batch_textures doesn't also
                // spawn its own prep on this same frame.
                item.result_tex_pending = true;
                // Invalidate the sidebar thumbnail — it was built from the
                // previous result_rgba and may have different line colors.
                // Sidebar's render loop will see `None + !pending` and queue
                // a fresh thumb generation on the next frame.
                item.thumb_texture = None;
                item.thumb_pending = false;
                let item_id = item.id;
                let switch = self.result_switch_id;
                let bg = item.settings.bg_rgb();
                Self::spawn_tex_prep(
                    new_rgba,
                    item_id,
                    format!("result_{item_id}_{switch}"),
                    true,
                    bg,
                    tex_prep_tx.clone(),
                    ctx.clone(),
                );
            }
            ctx.request_repaint();
        }
    }

    /// 512 MB budget for compressed tensor caches across all items.
    /// Budget covers BOTH segmentation (`cached_tensor`) and DexiNed
    /// (`cached_edge_tensor`) caches combined.
    const TENSOR_BUDGET: usize = 512 * 1024 * 1024;

    /// Total compressed bytes across both caches for a single item.
    fn item_cache_size(item: &BatchItem) -> usize {
        let seg = item.cached_tensor.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
        let edge = item.cached_edge_tensor.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
        seg + edge
    }

    /// Evict tensor caches from oldest-loaded items until under budget.
    /// Iterates front-to-back (oldest first) to preserve recently-processed items.
    /// Drops BOTH caches on eviction — partial eviction would leave a partially-stale
    /// item (segmentation cached but edges gone, or vice versa) which is useless.
    fn enforce_tensor_budget(&mut self) {
        let total: usize = self.batch_items.iter().map(Self::item_cache_size).sum();
        if total <= Self::TENSOR_BUDGET { return; }
        let selected_id = self.selected_item().map(|b| b.id);
        let mut remaining = total;
        for item in &mut self.batch_items {
            if remaining <= Self::TENSOR_BUDGET { break; }
            // Preserve the selected item's tensors (most likely to be reused)
            if Some(item.id) == selected_id { continue; }
            remaining -= Self::item_cache_size(item);
            item.cached_tensor = None;
            item.cached_edge_tensor = None;
        }
    }

    /// Evict all tensor caches except the selected item (called under memory pressure).
    fn evict_all_tensors(&mut self) {
        let selected_id = self.selected_item().map(|b| b.id);
        for item in &mut self.batch_items {
            if Some(item.id) != selected_id {
                item.cached_tensor = None;
                item.cached_edge_tensor = None;
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
                    let recipe = item.applied_recipe.clone();
                    if let Some(back) = item.history.back_mut() {
                        back.cleanup();
                        *back = HistoryEntry::new(rgba, recipe);
                    } else {
                        item.history.push_back(HistoryEntry::new(rgba, recipe));
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
            if let Some(entry) = self.batch_items[idx].history.back() {
                // Peek at the latest history entry — decompress without removing
                let restored = match &entry.slot {
                    HistorySlot::InMemory(rgba) => Some(rgba.clone()),
                    HistorySlot::Compressed(ce) => {
                        super::history_disk::decompress_from_ram(ce)
                            .ok()
                            .map(|img| Arc::new(img))
                    }
                    HistorySlot::OnDisk(de) => {
                        super::history_disk::read_history(de)
                            .ok()
                            .map(|img| Arc::new(img))
                    }
                };
                let recipe = entry.recipe.clone();
                self.batch_items[idx].result_rgba = restored;
                self.batch_items[idx].applied_recipe = recipe;
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
                    None, self.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }

        if self.batch_items[idx].result_texture.is_none()
            && !self.batch_items[idx].result_tex_pending
        {
            if let Some(rgba) = self.batch_items[idx].result_rgba.clone() {
                let switch = self.result_switch_id;
                let bg = self.batch_items[idx].settings.bg_rgb();
                self.batch_items[idx].result_tex_pending = true;
                Self::spawn_tex_prep(
                    rgba, item_id, format!("result_{item_id}_{switch}"), true,
                    bg, self.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }

        // Sync app-level state for canvas rendering.
        // Keep the previous texture visible until the new item's texture is ready
        // (avoids a blank flash on sidebar click, especially for lazy-decoded items).
        let item = &self.batch_items[idx];
        if item.source_texture.is_some() {
            self.source_texture = item.source_texture.clone();
        }
        // Only update dimensions/filename when the item actually has something to show,
        // or when switching away from a completely different state.
        self.loaded_filename = Some(item.filename.clone());
        self.image_dimensions = Some(item.dimensions);
        self.show_original = false;

        let item_status = item.status.clone();
        let result_texture = item.result_texture.clone();
        let result_rgba = item.result_rgba.clone();
        match item_status {
            BatchStatus::Done => {
                if item.result_texture.is_some() {
                    self.result_texture = result_texture;
                }
                if item.result_rgba.is_some() {
                    self.result_rgba = result_rgba;
                }
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
        bg_apply: Option<[u8; 3]>,
        tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
        ctx: egui::Context,
    ) {
        std::thread::spawn(move || {
            let (w, h) = (rgba.width(), rgba.height());
            let ci = if let Some(bg) = bg_apply {
                // Background compositing runs off the UI thread: clone the
                // RGBA (~48 MB for 4000×3000) and the per-pixel blend both
                // happen here instead of blocking the canvas.
                let mut composed = (*rgba).clone();
                prunr_core::apply_background_color(&mut composed, bg);
                egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    composed.as_flat_samples().as_slice(),
                )
            } else {
                egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    rgba.as_flat_samples().as_slice(),
                )
            };
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
                            ProgressStage::LoadingModel => {
                                if cfg!(target_os = "macos") {
                                    "Loading model (first run may take a few minutes)...".into()
                                } else {
                                    "Loading model...".into()
                                }
                            }
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
                WorkerResult::BatchItemDone { item_id, result, tensor_cache, edge_cache } => {
                    let is_selected = self.selected_item()
                        .map_or(false, |b| b.id == item_id);
                    // Fall back to the target item's own settings if dispatch_recipe
                    // wasn't snapshot (shouldn't happen in normal flow; defensive).
                    let recipe_snapshot = self.dispatch_recipe.clone().unwrap_or_else(|| {
                        let model: prunr_core::ModelKind = self.settings.model.into();
                        let chain = self.settings.chain_mode;
                        self.batch_items.iter()
                            .find(|b| b.id == item_id)
                            .map(|b| b.settings.current_recipe(model, chain))
                            .unwrap_or_else(|| self.settings.item_defaults.current_recipe(model, chain))
                    });
                    if let Some(item) = self.batch_items.iter_mut().find(|b| b.id == item_id) {
                        // Skip results for items that were already cancelled (reset to Pending)
                        if item.status != BatchStatus::Processing {
                            continue;
                        }
                        match result {
                            Ok(pr) => {
                                item.result_rgba = Some(Arc::new(pr.rgba_image));
                                item.status = BatchStatus::Done;
                                // Store recipe + compressed tensors for future tier routing.
                                // Both caches are zstd-compressed on the bridge thread before
                                // arriving here; storing is a zero-cost move.
                                item.applied_recipe = Some(recipe_snapshot);
                                item.cached_tensor = tensor_cache
                                    .and_then(super::worker::CompressedTensor::from_raw);
                                item.cached_edge_tensor = edge_cache
                                    .and_then(super::worker::CompressedTensor::from_raw);
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
                                // Tier 2 reruns report empty active_provider (no inference).
                                // Only update backend label on Tier 1 results that actually ran inference.
                                if !pr.active_provider.is_empty() {
                                    let backend_changed = self.settings.active_backend != pr.active_provider;
                                    self.settings.active_backend = pr.active_provider;
                                    if backend_changed {
                                        self.settings.parallel_jobs = self.settings.default_jobs();
                                    }
                                }
                            }
                            Err(e) => {
                                // Clear recipe + tensors so retry runs a fresh Tier 1
                                // (otherwise resolve_tier might return Skip for an errored item).
                                item.status = BatchStatus::Error(e);
                                item.cached_tensor = None;
                                item.cached_edge_tensor = None;
                                item.applied_recipe = None;
                            }
                        }
                    }
                    // Update progress info — count only items involved in this batch
                    let done = self.batch_items.iter().filter(|i| i.status == BatchStatus::Done).count();
                    let processing = self.batch_items.iter().filter(|i| i.status == BatchStatus::Processing).count();
                    let errored = self.batch_items.iter().filter(|i| matches!(i.status, BatchStatus::Error(_))).count();
                    let batch_total = done + processing + errored;
                    if processing > 0 {
                        self.status.stage = format!("Processing {done}/{batch_total}");
                    } else {
                        self.status.stage = "Finishing up".to_string();
                    }
                    self.status.pct = done as f32 / batch_total.max(1) as f32;

                    if is_selected {
                        self.result_switch_id += 1;
                        self.sync_selected_batch_textures(ctx);
                    }

                    // Memory admission: release budget and admit next items
                    self.admission_release_and_admit(item_id);

                    // Enforce tensor cache budget (evict oldest when over 512 MB)
                    self.enforce_tensor_budget();

                    // Under memory pressure: demote history to disk + evict tensors
                    if super::memory::under_memory_pressure() {
                        self.demote_history_to_disk();
                        self.evict_all_tensors();
                    }
                }
                WorkerResult::BatchComplete => {
                    self.dispatch_recipe = None;
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
                    self.dispatch_recipe = None;
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
                if self.settings.auto_process_on_import && self.next_batch_id > id_floor {
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
        let mut toggle_adjustments = false;
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
                // H = toggle sidebar, Shift+H = toggle adjustments toolbar.
                // Tab stays reserved for egui's focus traversal (accessibility).
                // Kept Tab as a fallback for v1 muscle memory — will be removed
                // when Settings → Hotkeys rebinding UI lands (Phase 5).
                if i.key_pressed(Key::H) {
                    if i.modifiers.shift {
                        toggle_adjustments = true;
                    } else {
                        toggle_sidebar = true;
                    }
                }
                if i.key_pressed(Key::Tab) && !i.modifiers.shift {
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
            self.close_settings(ctx);
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
                self.close_settings(ctx);
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
            self.zoom_state.reset();
            self.sync_selected_batch_textures(ctx);
            self.show_original = false;
        }
        if nav_next && !self.batch_items.is_empty() {
            self.selected_batch_index = (self.selected_batch_index + 1) % self.batch_items.len();
            self.zoom_state.reset();
            self.sync_selected_batch_textures(ctx);
            self.show_original = false;
        }
        if toggle_sidebar {
            self.sidebar_hidden = !self.sidebar_hidden;
        }
        if toggle_adjustments {
            self.adjustments_hidden = !self.adjustments_hidden;
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
            // Re-trigger fit zoom when a freshly decoded image arrives
            // (the previous pending_fit_zoom may have been consumed by the old texture)
            self.zoom_state.pending_fit_zoom = true;
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
            if self.settings.auto_process_on_import && self.next_batch_id > id_floor {
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
        // Phase 3: dispatch any debounced previews + apply completed ones
        // before we render. Tick returns a hint of how long to wait before
        // the next scheduled dispatch, so we schedule a repaint accordingly.
        self.pump_live_preview(ui.ctx());

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

        // Row 2 + 3: persistent adjustments toolbar. Only renders when the
        // batch has an item to bind to. Shift+H hides it; `auto_hide_adjustments`
        // hides it when the cursor is below a peek zone at the top of the window.
        //
        // Peek zone = first ~(TOOLBAR_HEIGHT + 60px) of the window vertically.
        // Toolbar stays visible while any chip/combo popup is open so that
        // the user interacting with a popover doesn't have the toolbar yanked
        // out from under their cursor.
        let show_adjustments = if self.adjustments_hidden || self.batch_items.is_empty() {
            false
        } else if self.settings.auto_hide_adjustments {
            let screen_rect = ui.ctx().content_rect();
            // Peek zone covers the main toolbar + ~half the adjustments toolbar
            // height. Generous enough to catch the user heading up toward the
            // chips, tight enough not to trigger on ordinary canvas work.
            let peek_zone = egui::Rect::from_min_size(
                screen_rect.min,
                egui::vec2(screen_rect.width(), theme::TOOLBAR_HEIGHT + 32.0),
            );
            let hover_in_peek = ui
                .ctx()
                .input(|i| i.pointer.hover_pos().is_some_and(|p| peek_zone.contains(p)));
            #[allow(deprecated)]
            let popup_open = ui.ctx().memory(|m| m.any_popup_open());
            hover_in_peek || popup_open
        } else {
            true
        };
        if show_adjustments {
            let line_mode = self.batch_items
                .get(self.selected_batch_index.min(self.batch_items.len() - 1))
                .map(|i| i.settings.line_mode)
                .unwrap_or(prunr_core::LineMode::Off);
            let height = if line_mode == prunr_core::LineMode::Off {
                chip::CHIP_HEIGHT + theme::SPACE_SM * 2.0
            } else {
                chip::CHIP_HEIGHT * 2.0 + theme::SPACE_XS + theme::SPACE_SM * 2.0
            };
            let mut bg_changed = false;
            let mut toolbar_change = adjustments_toolbar::ToolbarChange::default();
            let mut new_applied_preset: Option<String> = None;
            let is_processing = self.state == AppState::Processing;
            egui::Panel::top("adjustments_toolbar")
                .exact_size(height)
                .frame(panel_frame)
                .show_inside(ui, |ui| {
                    let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                    let applied_preset = self.batch_items[idx].applied_preset.clone();
                    let settings_ref: &mut crate::gui::settings::Settings = &mut self.settings;
                    let item_settings_ref: &mut crate::gui::item_settings::ItemSettings =
                        &mut self.batch_items[idx].settings;
                    toolbar_change = adjustments_toolbar::render(
                        ui,
                        item_settings_ref,
                        settings_ref,
                        &applied_preset,
                        &mut new_applied_preset,
                        is_processing,
                    );
                    // bg is applied at display-time, not via Tier 2 — rebuild
                    // texture immediately. Phase 4 moves this to GPU-side fill.
                    bg_changed = toolbar_change.bg;
                });
            if let Some(name) = new_applied_preset {
                let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                self.batch_items[idx].applied_preset = name;
            }
            // Model swap: persist the new selection, show toast, invalidate caches.
            if toolbar_change.model_changed {
                self.settings.save();
                self.toasts.info(format!(
                    "{} loaded",
                    crate::gui::views::model_name(self.settings.model),
                ));
            }
            // Granular cache invalidation — only clear the tensors whose
            // INPUT actually changed. Preset applies that keep line_mode the
            // same leave the edge cache valid (line_strength tweaks can still
            // live-preview). Model swaps clear only the seg cache (edge
            // tensor is model-independent).
            if toolbar_change.seg_cache_invalid || toolbar_change.edge_cache_invalid {
                let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                if toolbar_change.seg_cache_invalid {
                    self.batch_items[idx].cached_tensor = None;
                }
                if toolbar_change.edge_cache_invalid {
                    self.batch_items[idx].cached_edge_tensor = None;
                }
                // Live preview for the newly-invalidated tier can no longer run
                // against the cache — user needs to Process to repopulate.
                // Only hint on preset apply (for chip-level edits the user
                // already knows they tweaked a cross-tier setting).
                if toolbar_change.preset_applied
                    && self.batch_items[idx].status == BatchStatus::Done
                {
                    self.toasts.info("Preset applied — click Process to see the new result.");
                }
            }
            // Register mask / edge tweaks with the live preview dispatcher.
            // `mark_tweak` debounces; `flush` fires immediately when a chip
            // signals the edit has settled (slider released, toggle flipped,
            // color picked). toolbar_change.commit is OR-ed across all chips
            // touched this frame, so toggles + color picks always commit now
            // and mid-slider-drag changes debounce.
            if self.settings.live_preview && (toolbar_change.mask || toolbar_change.edge) {
                let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                let item_id = self.batch_items[idx].id;
                let kind = if toolbar_change.mask {
                    crate::gui::live_preview::PreviewKind::Mask
                } else {
                    crate::gui::live_preview::PreviewKind::Edge
                };
                self.live_preview.mark_tweak(item_id, kind);
                if toolbar_change.commit {
                    self.live_preview.flush(item_id);
                    // Immediate dispatch — no debounce wait needed.
                    ui.ctx().request_repaint();
                } else {
                    // Wake up after the debounce elapses so tick() can dispatch.
                    ui.ctx().request_repaint_after(
                        crate::gui::live_preview::DEBOUNCE,
                    );
                }
            }
            if bg_changed {
                let idx = self.selected_batch_index.min(self.batch_items.len() - 1);
                self.batch_items[idx].result_texture = None;
                self.batch_items[idx].result_tex_pending = false;
                // Thumbnail bakes bg in during generation; needs regeneration
                // so the sidebar preview matches the canvas.
                self.batch_items[idx].thumb_texture = None;
                self.batch_items[idx].thumb_pending = false;
                self.result_texture = None;
                self.result_switch_id += 1;
                self.sync_selected_batch_textures(ui.ctx());
            }
        }

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
