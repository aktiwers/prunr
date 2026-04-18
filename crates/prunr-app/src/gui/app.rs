use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

use egui::{Key, ViewportCommand};

use prunr_core::ProgressStage;
use super::drag_export_state::DragExportState;
use super::history_manager::{HistoryDir, HistoryManager};
use super::item::{BatchItem, BatchStatus, HistoryEntry, HistorySlot, ImageSource, PresetSnapshot};
use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{adjustments_toolbar, canvas, chip, cli_help, settings, shortcuts, sidebar, statusbar, toolbar};

pub struct PrunrApp {
    // State
    pub(crate) state: AppState,
    pub(crate) loaded_filename: Option<String>,
    /// Directory of the most recently opened file (for save dialog default)
    pub(crate) last_open_dir: Option<std::path::PathBuf>,
    pub(crate) image_dimensions: Option<(u32, u32)>,

    /// Processing pipeline — worker channels, admission, live preview, dispatch state.
    pub(crate) processor: super::processor::Processor,

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

    // Batch state — items, selection, lifecycle, memory, textures, bg_io
    pub(crate) batch: super::batch_manager::BatchManager,
    /// User explicitly hid the sidebar via Tab / configured hotkey.
    pub(crate) sidebar_hidden: bool,
    /// User explicitly hid the adjustments toolbar (rows 2 + 3) via Shift+H.
    pub(crate) adjustments_hidden: bool,

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
    /// Toast notification system
    pub(crate) toasts: egui_notify::Toasts,

    // ── Drag-out (OS drag to external apps) ────────────────────────────────
    pub(crate) drag_export: super::drag_export_state::DragExportState,

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

        // Housekeeping: clean up stale temp files from prior sessions.
        super::drag_export::cleanup_stale();
        super::history_disk::cleanup_stale();

        let mut settings = Settings::load();
        settings.active_backend = prunr_core::OrtEngine::detect_active_provider();

        // Subprocess worker: inference runs in a child process for OOM isolation.
        // No prewarm needed — the subprocess creates its own engine pool.
        let (worker_tx, worker_rx) = spawn_worker(worker_ctx);

        Self::init_state(settings, clipboard, worker_tx, worker_rx)
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        Self::init_state(Settings::default(), None, worker_tx, worker_rx)
    }

    /// Shared field-init for both `new` and `new_for_test`. Clipboard and
    /// worker channels are the only inputs that differ between runtime and test.
    fn init_state(
        settings: Settings,
        clipboard: Option<arboard::Clipboard>,
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        Self {
            state: AppState::Empty,
            loaded_filename: None,
            last_open_dir: None,
            image_dimensions: None,
            processor: super::processor::Processor::new(worker_tx, worker_rx),
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
            batch: super::batch_manager::BatchManager::new(),
            sidebar_hidden: false,
            adjustments_hidden: false,
            show_settings: false,
            settings_opened_at: 0.0,
            settings,
            canvas_switch_id: 0,
            result_switch_id: 0,
            pending_batch_sync: false,
            pending_open_dialog: false,
            toasts: egui_notify::Toasts::default()
                    .with_anchor(egui_notify::Anchor::BottomLeft)
                    .with_margin(egui::vec2(theme::SPACE_SM, theme::STATUS_BAR_HEIGHT + theme::SPACE_SM)),
            drag_export: super::drag_export_state::DragExportState::new(),
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
        self.batch.selected_index = 0;
    }

    /// Sync after batch modification — clamp index and refresh canvas.
    fn sync_after_batch_change(&mut self) {
        if self.batch.items.is_empty() {
            self.clear_to_empty();
        } else {
            self.batch.selected_index = self.batch.selected_index.min(self.batch.items.len() - 1);
            self.pending_batch_sync = true;
        }
    }

    /// Core image loading: creates a BatchItem from a source + dimensions.
    fn load_image_source(&mut self, source: ImageSource, dims: (u32, u32), name: String) {
        let id = self.batch.next_id;
        self.batch.next_id += 1;
        let do_decode = matches!(&source, ImageSource::Bytes(_)); // decode eagerly for in-memory
        let new_settings = self.settings.item_defaults_for_new_item();
        self.batch.items.push(BatchItem::new(
            id,
            name.clone(),
            source,
            dims,
            new_settings,
            self.settings.default_preset.clone(),
        ));
        self.batch.selected_index = self.batch.items.len() - 1;
        if do_decode {
            // invariant: push occurred above, so batch.items is non-empty.
            if let Ok(bytes) = self.batch.items.last().unwrap().source.load_bytes() {
                self.batch.request_decode_bytes(id, bytes);
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
    }

    pub fn handle_remove_bg(&mut self) {
        let has_selected = self.batch.items.iter().any(|i| i.selected);
        if has_selected {
            self.process_items(|item| item.selected);
        } else {
            // Process current image via batch path (item_id tracking ensures
            // result goes to correct image even if user switches during processing)
            let idx = self.batch.selected_index.min(self.batch.items.len().saturating_sub(1));
            if let Some(target_id) = self.batch.items.get(idx).map(|b| b.id) {
                self.process_items(|item| item.id == target_id);
            }
        }
    }

    pub(crate) fn close_settings(&mut self, _ctx: &egui::Context) {
        self.show_settings = false;
        self.settings.save();
        self.toasts.info("Settings saved");
    }

    pub(crate) fn any_modal_open(&self) -> bool {
        self.show_settings || self.show_shortcuts || self.show_cli_help
    }

    /// Undo background removal on selected items (or current item if none selected).
    /// Reverts Done/Error items back to Pending, clearing their results.
    fn handle_undo(&mut self, ctx: &egui::Context) {
        let has_selected = self.batch.items.iter().any(|i| i.selected);
        let current_id = self.batch.selected_item().map(|b| b.id);
        let mut undone = 0u32;
        for item in &mut self.batch.items {
            let target = if has_selected { item.selected } else { Some(item.id) == current_id };
            if target && HistoryManager::undo_result(item) {
                item.reset_result_caches();
                // Undo also needs the SOURCE view rebuilt — the canvas may
                // now show the unprocessed source instead of a result.
                item.source_texture = None;
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
        let has_selected = self.batch.items.iter().any(|i| i.selected);
        let current_id = self.batch.selected_item().map(|b| b.id);
        let mut redone = 0u32;
        for item in &mut self.batch.items {
            let target = if has_selected { item.selected } else { Some(item.id) == current_id };
            if target && HistoryManager::redo_result(item) {
                item.reset_result_caches();
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

    /// Preset undo/redo: rolls back (or re-applies) a preset swap on the
    /// current image. Does NOT touch the image-result history — that stays on
    /// Ctrl+Z. Kicks an auto-reprocess on a Done item so the restored settings
    /// produce a fresh result (same path a live preset apply would take).
    fn swap_preset_history(&mut self, dir: HistoryDir) {
        if self.batch.items.is_empty() { return; }
        let idx = self.batch.selected_index.min(self.batch.items.len() - 1);
        let item = &mut self.batch.items[idx];
        if !HistoryManager::swap_preset(item, dir) { return; }
        let target_id = item.id;
        let should_reprocess = item.status == BatchStatus::Done;
        if should_reprocess {
            self.process_items(|i| i.id == target_id);
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    /// Uses tier routing: compares each item's applied_recipe against current
    /// settings to determine the minimum work needed (skip / mask rerun / full).
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        let chain = self.settings.chain_mode;
        let model: prunr_core::ModelKind = self.settings.model.into();

        let candidate_ids: HashSet<u64> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }

        let tiers = self.classify_candidates(&candidate_ids, model, chain);
        let process_count = tiers.tier1.len() + tiers.tier2.len();
        self.notify_skip(tiers.skip_count, process_count);
        if process_count == 0 { return; }

        self.seed_history_for_reprocess(&tiers.all_process_ids(), chain);

        let tier2_work = self.build_tier2_work(&tiers.tier2);

        let jobs = self.settings.parallel_jobs.min(super::memory::safe_max_jobs(model));
        if tiers.tier1.len() > 1 {
            self.dispatch_with_admission(&tiers.tier1, tier2_work, model, jobs, chain);
        } else {
            self.dispatch_small_batch(&tiers.tier1, tier2_work, model, jobs, chain);
        }
    }

    /// Classify each candidate into Tier 1 (full pipeline), Tier 2 (mask
    /// rerun from cached tensor), or Skip (already up to date).
    /// Mutates items in-place: invalidates stale caches, syncs composite-only
    /// recipe changes, and never downgrades Tier 2 items without a tensor.
    fn classify_candidates(
        &mut self,
        candidate_ids: &HashSet<u64>,
        model: prunr_core::ModelKind,
        chain: bool,
    ) -> ClassifiedTiers {
        use prunr_core::RequiredTier;
        let mut tiers = ClassifiedTiers::default();

        for item in &mut self.batch.items {
            if !candidate_ids.contains(&item.id) { continue; }

            // Never-processed items always need the full pipeline.
            let Some(ref old_recipe) = item.applied_recipe else {
                tiers.tier1.insert(item.id);
                continue;
            };

            // Chain mode with an existing result feeds the output back in,
            // so the input changes each time → always full.
            if chain && item.result_rgba.is_some() {
                item.cached_tensor = None;
                tiers.tier1.insert(item.id);
                continue;
            }

            let current_recipe = item.settings.current_recipe(model, chain);
            match prunr_core::resolve_tier(old_recipe, &current_recipe) {
                RequiredTier::Skip | RequiredTier::CompositeOnly => {
                    // CompositeOnly (bg_color) is handled at display/export time;
                    // sync the stored composite so status reads stay accurate.
                    if let Some(ref mut recipe) = item.applied_recipe {
                        recipe.composite = current_recipe.composite.clone();
                    }
                    tiers.skip_count += 1;
                }
                RequiredTier::MaskRerun => {
                    if item.cached_tensor.is_some() {
                        tiers.tier2.insert(item.id);
                    } else {
                        tiers.tier1.insert(item.id);
                    }
                }
                RequiredTier::EdgeRerun => {
                    // Edge rerun via the subprocess isn't wired for batch
                    // dispatch yet — fall through to a full pipeline run.
                    // (Live-preview edge reruns DO work in-process via
                    // `pump_live_preview` + `finalize_edges`; this branch is
                    // for the explicit Process click path only.)
                    item.invalidate_edge_cache();
                    tiers.tier1.insert(item.id);
                }
                RequiredTier::FullPipeline => {
                    // Model / chain / mode changed — both caches invalid.
                    item.cached_tensor = None;
                    item.invalidate_edge_cache();
                    tiers.tier1.insert(item.id);
                }
            }
        }
        tiers
    }

    /// Tell the user when some items were skipped. Three shapes:
    /// nothing-skipped (silent), all-skipped, partially-skipped.
    fn notify_skip(&mut self, skipped: usize, processing: usize) {
        if skipped == 0 { return; }
        let msg = if processing == 0 {
            if skipped == 1 {
                "Already up to date".to_string()
            } else {
                format!("{skipped} images already up to date")
            }
        } else {
            format!("{skipped} up to date, processing {processing}")
        };
        self.toasts.info(msg);
    }

    fn seed_history_for_reprocess(&mut self, process_ids: &HashSet<u64>, chain: bool) {
        let max_depth = self.settings.history_depth;
        for item in &mut self.batch.items {
            if !process_ids.contains(&item.id) { continue; }
            HistoryManager::seed_with_source(item);
            let was_done = item.status == BatchStatus::Done;
            HistoryManager::archive_current_result(item, max_depth, chain);
            if was_done {
                // Rebuild textures from the (possibly new) result. In chain
                // mode, result_rgba stays populated for the next chain step,
                // so keep result_texture; otherwise drop it.
                if !chain {
                    item.result_texture = None;
                }
                item.thumb_texture = None;
                item.thumb_pending = false;
                item.source_tex_pending = false;
                item.result_tex_pending = false;
            }
        }
    }

    fn build_tier2_work(&mut self, tier2_ids: &HashSet<u64>) -> Vec<super::worker::Tier2WorkItem> {
        let mut out = Vec::new();
        for item in &mut self.batch.items {
            if !tier2_ids.contains(&item.id) { continue; }
            let Some(ref ct) = item.cached_tensor else { continue };
            let tensor_data = ct.decompress();
            let mask = item.settings.mask_settings();
            match (tensor_data, item.source.load_bytes()) {
                (Some(data), Ok(bytes)) => {
                    out.push(super::worker::Tier2WorkItem {
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
        out
    }

    /// Dispatch with streaming admission — used when >1 Tier 1 items so the
    /// batch doesn't blow past memory limits. Admits what fits now; the
    /// worker bridge receives more items via `admission_tx` as earlier items
    /// complete and free memory.
    fn dispatch_with_admission(
        &mut self,
        tier1_ids: &HashSet<u64>,
        tier2_work: Vec<super::worker::Tier2WorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        chain: bool,
    ) {
        use super::memory::{AdmissionController, ImageMemCost};

        let mut ctrl = AdmissionController::new(model, jobs);
        let costs: Vec<ImageMemCost> = self.batch.items.iter()
            .filter(|i| tier1_ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(i.id, i.dimensions, i.source.estimated_size()))
            .collect();
        ctrl.enqueue(costs);

        let mut initial_items = Vec::new();
        while let Some(admitted_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.items.iter_mut().find(|b| b.id == admitted_id) {
                if let Ok(bytes) = item.source.load_bytes() {
                    let chain_input = if chain { item.result_rgba.clone() } else { None };
                    initial_items.push((item.id, bytes, chain_input));
                    item.status = BatchStatus::Processing;
                }
            }
        }

        for item in &mut self.batch.items {
            if tier1_ids.contains(&item.id) && item.status != BatchStatus::Processing {
                item.status = BatchStatus::Pending;
            }
        }

        // All Tier 1 items failed load_bytes AND no Tier 2 work — skip dispatch.
        if initial_items.is_empty() && tier2_work.is_empty() { return; }

        let (atx, arx) = mpsc::channel();
        self.processor.admission = Some(ctrl);
        self.processor.admission_tx = Some(atx);
        self.dispatch_batch(initial_items, tier2_work, model, jobs, Some(arx));
    }

    /// Single-item or small-batch fast path — skip admission entirely.
    fn dispatch_small_batch(
        &mut self,
        tier1_ids: &HashSet<u64>,
        tier2_work: Vec<super::worker::Tier2WorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        chain: bool,
    ) {
        let items: Vec<_> = self.batch.items.iter_mut()
            .filter(|i| tier1_ids.contains(&i.id))
            .filter_map(|i| {
                let bytes = i.source.load_bytes().ok()?;
                let chain_input = if chain { i.result_rgba.clone() } else { None };
                i.status = BatchStatus::Processing;
                Some((i.id, bytes, chain_input))
            })
            .collect();

        // If all prep failed (Tier 2 decompress + Tier 1 load_bytes), don't
        // dispatch an empty batch — the subprocess would spin up for nothing.
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
        self.processor.cancel_flag.store(false, Ordering::Release);
        self.state = AppState::Processing;

        // Use the currently-viewed item's settings for the batch — matches
        // "what you see is what you process." Fallback to factory defaults
        // only if the batch is somehow empty at dispatch time (defensive).
        let idx = self.batch.selected_index.min(self.batch.items.len().saturating_sub(1));
        let current_settings = self.batch.items.get(idx)
            .map(|b| b.settings)
            .unwrap_or_default();

        // Broadcast: every item about to be processed inherits current.settings
        // so their `applied_recipe` ends up consistent with what ran.
        let process_ids: std::collections::HashSet<u64> = items.iter()
            .map(|wi| wi.0)
            .chain(tier2_items.iter().map(|ti| ti.item_id))
            .collect();
        for item in &mut self.batch.items {
            if process_ids.contains(&item.id) {
                item.settings = current_settings;
            }
        }

        self.processor.dispatch_recipe = Some(current_settings.current_recipe(model, self.settings.chain_mode));

        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.processor.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            tier2_items,
            config: super::worker::ProcessingConfig {
                model,
                jobs,
                mask: current_settings.mask_settings(),
                force_cpu: self.settings.force_cpu,
                line_mode: current_settings.line_mode,
                edge: current_settings.edge_settings(),
            },
            cancel: self.processor.cancel_flag.clone(),
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
        let has_selection = self.batch.items.iter()
            .any(|i| i.selected && i.status == BatchStatus::Done && i.result_rgba.is_some());
        if has_selection {
            self.save_selected_to_folder();
        } else {
            self.save_current_to_file();
        }
    }

    /// No checkboxes selected — save just the currently-viewed result via a
    /// save-as dialog. Suggests a `<stem>-nobg.png` name based on the source.
    fn save_current_to_file(&mut self) {
        let Some(rgba) = self.result_rgba.as_ref() else { return };
        let default_name = self.loaded_filename.as_deref()
            .and_then(|name| Path::new(name).file_stem()?.to_str())
            .map(|stem| format!("{stem}-nobg.png"))
            .unwrap_or_else(|| "result-nobg.png".to_string());
        let Some(path) = self.save_dialog()
            .add_filter("PNG Image", &["png"])
            .set_file_name(&default_name)
            .set_title("Save PNG")
            .save_file() else { return };
        let bg = self.batch.selected_item().and_then(|i| i.settings.bg_rgb());
        let rgba = Self::apply_bg_for_export(rgba, bg);
        let tx = self.batch.bg_io.save_done_tx.clone();
        self.toasts.info("Saving...");
        spawn_save_single(path, rgba, tx);
    }

    /// Save one specific batch item (by index) via a save-as dialog. Used by
    /// the sidebar's per-row save button, which needs a non-selection-based
    /// entry point into the same encode-on-background pipeline as
    /// `save_current_to_file`.
    pub(crate) fn save_item_to_file(&mut self, idx: usize) {
        let Some(item) = self.batch.items.get(idx) else { return };
        let Some(rgba) = item.result_rgba.as_ref() else { return };
        let stem = Path::new(&item.filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let default_name = format!("{stem}-nobg.png");
        let bg = item.settings.bg_rgb();
        let Some(path) = self.save_dialog()
            .add_filter("PNG Image", &["png"])
            .set_file_name(&default_name)
            .set_title("Save PNG")
            .save_file() else { return };
        let rgba = Self::apply_bg_for_export(rgba, bg);
        let tx = self.batch.bg_io.save_done_tx.clone();
        self.toasts.info("Saving...");
        spawn_save_single(path, rgba, tx);
    }

    /// One or more sidebar checkboxes selected — pick a folder, write each
    /// as `<source-stem>-nobg.png`. Encode + write run on a background thread.
    fn save_selected_to_folder(&mut self) {
        let items: Vec<(String, Arc<image::RgbaImage>)> = self.batch.items.iter()
            .filter(|i| i.selected && i.status == BatchStatus::Done && i.result_rgba.is_some())
            .filter_map(|item| {
                let rgba = item.result_rgba.as_ref()?;
                Some((item.filename.clone(), Self::apply_bg_for_export(rgba, item.settings.bg_rgb())))
            })
            .collect();
        let Some(folder) = self.save_dialog()
            .set_title("Save Selected \u{2014} Choose Folder")
            .pick_folder() else { return };
        let count = items.len();
        self.toasts.info(format!("Saving {count} image(s)..."));
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_batch(folder, items, tx);
    }

    pub fn remove_selected(&mut self) {
        let count = self.batch.items.iter().filter(|i| i.selected).count();
        self.batch.items.retain(|item| !item.selected);
        self.sync_after_batch_change();
        if count > 0 {
            self.toasts.info(format!("Removed {count} image(s)"));
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
            if let Some(item) = self.batch.items.iter().find(|b| b.id == *id) {
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

        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            DragExportState::reset(&self.drag_export.active, &self.drag_export.items);
            if !self.drag_export.linux_notified {
                self.drag_export.linux_notified = true;
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
            .batch
            .items
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
        let bg = self.batch.selected_item().and_then(|i| i.settings.bg_rgb());
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
        self.processor.cancel_flag.store(true, Ordering::Release);
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
        let id = self.batch.next_id;
        self.batch.next_id += 1;

        let new_settings = self.settings.item_defaults_for_new_item();
        self.batch.items.push(BatchItem::new(
            id,
            filename,
            source,
            dims,
            new_settings,
            self.settings.default_preset.clone(),
        ));

        if self.state == AppState::Empty {
            self.state = AppState::Loaded;
        }
        // Fit-to-window on first import so images open at a sensible size
        // (matching Ctrl+0). Any subsequent image change also fits — the reset
        // happens via zoom_state.reset() when the user navigates away.
        self.zoom_state.pending_fit_zoom = true;
        self.pending_batch_sync = true;
    }

    pub fn remove_batch_item(&mut self, idx: usize) {
        if idx >= self.batch.items.len() { return; }
        let item = self.batch.items.remove(idx);
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
        let Some(ref mut ctrl) = self.processor.admission else { return; };
        ctrl.release(completed_id);

        let chain = self.settings.chain_mode;
        while let Some(next_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.items.iter_mut().find(|b| b.id == next_id) {
                let Ok(bytes) = item.source.load_bytes() else { continue; };
                let chain_input = if chain { item.result_rgba.clone() } else { None };
                let tuple = (next_id, bytes, chain_input);
                item.status = BatchStatus::Processing;

                if let Some(ref tx) = self.processor.admission_tx {
                    if tx.send(tuple).is_err() {
                        break; // worker gone
                    }
                }
            }
        }

        // If all items admitted and released, drop the sender to signal worker.
        if ctrl.is_complete() {
            self.processor.clear_admission();
        }
    }

    /// Demote Tier 2 (compressed RAM) history to Tier 3 (disk) under memory pressure.
    fn demote_history_to_disk(&mut self) {
        for item in &mut self.batch.items {
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
        // Dispatch phase: tick() invokes the closure for each item whose
        // debounce expired this frame. The closure captures `&self.batch.items`
        // for read-only access; it doesn't need `&mut self`.
        let batch_items = &self.batch.items;
        let wait = self.processor.live_preview.tick(|id, kind| {
            Self::build_preview_inputs(batch_items, id, kind)
        });
        // If a future dispatch is waiting, schedule a repaint when the
        // debounce elapses so tick() can fire on its own.
        if let Some(w) = wait {
            ctx.request_repaint_after(w);
        }

        let results = self.processor.live_preview.drain_results();
        if !results.is_empty() {
            self.apply_completed_previews(ctx, results);
        }
    }

    /// Snapshot the dispatch inputs for one preview job: original RGBA +
    /// decompressed tensors + settings + reusable edge mask if its
    /// line_strength still matches. Returns `None` to abort the dispatch
    /// when a required tensor cache isn't available (user must Process first).
    fn build_preview_inputs(
        items: &[BatchItem],
        id: u64,
        kind: super::live_preview::PreviewKind,
    ) -> Option<super::live_preview::DispatchInputs> {
        use super::live_preview::{DispatchInputs, PreviewKind, decompress_edge, decompress_seg};
        let item = items.iter().find(|b| b.id == id)?;
        let rgba = item.source_rgba.as_ref()?;
        // One clone here — live preview workers need an owned DynamicImage.
        // Clones the ~48 MB RGBA once per dispatch (not per frame); the user
        // has paused typing for 300ms so a single clone is acceptable.
        let original = Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone()));
        let seg_tensor = item.cached_tensor.as_ref().and_then(decompress_seg);
        let edge_tensor = item.cached_edge_tensor.as_ref().and_then(decompress_edge);
        match kind {
            PreviewKind::Mask if seg_tensor.is_none() => return None,
            PreviewKind::Edge if edge_tensor.is_none() => return None,
            _ => {}
        }
        let cached_edge_mask = item.cached_edge_mask.as_ref().and_then(|(m, bits)| {
            (*bits == item.settings.line_strength.to_bits()).then(|| m.clone())
        });
        Some(DispatchInputs {
            kind, original, settings: item.settings,
            seg_tensor, edge_tensor, cached_edge_mask,
        })
    }

    /// Critical: do NOT null `result_texture` here — the old texture must stay
    /// visible until the newly-built one lands via `drain_background_channels`.
    /// Clearing it causes the canvas to flash black for a frame (no texture
    /// to draw → BG_PRIMARY shows). Instead we spawn a tex prep for the new
    /// RGBA directly and let drain swap it in atomically when ready.
    fn apply_completed_previews(
        &mut self,
        ctx: &egui::Context,
        results: Vec<super::live_preview::PreviewResult>,
    ) {
        let tex_prep_tx = self.batch.bg_io.tex_prep_tx.clone();
        for r in results {
            let Some(item) = self.batch.items.iter_mut().find(|b| b.id == r.item_id) else {
                continue;
            };
            let new_rgba = Arc::new(r.rgba);
            item.result_rgba = Some(new_rgba.clone());
            // Mark pending so sync_selected_batch_textures doesn't also spawn
            // its own prep on this same frame.
            item.result_tex_pending = true;
            // Invalidate the sidebar thumbnail — it was built from the previous
            // result_rgba and may carry different line colors. Sidebar's render
            // loop will see `None + !pending` and queue a fresh generation.
            item.thumb_texture = None;
            item.thumb_pending = false;
            if let Some((mask, bits)) = r.new_edge_mask {
                item.cached_edge_mask = Some((mask, bits));
            }
            let item_id = item.id;
            let switch = self.result_switch_id;
            Self::spawn_tex_prep(
                new_rgba, item_id, format!("result_{item_id}_{switch}"),
                true, tex_prep_tx.clone(), ctx.clone(),
            );
        }
        ctx.request_repaint();
    }

    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        self.evict_result_rgba_for_background_items(idx);
        self.restore_selected_result_from_history(idx);
        self.ensure_selected_source_decoded(idx);
        self.request_selected_textures(idx, ctx);
        self.sync_app_state_to_selected_item(idx);
    }

    /// Free full-resolution `result_rgba` for every Done item that isn't the
    /// selected one. The compressed history copy stays, so a later selection
    /// can restore on demand (see `restore_selected_result_from_history`).
    fn evict_result_rgba_for_background_items(&mut self, selected_idx: usize) {
        for (i, item) in self.batch.items.iter_mut().enumerate() {
            if i == selected_idx || item.result_rgba.is_none() || item.status != BatchStatus::Done {
                continue;
            }
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

    /// Restore `result_rgba` for the selected item if it was previously evicted
    /// by the background-eviction pass. Peeks the latest history slot
    /// (non-destructive) and decompresses / reads from disk as needed.
    fn restore_selected_result_from_history(&mut self, idx: usize) {
        if self.batch.items[idx].status != BatchStatus::Done
            || self.batch.items[idx].result_rgba.is_some()
        {
            return;
        }
        let Some(entry) = self.batch.items[idx].history.back() else { return };
        let restored = match &entry.slot {
            HistorySlot::InMemory(rgba) => Some(rgba.clone()),
            HistorySlot::Compressed(ce) => {
                super::history_disk::decompress_from_ram(ce).ok().map(Arc::new)
            }
            HistorySlot::OnDisk(de) => {
                super::history_disk::read_history(de).ok().map(Arc::new)
            }
        };
        let recipe = entry.recipe.clone();
        self.batch.items[idx].result_rgba = restored;
        self.batch.items[idx].applied_recipe = recipe;
    }

    /// Kick off background decode if the selected item has no decoded source
    /// (lazy decode path — applies after sidebar navigation to a not-yet-decoded item).
    fn ensure_selected_source_decoded(&mut self, idx: usize) {
        if self.batch.items[idx].source_rgba.is_some() || self.batch.items[idx].decode_pending {
            return;
        }
        self.batch.items[idx].decode_pending = true;
        let id = self.batch.items[idx].id;
        self.batch.request_decode_source(id, &self.batch.items[idx].source);
    }

    /// Spawn ColorImage preparation threads for whichever textures are missing.
    /// The actual `ctx.load_texture()` call happens in `drain_background_channels`
    /// after the prepared ColorImage arrives on `tex_prep_rx`.
    fn request_selected_textures(&mut self, idx: usize, ctx: &egui::Context) {
        let item_id = self.batch.items[idx].id;

        if self.batch.items[idx].source_texture.is_none()
            && !self.batch.items[idx].source_tex_pending
        {
            if let Some(rgba) = self.batch.items[idx].source_rgba.clone() {
                self.batch.items[idx].source_tex_pending = true;
                Self::spawn_tex_prep(
                    rgba, item_id, format!("source_{item_id}"), false,
                    self.batch.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }

        if self.batch.items[idx].result_texture.is_none()
            && !self.batch.items[idx].result_tex_pending
        {
            if let Some(rgba) = self.batch.items[idx].result_rgba.clone() {
                let switch = self.result_switch_id;
                self.batch.items[idx].result_tex_pending = true;
                Self::spawn_tex_prep(
                    rgba, item_id, format!("result_{item_id}_{switch}"), true,
                    self.batch.bg_io.tex_prep_tx.clone(), ctx.clone(),
                );
            }
        }
    }

    /// Pull the selected item's textures + dimensions + filename onto the
    /// app-level fields the canvas reads. Keeps the previous texture visible
    /// until the new item's is ready (no blank flash on sidebar click).
    fn sync_app_state_to_selected_item(&mut self, idx: usize) {
        let item = &self.batch.items[idx];
        if item.source_texture.is_some() {
            self.source_texture = item.source_texture.clone();
        }
        self.loaded_filename = Some(item.filename.clone());
        self.image_dimensions = Some(item.dimensions);
        self.show_original = false;

        let result_texture = item.result_texture.clone();
        let result_rgba = item.result_rgba.clone();
        match item.status {
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

/// Cap on `WorkerResult` messages drained per frame from `processor.worker_rx`.
/// Keeps the UI responsive during heavy batch processing — remaining messages
/// are picked up on the next frame (request_repaint_after ensures continuity).
const WORKER_POLL_PER_FRAME: usize = 8;

/// Cap on file-load receipts drained per frame from `batch.bg_io.file_load_rx`.
/// Same rationale as `WORKER_POLL_PER_FRAME`.
const FILE_LOAD_DRAIN_PER_FRAME: usize = 5;

/// How often `eframe::App::logic` triggers a sweep of stale on-disk history
/// files (Tier 3). 10 minutes is conservative — short enough that a long
/// session doesn't accumulate, long enough that the sweep cost is invisible.
const HISTORY_CLEANUP_INTERVAL_SECS: u64 = 600;

impl PrunrApp {
    fn poll_worker_results(&mut self, ctx: &egui::Context) {
        for _ in 0..WORKER_POLL_PER_FRAME {
            let Ok(msg) = self.processor.worker_rx.try_recv() else { break };
            match msg {
                WorkerResult::BatchProgress { item_id, stage, pct } => {
                    self.on_batch_progress(item_id, stage, pct);
                }
                WorkerResult::BatchItemDone { item_id, result, tensor_cache, edge_cache } => {
                    self.on_batch_item_done(ctx, item_id, result, tensor_cache, edge_cache);
                }
                WorkerResult::BatchComplete => self.on_batch_complete(),
                WorkerResult::Cancelled => self.on_cancelled(),
                WorkerResult::SubprocessRetry { reduced_jobs, re_queued_count } => {
                    self.on_subprocess_retry(reduced_jobs, re_queued_count);
                }
            }
        }
    }

    fn on_batch_progress(&mut self, item_id: u64, stage: ProgressStage, pct: f32) {
        if !self.batch.is_selected(item_id) {
            return;
        }
        self.status.stage = match stage {
            ProgressStage::LoadingModel => self.loading_model_status_text(item_id),
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

    /// Include which models are loading so a DexiNed-only reload is
    /// distinguishable from a segmentation model load.
    fn loading_model_status_text(&self, item_id: u64) -> String {
        let line_mode = self.batch.items.iter()
            .find(|b| b.id == item_id)
            .map(|b| b.settings.line_mode)
            .unwrap_or(prunr_core::LineMode::Off);
        let seg_name = super::views::model_name(self.settings.model);
        let models = match line_mode {
            prunr_core::LineMode::Off => seg_name.to_string(),
            prunr_core::LineMode::EdgesOnly => "DexiNed".to_string(),
            prunr_core::LineMode::SubjectOutline => format!("{seg_name} + DexiNed"),
        };
        if cfg!(target_os = "macos") {
            format!("Loading {models} (first run may take a few minutes)...")
        } else {
            format!("Loading {models}...")
        }
    }

    fn on_batch_item_done(
        &mut self,
        ctx: &egui::Context,
        item_id: u64,
        result: Result<prunr_core::ProcessResult, String>,
        tensor_cache: Option<super::worker::TensorCache>,
        edge_cache: Option<super::worker::TensorCache>,
    ) {
        let is_selected = self.batch.is_selected(item_id);
        let recipe_snapshot = self.resolved_dispatch_recipe(item_id);

        let Some(item) = self.batch.items.iter_mut().find(|b| b.id == item_id) else { return };
        // Skip results for items that were already cancelled (reset to Pending).
        if item.status != BatchStatus::Processing {
            return;
        }
        let backend_update = item.apply_tier_result(result, tensor_cache, edge_cache, recipe_snapshot, is_selected);

        if let Some(provider) = backend_update {
            let backend_changed = self.settings.active_backend != provider;
            self.settings.active_backend = provider;
            if backend_changed {
                self.settings.parallel_jobs = self.settings.default_jobs();
            }
        }

        self.refresh_batch_progress_status();

        if is_selected {
            self.result_switch_id += 1;
            self.sync_selected_batch_textures(ctx);
        }

        self.admission_release_and_admit(item_id);
        self.batch.enforce_tensor_budget();

        if super::memory::under_memory_pressure() {
            self.demote_history_to_disk();
            self.batch.evict_all_tensors();
        }
    }

    /// Resolve the recipe to stamp on a finished item. Prefers the snapshot
    /// taken at dispatch time; falls back to the item's own settings if no
    /// snapshot is available (defensive — shouldn't happen in normal flow).
    fn resolved_dispatch_recipe(&self, item_id: u64) -> prunr_core::ProcessingRecipe {
        self.processor.dispatch_recipe.clone().unwrap_or_else(|| {
            let model: prunr_core::ModelKind = self.settings.model.into();
            let chain = self.settings.chain_mode;
            self.batch.items.iter()
                .find(|b| b.id == item_id)
                .map(|b| b.settings.current_recipe(model, chain))
                .unwrap_or_else(|| super::item_settings::ItemSettings::default().current_recipe(model, chain))
        })
    }

    fn refresh_batch_progress_status(&mut self) {
        let counts = self.batch.status_counts();
        let total = counts.batch_total();
        self.status.stage = if counts.processing > 0 {
            format!("Processing {}/{total}", counts.done)
        } else {
            "Finishing up".to_string()
        };
        self.status.pct = counts.done as f32 / total.max(1) as f32;
    }

    fn on_batch_complete(&mut self) {
        self.processor.dispatch_recipe = None;
        let counts = self.batch.status_counts();
        let still_processing = counts.processing > 0;
        if counts.errored > 0 {
            let msg = format!("{} image(s) failed to process", counts.errored);
            self.status.text = msg.clone();
            self.toasts.warning(msg);
        } else if !still_processing {
            let msg = format!("All done \u{2014} {} images processed", counts.done);
            self.status.text = msg.clone();
            self.toasts.success(msg);
        }
        // Update app state to match viewed item (textures already synced by BatchItemDone).
        if !still_processing {
            let idx = self.batch.selected_index.min(self.batch.items.len().saturating_sub(1));
            if let Some(item) = self.batch.items.get(idx) {
                match item.status {
                    BatchStatus::Done => self.state = AppState::Done,
                    BatchStatus::Processing => self.state = AppState::Processing,
                    _ => self.state = AppState::Loaded,
                }
            }
        }
    }

    fn on_cancelled(&mut self) {
        self.processor.dispatch_recipe = None;
        if self.state == AppState::Processing {
            self.state = AppState::Loaded;
            self.status.text = "Cancelled".to_string();
        }
        self.processor.clear_admission();
    }

    fn on_subprocess_retry(&mut self, reduced_jobs: usize, re_queued_count: usize) {
        let msg = format!(
            "Memory pressure \u{2014} retrying {re_queued_count} images with {reduced_jobs} parallel jobs"
        );
        self.toasts.warning(msg.clone());
        self.status.text = msg;
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
            DragExportState::reset(&self.drag_export.active, &self.drag_export.items);
            ctx.stop_dragging();
            return;
        }

        // Send file paths for lazy loading (avoids reading all into RAM upfront)
        if !paths.is_empty() {
            let tx = self.batch.bg_io.file_load_tx.clone();
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
                    self.pending_batch_sync = true;
                }
                if self.settings.auto_process_on_import && self.batch.next_id > id_floor {
                    self.process_items(|item| item.id >= id_floor);
                }
            }
        }
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        let intents = collect_shortcut_intents(ctx);
        let copy_requested = std::mem::take(&mut self.pending_copy);
        let pending_open = std::mem::take(&mut self.pending_open_dialog);

        if intents.open_requested || pending_open {
            self.handle_open_dialog();
        }
        if intents.remove_requested && matches!(self.state, AppState::Loaded | AppState::Done) {
            self.handle_remove_bg();
        }
        if intents.save_requested && self.state == AppState::Done {
            self.handle_save_selected();
        }
        if copy_requested && self.state == AppState::Done {
            self.handle_copy();
        }
        if intents.toggle_before_after && self.state == AppState::Done {
            self.show_original = !self.show_original;
        }
        if intents.fit_to_window { self.zoom_state.pending_fit_zoom = true; }
        if intents.actual_size   { self.zoom_state.pending_actual_size = true; }

        if intents.cancel_requested {
            self.apply_cancel_shortcut(ctx);
        }

        if intents.toggle_shortcuts { self.show_shortcuts = !self.show_shortcuts; }
        if intents.toggle_cli_help  { self.show_cli_help  = !self.show_cli_help;  }
        if intents.toggle_settings  { self.toggle_settings_panel(ctx); }

        if intents.nav_prev { self.navigate_batch(ctx, NavDir::Prev); }
        if intents.nav_next { self.navigate_batch(ctx, NavDir::Next); }

        if intents.toggle_sidebar     { self.sidebar_hidden     = !self.sidebar_hidden; }
        if intents.toggle_adjustments { self.adjustments_hidden = !self.adjustments_hidden; }

        if intents.undo_requested        { self.handle_undo(ctx); }
        if intents.redo_requested        { self.handle_redo(ctx); }
        if intents.preset_undo_requested { self.swap_preset_history(HistoryDir::Undo); }
        if intents.preset_redo_requested { self.swap_preset_history(HistoryDir::Redo); }

        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }
    }

    /// Escape dismisses the topmost interruptable state. Priority order:
    /// active processing → open modal (settings / shortcuts / cli help).
    fn apply_cancel_shortcut(&mut self, ctx: &egui::Context) {
        if self.batch.status_counts().processing > 0 {
            self.handle_cancel();
            self.processor.clear_admission();
            for item in &mut self.batch.items {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
        } else if self.state == AppState::Processing {
            self.handle_cancel();
            self.processor.clear_admission();
            self.state = AppState::Loaded;
            self.status.text = "Cancelled".to_string();
        } else if self.show_settings {
            self.close_settings(ctx);
        } else if self.show_shortcuts {
            self.show_shortcuts = false;
        } else if self.show_cli_help {
            self.show_cli_help = false;
        }
    }

    fn toggle_settings_panel(&mut self, ctx: &egui::Context) {
        if self.show_settings {
            self.close_settings(ctx);
        } else {
            self.show_settings = true;
            self.settings_opened_at = ctx.input(|i| i.time);
        }
    }

    fn navigate_batch(&mut self, ctx: &egui::Context, dir: NavDir) {
        let len = self.batch.items.len();
        if len == 0 {
            return;
        }
        self.batch.selected_index = match dir {
            NavDir::Prev if self.batch.selected_index == 0 => len - 1,
            NavDir::Prev => self.batch.selected_index - 1,
            NavDir::Next => (self.batch.selected_index + 1) % len,
        };
        self.zoom_state.reset();
        self.sync_selected_batch_textures(ctx);
        self.show_original = false;
    }

    fn drain_background_channels(&mut self, ctx: &egui::Context) {
        let mut decode_arrived = false;
        while let Ok((item_id, rgba)) = self.batch.bg_io.decode_rx.try_recv() {
            if let Some(item) = self.batch.items.iter_mut().find(|b| b.id == item_id) {
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
        while let Ok((item_id, name, color_image, is_result)) = self.batch.bg_io.tex_prep_rx.try_recv() {
            let tex = ctx.load_texture(name, color_image, egui::TextureOptions::default());
            if let Some(item) = self.batch.items.iter_mut().find(|b| b.id == item_id) {
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
        while let Ok(msg) = self.batch.bg_io.save_done_rx.try_recv() {
            if msg.contains("fail") {
                self.toasts.error(msg);
            } else {
                self.toasts.success(msg);
            }
        }

        let id_floor = self.batch.next_id;
        let mut loaded_count = 0u32;
        let mut channel_drained = false;
        for _ in 0..FILE_LOAD_DRAIN_PER_FRAME {
            match self.batch.bg_io.file_load_rx.try_recv() {
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
                self.batch.selected_index = self.batch.items.len() - 1;
                self.sync_selected_batch_textures(ctx);
            }
            if self.settings.auto_process_on_import && self.batch.next_id > id_floor {
                self.process_items(|item| item.id >= id_floor);
            }
        }
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let title = if self.batch.items.len() >= 2 {
            format!("Prunr \u{2014} {} images", self.batch.items.len())
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
        self.processor.cancel_flag.store(true, Ordering::Release);
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
        if self.source_texture.is_none() && !self.batch.items.is_empty() {
            self.sync_selected_batch_textures(ctx);
        }
        // Periodic cleanup of stale on-disk history files.
        if self.processor.last_history_cleanup.elapsed().as_secs() >= HISTORY_CLEANUP_INTERVAL_SECS {
            self.processor.last_history_cleanup = std::time::Instant::now();
            super::history_disk::cleanup_stale();
        }
    }


    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Dispatch any debounced previews + apply completed ones before we
        // render. tick() returns a hint of how long until the next scheduled
        // dispatch so we can schedule a repaint accordingly.
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

        if self.adjustments_should_show(ui.ctx()) {
            self.render_adjustments_toolbar(ui, panel_frame);
        }

        egui::Panel::bottom("statusbar")
            .exact_size(theme::STATUS_BAR_HEIGHT)
            .frame(panel_frame)
            .show_inside(ui, |ui| statusbar::render(ui, self));

        let sidebar_visible = !self.batch.items.is_empty() && !self.sidebar_hidden;
        if sidebar_visible {
            egui::Panel::right("sidebar")
                .exact_size(theme::SIDEBAR_WIDTH)
                .resizable(false)
                .show_inside(ui, |ui| sidebar::render(ui, self));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| canvas::render(ui, self));

        self.render_modal_overlays(ui.ctx());

        // Consume pending drag-out (sidebar set this when a drag escaped the sidebar).
        // Runs after sidebar renders so the user sees the drag cursor leave the area.
        if let Some(ids) = self.drag_export.pending.take() {
            self.initiate_drag_out(ids, frame);
            // Clear egui's internal drag state — the OS drag session has taken over.
            // Without this, egui keeps showing the DnD crosshair cursor because it
            // still thinks an internal drag is in progress.
            ui.ctx().stop_dragging();
        }
    }
}

impl PrunrApp {
    /// Whether the row 2+3 adjustments toolbar should render this frame.
    /// Shift+H and an empty batch always hide it. `auto_hide_adjustments`
    /// hides it unless the cursor is in a peek zone near the top of the
    /// window or a popup (combo / color picker) is open.
    fn adjustments_should_show(&self, ctx: &egui::Context) -> bool {
        if self.adjustments_hidden || self.batch.items.is_empty() {
            return false;
        }
        if !self.settings.auto_hide_adjustments {
            return true;
        }
        let screen_rect = ctx.content_rect();
        // Peek zone covers the main toolbar + ~half the adjustments toolbar
        // height. Generous enough to catch the user heading up toward the
        // chips, tight enough not to trigger on ordinary canvas work.
        let peek_zone = egui::Rect::from_min_size(
            screen_rect.min,
            egui::vec2(screen_rect.width(), theme::TOOLBAR_HEIGHT + 32.0),
        );
        let hover_in_peek = ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| peek_zone.contains(p)));
        hover_in_peek || egui::Popup::is_any_open(ctx)
    }

    fn render_adjustments_toolbar(&mut self, ui: &mut egui::Ui, panel_frame: egui::Frame) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        // Row 3 is always visible (Lines mode selector lives there), so the
        // toolbar always reserves two rows of height.
        let height = chip::CHIP_HEIGHT * 2.0 + theme::SPACE_XS + theme::SPACE_SM * 2.0;
        let mut toolbar_change = adjustments_toolbar::ToolbarChange::default();
        let is_processing = self.state == AppState::Processing;
        // Snapshot taken BEFORE adjustments_toolbar::render runs — if the
        // user ends up applying a preset this frame, the snapshot goes onto
        // the preset undo stack so Ctrl+Shift+Z can roll back an accidental pick.
        let item = &self.batch.items[idx];
        let pre_apply_snapshot = PresetSnapshot {
            settings: item.settings,
            applied_preset: item.applied_preset.clone(),
        };
        egui::Panel::top("adjustments_toolbar")
            .exact_size(height)
            .frame(panel_frame)
            .show_inside(ui, |ui| {
                // Split borrow: app.settings and app.batch are disjoint
                // fields of PrunrApp, and within the batch item its
                // `settings` and `applied_preset` are disjoint fields too.
                // Lets the toolbar mutate the preset string in place
                // without a clone + writeback round-trip.
                let settings_ref = &mut self.settings;
                let item = &mut self.batch.items[idx];
                toolbar_change = adjustments_toolbar::render(
                    ui,
                    &mut item.settings,
                    settings_ref,
                    &mut item.applied_preset,
                    is_processing,
                );
            });
        self.apply_toolbar_change(ui.ctx(), toolbar_change, pre_apply_snapshot);
    }

    fn apply_toolbar_change(
        &mut self,
        ctx: &egui::Context,
        toolbar_change: adjustments_toolbar::ToolbarChange,
        pre_apply_snapshot: PresetSnapshot,
    ) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        if toolbar_change.preset_applied {
            let item = &mut self.batch.items[idx];
            HistoryManager::push_preset(item, pre_apply_snapshot);
        }
        // Model swap: persist the new selection, show toast, invalidate caches.
        if toolbar_change.model_changed {
            self.settings.save();
            self.toasts.info(format!(
                "{} loaded",
                crate::gui::views::model_name(self.settings.model),
            ));
        }
        // Granular cache invalidation — only clear the tensors whose INPUT
        // actually changed. Preset applies that keep line_mode the same
        // leave the edge cache valid (line_strength tweaks can still
        // live-preview). Model swaps clear only the seg cache (edge tensor
        // is model-independent).
        if toolbar_change.seg_cache_invalid || toolbar_change.edge_cache_invalid {
            if toolbar_change.seg_cache_invalid {
                self.batch.items[idx].cached_tensor = None;
            }
            if toolbar_change.edge_cache_invalid {
                self.batch.items[idx].invalidate_edge_cache();
            }
            // A preset apply that invalidates the cache on an already-
            // processed item means live preview has no tensor to rerun
            // against. Auto-trigger a reprocess so the user doesn't have to
            // click Process; tier routing keeps this cheap when only one
            // tensor is stale.
            if toolbar_change.preset_applied
                && self.batch.items[idx].status == BatchStatus::Done
            {
                let target_id = self.batch.items[idx].id;
                self.process_items(|item| item.id == target_id);
            }
        }
        // Register mask / edge tweaks with the live preview dispatcher.
        // `mark_tweak` debounces; `flush` fires immediately when a chip
        // signals the edit has settled (slider released, toggle flipped,
        // color picked). toolbar_change.commit is OR-ed across all chips
        // touched this frame, so toggles + color picks always commit now
        // and mid-slider-drag changes debounce.
        if self.settings.live_preview && (toolbar_change.mask || toolbar_change.edge) {
            let item_id = self.batch.items[idx].id;
            let kind = if toolbar_change.mask {
                crate::gui::live_preview::PreviewKind::Mask
            } else {
                crate::gui::live_preview::PreviewKind::Edge
            };
            self.processor.live_preview.mark_tweak(item_id, kind);
            if toolbar_change.commit {
                self.processor.live_preview.flush(item_id);
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
            }
        }
        if toolbar_change.bg {
            // bg is rendered at draw time (GPU rect behind transparent result
            // texture) — no CPU compositing, no texture rebuild.
            ctx.request_repaint();
        }
    }

    fn render_modal_overlays(&mut self, ctx: &egui::Context) {
        if self.show_shortcuts && shortcuts::render(ctx) {
            self.show_shortcuts = false;
        }
        if self.show_cli_help && cli_help::render(ctx, &mut self.toasts) {
            self.show_cli_help = false;
        }
        if self.show_settings {
            settings::render(ctx, self);
        }
        // Toasts — rendered last as foreground overlay.
        self.toasts.show(ctx);
    }
}

/// Encode + write one PNG on a background thread, then send the result toast
/// text on `tx`. Stays at module scope so neither save method needs `&mut self`
/// once they've kicked off the work.
fn spawn_save_single(path: PathBuf, rgba: Arc<image::RgbaImage>, tx: mpsc::Sender<String>) {
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

/// Encode + write N PNGs into `folder` (named `<source-stem>-nobg.png`) on a
/// background thread; report aggregate counts when done.
fn spawn_save_batch(
    folder: PathBuf,
    items: Vec<(String, Arc<image::RgbaImage>)>,
    tx: mpsc::Sender<String>,
) {
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

#[derive(Copy, Clone)]
enum NavDir {
    Prev,
    Next,
}

/// Classification output from `classify_candidates`: which items go to the
/// full subprocess pipeline (tier1), which can do an in-subprocess mask
/// rerun from their cached tensor (tier2), and how many were already
/// up-to-date (skip_count, used only for the user-facing toast).
#[derive(Default)]
struct ClassifiedTiers {
    tier1: HashSet<u64>,
    tier2: HashSet<u64>,
    skip_count: usize,
}

impl ClassifiedTiers {
    /// Union of tier1 + tier2 — the set of items that will actually be
    /// reprocessed this batch (history seeding, progress counting).
    fn all_process_ids(&self) -> HashSet<u64> {
        self.tier1.iter().chain(self.tier2.iter()).copied().collect()
    }
}

/// Pure aggregate of "user pressed a shortcut this frame" flags, collected
/// from `ctx.input` in a single pass. Separating collect from apply lets the
/// apply phase freely borrow `&mut self` without fighting egui's input lock.
#[derive(Default)]
struct ShortcutIntents {
    open_requested: bool,
    remove_requested: bool,
    save_requested: bool,
    cancel_requested: bool,
    toggle_shortcuts: bool,
    toggle_cli_help: bool,
    toggle_before_after: bool,
    fit_to_window: bool,
    actual_size: bool,
    toggle_settings: bool,
    nav_prev: bool,
    nav_next: bool,
    toggle_sidebar: bool,
    toggle_adjustments: bool,
    undo_requested: bool,
    redo_requested: bool,
    preset_undo_requested: bool,
    preset_redo_requested: bool,
}

fn collect_shortcut_intents(ctx: &egui::Context) -> ShortcutIntents {
    // Suppress bare-key shortcuts when any widget has focus (e.g., hex color input).
    let text_focused = ctx.memory(|m| m.focused().is_some());
    let mut s = ShortcutIntents::default();
    ctx.input(|i| {
        // Modifier shortcuts always work, even with a text field focused.
        if i.modifiers.command && i.key_pressed(Key::O) { s.open_requested = true; }
        if i.modifiers.command && i.key_pressed(Key::R) { s.remove_requested = true; }
        if i.modifiers.command && i.key_pressed(Key::S) { s.save_requested = true; }
        if i.key_pressed(Key::Escape)                   { s.cancel_requested = true; }
        if i.key_pressed(Key::F1)                       { s.toggle_shortcuts = true; }
        if i.key_pressed(Key::F2)                       { s.toggle_cli_help = true; }
        if i.modifiers.command && i.key_pressed(Key::Num0)  { s.fit_to_window = true; }
        if i.modifiers.command && i.key_pressed(Key::Num1)  { s.actual_size = true; }
        if i.modifiers.command && i.key_pressed(Key::Space) { s.toggle_settings = true; }

        // Bare-key shortcuts — only when no text field is focused.
        if !text_focused {
            if i.key_pressed(Key::B) { s.toggle_before_after = true; }
            if i.key_pressed(Key::ArrowLeft)  || i.key_pressed(Key::A) { s.nav_prev = true; }
            if i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::D) { s.nav_next = true; }
            // H = toggle sidebar, Shift+H = toggle adjustments toolbar.
            // Tab stays reserved for egui's focus traversal (accessibility),
            // kept here as a fallback for v1 muscle memory.
            if i.key_pressed(Key::H) {
                if i.modifiers.shift { s.toggle_adjustments = true; }
                else                 { s.toggle_sidebar = true; }
            }
            if i.key_pressed(Key::Tab) && !i.modifiers.shift { s.toggle_sidebar = true; }
        }

        // Shift variants of Z/Y are preset-only undo/redo — roll back an
        // accidental preset apply without touching the image-result history.
        // Without-shift stays bound to image history.
        if i.modifiers.command && i.key_pressed(Key::Z) {
            if i.modifiers.shift { s.preset_undo_requested = true; }
            else                 { s.undo_requested = true; }
        }
        if i.modifiers.command && i.key_pressed(Key::Y) {
            if i.modifiers.shift { s.preset_redo_requested = true; }
            else                 { s.redo_requested = true; }
        }
    });
    s
}
