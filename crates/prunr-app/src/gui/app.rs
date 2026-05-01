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
use super::views::{adjustments_toolbar, canvas, cli_help, model_store, pipeline_flow, settings, shortcuts, sidebar, statusbar, toolbar};

/// Days the user is left alone after dismissing the first-launch
/// runtime prompt. 14 picked to balance "don't nag" with "remind on a
/// reasonable cadence as the SD experience improves."
const RUNTIME_PROMPT_SNOOZE_DAYS: i64 = 14;

/// Replaced wholesale (never mutated in place) so the borrow checker
/// stays happy with the receiver living in the struct.
pub(crate) struct RuntimeInstallProgress {
    pub(crate) runtime: crate::runtime_install::RuntimeId,
    pub(crate) rx: mpsc::Receiver<crate::runtime_install::InstallEvent>,
    pub(crate) last_event: crate::runtime_install::InstallEvent,
    pub(crate) cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default, PartialEq)]
enum TitleState { #[default] Empty, Single(String), Batch(usize) }

pub struct PrunrApp {
    /// Directory of the most recently opened file (for save dialog default)
    pub(crate) last_open_dir: Option<std::path::PathBuf>,

    /// Processing pipeline — worker channels, admission, live preview, dispatch state.
    pub(crate) processor: super::processor::Processor,

    pub(crate) status: super::status_state::StatusState,

    /// Platform I/O: file dialogs + clipboard. Shim around `rfd` + `arboard`
    /// so the rest of `PrunrApp` doesn't carry their import surface.
    pub(crate) system: super::system_bridge::SystemBridge,

    // UI state
    pub(crate) show_shortcuts: bool,
    pub(crate) show_cli_help: bool,
    pub(crate) show_pipeline_flow: bool,

    // Set by raw_input_hook — egui converts Ctrl+C to Event::Copy before we see it
    pending_copy: bool,

    pub(crate) zoom_state: super::zoom_state::ZoomState,
    pub(crate) brush_state: super::brush_state::BrushState,
    pub(crate) download_manager: super::download_manager::DownloadManager,

    // Before/After toggle
    pub(crate) show_original: bool,

    title_state: TitleState,

    /// Last `(item_id, MaskSettings)` `recipe_drift_tripwire` proved
    /// drift-free. Skipping `MaskRecipe::from` on a per-frame match
    /// avoids ~60 recipe constructions/sec while the user sits idle on
    /// a Done item with no settings tweaks. Cleared on drift or item
    /// switch — the worst case re-pays a single frame.
    last_drift_check: Option<(u64, prunr_core::MaskSettings)>,

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
    /// Which Settings tab the modal is showing. Transient UI state — not
    /// persisted; opening Settings always starts on General.
    pub(crate) settings_tab: super::views::settings::SettingsTab,
    /// Inline "Reset to defaults" confirmation. Set when the user clicks
    /// the reset button; the next render shows a confirm/cancel pair.
    pub(crate) pending_reset_confirm: bool,

    pub(crate) model_store: Option<super::views::adjustments_toolbar::ModelStoreRequest>,
    /// When `Some(id)`, the license-acceptance dialog is open for that
    /// model. Set by Model Store's Download click for any descriptor
    /// where `requires_license_acceptance() && !has_accepted_license`;
    /// cleared on Accept (then `start_download`) or Cancel.
    pub(crate) pending_license_request: Option<prunr_models::ModelId>,
    /// Upgrade path: saved settings may reference a model that's now
    /// OnDemand and not installed. Shown once, then `take()`d.
    pub(crate) pending_onboarding_toast: Option<String>,

    pub(crate) runtime_install: Option<RuntimeInstallProgress>,
    /// Snapshot of `RuntimeId::is_installed()` to avoid syscalling per-frame
    /// while the Settings → Hardware tab is open. Refreshed on settings open,
    /// install completion, and uninstall.
    pub(crate) hardware_install_cache: super::hardware_cache::HardwareInstallCache,

    pub(crate) runtime_prompt: Option<crate::runtime_install::RuntimeId>,
    /// Once-per-session guard so we don't re-evaluate hardware + snooze
    /// state every frame after the prompt is dismissed.
    runtime_prompt_evaluated: bool,

    // Canvas fade-in: incremented on every image switch
    pub(crate) canvas_switch_id: u64,
    /// Incremented when a result completes, drives crossfade in render_done
    pub(crate) result_switch_id: u64,

    /// Set by add_to_batch — triggers sync_selected_batch_textures in next logic()
    pending_batch_sync: bool,
    /// Set by toolbar Open button — processed in logic() where ctx is available
    pub(crate) pending_open_dialog: bool,
    /// Toast notification system
    pub(crate) toasts: super::toasts::Toasts,

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
        let subtle = egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x3a, 0x3a, 0x3a));
        visuals.widgets.noninteractive.bg_stroke = subtle;
        visuals.widgets.inactive.bg_stroke = subtle;
        visuals.widgets.active.bg_stroke = egui::Stroke::new(theme::STROKE_DEFAULT, theme::ACCENT);
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(theme::STROKE_DEFAULT, theme::WIDGET_INACTIVE_BG);
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

        // Housekeeping: clean up stale temp files from prior sessions.
        super::drag_export::cleanup_stale();
        super::history_disk::cleanup_stale();

        let mut settings = Settings::load();
        settings.active_backend = prunr_core::OrtEngine::detect_active_provider();
        // PRUNR_OPEN_MODEL must apply BEFORE the worker prewarm config below
        // — otherwise the wrong model loads at startup and the first Process
        // click pays a full subprocess respawn (~15 s on this CPU). The env
        // var is consumed here; PrunrApp::new no longer needs to re-read it.
        if let Some(name) = std::env::var_os("PRUNR_OPEN_MODEL") {
            unsafe { std::env::remove_var("PRUNR_OPEN_MODEL"); }
            if let Some(s) = name.to_str() {
                if let Some(m) = super::settings::SettingsModel::from_debug_name(s) {
                    settings.model = m;
                }
            }
        }
        // Test-harness escape hatch: flip auto-process so an imported image
        // runs the pipeline without xdotool driving Ctrl+R.
        if let Some(v) = super::env_overrides::auto_process_override() {
            settings.auto_process_on_import = v;
        }
        // Phase 17 upgrade path: if the user's saved model is now OnDemand
        // and the file isn't on disk, queue a one-time toast pointing
        // them to the Model Store. Bundled-only users see nothing.
        let onboarding_toast = settings.model.to_model_id()
            .filter(|id| !prunr_models::is_available(*id))
            .and_then(prunr_models::descriptor)
            .map(|d| format!(
                "{} is now an on-demand download — open the Model Store from the model dropdown.",
                d.display_name,
            ));

        // Subprocess worker: inference runs in a child process for OOM
        // isolation. Pre-warm a subprocess with the startup config so the
        // first Process click skips the 1–5s model-load cost. Filter-only
        // mode (SettingsModel::None) skips pre-warm — no ORT session
        // needed for pure CPU filters.
        let prewarm = Self::initial_processing_config(&settings);
        let (worker_tx, worker_rx) = spawn_worker(worker_ctx, prewarm);

        let mut app = Self::init_state(settings, super::system_bridge::SystemBridge::new(), worker_tx, worker_rx);
        app.pending_onboarding_toast = onboarding_toast;
        app
    }

    /// Build the pre-warm subprocess config for startup, or `None` when
    /// pre-warming doesn't make sense (e.g. user has "No model" selected —
    /// filter-only runs without ORT, so the subprocess would never be used).
    /// Uses `ItemSettings::default()` for mask/edge because startup has no
    /// selected item yet; Process clicks that use different settings will
    /// drop the warm sub and spawn fresh.
    fn initial_processing_config(settings: &Settings) -> Option<super::worker::ProcessingConfig> {
        let model = settings.model.to_model_kind()?;
        let item_defaults = super::item_settings::ItemSettings::default();
        Some(super::worker::ProcessingConfig {
            model,
            jobs: settings.parallel_jobs,
            mask: item_defaults.mask_settings(),
            force_cpu: settings.force_cpu,
            line_mode: item_defaults.line_mode,
            edge: item_defaults.edge_settings(),
        })
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        // Test stub: no real platform clipboard available; SystemBridge::new
        // gracefully no-ops copy_image when the clipboard handle failed to
        // initialize, so this is safe in headless test envs.
        Self::init_state(
            Settings::default(),
            super::system_bridge::SystemBridge::new(),
            worker_tx,
            worker_rx,
        )
    }

    /// Shared field-init for both `new` and `new_for_test`. SystemBridge and
    /// worker channels are the only inputs that differ between runtime and test.
    fn init_state(
        settings: Settings,
        system: super::system_bridge::SystemBridge,
        worker_tx: mpsc::Sender<WorkerMessage>,
        worker_rx: mpsc::Receiver<WorkerResult>,
    ) -> Self {
        let mut app = Self {
            last_open_dir: None,
            processor: super::processor::Processor::new(worker_tx, worker_rx),
            status: Default::default(),
            system,
            show_shortcuts: false,
            show_cli_help: false,
            show_pipeline_flow: false,
            pending_copy: false,
            zoom_state: Default::default(),
            brush_state: super::brush_state::BrushState::default(),
            download_manager: super::download_manager::DownloadManager::new(),
            show_original: false,
            title_state: TitleState::default(),
            last_drift_check: None,
            batch: super::batch_manager::BatchManager::new(),
            sidebar_hidden: false,
            adjustments_hidden: false,
            show_settings: false,
            settings_tab: super::views::settings::SettingsTab::General,
            pending_reset_confirm: false,
            model_store: None,
            pending_license_request: None,
            pending_onboarding_toast: None,
            runtime_install: None,
            hardware_install_cache: super::hardware_cache::HardwareInstallCache::default(),
            runtime_prompt: None,
            runtime_prompt_evaluated: false,
            settings_opened_at: 0.0,
            settings,
            canvas_switch_id: 0,
            result_switch_id: 0,
            pending_batch_sync: false,
            pending_open_dialog: false,
            toasts: super::toasts::Toasts::new(
                egui_notify::Anchor::BottomLeft,
                egui::vec2(theme::SPACE_SM, theme::STATUS_BAR_HEIGHT + theme::SPACE_SM),
            ),
            drag_export: super::drag_export_state::DragExportState::new(),
        };
        // `--open <path>` (or PRUNR_OPEN_FILE env var) — pre-load on launch.
        // Reads the env once; clears it so a child subprocess doesn't inherit
        // and re-load again on a worker spawn.
        let preload = std::env::var_os("PRUNR_OPEN_FILE")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        // Safety: we're still in `new`, before any worker thread spawns.
        unsafe { std::env::remove_var("PRUNR_OPEN_FILE"); }
        if let Some(path) = preload {
            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("untitled")
                .to_string();
            let _ = app.batch.bg_io.file_load_tx.send((path, name));
        }
        // Test-harness escape hatch: PRUNR_OPEN_TAB pre-selects a Settings
        // tab and auto-opens the modal at startup. Lets the harness capture
        // each tab without driving mouse clicks (unreliable under Xephyr's
        // coord-space mismatch). Values match SettingsTab labels: General /
        // Behavior / Hotkeys (case-sensitive). Not exposed on --help.
        if let Some(name) = std::env::var_os("PRUNR_OPEN_TAB") {
            unsafe { std::env::remove_var("PRUNR_OPEN_TAB"); }
            if let Some(s) = name.to_str() {
                if let Some(t) = super::views::settings::SettingsTab::from_label(s) {
                    app.settings_tab = t;
                    app.show_settings = true;
                    // settings_opened_at stays at the default 0.0 — first
                    // frame's backdrop-debounce check uses egui input time
                    // which won't be < 0.0, so no premature dismissal.
                }
            }
        }
        app
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

    /// Sync after batch modification — clamp index and refresh canvas.
    fn sync_after_batch_change(&mut self) {
        if self.batch.items.is_empty() {
            self.batch.selected_index = 0;
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
            name,
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
        let paths = self.system.open_files_dialog(self.last_open_dir.as_deref());
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

    /// Open a file picker for a per-item background image. Decode + hash
    /// run synchronously on the UI thread — once per pick, after a modal
    /// file dialog already blocked. An async pipeline would need a
    /// pending-state machine for marginal benefit on a user-initiated event.
    pub(crate) fn handle_pick_bg_image(&mut self, idx: usize, ctx: &egui::Context) {
        let Some(path) = self.system.pick_image_dialog(
            self.last_open_dir.as_deref(),
            "Choose background image",
        ) else { return };
        match prunr_core::load_image_from_path(&path) {
            Ok(img) => {
                if let Some(item) = self.batch.items.get_mut(idx) {
                    item.set_bg_image(img, Some(path.clone()));
                    // Remember the path keyed by the bg's content hash so a
                    // preset that captured this hash can reload the image
                    // when applied later (or after a restart).
                    if let Some(bg) = item.bg_image.as_ref() {
                        self.settings.bg_image_paths.insert(bg.hash, path);
                        self.settings.save();
                    }
                    ctx.request_repaint();
                }
            }
            Err(err) => {
                self.toasts.error(format!("Couldn't load background image: {err}"));
            }
        }
    }

    /// Reconcile `BatchItem.bg_image` with `settings.bg_image_hash` after a
    /// preset apply. Preset apply copies `ItemSettings` (which includes the
    /// hash) but doesn't touch the bytes on the BatchItem; this restores
    /// the lockstep `set_bg_image` / `clear_bg_image` invariant.
    fn reconcile_bg_image_after_preset(&mut self, idx: usize) {
        let Some(item) = self.batch.items.get_mut(idx) else { return };
        let want = item.settings.bg_image_hash;
        let have = item.bg_image.as_ref().map(|b| b.hash);
        if want == have { return; }
        let Some(want_hash) = want else {
            // Preset has no bg image — drop the stale bytes.
            item.clear_bg_image();
            return;
        };
        // New hash from preset — try to reload via the persisted path map.
        let path = self.settings.bg_image_paths.get(&want_hash).cloned();
        let Some(path) = path else {
            // Hash isn't in our path map (preset shared from another user,
            // or the path entry was wiped). Drop both bytes and hash so the
            // recipe diff stays consistent.
            item.clear_bg_image();
            self.toasts.info("Preset references a background image we don't have on disk.");
            return;
        };
        match prunr_core::load_image_from_path(&path) {
            Ok(img) => item.set_bg_image(img, Some(path)),
            Err(err) => {
                item.clear_bg_image();
                self.toasts.error(format!("Couldn't load preset background: {err}"));
            }
        }
    }

    pub fn handle_remove_bg(&mut self) {
        let ids: std::collections::HashSet<u64> =
            self.batch.items_to_process().into_iter().collect();
        if ids.is_empty() {
            return;
        }
        self.process_items(|item| ids.contains(&item.id));
    }

    pub(crate) fn close_settings(&mut self, _ctx: &egui::Context) {
        self.show_settings = false;
        // Don't carry the half-clicked reset confirm across modal opens.
        self.pending_reset_confirm = false;
        self.settings.save();
        self.toasts.info("Settings saved");
    }

    pub(crate) fn any_modal_open(&self) -> bool {
        self.show_settings
            || self.show_shortcuts
            || self.show_cli_help
            || self.show_pipeline_flow
    }

    /// Undo background removal on selected items (or current item if none selected).
    /// Reverts Done/Error items back to Pending, clearing their results.
    ///
    /// Brush mode override: while the brush is active, prefer popping a
    /// stroke off the active item's stroke history. Falls through to the
    /// result-undo when there are no strokes left to undo.
    pub fn handle_undo(&mut self, ctx: &egui::Context) {
        if self.try_brush_undo(ctx) {
            return;
        }
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

    pub fn handle_redo(&mut self, ctx: &egui::Context) {
        if self.try_brush_redo(ctx) {
            return;
        }
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

    /// If brush is active and the selected item has a stroke to undo,
    /// pop it and dispatch a Tier 2 rerun. Returns `true` when handled.
    fn try_brush_undo(&mut self, _ctx: &egui::Context) -> bool {
        if !self.brush_state.is_enabled() {
            return false;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return false };
        if !self.batch.items[idx].has_stroke_undo() {
            return false;
        }
        if self.batch.items[idx].undo_stroke() {
            self.dispatch_brush_rerun(idx);
            self.toasts.info("Stroke undone");
            true
        } else {
            false
        }
    }

    fn try_brush_redo(&mut self, _ctx: &egui::Context) -> bool {
        if !self.brush_state.is_enabled() {
            return false;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return false };
        if !self.batch.items[idx].has_stroke_redo() {
            return false;
        }
        if self.batch.items[idx].redo_stroke() {
            self.dispatch_brush_rerun(idx);
            self.toasts.info("Stroke restored");
            true
        } else {
            false
        }
    }

    fn dispatch_brush_rerun(&mut self, idx: usize) {
        use crate::gui::live_preview::PreviewKind;
        let item_id = self.batch.items[idx].id;
        // Unconditional — see canvas::handle_brush_input.
        self.processor.live_preview.mark_tweak(item_id, PreviewKind::Mask);
        self.processor.live_preview.flush(item_id);
    }

    pub(crate) fn dispatch_inpaint_for_item(&mut self, idx: usize) {
        let item = &self.batch.items[idx];
        let item_id = item.id;
        let Some(correction) = item.mask_correction.as_ref().cloned() else {
            tracing::debug!(item_id, "inpaint dispatch skipped: no correction");
            return;
        };
        // Stack-based inpaint: each new stroke runs against the
        // previous result (if any), not the original. This keeps
        // earlier strokes intact and — critically for SD — keeps the
        // mask bbox bounded to the current stroke instead of the
        // cumulative paint history. source_rgba may have been evicted
        // under memory pressure; rehydrate from cached DynamicImage.
        let source = item.result_rgba.as_ref().cloned()
            .or_else(|| item.source_rgba.as_ref().cloned())
            .or_else(|| item.source_dyn.as_ref().map(|d| Arc::new(d.to_rgba8())));
        let Some(source) = source else {
            tracing::warn!(item_id, "inpaint dispatch skipped: source RGBA unavailable");
            return;
        };
        let bs = &self.settings.brush;
        let raw_backend = self.settings.model.to_model_id()
            .unwrap_or(prunr_models::ModelId::LaMaFp32);
        let backend = if self.settings.lcm_routing_active(raw_backend) {
            prunr_models::ModelId::SdV15LcmInpaintFp16
        } else {
            raw_backend
        };
        tracing::info!(item_id, ?backend, ?raw_backend, "inpaint stroke committed; dispatching");
        let tuning = super::processor::InpaintTuning {
            sharpen: bs.inpaint_sharpen,
            feather_px: bs.inpaint_feather,
            grow_px: bs.inpaint_grow,
            backend,
            sd_prompt: bs.sd_prompt.clone(),
            sd_negative_prompt: bs.sd_negative_prompt.clone(),
            sd_guidance_scale: bs.sd_guidance_scale,
        };
        self.processor.dispatch_inpaint(item_id, source, correction, tuning);
    }

    fn pump_inpaint_results(&mut self, ctx: &egui::Context) {
        // Bridge → inpaint_rx fan-in must run before the drain.
        self.processor.pump_inpaint_subprocess();
        let (results, cancelled, errors) = self.processor.drain_inpaint_results();
        for _ in &cancelled {
            self.toasts.info("Erase cancelled");
        }
        for msg in &errors {
            // Show the worker's message verbatim — the SD RAM-guard
            // text is already user-friendly ("only X GB free, Y minimum
            // recommended. Close other apps or use LaMa..."). Future
            // CoreError variants that surface here can be reformatted
            // at this seam if their default Display is too technical.
            self.toasts.error(msg.clone());
        }
        if results.is_empty() {
            return;
        }
        let tex_prep_tx = self.batch.bg_io.tex_prep_tx.clone();
        let switch = self.result_switch_id;
        for r in results {
            // Keep old texture visible until tex_prep lands so the canvas
            // doesn't flash empty for one frame between RGBA arriving and
            // GPU upload finishing.
            let (item_id, source, result_rgba) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else { continue };
                let new_rgba = Arc::new(r.rgba);
                item.result_rgba = Some(new_rgba.clone());
                if item.status == BatchStatus::Pending {
                    item.status = BatchStatus::Done;
                }
                item.result_tex_pending = true;
                item.thumb_pending = true;
                // Stack-based inpaint: this stroke is now baked into
                // result_rgba; clear the correction so the NEXT stroke's
                // bbox is just that stroke. Push the pre-clear state to
                // the undo stack so Ctrl+Z restores it (UX seam: undo
                // restores the brush-correction overlay, not the result
                // pixels — full result-pixel undo is separate work).
                item.clear_correction();
                Self::spawn_tex_prep(
                    new_rgba.clone(), item.id, format!("inpaint_{}_{}", item.id, switch),
                    true, tex_prep_tx.clone(), ctx.clone(),
                );
                (item.id, item.source.clone(), Some(new_rgba))
            };
            self.batch.request_thumbnail(item_id, &source, result_rgba.as_ref());
        }
        ctx.request_repaint();
    }

    pub(crate) fn maybe_evaluate_runtime_prompt(&mut self) {
        if self.runtime_prompt_evaluated { return; }
        self.runtime_prompt_evaluated = true;
        use crate::runtime_install::RuntimeId;
        let rt = RuntimeId::OpenVino;
        let profile = crate::hardware::profile();
        if !profile.recommends_openvino() { return; }
        if rt.is_installed() { return; }
        if self.settings.is_runtime_prompt_snoozed(rt) { return; }
        self.runtime_prompt = Some(rt);
    }

    pub(crate) fn pump_runtime_install(&mut self, ctx: &egui::Context) {
        use crate::runtime_install::InstallEvent;
        let Some(progress) = self.runtime_install.as_mut() else { return };
        while let Ok(event) = progress.rx.try_recv() {
            progress.last_event = event.clone();
            match event {
                InstallEvent::Done { .. } => {
                    let name = progress.runtime.display_name();
                    self.toasts.success(format!("{name} installed"));
                    self.runtime_install = None;
                    self.hardware_install_cache =
                        super::hardware_cache::HardwareInstallCache::refresh();
                    return;
                }
                InstallEvent::Failed { error } => {
                    let name = progress.runtime.display_name();
                    self.toasts.error(format!("{name} install failed: {error}"));
                    self.runtime_install = None;
                    self.hardware_install_cache =
                        super::hardware_cache::HardwareInstallCache::refresh();
                    return;
                }
                _ => {}
            }
        }
        ctx.request_repaint();
    }

    fn pump_download_manager(&mut self, ctx: &egui::Context) {
        let events = self.download_manager.pump();
        if events.is_empty() {
            return;
        }
        for event in events {
            use super::download_manager::DownloadEvent;
            match event {
                DownloadEvent::Complete { id } => {
                    let name = prunr_models::descriptor(id)
                        .map_or("Model", |d| d.display_name);
                    self.toasts.success(format!("{name} ready"));
                }
                DownloadEvent::Failed { id, error, .. } => {
                    let name = prunr_models::descriptor(id)
                        .map_or("Model", |d| d.display_name);
                    self.toasts.error(format!("{name} download failed: {error}"));
                }
                DownloadEvent::Progress { .. } | DownloadEvent::Verifying { .. } => {}
            }
        }
        ctx.request_repaint();
    }

    /// Catch any drift between the active item's `applied_recipe.mask`
    /// and the recipe derived from current settings. Safety net for
    /// non-toolbar state mutations (brush commits, hotkeys) so they
    /// never silently fail to update the result.
    ///
    /// Active-item only by design — the only drift surfaces today
    /// (brush, undo/redo) act on the selected item; widening to the
    /// full batch would do per-frame work for items the user isn't
    /// editing for no current benefit.
    fn recipe_drift_tripwire(&mut self) {
        use crate::gui::live_preview::PreviewKind;
        if self.processor.live_preview.has_in_flight() {
            return;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let Some(applied) = item.applied_recipe.as_ref() else { return };
        let id = item.id;
        let current_settings = item.settings.mask_settings();
        if self.last_drift_check == Some((id, current_settings)) {
            return;
        }
        let current = prunr_core::MaskRecipe::from(&current_settings);
        if applied.mask == current {
            self.last_drift_check = Some((id, current_settings));
            return;
        }
        self.last_drift_check = None;
        tracing::debug!(item_id = id, "recipe-drift tripwire fired — dispatching Tier-2");
        self.processor.live_preview.mark_tweak(id, PreviewKind::Mask);
        self.processor.live_preview.flush(id);
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
        // Restoring a preset snapshot rewrites settings.bg_image_hash —
        // pull the matching bytes back into bg_image so the canvas paint
        // and recipe diff stay consistent.
        self.reconcile_bg_image_after_preset(idx);
        if should_reprocess {
            self.process_items(|i| i.id == target_id);
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    /// Uses tier routing: compares each item's applied_recipe against current
    /// settings to determine the minimum work needed (skip / mask rerun / full).
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        let chain = self.settings.chain_mode;
        // Pure filter mode: No model AND line_mode = Off. Skip inference +
        // subprocess entirely. (No model + EdgesOnly still needs DexiNed,
        // so falls through to the normal path with a dummy ModelKind — the
        // seg engine won't spawn because needs_segmentation is false.)
        let is_pure_filter = self.settings.model.to_model_kind().is_none()
            && self.batch.items.iter().all(|i| i.settings.line_mode == prunr_core::LineMode::Off);
        if is_pure_filter {
            self.process_filter_only(filter);
            return;
        }
        // No-model-but-EdgesOnly falls through with a placeholder. The
        // subprocess Init still receives this field; `needs_segmentation`
        // is false for EdgesOnly so the seg model never loads.
        let model: prunr_core::ModelKind = self.settings.model
            .to_model_kind()
            .unwrap_or(prunr_core::ModelKind::BiRefNetLite);

        let candidate_ids: HashSet<u64> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| i.id)
            .collect();
        if candidate_ids.is_empty() { return; }

        let tiers = self.classify_candidates(&candidate_ids, model, chain);
        let process_count = tiers.tier1.len() + tiers.tier2.len() + tiers.tier_add_edge.len();
        self.notify_skip(tiers.skip_count, process_count);
        if process_count == 0 { return; }

        self.seed_history_for_reprocess(&tiers.all_process_ids(), chain);

        let tier2_work = self.build_tier2_work(&tiers.tier2);
        let add_edge_work = self.build_add_edge_work(&tiers.tier_add_edge);

        let jobs = self.settings.parallel_jobs.min(super::memory::safe_max_jobs(model));
        if tiers.tier1.len() > 1 {
            self.dispatch_with_admission(&tiers.tier1, tier2_work, add_edge_work, model, jobs, chain);
        } else {
            self.dispatch_small_batch(&tiers.tier1, tier2_work, add_edge_work, model, jobs, chain);
        }
    }

    /// Filter-only Process path (model=`None`). Dispatches each target item
    /// to a background thread via `BatchManager::request_filter_only` so the
    /// UI stays responsive on large batches. Results land on
    /// `bg_io.filter_only_rx`, drained in `drain_background_channels`.
    fn process_filter_only(&mut self, filter: impl Fn(&BatchItem) -> bool) {
        let dispatches: Vec<(u64, ImageSource, prunr_core::FillStyle)> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| (i.id, i.source.clone(), i.settings.fill_style))
            .collect();
        for (id, source, fill_style) in dispatches {
            if let Some(item) = self.batch.find_by_id_mut(id) {
                item.status = BatchStatus::Processing;
            }
            self.batch.request_filter_only(id, &source, fill_style);
        }
    }

    fn build_add_edge_work(&mut self, ids: &HashSet<u64>) -> Vec<super::worker::AddEdgeWorkItem> {
        let mut out = Vec::new();
        for item in &mut self.batch.items {
            if !ids.contains(&item.id) { continue; }
            let Some(ref ct) = item.cached_tensor else { continue };
            let tensor_data = ct.decompress();
            let mask = item.settings.mask_settings();
            match (tensor_data, item.source.load_bytes()) {
                (Some(data), Ok(bytes)) => {
                    out.push(super::worker::AddEdgeWorkItem {
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
                    item.status = BatchStatus::Error("Seg tensor cache corrupt".into());
                    item.cached_tensor = None;
                    item.applied_recipe = None;
                }
                (_, Err(e)) => {
                    item.status = BatchStatus::Error(format!("Failed to load: {e}"));
                }
            }
        }
        out
    }

    /// Classify each candidate into Tier 1 (full pipeline), Tier 2 (mask
    /// rerun from cached tensor), or Skip (already up to date).
    /// Mutates items in-place: invalidates stale caches via the catalog,
    /// syncs composite-only recipe changes, and never downgrades Tier 2
    /// items without a tensor.
    fn classify_candidates(
        &mut self,
        candidate_ids: &HashSet<u64>,
        model: prunr_core::ModelKind,
        chain: bool,
    ) -> ClassifiedTiers {
        use crate::gui::knob_catalog;
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
            let tier = prunr_core::resolve_tier(old_recipe, &current_recipe);
            let impact = knob_catalog::cache_impact_for_recipe_diff(old_recipe, &current_recipe);
            item.apply_cache_impact(impact);

            match tier {
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
                        // SubjectOutline mask rerun needs DexiNed re-composite
                        // on the new masked base; Tier 2 RePostProcess only
                        // runs the mask side, which would drop the outline.
                        if current_recipe.inference.uses_edge_detection {
                            tiers.tier_add_edge.insert(item.id);
                        } else {
                            tiers.tier2.insert(item.id);
                        }
                    } else {
                        tiers.tier1.insert(item.id);
                    }
                }
                RequiredTier::EdgeRerun => {
                    // Edge rerun via the subprocess isn't wired for batch
                    // dispatch — fall through to a full pipeline run (live-
                    // preview handles the in-process path via finalize_edges).
                    tiers.tier1.insert(item.id);
                }
                RequiredTier::AddEdgeInference => {
                    if item.cached_tensor.is_some() {
                        tiers.tier_add_edge.insert(item.id);
                    } else {
                        tiers.tier1.insert(item.id);
                    }
                }
                RequiredTier::FullPipeline => {
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
        add_edge_work: Vec<super::worker::AddEdgeWorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        chain: bool,
    ) {
        use super::memory::{AdmissionController, ImageMemCost};

        let mut ctrl = AdmissionController::new(model, jobs);
        let history_depth = self.settings.history_depth;
        let costs: Vec<ImageMemCost> = self.batch.items.iter()
            .filter(|i| tier1_ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(
                i.id, i.dimensions, i.source.estimated_size(), history_depth,
            ))
            .collect();
        ctrl.enqueue(costs);

        let mut initial_items = Vec::new();
        while let Some(admitted_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(admitted_id) {
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

        // All Tier 1 items failed load_bytes AND no Tier 2 / add-edge work —
        // skip dispatch.
        if initial_items.is_empty() && tier2_work.is_empty() && add_edge_work.is_empty() { return; }

        let (atx, arx) = mpsc::channel();
        self.processor.admission = Some(ctrl);
        self.processor.admission_tx = Some(atx);
        self.dispatch_batch(initial_items, tier2_work, add_edge_work, model, jobs, Some(arx));
    }

    /// Single-item or small-batch fast path — skip admission entirely.
    fn dispatch_small_batch(
        &mut self,
        tier1_ids: &HashSet<u64>,
        tier2_work: Vec<super::worker::Tier2WorkItem>,
        add_edge_work: Vec<super::worker::AddEdgeWorkItem>,
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

        // If all prep failed, don't dispatch an empty batch.
        if items.is_empty() && tier2_work.is_empty() && add_edge_work.is_empty() { return; }

        self.dispatch_batch(items, tier2_work, add_edge_work, model, jobs, None);
    }

    /// Build and send a WorkerMessage::BatchProcess with current settings.
    fn dispatch_batch(
        &mut self,
        items: Vec<super::worker::WorkItem>,
        tier2_items: Vec<super::worker::Tier2WorkItem>,
        add_edge_items: Vec<super::worker::AddEdgeWorkItem>,
        model: prunr_core::ModelKind,
        jobs: usize,
        additional_items_rx: Option<mpsc::Receiver<super::worker::WorkItem>>,
    ) {
        self.processor.cancels.reset();

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
            .chain(add_edge_items.iter().map(|ai| ai.item_id))
            .collect();
        for item in &mut self.batch.items {
            if process_ids.contains(&item.id) {
                item.settings = current_settings;
            }
        }

        let recipe = current_settings.current_recipe(model, self.settings.chain_mode);
        self.processor.track_dispatch(recipe, process_ids.iter().copied());

        self.status.pct = 0.0;
        self.status.stage = "Starting".to_string();
        let _ = self.processor.worker_tx.send(WorkerMessage::BatchProcess {
            items,
            tier2_items,
            add_edge_items,
            config: super::worker::ProcessingConfig {
                model,
                jobs,
                mask: current_settings.mask_settings(),
                force_cpu: self.settings.force_cpu,
                line_mode: current_settings.line_mode,
                edge: current_settings.edge_settings(),
            },
            cancels: self.processor.cancels.clone(),
            additional_items_rx,
        });
    }


    pub fn handle_save_selected(&mut self) {
        let has_selection = self.batch.items.iter()
            .any(|i| i.selected && i.status == BatchStatus::Done && i.result_rgba.is_some());
        // Layers mode: always folder-picker, regardless of selection count.
        // The filenames are derived from each item's source stem + layer suffix.
        if self.settings.export_split_layers {
            self.save_layers_to_folder(has_selection);
            return;
        }
        if has_selection {
            self.save_selected_to_folder();
        } else {
            self.save_current_to_file();
        }
    }

    /// Folder-picker + multi-file save for layers mode. Renders up to 3 PNGs
    /// per target (subject / lines / mask) via `drag_export::render_layer_bytes`
    /// and writes them on a background thread. Per-item fallback to composite
    /// when tensors are missing (e.g. unprocessed item), so every target lands
    /// at least one file.
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

        let mut payload: Vec<(PathBuf, Vec<u8>)> = Vec::new();
        for id in &targets {
            let Some(item) = self.batch.find_by_id(*id) else { continue };
            let layers = super::drag_export::render_layer_bytes(item);
            if layers.is_empty() {
                // No cached tensors — fall back to composite PNG so the user
                // still lands a file per target.
                if let Some(rgba) = item.result_rgba.as_ref() {
                    let baked = item.bake_export_bg(rgba);
                    if let Ok(bytes) = prunr_core::encode_rgba_png(&baked) {
                        let stem = Path::new(&item.filename)
                            .file_stem().and_then(|s| s.to_str()).unwrap_or("image");
                        payload.push((folder.join(format!("{stem}-nobg.png")), bytes));
                    }
                }
                continue;
            }
            for (filename, bytes) in layers {
                payload.push((folder.join(filename), bytes));
            }
        }

        if payload.is_empty() {
            self.toasts.error("Nothing to save — process the image first");
            return;
        }
        let file_count = payload.len();
        self.toasts.info(format!("Saving {file_count} file(s)..."));
        let tx = self.batch.bg_io.save_done_tx.clone();
        spawn_save_prerendered(payload, tx);
    }

    /// No checkboxes selected — save just the currently-viewed result via a
    /// save-as dialog. Suggests a `<stem>-nobg.png` name based on the source.
    fn save_current_to_file(&mut self) {
        let Some(item) = self.batch.selected_item() else { return };
        let Some(rgba) = item.result_rgba.clone() else { return };
        let default_name = Path::new(&item.filename).file_stem()
            .and_then(|s| s.to_str())
            .map(|stem| format!("{stem}-nobg.png"))
            .unwrap_or_else(|| "result-nobg.png".to_string());
        let baked = item.bake_export_bg(&rgba);
        self.save_rgba_with_dialog(baked, &default_name);
    }

    /// Save one specific batch item (by index) via a save-as dialog. Used by
    /// the sidebar's per-row save button, which needs a non-selection-based
    /// entry point into the same encode-on-background pipeline as
    /// `save_current_to_file`.
    pub(crate) fn save_item_to_file(&mut self, idx: usize) {
        let (baked, default_name) = {
            let Some(item) = self.batch.items.get(idx) else { return };
            let Some(rgba) = item.result_rgba.as_ref() else { return };
            let stem = Path::new(&item.filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            (item.bake_export_bg(rgba), format!("{stem}-nobg.png"))
        };
        self.save_rgba_with_dialog(baked, &default_name);
    }

    /// Shared tail for the two single-item save paths: open the PNG save-as
    /// dialog, kick off the encode+write on a background thread. The caller
    /// has already baked any per-item bg into the rgba.
    fn save_rgba_with_dialog(
        &mut self,
        rgba: Arc<image::RgbaImage>,
        default_name: &str,
    ) {
        let Some(path) = self.system.save_png_dialog(
            self.last_open_dir.as_deref(),
            default_name,
        ) else { return };
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
                Some((item.filename.clone(), item.bake_export_bg(rgba)))
            })
            .collect();
        let Some(folder) = self.system.pick_folder_dialog(
            self.last_open_dir.as_deref(),
            "Save Selected \u{2014} Choose Folder",
        ) else { return };
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
        let split = self.settings.export_split_layers;
        let mut paths: Vec<PathBuf> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(item) = self.batch.find_by_id(*id) {
                match super::drag_export::prepare_for_drag(item, split) {
                    Ok(mut ps) => paths.append(&mut ps),
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
            0 => (
                self.batch.selected_item().and_then(|i| i.result_rgba.clone()),
                None,
            ),
            1 => (Some(selected_with_result[0].clone()), None),
            n => (
                Some(selected_with_result[0].clone()),
                Some(format!(
                    "Copied 1 of {n} selected. Drag thumbnails out of the window to export multiple.")
                ),
            ),
        };

        let Some(rgba) = rgba_to_copy else { return };
        // Bake bg into the clipboard image so it matches the canvas.
        let rgba = self.batch.selected_item()
            .map(|item| item.bake_export_bg(&rgba))
            .unwrap_or(rgba);

        if self.system.copy_image(&rgba) {
            if let Some(msg) = multi_hint {
                self.toasts.info(msg);
            } else {
                self.set_temporary_status("Copied to clipboard");
            }
        } else {
            self.set_temporary_status("Could not copy to clipboard. Try saving instead.");
        }
    }


    pub fn handle_cancel(&mut self) {
        self.processor.cancels.request_global_cancel();
        // Esc cancels both the batch path and any in-flight eraser
        // stroke (SD on CPU takes minutes — needs the same escape valve).
        self.processor.cancel_all_inpaints();
    }

    /// Cancel only the in-flight inpaint stroke for `item_id`. Used by
    /// the canvas banner's Cancel button — `handle_cancel` cancels
    /// everything (batch + all inpaint), this scopes to one stroke.
    pub fn cancel_inpaint_for(&mut self, item_id: u64) {
        if self.processor.is_inpaint_in_flight(item_id) {
            self.processor.cancel_inpaint(item_id);
        }
    }

    pub fn handle_cancel_selected(&mut self) {
        let targets = self.batch.selected_ids_with_status(BatchStatus::Processing);
        if targets.is_empty() {
            return;
        }
        for id in &targets {
            self.processor.cancels.request_item_cancel(*id);
        }
        // Flip status immediately so thumbnail / canvas spinners stop. Any late
        // ImageDone or ImageError from the subprocess gets ignored in
        // `on_batch_item_done` (the Processing-only guard).
        for id in &targets {
            if let Some(item) = self.batch.find_by_id_mut(*id) {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
        }
        self.toasts.info(format!("Cancelling {} image(s)", targets.len()));
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

        // NOTE: do NOT touch `zoom_state` here. The callers that actually
        // change selection to the newly-added item (DnD inline single-file,
        // force-select-after-drain) own the full `zoom_state.reset()`, so
        // `previous_zoom` and `pan_offset` are cleared along with the flag.
        // Setting only `pending_fit_zoom` here would leak stale toggle-state
        // (previous_zoom) from the prior image into the fit logic, which can
        // mis-fire `canvas.rs`'s "toggle-back to previous_zoom" branch.
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

    /// Release an item's memory budget and greedily admit the next fitting items.
    fn admission_release_and_admit(&mut self, completed_id: u64) {
        let Some(ref mut ctrl) = self.processor.admission else { return; };
        ctrl.release(completed_id);

        let chain = self.settings.chain_mode;
        let mut streamed_ids = Vec::new();
        while let Some(next_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(next_id) {
                let Ok(bytes) = item.source.load_bytes() else { continue; };
                let chain_input = if chain { item.result_rgba.clone() } else { None };
                let tuple = (next_id, bytes, chain_input);
                item.status = BatchStatus::Processing;

                if let Some(ref tx) = self.processor.admission_tx {
                    if tx.send(tuple).is_err() {
                        break; // worker gone
                    }
                    streamed_ids.push(next_id);
                }
            }
        }

        let admission_complete = ctrl.is_complete();
        // ctrl borrow ends here so we can re-borrow processor.
        for id in streamed_ids {
            self.processor.track_streamed(id);
        }
        if admission_complete {
            self.processor.clear_admission();
        }
    }

    /// Demote Tier 2 (compressed RAM) history to Tier 3 (disk) under memory pressure.
    fn demote_history_to_disk(&mut self) {
        for item in &mut self.batch.items {
            for (seq, entry) in item.history.iter_mut().chain(item.redo_stack.iter_mut()).enumerate() {
                if matches!(entry.slot, HistorySlot::Compressed(_)) {
                    *entry = std::mem::take(entry).demote_to_disk(item.id, seq);
                }
            }
        }
    }

    /// Live-preview pump: dispatch debounced Tier 2 reruns + apply completed ones.
    /// Called once per frame at the start of `ui()` so the current frame renders
    /// with the newest available results.
    fn pump_live_preview(&mut self, ctx: &egui::Context) {
        if let Some(msg) = self.pending_onboarding_toast.take() {
            self.toasts.info(msg);
        }
        self.pump_inpaint_results(ctx);
        self.pump_download_manager(ctx);
        self.pump_runtime_install(ctx);
        self.recipe_drift_tripwire();

        // Dispatch phase: tick() invokes the closure for each item whose
        // debounce expired this frame. The closure borrows `batch.items`
        // mutably so `build_preview_inputs` can lazily cache an
        // `Arc<DynamicImage>` on the item (built once, reused across every
        // subsequent dispatch of the drag session — see `source_dyn`).
        let batch_items = &mut self.batch.items;
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

        // Covers the worker-running / not-yet-drained window. Same
        // self-extinguishing 50ms poll as the tex_pending loop in `logic()`.
        if self.processor.live_preview.has_in_flight() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    /// Snapshot the dispatch inputs for one preview job: original RGBA +
    /// decompressed tensors + settings + reusable edge mask if its
    /// line_strength still matches. Returns `None` to abort the dispatch
    /// when a required tensor cache isn't available (user must Process first).
    ///
    /// Lazily builds (and then reuses) `item.source_dyn` — the first dispatch
    /// of a drag session pays the ~15ms / 48 MB clone to wrap `source_rgba`
    /// in a `DynamicImage`, every subsequent dispatch is just an `Arc::clone`.
    fn build_preview_inputs(
        items: &mut [BatchItem],
        id: u64,
        kind: super::live_preview::PreviewKind,
    ) -> Option<super::live_preview::DispatchInputs> {
        use super::live_preview::{DispatchInputs, PreviewKind, decompress_seg};
        let item = items.iter_mut().find(|b| b.id == id)?;
        // Lazily build the cached `Arc<DynamicImage>` once per decode.
        if item.source_dyn.is_none() {
            let rgba = item.source_rgba.as_ref()?;
            item.source_dyn = Some(Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone())));
        }
        // Unwrap: we just set it above if it was None.
        let original = Arc::clone(item.source_dyn.as_ref().unwrap());
        let seg_tensor = item.cached_tensor.as_ref().and_then(decompress_seg);
        let edge_tensor = Self::edge_tensor_for_active_scale(item);
        // DualScale needs TWO edge tensors — the primary (active scale, used
        // as the Fine layer) and a secondary Bold layer. Only decompress the
        // Bold tensor when the style actually uses it; skipping this keeps
        // single-scale dispatches fast.
        let secondary_edge_tensor = matches!(item.settings.line_style, prunr_core::LineStyle::DualScale { .. })
            .then(|| Self::edge_tensor_for_scale(item, prunr_core::EdgeScale::Bold))
            .flatten();
        match kind {
            // Mask kind without a seg tensor is the filter-only path:
            // `run_preview` applies fill_style to the raw source. No tensor
            // required. Edge kind still needs an edge tensor (nothing to
            // preview without one).
            PreviewKind::Edge if edge_tensor.is_none() => return None,
            _ => {}
        }
        let cached_edge_mask = item.cached_edge_mask.as_ref().and_then(|(m, bits, scale)| {
            let strength_match = *bits == item.settings.line_strength.to_bits();
            let scale_match = *scale == item.settings.edge_scale;
            (strength_match && scale_match).then(|| m.clone())
        });
        let cached_masked_base = item.cached_masked_base.as_ref().and_then(|(base, recipe, model)| {
            let current_recipe = prunr_core::MaskRecipe::from(&item.settings.mask_settings());
            let seg_model_match = seg_tensor.as_ref().is_some_and(|s| s.model == *model);
            (*recipe == current_recipe && seg_model_match).then(|| base.clone())
        });
        Some(DispatchInputs {
            kind, original, settings: item.settings,
            seg_tensor, edge_tensor, secondary_edge_tensor,
            cached_edge_mask, cached_masked_base,
            correction: item.mask_correction.clone(),
        })
    }

    /// Decompress the edge tensor for the item's currently-selected scale,
    /// using the `volatile_edge_tensor` hot cache to skip zstd work during a
    /// drag. The tensor rides through as `Arc<Vec<f32>>` so hot hits are a
    /// pointer bump, not a 1.2 MB memcpy per dispatch.
    fn edge_tensor_for_active_scale(item: &mut BatchItem) -> Option<super::live_preview::EdgeTensor> {
        Self::edge_tensor_for_scale(item, item.settings.edge_scale)
    }

    /// Decompress the edge tensor for a specific scale. Uses the
    /// `volatile_edge_tensor` hot cache when the requested scale matches
    /// the cached one; otherwise pays the zstd decompress. Only updates the
    /// hot cache when `scale` matches the item's active scale — otherwise a
    /// DualScale dispatch that asks for Bold would evict the active-scale
    /// entry and make the next Edge tweak miss. Returns `None` when the
    /// multi-scale cache isn't populated (user must Process first).
    fn edge_tensor_for_scale(
        item: &mut BatchItem,
        scale: prunr_core::EdgeScale,
    ) -> Option<super::live_preview::EdgeTensor> {
        let hot_hit = item.volatile_edge_tensor.as_ref()
            .filter(|(s, _)| *s == scale)
            .map(|(_, arc)| arc.clone());
        if let Some(arc) = hot_hit {
            let (height, width) = item.cached_edge_tensors.as_ref()
                .map(|c| (c.height, c.width))?;
            return Some(super::live_preview::EdgeTensor { data: arc, height, width });
        }

        let (data, height, width) = {
            let cache = item.cached_edge_tensors.as_ref()?;
            let d = Arc::new(cache.decompress(scale)?);
            (d, cache.height, cache.width)
        };
        if scale == item.settings.edge_scale {
            item.volatile_edge_tensor = Some((scale, data.clone()));
        }
        Some(super::live_preview::EdgeTensor { data, height, width })
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
            let (item_id, source, is_final) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else {
                    continue;
                };
                let new_rgba = Arc::new(r.rgba);
                item.result_rgba = Some(new_rgba.clone());
                // Filter-only mode (model=None) never clicks Process — the
                // first live preview result IS the processed result. Promote
                // a Pending item to Done so the canvas flips from source
                // view to result view. Processing / Done items untouched.
                if item.status == BatchStatus::Pending {
                    item.status = BatchStatus::Done;
                }
                // Mark pending so sync_selected_batch_textures doesn't also
                // spawn its own prep on this same frame.
                item.result_tex_pending = true;
                if let Some((mask, bits, scale)) = r.new_edge_mask {
                    item.cached_edge_mask = Some((mask, bits, scale));
                }
                if let Some((base, recipe, model)) = r.new_masked_base {
                    item.cached_masked_base = Some((base, recipe, model));
                }
                // Mark the dispatched MaskRecipe as applied so the
                // recipe-drift tripwire doesn't immediately re-fire on
                // a result it has already consumed.
                if let Some(applied) = item.applied_recipe.as_mut() {
                    applied.mask = r.applied_mask;
                }
                let switch = self.result_switch_id;
                Self::spawn_tex_prep(
                    new_rgba, item.id, format!("result_{}_{}", item.id, switch),
                    true, tex_prep_tx.clone(), ctx.clone(),
                );
                (item.id, item.source.clone(), r.is_final)
            };
            // On the drag-settled result only, regenerate the sidebar
            // thumbnail. Unlike nulling `thumb_texture` + letting the
            // sidebar re-request, we call request_thumbnail directly while
            // the old texture remains displayed — `pump_thumbnail_results`
            // swaps it atomically when the new thumb arrives, so the user
            // never sees a spinner-gap. Mid-drag results skip this entirely
            // (is_final false) to avoid regenerating thumbs every debounce.
            if is_final {
                let result_rgba = self.batch.find_by_id(item_id).and_then(|i| i.result_rgba.clone());
                if let Some(item) = self.batch.find_by_id_mut(item_id) {
                    item.thumb_pending = true;
                }
                self.batch.request_thumbnail(item_id, &source, result_rgba.as_ref());
            }
        }
        ctx.request_repaint();
    }

    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        self.evict_result_rgba_for_background_items(idx);
        self.restore_selected_result_from_history(idx);
        self.ensure_selected_source_decoded(idx);
        self.request_selected_textures(idx, ctx);
        self.show_original = false;
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
                WorkerResult::BackendReady(provider) => {
                    if !provider.is_empty() && self.settings.active_backend != provider {
                        self.settings.active_backend = provider;
                        self.settings.parallel_jobs = self.settings.default_jobs();
                    }
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
        let line_mode = self.batch.find_by_id(item_id)
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
        edge_cache: Option<super::worker::EdgeTensorCache>,
    ) {
        let is_selected = self.batch.is_selected(item_id);
        let Some(recipe_snapshot) = self.take_dispatch_recipe(item_id) else { return };

        // User-initiated cancel reverts to Pending (not Error) so caches and
        // recipe stay intact for a re-Process.
        if matches!(result.as_ref(), Err(e) if e == crate::subprocess::protocol::CANCELLED_ERR_MSG) {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
            self.admission_release_and_admit(item_id);
            self.refresh_batch_progress_status();
            return;
        }

        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        // Skip results for items that were already cancelled (reset to Pending).
        if item.status != BatchStatus::Processing {
            return;
        }
        let backend_update = item.apply_tier_result(result, tensor_cache, edge_cache, recipe_snapshot, is_selected);
        // Single completion event — useful for `RUST_LOG=prunr=debug` bug
        // reports (when did inference actually finish?) and as a stable
        // signal for any external observer (e.g. a smoke driver) that a
        // dispatch round-trip closed for this item.
        tracing::info!(item_id, status = ?item.status, "item processing complete");

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

    /// Recipe to stamp on a finished item. Pulled from the in-flight slot
    /// owned by `Processor` — name is `take_*` because each call removes
    /// the entry. Late deliveries after `drain_recipes` fall back to
    /// reconstructing from current item state; returns `None` when the item
    /// is no longer in the batch (already removed by the user).
    fn take_dispatch_recipe(&mut self, item_id: u64) -> Option<prunr_core::ProcessingRecipe> {
        if let Some(recipe) = self.processor.take_recipe(item_id) {
            return Some(recipe);
        }
        let model: prunr_core::ModelKind = self.settings.model
            .to_model_kind()
            .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
        let chain = self.settings.chain_mode;
        self.batch.find_by_id(item_id)
            .map(|b| b.settings.current_recipe(model, chain))
    }

    fn refresh_batch_progress_status(&mut self) {
        let report = self.batch.progress();
        self.status.stage = report.stage;
        self.status.pct = report.pct;
    }

    fn on_batch_complete(&mut self) {
        self.processor.drain_recipes();
        let counts = self.batch.status_counts();
        let still_processing = counts.processing > 0;
        if counts.errored > 0 {
            let first_err = self.batch.first_error_message().unwrap_or("unknown error");
            let msg = if counts.errored == 1 {
                format!("Image failed: {first_err}")
            } else {
                format!("{} image(s) failed — first: {first_err}", counts.errored)
            };
            self.status.text = msg.clone();
            self.toasts.warning(msg);
        } else if !still_processing {
            let msg = format!("All done \u{2014} {} images processed", counts.done);
            self.status.text = msg.clone();
            self.toasts.success(msg);
        }
    }

    fn on_cancelled(&mut self) {
        self.processor.drain_recipes();
        self.status.text = "Cancelled".to_string();
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
        let app_state = self.batch.app_state();
        if intents.remove_requested && matches!(app_state, AppState::Loaded | AppState::Done) {
            self.handle_remove_bg();
        }
        if intents.save_requested && app_state == AppState::Done {
            self.handle_save_selected();
        }
        if copy_requested && app_state == AppState::Done {
            self.handle_copy();
        }
        if intents.toggle_before_after && app_state == AppState::Done {
            self.show_original = !self.show_original;
        }
        if intents.fit_to_window { self.zoom_state.pending_fit_zoom = true; }
        if intents.actual_size   { self.zoom_state.pending_actual_size = true; }

        if intents.cancel_requested {
            self.apply_cancel_shortcut(ctx);
        }

        if intents.toggle_shortcuts { self.show_shortcuts = !self.show_shortcuts; }
        if intents.toggle_cli_help  { self.show_cli_help  = !self.show_cli_help;  }
        if intents.toggle_pipeline_flow { self.show_pipeline_flow = !self.show_pipeline_flow; }
        if intents.toggle_settings  { self.toggle_settings_panel(ctx); }

        if intents.nav_prev { self.navigate_batch(ctx, NavDir::Prev); }
        if intents.nav_next { self.navigate_batch(ctx, NavDir::Next); }

        if intents.toggle_sidebar     { self.sidebar_hidden     = !self.sidebar_hidden; }
        if intents.toggle_adjustments { self.adjustments_hidden = !self.adjustments_hidden; }

        if intents.undo_requested        { self.handle_undo(ctx); }
        if intents.redo_requested        { self.handle_redo(ctx); }
        if intents.preset_undo_requested { self.swap_preset_history(HistoryDir::Undo); }
        if intents.preset_redo_requested { self.swap_preset_history(HistoryDir::Redo); }
        if intents.screenshot_requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }

        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }
    }

    /// Escape dismisses the topmost interruptable state. Priority order:
    /// in-flight eraser stroke → active batch processing → open modal
    /// (settings / shortcuts / cli help). Eraser strokes win over batch
    /// because the canvas banner is the most prominent active operation
    /// when both are running, and SD strokes on CPU eat minutes of work.
    fn apply_cancel_shortcut(&mut self, ctx: &egui::Context) {
        if self.processor.any_inpaint_in_flight() {
            self.processor.cancel_all_inpaints();
        } else if self.batch.status_counts().processing > 0 {
            self.handle_cancel();
            self.processor.clear_admission();
            for item in &mut self.batch.items {
                if item.status == BatchStatus::Processing {
                    item.status = BatchStatus::Pending;
                }
            }
            self.status.text = "Cancelled".to_string();
        } else if self.show_settings {
            self.close_settings(ctx);
        } else if self.show_shortcuts {
            self.show_shortcuts = false;
        } else if self.show_cli_help {
            self.show_cli_help = false;
        } else if self.show_pipeline_flow {
            self.show_pipeline_flow = false;
        }
    }

    fn toggle_settings_panel(&mut self, ctx: &egui::Context) {
        if self.show_settings {
            self.close_settings(ctx);
        } else {
            self.open_settings(ctx);
        }
    }

    pub(crate) fn open_settings(&mut self, ctx: &egui::Context) {
        self.show_settings = true;
        self.settings_opened_at = ctx.input(|i| i.time);
        self.hardware_install_cache = super::hardware_cache::HardwareInstallCache::refresh();
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
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.source_rgba = Some(rgba);
                // Live-preview DynamicImage cache is built from source_rgba;
                // invalidate so the next dispatch rebuilds against the new one.
                item.source_dyn = None;
                item.decode_pending = false;
                decode_arrived = true;
            }
        }
        if decode_arrived {
            // Freshly-decoded RGBA for the viewed item: clear zoom state
            // entirely (not just re-arm the flag). A bare flag keeps
            // previous_zoom / pan_offset around, which lets `canvas.rs`'s
            // toggle-back branch fire against the old image's state.
            self.zoom_state.reset();
            self.sync_selected_batch_textures(ctx);
        }

        // `sync_selected_batch_textures` calls
        // `evict_result_rgba_for_background_items`, which races any tex_prep
        // running for a non-selected item — gate sync on the selected
        // item's own arrival.
        let mut tex_arrived_for_selected = false;
        while let Ok((item_id, name, color_image, is_result)) = self.batch.bg_io.tex_prep_rx.try_recv() {
            let tex = ctx.load_texture(name, color_image, egui::TextureOptions::default());
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                if is_result {
                    item.result_texture = Some(tex);
                    item.result_tex_pending = false;
                    tracing::info!(item_id, kind = "result", "texture uploaded");
                } else {
                    item.source_texture = Some(tex);
                    item.source_tex_pending = false;
                    tracing::info!(item_id, kind = "source", "texture uploaded");
                }
                if self.batch.is_selected(item_id) {
                    tex_arrived_for_selected = true;
                }
            }
        }
        if tex_arrived_for_selected {
            self.sync_selected_batch_textures(ctx);
        }

        // Drain filter-only Process results (model=None path).
        let mut filter_only_arrived = false;
        while let Ok((item_id, result)) = self.batch.bg_io.filter_only_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            match result {
                Ok(rgba) => {
                    item.result_rgba = Some(rgba);
                    item.result_texture = None;
                    item.thumb_texture = None;
                    item.status = BatchStatus::Done;
                }
                Err(msg) => {
                    item.status = BatchStatus::Error(msg);
                }
            }
            filter_only_arrived = true;
        }
        if filter_only_arrived {
            self.result_switch_id += 1;
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
                self.zoom_state.reset();
                self.sync_selected_batch_textures(ctx);
            }
            if self.settings.auto_process_on_import && self.batch.next_id > id_floor {
                self.process_items(|item| item.id >= id_floor);
            }
        }
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let count = self.batch.items.len();
        let selected_name = if count < 2 {
            self.batch.selected_item().map(|i| i.filename.as_str())
        } else {
            None
        };
        let unchanged = match (&self.title_state, count, selected_name) {
            (TitleState::Batch(n), c, _) if c >= 2 => *n == c,
            (TitleState::Single(s), c, Some(name)) if c < 2 => s == name,
            (TitleState::Empty, c, None) if c < 2 => true,
            _ => false,
        };
        if unchanged { return; }

        let (new_state, title) = if count >= 2 {
            (TitleState::Batch(count), format!("Prunr \u{2014} {count} images"))
        } else if let Some(name) = selected_name {
            (TitleState::Single(name.to_string()), format!("Prunr \u{2014} {name}"))
        } else {
            (TitleState::Empty, "Prunr".to_string())
        };
        self.title_state = new_state;
        ctx.send_viewport_cmd(ViewportCommand::Title(title));
    }
}

impl Drop for PrunrApp {
    fn drop(&mut self) {
        self.processor.cancels.request_global_cancel();
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
        drain_screenshot_replies(ctx);
        self.drain_background_channels(ctx);
        self.update_window_title(ctx);
        self.status.tick();
        if self.batch.selected_item().is_some_and(|i| i.source_texture.is_none()) {
            self.sync_selected_batch_textures(ctx);
        }
        // Keep the event loop awake while any async texture / decode work is
        // pending. `ctx.request_repaint()` from the tex_prep / decode threads
        // is supposed to wake egui, but some compositors (notably Wayland)
        // drop thread-initiated wake-ups when the window is idle, leaving the
        // canvas stuck on the old image until a mouse event fires. Polling
        // from the UI thread is reliable; it costs one frame per 50ms while
        // async work is in flight, then self-extinguishes. See item 6 in
        // `.planning/phases/12-ux-refinement-and-bugs/12-01-FINDINGS.md` for
        // the full incident analysis.
        let any_tex_pending = self.batch.items.iter().any(|it| {
            it.result_tex_pending || it.source_tex_pending || it.decode_pending
        });
        if any_tex_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
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
            stroke: egui::Stroke::new(theme::STROKE_DEFAULT, egui::Color32::from_rgb(0x2a, 0x2a, 0x2a)),
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
        let height = theme::CHIP_HEIGHT * 2.0 + theme::SPACE_XS + theme::SPACE_SM * 2.0;
        let mut toolbar_change = adjustments_toolbar::ToolbarChange::default();
        let is_processing = self.batch.app_state() == AppState::Processing;
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
                let brush_state_ref = &mut self.brush_state;
                let item = &mut self.batch.items[idx];
                // Inpaint mode operates on the source image directly — no
                // cached seg tensor required. For seg-removal models the
                // brush corrects the AI mask so it needs the tensor.
                let brush_available = settings_ref.model.is_inpaint() || item.cached_tensor.is_some();
                let has_bg_image = item.bg_image.is_some();
                let bg_image_label = item.bg_image.as_deref()
                    .and_then(|bg| bg.source_path.as_deref())
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str());
                toolbar_change = adjustments_toolbar::render(
                    ui,
                    &mut item.settings,
                    settings_ref,
                    &mut item.applied_preset,
                    brush_state_ref,
                    brush_available,
                    is_processing,
                    has_bg_image,
                    bg_image_label,
                );
            });
        if toolbar_change.clear_correction_requested {
            if let Some(idx) = self.batch.selected_idx_clamped() {
                self.batch.items[idx].clear_correction();
            }
        }
        self.apply_toolbar_change(ui.ctx(), toolbar_change, pre_apply_snapshot);
    }

    fn apply_toolbar_change(
        &mut self,
        ctx: &egui::Context,
        toolbar_change: adjustments_toolbar::ToolbarChange,
        pre_apply_snapshot: PresetSnapshot,
    ) {
        use crate::gui::knob_catalog::DispatchKind;
        use crate::gui::live_preview::PreviewKind;

        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        if toolbar_change.preset_applied {
            HistoryManager::push_preset(&mut self.batch.items[idx], pre_apply_snapshot);
            // Preset apply copied an ItemSettings into the item — including
            // any captured bg_image_hash. Reload the matching bytes from
            // the persisted path map (or clear if missing) so the recipe
            // diff invariant holds and the canvas paints the right image.
            self.reconcile_bg_image_after_preset(idx);
        }
        if toolbar_change.model_changed {
            self.settings.save();
            self.toasts.info(format!(
                "{} loaded",
                crate::gui::views::model_name(self.settings.model),
            ));
        }
        if toolbar_change.brush_settings_committed {
            self.settings.save();
        }
        if let Some(req) = toolbar_change.open_model_store {
            self.model_store = Some(req);
        }
        if toolbar_change.pick_bg_image {
            self.handle_pick_bg_image(idx, ctx);
        } else if toolbar_change.clear_bg_image {
            self.batch.items[idx].clear_bg_image();
            ctx.request_repaint();
        }

        self.batch.items[idx].apply_cache_impact(toolbar_change.cache_impact);

        let dispatch = self.resolve_auto_dispatch(idx, &toolbar_change);
        let item_id = self.batch.items[idx].id;
        let is_done = self.batch.items[idx].status == BatchStatus::Done;

        match dispatch {
            DispatchKind::None => {}
            DispatchKind::Render => {}
            DispatchKind::LivePreviewMask => {
                if self.settings.live_preview {
                    self.processor.live_preview.mark_tweak(item_id, PreviewKind::Mask);
                    if toolbar_change.commit {
                        self.processor.live_preview.flush(item_id);
                        ctx.request_repaint();
                    } else {
                        ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
                    }
                }
            }
            DispatchKind::LivePreviewEdge => {
                if self.settings.live_preview {
                    self.processor.live_preview.mark_tweak(item_id, PreviewKind::Edge);
                    if toolbar_change.commit {
                        self.processor.live_preview.flush(item_id);
                        ctx.request_repaint();
                    } else {
                        ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
                    }
                }
            }
            DispatchKind::SubprocessAddEdge | DispatchKind::SubprocessFullPipeline => {
                if is_done {
                    self.process_items(|item| item.id == item_id);
                }
            }
        }

        if toolbar_change.render_repaint {
            // bg paints as a GPU rect behind the transparent result texture —
            // no CPU composite, no texture rebuild. Also fires on preset
            // applies so a no-op preset still repaints for the commit toast.
            ctx.request_repaint();
        }
    }

    /// Resolve the auto-fire dispatch. Starts from the aggregated catalog
    /// dispatch (populated only by `auto_trigger_on_commit` knobs), then
    /// refines context-sensitive knobs with item state (cached tensors).
    /// Preset applies reduce to the actual recipe diff — a no-op preset
    /// pick returns `None`, so a trivial re-apply doesn't spawn a subprocess.
    fn resolve_auto_dispatch(
        &self,
        idx: usize,
        tc: &adjustments_toolbar::ToolbarChange,
    ) -> crate::gui::knob_catalog::DispatchKind {
        use crate::gui::knob_catalog::{self, LineModeChange};
        use prunr_core::RequiredTier;
        let item = &self.batch.items[idx];
        let cached_seg = item.cached_tensor.is_some();
        let cached_edge = item.cached_edge_tensors.is_some();

        let mut dispatch = tc.auto_dispatch;

        if let Some(from) = tc.line_mode_from {
            let change = LineModeChange { from, to: item.settings.line_mode };
            dispatch = dispatch.max(knob_catalog::line_mode_spec(change, cached_edge).dispatch);
        }
        if tc.input_transform_changed {
            dispatch = dispatch.max(
                knob_catalog::input_transform_spec(cached_seg).dispatch,
            );
        }

        if tc.preset_applied {
            // `None` model = filter-only mode; pick an arbitrary ModelKind for
            // the diff since the seg stage is skipped regardless.
            let model = self
                .settings
                .model
                .to_model_kind()
                .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
            let preset_dispatch = match &item.applied_recipe {
                None => knob_catalog::DispatchKind::SubprocessFullPipeline,
                Some(old) => {
                    let new = item.settings.current_recipe(model, self.settings.chain_mode);
                    match prunr_core::resolve_tier(old, &new) {
                        RequiredTier::Skip | RequiredTier::CompositeOnly => {
                            knob_catalog::DispatchKind::None
                        }
                        RequiredTier::EdgeRerun => knob_catalog::DispatchKind::LivePreviewEdge,
                        RequiredTier::MaskRerun => knob_catalog::DispatchKind::LivePreviewMask,
                        RequiredTier::AddEdgeInference => {
                            knob_catalog::DispatchKind::SubprocessAddEdge
                        }
                        RequiredTier::FullPipeline => {
                            knob_catalog::DispatchKind::SubprocessFullPipeline
                        }
                    }
                }
            };
            dispatch = dispatch.max(preset_dispatch);
        }
        dispatch
    }

    fn render_modal_overlays(&mut self, ctx: &egui::Context) {
        if self.show_shortcuts && shortcuts::render(ctx) {
            self.show_shortcuts = false;
        }
        if self.show_cli_help && cli_help::render(ctx, &mut self.toasts) {
            self.show_cli_help = false;
        }
        if self.show_pipeline_flow && pipeline_flow::render(ctx) {
            self.show_pipeline_flow = false;
        }
        if self.show_settings {
            settings::render(ctx, self);
        }
        if self.model_store.is_some() && model_store::render(ctx, self) {
            self.model_store = None;
        }
        if let Some(id) = self.pending_license_request {
            let (close, accepted) = model_store::render_license_dialog(ctx, id);
            if accepted {
                self.settings.accept_license(id);
                self.download_manager.start_download(id);
                self.pending_license_request = None;
            } else if close {
                self.pending_license_request = None;
            }
        }
        self.maybe_evaluate_runtime_prompt();
        if let Some(rt) = self.runtime_prompt {
            use super::views::runtime_prompt::{RuntimePromptAction, render_runtime_prompt};
            if let Some(action) = render_runtime_prompt(ctx, rt) {
                self.runtime_prompt = None;
                match action {
                    RuntimePromptAction::Install => {
                        let h = crate::runtime_install::start_install(rt);
                        self.runtime_install = Some(RuntimeInstallProgress {
                            runtime: rt,
                            rx: h.events,
                            cancel: h.cancel,
                            last_event: crate::runtime_install::InstallEvent::Preparing,
                        });
                    }
                    RuntimePromptAction::NotNow => {
                        self.settings.snooze_runtime_prompt(rt, RUNTIME_PROMPT_SNOOZE_DAYS);
                    }
                    RuntimePromptAction::OpenSettings => {
                        self.open_settings(ctx);
                    }
                }
            }
        }
        // Toasts — rendered last as foreground overlay.
        self.toasts.show(ctx);
    }
}

/// Drain any `Event::Screenshot` replies and persist them as PNGs. The
/// directory is `$PRUNR_SCREENSHOT_DIR` (test-harness sets this) or
/// `<temp>/prunr-screenshots/`. Filename is the unix-millis timestamp
/// so a scenario that fires Shift+F12 multiple times never collides.
fn drain_screenshot_replies(ctx: &egui::Context) {
    let images: Vec<std::sync::Arc<egui::ColorImage>> = ctx.input(|i| {
        i.events.iter().filter_map(|e| match e {
            egui::Event::Screenshot { image, .. } => Some(std::sync::Arc::clone(image)),
            _ => None,
        }).collect()
    });
    if images.is_empty() { return; }
    let dir = std::env::var_os("PRUNR_SCREENSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("prunr-screenshots"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(%e, ?dir, "screenshot dir create failed");
        return;
    }
    for image in images {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("{stamp}.png"));
        let [w, h] = image.size;
        let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
        let Some(rgba) = image::RgbaImage::from_raw(w as u32, h as u32, bytes) else {
            tracing::warn!("screenshot dims mismatch — skipping");
            continue;
        };
        match rgba.save(&path) {
            Ok(()) => tracing::info!(?path, "screenshot saved"),
            Err(e) => tracing::warn!(%e, ?path, "screenshot save failed"),
        }
    }
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

/// Write a vec of pre-encoded `(path, bytes)` pairs on a background thread.
/// Used by the layers-save path — rendering happened on the UI thread (from
/// cached tensors), this just does disk writes.
fn spawn_save_prerendered(payload: Vec<(PathBuf, Vec<u8>)>, tx: mpsc::Sender<String>) {
    std::thread::spawn(move || {
        let mut saved = 0usize;
        let mut failed = 0usize;
        for (path, bytes) in payload {
            match std::fs::write(&path, &bytes) {
                Ok(()) => saved += 1,
                Err(_) => failed += 1,
            }
        }
        let msg = if failed > 0 {
            format!("Saved {saved}, failed {failed}")
        } else {
            format!("Saved {saved} file(s)")
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
    /// AddEdgeInference items: seg tensor cached, run DexiNed only.
    tier_add_edge: HashSet<u64>,
    skip_count: usize,
}

impl ClassifiedTiers {
    /// Union of tier1 + tier2 + add-edge — the set of items that will actually
    /// be reprocessed this batch (history seeding, progress counting).
    fn all_process_ids(&self) -> HashSet<u64> {
        self.tier1.iter()
            .chain(self.tier2.iter())
            .chain(self.tier_add_edge.iter())
            .copied().collect()
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
    toggle_pipeline_flow: bool,
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
    screenshot_requested: bool,
}

fn collect_shortcut_intents(ctx: &egui::Context) -> ShortcutIntents {
    // Suppress bare-key shortcuts when any widget has focus (e.g., hex color input).
    let text_focused = ctx.memory(|m| m.focused().is_some());
    let mut s = ShortcutIntents::default();
    ctx.input(|i| {
        // Fresh-press filter: egui's `key_pressed` includes OS key-repeat
        // events, which flip toggle-shortcuts on/off many times a second if
        // the user holds the key for more than the repeat-delay. Bind toggle
        // intents via this instead.
        let fresh = |k: Key| i.events.iter().any(|e| matches!(
            e,
            egui::Event::Key { key, pressed: true, repeat: false, .. } if *key == k,
        ));

        // Modifier shortcuts always work, even with a text field focused.
        if i.modifiers.command && i.key_pressed(Key::O) { s.open_requested = true; }
        if i.modifiers.command && i.key_pressed(Key::R) { s.remove_requested = true; }
        if i.modifiers.command && i.key_pressed(Key::S) { s.save_requested = true; }
        if i.key_pressed(Key::Escape)                   { s.cancel_requested = true; }
        if fresh(Key::F1)                               { s.toggle_shortcuts = true; }
        if fresh(Key::F2)                               { s.toggle_cli_help = true; }
        if fresh(Key::F3)                               { s.toggle_pipeline_flow = true; }
        // Test-harness hook (also doubles as a bug-report capture).
        // Writes to `$PRUNR_SCREENSHOT_DIR` (default `<temp>/prunr-screenshots/`).
        if i.modifiers.shift && fresh(Key::F12)         { s.screenshot_requested = true; }
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
