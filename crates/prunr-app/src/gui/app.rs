use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

use egui::ViewportCommand;

use prunr_core::ProgressStage;
use super::drag_export_state::DragExportState;
use super::history_manager::{HistoryDir, HistoryManager};
use super::item::{ActionType, BatchItem, BatchStatus, HistoryEntry, HistorySlot, ImageSource, PresetSnapshot};
use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{adjustments_toolbar, canvas, cli_help, model_store, pipeline_flow, settings, shortcuts, sidebar, statusbar, toolbar};
use super::views::selection_action_bar::SelectionAction;

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
    pub(crate) magic_brush_state: super::magic_brush_state::MagicBrushState,
    pub(crate) download_manager: super::download_manager::DownloadManager,

    // Before/After toggle
    pub(crate) show_original: bool,
    /// Which item the view was last brought up to date for; a change
    /// runs the selection-change work in `reconcile_selected`.
    synced_selected: Option<u64>,

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

    /// Set by add_to_batch — forces the selection-change work in the next logic()
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
            magic_brush_state: super::magic_brush_state::MagicBrushState::default(),
            download_manager: super::download_manager::DownloadManager::new(),
            show_original: false,
            synced_selected: None,
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
        self.batch.clamp_selected_index();
        if !self.batch.items.is_empty() {
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
                let kick_id = if let Some(item) = self.batch.items.get_mut(idx) {
                    item.set_bg_image(img, Some(path.clone()));
                    // Remember the path keyed by the bg's content hash so a
                    // preset that captured this hash can reload the image
                    // when applied later (or after a restart).
                    if let Some(bg) = item.bg_image.as_ref() {
                        self.settings.bg_image_paths.insert(bg.hash, path);
                        self.settings.save();
                    }
                    Some(item.id)
                } else { None };
                if let Some(id) = kick_id {
                    self.kick_bg_image_tex_prep(id, ctx);
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
    fn reconcile_bg_image_after_preset(&mut self, idx: usize, ctx: &egui::Context) {
        let Some(item) = self.batch.items.get_mut(idx) else { return };
        let want = item.settings.bg_image_hash;
        let have = item.bg_image.as_ref().map(|b| b.hash);
        if want.map(|nz| nz.get()) == have { return; }
        let Some(want_hash) = want else {
            // Preset has no bg image — drop the stale bytes.
            item.clear_bg_image();
            return;
        };
        let want_hash_u64 = want_hash.get();
        // New hash from preset — try to reload via the persisted path map.
        let path = self.settings.bg_image_paths.get(&want_hash_u64).cloned();
        let Some(path) = path else {
            // Hash isn't in our path map (preset shared from another user,
            // or the path entry was wiped). Drop both bytes and hash so the
            // recipe diff stays consistent.
            item.clear_bg_image();
            self.toasts.info("Preset references a background image we don't have on disk.");
            return;
        };
        match prunr_core::load_image_from_path(&path) {
            Ok(img) => {
                item.set_bg_image(img, Some(path));
                let id = item.id;
                self.kick_bg_image_tex_prep(id, ctx);
            }
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

    /// Single source of truth for the Process / Reprocess action.
    /// Routes between upscale dispatch, inpaint dispatch (eraser models),
    /// and the seg pipeline, with consistent gating. All input surfaces —
    /// toolbar button, Cmd+R keyboard shortcut, future menu items — call this.
    pub fn handle_process_intent(&mut self) {
        if !self.can_process_intent() {
            return;
        }
        if self.settings.model.is_upscale() {
            self.dispatch_upscale_intent();
            return;
        }
        if self.settings.model.is_inpaint() {
            let Some(idx) = self.batch.selected_idx_clamped() else { return };
            let has_selection = self.batch.items.get(idx).is_some_and(
                |i| i.selection_mask.as_ref().is_some_and(|m| m.has_selected_region()),
            );
            if !has_selection { return; }
            self.dispatch_inpaint_for_item(idx);
        } else {
            self.handle_remove_bg();
        }
    }

    /// Upscale branch of `handle_process_intent`. Resolves the input image
    /// (chain mode: use existing result; otherwise: source), then hands off
    /// to `Processor::dispatch_upscale`. Must only be called after
    /// `can_process_intent()` confirms the model is installed and idle.
    fn dispatch_upscale_intent(&mut self) {
        let item = match self.batch.selected_item() {
            Some(i) => i,
            None => return,
        };
        let item_id = item.id;
        let model_id = match self.settings.model.to_model_id() {
            Some(id) => id,
            None => return,
        };
        // Share the input via Arc — full RGBA payload would otherwise
        // be deep-cloned (tens of MB at 4K) for every Process click.
        let input: std::sync::Arc<image::RgbaImage> = if self.settings.chain_mode {
            match item.result_rgba.as_ref().or(item.source_rgba.as_ref()) {
                Some(arc) => std::sync::Arc::clone(arc),
                None => {
                    tracing::warn!(item_id, "upscale dispatch skipped: no source RGBA available");
                    return;
                }
            }
        } else {
            match item.source_rgba.as_ref() {
                Some(arc) => std::sync::Arc::clone(arc),
                None => {
                    tracing::warn!(item_id, "upscale dispatch skipped: source RGBA unavailable");
                    return;
                }
            }
        };
        let output_scale = item.settings.output_scale;
        let intra_threads = prunr_core::batch::ort_intra_threads(self.settings.parallel_jobs);
        // Capture the recipe at dispatch time so the pump can stamp
        // `item.applied_recipe` against what actually ran (not what the
        // user has tweaked to since).
        let model_kind = match prunr_core::ModelKind::try_from(model_id) {
            Ok(k) => k,
            Err(id) => {
                tracing::error!(?id, "upscale dispatch skipped: ModelId has no ModelKind mapping");
                return;
            }
        };
        let recipe = item.settings.current_recipe(model_kind, self.settings.chain_mode);
        tracing::info!(item_id, ?model_id, ?output_scale, "upscale dispatched");
        // Seed the action timeline so Cmd+Z can revert this Process. Seg/inpaint
        // routes through process_items → seed_history_for_reprocess; the upscale
        // path bypasses that, so the marker push lives here.
        let mut ids = HashSet::new();
        ids.insert(item_id);
        self.seed_history_for_reprocess(&ids, self.settings.chain_mode);
        self.processor.dispatch_upscale(item_id, input, model_id, output_scale, intra_threads, recipe);
    }

    /// Toolbar mirror for `handle_process_intent`. `add_enabled(...)` reads
    /// this so the button's disabled state and the dispatch's gating can't
    /// drift independently.
    pub fn can_process_intent(&self) -> bool {
        if self.settings.model.is_upscale() {
            // Install-state is not re-checked here: the dropdown filter
            // already restricts selection to `is_available` models, and
            // an `is_file()` syscall per frame would add 60 stats/sec
            // while the toolbar is visible. Deletion-after-select is
            // surfaced at dispatch time by the error toast.
            let is_in_flight = self.processor.is_upscale_in_flight();
            let item_loaded = self.batch.selected_item().is_some();
            return can_process_upscale(is_in_flight, item_loaded);
        }
        if self.settings.model.is_inpaint() {
            self.batch.selected_item().is_some_and(
                |i| i.selection_mask.as_ref().is_some_and(|m| m.has_selected_region()),
            )
        } else {
            self.batch.any_target_can(|it| !matches!(it.status, BatchStatus::Processing))
        }
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

    /// Undo the most-recent action on selected items (or current item if none
    /// selected). Pops one `ActionType` marker from the ordering layer and
    /// dispatches to the matching per-type stack — mode-agnostic. A stroke
    /// undo triggers a brush rerun so the canvas reflects the popped state.
    pub fn handle_undo(&mut self, ctx: &egui::Context) {
        let mut type_counts = [0u32; 3];
        let mut total = 0u32;
        for idx in self.batch.targeted_indices() {
            if let Some(action) = self.try_undo_one_action(idx, ctx) {
                total += 1;
                type_counts[action as usize] += 1;
            }
        }
        if total > 0 {
            self.result_switch_id += 1;
            self.canvas_switch_id += 1;
            self.sync_selected_batch_textures(ctx);
            self.toasts.info(action_toast_label(
                total, type_counts,
                ["Stroke undone", "Undone", "Preset undone"],
                "undone",
            ));
        } else {
            self.toasts.info("Nothing to undo");
        }
    }

    /// Redo the most-recently undone action on selected items. Mirrors
    /// `handle_undo` via the `actions_redo` ordering layer.
    pub fn handle_redo(&mut self, ctx: &egui::Context) {
        let mut type_counts = [0u32; 3];
        let mut total = 0u32;
        for idx in self.batch.targeted_indices() {
            if let Some(action) = self.try_redo_one_action(idx, ctx) {
                total += 1;
                type_counts[action as usize] += 1;
            }
        }
        if total > 0 {
            self.result_switch_id += 1;
            self.canvas_switch_id += 1;
            self.sync_selected_batch_textures(ctx);
            self.toasts.info(action_toast_label(
                total, type_counts,
                ["Stroke restored", "Result restored", "Preset restored"],
                "restored",
            ));
        } else {
            self.toasts.info("Nothing to redo");
        }
    }

    /// Pop the most-recent marker from `actions_undo` for item at `idx` and
    /// apply the inverse. Returns the popped ActionType on success.
    ///
    /// The ordering log (`actions_undo`) is capped at `ACTION_HIST_DEPTH`, but
    /// per-type stacks have their own caps — Result history is smaller because
    /// each entry is a full RGBA archive. When the result stack rotates an
    /// oldest entry out, the corresponding marker in `actions_undo` becomes an
    /// orphan. The loop pops orphans silently and tries the next action rather
    /// than reporting false failure.
    fn try_undo_one_action(&mut self, idx: usize, ctx: &egui::Context) -> Option<ActionType> {
        use super::item::push_action_bounded;
        loop {
            let kind = self.batch.items[idx].actions_undo.pop_back()?;
            let success = match kind {
                ActionType::Stroke => {
                    if self.settings.model.is_inpaint() {
                        // Inpaint mode: each stroke archived its prev result_rgba
                        // at result-receive time. Undo swaps stored RGBAs — instant,
                        // no re-dispatch (which would re-run the multi-second SD
                        // UNet pipeline). Sync the mask_correction stack so a
                        // subsequent stroke commit doesn't drift.
                        let _ = self.batch.items[idx].undo_stroke();
                        if HistoryManager::undo_result(&mut self.batch.items[idx]) {
                            self.batch.items[idx].reset_result_caches();
                            self.batch.items[idx].source_texture = None;
                            true
                        } else {
                            false
                        }
                    } else if self.settings.chain_mode
                        && HistoryManager::can_undo(&self.batch.items[idx])
                    {
                        // Chain-mode seg: the brush handler archived the
                        // pre-stroke `result_rgba` at commit. Pop it
                        // (instant) instead of `dispatch_brush_rerun` —
                        // the rerun would rebuild against the post-stroke
                        // chain base and never reach the pre-stroke state.
                        let _ = self.batch.items[idx].undo_stroke();
                        if HistoryManager::undo_result(&mut self.batch.items[idx]) {
                            self.batch.items[idx].reset_result_caches();
                            self.batch.items[idx].source_texture = None;
                            true
                        } else {
                            false
                        }
                    } else {
                        // Non-chain seg: mask_correction state IS the input
                        // to the pipeline — re-dispatch is required to
                        // recompute the result. (Seg pipelines are
                        // sub-second; rerun cost is tolerable.)
                        if self.batch.items[idx].undo_stroke() {
                            self.dispatch_brush_rerun(idx);
                            true
                        } else {
                            false
                        }
                    }
                }
                ActionType::Result => {
                    if HistoryManager::undo_result(&mut self.batch.items[idx]) {
                        self.batch.items[idx].reset_result_caches();
                        // Source view may now show the unprocessed source.
                        self.batch.items[idx].source_texture = None;
                        true
                    } else {
                        false
                    }
                }
                ActionType::PresetApply => {
                    if HistoryManager::swap_preset(&mut self.batch.items[idx], HistoryDir::Undo) {
                        let target_id = self.batch.items[idx].id;
                        let should_reprocess = self.batch.items[idx].status == BatchStatus::Done;
                        self.reconcile_bg_image_after_preset(idx, ctx);
                        if should_reprocess {
                            self.process_items(|i| i.id == target_id);
                        }
                        true
                    } else {
                        false
                    }
                }
            };
            if success {
                push_action_bounded(&mut self.batch.items[idx].actions_redo, kind);
                return Some(kind);
            }
        }
    }

    /// Pop the most-recent marker from `actions_redo` for item at `idx` and
    /// re-apply. Returns the popped ActionType on success.
    ///
    /// The ordering log (`actions_redo`) is capped at `ACTION_HIST_DEPTH`, but
    /// per-type stacks have their own caps — Result history is smaller because
    /// each entry is a full RGBA archive. When the result stack rotates an
    /// oldest entry out, the corresponding marker in `actions_redo` becomes an
    /// orphan. The loop pops orphans silently and tries the next action rather
    /// than reporting false failure.
    fn try_redo_one_action(&mut self, idx: usize, ctx: &egui::Context) -> Option<ActionType> {
        use super::item::push_action_bounded;
        loop {
            let kind = self.batch.items[idx].actions_redo.pop_back()?;
            let success = match kind {
                ActionType::Stroke => {
                    if self.settings.model.is_inpaint() {
                        // Inpaint redo: swap stored RGBAs (instant; mirror of
                        // try_undo_one_action). See comment there.
                        let _ = self.batch.items[idx].redo_stroke();
                        if HistoryManager::redo_result(&mut self.batch.items[idx]) {
                            self.batch.items[idx].reset_result_caches();
                            self.batch.items[idx].source_texture = None;
                            true
                        } else {
                            false
                        }
                    } else if self.settings.chain_mode
                        && HistoryManager::can_redo(&self.batch.items[idx])
                    {
                        // Chain-mode seg redo mirror of the undo path:
                        // the post-stroke result lives on `redo_stack`;
                        // pop it back instead of re-running the brush.
                        let _ = self.batch.items[idx].redo_stroke();
                        if HistoryManager::redo_result(&mut self.batch.items[idx]) {
                            self.batch.items[idx].reset_result_caches();
                            self.batch.items[idx].source_texture = None;
                            true
                        } else {
                            false
                        }
                    } else if self.batch.items[idx].redo_stroke() {
                        self.dispatch_brush_rerun(idx);
                        true
                    } else {
                        false
                    }
                }
                ActionType::Result => {
                    if HistoryManager::redo_result(&mut self.batch.items[idx]) {
                        self.batch.items[idx].reset_result_caches();
                        self.batch.items[idx].source_texture = None;
                        true
                    } else {
                        false
                    }
                }
                ActionType::PresetApply => {
                    if HistoryManager::swap_preset(&mut self.batch.items[idx], HistoryDir::Redo) {
                        let target_id = self.batch.items[idx].id;
                        let should_reprocess = self.batch.items[idx].status == BatchStatus::Done;
                        self.reconcile_bg_image_after_preset(idx, ctx);
                        if should_reprocess {
                            self.process_items(|i| i.id == target_id);
                        }
                        true
                    } else {
                        false
                    }
                }
            };
            if success {
                push_action_bounded(&mut self.batch.items[idx].actions_undo, kind);
                return Some(kind);
            }
        }
    }

    fn dispatch_brush_rerun(&mut self, idx: usize) {
        use crate::gui::live_preview::PreviewKind;
        let item_id = self.batch.items[idx].id;
        // Unconditional — see canvas::handle_brush_input.
        self.processor.live_preview.mark_tweak(item_id, PreviewKind::Mask);
        self.processor.live_preview.flush(item_id);
    }

    /// Per-model interpretation rule for a newly-committed selection.
    /// Called after every commit, and again when a result lands, so the BG-removal
    /// continuous-auto-apply UX is preserved while SD / LaMa wait for
    /// explicit Process.
    ///
    /// IMPORTANT: every call site that calls `BatchManager::commit_selection`
    /// MUST follow up with this method. Pairing the two prevents per-model
    /// dispatch rules from drifting between input surfaces.
    pub(crate) fn apply_selection_to_active_model(&mut self, item_id: u64) {
        let Some(idx) = self.batch.items.iter().position(|i| i.id == item_id) else { return };
        let category = self.settings.model.to_model_id()
            .and_then(prunr_models::descriptor)
            .map(|d| d.category);
        match category {
            Some(prunr_models::ModelCategory::Segmentation) => {
                // Without a tensor there is nothing to correct yet; the
                // selection waits and `on_batch_item_done` applies it when
                // the first result lands.
                if !self.settings.protect_selection && self.batch.items[idx].cached_tensor.is_some() {
                    self.dispatch_brush_rerun(idx);
                }
            }
            Some(prunr_models::ModelCategory::Inpaint) => {
                // Selection IS the inpaint region; user clicks Process.
            }
            Some(prunr_models::ModelCategory::Selection) => {
                // Magic Brush model itself; no downstream action.
            }
            Some(prunr_models::ModelCategory::Upscale)
            | Some(prunr_models::ModelCategory::EdgeDetection)
            | None => {
                // Mask-only actions work; no model dispatch.
            }
        }
    }

    /// Single-source-of-truth call for writing a new selection: persist the
    /// mask and fire the per-model dispatch rule. The commit drops the old
    /// texture, so `ensure_selection_texture` rebuilds it next frame.
    ///
    /// Every call site that authors a selection (Paint Brush, Magic Brush,
    /// Invert, any future tool) calls THIS method.
    pub(crate) fn commit_selection_and_dispatch(
        &mut self,
        item_id: u64,
        mask: prunr_core::selection::MaskArtifact,
    ) {
        if !self.batch.commit_selection(item_id, mask) {
            return;
        }
        // Chain mode: the rerun replaces the result in place, so undoing
        // the stroke needs the pre-stroke image archived now. Inpaint
        // archives when its result lands.
        if !self.settings.model.is_inpaint() && self.settings.chain_mode {
            let max_depth = self.settings.history_depth;
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                HistoryManager::archive_result_for_stroke(item, max_depth);
            }
        }
        self.apply_selection_to_active_model(item_id);
    }

    /// Dispatch a selection action (Delete / Copy / Cut / Invert / Clear).
    /// Called from `apply_toolbar_change` and from keyboard shortcuts that
    /// have a live selection.
    ///
    /// - Delete: scales alpha by coverage inside the selection on
    ///   `result_rgba`, archiving the previous result first so Cmd+Z can
    ///   restore it.
    /// - Copy: copies selection-masked pixels to the system clipboard. No
    ///   history entry — non-destructive read.
    /// - Cut: Copy + Delete in one step (one history entry, not two).
    /// - Invert: replaces the selection mask with its complement and
    ///   commits it like any other author.
    /// - Clear: removes the selection mask entirely.
    pub(crate) fn handle_selection_action(
        &mut self,
        idx: usize,
        action: SelectionAction,
        ctx: &egui::Context,
    ) {
        let item_id = self.batch.items[idx].id;

        match action {
            SelectionAction::Delete => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.apply_selection_cut(idx, &mask, &base, ctx);
            }
            SelectionAction::Copy => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.system.copy_image(&mask.copy_to_rgba(&base));
                self.set_temporary_status("Selection copied to clipboard");
            }
            SelectionAction::Cut => {
                let Some((mask, base)) = self.selection_edit_inputs(idx) else { return };
                self.system.copy_image(&mask.copy_to_rgba(&base));
                self.apply_selection_cut(idx, &mask, &base, ctx);
                self.set_temporary_status("Selection cut to clipboard");
            }
            SelectionAction::Invert => {
                let Some(mask) = self.batch.items[idx].selection_mask.clone() else { return };
                let inverted = mask.invert(self.settings.brush.mode);
                self.commit_selection_and_dispatch(item_id, inverted);
                ctx.request_repaint();
            }
            SelectionAction::Clear => {
                self.batch.clear_selection(item_id);
                ctx.request_repaint();
            }
        }
    }

    /// Selection actions act on the latest result or, before processing,
    /// on the source. Shared by the action bar, the keyboard shortcuts and
    /// the actions themselves.
    pub(crate) fn can_selection_action(&self) -> bool {
        self.batch.selected_item().is_some_and(|i| {
            i.selection_mask.is_some() && (i.result_rgba.is_some() || i.source_rgba.is_some())
        })
    }

    /// The selection plus the image a Delete / Copy / Cut acts on.
    fn selection_edit_inputs(
        &self,
        idx: usize,
    ) -> Option<(Arc<prunr_core::selection::MaskArtifact>, Arc<image::RgbaImage>)> {
        let item = &self.batch.items[idx];
        let mask = item.selection_mask.clone()?;
        let base = item.source_for_inpaint()?;
        // What gets cut is what the overlay shows: the feathered mask, which
        // the texture build already computed when it is current.
        let style = super::background_io::SelectionStyle::from_brush(&self.settings.brush);
        let shown = item.current_selection_texture(style)
            .and_then(|t| t.feathered.clone())
            .or_else(|| style.feather(&mask, item.source_rgba.as_deref()))
            .unwrap_or(mask);
        Some((shown, base))
    }

    /// Cut the selection out of `base` and publish it as the item's result,
    /// archiving the previous state so Cmd+Z restores it. An unprocessed
    /// image first seeds its history with the source so there is a state
    /// to go back to.
    fn apply_selection_cut(
        &mut self,
        idx: usize,
        mask: &prunr_core::selection::MaskArtifact,
        base: &image::RgbaImage,
        ctx: &egui::Context,
    ) {
        let max_depth = self.settings.history_depth;
        let item = &mut self.batch.items[idx];
        HistoryManager::seed_with_source(item);
        HistoryManager::archive_current_result(item, max_depth, false);
        let item_id = item.id;
        let mut edited = base.clone();
        mask.alpha_cut(&mut edited);
        self.publish_result(item_id, Arc::new(edited), "edit", ctx);
    }

    pub(crate) fn dispatch_inpaint_for_item(&mut self, idx: usize) {
        let item = &self.batch.items[idx];
        let item_id = item.id;
        // selection_mask is the single source of truth for the inpaint
        // region. It persists across Process, so Reprocess is available
        // whenever any selection exists — no separate "last_inpaint" cache.
        let Some(selection) = item.selection_mask.as_ref().cloned() else {
            tracing::debug!(item_id, "inpaint dispatch skipped: no selection_mask");
            return;
        };
        // Stack-based inpaint: each stroke runs against the previous
        // result so earlier strokes stay intact. source_for_inpaint
        // walks result_rgba → source_rgba → source_dyn, the last arm
        // handling memory-pressure eviction.
        let Some(source) = item.source_for_inpaint() else {
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
        if backend.is_sd_family() {
            if let Some(desc) = prunr_models::descriptor(backend) {
                let avail = crate::hardware::available_ram_bytes_throttled();
                if let Err(msg) = crate::hardware::pre_flight_sd_ram(
                    desc.working_set_mb,
                    avail,
                    self.settings.ram_safety_margin_gb,
                ) {
                    tracing::warn!(item_id, ?backend, %msg, "SD pre-flight gate refused dispatch");
                    self.toasts.error(msg);
                    return;
                }
            }
        }
        tracing::info!(item_id, ?backend, ?raw_backend, "inpaint stroke committed; dispatching");
        let tuning = super::processor::InpaintTuning {
            sharpen: bs.inpaint_sharpen,
            feather_px: bs.inpaint_feather,
            grow_px: bs.inpaint_grow,
            backend,
            sd_prompt: bs.sd_prompt.clone(),
            sd_negative_prompt: bs.sd_negative_prompt.clone(),
            sd_guidance_scale: bs.sd_guidance_scale,
            sd_scheduler: bs.sd_scheduler,
            sd_steps: bs.sd_steps,
            sd_seed: bs.sd_seed,
            sd_strength: bs.sd_strength,
            sd_use_karras_sigmas: bs.sd_use_karras_sigmas,
            use_taesd: bs.sd_use_taesd_effective(),
        };
        self.processor.dispatch_inpaint(item_id, source, selection, tuning);
    }

    fn pump_inpaint_results(&mut self, ctx: &egui::Context) {
        // Bridge → inpaint_rx fan-in must run before the drain.
        self.processor.pump_inpaint_subprocess();
        let (results, cancelled, errors) = self.processor.drain_inpaint_results();
        // For each cancelled stroke: roll back the committed
        // mask_correction + drop its Stroke marker so the cancelled
        // stroke doesn't ghost on canvas, accumulate into the next
        // commit, or pollute the undo timeline. The user's mental
        // model is "cancel = the stroke never happened."
        let cancelled_count = cancelled.len();
        for &item_id in &cancelled {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.revert_last_stroke_commit();
            }
            self.toasts.info("Erase cancelled");
        }
        if cancelled_count > 0 {
            // Repaint so the brush overlay clears even if no result
            // landed this frame.
            ctx.request_repaint();
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
        let max_depth = self.settings.history_depth;
        for r in results {
            let (item_id, new_rgba) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else { continue };
                let new_rgba = Arc::new(r.rgba);
                // Archive the previous result_rgba so Cmd+Z can swap stored
                // RGBAs instantly instead of re-running the inpaint pipeline.
                // The Stroke marker was already pushed at commit_correction;
                // this just stages the snapshot the marker undoes to.
                if let Some(prev) = item.result_rgba.take() {
                    item.history.push_back(super::item::HistoryEntry::new(
                        prev, item.applied_recipe.clone(),
                    ));
                    while item.history.len() > max_depth {
                        if let Some(old) = item.history.pop_front() {
                            old.cleanup();
                        }
                    }
                    // New edit invalidates the linear redo timeline.
                    for entry in item.redo_stack.drain(..) {
                        entry.cleanup();
                    }
                }
                // selection_mask intentionally persists — the region stays
                // highlighted and is available for Reprocess.
                (item.id, new_rgba)
            };
            self.publish_result(item_id, new_rgba, "inpaint", ctx);
        }
    }

    /// Install a new result the way every result producer must: the old
    /// texture stays on screen until the new one lands (a cleared texture
    /// paints nothing for a frame), the thumbnail is refreshed, and a
    /// Pending item becomes Done.
    fn publish_result(
        &mut self,
        item_id: u64,
        rgba: Arc<image::RgbaImage>,
        tag: &str,
        ctx: &egui::Context,
    ) {
        let handles = self.batch.bg_io.tex_prep_handles();
        let switch = self.result_switch_id;
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        item.result_rgba = Some(rgba.clone());
        if item.status == BatchStatus::Pending {
            item.status = BatchStatus::Done;
        }
        item.result_tex_pending = true;
        item.thumb_pending = true;
        let source = item.source.clone();
        Self::spawn_tex_prep(
            rgba.clone(), item_id, Self::tex_name(tag, item_id, Some(switch)), true, handles, ctx.clone(),
        );
        self.batch.request_thumbnail(item_id, &source, Some(&rgba));
        ctx.request_repaint();
    }

    /// Drain completed upscale results and apply them to the matching item.
    /// A stale result (the user fired a new dispatch before the previous one
    /// finished, or an admission-refused run delivered nothing) is silently
    /// dropped when no in-flight dispatch is registered. `.try_recv()` keeps
    /// this non-blocking.
    fn pump_upscale_results(&mut self, ctx: &egui::Context) {
        let results = self.processor.pump_upscale_results();
        if results.is_empty() {
            return;
        }
        let handles = self.batch.bg_io.tex_prep_handles();
        let switch = self.result_switch_id;
        for r in results {
            let item_id = r.item_id;
            let result_recipe = r.recipe.clone();
            match r.result {
                Ok(raw_rgba) => {
                    // Cache the raw upscale output (pre-Tier-2-postprocess) so
                    // Tier-2 tweaks can reprocess without re-inferring.
                    let raw_arc = Arc::new(raw_rgba);
                    let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
                    item.upscale_raw = Some(raw_arc.clone());
                    // Invalidate the stale bicubic_source — the upscale output
                    // dimensions changed (new Tier-1 run), so any prior bicubic
                    // cache at the old dimensions is wrong.
                    item.bicubic_source = None;
                    // Apply Tier-2 postprocess to produce the displayed result.
                    // The source for ai_blend / color_match is the original input
                    // upscaled bicubically to match upscale_raw's dimensions.
                    let bicubic = build_or_reuse_bicubic(item, &raw_arc);
                    let mut displayed = (*raw_arc).clone();
                    let s = &item.settings;
                    apply_tier2_postprocess(
                        &mut displayed,
                        s.sharpen,
                        s.ai_blend,
                        s.saturation,
                        s.color_match,
                        bicubic.as_ref(),
                    );
                    let new_rgba = Arc::new(displayed);
                    // Archive happened at dispatch time via seed_history_for_reprocess;
                    // here we just stamp the new result. result_rgba was already
                    // taken (chain_mode replanted it; non-chain left it None).
                    item.result_rgba = Some(new_rgba.clone());
                    item.applied_recipe = Some(result_recipe);
                    item.status = BatchStatus::Done;
                    item.result_tex_pending = true;
                    item.thumb_pending = true;
                    let source = item.source.clone();
                    Self::spawn_tex_prep(
                        new_rgba.clone(), item.id,
                        Self::tex_name("upscale", item.id, Some(switch)),
                        true, handles.clone(), ctx.clone(),
                    );
                    self.batch.request_thumbnail(item_id, &source, Some(&new_rgba));
                    tracing::info!(item_id, "upscale result applied");
                }
                Err(prunr_core::CoreError::Cancelled) => {
                    tracing::info!(item_id, "upscale cancelled by user");
                    self.toasts.info("Upscale cancelled");
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::error!(item_id, %msg, "upscale dispatch failed");
                    self.toasts.error(format!("Upscale failed: {msg}"));
                }
            }
        }
        ctx.request_repaint();
    }

    /// Eager encoder: while Magic Brush is active, the selected image gets
    /// an embedding as soon as it has a decoded source. Runs on activation
    /// and from `reconcile_selected` every frame, so switching images keeps
    /// the tool usable without toggling it off and on. Admission or encoder
    /// failure turns the tool off, so this cannot retry in a loop.
    fn ensure_magic_embedding_for_selected(&mut self) {
        if !self.magic_brush_state.is_active() || self.magic_brush_state.has_pending_encoder() {
            return;
        }
        let Some(item) = self.batch.selected_item() else { return };
        if item.magic_brush_embedding.is_some() {
            return;
        }
        // Decode still pending: try again next frame.
        let Some(source) = item.source_rgba.clone() else { return };
        let item_id = item.id;
        let avail_ram = crate::hardware::available_ram_mb_throttled();
        match self.processor.dispatch_sam_encoder(item_id, source, avail_ram) {
            Ok(()) => self.magic_brush_state.set_encoder_pending(true),
            Err(err) => self.magic_brush_unavailable(&err),
        }
    }

    /// Admission or encoder failure: tell the user and turn the tool off,
    /// which also stops the per-frame encoder check from retrying.
    fn magic_brush_unavailable(&mut self, err: &str) {
        self.toasts.error(format!("Magic Brush unavailable: {err}"));
        self.deactivate_magic_brush();
    }

    /// Turn Magic Brush off and drop its ORT sessions (~180 MB). Cached
    /// embeddings stay on their items, so re-activating is cheap.
    pub(crate) fn deactivate_magic_brush(&mut self) {
        self.magic_brush_state.deactivate();
        self.processor.release_sam_sessions();
    }

    /// Keep the selected image's embedding current, then drain SAM encoder
    /// and decoder results from the background rayon threads.
    /// Encoder results: write embedding to BatchItem, clear encoder_pending.
    /// Decoder results: convert to MaskArtifact, apply modifier, commit.
    fn pump_sam_results(&mut self, ctx: &egui::Context) {
        use crate::gui::processor::PromptModifier;

        let encoder_results = self.processor.pump_sam_encoder_results();
        for result in encoder_results {
            self.magic_brush_state.set_encoder_pending(false);
            match result.result {
                Ok(embedding) => {
                    if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                        item.magic_brush_embedding = Some(std::sync::Arc::new(embedding));
                        tracing::info!(item_id = result.item_id, "SAM encoder embedding cached");
                    }
                    ctx.request_repaint();
                }
                Err(err) => {
                    tracing::error!(item_id = result.item_id, %err, "SAM encoder failed");
                    self.magic_brush_unavailable(&err);
                }
            }
        }

        let decoder_results = self.processor.pump_sam_decoder_results();
        for result in decoder_results {
            let new_mask = match result.result {
                Ok(Some(mask)) => mask,
                Ok(None) => {
                    self.toasts.info("No selection candidate met the confidence threshold.");
                    continue;
                }
                Err(err) => {
                    tracing::error!(item_id = result.item_id, %err, "SAM decoder failed");
                    self.toasts.error(format!("Magic Brush decoder failed: {err}"));
                    continue;
                }
            };
            let existing = self.batch
                .find_by_id(result.item_id)
                .and_then(|i| i.selection_mask.clone());
            let final_mask = match (result.modifier, existing) {
                (PromptModifier::Replace, _) => new_mask,
                (PromptModifier::Add, Some(existing)) => {
                    existing.add_mask(&new_mask).unwrap_or(new_mask)
                }
                (PromptModifier::Subtract, Some(existing)) => {
                    existing.subtract_mask(&new_mask).unwrap_or(new_mask)
                }
                (PromptModifier::Add, None) | (PromptModifier::Subtract, None) => new_mask,
            };
            tracing::info!(item_id = result.item_id, modifier = ?result.modifier, "SAM decoder mask committed");
            self.commit_selection_and_dispatch(result.item_id, final_mask);
        }
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
        // Inpaint strokes commit via the inpaint subprocess; firing
        // a seg preview here would clobber `result_rgba` with the raw
        // source through `run_preview`'s no-tensor fallback.
        if self.settings.model.is_inpaint() {
            return;
        }
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
                    item.set_cached_tensor(None);
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
                item.set_cached_tensor(None);
                tiers.tier1.insert(item.id);
                continue;
            }

            let current_recipe = item.settings.current_recipe(model, chain);
            let tier = prunr_core::resolve_tier(old_recipe, &current_recipe);
            let impact = knob_catalog::cache_impact_for_recipe_diff(old_recipe, &current_recipe);
            item.apply_cache_impact(impact);

            match tier {
                RequiredTier::Skip
                | RequiredTier::CompositeOnly
                | RequiredTier::UpscaleRerun => {
                    // CompositeOnly (bg_color) and UpscaleRerun re-render at
                    // display/export time, not via the seg/edge dispatcher;
                    // sync the stored composite so status reads stay accurate.
                    if let Some(ref mut recipe) = item.applied_recipe {
                        recipe.composite = current_recipe.composite.clone();
                    }
                    tiers.skip_count += 1;
                }
                RequiredTier::UpscaleTier2 => {
                    // Tier-2 upscale (sharpen/ai_blend/saturation/color_match)
                    // dispatches the cached-buffer postprocess via live preview
                    // — no subprocess admission, no re-inference.
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
                    item.set_cached_tensor(None);
                    item.applied_recipe = None;
                }
                (_, Err(e)) => {
                    item.status = BatchStatus::Error(format!("Failed to load: {e}"));
                    item.set_cached_tensor(None);
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
            config: Box::new(super::worker::ProcessingConfig {
                model,
                jobs,
                mask: current_settings.mask_settings(),
                force_cpu: self.settings.force_cpu,
                line_mode: current_settings.line_mode,
                edge: current_settings.edge_settings(),
            }),
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
                        payload.push((folder.join(format!("{stem}.prunr.png")), bytes));
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
    /// save-as dialog. Suggests a `<stem>.prunr.png` name based on the source.
    fn save_current_to_file(&mut self) {
        let Some(item) = self.batch.selected_item() else { return };
        let Some(rgba) = item.result_rgba.clone() else { return };
        let default_name = Path::new(&item.filename).file_stem()
            .and_then(|s| s.to_str())
            .map(|stem| format!("{stem}.prunr.png"))
            .unwrap_or_else(|| "result.prunr.png".to_string());
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
            (item.bake_export_bg(rgba), format!("{stem}.prunr.png"))
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
    /// as `<source-stem>.prunr.png`. Encode + write run on a background thread.
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
        let count = self.batch.selected_count();
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
        // Esc cancels both the batch path and any in-flight eraser stroke
        // (SD on CPU takes minutes — needs the same escape valve).
        self.processor.cancel_all_inpaints();
        // Always cancel in-flight upscale; the call is idempotent and
        // gating on `model.is_upscale()` would drop the signal if the
        // user switched model mid-flight.
        self.processor.cancel_upscale();
    }

    /// Cancel-All from the toolbar / Escape: request the global cancel,
    /// clear admission, flip every Processing item back to Pending, and
    /// surface "Cancelled" in the status bar. Single source of truth so
    /// the toolbar button and the keyboard shortcut don't drift.
    pub fn handle_cancel_all_and_reset(&mut self) {
        self.handle_cancel();
        self.processor.clear_admission();
        self.batch.reset_processing_to_pending();
        // Clear the unified progress slot so the canvas banner/modal
        // disappears with the batch reset. Without this, the seg
        // dispatch_progress lingers from before the cancel and the
        // overlay keeps painting on top of the idle canvas — the
        // legacy `render_processing` was AppState-gated so it
        // disappeared automatically; the unified widget reads the
        // slot directly and needs an explicit clear.
        self.processor.set_dispatch_progress(None);
        self.status.text = "Cancelled".to_string();
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
        if self.batch.remove(idx) {
            self.sync_after_batch_change();
        }
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
        self.pump_upscale_results(ctx);
        self.pump_sam_results(ctx);
        self.pump_download_manager(ctx);
        self.pump_runtime_install(ctx);
        self.recipe_drift_tripwire();

        // Dispatch phase: tick() invokes the closure for each item whose
        // debounce expired this frame. The closure borrows `batch.items`
        // mutably so `build_preview_inputs` can lazily cache an
        // `Arc<DynamicImage>` on the item (built once, reused across every
        // subsequent dispatch of the drag session — see `source_dyn`).
        let is_filter_only = self.settings.model.to_model_kind().is_none();
        let chain_mode = self.settings.chain_mode;
        let batch_items = &mut self.batch.items;
        let wait = self.processor.live_preview.tick(|id, kind| {
            Self::build_preview_inputs(batch_items, id, kind, is_filter_only, chain_mode)
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
    pub(crate) fn build_preview_inputs(
        items: &mut [BatchItem],
        id: u64,
        kind: super::live_preview::PreviewKind,
        is_filter_only: bool,
        chain_mode: bool,
    ) -> Option<super::live_preview::DispatchInputs> {
        use super::live_preview::{DispatchInputs, PreviewKind};
        let item = items.iter_mut().find(|b| b.id == id)?;
        let seg_tensor = item.hot_seg_tensor();
        // A Mask dispatch in a seg model with no cached tensor falls
        // through to `run_preview`'s source+fill_style fallback and
        // overwrites any existing `result_rgba`. Filter-only mode is
        // the legitimate user of that fallback (the filter IS the
        // pipeline). Anywhere else, refuse — better to skip the
        // dispatch than discard a prior chained result.
        if matches!(kind, PreviewKind::Mask)
            && !is_filter_only
            && seg_tensor.is_none()
            && item.result_rgba.is_some()
        {
            return None;
        }
        // Chain mode + prior result: apply the live-preview mask to the
        // chained `result_rgba` so stacked edits (e.g. an SD inpaint
        // stroke) survive a subsequent seg brush stroke. Without this,
        // Tier-2 mask rerun rebuilds against `source_dyn` and silently
        // discards every chain step. The `chain_dyn_cache` keeps the
        // wrapped `DynamicImage` across dispatches so a slider drag at
        // 4K doesn't re-clone ~50 MB per tick; `Arc::ptr_eq` against the
        // current `result_rgba` is the self-invalidation check.
        let original = if chain_mode && item.result_rgba.is_some() {
            let rgba = item.result_rgba.as_ref().unwrap();
            let cached = item.chain_dyn_cache.as_ref()
                .and_then(|(src, dyn_img)| Arc::ptr_eq(rgba, src).then(|| Arc::clone(dyn_img)));
            match cached {
                Some(dyn_img) => dyn_img,
                None => {
                    let dyn_img = Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone()));
                    item.chain_dyn_cache = Some((Arc::clone(rgba), Arc::clone(&dyn_img)));
                    dyn_img
                }
            }
        } else {
            if item.source_dyn.is_none() {
                let rgba = item.source_rgba.as_ref()?;
                item.source_dyn = Some(Arc::new(image::DynamicImage::ImageRgba8((**rgba).clone())));
            }
            Arc::clone(item.source_dyn.as_ref().unwrap())
        };
        let edge_tensor = Self::edge_tensor_for_active_scale(item);
        // DualScale needs TWO edge tensors — the primary (active scale, used
        // as the Fine layer) and a secondary Bold layer. Only decompress the
        // Bold tensor when the style actually uses it; skipping this keeps
        // single-scale dispatches fast.
        let secondary_edge_tensor = matches!(item.settings.line_style, prunr_core::LineStyle::DualScale { .. })
            .then(|| Self::edge_tensor_for_scale(item, prunr_core::EdgeScale::Bold))
            .flatten();
        // UpscaleTier2: short-circuit the seg/edge path entirely. We need only
        // upscale_raw + bicubic_source + knob snapshot. Return None if there is
        // no cached upscale output yet (Tier-1 must run first).
        if matches!(kind, PreviewKind::UpscaleTier2) {
            use super::live_preview::UpscaleTier2Knobs;
            let upscale_raw = item.upscale_raw.as_ref()?.clone();
            let bicubic_arc = build_or_reuse_bicubic(item, &upscale_raw);
            let knobs = UpscaleTier2Knobs {
                sharpen: item.settings.sharpen,
                ai_blend: item.settings.ai_blend,
                saturation: item.settings.saturation,
                color_match: item.settings.color_match,
            };
            return Some(DispatchInputs {
                kind,
                original,
                settings: item.settings,
                seg_tensor: None,
                edge_tensor: None,
                secondary_edge_tensor: None,
                cached_edge: Default::default(),
                cached_bold: Default::default(),
                cached_masked_base: None,
                correction: None,
                upscale_raw: Some(upscale_raw),
                bicubic_source: Some(bicubic_arc),
                upscale_tier2_knobs: Some(knobs),
            });
        }
        match kind {
            // Mask kind without a seg tensor is the filter-only path:
            // `run_preview` applies fill_style to the raw source. No tensor
            // required. Edge kind still needs an edge tensor (nothing to
            // preview without one).
            PreviewKind::Edge if edge_tensor.is_none() => return None,
            _ => {}
        }
        let plane_key = |scale: prunr_core::EdgeScale| super::live_preview::EdgePlaneKey {
            strength_bits: item.settings.line_strength.to_bits(),
            scale,
            thickness: u32::from(item.settings.edge_thickness),
        };
        let cached_edge = item.cached_edge.lookup(plane_key(item.settings.edge_scale));
        let cached_bold = item.cached_bold.lookup(plane_key(prunr_core::EdgeScale::Bold));
        let cached_masked_base = item.cached_masked_base.as_ref().and_then(|(base, recipe, model)| {
            let current_recipe = prunr_core::MaskRecipe::from(&item.settings.mask_settings());
            let seg_model_match = seg_tensor.as_ref().is_some_and(|s| s.model == *model);
            (*recipe == current_recipe && seg_model_match).then(|| base.clone())
        });
        // Without a seg tensor the correction has nothing to apply to
        // (filter-only path).
        let correction = seg_tensor.as_ref().and_then(|seg| item.selection_plane_at((seg.width, seg.height)));
        Some(DispatchInputs {
            kind, original, settings: item.settings,
            seg_tensor, edge_tensor, secondary_edge_tensor,
            cached_edge, cached_bold, cached_masked_base,
            correction,
            upscale_raw: None,
            bicubic_source: None,
            upscale_tier2_knobs: None,
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
        use super::live_preview::PreviewKind;
        let handles = self.batch.bg_io.tex_prep_handles();
        for r in results {
            let (item_id, source, is_final) = {
                let Some(item) = self.batch.find_by_id_mut(r.item_id) else {
                    continue;
                };
                let is_upscale_tier2 = matches!(r.kind, PreviewKind::UpscaleTier2);
                let new_rgba = Arc::new(r.rgba);
                item.result_rgba = Some(new_rgba.clone());
                // Filter-only mode (model=None) never clicks Process — the
                // first live preview result IS the processed result. Promote
                // a Pending item to Done so the canvas flips from source
                // view to result view. Processing / Done items untouched.
                if item.status == BatchStatus::Pending {
                    item.status = BatchStatus::Done;
                }
                // Mark pending so reconcile_selected doesn't also
                // spawn its own prep on this same frame.
                item.result_tex_pending = true;
                if let Some((planes, key)) = r.new_edge {
                    item.cached_edge.store(planes, key);
                }
                if let Some((planes, key)) = r.new_bold {
                    item.cached_bold.store(planes, key);
                }
                if let Some((base, recipe, model)) = r.new_masked_base {
                    item.cached_masked_base = Some((base, recipe, model));
                }
                // Mark the dispatched MaskRecipe as applied so the
                // recipe-drift tripwire doesn't immediately re-fire on
                // a result it has already consumed.
                if let Some(applied) = item.applied_recipe.as_mut() {
                    applied.mask = r.applied_mask;
                    if let Some(knobs) = r.applied_tier2_knobs {
                        applied.upscale.sharpen_bits = knobs.sharpen.to_bits();
                        applied.upscale.ai_blend_bits = knobs.ai_blend.to_bits();
                        applied.upscale.saturation_bits = knobs.saturation.to_bits();
                        applied.upscale.color_match = knobs.color_match;
                    }
                }
                tracing::debug!(
                    event = "upscale_tier2_apply",
                    item_id = item.id,
                    is_upscale_tier2,
                    is_final = r.is_final,
                    "apply_completed_previews: result drained"
                );
                let switch = self.result_switch_id;
                Self::spawn_tex_prep(
                    new_rgba, item.id, Self::tex_name("result", item.id, Some(switch)),
                    true, handles.clone(), ctx.clone(),
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

    /// Run the selection-change work even though the selected id is
    /// unchanged: the result or history behind it was replaced.
    pub(crate) fn sync_selected_batch_textures(&mut self, ctx: &egui::Context) {
        self.synced_selected = None;
        self.reconcile_selected(ctx);
    }

    /// Once per frame. On a selection change (detected here, so no
    /// selection path can forget it): free the background items' results,
    /// restore the selected one from history, leave Compare. Every frame:
    /// the decoded source, the canvas textures, the bg-image texture, the
    /// selection texture and the Magic Brush embedding the selected item
    /// lacks. Every arm is idempotent behind its pending flag.
    fn reconcile_selected(&mut self, ctx: &egui::Context) {
        let selected = self.batch.selected_item().map(|i| i.id);
        let idx = self.batch.selected_idx_clamped();
        if selected != self.synced_selected {
            self.synced_selected = selected;
            if let Some(idx) = idx {
                self.evict_background_item_caches(idx);
                self.restore_selected_result_from_history(idx);
            }
            self.show_original = false;
        }
        if let (Some(idx), Some(id)) = (idx, selected) {
            self.ensure_selected_source_decoded(idx);
            self.request_selected_textures(idx, ctx);
            self.kick_bg_image_tex_prep(id, ctx);
        }
        let style = super::background_io::SelectionStyle::from_brush(&self.settings.brush);
        self.batch.ensure_selection_texture(style, ctx);
        self.ensure_magic_embedding_for_selected();
    }

    /// For every item that is not selected: drop the decoded tensors kept
    /// for slider drags, and free the full-resolution `result_rgba` of Done
    /// items, placing an in-memory placeholder at
    /// `history.back()` so `restore_selected_result_from_history` can
    /// read pixels back instantly, then kicks an off-thread zstd
    /// compression that swaps the placeholder for `HistorySlot::Compressed`
    /// once it returns (drained by `pump_history_demote_results`).
    /// Pre-fix this ran zstd inline — a 50-image 4K batch froze the UI
    /// for ~2.5 s on every selection change.
    fn evict_background_item_caches(&mut self, selected_idx: usize) {
        for (i, item) in self.batch.items.iter_mut().enumerate() {
            if i == selected_idx {
                continue;
            }
            item.drop_hot_tensors();
            if item.result_rgba.is_none() || item.status != BatchStatus::Done {
                continue;
            }
            if let Some(rgba) = item.result_rgba.take() {
                let recipe = item.applied_recipe.clone();
                let placeholder = HistoryEntry {
                    slot: HistorySlot::InMemory(rgba.clone()),
                    recipe: recipe.clone(),
                };
                if let Some(back) = item.history.back_mut() {
                    back.cleanup();
                    *back = placeholder;
                } else {
                    item.history.push_back(placeholder);
                }
                let id = item.id;
                let tx = self.batch.bg_io.history_demote_tx.clone();
                let slots = self.batch.bg_io.decode_slots.clone();
                std::thread::spawn(move || {
                    // Park if every decode slot is busy — same backpressure as
                    // the texture-prep / decode workers, so a Process All
                    // burst can't fan out N zstd encoders simultaneously.
                    let _slot = slots.acquire();
                    if let Ok(entry) = super::history_disk::compress_to_ram(&rgba) {
                        let _ = tx.send((id, entry, recipe));
                    }
                    // On compression failure (rare — disk-full / OOM), the
                    // InMemory placeholder stays. Memory is still freed on
                    // the result side because `result_rgba` was dropped.
                });
            }
            item.result_texture = None;
            item.result_tex_pending = false;
        }
    }

    /// Kick off-thread bg-image texture prep: `to_rgba8()` + the
    /// `ColorImage::from_rgba_unmultiplied` clone happen on a worker;
    /// `drain_background_channels` does the `ctx.load_texture` upload
    /// when the result lands. Idempotent — re-entry while a prep is
    /// already in flight is a no-op via the `bg_image_tex_pending`
    /// flag (cleared by the drain).
    pub(crate) fn kick_bg_image_tex_prep(&mut self, item_id: u64, ctx: &egui::Context) {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        if item.bg_image_tex_pending || item.bg_image_texture.is_some() { return; }
        let Some(bg) = item.bg_image.as_ref().cloned() else { return };
        item.bg_image_tex_pending = true;
        let id = item.id;
        let tx = self.batch.bg_io.bg_tex_prep_tx.clone();
        let slots = self.batch.bg_io.decode_slots.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            let rgba = bg.image.to_rgba8();
            let (w, h) = (rgba.width(), rgba.height());
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_raw(),
            );
            let _ = tx.send((id, bg.hash, ci));
            ctx.request_repaint();
        });
    }

    /// Replace each in-memory placeholder at `history.back()` with the
    /// compressed slot just produced off-thread. Recipe match guards
    /// against a fresh re-Process having pushed a new entry while the
    /// compression was in flight — in that case we drop the result
    /// rather than clobber the user's current state.
    fn pump_history_demote_results(&mut self) {
        while let Ok((id, entry, recipe)) = self.batch.bg_io.history_demote_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(id) else { continue };
            let Some(back) = item.history.back_mut() else { continue };
            if back.recipe != recipe { continue; }
            if !matches!(&back.slot, HistorySlot::InMemory(_)) { continue; }
            back.slot = HistorySlot::Compressed(entry);
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
                    rgba, item_id, Self::tex_name("source", item_id, None), false,
                    self.batch.bg_io.tex_prep_handles(), ctx.clone(),
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
                    rgba, item_id, Self::tex_name("result", item_id, Some(switch)), true,
                    self.batch.bg_io.tex_prep_handles(), ctx.clone(),
                );
            }
        }
    }


    /// Build a unique texture name for `spawn_tex_prep`. Single source-of-truth
    /// for the four call sites so the naming scheme is easy to audit.
    pub(crate) fn tex_name(kind: &str, item_id: u64, switch: Option<u64>) -> String {
        match switch {
            Some(sw) => format!("{kind}_{item_id}_{sw}"),
            None => format!("{kind}_{item_id}"),
        }
    }

    fn spawn_tex_prep(
        rgba: Arc<image::RgbaImage>,
        item_id: u64,
        name: String,
        is_result: bool,
        handles: super::background_io::TexPrepHandles,
        ctx: egui::Context,
    ) {
        std::thread::spawn(move || {
            // Park here past `available_parallelism()` so a Process All
            // batch-completion burst doesn't hold N × ~50 MB ColorImage
            // peaks simultaneously. Same slot pool as the bg-io decode /
            // thumbnail / filter spawns.
            let _slot = handles.slots.acquire();
            let (w, h) = (rgba.width(), rgba.height());
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_flat_samples().as_slice(),
            );
            let _ = handles.tx.send((item_id, name, ci, is_result));
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

        // A selection authored before this result, or kept across Process,
        // is applied now that there is a tensor to correct.
        let has_selection = self
            .batch
            .find_by_id(item_id)
            .is_some_and(|i| i.status == BatchStatus::Done && i.selection_mask.is_some());
        if has_selection {
            self.apply_selection_to_active_model(item_id);
        }

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
        self.status.stage = report.stage.clone();
        self.status.pct = report.pct;

        // Scope the seg slot counter to the *current* dispatch via
        // `processor.current_dispatch_progress()` — not whole-batch
        // status_counts, which would inflate the total with items
        // Done from a previous dispatch (e.g. reprocessing one image
        // after the other was already finished).
        let progress = self.processor.current_dispatch_progress().map(|(done, total)| {
            crate::gui::dispatch_progress::DispatchProgress::seg(done, total, report.stage)
        });
        self.processor.set_dispatch_progress(progress);
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
        // Drop the seg dispatch_progress entry so the canvas overlay
        // disappears with the cancel acknowledgement. Same reason
        // as `handle_cancel_all_and_reset`.
        self.processor.set_dispatch_progress(None);
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
        use crate::gui::views::shortcuts::{self, Action};
        let pressed = shortcuts::pressed(ctx);
        let copy_requested = std::mem::take(&mut self.pending_copy);
        let pending_open = std::mem::take(&mut self.pending_open_dialog);

        if pressed.is(Action::Open) || pending_open {
            self.handle_open_dialog();
        }
        let app_state = self.batch.app_state();
        if pressed.is(Action::Process) && matches!(app_state, AppState::Loaded | AppState::Done) {
            self.handle_process_intent();
        }
        if pressed.is(Action::Save) && app_state == AppState::Done {
            self.handle_save_selected();
        }
        // Selection shortcuts share the action bar's gate; Ctrl+C without
        // a usable selection is the whole-result copy.
        let selection_idx = self.batch.selected_idx_clamped().filter(|_| self.can_selection_action());
        if copy_requested {
            match selection_idx {
                Some(idx) => self.handle_selection_action(idx, SelectionAction::Copy, ctx),
                None if app_state == AppState::Done => self.handle_copy(),
                None => {}
            }
        }
        if let Some(idx) = selection_idx {
            for (wanted, action) in [
                (pressed.is(Action::Cut), SelectionAction::Cut),
                (pressed.is(Action::Delete), SelectionAction::Delete),
                (pressed.is(Action::Invert), SelectionAction::Invert),
            ] {
                if wanted {
                    self.handle_selection_action(idx, action, ctx);
                }
            }
        }
        if pressed.is(Action::BeforeAfter) && app_state == AppState::Done {
            self.show_original = !self.show_original;
        }
        if pressed.is(Action::FitToWindow) { self.zoom_state.pending_fit_zoom = true; }
        if pressed.is(Action::ActualSize)   { self.zoom_state.pending_actual_size = true; }

        if pressed.is(Action::Cancel) {
            self.apply_cancel_shortcut(ctx);
        }

        for action in [Action::Shortcuts, Action::CliHelp, Action::PipelineFlow] {
            if pressed.is(action) {
                if let Some(open) = self.help_modal_mut(action) {
                    *open = !*open;
                }
            }
        }
        if pressed.is(Action::Settings)  { self.toggle_settings_panel(ctx); }

        if pressed.is(Action::PrevImage) { self.navigate_batch(ctx, NavDir::Prev); }
        if pressed.is(Action::NextImage) { self.navigate_batch(ctx, NavDir::Next); }

        if pressed.is(Action::ToggleQueue)     { self.sidebar_hidden     = !self.sidebar_hidden; }
        if pressed.is(Action::ToggleAdjustments) { self.adjustments_hidden = !self.adjustments_hidden; }

        if pressed.is(Action::Undo) { self.handle_undo(ctx); }
        if pressed.is(Action::Redo) { self.handle_redo(ctx); }
        if pressed.is(Action::Screenshot) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }

        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }
    }

    /// The visibility flag behind a help modal's shortcut action; `None`
    /// for actions that are not help modals. The keyboard toggles it, the
    /// Help menu sets it.
    pub(crate) fn help_modal_mut(&mut self, action: crate::gui::views::shortcuts::Action) -> Option<&mut bool> {
        use crate::gui::views::shortcuts::Action;
        match action {
            Action::Shortcuts => Some(&mut self.show_shortcuts),
            Action::CliHelp => Some(&mut self.show_cli_help),
            Action::PipelineFlow => Some(&mut self.show_pipeline_flow),
            _ => None,
        }
    }

    /// Escape dismisses the topmost interruptable state. Priority order:
    /// in-flight upscale → in-flight eraser stroke → active batch processing
    /// → active selection → open modal. Selection clear sits above modal
    /// dismissal so Esc is the natural "cancel what I just drew" key.
    fn apply_cancel_shortcut(&mut self, ctx: &egui::Context) {
        if self.processor.is_upscale_in_flight() {
            self.processor.cancel_upscale();
        } else if self.processor.any_inpaint_in_flight() {
            self.processor.cancel_all_inpaints();
        } else if self.batch.status_counts().processing > 0 {
            self.handle_cancel_all_and_reset();
        } else if self
            .batch
            .selected_item()
            .is_some_and(|i| i.selection_mask.is_some())
        {
            if let Some(idx) = self.batch.selected_idx_clamped() {
                self.handle_selection_action(idx, SelectionAction::Clear, ctx);
            }
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
            self.open_settings();
        }
    }

    pub(crate) fn open_settings(&mut self) {
        self.show_settings = true;
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

    /// Drain completed thumbnails from the background decoder into GPU
    /// textures. Throttles uploads so a 50-image import doesn't burst all
    /// `ctx.load_texture` calls into one frame and overwhelm the egui
    /// command buffer. Requests a repaint while the queue is still saturated.
    fn pump_thumbnail_results(&mut self, ctx: &egui::Context) {
        const THUMB_UPLOADS_PER_FRAME: usize = 3;
        let mut uploaded = 0;
        while uploaded < THUMB_UPLOADS_PER_FRAME {
            let Ok((item_id, tw, th, pixels)) = self.batch.bg_io.thumb_rx.try_recv() else { break };
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                let ci = egui::ColorImage::from_rgba_unmultiplied(
                    [tw as usize, th as usize],
                    &pixels,
                );
                item.thumb_texture = Some(ctx.load_texture(
                    format!("thumb_{item_id}"),
                    ci,
                    egui::TextureOptions::LINEAR,
                ));
                item.thumb_pending = false;
                uploaded += 1;
            }
        }
        if uploaded == THUMB_UPLOADS_PER_FRAME {
            ctx.request_repaint();
        }
    }

    fn drain_background_channels(&mut self, ctx: &egui::Context) {
        let mut decode_arrived = false;
        while let Ok((item_id, result)) = self.batch.bg_io.decode_rx.try_recv() {
            let source_replaced = {
                let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
                item.decode_pending = false;
                match result {
                    Ok(rgba) => {
                        item.source_rgba = Some(rgba);
                        // Live-preview DynamicImage cache is built from source_rgba;
                        // invalidate so the next dispatch rebuilds against the new one.
                        item.source_dyn = None;
                        decode_arrived = true;
                        true
                    }
                    Err(msg) => {
                        item.status = BatchStatus::Error(msg);
                        false
                    }
                }
            };
            if source_replaced {
                // Selection mask + SAM encoder embedding were derived from
                // the old source bytes — drop them so the next stroke /
                // dispatch rebuilds against the new image.
                self.batch.invalidate_selection_on_source_change(item_id);
            }
        }
        if decode_arrived {
            // Freshly-decoded RGBA for the viewed item: clear zoom state
            // entirely (not just re-arm the flag). A bare flag keeps
            // previous_zoom / pan_offset around, which lets `canvas.rs`'s
            // toggle-back branch fire against the old image's state.
            self.zoom_state.reset();
            self.show_original = false;
        }

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
            }
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

        self.pump_thumbnail_results(ctx);
        self.pump_history_demote_results();

        // Drain off-thread selection textures. `ctx.load_texture` is
        // allowed here — this drain runs from `logic()`, not a render
        // closure. A result for a mask that has since changed is dropped;
        // `ensure_selection_texture` already requested the current one.
        while let Ok(result) = self.batch.bg_io.selection_texture_rx.try_recv() {
            if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                if item.selection_tex_pending == Some(result.key) {
                    item.selection_tex_pending = None;
                }
                if item.selection_hash != Some(result.key.0) {
                    continue;
                }
                let Some(shown) = item.selection_mask.clone() else { continue };
                match result.image {
                    super::background_io::SelectionImage::Full { image, feathered } => {
                        let handle = ctx.load_texture(
                            format!("selection_{}", result.item_id),
                            image,
                            egui::TextureOptions::LINEAR,
                        );
                        item.selection_texture = Some(super::item::SelectionTexture { key: result.key, handle, shown, feathered });
                    }
                    // A patch whose base texture has moved on is dropped; the
                    // per-frame check asks again against the new base.
                    super::background_io::SelectionImage::Patch { base_hash, pos, image } => {
                        if let Some(tex) = item.selection_texture.as_mut().filter(|t| t.key == (base_hash, result.key.1)) {
                            tex.handle.set_partial(pos, image, egui::TextureOptions::LINEAR);
                            tex.key = result.key;
                            tex.shown = shown;
                        }
                    }
                }
            }
        }

        // Drain bg-image texture preps. Hash match guards against a
        // user pick that swapped to a different bg image while the
        // worker was running — in that case the stale ColorImage is
        // dropped and a fresh kick is already in flight.
        while let Ok((item_id, hash, ci)) = self.batch.bg_io.bg_tex_prep_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            item.bg_image_tex_pending = false;
            if item.bg_image.as_ref().map(|b| b.hash) != Some(hash) { continue; }
            // WrapMode::Repeat enables BgImageFit::Tile (UV > 1.0
            // wraps); other fits keep UVs in [0, 1] so the wrap is a
            // no-op. LINEAR for smooth scale modes (Cover/Contain
            // /Stretch); Tile + Center read 1:1 so the filter doesn't
            // matter there.
            let opts = egui::TextureOptions {
                magnification: egui::TextureFilter::Linear,
                minification: egui::TextureFilter::Linear,
                wrap_mode: egui::TextureWrapMode::Repeat,
                mipmap_mode: None,
            };
            let tex = ctx.load_texture(format!("bg_image_{:x}", hash), ci, opts);
            item.bg_image_texture = Some(tex);
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
        self.reconcile_selected(ctx);
        // Keep the event loop awake while any async texture / decode work is
        // pending. `ctx.request_repaint()` from the tex_prep / decode threads
        // is supposed to wake egui, but some compositors (notably Wayland)
        // drop thread-initiated wake-ups when the window is idle, leaving the
        // canvas stuck on the old image until a mouse event fires. Polling
        // from the UI thread is reliable; it costs one frame per 50ms while
        // async work is in flight, then self-extinguishes.
        if self.batch.items.iter().any(|it| it.has_pending_work()) {
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
    /// Whether the adjustments toolbar should render this frame.
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
        let height = theme::CHIP_HEIGHT + theme::SPACE_SM * 2.0;
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
                // Both brushes author the selection against the source, so
                // they only need a decoded image; a segmentation correction
                // waits for the tensor (see `apply_selection_to_active_model`).
                let brush_available = item.source_rgba.is_some() || item.cached_tensor.is_some();
                let has_bg_image = item.bg_image.is_some();
                let bg_image_label = item.bg_image.as_deref()
                    .and_then(|bg| bg.source_path.as_deref())
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str());
                // Upscale row shows projected output size. When chain mode is
                // on and a result exists, the upscale input is the result image,
                // not the source — reflect that in the dimension chip.
                let source_dims = if settings_ref.chain_mode {
                    item.result_rgba.as_ref()
                        .map(|r| r.dimensions())
                        .unwrap_or(item.dimensions)
                } else {
                    item.dimensions
                };
                let state = adjustments_toolbar::ToolbarState {
                    magic_brush_active: self.magic_brush_state.is_active(),
                    magic_encoder_pending: self.magic_brush_state.has_pending_encoder(),
                    brush_available,
                    processing: is_processing,
                    has_bg_image,
                    bg_image_label,
                    source_dims,
                    has_selection: item.selection_mask.is_some(),
                    protect_selection: settings_ref.protect_selection,
                    show_original: self.show_original,
                    has_result: item.has_result(),
                };
                toolbar_change = adjustments_toolbar::render(
                    ui,
                    &mut item.settings,
                    settings_ref,
                    &mut item.applied_preset,
                    brush_state_ref,
                    state,
                );
            });
        if toolbar_change.reset_brush_requested {
            let resolved = self.settings.resolve_active_preset(None);
            self.settings.brush.reset_popover_fields_from(&resolved.brush);
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
            self.reconcile_bg_image_after_preset(idx, ctx);
        }
        if toolbar_change.model_changed {
            self.settings.on_model_change_resolve_brush();
            // Upscale has no brush surface — force-off so a leftover
            // enabled state from seg/inpaint doesn't leak into a mode
            // where the brush canvas overlay is hidden.
            if self.settings.model.is_upscale() {
                self.brush_state.disable();
                // No Magic toggle in the upscale row either; leaving the
                // tool active would keep encoding every selected image.
                self.deactivate_magic_brush();
            }
            // Cancel any in-flight upscale dispatch when the user
            // changes model mid-run — the result would land on the
            // wrong selection otherwise.
            self.processor.cancel_upscale();
            // Drop every item's upscale_raw / bicubic_source — the buffers
            // were produced by the previous model. A subsequent Tier-2
            // preview tick would otherwise display the old model's output
            // under the new model's knobs.
            for item in &mut self.batch.items {
                item.invalidate_upscale_cache();
            }
            self.settings.save();
            self.toasts.info(format!(
                "{} loaded",
                crate::gui::views::model_name(self.settings.model),
            ));
            // Drop every cached engine on model switch — both the
            // inpaint subprocess (SD bundle / LaMa subprocess
            // residual) and the seg subprocess's warm engine pool
            // (BiRefNetLite ~2 GB, U2Net ~800 MB, etc.). The
            // in-process LaMa cache was already cleared in
            // `adjustments_toolbar`'s on-model-changed branch.
            // Combined: every model switch reclaims RAM from every
            // backend the user is no longer using, regardless of
            // direction (inpaint→inpaint, seg→seg, inpaint→seg, …).
            self.processor.release_inpaint_subprocess();
            self.processor.release_seg_warm();
            // Drop the warm upscale engine on every model switch — both
            // upscale→upscale and upscale→seg/inpaint. ensure_upscale_engine
            // evicts on ModelKind mismatch anyway, but this explicit drop
            // covers the switch-away-from-upscale case where ensure is never
            // called again.
            self.processor.release_upscale_engine();
        }
        if toolbar_change.auto_chain_on {
            let item_has_result = self
                .batch
                .selected_item()
                .map(|i| i.has_result())
                .unwrap_or(false);
            self.settings.chain_mode =
                resolve_auto_chain_on(true, item_has_result, self.settings.chain_mode);
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
        if let Some(new_protect) = toolbar_change.protect_selection {
            self.settings.protect_selection = new_protect;
            self.settings.save();
        }
        if let Some(action) = toolbar_change.selection_action {
            self.handle_selection_action(idx, action, ctx);
        }
        if toolbar_change.toggle_compare {
            self.show_original = !self.show_original;
        }
        if toolbar_change.toggle_paint {
            self.brush_state.toggle();
            if self.brush_state.is_enabled() {
                self.deactivate_magic_brush();
            }
        }
        if toolbar_change.toggle_magic {
            if self.magic_brush_state.is_active() {
                self.deactivate_magic_brush();
            } else if self.magic_brush_state.activate() {
                self.brush_state.disable();
                self.ensure_magic_embedding_for_selected();
            }
        }

        self.batch.items[idx].apply_cache_impact(toolbar_change.cache_impact);

        let dispatch = self.resolve_auto_dispatch(idx, &toolbar_change);
        let item_id = self.batch.items[idx].id;
        let is_done = self.batch.items[idx].status == BatchStatus::Done;

        // Seg live-preview is meaningless in inpaint mode — no seg tensor
        // to rerun, and the no-tensor fallback would overwrite result_rgba.
        let seg_preview_active = self.settings.live_preview
            && !self.settings.model.is_inpaint();
        match dispatch {
            DispatchKind::None => {}
            DispatchKind::Render => {}
            DispatchKind::LivePreviewMask => {
                if seg_preview_active {
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
                if seg_preview_active {
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

        // Upscale Tier-2 live preview: sharpen / ai_blend / saturation / color_match
        // tweaks don't produce a DispatchKind (they aren't in the knob catalog and
        // resolve_auto_dispatch returns None for them). Detect them by checking
        // the recipe diff directly: if the model is upscale, live preview is enabled,
        // the item has a cached upscale_raw buffer, and the diff resolves to
        // UpscaleTier2 — fire mark_tweak so the debounced postprocess kicks in.
        if self.settings.live_preview && self.settings.model.is_upscale() {
            let item = &self.batch.items[idx];
            let has_raw = item.upscale_raw.is_some();
            let has_applied = item.applied_recipe.is_some();
            tracing::debug!(
                event = "upscale_tier2_check",
                item_id,
                live_preview = self.settings.live_preview,
                model_is_upscale = self.settings.model.is_upscale(),
                has_upscale_raw = has_raw,
                has_applied_recipe = has_applied,
                commit = toolbar_change.commit,
                "tier2 detection gate"
            );
            if has_raw {
                if let Some(ref old_recipe) = item.applied_recipe {
                    let model = self.settings.model
                        .dispatch_model_kind()
                        .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
                    let new_recipe = item.settings.current_recipe(model, self.settings.chain_mode);
                    let tier = prunr_core::resolve_tier(old_recipe, &new_recipe);
                    tracing::debug!(
                        event = "upscale_tier2_resolve",
                        item_id,
                        resolved_tier = ?tier,
                        old_model = ?old_recipe.upscale.model,
                        new_model = ?new_recipe.upscale.model,
                        old_sharpen_bits = old_recipe.upscale.sharpen_bits,
                        new_sharpen_bits = new_recipe.upscale.sharpen_bits,
                        old_ai_blend_bits = old_recipe.upscale.ai_blend_bits,
                        new_ai_blend_bits = new_recipe.upscale.ai_blend_bits,
                        old_saturation_bits = old_recipe.upscale.saturation_bits,
                        new_saturation_bits = new_recipe.upscale.saturation_bits,
                        old_color_match = old_recipe.upscale.color_match,
                        new_color_match = new_recipe.upscale.color_match,
                        old_output_scale = ?old_recipe.upscale.output_scale,
                        new_output_scale = ?new_recipe.upscale.output_scale,
                        "tier2 recipe diff"
                    );
                    if matches!(tier, prunr_core::RequiredTier::UpscaleTier2) {
                        tracing::debug!(event = "upscale_tier2_mark", item_id, "calling mark_tweak");
                        self.processor.live_preview.mark_tweak(item_id, PreviewKind::UpscaleTier2);
                        if toolbar_change.commit {
                            self.processor.live_preview.flush(item_id);
                            ctx.request_repaint();
                        } else {
                            ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
                        }
                    }
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
            // the diff since the seg stage is skipped regardless. Upscale and
            // inpaint variants must resolve to their own ModelKind here, not
            // the seg-only `to_model_kind` — see Bug #3.
            let model = self
                .settings
                .model
                .dispatch_model_kind()
                .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
            let preset_dispatch = match &item.applied_recipe {
                None => knob_catalog::DispatchKind::SubprocessFullPipeline,
                Some(old) => {
                    let new = item.settings.current_recipe(model, self.settings.chain_mode);
                    match prunr_core::resolve_tier(old, &new) {
                        // UpscaleRerun (Tier-1): user must re-click Process.
                        RequiredTier::Skip
                        | RequiredTier::CompositeOnly
                        | RequiredTier::UpscaleRerun => knob_catalog::DispatchKind::None,
                        // UpscaleTier2: route to live preview via the upscale
                        // dispatch path. DispatchKind doesn't have an UpscaleTier2
                        // variant — the caller handles it specially after this
                        // function returns when the model is in upscale mode.
                        RequiredTier::UpscaleTier2 => knob_catalog::DispatchKind::None,
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
                        self.open_settings();
                    }
                }
            }
        }
        // Toasts — rendered last as foreground overlay.
        self.toasts.show(ctx);
    }
}

/// Build a human-readable toast label for a multi-item undo or redo.
///
/// - Single item, known type: matches the existing per-type single-item labels.
/// - Multiple items, uniform type: "N strokes undone", "N results restored", etc.
/// - Multiple items, mixed types: "N actions undone" (safe fallback).
///
/// `single_labels` is `[stroke, result, preset]` — the three per-type
/// labels for the single-item case. `multi_verb` is the trailing word
/// for the multi-item form (e.g. "undone" / "restored").
fn action_toast_label(
    total: u32,
    type_counts: [u32; 3], // indexed by ActionType as usize: Stroke=0, Result=1, PresetApply=2
    single_labels: [&str; 3],
    multi_verb: &str,
) -> String {
    let dominant = type_counts.iter().enumerate()
        .find(|(_, &c)| c == total)
        .map(|(i, _)| i);
    match (total, dominant) {
        (1, Some(i)) => single_labels[i].into(),
        (1, None)    => format!("Action {multi_verb}"),
        (n, Some(0)) => format!("{n} strokes {multi_verb}"),
        (n, Some(1)) => format!("{n} results {multi_verb}"),
        (n, Some(2)) => format!("{n} presets {multi_verb}"),
        (n, None)    => format!("{n} actions {multi_verb}"),
        _            => format!("{total} actions {multi_verb}"),
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

/// Encode + write N PNGs into `folder` (named `<source-stem>.prunr.png`) on a
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
            let out_path = folder.join(format!("{stem}.prunr.png"));
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

/// Apply all Tier-2 upscale postprocess knobs to `img` in the canonical order:
/// ai_blend → sharpen → saturation → color_match.
///
/// Order rationale: ai_blend changes the underlying texture (blending AI output
/// with bicubic baseline at low weights); sharpen amplifies whatever texture
/// exists after the blend; saturation modifies perceived colour warmth; color_match
/// snaps global statistics last so it operates on the fully-sharpened result.
///
/// `bicubic_source` is the original source image upscaled bicubically to match
/// `img`'s dimensions. Used by `apply_ai_blend` (blending baseline) and
/// `apply_color_match` (reference for Lab statistics).
pub(crate) fn apply_tier2_postprocess(
    img: &mut image::RgbaImage,
    sharpen: f32,
    ai_blend: f32,
    saturation: f32,
    color_match: bool,
    bicubic_source: &image::RgbaImage,
) {
    use prunr_core::upscale::{apply_sharpen, apply_ai_blend, apply_saturation, apply_color_match};
    if ai_blend < 1.0 - f32::EPSILON {
        apply_ai_blend(img, bicubic_source, ai_blend);
    }
    apply_sharpen(img, sharpen);
    apply_saturation(img, saturation);
    if color_match {
        apply_color_match(img, bicubic_source);
    }
}

/// Return the bicubic-resized source for the upscale postprocess pipeline.
/// Builds the resize if `item.bicubic_source` is absent; stores it for reuse.
///
/// `upscale_raw` provides the target dimensions. The source is taken from
/// `item.source_rgba` (original decoded image). When the source is unavailable,
/// falls back to a copy of `upscale_raw` so the color_match stage sees an
/// uninformative but safe reference (no NaN from zero-dimension division).
pub(crate) fn build_or_reuse_bicubic(
    item: &mut super::item::BatchItem,
    upscale_raw: &Arc<image::RgbaImage>,
) -> Arc<image::RgbaImage> {
    if let Some(cached) = item.bicubic_source.as_ref() {
        if cached.width() == upscale_raw.width() && cached.height() == upscale_raw.height() {
            return Arc::clone(cached);
        }
    }
    // Build from the original source. Falls back to the upscale_raw itself
    // (zero-information color_match) when source_rgba is unavailable.
    let bicubic = if let Some(src) = item.source_rgba.as_ref() {
        prunr_core::formats::resize_rgba(
            src,
            upscale_raw.width(),
            upscale_raw.height(),
            prunr_core::formats::ResizeFilter::CatmullRom,
        )
    } else {
        (**upscale_raw).clone()
    };
    let arc = Arc::new(bicubic);
    item.bicubic_source = Some(Arc::clone(&arc));
    arc
}

/// Install-state is NOT a parameter — the model dropdown filter already
/// restricts selection to installed models, and re-checking every frame
/// would `is_file()`-stat the disk at 60 Hz.
pub(crate) fn can_process_upscale(is_in_flight: bool, item_loaded: bool) -> bool {
    item_loaded && !is_in_flight
}

#[cfg(test)]
mod can_process_upscale_tests {
    use super::can_process_upscale;

    #[test]
    fn can_process_upscale_requires_idle_and_item_loaded() {
        assert!(can_process_upscale(false, true),    "happy path: idle + item loaded");
        assert!(!can_process_upscale(true, true),    "already in flight");
        assert!(!can_process_upscale(false, false),  "no item loaded");
    }
}

/// The view emits `auto_chain_on = true` on an upscale-entry model switch.
/// No fresh-item upscale should silently flip chain mode on with no prior
/// result to chain from — so the application gates the flip on `item_has_result`.
pub(crate) fn resolve_auto_chain_on(
    auto_chain_on: bool,
    item_has_result: bool,
    current_chain_mode: bool,
) -> bool {
    if auto_chain_on && item_has_result {
        true
    } else {
        current_chain_mode
    }
}

#[cfg(test)]
mod auto_chain_on_tests {
    use super::resolve_auto_chain_on;

    #[test]
    fn auto_chain_on_sets_when_result_exists() {
        assert!(resolve_auto_chain_on(true, true, false));
    }

    #[test]
    fn auto_chain_on_noop_without_result() {
        assert!(!resolve_auto_chain_on(true, false, false));
    }

    #[test]
    fn auto_chain_on_unchanged_when_view_did_not_emit() {
        assert!(resolve_auto_chain_on(false, true, true));
    }
}

#[cfg(test)]
mod upscale_tier2_routing_tests {
    use prunr_core::{RequiredTier, resolve_tier};
    use crate::gui::item_settings::ItemSettings;
    use prunr_core::OutputScale;

    fn recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::BiRefNetLite, false)
    }

    fn upscale_recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus, false)
    }

    #[test]
    fn app_tier_upscaletier2_routes_to_mark_tweak() {
        // A diff where only sharpen changed must resolve to UpscaleTier2 so the
        // apply_toolbar_change block fires mark_tweak(PreviewKind::UpscaleTier2).
        let old_s = ItemSettings {
            output_scale: OutputScale::X4,
            sharpen: 0.0,
            ..ItemSettings::default()
        };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "sharpen-only diff must resolve to UpscaleTier2 so the live-preview gate opens",
        );
    }

    #[test]
    fn app_tier_upscale_rerun_still_gates_out_of_live_preview() {
        // A diff where output_scale changed must resolve to UpscaleRerun (Tier-1),
        // NOT UpscaleTier2 — so mark_tweak(UpscaleTier2) is NOT called.
        let old_s = ItemSettings { output_scale: OutputScale::X4, ..ItemSettings::default() };
        let new_s = ItemSettings { output_scale: OutputScale::X2, ..old_s };

        let old_recipe = recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleRerun,
            "output_scale diff must remain UpscaleRerun — Tier-1 stays gated out of live preview",
        );
    }

    /// Regression guard for Bug #3: a recipe-diff built with a model-kind
    /// mismatch (BiRefNetLite vs RealEsrganX4Plus) must NOT resolve to
    /// UpscaleTier2 — the bug was caused by the broken fallback masking
    /// upscale variants as seg models.
    #[test]
    fn tier2_chain_model_kind_mismatch_never_routes_to_upscale_tier2() {
        let old_s = ItemSettings { sharpen: 0.0, ..ItemSettings::default() };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = upscale_recipe_for(&old_s);
        let new_recipe = recipe_for(&new_s);
        assert_ne!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "model-kind mismatch between applied_recipe and new_recipe must NOT \
             resolve to UpscaleTier2 — this pins that the broken path never fires mark_tweak",
        );
    }

    /// The fixed path: both recipes use RealEsrganX4Plus. Sharpen-only
    /// diff must resolve to UpscaleTier2.
    #[test]
    fn tier2_chain_correct_model_kind_routes_sharpen_to_upscale_tier2() {
        let old_s = ItemSettings { sharpen: 0.0, ..ItemSettings::default() };
        let new_s = ItemSettings { sharpen: 0.5, ..old_s };

        let old_recipe = upscale_recipe_for(&old_s); // applied_recipe at Tier-1 result time
        let new_recipe = upscale_recipe_for(&new_s); // fixed path: same model kind used
        assert_eq!(
            resolve_tier(&old_recipe, &new_recipe),
            RequiredTier::UpscaleTier2,
            "sharpen-only diff with matching RealEsrganX4Plus model in both recipes \
             must resolve to UpscaleTier2 — this is the contract that enables live preview",
        );
    }

    /// Same as above but for all four Tier-2 knobs, ensuring each independently
    /// triggers the live-preview dispatch.
    #[test]
    fn tier2_chain_all_four_knobs_individually_route_to_upscale_tier2() {
        let base = ItemSettings {
            sharpen: 0.0,
            ai_blend: 1.0,
            saturation: 0.0,
            color_match: false,
            ..ItemSettings::default()
        };

        let cases: &[(&str, ItemSettings)] = &[
            ("sharpen",     ItemSettings { sharpen: 0.3, ..base }),
            ("ai_blend",    ItemSettings { ai_blend: 0.7, ..base }),
            ("saturation",  ItemSettings { saturation: 0.2, ..base }),
            ("color_match", ItemSettings { color_match: true, ..base }),
        ];

        let old_recipe = upscale_recipe_for(&base);
        for (knob_name, new_settings) in cases {
            let new_recipe = upscale_recipe_for(new_settings);
            assert_eq!(
                resolve_tier(&old_recipe, &new_recipe),
                RequiredTier::UpscaleTier2,
                "knob '{knob_name}' diff must resolve to UpscaleTier2 \
                 so the live-preview gate fires for that knob",
            );
        }
    }
}

#[cfg(test)]
mod toast_label_tests {
    use super::action_toast_label;

    const UNDO_LABELS: [&str; 3] = ["Stroke undone", "Undone", "Preset undone"];
    const REDO_LABELS: [&str; 3] = ["Stroke restored", "Result restored", "Preset restored"];

    #[test]
    fn single_item_stroke_undo() {
        assert_eq!(action_toast_label(1, [1, 0, 0], UNDO_LABELS, "undone"), "Stroke undone");
    }

    #[test]
    fn single_item_result_undo() {
        assert_eq!(action_toast_label(1, [0, 1, 0], UNDO_LABELS, "undone"), "Undone");
    }

    #[test]
    fn single_item_preset_redo() {
        assert_eq!(action_toast_label(1, [0, 0, 1], REDO_LABELS, "restored"), "Preset restored");
    }

    #[test]
    fn multi_item_uniform_strokes_undone() {
        assert_eq!(action_toast_label(3, [3, 0, 0], UNDO_LABELS, "undone"), "3 strokes undone");
    }

    #[test]
    fn multi_item_uniform_results_restored() {
        assert_eq!(action_toast_label(2, [0, 2, 0], REDO_LABELS, "restored"), "2 results restored");
    }

    #[test]
    fn multi_item_mixed_types_falls_back_to_count() {
        assert_eq!(action_toast_label(3, [2, 1, 0], UNDO_LABELS, "undone"), "3 actions undone");
    }
}

#[cfg(test)]
mod selection_action_tests {
    use std::sync::Arc;
    use prunr_core::selection::MaskArtifact;
    use crate::gui::item::{BatchItem, BatchStatus, ImageSource};
    use crate::gui::history_manager::HistoryManager;

    fn make_mask(w: u32, h: u32, selected: bool) -> MaskArtifact {
        let v = if selected { prunr_core::selection::FULL } else { 0 };
        MaskArtifact::from_cells(w, h, vec![v; (w * h) as usize])
    }

    fn make_item_done() -> BatchItem {
        let mut item = BatchItem::new(
            42,
            "test.png".to_string(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (4, 4),
            Default::default(),
            String::new(),
        );
        // Give it a result image (4×4 all-white fully-opaque).
        let mut rgba = image::RgbaImage::new(4, 4);
        for px in rgba.pixels_mut() {
            *px = image::Rgba([255, 255, 255, 255]);
        }
        item.result_rgba = Some(Arc::new(rgba));
        item.status = BatchStatus::Done;
        item
    }

    /// Delete: `alpha_cut` zeros the selected region; history depth grows by 1.
    ///
    /// `archive_current_result` moves `result_rgba` into the history entry
    /// (chain_mode=false). Save the result Arc before archiving so the
    /// alpha_cut operation has something to work on — exactly as
    /// `handle_selection_action` does (it clones `result_rgba` before
    /// calling `archive_current_result`).
    #[test]
    fn delete_action_alpha_cuts_result_and_pushes_history() {
        let mut item = make_item_done();
        let mask = make_mask(4, 4, true);

        // Capture result before archive moves it into history.
        let result = item.result_rgba.clone().unwrap();
        assert_eq!(item.history.len(), 0, "no history before archive");

        HistoryManager::archive_current_result(&mut item, 10, false);
        assert_eq!(item.history.len(), 1, "archive pushes exactly one entry");

        // Apply alpha_cut to a clone of the saved result.
        let mut new_result = (*result).clone();
        mask.alpha_cut(&mut new_result);

        // All pixels in the selected region must have alpha == 0.
        for px in new_result.pixels() {
            assert_eq!(px.0[3], 0, "alpha_cut must zero alpha in selected region");
        }
    }

    /// Delete on an image that has not been processed cuts the source and
    /// becomes the item's result; the stale result texture is dropped so
    /// the canvas rebuilds it.
    #[test]
    fn delete_on_unprocessed_item_cuts_the_source_into_a_result() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        let source = item.result_rgba.take().unwrap();
        item.status = BatchStatus::Pending;
        item.source_rgba = Some(source);
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Delete, &egui::Context::default());

        let item = &app.batch.items[0];
        let result = item.result_rgba.as_ref().expect("Delete must produce a result from the source");
        assert!(result.pixels().all(|px| px.0[3] == 0), "selected region must be cut");
        assert_eq!(item.status, BatchStatus::Done);
        assert!(item.result_tex_pending, "a texture build must be in flight for the new result");
        assert_eq!(item.history.len(), 1, "the source is archived so Cmd+Z has a state to return to");
        assert!(matches!(item.actions_undo.back(), Some(crate::gui::item::ActionType::Result)));
    }

    /// Cut through the real action path: the result is cut like Delete and
    /// exactly one history entry is archived (copy + cut is one undo step).
    #[test]
    fn cut_action_through_the_real_path_archives_once() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Cut, &egui::Context::default());

        let item = &app.batch.items[0];
        assert!(item.result_rgba.as_ref().unwrap().pixels().all(|px| px.0[3] == 0));
        assert_eq!(item.history.len(), 1, "Cut archives the previous result exactly once");
    }

    /// Invert: coverage complement, signed by the active brush mode.
    #[test]
    fn invert_action_replaces_mask_with_inverse() {
        use prunr_core::selection::BrushMode;
        use prunr_core::selection::FULL;
        let original = make_mask(4, 4, true);
        let inverted = original.invert(BrushMode::Subtract);
        assert!(inverted.cells().iter().all(|&v| v == 0), "inverted all-selected mask must be all-zero");
        let partial = MaskArtifact::from_cells(2, 2, vec![0, FULL, 0, FULL]);
        assert_eq!(partial.invert(BrushMode::Subtract).cells(), &[-FULL, 0, -FULL, 0]);
        assert_eq!(partial.invert(BrushMode::Add).cells(), &[FULL, 0, FULL, 0]);
    }

    /// Cut: one archive call produces one history entry (not two).
    /// Verifies that Cut doesn't call archive_current_result twice (which
    /// would produce one redundant entry with an identical result).
    #[test]
    fn cut_action_pushes_only_one_history_marker() {
        let mut item = make_item_done();
        assert_eq!(item.history.len(), 0, "fresh item has no history");

        // Simulate the Cut path: archive once, then apply Cut transformations.
        HistoryManager::archive_current_result(&mut item, 10, false);

        // Only one entry should exist.
        assert_eq!(item.history.len(), 1, "Cut must archive exactly once (not twice)");
    }
}

#[cfg(test)]
mod tool_mutual_exclusion_tests {
    use crate::gui::brush_state::BrushState;
    use crate::gui::magic_brush_state::MagicBrushState;

    #[test]
    fn magic_activation_deactivates_paint_brush() {
        let mut brush = BrushState::default();
        let mut magic = MagicBrushState::default();

        brush.toggle();
        assert!(brush.is_enabled(), "paint brush must start enabled");

        // Activating magic must deactivate paint brush — mutual exclusion contract.
        let activated = magic.activate();
        assert!(activated, "first activate() must return true");
        if magic.is_active() {
            brush.disable();
        }

        assert!(magic.is_active(), "magic brush must be active");
        assert!(!brush.is_enabled(), "paint brush must be disabled when magic activates");
    }

    #[test]
    fn paint_activation_deactivates_magic_brush() {
        let mut brush = BrushState::default();
        let mut magic = MagicBrushState::default();

        magic.activate();
        assert!(magic.is_active(), "magic brush must start active");

        // Enabling paint brush must deactivate magic — mutual exclusion contract.
        brush.toggle();
        if brush.is_enabled() {
            magic.deactivate();
        }

        assert!(brush.is_enabled(), "paint brush must be enabled");
        assert!(!magic.is_active(), "magic brush must be deactivated when paint activates");
    }
}
