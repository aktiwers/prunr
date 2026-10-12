use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

use egui::ViewportCommand;

use super::drag_export_state::DragExportState;
use super::item::{BatchItem, BatchStatus, Engine, ImageSource};
use super::knob_catalog::{shows, Surface};
use super::background_io::TexTarget;
use super::timeline::StepKind;
use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{adjustments_toolbar, canvas, cli_help, model_store, pipeline_flow, settings, sidebar, statusbar, toolbar};
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
    /// The Hotkeys tab is waiting for a key press for this action's slot.
    pub(crate) hotkey_capture: Option<(super::views::shortcuts::Action, usize)>,
    /// The live key bindings: the shipped table plus `settings.hotkeys`.
    /// Installed into the egui context each frame for the views.
    pub(crate) bindings: Arc<super::views::shortcuts::Bindings>,

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
    /// A popover was open when this frame began, or the click that closed
    /// one is still held down. A click outside a popover closes it while
    /// the toolbar renders, before the canvas runs, so the canvas needs
    /// this to keep that click, and a drag starting from it, off the image.
    pub(crate) popup_open_at_frame_start: bool,

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

    /// The control socket, when `PRUNR_CONTROL_PORT` is set.
    pub(crate) automation: Option<super::automation::Automation>,

    /// The Save-preset dialog's name field; `Some` while the dialog is up.
    pub(crate) save_preset: Option<String>,
    /// When a pending settings save falls due (see `save_settings_soon`).
    settings_save_due: Option<std::time::Instant>,
}

const SETTINGS_SAVE_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

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
        theme::apply_building_blocks(&mut style.visuals, &mut style.spacing);
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
        if app.magic_brush_wanted() {
            app.processor.warm_sam_sessions();
        }
        app.pending_onboarding_toast = onboarding_toast;
        app.automation = super::env_overrides::control_port()
            .and_then(|port| super::automation::Automation::start(&cc.egui_ctx, port));
        app
    }

    /// What the tool strip or a brush panel changed: persist on a
    /// settle, reset from the active preset, flip Auto-apply.
    pub(crate) fn apply_brush_change(&mut self, change: super::views::brush_chip::BrushChipOutcome) {
        if change.reset_brush_requested {
            let resolved = self.settings.resolve_active_preset(None);
            self.settings.brush.reset_popover_fields_from(&resolved.brush);
        }
        if change.committed {
            self.settings.save();
        }
        if change.action_changed {
            if let Some(id) = self.batch.selected_item().map(|i| i.id) {
                self.recut_with_selection(id);
            }
        }
    }

    /// Persists the settings once input has paused: a held key repeats
    /// at 30 Hz, and each save rewrites the whole file.
    fn save_settings_soon(&mut self) {
        self.settings_save_due = Some(std::time::Instant::now() + SETTINGS_SAVE_DELAY);
    }

    fn flush_settings_save(&mut self, ctx: &egui::Context) {
        let Some(due) = self.settings_save_due else { return };
        let now = std::time::Instant::now();
        if now >= due {
            self.settings_save_due = None;
            self.settings.save();
        } else {
            ctx.request_repaint_after(due - now);
        }
    }

    /// Every user intent with its gate, in one place: the keyboard, the
    /// control socket and any menu route here, so a gate cannot drift
    /// between surfaces.
    pub(crate) fn perform(&mut self, action: crate::gui::views::shortcuts::Action, ctx: &egui::Context) {
        use crate::gui::views::shortcuts::Action;
        let app_state = self.batch.app_state();
        // Selection actions share the action bar's gate; Copy without a
        // usable selection is the whole-result copy.
        let selection_idx = self.batch.selected_idx_clamped().filter(|_| self.can_selection_action());
        match action {
            Action::Open => self.handle_open_dialog(),
            Action::Process => {
                if matches!(app_state, AppState::Loaded | AppState::Done) {
                    self.handle_process_intent();
                }
            }
            Action::Save => {
                if self.can_save_intent() {
                    self.handle_save_selected();
                }
            }
            Action::Copy => match selection_idx {
                Some(idx) => self.handle_selection_action(idx, SelectionAction::Copy, ctx),
                None if app_state == AppState::Done => self.handle_copy(),
                None => {}
            },
            Action::Cut | Action::Delete | Action::Invert => {
                if let Some(idx) = selection_idx {
                    let selection = match action {
                        Action::Cut => SelectionAction::Cut,
                        Action::Delete => SelectionAction::Delete,
                        _ => SelectionAction::Invert,
                    };
                    self.handle_selection_action(idx, selection, ctx);
                }
            }
            Action::BeforeAfter => {
                if self.can_compare_intent() {
                    self.show_original = !self.show_original;
                }
            }
            Action::FitToWindow => self.zoom_state.pending_fit_zoom = true,
            Action::ActualSize => self.zoom_state.pending_actual_size = true,
            Action::Cancel => {
                if super::theme::any_popup_open(ctx) {
                    egui::Popup::close_all(ctx);
                } else {
                    self.apply_cancel_shortcut(ctx);
                }
            }
            Action::Shortcuts => self.toggle_hotkeys_tab(ctx),
            Action::CliHelp | Action::PipelineFlow => {
                if let Some(open) = self.help_modal_mut(action) {
                    *open = !*open;
                }
            }
            Action::Settings => self.toggle_settings_panel(ctx),
            Action::PrevImage => self.navigate_batch(ctx, NavDir::Prev),
            Action::NextImage => self.navigate_batch(ctx, NavDir::Next),
            Action::ToggleQueue => self.sidebar_hidden = !self.sidebar_hidden,
            Action::ToggleAdjustments => self.adjustments_hidden = !self.adjustments_hidden,
            Action::Undo => self.handle_undo(ctx),
            Action::Redo => self.handle_redo(ctx),
            Action::Screenshot => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            Action::BrushSmaller | Action::BrushLarger => {
                self.settings.brush.step_radius(action == Action::BrushLarger);
                self.save_settings_soon();
            }
        }
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
            force_cpu: settings.cpu_only(),
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
            settings_tab: super::views::settings::SettingsTab::Performance,
            pending_reset_confirm: false,
            hotkey_capture: None,
            bindings: Arc::new(super::views::shortcuts::Bindings::from_overrides(&settings.hotkeys)),
            model_store: None,
            pending_license_request: None,
            pending_onboarding_toast: None,
            runtime_install: None,
            hardware_install_cache: Default::default(),
            runtime_prompt: None,
            runtime_prompt_evaluated: false,
            popup_open_at_frame_start: false,
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
            automation: None,
            save_preset: None,
            settings_save_due: None,
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
        app.refresh_install_cache();
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

    /// Pick a background picture for the image at `idx`. It loads off the
    /// UI thread and becomes the background when it lands
    /// (`land_bg_image`); a cancelled pick changes nothing.
    pub(crate) fn handle_pick_bg_image(&mut self, idx: usize) {
        let Some(path) = self.system.pick_image_dialog(
            self.last_open_dir.as_deref(),
            "Choose background image",
        ) else { return };
        if let Some(id) = self.batch.items.get(idx).map(|i| i.id) {
            self.batch.request_bg_image(id, path, true);
        }
    }

    /// A background picture finished loading. A picked one replaces the
    /// background as one undo step; a preset's fills in the picture its
    /// settings name, if they still do.
    pub(crate) fn land_bg_image(&mut self, load: super::background_io::BgImageLoad) {
        use super::batch_manager::BgLanded;
        match self.batch.land_bg_image(load) {
            BgLanded::Picked { hash, path } => {
                // A preset that captured this hash can reload the picture
                // later, or after a restart.
                self.settings.bg_image_paths.insert(hash, path);
                self.settings.save();
            }
            BgLanded::Failed { picked: true, error } => {
                self.toasts.error(format!("Couldn't load background image: {error}"));
            }
            BgLanded::Failed { picked: false, error } => {
                self.toasts.error(format!("Couldn't load preset background: {error}"));
            }
            BgLanded::Filled | BgLanded::Gone => {}
        }
    }

    /// Reconcile `BatchItem.bg_image` with `settings.bg_image_hash` after a
    /// preset apply. Preset apply copies `ItemSettings` (which includes the
    /// hash) but not the picture; this loads it from the path the hash was
    /// saved with, or drops the hash.
    fn reconcile_bg_image_after_preset(&mut self, idx: usize) {
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
        let id = item.id;
        self.batch.request_bg_image(id, path, false);
    }

    pub fn handle_remove_bg(&mut self) {
        let ids: std::collections::HashSet<u64> =
            self.batch.items_to_process().into_iter().collect();
        if ids.is_empty() {
            return;
        }
        self.process_items(|item| ids.contains(&item.id), Some(StepKind::Process));
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
    /// from the timeline (`BatchItem::run_input`), then hands off
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
        let input = item.run_input(Engine::Upscale).or_else(|| item.source_rgba.clone())
            .map_or_else(|| super::processor::UpscaleInput::Source(item.source.clone()), super::processor::UpscaleInput::Decoded);
        let output_scale = item.settings.output_scale;
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
        let recipe = item.settings.current_recipe(model_kind);
        let run_model = prunr_core::upscale::model_for_scale(model_id, output_scale, self.hardware_install_cache.x2plus);
        tracing::info!(item_id, ?model_id, ?run_model, ?output_scale, "upscale dispatched");
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        }
        self.processor.dispatch_upscale(item_id, input, run_model, output_scale, self.settings.cpu_only(), recipe);
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
        !self.open_modals().is_empty()
    }

    /// The modals up this frame, by name.
    pub(crate) fn open_modals(&self) -> Vec<&'static str> {
        [
            (self.show_settings, "settings"),
            (self.show_cli_help, "cli_help"),
            (self.show_pipeline_flow, "pipeline_flow"),
            (self.model_store.is_some(), "model_store"),
            (self.pending_license_request.is_some(), "license"),
            (self.save_preset.is_some(), "save_preset"),
            (self.runtime_prompt.is_some(), "runtime_prompt"),
            (self.pending_reset_confirm, "reset_confirm"),
        ]
        .into_iter()
        .filter(|(open, _)| *open)
        .map(|(_, name)| name)
        .collect()
    }

    /// Step the targeted images (the checked ones, else the shown one) one
    /// state back on their timelines. A run still going on one is cancelled
    /// first: its result belongs to the state being left.
    pub fn handle_undo(&mut self, ctx: &egui::Context) {
        self.step_targets(ctx, BatchItem::undo_step, "undone", "Nothing to undo");
    }

    /// Mirror of `handle_undo`.
    pub fn handle_redo(&mut self, ctx: &egui::Context) {
        self.step_targets(ctx, BatchItem::redo_step, "redone", "Nothing to redo");
    }

    fn step_targets(
        &mut self,
        ctx: &egui::Context,
        step: fn(&mut BatchItem) -> Option<StepKind>,
        verb: &str,
        nothing: &str,
    ) {
        let mut stepped = Vec::new();
        for idx in self.batch.targeted_indices() {
            let id = self.batch.items[idx].id;
            let preset_before = self.batch.items[idx].settings.bg_image_hash;
            self.cancel_runs_on(id);
            self.processor.live_preview.cancel(id);
            if let Some(kind) = step(&mut self.batch.items[idx]) {
                if self.batch.items[idx].settings.bg_image_hash != preset_before {
                    self.reconcile_bg_image_after_preset(idx);
                }
                // The step's image shows its selection as the switch stood
                // then; the switch may have moved since.
                let item = &self.batch.items[idx];
                if item.selection_mask.is_some() && item.corrects_cutout(self.settings.model) && item.tensor_made_shown_step() {
                    self.dispatch_brush_rerun(idx);
                }
                stepped.push(kind);
            }
        }
        if stepped.is_empty() {
            self.toasts.info(nothing);
            return;
        }
        self.result_switch_id += 1;
        self.sync_selected_batch_textures(ctx);
        self.toasts.info(step_toast(&stepped, verb));
    }

    /// Stop whatever run `item_id` is waiting on, so its result cannot land.
    fn cancel_runs_on(&mut self, item_id: u64) {
        let Some(item) = self.batch.find_by_id(item_id) else { return };
        if item.status != BatchStatus::Processing && item.job.is_none() {
            return;
        }
        match item.job.as_ref().map(|j| j.engine) {
            Some(Engine::Erase) => self.cancel_inpaint_for(item_id),
            Some(Engine::Upscale) => self.processor.cancel_upscale(),
            Some(Engine::Segment) | None => self.processor.cancels.request_item_cancel(item_id),
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
        let Some(idx) = self.batch.index_of(item_id) else { return };
        // A cut-out re-cuts at once; the eraser's region waits for Process;
        // elsewhere the selection only feeds Delete / Copy / Cut.
        if !self.settings.model.edits_cutout() {
            return;
        }
        // With Feather on, the feathered plane is built off the UI thread
        // and its landing re-cuts; cutting now would flash the hard edge.
        let waits_for_feather = self.settings.brush.edge_feather >= 0.5 && self.batch.items[idx].selection_mask.is_some();
        if !waits_for_feather {
            self.rerun_with_strokes(idx);
        }
    }

    /// Re-cut a shown cut-out with its selection: when a result lands with a
    /// selection kept across Process (part of that run, so no step of its
    /// own), and when the selection's feathered plane lands.
    pub(crate) fn recut_with_selection(&mut self, item_id: u64) {
        let Some(idx) = self.batch.index_of(item_id) else { return };
        let item = &self.batch.items[idx];
        if (item.selection_mask.is_some() || item.edits.is_some()) && item.shows_cutout() && self.settings.model.edits_cutout() {
            self.dispatch_brush_rerun(idx);
        }
    }

    /// Re-cuts the image with the painted selection: Restore and Erase
    /// strokes take effect at once, from the cached tensor. Without a
    /// cut-out on screen there is nothing to correct yet; the selection
    /// waits and `on_batch_item_done` applies it when a result lands.
    fn rerun_with_strokes(&mut self, idx: usize) {
        if !self.batch.items[idx].shows_cutout() {
            return;
        }
        self.dispatch_brush_rerun(idx);
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
        kind: StepKind,
    ) {
        if !self.batch.commit_selection(item_id, mask, kind) {
            return;
        }
        self.apply_selection_to_active_model(item_id);
    }

    /// Dispatch a selection action (Delete / Copy / Cut / Invert / Clear).
    /// Called from `apply_toolbar_change` and from keyboard shortcuts that
    /// have a live selection.
    ///
    /// - Delete: scales alpha by coverage inside the selection on
    ///   `result_rgba`, as one undo step so Cmd+Z can
    ///   restore it.
    /// - Copy: copies selection-masked pixels to the system clipboard. No
    ///   undo step — non-destructive read.
    /// - Cut: Copy + Delete in one undo step, not two.
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

        // On a cut-out the selection is folded into the cut-out's edits:
        // Delete erases it, Restore brings it back, Clear keeps what the
        // switch is doing to it and starts a new area. Before the cut-out
        // exists, Delete cuts the photo itself.
        if self.batch.items[idx].corrects_cutout(self.settings.model) {
            use super::brush_state::CutoutAction;
            let fold = match action {
                SelectionAction::Restore => Some(CutoutAction::Restore),
                SelectionAction::Delete => Some(CutoutAction::Erase),
                SelectionAction::Clear => Some(self.settings.brush.cutout_action),
                SelectionAction::Cut => {
                    if let Some((mask, base)) = self.selection_edit_inputs(idx) {
                        self.system.copy_image(&mask.copy_to_rgba(&base));
                        self.set_temporary_status("Selection cut to clipboard");
                    }
                    Some(CutoutAction::Erase)
                }
                SelectionAction::Copy | SelectionAction::Invert => None,
            };
            if let Some(fold) = fold {
                let style = self.selection_style();
                if self.batch.bake_selection(item_id, style, fold) {
                    self.apply_selection_to_active_model(item_id);
                }
                ctx.request_repaint();
                return;
            }
        }

        match action {
            // Only a cut-out offers it; handled above.
            SelectionAction::Restore => {}
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
                let inverted = mask.invert();
                self.commit_selection_and_dispatch(item_id, inverted, StepKind::Selection);
                ctx.request_repaint();
            }
            SelectionAction::Clear => {
                if self.batch.clear_selection(item_id) {
                    self.apply_selection_to_active_model(item_id);
                }
                ctx.request_repaint();
            }
        }
    }

    /// Compare shows the original in place of a result, so it needs one.
    /// The toolbar button and the B key both read this.
    pub(crate) fn can_compare_intent(&self) -> bool {
        self.batch.selected_item().is_some_and(|i| i.has_result())
    }

    /// Selection actions act on the latest result or, before processing,
    /// on the source. Shared by the action bar, the keyboard shortcuts and
    /// the actions themselves.
    pub(crate) fn can_selection_action(&self) -> bool {
        shows(self.settings.model, Surface::Brushes)
            && self.batch.selected_item().is_some_and(|i| {
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
        let style = self.selection_style();
        let shown = item.current_selection_texture(style)
            .and_then(|t| t.feathered.clone())
            .or_else(|| style.feather(&mask, item.source_rgba.as_deref()))
            .unwrap_or(mask);
        Some((shown, base))
    }

    /// Cut the selection out of `base` and publish it as the item's result,
    /// one undo step.
    fn apply_selection_cut(
        &mut self,
        idx: usize,
        mask: &prunr_core::selection::MaskArtifact,
        base: &image::RgbaImage,
        ctx: &egui::Context,
    ) {
        let item_id = self.batch.items[idx].id;
        let mut edited = base.clone();
        mask.alpha_cut(&mut edited);
        self.publish_result(item_id, Arc::new(edited), "edit", ctx);
        self.batch.items[idx].commit(StepKind::Edit);
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
        let raw_backend = self.erase_backend();
        let seed = raw_backend.is_sd_family()
            .then(|| self.settings.brush.sd_seed.unwrap_or_else(prunr_core::inpaint_sd::fresh_seed));
        tracing::info!(item_id, ?raw_backend, ?seed, "inpaint stroke committed; dispatching");
        let tuning = self.erase_tuning(seed);
        self.processor.dispatch_inpaint(item_id, source, selection, tuning);
        self.batch.items[idx].start_erase(seed);
    }

    /// The eraser model the Model picker names (LaMa without one).
    fn erase_backend(&self) -> prunr_models::ModelId {
        self.settings.model.to_model_id().unwrap_or(prunr_models::ModelId::LaMaFp32)
    }

    /// How a fill runs under the current settings, with `seed` for a
    /// generated one.
    fn erase_tuning(&mut self, seed: Option<u64>) -> super::processor::InpaintTuning {
        self.settings.fit_sd_quality_to_install(self.hardware_install_cache.lcm_bundle);
        let raw_backend = self.erase_backend();
        let backend = if self.settings.lcm_routing_active(raw_backend) {
            prunr_models::ModelId::SdV15LcmInpaintFp16
        } else {
            raw_backend
        };
        let bs = &self.settings.brush;
        super::processor::InpaintTuning {
            sharpen: bs.inpaint_sharpen,
            feather_px: bs.inpaint_feather,
            grow_px: bs.inpaint_grow,
            fill_scale: bs.inpaint_fill_scale,
            backend,
            sd_prompt: bs.sd_prompt_for_run(),
            sd_negative_prompt: bs.sd_negative_prompt.clone(),
            sd_guidance_scale: bs.sd_guidance_scale,
            sd_scheduler: bs.sd_scheduler,
            sd_steps: bs.sd_steps,
            sd_seed: seed,
            use_taesd: bs.sd_use_taesd_effective(),
            sd_tuning: super::env_overrides::sd_tuning(prunr_core::inpaint_sd::plan_tuning(
                self.settings.keep_sd_loaded,
                self.settings.sd_memory.margin_mb(),
            )),
            cpu_only: self.settings.cpu_only(),
        }
    }

    /// "Try another" is possible: the SD eraser is chosen and the shown
    /// step is a generated fill, with nothing running on the image.
    pub fn can_reroll_intent(&self) -> bool {
        self.settings.model == super::settings::SettingsModel::SdInpaint
            && self.batch.selected_item().is_some_and(|i| !i.is_busy() && i.reroll().is_some())
    }

    /// Fill the shown fill's region again from the image it started on,
    /// with a new seed, as a new step (undo shows the fill before). A
    /// pinned seed follows the fill on screen.
    pub fn handle_reroll_intent(&mut self) {
        if !self.can_reroll_intent() {
            return;
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let Some(reroll) = item.reroll() else { return };
        let start = match reroll.start.as_ref() {
            Some(s) => s.image.load(),
            None => item.source_rgba.clone().or_else(|| item.source_dyn.as_ref().map(|d| Arc::new(d.to_rgba8()))),
        };
        let (Some(source), Some(region)) = (start, reroll.region.load()) else {
            tracing::warn!(item_id = item.id, "try another skipped: its start image or region is unavailable");
            return;
        };
        let item_id = item.id;
        let seed = prunr_core::inpaint_sd::fresh_seed();
        if self.settings.brush.sd_seed.is_some() {
            self.settings.brush.sd_seed = Some(seed);
        }
        tracing::info!(item_id, seed, "try another; dispatching");
        let tuning = self.erase_tuning(Some(seed));
        self.processor.dispatch_inpaint(item_id, source, region, tuning);
        self.batch.items[idx].start_reroll(reroll, seed);
    }

    fn pump_inpaint_results(&mut self, ctx: &egui::Context) {
        // Bridge → inpaint_rx fan-in must run before the drain.
        self.processor.pump_inpaint_subprocess();
        let (results, cancelled, errors) = self.processor.drain_inpaint_results();
        // A cancelled or failed erase records nothing; the painted region
        // stays for another try.
        for &item_id in cancelled.iter().chain(errors.iter().map(|(id, _)| id)) {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.abandon_job();
            }
        }
        if !cancelled.is_empty() || !errors.is_empty() {
            ctx.request_repaint();
        }
        for _ in &cancelled {
            self.toasts.info("Erase cancelled");
        }
        for (item_id, msg) in &errors {
            tracing::warn!(item_id, %msg, "erase failed");
            // The worker's message is user-facing already (the SD RAM
            // gate names the free and needed memory).
            self.toasts.error(msg.clone());
        }
        if results.is_empty() {
            return;
        }
        for r in results {
            let Some(item) = self.batch.find_by_id_mut(r.item_id) else { continue };
            if !item.job_is_current() {
                // Undone, redone or edited since the run started.
                continue;
            }
            self.publish_result(r.item_id, Arc::new(r.rgba), "inpaint", ctx);
            if let Some(item) = self.batch.find_by_id_mut(r.item_id) {
                // The region is filled: the step shows the result without
                // its brush marks; undoing it brings both back.
                item.invalidate_selection();
                item.finish_job();
            }
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
        item.hold_for_reveal();
        item.unpark_result();
        item.result_rgba = Some(rgba.clone());
        if item.status == BatchStatus::Pending {
            item.status = BatchStatus::Done;
        }
        item.result_tex_pending = true;
        item.thumb_pending = true;
        let source = item.source.clone();
        Self::spawn_tex_prep(
            rgba.clone(), item_id, Self::tex_name(tag, item_id, Some(switch)), TexTarget::Result, handles, ctx.clone(),
        );
        self.batch.request_thumbnail(item_id, &source, Some(&rgba));
        ctx.request_repaint();
    }

    /// Paint each finished upscale tile into its square of the item's
    /// preview texture, made on the run's first tile. Only the tile is
    /// uploaded (`set_partial`).
    fn pump_upscale_previews(&mut self, ctx: &egui::Context) {
        let tiles = self.processor.pump_upscale_previews();
        if tiles.is_empty() {
            return;
        }
        for (item_id, tile) in tiles {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if !item.job_is_current() {
                continue;
            }
            let full = [tile.full_width as usize, tile.full_height as usize];
            let texture = match &mut item.tile_preview {
                Some(t) if t.size() == full => t,
                slot => slot.insert(ctx.load_texture(
                    format!("tile-preview-{item_id}"),
                    egui::ColorImage::new(full, vec![egui::Color32::TRANSPARENT; full[0] * full[1]]),
                    egui::TextureOptions::LINEAR,
                )),
            };
            let image = egui::ColorImage::from_rgba_unmultiplied([tile.width as usize, tile.height as usize], &tile.rgba);
            texture.set_partial([tile.x as usize, tile.y as usize], image, egui::TextureOptions::LINEAR);
        }
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
                Ok(output) => {
                    let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
                    if !item.job_is_current() {
                        continue;
                    }
                    let new_rgba = output.shown;
                    item.result_rgba = Some(new_rgba.clone());
                    item.applied_recipe = Some(result_recipe);
                    item.status = BatchStatus::Done;
                    item.hold_tile_preview_for_reveal();
                    item.result_tex_pending = true;
                    item.thumb_pending = true;
                    let source = item.source.clone();
                    Self::spawn_tex_prep(
                        new_rgba.clone(), item.id,
                        Self::tex_name("upscale", item.id, Some(switch)),
                        TexTarget::Result, handles.clone(), ctx.clone(),
                    );
                    self.batch.request_thumbnail(item_id, &source, Some(&new_rgba));
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.land_upscale(output.run);
                    }
                    self.after_result_landed(item_id);
                    tracing::info!(item_id, "upscale result applied");
                }
                Err(prunr_core::CoreError::Cancelled) => {
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.job = None;
                        item.tile_preview = None;
                    }
                    tracing::info!(item_id, "upscale cancelled by user");
                    self.toasts.info("Upscale cancelled");
                }
                Err(e) => {
                    if let Some(item) = self.batch.find_by_id_mut(item_id) {
                        item.tile_preview = None;
                    }
                    let msg = e.to_string();
                    tracing::error!(item_id, %msg, "upscale dispatch failed");
                    self.toasts.error(format!("Upscale failed: {msg}"));
                }
            }
        }
        ctx.request_repaint();
    }

    /// Eager encoder: the selected image gets an embedding as soon as it
    /// has a decoded source, so Magic Brush is ready the moment it is
    /// turned on. Runs from `reconcile_selected` every frame. With the tool
    /// off it waits for the warm sessions and an idle item, and a refusal
    /// is remembered instead of retried; with the tool on, admission or
    /// encoder failure turns the tool off, so this cannot retry in a loop.
    fn ensure_magic_embedding_for_selected(&mut self, ctx: &egui::Context) {
        if !self.magic_brush_wanted() || self.magic_brush_state.has_pending_encoder() {
            return;
        }
        if let Some(wait) = self.magic_brush_state.settle_remaining(std::time::Instant::now()) {
            ctx.request_repaint_after(wait);
            return;
        }
        let Some(item) = self.batch.selected_item() else { return };
        if item.magic_brush_embedding.is_some() {
            return;
        }
        let active = self.magic_brush_state.is_active();
        if !active
            && (item.magic_preencode_refused
                || !self.processor.sam_sessions_ready()
                || self.batch.app_state() == AppState::Processing)
        {
            return;
        }
        // Decode still pending: try again next frame.
        let Some(source) = item.source_rgba.clone() else { return };
        let item_id = item.id;
        let avail_ram = crate::hardware::available_ram_mb_throttled();
        match self.processor.dispatch_sam_encoder(item_id, source, avail_ram) {
            Ok(()) => self.magic_brush_state.set_encoder_pending(true),
            Err(err) => self.on_encoder_failure(item_id, &err),
        }
    }

    /// Whether the selected model has brushes, so Magic Brush's models
    /// belong in memory.
    pub(crate) fn magic_brush_wanted(&self) -> bool {
        shows(self.settings.model, Surface::Brushes)
    }

    /// After a model switch: a model without brushes turns them off and
    /// frees Magic Brush (its sessions and each image's 16 MB embedding),
    /// so nothing encodes behind an upscale; one with brushes warms it.
    fn fit_brush_models_to_model(&mut self) {
        if self.magic_brush_wanted() {
            self.processor.warm_sam_sessions();
            return;
        }
        self.brush_state.disable();
        self.deactivate_magic_brush();
        self.processor.release_sam_sessions();
        self.batch.clear_magic_embeddings();
    }

    /// The selection overlay's style at the current zoom. The overlay and
    /// Delete / Copy / Cut read this one value, so the actions find the
    /// feathered plane the overlay built.
    pub(crate) fn selection_style(&self) -> super::background_io::SelectionStyle {
        super::background_io::SelectionStyle::from_settings(&self.settings).at_zoom(self.zoom_state.zoom)
    }

    pub(crate) fn magic_brush_preparing(&self) -> bool {
        let has_embedding = self.batch.selected_item().is_none_or(|i| i.magic_brush_embedding.is_some());
        self.magic_brush_state.preparing(has_embedding)
    }

    /// With the tool on, the user asked for it: say so and turn it off.
    /// With it off, the pre-encode was speculative: remember the refusal
    /// on the item so it is not retried every frame.
    fn on_encoder_failure(&mut self, item_id: u64, err: &str) {
        if self.magic_brush_state.is_active() {
            tracing::error!(item_id, %err, "SAM encoder failed");
            self.magic_brush_unavailable(err);
        } else {
            tracing::debug!(item_id, %err, "Magic Brush pre-encode skipped");
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                item.magic_preencode_refused = true;
            }
        }
    }

    /// Admission or encoder failure: tell the user and turn the tool off,
    /// which also stops the per-frame encoder check from retrying.
    fn magic_brush_unavailable(&mut self, err: &str) {
        self.toasts.error(format!("Magic Brush unavailable: {err}"));
        self.deactivate_magic_brush();
    }

    /// Make `bindings` the live ones and persist what differs from the
    /// shipped table.
    pub(crate) fn apply_hotkeys(&mut self, ctx: &egui::Context, bindings: super::views::shortcuts::Bindings) {
        self.settings.hotkeys = bindings.to_overrides();
        self.settings.save();
        self.bindings = Arc::new(bindings);
        super::views::shortcuts::install(ctx, Arc::clone(&self.bindings));
    }

    /// Turn Magic Brush off. The sessions stay warm and the selected
    /// image keeps its embedding, so turning it back on is instant;
    /// background images drop theirs on the next eviction pass.
    pub(crate) fn deactivate_magic_brush(&mut self) {
        self.magic_brush_state.deactivate();
    }

    /// Keep the selected image's embedding current, then drain SAM encoder
    /// and decoder results from the background rayon threads.
    /// Encoder results: write embedding to BatchItem, clear encoder_pending.
    /// Decoder results: convert to MaskArtifact, apply modifier, commit.
    fn pump_sam_results(&mut self, ctx: &egui::Context) {
        use super::processor::{merge_stroke, DecodedSelection};
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
                Err(err) => self.on_encoder_failure(result.item_id, &err),
            }
        }

        self.maybe_rethreshold_last_stroke();
        let decoder_results = self.processor.pump_sam_decoder_results();
        for result in decoder_results {
            match result.result {
                Ok(DecodedSelection::Fresh { mask, output }) => {
                    let existing = self.batch
                        .find_by_id(result.item_id)
                        .and_then(|i| i.selection_mask.clone());
                    let final_mask = merge_stroke(existing, mask);
                    tracing::info!(item_id = result.item_id, mode = ?result.mode, "SAM decoder mask committed");
                    self.commit_selection_and_dispatch(result.item_id, final_mask, StepKind::Stroke);
                    if let Some(item) = self.batch.find_by_id_mut(result.item_id) {
                        item.last_decode = item.selection_hash.map(|committed_hash| super::item::LastDecode {
                            output, mode: result.mode, committed_hash,
                            reading: result.reading,
                        });
                    }
                }
                Ok(DecodedSelection::Retuned { merged, hash }) => {
                    self.magic_brush_state.rethreshold_in_flight = false;
                    self.apply_rethreshold(result.item_id, merged, hash, result.reading);
                }
                Err(err) => {
                    tracing::error!(item_id = result.item_id, %err, "SAM decoder failed");
                    self.toasts.error(format!("Magic Brush decoder failed: {err}"));
                }
            }
        }
    }

    /// The selected image's last stroke still owns the selection and the
    /// Confidence knob moved: re-threshold its logits at the new value.
    pub(crate) fn maybe_rethreshold_last_stroke(&mut self) {
        if self.magic_brush_state.rethreshold_in_flight {
            return;
        }
        let want = self.settings.brush.magic_reading();
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &mut self.batch.items[idx];
        let (item_id, source_dims, base) = (item.id, item.dimensions, item.pre_stroke_selection());
        let Some(ld) = item.retunable_decode() else { return };
        if ld.reading == want {
            return;
        }
        self.processor.dispatch_sam_rethreshold(super::processor::SamRethresholdRequest {
            item_id, output: Arc::clone(&ld.output), mode: ld.mode,
            source_dims, reading: want, base,
        });
        self.magic_brush_state.rethreshold_in_flight = true;
    }

    /// Swap the last stroke's selection for its re-thresholded version,
    /// replaced in place so the stroke's undo entry stays the one entry.
    pub(crate) fn apply_rethreshold(&mut self, item_id: u64, merged: Arc<prunr_core::selection::MaskArtifact>, hash: u64, reading: prunr_core::MaskReading) {
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        if item.retunable_decode().is_none() {
            return;
        }
        let changed = item.replace_selection_in_place(merged, hash);
        if let Some(ld) = item.last_decode.as_mut() {
            ld.reading = reading;
            ld.committed_hash = hash;
        }
        if changed {
            self.apply_selection_to_active_model(item_id);
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
                    self.refresh_install_cache();
                    return;
                }
                InstallEvent::Failed { error } => {
                    let name = progress.runtime.display_name();
                    self.toasts.error(format!("{name} install failed: {error}"));
                    self.runtime_install = None;
                    self.refresh_install_cache();
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
                    self.refresh_install_cache();
                }
                DownloadEvent::Failed { id, error } => {
                    let name = prunr_models::descriptor(id)
                        .map_or("Model", |d| d.display_name);
                    self.toasts.error(format!("{name} download failed: {error}"));
                    self.refresh_install_cache();
                }
                DownloadEvent::Progress { .. } | DownloadEvent::Verifying { .. } | DownloadEvent::Cancelled { .. } => {}
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
    pub(crate) fn recipe_drift_tripwire(&mut self) {
        use crate::gui::live_preview::PreviewKind;
        // Only a cut-out or the filters apply the mask settings; firing a
        // preview over an erase or an upscale would clobber it with the
        // source through `run_preview`'s no-tensor fallback.
        if !shows(self.settings.model, Surface::Effect) {
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
        // The mask as the run applies it: pinned to the defaults where the
        // run uses none (Sketch on Full), so its settings never drift.
        let model = self.settings.model.to_model_kind().unwrap_or(prunr_core::ModelKind::BiRefNetLite);
        let current = item.settings.current_recipe(model).mask;
        if applied.mask == current {
            self.last_drift_check = Some((id, current_settings));
            return;
        }
        self.last_drift_check = None;
        tracing::debug!(item_id = id, "recipe-drift tripwire fired — dispatching Tier-2");
        self.processor.live_preview.mark_tweak(id, PreviewKind::Mask);
        self.processor.live_preview.flush(id);
    }

    /// Run background removal (or the No-model filter) on images imported
    /// from `id_floor` on, when the setting is on. Erasers wait for a
    /// painted region and upscales for the image to enlarge, so they
    /// never start on import.
    fn auto_process_imported(&mut self, id_floor: u64) {
        let model = self.settings.model;
        if !self.settings.auto_process_on_import || model.is_inpaint() || model.is_upscale() {
            return;
        }
        if self.batch.items.iter().any(|i| i.id >= id_floor) {
            self.process_items(|item| item.id >= id_floor, Some(StepKind::Process));
        }
    }

    /// Collect and send batch items matching `filter` for processing.
    /// Uses tier routing: compares each item's applied_recipe against current
    /// settings to determine the minimum work needed (skip / mask rerun / full).
    /// `step`: the undo step the results record, or `None` to fill in the
    /// shown one (a re-run a knob change asked for).
    fn process_items(&mut self, filter: impl Fn(&BatchItem) -> bool, step: Option<StepKind>) {
        self.batch.fit_line_modes_to(self.settings.model);
        // Pure filter mode: No model AND line_mode = Off. Skip inference +
        // subprocess entirely. (No model + EdgesOnly still needs DexiNed,
        // so falls through to the normal path with a dummy ModelKind — the
        // seg engine won't spawn because needs_segmentation is false.)
        let is_pure_filter = self.settings.model.to_model_kind().is_none()
            && self.run_settings().line_mode == prunr_core::LineMode::Off;
        if is_pure_filter {
            self.process_filter_only(filter, step);
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
        // A background image's result is released; a run may start from it.
        for item in self.batch.items.iter_mut().filter(|i| candidate_ids.contains(&i.id)) {
            item.unpark_result();
        }

        let tiers = self.classify_candidates(&candidate_ids, model);
        let process_count = tiers.tier1.len() + tiers.tier2.len() + tiers.tier_add_edge.len();
        self.notify_skip(tiers.skip_count, process_count);
        if process_count == 0 { return; }

        for id in tiers.all_process_ids() {
            if let Some(item) = self.batch.find_by_id_mut(id) {
                item.start_job(step, Engine::Segment);
            }
        }

        let tier2_work = self.build_tier2_work(&tiers.tier2);
        let add_edge_work = self.build_add_edge_work(&tiers.tier_add_edge);

        let jobs = self.settings.parallel_jobs.min(super::memory::safe_max_jobs(model));
        if tiers.tier1.len() > 1 {
            self.dispatch_with_admission(&tiers.tier1, tier2_work, add_edge_work, model, jobs);
        } else {
            self.dispatch_small_batch(&tiers.tier1, tier2_work, add_edge_work, model, jobs);
        }
    }

    /// Filter-only Process path (model=`None`). Dispatches each target item
    /// to a background thread via `BatchManager::request_filter_only` so the
    /// UI stays responsive on large batches. Results land on
    /// `bg_io.filter_only_rx`, drained in `drain_background_channels`.
    fn process_filter_only(&mut self, filter: impl Fn(&BatchItem) -> bool, step: Option<StepKind>) {
        let settings = self.run_settings();
        let dispatches: Vec<(u64, ImageSource)> = self.batch.items.iter()
            .filter(|i| filter(i) && !matches!(i.status, BatchStatus::Processing))
            .map(|i| (i.id, i.source.clone()))
            .collect();
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &dispatches.iter().map(|(id, _)| *id).collect());
        }
        for (id, source) in dispatches {
            let Some(item) = self.batch.find_by_id_mut(id) else { continue };
            item.status = BatchStatus::Processing;
            item.start_job(step, Engine::Segment);
            let input = item.job_input();
            self.batch.request_filter_only(id, &source, input, settings.fill_style);
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

            // A run that starts from an earlier result works on pixels the
            // cached tensor never saw, unless it redoes the shown cut-out
            // from that cut-out's own starting image.
            if item.starts_from_result(Engine::Segment) {
                if !item.redoes_shown_step(Engine::Segment) {
                    item.set_cached_tensor(None);
                }
                tiers.tier1.insert(item.id);
                continue;
            }

            let current_recipe = item.settings.current_recipe(model);
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
                    // Tier-2 upscale (sharpen/ai_blend/color_match)
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
                    // dispatch — fall through to a full pipeline run (live
                    // preview handles the in-process path).
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
    ) {
        use super::memory::{AdmissionController, ImageMemCost};

        let mut ctrl = AdmissionController::new(model, jobs);
        let costs: Vec<ImageMemCost> = self.batch.items.iter()
            .filter(|i| tier1_ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(
                i.id, i.dimensions, i.source.estimated_size(), super::memory::RESIDENT_HISTORY_STEPS,
            ))
            .collect();
        ctrl.enqueue(costs);

        let mut initial_items = Vec::new();
        while let Some(admitted_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(admitted_id) {
                if let Ok(bytes) = item.source.load_bytes() {
                    let input = item.job_input();
                    initial_items.push((item.id, bytes, input));
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
    ) {
        let items: Vec<_> = self.batch.items.iter_mut()
            .filter(|i| tier1_ids.contains(&i.id))
            .filter_map(|i| {
                let bytes = i.source.load_bytes().ok()?;
                let input = i.job_input();
                i.status = BatchStatus::Processing;
                Some((i.id, bytes, input))
            })
            .collect();

        // If all prep failed, don't dispatch an empty batch.
        if items.is_empty() && tier2_work.is_empty() && add_edge_work.is_empty() { return; }

        self.dispatch_batch(items, tier2_work, add_edge_work, model, jobs, None);
    }

    /// The settings a batch runs with: the shown image's — what you see
    /// is what you process. Factory defaults on an empty batch.
    fn run_settings(&self) -> super::item_settings::ItemSettings {
        self.batch.selected_idx_clamped().map(|i| self.batch.items[i].settings).unwrap_or_default()
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

        let current_settings = self.run_settings();

        // Broadcast: every item about to be processed inherits current.settings
        // so their `applied_recipe` ends up consistent with what ran.
        let process_ids: std::collections::HashSet<u64> = items.iter()
            .map(|wi| wi.0)
            .chain(tier2_items.iter().map(|ti| ti.item_id))
            .chain(add_edge_items.iter().map(|ai| ai.item_id))
            .collect();
        if let Some(shown) = self.batch.selected_idx_clamped() {
            self.batch.share_settings(shown, &process_ids);
        }

        let recipe = current_settings.current_recipe(model);
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
                force_cpu: self.settings.cpu_only(),
                line_mode: current_settings.line_mode,
                edge: current_settings.edge_settings(),
            }),
            cancels: self.processor.cancels.clone(),
            additional_items_rx,
        });
    }


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
                layers: super::drag_export::LayerJob::of(item, style, self.settings.brush.cutout_action),
                composite: item.result_ref().map(|r| (r, item.export_bg())),
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
        let default_name = super::drag_export::export_name(&item.filename, self.settings.export_format);
        let bg = item.export_bg();
        self.save_rgba_with_dialog(rgba, bg, &default_name);
    }

    /// Save one specific batch item (by index) via a save-as dialog. Used by
    /// the sidebar's per-row save button, which needs a non-selection-based
    /// entry point into the same encode-on-background pipeline as
    /// `save_current_to_file`.
    pub(crate) fn save_item_to_file(&mut self, idx: usize) {
        let (rgba, bg, default_name) = {
            let Some(item) = self.batch.items.get(idx) else { return };
            let Some(rgba) = item.result_rgba.clone() else { return };
            (rgba, item.export_bg(), super::drag_export::export_name(&item.filename, self.settings.export_format))
        };
        self.save_rgba_with_dialog(rgba, bg, &default_name);
    }

    /// Shared tail for the two single-item save paths: open the save-as
    /// dialog, then bake the background, encode in the format the chosen
    /// name asks for and write on a background thread.
    fn save_rgba_with_dialog(
        &mut self,
        rgba: Arc<image::RgbaImage>,
        bg: super::item::ExportBg,
        default_name: &str,
    ) {
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
        spawn_save_single(path, rgba, bg, format, tx);
    }

    /// One or more sidebar checkboxes selected — pick a folder, write each
    /// as `<source-stem>.prunr.png`. Encode + write run on a background thread.
    fn save_selected_to_folder(&mut self) {
        let items: Vec<(String, super::item::ResultRef, super::item::ExportBg)> = self.batch.items.iter()
            .filter(|i| i.selected && i.status == BatchStatus::Done)
            .filter_map(|item| Some((item.filename.clone(), item.result_ref()?, item.export_bg())))
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
    /// On Linux: a one-time fallback toast, no files written (winit + drag
    /// crate incompatibility; see Cargo.toml comment).
    #[allow(unused_variables)]
    pub fn initiate_drag_out(&mut self, ids: Vec<u64>, frame: &eframe::Frame) {
        if !super::drag_export::DRAG_OUT_SUPPORTED {
            if !self.drag_export.linux_notified {
                self.drag_export.linux_notified = true;
                self.toasts.info(
                    "Drag to external apps isn't supported on Linux yet.\n\
                     Use Ctrl+C to copy to clipboard, or the Save button to export."
                );
            }
            return;
        }
        let spec = self.drag_spec();
        let mut paths: Vec<PathBuf> = Vec::with_capacity(ids.len());
        for id in &ids {
            let Some(item) = self.batch.find_by_id(*id) else { continue };
            let key = super::drag_export::DragKey::of(item, spec);
            let written = self.drag_export.take_written(*id, &key)
                .map_or_else(|| super::drag_export::DragFiles::of(item, spec).write(), Ok);
            match written {
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

    /// Write the drag files of the thumbnail under the pointer ahead, on
    /// a worker, so a drag out of the window starts at once.
    fn prewrite_drag_files(&mut self) {
        if !super::drag_export::DRAG_OUT_SUPPORTED {
            return;
        }
        let Some(id) = self.drag_export.due(std::time::Instant::now()) else { return };
        let Some(idx) = self.batch.index_of(id) else { return };
        let targets = self.batch.drag_targets(idx);
        let spec = self.drag_spec();
        for id in targets {
            if self.drag_export.is_writing(id) {
                continue;
            }
            let Some(item) = self.batch.find_by_id(id) else { continue };
            let key = super::drag_export::DragKey::of(item, spec);
            if !self.drag_export.has_current(id, &key) {
                let files = super::drag_export::DragFiles::of(item, spec);
                self.drag_export.prewrite(id, key, files);
            }
        }
    }

    /// How a drag writes its files under the current settings.
    fn drag_spec(&self) -> super::drag_export::DragSpec {
        super::drag_export::DragSpec { split: self.saves_layers(), style: self.selection_style(), action: self.settings.brush.cutout_action }
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
        if let (true, super::item::ResultRef::Shown(rgba)) = (bg.is_empty(), &result) {
            let rgba = Arc::clone(rgba);
            self.finish_copy(&rgba, multi_hint);
            return;
        }
        let tx = self.batch.bg_io.copy_ready_tx.clone();
        std::thread::spawn(move || {
            if let Some(rgba) = result.load() {
                let _ = tx.send((bg.bake(&rgba), multi_hint));
            }
        });
    }

    /// Hand a copy to the clipboard and say how it went.
    fn finish_copy(&mut self, rgba: &image::RgbaImage, multi_hint: Option<String>) {
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
        // The capsule reads the slot, not the app state, so the reset
        // has to clear it.
        self.processor.end_seg_run();
        self.status.text = "Cancelled".to_string();
    }

    /// Cancel only the in-flight inpaint stroke for `item_id`. Used by
    /// the capsule's Cancel button — `handle_cancel` cancels
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

        let mut streamed_ids = Vec::new();
        while let Some(next_id) = ctrl.try_admit_next() {
            if let Some(item) = self.batch.find_by_id_mut(next_id) {
                let Ok(bytes) = item.source.load_bytes() else { continue; };
                let input = item.job_input();
                let tuple = (next_id, bytes, input);
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

    /// Keep faces found for the photo and model the image still wants, and
    /// blend them in; a search for anything else is dropped.
    fn land_faces(&mut self, landed: super::processor::FacesResult) {
        let Some(idx) = self.batch.index_of(landed.item_id) else { return };
        let stage = self.face_stage();
        let item = &mut self.batch.items[idx];
        if item.pending_for(&landed.photo, landed.model) {
            item.faces_pending = None;
        }
        if !item.wants_faces(&landed.photo, landed.model, stage) {
            return;
        }
        let faces = match landed.result {
            Ok(faces) if faces.is_empty() => {
                self.set_temporary_status("No faces found");
                faces
            }
            Ok(faces) => faces,
            Err(err) => {
                tracing::warn!(item_id = landed.item_id, %err, "face restore failed");
                self.toasts.error(format!("Restore faces failed: {err}"));
                // Kept empty so the same photo is not searched again.
                Vec::new()
            }
        };
        let item = &mut self.batch.items[idx];
        item.found_faces = Some(super::item::FoundFaces { photo: landed.photo, model: landed.model, faces: Arc::new(faces) });
        item.faces_changed();
        self.redo_faces_blend(idx);
    }

    /// The Faces chip changed: a blend change re-blends what was found; a
    /// search change re-blends now (or takes the faces out) and
    /// `pump_faces` finds them again where needed.
    fn on_faces_change(&mut self, idx: usize, change: super::views::faces_chip::FacesChange) {
        if change.search || change.blend {
            self.batch.items[idx].faces_changed();
            // Before a result exists there is nothing to blend into, and a
            // re-cut would publish the plain photo as one.
            if self.shows_face_result(idx) {
                self.redo_faces_blend(idx);
            }
        }
    }

    /// Restore faces works on the active model's stage.
    fn face_stage(&self) -> super::item::FaceStage {
        super::item::FaceStage::of(self.settings.model.role())
    }

    /// The image shows what Restore faces blends into: a finished result,
    /// or No model's photo, which is its own result.
    fn shows_face_result(&self, idx: usize) -> bool {
        let item = &self.batch.items[idx];
        let photo_is_result = self.settings.model.role() == super::knob_catalog::ModelRole::Filters;
        shows(self.settings.model, Surface::Faces) && (item.status == BatchStatus::Done || photo_is_result)
    }

    /// Blend the faces again: a Refine re-finish for a shown upscale, a
    /// re-cut otherwise.
    fn redo_faces_blend(&mut self, idx: usize) {
        if self.batch.items[idx].kept_upscale().is_some() {
            let id = self.batch.items[idx].id;
            self.processor.live_preview.mark_tweak(id, super::live_preview::PreviewKind::UpscaleTier2);
            self.processor.live_preview.flush(id);
        } else {
            self.dispatch_brush_rerun(idx);
        }
    }

    /// A run's result landed (a cut-out, No model's photo, an upscale):
    /// what the model can't know goes on top. The selection re-cuts it,
    /// and the faces found for its photo blend in.
    fn after_result_landed(&mut self, item_id: u64) {
        self.recut_with_selection(item_id);
        let Some(idx) = self.batch.index_of(item_id) else { return };
        if self.batch.items[idx].faces_input(self.face_stage()).is_some() {
            self.redo_faces_blend(idx);
        }
    }

    /// Restore faces: keep the shown image's faces found for its photo,
    /// and blend in the ones that land.
    fn pump_faces(&mut self) {
        for landed in self.processor.pump_faces_results() {
            self.land_faces(landed);
        }
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let item = &self.batch.items[idx];
        let installed = self.hardware_install_cache.has_face_model(item.settings.faces.model);
        if !item.settings.faces.on || !installed || !self.shows_face_result(idx) {
            return;
        }
        let Some(photo) = item.face_photo(self.face_stage()) else { return };
        if let Some(model) = item.faces_to_find(&photo) {
            let id = item.id;
            self.batch.items[idx].faces_pending = Some((photo.clone(), model));
            self.processor.dispatch_find_faces(id, photo, model);
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
        self.pump_upscale_previews(ctx);
        self.pump_upscale_results(ctx);
        self.pump_sam_results(ctx);
        self.pump_faces();
        self.pump_download_manager(ctx);
        self.pump_runtime_install(ctx);
        self.recipe_drift_tripwire();

        // Dispatch phase: tick() invokes the closure for each item whose
        // debounce expired this frame. The closure borrows `batch.items`
        // mutably so `build_preview_inputs` can lazily cache an
        // `Arc<DynamicImage>` on the item (built once, reused across every
        // subsequent dispatch of the drag session — see `source_dyn`).
        let is_filter_only = self.settings.model == super::settings::SettingsModel::None;
        let style = self.selection_style();
        let action = self.settings.brush.cutout_action;
        let batch_items = &mut self.batch.items;
        let wait = self.processor.live_preview.tick(|id, kind| {
            Self::build_preview_inputs(batch_items, id, kind, is_filter_only, style, action)
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
        style: super::background_io::SelectionStyle,
        action: super::brush_state::CutoutAction,
    ) -> Option<super::live_preview::DispatchInputs> {
        use super::live_preview::{DispatchInputs, PreviewKind};
        let item = items.iter_mut().find(|b| b.id == id)?;
        let version = item.timeline.version();
        // UpscaleTier2: short-circuit the seg/edge path entirely. We need only
        // the kept upscale run + knob snapshot. Return None if there is no
        // upscale run yet (Tier-1 must run first).
        if matches!(kind, PreviewKind::UpscaleTier2) {
            let upscale = item.kept_upscale()?.clone();
            let knobs = item.settings.upscale_refine();
            return Some(DispatchInputs {
                version,
                kind,
                // Unread by a Refine preview, which re-finishes the kept run.
                original: Arc::new(image::DynamicImage::new_rgba8(0, 0)),
                start: None,
                settings: item.settings,
                seg_tensor: None,
                edge_tensor: None,
                secondary_edge_tensor: None,
                cached_edge: Default::default(),
                cached_bold: Default::default(),
                cached_masked_base: None,
                correction: None,
                upscale: Some(upscale),
                upscale_tier2_knobs: Some(knobs),
                faces: item.faces_input(super::item::FaceStage::Upscale),
            });
        }
        let seg_tensor = item.hot_seg_tensor();
        // A Mask dispatch in a seg model with no cached tensor falls
        // through to `run_preview`'s source+fill_style fallback and
        // overwrites any existing `result_rgba`. Filter-only mode is
        // the legitimate user of that fallback (the filter IS the
        // pipeline). Anywhere else, refuse — better to skip the
        // dispatch than discard a prior result.
        if matches!(kind, PreviewKind::Mask)
            && !is_filter_only
            && seg_tensor.is_none()
            && item.result_rgba.is_some()
        {
            return None;
        }
        // A run on top of an earlier result (an erase, a hand edit) keeps
        // it: the preview starts where Process would, the shown step's own
        // start when it redoes that step.
        let start = item.run_start(Engine::Segment);
        let original = if let Some(rgba) = start.as_ref().and_then(|s| s.image.load()) {
            let cached = item.start_dyn_cache.as_ref()
                .and_then(|(src, dyn_img)| Arc::ptr_eq(&rgba, src).then(|| Arc::clone(dyn_img)));
            match cached {
                Some(dyn_img) => dyn_img,
                None => {
                    let dyn_img = Arc::new(image::DynamicImage::ImageRgba8((*rgba).clone()));
                    item.start_dyn_cache = Some((rgba, Arc::clone(&dyn_img)));
                    dyn_img
                }
            }
        } else {
            item.start_dyn_cache = None;
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
        let cached_edge = item.edge_planes.lookup(plane_key(item.settings.edge_settings().primary_scale()));
        let cached_bold = item.edge_planes.lookup(plane_key(prunr_core::EdgeScale::Bold));
        let cached_masked_base = item.cached_masked_base.as_ref().and_then(|(base, recipe, model)| {
            let current_recipe = prunr_core::MaskRecipe::from(&item.settings.mask_settings());
            let seg_model_match = seg_tensor.as_ref().is_some_and(|s| s.model == *model);
            (*recipe == current_recipe && seg_model_match).then(|| base.clone())
        });
        // Without a seg tensor the correction has nothing to apply to
        // (filter-only path).
        let correction = seg_tensor.as_ref().and_then(|_| item.correction_plane(style, action));
        let faces = item.faces_input(super::item::FaceStage::Photo);
        Some(DispatchInputs {
            version,
            kind, original, start, settings: item.settings,
            seg_tensor, edge_tensor, secondary_edge_tensor,
            cached_edge, cached_bold, cached_masked_base,
            correction,
            upscale: None,
            upscale_tier2_knobs: None,
            faces,
        })
    }

    /// Decompress the edge tensor for the item's currently-selected scale,
    /// using the `volatile_edge_tensor` hot cache to skip zstd work during a
    /// drag. The tensor rides through as `Arc<Vec<f32>>` so hot hits are a
    /// pointer bump, not a 1.2 MB memcpy per dispatch.
    fn edge_tensor_for_active_scale(item: &mut BatchItem) -> Option<super::live_preview::EdgeTensor> {
        Self::edge_tensor_for_scale(item, item.settings.edge_settings().primary_scale())
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
        if scale == item.settings.edge_settings().primary_scale() {
            item.volatile_edge_tensor = Some((scale, data.clone()));
        }
        Some(super::live_preview::EdgeTensor { data, height, width })
    }

    /// Critical: do NOT null `result_texture` here — the old texture must stay
    /// visible until the newly-built one lands via `drain_background_channels`.
    /// Clearing it causes the canvas to flash black for a frame (no texture
    /// to draw → BG_PRIMARY shows). Instead we spawn a tex prep for the new
    /// RGBA directly and let drain swap it in atomically when ready.
    pub(crate) fn apply_completed_previews(
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
                if r.version != item.timeline.version() {
                    // Computed for a state the item has since left.
                    continue;
                }
                let is_upscale_tier2 = matches!(r.kind, PreviewKind::UpscaleTier2);
                let new_rgba = r.rgba;
                item.unpark_result();
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
                for (planes, key) in r.new_edge.into_iter().chain(r.new_bold) {
                    item.edge_planes.store(planes, key);
                }
                if let Some(faced) = r.new_faced {
                    item.faced_photo = Some(faced);
                }
                if let Some((base, recipe, model)) = r.new_masked_base {
                    item.cached_masked_base = Some((base, recipe, model));
                }
                if let (Some(reference), Some(run)) = (r.new_reference, item.upscale.as_mut()) {
                    run.reference = Some(reference);
                }
                // Mark the dispatched MaskRecipe as applied so the
                // recipe-drift tripwire doesn't immediately re-fire on
                // a result it has already consumed.
                if let Some(applied) = item.applied_recipe.as_mut() {
                    applied.mask = r.applied_mask;
                    if let Some(knobs) = r.applied_tier2_knobs {
                        applied.upscale.sharpen_bits = knobs.sharpen.to_bits();
                        applied.upscale.ai_blend_bits = knobs.ai_blend.to_bits();
                        applied.upscale.color_match = knobs.color_match;
                    }
                }
                item.amend();
                if !is_upscale_tier2 {
                    item.record_origin(Engine::Segment, r.start);
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
                    TexTarget::LiveResult, handles.clone(), ctx.clone(),
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
        self.magic_brush_state.note_shown(selected, std::time::Instant::now());
        if selected != self.synced_selected {
            self.synced_selected = selected;
            if let Some(idx) = idx {
                self.evict_background_item_caches(idx);
                self.batch.items[idx].unpark_result();
                self.batch.items[idx].set_on_screen(true);
            }
            self.show_original = false;
        }
        if let (Some(idx), Some(id)) = (idx, selected) {
            self.batch.items[idx].settle_reveal();
            self.ensure_selected_source_decoded(idx);
            self.request_selected_textures(idx, ctx);
            self.kick_backdrop_tex_prep(id, ctx);
        }
        let style = self.selection_style();
        self.batch.ensure_selection_texture(style, ctx);
        self.ensure_magic_embedding_for_selected(ctx);
    }

    /// For every item that is not selected: drop the decoded tensors kept
    /// for slider drags, and release the full-resolution result of Done
    /// items (`BatchItem::park_result`); the timeline keeps it compressed.
    fn evict_background_item_caches(&mut self, selected_idx: usize) {
        for (i, item) in self.batch.items.iter_mut().enumerate() {
            if i == selected_idx {
                continue;
            }
            item.drop_hot_tensors();
            item.drop_upscale_reference();
            item.reveal = None;
            item.subject = None;
            // The photo with faces blended in, one image of RAM; the faces
            // themselves (a few MB) stay, so it is rebuilt without a search.
            item.faced_photo = None;
            // Up to two images; warmed again when the image is shown.
            item.set_on_screen(false);
            // A full-size picture made from the photo; rebuilt when shown.
            if matches!(item.settings.backdrop(), super::item_settings::Backdrop::Effect(_)) {
                item.backdrop_texture = None;
            }
            // 16 MB each: the shown image keeps its embedding, and with the
            // tool on so does the one shown before it.
            if !self.magic_brush_state.keeps_embedding(item.id) {
                item.magic_brush_embedding = None;
            }
            if item.park_result() {
                item.result_texture = None;
                item.result_tex_pending = false;
            }
        }
    }

    /// Build the picture behind the result off the UI thread: the
    /// background image, or the effect made from the photo (once it is
    /// decoded). `drain_background_channels` uploads it. One build at a
    /// time: a blur-radius drag builds the latest value when the one in
    /// flight lands, not a blur per frame.
    pub(crate) fn kick_backdrop_tex_prep(&mut self, item_id: u64, ctx: &egui::Context) {
        use super::item_settings::Backdrop;
        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        let backdrop = item.settings.backdrop();
        let Some(key) = item.backdrop_key() else {
            item.backdrop_texture = None;
            return;
        };
        if item.backdrop_tex_pending.is_some() || item.backdrop_texture.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let build: Box<dyn FnOnce() -> Option<image::RgbaImage> + Send> = match backdrop {
            Backdrop::Image(_) => {
                let Some(bg) = item.bg_image.clone() else { return };
                Box::new(move || Some(bg.image.to_rgba8()))
            }
            Backdrop::Effect(effect) => {
                let Some(photo) = item.backdrop_photo() else { return };
                Box::new(move || photo.load().and_then(|p| prunr_core::effect_backdrop(&p, effect)))
            }
            Backdrop::Transparent | Backdrop::Color(_) => return,
        };
        item.backdrop_tex_pending = Some(key);
        let tx = self.batch.bg_io.bg_tex_prep_tx.clone();
        let slots = self.batch.bg_io.decode_slots.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            let ci = build().map(|rgba| egui::ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw()));
            let _ = tx.send((item_id, key, ci));
            ctx.request_repaint();
        });
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
                    rgba, item_id, Self::tex_name("source", item_id, None), TexTarget::Source,
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
                    rgba, item_id, Self::tex_name("result", item_id, Some(switch)), TexTarget::Result,
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
        target: TexTarget,
        handles: super::background_io::TexPrepHandles,
        ctx: egui::Context,
    ) {
        std::thread::spawn(move || {
            // Park here past `available_parallelism()` so a Process All
            // batch-completion burst doesn't hold N × ~50 MB ColorImage
            // peaks simultaneously. Same slot pool as the bg-io decode /
            // thumbnail / filter spawns.
            let queued = std::time::Instant::now();
            let _slot = handles.slots.acquire();
            let waited_ms = queued.elapsed().as_millis() as u64;
            let (w, h) = (rgba.width(), rgba.height());
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_flat_samples().as_slice(),
            );
            tracing::debug!(item_id, waited_ms, convert_ms = queued.elapsed().as_millis() as u64 - waited_ms, "texture prepared");
            let _ = handles.tx.send((item_id, name, ci, target));
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


impl PrunrApp {
    fn poll_worker_results(&mut self, ctx: &egui::Context) {
        for _ in 0..WORKER_POLL_PER_FRAME {
            let Ok(msg) = self.processor.worker_rx.try_recv() else { break };
            match msg {
                WorkerResult::Report { item_id, update } => self.on_seg_report(item_id, update, ctx),
                WorkerResult::BatchItemDone { item_id, result, tensor_cache, edge_cache } => {
                    self.on_batch_item_done(ctx, item_id, result, tensor_cache, edge_cache);
                }
                WorkerResult::BatchComplete => self.on_batch_complete(),
                WorkerResult::Cancelled => self.on_cancelled(),
                WorkerResult::SubprocessRetry { reduced_jobs, re_queued_count } => {
                    self.on_subprocess_retry(reduced_jobs, re_queued_count);
                }
                WorkerResult::BackendReady(provider) => self.settings.note_backend(provider),
            }
        }
    }

    /// A background-removal step: the shared step words, except Loading
    /// the model, which names the models being loaded.
    fn on_seg_report(&mut self, item_id: u64, update: prunr_core::ProgressUpdate, ctx: &egui::Context) {
        if let prunr_core::ProgressUpdate::Subject(hint) = &update {
            self.show_subject(item_id, hint, ctx);
        } else if update == prunr_core::ProgressUpdate::Step(prunr_core::Step::LoadingModel) {
            let text = self.loading_model_status_text(item_id);
            self.processor.set_seg_step_text(text);
        } else {
            self.processor.apply_seg_report(update);
        }
    }

    /// Outline the subject a run has just found on the shown image, while
    /// its edges are refined. A 128 px hint costs well under a millisecond.
    fn show_subject(&mut self, item_id: u64, hint: &prunr_core::SubjectHint, ctx: &egui::Context) {
        if !self.settings.animations {
            return;
        }
        let awaited = self.batch.find_by_id(item_id).is_some_and(|i| i.status == BatchStatus::Processing && i.job_is_current());
        if !self.batch.is_selected(item_id) || !awaited {
            return;
        }
        let Some((image, order)) = super::views::effects::subject_glow(hint) else { return };
        let texture = ctx.load_texture(format!("subject-{item_id}"), image, egui::TextureOptions::LINEAR);
        if let Some(item) = self.batch.find_by_id_mut(item_id) {
            item.subject = Some(super::item::SubjectOutline { texture, shown_at: std::time::Instant::now(), order: order.into() });
        }
        ctx.request_repaint();
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

        // A cancelled run leaves the item on the state it shows.
        if matches!(result.as_ref(), Err(e) if e == crate::subprocess::protocol::CANCELLED_ERR_MSG) {
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                if item.status == BatchStatus::Processing {
                    item.abandon_job();
                }
            }
            self.admission_release_and_admit(item_id);
            self.refresh_batch_progress_status();
            return;
        }

        let Some(item) = self.batch.find_by_id_mut(item_id) else { return };
        // Skip results the item no longer waits for: cancelled, or undone,
        // redone or edited since the run started.
        if item.status != BatchStatus::Processing || !item.job_is_current() {
            return;
        }
        let succeeded = result.is_ok();
        if succeeded {
            item.hold_for_reveal();
        }
        let backend_update = item.apply_tier_result(result, tensor_cache, edge_cache, recipe_snapshot, is_selected);
        if succeeded {
            item.finish_job();
        } else {
            item.job = None;
            item.reveal = None;
            item.subject = None;
        }
        // Single completion event — useful for `RUST_LOG=prunr=debug` bug
        // reports (when did inference actually finish?) and as a stable
        // signal for any external observer (e.g. a smoke driver) that a
        // dispatch round-trip closed for this item.
        tracing::info!(item_id, status = ?item.status, "item processing complete");

        if let Some(provider) = backend_update {
            self.settings.note_backend(provider);
        }

        self.refresh_batch_progress_status();

        self.after_result_landed(item_id);

        if is_selected {
            self.result_switch_id += 1;
            self.sync_selected_batch_textures(ctx);
        }

        self.admission_release_and_admit(item_id);
        self.batch.enforce_tensor_budget();

        if super::memory::under_memory_pressure() {
            self.batch.demote_history_to_disk();
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
        self.batch.find_by_id(item_id)
            .map(|b| b.settings.current_recipe(model))
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
                format!("{} failed — first: {first_err}", super::views::fmt::count(counts.errored, "image", "images"))
            };
            self.status.text = msg.clone();
            self.toasts.warning(msg);
        } else if !still_processing {
            let msg = format!("All done \u{2014} {} processed", super::views::fmt::count(counts.done, "image", "images"));
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
                self.auto_process_imported(id_floor);
            }
        }
    }

    /// While the Hotkeys tab waits for a key, this frame's press is the
    /// new chord (Escape cancels, Backspace clears the slot), never a
    /// command. Returns true while capture is active.
    fn handle_hotkey_capture(&mut self, ctx: &egui::Context) -> bool {
        use crate::gui::views::shortcuts::{self, Chord};
        let Some((action, slot)) = self.hotkey_capture else { return false };
        let (cancel, clear, chord) = ctx.input(|i| {
            (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::Backspace), Chord::from_input(i))
        });
        if cancel {
            self.hotkey_capture = None;
        } else if clear || chord.is_some() {
            let mut next = (*self.bindings).clone();
            if let Some(c) = chord {
                if let Some(other) = next.bind(action, slot, c) {
                    self.toasts.info(format!(
                        "{} moved from \"{}\"", shortcuts::chord_display(c), shortcuts::label(other),
                    ));
                }
            } else {
                next.clear_slot(action, slot);
            }
            self.apply_hotkeys(ctx, next);
            self.hotkey_capture = None;
        }
        true
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        use crate::gui::views::shortcuts::{self, Action};
        let pressed = shortcuts::pressed(ctx);
        if std::mem::take(&mut self.pending_open_dialog) {
            self.handle_open_dialog();
        }
        // Ctrl+C arrives as egui's Copy event, intercepted in raw_input_hook.
        if std::mem::take(&mut self.pending_copy) {
            self.perform(Action::Copy, ctx);
        }
        for action in Action::ALL {
            if pressed.is(action) {
                self.perform(action, ctx);
            }
        }

        if self.pending_batch_sync {
            self.pending_batch_sync = false;
            self.sync_selected_batch_textures(ctx);
        }
    }

    /// The visibility flag behind a help modal's shortcut action; `None`
    /// for actions that are not help modals.
    fn help_modal_mut(&mut self, action: crate::gui::views::shortcuts::Action) -> Option<&mut bool> {
        use crate::gui::views::shortcuts::Action;
        match action {
            Action::CliHelp => Some(&mut self.show_cli_help),
            Action::PipelineFlow => Some(&mut self.show_pipeline_flow),
            _ => None,
        }
    }

    /// Escape dismisses the topmost interruptable state. Priority order:
    /// in-flight upscale → in-flight eraser stroke → active batch processing
    /// → active selection → open modal. Selection clear sits above modal
    /// dismissal so Esc is the natural "cancel what I just drew" key.
    /// Escape: the top-most modal first, then a run in flight, then
    /// the selection. One press does one thing.
    fn apply_cancel_shortcut(&mut self, ctx: &egui::Context) {
        if self.close_top_modal(ctx) {
            return;
        }
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
        }
    }

    /// Closes the modal on top, in stacking order; false when none is open.
    fn close_top_modal(&mut self, ctx: &egui::Context) -> bool {
        if self.save_preset.is_some() {
            self.save_preset = None;
        } else if self.pending_license_request.is_some() {
            self.pending_license_request = None;
        } else if self.model_store.is_some() {
            self.model_store = None;
        } else if let Some(rt) = self.runtime_prompt.take() {
            self.settings.snooze_runtime_prompt(rt, RUNTIME_PROMPT_SNOOZE_DAYS);
        } else if self.show_settings {
            self.close_settings(ctx);
        } else if self.show_cli_help {
            self.show_cli_help = false;
        } else if self.show_pipeline_flow {
            self.show_pipeline_flow = false;
        } else {
            return false;
        }
        true
    }

    /// Settings on the Hotkeys tab, the one list of shortcuts; closes it
    /// when that tab is already showing.
    pub(crate) fn toggle_hotkeys_tab(&mut self, ctx: &egui::Context) {
        use super::views::settings::SettingsTab;
        if self.show_settings && self.settings_tab == SettingsTab::Hotkeys {
            self.close_settings(ctx);
        } else {
            self.settings_tab = SettingsTab::Hotkeys;
            self.open_settings();
        }
    }

    fn toggle_settings_panel(&mut self, ctx: &egui::Context) {
        if self.show_settings {
            self.close_settings(ctx);
        } else {
            self.open_settings();
        }
    }

    /// Apply a changed Processor setting. Every run already carries it and
    /// each session cache keys on it; this frees the sessions built under
    /// the old choice now instead of at their idle timeout.
    pub(crate) fn apply_processor(&mut self) {
        self.processor.release_all_sessions();
        tracing::info!(processor = ?self.settings.processor, "processor setting applied");
    }

    /// Re-read what is installed, and move the eraser off a quality whose
    /// download is gone.
    pub(crate) fn refresh_install_cache(&mut self) {
        self.hardware_install_cache = super::hardware_cache::HardwareInstallCache::refresh();
        self.settings.fit_sd_quality_to_install(self.hardware_install_cache.lcm_bundle);
    }

    pub(crate) fn open_settings(&mut self) {
        self.show_settings = true;
        self.refresh_install_cache();
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

        while let Ok((item_id, name, color_image, target)) = self.batch.bg_io.tex_prep_rx.try_recv() {
            let tex = ctx.load_texture(name, color_image, egui::TextureOptions::default());
            if let Some(item) = self.batch.find_by_id_mut(item_id) {
                if target != TexTarget::Source {
                    item.result_texture = Some(tex);
                    item.result_tex_pending = false;
                    if target == TexTarget::Result {
                        item.start_reveal();
                    }
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
        let mut filtered = Vec::new();
        while let Ok((item_id, result)) = self.batch.bg_io.filter_only_rx.try_recv() {
            filtered.push(item_id);
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            match result {
                _ if !item.job_is_current() => continue,
                Ok(rgba) => {
                    item.hold_for_reveal();
                    item.unpark_result();
                    item.result_rgba = Some(rgba);
                    item.result_texture = None;
                    item.thumb_texture = None;
                    item.status = BatchStatus::Done;
                    item.finish_job();
                }
                Err(msg) => {
                    item.job = None;
                    item.status = BatchStatus::Error(msg);
                }
            }
            filter_only_arrived = true;
        }
        if filter_only_arrived {
            self.result_switch_id += 1;
        }
        for item_id in filtered {
            self.after_result_landed(item_id);
        }

        // Drain files loaded by background thread (max 5 per frame to stay responsive)
        while let Ok((rgba, hint)) = self.batch.bg_io.copy_ready_rx.try_recv() {
            self.finish_copy(&rgba, hint);
        }
        // Drain save completion notifications
        while let Ok(msg) = self.batch.bg_io.save_done_rx.try_recv() {
            if msg.contains("fail") {
                self.toasts.error(msg);
            } else {
                self.toasts.success(msg);
            }
        }

        self.pump_thumbnail_results(ctx);
        self.batch.compress_history_off_thread();

        // Drain off-thread selection textures. `ctx.load_texture` is
        // allowed here — this drain runs from `logic()`, not a render
        // closure. A result for a mask that has since changed is dropped;
        // `ensure_selection_texture` already requested the current one.
        let mut feathered_landed = Vec::new();
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
                        // Re-cut only for a feathered plane the cut-out has not
                        // had yet; a rebuild for zoom keeps the same one.
                        let had_it = item.selection_texture.as_ref().is_some_and(|t| {
                            t.feathered.is_some() && t.key.0 == result.key.0 && t.key.1.edge_feather_px == result.key.1.edge_feather_px
                        });
                        if feathered.is_some() && !had_it {
                            feathered_landed.push(result.item_id);
                        }
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

        // A feathered selection that just landed changes what the cut-out
        // applies; re-cut with it.
        for id in feathered_landed {
            self.recut_with_selection(id);
        }

        while let Ok(load) = self.batch.bg_io.bg_image_rx.try_recv() {
            self.land_bg_image(load);
            ctx.request_repaint();
        }

        // A picture that is no longer the item's backdrop (changed while it
        // was built) is dropped; the next kick builds the current one.
        while let Ok((item_id, key, ci)) = self.batch.bg_io.bg_tex_prep_rx.try_recv() {
            let Some(item) = self.batch.find_by_id_mut(item_id) else { continue };
            if item.backdrop_tex_pending == Some(key) {
                item.backdrop_tex_pending = None;
            }
            // A stale or empty build frees the slot: the next frame builds
            // the current picture.
            let Some(ci) = ci.filter(|_| item.backdrop_key() == Some(key)) else {
                ctx.request_repaint();
                continue;
            };
            // WrapMode::Repeat enables BgImageFit::Tile (UV > 1.0
            // wraps); other fits keep UVs in [0, 1] so the wrap is a
            // no-op.
            let opts = egui::TextureOptions {
                magnification: egui::TextureFilter::Linear,
                minification: egui::TextureFilter::Linear,
                wrap_mode: egui::TextureWrapMode::Repeat,
                mipmap_mode: None,
            };
            let tex = ctx.load_texture(format!("backdrop_{key:x}"), ci, opts);
            item.backdrop_texture = Some((key, tex));
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
            self.auto_process_imported(id_floor);
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
    }
}

impl eframe::App for PrunrApp {
    /// The session's temp files go on a graceful exit. Not in `Drop`:
    /// tests build many apps in one process, and each drop emptied the
    /// process-wide history dir under the others' feet.
    fn on_exit(&mut self) {
        super::drag_export::cleanup_all();
        super::history_disk::cleanup_all();
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        if let Some(frame) = self.automation.as_mut().and_then(|auto| auto.take_frame()) {
            raw_input.events.extend(frame.events);
            if let Some(modifiers) = frame.modifiers {
                raw_input.modifiers = modifiers;
            }
        }
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
        super::views::shortcuts::install(ctx, Arc::clone(&self.bindings));
        super::theme::apply_motion(ctx, self.settings.animations);
        // Held while the pointer stays down, so the press that dismissed a
        // popover is spent in full (see canvas.rs).
        let popup_open = super::theme::dismissable_popup_open(ctx);
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        self.popup_open_at_frame_start = popup_open || (self.popup_open_at_frame_start && pointer_down);
        self.poll_worker_results(ctx);
        self.handle_drag_and_drop(ctx);
        if !self.handle_hotkey_capture(ctx) {
            self.handle_keyboard_shortcuts(ctx);
        }
        super::automation::pump(self, ctx);
        self.flush_settings_save(ctx);
        drain_screenshot_replies(ctx);
        self.drain_background_channels(ctx);
        self.prewrite_drag_files();
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
        } else if self.batch.items.is_empty() && !self.adjustments_hidden {
            self.render_empty_model_bar(ui, panel_frame);
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

    fn render_empty_model_bar(&mut self, ui: &mut egui::Ui, panel_frame: egui::Frame) {
        let installed = self.hardware_install_cache;
        let change = adjustments_panel(panel_frame)
            .show_inside(ui, |ui| adjustments_toolbar::render_model_only(ui, &mut self.settings, installed))
            .inner;
        if change.model_changed {
            self.on_model_changed();
        }
        if let Some(req) = change.open_model_store {
            self.open_model_store(req, ui.ctx());
        }
    }

    /// Open the Model Store; the flyout or popover that asked for it
    /// closes, or it stays drawn over the store.
    fn open_model_store(&mut self, req: adjustments_toolbar::ModelStoreRequest, ctx: &egui::Context) {
        egui::Popup::close_all(ctx);
        self.model_store = Some(req);
    }

    fn render_adjustments_toolbar(&mut self, ui: &mut egui::Ui, panel_frame: egui::Frame) {
        let Some(idx) = self.batch.selected_idx_clamped() else { return };
        let mut toolbar_change = adjustments_toolbar::ToolbarChange::default();
        let is_processing = self.batch.app_state() == AppState::Processing || self.processor.is_upscale_in_flight();
        // Compared after the toolbar ran: a changed knob or preset is a step.
        let settings_before = self.batch.items[idx].settings;
        adjustments_panel(panel_frame)
            .show_inside(ui, |ui| {
                let can_compare = self.can_compare_intent();
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
                // Upscale row shows projected output size: the size of the
                // image the next upscale starts from.
                let source_dims = item.run_input_dims(Engine::Upscale);
                let state = adjustments_toolbar::ToolbarState {
                    magic_brush_active: self.magic_brush_state.is_active(),
                    brush_available,
                    processing: is_processing,
                    has_bg_image,
                    bg_image_label,
                    source_dims,
                    has_selection: item.selection_mask.is_some(),
                    corrects_cutout: item.corrects_cutout(settings_ref.model),
                    installed: self.hardware_install_cache,
                    show_original: self.show_original,
                    can_compare,
                    shown_seed: item.shown_seed(),
                    faces_found: item.face_photo(super::item::FaceStage::of(settings_ref.model.role())).zip(item.settings.faces.wanted())
                        .and_then(|(photo, model)| item.found_for(&photo, model)).map(|f| f.faces.len()),
                };
                toolbar_change = adjustments_toolbar::render(
                    ui,
                    &mut item.settings,
                    settings_ref,
                    super::views::preset_dropdown::PresetUi { applied: &mut item.applied_preset, save_dialog: &mut self.save_preset },
                    brush_state_ref,
                    state,
                );
            });
        self.apply_toolbar_change(ui.ctx(), toolbar_change, settings_before);
    }

    /// Whether the image at `idx` changed only in Refine's after-the-model
    /// knobs since its upscale, so its kept run can be re-finished.
    fn wants_upscale_tier2(&self, idx: usize) -> bool {
        let model = self.settings.model.dispatch_model_kind();
        self.batch.items.get(idx).zip(model).is_some_and(|(item, model)| item.refine_only_changed(model))
    }

    /// What a model switch does whatever images are loaded: fit lines and
    /// brushes to the model, drop runs and cached sessions of the old one.
    fn on_model_changed(&mut self) {
        self.batch.fit_line_modes_to(self.settings.model);
        self.settings.on_model_change_resolve_brush();
        self.fit_brush_models_to_model();
        // Cancel any in-flight upscale dispatch when the user
        // changes model mid-run — the result would land on the
        // wrong selection otherwise.
        self.processor.cancel_upscale();
        // Drop every item's kept upscale run — the buffers
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
        // Every model switch reclaims RAM from every backend the user
        // is no longer using, whatever the direction.
        self.processor.release_all_sessions();
        self.prewarm_in_process_eraser();
    }

    /// Build the LaMa / Big-LaMa / MI-GAN session in the background, so
    /// the first stroke skips the 5-10 s build. Not SD: it runs in its
    /// own subprocess, and a session here would hold ~9 GB no run reads.
    fn prewarm_in_process_eraser(&self) {
        use prunr_models::ModelId;
        let Some(id) = self.settings.model.to_model_id() else { return };
        if !matches!(id, ModelId::LaMaFp32 | ModelId::BigLaMa | ModelId::Migan) {
            return;
        }
        let cpu_only = self.settings.cpu_only();
        rayon::spawn(move || {
            if let Err(e) = prunr_core::inpaint::prewarm(id, cpu_only) {
                tracing::warn!(?id, %e, "Inpaint prewarm failed");
            }
        });
    }

    fn apply_toolbar_change(
        &mut self,
        ctx: &egui::Context,
        toolbar_change: adjustments_toolbar::ToolbarChange,
        settings_before: super::item_settings::ItemSettings,
    ) {
        use crate::gui::knob_catalog::DispatchKind;
        use crate::gui::live_preview::PreviewKind;

        let Some(idx) = self.batch.selected_idx_clamped() else { return };

        if toolbar_change.preset_applied {
            self.batch.items[idx].commit(StepKind::Preset);
            // Preset apply copied an ItemSettings into the item — including
            // any captured bg_image_hash. Reload the matching bytes from
            // the persisted path map (or clear if missing) so the recipe
            // diff invariant holds and the canvas paints the right image.
            self.reconcile_bg_image_after_preset(idx);
        }
        if toolbar_change.model_changed {
            self.on_model_changed();
        }
        if toolbar_change.brush_settings_committed {
            self.settings.save();
        }
        if let Some(req) = toolbar_change.open_model_store {
            self.open_model_store(req, ctx);
        }
        if toolbar_change.pick_bg_image {
            self.handle_pick_bg_image(idx);
        } else if toolbar_change.clear_bg_image {
            self.batch.items[idx].clear_bg_image();
            ctx.request_repaint();
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
                self.ensure_magic_embedding_for_selected(ctx);
            }
        }

        self.batch.items[idx].apply_cache_impact(toolbar_change.cache_impact);
        if !toolbar_change.preset_applied && self.batch.items[idx].settings != settings_before {
            self.batch.items[idx].commit(StepKind::Adjust);
        }
        self.on_faces_change(idx, toolbar_change.faces);

        let dispatch = self.resolve_auto_dispatch(idx, &toolbar_change);
        let item_id = self.batch.items[idx].id;
        let is_done = self.batch.items[idx].status == BatchStatus::Done;

        // Seg live-preview is meaningless in inpaint mode — no seg tensor
        // to rerun, and the no-tensor fallback would overwrite result_rgba.
        let seg_preview_active = !self.settings.model.is_inpaint();
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
                    self.process_items(|item| item.id == item_id, None);
                }
            }
        }

        // Refine's after-the-model knobs need no model, so they preview
        // whatever the Live preview setting. Checked only on a change (or
        // a slider's release), not every frame.
        let changed = self.batch.items[idx].settings != settings_before;
        if (changed || toolbar_change.commit) && self.wants_upscale_tier2(idx) {
            self.processor.live_preview.mark_tweak(item_id, PreviewKind::UpscaleTier2);
            if toolbar_change.commit {
                self.processor.live_preview.flush(item_id);
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(crate::gui::live_preview::DEBOUNCE);
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
                    let new = item.settings.current_recipe(model);
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

/// "Stroke undone" for one image; "3 steps undone" across several.
fn step_toast(steps: &[StepKind], verb: &str) -> String {
    match steps {
        [one] => format!("{} {verb}", one.noun()),
        many => format!("{} steps {verb}", many.len()),
    }
}

/// Drain any `Event::Screenshot` replies and persist them as PNGs. The
/// directory is `$PRUNR_SCREENSHOT_DIR` (test-harness sets this) or
/// `<temp>/prunr-screenshots/`. Filename is the unix-millis timestamp
/// so a scenario that fires Shift+F12 multiple times never collides.
/// Where `Action::Screenshot` writes: `PRUNR_SCREENSHOT_DIR` or the temp dir.
pub(crate) fn screenshot_dir() -> PathBuf {
    std::env::var_os("PRUNR_SCREENSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("prunr-screenshots"))
}

fn drain_screenshot_replies(ctx: &egui::Context) {
    let images: Vec<std::sync::Arc<egui::ColorImage>> = ctx.input(|i| {
        i.events.iter().filter_map(|e| match e {
            egui::Event::Screenshot { image, .. } => Some(std::sync::Arc::clone(image)),
            _ => None,
        }).collect()
    });
    if images.is_empty() { return; }
    let dir = screenshot_dir();
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

/// The second toolbar row's panel, with or without an image loaded.
fn adjustments_panel(frame: egui::Frame) -> egui::Panel {
    egui::Panel::top("adjustments_toolbar")
        .exact_size(theme::CHIP_HEIGHT + theme::SPACE_SM * 2.0)
        .frame(frame)
}

/// Encode + write one image on a background thread, then send the result
/// toast text on `tx`. Stays at module scope so neither save method needs
/// `&mut self` once they've kicked off the work.
fn spawn_save_single(
    path: PathBuf,
    rgba: Arc<image::RgbaImage>,
    bg: super::item::ExportBg,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<String>,
) {
    std::thread::spawn(move || {
        let msg = match format.encode(&bg.bake(&rgba)) {
            Ok(bytes) => match std::fs::write(&path, &bytes) {
                Ok(()) => "Saved".into(),
                Err(e) => format!("Save failed: {e}"),
            },
            Err(e) => format!("Save failed: {e}"),
        };
        let _ = tx.send(msg);
    });
}

/// One image of a layers save: its layers, and the composite to save
/// instead when it has none.
struct LayerSave {
    layers: super::drag_export::LayerJob,
    composite: Option<(super::item::ResultRef, super::item::ExportBg)>,
}

/// Render and write each image's layers into `folder` on a background
/// thread (the composite where an image has none); report the counts.
fn spawn_save_layers(folder: PathBuf, jobs: Vec<LayerSave>, format: prunr_core::ExportFormat, tx: mpsc::Sender<String>) {
    std::thread::spawn(move || {
        let mut saved = 0usize;
        let mut failed = 0usize;
        for job in jobs {
            let mut files = job.layers.render_bytes();
            if files.is_empty() {
                if let Some((rgba, bg)) = job.composite.as_ref().and_then(|(r, bg)| Some((r.load()?, bg))) {
                    if let Ok(bytes) = format.encode(&bg.bake(&rgba)) {
                        files.push((super::drag_export::export_name(job.layers.filename(), format), bytes));
                    }
                }
            }
            for (name, bytes) in files {
                match std::fs::write(folder.join(name), &bytes) {
                    Ok(()) => saved += 1,
                    Err(_) => failed += 1,
                }
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

/// Encode + write N images into `folder` (named by `drag_export::export_name`) on a
/// background thread; report aggregate counts when done.
fn spawn_save_batch(
    folder: PathBuf,
    items: Vec<(String, super::item::ResultRef, super::item::ExportBg)>,
    format: prunr_core::ExportFormat,
    tx: mpsc::Sender<String>,
) {
    std::thread::spawn(move || {
        let mut saved = 0usize;
        let mut failed = 0usize;
        for (filename, result, bg) in items {
            let out_path = folder.join(super::drag_export::export_name(&filename, format));
            let Some(rgba) = result.load() else {
                failed += 1;
                continue;
            };
            match format.encode(&bg.bake(&rgba)) {
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

/// Install-state is NOT a parameter — the model dropdown filter already
/// restricts selection to installed models, and re-checking every frame
/// would `is_file()`-stat the disk at 60 Hz.
pub(crate) fn can_process_upscale(is_in_flight: bool, item_loaded: bool) -> bool {
    item_loaded && !is_in_flight
}

#[cfg(test)]
mod background_item_tests {
    use std::sync::Arc;
    use crate::gui::item::UpscaleCache;
    use crate::gui::tests::fixtures::push_test_item;

    /// An image not on screen drops its upscale's enlarged reference (the
    /// size of the output); the model's output and input stay, so Refine
    /// rebuilds it on return. The shown image keeps its own.
    #[test]
    fn images_not_on_screen_drop_the_upscale_reference() {
        let mut app = super::PrunrApp::new_for_test();
        let image = || Arc::new(image::RgbaImage::new(8, 8));
        for id in [1, 2] {
            push_test_item(&mut app, id).upscale = Some(UpscaleCache::new(image(), image(), Some(image())));
        }
        app.batch.select_item(0);
        app.evict_background_item_caches(0);
        let kept = |i: usize| app.batch.items[i].upscale.as_ref().map(|u| u.reference.is_some());
        assert_eq!(kept(0), Some(true), "the shown image keeps its reference");
        assert_eq!(kept(1), Some(false), "the other drops it but keeps its run");
    }
}

#[cfg(test)]
mod faces_tests {
    use std::sync::Arc;
    use crate::gui::item::{FacePhoto, FoundFaces};
    use crate::gui::item_settings::FaceModel;
    use crate::gui::processor::FacesResult;
    use crate::gui::tests::fixtures::push_test_item;

    /// An item showing a cut-out of its source, Restore faces on.
    fn app_with_faces_on() -> (super::PrunrApp, Arc<image::RgbaImage>) {
        let mut app = super::PrunrApp::new_for_test();
        let source = Arc::new(image::RgbaImage::new(8, 8));
        let item = push_test_item(&mut app, 1);
        item.source_rgba = Some(Arc::clone(&source));
        item.settings.faces.on = true;
        (app, source)
    }

    fn landed(photo: &Arc<image::RgbaImage>, model: FaceModel) -> FacesResult {
        FacesResult { item_id: 1, photo: FacePhoto::Source(Arc::clone(photo)), model: model.id(), result: Ok(Vec::new()) }
    }

    /// Faces count only for the photo and model they were found with:
    /// another model or photo needs a search, and off needs nothing.
    #[test]
    fn found_faces_belong_to_one_photo_and_model() {
        let (mut app, source) = app_with_faces_on();
        let item = &mut app.batch.items[0];
        let photo = item.face_photo(crate::gui::item::FaceStage::Photo).expect("the source");
        assert_eq!(item.faces_to_find(&photo), Some(FaceModel::Gfpgan.id()));
        item.found_faces = Some(FoundFaces { photo: FacePhoto::Source(Arc::clone(&source)), model: FaceModel::Gfpgan.id(), faces: Arc::default() });
        assert!(item.faces_for(&photo).is_some() && item.faces_to_find(&photo).is_none());
        item.settings.faces.model = FaceModel::RestoreFormerPp;
        assert!(item.faces_for(&photo).is_none());
        assert_eq!(item.faces_to_find(&photo), Some(FaceModel::RestoreFormerPp.id()));
        let other = FacePhoto::Source(Arc::new(image::RgbaImage::new(8, 8)));
        assert!(item.faces_for(&other).is_none(), "another photo");
        item.settings.faces.on = false;
        assert!(item.faces_to_find(&photo).is_none() && item.faces_for(&photo).is_none());
    }

    /// Faces that land for the photo shown are kept and blended in; a
    /// search for a model no longer chosen is dropped.
    #[test]
    fn landed_faces_are_kept_only_when_still_wanted() {
        let (mut app, source) = app_with_faces_on();
        app.land_faces(landed(&source, FaceModel::RestoreFormerPp));
        assert!(app.batch.items[0].found_faces.is_none(), "not the chosen model");
        app.land_faces(landed(&source, FaceModel::Gfpgan));
        assert!(app.batch.items[0].found_faces.is_some());
        assert!(app.processor.live_preview.is_pending_for(1), "the re-blend is queued");
    }

    /// A blend change drops the faced photo built for the old blend and
    /// re-blends.
    #[test]
    fn a_blend_change_reblends() {
        let (mut app, _) = app_with_faces_on();
        app.batch.items[0].status = crate::gui::item::BatchStatus::Done;
        let faced = Arc::new(image::DynamicImage::new_rgba8(8, 8));
        let key = crate::gui::live_preview::FacedKey { faces: Arc::default(), blend: Default::default() };
        app.batch.items[0].faced_photo = Some((faced, key));
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { blend: true, ..Default::default() });
        assert!(app.batch.items[0].faced_photo.is_none());
        assert!(app.processor.live_preview.is_pending_for(1));
    }

    /// Before an image is processed a Faces change has nothing to blend
    /// into; a re-cut would publish the plain photo as the result.
    #[test]
    fn a_faces_change_before_processing_publishes_nothing() {
        let (mut app, _) = app_with_faces_on();
        app.on_faces_change(0, crate::gui::views::faces_chip::FacesChange { search: true, ..Default::default() });
        assert!(!app.processor.live_preview.is_pending_for(1));
    }

    /// Undo / redo can bring back other face settings: the faced photo and
    /// the cut-out over it are dropped, not reused.
    #[test]
    fn undo_drops_what_was_built_with_other_face_settings() {
        let (mut app, _) = app_with_faces_on();
        let item = &mut app.batch.items[0];
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        item.commit(crate::gui::timeline::StepKind::Process);
        item.settings.faces.on = false;
        item.commit(crate::gui::timeline::StepKind::Adjust);
        item.cached_masked_base = Some((Arc::new(image::RgbaImage::new(8, 8)), prunr_core::MaskRecipe::from(&item.settings.mask_settings()), prunr_core::ModelKind::Silueta));
        item.undo_step();
        assert!(item.cached_masked_base.is_none());
    }

    /// An image not on screen drops its faced photo; its faces stay.
    #[test]
    fn images_not_on_screen_drop_the_faced_photo() {
        let (mut app, source) = app_with_faces_on();
        push_test_item(&mut app, 2);
        let found = FoundFaces { photo: FacePhoto::Source(source), model: FaceModel::Gfpgan.id(), faces: Arc::default() };
        let key = crate::gui::live_preview::FacedKey { faces: Arc::default(), blend: Default::default() };
        let item = &mut app.batch.items[0];
        item.found_faces = Some(found);
        item.faced_photo = Some((Arc::new(image::DynamicImage::new_rgba8(8, 8)), key));
        app.batch.select_item(1);
        app.evict_background_item_caches(1);
        assert!(app.batch.items[0].faced_photo.is_none());
        assert!(app.batch.items[0].found_faces.is_some());
    }

    /// An image not on screen lets go of its unpacked undo / redo steps
    /// (up to two images each across a whole batch).
    #[test]
    fn images_not_on_screen_release_their_warm_steps() {
        let mut app = super::PrunrApp::new_for_test();
        for id in [1, 2] {
            let item = push_test_item(&mut app, id);
            item.on_screen = true;
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            item.commit(crate::gui::timeline::StepKind::Process);
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            item.commit(crate::gui::timeline::StepKind::Adjust);
        }
        let held = |app: &super::PrunrApp, i: usize| app.batch.items[i].warm_steps.lock().unwrap().iter().any(Option::is_some);
        assert!(held(&app, 0) && held(&app, 1));
        app.evict_background_item_caches(1);
        assert!(!held(&app, 0), "off screen: released");
        assert!(held(&app, 1), "the shown image keeps its own");
    }
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

#[cfg(test)]
mod upscale_tier2_routing_tests {
    use prunr_core::{RequiredTier, resolve_tier};
    use crate::gui::item_settings::ItemSettings;
    use prunr_core::OutputScale;

    fn recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::BiRefNetLite)
    }

    fn upscale_recipe_for(settings: &ItemSettings) -> prunr_core::ProcessingRecipe {
        settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus)
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
    fn tier2_chain_each_knob_individually_routes_to_upscale_tier2() {
        let base = ItemSettings {
            sharpen: 0.0,
            ai_blend: 1.0,
            color_match: false,
            ..ItemSettings::default()
        };

        let cases: &[(&str, ItemSettings)] = &[
            ("sharpen",     ItemSettings { sharpen: 0.3, ..base }),
            ("ai_blend",    ItemSettings { ai_blend: 0.7, ..base }),
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

/// Every kind of step on the one timeline: what it records, what undo
/// brings back, and that a result computed for a state left behind never
/// lands.
#[cfg(test)]
mod timeline_contract_tests {
    use std::sync::Arc;
    use super::PrunrApp;
    use crate::gui::item::{BatchStatus, Engine};
    use crate::gui::settings::SettingsModel;
    use crate::gui::tests::fixtures::push_test_item;
    use crate::gui::timeline::StepKind;

    fn px(v: u8) -> Arc<image::RgbaImage> {
        Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([v, v, v, 255])))
    }

    fn shown(app: &PrunrApp) -> Option<u8> {
        app.batch.items[0].result_rgba.as_ref().map(|r| r.get_pixel(0, 0)[0])
    }

    fn mask() -> prunr_core::selection::MaskArtifact {
        let mut cells = vec![0i8; 16];
        cells[5] = prunr_core::selection::FULL;
        prunr_core::selection::MaskArtifact::from_cells(4, 4, cells)
    }

    fn app_with_processed_item(model: SettingsModel) -> PrunrApp {
        let mut app = PrunrApp::new_for_test();
        app.settings.model = model;
        let item = push_test_item(&mut app, 1);
        item.dimensions = (4, 4);
        item.source_rgba = Some(px(0));
        item.result_rgba = Some(px(1));
        item.status = BatchStatus::Done;
        item.commit(StepKind::Process);
        app.batch.select_item(0);
        app
    }

    /// A saved result is named after its source, with the format's
    /// extension.
    #[test]
    fn a_saved_result_is_named_after_its_source_and_format() {
        use prunr_core::ExportFormat;
        assert_eq!(crate::gui::drag_export::export_name("beach.jpeg", ExportFormat::Png), "beach.prunr.png");
        assert_eq!(crate::gui::drag_export::export_name("beach.jpeg", ExportFormat::Jpeg), "beach.prunr.jpg");
        assert_eq!(crate::gui::drag_export::export_name("noext", ExportFormat::Webp), "noext.prunr.webp");
    }

    /// Cmd+S works whenever the Save button shows: a ticked finished
    /// image is enough even while the shown one is still unprocessed.
    #[test]
    fn save_is_possible_with_a_ticked_finished_image() {
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        push_test_item(&mut app, 2);
        app.batch.select_item(1);
        app.batch.items[0].selected = true;
        assert!(app.can_save_intent(), "the ticked image is done");
        app.batch.items[0].selected = false;
        assert!(!app.can_save_intent(), "nothing finished to save");
    }

    /// Auto-process on import runs background removal (or the No-model
    /// filter) on new images; an eraser needs a painted region and an
    /// upscale runs on the image you pick, so those wait for Process.
    #[test]
    fn auto_process_on_import_runs_only_for_cut_out_and_filter_models() {
        for model in [SettingsModel::SdInpaint, SettingsModel::Inpaint, SettingsModel::RealEsrganUpscale, SettingsModel::BiRefNetLite, SettingsModel::None] {
            let mut app = PrunrApp::new_for_test();
            app.settings.model = model;
            app.settings.auto_process_on_import = true;
            push_test_item(&mut app, 5);
            app.auto_process_imported(5);
            let ran = matches!(app.batch.items[0].status, BatchStatus::Processing);
            assert_eq!(ran, !(model.is_inpaint() || model.is_upscale()), "{model:?}");
        }
    }

    /// A batch runs with the selected image's settings, so the No-model
    /// filter path is chosen from those: another image's lines must not
    /// send the whole batch to a cut-out model.
    #[test]
    fn no_model_with_no_lines_selected_runs_the_filter_for_the_batch() {
        let mut app = app_with_processed_item(SettingsModel::None);
        push_test_item(&mut app, 2).settings.line_mode = prunr_core::LineMode::EdgesOnly;
        app.batch.select_item(0);
        app.process_items(|_| true, Some(StepKind::Process));
        assert!(app.batch.items.iter().all(|i| matches!(i.status, BatchStatus::Processing)), "both run as filters");
        assert_eq!(app.batch.items[1].settings.line_mode, prunr_core::LineMode::Off, "and show the settings they ran with");
    }

    /// A picked background picture replaces the background as one step;
    /// a pick that fails to load leaves the old one; a preset's picture
    /// lands only while the settings still name it.
    #[test]
    fn a_background_picture_lands_as_one_step_or_not_at_all() {
        use crate::gui::background_io::BgImageLoad;
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        app.batch.items[0].settings.bg = Some([9, 9, 9, 255]);
        let load = |picked, result| BgImageLoad { item_id: 1, path: "bg.png".into(), picked, result };

        let steps = app.batch.items[0].timeline.undo_steps();
        app.land_bg_image(load(true, Err("unreadable".into())));
        assert_eq!(app.batch.items[0].settings.bg, Some([9, 9, 9, 255]), "a failed pick keeps the colour");
        assert_eq!(app.batch.items[0].timeline.undo_steps(), steps);

        app.land_bg_image(load(true, Ok((image::DynamicImage::new_rgba8(2, 2), 42))));
        let item = &app.batch.items[0];
        assert_eq!(item.settings.backdrop(), super::super::item_settings::Backdrop::Image(std::num::NonZeroU64::new(42).unwrap()));
        assert_eq!(item.settings.bg, None, "the picture replaces the colour");
        assert_eq!(item.timeline.undo_steps(), steps + 1, "as one step");

        app.land_bg_image(load(false, Ok((image::DynamicImage::new_rgba8(3, 3), 77))));
        assert_eq!(app.batch.items[0].bg_image.as_ref().map(|b| b.hash), Some(42), "a picture no setting names is dropped");
    }

    /// Refine's after-the-model knobs re-finish the kept upscale; a Scale
    /// change needs a new run.
    #[test]
    fn refine_knobs_refinish_the_upscale() {
        use crate::gui::item::UpscaleCache;
        let mut app = app_with_processed_item(SettingsModel::RealEsrganUpscale);
        let item = &mut app.batch.items[0];
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus));
        item.settings.sharpen = 0.4;
        assert!(!app.wants_upscale_tier2(0), "no kept run yet");

        let img = Arc::new(image::RgbaImage::new(4, 4));
        app.batch.items[0].start_job(Some(StepKind::Upscale), Engine::Upscale);
        app.batch.items[0].land_upscale(UpscaleCache::new(img.clone(), img, None));
        assert!(app.wants_upscale_tier2(0), "a Sharpen change re-finishes the kept run");
        app.batch.items[0].settings.output_scale = prunr_core::OutputScale::X2;
        assert!(!app.wants_upscale_tier2(0), "a Scale change needs a new run");
    }

    /// A Refine preview re-finishes the kept upscale run: it needs neither
    /// the decoded photo nor a copy of it.
    #[test]
    fn a_refine_preview_needs_only_the_kept_upscale() {
        use crate::gui::item::UpscaleCache;
        use crate::gui::live_preview::PreviewKind;
        let mut app = app_with_processed_item(SettingsModel::RealEsrganUpscale);
        let item = &mut app.batch.items[0];
        item.source_rgba = None;
        let img = Arc::new(image::RgbaImage::new(4, 4));
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.land_upscale(UpscaleCache::new(img.clone(), img, None));
        let style = app.selection_style();
        let inputs = PrunrApp::build_preview_inputs(&mut app.batch.items, 1, PreviewKind::UpscaleTier2, false, style, app.settings.brush.cutout_action);
        assert!(inputs.is_some_and(|i| i.upscale.is_some()), "dispatched from the kept run");
        assert!(app.batch.items[0].source_dyn.is_none() && app.batch.items[0].start_dyn_cache.is_none(), "without decoding or copying the photo");
    }

    /// A model without brushes loads none of Magic Brush: switching to an
    /// upscaler drops every image's embedding and the encoder stays off.
    #[test]
    fn an_upscale_model_keeps_magic_brush_unloaded() {
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        app.batch.items[0].magic_brush_embedding = Some(Arc::new(prunr_core::sam::SamEmbedding {
            image_embed: vec![0.0; 4],
            high_res_feats_0: vec![0.0; 4],
            high_res_feats_1: vec![0.0; 4],
        }));
        app.settings.model = SettingsModel::RealEsrganUpscale;
        app.fit_brush_models_to_model();
        assert!(app.batch.items[0].magic_brush_embedding.is_none(), "its embedding is freed");
        assert!(!app.magic_brush_wanted(), "and the encoder stays off");
        app.settings.model = SettingsModel::Silueta;
        assert!(app.magic_brush_wanted());
    }

    /// A Full-lines run draws on the whole photo: a Pre-filter change must
    /// run the lines again, never compose them onto a cut-out left in the
    /// cache by an earlier Off run.
    #[test]
    fn a_full_lines_rerun_never_composes_onto_an_old_cutout() {
        use crate::gui::worker::{CompressedTensor, TensorCache};
        let mut app = app_with_processed_item(SettingsModel::Silueta);
        let item = &mut app.batch.items[0];
        item.record_origin(Engine::Segment, None);
        item.set_cached_tensor(CompressedTensor::from_raw(TensorCache { data: vec![0.5; 16], height: 4, width: 4, model: prunr_core::ModelKind::Silueta }));
        item.settings.line_mode = prunr_core::LineMode::EdgesOnly;
        item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
        item.settings.input_transform = prunr_core::InputTransform::Grayscale;
        let ids = std::collections::HashSet::from([1]);
        let tiers = app.classify_candidates(&ids, prunr_core::ModelKind::Silueta);
        assert!(tiers.tier_add_edge.is_empty(), "no lines composed onto the cached cut-out");
        assert!(tiers.tier1.contains(&1), "the lines run again on the whole photo");
    }

    /// An erase lands as one step that shows the filled image without its
    /// brush marks; undoing it brings back both the image and the marks.
    #[test]
    fn an_erase_lands_as_one_step_and_undo_brings_the_strokes_back() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);

        let item = &app.batch.items[0];
        assert_eq!(shown(&app), Some(7));
        assert!(item.selection_mask.is_none(), "the filled region's marks are gone");
        assert_eq!(item.timeline.undo_steps(), 3);

        app.handle_undo(&ctx);
        assert_eq!(shown(&app), Some(1));
        assert!(app.batch.items[0].selection_mask.is_some(), "undo brings the marks back with the image");
    }

    /// The control socket reports the app busy while an erase runs, so a
    /// driver waits for the fill instead of reading the working overlay.
    #[test]
    fn the_app_is_busy_while_an_erase_runs() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        assert!(!super::super::automation::state::dump(&app).busy);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        assert!(super::super::automation::state::dump(&app).busy, "an erase in flight");
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);
        assert!(app.batch.items[0].job.is_none(), "landed: only the texture upload is left");
    }

    /// "Try another" reruns a generated fill from the image it started
    /// on, with the region painted for it and a new seed, as a new step:
    /// undo shows the fill before. It keeps working after a re-roll.
    #[test]
    fn try_another_reruns_the_fill_from_its_start_as_a_new_step() {
        let mut app = app_with_processed_item(SettingsModel::SdInpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        let painted = app.batch.items[0].timeline.current().selection.clone().expect("the stroke's region");
        let item = &mut app.batch.items[0];
        assert!(item.reroll().is_none(), "nothing generated yet");
        let first_start = item.run_start(Engine::Erase).map(|s| s.dims);
        item.start_erase(Some(1));
        item.result_rgba = Some(px(10));
        item.invalidate_selection();
        item.finish_job();
        let r = item.reroll().expect("a generated fill can be re-rolled");
        assert!(Arc::ptr_eq(&r.region, &painted), "the region painted for it");
        assert_eq!(r.start.as_ref().map(|s| s.dims), first_start);
        item.start_reroll(r, 2);
        item.result_rgba = Some(px(20));
        item.finish_job();
        assert_eq!(item.shown_seed(), Some(2));
        let again = item.reroll().expect("re-rolls chain");
        assert!(Arc::ptr_eq(&again.region, &painted));
        assert_eq!(again.start.as_ref().map(|s| s.dims), first_start, "still from the pre-erase image");
        item.undo_step();
        assert_eq!(item.shown_seed(), Some(1), "undo shows the fill before");
    }

    /// "Try another" is offered exactly when the SD eraser is chosen and
    /// the shown step is a generated fill with nothing running on it.
    #[test]
    fn try_another_is_offered_only_for_a_shown_generated_fill() {
        let mut app = app_with_processed_item(SettingsModel::SdInpaint);
        assert!(!app.can_reroll_intent(), "nothing generated yet");
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        let item = &mut app.batch.items[0];
        item.start_erase(Some(1));
        assert!(!app.can_reroll_intent(), "not while a fill runs");
        let item = &mut app.batch.items[0];
        item.result_rgba = Some(px(10));
        item.invalidate_selection();
        item.finish_job();
        assert!(app.can_reroll_intent());
        app.settings.model = SettingsModel::Inpaint;
        assert!(!app.can_reroll_intent(), "LaMa fills are not generated");
        app.settings.model = SettingsModel::SdInpaint;
        app.batch.items[0].undo_step();
        assert!(!app.can_reroll_intent(), "the shown step is the stroke");
    }

    /// A refused or failed erase ends its run: the item stops waiting,
    /// so it is not shown working forever.
    #[test]
    fn a_failed_erase_ends_its_run() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.processor.fail_inpaint_for_test(1, "refused to load: only 6 GB RAM free");
        app.pump_inpaint_results(&ctx);
        assert!(app.batch.items[0].job.is_none(), "the run is over");
        assert!(app.batch.items[0].selection_mask.is_some(), "the painted region stays for another try");
    }

    /// An erase result that lands after an undo is dropped.
    #[test]
    fn an_erase_after_an_undo_does_not_land() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        app.batch.items[0].start_job(Some(StepKind::Erase), Engine::Erase);
        app.handle_undo(&ctx);
        app.processor.land_inpaint_for_test(1, (*px(7)).clone());
        app.pump_inpaint_results(&ctx);
        assert_eq!(shown(&app), Some(1));
        assert!(app.batch.items[0].timeline.can_redo(), "the undone stroke is still there to redo");
    }

    /// A Process result that lands after an undo is dropped, and the undo
    /// asked the worker to stop.
    #[test]
    fn a_process_result_after_an_undo_does_not_land() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        app.handle_undo(&ctx);
        assert!(app.processor.cancels.is_cancelled(1), "the run is asked to stop");

        let late = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: String::new() };
        app.on_batch_item_done(&ctx, 1, Ok(late), None, None);
        assert_eq!(shown(&app), None, "undo went back to the original and stays there");
        assert_eq!(app.batch.items[0].status, BatchStatus::Pending);
    }

    /// A Process result that lands on the state it started from is a step.
    #[test]
    fn a_process_result_lands_as_its_own_step() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let item = &mut app.batch.items[0];
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.status = BatchStatus::Processing;
        let fresh = prunr_core::ProcessResult { rgba_image: (*px(9)).clone(), active_provider: String::new() };
        app.on_batch_item_done(&ctx, 1, Ok(fresh), None, None);
        assert_eq!(shown(&app), Some(9));
        app.handle_undo(&ctx);
        assert_eq!(shown(&app), Some(1), "undo shows the result it replaced");
    }

    /// A knob change is an Adjust step; undo brings the old value back,
    /// with the image that went with it.
    #[test]
    fn a_knob_change_is_a_step_and_undo_restores_it() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::BiRefNetLite);
        let before = app.batch.items[0].settings;
        app.batch.items[0].settings.gamma = 1.8;
        app.apply_toolbar_change(&ctx, Default::default(), before);
        assert_eq!(app.batch.items[0].timeline.undo_steps(), 2);

        app.handle_undo(&ctx);
        assert_eq!(app.batch.items[0].settings.gamma, before.gamma);
        app.handle_redo(&ctx);
        assert_eq!(app.batch.items[0].settings.gamma, 1.8);
    }

    /// Clearing the selection is a step: undo brings it back.
    #[test]
    fn clearing_the_selection_is_a_step() {
        let ctx = egui::Context::default();
        let mut app = app_with_processed_item(SettingsModel::Inpaint);
        app.commit_selection_and_dispatch(1, mask(), StepKind::Stroke);
        assert!(app.batch.clear_selection(1));
        assert!(app.batch.items[0].selection_mask.is_none());
        app.handle_undo(&ctx);
        assert!(app.batch.items[0].selection_mask.is_some());
    }
}

#[cfg(test)]
mod toast_label_tests {
    use super::step_toast;
    use crate::gui::timeline::StepKind;

    #[test]
    fn one_image_names_the_step_and_several_count_them() {
        assert_eq!(step_toast(&[StepKind::Stroke], "undone"), "Stroke undone");
        assert_eq!(step_toast(&[StepKind::Erase], "redone"), "Erase redone");
        assert_eq!(step_toast(&[StepKind::Stroke, StepKind::Process], "undone"), "2 steps undone");
    }
}

#[cfg(test)]
mod selection_action_tests {
    use std::sync::Arc;
    use prunr_core::selection::MaskArtifact;
    use crate::gui::item::{BatchItem, BatchStatus, ImageSource};

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
        assert_eq!(item.timeline.undo_steps(), 1, "one Edit step, back to the unprocessed source");
    }

    /// Cut through the real action path: the result is cut like Delete and
    /// copy + cut is one undo step.
    #[test]
    fn cut_through_the_real_path_is_one_step() {
        use crate::gui::views::selection_action_bar::SelectionAction;
        let mut app = super::PrunrApp::new_for_test();
        let mut item = make_item_done();
        item.selection_mask = Some(Arc::new(make_mask(4, 4, true)));
        app.batch.items.push(item);

        app.handle_selection_action(0, SelectionAction::Cut, &egui::Context::default());

        let item = &app.batch.items[0];
        assert!(item.result_rgba.as_ref().unwrap().pixels().all(|px| px.0[3] == 0));
        assert_eq!(item.timeline.undo_steps(), 1, "Cut is exactly one step");
    }

    /// Invert: coverage complement, signed by the active brush mode.
    #[test]
    fn invert_action_replaces_mask_with_inverse() {
        use prunr_core::selection::FULL;
        let original = make_mask(4, 4, true);
        let inverted = original.invert();
        assert!(inverted.cells().iter().all(|&v| v == 0), "inverted all-selected mask must be all-zero");
        let partial = MaskArtifact::from_cells(2, 2, vec![0, FULL, 0, FULL]);
        assert_eq!(partial.invert().cells(), &[-FULL, 0, -FULL, 0]);
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

#[cfg(test)]
mod hotkey_capture_tests {
    use super::PrunrApp;
    use crate::gui::views::shortcuts::Action;

    /// A key pressed for the Hotkeys capture field never reaches the
    /// command handler in that frame: Escape cancels the capture and
    /// leaves the settings modal open.
    #[test]
    fn escape_during_hotkey_capture_cancels_without_closing_settings() {
        let mut app = PrunrApp::new_for_test();
        app.show_settings = true;
        app.hotkey_capture = Some((Action::Undo, 0));
        let ctx = egui::Context::default();
        let modifiers = egui::Modifiers::NONE;
        ctx.begin_pass(egui::RawInput {
            modifiers,
            events: vec![egui::Event::Key {
                key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers,
            }],
            ..Default::default()
        });
        let captured = app.handle_hotkey_capture(&ctx);
        if !captured {
            app.handle_keyboard_shortcuts(&ctx);
        }
        let _ = ctx.end_pass();
        assert!(captured);
        assert!(app.hotkey_capture.is_none());
        assert!(app.show_settings, "Escape went to the capture field, not to the modal");
    }
}
