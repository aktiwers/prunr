//! The GUI coordinator: `PrunrApp`'s state, frame loop, intent routing, modals and model switches.

use std::path::PathBuf;
use std::sync::{mpsc, Arc};

use egui::ViewportCommand;

use super::item::{BatchItem, BatchStatus, Engine};
use super::timeline::StepKind;
use super::settings::Settings;
use super::state::AppState;
use super::system_bridge::{DialogAnswer, DialogRequest};
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, cli_help, model_store, pipeline_flow, settings, sidebar, statusbar, toolbar};
use super::views::selection_action_bar::SelectionAction;

mod adjustments;
mod dispatch;
mod export;
mod import;
mod preview;
mod results;
mod textures;
mod tools;

/// Quiet work shorter than this needs no pill; longer, the user wonders
/// whether anything is happening.
const BUSY_PILL_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

/// How often the frame loop looks for a file dialog's answer.
const DIALOG_POLL: std::time::Duration = std::time::Duration::from_millis(250);

/// How often the frame loop wakes itself while async work is in flight.
/// The poll stops on its own once the work drains.
const ASYNC_POLL: std::time::Duration = std::time::Duration::from_millis(50);

/// The model a No-model recipe is built and diffed with. Any seg model
/// would do, since No model never runs one, but every site that builds or
/// diffs a recipe must pick the same one or `applied_recipe` reads as drift.
const NO_MODEL_RECIPE_KIND: prunr_core::ModelKind = prunr_core::ModelKind::BiRefNetLite;

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
    /// "Set focus" was clicked on the Depth chip: the next click on the
    /// image sets the focus point instead of panning or painting.
    pub(crate) picking_depth_focus: bool,
    /// Since when quiet background work has gone on with nothing landing.
    busy_since: Option<std::time::Instant>,
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
    /// Which Settings tab the modal is showing; not persisted.
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
        // Worker is spawned below after prewarm is created
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
            egui::FontId::monospace(theme::FONT_SIZE_SECONDARY),
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
        // click pays a full subprocess respawn (~15 s on this CPU).
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
        // A saved model that has since become OnDemand may not be on disk:
        // queue a one-time toast pointing at the Model Store. Bundled-only
        // users see nothing.
        let onboarding_toast = settings.model.to_model_id()
            .filter(|id| !prunr_models::is_available(*id))
            .and_then(prunr_models::descriptor)
            .map(|d| format!(
                "{} is a download now. {} in the Model Store.",
                d.display_name,
                crate::gui::views::get_model::label(d.id),
            ));

        // Subprocess worker: inference runs in a child process for OOM
        // isolation. Pre-warm a subprocess with the startup config so the
        // first Process click skips the 1–5s model-load cost. Filter-only
        // mode (SettingsModel::None) skips pre-warm — no ORT session
        // needed for pure CPU filters.
        let prewarm = Self::initial_engines(&settings);
        let (worker_tx, worker_rx) = spawn_worker(worker_ctx, prewarm);

        let mut app = Self::init_state(settings, super::system_bridge::SystemBridge::new(), worker_tx, worker_rx);
        // The saved model's tools, as a switch to it would set them: an
        // eraser starts with its brush on, a cut-out warms Magic Brush.
        app.fit_brush_models_to_model();
        app.pending_onboarding_toast = onboarding_toast;
        app.automation = super::env_overrides::control_port()
            .and_then(|port| super::automation::Automation::start(&cc.egui_ctx, port));
        app
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
            Action::Shortcuts => self.toggle_hotkeys_tab(),
            Action::CliHelp => self.show_cli_help ^= true,
            Action::PipelineFlow => self.show_pipeline_flow ^= true,
            Action::Settings => self.toggle_settings_panel(),
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

    /// The engines to load ahead of Process, at start-up or on a switch;
    /// `None` for a model that is not a cut-out (No model's filters run
    /// without a worker).
    fn initial_engines(settings: &Settings) -> Option<super::worker::EngineSpec> {
        Some(super::worker::EngineSpec {
            model: settings.model.to_model_kind()?,
            jobs: settings.parallel_jobs,
            force_cpu: settings.cpu_only(),
        })
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        // Test stub: no real platform clipboard available; SystemBridge::new
        // gracefully no-ops copy_image when the clipboard handle failed to
        // initialize, so this is safe in headless test envs. File dialogs
        // never show; tests answer them.
        let mut app = Self::init_state(
            Settings::default(),
            super::system_bridge::SystemBridge::new_for_test(),
            worker_tx,
            worker_rx,
        );
        app.hardware_install_cache = super::hardware_cache::HardwareInstallCache::all_installed();
        app.processor.stub_model_runs();
        app.fit_brush_models_to_model();
        app
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
            picking_depth_focus: false,
            busy_since: None,
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
            // Clears the brush's modifier hint in the same corner.
            toasts: super::toasts::Toasts::new(
                egui_notify::Anchor::BottomLeft,
                egui::vec2(theme::SPACE_MD, theme::STATUS_BAR_HEIGHT + theme::SPACE_SM + theme::SPACE_LG),
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
            let name = import::display_name(&path, import::UNTITLED);
            let _ = app.batch.bg_io.file_load_tx.send((path, name));
        }
        // Test-harness escape hatch: PRUNR_OPEN_TAB pre-selects a Settings
        // tab and auto-opens the modal at startup. Lets the harness capture
        // each tab without driving mouse clicks (unreliable under Xephyr's
        // coord-space mismatch). Values are `SettingsTab` labels
        // (case-sensitive). Not exposed on --help.
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

    /// A success toast, echoed on the status line for a while.
    fn status_success(&mut self, text: &str) {
        self.toasts.success(text);
        self.status.set_temporary(text);
    }

    /// An error toast, echoed on the status line for a while.
    fn status_error(&mut self, text: &str) {
        self.toasts.error(text);
        self.status.set_temporary(text);
    }

    /// Pick a background picture for the image at `idx`. It loads off the
    /// UI thread and becomes the background when it lands
    /// (`land_bg_image`); a cancelled pick changes nothing.
    pub(crate) fn handle_pick_bg_image(&mut self, idx: usize) {
        if let Some(item_id) = self.batch.items.get(idx).map(|i| i.id) {
            self.ask_dialog(DialogRequest::Background { item_id });
        }
    }

    fn on_background_picked(&mut self, item_id: u64, path: PathBuf) {
        if self.batch.find_by_id(item_id).is_some() {
            self.batch.request_bg_image(item_id, path, true);
        }
    }

    /// Ask for a file dialog; every one starts in the last opened folder.
    fn ask_dialog(&mut self, request: DialogRequest) {
        if !self.system.ask_dialog(request, self.last_open_dir.clone()) {
            self.toasts.info("A file dialog is already open. Close it first.");
        }
    }

    /// Resume the action whose file dialog the user answered.
    pub(crate) fn drain_dialog_answer(&mut self) {
        let Some(DialogAnswer { request, paths }) = self.system.take_dialog_answer() else { return };
        let Some(path) = paths.first().cloned() else { return };
        match request {
            DialogRequest::OpenImages => self.on_open_picked(paths),
            DialogRequest::Background { item_id } => self.on_background_picked(item_id, path),
            DialogRequest::SaveImage { item_id, preferred, .. } => self.on_save_image_picked(item_id, preferred, path),
            DialogRequest::SaveLayers { targets } => self.on_save_layers_picked(&targets, path),
            DialogRequest::SaveSelected { targets } => self.on_save_selected_picked(&targets, path),
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
    /// the toolbar button and the Cmd+R shortcut — call this.
    pub fn handle_process_intent(&mut self) {
        if !self.can_process_intent() {
            return;
        }
        let model = self.settings.model;
        if model.is_upscale() || model.is_colorize() {
            self.process_kept_run_items();
            return;
        }
        if !model.is_inpaint() {
            self.handle_remove_bg();
            return;
        }
        let (with_region, without) = self.erase_targets();
        if without > 0 {
            self.toasts.info(format!("{} had no painted region", super::views::fmt::count(without, "image", "images")));
        }
        // One image at a time through the run queue: a fill's working set
        // can be most of the machine's RAM.
        self.processor.queue_runs(with_region.iter().map(|&idx| self.batch.items[idx].id).collect());
        self.dispatch_next_queued_run();
    }

    /// Toolbar mirror for `handle_process_intent`. `add_enabled(...)` reads
    /// this so the button's disabled state and the dispatch's gating can't
    /// drift independently.
    pub fn can_process_intent(&self) -> bool {
        // A saved model that is gone from disk: the picker says Get.
        if !self.hardware_install_cache.has_model(self.settings.model) {
            return false;
        }
        if self.settings.model.is_upscale() {
            return can_start_run(self.processor.is_upscale_in_flight(), self.batch.any_target_can(|_| true));
        }
        if self.settings.model.is_colorize() {
            return can_start_run(self.processor.is_colorize_in_flight(), self.batch.any_target_can(|_| true));
        }
        if self.settings.model.is_inpaint() {
            can_start_run(self.processor.any_inpaint_in_flight(), self.batch.any_target_can(BatchItem::has_painted_region))
        } else {
            // A batch still admitting images into its RAM budget takes the
            // next Process once every one of them is on its way; the worker
            // queues batches, the budget cannot feed two.
            !self.processor.admission_still_feeding()
                && self.batch.any_target_can(|it| !matches!(it.status, BatchStatus::Processing))
        }
    }

    pub(crate) fn close_settings(&mut self) {
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
            (self.system.dialog_open(), "file_dialog"),
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
            Some(Engine::Colorize) => self.processor.cancel_colorize(),
            Some(Engine::Segment) | None => self.processor.cancels.request_item_cancel(item_id),
        }
    }

    /// Compare shows the original in place of a result, so it needs one.
    /// The toolbar button and the B key both read this.
    pub(crate) fn can_compare_intent(&self) -> bool {
        self.batch.selected_item().is_some_and(|i| i.has_result())
    }

    /// "Try another" is possible: the SD eraser is chosen and the shown
    /// step is a generated fill, with nothing running on the image.
    pub fn can_reroll_intent(&self) -> bool {
        self.settings.model.generates()
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
            None => item.decoded_source(),
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

    /// Make `bindings` the live ones and persist what differs from the
    /// shipped table.
    pub(crate) fn apply_hotkeys(&mut self, ctx: &egui::Context, bindings: super::views::shortcuts::Bindings) {
        self.settings.hotkeys = bindings.to_overrides();
        self.settings.save();
        self.bindings = Arc::new(bindings);
        super::views::shortcuts::install(ctx, Arc::clone(&self.bindings));
    }

    pub(crate) fn maybe_evaluate_runtime_prompt(&mut self) {
        if self.runtime_prompt_evaluated { return; }
        self.runtime_prompt_evaluated = true;
        let Some(rt) = crate::hardware::profile().recommended_runtime() else { return };
        if rt.is_installed() { return; }
        if self.settings.is_runtime_prompt_snoozed(rt) { return; }
        self.runtime_prompt = Some(rt);
    }

    /// Install `runtime` in the background; Settings › Hardware and the
    /// first-launch prompt both start it here. One install at a time.
    pub(crate) fn start_runtime_install(&mut self, runtime: crate::runtime_install::RuntimeId) {
        if self.runtime_install.is_some() { return; }
        let h = crate::runtime_install::start_install(runtime);
        self.runtime_install = Some(RuntimeInstallProgress {
            runtime,
            rx: h.events,
            cancel: h.cancel,
            last_event: crate::runtime_install::InstallEvent::Preparing,
        });
    }

    pub fn handle_cancel(&mut self) {
        self.processor.cancels.request_global_cancel();
        // Esc cancels both the batch path and any in-flight eraser stroke
        // (SD on CPU takes minutes — needs the same escape valve).
        self.processor.cancel_all_inpaints();
        // Always cancel an in-flight upscale or colorize; the call is
        // idempotent and gating on the model would drop the signal if the
        // user switched model mid-flight.
        self.processor.cancel_runs();
        self.processor.cancel_faces();
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
        let targets = self.batch.ticked_ids_with_status(BatchStatus::Processing);
        if targets.is_empty() {
            return;
        }
        for id in &targets {
            self.processor.cancels.request_item_cancel(*id);
        }
        // The image shows its state again at once; a late answer from the
        // cancelled run finds it no longer processing, and a newer batch
        // running it again takes it from the cancelled one's.
        for id in &targets {
            if let Some(item) = self.batch.find_by_id_mut(*id) {
                item.abandon_job();
            }
        }
        self.toasts.info(format!("Cancelling {} image(s)", targets.len()));
    }
}

impl PrunrApp {
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

    /// Escape: the top-most modal first, then a run in flight, then
    /// the selection. One press does one thing.
    fn apply_cancel_shortcut(&mut self, ctx: &egui::Context) {
        if self.close_top_modal() {
            return;
        }
        let finding_faces = self.processor.dispatch_progress().is_some_and(|p| p.kind == super::dispatch_progress::ProgressKind::Faces);
        if self.processor.is_run_in_flight() {
            self.processor.cancel_runs();
        } else if finding_faces {
            self.processor.cancel_faces();
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
    fn close_top_modal(&mut self) -> bool {
        if self.save_preset.is_some() {
            self.save_preset = None;
        } else if self.pending_license_request.is_some() {
            self.pending_license_request = None;
        } else if self.model_store.is_some() {
            self.model_store = None;
        } else if let Some(rt) = self.runtime_prompt.take() {
            self.settings.snooze_runtime_prompt(rt, RUNTIME_PROMPT_SNOOZE_DAYS);
        } else if self.show_settings {
            self.close_settings();
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
    pub(crate) fn toggle_hotkeys_tab(&mut self) {
        use super::views::settings::SettingsTab;
        if self.show_settings && self.settings_tab == SettingsTab::Hotkeys {
            self.close_settings();
        } else {
            self.settings_tab = SettingsTab::Hotkeys;
            self.open_settings();
        }
    }

    fn toggle_settings_panel(&mut self) {
        if self.show_settings {
            self.close_settings();
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

    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        super::views::shortcuts::install(ctx, Arc::clone(&self.bindings));
        let motion_changed = super::theme::apply_motion(ctx, self.settings.animations);
        if motion_changed || self.processor.batch_in_flight() {
            self.update_on_screen();
        }
        // Held while the pointer stays down, so the press that dismissed a
        // popover is spent in full.
        let popup_open = super::theme::dismissable_popup_open(ctx);
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        self.popup_open_at_frame_start = popup_open || (self.popup_open_at_frame_start && pointer_down);
        self.poll_worker_results(ctx);
        self.handle_drag_and_drop(ctx);
        if !self.handle_hotkey_capture(ctx) {
            self.handle_keyboard_shortcuts(ctx);
        }
        super::automation::pump(self, ctx);
        self.system.show_dialog(ctx, frame);
        self.drain_dialog_answer();
        if self.system.dialog_open() {
            // An idle Wayland window can drop the dialog thread's
            // wake-up; polling lands the answer regardless.
            ctx.request_repaint_after(DIALOG_POLL);
        }
        self.flush_settings_save(ctx);
        drain_screenshot_replies(ctx);
        // Anything landing restarts the quiet clock: a long import keeps
        // landing images, so it never reads as stuck.
        if self.drain_background_channels(ctx) {
            self.busy_since = None;
        }
        if self.prewrite_drag_files() {
            ctx.request_repaint_after(ASYNC_POLL);
        }
        self.update_window_title(ctx);
        self.status.tick();
        self.reconcile_selected(ctx);
        let now = std::time::Instant::now();
        if let Some(wait) = self.processor.pump_seg_warm(now) {
            ctx.request_repaint_after(wait);
        }
        self.note_quiet_work(now);
        // A frame at the one-second mark, so the pill appears without
        // waiting for input; once it shows, the poll below and the pill's
        // own repaint are enough.
        let until_pill = self.busy_since.map(|since| BUSY_PILL_AFTER.saturating_sub(now.duration_since(since)));
        if let Some(wait) = until_pill.filter(|wait| !wait.is_zero()) {
            ctx.request_repaint_after(wait);
        }
        // Keep the event loop awake while any async texture / decode work is
        // pending. `ctx.request_repaint()` from the tex_prep / decode threads
        // is supposed to wake egui, but some compositors (notably Wayland)
        // drop thread-initiated wake-ups when the window is idle, leaving the
        // canvas stuck on the old image until a mouse event fires. Polling
        // from the UI thread is reliable.
        if self.batch.has_pending_work() {
            ctx.request_repaint_after(ASYNC_POLL);
        }
    }


    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Land finished work, then the live previews it may have queued,
        // before rendering, so this frame shows the newest results.
        self.pump_results(ui.ctx());
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
        // A dialog asked for while rendering shows from the next `logic()`.
        if self.system.dialog_unshown() {
            ui.ctx().request_repaint();
        }
    }
}

impl PrunrApp {
    /// What a model switch does whatever images are loaded: fit lines and
    /// brushes to the model, drop runs and cached sessions of the old one.
    fn on_model_changed(&mut self) {
        self.batch.fit_line_modes_to(self.settings.model);
        self.settings.on_model_change_resolve_brush();
        self.fit_brush_models_to_model();
        // Cancel any in-flight upscale or colorize when the user
        // changes model mid-run — the result would land on the
        // wrong selection otherwise.
        self.processor.cancel_runs();
        // Drop every item's kept upscale and colorize run — the buffers
        // were produced by the previous model. A subsequent Tier-2
        // preview tick would otherwise display the old model's output
        // under the new model's knobs.
        for item in &mut self.batch.items {
            item.invalidate_upscale_cache();
            item.invalidate_colorize_cache();
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
        let installed = self.hardware_install_cache.has_model(self.settings.model);
        if let Some(spec) = Self::initial_engines(&self.settings).filter(|_| installed) {
            self.processor.schedule_seg_warm(spec, std::time::Instant::now());
        }
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
        self.processor.prewarm_eraser(id, self.settings.cpu_only());
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
                    RuntimePromptAction::Install => self.start_runtime_install(rt),
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

/// Where `Action::Screenshot` writes: `PRUNR_SCREENSHOT_DIR` or the temp dir.
pub(crate) fn screenshot_dir() -> PathBuf {
    std::env::var_os("PRUNR_SCREENSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("prunr-screenshots"))
}

/// Drain any `Event::Screenshot` replies and persist them as PNGs into
/// `screenshot_dir()`. Filename is the unix-millis timestamp so a
/// scenario that fires Shift+F12 multiple times never collides.
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

#[derive(Copy, Clone)]
enum NavDir {
    Prev,
    Next,
}

/// A run can start when some image qualifies and no run of its kind is
/// under way.
pub(crate) fn can_start_run(is_in_flight: bool, has_target: bool) -> bool {
    has_target && !is_in_flight
}

#[cfg(test)]
mod can_start_run_tests {
    use super::can_start_run;

    #[test]
    fn can_start_run_requires_idle_and_a_target() {
        assert!(can_start_run(false, true),    "happy path: idle + a target");
        assert!(!can_start_run(true, true),    "already in flight");
        assert!(!can_start_run(false, false),  "no target");
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
