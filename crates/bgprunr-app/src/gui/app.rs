use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use egui::{Key, ViewportCommand};
use image::GenericImageView;

use bgprunr_core::{ModelKind, ProgressStage};

use super::settings::Settings;
use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, settings, shortcuts, statusbar, toolbar};

pub(crate) struct BatchItem {
    pub id: u64,
    pub filename: String,
    pub source_bytes: Vec<u8>,
    pub source_texture: Option<egui::TextureHandle>,
    pub thumb_texture: Option<egui::TextureHandle>,
    pub result_rgba: Option<image::RgbaImage>,
    pub result_texture: Option<egui::TextureHandle>,
    pub status: BatchStatus,
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
    pub(crate) selected_model: ModelKind,
    pub(crate) active_backend: String,

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
}

impl BgPrunrApp {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let (worker_tx, worker_rx) = spawn_worker(cc.egui_ctx.clone());

        // Set dark visuals
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        // Customize visuals
        let mut visuals = cc.egui_ctx.global_style().visuals.clone();
        visuals.window_fill = theme::BG_PRIMARY;
        visuals.panel_fill = theme::BG_SECONDARY;
        cc.egui_ctx.set_visuals(visuals);

        // Override font sizes
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
        cc.egui_ctx.set_global_style(style);

        let clipboard = arboard::Clipboard::new().ok();

        let settings = Settings::default();
        let selected_model = settings.model.into();
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
            selected_model,
            active_backend: "CPU".to_string(),
            pending_copy: false,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            previous_zoom: 1.0,
            is_panning: false,
            show_original: false,
            anim_progress: 0.0,
            anim_mask: None,
            batch_items: Vec::new(),
            selected_batch_index: 0,
            show_sidebar: false,
            next_batch_id: 0,
            show_settings: false,
            settings,
            pending_fit_zoom: false,
            pending_actual_size: false,
        }
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
        let settings = Settings::default();
        let selected_model = settings.model.into();
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
            selected_model,
            active_backend: "CPU".to_string(),
            pending_copy: false,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            previous_zoom: 1.0,
            is_panning: false,
            show_original: false,
            anim_progress: 0.0,
            anim_mask: None,
            batch_items: Vec::new(),
            selected_batch_index: 0,
            show_sidebar: false,
            next_batch_id: 0,
            show_settings: false,
            settings,
            pending_fit_zoom: false,
            pending_actual_size: false,
        }
    }

    fn set_temporary_status(&mut self, text: impl Into<String>) {
        self.status_text = text.into();
        self.status_is_temporary = true;
        self.status_set_at = Some(std::time::Instant::now());
    }

    fn load_image(&mut self, bytes: Vec<u8>, filename: Option<String>) {
        match image::load_from_memory(&bytes) {
            Ok(img) => {
                self.image_dimensions = Some(img.dimensions());
                self.source_bytes = Some(bytes);
                self.loaded_filename = filename;
                self.source_texture = None;
                self.result_texture = None;
                self.result_rgba = None;
                self.state = AppState::Loaded;
                self.status_text = "Ready".to_string();
                // Reset zoom/pan for new image
                self.zoom = 1.0;
                self.pan_offset = egui::Vec2::ZERO;
                self.previous_zoom = 1.0;
                self.show_original = false;
            }
            Err(e) => {
                self.set_temporary_status(format!("Could not load image: {e}"));
            }
        }
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
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .set_title("Open Image")
            .pick_file()
        {
            self.handle_open_path(path);
        }
    }

    pub fn handle_remove_bg(&mut self) {
        if let Some(ref bytes) = self.source_bytes {
            self.cancel_flag.store(false, Ordering::Relaxed);
            self.state = AppState::Processing;
            self.progress_pct = 0.0;
            self.progress_stage = "Starting".to_string();
            let _ = self.worker_tx.send(WorkerMessage::ProcessImage {
                img_bytes: bytes.clone(),
                model: self.selected_model,
                cancel: self.cancel_flag.clone(),
            });
        }
    }

    pub fn handle_save(&mut self) {
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
                        Err(_) => self.set_temporary_status(
                            "Could not save file. Check disk space and permissions.",
                        ),
                    },
                    Err(_) => self.set_temporary_status(
                        "Could not save file. Check disk space and permissions.",
                    ),
                }
            }
        }
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
                        let (w, h) = (rgba_img.width(), rgba_img.height());
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(
                            [w as usize, h as usize],
                            rgba_img.as_flat_samples().as_slice(),
                        );
                        self.result_texture = Some(ctx.load_texture(
                            "result",
                            color_image,
                            egui::TextureOptions::default(),
                        ));
                        // Extract alpha mask for animation
                        let alpha_mask: Vec<u8> = rgba_img.pixels().map(|p| p[3]).collect();
                        self.anim_mask = Some(alpha_mask);
                        self.result_rgba = Some(rgba_img);
                    }
                    self.active_backend = result.active_provider.clone();
                    self.settings.active_backend = result.active_provider;

                    // Transition to Animating if enabled, else straight to Done
                    if self.settings.reveal_animation_enabled {
                        self.state = AppState::Animating;
                        self.anim_progress = 0.0;
                        self.show_original = false;
                    } else {
                        self.state = AppState::Done;
                        self.status_text = "Done".to_string();
                    }
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
            }
        }

        // b. Handle drag-and-drop (works on X11; no-op on native Wayland — winit#1881)
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(path) = file.path {
                self.handle_open_path(path);
            } else if let Some(bytes) = file.bytes {
                self.handle_open_bytes(
                    bytes.to_vec(),
                    file.name.clone(),
                );
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
        });

        let copy_requested = std::mem::take(&mut self.pending_copy);

        if open_requested {
            self.handle_open_dialog();
        }
        if remove_requested && matches!(self.state, AppState::Loaded | AppState::Done) {
            self.handle_remove_bg();
        }
        if save_requested && self.state == AppState::Done {
            self.handle_save();
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
        if cancel_requested && self.state == AppState::Animating {
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

        // Sync settings model with selected_model
        self.selected_model = self.settings.model.into();

        // d. Update window title
        let title = match &self.loaded_filename {
            Some(name) => format!("BgPrunR \u{2014} {name}"),
            None => "BgPrunR".to_string(),
        };
        ctx.send_viewport_cmd(ViewportCommand::Title(title));

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
                    let (w, h) = (rgba.width(), rgba.height());
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        rgba.as_flat_samples().as_slice(),
                    );
                    self.source_texture = Some(ctx.load_texture(
                        "source",
                        color_image,
                        egui::TextureOptions::default(),
                    ));
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("toolbar")
            .exact_size(theme::TOOLBAR_HEIGHT)
            .show_inside(ui, |ui| toolbar::render(ui, self));

        egui::Panel::bottom("statusbar")
            .exact_size(theme::STATUS_BAR_HEIGHT)
            .show_inside(ui, |ui| statusbar::render(ui, self));

        egui::CentralPanel::default().show_inside(ui, |ui| canvas::render(ui, self));

        if self.show_shortcuts {
            shortcuts::render(ui.ctx());
        }
        if self.show_settings {
            settings::render(ui.ctx(), self);
        }
    }
}
