use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use egui::{Key, ViewportCommand};
use image::GenericImageView;

use bgprunr_core::{ModelKind, ProgressStage};

use super::state::AppState;
use super::theme;
use super::worker::{WorkerMessage, WorkerResult, spawn_worker};
use super::views::{canvas, shortcuts, statusbar, toolbar};

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
            selected_model: ModelKind::Silueta,
            active_backend: "CPU".to_string(),
        }
    }

    /// Test constructor that skips eframe setup (for unit tests)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        let (worker_tx, _worker_msg_rx) = mpsc::channel::<WorkerMessage>();
        let (_result_tx, worker_rx) = mpsc::channel::<WorkerResult>();
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
            selected_model: ModelKind::Silueta,
            active_backend: "CPU".to_string(),
        }
    }

    fn set_temporary_status(&mut self, text: impl Into<String>) {
        self.status_text = text.into();
        self.status_is_temporary = true;
        self.status_set_at = Some(std::time::Instant::now());
    }

    pub fn handle_open_path(&mut self, path: PathBuf) {
        match std::fs::read(&path) {
            Ok(bytes) => {
                match image::load_from_memory(&bytes) {
                    Ok(img) => {
                        let (w, h) = img.dimensions();
                        self.image_dimensions = Some((w, h));
                        self.source_bytes = Some(bytes);
                        self.loaded_filename = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_string());
                        // Texture will be created lazily in canvas render or we create it here
                        // We store the raw rgba for texture creation; canvas will create textures
                        // Since we don't have egui context here, store dims; canvas creates texture on first paint
                        self.source_texture = None; // reset so canvas re-creates
                        self.result_texture = None;
                        self.result_rgba = None;
                        self.state = AppState::Loaded;
                        self.status_text = "Ready".to_string();
                    }
                    Err(e) => {
                        self.set_temporary_status(format!("Could not load image: {e}"));
                    }
                }
            }
            Err(e) => {
                self.set_temporary_status(format!("Could not read file: {e}"));
            }
        }
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
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG Image", &["png"])
                .set_file_name("result.png")
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

    pub fn handle_cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }
}

impl eframe::App for BgPrunrApp {
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
                        self.result_rgba = Some(rgba_img);
                    }
                    self.active_backend = result.active_provider;
                    self.state = AppState::Done;
                    self.status_text = "Done".to_string();
                }
                WorkerResult::Cancelled => {
                    self.state = AppState::Loaded;
                    self.status_text = "Cancelled".to_string();
                }
                WorkerResult::Error(msg) => {
                    self.state = AppState::Loaded;
                    self.status_text =
                        "Processing failed. Try a different image or restart the app.".to_string();
                    eprintln!("Worker error: {msg}");
                }
            }
        }

        // b. Handle drag-and-drop
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(path) = file.path {
                self.handle_open_path(path);
            }
        }

        // c. Keyboard shortcut detection via bool flags (CRITICAL: no blocking calls inside closure)
        let (mut open_requested, mut remove_requested, mut save_requested, mut copy_requested) =
            (false, false, false, false);
        let (mut cancel_requested, mut toggle_shortcuts) = (false, false);

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
            if i.modifiers.command && i.key_pressed(Key::C) {
                copy_requested = true;
            }
            if i.key_pressed(Key::Escape) {
                cancel_requested = true;
            }
            // '?' key
            if i.key_pressed(Key::Questionmark) {
                toggle_shortcuts = true;
            }
        });

        // Act on flags AFTER input closure has returned
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
        if cancel_requested {
            if self.state == AppState::Processing {
                self.handle_cancel();
            } else if self.show_shortcuts {
                self.show_shortcuts = false;
            }
        }
        if toggle_shortcuts {
            self.show_shortcuts = !self.show_shortcuts;
        }

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
            if let Some(ref bytes) = self.source_bytes.clone() {
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
    }
}
