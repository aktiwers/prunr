use egui::{Color32, Pos2, Rect, Stroke, Vec2};

use crate::gui::app::BgPrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;

pub fn render(ui: &mut egui::Ui, app: &BgPrunrApp) {
    // Set background
    let avail_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(avail_rect, 0.0, theme::BG_PRIMARY);

    match app.state {
        AppState::Empty => render_empty(ui, app),
        AppState::Loaded => render_loaded(ui, app),
        AppState::Processing => render_processing(ui, app),
        AppState::Animating => render_done(ui, app), // placeholder until animation.rs is wired
        AppState::Done => render_done(ui, app),
    }
}

fn render_empty(ui: &mut egui::Ui, _app: &BgPrunrApp) {
    let avail = ui.available_size();
    let is_hovered = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

    // Centered drop zone
    let zone_w = (avail.x * 0.5).min(400.0).max(200.0);
    let zone_h = 200.0_f32;
    let center = ui.available_rect_before_wrap().center();
    let zone_rect = Rect::from_center_size(center, Vec2::new(zone_w, zone_h));

    // Draw drop zone border
    let border_color = if is_hovered {
        theme::DROP_HOVER_BORDER
    } else {
        theme::DROP_BORDER
    };
    ui.painter().rect_stroke(
        zone_rect,
        theme::DROP_ZONE_ROUNDING,
        Stroke::new(theme::DROP_ZONE_BORDER_WIDTH, border_color),
        egui::StrokeKind::Outside,
    );

    let painter = ui.painter();

    let heading = "Drop an image here";
    painter.text(
        center - Vec2::new(0.0, 20.0),
        egui::Align2::CENTER_CENTER,
        heading,
        egui::FontId::proportional(theme::FONT_SIZE_HEADING),
        theme::TEXT_PRIMARY,
    );

    let hint = if cfg!(target_os = "macos") {
        "or press Cmd+O to open a file"
    } else {
        "or press Ctrl+O to open a file"
    };
    painter.text(
        center + Vec2::new(0.0, 20.0),
        egui::Align2::CENTER_CENTER,
        hint,
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_SECONDARY,
    );

    // Wayland DnD caveat
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        painter.text(
            center + Vec2::new(0.0, 55.0),
            egui::Align2::CENTER_CENTER,
            "(Drag and drop not supported in Wayland yet)",
            egui::FontId::proportional(theme::FONT_SIZE_BODY * 0.85),
            theme::TEXT_SECONDARY,
        );
    }
}

fn render_loaded(ui: &mut egui::Ui, app: &BgPrunrApp) {
    if let Some(ref texture) = app.source_texture {
        render_image_centered(ui, texture, 1.0);
    }
}

fn render_processing(ui: &mut egui::Ui, app: &BgPrunrApp) {
    if let Some(ref texture) = app.source_texture {
        render_image_centered(ui, texture, 0.5);
    }

    // Center overlay text
    let center = ui.available_rect_before_wrap().center();
    ui.painter().text(
        center + egui::Vec2::new(0.0, 30.0),
        egui::Align2::CENTER_CENTER,
        "Press Escape to cancel",
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_PRIMARY,
    );
}

fn render_done(ui: &mut egui::Ui, app: &BgPrunrApp) {
    if let Some(ref texture) = app.result_texture {
        let avail = ui.available_size();
        let tex_size = texture.size_vec2();
        let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).min(1.0);
        let img_size = tex_size * scale;
        let center = ui.available_rect_before_wrap().center();
        let img_rect = Rect::from_center_size(center, img_size);

        // Draw checkerboard pattern within image bounds
        draw_checkerboard(ui, img_rect);

        // Draw result image on top
        ui.painter().image(
            texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
}

fn render_image_centered(ui: &mut egui::Ui, texture: &egui::TextureHandle, opacity: f32) {
    let avail = ui.available_size();
    let tex_size = texture.size_vec2();
    let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).min(1.0);
    let img_size = tex_size * scale;
    let center = ui.available_rect_before_wrap().center();
    let img_rect = Rect::from_center_size(center, img_size);

    let alpha = (opacity * 255.0) as u8;
    ui.painter().image(
        texture.id(),
        img_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
    );
}

fn draw_checkerboard(ui: &egui::Ui, bounds: Rect) {
    let checker_size = theme::CHECKER_SIZE;
    let painter = ui.painter();

    let start_x = bounds.min.x;
    let start_y = bounds.min.y;
    let end_x = bounds.max.x;
    let end_y = bounds.max.y;

    let mut y = start_y;
    let mut row = 0usize;
    while y < end_y {
        let mut x = start_x;
        let mut col = 0usize;
        while x < end_x {
            let color = if (row + col) % 2 == 0 {
                theme::CHECKER_LIGHT
            } else {
                theme::CHECKER_DARK
            };
            let rect = Rect::from_min_max(
                Pos2::new(x, y),
                Pos2::new((x + checker_size).min(end_x), (y + checker_size).min(end_y)),
            );
            painter.rect_filled(rect, 0.0, color);
            x += checker_size;
            col += 1;
        }
        y += checker_size;
        row += 1;
    }
}
