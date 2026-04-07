use egui::{Align2, RichText, Stroke};

use crate::gui::theme;

pub fn render(ctx: &egui::Context) {
    // Draw semi-transparent backdrop to dim the app
    let screen_rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("shortcuts_backdrop"),
    ));
    painter.rect_filled(
        screen_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
    );

    // Platform-aware modifier key
    let modifier = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };

    egui::Window::new("Keyboard Shortcuts")
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SHORTCUT_OVERLAY_WIDTH, theme::SHORTCUT_OVERLAY_HEIGHT])
        .frame(egui::Frame {
            fill: theme::OVERLAY_BG,
            stroke: Stroke::new(1.0, theme::OVERLAY_BORDER),
            corner_radius: egui::CornerRadius::same(8),
            inner_margin: egui::Margin::same(theme::SPACE_MD as i8),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    RichText::new("Keyboard Shortcuts")
                        .size(theme::FONT_SIZE_HEADING)
                        .strong()
                        .color(theme::TEXT_PRIMARY),
                );
                ui.add_space(theme::SPACE_MD);

                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([theme::SPACE_LG, theme::SPACE_SM])
                    .show(ui, |ui| {
                        shortcut_row(ui, &format!("{modifier}+O"), "Open file");
                        shortcut_row(ui, &format!("{modifier}+R"), "Remove background");
                        shortcut_row(ui, &format!("{modifier}+S"), "Save result");
                        shortcut_row(ui, &format!("{modifier}+C"), "Copy to clipboard");
                        shortcut_row(ui, "Escape", "Cancel / Close");
                        shortcut_row(ui, "?", "Show this help");
                    });
            });
        });
}

fn shortcut_row(ui: &mut egui::Ui, key: &str, action: &str) {
    ui.label(
        RichText::new(key)
            .monospace()
            .size(theme::FONT_SIZE_MONO)
            .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        RichText::new(action)
            .size(theme::FONT_SIZE_BODY)
            .color(theme::TEXT_SECONDARY),
    );
    ui.end_row();
}
