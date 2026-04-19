use egui::{Align2, RichText};

use crate::gui::theme;

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context) -> bool {
    theme::draw_modal_backdrop(ctx, "shortcuts_backdrop");

    let modifier = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };

    let mut open = true;
    let window_response = egui::Window::new("Keyboard Shortcuts")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SHORTCUT_OVERLAY_WIDTH, theme::SHORTCUT_OVERLAY_HEIGHT])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);

                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([theme::SPACE_LG, theme::SPACE_SM])
                    .show(ui, |ui| {
                        shortcut_row(ui, &format!("{modifier}+O"), "Open file(s)");
                        shortcut_row(ui, &format!("{modifier}+R"), "Remove background");
                        shortcut_row(ui, &format!("{modifier}+S"), "Save result");
                        shortcut_row(ui, &format!("{modifier}+C"), "Copy result");
                        shortcut_row(ui, &format!("{modifier}+Z"), "Undo removal");
                        shortcut_row(ui, &format!("{modifier}+Y"), "Redo removal");
                        shortcut_row(ui, "Escape", "Cancel / Close");
                        shortcut_row(ui, "F1", "Keyboard shortcuts");
                        shortcut_row(ui, "F2", "CLI reference");
                        shortcut_row(ui, "F3", "Mask pipeline diagram");
                        shortcut_row(ui, "B", "Toggle before/after");
                        shortcut_row(ui, "← / → or A / D", "Previous / Next image");
                        shortcut_row(ui, &format!("{modifier}+0"), "Fit to window");
                        shortcut_row(ui, &format!("{modifier}+1"), "Actual size");
                        shortcut_row(ui, "Tab", "Show/hide queue");
                        shortcut_row(ui, &format!("{modifier}+Space"), "Settings");
                        shortcut_row(ui, "Drag", "Pan image");
                        shortcut_row(ui, "Scroll", "Zoom in/out");
                    });
            });
        });

    !open || theme::backdrop_clicked(ctx, &window_response)
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
            .color(theme::TEXT_PRIMARY),
    );
    ui.end_row();
}
