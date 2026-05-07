use crate::gui::theme;

use super::kv_row;

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context) -> bool {
    theme::standard_modal_window(
        ctx, "shortcuts", "Keyboard Shortcuts",
        [theme::SHORTCUT_OVERLAY_WIDTH, theme::SHORTCUT_OVERLAY_HEIGHT],
        |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);
                render_shortcut_grid(ui);
            });
        },
    )
}

/// Two-column key/action grid. Used by the F1 overlay and the Settings
/// Hotkeys tab — single source of truth for the shipped shortcut list.
pub fn render_shortcut_grid(ui: &mut egui::Ui) {
    let modifier = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };
    egui::Grid::new("shortcuts_grid")
        .num_columns(2)
        .spacing([theme::SPACE_LG, theme::SPACE_SM])
        .show(ui, |ui| {
            let row = |ui: &mut egui::Ui, k: &str, a: &str|
                kv_row(ui, k, a, theme::TEXT_PRIMARY);
            row(ui, &format!("{modifier}+O"), "Open file(s)");
            row(ui, &format!("{modifier}+R"), "Remove background");
            row(ui, &format!("{modifier}+S"), "Save result");
            row(ui, &format!("{modifier}+C"), "Copy result");
            row(ui, &format!("{modifier}+Z"), "Undo last action (stroke, result, or preset)");
            row(ui, &format!("{modifier}+Shift+Z / {modifier}+Y"), "Redo last undone action");
            row(ui, "Escape", "Cancel / Close");
            row(ui, "F1", "Keyboard shortcuts");
            row(ui, "F2", "CLI reference");
            row(ui, "F3", "Mask pipeline diagram");
            row(ui, "Shift+F12", "Capture window screenshot");
            row(ui, "B", "Toggle before/after");
            row(ui, "← / → or A / D", "Previous / Next image");
            row(ui, &format!("{modifier}+0"), "Fit to window");
            row(ui, &format!("{modifier}+1"), "Actual size");
            row(ui, "Tab", "Show/hide queue");
            row(ui, &format!("{modifier}+Space"), "Settings");
            row(ui, "Drag", "Pan image");
            row(ui, "Scroll", "Zoom in/out");
        });
}
