use egui::{Align2, RichText};

use crate::gui::animation_sweep::SweepKnob;
use crate::gui::app::PrunrApp;
use crate::gui::theme;

use super::{hint, section_heading};

pub fn render(ctx: &egui::Context, app: &mut PrunrApp) {
    theme::draw_modal_backdrop(ctx, "animation_sweep_backdrop");

    let mut open = true;
    let mut do_start = false;
    let window_response = egui::Window::new("Export animation")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([theme::SETTINGS_DIALOG_WIDTH, 420.0])
        .frame(theme::overlay_frame())
        .show(ctx, |ui| {
            section_heading(ui, "Sweep");
            ui.horizontal(|ui| {
                ui.label(RichText::new("Knob").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
                let current = app.animation_sweep_ui.knob;
                egui::ComboBox::from_id_salt("sweep_knob")
                    .selected_text(RichText::new(current.label()).color(theme::TEXT_PRIMARY))
                    .show_ui(ui, |ui| {
                        for k in SweepKnob::ALL {
                            if ui.selectable_label(k == current, k.label()).clicked() {
                                app.animation_sweep_ui.knob = k;
                            }
                        }
                    });
            });
            hint(ui, "Each frame applies one sweep step; others keep the item's current settings.");
            ui.add_space(theme::SPACE_MD);

            section_heading(ui, "Frames");
            match app.animation_sweep_ui.knob.cycle_len() {
                Some(n) => {
                    hint(ui, &format!("Cycle sweep — {n} frames (one per variant)."));
                }
                None => {
                    let mut frames = app.animation_sweep_ui.frames as f32;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Count").color(theme::TEXT_PRIMARY).size(theme::FONT_SIZE_BODY));
                        let avail = ui.available_width() - 52.0;
                        let slider = egui::Slider::new(&mut frames, 5.0..=60.0)
                            .step_by(1.0)
                            .show_value(false);
                        ui.add_sized([avail.max(100.0), 18.0], slider);
                        ui.label(
                            RichText::new(format!("{}", frames as usize))
                                .monospace()
                                .size(theme::FONT_SIZE_MONO)
                                .color(theme::TEXT_PRIMARY),
                        );
                    });
                    app.animation_sweep_ui.frames = (frames as usize).clamp(5, 60);
                    hint(ui, "Frames lerp between the knob's start and end values.");
                }
            }
            ui.add_space(theme::SPACE_MD);

            section_heading(ui, "Output folder");
            ui.horizontal(|ui| {
                let label = app.animation_sweep_ui.out_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "— not selected —".to_string());
                ui.label(
                    RichText::new(label)
                        .monospace()
                        .size(theme::FONT_SIZE_MONO)
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(RichText::new("Browse…").color(theme::TEXT_PRIMARY)).clicked() {
                        if let Some(dir) = app.save_dialog()
                            .set_title("Export animation — Choose Folder")
                            .pick_folder()
                        {
                            app.animation_sweep_ui.out_dir = Some(dir);
                        }
                    }
                });
            });
            hint(ui, "Frames land as 0000.png, 0001.png, …");
            ui.add_space(theme::SPACE_LG);

            ui.horizontal(|ui| {
                let ready = app.animation_sweep_ui.is_ready() && app.sweep_progress.is_none();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let btn = egui::Button::new(
                        RichText::new("Start export").color(theme::TEXT_PRIMARY),
                    )
                    .fill(theme::ACCENT)
                    .corner_radius(theme::BUTTON_ROUNDING);
                    if ui.add_enabled(ready, btn).clicked() {
                        do_start = true;
                    }
                });
            });

            if let Some((done, total)) = app.sweep_progress {
                ui.add_space(theme::SPACE_MD);
                ui.label(
                    RichText::new(format!("Rendering {done}/{total}…"))
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_BODY),
                );
            }
        });

    if do_start {
        app.start_animation_sweep();
    }

    let close_via_backdrop = theme::backdrop_clicked(ctx, &window_response);
    if !open || close_via_backdrop {
        app.show_animation_sweep = false;
    }
}
