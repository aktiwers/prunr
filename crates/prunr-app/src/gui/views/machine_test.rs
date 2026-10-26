//! The modal a test of this computer shows while it runs.

use egui::RichText;

use super::chip::{button, ButtonKind};
use crate::gui::machine_test::Step;
use crate::gui::theme;

const BAR_HEIGHT: f32 = 6.0;
const SIZE: [f32; 2] = [theme::STANDARD_MODAL_WIDTH, 196.0];

/// Its progress, and Cancel; `true` when Cancel was clicked. No close
/// button, and a click beside it does nothing: a test takes minutes to
/// start over.
pub(crate) fn render(ctx: &egui::Context, step: Option<Step>, accelerator: &str) -> bool {
    let mut cancel = false;
    let _ = theme::bare_modal_window(ctx, "machine_test", "Testing this computer", SIZE, |ui| {
        egui::Frame::NONE.inner_margin(theme::MODAL_PAGE_MARGIN).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Testing this computer").size(theme::FONT_SIZE_MODAL_TITLE).color(theme::TEXT_PRIMARY));
            ui.add_space(theme::SPACE_MD);
            let (line, done) = match step {
                None => ("Starting\u{2026}".to_string(), 0.0),
                Some(step) => (step_line(step, accelerator), fraction(step)),
            };
            ui.label(RichText::new(line).size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY));
            ui.add_space(theme::SPACE_SM);
            let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), BAR_HEIGHT), egui::Sense::hover());
            ui.painter().rect_filled(bar, 2.0, theme::RAIL);
            ui.painter().rect_filled(egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * done, BAR_HEIGHT)), 2.0, theme::ACCENT_BRIGHT);
            ui.add_space(theme::SPACE_SM);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Prunr is busy until it ends. Cancel keeps what it measured so far.")
                        .size(theme::FONT_SIZE_SECONDARY)
                        .color(theme::TEXT_SECONDARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        });
        false
    });
    cancel
}

/// "3 of 12: BiRefNet, with CUDA".
fn step_line(step: Step, accelerator: &str) -> String {
    let side = if step.on_gpu { "with" } else { "without" };
    format!("{} of {}: {}, {side} {accelerator}", step.index, step.total, prunr_models::display_name(step.model))
}

/// How far along: each model's run without the accelerator is its first
/// half.
fn fraction(step: Step) -> f32 {
    let half = if step.on_gpu { 0.5 } else { 0.0 };
    ((step.index.saturating_sub(1) as f32 + half) / step.total.max(1) as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::{Column, ModelId};

    fn step(index: usize, on_gpu: bool) -> Step {
        Step { index, total: 4, model: ModelId::U2net, on_gpu, column: Column::Cuda }
    }

    #[test]
    fn the_bar_moves_half_a_model_per_side() {
        assert_eq!(fraction(step(1, false)), 0.0);
        assert_eq!(fraction(step(1, true)), 0.125);
        assert_eq!(fraction(step(4, true)), 0.875);
    }

    #[test]
    fn the_line_names_the_model_its_place_and_the_side() {
        assert_eq!(step_line(step(3, true), "CUDA"), "3 of 4: U2Net, with CUDA");
        assert_eq!(step_line(step(3, false), "CUDA"), "3 of 4: U2Net, without CUDA");
    }
}
