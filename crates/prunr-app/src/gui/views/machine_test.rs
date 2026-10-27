//! The modal a test of this computer shows while it runs.

use egui::RichText;

use super::chip::{button, ButtonKind};
use crate::bench::here::Step;
use crate::gui::hardware_cache::Engine;
use crate::gui::machine_test::{MachineTest, Measuring};
use crate::gui::theme;

const BAR_HEIGHT: f32 = 10.0;
const SIZE: [f32; 2] = [760.0, 340.0];
/// The finished models it lists, the latest last.
const SHOWN: usize = 3;
const ROW_HEIGHT: f32 = 24.0;

/// Its progress, the models it finished, and Cancel; `true` when Cancel
/// was clicked. It holds the window: nothing under it takes a click, and
/// no click beside it closes it, as a test takes minutes to start over.
/// `engines`: this computer's, which it tries each model on beside the
/// processor.
pub(crate) fn render(ctx: &egui::Context, test: &MachineTest, engines: &[Engine]) -> bool {
    let mut cancel = false;
    theme::holding_modal_window(ctx, "machine_test", "Testing this computer", SIZE, |ui| {
        super::chip::modal_frame(ui, |ui| {
            ui.label(RichText::new("Testing this computer").size(theme::FONT_SIZE_MODAL_TITLE).color(theme::TEXT_PRIMARY));
            ui.add_space(theme::SPACE_MD);
            let step = test.step();
            let line = step.map_or_else(|| "Starting\u{2026}".to_string(), step_line);
            ui.label(RichText::new(line).size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY));
            if let Some(step) = step {
                ui.label(RichText::new(where_line(step, engines.iter().filter(|e| e.installed).map(|e| e.name))).size(theme::FONT_SIZE_SECONDARY).color(theme::TEXT_SECONDARY));
            }
            ui.add_space(theme::SPACE_SM);
            super::chip::progress_bar(ui, step.map_or(0.0, |s| s.fraction()), BAR_HEIGHT);
            ui.add_space(theme::SPACE_SM);
            let latest = test.done().skip(test.done().count().saturating_sub(SHOWN));
            let (rows, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), SHOWN as f32 * ROW_HEIGHT), egui::Sense::hover());
            let width = rows.width() / 4.0;
            for (i, m) in latest.enumerate() {
                let y = rows.top() + (i as f32 + 0.5) * ROW_HEIGHT;
                let [name, cpu, fastest, engine] = finished_cells(m);
                for (col, (text, color)) in [(name, theme::TEXT_SECONDARY), (cpu, theme::TEXT_SECONDARY), (fastest, theme::TEXT_SECONDARY), (engine, theme::TEXT_PRIMARY)].into_iter().enumerate() {
                    let at = egui::pos2(rows.left() + col as f32 * width, y);
                    ui.painter().text(at, egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(theme::FONT_SIZE_SECONDARY), color);
                }
            }
            ui.add_space(theme::SPACE_SM);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Prunr waits until it ends. Cancel keeps what it measured so far.")
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
    });
    cancel
}

/// "BiRefNet-lite with CUDA", or "on the processor" while its CPU side runs.
fn step_line(step: Step) -> String {
    let name = prunr_models::display_name(step.model);
    match step.on_gpu() {
        true => format!("{name} with {}", step.column.provider()),
        false => format!("{name} on the processor"),
    }
}

/// "Model 4 of 13 · the processor, DirectML and CUDA each in turn".
fn where_line<'a>(step: Step, engines: impl Iterator<Item = &'a str>) -> String {
    format!("Model {} of {} \u{00b7} {} each in turn", step.index, step.total, super::fmt::and_list(std::iter::once("the processor").chain(engines)))
}

/// A finished model's line: its name, its time on the processor, and its
/// fastest engine with that engine's time; the processor when none beat it.
fn finished_cells(m: &Measuring) -> [String; 4] {
    let cpu = m.cpu_ms.map_or_else(|| "\u{2014}".to_string(), super::fmt::seconds);
    let (fastest, engine) = match m.fastest {
        Some((engine, ms)) => (super::fmt::seconds(ms), engine.to_string()),
        None => ("\u{2014}".to_string(), "Processor".to_string()),
    };
    [prunr_models::display_name(m.model).to_string(), cpu, fastest, engine]
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::{Column, ModelId};

    #[test]
    fn the_lines_name_the_model_its_engine_and_its_turn() {
        let step = |side| Step { index: 4, total: 13, model: ModelId::U2net, side, sides: 3, column: Column::Cuda };
        assert_eq!(step_line(step(2)), "U2Net with CUDA");
        assert_eq!(step_line(step(0)), "U2Net on the processor");
        assert_eq!(where_line(step(1), ["DirectML", "CUDA"].into_iter()), "Model 4 of 13 \u{00b7} the processor, DirectML and CUDA each in turn");
    }

    #[test]
    fn a_finished_model_without_a_faster_engine_runs_on_the_processor() {
        let m = Measuring { model: ModelId::Silueta, cpu_ms: Some(670.0), fastest: None };
        assert_eq!(finished_cells(&m), ["Silueta".to_string(), "0.67 s".into(), "\u{2014}".into(), "Processor".into()]);
        let m = Measuring { model: ModelId::Silueta, cpu_ms: Some(670.0), fastest: Some(("DirectML", 40.0)) };
        assert_eq!(finished_cells(&m)[2..], ["0.04 s".to_string(), "DirectML".into()]);
    }
}
