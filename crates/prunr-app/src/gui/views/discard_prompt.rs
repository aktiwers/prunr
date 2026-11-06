//! Asked before work no save wrote is lost: closing the app, or taking
//! images out of the queue.

use egui::RichText;

use super::chip::{button, ButtonKind};
use crate::gui::theme;

/// What would lose the work.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Discard {
    Close,
    /// `announce`: a toast says how many went.
    Remove { ids: Vec<u64>, announce: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DiscardChoice {
    /// Close or remove all the same.
    Go,
    /// Keep the work: nothing happens.
    Keep,
}

/// The prompt for `discard`, with `unsaved` images at stake; `None` while
/// it waits for an answer. The backdrop keeps the work, as Cancel does.
pub(crate) fn render(ctx: &egui::Context, discard: &Discard, unsaved: usize) -> Option<DiscardChoice> {
    let (title, verb, go) = match discard {
        Discard::Close => ("Close without saving?", "Closing", "Close without saving"),
        Discard::Remove { .. } => ("Remove without saving?", "Removing", "Remove"),
    };
    let images = super::fmt::count(unsaved, "image", "images");
    let have = if unsaved == 1 { "has" } else { "have" };
    let mut choice = None;
    let backdrop_closed = theme::standard_modal_window(ctx, "discard_prompt", title, [theme::STANDARD_MODAL_WIDTH, 170.0], |ui| {
        ui.label(
            RichText::new(format!("{images} {have} work that is not saved. {verb} loses it."))
                .size(theme::FONT_SIZE_BODY)
                .color(theme::TEXT_PRIMARY),
        );
        ui.add_space(theme::SPACE_MD);
        ui.horizontal(|ui| {
            if button(ui, ButtonKind::Secondary, "Cancel").clicked() {
                choice = Some(DiscardChoice::Keep);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if button(ui, ButtonKind::Destructive, go).clicked() {
                    choice = Some(DiscardChoice::Go);
                }
            });
        });
    });
    if backdrop_closed && choice.is_none() {
        choice = Some(DiscardChoice::Keep);
    }
    choice
}
