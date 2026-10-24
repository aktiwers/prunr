//! The provider check's notices (`prunr_core::provider_check`), from this
//! process's sessions and the worker's, held until the window shows them.

use std::sync::{Mutex, OnceLock};

use prunr_core::provider_check::Notice;

static WAITING: Mutex<Vec<Notice>> = Mutex::new(Vec::new());
static WINDOW: OnceLock<egui::Context> = OnceLock::new();

/// The observer for this process, and the worker link's landing.
pub fn observe(notice: Notice) {
    WAITING.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(notice);
    if let Some(ctx) = WINDOW.get() {
        ctx.request_repaint();
    }
}

/// The notices so far, for the window to show; the window is woken for
/// the next.
pub(crate) fn take(ctx: &egui::Context) -> Vec<Notice> {
    let _ = WINDOW.set(ctx.clone());
    std::mem::take(&mut *WAITING.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// How long a notice stays up: a check runs the model on the CPU, about
/// 17-19 s for BiRefNet on an i5-11600KF, so its notice lasts as long.
pub(crate) fn shown_for(notice: &Notice) -> std::time::Duration {
    std::time::Duration::from_secs(match notice {
        Notice::Checking { .. } => 20,
        Notice::PassedOver { .. } => 10,
    })
}

pub(crate) fn text(notice: &Notice) -> String {
    let name = |model: prunr_models::ModelId| prunr_models::descriptor(model).map_or("The model", |d| d.display_name);
    match notice {
        Notice::Checking { model } => format!("Checking the graphics card with {}, once. This takes a few seconds.", name(*model)),
        Notice::PassedOver { model, reason } => format!("{} runs on the CPU here: {reason}.", name(*model)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_notice_names_the_model_and_the_reason() {
        let model = prunr_models::ModelId::DdcolorTiny;
        assert_eq!(text(&Notice::Checking { model }), "Checking the graphics card with DDColor, once. This takes a few seconds.");
        let passed_over = Notice::PassedOver { model, reason: "it runs slower on CoreML than on the CPU (950 ms against 1000 ms)".into() };
        assert_eq!(text(&passed_over), "DDColor runs on the CPU here: it runs slower on CoreML than on the CPU (950 ms against 1000 ms).");
    }
}
