//! Unified progress widgets — banner (top strip) and modal (centered
//! pill) — that read from `Processor.dispatch_progress`. One source of
//! truth replaces the three legacy renderers (`render_processing`,
//! `render_inpaint_progress`, the inline upscale toolbar bar).
//!
//! Both renderers take the same `DispatchProgress` snapshot and a
//! `cancelling` flag, and return `true` when the user clicked the
//! Cancel button this frame. Visual differences:
//!
//!   - **Banner** (top 44 px strip): minimal canvas obscuring,
//!     single-line counter, inline Cancel button. Uses the flat
//!     counter ("step 13 of 24") because vertical room is tight.
//!   - **Modal** (centered pill, ~320×120): more room for richer
//!     text — nested counter ("Tile 2 of 3 — step 5 of 8") on its
//!     own line, plus the `step_label`. Cancel hint as text
//!     ("Press Esc to cancel") since the modal is non-interactive
//!     to keep the canvas clean for the breathing/shimmer effect
//!     painted underneath.
//!
//! The user picks the style in Settings → Behavior. The render
//! callsite passes the user's choice; both functions accept the
//! same inputs so the swap is a one-line branch.

use egui::{Color32, FontId, Pos2, Rect, RichText, Vec2};

use crate::gui::dispatch_progress::DispatchProgress;
use crate::gui::theme;

const BANNER_HEIGHT_PX: f32 = 44.0;
const MODAL_WIDTH_PX: f32 = 320.0;
const MODAL_HEIGHT_PX: f32 = 120.0;
const PULSE_DOT_RADIUS_PX: f32 = 3.5;
const PULSE_DOTS_SPACING_PX: f32 = 10.0;
const CANCEL_BUTTON_WIDTH_PX: f32 = 90.0;
const CANCEL_BUTTON_MARGIN_PX: f32 = 16.0;

/// Top-of-canvas progress banner. Single-line layout: pulse dots,
/// headline + counter + step label, Cancel button. Returns `true`
/// when the user clicked Cancel this frame.
///
/// Hides the Cancel button while `cancelling` is true — a still-
/// clickable button reads as "the first click didn't take" while
/// the worker finishes its current step.
pub(crate) fn render_banner(
    ui: &mut egui::Ui,
    canvas_rect: Rect,
    progress: &DispatchProgress,
    cancelling: bool,
) -> bool {
    let t = ui.ctx().input(|i| i.time) as f32;
    let banner = Rect::from_min_size(
        canvas_rect.min,
        Vec2::new(canvas_rect.width(), BANNER_HEIGHT_PX),
    );
    ui.painter().rect_filled(banner, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 160));
    let center = banner.center();

    paint_pulse_dots(ui, Pos2::new(center.x - 56.0, center.y), t);

    let label = banner_label(progress, cancelling);
    ui.painter().text(
        Pos2::new(center.x - 22.0, center.y),
        egui::Align2::LEFT_CENTER,
        &label,
        FontId::proportional(14.0),
        theme::TEXT_PRIMARY,
    );

    if cancelling {
        return false;
    }
    render_cancel_button(ui, canvas_rect, center.y)
}

/// Centered-modal progress pill. Three-line layout: animated
/// headline ("Erasing.."), counter + step label, Esc hint.
/// Painted only — no inline Cancel button (the Esc key handles
/// cancellation and the modal is meant to look "calm").
///
/// Returns `false` always (no inline cancel). The render-site
/// keeps the return for symmetry with `render_banner` so the
/// settings toggle swap is a one-line change.
pub(crate) fn render_modal(
    ui: &mut egui::Ui,
    canvas_rect: Rect,
    progress: &DispatchProgress,
    cancelling: bool,
) -> bool {
    let t = ui.ctx().input(|i| i.time) as f32;
    let center = canvas_rect.center();
    let pill_rect = Rect::from_center_size(center, Vec2::new(MODAL_WIDTH_PX, MODAL_HEIGHT_PX));
    ui.painter().rect_filled(pill_rect, 14.0, Color32::from_rgba_unmultiplied(0, 0, 0, 180));

    let headline = modal_headline(progress, cancelling, t);
    ui.painter().text(
        center - Vec2::new(0.0, 28.0),
        egui::Align2::CENTER_CENTER,
        &headline,
        FontId::proportional(theme::FONT_SIZE_HEADING),
        theme::TEXT_PRIMARY,
    );

    // Counter line (nested form when outer is set, flat otherwise).
    if let Some(counter) = modal_counter_text(progress) {
        ui.painter().text(
            center,
            egui::Align2::CENTER_CENTER,
            &counter,
            FontId::proportional(theme::FONT_SIZE_BODY),
            theme::TEXT_SECONDARY,
        );
    }

    // Step label line — the rich "what's happening" text.
    ui.painter().text(
        center + Vec2::new(0.0, 22.0),
        egui::Align2::CENTER_CENTER,
        progress.step_label.as_ref(),
        FontId::proportional(theme::FONT_SIZE_BODY - 1.0),
        theme::TEXT_SECONDARY,
    );

    ui.painter().text(
        center + Vec2::new(0.0, 44.0),
        egui::Align2::CENTER_CENTER,
        "Press Esc to cancel",
        FontId::proportional(theme::FONT_SIZE_MONO),
        theme::TEXT_SECONDARY,
    );

    ui.ctx().request_repaint_after(std::time::Duration::from_millis(66));
    false
}

fn paint_pulse_dots(ui: &egui::Ui, anchor_left: Pos2, t: f32) {
    for i in 0..3 {
        let phase = t * 3.0 - i as f32 * 0.6;
        let a = (phase.sin() * 0.5 + 0.5).clamp(0.3, 1.0);
        let dot_x = anchor_left.x + i as f32 * PULSE_DOTS_SPACING_PX;
        ui.painter().circle_filled(
            Pos2::new(dot_x, anchor_left.y),
            PULSE_DOT_RADIUS_PX,
            theme::ACCENT.gamma_multiply(a),
        );
    }
}

fn banner_label(progress: &DispatchProgress, cancelling: bool) -> String {
    if cancelling {
        return "Cancelling…".to_string();
    }
    let headline = progress.kind.headline();
    match progress.flat_counter_text() {
        // "Upscaling — tile 12 of 49"
        Some(counter) => format!("{headline} \u{2014} {counter}"),
        // Indeterminate (no inner total yet): "Upscaling… (step label)"
        None if progress.step_label.is_empty() => format!("{headline}\u{2026}"),
        None => format!("{headline}\u{2026} {}", progress.step_label),
    }
}

fn modal_headline(progress: &DispatchProgress, cancelling: bool, t: f32) -> String {
    if cancelling {
        return "Cancelling…".to_string();
    }
    // Animated trailing dots — matches the legacy `render_processing`
    // feel so the modal style continues to read as "active work".
    let dots = match (t * 2.0) as usize % 4 {
        0 => ".",
        1 => "..",
        2 => "...",
        _ => "....",
    };
    format!("{}{}", progress.kind.headline(), dots)
}

fn modal_counter_text(progress: &DispatchProgress) -> Option<String> {
    // Prefer nested when available: it carries the most info.
    if let Some(nested) = progress.nested_counter_text() {
        return Some(nested);
    }
    progress.flat_counter_text()
}

fn render_cancel_button(ui: &mut egui::Ui, canvas_rect: Rect, center_y: f32) -> bool {
    let btn_size = Vec2::new(CANCEL_BUTTON_WIDTH_PX, theme::CHIP_HEIGHT);
    let btn_rect = Rect::from_center_size(
        Pos2::new(
            canvas_rect.right() - CANCEL_BUTTON_MARGIN_PX - CANCEL_BUTTON_WIDTH_PX / 2.0,
            center_y,
        ),
        btn_size,
    );
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(btn_rect)
            .layout(egui::Layout::centered_and_justified(egui::Direction::LeftToRight)),
    );
    let resp = child.add(
        egui::Button::new(
            RichText::new("Cancel (Esc)")
                .size(theme::FONT_SIZE_BODY - 1.0)
                .color(theme::TEXT_PRIMARY),
        )
        .min_size(btn_size),
    );
    resp.clicked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::dispatch_progress::ProgressKind;
    use std::borrow::Cow;

    fn upscale(inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress {
            kind: ProgressKind::Upscale,
            outer: None,
            inner,
            step_label: Cow::Borrowed("Tile inference"),
        }
    }

    fn sd_nested(outer: (u32, u32), inner: (u32, u32)) -> DispatchProgress {
        DispatchProgress {
            kind: ProgressKind::SdInpaint,
            outer: Some(outer),
            inner,
            step_label: Cow::Borrowed("Denoising"),
        }
    }

    #[test]
    fn banner_label_uses_flat_counter_for_unnested() {
        let p = upscale((12, 49));
        let label = banner_label(&p, false);
        assert!(label.contains("Upscaling"), "got: {label}");
        assert!(label.contains("tile 12 of 49"), "got: {label}");
    }

    #[test]
    fn banner_label_uses_flat_counter_for_nested() {
        // 24-tile-stroke version of the user's example.
        let p = sd_nested((2, 3), (5, 8));
        let label = banner_label(&p, false);
        assert!(label.contains("Erasing"), "got: {label}");
        assert!(label.contains("step 13 of 24"), "got: {label}");
    }

    #[test]
    fn banner_label_swaps_to_cancelling() {
        let p = upscale((5, 10));
        let label = banner_label(&p, true);
        assert_eq!(label, "Cancelling\u{2026}");
    }

    #[test]
    fn banner_label_indeterminate_uses_step_label() {
        let p = DispatchProgress {
            kind: ProgressKind::Seg,
            outer: None,
            inner: (0, 0),
            step_label: Cow::Borrowed("Loading model"),
        };
        let label = banner_label(&p, false);
        assert!(label.contains("Processing"), "got: {label}");
        assert!(label.contains("Loading model"), "got: {label}");
    }

    #[test]
    fn modal_counter_prefers_nested_when_outer_set() {
        let p = sd_nested((2, 3), (5, 8));
        let counter = modal_counter_text(&p).expect("nested has counter");
        assert!(counter.contains("Tile 2 of 3"), "got: {counter}");
        assert!(counter.contains("step 5 of 8"), "got: {counter}");
    }

    #[test]
    fn modal_counter_falls_back_to_flat_without_outer() {
        let p = upscale((12, 49));
        let counter = modal_counter_text(&p).expect("flat counter");
        assert_eq!(counter, "tile 12 of 49");
    }

    #[test]
    fn modal_counter_none_when_indeterminate() {
        let p = upscale((0, 0));
        assert!(modal_counter_text(&p).is_none());
    }

    #[test]
    fn modal_headline_cycles_through_dot_states() {
        let p = upscale((1, 10));
        // 4 phases × 1.0 / 2.0 = 2 seconds per cycle. Sample each.
        for (t_phase, expected_dots) in [(0.0, "."), (0.6, ".."), (1.1, "..."), (1.6, "....")] {
            let h = modal_headline(&p, false, t_phase);
            assert_eq!(h, format!("Upscaling{expected_dots}"), "t={t_phase}");
        }
    }

    #[test]
    fn modal_headline_cancelling_overrides_dots() {
        let p = upscale((1, 10));
        assert_eq!(modal_headline(&p, true, 0.5), "Cancelling\u{2026}");
    }
}
