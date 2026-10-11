//! The progress capsule at the bottom of the canvas, the tile map over the
//! image, and the small pill that follows the cursor while Magic Brush
//! loads. Everything reads the one `DispatchProgress` snapshot, so a
//! pipeline that reports through `prunr_core::Progress` shows up here
//! without widget changes.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::f32::consts::{FRAC_PI_2, TAU};
use std::time::Instant;

use egui::epaint::{PathShape, Shadow};
use egui::text::{LayoutJob, TextWrapping};
use egui::{Color32, FontId, Galley, Painter, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::gui::dispatch_progress::{DispatchProgress, TileState};
use crate::gui::theme;

const CAPSULE_HEIGHT: f32 = 56.0;
const CAPSULE_BOTTOM_INSET: f32 = 28.0;
const CAPSULE_PAD_LEFT: f32 = 12.0;
const CAPSULE_PAD_RIGHT: f32 = 8.0;
const CAPSULE_GAP: f32 = 14.0;
const CAPSULE_MIN_TEXT_WIDTH: f32 = 180.0;
const RING_RADIUS: f32 = 12.5;
const RING_STROKE: f32 = 3.0;
const BUTTON_HEIGHT: f32 = 40.0;
const TITLE_SIZE: f32 = 13.5;
const SUBTITLE_SIZE: f32 = 12.0;
const KEYCAP_SIZE: f32 = 11.0;

const STOPPING: &str = "Stopping";
const FINISHING_CURRENT: &str = "Finishing the current tile";

/// The capsule's two lines: what is happening, then how far along.
fn capsule_lines(progress: &DispatchProgress, now: Instant) -> (&str, Option<String>) {
    if progress.cancelling {
        return (STOPPING, Some(FINISHING_CURRENT.to_string()));
    }
    let detail: Vec<String> = progress.counter_text().into_iter().chain(progress.remaining_text(now)).collect();
    (progress.step_label.as_ref(), (!detail.is_empty()).then(|| detail.join(" \u{b7} ")))
}

/// The capsule's laid-out text, kept between frames: it changes a few
/// times a second at most, the capsule paints sixty.
#[derive(Clone)]
struct CapsuleText {
    key: u64,
    title: Arc<Galley>,
    subtitle: Option<Arc<Galley>>,
    cancel: Arc<Galley>,
    keycap: Arc<Galley>,
}

fn capsule_text(ui: &egui::Ui, painter: &Painter, progress: &DispatchProgress, max_text: f32, now: Instant) -> CapsuleText {
    let keys = super::shortcuts::keys(ui.ctx(), super::shortcuts::Action::Cancel);
    let mut h = std::hash::DefaultHasher::new();
    (progress.step_label.as_ref(), progress.cancelling, progress.outer, progress.inner).hash(&mut h);
    (progress.outer_unit, progress.inner_unit, progress.time_left(now), keys.as_ref(), max_text.to_bits()).hash(&mut h);
    let key = h.finish();
    let id = egui::Id::new("progress_capsule_text");
    let cached = ui.ctx().data(|d| d.get_temp::<CapsuleText>(id));
    if let Some(text) = cached.filter(|c| c.key == key) {
        return text;
    }
    let (title, subtitle) = capsule_lines(progress, now);
    let subtitle_color = if progress.cancelling { theme::CAUTION_TEXT } else { theme::TEXT_SECONDARY };
    let text = CapsuleText {
        key,
        title: single_line(painter, title.to_string(), TITLE_SIZE, theme::TEXT_PRIMARY, max_text),
        subtitle: subtitle.map(|s| single_line(painter, s, SUBTITLE_SIZE, subtitle_color, max_text)),
        cancel: single_line(painter, "Cancel".into(), TITLE_SIZE, theme::TEXT_PRIMARY, f32::INFINITY),
        keycap: single_line(painter, keys.to_string(), KEYCAP_SIZE, theme::TEXT_SECONDARY, f32::INFINITY),
    };
    ui.ctx().data_mut(|d| d.insert_temp(id, text.clone()));
    text
}

/// Paint the tile map and the capsule. Returns `true` when Cancel was
/// clicked this frame.
pub(crate) fn render(
    ui: &mut egui::Ui,
    canvas_rect: Rect,
    img_rect: Option<Rect>,
    progress: &DispatchProgress,
) -> bool {
    let cancelling = progress.cancelling;
    let t = ui.ctx().input(|i| i.time) as f32;
    let painter = ui.painter().with_clip_rect(canvas_rect);
    if let Some(img_rect) = img_rect {
        paint_tile_map(&painter, img_rect, progress, t);
    }

    let max_text = (canvas_rect.width() - 260.0).max(CAPSULE_MIN_TEXT_WIDTH);
    let CapsuleText { title, subtitle, cancel: cancel_label, keycap, .. } = capsule_text(ui, &painter, progress, max_text, Instant::now());
    let subtitle_color = if cancelling { theme::CAUTION_TEXT } else { theme::TEXT_SECONDARY };
    let text_w = subtitle.as_ref().map_or(0.0, |g| g.size().x).max(title.size().x).max(CAPSULE_MIN_TEXT_WIDTH);

    let keycap_size = keycap.size() + Vec2::new(12.0, 2.0);
    let button_w = 14.0 + cancel_label.size().x + 8.0 + keycap_size.x + 12.0;
    let with_button = !cancelling;

    let width = CAPSULE_PAD_LEFT + 2.0 * (RING_RADIUS + RING_STROKE / 2.0) + CAPSULE_GAP + text_w
        + if with_button { CAPSULE_GAP + button_w + CAPSULE_PAD_RIGHT } else { 18.0 };
    let capsule = Rect::from_center_size(
        Pos2::new(canvas_rect.center().x, canvas_rect.bottom() - CAPSULE_BOTTOM_INSET - CAPSULE_HEIGHT / 2.0),
        Vec2::new(width, CAPSULE_HEIGHT),
    );
    let radius = CAPSULE_HEIGHT / 2.0;
    let shadow = Shadow { offset: [0, 14], blur: 36, spread: 0, color: Color32::from_black_alpha(128) };
    painter.add(shadow.as_shape(capsule, radius as u8));
    let border = if cancelling { theme::CAUTION.gamma_multiply(0.35) } else { theme::OVERLAY_CAPSULE_BORDER };
    painter.rect(capsule, radius, theme::OVERLAY_CAPSULE_FILL, Stroke::new(1.0, border), StrokeKind::Inside);

    let ring_center = Pos2::new(capsule.left() + CAPSULE_PAD_LEFT + RING_RADIUS + RING_STROKE / 2.0, capsule.center().y);
    let ring_color = if cancelling { theme::CAUTION } else { theme::ACCENT_BRIGHT };
    // An empty ring would read as stuck; it turns until work moves.
    let fill = progress.fraction().filter(|&f| f > 0.0 && !cancelling);
    paint_ring(&painter, ring_center, RING_RADIUS, RING_STROKE, fill, t, ring_color);

    let text_left = ring_center.x + RING_RADIUS + RING_STROKE / 2.0 + CAPSULE_GAP;
    match subtitle {
        Some(sub) => {
            let top = capsule.center().y - (title.size().y + 2.0 + sub.size().y) / 2.0;
            let below = top + title.size().y + 2.0;
            painter.galley(Pos2::new(text_left, top), title, theme::TEXT_PRIMARY);
            painter.galley(Pos2::new(text_left, below), sub, subtitle_color);
        }
        None => {
            let top = capsule.center().y - title.size().y / 2.0;
            painter.galley(Pos2::new(text_left, top), title, theme::TEXT_PRIMARY);
        }
    }

    if !with_button {
        return false;
    }
    let button = Rect::from_min_size(
        Pos2::new(capsule.right() - CAPSULE_PAD_RIGHT - button_w, capsule.center().y - BUTTON_HEIGHT / 2.0),
        Vec2::new(button_w, BUTTON_HEIGHT),
    );
    let resp = ui.interact(button, ui.id().with("progress_cancel"), Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Cancel"));
    let fill = if resp.hovered() { theme::OVERLAY_BUTTON_HOVER_FILL } else { theme::OVERLAY_BUTTON_FILL };
    painter.rect_filled(button, BUTTON_HEIGHT / 2.0, fill);
    let label_pos = Pos2::new(button.left() + 14.0, button.center().y - cancel_label.size().y / 2.0);
    let cap = Rect::from_min_size(
        Pos2::new(label_pos.x + cancel_label.size().x + 8.0, button.center().y - keycap_size.y / 2.0),
        keycap_size,
    );
    painter.galley(label_pos, cancel_label, theme::TEXT_PRIMARY);
    painter.rect_stroke(cap, 4.0, Stroke::new(1.0, theme::KEYCAP_BORDER), StrokeKind::Inside);
    painter.galley(cap.center() - keycap.size() / 2.0, keycap, theme::TEXT_SECONDARY);
    resp.clicked()
}

/// A small pill beside the cursor (or mid-canvas when the cursor is
/// elsewhere) for a wait the user started by picking a tool.
pub(crate) fn render_loading_pill(ui: &egui::Ui, canvas_rect: Rect, cursor: Option<Pos2>, label: &str) {
    let t = ui.ctx().input(|i| i.time) as f32;
    let painter = ui.painter().with_clip_rect(canvas_rect);
    let text = single_line(&painter, label.to_string(), SUBTITLE_SIZE, theme::TEXT_PRIMARY, f32::INFINITY);
    let size = Vec2::new(9.0 + 16.0 + 8.0 + text.size().x + 12.0, 30.0);
    let anchor = cursor.map_or(canvas_rect.center() - size / 2.0, |c| c + Vec2::new(18.0, 4.0));
    let min = anchor.clamp(canvas_rect.min, (canvas_rect.max - size).max(canvas_rect.min));
    let pill = Rect::from_min_size(min, size);
    let shadow = Shadow { offset: [0, 8], blur: 22, spread: 0, color: Color32::from_black_alpha(115) };
    painter.add(shadow.as_shape(pill, 15));
    painter.rect(pill, 15.0, theme::OVERLAY_CAPSULE_FILL, Stroke::new(1.0, theme::OVERLAY_CAPSULE_BORDER), StrokeKind::Inside);
    paint_ring(&painter, Pos2::new(pill.left() + 9.0 + 8.0, pill.center().y), 6.0, 2.0, None, t, theme::ACCENT_BRIGHT);
    painter.galley(Pos2::new(pill.left() + 33.0, pill.center().y - text.size().y / 2.0), text, theme::TEXT_PRIMARY);
}

fn single_line(painter: &Painter, text: String, size: f32, color: Color32, max_width: f32) -> Arc<Galley> {
    let mut job = LayoutJob::simple_singleline(text, FontId::proportional(size), color);
    job.wrap = TextWrapping::truncate_at_width(max_width);
    painter.layout_job(job)
}

/// Dim the tiles still to come, light the one running, leave done ones clear.
fn paint_tile_map(painter: &Painter, img_rect: Rect, progress: &DispatchProgress, t: f32) {
    let to_screen = |r: &prunr_core::TileRect| Rect::from_min_size(
        img_rect.min + Vec2::new(r.x * img_rect.width(), r.y * img_rect.height()),
        Vec2::new(r.w * img_rect.width(), r.h * img_rect.height()),
    );
    let pulse = 0.75 + 0.25 * (t * 3.0).sin();
    for (cell, state) in progress.tiles.iter().zip(progress.tile_states.iter()) {
        let r = to_screen(cell);
        match state {
            TileState::Waiting => {
                painter.rect(r, 0.0, theme::TILE_WAITING, Stroke::new(0.5, theme::TILE_GRID), StrokeKind::Inside);
            }
            TileState::Running => {
                for (width, alpha) in [(10.0, 0.08), (5.0, 0.18)] {
                    painter.rect_stroke(r, 0.0, Stroke::new(width, theme::ACCENT_BRIGHT.gamma_multiply(alpha * pulse)), StrokeKind::Outside);
                }
                painter.rect(r, 0.0, theme::ACCENT_BRIGHT.gamma_multiply(0.12 * pulse), Stroke::new(2.0, theme::ACCENT_BRIGHT), StrokeKind::Inside);
            }
            TileState::Done => {}
        }
    }
}

/// A ring filled to `fraction` from twelve o'clock, or a turning arc
/// when there is nothing to count yet.
fn paint_ring(painter: &Painter, center: Pos2, radius: f32, width: f32, fraction: Option<f32>, t: f32, color: Color32) {
    painter.circle_stroke(center, radius, Stroke::new(width, theme::PROGRESS_RING_TRACK));
    let (start, sweep) = match fraction {
        Some(f) => (-FRAC_PI_2, f * TAU),
        None => (t * TAU * 0.8, 0.28 * TAU),
    };
    if sweep <= 0.0 {
        return;
    }
    let n = ((sweep / TAU) * 32.0).ceil() as usize + 1;
    let points = (0..=n)
        .map(|i| {
            let a = start + sweep * i as f32 / n as f32;
            center + radius * Vec2::new(a.cos(), a.sin())
        })
        .collect();
    painter.add(PathShape::line(points, Stroke::new(width, color)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::dispatch_progress::ProgressKind;
    use prunr_core::Step;

    #[test]
    fn the_capsule_names_the_step_then_the_counts() {
        let p = DispatchProgress { inner: (12, 49), ..DispatchProgress::new(ProgressKind::Upscale, Step::Upscaling.label()) };
        let (title, sub) = capsule_lines(&p, Instant::now());
        assert_eq!(title, "Upscaling");
        assert_eq!(sub.as_deref(), Some("Step 12 of 49"));
    }

    #[test]
    fn a_step_with_nothing_counted_is_one_line() {
        let p = DispatchProgress::new(ProgressKind::Seg, Step::LoadingModel.label());
        assert_eq!(capsule_lines(&p, Instant::now()), ("Loading the model", None));
    }

    #[test]
    fn a_cancel_says_stopping_whatever_the_step() {
        let p = DispatchProgress { inner: (3, 8), cancelling: true, ..DispatchProgress::new(ProgressKind::Inpaint, Step::Denoising.label()) };
        let (title, sub) = capsule_lines(&p, Instant::now());
        assert_eq!(title, STOPPING);
        assert_eq!(sub.as_deref(), Some(FINISHING_CURRENT));
    }
}
