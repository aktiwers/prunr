use std::collections::HashSet;
use std::sync::atomic::Ordering;

use egui::{Color32, Pos2, Rect, RichText, Stroke, Vec2};
use egui_material_icons::icons::*;
use crate::gui::app::PrunrApp;
use crate::gui::item::{BatchItem, BatchStatus};
use crate::gui::theme;

/// Manual pointer hit test for a rect — returns (hovered, clicked).
fn hit_test(ui: &egui::Ui, rect: Rect) -> (bool, bool) {
    ui.ctx().input(|i| {
        let hover = i.pointer.hover_pos().map_or(false, |p| rect.contains(p));
        (hover, hover && i.pointer.primary_clicked())
    })
}

/// Collected per-frame actions produced by the item rows, applied after the
/// scroll-area loop to avoid mutating the batch during iteration.
#[derive(Default)]
struct SidebarActions {
    swap_from: Option<usize>,
    swap_to: Option<usize>,
    remove_idx: Option<usize>,
    save_idx: Option<usize>,
    needs_repaint: bool,
}

/// Values that are constant for the whole item loop — computed once and
/// passed by reference to avoid re-reading per-row.
struct RowContext<'a> {
    dragging_ids: Option<&'a HashSet<u64>>,
    sidebar_escape_rect: Rect,
    anim_time: f32,
    visible_rect: Rect,
    item_width: f32,
    item_height: f32,
}

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    // Sidebar rect — used to detect when a drag escapes the sidebar (= drag-out to external app).
    // The 12px expansion is a dead-zone so small drifts while reordering don't trigger drag-out.
    let sidebar_escape_rect = ui.clip_rect().expand(12.0);
    let dragging_ids = snapshot_dragging_ids(app);

    ui.vertical(|ui| {
        if app.batch.items.is_empty() {
            render_empty_state(ui);
            return;
        }

        render_header(ui, app);
        ui.add_space(theme::SPACE_XS);

        pump_thumbnail_results(ui, app);

        let anim_time = ui.ctx().input(|i| i.time) as f32;
        let actions = render_item_list(ui, app, dragging_ids.as_ref(), sidebar_escape_rect, anim_time);

        // Request repaint for animations/pending thumbnails — throttled to ~15fps.
        if actions.needs_repaint || app.batch.items.iter().any(|i| i.thumb_pending) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(66));
        }
        apply_actions(app, &actions);
    });
}

/// Snapshot the drag-out set once per frame — locks/clones only when a drag
/// is actually active, so idle frames are free.
fn snapshot_dragging_ids(app: &PrunrApp) -> Option<HashSet<u64>> {
    if !app.drag_export.active.load(Ordering::Acquire) {
        return None;
    }
    app.drag_export.items.lock().ok().map(|s| s.clone())
}

fn render_empty_state(ui: &mut egui::Ui) {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            ui.label(
                RichText::new("Drop images here\nto queue them")
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );
        },
    );
}

/// Select-all checkbox + "N/M" selected counter.
fn render_header(ui: &mut egui::Ui, app: &mut PrunrApp) {
    let count = app.batch.items.len();
    let all_selected = count > 0 && app.batch.items.iter().all(|i| i.selected);
    let mut select_all = all_selected;
    ui.horizontal(|ui| {
        ui.spacing_mut().icon_width = 20.0;
        ui.spacing_mut().icon_spacing = 8.0;
        if ui
            .checkbox(
                &mut select_all,
                RichText::new("Select All")
                    .size(theme::FONT_SIZE_BODY)
                    .color(theme::TEXT_PRIMARY),
            )
            .changed()
        {
            for item in &mut app.batch.items {
                item.selected = select_all;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Always show "N/M" for consistency — "0/4" reads as "0 of 4
            // selected" rather than the ambiguous "4" when nothing's checked.
            let selected = app.batch.items.iter().filter(|i| i.selected).count();
            ui.label(
                RichText::new(format!("{selected}/{count}"))
                    .size(theme::FONT_SIZE_MONO)
                    .color(theme::TEXT_SECONDARY),
            );
        });
    });
}

/// Drain completed thumbnails from the background decoder into textures.
fn pump_thumbnail_results(ui: &egui::Ui, app: &mut PrunrApp) {
    while let Ok((item_id, tw, th, pixels)) = app.batch.bg_io.thumb_rx.try_recv() {
        if let Some(item) = app.batch.items.iter_mut().find(|b| b.id == item_id) {
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [tw as usize, th as usize],
                &pixels,
            );
            item.thumb_texture = Some(ui.ctx().load_texture(
                format!("thumb_{item_id}"),
                ci,
                egui::TextureOptions::LINEAR,
            ));
            item.thumb_pending = false;
        }
    }
}

fn render_item_list(
    ui: &mut egui::Ui,
    app: &mut PrunrApp,
    dragging_ids: Option<&HashSet<u64>>,
    sidebar_escape_rect: Rect,
    anim_time: f32,
) -> SidebarActions {
    let mut actions = SidebarActions::default();
    let item_width = ui.available_width() - theme::SPACE_SM;
    let item_height = theme::THUMBNAIL_SIZE + theme::SPACE_SM;

    egui::ScrollArea::vertical().show(ui, |ui| {
        let ctx = RowContext {
            dragging_ids,
            sidebar_escape_rect,
            anim_time,
            visible_rect: ui.clip_rect(),
            item_width,
            item_height,
        };
        for i in 0..app.batch.items.len() {
            render_item_row(ui, app, i, &ctx, &mut actions);
        }
    });
    actions
}

fn render_item_row(
    ui: &mut egui::Ui,
    app: &mut PrunrApp,
    i: usize,
    ctx: &RowContext,
    actions: &mut SidebarActions,
) {
    let is_selected = i == app.batch.selected_index;

    // Allocate space for item (always — keeps layout/scrollbar correct).
    let (item_rect, item_response) = ui.allocate_exact_size(
        Vec2::new(ctx.item_width, ctx.item_height),
        egui::Sense::click_and_drag(),
    );

    // Skip painting for off-screen items (virtualization).
    if item_rect.max.y < ctx.visible_rect.min.y || item_rect.min.y > ctx.visible_rect.max.y {
        return;
    }

    paint_item_background(ui, item_rect, is_selected);

    if render_thumbnail_layer(ui, app, i, item_rect, ctx.dragging_ids) {
        actions.needs_repaint = true;
    }

    paint_selection_checkbox(ui, &mut app.batch.items[i], item_rect);

    if matches!(app.batch.items[i].status, BatchStatus::Processing) {
        paint_processing_animation(ui, item_rect, ctx.anim_time);
        actions.needs_repaint = true;
    }

    if paint_status_indicator(ui, &app.batch.items[i].status, item_rect, ctx.anim_time) {
        actions.needs_repaint = true;
    }

    let close_clicked =
        paint_hover_buttons(ui, &app.batch.items[i], i, item_rect, &item_response, actions);

    // DnD: set drag payload (intra-sidebar reorder) and check for drops.
    item_response.dnd_set_drag_payload(i);
    if let Some(src_idx) = item_response.dnd_release_payload::<usize>() {
        actions.swap_from = Some(*src_idx);
        actions.swap_to = Some(i);
    }

    detect_drag_escape(ui, app, i, &item_response, ctx.sidebar_escape_rect);

    // Click to select (skip if a hover button handled the click).
    if !close_clicked && item_response.clicked() && app.batch.selected_index != i {
        app.batch.selected_index = i;
        app.canvas_switch_id += 1;
        app.zoom_state.reset();
        let egui_ctx = ui.ctx().clone();
        app.sync_selected_batch_textures(&egui_ctx);
    }

    paint_insertion_line(ui, &item_response, item_rect);

    // Filename tooltip (must be last — `on_hover_text` consumes the response).
    item_response.on_hover_text(&app.batch.items[i].filename);

    ui.add_space(theme::SPACE_XS);
}

/// Request on demand, show a spinner while pending, then fade the texture in.
/// Returns `true` if anything painted here is still animating and the frame
/// needs another repaint.
fn render_thumbnail_layer(
    ui: &mut egui::Ui,
    app: &mut PrunrApp,
    i: usize,
    item_rect: Rect,
    dragging_ids: Option<&HashSet<u64>>,
) -> bool {
    let mut needs_repaint = false;
    if should_request_thumbnail(&app.batch.items[i]) {
        request_item_thumbnail(app, i);
        needs_repaint = true;
    }
    if app.batch.items[i].thumb_texture.is_none() && app.batch.items[i].thumb_pending {
        draw_loading_spinner(ui, item_rect);
        needs_repaint = true;
    }
    // Compute fade unconditionally so the animation bookkeeping stays in sync
    // whether or not the texture is available yet.
    let has_thumb = app.batch.items[i].thumb_texture.is_some();
    let fade = ui.ctx().animate_bool_with_time(
        egui::Id::new(("thumb_fade", app.batch.items[i].id)),
        has_thumb,
        0.2,
    );
    let is_dragging_out = dragging_ids.is_some_and(|s| s.contains(&app.batch.items[i].id));
    paint_thumbnail(ui, &app.batch.items[i], item_rect, is_dragging_out, fade);
    needs_repaint || fade < 1.0
}

/// True when the item needs a thumbnail but hasn't asked for one yet. Phase
/// 10-07 will gate this on viewport visibility; today the gate is just
/// "no texture, not in flight".
fn should_request_thumbnail(item: &BatchItem) -> bool {
    item.thumb_texture.is_none() && !item.thumb_pending
}

fn request_item_thumbnail(app: &mut PrunrApp, i: usize) {
    app.batch.items[i].thumb_pending = true;
    let item_id = app.batch.items[i].id;
    app.batch.request_thumbnail(
        item_id,
        &app.batch.items[i].source,
        app.batch.items[i].result_rgba.as_ref(),
    );
}

fn paint_item_background(ui: &egui::Ui, item_rect: Rect, is_selected: bool) {
    let bg_color = if is_selected {
        theme::SIDEBAR_ITEM_SELECTED
    } else {
        theme::SIDEBAR_ITEM_BG
    };
    ui.painter()
        .rect_filled(item_rect, theme::THUMBNAIL_ROUNDING, bg_color);
    if is_selected {
        ui.painter().rect_stroke(
            item_rect,
            theme::THUMBNAIL_ROUNDING,
            Stroke::new(2.0, theme::SIDEBAR_SELECTED_BORDER),
            egui::StrokeKind::Inside,
        );
    }
}

fn draw_loading_spinner(ui: &mut egui::Ui, item_rect: Rect) {
    let spinner_rect = Rect::from_center_size(item_rect.center(), Vec2::splat(20.0));
    ui.put(spinner_rect, egui::Spinner::new().size(20.0).color(theme::ACCENT));
}

fn paint_thumbnail(
    ui: &egui::Ui,
    item: &BatchItem,
    item_rect: Rect,
    is_dragging_out: bool,
    fade: f32,
) {
    let Some(ref thumb_tex) = item.thumb_texture else {
        return;
    };
    let tex_size = thumb_tex.size_vec2();
    // 4px padding so the thumbnail doesn't touch the selected-border stroke.
    let pad = 4.0;
    let max_w = item_rect.width() - pad;
    let max_h = item_rect.height() - pad;
    let scale = (max_w / tex_size.x).min(max_h / tex_size.y).min(1.0);
    let fitted = tex_size * scale;
    let thumb_rect = Rect::from_center_size(item_rect.center(), fitted);

    // Render-time bg fill: result thumbs are transparent where pixels were
    // removed; paint the bg color behind so the sidebar matches the canvas.
    // Source-only thumbs are opaque and don't need this fill.
    if item.result_rgba.is_some() {
        if let Some(bg) = item.settings.bg_rgb() {
            ui.painter()
                .rect_filled(thumb_rect, 0.0, Color32::from_rgb(bg[0], bg[1], bg[2]));
        }
    }

    // Dim to ~40% while being dragged out to an external app.
    let dim = if is_dragging_out { 0.4 } else { 1.0 };
    let alpha = (fade * dim * 255.0) as u8;
    ui.painter().image(
        thumb_tex.id(),
        thumb_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
    );
}

fn paint_selection_checkbox(ui: &egui::Ui, item: &mut BatchItem, item_rect: Rect) {
    let cb_size = 16.0;
    let cb_center = Pos2::new(
        item_rect.min.x + 4.0 + cb_size * 0.5,
        item_rect.min.y + 4.0 + cb_size * 0.5,
    );
    let cb_rect = Rect::from_center_size(cb_center, Vec2::splat(cb_size));

    let (cb_hovered, cb_clicked) = hit_test(ui, cb_rect);
    if cb_clicked {
        item.selected = !item.selected;
    }

    let cb_bg = if item.selected {
        theme::ACCENT
    } else if cb_hovered {
        Color32::from_rgb(0x50, 0x50, 0x50)
    } else {
        Color32::from_rgba_unmultiplied(0, 0, 0, 160)
    };
    ui.painter().rect_filled(cb_rect, 3.0, cb_bg);
    ui.painter().rect_stroke(
        cb_rect,
        3.0,
        Stroke::new(1.0, theme::TEXT_SECONDARY),
        egui::StrokeKind::Outside,
    );
    if item.selected {
        ui.painter().text(
            cb_center,
            egui::Align2::CENTER_CENTER,
            ICON_CHECK.codepoint,
            egui::FontId::proportional(12.0),
            Color32::WHITE,
        );
    }
}

fn paint_processing_animation(ui: &egui::Ui, item_rect: Rect, anim_time: f32) {
    // Shimmer sweep.
    let sweep =
        ((anim_time * 0.7).fract()) * (item_rect.width() + 40.0) - 20.0 + item_rect.min.x;
    let shimmer = Rect::from_min_max(
        Pos2::new(sweep - 20.0, item_rect.min.y),
        Pos2::new(sweep + 20.0, item_rect.max.y),
    )
    .intersect(item_rect);
    if shimmer.width() > 0.0 {
        ui.painter().rect_filled(
            shimmer,
            theme::THUMBNAIL_ROUNDING,
            Color32::from_rgba_unmultiplied(0x7b, 0x2d, 0x8e, 50),
        );
    }
    // Pulsing blue border.
    let pulse = (anim_time * 2.5).sin() * 0.5 + 0.5;
    let border_alpha = (40.0 + pulse * 80.0) as u8;
    ui.painter().rect_stroke(
        item_rect,
        theme::THUMBNAIL_ROUNDING,
        Stroke::new(
            2.0,
            Color32::from_rgba_unmultiplied(0x7b, 0x2d, 0x8e, border_alpha),
        ),
        egui::StrokeKind::Inside,
    );
}

/// Bottom-right status dot / checkmark / error icon. Returns true if the
/// indicator animates and the frame needs a repaint.
fn paint_status_indicator(
    ui: &egui::Ui,
    status: &BatchStatus,
    item_rect: Rect,
    anim_time: f32,
) -> bool {
    match status {
        BatchStatus::Pending => {
            let dot = Pos2::new(item_rect.max.x - 8.0, item_rect.max.y - 8.0);
            ui.painter().circle_filled(dot, 3.0, theme::STATUS_ICON_PENDING);
            false
        }
        BatchStatus::Processing => {
            let dot = Pos2::new(item_rect.max.x - 8.0, item_rect.max.y - 8.0);
            let pulse = (anim_time * 3.0).sin() * 0.5 + 0.5;
            let size = 3.0 + pulse * 2.0;
            ui.painter().circle_filled(dot, size, theme::ACCENT);
            true
        }
        BatchStatus::Done => {
            let pos = Pos2::new(item_rect.max.x - 10.0, item_rect.max.y - 10.0);
            ui.painter().text(
                pos,
                egui::Align2::CENTER_CENTER,
                ICON_CHECK.codepoint,
                egui::FontId::proportional(14.0),
                theme::ACCENT_GREEN,
            );
            false
        }
        BatchStatus::Error(_) => {
            let pos = Pos2::new(item_rect.max.x - 10.0, item_rect.max.y - 10.0);
            ui.painter().text(
                pos,
                egui::Align2::CENTER_CENTER,
                ICON_ERROR.codepoint,
                egui::FontId::proportional(14.0),
                theme::DESTRUCTIVE,
            );
            false
        }
    }
}

/// Delete (top-right) and Save (bottom-left, Done-only) buttons shown on
/// hover. Returns true if either was pressed, so the caller can suppress the
/// row's own click-to-select.
fn paint_hover_buttons(
    ui: &egui::Ui,
    item: &BatchItem,
    i: usize,
    item_rect: Rect,
    item_response: &egui::Response,
    actions: &mut SidebarActions,
) -> bool {
    if !item_response.hovered() || ui.ctx().dragged_id().is_some() {
        return false;
    }
    let mut close_clicked = false;
    let btn_size = 20.0;

    // Delete — top-right.
    let del_center = Pos2::new(
        item_rect.max.x - 4.0 - btn_size * 0.5,
        item_rect.min.y + 4.0 + btn_size * 0.5,
    );
    if paint_hover_circle_btn(ui, del_center, btn_size, ICON_DELETE.codepoint, theme::DESTRUCTIVE) {
        actions.remove_idx = Some(i);
        close_clicked = true;
    }

    // Save — bottom-left, Done items only.
    if matches!(item.status, BatchStatus::Done) {
        let save_center = Pos2::new(
            item_rect.min.x + 4.0 + btn_size * 0.5,
            item_rect.max.y - 4.0 - btn_size * 0.5,
        );
        if paint_hover_circle_btn(ui, save_center, btn_size, ICON_SAVE.codepoint, theme::ACCENT) {
            actions.save_idx = Some(i);
            close_clicked = true;
        }
    }
    close_clicked
}

/// Circular hover button with an icon. Returns true if pressed this frame.
/// `hover_color` is the fill shown on hover; idle is translucent black.
fn paint_hover_circle_btn(
    ui: &egui::Ui,
    center: Pos2,
    size: f32,
    icon: &str,
    hover_color: Color32,
) -> bool {
    let rect = Rect::from_center_size(center, Vec2::splat(size));
    let (hovered, pressed) = hit_test(ui, rect);
    let bg = if hovered {
        hover_color
    } else {
        Color32::from_rgba_unmultiplied(0, 0, 0, 200)
    };
    ui.painter().circle_filled(center, size * 0.5, bg);
    ui.painter().text(
        center,
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(12.0),
        Color32::WHITE,
    );
    pressed
}

/// If this row is being dragged AND the pointer has crossed outside the
/// sidebar's dead-zone, queue an OS-level drag of the item (or the whole
/// checkbox-selected group). Skips items still processing.
fn detect_drag_escape(
    ui: &egui::Ui,
    app: &mut PrunrApp,
    i: usize,
    item_response: &egui::Response,
    sidebar_escape_rect: Rect,
) {
    if !ui.ctx().is_being_dragged(item_response.id) {
        return;
    }
    if app.drag_export.pending.is_some() || app.drag_export.active.load(Ordering::Acquire) {
        return;
    }
    let Some(pos) = ui.ctx().pointer_hover_pos() else {
        return;
    };
    if sidebar_escape_rect.contains(pos) {
        return;
    }

    let source_selected = app.batch.items[i].selected;
    let ids: Vec<u64> = if source_selected {
        app.batch
            .items
            .iter()
            .filter(|b| b.selected && !matches!(b.status, BatchStatus::Processing))
            .map(|b| b.id)
            .collect()
    } else if !matches!(app.batch.items[i].status, BatchStatus::Processing) {
        vec![app.batch.items[i].id]
    } else {
        Vec::new()
    };
    if !ids.is_empty() {
        app.drag_export.pending = Some(ids);
    }
}

/// Hover insertion line while another row is being dragged over this one.
fn paint_insertion_line(ui: &egui::Ui, item_response: &egui::Response, item_rect: Rect) {
    if ui.ctx().is_being_dragged(item_response.id) {
        return; // this row is the one being dragged
    }
    if item_response.hovered() && ui.ctx().dragged_id().is_some() {
        ui.painter().hline(
            item_rect.x_range(),
            item_rect.min.y,
            Stroke::new(2.0, theme::INSERTION_LINE),
        );
    }
}

/// Apply everything collected during the row loop. Ordering mirrors the
/// pre-refactor code: remove → save → reorder.
fn apply_actions(app: &mut PrunrApp, actions: &SidebarActions) {
    if let Some(idx) = actions.remove_idx {
        app.remove_batch_item(idx);
    }
    if let Some(idx) = actions.save_idx {
        app.save_item_to_file(idx);
    }
    if let (Some(from), Some(to)) = (actions.swap_from, actions.swap_to) {
        if from != to {
            apply_reorder(app, from, to);
        }
    }
}

fn apply_reorder(app: &mut PrunrApp, from: usize, to: usize) {
    let item = app.batch.items.remove(from);
    let dst = if from < to { to - 1 } else { to };
    app.batch.items.insert(dst, item);
    // Adjust selected index to follow the moved item or stay in place.
    if app.batch.selected_index == from {
        app.batch.selected_index = dst;
    } else if from < app.batch.selected_index && app.batch.selected_index <= to {
        app.batch.selected_index -= 1;
    } else if to <= app.batch.selected_index && app.batch.selected_index < from {
        app.batch.selected_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item::ImageSource;
    use crate::gui::item_settings::ItemSettings;
    use std::sync::Arc;

    fn fresh_item() -> BatchItem {
        BatchItem::new(
            0,
            "t.png".into(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (10, 10),
            ItemSettings::default(),
            String::new(),
        )
    }

    #[test]
    fn should_request_thumbnail_when_absent_and_not_pending() {
        let item = fresh_item();
        assert!(should_request_thumbnail(&item));
    }

    #[test]
    fn should_not_request_when_already_pending() {
        let mut item = fresh_item();
        item.thumb_pending = true;
        assert!(!should_request_thumbnail(&item));
    }
}
