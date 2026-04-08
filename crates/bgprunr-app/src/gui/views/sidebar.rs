use egui::{Color32, Pos2, Rect, RichText, Stroke, Vec2};
use crate::gui::app::{BgPrunrApp, BatchStatus};
use crate::gui::theme;

/// Manual pointer hit test for a rect — returns (hovered, clicked).
fn hit_test(ui: &egui::Ui, rect: Rect) -> (bool, bool) {
    ui.ctx().input(|i| {
        let hover = i.pointer.hover_pos().map_or(false, |p| rect.contains(p));
        (hover, hover && i.pointer.primary_clicked())
    })
}

pub fn render(ui: &mut egui::Ui, app: &mut BgPrunrApp) {
    ui.vertical(|ui| {
        if app.batch_items.is_empty() {
            // Empty state — centered
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
            return;
        }

        // Select All / Clear controls
        let any_selected = app.batch_items.iter().any(|i| i.selected);
        let all_selected = !app.batch_items.is_empty() && app.batch_items.iter().all(|i| i.selected);
        ui.horizontal(|ui| {
            let mut select_all = all_selected;
            if ui.checkbox(&mut select_all, RichText::new("All").size(theme::FONT_SIZE_MONO).color(theme::TEXT_SECONDARY)).changed() {
                for item in &mut app.batch_items {
                    item.selected = select_all;
                }
            }
            if any_selected {
                if ui.small_button("Clear").clicked() {
                    for item in &mut app.batch_items {
                        item.selected = false;
                    }
                }
            }
        });
        ui.add_space(theme::SPACE_XS);

        // Thumbnail list with drag-to-reorder
        let mut swap_from: Option<usize> = None;
        let mut swap_to: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let item_width = ui.available_width() - theme::SPACE_SM;
        let item_height = theme::THUMBNAIL_SIZE + theme::SPACE_SM;

        // Pick up completed thumbnails from background threads
        while let Ok((item_id, tw, th, pixels)) = app.thumb_rx.try_recv() {
            if let Some(item) = app.batch_items.iter_mut().find(|b| b.id == item_id) {
                let ci = egui::ColorImage::from_rgba_unmultiplied(
                    [tw as usize, th as usize], &pixels,
                );
                item.thumb_texture = Some(
                    ui.ctx().load_texture(format!("thumb_{item_id}"), ci, egui::TextureOptions::LINEAR),
                );
            }
        }

        let anim_time = ui.ctx().input(|i| i.time) as f32;
        let mut needs_repaint = false;

        egui::ScrollArea::vertical().show(ui, |ui| {
            for i in 0..app.batch_items.len() {
                let is_selected = i == app.selected_batch_index;

                // Allocate space for item
                let (item_rect, item_response) = ui.allocate_exact_size(
                    Vec2::new(item_width, item_height),
                    egui::Sense::click_and_drag(),
                );

                // Background
                let bg_color = if is_selected {
                    theme::SIDEBAR_ITEM_SELECTED
                } else {
                    theme::SIDEBAR_ITEM_BG
                };
                ui.painter().rect_filled(item_rect, theme::THUMBNAIL_ROUNDING, bg_color);

                // Selected border
                if is_selected {
                    ui.painter().rect_stroke(
                        item_rect, theme::THUMBNAIL_ROUNDING,
                        Stroke::new(2.0, theme::SIDEBAR_SELECTED_BORDER),
                        egui::StrokeKind::Inside,
                    );
                }

                // Request thumbnail generation on background thread if needed
                if app.batch_items[i].thumb_texture.is_none()
                    && !app.batch_items[i].thumb_pending
                {
                    app.batch_items[i].thumb_pending = true;
                    let item_id = app.batch_items[i].id;
                    app.request_thumbnail(
                        item_id,
                        &app.batch_items[i].source_bytes,
                        app.batch_items[i].result_rgba.as_ref(),
                    );
                    needs_repaint = true;
                }

                // Draw thumbnail if available (preserve aspect ratio)
                if let Some(ref thumb_tex) = app.batch_items[i].thumb_texture {
                    let tex_size = thumb_tex.size_vec2();
                    let scale = (item_width / tex_size.x)
                        .min(item_height / tex_size.y);
                    let fitted = tex_size * scale;
                    let thumb_rect = Rect::from_center_size(
                        item_rect.center(),
                        fitted,
                    );
                    ui.painter().image(
                        thumb_tex.id(), thumb_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }

                // Selection checkbox — top-left corner
                {
                    let cb_size = 16.0;
                    let cb_center = Pos2::new(
                        item_rect.min.x + 4.0 + cb_size * 0.5,
                        item_rect.min.y + 4.0 + cb_size * 0.5,
                    );
                    let cb_rect = Rect::from_center_size(cb_center, Vec2::splat(cb_size));

                    let (cb_hovered, cb_clicked) = hit_test(ui, cb_rect);
                    if cb_clicked {
                        app.batch_items[i].selected = !app.batch_items[i].selected;
                    }

                    // Draw checkbox background
                    let cb_bg = if app.batch_items[i].selected {
                        theme::ACCENT
                    } else if cb_hovered {
                        Color32::from_rgb(0x50, 0x50, 0x50)
                    } else {
                        Color32::from_rgba_unmultiplied(0, 0, 0, 160)
                    };
                    ui.painter().rect_filled(cb_rect, 3.0, cb_bg);
                    ui.painter().rect_stroke(cb_rect, 3.0,
                        Stroke::new(1.0, Color32::from_rgb(0x70, 0x70, 0x70)),
                        egui::StrokeKind::Outside);
                    if app.batch_items[i].selected {
                        ui.painter().text(cb_center, egui::Align2::CENTER_CENTER,
                            "\u{2713}", egui::FontId::proportional(11.0), Color32::WHITE);
                    }
                }

                // Processing animation on thumbnail: shimmer sweep + pulsing blue border
                if matches!(app.batch_items[i].status, BatchStatus::Processing) {
                    // Shimmer sweep
                    let sweep = ((anim_time * 0.7).fract()) * (item_rect.width() + 40.0) - 20.0 + item_rect.min.x;
                    let shimmer = Rect::from_min_max(
                        Pos2::new(sweep - 20.0, item_rect.min.y),
                        Pos2::new(sweep + 20.0, item_rect.max.y),
                    ).intersect(item_rect);
                    if shimmer.width() > 0.0 {
                        ui.painter().rect_filled(shimmer, theme::THUMBNAIL_ROUNDING,
                            Color32::from_rgba_unmultiplied(0x3b, 0x82, 0xf6, 50));
                    }
                    // Pulsing blue border
                    let pulse = (anim_time * 2.5).sin() * 0.5 + 0.5;
                    let border_alpha = (40.0 + pulse * 80.0) as u8;
                    ui.painter().rect_stroke(
                        item_rect, theme::THUMBNAIL_ROUNDING,
                        Stroke::new(2.0, Color32::from_rgba_unmultiplied(0x3b, 0x82, 0xf6, border_alpha)),
                        egui::StrokeKind::Inside,
                    );
                    needs_repaint = true;
                }

                // Status badge — colored dot with pill background in bottom-right
                {
                    let badge_radius = 5.0;
                    let badge_center = Pos2::new(
                        item_rect.max.x - theme::SPACE_SM - badge_radius,
                        item_rect.max.y - theme::SPACE_SM - badge_radius,
                    );
                    let (dot_color, show_ring) = match &app.batch_items[i].status {
                        BatchStatus::Pending => (theme::STATUS_ICON_PENDING, false),
                        BatchStatus::Processing => (theme::STATUS_ICON_PROCESSING, true),
                        BatchStatus::Done => (theme::STATUS_ICON_DONE, false),
                        BatchStatus::Error(_) => (theme::DESTRUCTIVE, false),
                    };
                    ui.painter().circle_filled(badge_center, badge_radius + 3.0,
                        Color32::from_rgba_unmultiplied(0, 0, 0, 180));
                    ui.painter().circle_filled(badge_center, badge_radius, dot_color);
                    if show_ring {
                        let pulse = (anim_time * 3.0).sin() * 0.5 + 0.5;
                        let ring_alpha = (pulse * 180.0) as u8;
                        let ring_color = Color32::from_rgba_unmultiplied(0x3b, 0x82, 0xf6, ring_alpha);
                        ui.painter().circle_stroke(badge_center, badge_radius + 3.0,
                            Stroke::new(1.5, ring_color));
                        needs_repaint = true;
                    }
                    if matches!(app.batch_items[i].status, BatchStatus::Done) {
                        ui.painter().text(badge_center, egui::Align2::CENTER_CENTER,
                            "\u{2713}", egui::FontId::proportional(9.0), Color32::WHITE);
                    }
                    if matches!(app.batch_items[i].status, BatchStatus::Error(_)) {
                        ui.painter().text(badge_center, egui::Align2::CENTER_CENTER,
                            "\u{2715}", egui::FontId::proportional(9.0), Color32::WHITE);
                    }
                }

                // Remove button — top-right, shown on hover (manual hit test)
                let mut close_clicked = false;
                if item_response.hovered() && ui.ctx().dragged_id().is_none() {
                    let btn_size = 18.0;
                    let btn_center = Pos2::new(
                        item_rect.max.x - 4.0 - btn_size * 0.5,
                        item_rect.min.y + 4.0 + btn_size * 0.5,
                    );
                    let btn_rect = Rect::from_center_size(btn_center, Vec2::splat(btn_size));

                    let (btn_hovered, btn_pressed) = hit_test(ui, btn_rect);

                    let bg = if btn_hovered {
                        Color32::from_rgb(0xdc, 0x26, 0x26)
                    } else {
                        Color32::from_rgba_unmultiplied(0, 0, 0, 200)
                    };
                    ui.painter().circle_filled(btn_center, btn_size * 0.5, bg);
                    ui.painter().text(btn_center, egui::Align2::CENTER_CENTER,
                        "\u{2715}", egui::FontId::proportional(10.0), Color32::WHITE);

                    if btn_pressed {
                        remove_idx = Some(i);
                        close_clicked = true;
                    }
                }

                // DnD: set drag payload
                item_response.dnd_set_drag_payload(i);

                // DnD: check if something dropped here
                if let Some(src_idx) = item_response.dnd_release_payload::<usize>() {
                    swap_from = Some(*src_idx);
                    swap_to = Some(i);
                }

                // Click to select (skip if close button was clicked)
                if !close_clicked && item_response.clicked() && app.selected_batch_index != i {
                    app.selected_batch_index = i;
                    let ctx = ui.ctx().clone();
                    app.sync_selected_batch_textures(&ctx);
                }

                // Hover insertion line for DnD
                if ui.ctx().is_being_dragged(item_response.id) {
                    // Item is being dragged — show ghost
                } else if item_response.hovered() && ui.ctx().dragged_id().is_some() {
                    // Something being dragged over this item — show insertion line
                    let line_y = item_rect.min.y;
                    ui.painter().hline(
                        item_rect.x_range(), line_y,
                        Stroke::new(2.0, theme::INSERTION_LINE),
                    );
                }

                ui.add_space(theme::SPACE_XS); // gap between items
            }
        });

        // Request repaint if animations are running or thumbnails still pending
        if needs_repaint || app.batch_items.iter().any(|i| i.thumb_pending) {
            ui.ctx().request_repaint();
        }

        // Apply remove
        if let Some(idx) = remove_idx {
            app.remove_batch_item(idx);
        }

        // Apply reorder after iteration
        if let (Some(from), Some(to)) = (swap_from, swap_to) {
            if from != to {
                let item = app.batch_items.remove(from);
                let dst = if from < to { to - 1 } else { to };
                app.batch_items.insert(dst, item);
                // Adjust selected index
                if app.selected_batch_index == from {
                    app.selected_batch_index = dst;
                } else if from < app.selected_batch_index && app.selected_batch_index <= to {
                    app.selected_batch_index -= 1;
                } else if to <= app.selected_batch_index && app.selected_batch_index < from {
                    app.selected_batch_index += 1;
                }
            }
        }
    });
}
