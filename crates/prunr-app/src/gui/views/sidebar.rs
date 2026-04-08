use egui::{Color32, Pos2, Rect, RichText, Stroke, Vec2};
use egui_material_icons::icons::*;
use crate::gui::app::{PrunrApp, BatchStatus};
use crate::gui::theme;

/// Manual pointer hit test for a rect — returns (hovered, clicked).
fn hit_test(ui: &egui::Ui, rect: Rect) -> (bool, bool) {
    ui.ctx().input(|i| {
        let hover = i.pointer.hover_pos().map_or(false, |p| rect.contains(p));
        (hover, hover && i.pointer.primary_clicked())
    })
}

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
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

        let all_selected = !app.batch_items.is_empty() && app.batch_items.iter().all(|i| i.selected);
        let mut select_all = all_selected;
        if ui.checkbox(&mut select_all, RichText::new("Select All").size(theme::FONT_SIZE_BODY).color(theme::TEXT_PRIMARY)).changed() {
            for item in &mut app.batch_items {
                item.selected = select_all;
            }
        }
        ui.add_space(theme::SPACE_XS);

        // Thumbnail list with drag-to-reorder
        let mut swap_from: Option<usize> = None;
        let mut swap_to: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let mut save_idx: Option<usize> = None;
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

                // Loading spinner while thumbnail is being generated
                if app.batch_items[i].thumb_texture.is_none() && app.batch_items[i].thumb_pending {
                    let spinner_rect = Rect::from_center_size(item_rect.center(), Vec2::splat(20.0));
                    ui.put(spinner_rect, egui::Spinner::new().size(20.0).color(theme::ACCENT));
                    needs_repaint = true;
                }

                // Draw thumbnail with fade-in
                let has_thumb = app.batch_items[i].thumb_texture.is_some();
                let fade = ui.ctx().animate_bool_with_time(
                    egui::Id::new(("thumb_fade", app.batch_items[i].id)),
                    has_thumb,
                    0.2,
                );
                if let Some(ref thumb_tex) = app.batch_items[i].thumb_texture {
                    let tex_size = thumb_tex.size_vec2();
                    let scale = (item_width / tex_size.x)
                        .min(item_height / tex_size.y)
                        .min(1.0);
                    let fitted = tex_size * scale;
                    let thumb_rect = Rect::from_center_size(
                        item_rect.center(),
                        fitted,
                    );
                    let alpha = (fade * 255.0) as u8;
                    ui.painter().image(
                        thumb_tex.id(), thumb_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
                    );
                    if fade < 1.0 { needs_repaint = true; }
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
                        Stroke::new(1.0, theme::TEXT_SECONDARY),
                        egui::StrokeKind::Outside);
                    if app.batch_items[i].selected {
                        ui.painter().text(cb_center, egui::Align2::CENTER_CENTER,
                            ICON_CHECK.codepoint, egui::FontId::proportional(12.0), Color32::WHITE);
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
                            Color32::from_rgba_unmultiplied(0x7b, 0x2d, 0x8e, 50));
                    }
                    // Pulsing blue border
                    let pulse = (anim_time * 2.5).sin() * 0.5 + 0.5;
                    let border_alpha = (40.0 + pulse * 80.0) as u8;
                    ui.painter().rect_stroke(
                        item_rect, theme::THUMBNAIL_ROUNDING,
                        Stroke::new(2.0, Color32::from_rgba_unmultiplied(0x7b, 0x2d, 0x8e, border_alpha)),
                        egui::StrokeKind::Inside,
                    );
                    needs_repaint = true;
                }

                // Status indicator — bottom-right corner, minimal design
                match &app.batch_items[i].status {
                    BatchStatus::Pending => {
                        // Small gray dot
                        let dot = Pos2::new(item_rect.max.x - 8.0, item_rect.max.y - 8.0);
                        ui.painter().circle_filled(dot, 3.0, theme::STATUS_ICON_PENDING);
                    }
                    BatchStatus::Processing => {
                        // Pulsing purple dot
                        let dot = Pos2::new(item_rect.max.x - 8.0, item_rect.max.y - 8.0);
                        let pulse = (anim_time * 3.0).sin() * 0.5 + 0.5;
                        let size = 3.0 + pulse * 2.0;
                        ui.painter().circle_filled(dot, size, theme::ACCENT);
                        needs_repaint = true;
                    }
                    BatchStatus::Done => {
                        // Green checkmark icon (no circle background)
                        let pos = Pos2::new(item_rect.max.x - 10.0, item_rect.max.y - 10.0);
                        ui.painter().text(pos, egui::Align2::CENTER_CENTER,
                            ICON_CHECK.codepoint, egui::FontId::proportional(14.0),
                            theme::ACCENT_GREEN);
                    }
                    BatchStatus::Error(_) => {
                        // Red X icon
                        let pos = Pos2::new(item_rect.max.x - 10.0, item_rect.max.y - 10.0);
                        ui.painter().text(pos, egui::Align2::CENTER_CENTER,
                            ICON_ERROR.codepoint, egui::FontId::proportional(14.0),
                            theme::DESTRUCTIVE);
                    }
                }

                // Hover action buttons (delete top-right, save bottom-left)
                let mut close_clicked = false;
                if item_response.hovered() && ui.ctx().dragged_id().is_none() {
                    let btn_size = 20.0;

                    // Delete button — top-right (trash icon)
                    let del_center = Pos2::new(
                        item_rect.max.x - 4.0 - btn_size * 0.5,
                        item_rect.min.y + 4.0 + btn_size * 0.5,
                    );
                    let del_rect = Rect::from_center_size(del_center, Vec2::splat(btn_size));
                    let (del_hover, del_press) = hit_test(ui, del_rect);
                    let del_bg = if del_hover {
                        theme::DESTRUCTIVE
                    } else {
                        Color32::from_rgba_unmultiplied(0, 0, 0, 200)
                    };
                    ui.painter().circle_filled(del_center, btn_size * 0.5, del_bg);
                    ui.painter().text(del_center, egui::Align2::CENTER_CENTER,
                        ICON_DELETE.codepoint, egui::FontId::proportional(12.0), Color32::WHITE);
                    if del_press {
                        remove_idx = Some(i);
                        close_clicked = true;
                    }

                    // Save button — bottom-left (only for Done items)
                    if matches!(app.batch_items[i].status, BatchStatus::Done) {
                        let save_center = Pos2::new(
                            item_rect.min.x + 4.0 + btn_size * 0.5,
                            item_rect.max.y - 4.0 - btn_size * 0.5,
                        );
                        let save_rect = Rect::from_center_size(save_center, Vec2::splat(btn_size));
                        let (save_hover, save_press) = hit_test(ui, save_rect);
                        let save_bg = if save_hover {
                            theme::ACCENT
                        } else {
                            Color32::from_rgba_unmultiplied(0, 0, 0, 200)
                        };
                        ui.painter().circle_filled(save_center, btn_size * 0.5, save_bg);
                        ui.painter().text(save_center, egui::Align2::CENTER_CENTER,
                            ICON_SAVE.codepoint, egui::FontId::proportional(12.0), Color32::WHITE);
                        if save_press {
                            save_idx = Some(i);
                            close_clicked = true;
                        }
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
                    app.canvas_switch_id += 1; // trigger canvas fade-in
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

        // Apply save single item
        if let Some(idx) = save_idx {
            if idx < app.batch_items.len() {
                if let Some(ref rgba) = app.batch_items[idx].result_rgba {
                    let stem = std::path::Path::new(&app.batch_items[idx].filename)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("image");
                    let default_name = format!("{stem}-nobg.png");
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("PNG Image", &["png"])
                        .set_file_name(&default_name)
                        .set_title("Save PNG")
                        .save_file()
                    {
                        let rgba = rgba.clone();
                        let tx = app.save_done_tx.clone();
                        app.toasts.info("Saving...");
                        std::thread::spawn(move || {
                            let msg = match prunr_core::encode_rgba_png(&rgba) {
                                Ok(png_bytes) => match std::fs::write(&path, &png_bytes) {
                                    Ok(()) => "Saved".into(),
                                    Err(e) => format!("Save failed: {e}"),
                                },
                                Err(e) => format!("Save failed: {e}"),
                            };
                            let _ = tx.send(msg);
                        });
                    }
                }
            }
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
