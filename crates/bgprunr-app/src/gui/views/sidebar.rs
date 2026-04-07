use egui::{Color32, Pos2, Rect, RichText, Stroke, Vec2};
use crate::gui::app::{BgPrunrApp, BatchStatus};
use crate::gui::theme;

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

        // Thumbnail list with drag-to-reorder
        let mut swap_from: Option<usize> = None;
        let mut swap_to: Option<usize> = None;
        let mut decoded_this_frame = false; // limit to 1 thumbnail decode per frame
        let item_width = theme::SIDEBAR_WIDTH - theme::SPACE_SM * 2.0;
        let item_height = theme::THUMBNAIL_SIZE + theme::SPACE_SM;

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

                // Lazily create thumbnail (max 1 decode per frame to stay responsive)
                // Use result image if available, otherwise source
                if app.batch_items[i].thumb_texture.is_none() && !decoded_this_frame {
                    let rgba = if let Some(ref result) = app.batch_items[i].result_rgba {
                        decoded_this_frame = true;
                        Some(image::imageops::thumbnail(result, 80, 80))
                    } else if let Ok(img) = image::load_from_memory(&app.batch_items[i].source_bytes) {
                        decoded_this_frame = true;
                        Some(image::imageops::thumbnail(&img.to_rgba8(), 80, 80))
                    } else {
                        None
                    };
                    if let Some(thumb) = rgba {
                        let (tw, th) = (thumb.width(), thumb.height());
                        let ci = egui::ColorImage::from_rgba_unmultiplied(
                            [tw as usize, th as usize],
                            thumb.as_flat_samples().as_slice(),
                        );
                        let id = app.batch_items[i].id;
                        app.batch_items[i].thumb_texture = Some(
                            ui.ctx().load_texture(format!("thumb_{id}"), ci, egui::TextureOptions::LINEAR),
                        );
                    }
                }

                // Draw thumbnail if available (preserve aspect ratio)
                if let Some(ref thumb_tex) = app.batch_items[i].thumb_texture {
                    let tex_size = thumb_tex.size_vec2();
                    let scale = (theme::THUMBNAIL_SIZE / tex_size.x)
                        .min(theme::THUMBNAIL_SIZE / tex_size.y);
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

                // Status icon overlay in bottom-right corner
                let icon_pos = Pos2::new(
                    item_rect.max.x - theme::SPACE_XS - 10.0,
                    item_rect.max.y - theme::SPACE_XS - 12.0,
                );
                let (icon_text, icon_color) = match &app.batch_items[i].status {
                    BatchStatus::Pending => ("\u{25CB}", theme::STATUS_ICON_PENDING),      // ○
                    BatchStatus::Processing => ("\u{25C6}", theme::STATUS_ICON_PROCESSING), // ◆
                    BatchStatus::Done => ("\u{2713}", theme::STATUS_ICON_DONE),             // ✓
                    BatchStatus::Error(_) => ("\u{2717}", theme::DESTRUCTIVE),               // ✗
                };
                ui.painter().text(
                    icon_pos, egui::Align2::LEFT_TOP,
                    icon_text,
                    egui::FontId::monospace(theme::FONT_SIZE_MONO),
                    icon_color,
                );

                // DnD: set drag payload
                item_response.dnd_set_drag_payload(i);

                // DnD: check if something dropped here
                if let Some(src_idx) = item_response.dnd_release_payload::<usize>() {
                    swap_from = Some(*src_idx);
                    swap_to = Some(i);
                }

                // Click to select — sync immediately (no frame delay)
                if item_response.clicked() && app.selected_batch_index != i {
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

        // If we decoded a thumbnail this frame, request repaint for the next ones
        if decoded_this_frame && app.batch_items.iter().any(|i| i.thumb_texture.is_none()) {
            ui.ctx().request_repaint();
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
