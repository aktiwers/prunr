use std::sync::LazyLock;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

use crate::gui::app::BgPrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;
use super::animation;

static IS_WAYLAND: LazyLock<bool> = LazyLock::new(|| std::env::var_os("WAYLAND_DISPLAY").is_some());

pub fn render(ui: &mut egui::Ui, app: &mut BgPrunrApp) {
    // Set background
    let avail_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(avail_rect, 0.0, theme::BG_PRIMARY);

    let canvas_rect = ui.available_rect_before_wrap();

    // Handle scroll-wheel zoom (cursor-centered)
    ui.ctx().input(|i| {
        for event in &i.events {
            if let egui::Event::MouseWheel { delta, modifiers, .. } = event {
                if !modifiers.any() {
                    let scroll_y = delta.y;
                    let zoom_delta = theme::ZOOM_STEP.powf(scroll_y);
                    let new_zoom = (app.zoom * zoom_delta).clamp(theme::ZOOM_MIN, theme::ZOOM_MAX);
                    if let Some(cursor) = i.pointer.hover_pos() {
                        if canvas_rect.contains(cursor) {
                            let cursor_rel = cursor - canvas_rect.center();
                            app.pan_offset =
                                cursor_rel / app.zoom - cursor_rel / new_zoom + app.pan_offset;
                            app.zoom = new_zoom;
                        }
                    }
                }
            }
        }
        // Space+drag pan
        let space_held = i.keys_down.contains(&egui::Key::Space);
        let dragging = i.pointer.primary_down();
        if space_held && dragging {
            app.pan_offset += i.pointer.delta();
            app.is_panning = true;
        } else {
            app.is_panning = false;
        }
    });

    // Handle pending Ctrl+0 (fit to window) / Ctrl+1 (actual size)
    if let Some(ref tex) = app.source_texture {
        let tex_size = tex.size_vec2();
        let canvas_size = canvas_rect.size();

        if app.pending_fit_zoom {
            app.pending_fit_zoom = false;
            let fit = fit_zoom(canvas_size, tex_size);
            // Only toggle back if zoom is already at fit (keyboard shortcut).
            // previous_zoom == 1.0 means this is a fresh image switch — always fit.
            if (app.zoom - fit).abs() < 0.001 && app.previous_zoom != 1.0 {
                app.zoom = app.previous_zoom;
            } else {
                app.previous_zoom = app.zoom;
                app.zoom = fit;
                app.pan_offset = Vec2::ZERO;
            }
        }
        if app.pending_actual_size {
            app.pending_actual_size = false;
            if (app.zoom - 1.0).abs() < 0.001 {
                app.zoom = app.previous_zoom;
            } else {
                app.previous_zoom = app.zoom;
                app.zoom = 1.0;
                app.pan_offset = Vec2::ZERO;
            }
        }
    }

    match app.state {
        AppState::Empty => render_empty(ui, app),
        AppState::Loaded => render_loaded(ui, app),
        AppState::Processing => render_processing(ui, app),
        AppState::Animating => render_animating(ui, app),
        AppState::Done => render_done(ui, app),
    }
}

/// Compute the image rectangle given canvas bounds, texture size, zoom, and pan offset.
fn compute_img_rect(canvas_rect: Rect, tex_size: Vec2, zoom: f32, pan: Vec2) -> Rect {
    let img_size = tex_size * zoom;
    let center = canvas_rect.center() + pan;
    Rect::from_center_size(center, img_size)
}

/// Compute fit-to-window zoom (never upscale beyond 1:1).
fn fit_zoom(canvas_size: Vec2, tex_size: Vec2) -> f32 {
    (canvas_size.x / tex_size.x)
        .min(canvas_size.y / tex_size.y)
        .min(1.0)
}

fn render_empty(ui: &mut egui::Ui, _app: &BgPrunrApp) {
    let avail = ui.available_size();
    let is_hovered = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

    // Centered drop zone
    let zone_w = (avail.x * 0.5).min(400.0).max(200.0);
    let zone_h = 200.0_f32;
    let center = ui.available_rect_before_wrap().center();
    let zone_rect = Rect::from_center_size(center, Vec2::new(zone_w, zone_h));

    // Draw drop zone border
    let border_color = if is_hovered {
        theme::DROP_HOVER_BORDER
    } else {
        theme::DROP_BORDER
    };
    ui.painter().rect_stroke(
        zone_rect,
        theme::DROP_ZONE_ROUNDING,
        Stroke::new(theme::DROP_ZONE_BORDER_WIDTH, border_color),
        egui::StrokeKind::Outside,
    );

    let painter = ui.painter();

    let heading = "Drop an image here";
    painter.text(
        center - Vec2::new(0.0, 20.0),
        egui::Align2::CENTER_CENTER,
        heading,
        egui::FontId::proportional(theme::FONT_SIZE_HEADING),
        theme::TEXT_PRIMARY,
    );

    let hint = if cfg!(target_os = "macos") {
        "or press Cmd+O to open a file"
    } else {
        "or press Ctrl+O to open a file"
    };
    painter.text(
        center + Vec2::new(0.0, 20.0),
        egui::Align2::CENTER_CENTER,
        hint,
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_SECONDARY,
    );

    // Wayland DnD caveat
    if *IS_WAYLAND {
        painter.text(
            center + Vec2::new(0.0, 55.0),
            egui::Align2::CENTER_CENTER,
            "(Drag and drop not supported in Wayland yet)",
            egui::FontId::proportional(theme::FONT_SIZE_BODY * 0.85),
            theme::TEXT_SECONDARY,
        );
    }
}

fn render_loaded(ui: &mut egui::Ui, app: &BgPrunrApp) {
    if let Some(ref texture) = app.source_texture {
        let canvas_rect = ui.available_rect_before_wrap();
        let img_rect = compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom, app.pan_offset);
        ui.painter().image(
            texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
}

fn render_processing(ui: &mut egui::Ui, app: &BgPrunrApp) {
    if let Some(ref texture) = app.source_texture {
        let canvas_rect = ui.available_rect_before_wrap();
        let img_rect = compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom, app.pan_offset);
        ui.painter().image(
            texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, 128),
        );
    }

    // Center overlay text
    let center = ui.available_rect_before_wrap().center();
    ui.painter().text(
        center + egui::Vec2::new(0.0, 30.0),
        egui::Align2::CENTER_CENTER,
        "Press Escape to cancel",
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_PRIMARY,
    );
}

fn render_done(ui: &mut egui::Ui, app: &BgPrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();
    if app.show_original {
        // Show original image without checkerboard
        if let Some(ref texture) = app.source_texture {
            let img_rect =
                compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom, app.pan_offset);
            ui.painter().image(
                texture.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    } else {
        // Show result with checkerboard
        if let Some(ref texture) = app.result_texture {
            let img_rect =
                compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom, app.pan_offset);
            draw_checkerboard(ui, img_rect);
            ui.painter().image(
                texture.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }

    // Before/after indicator label in top-left of canvas
    let label = if app.show_original { "Original" } else { "Result" };
    ui.painter().text(
        canvas_rect.min + Vec2::new(theme::SPACE_SM, theme::SPACE_SM),
        egui::Align2::LEFT_TOP,
        label,
        egui::FontId::monospace(theme::FONT_SIZE_MONO),
        theme::TEXT_SECONDARY,
    );
}

fn render_animating(ui: &mut egui::Ui, app: &mut BgPrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();

    // Populate cache if empty (decoded once, reused every frame)
    if app.source_rgba_cache.is_none() {
        if let Some(ref bytes) = app.source_bytes {
            app.source_rgba_cache = image::load_from_memory(bytes)
                .ok()
                .map(|img| img.to_rgba8());
        }
    }

    if let (Some(ref source), Some(ref result), Some(ref mask)) =
        (&app.source_rgba_cache, &app.result_rgba, &app.anim_mask)
    {
        // Cap animation texture to canvas size for performance
        let max_w = canvas_rect.width() as u32;
        let max_h = canvas_rect.height() as u32;

        let frame = animation::build_animation_frame(
            &source, result, mask, app.anim_progress, max_w, max_h,
        );

        // Compute image rect with fit zoom (ignore user zoom to keep it simple during animation)
        let tex_size = egui::Vec2::new(frame.size[0] as f32, frame.size[1] as f32);
        let fit = fit_zoom(canvas_rect.size(), tex_size);
        let img_rect = Rect::from_center_size(canvas_rect.center(), tex_size * fit);

        // Draw checkerboard behind (visible through fading background)
        draw_checkerboard(ui, img_rect);

        // Upload animation frame as texture
        let anim_texture = ui.ctx().load_texture(
            "anim_frame",
            frame,
            egui::TextureOptions::LINEAR,
        );
        ui.painter().image(
            anim_texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        // Fallback: just show result
        render_done(ui, app);
    }
}

fn draw_checkerboard(ui: &egui::Ui, bounds: Rect) {
    let checker_size = theme::CHECKER_SIZE;
    let painter = ui.painter();

    let start_x = bounds.min.x;
    let start_y = bounds.min.y;
    let end_x = bounds.max.x;
    let end_y = bounds.max.y;

    let mut y = start_y;
    let mut row = 0usize;
    while y < end_y {
        let mut x = start_x;
        let mut col = 0usize;
        while x < end_x {
            let color = if (row + col) % 2 == 0 {
                theme::CHECKER_LIGHT
            } else {
                theme::CHECKER_DARK
            };
            let rect = Rect::from_min_max(
                Pos2::new(x, y),
                Pos2::new((x + checker_size).min(end_x), (y + checker_size).min(end_y)),
            );
            painter.rect_filled(rect, 0.0, color);
            x += checker_size;
            col += 1;
        }
        y += checker_size;
        row += 1;
    }
}
