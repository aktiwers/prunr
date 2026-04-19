use std::sync::{LazyLock, OnceLock};
use egui::{Color32, Pos2, Rect, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::gui::app::PrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;

/// Number of checker squares per texture edge (16×16 grid).
const CHECKER_TEX_TILES: usize = 16;

static IS_WAYLAND: LazyLock<bool> = LazyLock::new(|| std::env::var_os("WAYLAND_DISPLAY").is_some());

pub fn render(ui: &mut egui::Ui, app: &mut PrunrApp) {
    // Set background
    let avail_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(avail_rect, 0.0, theme::BG_PRIMARY);

    let canvas_rect = ui.available_rect_before_wrap();

    let modal_open = app.any_modal_open();
    // Block canvas pan/zoom when an egui popup (chip popover, combo box,
    // dropdown) is open — otherwise the press that lands on a slider inside
    // the popover would also start panning the canvas, and slider drag would
    // drag both the slider AND the image.
    #[allow(deprecated)]
    let popup_open = ui.ctx().memory(|m| m.any_popup_open());
    // Also check egui's global "wants pointer input" — this is true when any
    // widget (slider, button, text field) is currently capturing the pointer.
    let widget_has_pointer = ui.ctx().egui_wants_pointer_input();
    let canvas_gets_input = !modal_open && !popup_open && !widget_has_pointer;
    if canvas_gets_input {
        // Handle scroll-wheel zoom (cursor-centered)
        ui.ctx().input(|i| {
            for event in &i.events {
                if let egui::Event::MouseWheel { delta, modifiers, .. } = event {
                    if !modifiers.any() {
                        let scroll_y = delta.y;
                        let zoom_delta = theme::ZOOM_STEP.powf(scroll_y);
                        let new_zoom = (app.zoom_state.zoom * zoom_delta).clamp(theme::ZOOM_MIN, theme::ZOOM_MAX);
                        if let Some(cursor) = i.pointer.hover_pos() {
                            if canvas_rect.contains(cursor) {
                                let cursor_rel = cursor - canvas_rect.center();
                                app.zoom_state.pan_offset =
                                    cursor_rel / app.zoom_state.zoom - cursor_rel / new_zoom + app.zoom_state.pan_offset;
                                app.zoom_state.zoom = new_zoom;
                            }
                        }
                    }
                }
            }
            // Click+drag pan: enter panning ONLY on a fresh press inside the canvas.
            // This avoids false pan when egui's pointer state is desynced — e.g. after
            // an OS drag-out session where Prunr's window never saw the mouse-up.
            let hovered_inside = i.pointer.hover_pos().is_some_and(|p| canvas_rect.contains(p));
            if i.pointer.primary_pressed() && hovered_inside {
                app.zoom_state.is_panning = true;
            }
            if !i.pointer.primary_down() {
                app.zoom_state.is_panning = false;
            }
            if app.zoom_state.is_panning && i.pointer.delta() != egui::Vec2::ZERO {
                app.zoom_state.pan_offset += i.pointer.delta();
            }
        });
    } else {
        // Cancel any ongoing pan when a widget / popup takes over. Prevents
        // stuck "is_panning=true" state after clicking from canvas onto a chip.
        app.zoom_state.is_panning = false;
    }

    // Handle pending Ctrl+0 (fit to window) / Ctrl+1 (actual size).
    // Only consume the flag when the texture is ready — with lazy decode,
    // the texture may not exist on the first frame after opening.
    if let Some(ref tex) = app.source_texture {
        let tex_size = tex.size_vec2();
        let canvas_size = canvas_rect.size();

        if app.zoom_state.pending_fit_zoom {
            app.zoom_state.pending_fit_zoom = false;
            let fit = fit_zoom(canvas_size, tex_size);
            // Only toggle back if zoom is already at fit (keyboard shortcut).
            // previous_zoom == 1.0 means this is a fresh image switch — always fit.
            if (app.zoom_state.zoom - fit).abs() < 0.001 && app.zoom_state.previous_zoom != 1.0 {
                app.zoom_state.zoom = app.zoom_state.previous_zoom;
            } else {
                app.zoom_state.previous_zoom = app.zoom_state.zoom;
                app.zoom_state.zoom = fit;
                app.zoom_state.pan_offset = Vec2::ZERO;
            }
        }
        if app.zoom_state.pending_actual_size {
            app.zoom_state.pending_actual_size = false;
            if (app.zoom_state.zoom - 1.0).abs() < 0.001 {
                app.zoom_state.zoom = app.zoom_state.previous_zoom;
            } else {
                app.zoom_state.previous_zoom = app.zoom_state.zoom;
                app.zoom_state.zoom = 1.0;
                app.zoom_state.pan_offset = Vec2::ZERO;
            }
        }
    }

    match app.state {
        AppState::Empty => render_empty(ui, app),
        AppState::Loaded => render_loaded(ui, app),
        AppState::Processing => render_processing(ui, app),
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

const TIP_CYCLE_SECS: f64 = 5.0;
const TIP_FADE_SECS: f64 = 0.5;

static TIPS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let m = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };
    vec![
        "Press F1 to view keyboard shortcuts".to_string(),
        "Press F2 to view CLI usage examples".to_string(),
        format!("Press {m}+R to remove the background"),
        "Press B to toggle before/after comparison".to_string(),
        "Use Arrow keys or A/D to navigate between images".to_string(),
        format!("Press {m}+0 to fit image to window"),
        "Scroll to zoom, drag to pan".to_string(),
        "Press Tab to show/hide the image queue".to_string(),
        format!("Press {m}+Space to open settings"),
        format!("Press {m}+S to save the result"),
        format!("Press {m}+C to copy result to clipboard"),
        format!("Press {m}+Z to undo background removal"),
        "Open multiple images for batch processing".to_string(),
    ]
});

fn render_empty(ui: &mut egui::Ui, _app: &PrunrApp) {
    let avail = ui.available_size();
    let is_hovered = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
    let canvas_rect = ui.available_rect_before_wrap();
    let center = canvas_rect.center();

    // Larger drop zone with logo
    let zone_w = (avail.x * 0.6).min(500.0).max(280.0);
    let zone_h = 380.0_f32;
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

    // Logo at center-top of drop zone (preserve aspect ratio)
    let logo_max_h = 140.0;
    let logo_aspect = theme::LOGO_ASPECT;
    let logo_w = logo_max_h * logo_aspect;
    let logo_size = Vec2::new(logo_w, logo_max_h);
    let logo_rect = Rect::from_center_size(
        Pos2::new(center.x, zone_rect.min.y + 16.0 + logo_size.y * 0.5),
        logo_size,
    );
    let logo_image = egui::Image::new(egui::include_image!("../../../../../img/logo-nobg.png"))
        .fit_to_exact_size(logo_size);
    logo_image.paint_at(ui, logo_rect);

    // Text below logo
    let text_y = logo_rect.max.y + 20.0;
    let painter = ui.painter();

    painter.text(
        Pos2::new(center.x, text_y),
        egui::Align2::CENTER_CENTER,
        "Drop an image here",
        egui::FontId::proportional(theme::FONT_SIZE_HEADING),
        theme::TEXT_PRIMARY,
    );

    let hint = if cfg!(target_os = "macos") {
        "or press Cmd+O to open a file"
    } else {
        "or press Ctrl+O to open a file"
    };
    painter.text(
        Pos2::new(center.x, text_y + 28.0),
        egui::Align2::CENTER_CENTER,
        hint,
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_SECONDARY,
    );

    let wayland_offset = if *IS_WAYLAND {
        painter.text(
            Pos2::new(center.x, text_y + 58.0),
            egui::Align2::CENTER_CENTER,
            "(Drag and drop not supported in Wayland yet)",
            egui::FontId::proportional(theme::FONT_SIZE_BODY * 0.85),
            theme::TEXT_SECONDARY,
        );
        30.0
    } else {
        0.0
    };

    let time = ui.ctx().input(|i| i.time);
    let tip_index = ((time / TIP_CYCLE_SECS) as usize) % TIPS.len();
    let phase = time % TIP_CYCLE_SECS; // 0..TIP_CYCLE_SECS

    let alpha = if phase < TIP_FADE_SECS {
        // Fading in
        (phase / TIP_FADE_SECS) as f32
    } else if phase > TIP_CYCLE_SECS - TIP_FADE_SECS {
        // Fading out
        ((TIP_CYCLE_SECS - phase) / TIP_FADE_SECS) as f32
    } else {
        1.0
    };

    let tip_color = egui::Color32::from_rgba_unmultiplied(
        theme::TEXT_SECONDARY.r(),
        theme::TEXT_SECONDARY.g(),
        theme::TEXT_SECONDARY.b(),
        (alpha * theme::TEXT_SECONDARY.a() as f32) as u8,
    );

    let tip_y = text_y + 68.0 + wayland_offset;
    painter.text(
        Pos2::new(center.x, tip_y),
        egui::Align2::CENTER_CENTER,
        &TIPS[tip_index],
        egui::FontId::proportional(theme::FONT_SIZE_BODY * 0.9),
        tip_color,
    );

    // Repaint for tip animation — during fades repaint frequently, otherwise schedule next transition
    if alpha < 1.0 {
        ui.ctx().request_repaint(); // smooth fade
    } else {
        let secs_until_fade_out = TIP_CYCLE_SECS - TIP_FADE_SECS - phase;
        ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(secs_until_fade_out.max(0.016)));
    }
}

fn render_loaded(ui: &mut egui::Ui, app: &PrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();
    if let Some(ref texture) = app.source_texture {
        let img_rect = compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);
        let fade = ui.ctx().animate_bool_with_time(
            egui::Id::new(("canvas_fade", app.canvas_switch_id)),
            true,
            0.2,
        );
        let alpha = (fade * 255.0) as u8;
        if fade < 1.0 { ui.ctx().request_repaint(); }
        ui.painter().image(
            texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
        );
    } else {
        // Source not decoded yet — show spinner
        let center = canvas_rect.center();
        ui.put(
            Rect::from_center_size(center, Vec2::splat(40.0)),
            egui::Spinner::new().size(40.0).color(theme::ACCENT),
        );
        ui.ctx().request_repaint();
    }
}

fn render_processing(ui: &mut egui::Ui, app: &PrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();

    let t = ui.ctx().input(|i| i.time) as f32;

    // In chain mode with an existing result, show the result being processed (not the original)
    let display_texture = if app.settings.chain_mode && app.result_texture.is_some() {
        app.result_texture.as_ref()
    } else {
        app.source_texture.as_ref()
    };

    if let Some(texture) = display_texture {
        let tex_size = texture.size_vec2();
        // Gentle wiggle/breathing effect on background image
        let wiggle_zoom = 1.0 + (t * 1.5).sin() * 0.003;
        let wiggle_x = (t * 0.8).sin() * 2.0;
        let wiggle_y = (t * 1.1).cos() * 1.5;
        let wiggle_offset = app.zoom_state.pan_offset + Vec2::new(wiggle_x, wiggle_y);
        let img_rect = compute_img_rect(canvas_rect, tex_size, app.zoom_state.zoom * wiggle_zoom, wiggle_offset);
        ui.painter().image(
            texture.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, 140),
        );

        // Shimmer sweep
        let sweep_x = ((t * 0.6).fract()) * (img_rect.width() + 80.0) - 40.0 + img_rect.min.x;
        let shimmer_rect = Rect::from_min_max(
            Pos2::new(sweep_x - 40.0, img_rect.min.y),
            Pos2::new(sweep_x + 40.0, img_rect.max.y),
        ).intersect(img_rect);
        if shimmer_rect.width() > 0.0 && shimmer_rect.height() > 0.0 {
            ui.painter().rect_filled(shimmer_rect, 0.0,
                Color32::from_rgba_unmultiplied(255, 255, 255, 18));
        }
    }

    const PROCESSING_DOTS: [&str; 4] = ["Processing.", "Processing..", "Processing...", "Processing...."];
    let label = PROCESSING_DOTS[(t * 2.0) as usize % 4];
    let center = canvas_rect.center();

    // Dark pill behind text
    let pill_rect = Rect::from_center_size(center + Vec2::new(0.0, 14.0), Vec2::new(280.0, 80.0));
    ui.painter().rect_filled(pill_rect, 14.0,
        Color32::from_rgba_unmultiplied(0, 0, 0, 180));

    ui.painter().text(
        center - Vec2::new(0.0, 6.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(theme::FONT_SIZE_HEADING),
        theme::TEXT_PRIMARY,
    );
    ui.painter().text(
        center + Vec2::new(0.0, 14.0),
        egui::Align2::CENTER_CENTER,
        &app.status.stage,
        egui::FontId::proportional(theme::FONT_SIZE_BODY),
        theme::TEXT_SECONDARY,
    );
    ui.painter().text(
        center + Vec2::new(0.0, 34.0),
        egui::Align2::CENTER_CENTER,
        "Press Escape to cancel",
        egui::FontId::proportional(theme::FONT_SIZE_MONO),
        theme::TEXT_SECONDARY,
    );

    ui.ctx().request_repaint_after(std::time::Duration::from_millis(66));
}

fn render_done(ui: &mut egui::Ui, app: &PrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();

    // Crossfade: result fades in over 0.4s when processing completes
    let fade = ui.ctx().animate_bool_with_time(
        egui::Id::new(("result_fade", app.result_switch_id)),
        true,
        0.4,
    );
    if fade < 1.0 { ui.ctx().request_repaint(); }

    if app.show_original {
        if let Some(ref texture) = app.source_texture {
            let img_rect =
                compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);
            ui.painter().image(
                texture.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    } else if let Some(ref result_tex) = app.result_texture {
        let img_rect =
            compute_img_rect(canvas_rect, result_tex.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);

        // During crossfade: show source fading out behind checkerboard + result fading in
        if fade < 1.0 {
            if let Some(ref source_tex) = app.source_texture {
                let src_alpha = ((1.0 - fade) * 255.0) as u8;
                ui.painter().image(
                    source_tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::from_rgba_unmultiplied(255, 255, 255, src_alpha),
                );
            }
        }

        if fade >= 1.0 {
            // Always paint the checkerboard underneath. Bg color (if set)
            // composites over it — so a partially-transparent bg lets the
            // checkerboard show through, mirroring how PNG transparency
            // would read the canvas.
            draw_checkerboard(ui, img_rect, app.settings.dark_checker);
            if let Some(bg) = app.batch.selected_item().and_then(|it| it.settings.bg) {
                ui.painter().rect_filled(
                    img_rect,
                    0.0,
                    Color32::from_rgba_unmultiplied(bg[0], bg[1], bg[2], bg[3]),
                );
            }
        }
        let result_alpha = (fade * 255.0) as u8;
        let tex_id = result_tex.id();
        let last = app.last_painted_tex_id.get();
        if last != Some(tex_id) {
            tracing::info!(?tex_id, ?last, fade, "canvas: painting result_texture (id changed)");
            app.last_painted_tex_id.set(Some(tex_id));
        }
        ui.painter().image(
            tex_id,
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, result_alpha),
        );
    }

    if app.show_original {
        ui.painter().text(
            canvas_rect.min + Vec2::new(theme::SPACE_SM, theme::SPACE_SM),
            egui::Align2::LEFT_TOP,
            "Original",
            egui::FontId::monospace(theme::FONT_SIZE_MONO),
            theme::TEXT_SECONDARY,
        );
    }
}

fn build_checker_image(light: Color32, dark: Color32) -> egui::ColorImage {
    let cell = theme::CHECKER_SIZE as usize;
    let px = CHECKER_TEX_TILES * cell;
    let mut img = egui::ColorImage::filled([px, px], light);
    for row in 0..CHECKER_TEX_TILES {
        for col in 0..CHECKER_TEX_TILES {
            if (row + col) % 2 != 0 {
                for dy in 0..cell {
                    let y = row * cell + dy;
                    let start = y * px + col * cell;
                    for dx in 0..cell {
                        img.pixels[start + dx] = dark;
                    }
                }
            }
        }
    }
    img
}

fn checker_texture(ctx: &egui::Context, dark: bool) -> TextureHandle {
    static LIGHT: OnceLock<TextureHandle> = OnceLock::new();
    static DARK: OnceLock<TextureHandle> = OnceLock::new();

    if dark {
        DARK.get_or_init(|| {
            let (light, dark) = theme::CHECKER_DARK_MODE;
            ctx.load_texture("checker_dark", build_checker_image(light, dark), TextureOptions::NEAREST)
        }).clone()
    } else {
        LIGHT.get_or_init(|| {
            let (light, dark) = theme::CHECKER_LIGHT_MODE;
            ctx.load_texture("checker_light", build_checker_image(light, dark), TextureOptions::NEAREST)
        }).clone()
    }
}

fn draw_checkerboard(ui: &egui::Ui, bounds: Rect, dark: bool) {
    let tex = checker_texture(ui.ctx(), dark);
    let tile_px = CHECKER_TEX_TILES as f32 * theme::CHECKER_SIZE; // screen-space size of one tile
    let painter = ui.painter();

    let mut y = bounds.min.y;
    while y < bounds.max.y {
        let mut x = bounds.min.x;
        let row_h = tile_px.min(bounds.max.y - y);
        while x < bounds.max.x {
            let col_w = tile_px.min(bounds.max.x - x);
            let tile_rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(col_w, row_h));
            // UV: fraction of tile actually visible (for edge tiles that are clipped)
            let uv_max = Pos2::new(col_w / tile_px, row_h / tile_px);
            painter.image(
                tex.id(), tile_rect,
                Rect::from_min_max(Pos2::ZERO, uv_max),
                Color32::WHITE,
            );
            x += tile_px;
        }
        y += tile_px;
    }
}
