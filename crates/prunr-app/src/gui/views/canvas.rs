use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use crate::gui::image_texture::ImageTexture;
use egui::{Color32, Pos2, Rect, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::gui::app::PrunrApp;
use crate::gui::knob_catalog::{shows, Surface};
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
    // A click-outside popover holds the canvas still, so the press that
    // dismisses it (and the drag after it) never pans or paints. A pinned
    // flyout does not: the canvas stays live under it. A pointer over any
    // popup, or a slider drag that leaves the popup's rectangle, is
    // egui's, which `egui_wants_pointer_input` reports.
    let dismissable_popup = theme::dismissable_popup_open(ui.ctx()) || app.popup_open_at_frame_start;
    let widget_has_pointer = ui.ctx().egui_wants_pointer_input();
    let pointer_blocked = modal_open || widget_has_pointer || dismissable_popup;
    // The brush authors the selection against the source, so it is live
    // as soon as an image is loaded; segmentation corrections apply once
    // a result exists.
    //
    // While an inpaint stroke is in flight on the selected item, the
    // brush is locked: another stroke would silently supersede the
    // first via dispatch_inpaint's generation counter, wasting the
    // 5+ minutes of CPU work just performed. The capsule's Cancel is
    // the way out.
    // A stroke, a Magic Brush click or a focus pick is a step: one taken
    // while a run is queued or under way on the image would make that
    // run's result stale, so they wait for it.
    let editable = app.batch.selected_item().is_some_and(crate::gui::item::BatchItem::is_idle);
    let app_state = app.batch.app_state();
    let model_has_brushes = shows(app.settings.model, Surface::Brushes);
    let brush_active = model_has_brushes && app.brush_state.is_enabled() && editable;
    // Magic Brush is treated as a brush tool for pan-gate purposes: left
    // drag belongs to the brush and pan moves to the secondary (right)
    // button, matching Paint Brush. That holds while the image is still
    // being prepared too: a click or stroke then waits and runs once it is
    // ready, rather than being lost.
    let magic_brush_active = model_has_brushes
        && app.magic_brush_state.is_active()
        && editable;
    // Picking the Depth chip's focus takes the next left click too.
    let picking_focus = app.picking_depth_focus && editable;
    let any_brush_tool_active = brush_active || magic_brush_active || picking_focus;
    // Scroll-zoom always works: it doesn't conflict with brush strokes
    // and the zoom feedback is reassuring even mid-painting.
    let canvas_gets_zoom = !pointer_blocked;
    // Pan binding: primary (left) drag normally; secondary (right) drag
    // in any brush-tool mode so left-click is free for the brush stroke.
    let canvas_gets_pan = canvas_gets_zoom;
    if canvas_gets_zoom {
        // Handle scroll-wheel zoom (cursor-centered)
        ui.ctx().input(|i| {
            for event in &i.events {
                if let egui::Event::MouseWheel { delta, modifiers, .. } = event {
                    if !modifiers.any() {
                        let scroll_y = delta.y;
                        let zoom_delta = theme::ZOOM_STEP.powf(scroll_y);
                        if let Some(cursor) = i.pointer.hover_pos() {
                            if canvas_rect.contains(cursor) {
                                app.zoom_state.zoom_about(cursor - canvas_rect.center(), app.zoom_state.zoom * zoom_delta);
                            }
                        }
                    }
                }
            }
        });
    }
    if canvas_gets_pan {
        ui.ctx().input(|i| {
            // Click+drag pan: enter panning ONLY on a fresh press inside the canvas.
            // This avoids false pan when egui's pointer state is desynced — e.g. after
            // an OS drag-out session where Prunr's window never saw the mouse-up.
            let hovered_inside = i.pointer.hover_pos().is_some_and(|p| canvas_rect.contains(p));
            let (pressed, down) = if any_brush_tool_active {
                (i.pointer.secondary_pressed(), i.pointer.button_down(egui::PointerButton::Secondary))
            } else {
                (i.pointer.primary_pressed(), i.pointer.primary_down())
            };
            if pressed && hovered_inside {
                app.zoom_state.is_panning = true;
            }
            if !down {
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
    let frame = app.batch.selected_item()
        .filter(|i| i.source_texture.is_some())
        .map(|i| (i.id, i.frame_size()));
    if let Some((item_id, tex_size)) = frame {
        app.zoom_state.follow_frame(item_id, tex_size);
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

    let img_rect = match app_state {
        AppState::Empty => {
            render_empty(ui);
            None
        }
        AppState::Loaded => render_loaded(ui, app),
        AppState::Processing => render_processing_canvas(ui, app),
        AppState::Done => render_done(ui, app),
    };

    if let Some(progress) = app.processor.dispatch_progress() {
        // A one-image run's tile map belongs over its own image; over
        // another, only the capsule shows, naming the image it works on.
        let shown = app.batch.selected_item().map(|i| i.id);
        let here = progress.is_over(shown);
        let elsewhere = progress.on.filter(|_| !here).and_then(|id| app.batch.find_by_id(id)).map(|i| i.filename.as_str());
        let preview = app.batch.selected_item().filter(|_| here).and_then(|i| i.tile_preview.as_ref()).map(|t| t.id());
        if super::progress_widget::render(ui, canvas_rect, img_rect.filter(|_| here), &progress, preview, elsewhere) {
            // Routing logic is the pure `cancel_target_for` fn — one
            // table-tested match instead of an inline copy.
            use crate::gui::dispatch_progress::{cancel_target_for, CancelTarget};
            if let Some(target) = cancel_target_for(progress.kind) {
                match target {
                    CancelTarget::Inpaint => app.cancel_erases(),
                    // The capsule counts the queued images too: its Cancel ends them all.
                    CancelTarget::Upscale | CancelTarget::Colorize => app.processor.cancel_runs(),
                    CancelTarget::SegBatchAndReset => app.handle_cancel_all_and_reset(),
                    CancelTarget::Faces => app.processor.cancel_faces(),
                }
            }
        }
        ui.ctx().request_repaint_after(theme::WORK_FRAME);
    } else if let Some(label) = app.busy_pill(std::time::Instant::now()) {
        super::progress_widget::render_working_pill(ui, canvas_rect, label);
        ui.ctx().request_repaint_after(theme::WORK_FRAME);
    }

    // The strip sits on the foreground layer, above the modals' level:
    // under an open modal it would draw over the modal.
    if let Some(tool) = app.active_tool().filter(|_| !modal_open) {
        let facts = super::tool_strip::StripFacts {
            model: app.settings.model,
            encoder_pending: app.magic_brush_preparing(),
            reroll_refusal: app.reroll_refusal(),
        };
        let depth_before = app.settings.brush.magic_depth;
        let change = super::tool_strip::render(ui, canvas_rect, tool, &mut app.settings.brush, facts);
        app.apply_brush_change(change.brush);
        if app.settings.brush.magic_depth && !depth_before {
            app.offer_depth_model(ui.ctx());
        }
        if change.try_another {
            app.handle_reroll_intent();
        }
    }

    if picking_focus && !pointer_blocked {
        handle_focus_pick(ui, app, canvas_rect);
    } else {
        if brush_active && !pointer_blocked {
            handle_brush_input(ui, app, canvas_rect);
        }

        // Magic Brush click/stroke input; one made while the image is still
        // being prepared waits for it.
        if magic_brush_active && !pointer_blocked {
            handle_magic_brush_input(ui, app, canvas_rect);
        }
    }

    if app.magic_brush_state.is_active() && app.magic_brush_preparing() {
        let hover = ui.ctx().input(|i| i.pointer.hover_pos()).filter(|p| canvas_rect.contains(*p));
        if hover.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Wait);
        }
        super::progress_widget::render_loading_pill(ui, canvas_rect, hover, prunr_core::Step::LoadingMagicBrush.label());
        ui.ctx().request_repaint_after(theme::WORK_FRAME);
    }

    // Bottom-left modifier hint — shown whenever Magic Brush is the active
    // tool (regardless of encoder state) so users discover Shift / Alt
    // before they need them. Active modifier tints to ACCENT; the rest stays
    // TEXT_SECONDARY.
    if app.magic_brush_state.is_active() || app.brush_state.is_enabled() {
        let (shift, alt) = modifier_state(ui);
        let base_color = theme::TEXT_SECONDARY;
        let active_color = theme::ACCENT;
        let pos = egui::pos2(
            canvas_rect.min.x + theme::SPACE_SM,
            canvas_rect.max.y - theme::SPACE_SM - theme::FONT_SIZE_SECONDARY,
        );
        let font = egui::FontId::monospace(theme::FONT_SIZE_SECONDARY);
        let painter = ui.painter();
        let (shift_hint, alt_hint) = super::shortcuts::MODIFIER_HINTS;
        let shift_color = if shift { active_color } else { base_color };
        let shift_rect = painter.text(
            pos, egui::Align2::LEFT_TOP, shift_hint, font.clone(), shift_color,
        );
        let sep_rect = painter.text(
            egui::pos2(shift_rect.max.x, pos.y), egui::Align2::LEFT_TOP,
            "  ", font.clone(), base_color,
        );
        let alt_color = if alt { active_color } else { base_color };
        painter.text(
            egui::pos2(sep_rect.max.x, pos.y), egui::Align2::LEFT_TOP,
            alt_hint, font, alt_color,
        );
    }
}

/// Whether Shift and the subtract modifier are held.
fn modifier_state(ui: &egui::Ui) -> (bool, bool) {
    ui.ctx().input(|i| (i.modifiers.shift, super::shortcuts::is_subtract_modifier(&i.modifiers)))
}

/// The held modifier's glyph (+ for Shift, − for subtract) below-right of
/// the pointer, the macOS modifier-indicator convention.
fn draw_modifier_glyph(ui: &egui::Ui, canvas_rect: Rect, (shift, subtract): (bool, bool)) {
    if !(shift || subtract) {
        return;
    }
    let glyph = if shift { "+" } else { "\u{2212}" };
    let Some(p) = ui.ctx().input(|i| i.pointer.hover_pos()).filter(|p| canvas_rect.contains(*p)) else { return };
    ui.painter().text(p + egui::vec2(12.0, 12.0), egui::Align2::CENTER_CENTER, glyph, egui::FontId::proportional(14.0), theme::ACCENT);
}

/// The item a brush acts on and where its image sits on the canvas;
/// `None` with no item, or none with a texture to paint over yet.
fn brush_target(app: &PrunrApp, canvas_rect: Rect) -> Option<(usize, Rect)> {
    let item = app.batch.selected_item()?;
    if item.result_texture.is_none() && item.source_texture.is_none() {
        return None;
    }
    let img_rect = compute_img_rect(canvas_rect, item.frame_size(), app.zoom_state.zoom, app.zoom_state.pan_offset);
    Some((app.batch.selected_idx_clamped()?, img_rect))
}

/// Run the brush overlay (cursor + pointer events) and commit any
/// finished stroke onto the active item.
fn handle_brush_input(ui: &mut egui::Ui, app: &mut PrunrApp, canvas_rect: Rect) {
    use super::brush_overlay::{self, BrushAction};
    let Some((idx, img_rect)) = brush_target(app, canvas_rect) else { return };
    let (shift, subtract) = modifier_state(ui);
    let selection_strokes = shows(app.settings.model, Surface::SelectionStrokes);
    let stamp = if selection_strokes {
        let mut stamp = app.settings.brush.stamp();
        stamp.mode = app.settings.brush.mode_under(app.settings.model.edits_cutout(), shift, subtract);
        stamp
    } else {
        app.settings.brush.region_stamp(subtract && !shift)
    };
    let action = {
        let item = &app.batch.items[idx];
        brush_overlay::handle_input(ui, &mut app.brush_state, &app.settings, stamp, item, img_rect)
    };
    draw_modifier_glyph(ui, canvas_rect, (shift, subtract));
    if let BrushAction::Committed(stroke_mask) = action {
        let item_id = app.batch.items[idx].id;

        tracing::info!(item_id, "brush stroke committed; merging into the selection");
        let op = crate::gui::selection_queue::SelectionOp::Merge(std::sync::Arc::new(stroke_mask));
        app.queue_selection_op(item_id, op, crate::gui::selection_queue::AfterLanding::Nothing);
    }
}

/// The Depth chip's "Set focus": a crosshair over the image, and the
/// next click on it focuses there; a click beside it gives up.
fn handle_focus_pick(ui: &mut egui::Ui, app: &mut PrunrApp, canvas_rect: Rect) {
    let Some(item) = app.batch.selected_item() else { return };
    let img_rect = compute_img_rect(canvas_rect, item.frame_size(), app.zoom_state.zoom, app.zoom_state.pan_offset);
    let (clicked, hover) = ui.ctx().input(|i| (i.pointer.primary_clicked(), i.pointer.hover_pos()));
    if hover.is_some_and(|p| canvas_rect.contains(p)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    if !clicked {
        return;
    }
    match hover.filter(|p| img_rect.contains(*p)) {
        Some(p) => {
            let f = super::brush_overlay::screen_to_model(p, img_rect, 1, 1);
            let (fx, fy) = (f.x.clamp(0.0, 1.0), f.y.clamp(0.0, 1.0));
            app.set_depth_focus(fx, fy);
        }
        None if hover.is_some_and(|p| canvas_rect.contains(p)) => app.picking_depth_focus = false,
        None => {}
    }
}

/// Handle Magic Brush pointer events (click + drag-stroke) on the canvas.
/// Suppressed when encoder is pending — clicks during "Preparing..." are
/// silently ignored. Dispatches a SAM decoder job on
/// every click or completed stroke; modifier keys control mask combination.
///
/// Also owns the per-frame visual feedback for the tool: crosshair cursor
/// icon, brush-shape cursor outline (via `brush_overlay::draw_cursor`),
/// trail rendering during drag (via `brush_overlay::draw_trail_for`),
/// and the modifier glyph (+ / −) at the cursor when Shift / Alt is held.
fn handle_magic_brush_input(ui: &mut egui::Ui, app: &mut PrunrApp, canvas_rect: Rect) {
    let Some((idx, img_rect)) = brush_target(app, canvas_rect) else { return };
    let item = &app.batch.items[idx];
    let item_id = item.id;
    let (source_w, source_h) = item.dimensions;
    // `None` while the encoder runs; a pick made then waits for it.
    let embedding = item.magic_brush_embedding.clone();

    // Cursor icon: Crosshair while over the canvas, default elsewhere.
    let hover_pos_for_cursor = ui.ctx().input(|i| i.pointer.hover_pos());
    if hover_pos_for_cursor.is_some_and(|p| canvas_rect.contains(p)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    let look = super::brush_overlay::CursorLook { erase: app.settings.brush_erases(), ..super::brush_overlay::MAGIC_CURSOR };
    let pointer_on_img = hover_pos_for_cursor.filter(|p| img_rect.contains(*p));
    super::brush_overlay::draw_trail_for(ui, look, look.shape, app.magic_brush_state.active_trail.stamps());
    super::brush_overlay::draw_cursor(ui, img_rect, look, pointer_on_img.is_some());

    let (shift, subtract) = modifier_state(ui);
    let mode = app.settings.brush.mode_under(app.settings.model.edits_cutout(), shift, subtract);
    draw_modifier_glyph(ui, canvas_rect, (shift, subtract));

    let screen_to_src = |pos: egui::Pos2| source_point(pos, img_rect, (source_w, source_h));

    let (clicked, drag_started, dragging, released, hover_pos) = ui.ctx().input(|i| {
        let hover = i.pointer.hover_pos();
        let on_image = hover.is_some_and(|p| img_rect.contains(p));
        let clicked = i.pointer.primary_clicked() && on_image;
        let drag_started = i.pointer.primary_pressed() && on_image;
        let dragging = i.pointer.primary_down() && on_image;
        let released = i.pointer.primary_released();
        (clicked, drag_started, dragging, released, hover)
    });

    // What a click or stroke selects with: the depth map, or SAM. A SAM
    // pick made before the encoder has finished waits for the embedding.
    let by_depth = app.settings.brush.magic_depth;

    let screen_radius = look.radius;

    if drag_started {
        app.magic_brush_state.clear_stroke();
    }

    if dragging {
        if let Some(pos) = hover_pos {
            if let Some((px, py)) = screen_to_src(pos) {
                let state = &mut app.magic_brush_state;
                // Dedup the source-coords list on bit-exact repeats; SAM
                // doesn't benefit from duplicate points and we cap at 8
                // anyway in build_stroke_prompt.
                if state.active_stroke.last() != Some(&(px, py)) {
                    state.active_stroke.push((px, py));
                }
                state.active_trail.push_spaced(pos.x, pos.y, screen_radius);
            }
        }
    }

    if released && app.magic_brush_state.active_stroke.len() > 1 {
        // Stroke completed — dispatch decoder with stroke prompt.
        let stroke_pts = std::mem::take(&mut app.magic_brush_state.active_stroke);
        app.magic_brush_state.active_trail.clear();
        if by_depth {
            // With Depth on a stroke selects like a click where it began.
            app.magic_depth_click(item_id, stroke_pts[0], (source_w, source_h), mode);
        } else {
            match prunr_core::sam::prompt::build_stroke_prompt(&stroke_pts, source_w, source_h) {
                Ok(prompt) => app.pick_with_sam(item_id, embedding, prompt, mode),
                Err(e) => tracing::warn!(item_id, "SAM stroke prompt failed: {e:?}"),
            }
        }
    } else if clicked {
        if let Some((px, py)) = hover_pos.and_then(screen_to_src) {
            // A stroke's own release was taken by the branch above, so
            // what is left is a click, even when its press and release
            // fell in different frames and left one point behind.
            if by_depth {
                app.magic_depth_click(item_id, (px, py), (source_w, source_h), mode);
            } else {
                let prompt = prunr_core::sam::prompt::build_click_prompt(px, py, source_w, source_h);
                app.pick_with_sam(item_id, embedding, prompt, mode);
            }
        }
        app.magic_brush_state.clear_stroke();
    }
}

/// A screen position as a source pixel, or `None` off the image: only
/// the image takes Magic Brush prompts, since a press beside it would
/// otherwise clamp onto the nearest edge and select whatever sits there.
fn source_point(pos: egui::Pos2, img_rect: Rect, (w, h): (u32, u32)) -> Option<(f32, f32)> {
    img_rect.contains(pos).then(|| {
        let p = super::brush_overlay::screen_to_model(pos, img_rect, w, h);
        (p.x.min(w as f32 - 1.0), p.y.min(h as f32 - 1.0))
    })
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

/// The empty-canvas texts, built from the bindings they name and rebuilt
/// when those change, so a rebind shows here too.
struct TipText {
    bindings: Arc<super::shortcuts::Bindings>,
    tips: Vec<String>,
    open_hint: String,
}

impl TipText {
    fn build(bindings: Arc<super::shortcuts::Bindings>) -> Self {
        use super::shortcuts::{label, Action};
        let keys = |a: Action| Arc::clone(bindings.display(a));
        let press = |a: Action| {
            let mut chars = label(a).chars();
            let first = chars.next().map(|c| c.to_lowercase().to_string()).unwrap_or_default();
            format!("Press {} to {first}{}", keys(a), chars.as_str())
        };
        let tips = vec![
            press(Action::Shortcuts),
            press(Action::CliHelp),
            press(Action::Process),
            press(Action::BeforeAfter),
            format!("Use {} or {} to switch images", keys(Action::PrevImage), keys(Action::NextImage)),
            press(Action::FitToWindow),
            "Scroll to zoom, drag to pan".to_string(),
            press(Action::ToggleQueue),
            press(Action::Settings),
            press(Action::Save),
            press(Action::Copy),
            press(Action::Undo),
            "Open several images to queue them".to_string(),
        ];
        let open_hint = format!("or press {} to open a file", keys(Action::Open));
        Self { bindings, tips, open_hint }
    }
}

fn tip_text(ctx: &egui::Context) -> Arc<TipText> {
    static CACHE: Mutex<Option<Arc<TipText>>> = Mutex::new(None);
    let live = super::shortcuts::current(ctx);
    let mut cache = CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match cache.as_ref() {
        Some(t) if Arc::ptr_eq(&t.bindings, &live) => Arc::clone(t),
        _ => {
            let built = Arc::new(TipText::build(live));
            *cache = Some(Arc::clone(&built));
            built
        }
    }
}

fn render_empty(ui: &mut egui::Ui) {
    let avail = ui.available_size();
    let is_hovered = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
    let canvas_rect = ui.available_rect_before_wrap();
    let center = canvas_rect.center();

    // Larger drop zone with logo
    let zone_w = (avail.x * 0.6).clamp(280.0, 500.0);
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

    painter.text(
        Pos2::new(center.x, text_y + 28.0),
        egui::Align2::CENTER_CENTER,
        tip_text(ui.ctx()).open_hint.as_str(),
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
    let tips = tip_text(ui.ctx());
    let tip_index = ((time / TIP_CYCLE_SECS) as usize) % tips.tips.len();
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
        &tips.tips[tip_index],
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

/// Points round a found face's outline.
const FACE_SEGMENTS: usize = 72;

/// `area`'s outline on screen, `img_rect` showing the photo it is in.
fn face_outline(area: prunr_core::face::FaceArea, img_rect: Rect) -> [Pos2; FACE_SEGMENTS] {
    let scale = img_rect.width() / area.photo_size[0].max(1) as f32;
    let (sin, cos) = area.angle.sin_cos();
    std::array::from_fn(|k| {
        let (s, c) = (k as f32 / FACE_SEGMENTS as f32 * std::f32::consts::TAU).sin_cos();
        let (x, y) = (area.radii[0] * c, area.radii[1] * s);
        img_rect.min + Vec2::new(area.center[0] + x * cos - y * sin, area.center[1] + x * sin + y * cos) * scale
    })
}

/// While Faces and its Outline switch are on, the faces found in the
/// shown image outlined: what Faces redraws, so a face it missed shows by
/// having none. They fade in as soon as the search finds them, before they are
/// redrawn, and glow once; under "Largest" the faces it leaves alone are
/// faint.
fn draw_found_faces(ui: &egui::Ui, app: &PrunrApp, item: &crate::gui::item::BatchItem, img_rect: Rect) {
    use super::effects::{face_arrival, paint_glow_ring, FACE_ARRIVAL_SECS, FACE_FADE_SECS};
    let faces = &item.settings.faces;
    let ctx = ui.ctx();
    let open = faces.on && app.settings.face_outlines && shows(app.settings.model, Surface::Faces);
    let shown = ctx.animate_bool_with_time(egui::Id::new("found_faces_shown"), open, theme::fade_secs(ctx, FACE_FADE_SECS));
    if shown <= 0.0 {
        return;
    }
    let Some(found) = item.outlined_faces(crate::gui::item::FaceStage::of(app.settings.model.role())) else { return };
    let since = if theme::motion(ctx) { found.found_at.elapsed().as_secs_f32() } else { f32::MAX };
    if since < FACE_ARRIVAL_SECS {
        ctx.request_repaint();
    }
    let (size, glow) = face_arrival(since);
    let opacity = shown * prunr_core::math::smoothstep((since / FACE_FADE_SECS).min(1.0));
    let redrawn = faces.blend().redrawn(found.places());
    for (i, face) in found.places().enumerate() {
        let (opacity, glow) = if redrawn(i) { (opacity, glow) } else { (0.4 * opacity, 0.0) };
        let area = face.area(faces.hair);
        let area = prunr_core::face::FaceArea { radii: area.radii.map(|r| r * size), ..area };
        paint_glow_ring(ui.painter(), &face_outline(area, img_rect), opacity, glow);
    }
}

/// The faces a search has redrawn so far, each over the image in its
/// place, until the image with them blended in is shown.
fn draw_face_preview(ui: &egui::Ui, item: &crate::gui::item::BatchItem, img_rect: Rect) {
    if let Some(texture) = item.face_preview_texture() {
        ui.painter().image(texture.id(), img_rect, super::effects::UNIT_UV, Color32::WHITE);
    }
}

/// The item's selection over the image, on models that use one.
fn draw_selection(ui: &egui::Ui, app: &PrunrApp, item: &crate::gui::item::BatchItem, img_rect: Rect) {
    if shows(app.settings.model, Surface::Brushes) {
        super::selection_overlay::render_selection_overlay(ui, item, img_rect, app.processor.is_inpaint_in_flight(item.id));
    }
}

/// `texture` in `img_rect`, revealing over the held image while the
/// item's reveal runs.
fn paint_image_or_reveal(ui: &egui::Ui, img_rect: Rect, item: &crate::gui::item::BatchItem, texture: &ImageTexture, tint: Color32) {
    match item.reveal_progress().filter(|_| theme::motion(ui.ctx())) {
        Some((from, p, order)) => {
            super::effects::paint_reveal(ui.painter(), img_rect, from, texture, p, order);
            // The subject's outline is gone by the time the cut-out has
            // grown halfway out from it.
            if let Some(subject) = item.subject.as_ref() {
                let fade = (1.0 - 2.0 * p).max(0.0);
                ui.painter().image(subject.texture.id(), img_rect, super::effects::UNIT_UV, Color32::WHITE.gamma_multiply(fade * fade));
            }
            ui.ctx().request_repaint();
        }
        None => texture.paint(ui.painter(), img_rect, tint),
    }
}

fn render_loaded(ui: &mut egui::Ui, app: &PrunrApp) -> Option<Rect> {
    let canvas_rect = ui.available_rect_before_wrap();
    if let Some((item, texture)) = app.batch.selected_item().and_then(|i| Some((i, i.source_texture.as_ref()?))) {
        let img_rect = compute_img_rect(canvas_rect, item.frame_size(), app.zoom_state.zoom, app.zoom_state.pan_offset);
        let fade = ui.ctx().animate_bool_with_time(
            egui::Id::new(("canvas_fade", app.canvas_switch_id)),
            true,
            theme::fade_secs(ui.ctx(), theme::IMAGE_FADE_SECS),
        );
        let alpha = prunr_core::math::round_u8(fade * 255.0);
        if fade < 1.0 { ui.ctx().request_repaint(); }
        if let Some(item) = app.batch.selected_item() {
            // Undone back to the original: the source reveals over the cut.
            if item.reveal_progress().is_some() {
                draw_checkerboard(ui, img_rect, app.settings.dark_checker);
            }
            // A drag on a photo with no result yet (No model's first one)
            // shows its quick frames over it.
            let texture = item.quick.texture().or(item.photo_look.as_ref()).unwrap_or(texture);
            paint_image_or_reveal(ui, img_rect, item, texture, Color32::from_rgba_unmultiplied(255, 255, 255, alpha));
        }

        // Selection overlay — same call as `render_done`. Required here too
        // because Magic Brush authors selections directly on the source
        // image, before any Process click; without this the action bar
        // shows but the user sees no mask.
        if let Some(it) = app.batch.selected_item() {
            draw_face_preview(ui, it, img_rect);
            draw_selection(ui, app, it, img_rect);
            draw_found_faces(ui, app, it, img_rect);
        }
        Some(img_rect)
    } else {
        // Source not decoded yet — show spinner
        let center = canvas_rect.center();
        ui.put(
            Rect::from_center_size(center, Vec2::splat(40.0)),
            egui::Spinner::new().size(40.0).color(theme::ACCENT),
        );
        ui.ctx().request_repaint();
        None
    }
}

/// Paint the canvas-background "work in progress" effect: gentle
/// wiggle/breathing of the source image + horizontal shimmer sweep.
fn render_processing_canvas(ui: &mut egui::Ui, app: &PrunrApp) -> Option<Rect> {
    let canvas_rect = ui.available_rect_before_wrap();
    let t = theme::motion_clock(ui.ctx());

    let item = app.batch.selected_item();
    let display_texture = item.and_then(processing_texture);

    let painted = display_texture.map(|texture| {
        let tex_size = item.map_or(texture.size_vec2(), |i| i.frame_size());
        let wiggle_zoom = 1.0 + (t * 1.5).sin() * 0.003;
        let wiggle_x = (t * 0.8).sin() * 2.0;
        let wiggle_y = (t * 1.1).cos() * 1.5;
        let wiggle_offset = app.zoom_state.pan_offset + Vec2::new(wiggle_x, wiggle_y);
        let img_rect = compute_img_rect(canvas_rect, tex_size, app.zoom_state.zoom * wiggle_zoom, wiggle_offset);
        texture.paint(ui.painter(), img_rect, Color32::from_rgba_unmultiplied(255, 255, 255, 140));

        let sweep_x = ((t * 0.6).fract()) * (img_rect.width() + 80.0) - 40.0 + img_rect.min.x;
        let shimmer_rect = Rect::from_min_max(
            Pos2::new(sweep_x - 40.0, img_rect.min.y),
            Pos2::new(sweep_x + 40.0, img_rect.max.y),
        ).intersect(img_rect);
        // A sweep held still is just a stripe: it only shows while moving.
        if theme::motion(ui.ctx()) && shimmer_rect.width() > 0.0 && shimmer_rect.height() > 0.0 {
            ui.painter().rect_filled(shimmer_rect, 0.0, theme::PROCESSING_SWEEP);
        }
        // The subject, outlined as soon as the model has found it.
        if let Some(subject) = item.and_then(|i| i.subject.as_ref()) {
            let shown = (subject.shown_at.elapsed().as_secs_f32() / super::effects::SUBJECT_FADE_SECS).min(1.0);
            let breath = 0.75 + 0.25 * super::effects::breath(t);
            ui.painter().image(subject.texture.id(), img_rect, super::effects::UNIT_UV, Color32::WHITE.gamma_multiply(shown * breath));
            ui.ctx().request_repaint_after(theme::WORK_FRAME);
        }
        img_rect
    });

    if theme::motion(ui.ctx()) {
        ui.ctx().request_repaint_after(theme::ANIM_FRAME);
    }
    painted
}

/// What the canvas shows while a run works: the image being edited, the
/// newest result, whatever the run reads; the photo before there is one.
fn processing_texture(item: &crate::gui::item::BatchItem) -> Option<&ImageTexture> {
    item.result_texture.as_ref().or(item.source_texture.as_ref())
}

/// What the canvas shows for a result. Until a new result's texture
/// lands, the image it reveals over stays: the one it replaces, or the
/// run's tile preview. A drag's quick frame shows in place of the result.
fn done_texture(item: &crate::gui::item::BatchItem) -> Option<&ImageTexture> {
    item.held_texture().or(item.quick.texture()).or(item.result_texture.as_ref())
}

fn render_done(ui: &mut egui::Ui, app: &PrunrApp) -> Option<Rect> {
    let canvas_rect = ui.available_rect_before_wrap();
    let item = app.batch.selected_item();
    let mut painted = None;
    if app.show_original {
        if let Some(texture) = item.and_then(|i| i.source_texture.as_ref()) {
            // Drawn in the result's frame: after an upscale the original
            // covers the same area instead of a quarter of it.
            let frame = item.map_or(texture.size_vec2(), |i| i.frame_size());
            let img_rect = compute_img_rect(canvas_rect, frame, app.zoom_state.zoom, app.zoom_state.pan_offset);
            texture.paint(ui.painter(), img_rect, Color32::WHITE);
            // The selection is authored against the source, so it shows
            // while comparing too.
            if let Some(it) = item {
                draw_selection(ui, app, it, img_rect);
            }
            painted = Some(img_rect);
        }
    } else if let Some(it) = item {
        let shown = done_texture(it)?;
        let img_rect = compute_img_rect(canvas_rect, it.frame_size(), app.zoom_state.zoom, app.zoom_state.pan_offset);
        draw_checkerboard(ui, img_rect, app.settings.dark_checker);
        paint_backdrop(ui, img_rect, it, it.frame_size());
        paint_image_or_reveal(ui, img_rect, it, shown, Color32::WHITE);
        draw_face_preview(ui, it, img_rect);
        draw_selection(ui, app, it, img_rect);
        draw_found_faces(ui, app, it, img_rect);
        painted = Some(img_rect);
    }

    if app.show_original {
        ui.painter().text(
            canvas_rect.min + Vec2::new(theme::SPACE_SM, theme::SPACE_SM),
            egui::Align2::LEFT_TOP,
            "Original",
            egui::FontId::monospace(theme::FONT_SIZE_SECONDARY),
            theme::TEXT_SECONDARY,
        );
    }
    painted
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

/// What sits behind `item`'s result, painted into `rect` under it. Until a
/// picture's texture lands, nothing (the checkerboard) shows.
pub(super) fn paint_backdrop(ui: &egui::Ui, rect: Rect, item: &crate::gui::item::BatchItem, image_px: Vec2) {
    use crate::gui::item_settings::Backdrop;
    let backdrop = item.settings.backdrop();
    match backdrop {
        Backdrop::Transparent => {}
        Backdrop::Color([r, g, b]) => {
            ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(r, g, b));
        }
        Backdrop::Image(_) | Backdrop::Effect(_) => {
            let Some((_, tex)) = item.backdrop_texture.as_ref() else { return };
            paint_bg_image(ui, rect, tex, backdrop.fit(item.settings.bg_image_fit), image_px);
        }
    }
}

/// The region that repeats a `tex_px` picture at its own pixel size over
/// an image of `image_px` pixels, as export tiles it: the number of tiles
/// follows the image, not the zoom. Past 1 the picture repeats.
fn tile_uv(image_px: Vec2, tex_px: Vec2) -> Rect {
    Rect::from_min_max(Pos2::ZERO, Pos2::new(image_px.x / tex_px.x, image_px.y / tex_px.y))
}

/// Paint `tex` into `bounds` per `fit`: which region of the picture shows
/// where, with no texture re-upload.
pub(super) fn paint_bg_image(
    ui: &egui::Ui,
    bounds: Rect,
    tex: &ImageTexture,
    fit: prunr_core::BgImageFit,
    image_px: Vec2,
) {
    use prunr_core::BgImageFit;
    let tex_size = tex.size_vec2();
    let (tw, th) = (tex_size.x, tex_size.y);
    let (bw, bh) = (bounds.width(), bounds.height());
    if tw <= 0.0 || th <= 0.0 || bw <= 0.0 || bh <= 0.0 {
        return;
    }
    match fit {
        BgImageFit::Cover => {
            let tex_aspect = tw / th;
            let bounds_aspect = bw / bh;
            let (uv_min, uv_max) = if tex_aspect > bounds_aspect {
                let visible = bounds_aspect / tex_aspect;
                let pad = (1.0 - visible) * 0.5;
                (Pos2::new(pad, 0.0), Pos2::new(1.0 - pad, 1.0))
            } else {
                let visible = tex_aspect / bounds_aspect;
                let pad = (1.0 - visible) * 0.5;
                (Pos2::new(0.0, pad), Pos2::new(1.0, 1.0 - pad))
            };
            tex.paint_region(ui.painter(), bounds, Rect::from_min_max(uv_min, uv_max), Color32::WHITE);
        }
        BgImageFit::Contain => {
            let scale = (bw / tw).min(bh / th);
            let inner = egui::vec2(tw * scale, th * scale);
            let inner_rect = Rect::from_center_size(bounds.center(), inner);
            tex.paint(ui.painter(), inner_rect, Color32::WHITE);
        }
        BgImageFit::Tile => {
            tex.paint_region(ui.painter(), bounds, tile_uv(image_px, tex_size), Color32::WHITE);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A found face's outline sits where the face is in the photo, at
    /// the zoom the image is shown, turned with the face.
    #[test]
    fn a_face_outline_follows_the_photo_onto_the_screen() {
        use prunr_core::face::FaceArea;
        let shown = Rect::from_min_size(Pos2::new(10.0, 0.0), Vec2::new(400.0, 200.0));
        let upright = FaceArea { center: [100.0, 50.0], radii: [10.0, 20.0], angle: 0.0, photo_size: [200, 100] };
        let pts = face_outline(upright, shown);
        let near = |p: Pos2, x: f32, y: f32| (p.x - x).abs() < 1e-3 && (p.y - y).abs() < 1e-3;
        assert!(near(pts[0], 10.0 + 220.0, 100.0), "{:?}", pts[0]);
        assert!(near(pts[FACE_SEGMENTS / 4], 10.0 + 200.0, 140.0), "{:?}", pts[FACE_SEGMENTS / 4]);
        let turned = FaceArea { angle: std::f32::consts::FRAC_PI_2, ..upright };
        assert!(near(face_outline(turned, shown)[0], 10.0 + 200.0, 120.0), "a quarter turn");
    }

    /// Tile repeats a picture at its own pixel size over the image, as
    /// export does: the count follows the image, whatever the zoom.
    #[test]
    fn tiling_counts_tiles_in_image_pixels() {
        let uv = tile_uv(Vec2::new(800.0, 600.0), Vec2::new(200.0, 300.0));
        assert_eq!(uv.max, Pos2::new(4.0, 2.0));
        assert_eq!(uv.min, Pos2::ZERO);
    }

    /// A run shows the image being edited, the newest result, even when
    /// it redoes that step from the photo (a Sketch mode change, a second
    /// upscale); the photo shows only before there is a result.
    #[test]
    fn a_run_shows_the_newest_image_whatever_it_reads() {
        use crate::gui::item::{BatchItem, Cutter};
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]), Default::default());
        let mut item = BatchItem::new_for_test();
        item.source_texture = Some(tex("photo").into());
        assert_eq!(processing_texture(&item).map(|t| t.id()), item.source_texture.as_ref().map(|t| t.id()), "nothing processed yet");

        item.land_cut_for_test(Cutter::Model(prunr_core::ModelKind::Silueta), std::sync::Arc::new(image::RgbaImage::new(1, 1)));
        item.result_texture = Some(tex("cut-out").into());
        item.start_cut(None, Cutter::Model(prunr_core::ModelKind::Silueta));
        assert!(item.run_input(crate::gui::item::Engine::Segment).is_none(), "the re-run reads the photo");
        assert_eq!(processing_texture(&item).map(|t| t.id()), item.result_texture.as_ref().map(|t| t.id()), "and the cut-out stays on screen");
    }

    /// A drag's quick frame shows in place of the result; an image held
    /// for a reveal still shows over both.
    #[test]
    fn a_quick_frame_shows_in_place_of_the_result() {
        use crate::gui::item::{BatchItem, Reveal};
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]), Default::default());
        let shown = |item: &BatchItem| done_texture(item).map(|t| t.id());
        let mut item = BatchItem::new_for_test();
        let result = tex("result");
        item.result_texture = Some(result.clone().into());
        assert_eq!(shown(&item), Some(result.id()));
        let frame = tex("frame");
        item.quick.expect(1);
        item.quick.land(frame.clone().into(), 1);
        assert_eq!(shown(&item), Some(frame.id()), "the frame, not the result");
        let held = tex("held");
        item.reveal = Some(Reveal::Held(held.clone().into()));
        assert_eq!(shown(&item), Some(held.id()));
    }

    #[test]
    fn only_points_on_the_image_become_source_pixels() {
        let img = Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(200.0, 100.0));
        let dims = (400, 200);
        assert_eq!(source_point(egui::pos2(100.0, 50.0), img, dims), Some((0.0, 0.0)));
        assert_eq!(source_point(egui::pos2(200.0, 100.0), img, dims), Some((200.0, 100.0)));
        assert_eq!(source_point(egui::pos2(300.0, 150.0), img, dims), Some((399.0, 199.0)), "the far edge stays in range");
        assert_eq!(source_point(egui::pos2(301.0, 100.0), img, dims), None, "beside the image");
        assert_eq!(source_point(egui::pos2(200.0, 10.0), img, dims), None, "above it");
    }
}
