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

    // Bg-image texture is built off-thread by `kick_bg_image_tex_prep`
    // (called from the file-pick + preset paths). The render path
    // reads `bg_image_texture` directly; if it's still pending the
    // existing bg-color fallback paints below.

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
    // The brush authors the selection against the source, so it is live
    // as soon as an image is loaded; segmentation corrections apply once
    // a result exists.
    //
    // While an inpaint stroke is in flight on the selected item, the
    // brush is locked: another stroke would silently supersede the
    // first via dispatch_inpaint's generation counter, wasting the
    // 5+ minutes of CPU work just performed. Visible signal is the
    // banner with "Cancel (Esc)" button rendered below.
    let inpaint_in_flight_for_selected = app.batch
        .selected_idx_clamped()
        .map(|idx| app.batch.items[idx].id)
        .is_some_and(|id| app.processor.is_inpaint_in_flight(id));
    let app_state = app.batch.app_state();
    let brush_active = app.brush_state.is_enabled()
        && !inpaint_in_flight_for_selected
        && matches!(app_state, AppState::Loaded | AppState::Done);
    // Magic Brush is treated as a brush tool for pan-gate purposes — when
    // it's active and the encoder is ready, left-drag belongs to the
    // brush and pan moves to secondary (right) button, matching Paint
    // Brush. Suppressed while encoder is pending so the "Preparing..."
    // overlay doesn't accidentally steal pan from a frustrated user.
    let magic_brush_active = app.magic_brush_state.is_active()
        && !app.magic_brush_state.has_pending_encoder()
        && matches!(app_state, AppState::Loaded | AppState::Done);
    let any_brush_tool_active = brush_active || magic_brush_active;
    // Scroll-zoom always works: it doesn't conflict with brush strokes
    // and the zoom feedback is reassuring even mid-painting.
    let canvas_gets_zoom = !modal_open && !popup_open && !widget_has_pointer;
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
    if let Some(tex) = app.batch.selected_item().and_then(|i| i.source_texture.as_ref()) {
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

    match app_state {
        AppState::Empty => render_empty(ui, app),
        AppState::Loaded => render_loaded(ui, app),
        AppState::Processing => render_processing_canvas(ui, app),
        AppState::Done => render_done(ui, app),
    }

    // Unified progress overlay — reads `Processor.dispatch_progress`
    // for seg / SD inpaint / upscale, falls back to synthesising from
    // the in-process LaMa inpaint state (the LaMa rayon path writes
    // `InpaintProgress` directly without going through the
    // subprocess bridge that populates the slot).
    if let Some(progress) = read_dispatch_progress(app) {
        let active_inpaint_item = app.batch.selected_idx_clamped()
            .map(|idx| app.batch.items[idx].id)
            .filter(|&id| app.processor.is_inpaint_in_flight(id));
        let cancelling = active_inpaint_item
            .map(|id| app.processor.is_inpaint_cancelling(id))
            .unwrap_or(false);
        let cancelled = match app.settings.progress_style {
            crate::gui::settings::ProgressStyle::Banner => {
                super::progress_widget::render_banner(ui, canvas_rect, &progress, cancelling)
            }
            crate::gui::settings::ProgressStyle::Modal => {
                super::progress_widget::render_modal(ui, canvas_rect, &progress, cancelling)
            }
        };
        if cancelled {
            // Routing logic is the pure `cancel_target_for` fn — one
            // table-tested match instead of an inline copy.
            use crate::gui::dispatch_progress::{cancel_target_for, CancelTarget};
            if let Some(target) = cancel_target_for(progress.kind, active_inpaint_item) {
                match target {
                    CancelTarget::InpaintForItem(id) => app.cancel_inpaint_for(id),
                    CancelTarget::Upscale => app.processor.cancel_upscale(),
                    CancelTarget::SegBatchAndReset => app.handle_cancel_all_and_reset(),
                }
            }
        }
        ui.ctx().request_repaint();
    }

    // Suppress brush input + cursor + trail when a popup or widget has
    // claimed the pointer — the brush popover floats over the canvas, so
    // its sliders sit inside `img_rect` geographically and would otherwise
    // get painted over and double-handle clicks.
    if brush_active && !modal_open && !popup_open && !widget_has_pointer {
        handle_brush_input(ui, app, canvas_rect);
    }

    // Magic Brush click/stroke input. Suppressed during encoder run (silently —
    // per UI-SPEC, clicks while encoder_pending are ignored without a toast).
    if app.magic_brush_state.is_active()
        && !app.magic_brush_state.has_pending_encoder()
        && !modal_open
        && !popup_open
        && !widget_has_pointer
    {
        handle_magic_brush_input(ui, app, canvas_rect);
    }

    // Magic Brush "Preparing..." overlay — shown while the SAM encoder is
    // in flight for the selected item. Canvas centre, TEXT_SECONDARY text +
    // ACCENT spinner below.
    if app.magic_brush_state.is_active() && app.magic_brush_state.has_pending_encoder() {
        // Cursor over the canvas reads as Wait while the encoder is in
        // flight (secondary cue to the centered "Preparing…" overlay).
        if ui.ctx().input(|i| i.pointer.hover_pos().is_some_and(|p| canvas_rect.contains(p))) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Wait);
        }
        let center = canvas_rect.center();
        ui.painter().text(
            center,
            egui::Align2::CENTER_CENTER,
            "Preparing\u{2026}",
            egui::FontId::proportional(14.0),
            theme::TEXT_SECONDARY,
        );
        egui::Area::new(egui::Id::new("magic_preparing_spinner"))
            .fixed_pos(center + egui::vec2(0.0, 24.0))
            .show(ui.ctx(), |ui| {
                ui.spinner();
            });
        // 10 Hz polling while the encoder runs — a 60 Hz busy-render of a
        // static spinner spins up the fan for no visible benefit.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }

    // Bottom-left modifier hint — shown whenever Magic Brush is the active
    // tool (regardless of encoder state) so users discover Shift / Alt
    // before they need them. Active modifier tints to ACCENT; the rest stays
    // TEXT_SECONDARY. Mono 12px per 33-UI-SPEC §Selection Visualization.
    if app.magic_brush_state.is_active() {
        let (shift, alt) = ui.ctx().input(|i| (i.modifiers.shift, i.modifiers.alt));
        let base_color = theme::TEXT_SECONDARY;
        let active_color = theme::ACCENT;
        let pos = egui::pos2(
            canvas_rect.min.x + theme::SPACE_SM,
            canvas_rect.max.y - theme::SPACE_SM - theme::FONT_SIZE_MONO,
        );
        let font = egui::FontId::monospace(theme::FONT_SIZE_MONO);
        let painter = ui.painter();
        let shift_color = if shift { active_color } else { base_color };
        let shift_rect = painter.text(
            pos, egui::Align2::LEFT_TOP, "Shift = add", font.clone(), shift_color,
        );
        let sep_rect = painter.text(
            egui::pos2(shift_rect.max.x, pos.y), egui::Align2::LEFT_TOP,
            "  ", font.clone(), base_color,
        );
        let alt_color = if alt { active_color } else { base_color };
        painter.text(
            egui::pos2(sep_rect.max.x, pos.y), egui::Align2::LEFT_TOP,
            "Alt = subtract", font, alt_color,
        );
    }
}

/// Run the brush overlay (cursor + pointer events) and commit any
/// finished stroke onto the active item. Pan is suppressed in this mode
/// — `render_done`'s pan logic already runs with `is_panning = false`
/// because brush mode disables the press tracking via the `canvas_gets_input`
/// gate.
fn handle_brush_input(ui: &mut egui::Ui, app: &mut PrunrApp, canvas_rect: Rect) {
    use super::brush_overlay::{self, BrushAction};
    let Some(item) = app.batch.selected_item() else { return };
    let Some(tex) = item.result_texture.as_ref().or(item.source_texture.as_ref()) else { return };
    let img_rect = compute_img_rect(canvas_rect, tex.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);

    let Some(idx) = app.batch.selected_idx_clamped() else { return };
    let stamp = if app.settings.model.is_inpaint() {
        app.settings.brush.region_stamp()
    } else {
        app.settings.brush.stamp()
    };
    let action = {
        let item = &app.batch.items[idx];
        brush_overlay::handle_input(ui, &mut app.brush_state, &app.settings.brush, stamp, item, img_rect)
    };
    if let BrushAction::Committed(stroke_mask) = action {
        let item_id = app.batch.items[idx].id;

        // Snapshot pre-state onto undo stack BEFORE writing the new mask.
        // begin_stroke_commit: pushes current selection_mask as pre-state,
        // clears redo stack, pushes Stroke marker on the action timeline.
        app.batch.items[idx].begin_stroke_commit();

        // Merge this stroke onto the existing selection: same-direction
        // overlap keeps the stronger stroke, an opposite-direction stroke
        // wins where it lands.
        let merged = match app.batch.items[idx].selection_mask.clone() {
            Some(existing) => existing.add_mask(&stroke_mask)
                .unwrap_or(stroke_mask),
            None => stroke_mask,
        };

        // Chain mode: archive the pre-stroke result before dispatch so
        // undo can swap stored RGBAs instantly. Mirrors the seg-path
        // archive from Phase 15; inpaint archives at result-land time.
        let is_inpaint = app.settings.model.is_inpaint();
        if !is_inpaint && app.settings.chain_mode && app.batch.items[idx].result_rgba.is_some() {
            let max_depth = app.settings.history_depth;
            let item = &mut app.batch.items[idx];
            if let Some(rgba) = item.result_rgba.as_ref().cloned() {
                item.history.push_back(crate::gui::item::HistoryEntry::new(
                    rgba, item.applied_recipe.clone(),
                ));
                while item.history.len() > max_depth {
                    if let Some(old) = item.history.pop_front() {
                        old.cleanup();
                    }
                }
                for entry in item.redo_stack.drain(..) {
                    entry.cleanup();
                }
            }
        }

        tracing::info!(item_id, "brush stroke committed; writing to selection_mask");
        // commit_selection_and_dispatch is the single-source-of-truth entry:
        // persists mask, fires per-model dispatch rule, spawns off-thread
        // outline + texture visualization build.
        app.commit_selection_and_dispatch(item_id, merged);
    }
}

/// Handle Magic Brush pointer events (click + drag-stroke) on the canvas.
/// Suppressed when encoder is pending — clicks during "Preparing..." are
/// silently ignored (per 33-UI-SPEC). Dispatches a SAM decoder job on
/// every click or completed stroke; modifier keys control mask combination.
///
/// Also owns the per-frame visual feedback for the tool: crosshair cursor
/// icon, brush-shape cursor outline (via `brush_overlay::draw_cursor`),
/// trail rendering during drag (via `brush_overlay::draw_trail_for`),
/// and the modifier glyph (+ / −) at the cursor when Shift / Alt is held.
fn handle_magic_brush_input(ui: &mut egui::Ui, app: &mut PrunrApp, canvas_rect: Rect) {
    use crate::gui::processor::PromptModifier;

    let Some(item) = app.batch.selected_item() else { return };
    let Some(tex) = item.result_texture.as_ref().or(item.source_texture.as_ref()) else { return };
    let img_rect = compute_img_rect(
        canvas_rect,
        tex.size_vec2(),
        app.zoom_state.zoom,
        app.zoom_state.pan_offset,
    );

    let Some(idx) = app.batch.selected_idx_clamped() else { return };
    let item = &app.batch.items[idx];
    let item_id = item.id;
    let (source_w, source_h) = item.dimensions;
    // Embedding may be None for one frame between encoder dispatch and
    // result pump. Cursor + trail should still render so the user gets
    // immediate feedback that Magic Brush is the active tool; only the
    // dispatch is gated on a real embedding.
    let embedding = item.magic_brush_embedding.clone();

    // Cursor icon: Crosshair while over the canvas, default elsewhere.
    let hover_pos_for_cursor = ui.ctx().input(|i| i.pointer.hover_pos());
    if hover_pos_for_cursor.is_some_and(|p| canvas_rect.contains(p)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    // Brush-shape cursor outline + in-progress trail. Same helpers Paint
    // Brush uses, reading the same BrushSettings — single source of truth
    // for size / shape / hardness / strength means a tweak in either
    // popover applies to both tools.
    let pointer_on_img = hover_pos_for_cursor.filter(|p| img_rect.contains(*p));
    super::brush_overlay::draw_trail_for(
        ui,
        &app.settings.brush,
        app.settings.brush.shape,
        app.magic_brush_state.active_trail.iter().copied(),
    );
    super::brush_overlay::draw_cursor(
        ui, img_rect, &app.settings.brush, pointer_on_img.is_some(),
    );

    // Determine modifier from held keys. Read once to avoid per-call InputState borrow.
    let modifier = ui.ctx().input(|i| {
        if i.modifiers.shift {
            PromptModifier::Add
        } else if i.modifiers.alt {
            PromptModifier::Subtract
        } else {
            PromptModifier::Replace
        }
    });

    // Modifier glyph at the cursor — `+` for Shift, `−` for Alt. Painted
    // below-right of the hover position per 33-UI-SPEC §"Modifier cursor
    // annotation" (offset matches the macOS modifier-indicator convention).
    if let Some(p) = hover_pos_for_cursor {
        if canvas_rect.contains(p) {
            let glyph = match modifier {
                PromptModifier::Add => Some("+"),
                PromptModifier::Subtract => Some("\u{2212}"), // U+2212 MINUS SIGN — visually heavier than ASCII hyphen.
                PromptModifier::Replace => None,
            };
            if let Some(g) = glyph {
                ui.painter().text(
                    p + egui::vec2(12.0, 12.0),
                    egui::Align2::CENTER_CENTER,
                    g,
                    egui::FontId::proportional(14.0),
                    theme::ACCENT,
                );
            }
        }
    }

    // Convert screen position to source-image pixel coordinates.
    let screen_to_src = |pos: egui::Pos2| -> (f32, f32) {
        if img_rect.width() <= 0.0 || img_rect.height() <= 0.0 {
            return (0.0, 0.0);
        }
        let rel_x = (pos.x - img_rect.min.x) / img_rect.width();
        let rel_y = (pos.y - img_rect.min.y) / img_rect.height();
        let px = rel_x * source_w as f32;
        let py = rel_y * source_h as f32;
        (px.clamp(0.0, source_w as f32 - 1.0), py.clamp(0.0, source_h as f32 - 1.0))
    };

    let (clicked, drag_started, dragging, released, hover_pos) = ui.ctx().input(|i| {
        let hover = i.pointer.hover_pos();
        let clicked = i.pointer.primary_clicked()
            && hover.is_some_and(|p| canvas_rect.contains(p));
        let drag_started = i.pointer.primary_pressed()
            && hover.is_some_and(|p| canvas_rect.contains(p));
        let dragging = i.pointer.primary_down()
            && hover.is_some_and(|p| canvas_rect.contains(p));
        let released = i.pointer.primary_released();
        (clicked, drag_started, dragging, released, hover)
    });

    // No embedding yet? Render cursor + trail above but skip dispatch —
    // the encoder must finish before a SAM prompt can run.
    let Some(embedding) = embedding else { return };

    let screen_radius = app.settings.brush.radius;
    let min_trail_step_sq = (screen_radius * 0.5).max(1.0).powi(2);

    if drag_started {
        app.magic_brush_state.clear_stroke();
    }

    if dragging {
        if let Some(pos) = hover_pos {
            if canvas_rect.contains(pos) {
                let (px, py) = screen_to_src(pos);
                let state = &mut app.magic_brush_state;
                // Dedup the source-coords list on bit-exact repeats; SAM
                // doesn't benefit from duplicate points and we cap at 8
                // anyway in build_stroke_prompt.
                if state.active_stroke.last() != Some(&(px, py)) {
                    state.active_stroke.push((px, py));
                }
                // Dedup the screen-coords trail on half-radius steps —
                // matches BrushState.record_trail_stamp's policy so the
                // two tools render visually identical trails.
                let push_trail = state.active_trail.last().is_none_or(|&(lx, ly, _)| {
                    let dx = pos.x - lx;
                    let dy = pos.y - ly;
                    dx * dx + dy * dy >= min_trail_step_sq
                });
                if push_trail {
                    state.active_trail.push((pos.x, pos.y, screen_radius));
                }
            }
        }
    }

    if released && app.magic_brush_state.active_stroke.len() > 1 {
        // Stroke completed — dispatch decoder with stroke prompt.
        let stroke_pts = std::mem::take(&mut app.magic_brush_state.active_stroke);
        app.magic_brush_state.active_trail.clear();
        match prunr_core::sam::prompt::build_stroke_prompt(
            &stroke_pts,
            source_w,
            source_h,
        ) {
            Ok(prompt) => {
                app.processor.dispatch_sam_decoder(sam_request(app, item_id, embedding, prompt, modifier));
            }
            Err(e) => {
                tracing::warn!(item_id, "SAM stroke prompt failed: {e:?}");
            }
        }
    } else if clicked {
        if let Some(pos) = hover_pos {
            let (px, py) = screen_to_src(pos);
            // The `.is_empty()` guard below distinguishes a real click from
            // the synthetic tail-click that fires at drag release.
            let prompt = match modifier {
                PromptModifier::Subtract => prunr_core::sam::prompt::build_alt_modifier_prompt(
                    px, py, source_w, source_h,
                ),
                PromptModifier::Add | PromptModifier::Replace => {
                    prunr_core::sam::prompt::build_click_prompt(
                        px, py, source_w, source_h,
                    )
                }
            };
            if app.magic_brush_state.active_stroke.is_empty() {
                app.processor.dispatch_sam_decoder(sam_request(app, item_id, embedding, prompt, modifier));
            }
        }
        app.magic_brush_state.clear_stroke();
    }
}

/// Snapshot everything a decoder run needs at click time: a chip change
/// while SAM runs must not re-polarise or re-threshold the result.
fn sam_request(
    app: &PrunrApp,
    item_id: u64,
    embedding: std::sync::Arc<prunr_core::sam::SamEmbedding>,
    prompt: prunr_core::sam::prompt::SamPrompt,
    modifier: crate::gui::processor::PromptModifier,
) -> crate::gui::processor::SamDecodeRequest {
    let source_dims = app.batch.find_by_id(item_id).map(|i| i.dimensions).unwrap_or((0, 0));
    crate::gui::processor::SamDecodeRequest {
        item_id,
        embedding,
        prompt,
        modifier,
        mode: app.settings.brush.mode,
        source_dims,
        confidence_threshold: app.settings.brush.magic_confidence_threshold,
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
    let m = super::KB_MOD;
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
    if let Some(texture) = app.batch.selected_item().and_then(|i| i.source_texture.as_ref()) {
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

        // Selection overlay — same call as `render_done`. Required here too
        // because Magic Brush authors selections directly on the source
        // image, before any Process click; without this the action bar
        // shows but the user sees no mask. See `33-UI-SPEC.md` §"Selection
        // Visualization — Render gating".
        if let Some(it) = app.batch.selected_item() {
            let (src_w, src_h) = it.dimensions;
            super::selection_overlay::render_selection_overlay(
                ui, it, &app.settings.brush, img_rect, src_w, src_h,
            );
        }

        // Empty-state hint — guides discovery when a brush tool is active
        // but no selection exists yet. Per 33-UI-SPEC §"Empty-state hint":
        // 50% alpha so it sits under the imagery without competing.
        let has_selection = app.batch
            .selected_item()
            .is_some_and(|i| i.selection_mask.is_some());
        if !has_selection {
            let hint = if app.magic_brush_state.is_active()
                && !app.magic_brush_state.has_pending_encoder()
            {
                Some("Click or stroke to select")
            } else if app.brush_state.is_enabled() {
                Some("Paint to select")
            } else {
                None
            };
            if let Some(text) = hint {
                let color = theme::TEXT_SECONDARY.linear_multiply(0.5);
                ui.painter().text(
                    img_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    text,
                    egui::FontId::monospace(theme::FONT_SIZE_MONO),
                    color,
                );
            }
        }
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

/// Paint the canvas-background "work in progress" effect: gentle
/// wiggle/breathing of the source image + horizontal shimmer sweep.
/// The progress pill that used to live here moved to the unified
/// `progress_widget` (banner / modal); the canvas effect stays as
/// its own visual cue.
fn render_processing_canvas(ui: &mut egui::Ui, app: &PrunrApp) {
    let canvas_rect = ui.available_rect_before_wrap();
    let t = ui.ctx().input(|i| i.time) as f32;

    // In chain mode with an existing result, show the result being processed
    // (not the original).
    let item = app.batch.selected_item();
    let result_tex = item.and_then(|i| i.result_texture.as_ref());
    let display_texture = if app.settings.chain_mode && result_tex.is_some() {
        result_tex
    } else {
        item.and_then(|i| i.source_texture.as_ref())
    };

    if let Some(texture) = display_texture {
        let tex_size = texture.size_vec2();
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

    ui.ctx().request_repaint_after(std::time::Duration::from_millis(66));
}

/// Synthesise a `DispatchProgress` for the canvas overlay. Reads the
/// unified slot first; if empty, falls back to the in-process LaMa
/// inpaint state for the selected item (LaMa's rayon dispatch writes
/// `InpaintProgress` directly and never publishes to the slot).
fn read_dispatch_progress(app: &PrunrApp) -> Option<crate::gui::dispatch_progress::DispatchProgress> {
    if let Some(slot) = app.processor.dispatch_progress() {
        return Some(slot);
    }

    let idx = app.batch.selected_idx_clamped()?;
    let item_id = app.batch.items[idx].id;
    if !app.processor.is_inpaint_in_flight(item_id) {
        return None;
    }
    let ((oc, ot), inner) = app.processor.inpaint_progress_nested(item_id);
    Some(crate::gui::dispatch_progress::DispatchProgress::lama_inpaint(oc, ot, inner))
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

    let item = app.batch.selected_item();
    if app.show_original {
        if let Some(texture) = item.and_then(|i| i.source_texture.as_ref()) {
            let img_rect =
                compute_img_rect(canvas_rect, texture.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);
            ui.painter().image(
                texture.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            // Selection overlay also renders in show_original — the user
            // is reviewing the source pre-mask, and the selection is
            // authored against source coordinates, so it should be visible
            // here too. Per 33-UI-SPEC §"Selection Visualization — Render
            // gating".
            if let Some(it) = item {
                let (src_w, src_h) = it.dimensions;
                super::selection_overlay::render_selection_overlay(
                    ui, it, &app.settings.brush, img_rect, src_w, src_h,
                );
            }
        }
    } else if let Some(result_tex) = item.and_then(|i| i.result_texture.as_ref()) {
        let img_rect =
            compute_img_rect(canvas_rect, result_tex.size_vec2(), app.zoom_state.zoom, app.zoom_state.pan_offset);

        // During crossfade: show source fading out behind checkerboard + result fading in
        if fade < 1.0 {
            if let Some(source_tex) = item.and_then(|i| i.source_texture.as_ref()) {
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
            // Image bg wins over color bg when both are set — they're
            // mutually exclusive in the UI but a stale bg color could
            // remain on the recipe.
            if let Some(it) = app.batch.selected_item() {
                if let Some(tex) = it.bg_image_texture.as_ref() {
                    paint_bg_image(ui, img_rect, tex, it.settings.bg_image_fit);
                } else if let Some(bg) = it.settings.bg {
                    ui.painter().rect_filled(
                        img_rect,
                        0.0,
                        Color32::from_rgba_unmultiplied(bg[0], bg[1], bg[2], bg[3]),
                    );
                }
            }
        }
        let result_alpha = (fade * 255.0) as u8;
        ui.painter().image(
            result_tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, result_alpha),
        );

        // Selection overlay: outline + fill, zero-alloc read of pre-built texture +
        // polyline. render_selection_overlay is RENDER-ONLY — no I/O, no decode,
        // no GPU upload (all pre-computed via background_io channels).
        if let Some(it) = app.batch.selected_item() {
            let (src_w, src_h) = it.dimensions;
            super::selection_overlay::render_selection_overlay(
                ui, it, &app.settings.brush, img_rect, src_w, src_h,
            );
        }
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

/// Paint `tex` into `bounds` per `fit` — UV-only math (no texture
/// re-upload; the texture must be uploaded with `WrapMode::Repeat` for
/// `Tile` to wrap rather than clamp).
pub(super) fn paint_bg_image(
    ui: &egui::Ui,
    bounds: Rect,
    tex: &TextureHandle,
    fit: prunr_core::BgImageFit,
) {
    use prunr_core::BgImageFit;
    let tex_size = tex.size_vec2();
    let (tw, th) = (tex_size.x, tex_size.y);
    let (bw, bh) = (bounds.width(), bounds.height());
    if tw <= 0.0 || th <= 0.0 || bw <= 0.0 || bh <= 0.0 {
        return;
    }
    let unit_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
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
            ui.painter().image(tex.id(), bounds, Rect::from_min_max(uv_min, uv_max), Color32::WHITE);
        }
        BgImageFit::Stretch => {
            ui.painter().image(tex.id(), bounds, unit_uv, Color32::WHITE);
        }
        BgImageFit::Contain => {
            let scale = (bw / tw).min(bh / th);
            let inner = egui::vec2(tw * scale, th * scale);
            let inner_rect = Rect::from_center_size(bounds.center(), inner);
            ui.painter().image(tex.id(), inner_rect, unit_uv, Color32::WHITE);
        }
        BgImageFit::Center => {
            // 1:1 native size centred. May overflow `bounds` but the
            // canvas painter clips to the painter's clip rect (egui's
            // panel clipping handles overflow).
            let inner_rect = Rect::from_center_size(bounds.center(), egui::vec2(tw, th));
            ui.painter().image(tex.id(), inner_rect, unit_uv, Color32::WHITE);
        }
        BgImageFit::Tile => {
            // UV scaled by bounds/tex_size triggers WrapMode::Repeat in
            // egui — the bg texture is uploaded with that wrap mode set.
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(bw / tw, bh / th));
            ui.painter().image(tex.id(), bounds, uv, Color32::WHITE);
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
