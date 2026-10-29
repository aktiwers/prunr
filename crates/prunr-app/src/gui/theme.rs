/// The app's fonts: egui's, with the icon font, and the monospace font
/// behind the body font for the symbols only it has (the arrow keys' ←
/// and →).
pub(crate) fn install_fonts(ctx: &egui::Context) {
    const FALLBACK: &str = "Hack";
    egui_material_icons::initialize(ctx);
    if let Some(data) = egui::FontDefinitions::default().font_data.get(FALLBACK) {
        ctx.add_font(egui::epaint::text::FontInsert::new(
            "body-fallback",
            (**data).clone(),
            vec![egui::epaint::text::InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Lowest,
            }],
        ));
    }
}

/// Whether an egui popup (chip popover, combo box, color picker) is open.
/// The legacy popup API is the one the chip popovers use; the deprecation
/// is isolated here.
#[allow(deprecated)]
pub(crate) fn any_popup_open(ctx: &egui::Context) -> bool {
    ctx.memory(|m| m.any_popup_open())
}

/// The popup ids that are flyouts, so the canvas can tell them from
/// popovers. egui only knows "a popup is open".
#[derive(Clone, Default)]
struct Flyouts(std::sync::Arc<std::sync::Mutex<std::collections::HashSet<egui::Id>>>);

fn flyouts(ctx: &egui::Context) -> Flyouts {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Flyouts>(egui::Id::new("flyout_ids")).clone())
}

pub(crate) fn register_flyout(ctx: &egui::Context, id: egui::Id) {
    flyouts(ctx).0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(id);
}

/// Close every open flyout: for a click the whole image must take.
pub(crate) fn close_flyouts(ctx: &egui::Context) {
    let flyouts = flyouts(ctx);
    let ids = flyouts.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for id in ids.iter() {
        egui::Popup::close_id(ctx, *id);
    }
}

/// Whether the open popup is a flyout the canvas stays live under.
pub(crate) fn pinned_popup_open(ctx: &egui::Context) -> bool {
    let flyouts = flyouts(ctx);
    let ids = flyouts.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    ids.iter().any(|id| egui::Popup::is_id_open(ctx, *id))
}

/// A popup that closes on a click outside is open: the canvas holds
/// still so that click, and the drag after it, is spent on the dismissal.
pub(crate) fn dismissable_popup_open(ctx: &egui::Context) -> bool {
    any_popup_open(ctx) && !pinned_popup_open(ctx)
}

use egui::{Color32, Stroke};

/// The dimming over the app behind a modal.
const MODAL_SCRIM: Color32 = Color32::from_rgba_unmultiplied_const(0, 0, 0, 56);

/// Check if the user clicked the backdrop (outside the modal window).
/// Must be called AFTER the window is rendered so we know the window rect.
/// One click closes one modal, the topmost, as one Escape does.
pub fn backdrop_clicked(ctx: &egui::Context, window_response: &Option<egui::InnerResponse<Option<()>>>) -> bool {
    let clicked = ctx.input(|i| i.pointer.primary_clicked());
    if !clicked { return false; }

    if any_popup_open(ctx) { return false; }

    let click_pos = ctx.input(|i| i.pointer.interact_pos());
    if let (Some(pos), Some(resp)) = (click_pos, window_response) {
        !resp.response.rect.contains(pos) && modal_on_top(ctx, resp.response.layer_id)
    } else {
        false
    }
}

/// The window layers of the modals shown lately, with the pass each was
/// last shown in: a modal shown before another in a pass learns of it
/// from the pass before.
fn modal_layers() -> egui::Id {
    egui::Id::new("prunr_modal_layers")
}

fn note_modal_shown(ctx: &egui::Context, layer: egui::LayerId) {
    let pass = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        let shown: &mut Vec<(egui::LayerId, u64)> = d.get_temp_mut_or_default(modal_layers());
        shown.retain(|&(other, at)| other != layer && at + 1 >= pass);
        shown.push((layer, pass));
    });
}

/// No modal shown lately sits over the one on `layer`.
fn modal_on_top(ctx: &egui::Context, layer: egui::LayerId) -> bool {
    let pass = ctx.cumulative_pass_nr();
    let shown: Vec<(egui::LayerId, u64)> = ctx.data(|d| d.get_temp(modal_layers())).unwrap_or_default();
    let order: Vec<egui::LayerId> = ctx.memory(|m| m.layer_ids().collect());
    let place = |l: egui::LayerId| order.iter().position(|&o| o == l);
    let own = place(layer);
    !shown.iter().any(|&(other, at)| other != layer && at + 1 >= pass && place(other) > own)
}


// === Building blocks ===
// Everything floats like the progress pill: dark glass, a hairline border,
// a soft deep shadow. Panels round at 18 px, strips and buttons are pills,
// inputs round at 10 px.

/// The glass fill: a raised surface (#2A2A2E) at 94 %, lighter than the
/// toolbar it floats under.
pub const GLASS: Color32 = Color32::from_rgba_premultiplied(40, 40, 43, 240);
/// The hairline border around glass: white at 8 %.
pub const HAIRLINE: Color32 = Color32::from_rgba_premultiplied(20, 20, 20, 20);
/// A slider's rail, an off switch, the progress ring's track.
pub const RAIL: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x46);
/// The Erase colour of brushes and strokes.
pub const ERASE: Color32 = Color32::from_rgb(0xe2, 0x70, 0x4f);
/// A field, a segmented control's track, a round icon button: white at 6 %.
pub const FIELD_FILL: Color32 = Color32::from_rgba_premultiplied(15, 15, 15, 15);
pub const FIELD_FILL_HOVER: Color32 = Color32::from_rgba_premultiplied(24, 24, 24, 24);
/// A list row under the pointer.
pub const ROW_HOVER: Color32 = Color32::from_rgba_premultiplied(10, 10, 10, 10);
/// The purple wash behind the picked row of a list.
pub const ROW_PICKED: Color32 = Color32::from_rgba_premultiplied(31, 16, 42, 51);
/// A card or a group of rows: a faint lift with a faint edge.
pub const CARD_FILL: Color32 = Color32::from_rgba_premultiplied(9, 9, 9, 9);
pub const CARD_EDGE: Color32 = Color32::from_rgba_premultiplied(13, 13, 13, 13);

/// egui's own animation time, while animations are on.
const ANIMATION_TIME: f32 = 1.0 / 12.0;

/// A moment the decorative effects hold at while animations are off: the
/// working glow half lit, its sheen off the image.
const STILL_MOMENT: f32 = 5.7;

/// Settings › Animations, carried as egui's animation time so egui's own
/// fades and slides stop with ours. Returns whether it changed.
pub fn apply_motion(ctx: &egui::Context, on: bool) -> bool {
    let changed = motion(ctx) != on;
    if changed {
        ctx.global_style_mut(|s| s.animation_time = if on { ANIMATION_TIME } else { 0.0 });
    }
    changed
}

/// Decorative motion (reveals, glows, sweeps) is on.
pub fn motion(ctx: &egui::Context) -> bool {
    ctx.global_style().animation_time > 0.0
}

/// The clock decorative motion runs on: the time, or one still moment
/// while animations are off. Spinners that say the app is busy keep the
/// real time.
pub fn motion_clock(ctx: &egui::Context) -> f32 {
    if motion(ctx) { ctx.input(|i| i.time) as f32 } else { STILL_MOMENT }
}

/// How opaque a flyout stays while one of its sliders is held: clear
/// enough to watch the image change under it, solid enough to read.
pub const SCRUB_OPACITY: f32 = 0.7;

/// Frame pace of animations that last a whole run (capsule, working
/// canvas, busy pills): 30 fps keeps a ring smooth at half the drawing
/// of 60, which a laptop on battery feels over minutes.
pub const WORK_FRAME: std::time::Duration = std::time::Duration::from_millis(33);

/// How often the slow ambient animations draw (the processing canvas,
/// the sidebar's sweeps and pending thumbnails) and a download's or an
/// install's progress: about fifteen a second.
pub const ANIM_FRAME: std::time::Duration = std::time::Duration::from_millis(66);

/// A thumbnail's fade-in once its picture is built.
pub const IMAGE_FADE_SECS: f32 = 0.2;

/// A fade's length: `secs`, or none while animations are off.
pub fn fade_secs(ctx: &egui::Context, secs: f32) -> f32 {
    if motion(ctx) { secs } else { 0.0 }
}

/// What a brush mark is drawn in: Erase's warm red, else the accent.
pub fn mark_color(erase: bool) -> Color32 {
    if erase { ERASE } else { ACCENT }
}
/// Erase picked in a mode switch: the warm red a shade deeper, under white text.
pub const ERASE_PICKED: Color32 = Color32::from_rgb(0xc9, 0x57, 0x3a);
/// A hint under its row: smaller and greyer than a label.
pub const HINT: Color32 = Color32::from_rgb(0x9c, 0x9c, 0x9c);
pub const PANEL_RADIUS: u8 = 18;
pub const PILL_RADIUS: u8 = 255;
pub const INPUT_RADIUS: u8 = 10;

/// The deep soft shadow under every floating surface.
pub fn glass_shadow() -> egui::epaint::Shadow {
    egui::epaint::Shadow { offset: [0, 14], blur: 36, spread: 0, color: Color32::from_black_alpha(128) }
}

/// A floating panel: modals, popovers, flyouts.
pub fn panel_frame() -> egui::Frame {
    egui::Frame {
        fill: GLASS,
        stroke: Stroke::new(STROKE_DEFAULT, HAIRLINE),
        corner_radius: egui::CornerRadius::same(PANEL_RADIUS),
        inner_margin: egui::Margin::same(SPACE_MD as i8),
        shadow: glass_shadow(),
        ..Default::default()
    }
}

/// A floating pill: the tool strip, the quick strips, the progress pill.
pub fn pill_frame() -> egui::Frame {
    egui::Frame {
        corner_radius: egui::CornerRadius::same(PILL_RADIUS),
        inner_margin: egui::Margin { left: 18, right: 8, top: 8, bottom: 8 },
        ..panel_frame()
    }
}

/// The app-wide look of the building blocks: glass popups and windows,
/// purple selections and slider fills on a thin rail, round inputs.
pub fn apply_building_blocks(visuals: &mut egui::Visuals, spacing: &mut egui::style::Spacing) {
    visuals.window_fill = GLASS;
    visuals.window_stroke = Stroke::new(STROKE_DEFAULT, HAIRLINE);
    visuals.window_corner_radius = egui::CornerRadius::same(PANEL_RADIUS);
    visuals.menu_corner_radius = egui::CornerRadius::same(PANEL_RADIUS);
    spacing.menu_margin = egui::Margin::same(12);
    // Every button is a pill: its label needs room from the rounded ends.
    spacing.button_padding = egui::vec2(12.0, 4.0);
    visuals.window_shadow = glass_shadow();
    visuals.popup_shadow = glass_shadow();
    visuals.selection.bg_fill = ACCENT_BRIGHT;
    visuals.hyperlink_color = ACCENT_RING;
    visuals.slider_trailing_fill = true;
    visuals.handle_shape = egui::style::HandleShape::Circle;
    for w in [&mut visuals.widgets.noninteractive, &mut visuals.widgets.inactive, &mut visuals.widgets.hovered, &mut visuals.widgets.active, &mut visuals.widgets.open] {
        w.corner_radius = egui::CornerRadius::same(INPUT_RADIUS);
    }
    visuals.widgets.inactive.bg_fill = RAIL;
    spacing.slider_rail_height = 4.0;
    // A slim, translucent thumb with no rail behind it: egui's default
    // 10 px white bar reads as a second panel edge on the dark glass.
    spacing.scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        bar_outer_margin: 2.0,
        active_background_opacity: 0.0,
        interact_background_opacity: 0.0,
        active_handle_opacity: 0.35,
        interact_handle_opacity: 0.6,
        ..egui::style::ScrollStyle::floating()
    };
}

/// Tighten egui's default visuals for a modal: subdued borders + the
/// button-fill colors used elsewhere in the app. Call once per frame
/// inside the modal's body.
pub fn apply_modal_visuals(ui: &mut egui::Ui) {
    let vis = ui.visuals_mut();
    vis.widgets.inactive.bg_stroke =
        Stroke::new(STROKE_DEFAULT, Color32::from_rgb(0x60, 0x60, 0x60));
    vis.widgets.hovered.bg_stroke =
        Stroke::new(STROKE_DEFAULT, Color32::from_rgb(0x80, 0x80, 0x80));
    vis.widgets.inactive.bg_fill = WIDGET_INACTIVE_BG;
    vis.widgets.inactive.fg_stroke = Stroke::new(STROKE_DEFAULT, TEXT_PRIMARY);
    vis.widgets.hovered.bg_fill = WIDGET_HOVER_BG;
}

/// Without it, the click that opens a modal also lands outside its rect
/// and closes it on the first frame.
pub const MODAL_BACKDROP_DEBOUNCE_SECS: f64 = 0.2;

/// What a modal keeps clear of the window's edges when it has to shrink.
const MODAL_SCREEN_MARGIN: f32 = 16.0;
const MODAL_MIN_SIZE: f32 = 240.0;

/// `true` when the user closed the modal this frame (X or backdrop).
/// `id` must be unique across modals: it keys the backdrop layer, the
/// Window and the debounce memory.
pub fn standard_modal_window(
    ctx: &egui::Context,
    id: &str,
    title: &str,
    size: [f32; 2],
    body: impl FnOnce(&mut egui::Ui),
) -> bool {
    let window = egui::Window::new(title).collapsible(false).frame(panel_frame());
    show_modal(ctx, id, window, size, false, body)
}

/// A modal like `standard_modal_window` with no title bar and its frame
/// flush to the edges: the body draws its own header (a sidebar, a close
/// button) and returns true when its close was clicked.
pub fn bare_modal_window(
    ctx: &egui::Context,
    id: &str,
    title: &str,
    size: [f32; 2],
    body: impl FnOnce(&mut egui::Ui) -> bool,
) -> bool {
    bare_modal_window_holding(ctx, id, title, size, false, body)
}

/// A bare modal that, while `holds`, holds the window: a click anywhere
/// else lands on nothing, and neither it nor the body's close closes it.
pub fn bare_modal_window_holding(
    ctx: &egui::Context,
    id: &str,
    title: &str,
    size: [f32; 2],
    holds: bool,
    body: impl FnOnce(&mut egui::Ui) -> bool,
) -> bool {
    // Opaque: a page of text over a busy photo should not let it through.
    let window = egui::Window::new(title).title_bar(false).frame(panel_frame().inner_margin(0).fill(MODAL_FILL));
    let mut close_clicked = false;
    let closed = show_modal(ctx, id, window, size, holds, |ui| close_clicked = body(ui));
    !holds && (closed || close_clicked)
}

/// A bare modal that holds the window while it is up. The body says how
/// it ends.
pub fn holding_modal_window(ctx: &egui::Context, id: &str, title: &str, size: [f32; 2], body: impl FnOnce(&mut egui::Ui)) {
    bare_modal_window_holding(ctx, id, title, size, true, |ui| {
        body(ui);
        false
    });
}

fn show_modal(
    ctx: &egui::Context,
    id: &str,
    window: egui::Window<'_>,
    size: [f32; 2],
    holds: bool,
    body: impl FnOnce(&mut egui::Ui),
) -> bool {
    let opened_key = egui::Id::new(id).with("opened_at");
    let now = ctx.input(|i| i.time);
    let pass = ctx.cumulative_pass_nr();
    // (opened at, last pass shown). A modal missing from the last pass was
    // closed some other way (its shortcut, Escape) and opens now: the
    // click that opened it must not count as a click on its backdrop.
    let opened_at = ctx.memory_mut(|m| {
        let shown = m.data.get_temp_mut_or_insert_with(opened_key, || (now, pass));
        if shown.1 + 1 < pass {
            shown.0 = now;
        }
        shown.1 = pass;
        shown.0
    });
    let debounce_passed = (now - opened_at) >= MODAL_BACKDROP_DEBOUNCE_SECS;

    // An area on the window's own level with the window as its sublayer,
    // which egui draws directly after it: the modal over the dimming over
    // the rest. (A bare painter layer is drawn after every area, on top.)
    let screen = ctx.content_rect();
    let backdrop = egui::Area::new(egui::Id::new(id).with("backdrop"))
        .order(egui::Order::Middle)
        .fixed_pos(screen.min)
        .interactable(holds)
        .show(ctx, |ui| {
            ui.painter().rect_filled(screen, 0.0, MODAL_SCRIM);
            if holds {
                ui.allocate_rect(screen, egui::Sense::click());
            }
        })
        .response
        .layer_id;

    // A short window: the modal shrinks to fit it, and its page scrolls.
    let room = screen.size() - egui::vec2(2.0 * MODAL_SCREEN_MARGIN, 2.0 * MODAL_SCREEN_MARGIN);
    let size = [size[0].min(room.x).max(MODAL_MIN_SIZE), size[1].min(room.y).max(MODAL_MIN_SIZE)];
    let mut open = true;
    let resp = window
        .id(egui::Id::new(id))
        .open(&mut open)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size(size)
        .show(ctx, body);
    if let Some(shown) = &resp {
        ctx.set_sublayer(backdrop, shown.response.layer_id);
        note_modal_shown(ctx, shown.response.layer_id);
    }

    !open || (debounce_passed && backdrop_clicked(ctx, &resp))
}

// === Colors (plum logo palette) ===

/// Main window/canvas background — dark charcoal from logo bg
pub const BG_PRIMARY: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1e);

/// Toolbar and status bar background
pub const BG_SECONDARY: Color32 = Color32::from_rgb(0x26, 0x24, 0x28);

/// The accent lifted for marks drawn over a photo, where plum is too dark.
pub const ACCENT_BRIGHT: Color32 = Color32::from_rgb(0x9b, 0x4f, 0xd1);

/// Lavender light: the accent lifted toward white, for glows over a photo.
pub const GLOW: Color32 = Color32::from_rgb(0xe6, 0xd2, 0xfa);
/// The ring around a picked tile.
pub const ACCENT_RING: Color32 = Color32::from_rgb(0xb2, 0x7a, 0xe0);

/// Amber for a run that is stopping.
pub const CAUTION: Color32 = Color32::from_rgb(0xe8, 0xa8, 0x4c);
/// Amber text, softer than `CAUTION` so a line of it stays readable.
pub const CAUTION_TEXT: Color32 = Color32::from_rgb(0xc9, 0xa7, 0x7a);
/// Text for a choice that works with a caveat (a model that runs, but
/// slowly or in tiles).
pub const ADVISORY_TEXT: Color32 = Color32::from_rgb(0xe1, 0xb3, 0x4e);

// Floating overlays over the canvas: the progress capsule and its pills.
pub const OVERLAY_CAPSULE_FILL: Color32 = Color32::from_rgba_premultiplied(18, 18, 22, 214);
pub const OVERLAY_BUTTON_FILL: Color32 = Color32::from_rgba_premultiplied(18, 18, 18, 18);
pub const OVERLAY_BUTTON_HOVER_FILL: Color32 = Color32::from_rgba_premultiplied(36, 36, 36, 36);
pub const KEYCAP_BORDER: Color32 = Color32::from_rgb(0x52, 0x52, 0x5b);
/// Tiles of the tile map still to come: dimmed, with a faint grid.
pub const TILE_WAITING: Color32 = Color32::from_rgba_premultiplied(9, 9, 11, 140);
pub const TILE_GRID: Color32 = Color32::from_rgba_premultiplied(13, 13, 13, 13);

/// Primary accent — plum purple (buttons, selections, progress)
pub const ACCENT: Color32 = Color32::from_rgb(0x7b, 0x2d, 0x8e);

/// Secondary accent — leaf green (done/success states)
pub const ACCENT_GREEN: Color32 = Color32::from_rgb(0x5b, 0x8c, 0x3e);

/// Destructive/error
pub const DESTRUCTIVE: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xf0, 0xf0, 0xf0);

pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xa8, 0xa8, 0xa8);

/// Checkerboard colors for light mode: (lighter square, darker square).
pub const CHECKER_LIGHT_MODE: (Color32, Color32) = (
    Color32::from_rgb(0xcc, 0xcc, 0xcc),
    Color32::from_rgb(0x88, 0x88, 0x88),
);

/// Checkerboard colors for dark mode: (lighter square, darker square).
/// Used when viewing light-colored subjects against the transparency pattern.
pub const CHECKER_DARK_MODE: (Color32, Color32) = (
    Color32::from_rgb(0x2a, 0x2a, 0x2a),
    Color32::from_rgb(0x1f, 0x1f, 0x1f),
);

pub const PROGRESS_FILL: Color32 = ACCENT;

/// The sweep over an image being processed, on the canvas and its thumbnail:
/// `ACCENT` at about 20% opacity.
pub const PROCESSING_SWEEP: Color32 = Color32::from_rgba_premultiplied(24, 9, 28, 50);

/// Drop zone border on drag hover
pub const DROP_HOVER_BORDER: Color32 = ACCENT;

pub const DROP_BORDER: Color32 = Color32::from_rgba_unmultiplied_const(255, 255, 255, 64);

// === Spacing (multiples of 4) ===

pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 16.0;
pub const SPACE_LG: f32 = 24.0;

// === Layout ===

pub const TOOLBAR_HEIGHT: f32 = 48.0;
pub const STATUS_BAR_HEIGHT: f32 = 24.0;
pub const PROGRESS_BAR_HEIGHT: f32 = 4.0;
pub const PROGRESS_BAR_BG: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
/// Buttons and chips are pills.
pub const BUTTON_ROUNDING: f32 = PILL_RADIUS as f32;
pub const DROP_ZONE_ROUNDING: f32 = 8.0;
pub const DROP_ZONE_BORDER_WIDTH: f32 = 2.0;

/// Toolbar button height (Open, Settings, Process, Undo/Redo icons).
pub const BTN_HEIGHT: f32 = 32.0;
/// Chip / compact-button height on the adjustments rows and the CLI help
/// tab strip.
pub const CHIP_HEIGHT: f32 = 28.0;
/// A key slot in Settings › Shortcuts, wide enough for "Ctrl+Shift+F12".
pub const HOTKEY_SLOT_WIDTH: f32 = 110.0;
/// Shared popover width for chip / preset dropdowns.
pub const POPOVER_WIDTH: f32 = 260.0;
/// Wider popover for chips that host multi-line text (prompt + negative).
/// Sized so a 28-char default-font line renders without horizontal wrap.
pub const POPOVER_WIDTH_WIDE: f32 = 480.0;
/// Default stroke width for widget borders and accent outlines.
pub const STROKE_DEFAULT: f32 = 1.0;
/// Icon glyph size on toolbar icon-buttons (settings gear, undo, redo).
pub const ICON_SIZE_BUTTON: f32 = 20.0;
/// Smaller icon glyph size — secondary-action buttons like reset-all.
pub const ICON_SIZE_SMALL: f32 = 18.0;

/// Inactive widget fill and the app-wide hovered widget border.
pub const WIDGET_INACTIVE_BG: Color32 = Color32::from_rgb(0x50, 0x50, 0x50);
/// Modal widget hover fill — one step brighter than inactive.
pub const WIDGET_HOVER_BG: Color32 = Color32::from_rgb(0x5a, 0x5a, 0x5a);

/// Accent at ~40% opacity for disabled buttons
/// Premultiplied: 0x7b*102/255=49, 0x2d*102/255=18, 0x8e*102/255=57
pub const ACCENT_DISABLED: Color32 = Color32::from_rgba_premultiplied(49, 18, 57, 102);

// === Window ===

pub const DEFAULT_WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];
// Floor set by the widest floating surface: the Mask quick strip (~860 px)
// and Settings. The toolbar shows its chips as icons when narrow and fits
// down to 640 (the `narrowest_toolbar` test).
pub const MIN_WINDOW_SIZE: [f32; 2] = [900.0, 540.0];

// === Typography (sizes for egui TextStyle overrides) ===

// The type scale: five steps.
pub const FONT_SIZE_CAPTION: f32 = 11.0;
pub const FONT_SIZE_SECONDARY: f32 = 12.0;
pub const FONT_SIZE_BODY: f32 = 14.0;
pub const FONT_SIZE_MODAL_TITLE: f32 = 20.0;

/// A list row's corners (pickers, menus, choice lists).
pub const ROW_ROUNDING: f32 = 12.0;
/// A bare modal page's margin on every side.
pub const MODAL_PAGE_MARGIN: egui::Margin = egui::Margin::same(24);
pub const FONT_SIZE_HEADING: f32 = 16.0;

// === Checkerboard ===

pub const CHECKER_SIZE: f32 = 16.0;

// === Logo ===
pub const LOGO_ASPECT: f32 = 416.0 / 512.0;

// === Sidebar ===
pub const SIDEBAR_WIDTH: f32 = 170.0;
pub const THUMBNAIL_SIZE: f32 = 120.0;
pub const THUMBNAIL_ROUNDING: f32 = 4.0;

// === Modals ===
pub const STANDARD_MODAL_WIDTH: f32 = 600.0;
/// The glass panel's colour at full cover, for a modal page of text.
pub const MODAL_FILL: Color32 = Color32::from_rgb(42, 42, 46);
/// Settings: its sidebar and the page beside it.
pub const SETTINGS_MODAL_SIZE: [f32; 2] = [860.0, 600.0];

pub const CLI_HELP_DIALOG_HEIGHT: f32 = 700.0;

/// Hint text for settings modal (readable on dark overlay)
pub const TEXT_HINT: Color32 = Color32::from_rgb(0xb8, 0xb8, 0xb8);

// === Sidebar Colors ===
pub const SIDEBAR_ITEM_BG: Color32 = Color32::from_rgb(0x26, 0x24, 0x28);
pub const SIDEBAR_ITEM_SELECTED: Color32 = Color32::from_rgb(0x32, 0x28, 0x36);
pub const SIDEBAR_SELECTED_BORDER: Color32 = ACCENT;
pub const STATUS_ICON_PENDING: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
pub const INSERTION_LINE: Color32 = ACCENT;


// === Zoom ===
pub const ZOOM_MIN: f32 = 0.10;
pub const ZOOM_MAX: f32 = 20.0;
pub const ZOOM_STEP: f32 = 1.1;

#[cfg(test)]
mod flyout_tests {
    use super::*;

    #[test]
    fn close_flyouts_closes_every_registered_flyout_and_nothing_else() {
        let ctx = egui::Context::default();
        let (strip, popover) = (egui::Id::new("strip"), egui::Id::new("popover"));
        register_flyout(&ctx, strip);
        // egui holds one open popup at a time, so each case opens its own.
        egui::Popup::open_id(&ctx, strip);
        assert!(pinned_popup_open(&ctx));
        close_flyouts(&ctx);
        assert!(!egui::Popup::is_id_open(&ctx, strip), "the flyout is closed");
        egui::Popup::open_id(&ctx, popover);
        close_flyouts(&ctx);
        assert!(egui::Popup::is_id_open(&ctx, popover), "a popover is left alone");
    }
}

#[cfg(test)]
mod motion_tests {
    use super::*;

    /// With animations off, egui's animations stop and the decorative
    /// effects hold still; on again, both move.
    #[test]
    fn animations_off_holds_every_effect_still() {
        let ctx = egui::Context::default();
        apply_motion(&ctx, false);
        assert!(!motion(&ctx));
        assert_eq!(ctx.global_style().animation_time, 0.0, "egui's own fades stop too");
        assert_eq!(motion_clock(&ctx), STILL_MOMENT);
        assert_eq!(fade_secs(&ctx, 0.2), 0.0);
        apply_motion(&ctx, true);
        assert!(motion(&ctx));
        assert_eq!(fade_secs(&ctx, 0.2), 0.2);
    }
}

#[cfg(test)]
mod modal_tests {
    use super::*;

    /// The dimming goes under the modal, not over it.
    #[test]
    fn the_backdrop_is_painted_under_the_modal() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))),
            ..Default::default()
        };
        let mut out = None;
        for _ in 0..3 {
            out = Some(ctx.run_ui(input.clone(), |ui| {
                standard_modal_window(ui.ctx(), "modal_test", "Test", [200.0, 100.0], |ui| {
                    ui.label("inside the modal");
                });
            }));
        }
        let shapes = out.expect("ran").shapes;
        let is_backdrop = |s: &egui::epaint::ClippedShape| matches!(&s.shape,
            egui::Shape::Rect(r) if r.rect.width() >= 800.0 && r.fill.a() > 0 && r.fill.r() == 0);
        let is_text = |s: &egui::epaint::ClippedShape| matches!(&s.shape, egui::Shape::Text(_));
        let backdrop = shapes.iter().position(is_backdrop).expect("a backdrop is painted");
        let text = shapes.iter().position(is_text).expect("the modal's text is painted");
        assert!(backdrop < text, "backdrop at {backdrop}, modal text at {text}: the backdrop covers the modal");
    }
}
