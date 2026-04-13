use egui::{Color32, Stroke};

/// Draw a semi-transparent backdrop behind a modal.
/// Call `backdrop_clicked` after the modal window is rendered to check if
/// the user clicked the backdrop (outside the window).
pub fn draw_modal_backdrop(ctx: &egui::Context, id: &str) {
    let screen_rect = ctx.content_rect();
    let layer_id = egui::LayerId::new(egui::Order::Foreground, egui::Id::new(id));
    let painter = ctx.layer_painter(layer_id);
    painter.rect_filled(screen_rect, 0.0,
        Color32::from_rgba_unmultiplied(0, 0, 0, 100));
}

/// Check if the user clicked the backdrop (outside the modal window).
/// Must be called AFTER the window is rendered so we know the window rect.
pub fn backdrop_clicked(ctx: &egui::Context, window_response: &Option<egui::InnerResponse<Option<()>>>) -> bool {
    let clicked = ctx.input(|i| i.pointer.primary_clicked());
    if !clicked { return false; }

    // If any popup is open (color picker, combo box), don't close
    #[allow(deprecated)]
    let popup_open = ctx.memory(|m| m.any_popup_open());
    if popup_open { return false; }

    // Check click position is outside the window rect
    let click_pos = ctx.input(|i| i.pointer.interact_pos());
    if let (Some(pos), Some(resp)) = (click_pos, window_response) {
        !resp.response.rect.contains(pos)
    } else {
        false
    }
}


/// Standard frame for modal overlay windows.
pub fn overlay_frame() -> egui::Frame {
    egui::Frame {
        fill: OVERLAY_BG,
        stroke: Stroke::new(1.0, OVERLAY_BORDER),
        corner_radius: egui::CornerRadius::same(8),
        inner_margin: egui::Margin::same(SPACE_MD as i8),
        ..Default::default()
    }
}

// === Colors (plum logo palette) ===

/// Main window/canvas background — dark charcoal from logo bg
pub const BG_PRIMARY: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1e);

/// Toolbar and status bar background
pub const BG_SECONDARY: Color32 = Color32::from_rgb(0x26, 0x24, 0x28);

/// Primary accent — plum purple (buttons, selections, progress)
pub const ACCENT: Color32 = Color32::from_rgb(0x7b, 0x2d, 0x8e);

/// Secondary accent — leaf green (done/success states)
pub const ACCENT_GREEN: Color32 = Color32::from_rgb(0x5b, 0x8c, 0x3e);

/// Destructive/error
pub const DESTRUCTIVE: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);

/// Primary text
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xf0, 0xf0, 0xf0);

/// Secondary text
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

/// Checkerboard light squares (light mode)
pub const CHECKER_LIGHT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

/// Checkerboard dark squares (light mode)
pub const CHECKER_DARK: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

/// Checkerboard light squares (dark mode — for viewing results on dark background)
pub const CHECKER_LIGHT_DARK_MODE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);

/// Checkerboard dark squares (dark mode)
pub const CHECKER_DARK_DARK_MODE: Color32 = Color32::from_rgb(0x1f, 0x1f, 0x1f);

/// Progress bar fill
pub const PROGRESS_FILL: Color32 = ACCENT;

/// Drop zone border on drag hover
pub const DROP_HOVER_BORDER: Color32 = ACCENT;

/// Drop zone border default
pub const DROP_BORDER: Color32 = Color32::from_rgba_premultiplied(0xff, 0xff, 0xff, 0x40);

// === Spacing (from UI-SPEC, multiples of 4) ===

pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 16.0;
pub const SPACE_LG: f32 = 24.0;

// === Layout ===

pub const TOOLBAR_HEIGHT: f32 = 48.0;
pub const STATUS_BAR_HEIGHT: f32 = 24.0;
pub const PROGRESS_BAR_HEIGHT: f32 = 4.0;
pub const PROGRESS_BAR_BG: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
pub const BUTTON_ROUNDING: f32 = 4.0;
pub const DROP_ZONE_ROUNDING: f32 = 8.0;
pub const DROP_ZONE_BORDER_WIDTH: f32 = 2.0;

/// Accent at ~40% opacity for disabled buttons
/// Premultiplied: 0x7b*102/255=49, 0x2d*102/255=18, 0x8e*102/255=57
pub const ACCENT_DISABLED: Color32 = Color32::from_rgba_premultiplied(49, 18, 57, 102);

// === Window ===

pub const DEFAULT_WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];
pub const MIN_WINDOW_SIZE: [f32; 2] = [640.0, 480.0];

// === Typography (sizes for egui TextStyle overrides) ===

pub const FONT_SIZE_BODY: f32 = 14.0;
pub const FONT_SIZE_HEADING: f32 = 16.0;
pub const FONT_SIZE_MONO: f32 = 12.0;

// === Shortcut Overlay ===

pub const SHORTCUT_OVERLAY_WIDTH: f32 = 320.0;
pub const SHORTCUT_OVERLAY_HEIGHT: f32 = 420.0;
/// Overlay background: #1a1a1a at 95% alpha
pub const OVERLAY_BG: Color32 = Color32::from_rgba_premultiplied(0x3a, 0x3a, 0x3a, 0xf8);
/// Overlay border: #ffffff20 (white 12.5%)
pub const OVERLAY_BORDER: Color32 = Color32::from_rgba_premultiplied(0xff, 0xff, 0xff, 0x20);

// === Checkerboard ===

/// Checkerboard square size in pixels
pub const CHECKER_SIZE: f32 = 16.0;

// === Logo ===
pub const LOGO_ASPECT: f32 = 416.0 / 512.0;

// === Phase 5: Sidebar ===
pub const SIDEBAR_WIDTH: f32 = 170.0;
pub const THUMBNAIL_SIZE: f32 = 120.0;
pub const THUMBNAIL_ROUNDING: f32 = 4.0;

// === Phase 5: Settings Dialog ===
pub const SETTINGS_DIALOG_WIDTH: f32 = 480.0;
pub const SETTINGS_DIALOG_HEIGHT: f32 = 700.0;

/// Hint text for settings modal (readable on dark overlay)
pub const TEXT_HINT: Color32 = Color32::from_rgb(0xb8, 0xb8, 0xb8);

// === Sidebar Colors ===
pub const SIDEBAR_ITEM_BG: Color32 = Color32::from_rgb(0x26, 0x24, 0x28);
pub const SIDEBAR_ITEM_SELECTED: Color32 = Color32::from_rgb(0x32, 0x28, 0x36);
pub const SIDEBAR_SELECTED_BORDER: Color32 = ACCENT;
pub const STATUS_ICON_PENDING: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
pub const INSERTION_LINE: Color32 = ACCENT;


// === Phase 5: Zoom ===
pub const ZOOM_MIN: f32 = 0.10;
pub const ZOOM_MAX: f32 = 20.0;
pub const ZOOM_STEP: f32 = 1.1;
