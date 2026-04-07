use egui::Color32;

// === Colors (from UI-SPEC) ===

/// Main window/canvas background (#1a1a1a)
pub const BG_PRIMARY: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);

/// Toolbar and status bar background (#252525)
pub const BG_SECONDARY: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);

/// Primary accent -- Remove BG button only (#3b82f6)
pub const ACCENT: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);

/// Destructive/error (#ef4444)
pub const DESTRUCTIVE: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);

/// Surface overlay -- disabled tint, drop zone border (#ffffff14 = white 8%)
pub const SURFACE_OVERLAY: Color32 = Color32::from_rgba_premultiplied(0xff, 0xff, 0xff, 0x14);

/// Primary text (#f0f0f0)
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xf0, 0xf0, 0xf0);

/// Secondary text (#888888)
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

/// Checkerboard light squares (#cccccc)
pub const CHECKER_LIGHT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

/// Checkerboard dark squares (#888888)
pub const CHECKER_DARK: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

/// Progress bar fill (same as ACCENT)
pub const PROGRESS_FILL: Color32 = ACCENT;

/// Drop zone border on drag hover (#3b82f6 -- accent)
pub const DROP_HOVER_BORDER: Color32 = ACCENT;

/// Drop zone border default (#ffffff40 = white 25%)
pub const DROP_BORDER: Color32 = Color32::from_rgba_premultiplied(0xff, 0xff, 0xff, 0x40);

// === Spacing (from UI-SPEC, multiples of 4) ===

pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 16.0;
pub const SPACE_LG: f32 = 24.0;
pub const SPACE_XL: f32 = 32.0;

// === Layout ===

pub const TOOLBAR_HEIGHT: f32 = 36.0;
pub const STATUS_BAR_HEIGHT: f32 = 24.0;
pub const PROGRESS_BAR_HEIGHT: f32 = 4.0;
pub const BUTTON_ROUNDING: f32 = 4.0;
pub const DROP_ZONE_ROUNDING: f32 = 8.0;
pub const DROP_ZONE_BORDER_WIDTH: f32 = 2.0;

/// Disabled button opacity (40%)
pub const DISABLED_OPACITY: f32 = 0.4;

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
pub const OVERLAY_BG: Color32 = Color32::from_rgba_premultiplied(0x1a, 0x1a, 0x1a, 0xf2);
/// Overlay border: #ffffff20 (white 12.5%)
pub const OVERLAY_BORDER: Color32 = Color32::from_rgba_premultiplied(0xff, 0xff, 0xff, 0x20);

// === Checkerboard ===

/// Checkerboard square size in pixels
pub const CHECKER_SIZE: f32 = 16.0;

// === Phase 5: Sidebar ===
pub const SIDEBAR_WIDTH: f32 = 96.0;
pub const THUMBNAIL_SIZE: f32 = 80.0;
pub const THUMBNAIL_ROUNDING: f32 = 4.0;

// === Phase 5: Settings Dialog ===
pub const SETTINGS_DIALOG_WIDTH: f32 = 400.0;
pub const SETTINGS_DIALOG_HEIGHT: f32 = 320.0;

// === Phase 5: Sidebar Colors ===
pub const SIDEBAR_ITEM_BG: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);
pub const SIDEBAR_ITEM_SELECTED: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
pub const SIDEBAR_SELECTED_BORDER: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);
pub const STATUS_ICON_PENDING: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
pub const STATUS_ICON_PROCESSING: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);
pub const STATUS_ICON_DONE: Color32 = Color32::from_rgb(0x22, 0xc5, 0x5e);
pub const DRAG_GHOST_ALPHA: u8 = 0x80;
pub const INSERTION_LINE: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);

// === Phase 5: Animation ===
pub const ANIM_DURATION_SECS: f32 = 0.75;
pub const ANIM_MASK_THRESHOLD: u8 = 128;

// === Phase 5: Zoom ===
pub const ZOOM_MIN: f32 = 0.10;
pub const ZOOM_MAX: f32 = 20.0;
pub const ZOOM_STEP: f32 = 1.1;
