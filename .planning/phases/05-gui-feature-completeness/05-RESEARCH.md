# Phase 5: GUI Feature Completeness - Research

**Researched:** 2026-04-07
**Domain:** egui 0.34 GUI — zoom/pan, before/after toggle, reveal animation, batch sidebar with drag-reorder, settings dialog
**Confidence:** HIGH (verified from egui 0.34.1 source on disk and existing codebase)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Zoom & Pan**
- Scroll-wheel zoom centers on cursor position (Photoshop/Figma style)
- Exponential zoom steps: each scroll tick multiplies by ~1.1x
- Zoom range: 10% to 2000%
- Space+drag to pan the image
- Ctrl/Cmd+0 fits image to window; if already fitted, toggles back to previous zoom level
- Ctrl/Cmd+1 shows 1:1 pixel size; if already at 1:1, toggles back to previous zoom level
- Current zoom percentage displayed in the status bar alongside image dimensions

**Before/After Toggle**
- B key performs instant swap between original (source_texture) and result (result_texture)
- No split-slider — simple toggle
- When viewing result: checkerboard behind transparent areas
- When viewing original in Done state: no checkerboard, just original on dark background

**Reveal Animation**
- Mask-aware dissolve: only background pixels dissolve (subject stays 100% sharp)
- Duration: 0.5–1s, smooth alpha interpolation on background regions
- Checkerboard gradually appears through dissolving background areas
- Any key press or mouse click immediately skips to final state
- No "press to skip" hint text during animation
- Settings toggle to enable/disable (default: enabled)
- Plays once per processing completion, not on every before/after toggle

**Batch Sidebar**
- Left side panel with vertical thumbnail strip
- Auto-shows when 2+ images are loaded, hidden for single image
- Can be manually toggled (sidebar visibility shortcut)
- Thumbnail for each image with corner status icon overlay: ○ pending, ◆ processing, ✓ done
- Clicking a thumbnail switches main canvas to that image
- Drag-to-reorder items in sidebar
- [ and ] keys navigate previous/next image
- Each image's result is cached — no re-inference on switch
- Dropping multiple images populates the sidebar queue

**Batch Processing**
- "Process All" action processes all unprocessed images in the queue
- Uses batch_process() from bgprunr-core with rayon thread pool
- Parallelism level comes from settings (default: half available cores)
- Individual image progress via status icons in sidebar
- Overall batch progress in status bar

**Settings Dialog**
- Centered modal overlay (same pattern as shortcuts overlay)
- Escape or X button closes it
- Ctrl/Cmd+, opens it
- Settings: model selection dropdown, auto-remove checkbox, parallel jobs slider, reveal animation checkbox, active backend label (read-only)
- All settings must be remembered across restarts
- Settings struct serializable (serde) — Phase 6 wires file load/save, structure established now

### Claude's Discretion
- Sidebar width and thumbnail dimensions
- Sidebar toggle keyboard shortcut
- Exact animation easing curve
- Drag-to-reorder visual feedback style (ghost item, insertion line, etc.)
- Settings dialog dimensions and internal layout
- Zoom step multiplier value (approximately 1.1x but exact value flexible)
- How batch "Process All" is triggered (toolbar button, shortcut, or context menu)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VIEW-01 | User can zoom in/out with scroll wheel | `i.smooth_scroll_delta()` + `zoom_factor_delta`; cursor-centered zoom via pointer.hover_pos() |
| VIEW-02 | User can pan by holding Space and dragging | `i.keys_down.contains(Key::Space)` + `pointer.primary_down()` + `pointer.delta()` |
| VIEW-03 | Transparency displayed as checkerboard pattern | `draw_checkerboard()` already exists in canvas.rs; reuse for zoomed view |
| VIEW-04 | User can toggle between original and processed image (B key) | `i.key_pressed(Key::B)` toggles `show_original: bool` field; switches rendered texture |
| VIEW-05 | Fit image to window (Ctrl+0) or actual size (Ctrl+1) | Toggle pattern: compare current zoom to fit/1.0, swap with `previous_zoom` field |
| ANIM-01 | Background removal completion dissolves removed areas | `AppState::Animating` sub-state; per-pixel alpha animation driven by result mask |
| ANIM-02 | Animation plays over 0.5-1s, transitions to checkerboard | `i.stable_dt` for frame-accurate time; `ctx.request_repaint()` keeps loop alive |
| ANIM-03 | User can skip animation by pressing any key or clicking | Check `i.pointer.any_pressed() || !i.keys_pressed.is_empty()` in Animating state |
| BATCH-01 | Drop multiple images; appear in sidebar queue | Extend dropped_files handler to collect Vec; add `Panel::left` sidebar |
| BATCH-02 | Click between images in sidebar to view each | `selected_index: usize` field; thumbnail click sets it; canvas renders current item |
| BATCH-03 | Reorder images by dragging items in sidebar | `response.dnd_set_drag_payload()` / `dnd_release_payload()` — egui 0.34 built-in DnD |
| BATCH-04 | Process all queued images at once with parallel inference | `batch_process()` from bgprunr-core; WorkerMessage::BatchProcess variant |
| BATCH-05 | Results cached; switching does not re-process | `Vec<BatchItem>` each holding `Option<ProcessResult>` — switch = lookup only |
| BATCH-06 | Auto-remove on import setting processes images automatically | `Settings::auto_remove: bool`; after load, if true, enqueue and start processing |
| UX-02 | Settings dialog (Ctrl/Cmd+,) for model, auto-remove, parallelism | Modal window following shortcuts overlay pattern; `egui::Window` with `anchor(CENTER_CENTER)` |
| UX-05 | Navigate between images with [ and ] keys | `Key::OpenBracket` / `Key::CloseBracket` in keyboard handler; wraps around queue |
</phase_requirements>

---

## Summary

Phase 5 adds five major feature clusters to the existing egui 0.34.1 GUI: zoom/pan on the canvas, before/after toggle, a reveal animation on processing completion, a batch sidebar for multi-image workflows, and a settings dialog. All five clusters are fully achievable with the current dependency set — no new crates are required except adding `serde` and `serde_json` to the workspace (for the serializable Settings struct; file I/O deferred to Phase 6).

The existing codebase provides excellent foundations: `draw_checkerboard()`, the modal overlay pattern from `shortcuts.rs`, the worker channel pattern, `batch_process()` in bgprunr-core, and the AppState machine in state.rs. Phase 5 extends and connects these — it is primarily additive work with targeted refactors in canvas.rs, state.rs, app.rs, and worker.rs.

The most technically nuanced parts are: (1) zoom-toward-cursor math (requires the pointer position at the time of the scroll event); (2) the mask-aware reveal animation (requires storing the raw alpha mask alongside the result texture to drive per-pixel opacity during animation); and (3) the batch sidebar drag-to-reorder (egui 0.34 provides `Response::dnd_set_drag_payload` / `dnd_release_payload` natively — no external library needed).

**Primary recommendation:** Decompose Phase 5 into five implementation waves aligned to features (zoom/pan, before/after, animation, batch sidebar, settings), in that order. Each wave is independently shippable. Zoom/pan comes first because it is needed for quality inspection of results and has no dependencies on the other features.

---

## Standard Stack

### Core (all already in workspace — no new external crates needed for logic)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `egui` | 0.34.1 | All GUI widgets: Panel::left for sidebar, egui::Window for settings, DnD, input events | Already in use; 0.34.1 is the pinned workspace version |
| `eframe` | 0.34.1 | App lifecycle, repaint scheduling | Already in use |
| `image` | 0.25 | RgbaImage pixel access for animation mask | Already in bgprunr-app |
| `rayon` | 1.11 | Batch parallelism via `batch_process()` | Already in bgprunr-core |
| `num_cpus` | 1.17.0 | Default parallelism = half cores | Already in bgprunr-core |

### New Dependency (must add)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `serde` | 1.0.228 | Derive `Serialize`/`Deserialize` on `Settings` struct | Settings struct needs it now; file I/O in Phase 6 |
| `serde_json` | 1.0.149 | JSON serialization for settings file (Phase 6 consumes it) | Add now; `serde_json` is already an indirect dep |

**Installation (add to workspace Cargo.toml `[workspace.dependencies]` and bgprunr-app `[dependencies]`):**
```toml
# workspace Cargo.toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# crates/bgprunr-app/Cargo.toml [dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
```

Note: `serde_json` is already present as an indirect dependency (version 1.0.149 detected in `cargo metadata`). Adding it explicitly pins it and makes it available for Phase 6 settings persistence.

---

## Architecture Patterns

### Extended Project Structure

The planner should create these new files:
```
crates/bgprunr-app/src/gui/
├── views/
│   ├── canvas.rs          # EXTEND: add zoom/pan transform, before/after rendering
│   ├── sidebar.rs         # NEW: batch queue thumbnail strip
│   ├── settings.rs        # NEW: settings modal dialog
│   ├── animation.rs       # NEW: reveal animation rendering
│   ├── toolbar.rs         # EXTEND: batch "Process All" button
│   ├── statusbar.rs       # EXTEND: zoom % display
│   └── shortcuts.rs       # EXTEND: add new shortcuts to help overlay
├── app.rs                 # EXTEND: new fields for zoom, batch items, settings, animation
├── state.rs               # EXTEND: add Animating variant
├── worker.rs              # EXTEND: add BatchProcess message, BatchItemDone result
└── settings.rs            # NEW: Settings struct with serde derives
```

### Pattern 1: Zoom/Pan State Fields

**What:** Add zoom level and pan offset as fields in `BgPrunrApp`. Canvas render functions use these to compute `img_rect`.
**When to use:** All canvas render paths (Loaded, Processing, Done, Animating) must apply the same transform.

```rust
// In app.rs — new fields on BgPrunrApp
pub(crate) zoom: f32,           // 1.0 = 100%
pub(crate) pan_offset: Vec2,    // pixels from center
pub(crate) previous_zoom: f32,  // for Ctrl+0/Ctrl+1 toggle
pub(crate) is_space_held: bool,
```

**Zoom-toward-cursor math (cursor-centered zoom):**
```rust
// In canvas.rs render — inside the interaction zone
// Source pattern: standard pan-zoom for 2D canvases
let cursor_screen = i.pointer.hover_pos().unwrap_or(canvas_rect.center());
let cursor_canvas = cursor_screen - canvas_rect.center();

// Before: content at (cursor_canvas / zoom + pan_offset) was under cursor
// After zoom: (cursor_canvas / new_zoom + new_pan_offset) must equal same point
// Therefore:
app.pan_offset = cursor_canvas / app.zoom - cursor_canvas / new_zoom + app.pan_offset;
app.zoom = new_zoom;
```

**Image rect from zoom/pan:**
```rust
fn compute_img_rect(canvas_rect: Rect, tex_size: Vec2, zoom: f32, pan: Vec2) -> Rect {
    let img_size = tex_size * zoom;
    let center = canvas_rect.center() + pan;
    Rect::from_center_size(center, img_size)
}
```

**Fit-to-window zoom:**
```rust
fn fit_zoom(canvas_size: Vec2, tex_size: Vec2) -> f32 {
    (canvas_size.x / tex_size.x).min(canvas_size.y / tex_size.y).min(1.0)
}
```

**Ctrl+0 / Ctrl+1 toggle behavior:**
```rust
// Ctrl+0: fit-to-window, toggle back to previous if already fitted
let fit = fit_zoom(avail, tex_size);
if (app.zoom - fit).abs() < 0.001 {
    app.zoom = app.previous_zoom;
} else {
    app.previous_zoom = app.zoom;
    app.zoom = fit;
    app.pan_offset = Vec2::ZERO;
}
```

### Pattern 2: Before/After Toggle State

**What:** A single bool in `BgPrunrApp` controls which texture renders in Done state.
**When to use:** Only relevant in Done (and Animating) states.

```rust
// In app.rs
pub(crate) show_original: bool,  // false = show result (default)
```

```rust
// In canvas.rs render_done()
if app.show_original {
    // draw source_texture without checkerboard
    render_image_with_transform(ui, app, app.source_texture.as_ref().unwrap(), 1.0, false);
} else {
    // draw checkerboard + result_texture
    draw_checkerboard(ui, img_rect);
    render_image_with_transform(ui, app, app.result_texture.as_ref().unwrap(), 1.0, true);
}
```

### Pattern 3: Reveal Animation

**What:** New `AppState::Animating` state. Stores animation progress (0.0 → 1.0) and the raw alpha mask to know which pixels are background.
**When to use:** Entered from Processing when WorkerResult::Done is received (if animation enabled).

```rust
// In state.rs — extend AppState
pub enum AppState {
    Empty,
    Loaded,
    Processing,
    Animating,   // NEW
    Done,
}
```

```rust
// In app.rs — new fields
pub(crate) anim_progress: f32,         // 0.0 start, 1.0 done
pub(crate) anim_mask: Option<Vec<u8>>, // alpha channel of result, same dims as result_texture
                                        // 0 = background (dissolves), 255 = subject (stays)
```

**Animation rendering approach:**
The animation cannot be done purely with egui painter operations on a single texture — it requires updating a texture per-frame with modified alpha values. The correct approach is:

1. On `WorkerResult::Done`: store `result_rgba` as usual, extract alpha channel into `anim_mask`, set `state = Animating`, set `anim_progress = 0.0`.
2. Each frame in `Animating`: compute `t = anim_progress` (0→1 over ANIM_DURATION_SECS).
3. Build a per-frame `RgbaImage` where background pixels (mask alpha < threshold) have their alpha scaled by `(1.0 - t)`, subject pixels keep alpha=255.
4. Upload as a temporary egui texture for this frame using `ctx.load_texture(...)`.
5. Advance `anim_progress += stable_dt / ANIM_DURATION_SECS`.
6. Call `ctx.request_repaint()` to keep animation running.
7. When `anim_progress >= 1.0` or user presses key/clicks: `state = Done`.

```rust
// Animation constants
const ANIM_DURATION_SECS: f32 = 0.75;
const ANIM_MASK_THRESHOLD: u8 = 128; // pixels with alpha < 128 are background
```

**Performance note:** Building a new RgbaImage every frame for large images is non-trivial. Use `TextureOptions::NEAREST` and consider downsampling the animation texture to the screen pixel count (not source image size) for large images. A 4K source image does not need a 4K animation texture — the on-screen size is what matters.

**Skip animation:**
```rust
// In logic() while state == Animating
let skip = ctx.input(|i| {
    i.pointer.any_pressed() || i.keys_pressed.iter().next().is_some()
});
if skip { app.state = AppState::Done; app.anim_progress = 0.0; }
```

### Pattern 4: Batch Sidebar

**What:** `Panel::left` added in `ui()` before `CentralPanel`. Contains a `Vec<BatchItem>` list with thumbnails.
**When to use:** Auto-visible when `batch_items.len() >= 2`.

```rust
// In app.rs — new fields
pub(crate) batch_items: Vec<BatchItem>,
pub(crate) selected_batch_index: usize,
pub(crate) show_sidebar: bool, // manual override
```

```rust
// New struct (can live in app.rs or a new batch.rs module)
pub(crate) struct BatchItem {
    pub filename: String,
    pub source_bytes: Vec<u8>,
    pub source_texture: Option<egui::TextureHandle>,
    pub thumb_texture: Option<egui::TextureHandle>,  // small thumbnail for sidebar
    pub result: Option<image::RgbaImage>,
    pub result_texture: Option<egui::TextureHandle>,
    pub status: BatchStatus,
}

pub(crate) enum BatchStatus {
    Pending,
    Processing,
    Done,
    Error(String),
}
```

**Panel::left usage (egui 0.34):**
```rust
// In app.rs ui()
let sidebar_visible = app.show_sidebar || app.batch_items.len() >= 2;

if sidebar_visible {
    egui::Panel::left("sidebar")
        .exact_width(SIDEBAR_WIDTH)   // recommended: 160.0
        .resizable(false)
        .show_inside(ui, |ui| sidebar::render(ui, self));
}

egui::CentralPanel::default().show_inside(ui, |ui| canvas::render(ui, self));
```

**Drag-to-reorder in sidebar (egui 0.34 DnD API):**
```rust
// For each item at index `i`:
let item_response = ui.allocate_response(item_size, egui::Sense::drag());

// Set payload when drag starts
item_response.dnd_set_drag_payload(i); // payload = original index

// Detect drop target: each item checks if something is being hovered
if let Some(src_idx) = item_response.dnd_release_payload::<usize>() {
    // Reorder: move batch_items[*src_idx] to position i
    let moved = app.batch_items.remove(*src_idx);
    let dst = if *src_idx < i { i - 1 } else { i };
    app.batch_items.insert(dst, moved);
}
```

**Thumbnail generation:** When loading a batch item, generate a small thumbnail (e.g., 128×128 max) immediately on the UI thread using `image::imageops::thumbnail()` and upload as a texture. This keeps sidebar rendering fast.

### Pattern 5: Settings Dialog (Modal Window)

**What:** `egui::Window` anchored to CENTER_CENTER, same frame style as shortcuts overlay. Controlled by `show_settings: bool` on `BgPrunrApp`.
**When to use:** Opened by Ctrl/Cmd+, or X button.

```rust
// In app.rs
pub(crate) show_settings: bool,
pub(crate) settings: Settings,
```

```rust
// New file: crates/bgprunr-app/src/gui/settings.rs
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub model: ModelKind,
    pub auto_remove_on_import: bool,
    pub parallel_jobs: usize,
    pub reveal_animation_enabled: bool,
    // Not serialized — runtime-only
    #[serde(skip)]
    pub active_backend: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: ModelKind::Silueta,
            auto_remove_on_import: false,
            parallel_jobs: (num_cpus::get() / 2).max(1),
            reveal_animation_enabled: true,
            active_backend: "CPU".to_string(),
        }
    }
}
```

**Settings dialog rendering (views/settings.rs):**
```rust
// Source: mirrors shortcuts.rs pattern exactly
pub fn render(ctx: &egui::Context, app: &mut BgPrunrApp) {
    egui::Window::new("Settings")
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([SETTINGS_OVERLAY_WIDTH, SETTINGS_OVERLAY_HEIGHT])
        .frame(egui::Frame {
            fill: theme::OVERLAY_BG,
            stroke: Stroke::new(1.0, theme::OVERLAY_BORDER),
            corner_radius: egui::CornerRadius::same(8),
            inner_margin: egui::Margin::same(theme::SPACE_MD as i8),
            ..Default::default()
        })
        .show(ctx, |ui| { /* model combo, checkboxes, slider, backend label */ });
}
```

### Anti-Patterns to Avoid

- **Rebuilding OrtEngine per frame for animation:** Animation is purely a GPU/CPU-side render operation. The OrtEngine is only used during inference. Never call `OrtEngine::new()` in the render loop.
- **Blocking UI thread during batch:** `batch_process()` is blocking. It must run on a worker thread with `WorkerMessage::BatchProcess`. Never call it in `logic()` or `ui()` directly.
- **Sharing textures across batch items without `Id` uniqueness:** Each texture must have a unique string id (`ctx.load_texture(format!("source_{i}", i), ...)`). Reusing the same string key causes textures to overwrite each other.
- **Using `i.modifiers.ctrl` for Ctrl/Cmd:** Always use `i.modifiers.command` which maps to Cmd on macOS and Ctrl elsewhere (established in ARCHITECTURE.md).
- **Calling `ui.input()` in `logic()`:** In eframe, `logic()` receives `&egui::Context`, not `&Ui`. Use `ctx.input(|i| ...)` in `logic()`. The `ui.input()` shortcut is only available in `ui()`.
- **Allocating large RgbaImage per frame for animation at full source resolution:** Cap animation texture dimensions to `(avail_width as u32).min(source_width)` etc. A 500×500 canvas doesn't need a 4000×4000 animation texture.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Drag-to-reorder in list | Custom mouse tracking + index math | `response.dnd_set_drag_payload()` + `dnd_release_payload()` | egui 0.34 DnD plugin handles cursor states, payload lifecycle, pointer-up detection |
| Modal overlay backdrop | Custom `painter.rect_filled()` on backdrop | Copy shortcuts.rs pattern exactly | Pattern already tested; layer ordering via `ctx.layer_painter(Order::Background)` handles z-order |
| Thumbnail generation | Custom bilinear downscale | `image::imageops::thumbnail(src, max_w, max_h)` | Handles aspect ratio, picks correct filter algorithm |
| CPU count for default parallelism | `std::thread::available_parallelism()` | `num_cpus::get()` (already in workspace) | Consistent with batch.rs which already uses it |
| Easing function for animation | Custom polynomial | `t * t * (3.0 - 2.0 * t)` (smoothstep, one line) | Smoothstep provides perceptually smooth ease-in-out with no deps |

**Key insight:** egui's `DragAndDrop` plugin (added in ~0.27) is stable in 0.34.1 and handles the full drag lifecycle. Using `Response::dnd_set_drag_payload` (on mouse-down of an item) and `Response::dnd_release_payload` (on mouse-up over a target item) is the idiomatic approach — no need for manual mouse state tracking.

---

## Common Pitfalls

### Pitfall 1: Scroll Events Consumed by Egui Scroll Areas
**What goes wrong:** If any parent widget in the layout is a `ScrollArea`, scroll wheel events are consumed before reaching the canvas input handler.
**Why it happens:** `egui::ScrollArea` captures scroll events within its rect, preventing them from reaching the canvas zoom handler.
**How to avoid:** The canvas `CentralPanel` must NOT be wrapped in a `ScrollArea`. The canvas itself handles scroll directly via `ctx.input(|i| i.smooth_scroll_delta())` — not via egui's scroll widget system.
**Warning signs:** Scroll wheel pans the canvas view unexpectedly, or zoom factor is always 1.0.

### Pitfall 2: Zoom-Toward-Cursor Requires Pointer Position at Scroll Time
**What goes wrong:** Using `canvas_rect.center()` as the pivot instead of the cursor position produces zoom-toward-center behavior, which frustrates edge inspection workflows.
**Why it happens:** It's easier to implement, but it is not the expected behavior.
**How to avoid:** Read `ctx.input(|i| i.pointer.hover_pos())` within the same `ctx.input(|i| ...)` call that reads `smooth_scroll_delta`. The pointer position must be captured in the same frame as the scroll event.
**Warning signs:** Zoom works but always pulls image center to viewport center.

### Pitfall 3: Animation Texture Id Collision
**What goes wrong:** `ctx.load_texture("anim_frame", ...)` overwrites the result texture if it was also loaded as `"result"`. Conversely, if animation re-uses `"result"` key, the stored `result_texture` handle goes stale.
**Why it happens:** egui texture cache is keyed by string name. The same string = same texture slot.
**How to avoid:** Use a dedicated key like `"anim_frame"` for the animation texture, distinct from `"result"` and `"source"`. Drop the animation texture after animation ends by retaining only the `result_texture`.

### Pitfall 4: BatchItem Texture IDs Must Be Unique Per Item
**What goes wrong:** All thumbnails show the last-loaded image if all use the same texture key (`"thumb"`).
**Why it happens:** `ctx.load_texture("thumb", ...)` on each item overwrites the previous.
**How to avoid:** Use `format!("source_{idx}")`, `format!("thumb_{idx}")`, `format!("result_{idx}")` for all per-item textures, where `idx` is a stable ID (not the Vec position, which changes on reorder).
**Warning signs:** All sidebar thumbnails show the same image.

### Pitfall 5: Batch Drop Handling Conflicts With Single-Image Drop
**What goes wrong:** Existing drop handler calls `handle_open_path()` for each file, which replaces `source_bytes` and resets state — batching is destroyed.
**Why it happens:** The Phase 4 handler was written for single-image flow.
**How to avoid:** When implementing batch, replace the drop loop in `logic()`: if `batch_items` is non-empty or count > 1, route all drops to `add_to_batch()`. Provide a clear mode-switch: first drop with empty batch → single image flow; subsequent drops → batch flow.

### Pitfall 6: Space Key Intercepted by Egui Focusable Widgets
**What goes wrong:** Space key triggers button clicks if a button has focus.
**Why it happens:** egui buttons respond to Space when they have keyboard focus.
**How to avoid:** The Space-drag pan must be read in `logic()` → `ctx.input()` before ui() renders buttons. In `logic()`, consume the Space key event for pan if the canvas is the "active" focus context. Alternatively, since the canvas is not a widget, read Space from global input and suppress it from reaching widgets by checking that no text input widget is focused.

---

## Code Examples

Verified from egui 0.34.1 source in `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/`.

### Scroll-Wheel Zoom (Raw Event Approach)
```rust
// Source: egui 0.34.1 input_state/mod.rs — Event::MouseWheel
// Read raw scroll delta to compute zoom factor ourselves (not Ctrl+scroll)
// This intercepts ALL scroll events on the canvas, not just Ctrl+scroll

ctx.input(|i| {
    // Check for mouse wheel events directly for zoom (without Ctrl requirement)
    for event in &i.events {
        if let egui::Event::MouseWheel { delta, modifiers, .. } = event {
            if !modifiers.any() { // pure scroll = zoom (no Ctrl/Alt/Shift)
                let scroll_y = delta.y; // positive = scroll up = zoom in
                let zoom_delta = 1.1_f32.powf(scroll_y);
                let new_zoom = (app.zoom * zoom_delta).clamp(0.10, 20.0);
                // cursor-centered adjustment
                if let Some(cursor) = i.pointer.hover_pos() {
                    let cursor_rel = cursor - canvas_rect.center();
                    app.pan_offset = cursor_rel / app.zoom
                        - cursor_rel / new_zoom
                        + app.pan_offset;
                }
                app.zoom = new_zoom;
            }
        }
    }
});
```

### Space+Drag Pan
```rust
// Source: egui 0.34.1 input_state/mod.rs — pointer.delta(), keys_down
ctx.input(|i| {
    let space_held = i.keys_down.contains(&egui::Key::Space);
    let dragging = i.pointer.primary_down();
    if space_held && dragging {
        app.pan_offset += i.pointer.delta();
        app.is_panning = true;
    } else {
        app.is_panning = false;
    }
});
```

### DnD Drag-to-Reorder (sidebar.rs)
```rust
// Source: egui 0.34.1 response.rs — dnd_set_drag_payload, dnd_release_payload
// Draw items and handle reorder
let mut swap_from: Option<usize> = None;
let mut swap_to: Option<usize> = None;

for (i, item) in app.batch_items.iter().enumerate() {
    let item_response = ui.allocate_response(
        egui::Vec2::new(SIDEBAR_WIDTH - 8.0, THUMB_HEIGHT + 4.0),
        egui::Sense::click_and_drag(),
    );

    // Draw the thumbnail (painter calls here)

    // Set payload when this item starts being dragged
    item_response.dnd_set_drag_payload(i);

    // Detect if something is dropped onto this position
    if let Some(src) = item_response.dnd_release_payload::<usize>() {
        swap_from = Some(*src);
        swap_to = Some(i);
    }

    if item_response.clicked() {
        app.selected_batch_index = i;
    }
}

// Apply reorder after iteration
if let (Some(from), Some(to)) = (swap_from, swap_to) {
    if from != to {
        let item = app.batch_items.remove(from);
        let dst = if from < to { to - 1 } else { to };
        app.batch_items.insert(dst, item);
        // Adjust selected index if it was moved
    }
}
```

### Panel::left for Sidebar (app.rs ui())
```rust
// Source: egui 0.34.1 containers/panel.rs — Panel::left
let show = app.show_sidebar || app.batch_items.len() >= 2;
if show {
    egui::Panel::left("sidebar")
        .exact_width(theme::SIDEBAR_WIDTH)
        .resizable(false)
        .show_separator_line(true)
        .show_inside(ui, |ui| sidebar::render(ui, self));
}
egui::CentralPanel::default().show_inside(ui, |ui| canvas::render(ui, self));
```

### Animation Repaint Loop
```rust
// Source: egui 0.34.1 context.rs — request_repaint_after_secs
// In logic() while state == Animating:
let dt = ctx.input(|i| i.stable_dt);
app.anim_progress = (app.anim_progress + dt / ANIM_DURATION_SECS).min(1.0);
if app.anim_progress >= 1.0 {
    app.state = AppState::Done;
} else {
    ctx.request_repaint(); // keep animation loop alive
}
```

### Worker BatchProcess Extension
```rust
// In worker.rs — extend WorkerMessage and WorkerResult
pub enum WorkerMessage {
    ProcessImage { img_bytes: Vec<u8>, model: ModelKind, cancel: Arc<AtomicBool> },
    BatchProcess { images: Vec<Vec<u8>>, model: ModelKind, jobs: usize },
    Quit,
}

pub enum WorkerResult {
    Progress(ProgressStage, f32),
    Done(ProcessResult),
    BatchItemDone { index: usize, result: Result<ProcessResult, CoreError> },
    Cancelled,
    Error(String),
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SidePanel::left("id")` | `Panel::left("id")` | egui 0.29 (deprecated alias exists in 0.34) | Use `Panel::left`; `SidePanel` type alias still compiles but shows deprecation warning |
| `ui.ctx().input()` in `logic()` | `ctx.input()` directly | eframe split of `logic()` + `ui()` | In `logic()` there is no `ui`, use the `ctx: &egui::Context` parameter |
| Manual drag tracking with `pointer.delta()` | `response.dnd_set_drag_payload()` + `dnd_release_payload()` | egui ~0.27 | Built-in DnD plugin handles payload lifecycle cleanly |
| Global style via `set_visuals()` | `ctx.set_global_style()` / `ctx.set_visuals()` | egui 0.30+ (already applied in Phase 4) | Phase 4 already uses the correct API |

---

## Open Questions

1. **Animation texture performance for large images**
   - What we know: Building a per-frame `RgbaImage` and uploading to GPU has overhead proportional to pixel count.
   - What's unclear: Whether 4K images will cause frame rate drops on low-end hardware during the 0.75s animation.
   - Recommendation: During animation, scale the frame down to `avail_size * window_scale` before upload. The on-screen image is never larger than the canvas area, so a 4K source in a 1000×800 window only needs a ~1000×800 animation texture. Add a comment explaining the optimization.

2. **Batch progress reporting back to UI**
   - What we know: `batch_process()` takes a progress callback but returns all results only at completion (blocking call). Individual results must be sent as they complete.
   - What's unclear: `batch_process()` as currently written returns a `Vec` only after all items finish. For the GUI, we need per-item completion events.
   - Recommendation: The worker thread wraps `batch_process()` per-item, not as a single call. The worker spawns a rayon pool and sends `WorkerResult::BatchItemDone { index, result }` via the mpsc channel as each image finishes, rather than calling `batch_process()` wholesale. This matches the architecture diagram in ARCHITECTURE.md which shows "send per-image results via channel."

3. **Settings serde and ModelKind derive**
   - What we know: `ModelKind` is defined in bgprunr-core and does not currently derive `serde::Serialize/Deserialize`.
   - What's unclear: Whether to add serde derives to `ModelKind` in bgprunr-core, or create a parallel `SettingsModelKind` type in bgprunr-app with `From` conversion.
   - Recommendation: Add `serde` as an optional feature to bgprunr-core and derive on `ModelKind` only when the feature is active. This avoids forcing a serde dependency on the core library for CLI use. Alternatively, since bgprunr-app already has serde, define a mirrored enum in settings.rs with a `From<SettingsModelKind> for ModelKind` impl — simpler, no core changes needed.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test (`cargo test`) |
| Config file | none — `[lib]` section in bgprunr-app/Cargo.toml with `src/lib.rs` |
| Quick run command | `cargo test -p bgprunr-app --lib 2>&1 \| tail -20` |
| Full suite command | `cargo test --workspace --lib 2>&1 \| tail -30` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VIEW-01 | Zoom state field initialized to fit-zoom | unit | `cargo test -p bgprunr-app --lib zoom` | ❌ Wave 0 |
| VIEW-02 | Pan offset accumulates during space+drag | unit | `cargo test -p bgprunr-app --lib pan` | ❌ Wave 0 |
| VIEW-03 | Checkerboard already tested implicitly | unit | `cargo test -p bgprunr-app --lib canvas` | ❌ Wave 0 |
| VIEW-04 | before_after toggle switches show_original field | unit | `cargo test -p bgprunr-app --lib before_after` | ❌ Wave 0 |
| VIEW-05 | Ctrl+0 sets zoom to fit, second call restores previous | unit | `cargo test -p bgprunr-app --lib fit_zoom` | ❌ Wave 0 |
| ANIM-01 | AppState::Animating variant exists | unit | `cargo test -p bgprunr-app --lib state` | ❌ (extends state_tests.rs) |
| ANIM-02 | anim_progress advances by dt/DURATION | unit | `cargo test -p bgprunr-app --lib anim` | ❌ Wave 0 |
| ANIM-03 | Skip on click/key sets state to Done | unit | `cargo test -p bgprunr-app --lib anim_skip` | ❌ Wave 0 |
| BATCH-01 | Multiple dropped files → batch_items Vec populated | unit | `cargo test -p bgprunr-app --lib batch` | ❌ Wave 0 |
| BATCH-02 | selected_batch_index change switches active item | unit | `cargo test -p bgprunr-app --lib batch_select` | ❌ Wave 0 |
| BATCH-03 | Drag-reorder swaps items at correct indices | unit | `cargo test -p bgprunr-app --lib batch_reorder` | ❌ Wave 0 |
| BATCH-04 | WorkerMessage::BatchProcess variant exists and routes | unit | `cargo test -p bgprunr-app --lib worker` | ❌ Wave 0 |
| BATCH-05 | result cached in BatchItem; second select does not re-process | unit | `cargo test -p bgprunr-app --lib batch_cache` | ❌ Wave 0 |
| BATCH-06 | auto_remove_on_import triggers handle_remove_bg after load | unit | `cargo test -p bgprunr-app --lib auto_remove` | ❌ Wave 0 |
| UX-02 | Settings struct default values correct | unit | `cargo test -p bgprunr-app --lib settings` | ❌ Wave 0 |
| UX-05 | OpenBracket/CloseBracket keys advance selected_batch_index | unit | `cargo test -p bgprunr-app --lib nav_keys` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p bgprunr-app --lib 2>&1 | tail -20`
- **Per wave merge:** `cargo test --workspace --lib 2>&1 | tail -30`
- **Phase gate:** Full workspace test suite green before `/gsd:verify-work`

### Wave 0 Gaps
All Phase 5 test files are new. Existing test infrastructure (`tests/mod.rs`, `tests/state_tests.rs`, `tests/input_tests.rs`, `tests/clipboard_tests.rs`) provides the pattern.

New test files needed:
- [ ] `crates/bgprunr-app/src/gui/tests/zoom_pan_tests.rs` — covers VIEW-01, VIEW-02, VIEW-05
- [ ] `crates/bgprunr-app/src/gui/tests/batch_tests.rs` — covers BATCH-01 through BATCH-06, UX-05
- [ ] `crates/bgprunr-app/src/gui/tests/anim_tests.rs` — covers ANIM-01, ANIM-02, ANIM-03
- [ ] `crates/bgprunr-app/src/gui/tests/settings_tests.rs` — covers UX-02
- [ ] Extend `crates/bgprunr-app/src/gui/tests/state_tests.rs` — add Animating variant test

These are unit tests of state-machine logic only (same pattern as existing tests). They do not require a running egui context.

---

## Sources

### Primary (HIGH confidence)
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/input_state/mod.rs` — InputState struct, smooth_scroll_delta, stable_dt, pointer API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/drag_and_drop.rs` — DragAndDrop plugin, set_payload/take_payload
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/response.rs` — dnd_set_drag_payload, dnd_hover_payload, dnd_release_payload
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/containers/panel.rs` — Panel::left, exact_width, resizable, show_animated_inside
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/data/key.rs` — Key::Space, Key::OpenBracket, Key::CloseBracket, Key::Comma verified
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/context.rs` — request_repaint, request_repaint_after_secs
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-0.34.1/src/data/input.rs` — Event::MouseWheel structure, modifiers field
- `crates/bgprunr-app/src/gui/` — all existing Phase 4 GUI code read directly

### Secondary (MEDIUM confidence)
- `cargo metadata` output — confirmed serde 1.0.228, serde_json 1.0.149, num_cpus 1.17.0 already present in dependency graph
- `ARCHITECTURE.md` — batch processing data flow, threading model, state machine diagram (project-authored, highly authoritative)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified from local cargo registry and cargo metadata
- Architecture: HIGH — all egui APIs verified from egui 0.34.1 source on disk; patterns derived from existing Phase 4 code
- Pitfalls: HIGH — derived from reading actual egui source and existing codebase; not speculation
- Validation architecture: HIGH — existing test infrastructure verified from source files

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (egui 0.34 is pinned in workspace; unlikely to change within a month)
