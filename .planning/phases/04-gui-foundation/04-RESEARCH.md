# Phase 4: GUI Foundation - Research

**Researched:** 2026-04-07
**Domain:** egui/eframe 0.34 desktop GUI, worker-thread inference dispatch, file dialogs, clipboard, drag-and-drop
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Window Layout**
- Minimal toolbar + canvas: Thin top toolbar with [Open] [Remove BG] [Save] [Copy] buttons + Model selector dropdown
- Large image canvas area takes most of the window
- Status bar at bottom: state text ("Ready" / "Processing..." / "Done"), active backend ("CPU" / "CUDA"), image dimensions
- Dark theme — better for image editing, makes transparency checkerboard visible
- Default window size: 1280×800
- Remember last window size/position (store locally, Phase 6 adds full settings persistence)
- Window title: "BgPrunR" → "BgPrunR — filename.jpg" when image loaded

**Image Display**
- Empty state: Centered drop zone hint with dashed border — "Drop an image here or press Ctrl+O"
- Adapts hint to platform: "Cmd+O" on macOS
- Fit to window by default — scale image to fit canvas, maintaining aspect ratio
- Dark background behind image (consistent with dark theme)

**Progress Indicator**
- During inference: progress spinner/bar in status bar area with current stage name
- Toolbar buttons disabled during processing (except Escape to cancel)
- Status bar shows: "Processing... Inferring" → "Processing... Applying alpha" etc.

**Shortcut Overlay**
- Centered modal with dark semi-transparent background
- Lists Phase 4 shortcuts only (6 core shortcuts)
- Press ? or Escape to dismiss

**Worker Thread Architecture**
- Single long-lived worker thread communicating via std::sync::mpsc channels
- UI thread sends `WorkerMessage::ProcessImage(bytes, model)` to worker
- Worker sends `WorkerResult::Progress(stage, pct)` and `WorkerResult::Done(ProcessResult)` back
- UI thread polls via `try_recv()` in each frame's logic handler
- Worker calls `ctx.request_repaint()` when progress updates or completes
- Escape sends `WorkerMessage::Cancel` — worker checks a cancel flag between stages

**Clipboard**
- arboard crate with `wayland-data-control` feature for Wayland support
- Copy the processed RGBA image to clipboard as PNG
- Show brief status bar feedback: "Copied to clipboard"

### Claude's Discretion
- Exact egui widget layout code (ui.horizontal, ui.vertical, etc.)
- Status bar styling (colors, spacing)
- Drop zone visual design (dashed border style, text color)
- Progress bar vs spinner choice for status area
- arboard error handling details

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LOAD-01 | User can drag and drop an image file onto the app window to load it | eframe ViewportBuilder::with_drag_and_drop(true) + egui RawInput::dropped_files pattern |
| LOAD-02 | User can open an image via file browser dialog (Ctrl+O) | rfd 0.15.4 FileDialog::pick_file() with add_filter for image types; Modifiers::command + Key::O |
| OUT-01 | User can save the processed image as PNG with transparency (Ctrl+S) | rfd FileDialog::save_file() + bgprunr_core::encode_rgba_png(); only enabled in Done state |
| OUT-02 | User can copy the processed image to clipboard (Ctrl+C) | arboard 3.6.1 Clipboard::set_image(ImageData) with wayland-data-control feature; Clipboard stored on App struct |
| UX-01 | All keyboard shortcuts work as specified (Ctrl/Cmd+O, +R, +S, +C, Escape, ?) | egui Modifiers::command (not .ctrl) for cross-platform; all 6 shortcuts dispatched in logic() or input handler |
| UX-03 | User can cancel in-progress inference with Escape | WorkerMessage::Cancel sent via mpsc; AtomicBool cancel_flag polled between pipeline stages |
| UX-04 | User can press ? to see all keyboard shortcuts | show_shortcuts bool on App struct; modal rendered in ui() when true; dismissed by ? or Escape |
</phase_requirements>

---

## Summary

Phase 4 builds the entire interactive GUI layer on top of the already-complete inference core (Phase 2) and CLI binary (Phase 3). The central technical challenge is keeping egui's render loop non-blocking while dispatching inference (which can take 3–15 seconds on CPU with u2net). The solution is a single long-lived worker thread communicating via `std::sync::mpsc`, a pattern that is well-established in the egui community and directly supported by egui's `Context::request_repaint()` being `Send`.

A critical API discovery: **eframe 0.34 deprecated `App::update()` and replaced it with two methods: `App::ui()` (for rendering) and `App::logic()` (for non-rendering per-frame work like channel polling)**. The ARCHITECTURE.md references `App::update()` which is the old pre-0.34 API. All implementation must use the new `ui()` + `logic()` split. Channel polling via `try_recv()` goes in `logic()`, UI widget rendering goes in `ui()`.

The rfd file dialog is synchronous (blocking), which is correct for this use case — the native OS dialog runs modally and the user interaction is intentionally sequential. arboard's clipboard requires the `Clipboard` instance to be stored on the `App` struct for the lifetime of the app (Wayland ownership requirement), and must be constructed with the `wayland-data-control` feature.

**Primary recommendation:** Follow the exact worker-thread architecture from CONTEXT.md with the new eframe 0.34 `logic()` + `ui()` method split. Store `TextureHandle`, `Clipboard`, and `mpsc` channel ends as fields on the `App` struct. Never call rfd or arboard from the worker thread.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `egui` | 0.34.1 | Immediate-mode UI widgets, input handling, texture management | Already in workspace; confirmed latest as of 2026-03-27 |
| `eframe` | 0.34.1 | Native window harness, wgpu renderer, winit integration | Already in workspace; must match egui version exactly |
| `rfd` | 0.15.4 | Native file open/save dialogs | Already in workspace (pinned at "0.15", resolves to 0.15.4); synchronous FileDialog API; no tokio needed |
| `arboard` | 3.6.1 | Cross-platform clipboard image copy | Already in workspace with wayland-data-control feature; 1Password-maintained |
| `bgprunr-core` | workspace | Inference pipeline, image I/O | Already complete from Phase 2; process_image(), encode_rgba_png(), load_image_from_path() |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `egui_extras` | 0.34.1 | Image loaders for egui texture system | Optional — project loads images via `image` crate directly and constructs `ColorImage` manually; simpler to not use |
| `std::sync::mpsc` | stdlib | Worker thread communication | Channel ends stored on App struct; Sender<WorkerMessage> + Receiver<WorkerResult> |
| `std::sync::atomic::AtomicBool` | stdlib | Cancel flag polled by worker between pipeline stages | Shared via Arc<AtomicBool> between UI thread (sets) and worker thread (reads) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `std::sync::mpsc` | `crossbeam-channel` | crossbeam has select! and bounded channels; overkill for single worker, single result pair |
| `rfd` sync `FileDialog` | `rfd` async `AsyncFileDialog` | Async requires tokio or async-std runtime; project explicitly avoids tokio; sync is fine for modal dialogs |
| Manual `ColorImage` construction | `egui_extras::install_image_loaders` | install_image_loaders decodes at render time, harder to control caching; manual construction gives explicit texture lifecycle control |

**Installation (additions needed to bgprunr-app/Cargo.toml):**
```toml
[dependencies]
bgprunr-core = { path = "../bgprunr-core" }
clap = { workspace = true }
indicatif = { workspace = true }
egui = { workspace = true }
eframe = { workspace = true }
arboard = { workspace = true }
rfd = { workspace = true }
```

All versions are already pinned in the workspace `Cargo.toml`. No new workspace entries needed.

**Version verification:**
- eframe 0.34.1 — confirmed on crates.io (released 2026-03-27)
- arboard 3.6.1 — confirmed on crates.io (released 2025-08-23)
- rfd 0.15.4 — confirmed; latest is 0.17.2 but workspace pins 0.15 (acceptable, API is stable)

---

## Architecture Patterns

### Recommended Project Structure

```
crates/bgprunr-app/src/
├── main.rs          # Entry: no args → GUI via eframe::run_native
├── cli.rs           # Existing CLI module (unchanged)
└── gui/
    ├── mod.rs       # eframe::run_native entry + NativeOptions setup
    ├── app.rs       # BgPrunrApp struct + eframe::App impl (ui + logic)
    ├── worker.rs    # WorkerMessage / WorkerResult enums + spawn_worker()
    ├── state.rs     # AppState enum: Empty / Loaded / Processing / Done
    └── views/
        ├── toolbar.rs    # toolbar() function rendering top button row
        ├── canvas.rs     # canvas() function: drop zone or image display
        ├── statusbar.rs  # status_bar() function: text + progress + backend badge
        └── shortcuts.rs  # shortcuts_overlay() function: modal with shortcut table
```

### Pattern 1: eframe 0.34 App Trait — `logic()` + `ui()` Split

**What:** eframe 0.34 deprecated `App::update()`. Logic (channel polling, state mutation) moves to `logic()`; rendering moves to `ui()`.

**When to use:** Always in eframe 0.34. Using the deprecated `update()` still compiles but is wrong practice.

**Example:**
```rust
// Source: docs.rs/eframe/0.34.1/eframe/trait.App.html
impl eframe::App for BgPrunrApp {
    // Called before each ui() frame, also when UI is hidden.
    // MAY NOT render any widgets or paint.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll worker channel (non-blocking)
        while let Ok(msg) = self.worker_rx.try_recv() {
            match msg {
                WorkerResult::Progress(stage, pct) => {
                    self.progress = (stage, pct);
                }
                WorkerResult::Done(result) => {
                    self.state = AppState::Done(result);
                    // Load texture ONCE here, not in ui()
                    let color_image = result_to_color_image(&self.state);
                    self.result_texture = Some(ctx.load_texture(
                        "result",
                        color_image,
                        egui::TextureOptions::default(),
                    ));
                }
                WorkerResult::Cancelled => {
                    self.state = AppState::Loaded;
                }
                WorkerResult::Error(msg) => {
                    self.state = AppState::Error(msg);
                }
            }
        }
        // Handle global keyboard shortcuts here (context-level, not panel-level)
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Escape) { self.handle_escape(); }
            if i.key_pressed(egui::Key::QuestionMark) { self.show_shortcuts = !self.show_shortcuts; }
        });
    }

    // Called each frame to render UI. MUST NOT mutate significant state.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show_inside(ui, |ui| {
            toolbar::render(ui, &mut self.state, &mut self.worker_tx, ...);
        });
        egui::TopBottomPanel::bottom("statusbar").show_inside(ui, |ui| {
            statusbar::render(ui, &self.state, &self.progress, &self.active_backend);
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            canvas::render(ui, &self.state, &self.source_texture, &self.result_texture);
        });
        if self.show_shortcuts {
            shortcuts::render(ui.ctx());
        }
    }
}
```

### Pattern 2: Worker Thread with mpsc + `request_repaint`

**What:** Long-lived thread owns the inference pipeline; communicates results back via channel; signals UI to repaint.

**When to use:** Any time inference is dispatched from the GUI.

**Example:**
```rust
// Source: egui issue #484, CONTEXT.md architecture decisions
pub fn spawn_worker(ctx: egui::Context) -> (Sender<WorkerMessage>, Receiver<WorkerResult>) {
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<WorkerMessage>();
    let (res_tx, res_rx) = std::sync::mpsc::channel::<WorkerResult>();

    std::thread::spawn(move || {
        loop {
            match msg_rx.recv() {
                Ok(WorkerMessage::ProcessImage { bytes, model, cancel }) => {
                    let engine = OrtEngine::new(model).unwrap();
                    let img = load_image_from_bytes(&bytes).unwrap();

                    let result = process_image(&engine, img, |stage, pct| {
                        if cancel.load(Ordering::Relaxed) { return; }
                        res_tx.send(WorkerResult::Progress(stage, pct)).ok();
                        ctx.request_repaint();
                    });

                    if cancel.load(Ordering::Relaxed) {
                        res_tx.send(WorkerResult::Cancelled).ok();
                    } else {
                        res_tx.send(WorkerResult::Done(result.unwrap())).ok();
                    }
                    ctx.request_repaint();
                }
                Ok(WorkerMessage::Cancel) => { /* handled by AtomicBool */ }
                Ok(WorkerMessage::Quit) | Err(_) => break,
            }
        }
    });

    (msg_tx, res_rx)
}
```

### Pattern 3: File Drag-and-Drop via egui RawInput

**What:** egui surfaces dropped files via `ctx.input(|i| i.raw.dropped_files.clone())`.

**When to use:** In `logic()` to detect drops each frame.

**Example:**
```rust
// Source: docs.rs/egui/0.34.1/egui/struct.RawInput.html
// In logic():
let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
for file in dropped {
    if let Some(path) = file.path {
        // load_image_from_path is available from bgprunr_core
        match bgprunr_core::load_image_from_path(&path) {
            Ok(img) => { self.load_image(img, path); }
            Err(e) => { self.state = AppState::Error(e.to_string()); }
        }
    }
}

// Visual hover indicator (in canvas ui):
let hovered: bool = ctx.input(|i| !i.raw.hovered_files.is_empty());
// Use hovered to change drop zone border color
```

### Pattern 4: File Dialogs (rfd — synchronous, called from UI thread)

**What:** rfd's synchronous `FileDialog` runs the native OS dialog modally. Safe to call from the eframe render thread for user-triggered actions.

**When to use:** Triggered by button press or keyboard shortcut in `logic()`. Launch dialog, get path, load file.

**Example:**
```rust
// Source: docs.rs/rfd/0.15.4/rfd/struct.FileDialog.html
// Open dialog (Ctrl+O):
if let Some(path) = rfd::FileDialog::new()
    .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
    .set_title("Open Image")
    .pick_file()
{
    match bgprunr_core::load_image_from_path(&path) {
        Ok(img) => self.load_image(img, path),
        Err(e) => self.state = AppState::Error(e.to_string()),
    }
}

// Save dialog (Ctrl+S):
if let Some(path) = rfd::FileDialog::new()
    .add_filter("PNG Image", &["png"])
    .set_file_name("result.png")
    .set_title("Save PNG")
    .save_file()
{
    if let AppState::Done(ref result) = self.state {
        let bytes = bgprunr_core::encode_rgba_png(&result.rgba_image).unwrap();
        std::fs::write(&path, bytes).unwrap();
    }
}
```

**Warning:** Do NOT call rfd dialogs from the worker thread. rfd requires access to the window handle for parent window association (especially on Windows/macOS). Call from the UI thread only.

### Pattern 5: arboard Clipboard Image Copy

**What:** arboard `Clipboard::set_image()` copies RGBA pixels to the system clipboard. Clipboard must live on the App struct (not be dropped).

**When to use:** Ctrl+C or Copy button press in Done state.

**Example:**
```rust
// Source: docs.rs/arboard/3.6.1/arboard/struct.Clipboard.html
// In App struct:
// clipboard: arboard::Clipboard  (initialized in new(), stored for app lifetime)

// On copy action:
if let AppState::Done(ref result) = self.state {
    let rgba = result.rgba_image.as_flat_samples();
    let img_data = arboard::ImageData {
        width: result.rgba_image.width() as usize,
        height: result.rgba_image.height() as usize,
        bytes: std::borrow::Cow::from(rgba.as_slice()),
    };
    match self.clipboard.set_image(img_data) {
        Ok(_) => self.status_text = "Copied to clipboard".to_string(),
        Err(e) => self.status_text = format!("Could not copy: {e}"),
    }
}
```

### Pattern 6: Texture Upload (Once on State Transition)

**What:** `ctx.load_texture()` must be called ONCE, not every frame. Store `TextureHandle` on App struct.

**When to use:** Exactly when state transitions to `Done` (in `logic()`) and when a new source image is loaded.

**Example:**
```rust
// Source: docs.rs/egui/0.34.1/egui/struct.Context.html
// In logic(), on receiving WorkerResult::Done:
let rgba = result.rgba_image.as_flat_samples();
let color_image = egui::ColorImage::from_rgba_unmultiplied(
    [result.rgba_image.width() as usize, result.rgba_image.height() as usize],
    rgba.as_slice(),
);
self.result_texture = Some(ctx.load_texture(
    "result",
    color_image,
    egui::TextureOptions::default(),
));
```

### Pattern 7: Platform-Correct Keyboard Shortcuts

**What:** Use `Modifiers::command` (not `.ctrl`) for Ctrl/Cmd+Key shortcuts.

**When to use:** All shortcuts that should use Ctrl on Linux/Windows and Cmd on macOS.

**Example:**
```rust
// Source: ARCHITECTURE.md — Keyboard Shortcuts section
ctx.input(|i| {
    if i.modifiers.command && i.key_pressed(egui::Key::O) { /* open */ }
    if i.modifiers.command && i.key_pressed(egui::Key::R) { /* remove bg */ }
    if i.modifiers.command && i.key_pressed(egui::Key::S) { /* save */ }
    if i.modifiers.command && i.key_pressed(egui::Key::C) { /* copy */ }
    if i.key_pressed(egui::Key::Escape) { /* cancel / close */ }
    // ? key: egui Key::QuestionMark
    if i.key_pressed(egui::Key::QuestionMark) { /* shortcut overlay */ }
});
```

### Anti-Patterns to Avoid

- **Calling `ctx.load_texture()` inside `ui()`:** Uploads texture to GPU every frame. Always guard behind a "newly arrived result" check or call only in `logic()`.
- **Using `App::update()` instead of `App::ui()` + `App::logic()`:** Deprecated in eframe 0.34. Compiles with a warning; use the new split API.
- **Using `Modifiers::ctrl` for cross-platform shortcuts:** Breaks Cmd+Key on macOS. Always use `Modifiers::command`.
- **Dropping `arboard::Clipboard` immediately after set_image:** Clipboard data disappears on Wayland before other apps can paste. Store on App struct for app lifetime.
- **Calling `rfd::FileDialog` from the worker thread:** rfd requires a window handle; calling from a non-UI thread causes panics on macOS/Windows.
- **Storing `ctx: &Context` across frames:** `Context` is `Clone` and cheap to clone; store a clone if you need it outside `ui()`/`logic()`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Native file open/save dialogs | Custom file browser widget | `rfd::FileDialog` | Platform-native look, correct macOS NSOpenPanel / Windows IFileDialog behavior, portal support on Linux Wayland |
| Clipboard image copy | Raw X11/Wayland protocol calls | `arboard::Clipboard::set_image()` | Wayland ownership model (compositor serves paste requests from your process) is non-trivial; arboard handles it |
| Image format detection and decoding | Custom decoder | `bgprunr_core::load_image_from_path()` — already exists | Already handles PNG/JPEG/WebP/BMP, large-image check, downscale prompt; do not duplicate |
| PNG encoding for save | Manual PNG encoder | `bgprunr_core::encode_rgba_png()` — already exists | Already encodes RGBA with correct alpha channel |
| GPU texture management | Manual wgpu texture upload | `egui::Context::load_texture()` + `TextureHandle` | Handles GPU upload, mip-mapping, format conversion |
| Cross-platform keyboard modifier mapping | if cfg!(target_os = "macos") checks | `egui::Modifiers::command` | Already maps Ctrl → Cmd correctly per platform |

**Key insight:** The inference, image I/O, and encoding pipeline is entirely implemented in `bgprunr-core`. Phase 4 is a pure GUI layer — every computation delegates to the existing core.

---

## Common Pitfalls

### Pitfall 1: Using Deprecated `App::update()` Instead of `ui()` + `logic()`

**What goes wrong:** Code compiles and runs but uses deprecated API. Worse — `update()` receives `ctx: &Context` while `ui()` receives `ui: &mut Ui`. Mixing patterns leads to layout confusion where panels don't nest correctly.

**Why it happens:** All pre-0.34 egui examples, blog posts, and the ARCHITECTURE.md reference `App::update()`. The change landed in 0.34.0 (2026-03-26).

**How to avoid:** Implement `fn logic(&mut self, ctx: &Context, _frame: &mut Frame)` for channel polling and state mutation. Implement `fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame)` for all widget rendering using `show_inside()` on panels.

**Warning signs:** Compiler warning "use of deprecated method `update`" or panels appearing outside the expected area.

### Pitfall 2: Blocking the `ui()` Frame with File Dialog or Inference

**What goes wrong:** The window freezes while rfd shows a dialog or while an inference call is made directly in `ui()`. On macOS this triggers the spinning beachball. (Detailed in PITFALLS.md #3.)

**Why it happens:** `ui()` is called on the egui render thread. Any blocking call prevents frame delivery.

**How to avoid:** For inference — use worker thread (CONTEXT.md architecture). For file dialogs — rfd dialogs are modal and block until user action, which is the expected UX, but must be triggered from a user interaction (button press / shortcut), not looped or polled.

### Pitfall 3: Texture Re-Upload Every Frame

**What goes wrong:** After showing inference result, CPU/GPU usage stays at 100%. (Detailed in PITFALLS.md #4.)

**Why it happens:** `ctx.load_texture()` called unconditionally in `ui()` each frame.

**How to avoid:** Call `load_texture()` only in `logic()` when the state transitions to `Done`. Store `Option<TextureHandle>` on `App` struct. Set it to `None` when loading a new image.

### Pitfall 4: Wayland Clipboard Ownership Loss

**What goes wrong:** Copy appears to succeed but paste in another app returns nothing. (Detailed in PITFALLS.md #8.)

**Why it happens:** `arboard::Clipboard` dropped immediately after `set_image()`. On Wayland, the clipboard "server" is the app itself; dropping it ends the server.

**How to avoid:** `arboard::Clipboard` must be a field on `BgPrunrApp`. Initialize in `new()`, keep alive for app lifetime. The `wayland-data-control` feature must be enabled (it is in the workspace Cargo.toml).

### Pitfall 5: Drag-and-Drop Silently Disabled on Windows

**What goes wrong:** File drops do nothing on Windows with no error.

**Why it happens:** On Windows, `ViewportBuilder` drag-and-drop defaults to disabled (COM OLE drag-and-drop setup is opt-in). Quote from docs: "on Windows: enable drag and drop support. Drag and drop can not be disabled on other platforms."

**How to avoid:** Set `with_drag_and_drop(true)` on `ViewportBuilder` in `NativeOptions`.

```rust
let native_options = eframe::NativeOptions {
    viewport: egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([640.0, 480.0])
        .with_drag_and_drop(true),
    ..Default::default()
};
```

### Pitfall 6: Cancel Not Working Between Pipeline Stages

**What goes wrong:** Escape sends cancel but inference continues to completion anyway.

**Why it happens:** `bgprunr_core::process_image()` takes a progress callback but the current design does not thread a cancellation signal into that callback.

**How to avoid:** Pass an `Arc<AtomicBool>` cancel flag into the worker. In the progress callback closure, check `cancel.load(Ordering::Relaxed)`. After `process_image()` returns, check the flag again before sending `WorkerResult::Done`. If cancelled, send `WorkerResult::Cancelled` instead. UI resets to `AppState::Loaded` on receiving `Cancelled`.

**Note:** This requires the worker to check the flag, not the core. The `process_image()` API won't be modified — cancellation is worker-level.

### Pitfall 7: Window Title Not Updating When Image Loaded

**What goes wrong:** Title stays "BgPrunR" even when an image is loaded.

**Why it happens:** Window title in eframe 0.34 must be set via `ViewportCommand::Title` sent each frame, or the title is only set at creation.

**How to avoid:**
```rust
// In ui() when title needs updating:
ctx.send_viewport_cmd(egui::ViewportCommand::Title(
    format!("BgPrunR — {}", self.loaded_filename)
));
```

---

## Code Examples

### Minimal eframe 0.34 App Skeleton

```rust
// Source: docs.rs/eframe/0.34.1/eframe/index.html
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("BgPrunR")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "bgprunr",   // app_id — used for Wayland window grouping
        native_options,
        Box::new(|cc| Ok(Box::new(BgPrunrApp::new(cc))))
    )
}
```

### App Struct Fields

```rust
struct BgPrunrApp {
    // State
    state: AppState,
    loaded_filename: Option<String>,

    // Worker thread
    worker_tx: std::sync::mpsc::Sender<WorkerMessage>,
    worker_rx: std::sync::mpsc::Receiver<WorkerResult>,
    cancel_flag: Arc<AtomicBool>,

    // Progress
    progress_stage: String,
    progress_pct: f32,
    status_text: String,

    // Textures (loaded ONCE on state transition, not every frame)
    source_texture: Option<egui::TextureHandle>,
    result_texture: Option<egui::TextureHandle>,

    // Clipboard (must live for app lifetime — Wayland ownership)
    clipboard: arboard::Clipboard,

    // UI state
    show_shortcuts: bool,
    selected_model: bgprunr_core::ModelKind,
    active_backend: String,
}
```

### ColorImage Construction from `image::RgbaImage`

```rust
// Source: docs.rs/egui/0.34.1/egui/struct.ColorImage.html
fn rgba_to_color_image(img: &image::RgbaImage) -> egui::ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let pixels = img.as_flat_samples();
    egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice())
}
```

### arboard ImageData from `image::RgbaImage`

```rust
// Source: docs.rs/arboard/3.6.1/arboard/struct.ImageData.html
fn rgba_to_clipboard_image(img: &image::RgbaImage) -> arboard::ImageData<'_> {
    arboard::ImageData {
        width: img.width() as usize,
        height: img.height() as usize,
        bytes: std::borrow::Cow::from(img.as_flat_samples().as_slice()),
    }
}
```

### Drag-and-Drop Detection

```rust
// Source: docs.rs/egui/0.34.1/egui/struct.RawInput.html
// In logic():
let dropped = ctx.input(|i| i.raw.dropped_files.clone());
for file in &dropped {
    if let Some(path) = &file.path {
        self.handle_open_path(path.clone());
    }
}

// In canvas ui() — visual hover:
let is_hovered = ctx.input(|i| !i.raw.hovered_files.is_empty());
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `App::update(ctx, frame)` | `App::logic(ctx, frame)` + `App::ui(ui, frame)` | eframe 0.34.0 (2026-03-26) | Logic and rendering are now separate; `ui()` receives `&mut Ui` not `&Context` |
| `egui::TopBottomPanel::show(ctx, ...)` | `show_inside(ui, ...)` inside `ui()` method | eframe 0.34 | Panels attach to a parent `Ui`, not directly to `Context` |
| `glow` renderer (OpenGL) | `wgpu` renderer (default in eframe 0.30+) | eframe 0.30 | Better GPU texture upload performance; required for correct display on some platforms |

**Deprecated/outdated:**
- `App::update()`: Deprecated in 0.34. Compiles with warning. Use `logic()` + `ui()`.
- `egui::CtxRef`: Replaced with `egui::Context` years ago; not relevant but old docs reference it.
- `eframe::egui::Window::new(...).show(ctx, ...)` called from `update()`: Still works but should use `ctx.show_viewport_deferred()` for secondary windows.

---

## Open Questions

1. **`process_image()` cancellation granularity**
   - What we know: The core API takes a progress callback `|stage, pct|`. The worker can check `cancel_flag` in that callback, but cannot interrupt `session.run()` mid-inference.
   - What's unclear: How long does each stage take? If `Inferring` stage takes 10s and cancel is only checked between stages, the UX is laggy.
   - Recommendation: Check cancel flag in the progress callback AND between stages. Accept that mid-inference cancellation may have up to one stage of latency (typically 1-5s). This is adequate for Phase 4.

2. **`ViewportCommand::Title` update frequency**
   - What we know: Title must be set via `ctx.send_viewport_cmd(ViewportCommand::Title(...))`.
   - What's unclear: Should this be sent every frame or only on change? Sending every frame seems wasteful.
   - Recommendation: Only send when `loaded_filename` changes. Track a `title_dirty: bool` flag and clear after sending.

3. **rfd on Linux Wayland — portal vs GTK**
   - What we know: rfd has an `xdg-portal` feature for portal-based dialogs on Linux.
   - What's unclear: The workspace pins `rfd = "0.15"` without the xdg-portal feature. GTK dialogs work on most Linux desktops but may not work in all Wayland sandbox environments.
   - Recommendation: Accept GTK dialogs for Phase 4. The `xdg-portal` feature can be added in Phase 6 distribution hardening if needed.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `cargo test` (no external test framework) |
| Config file | none — standard cargo test discovery |
| Quick run command | `cargo test -p bgprunr-app --features dev-models` |
| Full suite command | `cargo test --workspace --features dev-models` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LOAD-01 | Dropped file path is parsed and image is loaded | unit | `cargo test -p bgprunr-app --features dev-models test_drop_zone_accepts_image_path` | ❌ Wave 0 |
| LOAD-02 | Keyboard shortcut Ctrl+O dispatches open action | unit (input handler) | `cargo test -p bgprunr-app --features dev-models test_ctrl_o_dispatches_open` | ❌ Wave 0 |
| OUT-01 | Save action writes valid RGBA PNG to temp path | unit | `cargo test -p bgprunr-app --features dev-models test_save_writes_png` | ❌ Wave 0 |
| OUT-02 | Clipboard copy constructs correct ImageData dimensions | unit | `cargo test -p bgprunr-app --features dev-models test_clipboard_image_data` | ❌ Wave 0 |
| UX-01 | All 6 shortcuts are dispatched correctly | unit (input handler) | `cargo test -p bgprunr-app --features dev-models test_keyboard_shortcuts` | ❌ Wave 0 |
| UX-03 | Escape during Processing sets cancel_flag and transitions to Loaded | unit (state machine) | `cargo test -p bgprunr-app --features dev-models test_escape_cancels_processing` | ❌ Wave 0 |
| UX-04 | `?` key toggles show_shortcuts bool | unit (state) | `cargo test -p bgprunr-app --features dev-models test_question_mark_toggles_overlay` | ❌ Wave 0 |

**Note on GUI testing:** egui does not support headless rendering in unit tests. The above tests are for pure logic — state machine transitions, input dispatch, data conversion — not for rendered output. Test functions will construct `BgPrunrApp` with a mock `egui::Context` where possible. Keyboard shortcut handling extracted to a `handle_input()` function makes it testable without egui context.

### Sampling Rate

- **Per task commit:** `cargo test -p bgprunr-app --features dev-models`
- **Per wave merge:** `cargo test --workspace --features dev-models`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/bgprunr-app/src/gui/tests/mod.rs` — test module for GUI logic covering all Phase 4 reqs
- [ ] `crates/bgprunr-app/src/gui/tests/state_tests.rs` — AppState transitions (Empty→Loaded→Processing→Done→Loaded)
- [ ] `crates/bgprunr-app/src/gui/tests/input_tests.rs` — keyboard shortcut dispatch, cancel, shortcut overlay
- [ ] `crates/bgprunr-app/src/gui/tests/clipboard_tests.rs` — ImageData construction from RgbaImage
- [ ] Worker thread integration test file if one does not exist

---

## Sources

### Primary (HIGH confidence)

- `docs.rs/eframe/0.34.1` — App trait `ui()` and `logic()` methods, `run_native` signature, `NativeOptions`, `ViewportBuilder::with_drag_and_drop()`
- `docs.rs/egui/0.34.1` — `RawInput::dropped_files`, `RawInput::hovered_files`, `ColorImage::from_rgba_unmultiplied`, `Context::load_texture`, `Context::request_repaint`, `Modifiers::command`
- `docs.rs/arboard/3.6.1` — `ImageData` struct (width, height, bytes: Cow<[u8]>), `Clipboard::set_image()`, Wayland notes
- `docs.rs/rfd/0.15.4` — `FileDialog::pick_file()`, `save_file()`, `add_filter()`, blocking behavior
- `github.com/emilk/egui CHANGELOG.md` — eframe 0.34.0 deprecation of `App::update()`, replacement with `ui()` + `logic()`
- `crates.io/crates/eframe` — 0.34.1 release date 2026-03-27 confirmed
- `.planning/research/STACK.md` — Version pins verified, arboard Wayland note, egui threading model
- `.planning/research/PITFALLS.md` — Pitfall #3 (UI freeze), #4 (texture re-upload), #7 (clipboard Wayland), cross-referenced
- `ARCHITECTURE.md` — Worker thread architecture, state machine, `Modifiers::command` pattern, module layout

### Secondary (MEDIUM confidence)

- `github.com/emilk/egui issue #484` — Community-validated channel-based worker pattern (referenced in PITFALLS.md)
- `crates.io/crates/rfd` — rfd 0.15.4 confirmed as latest 0.15.x; 0.17.2 is latest overall

### Tertiary (LOW confidence)

- None — all claims verified against official docs or existing project research.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all crates already in workspace, versions verified on crates.io
- Architecture: HIGH — eframe 0.34 API verified on docs.rs; `ui()`+`logic()` split confirmed from CHANGELOG
- Pitfalls: HIGH — cross-referenced with PITFALLS.md (which cites egui issue tracker and official docs)
- eframe 0.34 API changes: HIGH — verified on docs.rs and CHANGELOG

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (egui releases every few months; 0.35 would require re-verification)

**Critical implementation note for planner:** The ARCHITECTURE.md and CONTEXT.md reference `App::update()` throughout. ALL plans and tasks must use `App::logic()` + `App::ui()` instead. This is not optional — `update()` is deprecated in the exact version being used (0.34.1).
