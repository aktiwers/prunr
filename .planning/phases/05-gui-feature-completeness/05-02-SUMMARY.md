---
phase: 05-gui-feature-completeness
plan: 02
subsystem: ui
tags: [egui, settings-dialog, reveal-animation, dissolve, mask, egui-overlay]

# Dependency graph
requires:
  - phase: 05-01
    provides: Settings struct, AppState::Animating, anim_progress/anim_mask fields, theme constants (ANIM_DURATION_SECS, ANIM_MASK_THRESHOLD, SETTINGS_DIALOG_WIDTH/HEIGHT)

provides:
  - Settings modal dialog with 5 fields, centered overlay, Ctrl+, shortcut, Escape close
  - Reveal animation with ease-out cubic dissolve (background fades, subject stays sharp)
  - Animation state transitions in WorkerResult::Done (Animating vs Done based on settings)
  - Skip-animation support (any key or click skips to Done)
  - Animation disabled toggle in settings

affects: [05-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Centered modal overlay: backdrop painter + egui::Window with anchor(Align2::CENTER_CENTER)"
    - "Per-frame animation texture: build_animation_frame() builds ColorImage each frame, uploaded with unique key"
    - "Mask-based dissolve: alpha channel of result image classifies subject vs background pixels"
    - "Ease-out cubic: t = 1 - (1-t)^3 for smooth deceleration into final state"
    - "Animation skip: egui::Event::Key { pressed: true } scanned in input closure"

key-files:
  created:
    - crates/bgprunr-app/src/gui/views/settings.rs
    - crates/bgprunr-app/src/gui/views/animation.rs
    - crates/bgprunr-app/src/gui/tests/settings_tests.rs
    - crates/bgprunr-app/src/gui/tests/anim_tests.rs
  modified:
    - crates/bgprunr-app/src/gui/views/mod.rs
    - crates/bgprunr-app/src/gui/views/canvas.rs
    - crates/bgprunr-app/src/gui/app.rs
    - crates/bgprunr-app/src/gui/tests/mod.rs

key-decisions:
  - "egui InputState has no keys_pressed field — use i.events.iter().any(|e| matches!(e, Event::Key { pressed: true, .. })) for any-key-pressed detection"
  - "ColorImage in epaint-0.34 requires source_size: Vec2 field alongside size and pixels"
  - "Animation frame builds a downscaled ColorImage capped to canvas size to avoid GPU upload of full-resolution images during animation"
  - "Settings model synced to selected_model each logic() frame so model selection in dialog always reflects current processing intent"

patterns-established:
  - "Modal overlay pattern: backdrop rect_filled + egui::Window::new().anchor(CENTER_CENTER).frame(OVERLAY_BG/BORDER/corner_radius)"
  - "Animation as per-frame ColorImage: build fresh texture each frame, upload with load_texture(unique_key) — egui caches by key"

requirements-completed: [UX-02, ANIM-01, ANIM-02, ANIM-03]

# Metrics
duration: 4min
completed: 2026-04-07
---

# Phase 5 Plan 02: Settings Dialog and Reveal Animation Summary

**Settings modal (Ctrl+,) with 5 configuration fields plus mask-aware background-dissolve reveal animation using ease-out cubic over 0.75s**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-07T15:18:52Z
- **Completed:** 2026-04-07T15:22:52Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Settings dialog renders as centered modal overlay with model dropdown, auto-remove checkbox, parallel jobs slider, animation toggle, and read-only backend label
- Reveal animation plays on processing completion: background pixels dissolve from opaque to transparent while subject pixels stay 100% sharp, with ease-out cubic easing
- Animation state machine wired: WorkerResult::Done transitions to Animating (or Done directly if disabled), anim_progress advances each frame, any key/click skips to Done
- 13 new unit tests (7 settings + 6 animation), all passing alongside existing 16 tests

## Task Commits

1. **Task 1: Settings dialog view and wiring** - `08a0e22` (feat)
2. **Task 2: Reveal animation rendering and state transitions** - `4928d39` (feat)

## Files Created/Modified
- `crates/bgprunr-app/src/gui/views/settings.rs` - Centered modal overlay with 5 settings fields, uses ComboBox/Slider/checkbox widgets
- `crates/bgprunr-app/src/gui/views/animation.rs` - build_animation_frame() producing per-frame ColorImage with mask-based dissolve
- `crates/bgprunr-app/src/gui/views/canvas.rs` - Added render_animating() function with checkerboard + animation texture upload
- `crates/bgprunr-app/src/gui/app.rs` - Ctrl+, shortcut, settings::render call, animation advancement, WorkerResult::Done Animating transition
- `crates/bgprunr-app/src/gui/views/mod.rs` - Added pub mod settings; pub mod animation;
- `crates/bgprunr-app/src/gui/tests/settings_tests.rs` - 7 settings unit tests
- `crates/bgprunr-app/src/gui/tests/anim_tests.rs` - 6 animation unit tests
- `crates/bgprunr-app/src/gui/tests/mod.rs` - Added settings_tests and anim_tests modules

## Decisions Made
- `keys_pressed` field does not exist on egui 0.34 InputState — replaced with `i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))` for skip detection
- `ColorImage` in epaint 0.34 requires `source_size: Vec2` field — added with `Vec2::new(out_w, out_h)` matching output dimensions
- Animation downscales frame to canvas size for performance (avoids uploading multi-megapixel frames each render tick)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ColorImage missing source_size field**
- **Found during:** Task 2 (reveal animation rendering)
- **Issue:** epaint-0.34 ColorImage struct requires source_size: Vec2 field — plan's struct literal was missing it
- **Fix:** Added `source_size: Vec2::new(out_w as f32, out_h as f32)` to ColorImage construction
- **Files modified:** crates/bgprunr-app/src/gui/views/animation.rs
- **Verification:** cargo test passes
- **Committed in:** 4928d39

**2. [Rule 1 - Bug] Fixed invalid InputState field keys_pressed**
- **Found during:** Task 2 (animation skip detection)
- **Issue:** Plan used `i.keys_pressed.is_empty()` but egui 0.34 InputState has no `keys_pressed` field
- **Fix:** Replaced with event scan: `i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))`
- **Files modified:** crates/bgprunr-app/src/gui/app.rs
- **Verification:** cargo test passes, all 29 tests green
- **Committed in:** 4928d39

---

**Total deviations:** 2 auto-fixed (both Rule 1 - API compatibility bugs)
**Impact on plan:** Both fixes necessary for compilation; no scope creep.

## Issues Encountered
- egui 0.34 API differences from plan's expected API (keys_pressed, ColorImage source_size field) — both resolved with Rule 1 auto-fixes

## Next Phase Readiness
- Settings dialog and reveal animation complete — ready for Phase 5 Plan 03
- All 29 tests passing, cargo check clean

---
*Phase: 05-gui-feature-completeness*
*Completed: 2026-04-07*
