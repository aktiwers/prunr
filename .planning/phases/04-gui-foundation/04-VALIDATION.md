---
phase: 4
slug: gui-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml workspace |
| **Quick run command** | `cargo test -p prunr-app --lib` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p prunr-app --lib`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | LOAD-01 | integration | `cargo test -p prunr-app test_load_image` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | LOAD-02 | integration | `cargo test -p prunr-app test_drag_drop` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | OUT-01 | integration | `cargo test -p prunr-app test_save_png` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | OUT-02 | integration | `cargo test -p prunr-app test_clipboard` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | UX-01 | unit | `cargo test -p prunr-app test_state_machine` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | UX-03 | unit | `cargo test -p prunr-app test_cancel` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | UX-04 | unit | `cargo test -p prunr-app test_shortcuts` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/prunr-app/src/state.rs` — state machine unit tests (transitions, cancel)
- [ ] `crates/prunr-app/tests/` — integration test stubs for load/save/clipboard
- [ ] Test fixtures: sample PNG/JPEG images for load testing

*Note: egui does not support headless rendering — UI interaction tests are manual-only. Logic tests target the state machine and worker channel.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Drag-and-drop loads image | LOAD-01 | egui has no headless drag-drop simulation | Drop a PNG onto the window; verify image appears in viewer |
| UI stays responsive during inference | UX-01 | Requires visual confirmation of no freeze | Start inference on large image; move/resize window during processing |
| Progress spinner visible | UX-01 | Visual rendering check | Start inference; confirm spinner animation is visible |
| Clipboard paste in external app | OUT-02 | Requires cross-app interaction | Copy result, paste in GIMP/Inkscape; verify transparency preserved |
| Shortcut overlay on ? press | UX-04 | Visual overlay rendering | Press ?; verify overlay lists all shortcuts |
| Escape cancels inference | UX-03 | Requires active inference + UI interaction | Start inference; press Escape; confirm status shows "Cancelled" |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
