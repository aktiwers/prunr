---
phase: 5
slug: gui-feature-completeness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test (`cargo test`) |
| **Config file** | `[lib]` section in prunr-app/Cargo.toml with `src/lib.rs` |
| **Quick run command** | `cargo test -p prunr-app --lib 2>&1 \| tail -20` |
| **Full suite command** | `cargo test --workspace --lib 2>&1 \| tail -30` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p prunr-app --lib 2>&1 | tail -20`
- **After every plan wave:** Run `cargo test --workspace --lib 2>&1 | tail -30`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | VIEW-01 | unit | `cargo test -p prunr-app --lib zoom` | W0 | pending |
| 05-01-02 | 01 | 1 | VIEW-02 | unit | `cargo test -p prunr-app --lib pan` | W0 | pending |
| 05-01-03 | 01 | 1 | VIEW-03, VIEW-04, VIEW-05 | unit | `cargo test -p prunr-app --lib fit_zoom` | W0 | pending |
| 05-02-01 | 02 | 2 | UX-02 | unit | `cargo test -p prunr-app --lib settings` | W0 | pending |
| 05-02-02 | 02 | 2 | ANIM-01, ANIM-02, ANIM-03 | unit | `cargo test -p prunr-app --lib anim` | W0 | pending |
| 05-03-01 | 03 | 3 | BATCH-01, BATCH-02, BATCH-03 | unit | `cargo test -p prunr-app --lib batch` | W0 | pending |
| 05-03-02 | 03 | 3 | BATCH-04, BATCH-05, BATCH-06 | unit | `cargo test -p prunr-app --lib worker` | W0 | pending |
| 05-03-03 | 03 | 3 | UX-05 | unit | `cargo test -p prunr-app --lib nav_keys` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `crates/prunr-app/src/gui/tests/zoom_pan_tests.rs` — stubs for VIEW-01, VIEW-02, VIEW-05
- [ ] `crates/prunr-app/src/gui/tests/batch_tests.rs` — stubs for BATCH-01 through BATCH-06, UX-05
- [ ] `crates/prunr-app/src/gui/tests/anim_tests.rs` — stubs for ANIM-01, ANIM-02, ANIM-03
- [ ] `crates/prunr-app/src/gui/tests/settings_tests.rs` — stubs for UX-02
- [ ] Extend `crates/prunr-app/src/gui/tests/state_tests.rs` — add Animating variant test

*Existing infrastructure (`tests/mod.rs`, `tests/state_tests.rs`, `tests/input_tests.rs`, `tests/clipboard_tests.rs`) provides the pattern.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Scroll-wheel zoom visual centering | VIEW-01 | Requires live egui context with mouse events | Scroll on image, verify zoom centers on cursor |
| Space+drag visual pan | VIEW-02 | Requires live egui context with pointer interaction | Hold Space, drag, verify image pans |
| Checkerboard rendering appearance | VIEW-03 | Visual correctness check | Toggle to result, verify checkerboard behind transparent areas |
| Reveal animation visual smoothness | ANIM-02 | Animation rendering requires GPU context | Process image, watch dissolve animation |
| Sidebar thumbnail rendering | BATCH-01 | Visual thumbnail quality check | Drop 3+ images, verify sidebar thumbnails render |
| Drag-reorder visual feedback | BATCH-03 | DnD visual feedback requires live context | Drag sidebar item, verify reorder animation |
| Settings dialog layout | UX-02 | Dialog layout verification | Press Ctrl+, verify dialog layout and controls |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
