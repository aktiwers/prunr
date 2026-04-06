---
phase: 2
slug: core-inference-engine
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml (workspace) |
| **Quick run command** | `cargo test -p bgprunr-core` |
| **Full suite command** | `cargo test -p bgprunr-core --all-features` |
| **Estimated runtime** | ~15 seconds (includes inference on test images) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p bgprunr-core`
- **After every plan wave:** Run `cargo test -p bgprunr-core --all-features`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | CORE-01 | integration | `cargo test -p bgprunr-core test_process_image` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CORE-02 | unit | `cargo test -p bgprunr-core test_model_selection` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CORE-03 | unit | `cargo test -p bgprunr-core test_active_provider` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CORE-04 | unit | `cargo test -p bgprunr-core test_progress_callback` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CORE-05 | integration | `cargo test -p bgprunr-core test_rembg_reference` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | LOAD-03 | unit | `cargo test -p bgprunr-core test_format_support` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | LOAD-04 | unit | `cargo test -p bgprunr-core test_large_image_warning` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Reference test images downloaded/committed to `tests/references/`
- [ ] `scripts/generate_references.py` creates ground truth masks from rembg
- [ ] Test fixtures for all supported formats (PNG, JPEG, WebP, BMP)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU EP detection | CORE-03 | Requires CUDA/CoreML/DirectML hardware | Run on GPU machine, verify active_provider() returns non-CPU |
| Reference test accuracy | CORE-05 | Requires rembg Python + test images | Run generate_references.py, then cargo test |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
