---
phase: 1
slug: workspace-scaffolding
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml (workspace) |
| **Quick run command** | `cargo test --workspace` |
| **Full suite command** | `cargo test --workspace --all-features` |
| **Estimated runtime** | ~5 seconds (scaffolding phase, minimal tests) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --workspace`
- **After every plan wave:** Run `cargo test --workspace --all-features`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | DIST-01 | build | `cargo build --workspace` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | DIST-02 | unit | `cargo test -p bgprunr-models` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | DIST-03 | ci | GitHub Actions workflow | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | DIST-04 | build | `cargo build --release` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/bgprunr-core/src/lib.rs` — placeholder with version test
- [ ] `crates/bgprunr-models/src/lib.rs` — model bytes accessibility test
- [ ] Root `Cargo.toml` workspace configuration

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI builds on all platforms | DIST-03 | Requires GitHub Actions runners | Push to branch, verify all matrix jobs pass |
| cargo-dist produces artifacts | DIST-04 | Requires release tag trigger | Create test tag, verify artifacts in GitHub release |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
