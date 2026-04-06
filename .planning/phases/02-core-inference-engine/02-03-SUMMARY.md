---
phase: "02"
plan: "03"
subsystem: bgprunr-core
tags: [ort, inference, pipeline, session, execution-provider]
dependency_graph:
  requires: ["02-01", "02-02"]
  provides: ["OrtEngine", "process_image"]
  affects: ["bgprunr-cli", "bgprunr-app"]
tech_stack:
  added: ["ort session management", "Mutex<Session> interior mutability"]
  patterns: ["session-once-reuse-many", "progress-callback", "with_session closure API"]
key_files:
  created:
    - crates/bgprunr-core/src/pipeline.rs
  modified:
    - crates/bgprunr-core/src/engine.rs
    - crates/bgprunr-core/src/lib.rs
    - crates/bgprunr-core/Cargo.toml
    - Cargo.toml
decisions:
  - "OrtEngine wraps Mutex<Session> because ort::Session::run() requires &mut self; Mutex enables shared &OrtEngine references while satisfying mutability"
  - "with_session() closure API on OrtEngine gives pipeline.rs safe mutable access without exposing Session directly"
  - "ndarray upgraded from 0.16 to 0.17 to match ort 2.0-rc.12 dependency requirement"
  - "session.inputs()[0].name() queried at runtime per research pattern — never hardcoded"
  - "dev-models and cuda features added to bgprunr-core/Cargo.toml to propagate feature flags cleanly"
metrics:
  duration_seconds: 510
  completed_date: "2026-04-06"
  tasks_completed: 2
  files_changed: 5
---

# Phase 02 Plan 03: OrtEngine and process_image Pipeline Summary

**One-liner:** ORT session management with Mutex interior mutability and full process_image() pipeline (Decode → Resize → Normalize → Infer → Postprocess → Alpha) with per-stage progress callbacks.

## What Was Built

### Task 1: OrtEngine in engine.rs (commit 166097e)

Implemented `OrtEngine` struct in `crates/bgprunr-core/src/engine.rs`:

- `OrtEngine::new(ModelKind, intra_threads)` creates an ORT session with Level3 optimization
- EP priority list: CUDA (feature-gated) → CoreML (macOS) → DirectML (Windows) → CPU
- `Mutex<Session>` for interior mutability — ort's `Session::run()` requires `&mut self`
- `with_session()` closure API gives pipeline.rs mutable access via `&OrtEngine`
- `active_provider()` returns best-effort provider name from compile-time flags
- `dev-models` feature added to `bgprunr-core/Cargo.toml` propagating `bgprunr-models/dev-models`

### Task 2: pipeline.rs process_image() (commit de6834f)

Implemented `process_image()` in `crates/bgprunr-core/src/pipeline.rs`:

- Full pipeline: load_image_from_bytes → check_large_image → preprocess → ORT run → postprocess → encode_rgba_png
- Six progress stages with percentages: Decode(0.0), Resize(0.2), Normalize(0.4), Infer(0.5), Postprocess(0.8), Alpha(0.95)
- Input name queried at runtime via `session.inputs()[0].name()` — never hardcoded
- LargeImage guard runs before tensor allocation (fail fast, no memory waste)
- `lib.rs` updated: exports `OrtEngine`, `process_image`, and all format utilities

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ort::session::GraphOptimizationLevel import path corrected**
- **Found during:** Task 1 compilation
- **Issue:** Plan showed `ort::session::GraphOptimizationLevel` but actual path is `ort::session::builder::GraphOptimizationLevel`
- **Fix:** Updated import path
- **Files modified:** crates/bgprunr-core/src/engine.rs

**2. [Rule 1 - Bug] with_intra_op_threads() renamed to with_intra_threads() in ort 2.0**
- **Found during:** Task 1 compilation
- **Issue:** Plan used `with_intra_op_threads(n as i16)` but ort API is `with_intra_threads(usize)`
- **Fix:** Used correct method name and removed i16 cast
- **Files modified:** crates/bgprunr-core/src/engine.rs

**3. [Rule 1 - Bug] From<ort::Error> not implemented for CoreError**
- **Found during:** Task 1 compilation
- **Issue:** Plan used `?` operator on ort builder chain but `ort::Error<R>` variants are not `From`-convertible to `CoreError`
- **Fix:** Used `.map_err(|e| CoreError::Inference(format!(...)))` pattern instead of `?`
- **Files modified:** crates/bgprunr-core/src/engine.rs

**4. [Rule 1 - Bug] Session::run() requires &mut self — Mutex<Session> needed**
- **Found during:** Task 2 design
- **Issue:** Plan signature `process_image(img, engine: &OrtEngine)` requires shared access, but `session.run()` requires `&mut Session`
- **Fix:** Wrapped `Session` in `Mutex<Session>` for interior mutability; added `with_session()` closure API
- **Files modified:** crates/bgprunr-core/src/engine.rs

**5. [Rule 1 - Bug] ndarray version mismatch — 0.16 vs ort 2.0 required 0.17**
- **Found during:** Task 2 compilation
- **Issue:** Workspace had `ndarray = "0.16"` but ort 2.0-rc.12 depends on `ndarray = "0.17"`, causing `Dimension` trait incompatibility
- **Fix:** Updated workspace Cargo.toml `ndarray` from `"0.16"` to `"0.17"`
- **Files modified:** Cargo.toml, Cargo.lock

**6. [Rule 1 - Bug] session.inputs[0].name field access — both are methods in ort 2.0**
- **Found during:** Task 2 compilation
- **Issue:** Plan showed `session.inputs[0].name` but both `inputs` and `name` are methods: `session.inputs()[0].name()`
- **Fix:** Used method call syntax
- **Files modified:** crates/bgprunr-core/src/pipeline.rs

**7. [Rule 2 - Missing critical functionality] dev-models and cuda features missing from bgprunr-core**
- **Found during:** Task 1 compilation
- **Issue:** `#[cfg(feature = "dev-models")]` in bgprunr-core referenced a feature not declared in its Cargo.toml, causing `unexpected_cfg` warnings and incorrect conditional compilation
- **Fix:** Added `dev-models = ["bgprunr-models/dev-models"]` and `cuda = []` to bgprunr-core features
- **Files modified:** crates/bgprunr-core/Cargo.toml

## Test Results

| Test Suite | Result | Notes |
|------------|--------|-------|
| types tests | 5 pass | No model required |
| preprocess tests | 4 pass | No model required |
| postprocess tests | 4 pass | No model required |
| formats tests | 6 pass | No model required |
| engine unit tests | 1 pass | Object-safety test (no model) |
| pipeline unit tests | 2 pass | Large image + bad bytes (no model) |
| engine integration tests | 2 fail (expected) | Require `cargo xtask fetch-models` |
| pipeline integration tests | 2 fail (expected) | Require `cargo xtask fetch-models` |

All 22 unit tests pass without model files. 4 integration tests are gated behind `dev-models` feature and require downloaded models.

## Self-Check: PASSED

| Check | Status |
|-------|--------|
| crates/bgprunr-core/src/engine.rs exists | FOUND |
| crates/bgprunr-core/src/pipeline.rs exists | FOUND |
| .planning/phases/02-core-inference-engine/02-03-SUMMARY.md exists | FOUND |
| Commit 166097e (Task 1) exists | FOUND |
| Commit de6834f (Task 2) exists | FOUND |
