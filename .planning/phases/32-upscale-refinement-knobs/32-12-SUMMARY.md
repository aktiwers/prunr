---
phase: 32-upscale-refinement-knobs
plan: "12"
subsystem: upscale
tags: [ort, engine-cache, gap-closure, perf]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: upscale_rgba_with_engine / upscale_two_pass_with_engine entry points (Plan 32-12 Task 1, commit ac44ffe)
  - phase: 32-upscale-refinement-knobs
    provides: EP ladder constructor in upscale dispatch (Plan 32-10)
provides:
  - Processor.warm_upscale_engine — Option<(ModelKind, Arc<OrtEngine>)> slot
  - try_cached_upscale_engine / ensure_upscale_engine / release_upscale_engine on Processor
  - Cache-evict-on-model-swap hook in apply_toolbar_change
  - 4 boundary tests pinning the cache-hit / evict / release contract
affects: [32-upscale-refinement-knobs, gui processor, gui app toolbar-change]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-slot warm-engine cache: Option<(Key, Arc<Engine>)> on owning coordinator, ensure_*/release_* lifecycle"
    - "Mismatch-evict on lookup: try_cached_* drops the slot when the key doesn't match, so ensure_* doesn't need a separate eviction branch"
    - "Worker thread receives Arc::clone(&engine): in-flight dispatch survives cache eviction (e.g. mid-dispatch model swap)"

key-files:
  created: []
  modified:
    - crates/prunr-app/src/gui/processor.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-core/src/upscale/mod.rs (Task 1, committed in ac44ffe)

key-decisions:
  - "ModelKind is the cache key — not (ModelKind, intra_threads, level) — because intra_threads is process-stable in current Settings UI and level is deterministic from ModelKind via pick_optimization_level. If a future setting flips either at runtime, route the change through release_upscale_engine."
  - "Single-slot, not LRU map: caching multiple upscale engines simultaneously is YAGNI. Users swap models one at a time; the eviction-on-swap behaviour is what the user expects."
  - "pick_optimization_level promoted to `pub` (in prunr-core/src/upscale/mod.rs, committed in ac44ffe) so Processor derives the same level the dispatcher would — single source of truth at the descriptor level."
  - "Explicit release_upscale_engine call in apply_toolbar_change's model-swap branch, on top of the mismatch-evict in try_cached_*. Covers switch-away-from-upscale where ensure_* is never called again."
  - "Test sentinel uses Silueta (bundled, ~4 MB, no download). Tests gracefully skip if ort_runtime::init() fails — no model bundle required."

patterns-established:
  - "Coordinator-owned single-slot engine cache for GPU-state-heavy workloads where one-at-a-time is the user model"

# Verification
self-check:
  - cargo test --workspace --lib — 933 tests pass (545 + 334 + 34 + 20), including 4 new warm_cache_tests
  - grep -n 'warm_upscale_engine' crates/prunr-app/src/gui/processor.rs — 6+ matches (field + 3 fn impls + 1 test-only accessor + 4 test refs)
  - grep -n 'release_upscale_engine' crates/prunr-app/src/gui/app.rs — 1 match (model-swap branch in apply_toolbar_change)
  - All Arc::clone() in the dispatch path leave a strong_count >= 2 only for the brief window between cache lookup and thread::spawn — back to 1 after the worker completes

manual-verify-pending:
  - "Cold vs warm dispatch time on user's machine (RTX-class GPU with EP ladder active): cold click should be 1-3s + inference; warm click should be only inference. Quantify after user runs a 4K image through Real-ESRGAN x4plus twice."
  - "Memory: switch upscale → seg model and confirm RSS drops by ~64 MB (the cached ORT session is reclaimed)."

# What this enables
unlocks:
  - "User-perceptible: 'why is the second click slow too?' becomes 'second click is instant', closing Gap-2."
  - "Plan 32-14's fp16 variant load gets the same warm-cache benefit for free — the fp16-variant byte stream replaces the fp32 one inside OrtEngine construction, but the cache key (ModelKind) is the same."

# Deviations from plan
- "Plan called for `release_upscale_engine` on the model-swap branch only; this lands the call on every model-change branch (upscale→upscale, upscale→seg, upscale→inpaint, seg→upscale, etc.). The mismatch-evict in try_cached_* would handle upscale→upscale, but the unconditional drop is cheaper than the conditional plumbing and prevents the cache from surviving a sequence like upscale-x4plus → seg → upscale-x4plus where the second upscale would otherwise reuse a possibly-stale session."
- "Added a test-only accessor (`warm_upscale_engine_for_test`) gated by `#[cfg(test)]` so the 4 boundary tests can inject a sentinel Arc and assert pointer identity without exercising real upscale inference. Pattern matches the rest of the test suite."

# Plan 32-12 closeout
gap_closed: Gap-2 (warm-engine cache for upscale)
deferred: None from this plan.
