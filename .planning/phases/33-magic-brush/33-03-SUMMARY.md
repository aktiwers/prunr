---
phase: 33-magic-brush
plan: 03
subsystem: gui
tags: [batch-item, selection, mask, lifecycle, arc, egui]

# Dependency graph
requires:
  - phase: 33-magic-brush/33-01
    provides: MaskArtifact type (Arc<Vec<f32>> at source resolution) + content_hash
provides:
  - BatchItem.selection_mask / selection_hash / selection_outline / selection_texture fields
  - BatchItem.magic_brush_embedding placeholder field (Arc<()>, Plan 07 retypes)
  - BatchItem::invalidate_selection() + invalidate_magic_brush_embedding() methods
  - BatchManager::commit_selection / clear_selection / invalidate_selection_on_source_change helpers
  - Criterion 8 lifecycle contract: selection survives Process clicks
  - 8 boundary tests pinning all selection lifecycle contracts
affects:
  - 33-magic-brush/33-04 (Paint Brush migration reads selection_mask)
  - 33-magic-brush/33-05 (visualization reads selection_outline + selection_texture)
  - 33-magic-brush/33-07 (Magic Brush wires magic_brush_embedding; retypes Arc<()> to SamEmbedding)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Arc<MaskArtifact> for O(1) undo snapshot clones — same pattern as result_rgba"
    - "commit_selection lockstep: mask + hash set together, outline + texture cleared for off-thread rebuild"
    - "Single source of truth: BatchItem::invalidate_selection() is the only clear site; BatchManager delegates"
    - "test-utils feature in prunr-app Cargo.toml enables #[cfg(any(test, feature = 'test-utils'))] gating"

key-files:
  modified:
    - crates/prunr-app/src/gui/item.rs
    - crates/prunr-app/src/gui/batch_manager.rs
    - crates/prunr-app/Cargo.toml

key-decisions:
  - "magic_brush_embedding typed as Arc<()> placeholder — avoids prunr-core::sam dep before Plan 06 lands; Plan 07 retypes in one line"
  - "selection_outline typed as Arc<Vec<(u32, u32)>> matching outline_polyline return type from refine.rs"
  - "test-utils feature added to Cargo.toml so #[cfg(any(test, feature = 'test-utils'))] compiles warning-free per CLAUDE.md Panic safety"

requirements-completed:
  - "Criterion 1"
  - "Criterion 8"

# Metrics
duration: 6min
completed: 2026-05-16
---

# Phase 33 Plan 03: BatchItem Selection Fields + Lifecycle Summary

**BatchItem extended with 5 selection fields; BatchManager gains 3 lifecycle helpers; 8 boundary tests pin the Criterion 8 contract that selection survives Process clicks**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-05-16T10:43:40Z
- **Completed:** 2026-05-16T10:49:52Z
- **Tasks:** 2
- **Files modified:** 3 (2 modified with code, 1 Cargo.toml)

## Accomplishments

### Task 1 — BatchItem fields + cache_size accounting

Five new fields added to `BatchItem`:

| Field | Type | Lifecycle |
|---|---|---|
| `selection_mask` | `Option<Arc<prunr_core::selection::MaskArtifact>>` | Cleared by `clear_selection` / source change; survives Process clicks |
| `selection_hash` | `Option<u64>` | Set in lockstep with mask via `commit_selection` |
| `selection_outline` | `Option<Arc<Vec<(u32, u32)>>>` | Cleared on commit; Plan 05 rebuilds off-thread |
| `selection_texture` | `Option<egui::TextureHandle>` | Cleared on commit; Plan 05 rebuilds off-thread |
| `magic_brush_embedding` | `Option<Arc<()>>` | Placeholder; Plan 07 retypes to `Arc<prunr_core::sam::SamEmbedding>` |

- `cache_size()` includes selection bytes (mask data len × 4) + 16 MB constant for embedding when present
- `reset_result_caches()` does NOT touch any selection field (verified by grep + Criterion 8 test)
- `invalidate_selection()` clears all 4 selection fields atomically (single source of truth)
- `invalidate_magic_brush_embedding()` clears the SAM embedding cache
- `new_for_test()` added, gated `#[cfg(any(test, feature = "test-utils"))]`

### Task 2 — BatchManager lifecycle helpers + boundary tests

Three helpers added to `impl BatchManager`:

- **`commit_selection(item_id, MaskArtifact) -> bool`**: writes mask Arc, sets `selection_hash = mask.content_hash()` in lockstep, drops `selection_outline` + `selection_texture` (Plan 05 rebuilds off-thread). Returns `false` silently when item not found.
- **`clear_selection(item_id) -> bool`**: delegates to `item.invalidate_selection()`; returns `true` when anything was present (idempotent).
- **`invalidate_selection_on_source_change(item_id) -> bool`**: drops selection AND `magic_brush_embedding` — both are bound to the source bytes they were derived from.

## Task Commits

1. **Task 1: BatchItem fields + cache_size + lifecycle methods** - `875fd81` (feat)
2. **Task 2: BatchManager helpers + 4 boundary tests** - `2337be2` (feat)

**Plan metadata:** (docs commit follows)

## Files Modified

- `crates/prunr-app/src/gui/item.rs` — 5 new fields, `new_for_test()`, `invalidate_selection()`, `invalidate_magic_brush_embedding()`, cache_size update, 4 unit tests
- `crates/prunr-app/src/gui/batch_manager.rs` — 3 lifecycle helpers, 4 boundary tests
- `crates/prunr-app/Cargo.toml` — `test-utils = []` feature added

## Boundary Tests Shipped (8 total)

| Test | Location | Contract pinned |
|---|---|---|
| `default_batch_item_has_no_selection` | item.rs | All 5 fields are None at construction |
| `invalidate_selection_clears_all_four_fields` | item.rs | All selection fields cleared atomically |
| `cache_size_includes_selection_bytes` | item.rs | Mask bytes (len × 4) counted correctly |
| `cache_size_includes_embedding_when_present` | item.rs | 16 MB counted for embedding |
| `commit_selection_sets_hash_in_lockstep` | batch_manager.rs | selection_hash == mask.content_hash() |
| `clear_selection_is_idempotent` | batch_manager.rs | First=true, second=false |
| **`reset_result_caches_does_not_clear_selection`** | batch_manager.rs | **Criterion 8: selection survives Process clicks** |
| `invalidate_selection_on_source_change_drops_embedding` | batch_manager.rs | Both selection + embedding cleared on source change |

## Decisions Made

- `magic_brush_embedding` typed as `Arc<()>` placeholder (not `Arc<prunr_core::sam::SamEmbedding>`) because Plan 06 (SAM core) is a Wave 2 independent plan — Plan 03 must not block on Plan 06's landing. Plan 07 retypes in a one-line change with no semantic impact.
- `selection_outline` type is `Arc<Vec<(u32, u32)>>` matching `outline_polyline`'s actual return type from `refine.rs` (as shipped in Plan 01).
- `test-utils` feature added to `Cargo.toml` to suppress the `unexpected_cfg` warning that would otherwise appear in clean builds under strict lint settings.
- `BatchItem::invalidate_selection()` is the single clear site — `BatchManager::clear_selection` and `invalidate_selection_on_source_change` both delegate to it (CLAUDE.md ## One source of truth).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added test-utils feature to Cargo.toml**
- **Found during:** Task 2 test compilation
- **Issue:** `#[cfg(any(test, feature = "test-utils"))]` on `new_for_test()` generated an `unexpected_cfg` warning because `test-utils` was not declared as a feature in `Cargo.toml`. CLAUDE.md Panic safety mandates this gate.
- **Fix:** Added `test-utils = []` feature declaration to `crates/prunr-app/Cargo.toml`.
- **Files modified:** `crates/prunr-app/Cargo.toml`
- **Committed in:** `2337be2` (Task 2 commit)

None other — plan executed as written.

## Workspace Test Results

- `cargo test --workspace --lib`: 553 prunr-app + 361 prunr-core + 48 prunr-models + 20 prunr-runtime-install tests green
- `cargo build --workspace`: exits 0, no errors

## Self-Check: PASSED
