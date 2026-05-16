---
phase: 33-magic-brush
plan: "02"
subsystem: prunr-models
tags: [model-registry, sam2, selection, multi-part-ondemand]
dependency_graph:
  requires: []
  provides: [ModelCategory::Selection, ModelId::Sam2HieraSmall, resolve_part_bytes]
  affects: [prunr-models, prunr-app model-store]
tech_stack:
  added: [ResolveError enum, resolve_part_bytes fn]
  patterns: [MultiPartOnDemand with placeholder SHA256, 64-char hex placeholders]
key_files:
  modified:
    - crates/prunr-models/src/lib.rs
decisions:
  - "SHA256 placeholders are 64-char hex zeros (not TODO strings) so the existing multi_part_descriptors_have_at_least_one_part sha256.len()==64 invariant holds"
  - "ResolveError::UnknownPart checked before disk I/O so error-path tests need no on-disk model files"
  - "resolve_part_bytes returns Cow::Owned (not cached) — SAM parts are 200 MB; process-lifetime cache for a 200 MB file would hold RAM permanently; Plan 07 decides caching strategy"
  - "Added second non-multipart test (U2net OnDemand) alongside the Silueta Bundled test to cover both non-multipart source kinds"
metrics:
  duration: "~12 minutes"
  completed: "2026-05-16"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 1
  tests_added: 10
---

# Phase 33 Plan 02: SAM 2 Hiera Small Model Registry Summary

ModelCategory::Selection + ModelId::Sam2HieraSmall registered as a two-part MultiPartOnDemand bundle (Apache 2.0, Meta); resolve_part_bytes added for Plan 07 SAM dispatch.

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Add ModelCategory::Selection + Sam2HieraSmall variants | fce2c7c | Enum variant, ALL, stable_name, bundled_bytes, load_variant, exhaustiveness fence |
| 2 | REGISTRY entry + 4 boundary tests | 4ce9300 | MultiPartOnDemand descriptor, encoder+decoder parts, 4 sam2_* tests |
| 3 | resolve_part_bytes + ResolveError | 9fbf8a7 | ResolveError enum, resolve_part_bytes fn, 3 error-path tests |

## Variants Added

- `ModelCategory::Selection` — Magic Brush SAM dispatch; not routed through BG-removal pipeline
- `ModelId::Sam2HieraSmall` — stable_name `"sam2_hiera_small"`; is_sd_family() = false

## REGISTRY Entry

| Field | Value |
|-------|-------|
| category | ModelCategory::Selection |
| source | MultiPartOnDemand |
| subdir | "sam2-hiera-small-v1.0.0" |
| parts | encoder (~200 MB) + decoder (~20 MB) |
| license | Apache 2.0 / Meta / facebookresearch |
| license_acceptance_required | false |
| working_set_mb | 800 (conservative; empirical validation pending) |
| gpu | GpuRequirement::None |
| attribution_required | false |
| upscale | None |

## SHA256 Status

Both SHA256 fields are 64-char hex zeros (placeholder). The 64-char length satisfies the existing `multi_part_descriptors_have_at_least_one_part` invariant. Plan 07 Task 2 replaces them with real values from vietanhdev/segment-anything-2-onnx-models before the Model Store download gate ships.

## Descriptor Boundary Tests Added (Task 2)

1. `sam2_registry_entry` — category=Selection, source=MultiPartOnDemand, working_set_mb=800, attribution_required=false, upscale=None
2. `sam2_has_encoder_and_decoder_parts` — parts.len()==2, keys contain "encoder" and "decoder"
3. `sam2_license_is_apache_2_no_acceptance_required` — license=="Apache 2.0", !license_acceptance_required
4. `sam2_encoder_decoder_separate_engines` — distinct filenames per part (pins Criterion 12: separate OrtEngine sessions)

## resolve_part_bytes Signature

```rust
pub fn resolve_part_bytes(
    id: ModelId,
    part_key: &str,
) -> Result<Cow<'static, [u8]>, ResolveError>
```

### ResolveError Variants

| Variant | Trigger |
|---------|---------|
| `NotInRegistry` | ModelId not found in REGISTRY |
| `NotMultiPart` | Model is Bundled or single-file OnDemand |
| `UnknownPart(String)` | part_key not in bundle's parts list |
| `Io(String)` | on_demand_dir unresolvable or file read failure |

UnknownPart is checked before any disk I/O.

## Resolver Error-Path Tests Added (Task 3)

1. `resolve_part_bytes_non_multipart_returns_not_multipart` — Silueta (Bundled) → Err(NotMultiPart)
2. `resolve_part_bytes_non_multipart_ondemand_returns_not_multipart` — U2net (single-file OnDemand) → Err(NotMultiPart)
3. `resolve_part_bytes_unknown_key_returns_unknown_part` — Sam2HieraSmall + "nonexistent" → Err(UnknownPart(_))

Happy-path test not included — requires either the 200 MB encoder on disk or a fabricated temp dir. Plan 07 manual smoke covers the happy path end-to-end.

## Deviations from Plan

None — plan executed exactly as written.

The plan specified one non-multipart test (`resolve_part_bytes_non_multipart_returns_not_multipart`). A second was added (`resolve_part_bytes_non_multipart_ondemand_returns_not_multipart` with U2net/OnDemand) to cover both non-MultiPart source kinds (Bundled + OnDemand). This is Rule 2 (add missing critical functionality — testing both branches of the `_ => Err(NotMultiPart)` arm).

## Self-Check: PASSED

- crates/prunr-models/src/lib.rs: FOUND
- commit fce2c7c (Task 1): FOUND
- commit 4ce9300 (Task 2): FOUND
- commit 9fbf8a7 (Task 3): FOUND
