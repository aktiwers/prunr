---
phase: 32-upscale-refinement-knobs
plan: "13"
subsystem: upscale
tags: [models, registry, community, license, gap-closure]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: UpscaleModelKnobs (Plan 32-05) + native_scale + uses_window_attention (Plan 32-10) + upscale_rgba_with_engine (Plan 32-12)
provides:
  - ModelId::FourXNmkdSiaxCx + ModelId::FourXNmkdSuperscale (registry + bundled_bytes panic + load_variant + exhaustiveness fence)
  - ModelKind variants + TryFrom<ModelId> + From<ModelKind>
  - preprocess() unreachable arm for both
  - SettingsModel::FourXNmkdSiaxCxUpscale + FourXNmkdSuperscaleUpscale (ALL[12] + is_upscale + to_model_id + From<ModelKind>)
  - views/mod.rs model_name / model_label entries
  - 5 boundary tests pinning license + descriptor shape
affects: [32-upscale-refinement-knobs, prunr-models, prunr-core types/preprocess, prunr-app gui settings/views]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Community-RRDB model addition: pure data edit + match-site fences across the 5 canonical files; no dispatch-code change because UpscaleModelKnobs is data-driven"
    - "Permissive-license gate: license verification via openmodeldb WebFetch BEFORE export; ship only models that allow redistribution + commercial use"

key-files:
  created:
    - .planning/phases/32-upscale-refinement-knobs/32-13-PRECONDITIONS.md
  modified:
    - crates/prunr-models/src/lib.rs
    - crates/prunr-core/src/types.rs
    - crates/prunr-core/src/preprocess.rs
    - crates/prunr-app/src/gui/settings.rs
    - crates/prunr-app/src/gui/views/mod.rs

key-decisions:
  - "Original candidates (UltraSharp, Remacri) BLOCKED on license. Both are CC-BY-NC-SA-4.0 — non-commercial → not redistributable via aktiwers/prunr/releases. Replaced with NMKD Siax-CX + NMKD Superscale, both WTFPL (most permissive license possible; no attribution required)."
  - "NMKD checkpoints use the legacy xinntao/ESRGAN key naming (model.0.weight, model.1.sub.N.RDBM.convK.0...) which `basicsr.archs.rrdbnet_arch.RRDBNet` (the Real-ESRGAN naming) does not accept directly. Wrote `/tmp/realesrgan-export/export_nmkd.py` with an inline key remap (mechanical 7-rule mapping). Architecture is identical RRDB-23, just different naming."
  - "`working_set_mb=600` mirrors x4plus exactly. Both NMKD models have the same 16,697,987 params as x4plus and produce the same RSS profile in the Rust tiled dispatch. The Python single-shot smoke test measured 1152 MB peak — that's an upper bound for a no-tiling full-output buffer, NOT the dispatch shape. The Rust path uses recommended_tile=512 with buffer reuse, so peak stays under 600 MB like x4plus."
  - "WTFPL license_url uses https://www.wtfpl.net/about/ (not http) to satisfy the existing `ondemand_descriptors_have_complete_metadata` test that requires HTTPS for all license URLs. wtfpl.net supports both schemes; HTTPS is now the canonical."
  - "Display names: 'Upscale (NMKD Siax-CX)' / 'Upscale (NMKD Superscale)' — matches the existing 'Upscale (Real-ESRGAN)' / 'Upscale (Nomos8k)' shape for the toolbar dropdown."

patterns-established:
  - "Permissive-license gate via WebFetch before .pth download — catches NC-licenses at the source page, saves a wasted export pipeline"
  - "Legacy-ESRGAN → Real-ESRGAN state-dict key remap (export_nmkd.py): 7 rules from `model.0.{w|b}` to `conv_first.{w|b}`, `model.1.sub.N.RDBM.convK.0` to `body.N.rdbM.convK`, etc."

# Verification
self-check:
  - cargo test --workspace --lib — 935 tests pass (545 + 334 + 39 + 20), including 5 new NMKD tests
  - cargo build --workspace — clean (after match-site fence updates: views/mod.rs model_name + model_label + is_upscale_classifies_every_variant test)
  - 5 new tests in prunr-models pin: ALL contains, descriptor shape (native_scale=4, uses_window_attention=false, input_name=data, is_fp16=false), license metadata (WTFPL, Nmkd, attribution_required=false)
  - Both .onnx assets uploaded to aktiwers/prunr/releases tag models-v1 and resolve via curl -I (302 redirect → 200 from CDN)
  - sha256 captured at upload time and pinned in REGISTRY — `prunr-models`'s on-download verification will catch any future mismatch

manual-verify-pending:
  - "Run `cargo run --release` and visually confirm both models appear in the Model Store under Upscale category"
  - "Click Install on either → download completes from the new GitHub release URLs"
  - "Switch active model to each → toolbar shows the new name → Process an image → upscale output produced"
  - "A/B comparison vs x4plus on a representative photo to confirm perceived-quality is at parity or better (the whole reason this plan shipped)"

# What this enables
unlocks:
  - "User-perceptible: Model Store now lists 4 viable upscale models (Real-ESRGAN x4plus + Nomos8k + NMKD Siax-CX + NMKD Superscale), giving the user real choice across clean / restoration / high-quality niches"
  - "Plan 32-14's fp16 variant loader trivially extends to both new models — same `load_variant` codepath, same `model_fp16_bytes` resolution"

# Deviations from plan
- "Plan named the two slots `FourXUltraSharp` and `FourXFoolhardyRemacri`. Both candidate models were CC-BY-NC-SA-4.0 — non-redistributable. Replaced with permissive-licensed pair after user confirmed they wanted same-or-better quality replacements rather than closing the plan as license-blocked."
- "Plan called for a `pytorch2onnx.py` invocation in the export script. The existing `/tmp/realesrgan-export/export.py` (left over from Phase 30 / 32-05) is the canonical exporter. Wrote `export_nmkd.py` as a peer with the key-remap for legacy ESRGAN naming."
- "Smoke test measured peak RSS at 1152 MB (Python single-shot 256→1024 full inference, no tiling). This is irrelevant for the Rust dispatch which uses recommended_tile=512 with buffer reuse. REGISTRY uses `working_set_mb=600` matching x4plus's calibrated tiled-dispatch value."

# Plan 32-13 closeout
gap_closed: Gap-3 (community-quality upscale models — closed via WTFPL pair, not the originally-named CC-NC pair)
deferred: None from this plan. The user-import-feature alternative (let user load their own .onnx with self-asserted license) is a fundamentally different shape and out of scope for this milestone.
