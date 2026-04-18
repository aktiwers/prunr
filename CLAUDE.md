# BgPrunR — Claude Guidelines

## Honesty over cop-outs

When declining a task, skipping a refactor, or marking something "not worth it," **state the real reason**. Never invent risk or complexity narratives to cover "I didn't do the homework" or "I was pacing myself."

Acceptable:
- "I haven't verified X, would need to read Y first."
- "This requires plumbing through 3 call sites for a net-negative LOC change — not a win."
- "Out of scope for the current task — separate commit."

Not acceptable:
- "Not worth the risk" (generic hand-wave).
- "Medium refactor" (without naming the actual cost).
- Inventing trade-offs that don't exist to justify skipping work.

If the honest answer is "I was lazy," say that. The user would rather hear it and redirect than unpack a fake justification.

## ARCHITECTURE.md standards

ARCHITECTURE.md is a **technical reference** — someone reading it should quickly understand the codebase structure, the philosophy, and the non-obvious choices. It is **not** a dump of everything that changed in a phase.

Each section earns its place by answering one of:
- *How is this structured?* (workspace, process model, threading)
- *What's the philosophy or non-obvious trade-off?*
- *Where does data live?* (paths, caches, temp files)

Skip implementation detail (sanitization rules, serde attributes, snapshot struct layouts, comparison logic) — that belongs in code comments. Prefer one-sentence WHY over multi-paragraph HOW. Tables for scannable cross-cutting info. Prose for philosophy. When in doubt, cut.

## Panic safety

`.unwrap()` / `.expect()` in non-test code is allowed **only when the invariant is locally verifiable** — the line above proves it, or the type system enforces it. If kept, add a one-line comment naming the invariant (e.g., `// just pushed, non-empty`). If the invariant is not local, use `?` or `.ok_or_else(...)`.

`Mutex::lock()` must handle poison. Default pattern:
```rust
let guard = mtx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
```
Unless we explicitly want the panic to propagate (rare — say so in a comment).

`new_for_test` and similar test-only constructors must be gated behind `#[cfg(any(test, feature = "test-utils"))]` — not compiled into release binaries.

## Logging

Use `tracing::{info, warn, error, debug}!` with structured fields (`item_id`, `stage`, `rss_mb`, …). **Never `eprintln!` in library or GUI-internal app code.** Allowed exceptions:

- `main.rs` before `tracing_subscriber` is initialized.
- **CLI user-facing output** in `cli.rs` — messages intended for the user's terminal (`error: no input files`, per-file processing failures, skip notices). These are the CLI's UI; timestamps and log levels would be a UX regression, and shell scripts parse the format. **Diagnostic output (timing, internal state, debug dumps) still uses `tracing` even in `cli.rs`** — if it wouldn't show up on a polished release binary, it's not user UI.
- Tests and diagnostic tools where `stderr` capture matters.

When logging errors via `tracing`, include the failing identifier as a structured field, not interpolated into the message — `error!(item_id, %err, "decode failed")` beats `eprintln!("decode {item_id} failed: {err}")`.

## Test expectations

New **pure functions** (no I/O, no globals, no GUI) in `prunr-core` earn a unit test in the same file. Not negotiable for `recipe.rs`, `postprocess.rs`, `preprocess.rs`, `edge.rs`, `formats.rs`.

New subprocess IPC variants (`SubprocessCommand` / `SubprocessEvent`) earn a bincode round-trip test in `subprocess/protocol.rs`.

Changes to `resolve_tier` or `ItemSettings` fields require a test covering the new field's behaviour in the tier comparison table.

## Layers

Each layer has its own state-ownership rules. The **coordinator pattern** in the next section applies **only** to the GUI layer. Other layers have their own conventions:

| Layer                       | Location                                 | Convention                                             |
|-----------------------------|------------------------------------------|--------------------------------------------------------|
| Core inference pipeline     | `crates/prunr-core/src/`                 | Pure functions; caller-owned state. See `## prunr-core conventions`. |
| Model blobs                 | `crates/prunr-models/src/`               | Static `&[u8]` + `OnceLock<Vec<u8>>`; no mutable state |
| Subprocess child            | `crates/prunr-app/src/worker_process.rs` | One loop + IPC-event match; stateless by design        |
| Subprocess IPC (parent)     | `crates/prunr-app/src/subprocess/`       | `SubprocessManager` owns child handle + event queue; new wire variants earn bincode round-trip tests (see `## Test expectations`) |
| CLI frontend                | `crates/prunr-app/src/cli.rs`            | Thin wrapper over core + subprocess (`eprintln!` allowed for user-facing output — see `## Logging`) |
| **GUI frontend**            | `crates/prunr-app/src/gui/`              | **Coordinator pattern — see `## GUI state ownership`** |

If you're editing `prunr-core`, `prunr-models`, `worker_process.rs`, or the subprocess IPC, **do not apply the GUI coordinator decision table** — it is scoped to the GUI layer only.

## GUI state ownership (prunr-app/src/gui/)

`PrunrApp` is a **coordinator**, not an owner. It holds UI visibility flags and handles to domain coordinators — not business logic.

Business state lives in its own module with a clear owner:
- Batch state (items, selection, memory governance, BackgroundIO) → `BatchManager`
- Processing pipeline (worker channels, admission, live preview, dispatch state) → `Processor`
- Result history + preset undo → `HistoryManager` (unit struct; methods on `&mut BatchItem`)
- Drag-export lifecycle → `DragExportState`

Before adding a new `PrunrApp` field, ask: which coordinator owns this domain? Default-to-no.

### State ownership — decision table

Before adding a new field, method, or channel in the GUI layer, find the matching row:

| Need                                                | Owner                                              |
|-----------------------------------------------------|----------------------------------------------------|
| Add / remove / iterate batch items                  | `BatchManager`                                     |
| Selected index, clamp, look up by id                | `BatchManager`                                     |
| Thumbnail / decode / tex-prep / save-done channels  | `BatchManager.bg_io`                               |
| Per-status counts, progress totals                  | `BatchManager`                                     |
| Worker IPC (tx/rx), admission, job dispatch         | `Processor`                                        |
| Live-preview tick / debounce / cancellation         | `Processor`                                        |
| Result history, preset undo/redo                    | `HistoryManager` (methods on `&mut BatchItem`)     |
| Drag-out state (active, items set, pending)         | `DragExportState`                                  |
| Save-as dialog, clipboard, file picker              | `PrunrApp` (Phase 11 → `SystemBridge`)             |
| Canvas zoom/pan state                               | `PrunrApp.zoom_state`                              |
| Toasts, transient status text                       | `PrunrApp`                                         |

**Hard rule:** if a row matches, do NOT add the field/method/channel to `PrunrApp`. Default-to-the-coordinator.

### Use-before-hand-rolling helper menu

Reach for these before writing the equivalent inline:

| You need…                                      | Use                                        |
|------------------------------------------------|--------------------------------------------|
| Find a `BatchItem` by id                       | `BatchManager::find_by_id{,_mut}(id)`      |
| Selected item (or `None` when empty)           | `BatchManager::selected_item()`            |
| Selected index, clamped to batch size          | `BatchManager::selected_idx_clamped()`     |
| "Is `id` the selected one?"                    | `BatchManager::is_selected(id)`            |
| Per-status counts (done/processing/errored)    | `BatchManager::status_counts()`            |
| Clear all result-derived caches on an item     | `BatchItem::reset_result_caches()`         |
| Invalidate edge cache (tensor + mask together) | `BatchItem::invalidate_edge_cache()`       |
| Item's cache footprint (bytes)                 | `BatchItem::cache_size()`                  |
| Request a thumbnail build                      | `BatchManager::request_thumbnail(...)`     |
| Pre-decode source bytes                        | `BatchManager::request_decode_source(...)` |

If the helper you want doesn't exist, add it to the coordinator — not to `PrunrApp`.

### Anti-patterns (grep-rejectable)

Patterns that look fine but have a better home. Refuse PRs from your own past self:

- `self.batch.items.iter().find(|b| b.id == …)` → `self.batch.find_by_id(…)`
- `self.batch.items.iter_mut().find(|b| b.id == …)` → `self.batch.find_by_id_mut(…)`
- Computing `pct = done / total` on `PrunrApp` → compute on `BatchManager` and expose a helper
- New channel sender/receiver on `PrunrApp` → belongs in `BatchManager.bg_io` (I/O) or `Processor` (work)
- New `fn on_*_event(&mut self, …)` on `PrunrApp` that iterates `self.batch.items` — the inner logic belongs on `BatchManager`; `PrunrApp` just routes the event
- New `PrunrApp` field named after a domain noun (`save_state`, `filter_opts`, `export_prefs`, …) — that's a coordinator in disguise; either it fits an existing coordinator or it *is* a new one

**Before `Edit`/`Write` of `crates/prunr-app/src/gui/app.rs`:** check the decision table first. If the change touches batch items, it probably belongs in `batch_manager.rs`.

## prunr-core conventions

`prunr-core` is the **pure inference pipeline** — no GUI, no subprocess, no coordinators. State flows through function arguments; the only long-lived state is `OrtEngine` (owned by the caller) and the per-model `OnceLock<Vec<u8>>` decompressed-bytes caches in `prunr-models`.

**Rules:**
- Pure functions by default. The single allowed mutable global pattern is `OnceLock<T>` for caches. No `static mut`, no `lazy_static!`-ish stateful singletons. `Mutex<T>` is allowed only when it wraps something ort-owned that externally mandates `&mut` (`OrtEngine::session`).
- `tracing::{info,warn,error,debug}!` for diagnostics. No `eprintln!` (same rule as the rest of the workspace).
- New pure functions (no I/O, no globals) in `recipe.rs` / `postprocess.rs` / `preprocess.rs` / `edge.rs` / `formats.rs` earn a unit test in the same file. Already enforced by `## Test expectations`; restated here because it's easy to miss when adding a helper.

### Hot-path helper menu

Reach for these before reinventing:

| You need…                                               | Use                                                       |
|---------------------------------------------------------|-----------------------------------------------------------|
| Full pipeline (decode + infer + postprocess)            | `process_image_*` / `process_image_from_decoded`          |
| Tier 1-only (infer without postprocess)                 | `infer_only`                                              |
| Tier 2 re-postprocess from cached tensor                | `postprocess_from_flat`                                   |
| Tensor → mask only (without RGBA composite)             | `postprocess::tensor_to_mask`                             |
| Mask → RGBA composite on an existing image              | `postprocess::apply_mask`                                 |
| SIMD gray / RGB Lanczos3 resize                         | `formats::resize_gray_lanczos3` / `resize_rgb_lanczos3`   |
| Oversize guard before tensor allocation                 | `formats::check_large_image(img) -> Option<CoreError>`    |
| Downscale an oversized image to a max dimension         | `formats::downscale_image(img, max_dim)`                  |
| Composite bg color into pixels (export path)            | `formats::apply_background_color`                         |
| Encode PNG bytes with fast compression                  | `formats::encode_rgba_png`                                |
| Tier decision from (old, new) recipe                    | `recipe::resolve_tier(old, new) -> RequiredTier`          |
| Edge-refining alpha filter                              | `guided_filter::guided_filter_alpha(rgba, mask, r, eps)`  |

### Anti-patterns

- **Double RGBA conversion in one path:** `.to_rgba8()` followed by another `.to_rgba8()` on the same `DynamicImage` allocates twice. Share one buffer (see the comment on `postprocess::postprocess` — single RGBA allocation shared with guided filter and mask application).
- **Parallelising a previously-serial loop without measurement:** `apply_background_color` stays sequential on purpose — nested rayon inside the subprocess worker path caused deadlock/starvation (commit `b2306bb`). `apply_mask_inplace` was re-parallelised in 10-07 only after benchmarks confirmed row-parallel wins without regressing the subprocess path. If you touch either, read the inline comment + the 10-07 ARCHITECTURE benchmark row before flipping the switch.
- **Adding a new `RequiredTier` variant without extending `resolve_tier`'s table test.** The test is the contract; a variant without a test row is a silent fall-through bug waiting to ship. Same for any new `ItemSettings` field — add a tier comparison test case.
- **`Mutex<T>` for read-mostly caches.** Use `OnceLock<T>` — no contention, no poison handling, no runtime lock cost after the first init.
