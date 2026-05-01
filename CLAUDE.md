# BgPrunR — Claude Guidelines

## Committing — durable authorization

Commit autonomously when work reaches a logical stopping point. Don't
ask for a one-off green light each time. Defaults that make this
safe:

- **Tests must be green.** Run `cargo test --workspace --lib` (and
  any test that's a real correctness contract for the change) before
  committing. Never commit through a failing run.
- **One concept per commit.** A bug fix is one commit; a refactor is
  one commit; a feature is one commit. Don't bundle unrelated
  changes — that's bisect-hostile.
- **New commits only.** Never `--amend` published work. If a hook
  fails, fix and create a NEW commit.
- **Never skip hooks.** No `--no-verify`, no `--no-gpg-sign`. If a
  hook breaks, debug the underlying issue.
- **Stage by file name.** Avoid `git add -A` / `git add .` — they
  catch sensitive files (`.env`, credentials) or large blobs by
  accident.
- **Refuse to commit secrets / large binaries.** Anything that
  smells like a credential, a `.env`, a `target/` build artifact, or
  a multi-MB asset is a stop-and-ask.

Hard line — these still require an explicit ask each time:
- `git push` (visible to others, harder to reverse).
- Force-push, `git reset --hard`, branch deletion, `git checkout --`,
  any history-rewriting operation.
- Creating, closing, or commenting on PRs / issues.
- Anything that touches a remote, sends a notification, or is
  user-visible beyond the local working tree.

Commit message shape: short imperative title (≤ 70 chars) summarising
the *why*, body bullets for non-obvious detail when the diff alone
doesn't carry the reasoning, plus the standard
`Co-Authored-By: Claude Opus 4.7 (1M context)` trailer. If in doubt,
mirror the project's recent commit voice.

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

## Comments — what earns its place

Default to **no comments**. A comment earns its place only when the WHY is non-obvious — a hidden constraint, a subtle invariant, a workaround, behaviour that would surprise a reader. Good names carry intent; don't restate them.

Delete on sight:
- Narration of WHAT the code does (`"Iterate over items and find the matching id"` — the code already says that).
- **Restating the immediate neighbour.** A comment that paraphrases the case label, function name, attribute, or struct field directly below it. If the next line is `("input_transform+edges → add_edge", …)`, don't write `// input_transform with edges on invalidates edge tensor only.` above it. If the function is `_force_developer_to_update_ALL_too`, don't add `// SCREAMING_ALL is the signal — the name is the message` — the name *is* the message.
- **References to future work — including task-tracker IDs.** `"step 5 will wire this up"`, `"Phase 11 will move this"`, `"// REVIEW-FINDINGS H6: queued ChipMeta extraction"`, `"// see H6"`, `"M11 will refactor"`, `"fix lands once #123 merges"`. These rot when the queued work shifts, ships, or never happens. Planning IDs (REVIEW-FINDINGS, DEFERRED, phase numbers, issue/PR numbers) belong in the task list and the planning docs, not in source.
- **Cross-file "see also" / "mirror of X" pointers.** `"(mirror of inpaint_blend::seam_guided_blend's guard)"`, `"see also format_byte_size in views/mod.rs"`, `"as in fn foo above"`. File paths and symbol names rot when files move or symbols rename. The WHY should stand on its own; if the same logic lives in two places and that *is* the non-obvious bit, name the invariant ("flat-region variance can go sub-zero by f32 rounding"), not the sibling.
- Paragraphs about feature design rationale — goes in the commit message, not inline.
- Docstrings that echo the type signature (`"Returns Option<T>"`, `"Takes a &mut BatchItem"`).
- "Fixed in commit abcd" / "Added for issue #123" — lives in git blame, not in code.

Keep:
- One-line invariant notes above `.unwrap()` (`// just pushed, non-empty`).
- WHY for non-obvious policy (`// Feather runs AFTER refine — sharpen-then-soften`).
- Cautions against regressions (`// Do NOT parallelise: nested rayon deadlocks the subprocess path`).

Rule of thumb: if removing the comment wouldn't confuse a future reader who understands the surrounding code, delete it.

### `#[allow(...)]` rationale comments

When you `#[allow]` a lint, the rationale must name a *local, durable* reason — a constraint that's true at this site today and will still be true if the queued cleanup never lands. Generic "fits any site" excuses don't count.

Bad (rot-prone or boilerplate):
- `// see H6` / `// REVIEW-FINDINGS M11 will refactor` — points at unshipped work in a planning doc.
- `// args are the per-frame state surface; struct adds indirection` — fits any `too_many_arguments` site, not a real local rationale.
- `// will fix later` — every allow can claim that.

Good (local + stable):
- `// args mirror the IPC variant fields one-for-one — packing them adds indirection without consolidating call sites.`
- `// `Process*` prefix matches the user-visible button labels and the dispatcher arms — renaming would split that pairing.`
- `// SCREAMING_ALL deliberately matches `ModelId::ALL`; renaming erases that link.`

If the only honest rationale is "queued for refactor in task X," **don't add the allow** — fix the lint inline (split the function, rename, etc.) or accept the warning until X lands. An allow with a forward-reference rationale is worse than a TODO because it has no removal trigger.

## Hot paths

A **hot path** is any code inside:
- A per-frame `render` closure (egui runs at 60 Hz during interaction).
- A live-preview dispatch (~10 Hz during a slider drag).
- The subprocess worker's per-image loop.

Before you write a `.clone()` on anything larger than ~1 KB in a hot path, ask: *is this cloning the shell (Arc) or the payload?* If payload, store the value as `Arc<T>` upstream and `Arc::clone` through the hot path. Tooltip / label strings: prefer `&'static str` when the content doesn't vary; reach for `format!` only when a value interpolates in.

Before allocating (Vec, String, Box) inside a hot closure: can the allocation hoist out once per frame or once per session? Drag handlers, tooltip callbacks, and per-item render loops are the usual offenders.

### The render closure is sacred (no I/O, no decode, no GPU upload)

The egui render closure runs at 60 Hz during interaction. It must do **rendering only** — no synchronous file I/O, no image decode, no GPU upload, no heavy allocation. Data that isn't already on the item flows through `BatchManager.bg_io` / `Processor` channels and is consumed via `.try_recv()` in `logic()` (or a dedicated `drain_*` helper called from there). A single 30-ms decode stalls the frame and the user feels it.

Grep-rejectable patterns inside any `fn render`, closure passed to `Window::show` / `ui.allocate_*` / `egui::CentralPanel::default().show`, or any `pump_*` called from `ui()`:

- `fs::File::open` / `fs::read` / `image::ImageReader::*` / `to_rgba8()` — synchronous I/O or decode. Route through `BatchManager.request_decode_source(...)` or a fresh bg_io channel.
- `ctx.load_texture` / `TextureHandle::set` — GPU upload. Belongs in `drain_background_channels`, not row-render or popover-render. (This is what `pump_thumbnail_results` got wrong.)
- `.recv()` (blocking) on any `Receiver` — must be `.try_recv()`.
- `.to_string()` on a `&'static str` (icon codepoints, label literals) — pass the `&'static str` through directly.
- `format!("...{m}...")` where `m = cfg!(target_os = "macos")` resolves at compile time — `cfg!`-gated `&'static str` constants instead.
- Recipe / hash construction every frame for drift detection — cache the last-checked tuple on the coordinator and only recompute when the source changed.

## RAM discipline (without sacrificing quality)

Prunr already runs SD models hitting 2 GB+ resident; layering naïve full-image f32 buffers on top has caused near-OOMs. Bake RAM into the design **before** writing code — but **quality and user experience always win the tie-break**. The rule isn't "silently refuse trades", it's "surface them and let the user decide."

For any new function that operates on full-resolution `RgbaImage` (or comparable) buffers:

1. **Estimate peak working set in the doc-comment** — a small table (bbox vs RAM in MB). Be honest. If peak is 9×n, write 9×n; don't undercount to look better.
2. **Bbox-crop the work region** to the mask / ROI. Pixels outside are bit-identical, so the user can't tell the bbox version from a full-image one.
3. **Sequential per-channel** for multi-channel work, not parallel-channel. R/G/B in parallel triples peak RAM for invisible wall-clock gain when each channel needs scratch.
4. **Buffer reuse.** The `guided_filter_alpha` pattern is canonical: `gi`/`gg` are repurposed in-place as `a`/`b`. Annotate reuse: `// reused as ...`. Pre-allocated `_into` variants (see `box_filter_into`) drop alloc churn across loops.
5. **Explicit `drop()` of intermediates** before the next allocation. Rust NLL doesn't always free at last-use when control flow is complex.

### The two classes of optimization

**Always allowed — apply silently** (same math, smaller footprint, invisible to the user looking at the result):
- Bbox crop, sequential channels, buffer reuse, pre-allocated `_into` outputs.
- Numerical guards (e.g., `var.max(0.0)` before division) — these protect quality for free.
- Lifting allocations out of hot loops, parallel reduction of sequential scans, RAII tightening.

**Surface as a trade — present, don't silently apply** (anything with a non-zero cost on the user side):
- Lower-precision working buffers (f32 → f16/int8). Quality cost.
- Reducing kernel/filter radius below the algorithmic minimum. Quality cost.
- Skipping refinement passes when bbox is large. Quality cost.
- Output downscaling. Quality cost.
- Lossy approximations of the same operation. Quality cost.
- More-RAM-for-more-speed (parallel-channel, batched loads that hold all channels alive). Speed-vs-RAM trade.
- Less-RAM-for-less-speed (serializing what was parallel). Speed-vs-RAM trade in the other direction.

When `/simplify` or research turns up a `[TRADE]` finding, list it with location + gain + cost + recommendation. Don't apply it. Wait for the call.

Examples in tree:
- `crates/prunr-core/src/guided_filter.rs` — buffer reuse + `box_filter_into` for alloc reuse.
- `crates/prunr-core/src/inpaint_blend.rs` — bbox crop + sequential channels + reused mean buffers + numerical guard. All quality-neutral.

## Code-writing defaults (compounded from /simplify passes)

These are the patterns `/simplify` agents repeatedly flag. Apply them on the first attempt:

- **Size-aware cloning.** `.clone()` on a type >1 KB deserves a second look — `Arc::clone` is always free. If the data is shared read-only, store as `Arc<T>` at the point of creation, not at the point of use.
- **Parameter count alarm at 6.** When a new param pushes a function past 6, either the function does two things or the params want to be a struct. `chip_option_rgba` at 9 is the ceiling this codebase has tolerated.
- **Stringly-typed smell.** If you're `match`-ing on a set of literal strings (model names, scale names, mode names), extract an enum with `FromStr` + `Display` — matches the `EdgeScale` / `ModelKind` / `LineMode` idiom.
- **Option / Result combinators.** Prefer `.map`, `.and_then`, `.map_or`, `.filter` over `if let Some(x) = foo { ... } else { ... }` when one arm is trivial. Stay in expression-land; reach for `match` only when both arms carry real work.
- **Stay in iterator-land.** Avoid `.collect::<Vec<_>>().iter().map(...)` — the intermediate Vec is wasted allocation. Chain the iterators.
- **Helper-before-hand-rolling.** Before writing inline code that looks like something else in the file (chip layout, popup wiring, file-read-and-delete, mask-cache key), grep for a helper. The `## Use-before-hand-rolling helper menu` tables below are canonical for GUI / core; when you see duplication across two call sites, extract a `pub(super) fn` rather than triple it.
- **Tiered fields → tiered cache keys.** When a cached artifact depends on N inputs, the cache key must include all N. A mask cached on `(line_strength)` silently broke on scale change; now it's `(line_strength, edge_scale)`. If adding an input that affects a cache, audit every cache-key tuple reading from it.

## Verify before bundling / wiring

Before adding a CI step (or `xtask`, or shell snippet) that copies, references, or globs a build-output file, run the equivalent local check first: `ls target/<target>/release/<file>`, or `ldd <binary>`, or whatever proves the assumption. Don't trust documentation that says "this feature drops the file there" — verify. The 60-second local check beats a 30-minute CI feedback loop. This applies double to any step you write `cp ... || exit 1` around: that error annotation is only useful if the cp ever has a chance of working.

Specific case that prompted this rule: Phase 6-01 assumed `pykeio/ort` `download-binaries` placed `libonnxruntime.so` next to the binary on Linux/Windows. It doesn't — that runtime is statically linked into the binary on those platforms; only GPU-provider plugins ship as separate `.so` files. A single `ldd target/release/prunr` would have shown the assumption was wrong before any YAML got written.

## Goal-driven for bug fixes

When fixing a perf or correctness bug, write the test that **reproduces** it first. Pin the invariant in code, then make the fix turn the test green. Without the failing-then-passing transition, you've shipped a fix-shaped change, not a verified one — and the regression has nothing guarding against its return.

This isn't TDD evangelism. It's specifically about bug fixes: the test you write is the contract that the bug is gone. The Phase 14 `cold_line_mode_dispatch_outranks_warm_under_max` test is the right pattern — it pins the exact `.max()` ordering that caused the Off↔Subject toggle slowdown, so any future refactor that re-poisons the warm path fails loudly.

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

**Cross-subsystem behaviour contracts** earn a behaviour test enumerating every variant of the trigger event. Wire-format tests are not enough — the IPC bincode roundtrip confirms bytes encode/decode correctly, but a manager's *handling* of each variant is a separate contract. The pattern: when subsystem A's state must respond to subsystem B's events (or to a lifecycle phase, slow frame, file-system race, etc.), pin the contract with a test that fails when the response drifts.

Concrete shapes that count as boundary tests:

- An event-handler that branches on an enum: one test row per variant. "Tested it once with `ImageDone`" is not enough when `InpaintDone` exists too.
- A lifecycle ordering invariant (file written before spawn must survive spawn; cleanup-on-crash must not touch a sibling subprocess's in-flight files): test writes the marker, drives the lifecycle, asserts the invariant.
- A timing/debounce contract: drive a fake clock through the worst case (slow frames, rapid bursts), assert the dispatch arrives within bounded time.
- A scope/prefix contract on a destructive operation (cleanup, eviction, kill-all): seed inputs both inside and outside the scope, run the operation, assert only the in-scope items are affected.

If a regression you ship has the shape "subsystem A changed and subsystem B broke quietly," the boundary test that would have caught it is the one to write — alongside the fix, not after. CLAUDE.md `## Goal-driven for bug fixes` already requires this for one-off bugs; the boundary-test rule is the same idea applied across every subsystem seam.

## Layers

Each layer has its own state-ownership rules. The **coordinator pattern** in the next section applies **only** to the GUI layer. Other layers have their own conventions:

| Layer                       | Location                                 | Convention                                             |
|-----------------------------|------------------------------------------|--------------------------------------------------------|
| Core inference pipeline     | `crates/prunr-core/src/`                 | Pure functions; caller-owned state. See `## prunr-core conventions`. |
| Model blobs + registry      | `crates/prunr-models/src/`               | `REGISTRY` of `ModelDescriptor`; `Bundled` via `include_bytes!` + `OnceLock<Vec<u8>>`, `OnDemand` via user data dir. Entry: `resolve_bytes(id)`. No deps on other workspace crates. |
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
- On-demand model downloads → `DownloadManager` (Phase 17)

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
| Model download lifecycle (state, queue, progress)   | `DownloadManager`                                  |
| Save-as dialog, clipboard, file picker              | `SystemBridge`                                     |
| Canvas zoom/pan state                               | `PrunrApp.zoom_state`                              |
| Brush tool toggle / settings / active stroke / trail | `PrunrApp.brush_state` (see `BrushState`)         |
| Per-item brush correction bytes + hash               | `BatchItem.mask_correction` + `correction_hash`    |
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
| Merge brush strokes into the per-item correction | `BatchItem::commit_correction(strokes)`  |
| Wipe per-item brush correction (undoable)      | `BatchItem::clear_correction()`            |
| Pop / push brush stroke history                | `BatchItem::undo_stroke()` / `redo_stroke()` |
| Read/write brush settings                      | `BrushState::settings()` / `settings_mut()` |

If the helper you want doesn't exist, add it to the coordinator — not to `PrunrApp`.

### View-layer helper menu

Row 2 / 3 chip rendering has its own set of pub(super) helpers in `chip.rs`. Before hand-rolling a chip / popover layout:

| You need…                                      | Use                                         |
|------------------------------------------------|---------------------------------------------|
| Chip button (icon + value + accent border)     | `chip::chip_button(ui, icon, value, accent)` |
| Popover attached to a chip button              | `chip::popup_for(ui, id, &resp, body)`       |
| Strong-headline tooltip on any response        | `chip::chip_tooltip(resp, label, body)`      |
| Slider row without a chip wrapper              | `chip::slider_row_f32` / `slider_row_u32`    |
| Centred non-resizable modal (backdrop + close) | `theme::standard_modal_window(ctx, id, title, [w, h], body)` — returns `bool` |
| Format byte counts for display                  | `views::format_byte_size(bytes)` |

Any new chip-shaped control (e.g. the Scale chip in `lines_popover.rs`) uses these three primitives — matches the visual rhythm of every other chip and keeps stroke / rounding / padding in one file.

### View → app intent pattern (Phase 17)

Views shouldn't poke `PrunrApp` fields directly. Return an intent on the corresponding `*Change` struct (e.g. `ToolbarChange.open_model_store: Option<ModelStoreRequest>`) and let `apply_*_change` decide what to do. Same shape for any future "view requests app open a modal / fire a coordinator action" — `Option<Request>` where `Request` carries the args.

### First-frame one-shot toasts (Phase 17)

If you need to show a toast message produced during construction (e.g. `Self::new`), don't try to push to `toasts` directly — frame 0 of egui isn't ready. Use a `pending_*: Option<String>` field, take it in the first `pump_*` call: `if let Some(msg) = self.pending_x.take() { self.toasts.info(msg); }`.

### Subprocess helper menu

IPC readers / writers share patterns. Before hand-rolling:

| You need…                                      | Use                                         |
|------------------------------------------------|---------------------------------------------|
| Read a temp file and delete it                 | `worker::read_and_delete(path) -> Option<Vec<u8>>` |
| f32 slice → LE bytes (for temp-file write)     | `subprocess::ipc::f32s_to_le_bytes(&[f32])` |
| LE bytes → Vec<f32> (for temp-file read)       | `subprocess::ipc::le_bytes_to_f32s(&[u8])`  |

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

### Numerical & color invariants

Image-math regressions are silent — a wrong gamma, alpha-blend, or denominator change ships without any unit test failing because the result is "still an image, just slightly different." These rules pin the invariants explicitly so refactors don't drift them.

- **Straight (un-premultiplied) sRGB throughout.** Every `RgbaImage` in core holds straight alpha in sRGB-encoded u8. Do **not** introduce premultiplied intermediates or linear-light conversion inside core math without an explicit `// rationale: ...` comment naming why and where the value is converted back. An agent's instinct to "premultiply for correct blending" or "work in linear-light because that's physically correct" is almost always wrong here — the existing math is calibrated to straight sRGB, and changing the working space silently shifts pixel output. If a real need surfaces (e.g. a future linear-light bloom effect), it ships as its own commit with golden-suite updates.

- **f32 working buffers for math-heavy paths.** Mask refinement, guided filtering, alpha blending, color matching, edge feathering all operate on `Vec<f32>` (or per-pixel f32). Clamp + cast to u8 only at the final write back to `RgbaImage` (`(x * 255.0).clamp(0.0, 255.0) as u8`, or via `image::Rgba` constructors that clamp internally). Do not roll u8 arithmetic in working buffers — halos and color banding are cheap to introduce, expensive to spot.

- **Guard the denominator.** Where division by a variance / sum / count is involved, guard with `.max(0.0)` (or an epsilon-floor when zero is also pathological). Canonical: `inpaint_blend::seam_guided_blend:262` — `let var = (mgg_v - mg_v * mg_v).max(0.0);`. f32 rounding on flat regions of the guide image can produce sub-zero variance values that, fed through `1.0 / (var + eps)`, amplify f32 noise into the output. Apply the same guard in any new `guided_filter`-shaped math.

- **Saturating cast at the boundary, not in the middle.** Converting f32 → u8 with `as u8` silently wraps negatives and truncates above 255. Always go through `.clamp(0.0, 255.0) as u8` (or the `image` crate's clamping constructors). The same goes for sums of u8 channels: prefer f32 working math over `saturating_add` chains that mask off valid out-of-range intermediate states.

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
