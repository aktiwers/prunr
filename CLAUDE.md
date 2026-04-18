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

## State ownership

`PrunrApp` is a **coordinator**, not an owner. It holds UI visibility flags and handles to domain coordinators — not business logic.

Business state lives in its own module with a clear owner:
- Batch orchestration → `BatchDispatcher`
- Result history + preset undo → `HistoryManager`
- Drag-export lifecycle → `DragExportState`
- Live preview → `LivePreviewDispatcher`

Before adding a new `PrunrApp` field, ask: which coordinator owns this domain? Default-to-no.
