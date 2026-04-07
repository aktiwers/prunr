---
phase: 03-cli-binary
verified: 2026-04-06T00:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 3: CLI Binary Verification Report

**Phase Goal:** A user with no GUI can process single images and batches via the terminal, select models, tune parallelism, and get correct exit codes — the full core API is exercised under real scripting conditions
**Verified:** 2026-04-06
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `bgprunr remove --help` prints usage with all documented flags | VERIFIED (human) | Human-verified: --help shows all flags |
| 2  | All CLI flags parse correctly: --model, --jobs, --large-image, --output-dir, --force, --quiet, -o | VERIFIED | RemoveArgs in cli.rs lines 18–54 declares all 7 flags with correct clap attributes |
| 3  | ModelKind and LargeImagePolicy parse from string (clap ValueEnum) | VERIFIED | `#[derive(ValueEnum)]` on both CliModel (line 57) and LargeImagePolicy (line 75); `From<CliModel>` for `bgprunr_core::ModelKind` at lines 65–72 |
| 4  | bgprunr-app compiles with clap and indicatif as dependencies | VERIFIED | Cargo.toml lines 17–18 declare `clap = { workspace = true }` and `indicatif = { workspace = true }` |
| 5  | Single image processes correctly and exits 0 | VERIFIED (human) | Human test: `bgprunr remove car-1.jpg --force` produced transparent PNG, exit 0 |
| 6  | Batch processes multiple images with correct output-dir, exits 0/1/2 | VERIFIED (human) | Human test: `bgprunr remove *.jpg --output-dir /tmp/out --force` — 3 succeeded, exit 0 |
| 7  | --model u2net and --model silueta both accepted and produce output | VERIFIED (human) | Human test: `--model u2net` confirmed working |
| 8  | --jobs N controls parallelism, no crash | VERIFIED | `run_batch()` passes `args.jobs` directly to `batch_process()` (cli.rs line 366) |
| 9  | --quiet produces no stdout on success | VERIFIED | All `println!` and progress output gated on `!args.quiet` throughout cli.rs |
| 10 | Without --force, bgprunr refuses to overwrite existing output files | VERIFIED | `check_overwrite()` at cli.rs lines 135–144 returns Err when `out.exists() && !force` |
| 11 | Exit codes 0/1/2 correct for success/partial/failure | VERIFIED (human) | Human test: nonexistent.jpg → exit 1; code logic at cli.rs lines 442–448 |
| 12 | bgprunr (no args) prints hint and exits 0 | VERIFIED | main.rs lines 13–18: None arm prints GUI stub message and calls `std::process::exit(0)` |

**Score:** 12/12 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/bgprunr-app/src/cli.rs` | Clap derive structs: Cli, Commands, RemoveArgs; run_remove() orchestration | VERIFIED, WIRED | 449-line file; exports Cli, Commands, RemoveArgs, CliModel, LargeImagePolicy, run_remove, run_single, run_batch, all helpers |
| `crates/bgprunr-app/Cargo.toml` | clap, indicatif dependencies | VERIFIED | Lines 17–18 declare both workspace deps; dev-models feature at lines 11–13 |
| `crates/bgprunr-core/src/pipeline.rs` | process_image_unchecked for --large-image=process bypass | VERIFIED, WIRED | Lines 66–76: substantive implementation delegating to `process_image_from_decoded`; no large-image guard |
| `crates/bgprunr-app/src/main.rs` | Entry point dispatching to remove subcommand | VERIFIED, WIRED | 20-line file; calls `Cli::parse()`, dispatches `Commands::Remove(args)` to `cli::run_remove(args)`, captures exit code, calls `std::process::exit` |
| `crates/bgprunr-core/src/lib.rs` | Re-exports process_image_unchecked | VERIFIED | Line 14: `pub use pipeline::{process_image, process_image_unchecked};` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `main.rs` | `cli::run_remove()` | `match Cli::parse().command` → `Commands::Remove(args)` | WIRED | main.rs line 10: `cli::run_remove(args)` called and exit code captured |
| `cli::run_remove()` | `bgprunr_core::process_image` | single-image path with indicatif spinner | WIRED | cli.rs line 255: `process_image(&img_bytes, &engine, progress)` |
| `cli::run_remove()` | `bgprunr_core::process_image_unchecked` | `--large-image=process` branch | WIRED | cli.rs line 253: `process_image_unchecked(&img_bytes, &engine, progress)` when `args.large_image == LargeImagePolicy::Process` |
| `cli::run_remove()` | `bgprunr_core::batch_process` | batch path with MultiProgress | WIRED | cli.rs line 365: `batch_process(&valid_refs, model, args.jobs, Some(progress_cb))` |
| `cli::run_remove()` | `std::process::exit` | exit code returned to main.rs, dispatched there | WIRED | main.rs line 11: `std::process::exit(exit_code)` |
| `pipeline.rs::process_image_unchecked` | `lib.rs` re-export | `pub use pipeline::{..., process_image_unchecked}` | WIRED | lib.rs line 14 confirmed |

---

### Requirements Coverage

| Requirement | Description | Source Plans | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| CLI-01 | User can process a single image via `bgprunr input.jpg -o output.png` | 03-01, 03-02, 03-03 | SATISFIED | `run_single()` in cli.rs; human test confirmed transparent PNG, exit 0 |
| CLI-02 | User can batch process via glob/directory: `bgprunr *.jpg --output-dir ./results/` | 03-01, 03-02, 03-03 | SATISFIED | `run_batch()` in cli.rs; output dir auto-created via `create_dir_all`; human test: 3 images processed, exit 0 |
| CLI-03 | User can select model with `--model silueta\|u2net` | 03-01, 03-02, 03-03 | SATISFIED | `CliModel` enum with `ValueEnum`; `From<CliModel>` bridges to `ModelKind`; human test: `--model u2net` confirmed |
| CLI-04 | User can control parallelism with `--jobs N` | 03-01, 03-02, 03-03 | SATISFIED | `args.jobs` passed directly to `batch_process()` which uses rayon |
| CLI-05 | CLI exits with appropriate exit codes (0 success, 1 error, 2 partial failure in batch) | 03-01, 03-02, 03-03 | SATISFIED | Exit logic at cli.rs lines 442–448; human test: nonexistent.jpg → exit 1 confirmed |

No orphaned Phase 3 requirements found. REQUIREMENTS.md traceability maps CLI-01 through CLI-05 exclusively to Phase 3, all accounted for.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/bgprunr-app/src/cli.rs` | 451–452 | Trailing blank lines at EOF | Info | No functional impact |

No TODOs, FIXMEs, placeholder returns, stub handlers, or empty implementations found in any phase-modified file.

---

### Human Verification

Human verification was completed as part of Plan 03-03 (a blocking checkpoint gate). The following were confirmed working with real ONNX models and real JPEG images:

1. **Single image** — `bgprunr remove car-1.jpg --force` produced transparent PNG, exit 0
2. **Batch mode** — `bgprunr remove *.jpg --output-dir /tmp/out --force` processed 3 images, exit 0
3. **Model selection** — `--model u2net` accepted and produced output
4. **Error exit code** — `nonexistent.jpg` → exit 1
5. **Help** — `--help` shows all flags with descriptions

One bug was discovered and fixed during human verification: `--output-dir` would fail if the directory did not exist. Fixed by adding `std::fs::create_dir_all()` in `run_remove()` before dispatching to `run_single`/`run_batch`. The fix is present in cli.rs at lines 101–106.

---

### Gaps Summary

No gaps. All 12 observable truths verified. All 5 requirement IDs satisfied. All key links confirmed wired. No blocker anti-patterns.

---

_Verified: 2026-04-06_
_Verifier: Claude (gsd-verifier)_
