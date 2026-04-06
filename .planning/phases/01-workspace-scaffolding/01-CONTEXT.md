# Phase 1: Workspace Scaffolding - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Cargo workspace structure, CI pipeline, model embedding foundation, and release tooling exist — any developer can clone, run `cargo xtask fetch-models`, build, and run a placeholder binary on all three platforms. No inference logic, no GUI, no CLI commands — just the skeleton that all subsequent phases build on.

</domain>

<decisions>
## Implementation Decisions

### Model Acquisition
- Models are NOT committed to git and NOT downloaded at runtime by end users
- Developer runs `cargo xtask fetch-models` once after cloning — downloads silueta + u2net from HuggingFace
- Models cached in `models/` directory (gitignored)
- SHA256 checksums hardcoded in xtask — verified on download
- At compile time, `include-bytes-zstd` embeds models into the binary (level 19 compression)
- `dev-models` feature flag loads from filesystem instead (avoids recompilation during development)
- `bgprunr-models` is an isolated crate so model embedding only recompiles when model files change

### CI Strategy
- GitHub Actions with native runners for all platforms
- macOS: macos-14 (arm64) + macos-13 (x86_64) native runners — no cross-compilation (CoreML/Metal requires native SDK)
- Linux: ubuntu-latest x86_64
- Windows: windows-latest x86_64
- Models cached via actions/cache keyed by SHA256 — download once, reuse across builds
- `cargo xtask fetch-models` runs in CI before build step

### Release Tooling
- cargo-dist for generating release workflow and per-platform binary artifacts
- Even Phase 1 should produce a placeholder binary artifact in CI (validates the pipeline)
- Single workspace version via `workspace.package.version` — all crates share one version

### Crate API Surface
- **Single binary with mode flag**: `bgprunr` with no args opens GUI; `bgprunr remove ...` runs CLI mode. One binary to distribute, not two.
- This means the workspace has: `bgprunr` (binary crate), `bgprunr-core` (lib), `bgprunr-models` (lib)
- Key traits defined in bgprunr-core: `InferenceEngine`, `ImageProcessor` — but only the ORT concrete implementation exists. SOLID-ready without over-engineering.
- Error handling: `thiserror` enums per crate with `#[from]` conversions. Typed errors throughout.
- Binary name: `bgprunr` (lowercase)

### Claude's Discretion
- Exact xtask implementation details (reqwest vs ureq for download)
- CI workflow file structure (single vs matrix)
- cargo-dist configuration specifics
- Placeholder binary content (can be minimal "hello world" that proves the build works)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture
- `ARCHITECTURE.md` — Workspace structure, crate dependency graph, model embedding pattern, platform-specific notes

### Research
- `.planning/research/STACK.md` — Crate versions, ort feature flags, include-bytes-zstd usage
- `.planning/research/PITFALLS.md` — Windows DLL hell, model embed compile time, isolated model crate rationale
- `.planning/research/ARCHITECTURE.md` — Component boundaries, build order, Cargo workspace patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project, no existing code

### Established Patterns
- None yet — this phase establishes the foundational patterns

### Integration Points
- This phase creates the workspace that all subsequent phases build on
- bgprunr-core's trait definitions will be the API contract for Phase 2 (inference engine)
- The single-binary architecture means CLI and GUI code live in the same binary crate, dispatched by clap subcommands

</code_context>

<specifics>
## Specific Ideas

- Single binary approach: `bgprunr` (no args) = GUI, `bgprunr remove input.jpg -o output.png` = CLI mode. User distributes and downloads one file that does both.
- SOLID from day one: traits for inference engine abstraction even though only ORT backend exists. This is for clean architecture, not premature flexibility.
- The `xtask` pattern (cargo xtask fetch-models) is preferred over build.rs for model fetching because build.rs runs on every build — xtask is explicit and one-time.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-workspace-scaffolding*
*Context gathered: 2026-04-06*
