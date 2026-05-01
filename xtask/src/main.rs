use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

struct ModelSpec {
    /// Registry id; OnDemand mirroring uses the filename from
    /// `prunr_models::descriptor(id)` so xtask and registry can't drift.
    id: prunr_models::ModelId,
    /// Dev-mode unversioned filename in `models/`.
    name: &'static str,
    url: &'static str,
    sha256: &'static str, // Empty string = bootstrap mode (skip verification, print hash)
}

// After first run, replace empty strings with the printed SHA256 values.
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: prunr_models::ModelId::Silueta,
        name: "silueta.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        sha256: "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
    },
    ModelSpec {
        id: prunr_models::ModelId::U2net,
        name: "u2net.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        sha256: "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    },
    ModelSpec {
        id: prunr_models::ModelId::BiRefNetLite,
        name: "birefnet_lite.onnx",
        url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
        sha256: "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
    },
    // DexiNed is exported from PyTorch weights via scripts/export_dexined.py
    // and hosted on prunr's own releases (separate tag from app versions).
    ModelSpec {
        id: prunr_models::ModelId::DexiNed,
        name: "dexined.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/dexined.onnx",
        sha256: "cba9193b1e3fbcb5bd196001a9aae13bafaa309442f6cb074330c426cc61ec5a",
    },
    // LaMa for the Eraser tool. OnDemand: distributed via Model Store at
    // runtime. Dev-mode uses `models/lama_fp32.onnx`; this xtask also
    // mirrors it to the user data dir so the dev workflow exercises the
    // same code path as production.
    ModelSpec {
        id: prunr_models::ModelId::LaMaFp32,
        name: "lama_fp32.onnx",
        url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
    },
    // MI-GAN + Big-LaMa: xtask fetches the already-published artefacts
    // from prunr's own GitHub release rather than rebuilding from source
    // each time — the export scripts (`scripts/export_migan.py` /
    // `scripts/export_big_lama.py`) need a Python env that the Rust dev
    // workflow doesn't otherwise require. To regenerate from source,
    // run the export script directly and re-upload via gh release.
    ModelSpec {
        id: prunr_models::ModelId::Migan,
        name: "migan.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/migan-1.0.0.onnx",
        sha256: "17531b1604e56ff3179a22824c19debf12741dadc551b4500b035bcb216b58ba",
    },
    ModelSpec {
        id: prunr_models::ModelId::BigLaMa,
        name: "big_lama.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/big_lama-1.0.0.onnx",
        sha256: "523e84eb2ec2df933714cbab6983627a9909f9f23cd848fbbe977356c54bdaa0",
    },
];

/// OnDemand filename from the registry, or `None` for Bundled models.
/// Single source of truth — bumping a model's version in REGISTRY
/// automatically updates xtask's mirror target.
fn ondemand_target(id: prunr_models::ModelId) -> Option<&'static str> {
    match prunr_models::descriptor(id)?.source {
        prunr_models::ModelSource::OnDemand { filename, .. } => Some(filename),
        // Multi-part bundles aren't single-file targets — xtask doesn't
        // mirror them today; the production DownloadManager handles them.
        prunr_models::ModelSource::Bundled
        | prunr_models::ModelSource::MultiPartOnDemand { .. } => None,
    }
}

fn user_data_models_dir() -> Option<std::path::PathBuf> {
    prunr_models::data_dir().map(|d| d.join("models"))
}

/// Copy `src` to the user data dir under the registry-versioned filename
/// so the production OnDemand path (`prunr_models::resolve_bytes`)
/// resolves it without filesystem-shim heuristics.
fn mirror_to_user_data_dir(src: &Path, target: &str) -> anyhow::Result<()> {
    let Some(dir) = user_data_models_dir() else {
        eprintln!("  (skipping user-data-dir mirror: dirs::data_dir returned None)");
        return Ok(());
    };
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join(target);
    std::fs::copy(src, &dst)?;
    println!("  Mirrored to {}", dst.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let task = std::env::args().nth(1).unwrap_or_default();
    match task.as_str() {
        "fetch-models" => fetch_models(),
        "probe-load-dynamic" => probe_load_dynamic(),
        "install-runtime" => install_runtime(),
        "render-golden-diff" => render_golden_diff(),
        _ => {
            eprintln!("Usage: cargo xtask <task>");
            eprintln!("Tasks:");
            eprintln!("  fetch-models           Download and verify ONNX model files to models/");
            eprintln!("  probe-load-dynamic     [Phase 19-02b] Verify ort's `load-dynamic` mechanism");
            eprintln!("                         can swap libonnxruntime at runtime end-to-end.");
            eprintln!("                         Pass dylib path as second arg, or set ORT_DYLIB_PATH.");
            eprintln!("  install-runtime        [Phase 19-09] Download + extract an `onnxruntime-*`");
            eprintln!("                         PyPI wheel into the user runtime store. Args:");
            eprintln!("                         <package> <version> [target-name]");
            eprintln!("                         e.g. install-runtime onnxruntime-openvino 1.24.1");
            eprintln!("  render-golden-diff     [Phase 20-04] After UPDATE_GOLDEN regenerates");
            eprintln!("                         expected.png files, produce side-by-side");
            eprintln!("                         <before | after | diff> triptych PNGs for every");
            eprintln!("                         changed golden. Lets a reviewer visually inspect");
            eprintln!("                         math-changing refactors before commit.");
            eprintln!("                         Output: golden-diffs/<tier>/<id>.png");
            std::process::exit(1);
        }
    }
}

/// 19-02b verification: load an externally-provided libonnxruntime via
/// `ort::init_from`, build a Session against a real ONNX model, run one
/// inference. Confirms the `load-dynamic` path that Phase 19's Runtime
/// Store will rely on.
///
/// Usage:
///   cargo xtask probe-load-dynamic <path-to-libonnxruntime.so>
///   cargo xtask probe-load-dynamic       # falls back to ORT_DYLIB_PATH
fn probe_load_dynamic() -> anyhow::Result<()> {
    use ort::{
        inputs as ort_inputs,
        session::{Session, builder::GraphOptimizationLevel},
        value::Tensor,
    };

    // ort::Error is generic; anyhow::Context's blanket impl doesn't fit,
    // so we wrap each ORT-returning call with anyhow!. Use a small alias
    // to keep the wrapping uniform.
    fn ort_err<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> anyhow::Error {
        move |e| anyhow::anyhow!("{stage}: {e}")
    }

    let dylib_path: std::path::PathBuf = std::env::args().nth(2)
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(std::path::PathBuf::from))
        .ok_or_else(|| anyhow::anyhow!(
            "no dylib path supplied — pass as 2nd arg or set ORT_DYLIB_PATH"
        ))?;

    if !dylib_path.is_file() {
        anyhow::bail!("dylib not found: {}", dylib_path.display());
    }

    println!("=== Phase 19-02b probe: load-dynamic verification ===");
    println!("Loading: {}", dylib_path.display());

    let env = ort::init_from(&dylib_path).map_err(ort_err("ort::init_from"))?;
    if !env.commit() {
        anyhow::bail!("ort env commit returned false (already committed?)");
    }
    println!("OK ort::init_from + commit");

    let silueta = prunr_models::silueta_bytes();
    println!("Building session against bundled Silueta ({} bytes)", silueta.len());

    let mut session = Session::builder().map_err(ort_err("Session::builder"))?
        .with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_err("opt level"))?
        .commit_from_memory(silueta).map_err(ort_err("commit_from_memory"))?;

    let inputs = session.inputs();
    let outputs = session.outputs();
    println!("OK Session built");
    println!("  Inputs:  {:?}", inputs.iter().map(|i| i.name()).collect::<Vec<_>>());
    println!("  Outputs: {:?}", outputs.iter().map(|o| o.name()).collect::<Vec<_>>());

    let input_name = inputs[0].name().to_string();
    // Silueta input: 1×3×320×320 f32 in [0, 1].
    let arr = ndarray::Array4::<f32>::zeros((1, 3, 320, 320));
    let t = Tensor::from_array(arr).map_err(ort_err("tensor build"))?;
    let out = session.run(ort_inputs![input_name.as_str() => &t]).map_err(ort_err("session.run"))?;
    println!("OK Inference ran, {} output(s)", out.len());

    println!();
    println!("=== load-dynamic VERIFIED ===");
    println!("`ort::init_from(path)` successfully loaded an external libonnxruntime");
    println!("and built+ran a Session against it. This is the mechanism Phase 19's");
    println!("Runtime Store will use to swap in EP-specific ORT bundles per-user.");
    Ok(())
}

fn fetch_models() -> anyhow::Result<()> {
    std::fs::create_dir_all("models")?;
    let client = reqwest::blocking::Client::builder()
        .user_agent("prunr-xtask/0.1")
        .build()?;

    for spec in MODELS {
        let dest = std::path::Path::new("models").join(spec.name);

        if dest.exists() {
            println!("{}: exists, verifying checksum...", spec.name);
            let bytes = std::fs::read(&dest)?;
            let hash = hex::encode(Sha256::digest(&bytes));

            if spec.sha256.is_empty() {
                println!("  Computed SHA256: {hash}");
                println!("  IMPORTANT: Hardcode this in xtask/src/main.rs");
                continue;
            }

            if hash == spec.sha256 {
                println!("  OK (cached)");
                if let Some(target) = ondemand_target(spec.id) {
                    mirror_to_user_data_dir(&dest, target)?;
                }
                continue;
            }
            println!("  Checksum mismatch — re-downloading");
        } else {
            println!("{}: downloading from {}", spec.name, spec.url);
        }

        let response = client.get(spec.url).send()?;
        if !response.status().is_success() {
            anyhow::bail!(
                "HTTP {} downloading {}",
                response.status(),
                spec.name
            );
        }
        let bytes = response.bytes()?;
        let hash = hex::encode(Sha256::digest(&bytes));

        if spec.sha256.is_empty() {
            println!("  Computed SHA256: {hash}");
            println!(
                "  IMPORTANT: Hardcode this in xtask/src/main.rs as {} constant",
                spec.name
            );
        } else if hash != spec.sha256 {
            anyhow::bail!(
                "SHA256 mismatch for {}:\n  expected: {}\n  got:      {}",
                spec.name,
                spec.sha256,
                hash
            );
        }

        let mut file = std::fs::File::create(&dest)?;
        file.write_all(&bytes)?;
        println!("  Saved to {}", dest.display());

        compress_to_zst(&dest, &bytes)?;
        if let Some(target) = ondemand_target(spec.id) {
            mirror_to_user_data_dir(&dest, target)?;
        }
    }

    // Catch any models with stale .zst (.onnx newer than .zst, or .zst missing).
    // Belt-and-braces for the case where someone fetched models with an older
    // xtask and the .zst step never ran.
    for spec in MODELS {
        let dest = std::path::Path::new("models").join(spec.name);
        if !dest.exists() { continue; }
        let zst_path = dest.with_extension("onnx.zst");
        let needs = match (dest.metadata(), zst_path.metadata()) {
            (Ok(_), Err(_)) => true,
            (Ok(o), Ok(z)) => o.modified().ok() > z.modified().ok(),
            _ => false,
        };
        if needs {
            let bytes = std::fs::read(&dest)?;
            compress_to_zst(&dest, &bytes)?;
        }
    }

    println!("\nDone. If any SHA256 values above say IMPORTANT, update xtask/src/main.rs.");
    Ok(())
}

/// zstd level 19 — matches the existing `.onnx.zst` artefacts that
/// `prunr-models` embeds via `include_bytes!`. Skips if the existing
/// `.zst` already matches (cheap mtime check via fetch_models).
fn compress_to_zst(onnx_path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let zst_path = onnx_path.with_extension("onnx.zst");
    let compressed = zstd::bulk::compress(bytes, 19)?;
    let onnx_mb = bytes.len() as f64 / 1024.0 / 1024.0;
    let zst_mb = compressed.len() as f64 / 1024.0 / 1024.0;
    std::fs::write(&zst_path, &compressed)?;
    println!(
        "  Compressed {} ({:.1} MB → {:.1} MB)",
        zst_path.display(), onnx_mb, zst_mb,
    );
    Ok(())
}

/// Phase 19-09 / J-7: download + extract a `onnxruntime-*` PyPI wheel.
///
/// Default target: `<user data>/prunr/runtimes/<package>-<version>-<rid>/`
/// (the runtime store, hot-loaded by `ort_runtime::resolve_dylib_path`).
///
/// `--stage-to <DIR>`: extract directly into `<DIR>/` instead of the
/// runtime store. CI release packaging uses this to bundle the CPU
/// runtime next to the binary, satisfying `bundled_dylib`'s
/// `<exe>/runtime/libonnxruntime.{so,dylib,dll}` lookup. Path-safety
/// guard (refuse to wipe a path whose parent isn't `runtimes`) does
/// NOT apply when staging — caller picks the path.
///
/// Usage:
///   cargo xtask install-runtime <package> <version> [target-name]
///   cargo xtask install-runtime <package> <version> --stage-to <DIR>
///
/// Examples:
///   cargo xtask install-runtime onnxruntime-openvino 1.24.1
///   cargo xtask install-runtime onnxruntime 1.24.1 --stage-to dist/runtime
///
/// Skips Python bindings (`onnxruntime_pybind11_state.cpython-*.so`,
/// `*.py` sources) since prunr links via Rust ort. Strips the version
/// suffix off `libonnxruntime.so.<ver>` (Linux) and
/// `libonnxruntime.<ver>.dylib` (macOS) so the canonical name is what
/// `prunr_app::ort_runtime::resolve_dylib_path()` picks up.
fn install_runtime() -> anyhow::Result<()> {
    use prunr_runtime_install as ri;
    let package = std::env::args().nth(2)
        .ok_or_else(|| anyhow::anyhow!("missing <package> arg, e.g. onnxruntime-openvino"))?;
    let version = std::env::args().nth(3)
        .ok_or_else(|| anyhow::anyhow!("missing <version> arg, e.g. 1.24.1"))?;

    // Either `--stage-to <DIR>` (CI bundle path) or a positional
    // target-name (runtime store path under data_dir).
    let mut stage_to: Option<PathBuf> = None;
    let mut target_name: Option<String> = None;
    let extra: Vec<String> = std::env::args().skip(4).collect();
    let mut i = 0;
    while i < extra.len() {
        let arg = &extra[i];
        if arg == "--stage-to" {
            stage_to = Some(PathBuf::from(extra.get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("--stage-to requires a directory argument"))?));
            i += 2;
        } else {
            target_name = Some(arg.clone());
            i += 1;
        }
    }
    if stage_to.is_some() && target_name.is_some() {
        anyhow::bail!("--stage-to and a positional target-name are mutually exclusive");
    }

    let short = package.strip_prefix("onnxruntime-").unwrap_or(&package);

    println!("=== install-runtime ===");
    println!("Package:  {package} {version}");

    let target_dir = if let Some(stage) = &stage_to {
        println!("Stage-to: {}", stage.display());
        stage.clone()
    } else {
        let target_name = target_name
            .unwrap_or_else(|| ri::install_subdir(short, &version));
        ri::validate_subdir(&target_name).map_err(|e| anyhow::anyhow!(e))?;
        let dir = prunr_models::data_dir()
            .ok_or_else(|| anyhow::anyhow!("could not resolve user data dir"))?
            .join("runtimes")
            .join(&target_name);
        println!("Target:   {}", dir.display());
        dir
    };

    let json_url = format!("https://pypi.org/pypi/{package}/{version}/json");
    println!("Querying {json_url}");
    let client = reqwest::blocking::Client::builder()
        .user_agent("prunr-xtask/0.1")
        .build()?;
    let metadata: serde_json::Value = client.get(&json_url).send()?.json()?;
    let urls = metadata["urls"].as_array()
        .ok_or_else(|| anyhow::anyhow!("PyPI metadata missing `urls`"))?;

    let wheel = ri::pick_wheel_for_host(urls).map_err(|e| anyhow::anyhow!(e))?;
    println!("Selected: {}", wheel.url);
    println!("SHA256:   {}", wheel.sha256);

    println!("Downloading…");
    let mut on_progress = |so_far: u64, total: u64| {
        if total > 0 && so_far == total {
            println!("Verified ({:.1} MB)", so_far as f64 / 1024.0 / 1024.0);
        }
    };
    let mut hooks = ri::DownloadHooks::progress_only(&mut on_progress);
    let bytes = ri::download_wheel(&wheel, &mut hooks).map_err(|e| anyhow::anyhow!(e))?;
    ri::verify_sha256(&bytes, &wheel.sha256).map_err(|e| anyhow::anyhow!(e))?;

    if target_dir.exists() {
        std::fs::remove_dir_all(&target_dir)?;
    }
    std::fs::create_dir_all(&target_dir)?;
    println!("Extracting to {}", target_dir.display());
    ri::extract_wheel(&bytes, &target_dir).map_err(|e| anyhow::anyhow!(e))?;

    println!();
    println!("=== install-runtime DONE ===");
    if stage_to.is_none() {
        println!("Try the doctor command to confirm pickup:");
        println!("  target/debug/prunr --doctor");
    }
    Ok(())
}

/// 20-04: render side-by-side `<before | after | diff>` triptych PNGs for any
/// `expected.png` under `crates/prunr-core/tests/golden_data/{postprocess,e2e}/`
/// that differs from its `git HEAD` version. Run after `UPDATE_GOLDEN=1 cargo
/// test ...` regenerates expected PNGs — the triptychs let a reviewer visually
/// inspect a math-changing refactor before committing the golden update.
///
/// "before" = bytes from `git show HEAD:<path>`
/// "after"  = bytes on disk (the regenerated expected.png)
/// "diff"   = per-pixel max channel diff, scaled ×4 + clamped to 255 for
///            visibility (small differences would otherwise be invisible).
///
/// Output: `golden-diffs/<tier>/<id>.png` (the directory is gitignored —
/// triptychs are reviewer artifacts, not committed).
///
/// Args (positional, all optional):
///   render-golden-diff [--phase postprocess|e2e|both] [--id <pattern>]
///
/// `--id` filters fixture ids by substring match.
fn render_golden_diff() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let mut phase_filter: Option<String> = None;
    let mut id_filter: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" => {
                phase_filter = args.get(i + 1).cloned();
                i += 2;
            }
            "--id" => {
                id_filter = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                anyhow::bail!("unknown arg `{other}` (expected `--phase` or `--id`)");
            }
        }
    }

    let phases: &[&str] = match phase_filter.as_deref() {
        None | Some("both") => &["postprocess", "e2e"],
        Some("postprocess") => &["postprocess"],
        Some("e2e") => &["e2e"],
        Some(other) => anyhow::bail!("unknown phase `{other}` (expected postprocess|e2e|both)"),
    };

    let workspace_root = std::env::current_dir()?;
    let output_root = workspace_root.join("golden-diffs");
    std::fs::create_dir_all(&output_root)?;

    let mut total_changed = 0usize;
    let mut total_unchanged = 0usize;
    let mut total_new = 0usize;

    for phase in phases {
        let fixture_root =
            workspace_root.join("crates/prunr-core/tests/golden_data").join(phase);
        if !fixture_root.is_dir() {
            continue;
        }
        let phase_out = output_root.join(phase);
        std::fs::create_dir_all(&phase_out)?;

        let entries = std::fs::read_dir(&fixture_root)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir());

        for fixture_dir in entries {
            let id = fixture_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if let Some(ref pat) = id_filter {
                if !id.contains(pat.as_str()) {
                    continue;
                }
            }

            let expected_path = fixture_dir.join("expected.png");
            if !expected_path.is_file() {
                continue;
            }

            let rel_path = expected_path
                .strip_prefix(&workspace_root)
                .unwrap_or(&expected_path);

            let head_bytes = match git_show_head(rel_path) {
                Ok(b) => Some(b),
                Err(GitShowErr::NotInIndex) => None,
                Err(GitShowErr::Other(msg)) => {
                    anyhow::bail!("git show failed for {}: {msg}", rel_path.display());
                }
            };

            let disk_bytes = std::fs::read(&expected_path)?;

            let head_bytes = match head_bytes {
                Some(b) => b,
                None => {
                    total_new += 1;
                    println!("[{phase}] {id}: NEW (no HEAD version) — skipping triptych");
                    continue;
                }
            };

            if head_bytes == disk_bytes {
                total_unchanged += 1;
                continue;
            }

            let before = image::load_from_memory(&head_bytes)
                .map_err(|e| anyhow::anyhow!("decode HEAD expected.png ({}): {e}", rel_path.display()))?
                .to_rgba8();
            let after = image::load_from_memory(&disk_bytes)
                .map_err(|e| anyhow::anyhow!("decode disk expected.png ({}): {e}", expected_path.display()))?
                .to_rgba8();

            if before.dimensions() != after.dimensions() {
                println!(
                    "[{phase}] {id}: DIM MISMATCH {:?} vs {:?} — writing side-by-side without diff panel",
                    before.dimensions(),
                    after.dimensions()
                );
                let triptych = compose_dim_mismatch(&before, &after);
                let out_path = phase_out.join(format!("{id}.png"));
                triptych.save(&out_path)?;
                total_changed += 1;
                continue;
            }

            let (w, h) = (before.width(), before.height());
            let diff = scaled_abs_diff(&before, &after);
            let triptych = compose_triptych(&before, &after, &diff);
            let out_path = phase_out.join(format!("{id}.png"));
            triptych.save(&out_path)?;
            let max_diff = max_channel_diff(&before, &after);
            println!(
                "[{phase}] {id}: CHANGED ({w}x{h}, max channel diff {max_diff}) → {}",
                out_path.display()
            );
            total_changed += 1;
        }
    }

    println!();
    println!(
        "=== render-golden-diff DONE: {total_changed} changed, {total_unchanged} unchanged, {total_new} new ===",
    );
    if total_changed > 0 {
        println!("Inspect triptychs under {}", output_root.display());
        println!("If the changes are intentional, commit the goldens as a separate commit:");
        println!("  git commit -m 'goldens: update for <reason>'");
    }
    Ok(())
}

enum GitShowErr {
    /// File doesn't exist in HEAD (a brand-new fixture). Caller should skip.
    NotInIndex,
    /// Anything else (binary not present, repo not initialized, etc).
    Other(String),
}

fn git_show_head(path: &Path) -> Result<Vec<u8>, GitShowErr> {
    let arg = format!("HEAD:{}", path.display());
    let output = std::process::Command::new("git")
        .arg("show")
        .arg(&arg)
        .output()
        .map_err(|e| GitShowErr::Other(format!("spawn git: {e}")))?;
    if output.status.success() {
        return Ok(output.stdout);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    // git show emits this message both for missing path and missing ref —
    // checking for the path-specific phrase is the cleanest signal.
    if stderr.contains("does not exist") || stderr.contains("exists on disk, but not in") {
        return Err(GitShowErr::NotInIndex);
    }
    Err(GitShowErr::Other(stderr.into_owned()))
}

/// Per-pixel max channel absolute diff, scaled ×4 + clamped to 255 so that
/// small differences (the common case after a math-neutral-ish refactor)
/// remain visible. Output is RGBA with alpha=255 throughout.
fn scaled_abs_diff(a: &image::RgbaImage, b: &image::RgbaImage) -> image::RgbaImage {
    let (w, h) = a.dimensions();
    let mut out = image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let pa = a.get_pixel(x, y).0;
            let pb = b.get_pixel(x, y).0;
            let mut d = 0u32;
            for c in 0..4 {
                let cd = pa[c].abs_diff(pb[c]) as u32;
                if cd > d {
                    d = cd;
                }
            }
            let scaled = (d * 4).min(255) as u8;
            out.put_pixel(x, y, image::Rgba([scaled, scaled, scaled, 255]));
        }
    }
    out
}

fn max_channel_diff(a: &image::RgbaImage, b: &image::RgbaImage) -> u8 {
    let mut max = 0u8;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        for c in 0..4 {
            let d = pa.0[c].abs_diff(pb.0[c]);
            if d > max {
                max = d;
            }
        }
    }
    max
}

/// Lay out three same-size images horizontally with thin separators.
fn compose_triptych(
    before: &image::RgbaImage,
    after: &image::RgbaImage,
    diff: &image::RgbaImage,
) -> image::RgbaImage {
    let (w, h) = before.dimensions();
    let sep = 2u32;
    let total_w = w * 3 + sep * 2;
    let mut out = image::ImageBuffer::new(total_w, h);

    // Fill with neutral gray so separators are visible.
    for p in out.pixels_mut() {
        *p = image::Rgba([200, 200, 200, 255]);
    }

    blit(&mut out, before, 0, 0);
    blit(&mut out, after, w + sep, 0);
    blit(&mut out, diff, (w + sep) * 2, 0);
    out
}

fn compose_dim_mismatch(
    before: &image::RgbaImage,
    after: &image::RgbaImage,
) -> image::RgbaImage {
    let (bw, bh) = before.dimensions();
    let (aw, ah) = after.dimensions();
    let sep = 2u32;
    let total_w = bw + aw + sep;
    let total_h = bh.max(ah);
    let mut out = image::ImageBuffer::new(total_w, total_h);
    for p in out.pixels_mut() {
        *p = image::Rgba([200, 200, 200, 255]);
    }
    blit(&mut out, before, 0, 0);
    blit(&mut out, after, bw + sep, 0);
    out
}

fn blit(dst: &mut image::RgbaImage, src: &image::RgbaImage, dx: u32, dy: u32) {
    let (sw, sh) = src.dimensions();
    let (dw, dh) = dst.dimensions();
    let xe = (dx + sw).min(dw);
    let ye = (dy + sh).min(dh);
    for y in dy..ye {
        for x in dx..xe {
            let p = *src.get_pixel(x - dx, y - dy);
            dst.put_pixel(x, y, p);
        }
    }
}

