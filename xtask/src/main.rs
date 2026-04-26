use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

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

/// Phase 19-09: download + extract a `onnxruntime-*` PyPI wheel into
/// `<user data>/prunr/runtimes/<package>-<version>-<rid>/`.
///
/// Usage:
///   cargo xtask install-runtime <package> <version> [target-name]
///
/// Example:
///   cargo xtask install-runtime onnxruntime-openvino 1.24.1
///   → installs to `<data>/prunr/runtimes/openvino-1.24.1-linux-x64/`
///
/// Skips Python bindings (`onnxruntime_pybind11_state.cpython-*.so`,
/// `*.py` sources) since prunr links via Rust ort. Renames
/// `libonnxruntime.so.<ver>` to `libonnxruntime.so` so
/// `prunr_app::ort_runtime::resolve_dylib_path()` picks it up.
fn install_runtime() -> anyhow::Result<()> {
    let package = std::env::args().nth(2)
        .ok_or_else(|| anyhow::anyhow!("missing <package> arg, e.g. onnxruntime-openvino"))?;
    let version = std::env::args().nth(3)
        .ok_or_else(|| anyhow::anyhow!("missing <version> arg, e.g. 1.24.1"))?;
    let target_name = std::env::args().nth(4)
        .unwrap_or_else(|| install_target_name(&package, &version));

    println!("=== Phase 19-09: install-runtime ===");
    println!("Package:  {package} {version}");
    println!("Target:   <data>/prunr/runtimes/{target_name}/");

    let json_url = format!("https://pypi.org/pypi/{package}/{version}/json");
    println!("Querying {json_url}");
    let client = reqwest::blocking::Client::builder()
        .user_agent("prunr-xtask/0.1")
        .build()?;
    let metadata: serde_json::Value = client.get(&json_url).send()?.json()?;
    let urls = metadata["urls"].as_array()
        .ok_or_else(|| anyhow::anyhow!("PyPI metadata missing `urls`"))?;

    let (wheel_url, expected_sha) = pick_wheel_for_host(urls)?;
    println!("Selected: {wheel_url}");
    println!("SHA256:   {expected_sha}");

    println!("Downloading…");
    let bytes = client.get(wheel_url).send()?.bytes()?;
    verify_sha256(&bytes, expected_sha)?;
    println!("Verified ({:.1} MB)", bytes.len() as f64 / 1024.0 / 1024.0);

    let target_dir = prunr_models::data_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve user data dir"))?
        .join("runtimes")
        .join(&target_name);
    // Sanity-guard the wipe: if a future refactor makes `data_dir()`
    // return something unexpected, this keeps `remove_dir_all` from
    // nuking the wrong tree.
    if !target_dir.parent().is_some_and(|p| p.ends_with("runtimes")) {
        anyhow::bail!(
            "refusing to wipe non-runtimes path: {}", target_dir.display(),
        );
    }
    if target_dir.exists() {
        std::fs::remove_dir_all(&target_dir)?;
    }
    std::fs::create_dir_all(&target_dir)?;
    println!("Extracting to {}", target_dir.display());

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes))?;
    let mut extracted = 0u32;
    let mut bytes_written = 0u64;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let raw_name = entry.name().to_string();
        let Some(target_filename) = repackage_target_filename(&raw_name) else {
            continue;
        };
        let dest = target_dir.join(target_filename);
        let mut out = std::fs::File::create(&dest)?;
        let n = std::io::copy(&mut entry, &mut out)?;
        bytes_written += n;
        extracted += 1;
    }
    println!(
        "Extracted {extracted} files ({:.1} MB)",
        bytes_written as f64 / 1024.0 / 1024.0,
    );

    let dylib = target_dir.join("libonnxruntime.so");
    if !dylib.is_file() {
        anyhow::bail!(
            "libonnxruntime.so missing after extract — wheel layout may have changed"
        );
    }
    println!();
    println!("=== install-runtime DONE ===");
    println!("Try the doctor command to confirm pickup:");
    println!("  target/debug/prunr --doctor");
    println!("Then run any prunr command — `ort_runtime::resolve` will pick");
    println!("up the runtime store entry without needing ORT_DYLIB_PATH.");
    Ok(())
}

/// Default install dir name when none is supplied. Strips the
/// `onnxruntime-` prefix and tags with the host RID so multiple
/// installs (Linux x64 + Windows x64) coexist.
fn install_target_name(package: &str, version: &str) -> String {
    let short = package.strip_prefix("onnxruntime-").unwrap_or(package);
    let rid = host_rid();
    format!("{short}-{version}-{rid}")
}

fn host_rid() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) { "linux-x64" }
    else if cfg!(all(target_os = "linux", target_arch = "aarch64")) { "linux-arm64" }
    else if cfg!(all(target_os = "windows", target_arch = "x86_64")) { "windows-x64" }
    else if cfg!(all(target_os = "windows", target_arch = "aarch64")) { "windows-arm64" }
    else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") { "macos-arm64" } else { "macos-x64" }
    }
    else { "unknown" }
}

/// Pick the wheel matching the host platform. PyPI's `urls` array has
/// per-platform entries; we match on filename suffix. Prefer cp313
/// (newest published wheel) but accept any Python ABI tag — the
/// `libonnxruntime.so` we extract is interpreter-agnostic.
fn pick_wheel_for_host<'a>(urls: &'a [serde_json::Value]) -> anyhow::Result<(&'a str, &'a str)> {
    let host_token = match host_rid() {
        "linux-x64" => "manylinux_2_28_x86_64",
        "linux-arm64" => "manylinux_2_28_aarch64",
        "windows-x64" => "win_amd64",
        "windows-arm64" => "win_arm64",
        "macos-arm64" => "macosx_11_0_arm64",
        _ => anyhow::bail!("unsupported host platform: {}", host_rid()),
    };
    let pick = |require_cp313: bool| urls.iter().find_map(|u| {
        let name = u["filename"].as_str()?;
        if !name.contains(host_token) { return None; }
        if require_cp313 && !name.contains("cp313") { return None; }
        Some((u["url"].as_str()?, u["digests"]["sha256"].as_str()?))
    });
    pick(true).or_else(|| pick(false)).ok_or_else(|| anyhow::anyhow!(
        "no wheel found for host platform `{}`", host_rid(),
    ))
}

/// Filter + rename for files extracted from an `onnxruntime-*` wheel.
/// Returns `Some(<filename in our runtime dir>)` for files we want to
/// keep, `None` for everything we skip. Renames `libonnxruntime.so.<ver>`
/// to canonical `libonnxruntime.so` so our resolver picks it up;
/// versioned symlink siblings (`libopenvino.so.2025.4.1`) keep their
/// names since ELF dynamic linking resolves either form.
fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if stripped.contains('/') { return None; }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
    if stripped.ends_with(".py") { return None; }
    if stripped.starts_with("libonnxruntime.so.") {
        return Some("libonnxruntime.so".to_string());
    }
    Some(stripped.to_string())
}

fn verify_sha256(bytes: &[u8], expected: &str) -> anyhow::Result<()> {
    let actual = hex::encode(Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(expected) {
        anyhow::bail!("SHA256 mismatch:\n  expected: {expected}\n  got:      {actual}");
    }
    Ok(())
}
