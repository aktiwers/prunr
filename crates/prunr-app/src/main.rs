// On Windows, launching a `console` subsystem binary always pops an empty cmd
// window — bad UX for a GUI app. Build as a Windows subsystem app (no console)
// and attach to the parent console at runtime when the user invokes CLI mode
// from cmd/PowerShell, so CLI output still reaches their terminal.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod cli;
use prunr_app::gui;
mod worker_process;

use clap::Parser;
use cli::Cli;

#[cfg(windows)]
fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    unsafe { AttachConsole(ATTACH_PARENT_PROCESS) };
}

fn main() {
    let cli = Cli::parse();

    // `--debug` propagates to subprocess workers via two env vars, set BEFORE
    // `init_tracing` reads them. Workers inherit the parent env at spawn.
    //   RUST_LOG=prunr=debug  → filter level
    //   PRUNR_DEBUG_LOG=<path> → also tee tracing to a file (bulletproof on
    //                            Windows where AttachConsole + GUI-subsystem
    //                            subprocess stderr is unreliable).
    if cli.debug {
        // SAFETY: single-threaded at this point in main.
        if std::env::var_os("RUST_LOG").is_none() {
            unsafe { std::env::set_var("RUST_LOG", "prunr=debug"); }
        }
        if std::env::var_os("PRUNR_DEBUG_LOG").is_none() {
            let log_path = std::env::temp_dir().join("prunr-debug.log");
            unsafe { std::env::set_var("PRUNR_DEBUG_LOG", &log_path); }
        }
    }

    // Attach the parent console BEFORE init_tracing so the stderr writer
    // captures a valid handle. The previous order inverted these, causing
    // parent stderr to work by luck and worker stderr (via Stdio::inherit)
    // to silently drop on Windows windows_subsystem=windows builds.
    #[cfg(windows)]
    if cli.debug || !cli.inputs.is_empty() {
        attach_parent_console();
    }

    init_tracing();

    if cli.worker {
        worker_process::run_worker();
    }

    if !cli.inputs.is_empty() {
        let exit_code = cli::run_remove(&cli);
        std::process::exit(exit_code);
    }

    if cli.debug {
        let log = std::env::var("PRUNR_DEBUG_LOG").unwrap_or_default();
        tracing::info!(log_file = %log, "debug mode — tracing at prunr=debug, tee to file");
    }
    if let Err(e) = gui::run() {
        tracing::error!(%e, "GUI launch failed");
        std::process::exit(1);
    }
}

/// Initialize the global tracing subscriber.
///
/// Writes to stderr (for CLI use and Linux/macOS debug) and, when
/// `PRUNR_DEBUG_LOG` names a file, tees to that file too. The file route is
/// the reliable path on Windows: `windows_subsystem = "windows"` subprocess
/// stderr doesn't cleanly reach an AttachConsole'd parent, so we sidestep it.
fn init_tracing() {
    use std::sync::Mutex;
    use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("prunr=info"));

    let stderr_layer = fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr);

    let file_layer = std::env::var_os("PRUNR_DEBUG_LOG")
        .and_then(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
                .ok()
        })
        .map(|f| {
            fmt::layer()
                .with_target(false)
                .with_ansi(false)
                .with_writer(Mutex::new(f))
        });

    Registry::default()
        .with(filter)
        .with(stderr_layer)
        .with(file_layer)
        .init();
}
