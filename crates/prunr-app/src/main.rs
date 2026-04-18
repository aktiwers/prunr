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
    init_tracing();

    let cli = Cli::parse();

    // Internal subprocess worker mode (launched by GUI batch processing)
    if cli.worker {
        worker_process::run_worker();
    }

    if !cli.inputs.is_empty() {
        #[cfg(windows)]
        attach_parent_console();
        let exit_code = cli::run_remove(&cli);
        std::process::exit(exit_code);
    }

    // No args → launch GUI
    if let Err(e) = gui::run() {
        tracing::error!(%e, "GUI launch failed");
        std::process::exit(1);
    }
}

/// Initialize the global tracing subscriber.
///
/// Writes to stderr so the subprocess worker's stdout stays clean for
/// bincode IPC framing. Default filter is `prunr=info`; override via
/// `RUST_LOG=prunr=debug` etc. Per-crate filters (`prunr_app=debug`,
/// `prunr_core=warn`) also work.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("prunr=info"));
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}
