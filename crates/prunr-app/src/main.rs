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
    // Parse CLI before tracing init so --debug can raise the filter level
    // and propagate to the subprocess worker via env.
    let cli = Cli::parse();

    // `--debug` propagates to the subprocess worker via RUST_LOG so its
    // tracing output matches the parent's. Set BEFORE init_tracing reads
    // the env (worker subprocess inherits env from parent).
    if cli.debug && std::env::var_os("RUST_LOG").is_none() {
        // SAFETY: single-threaded at this point in main, no other threads
        // reading env yet. The subprocess spawn inherits this env.
        unsafe { std::env::set_var("RUST_LOG", "prunr=debug"); }
    }

    init_tracing();

    // Internal subprocess worker mode (launched by GUI batch processing).
    if cli.worker {
        worker_process::run_worker();
    }

    if !cli.inputs.is_empty() {
        #[cfg(windows)]
        attach_parent_console();
        let exit_code = cli::run_remove(&cli);
        std::process::exit(exit_code);
    }

    // No args → launch GUI. With --debug on Windows, attach the parent
    // console so tracing stderr is visible in the launching terminal
    // (GUI subsystem has no console by default).
    #[cfg(windows)]
    if cli.debug {
        attach_parent_console();
    }
    if cli.debug {
        tracing::info!("debug mode active — GUI + subprocess tracing at prunr=debug");
    }
    if let Err(e) = gui::run() {
        tracing::error!(%e, "GUI launch failed");
        std::process::exit(1);
    }
}

/// Initialize the global tracing subscriber.
///
/// Writes to stderr so the subprocess worker's stdout stays clean for
/// bincode IPC framing. Default filter is `prunr=info`; override via
/// `RUST_LOG=prunr=debug` (or `--debug` which sets that env var).
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
