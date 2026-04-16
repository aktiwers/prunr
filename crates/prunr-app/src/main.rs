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
        eprintln!("prunr: GUI error: {e}");
        std::process::exit(1);
    }
}
