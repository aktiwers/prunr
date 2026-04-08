mod cli;
mod gui;

use clap::Parser;
use cli::Cli;

fn main() {
    let cli = Cli::parse();

    if !cli.inputs.is_empty() {
        let exit_code = cli::run_remove(&cli);
        std::process::exit(exit_code);
    }

    // No args → launch GUI
    if let Err(e) = gui::run() {
        eprintln!("prunr: GUI error: {e}");
        std::process::exit(1);
    }
}
