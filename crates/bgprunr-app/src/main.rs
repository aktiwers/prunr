mod cli;
mod gui;

use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let mut cli = Cli::parse();

    // `bgprunr remove file.jpg` — merge subcommand inputs into top-level
    if let Some(Commands::Remove(sub)) = cli.command.take() {
        cli.inputs = sub.inputs;
    }

    // Any inputs → process images
    if !cli.inputs.is_empty() {
        let exit_code = cli::run_remove(&cli);
        std::process::exit(exit_code);
    }

    // No args → launch GUI
    if let Err(e) = gui::run() {
        eprintln!("bgprunr: GUI error: {e}");
        std::process::exit(1);
    }
}
