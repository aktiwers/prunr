mod cli;
mod gui;

use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Remove(args)) => {
            let exit_code = cli::run_remove(args);
            std::process::exit(exit_code);
        }
        None => {
            if let Err(e) = gui::run() {
                eprintln!("bgprunr: GUI error: {e}");
                std::process::exit(1);
            }
        }
    }
}
