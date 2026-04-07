mod cli;

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
            // Phase 4 will replace this with eframe::run_native(...)
            eprintln!("bgprunr: no subcommand given. Run `bgprunr remove --help` for CLI usage.");
            eprintln!("GUI mode is not yet implemented (Phase 4).");
            std::process::exit(0);
        }
    }
}
