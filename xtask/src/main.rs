fn main() -> anyhow::Result<()> {
    let task = std::env::args().nth(1).unwrap_or_default();
    match task.as_str() {
        "fetch-models" => {
            eprintln!("fetch-models: not yet implemented — see Plan 02");
            std::process::exit(1);
        }
        _ => {
            eprintln!("Usage: cargo xtask <task>");
            eprintln!("Tasks: fetch-models");
            std::process::exit(1);
        }
    }
}
