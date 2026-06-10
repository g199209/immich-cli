use anyhow::Result;
use clap::{Parser, Subcommand};
use immich_cli::{commands, config};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "immich-cli",
    version,
    about = "Custom Immich CLI that resolves search hits to local NFS paths"
)]
struct Cli {
    /// Path to the config file. Defaults to ~/.config/immich-cli/config.toml
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Search assets by smart query, time range, and/or geo location
    Search(commands::search::SearchArgs),
    /// Show all available metadata for a single photo, given its local NFS path
    Info(commands::info::InfoArgs),
    /// Generate (or refresh) photo descriptions via a vision LLM, idempotently
    UpdateDescriptions(commands::update_descriptions::UpdateDescriptionsArgs),
    /// Collapse near-duplicate groups into stacks (vision-model assisted)
    Dedup(commands::dedup::DedupArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = config::Config::load(cli.config.as_deref())?;

    match cli.command {
        Command::Search(args) => commands::search::run(&cfg, args),
        Command::Info(args) => commands::info::run(&cfg, args),
        Command::UpdateDescriptions(args) => commands::update_descriptions::run(&cfg, args),
        Command::Dedup(args) => commands::dedup::run(&cfg, args),
    }
}
