use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod client;
mod commands;
mod config;
mod models;
mod path_map;

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = config::Config::load(cli.config.as_deref())?;

    match cli.command {
        Command::Search(args) => commands::search::run(&cfg, args),
    }
}
