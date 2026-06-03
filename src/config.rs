use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server_url: String,
    pub api_key: String,
    /// Path mappings from Immich server paths to local NFS paths.
    /// Matched in order; the first prefix match wins.
    #[serde(default)]
    pub path_map: Vec<PathMapEntry>,
    /// Optional request timeout in seconds (default 60).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PathMapEntry {
    pub server: String,
    pub local: String,
}

fn default_timeout() -> u64 {
    60
}

impl Config {
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => default_config_path()?,
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file at {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text)
            .with_context(|| format!("failed to parse config file at {}", path.display()))?;

        cfg.server_url = cfg.server_url.trim_end_matches('/').to_string();
        if cfg.server_url.is_empty() {
            bail!("config.server_url is empty");
        }
        if cfg.api_key.is_empty() {
            bail!("config.api_key is empty");
        }

        // Expand `~` in path map entries.
        for entry in &mut cfg.path_map {
            entry.local = expand_tilde(&entry.local);
            entry.server = entry.server.trim_end_matches('/').to_string();
            entry.local = entry.local.trim_end_matches('/').to_string();
        }

        Ok(cfg)
    }
}

fn default_config_path() -> Result<PathBuf> {
    let dirs = directories::ProjectDirs::from("", "", "immich-cli")
        .context("could not resolve a config directory for this platform")?;
    Ok(dirs.config_dir().join("config.toml"))
}

fn expand_tilde(input: &str) -> String {
    if let Some(rest) = input.strip_prefix("~/") {
        if let Some(home) = directories::UserDirs::new().and_then(|d| Some(d.home_dir().to_path_buf())) {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    if input == "~" {
        if let Some(home) = directories::UserDirs::new().map(|d| d.home_dir().to_path_buf()) {
            return home.to_string_lossy().into_owned();
        }
    }
    input.to_string()
}
