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
    /// OpenAI-compatible LLM endpoint used by `ask`. Optional: only the
    /// `ask` subcommand needs this; everything else ignores it.
    #[serde(default)]
    pub llm: Option<LlmConfig>,
}

/// LLM endpoint config. Currently expected to speak the OpenAI
/// `/v1/chat/completions` protocol — works with OneAPI, LiteLLM, Ollama
/// (via its OpenAI-compatible layer), OpenAI itself, and similar.
#[derive(Debug, Deserialize, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: String,
    /// Text-only model used by the `ask` subcommand for keyword
    /// expansion and rerank.
    pub model: String,
    /// Vision-capable model used by `update-descriptions` for captioning.
    /// Optional: only required when running that subcommand.
    #[serde(default)]
    pub vision_model: Option<String>,
    /// Per-call request timeout. Reranking long descriptions and
    /// generating captions both take time; default is generous on
    /// purpose.
    #[serde(default = "default_llm_timeout")]
    pub timeout_secs: u64,
}

fn default_llm_timeout() -> u64 {
    120
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

        if let Some(llm) = cfg.llm.as_mut() {
            llm.base_url = llm.base_url.trim_end_matches('/').to_string();
            if llm.base_url.is_empty() {
                bail!("config.llm.base_url is empty");
            }
            if llm.api_key.is_empty() {
                bail!("config.llm.api_key is empty");
            }
            if llm.model.is_empty() {
                bail!("config.llm.model is empty");
            }
            if let Some(vm) = llm.vision_model.as_ref() {
                if vm.is_empty() {
                    bail!("config.llm.vision_model is empty (omit the field if you don't want to set it)");
                }
            }
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
        if let Some(home) = directories::UserDirs::new().map(|d| d.home_dir().to_path_buf()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(text: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(text.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parses_minimum_config() {
        let f = write_config(
            r#"
server_url = "http://example.com:2283/"
api_key = "abc"
"#,
        );
        let cfg = Config::load(Some(f.path())).unwrap();
        // Trailing slash is stripped so URL joins are predictable.
        assert_eq!(cfg.server_url, "http://example.com:2283");
        assert_eq!(cfg.api_key, "abc");
        assert!(cfg.path_map.is_empty());
        assert_eq!(cfg.timeout_secs, 60);
    }

    #[test]
    fn parses_path_map_and_strips_trailing_slashes() {
        let f = write_config(
            r#"
server_url = "http://x"
api_key = "k"
timeout_secs = 5

[[path_map]]
server = "/mnt/a/"
local  = "/home/u/A/"

[[path_map]]
server = "/mnt/b"
local  = "/home/u/B"
"#,
        );
        let cfg = Config::load(Some(f.path())).unwrap();
        assert_eq!(cfg.timeout_secs, 5);
        assert_eq!(cfg.path_map.len(), 2);
        assert_eq!(cfg.path_map[0].server, "/mnt/a");
        assert_eq!(cfg.path_map[0].local, "/home/u/A");
        assert_eq!(cfg.path_map[1].server, "/mnt/b");
        assert_eq!(cfg.path_map[1].local, "/home/u/B");
    }

    #[test]
    fn tilde_in_local_path_is_expanded() {
        let f = write_config(
            r#"
server_url = "http://x"
api_key = "k"
[[path_map]]
server = "/mnt/a"
local  = "~/Pictures"
"#,
        );
        let cfg = Config::load(Some(f.path())).unwrap();
        let home = directories::UserDirs::new()
            .unwrap()
            .home_dir()
            .to_string_lossy()
            .into_owned();
        assert_eq!(cfg.path_map[0].local, format!("{home}/Pictures"));
    }

    #[test]
    fn rejects_empty_server_url() {
        let f = write_config(
            r#"
server_url = ""
api_key = "k"
"#,
        );
        let err = Config::load(Some(f.path())).unwrap_err().to_string();
        assert!(err.contains("server_url"), "got: {err}");
    }

    #[test]
    fn rejects_empty_api_key() {
        let f = write_config(
            r#"
server_url = "http://x"
api_key = ""
"#,
        );
        let err = Config::load(Some(f.path())).unwrap_err().to_string();
        assert!(err.contains("api_key"), "got: {err}");
    }

    #[test]
    fn missing_file_reports_path() {
        let err = Config::load(Some(std::path::Path::new("/no/such/cfg.toml")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("/no/such/cfg.toml"), "got: {err}");
    }

    #[test]
    fn rejects_unparseable_toml() {
        let f = write_config("server_url = 'http://x'\napi_key = 'k\n");
        assert!(Config::load(Some(f.path())).is_err());
    }
}
