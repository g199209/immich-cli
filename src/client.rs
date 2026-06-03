use crate::config::Config;
use crate::models::{SearchRequest, SearchResponse};
use anyhow::{bail, Context, Result};
use std::time::Duration;

/// Backend the search command talks to. The trait exists so tests can swap
/// in a fake without spinning up the real HTTP client.
pub trait SearchBackend {
    fn search(&self, req: &SearchRequest) -> Result<SearchResponse>;
}

pub struct ImmichClient {
    base_url: String,
    http: reqwest::blocking::Client,
}

impl ImmichClient {
    pub fn new(cfg: &Config) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut key = reqwest::header::HeaderValue::from_str(&cfg.api_key)
            .context("api_key contains invalid HTTP header bytes")?;
        key.set_sensitive(true);
        headers.insert("x-api-key", key);
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let http = reqwest::blocking::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            base_url: cfg.server_url.clone(),
            http,
        })
    }

    pub fn with_base_url(base_url: String, api_key: &str, timeout_secs: u64) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut key = reqwest::header::HeaderValue::from_str(api_key)
            .context("api_key contains invalid HTTP header bytes")?;
        key.set_sensitive(true);
        headers.insert("x-api-key", key);
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        let http = reqwest::blocking::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        })
    }
}

impl SearchBackend for ImmichClient {
    /// Run a search against either `/api/search/metadata` or `/api/search/smart`,
    /// depending on whether the request includes a smart query.
    fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
        let endpoint = if req.query.is_some() {
            "/api/search/smart"
        } else {
            "/api/search/metadata"
        };
        let url = format!("{}{}", self.base_url, endpoint);

        let resp = self
            .http
            .post(&url)
            .json(req)
            .send()
            .with_context(|| format!("HTTP request to {url} failed"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            bail!("Immich returned {status}: {body}");
        }

        let parsed: SearchResponse = resp
            .json()
            .with_context(|| format!("failed to decode response from {url}"))?;
        Ok(parsed)
    }
}
