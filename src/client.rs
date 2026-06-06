use crate::config::Config;
use crate::models::{SearchRequest, SearchResponse};
use crate::places::CityVocabEntry;
use anyhow::{anyhow, bail, Context, Result};
use std::time::Duration;

/// Backend the search command talks to. The trait exists so tests can swap
/// in a fake without spinning up the real HTTP client.
pub trait SearchBackend {
    fn search(&self, req: &SearchRequest) -> Result<SearchResponse>;
}

/// Thumbnail/write half of the `update-descriptions` backend. The command
/// also uses [`InfoBackend`] when configured people mappings require full
/// asset detail.
pub trait CaptionBackend {
    /// `GET /api/assets/{id}/thumbnail?size=thumbnail` — returns a small
    /// JPEG (~720x960) ready to send to a vision model.
    fn thumbnail(&self, id: &str) -> Result<Vec<u8>>;

    /// `PUT /api/assets/{id}` with `{"description": ...}`. Requires the
    /// API token to have the `asset.update` permission scope; without it
    /// the server returns 403.
    fn update_description(&self, id: &str, description: &str) -> Result<()>;
}

/// Backend exposing the library's geocoded vocabulary, used to resolve
/// the user's free-form `--place "..."` input to Immich's exact-match
/// city/state/country values. The endpoint `/api/search/cities` returns
/// one asset per distinct (city, state, country) tuple in the library —
/// effectively a free enumeration of every place name Immich knows
/// about, without us having to walk the whole asset table.
pub trait PlacesBackend {
    fn cities_vocabulary(&self) -> Result<Vec<CityVocabEntry>>;
}

/// Backend the `info` subcommand talks to. Separate from `SearchBackend`
/// because info needs raw asset+album JSON and would otherwise force every
/// fake search backend in the suite to implement these too.
pub trait InfoBackend {
    /// `GET /api/assets/{id}` — full asset detail, including EXIF, people,
    /// faces, tags, stack, duplicate id, etc. Returned as raw JSON so we
    /// stay forward-compatible with new Immich fields.
    fn get_asset(&self, id: &str) -> Result<serde_json::Value>;

    /// `GET /api/albums?assetId={id}` — list of albums the asset belongs to.
    fn albums_for_asset(&self, id: &str) -> Result<serde_json::Value>;

    /// `GET /api/assets/{id}/ocr` — list of OCR text regions. Each entry has
    /// the recognized `text`, a normalized 4-corner box (x1,y1 .. x4,y4),
    /// per-box and per-text confidence scores, and an `isVisible` flag.
    /// Returns an empty array when the asset has no detected text or when
    /// OCR is disabled server-side.
    fn ocr_for_asset(&self, id: &str) -> Result<serde_json::Value>;
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

impl InfoBackend for ImmichClient {
    fn get_asset(&self, id: &str) -> Result<serde_json::Value> {
        self.get_json(&format!("/api/assets/{id}"))
    }

    fn albums_for_asset(&self, id: &str) -> Result<serde_json::Value> {
        // assetId is passed as a query parameter; reqwest handles encoding.
        let url = format!("{}/api/albums", self.base_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("assetId", id)])
            .send()
            .with_context(|| format!("HTTP GET {url}?assetId={id} failed"))?;
        unpack_json(resp, &url)
    }

    fn ocr_for_asset(&self, id: &str) -> Result<serde_json::Value> {
        self.get_json(&format!("/api/assets/{id}/ocr"))
    }
}

impl PlacesBackend for ImmichClient {
    fn cities_vocabulary(&self) -> Result<Vec<CityVocabEntry>> {
        let raw = self.get_json("/api/search/cities")?;
        let arr = raw
            .as_array()
            .ok_or_else(|| anyhow!("/api/search/cities did not return an array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for asset in arr {
            let exif = &asset["exifInfo"];
            let city = exif["city"].as_str().unwrap_or("");
            let state = exif["state"].as_str().unwrap_or("");
            let country = exif["country"].as_str().unwrap_or("");
            // Immich's filter is exact-match on all three fields; an
            // entry with any field empty cannot be matched, so drop it.
            if city.is_empty() || state.is_empty() || country.is_empty() {
                continue;
            }
            out.push(CityVocabEntry {
                country: country.into(),
                state: state.into(),
                city: city.into(),
                admin2: None,
            });
        }
        Ok(out)
    }
}

impl ImmichClient {
    fn get_json(&self, path: &str) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .get(&url)
            .send()
            .with_context(|| format!("HTTP GET {url} failed"))?;
        unpack_json(resp, &url)
    }
}

impl CaptionBackend for ImmichClient {
    fn thumbnail(&self, id: &str) -> Result<Vec<u8>> {
        let url = format!("{}/api/assets/{id}/thumbnail", self.base_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("size", "thumbnail")])
            .send()
            .with_context(|| format!("HTTP GET {url}?size=thumbnail failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            bail!("Immich returned {status} for {url}: {body}");
        }
        resp.bytes()
            .map(|b| b.to_vec())
            .with_context(|| format!("failed to read thumbnail bytes from {url}"))
    }

    fn update_description(&self, id: &str, description: &str) -> Result<()> {
        let url = format!("{}/api/assets/{id}", self.base_url);
        let body = serde_json::json!({ "description": description });
        let resp = self
            .http
            .put(&url)
            .json(&body)
            .send()
            .with_context(|| format!("HTTP PUT {url} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            bail!("Immich returned {status} for PUT {url}: {body}");
        }
        Ok(())
    }
}

fn unpack_json(resp: reqwest::blocking::Response, url: &str) -> Result<serde_json::Value> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().unwrap_or_default();
        bail!("Immich returned {status} for {url}: {body}");
    }
    resp.json()
        .with_context(|| format!("failed to decode JSON response from {url}"))
}
