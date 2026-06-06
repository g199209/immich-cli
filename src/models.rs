use serde::{Deserialize, Serialize};

/// Request body for `/api/search/metadata` and `/api/search/smart`.
///
/// Only fields we actually populate are listed; everything else is left as
/// API defaults. `query` is required by smart search and ignored by metadata
/// search.
#[derive(Debug, Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Exact-match filter on `originalFileName`. Used by the `info`
    /// subcommand to locate an asset given its on-disk path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_file_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Substring filter on Immich's OCR-detected text. Both `/metadata`
    /// and `/smart` endpoints accept this; matching is case-sensitive
    /// substring and supports Unicode (Chinese, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr: Option<String>,

    /// Substring filter on the asset's description (EXIF). Available
    /// only on `/metadata`; used by the `ask` subcommand to fan out
    /// LLM-generated keywords against descriptions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub taken_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taken_before: Option<String>,

    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub asset_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub with_exif: Option<bool>,
}

/// Top-level response from search endpoints.
#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    pub assets: AssetsBucket,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetsBucket {
    #[allow(dead_code)]
    pub total: u32,
    #[allow(dead_code)]
    pub count: u32,
    pub items: Vec<Asset>,
    /// Page number to request next, as a string. `null` when finished.
    /// In some Immich versions this field is absent entirely.
    #[serde(default)]
    pub next_page: Option<serde_json::Value>,
}

/// Subset of asset fields we care about for the search command.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub id: String,
    pub original_path: String,
    pub original_file_name: String,
    #[serde(rename = "type")]
    pub asset_type: String,
    pub file_created_at: Option<String>,
    pub local_date_time: Option<String>,
    /// Base64-encoded checksum (SHA-1) of the file. Used by
    /// `update-descriptions` to detect when the underlying file has
    /// changed so previously-generated captions can be refreshed.
    #[serde(default)]
    pub checksum: String,
    #[serde(default)]
    pub exif_info: Option<ExifInfo>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExifInfo {
    pub city: Option<String>,
    pub state: Option<String>,
    pub country: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    /// Free-form text (EXIF UserComment / ImageDescription). The `ask`
    /// command reads this for LLM-mediated semantic search.
    #[serde(default)]
    pub description: Option<String>,
}
