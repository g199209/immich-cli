use crate::client::{ImmichClient, SearchBackend};
use crate::config::{Config, PathMapEntry};
use crate::models::{Asset, SearchRequest};
use crate::path_map;
use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Smart-search query (CLIP). When set, the smart-search endpoint is used.
    #[arg(short, long)]
    pub query: Option<String>,

    /// Earliest `localDateTime` to include. ISO 8601, or YYYY-MM-DD (UTC start of day).
    #[arg(long, value_name = "DATE")]
    pub taken_after: Option<String>,

    /// Latest `localDateTime` to include. ISO 8601, or YYYY-MM-DD (UTC end of day).
    #[arg(long, value_name = "DATE")]
    pub taken_before: Option<String>,

    /// Filter by city as recorded in EXIF (Immich does an exact match).
    #[arg(long)]
    pub city: Option<String>,

    /// Filter by state/province as recorded in EXIF (exact match).
    #[arg(long)]
    pub state: Option<String>,

    /// Filter by country as recorded in EXIF (exact match).
    #[arg(long)]
    pub country: Option<String>,

    /// Restrict by asset type.
    #[arg(long, value_enum)]
    pub r#type: Option<AssetTypeArg>,

    /// Maximum results to return across all pages.
    #[arg(long, default_value_t = 250)]
    pub limit: u32,

    /// Page size to request from Immich (max 1000 in current API).
    #[arg(long, default_value_t = 250)]
    pub page_size: u32,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Paths)]
    pub format: OutputFormat,

    /// Verify each translated path exists on the local filesystem; missing
    /// files are reported on stderr and (unless `--include-missing`) skipped.
    #[arg(long)]
    pub verify: bool,

    /// When verifying, still emit lines for missing files (prefixed with `MISSING\t` in paths/table mode).
    #[arg(long, requires = "verify")]
    pub include_missing: bool,

    /// Include server-side paths that have no matching local mapping in the
    /// output (otherwise they are skipped with a stderr warning).
    #[arg(long)]
    pub include_unmapped: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum AssetTypeArg {
    Image,
    Video,
    Audio,
    Other,
}

impl AssetTypeArg {
    fn as_api_str(self) -> &'static str {
        match self {
            AssetTypeArg::Image => "IMAGE",
            AssetTypeArg::Video => "VIDEO",
            AssetTypeArg::Audio => "AUDIO",
            AssetTypeArg::Other => "OTHER",
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// One local path per line. Unmapped/missing assets reported on stderr.
    Paths,
    /// One JSON object per asset (newline-delimited).
    Json,
    /// Aligned table with id, type, taken date, location, local path.
    Table,
}

impl SearchArgs {
    /// Returns true if at least one user-facing filter is set. We reject
    /// "empty" searches because dumping the entire library at random is
    /// almost certainly not what the caller intended.
    pub fn has_filter(&self) -> bool {
        self.query.is_some()
            || self.taken_after.is_some()
            || self.taken_before.is_some()
            || self.city.is_some()
            || self.state.is_some()
            || self.country.is_some()
            || self.r#type.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        if self.limit == 0 {
            bail!("--limit must be > 0");
        }
        if self.page_size == 0 {
            bail!("--page-size must be > 0");
        }
        if !self.has_filter() {
            bail!(
                "search requires at least one filter: --query, --taken-after, \
                 --taken-before, --city, --state, --country, or --type"
            );
        }
        Ok(())
    }
}

pub fn run(cfg: &Config, args: SearchArgs) -> Result<()> {
    args.validate()?;

    let client = ImmichClient::new(cfg)?;
    let assets = fetch_assets(&client, &args)?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    emit_to_writer(&cfg.path_map, &args, &assets, &mut out)
}

pub fn fetch_assets<B: SearchBackend>(backend: &B, args: &SearchArgs) -> Result<Vec<Asset>> {
    let taken_after = args
        .taken_after
        .as_deref()
        .map(normalize_date_start)
        .transpose()?;
    let taken_before = args
        .taken_before
        .as_deref()
        .map(normalize_date_end)
        .transpose()?;

    let mut collected = Vec::with_capacity(args.limit as usize);
    let mut page: u32 = 1;
    let page_size = args.page_size.min(args.limit).max(1);

    loop {
        let req = SearchRequest {
            query: args.query.clone(),
            city: args.city.clone(),
            state: args.state.clone(),
            country: args.country.clone(),
            taken_after: taken_after.clone(),
            taken_before: taken_before.clone(),
            asset_type: args.r#type.map(|t| t.as_api_str().to_string()),
            page: Some(page),
            size: Some(page_size),
            with_exif: Some(true),
        };

        let resp = backend.search(&req)?;
        let count = resp.assets.items.len();
        let has_more = resp.assets.next_page.as_ref().is_some_and(|v| !v.is_null());

        for asset in resp.assets.items {
            collected.push(asset);
            if collected.len() as u32 >= args.limit {
                return Ok(collected);
            }
        }

        if !has_more || count == 0 {
            break;
        }
        page += 1;
    }
    Ok(collected)
}

/// Accept either a full ISO 8601 timestamp or a bare `YYYY-MM-DD`. For the
/// bare form, expand to UTC 00:00:00 (start of that day).
pub fn normalize_date_start(input: &str) -> Result<String> {
    normalize_date(input, false)
}

/// Bare `YYYY-MM-DD` becomes UTC 23:59:59.999 of that day (end of day).
pub fn normalize_date_end(input: &str) -> Result<String> {
    normalize_date(input, true)
}

fn normalize_date(input: &str, end_of_day: bool) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.len() == 10 && trimmed.chars().nth(4) == Some('-') {
        let date = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d")
            .with_context(|| format!("invalid date `{trimmed}`, expected YYYY-MM-DD"))?;
        let time = if end_of_day {
            chrono::NaiveTime::from_hms_milli_opt(23, 59, 59, 999).unwrap()
        } else {
            chrono::NaiveTime::MIN
        };
        let dt = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            date.and_time(time),
            chrono::Utc,
        );
        return Ok(dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true));
    }

    // Otherwise assume the caller already passed something the server can parse.
    Ok(trimmed.to_string())
}

pub fn emit_to_writer<W: std::io::Write>(
    path_map: &[PathMapEntry],
    args: &SearchArgs,
    assets: &[Asset],
    out: &mut W,
) -> Result<()> {
    let mut rows: Vec<Row> = Vec::with_capacity(assets.len());
    for asset in assets {
        let local = path_map::translate(&asset.original_path, path_map);
        let unmapped = local.is_none();
        if unmapped && !args.include_unmapped {
            eprintln!(
                "warn: no path mapping for {} (asset {})",
                asset.original_path, asset.id
            );
            continue;
        }
        let missing = if args.verify {
            local.as_ref().map(|p| !p.exists()).unwrap_or(false)
        } else {
            false
        };
        if missing {
            eprintln!(
                "warn: local file missing for asset {}: {}",
                asset.id,
                local.as_ref().unwrap().display()
            );
            if !args.include_missing {
                continue;
            }
        }
        rows.push(Row {
            asset,
            local,
            missing,
            unmapped,
        });
    }

    match args.format {
        OutputFormat::Paths => {
            for row in &rows {
                let display = row
                    .local
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| row.asset.original_path.clone());
                if row.missing {
                    writeln!(out, "MISSING\t{display}")?;
                } else if row.unmapped {
                    writeln!(out, "UNMAPPED\t{display}")?;
                } else {
                    writeln!(out, "{display}")?;
                }
            }
        }
        OutputFormat::Json => {
            for row in &rows {
                let obj = serde_json::json!({
                    "id": row.asset.id,
                    "type": row.asset.asset_type,
                    "originalPath": row.asset.original_path,
                    "originalFileName": row.asset.original_file_name,
                    "localPath": row.local.as_ref().map(|p| p.to_string_lossy().into_owned()),
                    "localDateTime": row.asset.local_date_time,
                    "city": row.asset.exif_info.as_ref().and_then(|e| e.city.clone()),
                    "state": row.asset.exif_info.as_ref().and_then(|e| e.state.clone()),
                    "country": row.asset.exif_info.as_ref().and_then(|e| e.country.clone()),
                    "latitude": row.asset.exif_info.as_ref().and_then(|e| e.latitude),
                    "longitude": row.asset.exif_info.as_ref().and_then(|e| e.longitude),
                    "unmapped": row.unmapped,
                    "missing": row.missing,
                });
                writeln!(out, "{}", serde_json::to_string(&obj)?)?;
            }
        }
        OutputFormat::Table => {
            write_table(out, &rows)?;
        }
    }

    Ok(())
}

struct Row<'a> {
    asset: &'a Asset,
    local: Option<PathBuf>,
    missing: bool,
    unmapped: bool,
}

fn write_table(out: &mut impl std::io::Write, rows: &[Row<'_>]) -> Result<()> {
    let headers = ["TYPE", "TAKEN", "LOCATION", "PATH"];
    let mut widths = [headers[0].len(), headers[1].len(), headers[2].len(), 0];
    let mut data: Vec<[String; 4]> = Vec::with_capacity(rows.len());
    for row in rows {
        let taken = row
            .asset
            .local_date_time
            .as_deref()
            .map(|s| s.split('T').next().unwrap_or(s).to_string())
            .unwrap_or_default();
        let location = row
            .asset
            .exif_info
            .as_ref()
            .map(|e| {
                let parts: Vec<&str> = [&e.city, &e.state, &e.country]
                    .into_iter()
                    .filter_map(|x| x.as_deref())
                    .filter(|s| !s.is_empty())
                    .collect();
                parts.join(", ")
            })
            .unwrap_or_default();
        let path = row
            .local
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("(unmapped) {}", row.asset.original_path));
        let row_strs = [row.asset.asset_type.clone(), taken, location, path];
        for (i, s) in row_strs.iter().enumerate() {
            widths[i] = widths[i].max(s.chars().count());
        }
        data.push(row_strs);
    }
    writeln!(
        out,
        "{:<w0$}  {:<w1$}  {:<w2$}  {}",
        headers[0],
        headers[1],
        headers[2],
        headers[3],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
    )?;
    for r in data {
        writeln!(
            out,
            "{:<w0$}  {:<w1$}  {:<w2$}  {}",
            r[0],
            r[1],
            r[2],
            r[3],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AssetsBucket, ExifInfo, SearchResponse};
    use std::cell::RefCell;

    fn make_asset(id: &str, path: &str, asset_type: &str, taken: &str) -> Asset {
        Asset {
            id: id.into(),
            original_path: path.into(),
            original_file_name: path.rsplit('/').next().unwrap_or(path).into(),
            asset_type: asset_type.into(),
            file_created_at: Some(taken.into()),
            local_date_time: Some(taken.into()),
            exif_info: Some(ExifInfo {
                city: Some("Shanghai".into()),
                state: Some("Shanghai".into()),
                country: Some("China".into()),
                latitude: Some(31.0),
                longitude: Some(121.0),
            }),
        }
    }

    fn default_args() -> SearchArgs {
        SearchArgs {
            query: None,
            taken_after: None,
            taken_before: None,
            city: None,
            state: None,
            country: None,
            r#type: None,
            limit: 250,
            page_size: 250,
            format: OutputFormat::Paths,
            verify: false,
            include_missing: false,
            include_unmapped: false,
        }
    }

    /// Records each call and replays canned responses in order. Optional
    /// `assert_fn` lets a test verify the request body the backend sees.
    struct FakeBackend {
        responses: RefCell<Vec<SearchResponse>>,
        calls: RefCell<Vec<SearchRequest>>,
    }

    impl FakeBackend {
        fn new(responses: Vec<SearchResponse>) -> Self {
            Self {
                responses: RefCell::new(responses),
                calls: RefCell::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<SearchRequest> {
            self.calls.borrow().clone()
        }
    }

    impl SearchBackend for FakeBackend {
        fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
            self.calls.borrow_mut().push(SearchRequest {
                query: req.query.clone(),
                city: req.city.clone(),
                state: req.state.clone(),
                country: req.country.clone(),
                taken_after: req.taken_after.clone(),
                taken_before: req.taken_before.clone(),
                asset_type: req.asset_type.clone(),
                page: req.page,
                size: req.size,
                with_exif: req.with_exif,
            });
            let mut q = self.responses.borrow_mut();
            if q.is_empty() {
                anyhow::bail!("FakeBackend ran out of canned responses");
            }
            Ok(q.remove(0))
        }
    }

    fn resp(items: Vec<Asset>, next: Option<&str>) -> SearchResponse {
        let total = items.len() as u32;
        SearchResponse {
            assets: AssetsBucket {
                total,
                count: total,
                items,
                next_page: next.map(|s| serde_json::Value::String(s.into())),
            },
        }
    }

    // ---- Args validation -----------------------------------------------

    #[test]
    fn validate_rejects_empty_filter() {
        let args = default_args();
        let err = args.validate().unwrap_err().to_string();
        assert!(err.contains("at least one filter"), "got: {err}");
    }

    #[test]
    fn validate_accepts_query_only() {
        let mut args = default_args();
        args.query = Some("x".into());
        assert!(args.validate().is_ok());
    }

    #[test]
    fn validate_accepts_time_only() {
        let mut args = default_args();
        args.taken_after = Some("2025-01-01".into());
        assert!(args.validate().is_ok());
    }

    #[test]
    fn validate_accepts_type_only() {
        let mut args = default_args();
        args.r#type = Some(AssetTypeArg::Video);
        assert!(args.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_limit() {
        let mut args = default_args();
        args.query = Some("x".into());
        args.limit = 0;
        assert!(args.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_page_size() {
        let mut args = default_args();
        args.query = Some("x".into());
        args.page_size = 0;
        assert!(args.validate().is_err());
    }

    // ---- Date normalization --------------------------------------------

    #[test]
    fn date_start_expands_to_utc_midnight() {
        let got = normalize_date_start("2025-03-04").unwrap();
        assert_eq!(got, "2025-03-04T00:00:00.000Z");
    }

    #[test]
    fn date_end_expands_to_utc_eod() {
        let got = normalize_date_end("2025-03-04").unwrap();
        assert_eq!(got, "2025-03-04T23:59:59.999Z");
    }

    #[test]
    fn date_iso_passthrough() {
        let got = normalize_date_start("2025-03-04T12:34:56Z").unwrap();
        assert_eq!(got, "2025-03-04T12:34:56Z");
    }

    #[test]
    fn date_invalid_yyyymmdd_rejected() {
        let err = normalize_date_start("2025-13-99").unwrap_err().to_string();
        assert!(err.contains("invalid date"), "got: {err}");
    }

    // ---- fetch_assets --------------------------------------------------

    #[test]
    fn fetch_walks_pages_until_next_is_null() {
        let backend = FakeBackend::new(vec![
            resp(
                vec![
                    make_asset("a1", "/mnt/x/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                    make_asset("a2", "/mnt/x/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
                ],
                Some("2"),
            ),
            resp(
                vec![make_asset("a3", "/mnt/x/c.jpg", "IMAGE", "2025-01-03T00:00:00Z")],
                None,
            ),
        ]);
        let mut args = default_args();
        args.query = Some("anything".into());
        args.page_size = 2;
        args.limit = 10;

        let assets = fetch_assets(&backend, &args).unwrap();
        assert_eq!(assets.len(), 3);
        let calls = backend.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].page, Some(1));
        assert_eq!(calls[1].page, Some(2));
        // The query is propagated as-is, and with_exif is forced on so the
        // table/json output can show location.
        assert_eq!(calls[0].query.as_deref(), Some("anything"));
        assert_eq!(calls[0].with_exif, Some(true));
    }

    #[test]
    fn fetch_stops_at_limit_even_if_more_available() {
        let backend = FakeBackend::new(vec![resp(
            vec![
                make_asset("a1", "/mnt/x/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                make_asset("a2", "/mnt/x/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
                make_asset("a3", "/mnt/x/c.jpg", "IMAGE", "2025-01-03T00:00:00Z"),
            ],
            Some("2"),
        )]);
        let mut args = default_args();
        args.country = Some("China".into());
        args.page_size = 10;
        args.limit = 2;

        let assets = fetch_assets(&backend, &args).unwrap();
        assert_eq!(assets.len(), 2);
        // Only the first page should have been requested.
        assert_eq!(backend.calls().len(), 1);
    }

    #[test]
    fn fetch_sends_geo_and_type_filters_through() {
        let backend = FakeBackend::new(vec![resp(vec![], None)]);
        let mut args = default_args();
        args.city = Some("Kangqiao".into());
        args.state = Some("Shanghai".into());
        args.country = Some("China".into());
        args.r#type = Some(AssetTypeArg::Video);
        args.taken_after = Some("2025-01-01".into());
        args.taken_before = Some("2025-12-31".into());

        fetch_assets(&backend, &args).unwrap();
        let calls = backend.calls();
        assert_eq!(calls[0].city.as_deref(), Some("Kangqiao"));
        assert_eq!(calls[0].state.as_deref(), Some("Shanghai"));
        assert_eq!(calls[0].country.as_deref(), Some("China"));
        assert_eq!(calls[0].asset_type.as_deref(), Some("VIDEO"));
        assert_eq!(
            calls[0].taken_after.as_deref(),
            Some("2025-01-01T00:00:00.000Z")
        );
        assert_eq!(
            calls[0].taken_before.as_deref(),
            Some("2025-12-31T23:59:59.999Z")
        );
    }

    #[test]
    fn fetch_propagates_backend_errors() {
        struct ErrBackend;
        impl SearchBackend for ErrBackend {
            fn search(&self, _req: &SearchRequest) -> Result<SearchResponse> {
                anyhow::bail!("immich exploded")
            }
        }
        let mut args = default_args();
        args.query = Some("x".into());
        let err = fetch_assets(&ErrBackend, &args).unwrap_err().to_string();
        assert!(err.contains("immich exploded"), "got: {err}");
    }

    // ---- emit_to_writer ------------------------------------------------

    fn pmap() -> Vec<PathMapEntry> {
        vec![PathMapEntry {
            server: "/mnt/qnap".into(),
            local: "/home/u/Photos".into(),
        }]
    }

    #[test]
    fn emit_paths_default_skips_unmapped_with_warning() {
        let assets = vec![
            make_asset("a1", "/mnt/qnap/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
            make_asset("a2", "/other/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
        ];
        let mut args = default_args();
        args.query = Some("x".into());
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "/home/u/Photos/a.jpg");
    }

    #[test]
    fn emit_paths_include_unmapped_marks_them() {
        let assets = vec![make_asset(
            "a2",
            "/other/b.jpg",
            "IMAGE",
            "2025-01-02T00:00:00Z",
        )];
        let mut args = default_args();
        args.query = Some("x".into());
        args.include_unmapped = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "UNMAPPED\t/other/b.jpg");
    }

    #[test]
    fn emit_paths_verify_skips_missing_files() {
        let assets = vec![make_asset(
            "a1",
            "/mnt/qnap/nope.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )];
        let mut args = default_args();
        args.query = Some("x".into());
        args.verify = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        assert!(String::from_utf8(buf).unwrap().is_empty());
    }

    #[test]
    fn emit_paths_verify_with_include_missing_emits_marker() {
        let assets = vec![make_asset(
            "a1",
            "/mnt/qnap/nope.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )];
        let mut args = default_args();
        args.query = Some("x".into());
        args.verify = true;
        args.include_missing = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "MISSING\t/home/u/Photos/nope.jpg");
    }

    #[test]
    fn emit_json_is_ndjson() {
        let assets = vec![
            make_asset("a1", "/mnt/qnap/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
            make_asset("a2", "/mnt/qnap/b.jpg", "VIDEO", "2025-01-02T00:00:00Z"),
        ];
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Json;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["id"], "a1");
        assert_eq!(parsed["type"], "IMAGE");
        assert_eq!(parsed["localPath"], "/home/u/Photos/a.jpg");
        assert_eq!(parsed["country"], "China");
        assert_eq!(parsed["unmapped"], false);
        assert_eq!(parsed["missing"], false);
        let parsed2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(parsed2["type"], "VIDEO");
    }

    #[test]
    fn emit_table_has_header_and_aligned_columns() {
        let assets = vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )];
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Table;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &assets, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("TYPE"));
        assert!(lines[0].contains("TAKEN"));
        assert!(lines[0].contains("LOCATION"));
        assert!(lines[0].contains("PATH"));
        assert!(lines[1].contains("IMAGE"));
        assert!(lines[1].contains("2025-01-01"));
        assert!(lines[1].contains("Shanghai, Shanghai, China"));
        assert!(lines[1].contains("/home/u/Photos/a.jpg"));
    }
}

