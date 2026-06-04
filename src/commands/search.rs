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

    /// Maximum results to return across all pages. When the server has more
    /// matches than this, the output ends with a `......` marker.
    #[arg(long, default_value_t = 1000)]
    pub limit: u32,

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
    ///
    /// A whitespace-only string filter (e.g. `--query ""`, `--country " "`)
    /// is treated as if the flag was not passed at all — otherwise the
    /// caller could trivially bypass the "must filter" guard with an empty
    /// flag and dump the whole library.
    pub fn has_filter(&self) -> bool {
        non_blank(&self.query)
            || non_blank(&self.taken_after)
            || non_blank(&self.taken_before)
            || non_blank(&self.city)
            || non_blank(&self.state)
            || non_blank(&self.country)
            || self.r#type.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        if self.limit == 0 {
            bail!("--limit must be > 0");
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

/// `true` only when the option holds a non-empty, non-whitespace string.
/// Empty/whitespace clones of "set" are not real filters and must not count
/// toward the "at least one filter" requirement.
fn non_blank(opt: &Option<String>) -> bool {
    opt.as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

/// Same idea as `non_blank`, but returns the trimmed string for sending to
/// Immich, or `None` if the input is missing/blank. Use this when building
/// the API request so we never send `"city": ""` over the wire.
fn cleaned(opt: &Option<String>) -> Option<String> {
    opt.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

pub fn run(cfg: &Config, args: SearchArgs) -> Result<()> {
    args.validate()?;

    let client = ImmichClient::new(cfg)?;
    let result = fetch_assets(&client, &args)?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    emit_to_writer(&cfg.path_map, &args, &result, &mut out)
}

/// Outcome of a paginated fetch: the items we kept, plus whether the
/// server still had more matches that we didn't retrieve.
#[derive(Debug)]
pub struct FetchResult {
    pub assets: Vec<Asset>,
    pub truncated: bool,
}

/// Per-request page size for Immich's search endpoints. 1000 is the API's
/// documented maximum, so it's also the fewest round-trips we can make for
/// the default --limit of 1000.
const PAGE_SIZE: u32 = 1000;

pub fn fetch_assets<B: SearchBackend>(backend: &B, args: &SearchArgs) -> Result<FetchResult> {
    fetch_assets_inner(backend, args, PAGE_SIZE)
}

fn fetch_assets_inner<B: SearchBackend>(
    backend: &B,
    args: &SearchArgs,
    page_size: u32,
) -> Result<FetchResult> {
    let taken_after = cleaned(&args.taken_after)
        .as_deref()
        .map(normalize_date_start)
        .transpose()?;
    let taken_before = cleaned(&args.taken_before)
        .as_deref()
        .map(normalize_date_end)
        .transpose()?;
    let query = cleaned(&args.query);
    let city = cleaned(&args.city);
    let state = cleaned(&args.state);
    let country = cleaned(&args.country);

    let mut collected: Vec<Asset> = Vec::with_capacity(args.limit as usize);
    let mut page: u32 = 1;
    let page_size = page_size.min(args.limit).max(1);

    loop {
        let req = SearchRequest {
            query: query.clone(),
            city: city.clone(),
            state: state.clone(),
            country: country.clone(),
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

        let remaining = (args.limit as usize).saturating_sub(collected.len());
        let take = count.min(remaining);
        // Anything not consumed in the current page, plus a known next page,
        // both mean there's more we're not fetching.
        let leftover_in_page = count > take;
        collected.extend(resp.assets.items.into_iter().take(take));

        if collected.len() as u32 >= args.limit {
            return Ok(FetchResult {
                assets: collected,
                truncated: leftover_in_page || has_more,
            });
        }

        if !has_more || count == 0 {
            break;
        }
        page += 1;
    }
    Ok(FetchResult {
        assets: collected,
        truncated: false,
    })
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
    if trimmed.is_empty() {
        bail!("date filter is empty");
    }
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
    result: &FetchResult,
    out: &mut W,
) -> Result<()> {
    let assets = &result.assets;
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

    if result.truncated {
        // Signal that the server still had more matches than --limit allowed
        // through. NDJSON output stays parseable by using a structured marker.
        match args.format {
            OutputFormat::Json => writeln!(out, "{{\"truncated\":true}}")?,
            OutputFormat::Paths | OutputFormat::Table => writeln!(out, "......")?,
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
            limit: 1000,
            format: OutputFormat::Paths,
            verify: false,
            include_missing: false,
            include_unmapped: false,
        }
    }

    fn fr(assets: Vec<Asset>) -> FetchResult {
        FetchResult {
            assets,
            truncated: false,
        }
    }

    fn fr_truncated(assets: Vec<Asset>) -> FetchResult {
        FetchResult {
            assets,
            truncated: true,
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
    fn validate_rejects_empty_string_query() {
        // -q "" must not slip past the "at least one filter" guard.
        let mut args = default_args();
        args.query = Some(String::new());
        let err = args.validate().unwrap_err().to_string();
        assert!(err.contains("at least one filter"), "got: {err}");
    }

    #[test]
    fn validate_rejects_whitespace_only_country() {
        let mut args = default_args();
        args.country = Some("   ".into());
        let err = args.validate().unwrap_err().to_string();
        assert!(err.contains("at least one filter"), "got: {err}");
    }

    #[test]
    fn validate_rejects_whitespace_only_date() {
        let mut args = default_args();
        args.taken_after = Some(" ".into());
        let err = args.validate().unwrap_err().to_string();
        assert!(err.contains("at least one filter"), "got: {err}");
    }

    #[test]
    fn blank_strings_are_not_sent_over_the_wire() {
        // If a real filter is set, any *other* string flag that happens to
        // be blank must be stripped from the request body — never sent as
        // an empty match Immich would interpret literally.
        let backend = FakeBackend::new(vec![resp(vec![], None)]);
        let mut args = default_args();
        args.country = Some("China".into());
        args.city = Some("   ".into());
        args.query = Some("".into());
        fetch_assets(&backend, &args).unwrap();
        let req = &backend.calls()[0];
        assert_eq!(req.country.as_deref(), Some("China"));
        assert!(req.city.is_none(), "blank city should be stripped");
        assert!(req.query.is_none(), "blank query should be stripped");
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
                vec![make_asset(
                    "a3",
                    "/mnt/x/c.jpg",
                    "IMAGE",
                    "2025-01-03T00:00:00Z",
                )],
                None,
            ),
        ]);
        let mut args = default_args();
        args.query = Some("anything".into());
        args.limit = 10;

        // page_size is hard-coded in fetch_assets, so drive the inner
        // entry point directly to exercise multi-page behavior.
        let got = fetch_assets_inner(&backend, &args, 2).unwrap();
        assert_eq!(got.assets.len(), 3);
        assert!(!got.truncated, "exhausted result must not be truncated");
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
        args.limit = 2;

        let got = fetch_assets_inner(&backend, &args, 10).unwrap();
        assert_eq!(got.assets.len(), 2);
        // Leftover items in the same page mean the server still had more
        // to give us — the marker must be raised.
        assert!(got.truncated);
        // Only the first page should have been requested.
        assert_eq!(backend.calls().len(), 1);
    }

    #[test]
    fn fetch_exact_limit_at_page_boundary_signals_truncated_via_next_page() {
        // limit == items in page 1, but nextPage="2" tells us page 2 exists.
        let backend = FakeBackend::new(vec![resp(
            vec![
                make_asset("a1", "/mnt/x/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                make_asset("a2", "/mnt/x/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
            ],
            Some("2"),
        )]);
        let mut args = default_args();
        args.country = Some("China".into());
        args.limit = 2;
        let got = fetch_assets_inner(&backend, &args, 2).unwrap();
        assert_eq!(got.assets.len(), 2);
        assert!(got.truncated);
        assert_eq!(backend.calls().len(), 1);
    }

    #[test]
    fn fetch_exact_limit_at_end_of_results_is_not_truncated() {
        // limit == total available, and nextPage=null. Boundary case: must
        // not raise the truncation marker.
        let backend = FakeBackend::new(vec![resp(
            vec![
                make_asset("a1", "/mnt/x/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                make_asset("a2", "/mnt/x/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
            ],
            None,
        )]);
        let mut args = default_args();
        args.country = Some("China".into());
        args.limit = 2;
        let got = fetch_assets_inner(&backend, &args, 2).unwrap();
        assert_eq!(got.assets.len(), 2);
        assert!(!got.truncated);
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
        let result = fr(vec![
            make_asset("a1", "/mnt/qnap/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
            make_asset("a2", "/other/b.jpg", "IMAGE", "2025-01-02T00:00:00Z"),
        ]);
        let mut args = default_args();
        args.query = Some("x".into());
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "/home/u/Photos/a.jpg");
    }

    #[test]
    fn emit_paths_include_unmapped_marks_them() {
        let result = fr(vec![make_asset(
            "a2",
            "/other/b.jpg",
            "IMAGE",
            "2025-01-02T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.include_unmapped = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "UNMAPPED\t/other/b.jpg");
    }

    #[test]
    fn emit_paths_verify_skips_missing_files() {
        let result = fr(vec![make_asset(
            "a1",
            "/mnt/qnap/nope.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.verify = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        assert!(String::from_utf8(buf).unwrap().is_empty());
    }

    #[test]
    fn emit_paths_verify_with_include_missing_emits_marker() {
        let result = fr(vec![make_asset(
            "a1",
            "/mnt/qnap/nope.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.verify = true;
        args.include_missing = true;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "MISSING\t/home/u/Photos/nope.jpg");
    }

    #[test]
    fn emit_json_is_ndjson() {
        let result = fr(vec![
            make_asset("a1", "/mnt/qnap/a.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
            make_asset("a2", "/mnt/qnap/b.jpg", "VIDEO", "2025-01-02T00:00:00Z"),
        ]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Json;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
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
        let result = fr(vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Table;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
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

    // ---- truncation marker --------------------------------------------

    #[test]
    fn emit_paths_appends_dots_when_truncated() {
        let result = fr_truncated(vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["/home/u/Photos/a.jpg", "......"]);
    }

    #[test]
    fn emit_paths_no_marker_when_not_truncated() {
        let result = fr(vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(!out.contains("......"), "got: {out}");
    }

    #[test]
    fn emit_table_appends_dots_when_truncated() {
        let result = fr_truncated(vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Table;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let last_line = out.lines().last().unwrap();
        assert_eq!(last_line, "......");
    }

    #[test]
    fn emit_json_truncation_marker_is_parseable() {
        let result = fr_truncated(vec![make_asset(
            "a1",
            "/mnt/qnap/a.jpg",
            "IMAGE",
            "2025-01-01T00:00:00Z",
        )]);
        let mut args = default_args();
        args.query = Some("x".into());
        args.format = OutputFormat::Json;
        let mut buf = Vec::new();
        emit_to_writer(&pmap(), &args, &result, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        let marker: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(marker["truncated"], true);
    }
}
