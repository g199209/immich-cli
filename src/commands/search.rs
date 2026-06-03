use crate::client::ImmichClient;
use crate::config::Config;
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

pub fn run(cfg: &Config, args: SearchArgs) -> Result<()> {
    if args.limit == 0 {
        bail!("--limit must be > 0");
    }
    if args.page_size == 0 {
        bail!("--page-size must be > 0");
    }

    let client = ImmichClient::new(cfg)?;
    let assets = fetch_assets(&client, &args)?;

    emit(cfg, &args, &assets)
}

fn fetch_assets(client: &ImmichClient, args: &SearchArgs) -> Result<Vec<Asset>> {
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

        let resp = client.search(&req)?;
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
fn normalize_date_start(input: &str) -> Result<String> {
    normalize_date(input, false)
}

/// Bare `YYYY-MM-DD` becomes UTC 23:59:59.999 of that day (end of day).
fn normalize_date_end(input: &str) -> Result<String> {
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

fn emit(cfg: &Config, args: &SearchArgs, assets: &[Asset]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    use std::io::Write;

    let mut rows: Vec<Row> = Vec::with_capacity(assets.len());
    for asset in assets {
        let local = path_map::translate(&asset.original_path, &cfg.path_map);
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
            write_table(&mut out, &rows)?;
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
