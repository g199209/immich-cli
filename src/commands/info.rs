use crate::client::{ImmichClient, InfoBackend, SearchBackend};
use crate::config::Config;
use crate::models::SearchRequest;
use crate::path_map;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, ValueEnum};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Local NFS path to the photo or video to look up. Tilde (`~`) is
    /// expanded; relative paths are resolved against the current directory.
    pub path: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// Structured human-readable text with section headers. Also stays
    /// grep-friendly so AI assistants and shell pipelines can scan it.
    Text,
    /// Pretty-printed JSON of the full asset detail (plus the resolved
    /// localPath and any albums). Use this for automation.
    Json,
}

pub fn run(cfg: &Config, args: InfoArgs) -> Result<()> {
    let client = ImmichClient::new(cfg)?;
    run_with(cfg, &client, &client, args, &mut std::io::stdout())
}

/// Backend-agnostic entry point used by the binary and the test suite.
pub fn run_with<S, I, W>(
    cfg: &Config,
    search: &S,
    info: &I,
    args: InfoArgs,
    out: &mut W,
) -> Result<()>
where
    S: SearchBackend,
    I: InfoBackend,
    W: std::io::Write,
{
    let local_path = resolve_local_path(&args.path)?;
    let server_path = path_map::reverse_translate(&local_path.to_string_lossy(), &cfg.path_map)
        .ok_or_else(|| {
            anyhow!(
            "no path mapping matches {} — add a [[path_map]] entry whose `local` prefix covers it",
            local_path.display()
        )
        })?;

    let asset_id = find_asset_id(search, &server_path)?;
    let mut asset = info.get_asset(&asset_id)?;
    let albums = info
        .albums_for_asset(&asset_id)
        .unwrap_or(Value::Array(vec![]));
    // OCR may be absent on older Immich servers — silently treat as empty so
    // we don't fail the whole `info` call when only this extra is missing.
    let ocr = info
        .ocr_for_asset(&asset_id)
        .unwrap_or(Value::Array(vec![]));

    // Augment the asset JSON with the inputs the caller cares about: where
    // the file lives locally, plus the album and OCR side-data.
    if let Some(obj) = asset.as_object_mut() {
        obj.insert(
            "localPath".into(),
            Value::String(local_path.to_string_lossy().into_owned()),
        );
        obj.insert("albums".into(), albums.clone());
        obj.insert("ocr".into(), ocr.clone());
    }

    match args.format {
        OutputFormat::Json => {
            prune_for_json(&mut asset);
            emit_json(out, &asset)
        }
        OutputFormat::Text => emit_text(out, &asset, &albums, &ocr),
    }
}

/// Strip bookkeeping fields from the JSON payload so `--format json`
/// surfaces the same content the text format does. Whitelist over
/// denylist: when Immich adds new fields they're silently dropped
/// until we decide they're meaningful. The text format keeps the same
/// fields, so the two outputs stay in sync.
fn prune_for_json(asset: &mut Value) {
    const TOP_KEEP: &[&str] = &[
        // CLI-added.
        "localPath",
        "albums",
        "ocr",
        // Content we surface in text.
        "localDateTime",
        "originalMimeType",
        "width",
        "height",
        "duration",
        "exifInfo",
        "people",
        "unassignedFaces",
        "tags",
    ];
    const EXIF_KEEP: &[&str] = &[
        // Time zone for the local wall-clock time.
        "timeZone",
        // GPS + reverse-geocoded location.
        "latitude",
        "longitude",
        "city",
        "state",
        "country",
        // Camera.
        "make",
        "model",
        "lensModel",
        "fNumber",
        "focalLength",
        "iso",
        "exposureTime",
        "orientation",
        "projectionType",
        // File size.
        "fileSizeInByte",
    ];
    if let Some(obj) = asset.as_object_mut() {
        obj.retain(|k, _| TOP_KEEP.contains(&k.as_str()));
        if let Some(exif) = obj.get_mut("exifInfo").and_then(|v| v.as_object_mut()) {
            exif.retain(|k, _| EXIF_KEEP.contains(&k.as_str()));
        }
    }
}

/// Search by filename, then filter by exact `originalPath`. This is the
/// most reliable way to look up an asset from its on-disk path: there's no
/// path filter in Immich's search API, but filename collisions are rare
/// enough that this only fetches a handful of candidates.
fn find_asset_id<S: SearchBackend>(search: &S, server_path: &str) -> Result<String> {
    let filename = Path::new(server_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow!("could not extract filename from `{server_path}`"))?;

    let req = SearchRequest {
        original_file_name: Some(filename.clone()),
        size: Some(250),
        ..Default::default()
    };
    let resp = search.search(&req)?;
    let mut matches: Vec<_> = resp
        .assets
        .items
        .into_iter()
        .filter(|a| a.original_path == server_path)
        .collect();

    match matches.len() {
        0 => bail!(
            "no Immich asset has originalPath `{server_path}` \
             (filename `{filename}` was not found, or Immich knows it under a different path)"
        ),
        1 => Ok(matches.pop().unwrap().id),
        n => bail!(
            "{n} Immich assets share originalPath `{server_path}` — \
             this should never happen; reindex Immich or report a bug"
        ),
    }
}

fn resolve_local_path(input: &Path) -> Result<PathBuf> {
    let s = input.to_string_lossy();
    let expanded: PathBuf = if let Some(rest) = s.strip_prefix("~/") {
        directories::UserDirs::new()
            .ok_or_else(|| anyhow!("cannot resolve home directory for tilde expansion"))?
            .home_dir()
            .join(rest)
    } else if s.as_ref() == "~" {
        directories::UserDirs::new()
            .ok_or_else(|| anyhow!("cannot resolve home directory"))?
            .home_dir()
            .to_path_buf()
    } else {
        input.to_path_buf()
    };

    if expanded.is_absolute() {
        Ok(expanded)
    } else {
        std::env::current_dir()
            .context("cannot resolve current working directory")
            .map(|cwd| cwd.join(expanded))
    }
}

fn emit_json<W: std::io::Write>(out: &mut W, asset: &Value) -> Result<()> {
    serde_json::to_writer_pretty(&mut *out, asset)?;
    writeln!(out)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Text formatter — structured by section, key:value with aligned colons.
// Designed to be skim-readable for humans, grep-friendly for shell tooling,
// and unambiguous for an LLM doing pattern extraction.
// ---------------------------------------------------------------------------

fn emit_text<W: std::io::Write>(
    out: &mut W,
    asset: &Value,
    albums: &Value,
    ocr: &Value,
) -> Result<()> {
    let mut s = Section::new(out);

    s.heading("File")?;
    s.kv_opt("MIME", as_str(&asset["originalMimeType"]))?;
    if let Some(bytes) = as_u64(&asset["exifInfo"]["fileSizeInByte"]) {
        s.kv("Size", format!("{} ({} bytes)", human_size(bytes), bytes))?;
    }
    let w = asset["width"]
        .as_u64()
        .or_else(|| as_u64(&asset["exifInfo"]["exifImageWidth"]));
    let h = asset["height"]
        .as_u64()
        .or_else(|| as_u64(&asset["exifInfo"]["exifImageHeight"]));
    if let (Some(w), Some(h)) = (w, h) {
        s.kv("Dimensions", format!("{w}x{h}"))?;
    }
    // Duration only renders for video assets.
    if let Some(dur) = as_str(&asset["duration"]) {
        if dur != "0:00:00.00000" && !dur.is_empty() {
            s.kv("Duration", dur)?;
        }
    }

    s.heading("Time")?;
    // Single time, always wall-clock at the photo's location (GPS-derived,
    // or device EXIF zone if no GPS). Not the current machine's TZ. The
    // EXIF time zone is appended so the reader knows what "local" refers
    // to — Immich reports it as either an IANA name (Asia/Shanghai) or a
    // UTC offset (UTC+8), and we pass it through verbatim.
    if let Some(naive) = as_str(&asset["localDateTime"]).map(humanize_naive_iso) {
        let value = match as_nonempty_str(&asset["exifInfo"]["timeZone"]) {
            Some(tz) => format!("{naive} ({tz})"),
            None => naive,
        };
        s.kv("File created", value)?;
    }

    let has_location =
        asset["exifInfo"]["latitude"].is_number() || asset["exifInfo"]["longitude"].is_number();
    if has_location || asset["exifInfo"]["city"].is_string() {
        s.heading("Location")?;
        s.kv_opt("Latitude", as_number(&asset["exifInfo"]["latitude"]))?;
        s.kv_opt("Longitude", as_number(&asset["exifInfo"]["longitude"]))?;
        s.kv_opt("City", as_nonempty_str(&asset["exifInfo"]["city"]))?;
        s.kv_opt("State", as_nonempty_str(&asset["exifInfo"]["state"]))?;
        s.kv_opt("Country", as_nonempty_str(&asset["exifInfo"]["country"]))?;
    }

    let camera_keys = [
        "make",
        "model",
        "lensModel",
        "fNumber",
        "focalLength",
        "iso",
        "exposureTime",
        "orientation",
    ];
    if camera_keys.iter().any(|k| has_value(&asset["exifInfo"][k])) {
        s.heading("Camera")?;
        s.kv_opt("Make", as_nonempty_str(&asset["exifInfo"]["make"]))?;
        s.kv_opt("Model", as_nonempty_str(&asset["exifInfo"]["model"]))?;
        s.kv_opt("Lens", as_nonempty_str(&asset["exifInfo"]["lensModel"]))?;
        s.kv_opt(
            "Aperture",
            as_number(&asset["exifInfo"]["fNumber"]).map(|n| format!("f/{n}")),
        )?;
        s.kv_opt(
            "Focal length",
            as_number(&asset["exifInfo"]["focalLength"]).map(|n| format!("{n} mm")),
        )?;
        s.kv_opt("ISO", as_number(&asset["exifInfo"]["iso"]))?;
        s.kv_opt(
            "Exposure",
            as_nonempty_str(&asset["exifInfo"]["exposureTime"]),
        )?;
        s.kv_opt(
            "Orientation",
            as_nonempty_str(&asset["exifInfo"]["orientation"]),
        )?;
        s.kv_opt(
            "Projection",
            as_nonempty_str(&asset["exifInfo"]["projectionType"]),
        )?;
    }

    let people = asset["people"].as_array().cloned().unwrap_or_default();
    s.heading(&format!("People ({})", people.len()))?;
    if people.is_empty() {
        s.line("  (none)")?;
    } else {
        for p in &people {
            let name = as_nonempty_str(&p["name"]).unwrap_or_else(|| "(unnamed)".into());
            let id = as_str(&p["id"]).unwrap_or_default();
            let face_count = p["faces"].as_array().map(|a| a.len()).unwrap_or(0);
            let mut extras = Vec::new();
            extras.push(format!("id={id}"));
            extras.push(format!("faces={face_count}"));
            if let Some(birth) = as_nonempty_str(&p["birthDate"]) {
                extras.push(format!("birth={birth}"));
            }
            if p["isFavorite"].as_bool() == Some(true) {
                extras.push("favorite".into());
            }
            if p["isHidden"].as_bool() == Some(true) {
                extras.push("hidden".into());
            }
            s.line(&format!("  - {name} ({})", extras.join(", ")))?;
        }
    }
    let unassigned = asset["unassignedFaces"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    s.kv("Unassigned faces", unassigned.to_string())?;

    let tags = asset["tags"].as_array().cloned().unwrap_or_default();
    s.heading(&format!("Tags ({})", tags.len()))?;
    if tags.is_empty() {
        s.line("  (none)")?;
    } else {
        for t in &tags {
            let name = as_str(&t["value"])
                .or_else(|| as_str(&t["name"]))
                .unwrap_or_default();
            s.line(&format!("  - {name}"))?;
        }
    }

    // OCR-detected text. The `[NN%]` confidence prefix lets a reader (or
    // grep, or an LLM) cheaply filter out low-confidence noise.
    let ocr_entries = ocr.as_array().cloned().unwrap_or_default();
    s.heading(&format!("OCR ({} regions)", ocr_entries.len()))?;
    if ocr_entries.is_empty() {
        s.line("  (none)")?;
    } else {
        for entry in &ocr_entries {
            let text = as_nonempty_str(&entry["text"]).unwrap_or_default();
            let score = entry["textScore"]
                .as_f64()
                .map(|v| format!("[{:3.0}%] ", (v * 100.0).round()))
                .unwrap_or_default();
            let hidden = if entry["isVisible"].as_bool() == Some(false) {
                " (hidden)"
            } else {
                ""
            };
            s.line(&format!("  - {score}{text}{hidden}"))?;
        }
    }

    let albums_arr = albums.as_array().cloned().unwrap_or_default();
    s.heading(&format!("Albums ({})", albums_arr.len()))?;
    if albums_arr.is_empty() {
        s.line("  (none)")?;
    } else {
        for a in &albums_arr {
            let name = as_str(&a["albumName"]).unwrap_or_default();
            let id = as_str(&a["id"]).unwrap_or_default();
            let count = a["assetCount"].as_u64().unwrap_or(0);
            s.line(&format!("  - {name} (id={id}, assets={count})"))?;
        }
    }

    Ok(())
}

// ---- JSON value helpers ----------------------------------------------------

fn as_str(v: &Value) -> Option<String> {
    v.as_str().map(|s| s.to_owned())
}

fn as_nonempty_str(v: &Value) -> Option<String> {
    v.as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn as_u64(v: &Value) -> Option<u64> {
    v.as_u64()
}

fn as_number(v: &Value) -> Option<String> {
    if v.is_number() {
        Some(v.to_string())
    } else {
        None
    }
}

fn has_value(v: &Value) -> bool {
    !v.is_null() && as_nonempty_str(v).is_some() || v.is_number()
}

/// Format Immich's `localDateTime`. The value is wall-clock time at the
/// photo's location — derived from GPS timezone if available, otherwise
/// from the camera's EXIF time zone, never from the machine running this
/// CLI. Immich transports it as a Z-suffixed ISO string but it's not UTC,
/// so we strip the suffix.
fn humanize_naive_iso(s: String) -> String {
    chrono::DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or(s)
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.2} {}", UNITS[unit])
    }
}

// ---- Section writer --------------------------------------------------------

/// Tiny helper to keep section formatting consistent. Each heading starts
/// flush-left, each key/value is indented two spaces, and we pad keys to a
/// common width inside one heading so colons line up.
struct Section<'a, W: std::io::Write> {
    out: &'a mut W,
    first: bool,
}

impl<'a, W: std::io::Write> Section<'a, W> {
    fn new(out: &'a mut W) -> Self {
        Self { out, first: true }
    }

    fn heading(&mut self, title: &str) -> std::io::Result<()> {
        if !self.first {
            writeln!(self.out)?;
        }
        self.first = false;
        writeln!(self.out, "{title}")
    }

    fn kv<V: AsRef<str>>(&mut self, key: &str, value: V) -> std::io::Result<()> {
        writeln!(self.out, "  {:<18}{}", format!("{key}:"), value.as_ref())
    }

    fn kv_opt<V: AsRef<str>>(&mut self, key: &str, value: Option<V>) -> std::io::Result<()> {
        if let Some(v) = value {
            self.kv(key, v)?;
        }
        Ok(())
    }

    fn line(&mut self, raw: &str) -> std::io::Result<()> {
        writeln!(self.out, "{raw}")
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PathMapEntry;
    use crate::models::{Asset, AssetsBucket, SearchResponse};
    use std::cell::RefCell;

    // ---- fakes ----

    struct FakeSearch {
        responses: RefCell<Vec<SearchResponse>>,
        calls: RefCell<Vec<SearchRequest>>,
    }
    impl FakeSearch {
        fn new(responses: Vec<SearchResponse>) -> Self {
            Self {
                responses: RefCell::new(responses),
                calls: RefCell::new(vec![]),
            }
        }
    }
    impl SearchBackend for FakeSearch {
        fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
            self.calls.borrow_mut().push(req.clone());
            Ok(self.responses.borrow_mut().remove(0))
        }
    }

    struct FakeInfo {
        asset: serde_json::Value,
        albums: serde_json::Value,
        ocr: serde_json::Value,
    }
    impl Default for FakeInfo {
        fn default() -> Self {
            Self {
                asset: Value::Null,
                albums: serde_json::json!([]),
                ocr: serde_json::json!([]),
            }
        }
    }
    impl InfoBackend for FakeInfo {
        fn get_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(self.asset.clone())
        }
        fn albums_for_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(self.albums.clone())
        }
        fn ocr_for_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(self.ocr.clone())
        }
    }

    fn cfg() -> Config {
        Config {
            server_url: "http://x".into(),
            api_key: "k".into(),
            path_map: vec![PathMapEntry {
                server: "/mnt/qnap".into(),
                local: "/home/u/Photos".into(),
            }],
            timeout_secs: 60,
            llm: None,
            people: std::collections::BTreeMap::new(),
        }
    }

    fn search_hit(id: &str, server_path: &str) -> SearchResponse {
        SearchResponse {
            assets: AssetsBucket {
                total: 1,
                count: 1,
                items: vec![Asset {
                    id: id.into(),
                    original_path: server_path.into(),
                    original_file_name: Path::new(server_path)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    asset_type: "IMAGE".into(),
                    file_created_at: None,
                    local_date_time: None,
                    checksum: String::new(),
                    exif_info: None,
                }],
                next_page: None,
            },
        }
    }

    fn search_multi(items: Vec<(&str, &str)>) -> SearchResponse {
        let assets = items
            .into_iter()
            .map(|(id, path)| Asset {
                id: id.into(),
                original_path: path.into(),
                original_file_name: Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                asset_type: "IMAGE".into(),
                file_created_at: None,
                local_date_time: None,
                checksum: String::new(),
                exif_info: None,
            })
            .collect::<Vec<_>>();
        let n = assets.len() as u32;
        SearchResponse {
            assets: AssetsBucket {
                total: n,
                count: n,
                items: assets,
                next_page: None,
            },
        }
    }

    // ---- find_asset_id edge cases ----

    #[test]
    fn find_returns_id_when_single_match() {
        let s = FakeSearch::new(vec![search_hit("the-id", "/mnt/qnap/PYL/x.jpg")]);
        let id = find_asset_id(&s, "/mnt/qnap/PYL/x.jpg").unwrap();
        assert_eq!(id, "the-id");
        // filename used for the search filter:
        assert_eq!(
            s.calls.borrow()[0].original_file_name.as_deref(),
            Some("x.jpg")
        );
    }

    #[test]
    fn find_disambiguates_filename_collision_by_path() {
        let s = FakeSearch::new(vec![search_multi(vec![
            ("wrong-id", "/mnt/qnap/OTHER/x.jpg"),
            ("right-id", "/mnt/qnap/PYL/x.jpg"),
        ])]);
        let id = find_asset_id(&s, "/mnt/qnap/PYL/x.jpg").unwrap();
        assert_eq!(id, "right-id");
    }

    #[test]
    fn find_errors_when_no_match() {
        let s = FakeSearch::new(vec![search_multi(vec![])]);
        let err = find_asset_id(&s, "/mnt/qnap/PYL/x.jpg")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no Immich asset"), "got: {err}");
        assert!(err.contains("/mnt/qnap/PYL/x.jpg"), "got: {err}");
    }

    #[test]
    fn find_errors_when_duplicate_paths() {
        // Two distinct assets with the exact same server path is a server
        // misindex; report it loudly rather than silently picking one.
        let s = FakeSearch::new(vec![search_multi(vec![
            ("a", "/mnt/qnap/PYL/x.jpg"),
            ("b", "/mnt/qnap/PYL/x.jpg"),
        ])]);
        let err = find_asset_id(&s, "/mnt/qnap/PYL/x.jpg")
            .unwrap_err()
            .to_string();
        assert!(err.contains("share originalPath"), "got: {err}");
    }

    // ---- resolve_local_path ----

    #[test]
    fn resolve_keeps_absolute_paths_unchanged() {
        let got = resolve_local_path(Path::new("/home/u/Photos/x.jpg")).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/Photos/x.jpg"));
    }

    #[test]
    fn resolve_expands_leading_tilde() {
        let home = directories::UserDirs::new()
            .unwrap()
            .home_dir()
            .to_path_buf();
        let got = resolve_local_path(Path::new("~/QNAP-Photos/x.jpg")).unwrap();
        assert_eq!(got, home.join("QNAP-Photos/x.jpg"));
    }

    // ---- end-to-end run_with: full text output ----

    fn sample_asset() -> Value {
        serde_json::json!({
            "id": "asset-1",
            "originalPath": "/mnt/qnap/PYL/2018/IMG_20180908_185429.jpg",
            "originalFileName": "IMG_20180908_185429.jpg",
            "originalMimeType": "image/jpeg",
            "type": "IMAGE",
            "createdAt": "2026-05-21T14:48:59.239Z",
            "updatedAt": "2026-05-22T15:34:26.312Z",
            "fileCreatedAt": "2018-09-08T10:54:29.000Z",
            "fileModifiedAt": "2024-02-22T13:33:17.000Z",
            "localDateTime": "2018-09-08T18:54:29.000Z",
            "isFavorite": false, "isArchived": false, "isTrashed": false,
            "isOffline": false, "isEdited": false, "hasMetadata": true,
            "visibility": "timeline",
            "checksum": "abc123==",
            "thumbhash": "th",
            "width": 4032, "height": 3024,
            "duration": "0:00:00.00000",
            "libraryId": "lib-1",
            "owner": {"name": "mingfei", "email": "m@x.com"},
            "exifInfo": {
                "make": "HONOR", "model": "FNE-AN00", "lensModel": null,
                "fNumber": 4, "focalLength": 5.52, "iso": 452,
                "exposureTime": "1/33", "orientation": null,
                "fileSizeInByte": 2_741_923,
                "dateTimeOriginal": "2018-09-08T10:54:29.000+00:00",
                "modifyDate": "2018-09-08T10:54:29+00:00",
                "timeZone": "Asia/Shanghai",
                "latitude": 31.1269, "longitude": 121.5718,
                "city": "Kangqiao", "state": "Shanghai",
                "country": "People's Republic of China",
                "description": "", "rating": null,
                "exifImageWidth": 4032, "exifImageHeight": 3024,
                "projectionType": null
            },
            "tags": [{"value": "sunset"}, {"value": "beach"}],
            "people": [
                {"id": "p-1", "name": "张三", "birthDate": null,
                 "isFavorite": false, "isHidden": false,
                 "faces": [{"id": "f1"}, {"id": "f2"}]},
                {"id": "p-2", "name": "", "birthDate": null,
                 "isFavorite": false, "isHidden": false,
                 "faces": [{"id": "f3"}]}
            ],
            "unassignedFaces": [{"id": "uf"}],
            "stack": null,
            "duplicateId": null,
            "livePhotoVideoId": null
        })
    }

    fn run_collecting(cfg: &Config, asset: Value, format: OutputFormat) -> String {
        run_collecting_with_ocr(cfg, asset, serde_json::json!([]), format)
    }

    fn run_collecting_with_ocr(
        cfg: &Config,
        asset: Value,
        ocr: Value,
        format: OutputFormat,
    ) -> String {
        let server_path = asset["originalPath"].as_str().unwrap().to_owned();
        let local_path = "/home/u/Photos/PYL/2018/IMG_20180908_185429.jpg";
        let search = FakeSearch::new(vec![search_hit("asset-1", &server_path)]);
        let info = FakeInfo {
            asset,
            albums: serde_json::json!([]),
            ocr,
        };
        let mut buf = Vec::new();
        run_with(
            cfg,
            &search,
            &info,
            InfoArgs {
                path: PathBuf::from(local_path),
                format,
            },
            &mut buf,
        )
        .unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn text_output_contains_all_sections() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Text);
        for heading in [
            "File",
            "Time",
            "Location",
            "Camera",
            "People (2)",
            "Tags (2)",
            "OCR (0 regions)",
            "Albums (0)",
        ] {
            assert!(
                out.contains(heading),
                "missing heading `{heading}` in:\n{out}"
            );
        }
        // Bookkeeping `Immich` section was removed as not useful.
        assert!(
            !out.contains("\nImmich\n"),
            "Immich section should be gone:\n{out}"
        );
    }

    fn sample_ocr() -> Value {
        serde_json::json!([
            {
                "id": "o1", "assetId": "asset-1",
                "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.1,
                "x3": 0.3, "y3": 0.2, "x4": 0.1, "y4": 0.2,
                "boxScore": 0.88, "textScore": 0.99,
                "text": "DELL", "isVisible": true
            },
            {
                "id": "o2", "assetId": "asset-1",
                "x1": 0.1, "y1": 0.3, "x2": 0.8, "y2": 0.3,
                "x3": 0.8, "y3": 0.4, "x4": 0.1, "y4": 0.4,
                "boxScore": 0.7, "textScore": 0.85,
                "text": "浙江大学 电气工程学院", "isVisible": true
            },
            {
                "id": "o3", "assetId": "asset-1",
                "x1": 0.0, "y1": 0.0, "x2": 0.1, "y2": 0.0,
                "x3": 0.1, "y3": 0.1, "x4": 0.0, "y4": 0.1,
                "boxScore": 0.5, "textScore": 0.41,
                "text": "low-conf", "isVisible": false
            }
        ])
    }

    #[test]
    fn text_output_renders_ocr_with_confidence_and_hidden_flag() {
        let out = run_collecting_with_ocr(&cfg(), sample_asset(), sample_ocr(), OutputFormat::Text);
        assert!(
            out.contains("OCR (3 regions)"),
            "missing OCR heading in:\n{out}"
        );
        // Confidence is rendered as `[NN%]` prefix; helps grep / skim.
        assert!(
            out.contains("[ 99%] DELL"),
            "missing high-confidence OCR entry in:\n{out}"
        );
        assert!(
            out.contains("[ 85%] 浙江大学 电气工程学院"),
            "missing unicode OCR text in:\n{out}"
        );
        // Hidden (filtered) text is marked so callers can spot moderated data.
        assert!(
            out.contains("[ 41%] low-conf (hidden)"),
            "missing hidden marker in:\n{out}"
        );
    }

    #[test]
    fn json_output_includes_ocr_array_verbatim() {
        let out = run_collecting_with_ocr(&cfg(), sample_asset(), sample_ocr(), OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        let entries = parsed["ocr"].as_array().expect("ocr must be an array");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["text"], "DELL");
        assert_eq!(entries[1]["text"], "浙江大学 电气工程学院");
        // Bounding-box coords round-trip untouched — automation may want
        // them, and we never pretend to interpret them.
        assert_eq!(entries[0]["x1"], 0.1);
        assert_eq!(entries[2]["isVisible"], false);
    }

    #[test]
    fn text_output_shows_human_size_and_dimensions() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Text);
        assert!(out.contains("4032x3024"), "missing dimensions in:\n{out}");
        assert!(out.contains("2.61 MB"), "missing human size in:\n{out}");
        assert!(
            out.contains("2741923 bytes"),
            "missing raw byte count in:\n{out}"
        );
    }

    #[test]
    fn text_output_drops_redundant_path_filename_owner_fields() {
        // Path/filename are redundant with the input the user just typed,
        // and owner/library/IDs are bookkeeping — none of these should
        // appear in the trimmed text output.
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Text);
        for unwanted in [
            "Local path",
            "Server path",
            "Filename",
            "Type:",
            "Owner",
            "Asset ID",
            "Library",
            "Checksum",
            "Thumbhash",
            "Visibility",
        ] {
            assert!(
                !out.contains(unwanted),
                "unexpected field `{unwanted}` in trimmed output:\n{out}"
            );
        }
    }

    #[test]
    fn text_output_handles_people_with_and_without_names() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Text);
        assert!(out.contains("张三"), "missing person name in:\n{out}");
        assert!(
            out.contains("(unnamed)"),
            "missing unnamed-person placeholder in:\n{out}"
        );
        assert!(out.contains("faces=2"), "missing face count in:\n{out}");
        assert!(
            out.contains("Unassigned faces:"),
            "missing unassigned-faces line in:\n{out}"
        );
    }

    #[test]
    fn time_section_has_single_local_line_with_tz_in_parens() {
        // Only one time is shown — File created — in the photo's local
        // wall-clock time, followed by the EXIF time zone in parens so
        // the reader knows which zone "local" means.
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Text);
        let time_block: Vec<&str> = out
            .lines()
            .skip_while(|l| *l != "Time")
            .skip(1)
            .take_while(|l| !l.is_empty())
            .collect();
        assert_eq!(time_block.len(), 1, "got: {time_block:?}");
        let line = time_block[0];
        assert!(line.contains("File created"), "got: {line}");
        assert!(
            line.contains("2018-09-08 18:54:29 (Asia/Shanghai)"),
            "got: {line}"
        );
    }

    #[test]
    fn time_section_omits_tz_parens_when_absent() {
        let mut asset = sample_asset();
        asset["exifInfo"]["timeZone"] = serde_json::Value::Null;
        let out = run_collecting(&cfg(), asset, OutputFormat::Text);
        let line = out
            .lines()
            .find(|l| l.contains("File created"))
            .expect("missing File created line");
        assert!(line.contains("2018-09-08 18:54:29"), "got: {line}");
        assert!(
            !line.contains('('),
            "no parens should be shown when TZ is unknown: {line}"
        );
    }

    #[test]
    fn json_output_keeps_meaningful_fields() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["localPath"],
            "/home/u/Photos/PYL/2018/IMG_20180908_185429.jpg"
        );
        assert!(parsed["albums"].is_array());
        assert_eq!(parsed["originalMimeType"], "image/jpeg");
        assert_eq!(parsed["width"], 4032);
        assert_eq!(parsed["height"], 3024);
        assert_eq!(parsed["localDateTime"], "2018-09-08T18:54:29.000Z");
        // EXIF whitelist: location + camera kept, raw EXIF bookkeeping gone.
        let exif = &parsed["exifInfo"];
        assert_eq!(exif["latitude"], 31.1269);
        assert_eq!(exif["country"], "People's Republic of China");
        assert_eq!(exif["make"], "HONOR");
        assert_eq!(exif["fNumber"], 4);
        assert_eq!(exif["timeZone"], "Asia/Shanghai");
        assert_eq!(exif["fileSizeInByte"], 2_741_923);
        // People/tags arrays untouched.
        assert!(parsed["people"].is_array());
        assert!(parsed["tags"].is_array());
    }

    #[test]
    fn json_output_drops_bookkeeping_top_level_fields() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        for k in [
            "id",
            "originalPath",
            "originalFileName",
            "type",
            "owner",
            "ownerId",
            "libraryId",
            "deviceId",
            "deviceAssetId",
            "visibility",
            "isFavorite",
            "isArchived",
            "isTrashed",
            "isOffline",
            "isEdited",
            "hasMetadata",
            "thumbhash",
            "checksum",
            "createdAt",
            "updatedAt",
            "fileCreatedAt",
            "fileModifiedAt",
            "duplicateId",
            "stack",
            "livePhotoVideoId",
        ] {
            assert!(parsed.get(k).is_none(), "expected `{k}` to be pruned");
        }
    }

    #[test]
    fn json_output_drops_bookkeeping_exif_fields() {
        let out = run_collecting(&cfg(), sample_asset(), OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        let exif = parsed["exifInfo"].as_object().unwrap();
        for k in [
            "dateTimeOriginal",
            "modifyDate",
            "description",
            "rating",
            "exifImageWidth",
            "exifImageHeight",
        ] {
            assert!(exif.get(k).is_none(), "expected exif.{k} to be pruned");
        }
    }

    #[test]
    fn unmapped_local_path_errors_with_actionable_message() {
        let asset = sample_asset();
        let search = FakeSearch::new(vec![]);
        let info = FakeInfo {
            asset,
            ..Default::default()
        };
        let mut buf = Vec::new();
        let err = run_with(
            &cfg(),
            &search,
            &info,
            InfoArgs {
                path: PathBuf::from("/somewhere/else/x.jpg"),
                format: OutputFormat::Text,
            },
            &mut buf,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no path mapping"), "got: {err}");
        assert!(err.contains("/somewhere/else/x.jpg"), "got: {err}");
    }

    // ---- pure helpers ----

    #[test]
    fn human_size_picks_unit() {
        assert_eq!(human_size(500), "500 B");
        assert_eq!(human_size(1536), "1.50 KB");
        assert_eq!(human_size(2_741_923), "2.61 MB");
        assert_eq!(human_size(5_000_000_000), "4.66 GB");
    }

    #[test]
    fn humanize_naive_strips_zone_marker() {
        // Inputs Immich ships as Z-suffixed but semantically local must
        // come out without UTC / offset.
        assert_eq!(
            humanize_naive_iso("2018-09-08T18:54:29.000Z".into()),
            "2018-09-08 18:54:29"
        );
        assert_eq!(
            humanize_naive_iso("2018-09-08T18:54:29+08:00".into()),
            "2018-09-08 18:54:29"
        );
    }

    #[test]
    fn humanize_naive_keeps_unparseable_input() {
        assert_eq!(humanize_naive_iso("not-a-date".into()), "not-a-date");
    }
}
