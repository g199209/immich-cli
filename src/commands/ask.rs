use crate::client::{ImmichClient, SearchBackend};
use crate::config::{Config, PathMapEntry};
use crate::llm::{ChatBackend, Message, OpenAiClient};
use crate::models::{Asset, SearchRequest};
use crate::path_map;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, ValueEnum};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Hard caps on the LLM workflow. Both keep prompt sizes manageable and
/// bound how many Immich round-trips a single `ask` makes.
const MAX_KEYWORDS: usize = 16;
const MAX_CANDIDATES: usize = 100;
/// Truncate each candidate description before sending to the reranker so
/// one very long description doesn't blow the prompt budget.
const DESCRIPTION_EXCERPT_CHARS: usize = 1500;

#[derive(Args, Debug)]
pub struct AskArgs {
    /// Natural-language description of the photo you want to find,
    /// in Chinese or English. The query is fed to an LLM which extracts
    /// keywords, runs them against Immich's description index, and
    /// re-ranks the candidates.
    pub query: String,

    /// Maximum number of results to print. The LLM ranks all candidates
    /// by relevance; only the top N are kept.
    #[arg(long, default_value_t = 50)]
    pub limit: u32,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Paths)]
    pub format: OutputFormat,

    /// Confirm each translated local path exists on the filesystem;
    /// missing files are warned about on stderr and skipped.
    #[arg(long)]
    pub verify: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// One local path per line. Order is by LLM-assessed relevance.
    Paths,
    /// One JSON object per asset (newline-delimited) with localPath +
    /// description + rank.
    Json,
    /// Aligned table: RANK, TYPE, TAKEN, LOCATION, PATH.
    Table,
}

pub fn run(cfg: &Config, args: AskArgs) -> Result<()> {
    let llm_cfg = cfg.llm.as_ref().ok_or_else(|| {
        anyhow!(
            "ask requires an [llm] section in config.toml \
             (base_url, api_key, model)"
        )
    })?;
    let client = ImmichClient::new(cfg)?;
    let llm = OpenAiClient::new(llm_cfg)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    run_with(cfg, &client, &llm, args, &mut out)
}

pub fn run_with<S, L, W>(
    cfg: &Config,
    search: &S,
    llm: &L,
    args: AskArgs,
    out: &mut W,
) -> Result<()>
where
    S: SearchBackend,
    L: ChatBackend,
    W: std::io::Write,
{
    let q = args.query.trim();
    if q.is_empty() {
        bail!("ask: query is empty");
    }
    if args.limit == 0 {
        bail!("ask: --limit must be > 0");
    }

    eprintln!("ask: expanding query into keywords ...");
    let keywords = expand_keywords(llm, q)?;
    if keywords.is_empty() {
        bail!("LLM returned no keywords for the query");
    }
    eprintln!("ask: {} keywords: {}", keywords.len(), keywords.join(", "));

    let candidates = collect_candidates(search, &keywords)?;
    if candidates.is_empty() {
        eprintln!("ask: no description matched any keyword");
        return Ok(());
    }
    eprintln!(
        "ask: {} unique candidates from substring search, reranking ...",
        candidates.len()
    );

    let selected_ids = rerank(llm, q, &candidates)?;
    let ordered = take_in_order(&candidates, &selected_ids, args.limit as usize);

    emit(&cfg.path_map, &args, &ordered, out)
}

// ---- stage 1: keyword expansion ------------------------------------------

#[derive(Deserialize)]
struct KeywordsReply {
    #[serde(default)]
    keywords: Vec<String>,
}

const KEYWORD_SYSTEM_PROMPT: &str = "\
You extract substring search keywords from a natural-language photo-search query.\n\n\
IMPORTANT — the photo descriptions you'll match against are ALWAYS written in \
Chinese. Substring matching is literal, so keywords must be the Chinese forms \
that actually appear in the descriptions. Translate the user's intent into \
Chinese before extracting keywords, regardless of the query language.\n\n\
Your output MUST be a single JSON object of the form: \
{\"keywords\": [\"...\", \"...\"]}\n\n\
Rules:\n\
- Up to 16 keywords. Quality > quantity.\n\
- Output keywords in Chinese (e.g. sunset → 夕阳/日落, elephant → 大象, \
sailing boat → 帆船, savannah → 草原).\n\
- Each keyword is a short noun phrase (1-8 Chinese characters) likely to \
appear verbatim inside a Chinese description.\n\
- Include synonyms and lexical variants of central concepts (大象, 象群, \
一群大象). For sunset, include both 夕阳 and 日落. For night cityscapes, \
include 夜景 and 灯光.\n\
- The ONLY exception to Chinese-only output is proper nouns and brand names \
that are conventionally written in English even inside Chinese text \
(e.g. \"DELL\", \"iPhone\", \"BMW\", model numbers). Include both forms \
only if both are plausibly used in Chinese descriptions.\n\
- Do NOT include function words, abstract verbs, or generic words like \
\"照片\", \"图片\", \"看看\", \"想\".";

fn expand_keywords<L: ChatBackend>(llm: &L, query: &str) -> Result<Vec<String>> {
    let messages = [
        Message::system(KEYWORD_SYSTEM_PROMPT),
        Message::user(query.to_string()),
    ];
    let raw = llm.chat_json(&messages)?;
    let reply: KeywordsReply = serde_json::from_str(&raw)
        .with_context(|| format!("LLM keyword reply was not valid JSON: {raw}"))?;
    let mut keywords: Vec<String> = reply
        .keywords
        .into_iter()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .collect();
    keywords.truncate(MAX_KEYWORDS);
    Ok(dedup_preserving_order(keywords))
}

// ---- stage 2: substring search per keyword -------------------------------

fn collect_candidates<S: SearchBackend>(search: &S, keywords: &[String]) -> Result<Vec<Asset>> {
    // Insertion-ordered map: first time we see an asset id is its slot.
    let mut by_id: HashMap<String, Asset> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for kw in keywords {
        let req = SearchRequest {
            description: Some(kw.clone()),
            size: Some(250),
            with_exif: Some(true),
            ..Default::default()
        };
        let resp = search.search(&req)?;
        for asset in resp.assets.items {
            if !by_id.contains_key(&asset.id) {
                order.push(asset.id.clone());
                by_id.insert(asset.id.clone(), asset);
            }
            if by_id.len() >= MAX_CANDIDATES {
                break;
            }
        }
        if by_id.len() >= MAX_CANDIDATES {
            break;
        }
    }

    Ok(order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect())
}

// ---- stage 3: rerank -----------------------------------------------------

#[derive(Deserialize)]
struct RerankReply {
    #[serde(default)]
    ids: Vec<String>,
}

const RERANK_SYSTEM_PROMPT: &str = "\
You are filtering image-search candidates. Given a user query and a list of \
candidate photos (each with an id and a free-form description), identify the \
candidates that genuinely match the user's intent.\n\n\
Your output MUST be a single JSON object: {\"ids\": [\"id1\", \"id2\", ...]} \
ordered by descending relevance.\n\n\
Rules:\n\
- Be strict. Omit weak or tangential matches.\n\
- Only include ids you actually saw in the input.\n\
- If nothing matches, return {\"ids\": []}.";

fn rerank<L: ChatBackend>(llm: &L, query: &str, candidates: &[Asset]) -> Result<Vec<String>> {
    let items: Vec<serde_json::Value> = candidates
        .iter()
        .map(|a| {
            serde_json::json!({
                "id": a.id,
                "description": excerpt(
                    a.exif_info.as_ref().and_then(|e| e.description.as_deref()).unwrap_or(""),
                    DESCRIPTION_EXCERPT_CHARS,
                ),
            })
        })
        .collect();
    let user_msg = serde_json::json!({
        "query": query,
        "candidates": items,
    });
    let messages = [
        Message::system(RERANK_SYSTEM_PROMPT),
        Message::user(user_msg.to_string()),
    ];
    let raw = llm.chat_json(&messages)?;
    let reply: RerankReply = serde_json::from_str(&raw)
        .with_context(|| format!("LLM rerank reply was not valid JSON: {raw}"))?;
    Ok(reply.ids)
}

fn excerpt(s: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(max_chars.min(s.len()));
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

// ---- final assembly ------------------------------------------------------

fn take_in_order<'a>(
    candidates: &'a [Asset],
    selected_ids: &[String],
    limit: usize,
) -> Vec<&'a Asset> {
    let lookup: HashMap<&str, &Asset> = candidates.iter().map(|a| (a.id.as_str(), a)).collect();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut out = Vec::with_capacity(limit.min(selected_ids.len()));
    for id in selected_ids {
        if out.len() >= limit {
            break;
        }
        if seen.insert(id.as_str(), ()).is_some() {
            continue;
        }
        if let Some(asset) = lookup.get(id.as_str()) {
            out.push(*asset);
        }
    }
    out
}

fn dedup_preserving_order(items: Vec<String>) -> Vec<String> {
    let mut seen: HashMap<String, ()> = HashMap::new();
    let mut out = Vec::with_capacity(items.len());
    for s in items {
        if seen.insert(s.clone(), ()).is_none() {
            out.push(s);
        }
    }
    out
}

// ---- emit ----------------------------------------------------------------

fn emit<W: std::io::Write>(
    path_map: &[PathMapEntry],
    args: &AskArgs,
    selected: &[&Asset],
    out: &mut W,
) -> Result<()> {
    struct Row<'a> {
        rank: usize,
        asset: &'a Asset,
        local: PathBuf,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(selected.len());
    for (i, asset) in selected.iter().enumerate() {
        let Some(local) = path_map::translate(&asset.original_path, path_map) else {
            eprintln!(
                "warn: no path mapping for {} (asset {})",
                asset.original_path, asset.id
            );
            continue;
        };
        if args.verify && !local.exists() {
            eprintln!(
                "warn: local file missing for asset {}: {}",
                asset.id,
                local.display()
            );
            continue;
        }
        rows.push(Row {
            rank: i + 1,
            asset,
            local,
        });
    }

    match args.format {
        OutputFormat::Paths => {
            for row in &rows {
                writeln!(out, "{}", row.local.display())?;
            }
        }
        OutputFormat::Json => {
            for row in &rows {
                let exif = row.asset.exif_info.as_ref();
                let obj = serde_json::json!({
                    "rank": row.rank,
                    "id": row.asset.id,
                    "type": row.asset.asset_type,
                    "localPath": row.local.to_string_lossy(),
                    "originalPath": row.asset.original_path,
                    "originalFileName": row.asset.original_file_name,
                    "localDateTime": row.asset.local_date_time,
                    "city": exif.and_then(|e| e.city.clone()),
                    "state": exif.and_then(|e| e.state.clone()),
                    "country": exif.and_then(|e| e.country.clone()),
                    "description": exif.and_then(|e| e.description.clone()),
                });
                writeln!(out, "{}", serde_json::to_string(&obj)?)?;
            }
        }
        OutputFormat::Table => {
            let headers = ["#", "TYPE", "TAKEN", "LOCATION", "PATH"];
            let mut widths = [
                headers[0].len(),
                headers[1].len(),
                headers[2].len(),
                headers[3].len(),
                0,
            ];
            let mut data: Vec<[String; 5]> = Vec::with_capacity(rows.len());
            for row in &rows {
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
                        [&e.city, &e.state, &e.country]
                            .into_iter()
                            .filter_map(|x| x.as_deref())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                let r = [
                    row.rank.to_string(),
                    row.asset.asset_type.clone(),
                    taken,
                    location,
                    row.local.display().to_string(),
                ];
                for (i, s) in r.iter().enumerate() {
                    widths[i] = widths[i].max(s.chars().count());
                }
                data.push(r);
            }
            writeln!(
                out,
                "{:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}  {}",
                headers[0],
                headers[1],
                headers[2],
                headers[3],
                headers[4],
                w0 = widths[0],
                w1 = widths[1],
                w2 = widths[2],
                w3 = widths[3],
            )?;
            for r in &data {
                writeln!(
                    out,
                    "{:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}  {}",
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    w0 = widths[0],
                    w1 = widths[1],
                    w2 = widths[2],
                    w3 = widths[3],
                )?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AssetsBucket, ExifInfo, SearchResponse};
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

    struct FakeLlm {
        replies: RefCell<Vec<String>>,
        calls: RefCell<Vec<Vec<Message>>>,
    }
    impl FakeLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: RefCell::new(replies.iter().map(|s| s.to_string()).collect()),
                calls: RefCell::new(vec![]),
            }
        }
    }
    impl ChatBackend for FakeLlm {
        fn chat_json(&self, messages: &[Message]) -> Result<String> {
            self.calls.borrow_mut().push(messages.to_vec());
            if self.replies.borrow().is_empty() {
                bail!("FakeLlm out of canned replies");
            }
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    fn asset(id: &str, path: &str, description: &str) -> Asset {
        Asset {
            id: id.into(),
            original_path: path.into(),
            original_file_name: path.rsplit('/').next().unwrap_or(path).into(),
            asset_type: "IMAGE".into(),
            file_created_at: Some("2024-03-05T02:42:05Z".into()),
            local_date_time: Some("2024-03-05T10:42:05Z".into()),
            checksum: String::new(),
            exif_info: Some(ExifInfo {
                city: Some("Kangqiao".into()),
                state: Some("Shanghai".into()),
                country: Some("PRC".into()),
                latitude: None,
                longitude: None,
                description: Some(description.into()),
            }),
        }
    }

    fn bucket(items: Vec<Asset>) -> SearchResponse {
        let n = items.len() as u32;
        SearchResponse {
            assets: AssetsBucket {
                total: n,
                count: n,
                items,
                next_page: None,
            },
        }
    }

    fn cfg_with_map() -> Config {
        Config {
            server_url: "x".into(),
            api_key: "k".into(),
            path_map: vec![PathMapEntry {
                server: "/mnt/qnap".into(),
                local: "/home/u/Photos".into(),
            }],
            timeout_secs: 60,
            llm: None,
        }
    }

    // ---- keyword expansion parsing ----

    #[test]
    fn expand_parses_keywords_object() {
        let llm = FakeLlm::new(&[r#"{"keywords":["大象","草原","象群"]}"#]);
        let got = expand_keywords(&llm, "大象").unwrap();
        assert_eq!(got, vec!["大象", "草原", "象群"]);
    }

    #[test]
    fn expand_trims_and_drops_empties() {
        let llm = FakeLlm::new(&[r#"{"keywords":["  大象 ","",  "  "  ,"草原"]}"#]);
        let got = expand_keywords(&llm, "x").unwrap();
        assert_eq!(got, vec!["大象", "草原"]);
    }

    #[test]
    fn expand_caps_at_16() {
        let many: Vec<String> = (0..30).map(|i| format!("k{i}")).collect();
        let json = serde_json::json!({ "keywords": many }).to_string();
        let llm = FakeLlm::new(&[&json]);
        let got = expand_keywords(&llm, "x").unwrap();
        assert_eq!(got.len(), MAX_KEYWORDS);
    }

    #[test]
    fn expand_dedupes_after_trim() {
        let llm = FakeLlm::new(&[r#"{"keywords":["大象","大象","象群","大象"]}"#]);
        let got = expand_keywords(&llm, "x").unwrap();
        assert_eq!(got, vec!["大象", "象群"]);
    }

    #[test]
    fn expand_errors_on_garbage_reply() {
        let llm = FakeLlm::new(&["not json at all"]);
        let err = expand_keywords(&llm, "x").unwrap_err().to_string();
        assert!(err.contains("not valid JSON"), "got: {err}");
    }

    // ---- collect_candidates ----

    #[test]
    fn collect_unions_keywords_and_dedupes_by_id() {
        // kw "大象" → assets a, b
        // kw "草原" → assets b, c  (b is dup)
        let search = FakeSearch::new(vec![
            bucket(vec![
                asset("a", "/mnt/qnap/a.jpg", "大象在草原上"),
                asset("b", "/mnt/qnap/b.jpg", "象群迁徙"),
            ]),
            bucket(vec![
                asset("b", "/mnt/qnap/b.jpg", "象群迁徙"),
                asset("c", "/mnt/qnap/c.jpg", "非洲草原"),
            ]),
        ]);
        let got = collect_candidates(&search, &["大象".into(), "草原".into()]).unwrap();
        let ids: Vec<&str> = got.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
        let calls = search.calls.borrow();
        assert_eq!(calls.len(), 2);
        // Each call sends one keyword in description, withExif=true.
        assert_eq!(calls[0].description.as_deref(), Some("大象"));
        assert_eq!(calls[0].with_exif, Some(true));
        assert_eq!(calls[1].description.as_deref(), Some("草原"));
    }

    #[test]
    fn collect_stops_at_max_candidates() {
        // First keyword already overflows MAX_CANDIDATES.
        let many: Vec<Asset> = (0..(MAX_CANDIDATES + 50))
            .map(|i| asset(&format!("a{i}"), &format!("/mnt/qnap/{i}.jpg"), "x"))
            .collect();
        let search = FakeSearch::new(vec![bucket(many)]);
        let got = collect_candidates(&search, &["x".into(), "y".into()]).unwrap();
        assert_eq!(got.len(), MAX_CANDIDATES);
        // Should not have asked Immich for the 2nd keyword.
        assert_eq!(search.calls.borrow().len(), 1);
    }

    // ---- rerank ----

    #[test]
    fn rerank_parses_ids_and_includes_query_and_truncated_desc() {
        let big_desc = "象".repeat(DESCRIPTION_EXCERPT_CHARS + 500);
        let candidates = vec![
            asset("a", "/mnt/qnap/a.jpg", "短描述"),
            asset("b", "/mnt/qnap/b.jpg", &big_desc),
        ];
        let llm = FakeLlm::new(&[r#"{"ids":["b","a"]}"#]);
        let got = rerank(&llm, "找大象", &candidates).unwrap();
        assert_eq!(got, vec!["b", "a"]);
        // Verify the prompt content included query and truncated descriptions.
        let last = llm.calls.borrow().last().unwrap().clone();
        let user = last.iter().find(|m| m.role == "user").unwrap();
        assert!(user.content.contains("找大象"));
        // Truncated to DESCRIPTION_EXCERPT_CHARS + ellipsis.
        assert!(user.content.contains('…'));
        // Big desc should NOT be present in full.
        assert!(!user
            .content
            .contains(&"象".repeat(DESCRIPTION_EXCERPT_CHARS + 100)));
    }

    #[test]
    fn rerank_handles_empty_ids() {
        let candidates = vec![asset("a", "/mnt/qnap/a.jpg", "x")];
        let llm = FakeLlm::new(&[r#"{"ids":[]}"#]);
        let got = rerank(&llm, "q", &candidates).unwrap();
        assert!(got.is_empty());
    }

    // ---- take_in_order ----

    #[test]
    fn take_in_order_preserves_llm_ranking_and_drops_unknowns() {
        let a = asset("a", "/mnt/qnap/a.jpg", "x");
        let b = asset("b", "/mnt/qnap/b.jpg", "y");
        let c = asset("c", "/mnt/qnap/c.jpg", "z");
        let candidates = vec![a.clone(), b.clone(), c.clone()];
        let selected: Vec<String> = vec!["c".into(), "ghost".into(), "a".into(), "c".into()];
        let got = take_in_order(&candidates, &selected, 10);
        let ids: Vec<&str> = got.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "a"]);
    }

    #[test]
    fn take_in_order_respects_limit() {
        let candidates: Vec<Asset> = (0..5)
            .map(|i| asset(&format!("a{i}"), &format!("/mnt/qnap/{i}.jpg"), "x"))
            .collect();
        let selected: Vec<String> = candidates.iter().map(|a| a.id.clone()).collect();
        let got = take_in_order(&candidates, &selected, 2);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].id, "a0");
        assert_eq!(got[1].id, "a1");
    }

    // ---- end-to-end run_with ----

    fn args() -> AskArgs {
        AskArgs {
            query: "非洲草原大象".into(),
            limit: 50,
            format: OutputFormat::Paths,
            verify: false,
        }
    }

    #[test]
    fn run_with_full_flow_emits_paths() {
        let search = FakeSearch::new(vec![
            // for keyword "大象"
            bucket(vec![
                asset("a", "/mnt/qnap/elephants.jpg", "非洲草原大象在迁徙"),
                asset("b", "/mnt/qnap/zebras.jpg", "非洲斑马"),
            ]),
            // for keyword "草原"
            bucket(vec![
                asset("b", "/mnt/qnap/zebras.jpg", "非洲斑马"),
                asset("c", "/mnt/qnap/cat.jpg", "猫坐在窗台上"),
            ]),
        ]);
        let llm = FakeLlm::new(&[
            r#"{"keywords":["大象","草原"]}"#,
            // rerank: only 'a' truly matches the query.
            r#"{"ids":["a"]}"#,
        ]);
        let mut buf = Vec::new();
        run_with(&cfg_with_map(), &search, &llm, args(), &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out.trim(), "/home/u/Photos/elephants.jpg");
    }

    #[test]
    fn run_with_json_format_includes_rank_and_description() {
        let search = FakeSearch::new(vec![bucket(vec![asset(
            "a",
            "/mnt/qnap/elephants.jpg",
            "非洲草原大象迁徙的壮观场景",
        )])]);
        let llm = FakeLlm::new(&[r#"{"keywords":["大象"]}"#, r#"{"ids":["a"]}"#]);
        let mut a = args();
        a.format = OutputFormat::Json;
        let mut buf = Vec::new();
        run_with(&cfg_with_map(), &search, &llm, a, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let line = out.lines().next().expect("expected one line");
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(parsed["rank"], 1);
        assert_eq!(parsed["id"], "a");
        assert_eq!(parsed["localPath"], "/home/u/Photos/elephants.jpg");
        assert_eq!(parsed["description"], "非洲草原大象迁徙的壮观场景");
    }

    #[test]
    fn run_with_table_format_has_header_with_rank_column() {
        let search = FakeSearch::new(vec![bucket(vec![asset(
            "a",
            "/mnt/qnap/x.jpg",
            "non-empty",
        )])]);
        let llm = FakeLlm::new(&[r#"{"keywords":["x"]}"#, r#"{"ids":["a"]}"#]);
        let mut a = args();
        a.format = OutputFormat::Table;
        let mut buf = Vec::new();
        run_with(&cfg_with_map(), &search, &llm, a, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let first = out.lines().next().unwrap();
        assert!(first.starts_with('#'), "got: {first}");
        assert!(first.contains("TYPE") && first.contains("PATH"));
    }

    #[test]
    fn run_with_empty_query_errors() {
        let search = FakeSearch::new(vec![]);
        let llm = FakeLlm::new(&[]);
        let mut a = args();
        a.query = "   ".into();
        let mut buf = Vec::new();
        let err = run_with(&cfg_with_map(), &search, &llm, a, &mut buf)
            .unwrap_err()
            .to_string();
        assert!(err.contains("query is empty"));
    }

    #[test]
    fn run_with_no_candidates_returns_quietly() {
        // Keyword expansion succeeds, but every substring search misses.
        let search = FakeSearch::new(vec![bucket(vec![]), bucket(vec![])]);
        let llm = FakeLlm::new(&[r#"{"keywords":["a","b"]}"#]);
        let mut buf = Vec::new();
        run_with(&cfg_with_map(), &search, &llm, args(), &mut buf).unwrap();
        assert!(String::from_utf8(buf).unwrap().is_empty());
    }

    #[test]
    fn excerpt_truncates_long_strings_with_ellipsis() {
        let s = "a".repeat(100);
        let got = excerpt(&s, 20);
        // 20 chars + 1 ellipsis.
        assert_eq!(got.chars().count(), 21);
        assert!(got.ends_with('…'));
    }

    #[test]
    fn excerpt_keeps_short_strings_intact() {
        assert_eq!(excerpt("hi", 100), "hi");
    }
}
