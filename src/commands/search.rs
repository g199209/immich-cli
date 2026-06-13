use crate::client::{
    CaptionBackend, ImmichClient, InfoBackend, PlacesBackend, SearchBackend, StacksBackend,
};
use crate::config::{Config, PathMapEntry};
use crate::description_search::{self, HardFilters, RerankCandidate, RerankLocation};
use crate::llm::{ChatBackend, MultiImageVisionLlm, OpenAiClient};
use crate::models::{Asset, SearchRequest, SearchResponse, Stack};
use crate::path_map;
use crate::places::{self, Admin2Lookup, PlaceMatch};
use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

const DEFAULT_FILTER_LIMIT: u32 = 1000;
const DEFAULT_QUERY_LIMIT: u32 = 24;
const MAX_QUERY_LIMIT: u32 = 48;
/// Cap on per-keyword Immich hits before dedup. Independent of `--limit`
/// so a broad `--limit 48` query doesn't fan out to 48×N keyword pages
/// when the rerank pool stays bounded.
const MAX_PER_KEYWORD_HITS: u32 = 20;
/// Parallelism for the thumbnail-fetch phase before vision rerank.
/// Thumbnails are pre-rendered on disk server-side, so the call is
/// effectively a static file fetch. 32 workers comfortably saturates a
/// LAN/NFS deployment and keeps the round-trip from dominating overall
/// search latency at the new `MAX_QUERY_LIMIT` of 48.
const THUMBNAIL_FETCH_PARALLELISM: usize = 32;

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Natural-language query. With `[llm]` configured, the CLI collects
    /// candidates from Immich smart search (CLIP image similarity) plus
    /// LLM-expanded description keyword searches, dedupes them, enriches
    /// them with metadata, then asks the LLM to return the top results.
    ///
    /// With `--description-only`, the CLIP path is skipped. If no
    /// `[llm]` block is configured, it gracefully degrades to smart search only.
    #[arg(short, long)]
    pub query: Option<String>,

    /// Skip the CLIP path; only run the LLM-mediated description search.
    /// Requires `[llm]` in config; errors otherwise. Has no effect when
    /// `--query` is unset.
    #[arg(long, default_value_t = false)]
    pub description_only: bool,

    /// Earliest `localDateTime` to include. ISO 8601, or YYYY-MM-DD (UTC start of day).
    #[arg(long, value_name = "DATE")]
    pub taken_after: Option<String>,

    /// Latest `localDateTime` to include. ISO 8601, or YYYY-MM-DD (UTC end of day).
    #[arg(long, value_name = "DATE")]
    pub taken_before: Option<String>,

    /// Free-form natural-language place ("上海", "Shanghai Pudong",
    /// "中国 内蒙古", "Japan"). Resolved against the library's actual
    /// geocoded vocabulary via the LLM, then turned into one or more
    /// exact-match city/state/country queries against Immich. Requires
    /// `[llm]` in config.
    #[arg(long)]
    pub place: Option<String>,

    /// Substring match against text Immich's OCR detected in the image.
    /// Case-sensitive, Unicode-aware. Combines with --query and the
    /// other filters.
    #[arg(long)]
    pub ocr: Option<String>,

    /// Restrict by asset type.
    #[arg(long, value_enum)]
    pub r#type: Option<AssetTypeArg>,

    /// Maximum results to return. Defaults to 24 for -q searches and 1000
    /// for filter-only searches. With -q, the maximum accepted value is 48
    /// — the cap also bounds the vision rerank input, so raising it costs
    /// extra thumbnail fetches and vision tokens per query.
    #[arg(long)]
    pub limit: Option<u32>,

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

    /// Search the archive bucket instead of the timeline. By default the
    /// CLI lets the server fall back to its timeline default, so archived
    /// assets are hidden. Setting this sends `visibility=archive`, which
    /// returns ONLY archived assets — Immich ≥ v2.7.5 no longer supports a
    /// single request spanning both buckets, so "include" is a slight
    /// misnomer kept for CLI backwards compatibility.
    #[arg(long)]
    pub include_archived: bool,

    /// Include assets that are non-primary members of a stack. By default the
    /// CLI fetches /api/stacks and hides stacked members from results
    /// (mirroring how Immich's web timeline collapses a stack to its cover),
    /// so stacking redundant shots is a non-destructive way to keep them out
    /// of search. Pass this to disable that filtering and return every match.
    #[arg(long)]
    pub include_stacked: bool,

    /// Print a detailed trace of what the CLI did to stderr: vocabulary
    /// size, full LLM prompt + raw reply, parsed matches, and per-place
    /// Immich call counts. Useful when `--place` resolution is surprising.
    #[arg(short, long)]
    pub verbose: bool,
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
            || non_blank(&self.place)
            || non_blank(&self.ocr)
            || self.r#type.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        if self.limit == Some(0) {
            bail!("--limit must be > 0");
        }
        if !self.has_filter() {
            bail!(
                "search requires at least one filter: --query, --taken-after, \
                 --taken-before, --place, --ocr, or --type"
            );
        }
        if non_blank(&self.query) && self.effective_limit() > MAX_QUERY_LIMIT {
            bail!("--limit cannot exceed {MAX_QUERY_LIMIT} when -q/--query is set");
        }
        Ok(())
    }

    pub fn effective_limit(&self) -> u32 {
        self.limit.unwrap_or_else(|| {
            if non_blank(&self.query) {
                DEFAULT_QUERY_LIMIT
            } else {
                DEFAULT_FILTER_LIMIT
            }
        })
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

/// Collect the ids of every *non-primary* stack member. These are the shots
/// Immich's web timeline tucks behind a stack cover; the search command drops
/// them so stacked duplicates stop surfacing. The primary (cover) asset of
/// each stack is deliberately kept so the group still has one representative.
fn hidden_stacked_ids(stacks: &[Stack]) -> HashSet<String> {
    let mut hidden = HashSet::new();
    for stack in stacks {
        for member in &stack.assets {
            if member.id != stack.primary_asset_id {
                hidden.insert(member.id.clone());
            }
        }
    }
    hidden
}

/// Wraps a [`SearchBackend`] and strips stacked non-primary assets from every
/// response. Both the CLIP fetch and the description keyword fan-out funnel
/// through `SearchBackend::search`, so wrapping the backend filters every
/// search path in one place. The hidden set is pre-computed from `/api/stacks`
/// because the search endpoints never report stack membership themselves.
struct StackFilteredBackend<'a, S: SearchBackend> {
    inner: &'a S,
    hidden: HashSet<String>,
}

impl<'a, S: SearchBackend> StackFilteredBackend<'a, S> {
    fn new(inner: &'a S, hidden: HashSet<String>) -> Self {
        Self { inner, hidden }
    }
}

impl<S: SearchBackend> SearchBackend for StackFilteredBackend<'_, S> {
    fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
        let mut resp = self.inner.search(req)?;
        if !self.hidden.is_empty() {
            resp.assets.items.retain(|a| !self.hidden.contains(&a.id));
        }
        Ok(resp)
    }
}

pub fn run(cfg: &Config, args: SearchArgs) -> Result<()> {
    args.validate()?;
    let client = ImmichClient::new(cfg)?;
    let llm = cfg.llm.as_ref().map(OpenAiClient::new).transpose()?;
    let lookup = match places::default_admin2_lookup_path() {
        Some(p) => places::load_admin2_lookup(&p)?,
        None => Admin2Lookup::new(),
    };

    // Stacking is the user's lever for keeping near-duplicate shots out of
    // search: hide every non-primary stack member unless asked to keep them.
    // Failure here is non-fatal — degrade to an unfiltered search with a warning
    // rather than blocking the whole query on the stacks endpoint.
    let hidden = if args.include_stacked {
        HashSet::new()
    } else {
        match client.stacks() {
            Ok(stacks) => {
                let hidden = hidden_stacked_ids(&stacks);
                if args.verbose {
                    eprintln!(
                        "[verbose] search: {} stack(s) → hiding {} non-primary asset(s)",
                        stacks.len(),
                        hidden.len()
                    );
                }
                hidden
            }
            Err(e) => {
                eprintln!(
                    "warn: could not fetch /api/stacks ({e:#}); \
                     results may include stacked duplicates"
                );
                HashSet::new()
            }
        }
    };
    let search_be = StackFilteredBackend::new(&client, hidden);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let backends = Backends {
        search: &search_be,
        places: &client,
        info: &client,
        thumbs: &client,
        llm: llm.as_ref(),
    };
    run_with(cfg, &backends, &lookup, &args, &mut out)
}

/// Bundle of every backend the search pipeline talks to. Grouped to
/// keep argument lists manageable: layers below the top-level entry
/// each consume a subset (e.g. `perform_query_search` ignores `places`).
/// `llm` is optional because filter-only and CLIP-only paths never need it;
/// `thumbs` is always present because vision rerank needs it whenever the
/// query path runs, and in production it's the same `ImmichClient`.
pub struct Backends<'a, S, P, I, T, L> {
    pub search: &'a S,
    pub places: &'a P,
    pub info: &'a I,
    pub thumbs: &'a T,
    pub llm: Option<&'a L>,
}

/// Test/library entry. Decouples the runtime from concrete clients so
/// we can swap in fakes. `L` must implement both `ChatBackend` (for
/// keyword expansion / place resolution) and `MultiImageVisionLlm` (for
/// rerank) — in production both are the same `OpenAiClient`.
pub fn run_with<S, P, I, T, L, W>(
    cfg: &Config,
    backends: &Backends<'_, S, P, I, T, L>,
    admin2_lookup: &Admin2Lookup,
    args: &SearchArgs,
    out: &mut W,
) -> Result<()>
where
    S: SearchBackend,
    P: PlacesBackend,
    I: InfoBackend,
    T: CaptionBackend + Sync,
    L: ChatBackend + MultiImageVisionLlm,
    W: std::io::Write,
{
    let result = perform_search(cfg, backends, admin2_lookup, args)?;
    emit_to_writer(&cfg.path_map, args, &result, out)
}

/// Top-level dispatcher: resolve `--place` (if set) to one or more
/// concrete (city, state, country) tuples. Query searches build one
/// global candidate pool and ask the LLM to rerank it; filter-only
/// searches run one metadata fetch per place and merge those lists.
fn perform_search<S, P, I, T, L>(
    cfg: &Config,
    backends: &Backends<'_, S, P, I, T, L>,
    admin2_lookup: &Admin2Lookup,
    args: &SearchArgs,
) -> Result<FetchResult>
where
    S: SearchBackend,
    P: PlacesBackend,
    I: InfoBackend,
    T: CaptionBackend + Sync,
    L: ChatBackend + MultiImageVisionLlm,
{
    let places = resolve_places(backends.places, backends.llm, admin2_lookup, args)?;
    let query = cleaned(&args.query);
    if let Some(q) = query.as_deref() {
        return perform_query_search(cfg, backends, admin2_lookup, args, &places, q);
    }

    if places.len() == 1 {
        let r = fetch_assets(backends.search, args, &places[0])?;
        if args.verbose {
            eprintln!(
                "[verbose] search: returned {} asset(s){}",
                r.assets.len(),
                if r.truncated { " (truncated)" } else { "" }
            );
        }
        return Ok(r);
    }

    // Multiple resolved places: run the full flow per place and merge.
    let mut per_place: Vec<Vec<Asset>> = Vec::with_capacity(places.len());
    let mut any_truncated = false;
    for (i, p) in places.iter().enumerate() {
        let r = fetch_assets(backends.search, args, p)?;
        if args.verbose {
            eprintln!(
                "[verbose] search: place[{}] (country={:?} state={:?} city={:?}) returned {} asset(s){}",
                i,
                p.country,
                p.state,
                p.city,
                r.assets.len(),
                if r.truncated { " (truncated)" } else { "" }
            );
        }
        any_truncated |= r.truncated;
        per_place.push(r.assets);
    }
    let merged = description_search::rrf_merge(&per_place);
    let limit = args.effective_limit() as usize;
    let truncated = any_truncated || merged.len() > limit;
    let assets: Vec<Asset> = merged.into_iter().take(limit).collect();
    if args.verbose {
        eprintln!(
            "[verbose] search: RRF-merged {} place(s) → {} asset(s){}",
            places.len(),
            assets.len(),
            if truncated { " (truncated)" } else { "" }
        );
    }
    Ok(FetchResult { assets, truncated })
}

/// Resolve `--place "..."` to one or more concrete `PlaceMatch`es.
/// Returns a single empty match (no geo filter) when `--place` is unset.
fn resolve_places<P, L>(
    places_be: &P,
    llm: Option<&L>,
    admin2_lookup: &Admin2Lookup,
    args: &SearchArgs,
) -> Result<Vec<PlaceMatch>>
where
    P: PlacesBackend,
    L: ChatBackend,
{
    let Some(input) = cleaned(&args.place) else {
        return Ok(vec![PlaceMatch::default()]);
    };
    let llm = llm.ok_or_else(|| {
        anyhow::anyhow!(
            "--place requires an [llm] section in config.toml \
             (base_url, api_key, model) to resolve the free-form input"
        )
    })?;
    let matches = places::resolve_place(places_be, llm, &input, admin2_lookup, args.verbose)?;
    if matches.is_empty() {
        bail!(
            "no place in the library matches `{input}` — \
             try a different wording or check the library's geocoding"
        );
    }
    if args.verbose {
        eprintln!(
            "[verbose] search: --place {input:?} → {} place match(es) to query",
            matches.len()
        );
    }
    Ok(matches)
}

fn perform_query_search<S, P, I, T, L>(
    cfg: &Config,
    backends: &Backends<'_, S, P, I, T, L>,
    admin2_lookup: &Admin2Lookup,
    args: &SearchArgs,
    places: &[PlaceMatch],
    query: &str,
) -> Result<FetchResult>
where
    S: SearchBackend,
    P: PlacesBackend,
    I: InfoBackend,
    T: CaptionBackend + Sync,
    L: ChatBackend + MultiImageVisionLlm,
{
    let limit = args.effective_limit() as usize;
    let Some(llm) = backends.llm else {
        if args.description_only {
            bail!(
                "--description-only requires an [llm] section in config.toml \
                 (base_url, api_key, model)"
            );
        }
        return collect_smart_only(backends.search, args, places);
    };

    if args.verbose {
        eprintln!(
            "[verbose] search: query={query:?} description_only={} places={} limit={limit}",
            args.description_only,
            places.len()
        );
    }
    let keywords = description_search::expand_keywords(llm, query, args.verbose)?;
    let mut candidates: HashMap<String, QueryCandidate> = HashMap::new();
    let mut order = Vec::new();
    let mut any_truncated = false;

    for (pi, place) in places.iter().enumerate() {
        if !args.description_only {
            let smart = fetch_assets(backends.search, args, place)?;
            any_truncated |= smart.truncated;
            if args.verbose {
                eprintln!(
                    "[verbose] search: place[{pi}] CLIP smart search → {} asset(s){}",
                    smart.assets.len(),
                    if smart.truncated { " (truncated)" } else { "" }
                );
            }
            for asset in smart.assets {
                add_smart_candidate(&mut candidates, &mut order, asset);
            }
        }

        if !keywords.is_empty() {
            let per_keyword_limit = args.effective_limit().min(MAX_PER_KEYWORD_HITS);
            if args.verbose {
                eprintln!(
                    "[verbose] search: place[{pi}] description-keyword fan-out \
                     ({} keyword(s), per-keyword limit = {per_keyword_limit})",
                    keywords.len(),
                );
            }
            let filters = build_hard_filters(args, place)?;
            let hits = description_search::collect_keyword_hits(
                backends.search,
                &keywords,
                &filters,
                per_keyword_limit,
                args.verbose,
            )?;
            if args.verbose {
                eprintln!(
                    "[verbose] search: place[{pi}] description-keyword total → {} hit(s)",
                    hits.len()
                );
            }
            for hit in hits {
                add_keyword_candidate(&mut candidates, &mut order, hit.asset, hit.keyword);
            }
        }
    }

    if args.verbose {
        let (mut smart_only, mut desc_only, mut both) = (0usize, 0usize, 0usize);
        for c in candidates.values() {
            match (c.smart, !c.matched_keywords.is_empty()) {
                (true, true) => both += 1,
                (true, false) => smart_only += 1,
                (false, true) => desc_only += 1,
                (false, false) => {}
            }
        }
        eprintln!(
            "[verbose] search: candidate pool → {} unique \
             (smart-only={smart_only}, description-only={desc_only}, both={both})",
            candidates.len()
        );
    }

    if candidates.is_empty() {
        if args.verbose {
            eprintln!("[verbose] search: candidate pool empty, skipping rerank");
        }
        return Ok(FetchResult {
            assets: vec![],
            truncated: false,
        });
    }

    let enriched = build_rerank_candidates(cfg, backends.info, admin2_lookup, &candidates, &order)?;
    let prompt_candidates: Vec<RerankCandidate> =
        enriched.iter().map(|c| c.prompt.clone()).collect();
    let asset_ids: Vec<String> = enriched.iter().map(|c| c.asset.id.clone()).collect();
    let thumbnails = fetch_thumbnails_parallel(backends.thumbs, &asset_ids, args.verbose)?;
    let selected_ids = description_search::rerank_rich_vision(
        llm,
        query,
        limit,
        &prompt_candidates,
        &thumbnails,
        args.verbose,
    )?;
    let mut by_short_id: HashMap<&str, &RichQueryCandidate> = HashMap::new();
    for c in &enriched {
        by_short_id.insert(c.prompt.id.as_str(), c);
    }
    let mut seen = HashSet::new();
    let mut assets = Vec::new();
    for id in selected_ids {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(candidate) = by_short_id.get(id.as_str()) {
            assets.push(candidate.asset.clone());
        }
        if assets.len() >= limit {
            break;
        }
    }

    let truncated = assets.len() >= limit && (any_truncated || enriched.len() > limit);
    if args.verbose {
        eprintln!(
            "[verbose] search: returning {} asset(s){}",
            assets.len(),
            if truncated { " (truncated)" } else { "" }
        );
    }
    Ok(FetchResult { assets, truncated })
}

fn collect_smart_only<S: SearchBackend>(
    search_be: &S,
    args: &SearchArgs,
    places: &[PlaceMatch],
) -> Result<FetchResult> {
    let limit = args.effective_limit() as usize;
    if args.verbose {
        eprintln!(
            "[verbose] search: smart-only path (no [llm] configured) over {} place(s), limit={limit}",
            places.len()
        );
    }
    let mut by_id: HashMap<String, Asset> = HashMap::new();
    let mut order = Vec::new();
    let mut any_truncated = false;
    for (pi, place) in places.iter().enumerate() {
        let r = fetch_assets(search_be, args, place)?;
        any_truncated |= r.truncated;
        if args.verbose {
            eprintln!(
                "[verbose] search: place[{pi}] CLIP smart search → {} asset(s){}",
                r.assets.len(),
                if r.truncated { " (truncated)" } else { "" }
            );
        }
        for asset in r.assets {
            if !by_id.contains_key(&asset.id) {
                order.push(asset.id.clone());
                by_id.insert(asset.id.clone(), asset);
            }
        }
    }
    let mut assets = Vec::new();
    for id in order {
        if let Some(asset) = by_id.remove(&id) {
            assets.push(asset);
        }
        if assets.len() >= limit {
            break;
        }
    }
    let truncated = any_truncated || by_id.len() + assets.len() > limit;
    if args.verbose {
        eprintln!(
            "[verbose] search: returning {} asset(s){}",
            assets.len(),
            if truncated { " (truncated)" } else { "" }
        );
    }
    Ok(FetchResult { assets, truncated })
}

#[derive(Debug, Clone)]
struct QueryCandidate {
    asset: Asset,
    smart: bool,
    matched_keywords: Vec<String>,
}

fn add_smart_candidate(
    candidates: &mut HashMap<String, QueryCandidate>,
    order: &mut Vec<String>,
    asset: Asset,
) {
    let id = asset.id.clone();
    if let Some(existing) = candidates.get_mut(&id) {
        existing.smart = true;
        return;
    }
    order.push(id.clone());
    candidates.insert(
        id,
        QueryCandidate {
            asset,
            smart: true,
            matched_keywords: Vec::new(),
        },
    );
}

fn add_keyword_candidate(
    candidates: &mut HashMap<String, QueryCandidate>,
    order: &mut Vec<String>,
    asset: Asset,
    keyword: String,
) {
    let id = asset.id.clone();
    if let Some(existing) = candidates.get_mut(&id) {
        if !existing.matched_keywords.contains(&keyword) {
            existing.matched_keywords.push(keyword);
        }
        return;
    }
    order.push(id.clone());
    candidates.insert(
        id,
        QueryCandidate {
            asset,
            smart: false,
            matched_keywords: vec![keyword],
        },
    );
}

#[derive(Debug, Clone)]
struct RichQueryCandidate {
    prompt: RerankCandidate,
    asset: Asset,
}

fn build_rerank_candidates(
    cfg: &Config,
    info_be: &impl InfoBackend,
    admin2_lookup: &Admin2Lookup,
    candidates: &HashMap<String, QueryCandidate>,
    order: &[String],
) -> Result<Vec<RichQueryCandidate>> {
    let mut out = Vec::with_capacity(order.len());
    for (i, id) in order.iter().enumerate() {
        let Some(candidate) = candidates.get(id) else {
            continue;
        };
        let people = configured_people_for_asset(info_be, &cfg.people, &candidate.asset.id)?;
        let mut sources = Vec::new();
        if candidate.smart {
            sources.push("smart".to_string());
        }
        if !candidate.matched_keywords.is_empty() {
            sources.push("description".to_string());
        }
        out.push(RichQueryCandidate {
            prompt: RerankCandidate {
                id: format!("c{:03}", i + 1),
                asset_type: candidate.asset.asset_type.clone(),
                taken_at: candidate.asset.local_date_time.clone(),
                location: rerank_location(&candidate.asset, admin2_lookup),
                people,
                sources,
                matched_keywords: candidate.matched_keywords.clone(),
            },
            asset: candidate.asset.clone(),
        });
    }
    Ok(out)
}

fn configured_people_for_asset(
    info_be: &impl InfoBackend,
    people_map: &std::collections::BTreeMap<String, Vec<String>>,
    asset_id: &str,
) -> Result<Vec<String>> {
    if people_map.is_empty() {
        return Ok(Vec::new());
    }
    let full_asset = info_be
        .get_asset(asset_id)
        .with_context(|| format!("failed to fetch full asset detail for {asset_id}"))?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let Some(items) = full_asset.get("people").and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    for person in items {
        let Some(name) = person
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let Some(roles) = people_map.get(name) else {
            continue;
        };
        if !seen.insert(name.to_string()) {
            continue;
        }
        if roles.is_empty() {
            out.push(name.to_string());
        } else {
            out.push(format!("{}（{}）", name, roles.join("/")));
        }
    }
    Ok(out)
}

fn rerank_location(asset: &Asset, admin2_lookup: &Admin2Lookup) -> Option<RerankLocation> {
    let exif = asset.exif_info.as_ref()?;
    let country = cleaned(&exif.country);
    let state = cleaned(&exif.state);
    let city = cleaned(&exif.city);
    if country.is_none() && state.is_none() && city.is_none() {
        return None;
    }
    let admin2 = match (&country, &state, &city) {
        (Some(country), Some(state), Some(city)) => admin2_lookup
            .get(&(country.clone(), state.clone(), city.clone()))
            .cloned(),
        _ => None,
    };
    Some(RerankLocation {
        country,
        state,
        admin2,
        city,
    })
}

/// Fetch every candidate's `~720×960` JPEG thumbnail from Immich, in
/// `THUMBNAIL_FETCH_PARALLELISM` worker threads. Order of the returned
/// vector matches `ids`. Any single failure aborts the whole batch —
/// vision rerank can't paper over a missing image, so failing fast
/// surfaces the underlying problem (auth, server down) cleanly.
fn fetch_thumbnails_parallel<T: CaptionBackend + Sync>(
    thumbs_be: &T,
    ids: &[String],
    verbose: bool,
) -> Result<Vec<Vec<u8>>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let started = std::time::Instant::now();
    let slots: Vec<Mutex<Option<Result<Vec<u8>>>>> = (0..ids.len()).map(|_| Mutex::new(None)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parallel = THUMBNAIL_FETCH_PARALLELISM.min(ids.len()).max(1);

    std::thread::scope(|scope| {
        for _ in 0..parallel {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= ids.len() {
                    break;
                }
                let res = thumbs_be
                    .thumbnail(&ids[i])
                    .with_context(|| format!("failed to fetch thumbnail for {}", ids[i]));
                *slots[i].lock().unwrap() = Some(res);
            });
        }
    });

    let mut out = Vec::with_capacity(ids.len());
    for slot in slots {
        out.push(slot.into_inner().unwrap().expect("worker filled slot")?);
    }
    if verbose {
        let total_bytes: usize = out.iter().map(|b| b.len()).sum();
        eprintln!(
            "[verbose] search: fetched {} thumbnail(s) in {} ms ({} workers, {} bytes total)",
            out.len(),
            started.elapsed().as_millis(),
            parallel,
            total_bytes
        );
    }
    Ok(out)
}

/// Build the hard-filter bundle the description path needs. Geo fields
/// come from the resolved [`PlaceMatch`]; everything else is read from
/// `args` (with the same `cleaned` + date normalization the CLIP path
/// uses, so both see identical values).
fn build_hard_filters(args: &SearchArgs, place: &PlaceMatch) -> Result<HardFilters> {
    Ok(HardFilters {
        city: place.city.clone(),
        state: place.state.clone(),
        country: place.country.clone(),
        ocr: cleaned(&args.ocr),
        taken_after: cleaned(&args.taken_after)
            .as_deref()
            .map(normalize_date_start)
            .transpose()?,
        taken_before: cleaned(&args.taken_before)
            .as_deref()
            .map(normalize_date_end)
            .transpose()?,
        asset_type: args.r#type.map(|t| t.as_api_str().to_string()),
        visibility: if args.include_archived {
            Some("archive".to_string())
        } else {
            None
        },
    })
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

pub fn fetch_assets<B: SearchBackend>(
    backend: &B,
    args: &SearchArgs,
    place: &PlaceMatch,
) -> Result<FetchResult> {
    fetch_assets_inner(backend, args, PAGE_SIZE, place)
}

fn fetch_assets_inner<B: SearchBackend>(
    backend: &B,
    args: &SearchArgs,
    page_size: u32,
    place: &PlaceMatch,
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
    let ocr = cleaned(&args.ocr);

    let limit = args.effective_limit();
    let mut collected: Vec<Asset> = Vec::with_capacity(limit as usize);
    let mut page: u32 = 1;
    let page_size = page_size.min(limit).max(1);

    loop {
        let req = SearchRequest {
            query: query.clone(),
            original_file_name: None,
            city: place.city.clone(),
            state: place.state.clone(),
            country: place.country.clone(),
            ocr: ocr.clone(),
            description: None,
            taken_after: taken_after.clone(),
            taken_before: taken_before.clone(),
            asset_type: args.r#type.map(|t| t.as_api_str().to_string()),
            visibility: if args.include_archived {
                Some("archive".to_string())
            } else {
                None
            },
            page: Some(page),
            size: Some(page_size),
            with_exif: Some(true),
        };

        let resp = backend.search(&req)?;
        let count = resp.assets.items.len();
        let has_more = resp.assets.next_page.as_ref().is_some_and(|v| !v.is_null());

        let remaining = (limit as usize).saturating_sub(collected.len());
        let take = count.min(remaining);
        // Anything not consumed in the current page, plus a known next page,
        // both mean there's more we're not fetching.
        let leftover_in_page = count > take;
        collected.extend(resp.assets.items.into_iter().take(take));

        if collected.len() as u32 >= limit {
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
            checksum: String::new(),
            exif_info: Some(ExifInfo {
                city: Some("Shanghai".into()),
                state: Some("Shanghai".into()),
                country: Some("China".into()),
                latitude: Some(31.0),
                longitude: Some(121.0),
                ..Default::default()
            }),
        }
    }

    fn default_args() -> SearchArgs {
        SearchArgs {
            query: None,
            description_only: false,
            taken_after: None,
            taken_before: None,
            place: None,
            ocr: None,
            r#type: None,
            limit: None,
            format: OutputFormat::Paths,
            verify: false,
            include_missing: false,
            include_unmapped: false,
            include_archived: false,
            include_stacked: false,
            verbose: false,
        }
    }

    fn cfg() -> Config {
        Config {
            server_url: "http://x".into(),
            api_key: "k".into(),
            path_map: pmap(),
            timeout_secs: 60,
            llm: None,
            people: std::collections::BTreeMap::new(),
        }
    }

    fn no_place() -> PlaceMatch {
        PlaceMatch::default()
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

    /// Records each call and replays canned responses in order. Uses
    /// `Mutex` rather than `RefCell` so the test backend can be `Sync`,
    /// which the search pipeline now needs for the parallel
    /// thumbnail-fetch step.
    struct FakeBackend {
        responses: Mutex<Vec<SearchResponse>>,
        calls: Mutex<Vec<SearchRequest>>,
        vocab: Vec<crate::places::CityVocabEntry>,
    }

    impl FakeBackend {
        fn new(responses: Vec<SearchResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                calls: Mutex::new(Vec::new()),
                vocab: vec![],
            }
        }
        fn with_vocab(mut self, vocab: Vec<crate::places::CityVocabEntry>) -> Self {
            self.vocab = vocab;
            self
        }
        fn calls(&self) -> Vec<SearchRequest> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl SearchBackend for FakeBackend {
        fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
            self.calls.lock().unwrap().push(req.clone());
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                anyhow::bail!("FakeBackend ran out of canned responses");
            }
            Ok(q.remove(0))
        }
    }

    impl PlacesBackend for FakeBackend {
        fn cities_vocabulary(&self) -> Result<Vec<crate::places::CityVocabEntry>> {
            Ok(self.vocab.clone())
        }
    }

    impl InfoBackend for FakeBackend {
        fn get_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "people": [
                    { "name": "测试人物甲" },
                    { "name": "未配置测试人物" }
                ]
            }))
        }

        fn albums_for_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(serde_json::json!([]))
        }

        fn ocr_for_asset(&self, _id: &str) -> Result<serde_json::Value> {
            Ok(serde_json::json!([]))
        }
    }

    impl CaptionBackend for FakeBackend {
        fn thumbnail(&self, id: &str) -> Result<Vec<u8>> {
            // Tiny deterministic blob so tests can verify presence by id.
            Ok(id.as_bytes().to_vec())
        }
        fn update_description(&self, _id: &str, _description: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Test-only sugar: wrap a single FakeBackend (which implements every
    /// backend trait needed by `search`) plus an optional FakeLlm into a
    /// `Backends`. Avoids ten-line struct literals at every call site.
    fn fb<'a>(
        b: &'a FakeBackend,
        llm: Option<&'a FakeLlm>,
    ) -> Backends<'a, FakeBackend, FakeBackend, FakeBackend, FakeBackend, FakeLlm> {
        Backends {
            search: b,
            places: b,
            info: b,
            thumbs: b,
            llm,
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
    fn validate_accepts_ocr_only() {
        let mut args = default_args();
        args.ocr = Some("hello".into());
        assert!(args.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_limit() {
        let mut args = default_args();
        args.query = Some("x".into());
        args.limit = Some(0);
        assert!(args.validate().is_err());
    }

    #[test]
    fn effective_limit_depends_on_query_mode() {
        let mut args = default_args();
        args.taken_after = Some("2025-01-01".into());
        assert_eq!(args.effective_limit(), 1000);

        args.query = Some("x".into());
        assert_eq!(args.effective_limit(), 24);

        args.limit = Some(48);
        assert!(args.validate().is_ok());

        args.limit = Some(49);
        let err = args.validate().unwrap_err().to_string();
        assert!(err.contains("cannot exceed 48"), "got: {err}");

        args.query = None;
        args.limit = Some(5000);
        assert!(args.validate().is_ok());
        assert_eq!(args.effective_limit(), 5000);
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
    fn validate_rejects_whitespace_only_place() {
        let mut args = default_args();
        args.place = Some("   ".into());
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
        // Geo now comes from a resolved PlaceMatch, so verify the other
        // string fields are still stripped of blank values when handed
        // to Immich. The geo override carries any city/state/country.
        let backend = FakeBackend::new(vec![resp(vec![], None)]);
        let mut args = default_args();
        args.taken_after = Some("2025-01-01".into());
        args.query = Some("".into());
        args.ocr = Some("   ".into());
        let place = PlaceMatch {
            country: Some("People's Republic of China".into()),
            ..Default::default()
        };
        fetch_assets(&backend, &args, &place).unwrap();
        let req = &backend.calls()[0];
        assert_eq!(req.country.as_deref(), Some("People's Republic of China"));
        assert!(req.city.is_none());
        assert!(req.query.is_none(), "blank query should be stripped");
        assert!(req.ocr.is_none(), "blank ocr should be stripped");
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
        args.limit = Some(10);

        // page_size is hard-coded in fetch_assets, so drive the inner
        // entry point directly to exercise multi-page behavior.
        let got = fetch_assets_inner(&backend, &args, 2, &no_place()).unwrap();
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
        args.query = Some("x".into());
        args.limit = Some(2);

        let got = fetch_assets_inner(&backend, &args, 10, &no_place()).unwrap();
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
        args.query = Some("x".into());
        args.limit = Some(2);
        let got = fetch_assets_inner(&backend, &args, 2, &no_place()).unwrap();
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
        args.query = Some("x".into());
        args.limit = Some(2);
        let got = fetch_assets_inner(&backend, &args, 2, &no_place()).unwrap();
        assert_eq!(got.assets.len(), 2);
        assert!(!got.truncated);
    }

    #[test]
    fn fetch_sends_geo_and_type_filters_through() {
        let backend = FakeBackend::new(vec![resp(vec![], None)]);
        let mut args = default_args();
        args.r#type = Some(AssetTypeArg::Video);
        args.taken_after = Some("2025-01-01".into());
        args.taken_before = Some("2025-12-31".into());
        let place = PlaceMatch {
            country: Some("People's Republic of China".into()),
            state: Some("Shanghai".into()),
            city: Some("Kangqiao".into()),
        };

        fetch_assets(&backend, &args, &place).unwrap();
        let calls = backend.calls();
        assert_eq!(calls[0].city.as_deref(), Some("Kangqiao"));
        assert_eq!(calls[0].state.as_deref(), Some("Shanghai"));
        assert_eq!(
            calls[0].country.as_deref(),
            Some("People's Republic of China")
        );
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
    fn fetch_sends_ocr_filter_through_trimmed() {
        let backend = FakeBackend::new(vec![resp(vec![], None)]);
        let mut args = default_args();
        args.ocr = Some("  上海市老年基金会  ".into());
        fetch_assets(&backend, &args, &no_place()).unwrap();
        let calls = backend.calls();
        // Whitespace is stripped just like every other string filter.
        assert_eq!(calls[0].ocr.as_deref(), Some("上海市老年基金会"));
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
        let err = fetch_assets(&ErrBackend, &args, &no_place())
            .unwrap_err()
            .to_string();
        assert!(err.contains("immich exploded"), "got: {err}");
    }

    // ---- stack filtering -----------------------------------------------

    fn stack_member(id: &str) -> crate::models::StackMember {
        crate::models::StackMember { id: id.into() }
    }

    #[test]
    fn hidden_stacked_ids_keeps_primary_drops_rest() {
        let stacks = vec![
            Stack {
                primary_asset_id: "p1".into(),
                assets: vec![stack_member("p1"), stack_member("c1"), stack_member("c2")],
            },
            Stack {
                primary_asset_id: "p2".into(),
                assets: vec![stack_member("p2"), stack_member("c3")],
            },
        ];
        let hidden = hidden_stacked_ids(&stacks);
        assert_eq!(hidden.len(), 3);
        for child in ["c1", "c2", "c3"] {
            assert!(hidden.contains(child), "expected {child} hidden");
        }
        // Covers — the assets we keep — must never be hidden.
        assert!(!hidden.contains("p1"));
        assert!(!hidden.contains("p2"));
    }

    #[test]
    fn stack_filter_drops_hidden_assets_across_pages() {
        // Page 1: cover p1, stacked child c1, unrelated x. Page 2: stacked
        // child c2, unrelated y. The cover and the non-stacked assets survive;
        // the stacked children are filtered out, and pagination keeps walking
        // so the limit fills with real results.
        let backend = FakeBackend::new(vec![
            resp(
                vec![
                    make_asset("p1", "/mnt/x/p1.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                    make_asset("c1", "/mnt/x/c1.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                    make_asset("x", "/mnt/x/x.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                ],
                Some("2"),
            ),
            resp(
                vec![
                    make_asset("c2", "/mnt/x/c2.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                    make_asset("y", "/mnt/x/y.jpg", "IMAGE", "2025-01-01T00:00:00Z"),
                ],
                None,
            ),
        ]);
        let hidden: HashSet<String> = ["c1".to_string(), "c2".to_string()].into_iter().collect();
        let filtered = StackFilteredBackend::new(&backend, hidden);
        let mut args = default_args();
        args.query = Some("x".into());
        args.limit = Some(10);
        let got = fetch_assets_inner(&filtered, &args, 3, &no_place()).unwrap();
        let ids: Vec<&str> = got.assets.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["p1", "x", "y"]);
    }

    #[test]
    fn stack_filter_with_empty_set_is_passthrough() {
        let backend = FakeBackend::new(vec![resp(
            vec![make_asset(
                "a1",
                "/mnt/x/a.jpg",
                "IMAGE",
                "2025-01-01T00:00:00Z",
            )],
            None,
        )]);
        let filtered = StackFilteredBackend::new(&backend, HashSet::new());
        let mut args = default_args();
        args.query = Some("x".into());
        let got = fetch_assets(&filtered, &args, &no_place()).unwrap();
        assert_eq!(got.assets.len(), 1);
        assert_eq!(got.assets[0].id, "a1");
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

    // ---- perform_search: CLIP + description merge logic ------------------

    struct FakeLlm {
        replies: RefCell<Vec<String>>,
        messages: RefCell<Vec<Vec<crate::llm::Message>>>,
    }
    impl FakeLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: RefCell::new(replies.iter().map(|s| s.to_string()).collect()),
                messages: RefCell::new(Vec::new()),
            }
        }
        fn messages(&self) -> Vec<Vec<crate::llm::Message>> {
            self.messages.borrow().clone()
        }
    }
    impl crate::llm::ChatBackend for FakeLlm {
        fn chat_json(&self, messages: &[crate::llm::Message]) -> Result<String> {
            self.messages.borrow_mut().push(messages.to_vec());
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    impl crate::llm::MultiImageVisionLlm for FakeLlm {
        fn pick_best(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _images: &[(Vec<u8>, &str)],
            _max_tokens: u32,
        ) -> Result<String> {
            Ok(self.replies.borrow_mut().remove(0))
        }
        fn rank_images(
            &self,
            system_prompt: &str,
            user_prompt: &str,
            items: &[(String, Vec<u8>, &str)],
            _max_tokens: u32,
        ) -> Result<String> {
            // Record a synthetic "user message" so the existing
            // `messages()` assertion helpers can inspect what the rerank
            // saw — labels go in the last message's content, prefixed
            // with the per-image label so substring asserts keep working.
            let mut content = String::new();
            content.push_str(user_prompt);
            for (label, bytes, _mime) in items {
                content.push('\n');
                content.push_str(label);
                content.push_str(&format!(" [thumbnail {} bytes]", bytes.len()));
            }
            self.messages.borrow_mut().push(vec![
                crate::llm::Message::system(system_prompt.to_string()),
                crate::llm::Message::user(content),
            ]);
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    fn asset_with_desc(id: &str, path: &str, desc: &str) -> Asset {
        let mut a = make_asset(id, path, "IMAGE", "2025-01-01T00:00:00Z");
        if let Some(e) = a.exif_info.as_mut() {
            e.description = Some(desc.into());
        }
        a
    }

    fn sample_vocab() -> Vec<crate::places::CityVocabEntry> {
        vec![
            crate::places::CityVocabEntry {
                country: "People's Republic of China".into(),
                state: "Shanghai".into(),
                city: "Pudong".into(),
                admin2: None,
            },
            crate::places::CityVocabEntry {
                country: "People's Republic of China".into(),
                state: "Zhejiang".into(),
                city: "Andong".into(),
                admin2: None,
            },
        ]
    }

    #[test]
    fn perform_search_without_query_does_filter_only_no_llm() {
        let backend = FakeBackend::new(vec![resp(
            vec![asset_with_desc("a", "/mnt/x/a.jpg", "desc-a")],
            None,
        )]);
        let mut args = default_args();
        args.taken_after = Some("2025-01-01".into());
        // LLM provided but should NOT be used (no -q, no --place).
        let llm = FakeLlm::new(&[]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        assert_eq!(got.assets.len(), 1);
        // Only the CLIP/metadata path call, never the description workflow.
        assert_eq!(backend.calls().len(), 1);
    }

    #[test]
    fn perform_search_with_query_no_llm_runs_clip_only() {
        let backend = FakeBackend::new(vec![resp(
            vec![asset_with_desc("a", "/mnt/x/a.jpg", "desc-a")],
            None,
        )]);
        let mut args = default_args();
        args.query = Some("elephants".into());
        let got = perform_search(&cfg(), &fb(&backend, None), &Admin2Lookup::new(), &args).unwrap();
        assert_eq!(got.assets.len(), 1);
        assert_eq!(backend.calls().len(), 1);
        // The single call was to /smart (query is set).
        assert_eq!(backend.calls()[0].query.as_deref(), Some("elephants"));
    }

    #[test]
    fn perform_search_description_only_requires_llm() {
        let backend = FakeBackend::new(vec![]);
        let mut args = default_args();
        args.query = Some("elephants".into());
        args.description_only = true;
        let err = perform_search(&cfg(), &fb(&backend, None), &Admin2Lookup::new(), &args)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[llm]"), "got: {err}");
    }

    #[test]
    fn perform_search_description_only_skips_clip_path() {
        let backend = FakeBackend::new(vec![
            // Only the per-keyword metadata search; no CLIP call.
            resp(vec![asset_with_desc("a", "/mnt/x/a.jpg", "desc-a")], None),
        ]);
        let mut args = default_args();
        args.query = Some("elephants".into());
        args.description_only = true;
        let llm = FakeLlm::new(&[r#"{"keywords":["大象"]}"#, r#"{"ids":["c001"]}"#]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        assert_eq!(got.assets.len(), 1);
        assert_eq!(got.assets[0].id, "a");
        // Exactly 1 search call: per-keyword. No CLIP call.
        let calls = backend.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].description.as_deref(), Some("大象"));
        assert!(calls[0].query.is_none(), "CLIP query must not be sent");
    }

    #[test]
    fn perform_search_combined_uses_single_llm_rerank_over_deduped_candidates() {
        // CLIP returns: [a, b] (in that rank order)
        // Keyword search returns: [c, b]. Candidate order is a=c001,
        // b=c002, c=c003; b carries both smart + description evidence.
        let backend = FakeBackend::new(vec![
            // First call: CLIP smart search (query is set)
            resp(
                vec![
                    asset_with_desc("a", "/mnt/x/a.jpg", "x"),
                    asset_with_desc("b", "/mnt/x/b.jpg", "x"),
                ],
                None,
            ),
            // Then one metadata call per keyword (just one keyword here)
            resp(
                vec![
                    asset_with_desc("c", "/mnt/x/c.jpg", "desc-c"),
                    asset_with_desc("b", "/mnt/x/b.jpg", "desc-b"),
                ],
                None,
            ),
        ]);
        let mut args = default_args();
        args.query = Some("elephants".into());
        let llm = FakeLlm::new(&[r#"{"keywords":["大象"]}"#, r#"{"ids":["c003","c002"]}"#]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        let ids: Vec<&str> = got.assets.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "b"]);
    }

    #[test]
    fn perform_search_rerank_prompt_includes_metadata_people_and_evidence_without_paths() {
        let backend = FakeBackend::new(vec![
            resp(
                vec![asset_with_desc("a", "/mnt/x/a.jpg", "smart-desc")],
                None,
            ),
            resp(
                vec![asset_with_desc("a", "/mnt/x/a.jpg", "keyword-desc")],
                None,
            ),
        ]);
        let mut cfg = cfg();
        cfg.people
            .insert("测试人物甲".into(), vec!["女儿".into(), "孩子".into()]);
        let mut lookup = Admin2Lookup::new();
        lookup.insert(
            ("China".into(), "Shanghai".into(), "Shanghai".into()),
            "Shanghai Shi".into(),
        );
        let mut args = default_args();
        args.query = Some("孩子".into());
        let llm = FakeLlm::new(&[r#"{"keywords":["孩子"]}"#, r#"{"ids":["c001"]}"#]);

        let got = perform_search(&cfg, &fb(&backend, Some(&llm)), &lookup, &args).unwrap();
        assert_eq!(got.assets[0].id, "a");

        let messages = llm.messages();
        assert_eq!(messages.len(), 2);
        // messages[0] = keyword expansion (chat_json), messages[1] = vision rerank.
        let rerank_user = &messages[1][1].content;
        // Compact label format: `key=value | key=value`; thumbnails are
        // logged inline so we can still assert the candidate was sent.
        assert!(rerank_user.contains("id=c001"), "{rerank_user}");
        assert!(rerank_user.contains("type=IMAGE"), "{rerank_user}");
        assert!(rerank_user.contains("taken="), "{rerank_user}");
        assert!(
            rerank_user.contains("loc=China/Shanghai/Shanghai Shi/Shanghai"),
            "{rerank_user}"
        );
        assert!(
            rerank_user.contains("people=测试人物甲（女儿/孩子）"),
            "{rerank_user}"
        );
        assert!(
            rerank_user.contains("sources=smart+description"),
            "{rerank_user}"
        );
        assert!(rerank_user.contains("matched=孩子"), "{rerank_user}");
        assert!(rerank_user.contains("[thumbnail "), "{rerank_user}");
        assert!(!rerank_user.contains("/mnt/x/a.jpg"), "{rerank_user}");
        // The vision rerank prompt's text body should NOT contain the raw
        // description excerpt — that's the whole point of switching to the
        // thumbnail.
        assert!(!rerank_user.contains("smart-desc"), "{rerank_user}");
        assert!(!rerank_user.contains("keyword-desc"), "{rerank_user}");
    }

    #[test]
    fn perform_search_hard_filters_apply_to_both_paths() {
        // --place "上海浦东" resolves to a single match; both the CLIP
        // call and the description metadata call must carry the geo.
        let backend = FakeBackend::new(vec![
            resp(vec![asset_with_desc("a", "/mnt/x/a.jpg", "x")], None),
            resp(vec![asset_with_desc("b", "/mnt/x/b.jpg", "desc")], None),
        ])
        .with_vocab(sample_vocab());
        let mut args = default_args();
        args.query = Some("x".into());
        args.place = Some("上海浦东".into());
        let llm = FakeLlm::new(&[
            // 1. place resolution → 1 match
            r#"{"matches":[{"country":"People's Republic of China","state":"Shanghai","city":"Pudong"}]}"#,
            // 2. keyword expansion
            r#"{"keywords":["猫"]}"#,
            // 3. rerank
            r#"{"ids":["c002"]}"#,
        ]);
        perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args).unwrap();
        let calls = backend.calls();
        assert_eq!(calls.len(), 2);
        for c in &calls {
            assert_eq!(c.country.as_deref(), Some("People's Republic of China"));
            assert_eq!(c.state.as_deref(), Some("Shanghai"));
            assert_eq!(c.city.as_deref(), Some("Pudong"));
        }
    }

    #[test]
    fn perform_search_combined_reranks_smart_hits_when_description_empty() {
        let backend = FakeBackend::new(vec![
            // CLIP returns 1 hit
            resp(vec![asset_with_desc("a", "/mnt/x/a.jpg", "x")], None),
            // Description workflow's substring search hits nothing
            resp(vec![], None),
        ]);
        let mut args = default_args();
        args.query = Some("elephants".into());
        let llm = FakeLlm::new(&[r#"{"keywords":["大象"]}"#, r#"{"ids":["c001"]}"#]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        assert_eq!(got.assets.len(), 1);
        assert_eq!(got.assets[0].id, "a");
    }

    // ---- --place: resolution + multi-match merge -----------------------

    #[test]
    fn perform_search_place_requires_llm() {
        let backend = FakeBackend::new(vec![]);
        let mut args = default_args();
        args.place = Some("上海".into());
        let err = perform_search(&cfg(), &fb(&backend, None), &Admin2Lookup::new(), &args)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--place"), "got: {err}");
        assert!(err.contains("[llm]"), "got: {err}");
    }

    #[test]
    fn perform_search_place_with_no_query_uses_resolved_geo() {
        // --place "中国" → country-only match; no -q means filter-only fetch.
        let backend = FakeBackend::new(vec![resp(
            vec![asset_with_desc("a", "/mnt/x/a.jpg", "x")],
            None,
        )])
        .with_vocab(sample_vocab());
        let mut args = default_args();
        args.place = Some("中国".into());
        let llm = FakeLlm::new(&[r#"{"matches":[{"country":"People's Republic of China"}]}"#]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        assert_eq!(got.assets.len(), 1);
        let calls = backend.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].country.as_deref(),
            Some("People's Republic of China")
        );
        assert!(calls[0].state.is_none());
        assert!(calls[0].city.is_none());
    }

    #[test]
    fn perform_search_place_with_no_match_bails() {
        let backend = FakeBackend::new(vec![]).with_vocab(sample_vocab());
        let mut args = default_args();
        args.place = Some("Antarctica".into());
        let llm = FakeLlm::new(&[r#"{"matches":[]}"#]);
        let err = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no place"), "got: {err}");
    }

    #[test]
    fn perform_search_multi_place_runs_one_fetch_each_and_rrf_merges() {
        // Two matches → two filter-only Immich calls, results merged by RRF.
        let backend = FakeBackend::new(vec![
            resp(
                vec![
                    asset_with_desc("a", "/mnt/x/a.jpg", "x"),
                    asset_with_desc("b", "/mnt/x/b.jpg", "x"),
                ],
                None,
            ),
            resp(
                vec![
                    asset_with_desc("c", "/mnt/x/c.jpg", "x"),
                    asset_with_desc("b", "/mnt/x/b.jpg", "x"),
                ],
                None,
            ),
        ])
        .with_vocab(sample_vocab());
        let mut args = default_args();
        args.place = Some("Anping".into());
        // Pretend "Anping" matches two cities in two states.
        let llm = FakeLlm::new(&[r#"{"matches":[
            {"country":"People's Republic of China","state":"Shanghai","city":"Pudong"},
            {"country":"People's Republic of China","state":"Zhejiang","city":"Andong"}
        ]}"#]);
        let got = perform_search(&cfg(), &fb(&backend, Some(&llm)), &Admin2Lookup::new(), &args)
            .unwrap();
        // RRF: b appears in both lists (rank 2 in each) → top.
        let ids: Vec<&str> = got.assets.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids[0], "b");
        // Two Immich calls, one per match, each with its own geo.
        let calls = backend.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].city.as_deref(), Some("Pudong"));
        assert_eq!(calls[1].city.as_deref(), Some("Andong"));
    }
}
