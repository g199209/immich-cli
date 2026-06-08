//! LLM-mediated semantic search over photo descriptions.
//!
//! Used by `search` when an `-q` query is paired with a configured
//! `[llm]` block. Main responsibilities:
//!   1. **Keyword expansion**: LLM turns the natural-language query into
//!      up to 8 Chinese substrings that are likely to appear verbatim
//!      inside a photo description.
//!   2. **Substring search**: each keyword is fanned out against
//!      `/api/search/metadata` with the same hard filters the user
//!      passed to `search` (city, country, taken-after, …). Each keyword
//!      contributes up to 64 candidates, then all candidates are unioned
//!      and deduped by asset id.
//!   3. **Rerank**: the LLM is shown the query and enriched candidates
//!      (short id, metadata, sources, matched keywords, description) and
//!      returns the relevant short ids in relevance order.

use crate::client::SearchBackend;
use crate::llm::{ChatBackend, Message};
use crate::models::{Asset, SearchRequest};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// Hard upper bound on keywords returned by the LLM. Keeps prompt cost
/// linear and the substring-search fan-out bounded.
pub const MAX_KEYWORDS: usize = 8;

/// Per-keyword cap before deduping candidates for rerank. Keeping this
/// per keyword avoids one broad keyword filling the entire rerank pool
/// and starving later, more specific keywords.
pub const MAX_CANDIDATES_PER_KEYWORD: usize = 64;

/// Per-candidate description excerpt sent to the reranker. Long enough
/// to convey content, short enough to keep the rerank prompt bounded.
pub const DESCRIPTION_EXCERPT_CHARS: usize = 1500;

/// Same hard filters `search` accepts; we forward them to both the CLIP
/// path and the per-keyword metadata calls so the two ranked lists are
/// over the same universe of assets.
#[derive(Debug, Default, Clone)]
pub struct HardFilters {
    pub city: Option<String>,
    pub state: Option<String>,
    pub country: Option<String>,
    pub taken_after: Option<String>,
    pub taken_before: Option<String>,
    pub asset_type: Option<String>,
    pub ocr: Option<String>,
    /// See [`crate::models::SearchRequest::is_archived`] for semantics.
    /// `None` means "don't filter" (Immich returns archived + non-archived).
    pub is_archived: Option<bool>,
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
- Up to 8 keywords. Quality > quantity.\n\
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

pub fn expand_keywords<L: ChatBackend>(
    llm: &L,
    query: &str,
    verbose: bool,
) -> Result<Vec<String>> {
    if verbose {
        eprintln!(
            "[verbose] description_search: expanding keywords for query={query:?} \
             (system prompt = {} chars)",
            KEYWORD_SYSTEM_PROMPT.len()
        );
    }
    let messages = [
        Message::system(KEYWORD_SYSTEM_PROMPT),
        Message::user(query.to_string()),
    ];
    let raw = llm.chat_json(&messages)?;
    if verbose {
        eprintln!("[verbose] description_search: keyword raw LLM reply = {raw}");
    }
    let reply: KeywordsReply = serde_json::from_str(&raw)
        .with_context(|| format!("LLM keyword reply was not valid JSON: {raw}"))?;
    let mut keywords: Vec<String> = reply
        .keywords
        .into_iter()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .collect();
    keywords.truncate(MAX_KEYWORDS);
    let out = dedup_preserving_order(keywords);
    if verbose {
        eprintln!(
            "[verbose] description_search: parsed {} keyword(s): [{}]",
            out.len(),
            out.iter()
                .map(|k| format!("{k:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(out)
}

// ---- stage 2: substring search per keyword (with hard filters) ----------

pub fn collect_candidates<S: SearchBackend>(
    search: &S,
    keywords: &[String],
    filters: &HardFilters,
) -> Result<Vec<Asset>> {
    let hits = collect_keyword_hits(
        search,
        keywords,
        filters,
        MAX_CANDIDATES_PER_KEYWORD as u32,
        false,
    )?;
    let mut by_id: HashMap<String, Asset> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for hit in hits {
        if !by_id.contains_key(&hit.asset.id) {
            order.push(hit.asset.id.clone());
            by_id.insert(hit.asset.id.clone(), hit.asset);
        }
    }

    Ok(order
        .into_iter()
        .filter_map(|id| by_id.remove(&id))
        .collect())
}

#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub keyword: String,
    pub asset: Asset,
}

pub fn collect_keyword_hits<S: SearchBackend>(
    search: &S,
    keywords: &[String],
    filters: &HardFilters,
    per_keyword_limit: u32,
    verbose: bool,
) -> Result<Vec<KeywordHit>> {
    let mut hits = Vec::new();
    let per_keyword_limit = per_keyword_limit.max(1);
    for (i, kw) in keywords.iter().enumerate() {
        let req = SearchRequest {
            description: Some(kw.clone()),
            city: filters.city.clone(),
            state: filters.state.clone(),
            country: filters.country.clone(),
            taken_after: filters.taken_after.clone(),
            taken_before: filters.taken_before.clone(),
            asset_type: filters.asset_type.clone(),
            ocr: filters.ocr.clone(),
            is_archived: filters.is_archived,
            size: Some(per_keyword_limit),
            with_exif: Some(true),
            ..Default::default()
        };
        let resp = search.search(&req)?;
        let returned = resp.assets.items.len();
        let take = returned.min(per_keyword_limit as usize);
        for asset in resp.assets.items.into_iter().take(take) {
            hits.push(KeywordHit {
                keyword: kw.clone(),
                asset,
            });
        }
        if verbose {
            let capped = if returned > take { " (capped)" } else { "" };
            eprintln!(
                "[verbose] description_search: keyword {}/{} {kw:?} → {take} hit(s) \
                 (cap {per_keyword_limit}){capped}",
                i + 1,
                keywords.len()
            );
        }
    }
    Ok(hits)
}

// ---- stage 3: rerank -----------------------------------------------------

#[derive(Deserialize)]
struct RerankReply {
    #[serde(default)]
    ids: Vec<String>,
}

const RERANK_SYSTEM_PROMPT: &str = "\
You are ranking image-search candidates. Given a user query and a list of \
candidate photos, identify the candidates that genuinely match the user's \
intent and order them by relevance.\n\n\
Your output MUST be a single JSON object: {\"ids\": [\"id1\", \"id2\", ...]} \
ordered by descending relevance.\n\n\
Rules:\n\
- Be strict. Omit weak or tangential matches.\n\
- Only include candidate ids you actually saw in the input.\n\
- Return no more ids than the requested limit.\n\
- If nothing matches, return {\"ids\": []}.";

#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub id: String,
    pub asset_type: String,
    pub taken_at: Option<String>,
    pub location: Option<RerankLocation>,
    pub people: Vec<String>,
    pub sources: Vec<String>,
    pub matched_keywords: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct RerankLocation {
    pub country: Option<String>,
    pub state: Option<String>,
    pub admin2: Option<String>,
    pub city: Option<String>,
}

pub fn rerank_rich<L: ChatBackend>(
    llm: &L,
    query: &str,
    limit: usize,
    candidates: &[RerankCandidate],
    verbose: bool,
) -> Result<Vec<String>> {
    let items: Vec<serde_json::Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "type": c.asset_type,
                "takenAt": c.taken_at,
                "location": c.location.as_ref().map(|loc| serde_json::json!({
                    "country": loc.country,
                    "state": loc.state,
                    "admin2": loc.admin2,
                    "city": loc.city,
                })),
                "people": c.people,
                "sources": c.sources,
                "matchedKeywords": c.matched_keywords,
                "description": excerpt(&c.description, DESCRIPTION_EXCERPT_CHARS),
            })
        })
        .collect();
    let user_msg = serde_json::json!({
        "query": query,
        "limit": limit,
        "candidates": items,
    });
    let user_payload = user_msg.to_string();
    if verbose {
        eprintln!(
            "[verbose] description_search: rerank query={query:?} limit={limit} \
             candidates={} (user payload = {} chars)",
            candidates.len(),
            user_payload.len()
        );
    }
    let messages = [
        Message::system(RERANK_SYSTEM_PROMPT),
        Message::user(user_payload),
    ];
    let raw = llm.chat_json(&messages)?;
    if verbose {
        eprintln!("[verbose] description_search: rerank raw LLM reply = {raw}");
    }
    let reply: RerankReply = serde_json::from_str(&raw)
        .with_context(|| format!("LLM rerank reply was not valid JSON: {raw}"))?;
    let mut ids = reply.ids;
    ids.truncate(limit);
    if verbose {
        eprintln!(
            "[verbose] description_search: rerank kept {} id(s) after limit: [{}]",
            ids.len(),
            ids.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(ids)
}

pub fn rerank<L: ChatBackend>(llm: &L, query: &str, candidates: &[Asset]) -> Result<Vec<String>> {
    let items: Vec<serde_json::Value> = candidates
        .iter()
        .map(|a| {
            serde_json::json!({
                "id": a.id,
                "description": excerpt(
                    a.exif_info
                        .as_ref()
                        .and_then(|e| e.description.as_deref())
                        .unwrap_or(""),
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

/// Pull `candidates` in the order given by `selected_ids`, dropping ids
/// the LLM hallucinated (or repeated). Each id appears at most once.
pub fn take_in_order<'a>(candidates: &'a [Asset], selected_ids: &[String]) -> Vec<&'a Asset> {
    let lookup: HashMap<&str, &Asset> = candidates.iter().map(|a| (a.id.as_str(), a)).collect();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut out = Vec::with_capacity(selected_ids.len());
    for id in selected_ids {
        if seen.insert(id.as_str(), ()).is_some() {
            continue;
        }
        if let Some(asset) = lookup.get(id.as_str()) {
            out.push(*asset);
        }
    }
    out
}

// ---- helpers --------------------------------------------------------------

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

// ---- end-to-end convenience: query → ordered assets ----------------------

/// Wraps the three stages so `search` only has to call one function.
/// Returns the assets ordered by LLM-assessed relevance.
pub fn run<S, L>(search: &S, llm: &L, query: &str, filters: &HardFilters) -> Result<Vec<Asset>>
where
    S: SearchBackend,
    L: ChatBackend,
{
    let keywords = expand_keywords(llm, query, false)?;
    if keywords.is_empty() {
        return Ok(vec![]);
    }
    let candidates = collect_candidates(search, &keywords, filters)?;
    if candidates.is_empty() {
        return Ok(vec![]);
    }
    let selected_ids = rerank(llm, query, &candidates)?;
    Ok(take_in_order(&candidates, &selected_ids)
        .into_iter()
        .cloned()
        .collect())
}

// ---- Reciprocal Rank Fusion ----------------------------------------------

/// Standard RRF (k=60). Each ranked list contributes
/// `1 / (k + rank_in_list)` to each asset's score. Higher = better.
/// We use insertion order for stable ties.
pub const RRF_K: f64 = 60.0;

pub fn rrf_merge(lists: &[Vec<Asset>]) -> Vec<Asset> {
    let mut score: HashMap<String, f64> = HashMap::new();
    let mut first_seen: HashMap<String, usize> = HashMap::new();
    let mut id_to_asset: HashMap<String, Asset> = HashMap::new();
    let mut counter = 0usize;

    for list in lists {
        for (rank, asset) in list.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
            *score.entry(asset.id.clone()).or_insert(0.0) += contribution;
            first_seen.entry(asset.id.clone()).or_insert_with(|| {
                let v = counter;
                counter += 1;
                v
            });
            id_to_asset
                .entry(asset.id.clone())
                .or_insert_with(|| asset.clone());
        }
    }

    let mut ids: Vec<String> = score.keys().cloned().collect();
    ids.sort_by(|a, b| {
        let sa = score[a];
        let sb = score[b];
        // Descending score, tie-break by first_seen ascending (more
        // deterministic than HashMap iteration order).
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| first_seen[a].cmp(&first_seen[b]))
    });
    ids.into_iter()
        .map(|id| id_to_asset.remove(&id).unwrap())
        .collect()
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AssetsBucket, ExifInfo, SearchResponse};
    use std::cell::RefCell;

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
    }
    impl FakeLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: RefCell::new(replies.iter().map(|s| s.to_string()).collect()),
            }
        }
    }
    impl ChatBackend for FakeLlm {
        fn chat_json(&self, _messages: &[Message]) -> Result<String> {
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    fn asset(id: &str, path: &str) -> Asset {
        Asset {
            id: id.into(),
            original_path: path.into(),
            original_file_name: path.rsplit('/').next().unwrap_or(path).into(),
            asset_type: "IMAGE".into(),
            file_created_at: None,
            local_date_time: None,
            checksum: String::new(),
            exif_info: Some(ExifInfo {
                city: None,
                state: None,
                country: None,
                latitude: None,
                longitude: None,
                description: Some(format!("desc-{id}")),
            }),
        }
    }

    fn bucket(items: Vec<Asset>) -> SearchResponse {
        SearchResponse {
            assets: AssetsBucket {
                total: items.len() as u32,
                count: items.len() as u32,
                items,
                next_page: None,
            },
        }
    }

    #[test]
    fn collect_candidates_forwards_hard_filters_to_every_call() {
        let search = FakeSearch::new(vec![bucket(vec![]), bucket(vec![])]);
        let filters = HardFilters {
            country: Some("China".into()),
            asset_type: Some("IMAGE".into()),
            ..Default::default()
        };
        collect_candidates(&search, &["大象".into(), "草原".into()], &filters).unwrap();
        let calls = search.calls.borrow();
        assert_eq!(calls.len(), 2);
        for c in calls.iter() {
            assert_eq!(c.country.as_deref(), Some("China"));
            assert_eq!(c.asset_type.as_deref(), Some("IMAGE"));
            assert_eq!(c.size, Some(MAX_CANDIDATES_PER_KEYWORD as u32));
        }
        assert_eq!(calls[0].description.as_deref(), Some("大象"));
        assert_eq!(calls[1].description.as_deref(), Some("草原"));
    }

    #[test]
    fn collect_candidates_caps_each_keyword_without_starving_later_keywords() {
        let broad: Vec<Asset> = (0..70)
            .map(|i| asset(&format!("broad-{i}"), &format!("/x/broad-{i}.jpg")))
            .collect();
        let search = FakeSearch::new(vec![
            bucket(broad),
            bucket(vec![asset("exact", "/x/e.jpg")]),
        ]);

        let got = collect_candidates(
            &search,
            &["宽泛".into(), "精确".into()],
            &HardFilters::default(),
        )
        .unwrap();

        let calls = search.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].description.as_deref(), Some("宽泛"));
        assert_eq!(calls[1].description.as_deref(), Some("精确"));

        let ids: Vec<&str> = got.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids.len(), MAX_CANDIDATES_PER_KEYWORD + 1);
        assert!(ids.contains(&"broad-0"));
        assert!(ids.contains(&"broad-63"));
        assert!(!ids.contains(&"broad-64"));
        assert!(ids.contains(&"exact"));
    }

    #[test]
    fn run_returns_assets_in_rerank_order() {
        let search = FakeSearch::new(vec![
            bucket(vec![asset("a", "/x/a.jpg"), asset("b", "/x/b.jpg")]),
            bucket(vec![asset("c", "/x/c.jpg")]),
        ]);
        let llm = FakeLlm::new(&[r#"{"keywords":["大象","草原"]}"#, r#"{"ids":["c","a"]}"#]);
        let got = run(&search, &llm, "elephants", &HardFilters::default()).unwrap();
        let ids: Vec<&str> = got.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "a"]);
    }

    #[test]
    fn rrf_merge_prefers_assets_appearing_in_both_lists() {
        let a = asset("a", "/x/a.jpg");
        let b = asset("b", "/x/b.jpg");
        let c = asset("c", "/x/c.jpg");
        // List 1: a > b
        // List 2: b > c
        // RRF: a=1/61, b=1/62+1/61, c=1/62 → b wins (in both), then a, then c.
        let merged = rrf_merge(&[vec![a.clone(), b.clone()], vec![b.clone(), c.clone()]]);
        let ids: Vec<&str> = merged.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn rrf_merge_dedupes_across_lists() {
        let a = asset("a", "/x/a.jpg");
        let b = asset("b", "/x/b.jpg");
        let merged = rrf_merge(&[vec![a.clone(), b.clone()], vec![a.clone()]]);
        let ids: Vec<&str> = merged.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn rrf_merge_handles_empty_lists() {
        let a = asset("a", "/x/a.jpg");
        let merged = rrf_merge(&[vec![], vec![a.clone()]]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "a");
    }

    #[test]
    fn rrf_empty_inputs_returns_empty() {
        let merged: Vec<Asset> = rrf_merge(&[]);
        assert!(merged.is_empty());
        let merged: Vec<Asset> = rrf_merge(&[vec![], vec![]]);
        assert!(merged.is_empty());
    }
}
