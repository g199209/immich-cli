//! LLM-mediated semantic search over photo descriptions.
//!
//! Used by `search` when an `-q` query is paired with a configured
//! `[llm]` block. Main responsibilities:
//!   1. **Keyword expansion**: LLM turns the natural-language query into
//!      up to 3 Chinese substrings that are likely to appear verbatim
//!      inside a photo description.
//!   2. **Substring search**: each keyword is fanned out against
//!      `/api/search/metadata` with the same hard filters the user
//!      passed to `search` (city, country, taken-after, …). Each keyword
//!      contributes up to a per-keyword cap, then all candidates are
//!      unioned and deduped by asset id.
//!   3. **Rerank**: the vision LLM is shown the query, per-candidate
//!      metadata (short id, time, place, people, sources, matched
//!      keywords) and the candidate thumbnails, and returns the relevant
//!      short ids in relevance order. The thumbnail replaces what the
//!      old text-only path used to send as a description excerpt — the
//!      vision model can read appearance straight from pixels, which is
//!      cheaper *and* more accurate than routing through another model's
//!      generated caption.

use crate::client::SearchBackend;
use crate::llm::{ChatBackend, Message, MultiImageVisionLlm};
use crate::models::{Asset, SearchRequest};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// Hard upper bound on keywords returned by the LLM. Each keyword
/// triggers one sequential Immich `/metadata` round-trip, so this is
/// the dominant lever on description fan-out latency. Three covers
/// the central noun plus two close synonyms (夕阳 vs 日落, 夜景 vs
/// 灯光) — broader queries gain little from additional variants and
/// pay a full RTT for each.
pub const MAX_KEYWORDS: usize = 3;

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
    /// See [`crate::models::SearchRequest::visibility`] for semantics.
    /// `None` falls back to the server default (`timeline`, i.e.
    /// non-archived). Send `"archive"` to target archived assets.
    pub visibility: Option<String>,
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
- Up to 3 keywords. Quality > quantity.\n\
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
            visibility: filters.visibility.clone(),
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

// ---- stage 3: vision rerank ----------------------------------------------

#[derive(Deserialize)]
struct RerankReply {
    #[serde(default)]
    ids: Vec<String>,
}

const RERANK_SYSTEM_PROMPT: &str = "\
You are ranking image-search candidates. You receive a user query and a list \
of candidate photos — each candidate is presented as a one-line metadata \
header followed by the photo itself. Use both the visual content of the \
photo and the metadata (time, place, people, matched description keywords) \
to identify candidates that genuinely match the user's intent, ordered by \
relevance.\n\n\
Your output MUST be a single JSON object: {\"ids\": [\"id1\", \"id2\", ...]} \
ordered by descending relevance.\n\n\
Rules:\n\
- Be strict. Omit weak or tangential matches; visual mismatch with the query \
trumps a description-keyword hit.\n\
- Only include candidate ids you actually saw in the input.\n\
- Return no more ids than the requested limit.\n\
- If nothing matches, return {\"ids\": []}.";

/// Cap on output tokens for the rerank reply. The reply is just an id
/// list — even 64 short ids fit comfortably under a few hundred tokens.
/// Headroom for reasoning models that emit internal thinking before the
/// JSON, which would otherwise null out `content`.
const RERANK_MAX_OUTPUT_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub id: String,
    pub asset_type: String,
    pub taken_at: Option<String>,
    pub location: Option<RerankLocation>,
    pub people: Vec<String>,
    pub sources: Vec<String>,
    pub matched_keywords: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RerankLocation {
    pub country: Option<String>,
    pub state: Option<String>,
    pub admin2: Option<String>,
    pub city: Option<String>,
}

/// One-line, human-friendly label sent right before each candidate
/// thumbnail. Compact on purpose — every char is a vision input token.
pub fn label_for(c: &RerankCandidate) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(6);
    parts.push(format!("id={}", c.id));
    parts.push(format!("type={}", c.asset_type));
    if let Some(t) = c.taken_at.as_deref().filter(|s| !s.is_empty()) {
        parts.push(format!("taken={t}"));
    }
    if let Some(loc) = &c.location {
        let mut bits: Vec<&str> = Vec::with_capacity(4);
        for f in [&loc.country, &loc.state, &loc.admin2, &loc.city]
            .into_iter()
            .flatten()
        {
            if !f.is_empty() {
                bits.push(f);
            }
        }
        if !bits.is_empty() {
            parts.push(format!("loc={}", bits.join("/")));
        }
    }
    if !c.people.is_empty() {
        parts.push(format!("people={}", c.people.join(",")));
    }
    if !c.sources.is_empty() {
        parts.push(format!("sources={}", c.sources.join("+")));
    }
    if !c.matched_keywords.is_empty() {
        parts.push(format!("matched={}", c.matched_keywords.join(",")));
    }
    parts.join(" | ")
}

/// Vision rerank: each candidate is a `(metadata-label, thumbnail JPEG)`
/// pair, sent to the multimodal model alongside the query.
pub fn rerank_rich_vision<V: MultiImageVisionLlm>(
    llm: &V,
    query: &str,
    limit: usize,
    candidates: &[RerankCandidate],
    thumbnails: &[Vec<u8>],
    verbose: bool,
) -> Result<Vec<String>> {
    if candidates.len() != thumbnails.len() {
        anyhow::bail!(
            "rerank_rich_vision: {} candidate(s) but {} thumbnail(s)",
            candidates.len(),
            thumbnails.len()
        );
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let items: Vec<(String, Vec<u8>, &str)> = candidates
        .iter()
        .zip(thumbnails.iter())
        .map(|(c, bytes)| (label_for(c), bytes.clone(), "image/jpeg"))
        .collect();
    let user_prompt = format!(
        "query: {query}\nlimit: {limit}\nReturn the top candidate ids by relevance."
    );
    if verbose {
        let label_chars: usize = items.iter().map(|(l, _, _)| l.chars().count()).sum();
        eprintln!(
            "[verbose] description_search: vision rerank query={query:?} limit={limit} \
             candidates={} (labels total {label_chars} chars)",
            candidates.len()
        );
    }
    let raw = llm.rank_images(
        RERANK_SYSTEM_PROMPT,
        &user_prompt,
        &items,
        RERANK_MAX_OUTPUT_TOKENS,
    )?;
    if verbose {
        eprintln!("[verbose] description_search: vision rerank raw LLM reply = {raw}");
    }
    let reply: RerankReply = serde_json::from_str(&raw)
        .with_context(|| format!("LLM rerank reply was not valid JSON: {raw}"))?;
    let mut ids = reply.ids;
    ids.truncate(limit);
    if verbose {
        eprintln!(
            "[verbose] description_search: vision rerank kept {} id(s) after limit: [{}]",
            ids.len(),
            ids.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(ids)
}

// ---- helpers --------------------------------------------------------------

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
    use crate::models::ExifInfo;
    use std::cell::RefCell;

    struct FakeVisionLlm {
        replies: RefCell<Vec<String>>,
        calls: RefCell<Vec<(String, Vec<(String, usize)>)>>,
    }
    impl FakeVisionLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: RefCell::new(replies.iter().map(|s| s.to_string()).collect()),
                calls: RefCell::new(Vec::new()),
            }
        }
    }
    impl MultiImageVisionLlm for FakeVisionLlm {
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
            _system_prompt: &str,
            user_prompt: &str,
            items: &[(String, Vec<u8>, &str)],
            _max_tokens: u32,
        ) -> Result<String> {
            let summary: Vec<(String, usize)> =
                items.iter().map(|(l, b, _)| (l.clone(), b.len())).collect();
            self.calls
                .borrow_mut()
                .push((user_prompt.to_string(), summary));
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
                description: Some(format!("desc-{id}")),
                ..Default::default()
            }),
        }
    }

    fn rerank_candidate(id: &str) -> RerankCandidate {
        RerankCandidate {
            id: id.into(),
            asset_type: "IMAGE".into(),
            taken_at: Some("2024-05-01T10:00:00".into()),
            location: Some(RerankLocation {
                country: Some("China".into()),
                state: Some("Hainan".into()),
                admin2: None,
                city: Some("Haitangwan".into()),
            }),
            people: vec!["女儿（孩子）".into()],
            sources: vec!["smart".into(), "description".into()],
            matched_keywords: vec!["大海".into()],
        }
    }

    #[test]
    fn label_for_packs_compact_kv_metadata() {
        let label = label_for(&rerank_candidate("c001"));
        assert!(label.starts_with("id=c001"), "{label}");
        assert!(label.contains("type=IMAGE"), "{label}");
        assert!(label.contains("taken=2024-05-01"), "{label}");
        assert!(label.contains("loc=China/Hainan/Haitangwan"), "{label}");
        assert!(label.contains("people=女儿（孩子）"), "{label}");
        assert!(label.contains("sources=smart+description"), "{label}");
        assert!(label.contains("matched=大海"), "{label}");
    }

    #[test]
    fn rerank_rich_vision_passes_one_label_image_pair_per_candidate() {
        let llm = FakeVisionLlm::new(&[r#"{"ids":["c002","c001"]}"#]);
        let candidates = vec![rerank_candidate("c001"), rerank_candidate("c002")];
        let thumbs = vec![vec![0u8; 11], vec![0u8; 22]];
        let ids = rerank_rich_vision(&llm, "大海", 5, &candidates, &thumbs, false).unwrap();
        assert_eq!(ids, vec!["c002".to_string(), "c001".to_string()]);
        let calls = llm.calls.borrow();
        assert_eq!(calls.len(), 1);
        let (user_prompt, items) = &calls[0];
        assert!(user_prompt.contains("query: 大海"), "{user_prompt}");
        assert!(user_prompt.contains("limit: 5"), "{user_prompt}");
        assert_eq!(items.len(), 2);
        assert!(items[0].0.starts_with("id=c001"));
        assert_eq!(items[0].1, 11);
        assert!(items[1].0.starts_with("id=c002"));
        assert_eq!(items[1].1, 22);
    }

    #[test]
    fn rerank_rich_vision_truncates_to_limit() {
        let llm = FakeVisionLlm::new(&[r#"{"ids":["c001","c002","c003"]}"#]);
        let candidates = vec![
            rerank_candidate("c001"),
            rerank_candidate("c002"),
            rerank_candidate("c003"),
        ];
        let thumbs = vec![vec![], vec![], vec![]];
        let ids = rerank_rich_vision(&llm, "x", 2, &candidates, &thumbs, false).unwrap();
        assert_eq!(ids, vec!["c001".to_string(), "c002".to_string()]);
    }

    #[test]
    fn rerank_rich_vision_rejects_mismatched_thumbnail_count() {
        let llm = FakeVisionLlm::new(&[]);
        let err = rerank_rich_vision(
            &llm,
            "x",
            5,
            &[rerank_candidate("c001"), rerank_candidate("c002")],
            &[vec![0u8; 4]],
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("candidate"), "{err}");
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
