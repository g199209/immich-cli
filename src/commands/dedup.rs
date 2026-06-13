//! `dedup` — collapse near-duplicate asset groups into stacks.
//!
//! Pipeline per `GET /api/duplicates` group:
//!   1. **Skip** the group if any member is already in some stack
//!      (avoid double-stacking the same shot).
//!   1b.**KeepAll** if every asset has a capture time and earliest/latest
//!      differ by more than 6 hours. Hard guard that runs before
//!      auto-delete so a multi-hour spread can never trigger a delete
//!      or stack. In `--apply` mode this calls
//!      `DELETE /api/duplicates/{id}` to dismiss the group server-side
//!      so it no longer appears under `GET /api/duplicates` (same
//!      effect as the web UI's "keep all"). Assets themselves are
//!      untouched.
//!   2. **Skip** if any member is a video — the user has not asked us
//!      to make video judgement calls.
//!   3. **Auto-delete (images)** if every member is an image with the
//!      same filename, the same capture time, AND file sizes within
//!      0.1% of each other, but lives in a different folder. These
//!      are exact cross-folder copies the user filed in multiple
//!      places; keep the one with GPS (or any one, if all/none have
//!      GPS) and DELETE the rest. No stack is created. The size check
//!      is the safety belt: a `/raw/` vs `/retouched/` pair with
//!      identical EXIF time would otherwise be indistinguishable from
//!      a true copy and we'd wrongly nuke the retouched version. A
//!      0.1% tolerance absorbs tiny EXIF re-writes (e.g. an embedded
//!      thumbnail refresh) without admitting genuinely different
//!      exports.
//!   3b.**Auto-delete (videos)** if every member is a video with the
//!      same filename, the same capture time, different folders, and
//!      file sizes that are NOT all equal. Keep the largest (assumed
//!      original / highest-quality) and DELETE the rest. If the
//!      keeper lacks GPS but a sibling has it, copy the sibling's
//!      coords onto the keeper before deletion. Equal-size video
//!      groups fall through to the regular ContainsVideo skip for
//!      manual review.
//!   4. **Skip** if members come from different parent folders — a
//!      cross-folder match that isn't an exact-name+time duplicate is
//!      almost always a coincidence we don't want to act on.
//!   5. **Skip** if earliest/latest capture time differ by more than
//!      `--max-time-gap` (default 10 min). Combined with rule 1b: a
//!      10min–6h spread falls through here, while > 6h is hard-capped
//!      earlier and can't reach the auto-delete branches at all.
//!   6. **Skip** if ALL members have GPS and the pairwise great-circle
//!      distance exceeds `--max-distance-m` (default 500m). When some
//!      members are missing GPS we keep the group and copy coords later.
//!   7. **Pick a winner**:
//!        * If the largest file is more than `--size-ratio` larger than
//!          the smallest (default 50%), pick the largest. The assumption
//!          is the smaller siblings are downsamples/re-exports.
//!        * Otherwise, ask the vision LLM to compare composition,
//!          content, and liveliness — it returns the winner index in JSON.
//!   8. **Backfill GPS**: if the winner has coords but some siblings
//!      do not, PUT the winner's coords onto them. (Or, if the winner
//!      lacks coords but a sibling has them, copy from the sibling
//!      onto every missing one — winner included.)
//!   9. **Stack**: POST /api/stacks with the winner first so it becomes
//!      the cover.
//!
//! Default mode is dry-run; pass `--apply` to actually mutate Immich.

use crate::client::{
    CaptionBackend, DedupWriteBackend, DuplicatesBackend, ImmichClient, StacksBackend,
};
use crate::config::Config;
use crate::llm::{MultiImageVisionLlm, OpenAiClient};
use crate::models::{Asset, DuplicateGroup};
use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Token budget for the vision pick call. The model must only emit a
/// short JSON object — 1K leaves plenty of room for reasoning tokens on
/// reasoning models.
const MAX_PICK_TOKENS: u32 = 1024;

/// Tolerance for the auto-delete size-equality check. `(max - min) / max`
/// must be ≤ this. 0.001 = 0.1% — enough to absorb an EXIF rewrite
/// without admitting a different export.
const AUTO_DELETE_MAX_SIZE_GAP: f64 = 0.001;

/// Hard upper bound on the capture-time spread within a duplicate group.
/// If earliest/latest differ by more than this, we keep every member
/// untouched regardless of any other rule (no stack, no delete). This
/// guards against acting on groups that obviously span more than a
/// single shoot, even when `--max-time-gap-secs` is set very high.
const HARD_MAX_TIME_SPREAD_SECS: i64 = 6 * 60 * 60;

#[derive(Args, Debug)]
pub struct DedupArgs {
    /// Actually write changes back to Immich (PUT location, POST stacks).
    /// Without this flag, the command only prints what it would do.
    #[arg(long)]
    pub apply: bool,

    /// Maximum capture-time gap, in seconds, between the earliest and
    /// latest asset in a duplicate group. Groups exceeding this are
    /// skipped as likely false positives. Note: rule 1b applies a
    /// separate hard 6h cap that runs before auto-delete and cannot
    /// be relaxed by raising this value.
    #[arg(long, default_value_t = 600)]
    pub max_time_gap_secs: i64,

    /// Maximum great-circle distance, in meters, between any two GPS
    /// fixes in a duplicate group (only applied when every member has
    /// coordinates). Groups exceeding this are skipped.
    #[arg(long, default_value_t = 500.0)]
    pub max_distance_m: f64,

    /// File-size ratio threshold. The pick short-circuits to "largest"
    /// when `1 - smallest/largest > size_ratio`. Default 0.5 → a 50%
    /// size gap is enough to skip the vision call.
    #[arg(long, default_value_t = 0.5)]
    pub size_ratio: f64,

    /// Maximum number of groups to act on this run, counted after all
    /// skip filters. `0` = no cap.
    #[arg(long, default_value_t = 0)]
    pub limit: u32,

    /// Number of concurrent workers for the plan-and-apply phase. Each
    /// worker holds one group and does its thumbnail fetches, vision
    /// call, and HTTP writes. Raise carefully — the upstream vision
    /// gateway may rate-limit.
    #[arg(long, default_value_t = 4)]
    pub parallel: u32,

    /// Print the per-group decision (skip reason, winner, vision rationale).
    #[arg(long)]
    pub verbose: bool,
}

// ---- pure helpers --------------------------------------------------------

/// Reason a group was filtered out before we considered acting on it.
/// Surfaced in `--verbose` output and asserted on by tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    AlreadyStacked,
    TooFewAssets,
    ContainsVideo,
    DifferentFolders,
    TimeGapTooLarge,
    DistanceTooLarge,
    MissingCaptureTime,
}

/// What we plan to do with a single group: either skip with a reason
/// or act, pending the (LLM-resolved) winner selection.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupDecision {
    Skip(SkipReason),
    /// Exact cross-folder copies (same filename + same capture time +
    /// matching size or largest-of-videos, different folders). Keep
    /// `keeper_idx`, delete every other member. No vision call, no
    /// stack. If `gps_backfill` is `Some`, the keeper itself is
    /// missing GPS and we should PUT those coords onto it before
    /// deleting the source sibling. Used by the video rule, where
    /// the keeper is chosen by file size rather than by GPS presence.
    AutoDeleteDupes {
        keeper_idx: usize,
        gps_backfill: Option<(f64, f64)>,
    },
    /// Hard time-spread guard (> 6h) fired. Don't delete or stack — but
    /// don't just silently skip either, otherwise the same group keeps
    /// showing up on every `GET /api/duplicates`. In `--apply` mode we
    /// PUT `duplicateId: null` on every member to detach them from the
    /// duplicate group on the server side (the same effect as the
    /// web UI's "keep all" button).
    KeepAll,
    /// Pre-filter accepted the group; the winner still needs to be
    /// picked by [`pick_winner_by_size`] or the vision model.
    Consider,
}

pub fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// Great-circle distance in meters using the haversine formula.
pub fn haversine_m(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let earth_r = 6_371_000.0_f64;
    let to_rad = |d: f64| d.to_radians();
    let (phi1, phi2) = (to_rad(lat1), to_rad(lat2));
    let d_phi = to_rad(lat2 - lat1);
    let d_lambda = to_rad(lng2 - lng1);
    let a = (d_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (d_lambda / 2.0).sin().powi(2);
    2.0 * earth_r * a.sqrt().asin()
}

/// Parse the various ISO-8601 shapes Immich emits across `dateTimeOriginal`,
/// `localDateTime`, and `fileCreatedAt`. Returns UTC seconds.
///
/// Real-world examples we have to swallow:
///   * `2024-03-12T09:08:42.000Z`        — `fileCreatedAt` (UTC)
///   * `2024-03-12T17:08:42.000+08:00`   — `dateTimeOriginal` with offset
///   * `2024-03-12T17:08:42.000`         — `localDateTime` (no zone)
///   * `2024-03-12T17:08:42`             — same, sub-second omitted
///
/// For time-**gap** comparisons inside one duplicate group, all assets
/// share a single capture context, so treating the naive variants as
/// UTC is safe — the deltas are unchanged.
pub fn parse_capture_seconds(s: &str) -> Option<i64> {
    use chrono::{DateTime, NaiveDateTime};
    // RFC3339 covers `Z` and `±HH:MM` offsets; try it first since it
    // gives an absolute instant directly.
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    // Naive (zoneless) fallback — Immich emits this for `localDateTime`.
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(dt.and_utc().timestamp());
        }
    }
    None
}

fn asset_capture_time_seconds(a: &Asset) -> Option<i64> {
    a.exif_info
        .as_ref()
        .and_then(|e| e.date_time_original.as_deref())
        .and_then(parse_capture_seconds)
        .or_else(|| a.local_date_time.as_deref().and_then(parse_capture_seconds))
        .or_else(|| a.file_created_at.as_deref().and_then(parse_capture_seconds))
}

fn asset_gps(a: &Asset) -> Option<(f64, f64)> {
    let e = a.exif_info.as_ref()?;
    Some((e.latitude?, e.longitude?))
}

fn asset_size(a: &Asset) -> Option<u64> {
    a.exif_info.as_ref().and_then(|e| e.file_size_in_byte)
}

/// True iff `(max - min) / max <= max_gap`. Empty slices and an all-zero
/// `max` are treated as "close enough" so the caller doesn't need to
/// special-case them — the surrounding rule already requires every
/// member to report a size, so a 0 here is a real (if odd) value.
pub fn sizes_within_tolerance(sizes: &[u64], max_gap: f64) -> bool {
    let (Some(&min), Some(&max)) = (sizes.iter().min(), sizes.iter().max()) else {
        return true;
    };
    if max == 0 {
        return true;
    }
    (1.0 - (min as f64 / max as f64)) <= max_gap
}

/// Run all pre-filters from rules 1-5. Returns either a skip reason or
/// `Consider`, in which case the caller proceeds to picking a winner.
pub fn classify_group(
    group: &DuplicateGroup,
    stacked: &HashSet<String>,
    max_time_gap_secs: i64,
    max_distance_m: f64,
) -> GroupDecision {
    if group.assets.len() < 2 {
        return GroupDecision::Skip(SkipReason::TooFewAssets);
    }
    if group.assets.iter().any(|a| stacked.contains(&a.id)) {
        return GroupDecision::Skip(SkipReason::AlreadyStacked);
    }

    // Hard safety guard: when every asset has a capture time and the
    // earliest/latest differ by more than HARD_MAX_TIME_SPREAD_SECS,
    // bail out before any auto-delete branch. A multi-hour spread inside
    // one "duplicate" group is almost always two distinct events the
    // perceptual hash collided on, so we KeepAll — preserving every
    // asset AND detaching them from the duplicate group on the server
    // so they stop reappearing on future runs.
    if let Some(times) = group
        .assets
        .iter()
        .map(asset_capture_time_seconds)
        .collect::<Option<Vec<i64>>>()
    {
        let (min_t, max_t) = (
            *times.iter().min().unwrap(),
            *times.iter().max().unwrap(),
        );
        if max_t - min_t > HARD_MAX_TIME_SPREAD_SECS {
            return GroupDecision::KeepAll;
        }
    }

    let folders: HashSet<&str> = group
        .assets
        .iter()
        .map(|a| parent_dir(&a.original_path))
        .collect();

    // Rule 2.4 — all-videos cross-folder near-copy: same basename,
    // same capture time, different folders, file sizes NOT all equal.
    // Keep the largest (treated as the original / highest-quality
    // copy) and delete the rest. If the keeper lacks GPS but a sibling
    // has it, plan a backfill onto the keeper before the delete so
    // location metadata survives.
    let all_videos = group
        .assets
        .iter()
        .all(|a| a.asset_type.eq_ignore_ascii_case("VIDEO"));
    if all_videos && folders.len() > 1 {
        let names: HashSet<&str> = group
            .assets
            .iter()
            .map(|a| a.original_file_name.as_str())
            .collect();
        if names.len() == 1 {
            let times: Option<Vec<i64>> = group
                .assets
                .iter()
                .map(asset_capture_time_seconds)
                .collect();
            let sizes: Option<Vec<u64>> = group.assets.iter().map(asset_size).collect();
            if let (Some(times), Some(sizes)) = (times, sizes) {
                let times_equal = times.iter().min() == times.iter().max();
                let max_size = *sizes.iter().max().unwrap();
                let min_size = *sizes.iter().min().unwrap();
                if times_equal && max_size > min_size {
                    // First occurrence of the max wins on ties.
                    let keeper_idx = sizes.iter().position(|s| *s == max_size).unwrap();
                    let gps_backfill = if asset_gps(&group.assets[keeper_idx]).is_some() {
                        None
                    } else {
                        group
                            .assets
                            .iter()
                            .enumerate()
                            .find_map(|(i, a)| {
                                if i == keeper_idx {
                                    None
                                } else {
                                    asset_gps(a)
                                }
                            })
                    };
                    return GroupDecision::AutoDeleteDupes {
                        keeper_idx,
                        gps_backfill,
                    };
                }
            }
        }
    }

    if group
        .assets
        .iter()
        .any(|a| a.asset_type.eq_ignore_ascii_case("VIDEO"))
    {
        return GroupDecision::Skip(SkipReason::ContainsVideo);
    }

    // Rule 2.5 — exact cross-folder duplicates: same basename, same
    // capture-time-to-the-second, same byte size, only the folder
    // differs. The user typically created these by copying one shoot
    // into multiple archive folders. We don't need a vision pick:
    // keep one copy (preferring one that already has GPS so we don't
    // lose metadata) and delete the rest. Size equality is what
    // separates true copies from same-name+same-time edits (e.g.
    // `/raw/foo.jpg` vs `/retouched/foo.jpg`).
    if folders.len() > 1 {
        let names: HashSet<&str> = group
            .assets
            .iter()
            .map(|a| a.original_file_name.as_str())
            .collect();
        if names.len() == 1 {
            let times: Option<Vec<i64>> = group
                .assets
                .iter()
                .map(asset_capture_time_seconds)
                .collect();
            let sizes: Option<Vec<u64>> = group.assets.iter().map(asset_size).collect();
            if let (Some(times), Some(sizes)) = (times, sizes) {
                let times_equal = times.iter().min() == times.iter().max();
                let sizes_close = sizes_within_tolerance(&sizes, AUTO_DELETE_MAX_SIZE_GAP);
                if times_equal && sizes_close {
                    let keeper_idx = group
                        .assets
                        .iter()
                        .position(|a| asset_gps(a).is_some())
                        .unwrap_or(0);
                    return GroupDecision::AutoDeleteDupes {
                        keeper_idx,
                        gps_backfill: None,
                    };
                }
            }
        }
    }

    if folders.len() != 1 {
        return GroupDecision::Skip(SkipReason::DifferentFolders);
    }

    // Rule 3 — capture-time spread.
    let mut times: Vec<i64> = Vec::with_capacity(group.assets.len());
    for a in &group.assets {
        match asset_capture_time_seconds(a) {
            Some(t) => times.push(t),
            None => return GroupDecision::Skip(SkipReason::MissingCaptureTime),
        }
    }
    let (min_t, max_t) = (
        *times.iter().min().unwrap(),
        *times.iter().max().unwrap(),
    );
    if max_t - min_t > max_time_gap_secs {
        return GroupDecision::Skip(SkipReason::TimeGapTooLarge);
    }

    // Rule 4 — GPS spread, but only when every member has coords. A
    // partial-GPS group is kept (GPS will be backfilled later).
    let gpses: Vec<(f64, f64)> = group.assets.iter().filter_map(asset_gps).collect();
    if gpses.len() == group.assets.len() {
        for i in 0..gpses.len() {
            for j in (i + 1)..gpses.len() {
                let (a, b) = (gpses[i], gpses[j]);
                if haversine_m(a.0, a.1, b.0, b.1) > max_distance_m {
                    return GroupDecision::Skip(SkipReason::DistanceTooLarge);
                }
            }
        }
    }

    GroupDecision::Consider
}

/// Rule 6.a — short-circuit pick by file size. Returns `Some(winner_idx)`
/// when the largest file is more than `size_ratio` larger than the
/// smallest; `None` means "vision model needs to decide" (sizes too
/// close, or any size unknown).
pub fn pick_winner_by_size(group: &DuplicateGroup, size_ratio: f64) -> Option<usize> {
    let sizes: Vec<u64> = group.assets.iter().map(asset_size).collect::<Option<_>>()?;
    let (min_size, max_size) = (*sizes.iter().min()?, *sizes.iter().max()?);
    if max_size == 0 {
        return None;
    }
    let ratio = 1.0 - (min_size as f64 / max_size as f64);
    if ratio > size_ratio {
        sizes.iter().position(|s| *s == max_size)
    } else {
        None
    }
}

// ---- vision prompt -------------------------------------------------------

const PICK_SYSTEM_PROMPT: &str =
    "你是一位帮助用户在一组高度相似的照片中挑选\"最值得保留的那一张\"的助手。\
你的输出会作为脚本的输入，必须严格遵守 JSON 格式，且只输出一个 JSON 对象，不要附加任何额外说明。";

const PICK_USER_PROMPT_HEAD: &str = "下面给你一组照片，它们被 Immich 判定为视觉上几乎相同的重复。\
请只从以下几个角度比较，挑出**最值得保留**的一张：\n\
1. 构图（取景、主体位置、画面平衡、有无明显裁切失误或手抖）\n\
2. 内容（人物表情/姿态是否最自然，主体是否最完整、是否被遮挡）\n\
3. 生动程度（光线、清晰度、色彩、瞬间感）\n\n\
4. 人物漂亮程度，偏向美颜风格\n\n\
比较时请忽略文件名、文件大小、缩略图分辨率差异；只看图像本身。\n\n\
请输出如下 JSON：\n\
```\n\
{\n\
  \"winner_index\": <整数，从 0 开始>,\n\
  \"reason\": \"<不超过 80 字的中文理由>\"\n\
}\n\
```\n\
不要输出 JSON 之外的任何文字。";

fn build_pick_user_prompt(n: usize) -> String {
    format!(
        "{PICK_USER_PROMPT_HEAD}\n\n本组共 {n} 张候选，索引 0 到 {}。",
        n.saturating_sub(1)
    )
}

#[derive(Debug, Deserialize)]
struct PickResponse {
    winner_index: usize,
    #[serde(default)]
    reason: String,
}

/// Parse the vision model's JSON answer and validate that the index is
/// in range. Lenient about wrapping markdown fences just in case.
pub fn parse_pick_response(raw: &str, n: usize) -> Result<(usize, String)> {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|s| s.trim_start_matches('\n'))
        .and_then(|s| s.strip_suffix("```").map(str::trim_end))
        .unwrap_or(trimmed);
    let parsed: PickResponse = serde_json::from_str(stripped)
        .with_context(|| format!("vision pick reply was not valid JSON: {raw}"))?;
    if parsed.winner_index >= n {
        bail!(
            "vision pick returned winner_index={} but group only has {n} assets",
            parsed.winner_index
        );
    }
    Ok((parsed.winner_index, parsed.reason))
}

// ---- per-group planning --------------------------------------------------

/// What the dedup pipeline ultimately decided to do with one accepted
/// group, after the (optional) vision call. Returned by [`plan_group`]
/// so the caller can preview (dry-run) or execute (`--apply`) without
/// re-running the picker.
#[derive(Debug, Clone)]
pub struct ActionPlan {
    pub winner_idx: usize,
    pub pick_reason: PickReason,
    /// GPS coords to push, alongside the asset ids that need them.
    /// Empty when every member already has GPS, or when no member does.
    pub gps_backfill: Vec<GpsCopy>,
    /// Asset ids in stack order; index 0 is the cover (winner).
    pub stack_order: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PickReason {
    LargerFile {
        /// The fraction `1 - smallest/largest`, for logging.
        size_gap: f64,
    },
    VisionModel {
        /// The model's free-form rationale (verbatim).
        rationale: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpsCopy {
    pub asset_id: String,
    pub latitude: f64,
    pub longitude: f64,
}

fn compute_gps_backfill(group: &DuplicateGroup, winner_idx: usize) -> Vec<GpsCopy> {
    let with_gps: Vec<(usize, (f64, f64))> = group
        .assets
        .iter()
        .enumerate()
        .filter_map(|(i, a)| asset_gps(a).map(|g| (i, g)))
        .collect();
    let without_gps: Vec<usize> = group
        .assets
        .iter()
        .enumerate()
        .filter(|(_, a)| asset_gps(a).is_none())
        .map(|(i, _)| i)
        .collect();
    if with_gps.is_empty() || without_gps.is_empty() {
        return Vec::new();
    }
    // Prefer the winner's GPS if it has any; otherwise just pick the
    // first sibling with coords. The user said "随便找一张有的".
    let (_, (lat, lng)) = with_gps
        .iter()
        .find(|(i, _)| *i == winner_idx)
        .copied()
        .unwrap_or(with_gps[0]);
    without_gps
        .into_iter()
        .map(|i| GpsCopy {
            asset_id: group.assets[i].id.clone(),
            latitude: lat,
            longitude: lng,
        })
        .collect()
}

fn build_stack_order(group: &DuplicateGroup, winner_idx: usize) -> Vec<String> {
    let mut ids = Vec::with_capacity(group.assets.len());
    ids.push(group.assets[winner_idx].id.clone());
    for (i, a) in group.assets.iter().enumerate() {
        if i != winner_idx {
            ids.push(a.id.clone());
        }
    }
    ids
}

/// Pick winner + plan GPS backfill + plan stack order for one already-
/// accepted group. Calls the vision LLM iff the size short-circuit
/// doesn't fire.
pub fn plan_group<C, V>(
    caption_be: &C,
    llm: &V,
    group: &DuplicateGroup,
    size_ratio: f64,
) -> Result<ActionPlan>
where
    C: CaptionBackend,
    V: MultiImageVisionLlm,
{
    let n = group.assets.len();
    let (winner_idx, pick_reason) = if let Some(idx) = pick_winner_by_size(group, size_ratio) {
        let sizes: Vec<u64> = group.assets.iter().filter_map(asset_size).collect();
        let (min_size, max_size) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
        let gap = 1.0 - (min_size as f64 / max_size as f64);
        (idx, PickReason::LargerFile { size_gap: gap })
    } else {
        // Fetch every thumbnail, then send to the vision model.
        let mut images: Vec<(Vec<u8>, &str)> = Vec::with_capacity(n);
        for a in &group.assets {
            let bytes = caption_be
                .thumbnail(&a.id)
                .with_context(|| format!("failed to fetch thumbnail for {}", a.id))?;
            images.push((bytes, "image/jpeg"));
        }
        let raw = llm
            .pick_best(
                PICK_SYSTEM_PROMPT,
                &build_pick_user_prompt(n),
                &images,
                MAX_PICK_TOKENS,
            )
            .context("vision pick failed")?;
        let (idx, reason) = parse_pick_response(&raw, n)?;
        (idx, PickReason::VisionModel { rationale: reason })
    };

    Ok(ActionPlan {
        winner_idx,
        pick_reason,
        gps_backfill: compute_gps_backfill(group, winner_idx),
        stack_order: build_stack_order(group, winner_idx),
    })
}

// ---- orchestration -------------------------------------------------------

pub fn run(cfg: &Config, args: DedupArgs) -> Result<()> {
    let llm_cfg = cfg.llm.as_ref().ok_or_else(|| {
        anyhow!(
            "dedup requires an [llm] section in config.toml with \
             vision_model set (used to pick the best photo from each group)"
        )
    })?;
    if llm_cfg.vision_model.as_deref().unwrap_or("").is_empty() {
        bail!(
            "dedup requires config.llm.vision_model — set it to a \
             vision-capable model (e.g. \"mimo-v2.5\")"
        );
    }
    let client = ImmichClient::new(cfg)?;
    let llm = OpenAiClient::new(llm_cfg)?;
    let mut stderr = std::io::stderr();
    run_with(&client, &client, &client, &client, &llm, args, &mut stderr)
}

/// Test/library entry point: decouples the runtime from concrete backends.
pub fn run_with<D, S, C, W, V, Log>(
    duplicates_be: &D,
    stacks_be: &S,
    caption_be: &C,
    write_be: &W,
    llm: &V,
    args: DedupArgs,
    log: &mut Log,
) -> Result<()>
where
    D: DuplicatesBackend,
    S: StacksBackend,
    C: CaptionBackend + Sync,
    W: DedupWriteBackend + Sync,
    V: MultiImageVisionLlm + Sync,
    Log: std::io::Write + Send,
{
    if args.parallel == 0 {
        bail!("--parallel must be > 0");
    }
    if args.max_time_gap_secs < 0 {
        bail!("--max-time-gap-secs must be >= 0");
    }
    if args.max_distance_m < 0.0 {
        bail!("--max-distance-m must be >= 0");
    }
    if !(0.0..=1.0).contains(&args.size_ratio) {
        bail!("--size-ratio must be in [0, 1]");
    }

    writeln!(log, "fetching duplicates ...").ok();
    let groups = duplicates_be.duplicates()?;
    writeln!(log, "{} duplicate group(s) reported by server", groups.len()).ok();

    writeln!(log, "fetching existing stacks ...").ok();
    let stacks = stacks_be.stacks()?;
    let mut stacked: HashSet<String> = HashSet::new();
    for s in &stacks {
        stacked.insert(s.primary_asset_id.clone());
        for m in &s.assets {
            stacked.insert(m.id.clone());
        }
    }

    // Phase 1 — classify every group sequentially. Pure CPU work
    // (string ops, haversine, time parsing), no IO, so there's no
    // reason to parallelize it. Side effect: verbose skip lines are
    // emitted here, in source order, before any work output.
    let mut counts = Counts::default();
    let mut to_process: Vec<&DuplicateGroup> = Vec::new();
    let mut to_auto_delete: Vec<(&DuplicateGroup, usize, Option<(f64, f64)>)> = Vec::new();
    let mut to_keep_all: Vec<&DuplicateGroup> = Vec::new();
    for group in &groups {
        match classify_group(group, &stacked, args.max_time_gap_secs, args.max_distance_m) {
            GroupDecision::Skip(reason) => {
                counts.record_skip(&reason);
                if args.verbose {
                    log_skip(log, &reason, group).ok();
                }
            }
            GroupDecision::AutoDeleteDupes { keeper_idx, gps_backfill } => {
                to_auto_delete.push((group, keeper_idx, gps_backfill));
            }
            GroupDecision::KeepAll => to_keep_all.push(group),
            GroupDecision::Consider => to_process.push(group),
        }
    }

    if args.limit > 0 && (args.limit as usize) < to_process.len() {
        to_process.truncate(args.limit as usize);
    }

    // Phase 2a — auto-delete exact cross-folder duplicates. Run before
    // the vision pipeline so the user sees these cheap actions first
    // and so a small `--limit` doesn't starve them out. Sequential: the
    // work per group is a single DELETE, parallelism wouldn't help.
    let auto_del = run_auto_deletes(write_be, &to_auto_delete, args.apply, log);
    counts.merge_auto_delete(&auto_del);

    // Phase 2a' — detach KeepAll groups from their server-side duplicate
    // grouping so they stop reappearing. Pure PUT, no GPS/stack work.
    let keep = run_keep_all(write_be, &to_keep_all, args.apply, log);
    counts.merge_keep_all(&keep);

    // Phase 2b — plan + apply, parallelized across groups. Each worker
    // does N thumbnail fetches + 1 vision call + (in --apply mode) M
    // PUTs + 1 POST, so the work is almost entirely IO-bound and
    // parallelizing helps.
    let work = process_groups_in_parallel(
        caption_be,
        llm,
        write_be,
        &to_process,
        &args,
        log,
    );
    counts.merge_work(&work);

    counts.write_summary(log, args.apply).ok();
    Ok(())
}

fn log_skip<W: std::io::Write>(
    log: &mut W,
    reason: &SkipReason,
    group: &DuplicateGroup,
) -> std::io::Result<()> {
    writeln!(
        log,
        "skip [{:?}] group {} ({} assets)",
        reason,
        group.duplicate_id,
        group.assets.len()
    )?;
    if *reason == SkipReason::MissingCaptureTime {
        // Surface the strings we failed to parse so the user can spot
        // a new format we don't handle.
        for a in &group.assets {
            writeln!(
                log,
                "  asset {} times: dateTimeOriginal={:?} localDateTime={:?} fileCreatedAt={:?}",
                a.id,
                a.exif_info.as_ref().and_then(|e| e.date_time_original.as_deref()),
                a.local_date_time.as_deref(),
                a.file_created_at.as_deref(),
            )?;
        }
    }
    if *reason == SkipReason::TimeGapTooLarge {
        // Print earliest/latest and per-asset capture-time strings so
        // the user can tell which tier of the time guard they hit (the
        // 10-min user rule vs. the hard 6h cap) and spot wildly wrong
        // EXIF values.
        for a in &group.assets {
            writeln!(
                log,
                "  asset {} path={} dateTimeOriginal={:?} localDateTime={:?} fileCreatedAt={:?}",
                a.id,
                a.original_path,
                a.exif_info.as_ref().and_then(|e| e.date_time_original.as_deref()),
                a.local_date_time.as_deref(),
                a.file_created_at.as_deref(),
            )?;
        }
    }
    Ok(())
}

/// Counters for the auto-delete phase, folded into the run's [`Counts`]
/// by the caller.
#[derive(Default)]
struct AutoDeleteCounts {
    would_delete_groups: u32,
    would_delete_assets: u32,
    deleted_groups: u32,
    deleted_assets: u32,
    delete_failed: u32,
}

/// Counters for the KeepAll phase (groups beyond the 6h hard cap whose
/// `duplicateId` we detach on the server).
#[derive(Default)]
struct KeepAllCounts {
    would_keep_groups: u32,
    would_keep_assets: u32,
    kept_groups: u32,
    kept_assets: u32,
    keep_failed: u32,
}

/// Execute (or, in dry-run, log) the auto-delete decisions for exact
/// cross-folder duplicates. Errors on individual groups are logged but
/// do not abort the run.
fn run_auto_deletes<W, Log>(
    write_be: &W,
    groups: &[(&DuplicateGroup, usize, Option<(f64, f64)>)],
    apply: bool,
    log: &mut Log,
) -> AutoDeleteCounts
where
    W: DedupWriteBackend,
    Log: std::io::Write,
{
    let mut counts = AutoDeleteCounts::default();
    let prefix = if apply { "AUTO-DELETE" } else { "DRY-RUN AUTO-DELETE" };
    for (group, keeper_idx, gps_backfill) in groups {
        let keeper = &group.assets[*keeper_idx];
        let victims: Vec<String> = group
            .assets
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != *keeper_idx)
            .map(|(_, a)| a.id.clone())
            .collect();
        if victims.is_empty() {
            continue;
        }
        let keeper_gps = if asset_gps(keeper).is_some() { "GPS" } else { "no-GPS" };
        writeln!(
            log,
            "{prefix} group {} ({} assets) → keep {} {} [{keeper_gps}], delete {}",
            group.duplicate_id,
            group.assets.len(),
            keeper.id,
            keeper.original_path,
            victims.len(),
        )
        .ok();
        if let Some((lat, lng)) = gps_backfill {
            writeln!(
                log,
                "  {prefix} GPS {} → ({:.6}, {:.6})",
                keeper.id, lat, lng
            )
            .ok();
        }
        for (i, a) in group.assets.iter().enumerate() {
            if i != *keeper_idx {
                writeln!(log, "  {prefix} {} {}", a.id, a.original_path).ok();
            }
        }

        if !apply {
            counts.would_delete_groups += 1;
            counts.would_delete_assets += victims.len() as u32;
            continue;
        }
        // Backfill GPS onto the keeper first so the location survives
        // even if the GPS source is one of the victims we're about to
        // delete. A backfill failure is logged but does not abort the
        // delete — the dedup is still the user's primary intent.
        if let Some((lat, lng)) = gps_backfill {
            if let Err(e) = write_be.update_asset_location(&keeper.id, *lat, *lng) {
                writeln!(
                    log,
                    "FAIL backfill GPS for {} (group {}) — {e:#}",
                    keeper.id, group.duplicate_id
                )
                .ok();
            }
        }
        match write_be.delete_assets(&victims) {
            Ok(()) => {
                counts.deleted_groups += 1;
                counts.deleted_assets += victims.len() as u32;
            }
            Err(e) => {
                counts.delete_failed += 1;
                writeln!(
                    log,
                    "FAIL delete group {} — {e:#}",
                    group.duplicate_id
                )
                .ok();
            }
        }
    }
    counts
}

/// Execute (or, in dry-run, log) the KeepAll decisions. Each group is
/// dismissed server-side via `DELETE /api/duplicates/{id}` so its
/// members stop appearing in `GET /api/duplicates`. Errors on
/// individual groups are logged but do not abort the run.
fn run_keep_all<W, Log>(
    write_be: &W,
    groups: &[&DuplicateGroup],
    apply: bool,
    log: &mut Log,
) -> KeepAllCounts
where
    W: DedupWriteBackend,
    Log: std::io::Write,
{
    let mut counts = KeepAllCounts::default();
    let prefix = if apply { "KEEP-ALL" } else { "DRY-RUN KEEP-ALL" };
    for group in groups {
        let n_assets = group.assets.len() as u32;
        writeln!(
            log,
            "{prefix} group {} ({} assets) — capture-time spread > 6h, dismissing on server",
            group.duplicate_id, n_assets,
        )
        .ok();
        for a in &group.assets {
            writeln!(log, "  {prefix} {} {}", a.id, a.original_path).ok();
        }
        if !apply {
            counts.would_keep_groups += 1;
            counts.would_keep_assets += n_assets;
            continue;
        }
        match write_be.dismiss_duplicate_group(&group.duplicate_id) {
            Ok(()) => {
                counts.kept_groups += 1;
                counts.kept_assets += n_assets;
            }
            Err(e) => {
                counts.keep_failed += 1;
                writeln!(
                    log,
                    "FAIL keep-all group {} — {e:#}",
                    group.duplicate_id
                )
                .ok();
            }
        }
    }
    counts
}

/// Counters touched by the parallel workers. Returned by
/// [`process_groups_in_parallel`] and folded into the run's [`Counts`]
/// by the caller — keeps the parallel code free of borrowing concerns.
#[derive(Default)]
struct WorkCounts {
    plan_failed: u32,
    would_apply: u32,
    applied: u32,
    applied_with_warnings: u32,
    apply_failed: u32,
}

/// Plan + (optionally) apply each accepted group in parallel. Errors
/// on individual groups are logged but do not abort the whole run.
fn process_groups_in_parallel<C, V, W, Log>(
    caption_be: &C,
    llm: &V,
    write_be: &W,
    groups: &[&DuplicateGroup],
    args: &DedupArgs,
    log: &mut Log,
) -> WorkCounts
where
    C: CaptionBackend + Sync,
    W: DedupWriteBackend + Sync,
    V: MultiImageVisionLlm + Sync,
    Log: std::io::Write + Send,
{
    if groups.is_empty() {
        return WorkCounts::default();
    }
    let plan_failed = AtomicU32::new(0);
    let would_apply = AtomicU32::new(0);
    let applied = AtomicU32::new(0);
    let applied_with_warnings = AtomicU32::new(0);
    let apply_failed = AtomicU32::new(0);

    let next = AtomicUsize::new(0);
    let log_lock = Mutex::new(log);
    let parallel = (args.parallel as usize).min(groups.len()).max(1);
    // `args` field reads here so the closure captures only Copy values.
    let apply = args.apply;
    let size_ratio = args.size_ratio;

    std::thread::scope(|scope| {
        for _ in 0..parallel {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= groups.len() {
                    break;
                }
                let group = groups[i];

                let plan = match plan_group(caption_be, llm, group, size_ratio) {
                    Ok(p) => p,
                    Err(e) => {
                        plan_failed.fetch_add(1, Ordering::SeqCst);
                        let mut log = log_lock.lock().unwrap();
                        writeln!(*log, "FAIL plan group {} — {e:#}", group.duplicate_id).ok();
                        continue;
                    }
                };

                {
                    let mut log = log_lock.lock().unwrap();
                    log_plan(&mut *log, group, &plan, apply).ok();
                }

                if !apply {
                    would_apply.fetch_add(1, Ordering::SeqCst);
                    continue;
                }

                let mut any_failure = false;
                for copy in &plan.gps_backfill {
                    if let Err(e) = write_be.update_asset_location(
                        &copy.asset_id,
                        copy.latitude,
                        copy.longitude,
                    ) {
                        any_failure = true;
                        let mut log = log_lock.lock().unwrap();
                        writeln!(
                            *log,
                            "FAIL backfill GPS for {} (group {}) — {e:#}",
                            copy.asset_id, group.duplicate_id
                        )
                        .ok();
                    }
                }
                match write_be.create_stack(&plan.stack_order) {
                    Ok(_) => {
                        if any_failure {
                            applied_with_warnings.fetch_add(1, Ordering::SeqCst);
                        } else {
                            applied.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                    Err(e) => {
                        apply_failed.fetch_add(1, Ordering::SeqCst);
                        let mut log = log_lock.lock().unwrap();
                        writeln!(
                            *log,
                            "FAIL create stack for group {} — {e:#}",
                            group.duplicate_id
                        )
                        .ok();
                    }
                }
            });
        }
    });

    WorkCounts {
        plan_failed: plan_failed.into_inner(),
        would_apply: would_apply.into_inner(),
        applied: applied.into_inner(),
        applied_with_warnings: applied_with_warnings.into_inner(),
        apply_failed: apply_failed.into_inner(),
    }
}

fn log_plan<W: std::io::Write>(
    log: &mut W,
    group: &DuplicateGroup,
    plan: &ActionPlan,
    apply: bool,
) -> std::io::Result<()> {
    let prefix = if apply { "PLAN" } else { "DRY-RUN" };
    let winner = &group.assets[plan.winner_idx];
    let pick_desc = match &plan.pick_reason {
        PickReason::LargerFile { size_gap } => {
            format!("size (largest, {:.0}% gap)", size_gap * 100.0)
        }
        PickReason::VisionModel { rationale } => format!("vision: {rationale}"),
    };
    writeln!(
        log,
        "{prefix} stack group {} ({} assets) → winner {} {} [{}]",
        group.duplicate_id,
        group.assets.len(),
        winner.id,
        winner.original_path,
        pick_desc
    )?;
    for copy in &plan.gps_backfill {
        writeln!(
            log,
            "  {prefix} GPS {} → ({:.6}, {:.6})",
            copy.asset_id, copy.latitude, copy.longitude
        )?;
    }
    Ok(())
}

#[derive(Default)]
struct Counts {
    already_stacked: u32,
    too_few: u32,
    contains_video: u32,
    different_folders: u32,
    time_gap_too_large: u32,
    distance_too_large: u32,
    missing_capture_time: u32,
    plan_failed: u32,
    would_apply: u32,
    applied: u32,
    applied_with_warnings: u32,
    apply_failed: u32,
    would_delete_groups: u32,
    would_delete_assets: u32,
    deleted_groups: u32,
    deleted_assets: u32,
    delete_failed: u32,
    would_keep_groups: u32,
    would_keep_assets: u32,
    kept_groups: u32,
    kept_assets: u32,
    keep_failed: u32,
}

impl Counts {
    fn record_skip(&mut self, reason: &SkipReason) {
        match reason {
            SkipReason::AlreadyStacked => self.already_stacked += 1,
            SkipReason::TooFewAssets => self.too_few += 1,
            SkipReason::ContainsVideo => self.contains_video += 1,
            SkipReason::DifferentFolders => self.different_folders += 1,
            SkipReason::TimeGapTooLarge => self.time_gap_too_large += 1,
            SkipReason::DistanceTooLarge => self.distance_too_large += 1,
            SkipReason::MissingCaptureTime => self.missing_capture_time += 1,
        }
    }

    fn merge_work(&mut self, w: &WorkCounts) {
        self.plan_failed += w.plan_failed;
        self.would_apply += w.would_apply;
        self.applied += w.applied;
        self.applied_with_warnings += w.applied_with_warnings;
        self.apply_failed += w.apply_failed;
    }

    fn merge_auto_delete(&mut self, d: &AutoDeleteCounts) {
        self.would_delete_groups += d.would_delete_groups;
        self.would_delete_assets += d.would_delete_assets;
        self.deleted_groups += d.deleted_groups;
        self.deleted_assets += d.deleted_assets;
        self.delete_failed += d.delete_failed;
    }

    fn merge_keep_all(&mut self, k: &KeepAllCounts) {
        self.would_keep_groups += k.would_keep_groups;
        self.would_keep_assets += k.would_keep_assets;
        self.kept_groups += k.kept_groups;
        self.kept_assets += k.kept_assets;
        self.keep_failed += k.keep_failed;
    }

    fn write_summary<W: std::io::Write>(
        &self,
        log: &mut W,
        apply: bool,
    ) -> std::io::Result<()> {
        writeln!(
            log,
            "skipped: already_stacked={} too_few={} video={} folders={} time_gap={} distance={} no_time={}",
            self.already_stacked,
            self.too_few,
            self.contains_video,
            self.different_folders,
            self.time_gap_too_large,
            self.distance_too_large,
            self.missing_capture_time,
        )?;
        if apply {
            writeln!(
                log,
                "auto-deleted: {} group(s), {} asset(s); {} failed",
                self.deleted_groups, self.deleted_assets, self.delete_failed,
            )?;
            writeln!(
                log,
                "kept-all (>6h spread): {} group(s), {} asset(s); {} failed",
                self.kept_groups, self.kept_assets, self.keep_failed,
            )?;
            writeln!(
                log,
                "applied: {} ok, {} ok-with-warnings, {} stack failed, {} plan failed",
                self.applied, self.applied_with_warnings, self.apply_failed, self.plan_failed,
            )?;
        } else {
            writeln!(
                log,
                "would auto-delete: {} group(s), {} asset(s)",
                self.would_delete_groups, self.would_delete_assets,
            )?;
            writeln!(
                log,
                "would keep-all (>6h spread): {} group(s), {} asset(s)",
                self.would_keep_groups, self.would_keep_assets,
            )?;
            writeln!(
                log,
                "would apply: {} group(s); {} plan failed",
                self.would_apply, self.plan_failed,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ExifInfo, Stack, StackMember};
    use std::collections::HashMap;

    fn asset(id: &str, path: &str) -> Asset {
        Asset {
            id: id.into(),
            original_path: path.into(),
            original_file_name: path.rsplit('/').next().unwrap_or(path).into(),
            asset_type: "IMAGE".into(),
            file_created_at: None,
            local_date_time: Some("2024-03-12T17:08:42".into()),
            checksum: String::new(),
            exif_info: Some(ExifInfo {
                date_time_original: Some("2024-03-12T17:08:42".into()),
                ..Default::default()
            }),
        }
    }

    fn video(id: &str, path: &str) -> Asset {
        let mut a = asset(id, path);
        a.asset_type = "VIDEO".into();
        a
    }

    fn with_time(mut a: Asset, t: &str) -> Asset {
        if let Some(e) = a.exif_info.as_mut() {
            e.date_time_original = Some(t.into());
        }
        a.local_date_time = Some(t.into());
        a
    }

    fn with_size(mut a: Asset, sz: u64) -> Asset {
        if let Some(e) = a.exif_info.as_mut() {
            e.file_size_in_byte = Some(sz);
        }
        a
    }

    fn with_gps(mut a: Asset, lat: f64, lng: f64) -> Asset {
        if let Some(e) = a.exif_info.as_mut() {
            e.latitude = Some(lat);
            e.longitude = Some(lng);
        }
        a
    }

    fn group(id: &str, assets: Vec<Asset>) -> DuplicateGroup {
        DuplicateGroup {
            duplicate_id: id.into(),
            assets,
        }
    }

    #[test]
    fn parent_dir_strips_filename() {
        assert_eq!(parent_dir("/a/b/c.jpg"), "/a/b");
        assert_eq!(parent_dir("/a/b/"), "/a/b");
        assert_eq!(parent_dir("foo.jpg"), "");
    }

    #[test]
    fn haversine_zero_when_same_point() {
        assert!(haversine_m(31.0, 121.0, 31.0, 121.0).abs() < 1e-6);
    }

    #[test]
    fn haversine_roughly_correct_for_500m() {
        // ~0.0045 degrees latitude ≈ 500m
        let d = haversine_m(31.0, 121.0, 31.0 + 0.0045, 121.0);
        assert!((d - 500.0).abs() < 5.0, "got {d}");
    }

    #[test]
    fn parse_capture_seconds_accepts_both_with_and_without_millis() {
        let a = parse_capture_seconds("2024-03-12T17:08:42").unwrap();
        let b = parse_capture_seconds("2024-03-12T17:08:42.000").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn parse_capture_seconds_accepts_rfc3339_zulu_and_offset() {
        // Z and +08:00 must both parse to the same absolute instant.
        let zulu = parse_capture_seconds("2024-03-12T09:08:42.000Z").unwrap();
        let plus8 = parse_capture_seconds("2024-03-12T17:08:42.000+08:00").unwrap();
        assert_eq!(zulu, plus8, "Z and +08:00 should denote the same instant");
        // And without sub-second.
        let zulu_nofrac = parse_capture_seconds("2024-03-12T09:08:42Z").unwrap();
        assert_eq!(zulu_nofrac, zulu);
    }

    #[test]
    fn classify_skip_video() {
        let mut v = asset("v", "/p/a.mp4");
        v.asset_type = "VIDEO".into();
        let g = group("g", vec![asset("a", "/p/a.jpg"), v]);
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::ContainsVideo));
    }

    #[test]
    fn classify_skip_different_folders() {
        let g = group(
            "g",
            vec![asset("a", "/p/a.jpg"), asset("b", "/q/b.jpg")],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn classify_skip_time_gap() {
        let g = group(
            "g",
            vec![
                with_time(asset("a", "/p/a.jpg"), "2024-03-12T17:08:42"),
                with_time(asset("b", "/p/b.jpg"), "2024-03-12T17:30:00"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::TimeGapTooLarge));
    }

    #[test]
    fn classify_keep_all_when_time_spread_exceeds_six_hours() {
        // > 6h spread: KeepAll so the runner detaches duplicateId on
        // the server, regardless of an over-permissive --max-time-gap-secs.
        let g = group(
            "g",
            vec![
                with_time(asset("a", "/p/a.jpg"), "2024-03-12T10:00:00"),
                with_time(asset("b", "/p/b.jpg"), "2024-03-12T16:00:01"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), i64::MAX, 500.0);
        assert_eq!(dec, GroupDecision::KeepAll);
    }

    #[test]
    fn classify_six_hour_guard_blocks_auto_delete() {
        // Cross-folder same-name+same-size would normally auto-delete,
        // but with a > 6h capture-time spread we KeepAll instead.
        let g = group(
            "g",
            vec![
                with_time(
                    with_size(asset("a", "/x/IMG_1.jpg"), 2_000_000),
                    "2024-03-12T10:00:00",
                ),
                with_time(
                    with_size(asset("b", "/y/IMG_1.jpg"), 2_000_000),
                    "2024-03-12T17:00:00",
                ),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::KeepAll);
    }

    #[test]
    fn classify_skip_distance() {
        let g = group(
            "g",
            vec![
                with_gps(asset("a", "/p/a.jpg"), 31.0, 121.0),
                with_gps(asset("b", "/p/b.jpg"), 31.02, 121.0), // ~2.2km
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DistanceTooLarge));
    }

    #[test]
    fn classify_skip_already_stacked() {
        let g = group(
            "g",
            vec![asset("a", "/p/a.jpg"), asset("b", "/p/b.jpg")],
        );
        let mut stacked = HashSet::new();
        stacked.insert("b".to_string());
        let dec = classify_group(&g, &stacked, 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::AlreadyStacked));
    }

    #[test]
    fn classify_keeps_group_when_partial_gps() {
        // One GPS, one missing — should NOT skip on distance.
        let g = group(
            "g",
            vec![
                with_gps(asset("a", "/p/a.jpg"), 31.0, 121.0),
                asset("b", "/p/b.jpg"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Consider);
    }

    #[test]
    fn classify_keeps_well_formed_group() {
        let g = group(
            "g",
            vec![
                with_gps(asset("a", "/p/a.jpg"), 31.0, 121.0),
                with_gps(asset("b", "/p/b.jpg"), 31.0001, 121.0001),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Consider);
    }

    #[test]
    fn classify_auto_deletes_same_name_same_time_same_size_diff_folder_prefers_gps() {
        // Same basename, same capture time, same byte size, two folders.
        // Only b has GPS, so b must be the keeper even though a comes
        // first.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/IMG_001.jpg"), 2_000_000),
                with_size(with_gps(asset("b", "/y/IMG_001.jpg"), 31.0, 121.0), 2_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::AutoDeleteDupes { keeper_idx: 1, gps_backfill: None });
    }

    #[test]
    fn classify_auto_deletes_all_with_gps_picks_first() {
        // All three have GPS → "随便选一张" → the first (index 0).
        let g = group(
            "g",
            vec![
                with_size(with_gps(asset("a", "/x/IMG_1.jpg"), 31.0, 121.0), 2_000_000),
                with_size(with_gps(asset("b", "/y/IMG_1.jpg"), 31.0, 121.0), 2_000_000),
                with_size(with_gps(asset("c", "/z/IMG_1.jpg"), 31.0, 121.0), 2_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::AutoDeleteDupes { keeper_idx: 0, gps_backfill: None });
    }

    #[test]
    fn classify_auto_deletes_none_with_gps_picks_first() {
        // None have GPS — still safe to dedupe by hand, keep the first.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/IMG_1.jpg"), 2_000_000),
                with_size(asset("b", "/y/IMG_1.jpg"), 2_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::AutoDeleteDupes { keeper_idx: 0, gps_backfill: None });
    }

    #[test]
    fn classify_does_not_auto_delete_when_sizes_differ() {
        // Same name + same time + different folder but DIFFERENT sizes
        // — the classic /raw/ vs /retouched/ pair. Must fall through
        // to the normal folder skip, not auto-delete.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/raw/IMG_1.jpg"), 5_000_000),
                with_size(asset("b", "/retouched/IMG_1.jpg"), 3_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn classify_auto_deletes_when_sizes_differ_within_tolerance() {
        // 2_000_000 vs 2_001_000 → 0.05% gap, under the 0.1% ceiling.
        // EXIF-rewrite shouldn't block the rule.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/IMG_1.jpg"), 2_000_000),
                with_size(asset("b", "/y/IMG_1.jpg"), 2_001_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::AutoDeleteDupes { keeper_idx: 0, gps_backfill: None });
    }

    #[test]
    fn classify_does_not_auto_delete_when_sizes_just_outside_tolerance() {
        // 2_000_000 vs 2_010_000 → 0.5% gap, over the 0.1% ceiling.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/IMG_1.jpg"), 2_000_000),
                with_size(asset("b", "/y/IMG_1.jpg"), 2_010_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn classify_does_not_auto_delete_when_any_size_missing() {
        // Without a size we can't prove byte-equality, so be safe.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/IMG_1.jpg"), 2_000_000),
                asset("b", "/y/IMG_1.jpg"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn classify_auto_deletes_video_keeps_largest_no_backfill_needed() {
        // Two videos, same name + same time, different folders, sizes
        // differ. Keeper = larger (b). Both have GPS → no backfill.
        let g = group(
            "g",
            vec![
                with_size(with_gps(video("a", "/x/VID.mp4"), 31.0, 121.0), 5_000_000),
                with_size(with_gps(video("b", "/y/VID.mp4"), 31.0, 121.0), 9_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(
            dec,
            GroupDecision::AutoDeleteDupes { keeper_idx: 1, gps_backfill: None }
        );
    }

    #[test]
    fn classify_auto_deletes_video_backfills_gps_to_keeper() {
        // Keeper (larger) has no GPS; sibling does → must plan a
        // backfill onto the keeper before deletion.
        let g = group(
            "g",
            vec![
                with_size(with_gps(video("a", "/x/VID.mp4"), 31.5, 121.5), 5_000_000),
                with_size(video("b", "/y/VID.mp4"), 9_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(
            dec,
            GroupDecision::AutoDeleteDupes {
                keeper_idx: 1,
                gps_backfill: Some((31.5, 121.5)),
            }
        );
    }

    #[test]
    fn classify_does_not_auto_delete_videos_when_sizes_equal() {
        // Equal-size videos: user explicitly wanted size-DIFFERS as the
        // trigger, so this falls through to the ContainsVideo skip.
        let g = group(
            "g",
            vec![
                with_size(video("a", "/x/VID.mp4"), 5_000_000),
                with_size(video("b", "/y/VID.mp4"), 5_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::ContainsVideo));
    }

    #[test]
    fn classify_does_not_auto_delete_videos_when_names_differ() {
        let g = group(
            "g",
            vec![
                with_size(video("a", "/x/VID_1.mp4"), 5_000_000),
                with_size(video("b", "/y/VID_2.mp4"), 9_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::ContainsVideo));
    }

    #[test]
    fn classify_does_not_auto_delete_mixed_image_video_group() {
        // One image + one video → not all-videos → falls through to
        // ContainsVideo skip even with matching name/time.
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/x/foo.jpg"), 5_000_000),
                with_size(video("b", "/y/foo.jpg"), 9_000_000),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::ContainsVideo));
    }

    #[test]
    fn classify_does_not_auto_delete_when_filenames_differ() {
        // Same folder pair, different filenames → fall through to the
        // usual DifferentFolders skip.
        let g = group(
            "g",
            vec![
                asset("a", "/x/IMG_1.jpg"),
                asset("b", "/y/IMG_2.jpg"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn classify_does_not_auto_delete_when_capture_times_differ() {
        let g = group(
            "g",
            vec![
                with_time(asset("a", "/x/IMG_1.jpg"), "2024-03-12T17:08:42"),
                with_time(asset("b", "/y/IMG_1.jpg"), "2024-03-12T17:08:43"),
            ],
        );
        let dec = classify_group(&g, &HashSet::new(), 600, 500.0);
        // Same filename, but times differ by 1s → not an exact dup.
        // Two folders, two times in the same gap → falls into
        // DifferentFolders skip.
        assert_eq!(dec, GroupDecision::Skip(SkipReason::DifferentFolders));
    }

    #[test]
    fn size_pick_short_circuits_when_one_much_larger() {
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/p/a.jpg"), 1_000_000),
                with_size(asset("b", "/p/b.jpg"), 3_000_000),
                with_size(asset("c", "/p/c.jpg"), 1_500_000),
            ],
        );
        // 1 - 1/3 = 0.666... > 0.5 → pick index 1
        assert_eq!(pick_winner_by_size(&g, 0.5), Some(1));
    }

    #[test]
    fn size_pick_returns_none_when_close() {
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/p/a.jpg"), 1_000_000),
                with_size(asset("b", "/p/b.jpg"), 1_200_000),
            ],
        );
        assert_eq!(pick_winner_by_size(&g, 0.5), None);
    }

    #[test]
    fn size_pick_returns_none_when_any_size_missing() {
        let g = group(
            "g",
            vec![
                with_size(asset("a", "/p/a.jpg"), 1_000_000),
                asset("b", "/p/b.jpg"),
            ],
        );
        assert_eq!(pick_winner_by_size(&g, 0.5), None);
    }

    #[test]
    fn parse_pick_response_extracts_index_and_reason() {
        let raw = r#"{"winner_index": 1, "reason": "构图更稳"}"#;
        let (idx, reason) = parse_pick_response(raw, 3).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(reason, "构图更稳");
    }

    #[test]
    fn parse_pick_response_tolerates_markdown_fence() {
        let raw = "```json\n{\"winner_index\": 0, \"reason\": \"x\"}\n```";
        let (idx, _) = parse_pick_response(raw, 2).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn parse_pick_response_rejects_out_of_range() {
        let raw = r#"{"winner_index": 5, "reason": "x"}"#;
        assert!(parse_pick_response(raw, 2).is_err());
    }

    #[test]
    fn gps_backfill_when_some_missing() {
        let g = group(
            "g",
            vec![
                with_gps(asset("a", "/p/a.jpg"), 31.0, 121.0),
                asset("b", "/p/b.jpg"),
                asset("c", "/p/c.jpg"),
            ],
        );
        // winner is index 0 (which has GPS)
        let backfill = compute_gps_backfill(&g, 0);
        assert_eq!(backfill.len(), 2);
        assert_eq!(backfill[0].asset_id, "b");
        assert_eq!(backfill[0].latitude, 31.0);
        assert_eq!(backfill[1].asset_id, "c");
    }

    #[test]
    fn gps_backfill_uses_sibling_when_winner_missing() {
        let g = group(
            "g",
            vec![
                asset("a", "/p/a.jpg"),
                with_gps(asset("b", "/p/b.jpg"), 30.0, 120.0),
            ],
        );
        // winner is index 0 (no GPS); should still backfill onto 'a'
        // using b's coords. With "any with GPS" as source, we expect:
        //   - a gets backfilled from b
        let backfill = compute_gps_backfill(&g, 0);
        assert_eq!(backfill.len(), 1);
        assert_eq!(backfill[0].asset_id, "a");
        assert_eq!(backfill[0].latitude, 30.0);
    }

    #[test]
    fn gps_backfill_empty_when_none_have_gps() {
        let g = group("g", vec![asset("a", "/p/a.jpg"), asset("b", "/p/b.jpg")]);
        assert!(compute_gps_backfill(&g, 0).is_empty());
    }

    #[test]
    fn gps_backfill_empty_when_all_have_gps() {
        let g = group(
            "g",
            vec![
                with_gps(asset("a", "/p/a.jpg"), 31.0, 121.0),
                with_gps(asset("b", "/p/b.jpg"), 31.0001, 121.0001),
            ],
        );
        assert!(compute_gps_backfill(&g, 0).is_empty());
    }

    #[test]
    fn stack_order_puts_winner_first() {
        let g = group(
            "g",
            vec![
                asset("a", "/p/a.jpg"),
                asset("b", "/p/b.jpg"),
                asset("c", "/p/c.jpg"),
            ],
        );
        let order = build_stack_order(&g, 1);
        assert_eq!(order, vec!["b", "a", "c"]);
    }

    // ---- end-to-end via run_with --------------------------------------

    struct FakeDuplicates {
        groups: Vec<DuplicateGroup>,
    }
    impl DuplicatesBackend for FakeDuplicates {
        fn duplicates(&self) -> Result<Vec<DuplicateGroup>> {
            Ok(self.groups.clone())
        }
    }

    struct FakeStacks {
        stacks: Vec<Stack>,
    }
    impl StacksBackend for FakeStacks {
        fn stacks(&self) -> Result<Vec<Stack>> {
            Ok(self.stacks.clone())
        }
    }

    struct FakeCaption;
    impl CaptionBackend for FakeCaption {
        fn thumbnail(&self, _id: &str) -> Result<Vec<u8>> {
            Ok(vec![0u8; 16])
        }
        fn update_description(&self, _id: &str, _description: &str) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeWrite {
        locations: Mutex<Vec<(String, f64, f64)>>,
        stacks: Mutex<Vec<Vec<String>>>,
        deletes: Mutex<Vec<Vec<String>>>,
        dismissed_groups: Mutex<Vec<String>>,
    }
    impl DedupWriteBackend for FakeWrite {
        fn update_asset_location(
            &self,
            id: &str,
            latitude: f64,
            longitude: f64,
        ) -> Result<()> {
            self.locations
                .lock()
                .unwrap()
                .push((id.into(), latitude, longitude));
            Ok(())
        }
        fn create_stack(&self, asset_ids: &[String]) -> Result<Stack> {
            self.stacks.lock().unwrap().push(asset_ids.to_vec());
            Ok(Stack {
                primary_asset_id: asset_ids[0].clone(),
                assets: asset_ids
                    .iter()
                    .map(|id| StackMember { id: id.clone() })
                    .collect(),
            })
        }
        fn delete_assets(&self, ids: &[String]) -> Result<()> {
            self.deletes.lock().unwrap().push(ids.to_vec());
            Ok(())
        }
        fn dismiss_duplicate_group(&self, duplicate_id: &str) -> Result<()> {
            self.dismissed_groups
                .lock()
                .unwrap()
                .push(duplicate_id.into());
            Ok(())
        }
    }

    /// Vision fake: returns a winner_index per duplicate_id (or a default).
    struct FakeVision {
        replies: HashMap<String, usize>,
        default_idx: usize,
        calls: Mutex<u32>,
    }
    impl MultiImageVisionLlm for FakeVision {
        fn pick_best(
            &self,
            _system_prompt: &str,
            user_prompt: &str,
            images: &[(Vec<u8>, &str)],
            _max_tokens: u32,
        ) -> Result<String> {
            *self.calls.lock().unwrap() += 1;
            // The fake doesn't know which group it is; index by call count.
            let _ = user_prompt;
            let _ = images;
            let idx = self
                .replies
                .values()
                .next()
                .copied()
                .unwrap_or(self.default_idx);
            Ok(format!(r#"{{"winner_index": {idx}, "reason": "fake"}}"#))
        }
        fn rank_images(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _items: &[(String, Vec<u8>, &str)],
            _max_tokens: u32,
        ) -> Result<String> {
            unreachable!("dedup never calls rank_images")
        }
    }

    fn args(apply: bool) -> DedupArgs {
        DedupArgs {
            apply,
            max_time_gap_secs: 600,
            max_distance_m: 500.0,
            size_ratio: 0.5,
            limit: 0,
            parallel: 1,
            verbose: true,
        }
    }

    #[test]
    fn run_with_dry_run_does_not_write() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("a", "/p/a.jpg"), 3_000_000),
                    with_size(asset("b", "/p/b.jpg"), 1_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(false), &mut log).unwrap();
        assert!(write.locations.lock().unwrap().is_empty());
        assert!(write.stacks.lock().unwrap().is_empty());
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("DRY-RUN"), "{s}");
        // Size short-circuit fired → vision NOT called.
        assert_eq!(*vision.calls.lock().unwrap(), 0);
    }

    #[test]
    fn run_with_apply_creates_stack_winner_first() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("small", "/p/a.jpg"), 1_000_000),
                    with_size(asset("big", "/p/b.jpg"), 3_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();
        let posts = write.stacks.lock().unwrap().clone();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0][0], "big", "winner must be first (cover)");
        assert_eq!(posts[0].len(), 2);
    }

    #[test]
    fn run_with_apply_backfills_gps_then_stacks() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(with_gps(asset("withgps", "/p/a.jpg"), 31.0, 121.0), 3_000_000),
                    with_size(asset("nogps", "/p/b.jpg"), 1_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();
        let locs = write.locations.lock().unwrap().clone();
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].0, "nogps");
        assert_eq!(locs[0].1, 31.0);
        assert_eq!(locs[0].2, 121.0);
    }

    #[test]
    fn run_with_skips_video_group_without_calling_vision() {
        let mut vid = asset("v", "/p/a.mp4");
        vid.asset_type = "VIDEO".into();
        let dups = FakeDuplicates {
            groups: vec![group("g1", vec![asset("a", "/p/a.jpg"), vid])],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();
        assert!(write.stacks.lock().unwrap().is_empty());
        assert_eq!(*vision.calls.lock().unwrap(), 0);
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("ContainsVideo"), "{s}");
    }

    #[test]
    fn run_with_skips_already_stacked_groups() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("x", "/p/a.jpg"), 3_000_000),
                    with_size(asset("y", "/p/b.jpg"), 1_000_000),
                ],
            )],
        };
        let stacks = FakeStacks {
            stacks: vec![Stack {
                primary_asset_id: "x".into(),
                assets: vec![StackMember { id: "x".into() }, StackMember { id: "z".into() }],
            }],
        };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();
        assert!(write.stacks.lock().unwrap().is_empty());
    }

    #[test]
    fn run_with_vision_picks_when_sizes_close() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("a", "/p/a.jpg"), 2_000_000),
                    with_size(asset("b", "/p/b.jpg"), 2_100_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 1, // winner = b
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();
        assert_eq!(*vision.calls.lock().unwrap(), 1);
        let posts = write.stacks.lock().unwrap().clone();
        assert_eq!(posts[0][0], "b");
    }

    #[test]
    fn run_with_parallel_handles_many_groups() {
        // 8 size-decidable groups, 4 workers. Verifies the parallel
        // path doesn't lose work and that the totals add up.
        let mut groups = Vec::new();
        for i in 0..8 {
            groups.push(group(
                &format!("g{i}"),
                vec![
                    with_size(asset(&format!("a{i}"), &format!("/p/a{i}.jpg")), 3_000_000),
                    with_size(asset(&format!("b{i}"), &format!("/p/b{i}.jpg")), 1_000_000),
                ],
            ));
        }
        let dups = FakeDuplicates { groups };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut a = args(true);
        a.parallel = 4;
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, a, &mut log).unwrap();
        // Size short-circuits → no vision calls.
        assert_eq!(*vision.calls.lock().unwrap(), 0);
        // Every group produced exactly one stack, winner first.
        let posts = write.stacks.lock().unwrap().clone();
        assert_eq!(posts.len(), 8);
        for ids in &posts {
            assert_eq!(ids.len(), 2);
            assert!(ids[0].starts_with('a'), "winner must be the larger 'a' asset, got {:?}", ids);
        }
    }

    #[test]
    fn run_with_apply_auto_deletes_cross_folder_dupes_and_keeps_gps() {
        // Same basename + same time + same size, two folders, one has
        // GPS — auto delete must fire, the GPS asset stays, no stack
        // is created, no vision call is made.
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("plain", "/x/IMG_001.jpg"), 2_000_000),
                    with_size(with_gps(asset("withgps", "/y/IMG_001.jpg"), 31.0, 121.0), 2_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();

        let deletes = write.deletes.lock().unwrap().clone();
        assert_eq!(deletes, vec![vec!["plain".to_string()]]);
        assert!(write.stacks.lock().unwrap().is_empty());
        assert_eq!(*vision.calls.lock().unwrap(), 0);
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("AUTO-DELETE"), "{s}");
    }

    #[test]
    fn run_with_apply_video_auto_delete_backfills_gps_then_deletes() {
        // Two videos, same name+time+folders differ, sizes differ.
        // Keeper is the larger one (b). a has GPS, b doesn't → must
        // PUT a's coords onto b BEFORE deleting a. No stack, no
        // vision call.
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(with_gps(video("a", "/x/VID.mp4"), 31.5, 121.5), 5_000_000),
                    with_size(video("b", "/y/VID.mp4"), 9_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();

        let locs = write.locations.lock().unwrap().clone();
        assert_eq!(locs, vec![("b".to_string(), 31.5, 121.5)]);
        let deletes = write.deletes.lock().unwrap().clone();
        assert_eq!(deletes, vec![vec!["a".to_string()]]);
        assert!(write.stacks.lock().unwrap().is_empty());
        assert_eq!(*vision.calls.lock().unwrap(), 0);
    }

    #[test]
    fn run_with_dry_run_does_not_delete_cross_folder_dupes() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_size(asset("a", "/x/IMG.jpg"), 2_000_000),
                    with_size(asset("b", "/y/IMG.jpg"), 2_000_000),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(false), &mut log).unwrap();
        assert!(write.deletes.lock().unwrap().is_empty());
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("DRY-RUN AUTO-DELETE"), "{s}");
    }

    #[test]
    fn run_with_apply_dismisses_duplicate_group_for_keep_all() {
        // > 6h spread → KeepAll. In --apply mode we must DELETE
        // /api/duplicates/{group_id}; no asset delete, no stack, no
        // vision call.
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_time(asset("a", "/p/a.jpg"), "2024-03-12T10:00:00"),
                    with_time(asset("b", "/p/b.jpg"), "2024-03-12T17:00:00"),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(true), &mut log).unwrap();

        let dismissed = write.dismissed_groups.lock().unwrap().clone();
        assert_eq!(dismissed, vec!["g1".to_string()]);
        assert!(write.deletes.lock().unwrap().is_empty());
        assert!(write.stacks.lock().unwrap().is_empty());
        assert_eq!(*vision.calls.lock().unwrap(), 0);
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("KEEP-ALL"), "{s}");
    }

    #[test]
    fn run_with_dry_run_does_not_dismiss_duplicate_group_for_keep_all() {
        let dups = FakeDuplicates {
            groups: vec![group(
                "g1",
                vec![
                    with_time(asset("a", "/p/a.jpg"), "2024-03-12T10:00:00"),
                    with_time(asset("b", "/p/b.jpg"), "2024-03-12T17:00:00"),
                ],
            )],
        };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut log = Vec::new();
        run_with(&dups, &stacks, &caption, &write, &vision, args(false), &mut log).unwrap();
        assert!(write.dismissed_groups.lock().unwrap().is_empty());
        let s = String::from_utf8(log).unwrap();
        assert!(s.contains("DRY-RUN KEEP-ALL"), "{s}");
    }

    #[test]
    fn run_with_parallel_rejects_zero() {
        let dups = FakeDuplicates { groups: vec![] };
        let stacks = FakeStacks { stacks: vec![] };
        let caption = FakeCaption;
        let write = FakeWrite::default();
        let vision = FakeVision {
            replies: HashMap::new(),
            default_idx: 0,
            calls: Mutex::new(0),
        };
        let mut a = args(false);
        a.parallel = 0;
        let mut log = Vec::new();
        let err = run_with(&dups, &stacks, &caption, &write, &vision, a, &mut log)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--parallel"), "got: {err}");
    }
}
