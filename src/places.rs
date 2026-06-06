//! Natural-language place resolution.
//!
//! The CLI's `--place "..."` flag accepts free-form input — Chinese,
//! English, mixed, abbreviated, mis-cased. Immich, however, only filters
//! by exact-match `city` / `state` / `country` strings drawn from
//! whatever Mapbox/Photon used at ingest (e.g. `People's Republic of
//! China`, never `China` or `中国`).
//!
//! Bridge: pull the library's full geocoded vocabulary once
//! (`GET /api/search/cities` returns one asset per distinct (city, state,
//! country) tuple), hand it to an LLM as an indented outline, and ask it
//! to map the user's input to one or more concrete entries. The
//! resulting [`PlaceMatch`]es are passed back to the search command,
//! which issues one Immich request per match and RRF-merges them.

use crate::client::PlacesBackend;
use crate::llm::{ChatBackend, Message};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// A resolved geo filter ready to be applied to one Immich search. Any
/// of the three fields may be `None`, mirroring Immich's optional
/// `city`/`state`/`country` filter shape — broader inputs map to fewer
/// fields (e.g. `"中国"` → only `country` set).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaceMatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
}

impl PlaceMatch {
    pub fn is_empty(&self) -> bool {
        self.country.is_none() && self.state.is_none() && self.city.is_none()
    }

    /// Convert any whitespace-only string field to `None`. Some LLMs
    /// emit `"city": ""` instead of omitting the key — Immich's filter
    /// would then exact-match on the empty string and return zero hits,
    /// which is the opposite of the broader intent the LLM was trying
    /// (clumsily) to express.
    fn normalize_empties(&mut self) {
        for field in [&mut self.country, &mut self.state, &mut self.city] {
            if field.as_deref().is_some_and(|s| s.trim().is_empty()) {
                *field = None;
            }
        }
    }
}

/// One distinct (city, state, country) tuple drawn from the library,
/// optionally enriched with the GeoNames admin2 (prefecture / 县级市 /
/// county / district) it belongs to. `admin2` is `None` for entries
/// whose GeoNames row had no admin2 code — these "orphan" cities still
/// get put in front of the LLM, just in an `(uncategorized)` bucket so
/// the model has to fall back on its own geographic knowledge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CityVocabEntry {
    pub country: String,
    pub state: String,
    pub city: String,
    pub admin2: Option<String>,
}

/// Sentinel rendered for orphan cities in the LLM prompt. The LLM
/// understands `(uncategorized)` as "the GeoNames join couldn't place
/// these — figure it out from your own knowledge of geography."
const ORPHAN_BUCKET: &str = "(uncategorized)";

/// `(country, state, city)` → admin2 lookup. Built by
/// `scripts/build_places_index.sh` and loaded from
/// `~/.cache/immich-cli/places_index.tsv`. A missing file is fine —
/// `admin2` just stays `None` for every entry and the LLM operates on a
/// 3-level vocabulary like before.
pub type Admin2Lookup = HashMap<(String, String, String), String>;

/// Default path: `<XDG_CACHE_HOME>/immich-cli/places_index.tsv`.
pub fn default_admin2_lookup_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "immich-cli")
        .map(|d| d.cache_dir().join("places_index.tsv"))
}

/// Load the lookup from `path`. If the file doesn't exist, return an
/// empty map (the LLM-only path is the documented fallback). Malformed
/// rows are skipped with a warning but never abort the load.
pub fn load_admin2_lookup(path: &Path) -> Result<Admin2Lookup> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Admin2Lookup::new()),
        Err(e) => {
            return Err(e).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let mut out = Admin2Lookup::with_capacity(text.lines().count());
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut it = line.splitn(4, '\t');
        let country = it.next();
        let state = it.next();
        let city = it.next();
        let admin2 = it.next();
        match (country, state, city, admin2) {
            (Some(c), Some(s), Some(ci), Some(a2))
                if !c.is_empty() && !s.is_empty() && !ci.is_empty() && !a2.is_empty() =>
            {
                out.insert((c.to_string(), s.to_string(), ci.to_string()), a2.to_string());
            }
            _ => {
                eprintln!(
                    "warn: places_index.tsv line {}: skipping malformed row",
                    i + 1
                );
            }
        }
    }
    Ok(out)
}

/// Augment each entry's `admin2` field via the lookup table. Entries
/// missing from the lookup keep `admin2 = None` and become orphans.
pub fn enrich_with_admin2(vocab: &mut [CityVocabEntry], lookup: &Admin2Lookup) {
    for entry in vocab {
        let key = (entry.country.clone(), entry.state.clone(), entry.city.clone());
        if let Some(a2) = lookup.get(&key) {
            entry.admin2 = Some(a2.clone());
        }
    }
}

/// Format the vocabulary as an indented outline grouped by admin2
/// (prefecture / 县级市 / county / district):
///
/// ```text
/// People's Republic of China
///   Hainan
///     Haikou Shi: Haikou
///     Qionghai County: Bo'ao
///     Wenchang Shi: Dongjiao, Wenchang, Wenjiao
///     (uncategorized): Haitangwan
///   Shanghai
///     Pudong Xinqu: Pudong, ...
/// Japan
///   Tokyo
///     Chūō Ku: Chūō, Hatchōbori, Higashi-nihombashi, Nihonbashi-Kayabachō
///     Minato-ku: Azabu-jūban, Minato City
/// ```
///
/// Cities whose GeoNames row had no admin2 (orphans, ~5% in practice)
/// land in `(uncategorized)` so the LLM still sees them but knows it
/// needs its own geographic knowledge to place them.
///
/// Sorted alphabetically at every level. Cities deduped per admin2.
pub fn format_vocabulary(entries: &[CityVocabEntry]) -> String {
    type CitiesByAdmin2<'a> = BTreeMap<&'a str, Vec<&'a str>>;
    type StatesTree<'a> = BTreeMap<&'a str, CitiesByAdmin2<'a>>;
    let mut tree: BTreeMap<&str, StatesTree> = BTreeMap::new();
    for e in entries {
        let admin2 = e.admin2.as_deref().unwrap_or(ORPHAN_BUCKET);
        tree.entry(e.country.as_str())
            .or_default()
            .entry(e.state.as_str())
            .or_default()
            .entry(admin2)
            .or_default()
            .push(e.city.as_str());
    }
    let mut out = String::new();
    for (country, states) in &tree {
        out.push_str(country);
        out.push('\n');
        for (state, admin2s) in states {
            out.push_str("  ");
            out.push_str(state);
            out.push('\n');
            for (admin2, cities) in admin2s {
                let mut cs: Vec<&&str> = cities.iter().collect();
                cs.sort();
                cs.dedup();
                out.push_str("    ");
                out.push_str(admin2);
                out.push_str(": ");
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(c);
                }
                out.push('\n');
            }
        }
    }
    out
}

const SYSTEM_PROMPT_TEMPLATE: &str = "You map a user's free-form description of a place to the \
exact (city, state, country) tuples a photo library uses for filtering.\n\n\
Two halves to keep straight:\n\
1. The USER INPUT is free-form: Chinese, English, abbreviated, names of countries, provinces, \
districts, prefectures, neighborhoods. You INTERPRET it using your geographic knowledge.\n\
2. The OUTPUT strings (country, state, city) must come from the vocabulary below byte-for-byte. \
The library filters by exact match; an invented string returns nothing.\n\n\
What the vocabulary actually is — this is critical context:\n\
The library uses GeoNames as its reverse-geocoder. Each photo's GPS coords are mapped to \
the nearest GeoNames populated-place entry; THAT entry's name fills the `city` field, and \
a preprocessing step has joined each entry with its GeoNames admin2 (prefecture / 县级市 / \
county / district / ward).\n\
- `country` is GeoNames' long-form country name. Always use the long form as it appears in \
the vocabulary: `People's Republic of China` (NEVER `China`), `Russian Federation`, \
`United Arab Emirates`, etc.\n\
- `state` is GeoNames' admin1 division: province / 省 / direct-administered municipality \
in mainland China, prefecture in Japan, state in the US, region in Italy, federal subject \
in Russia.\n\
- The **admin2 group** (indented inside each state in the vocabulary, with `:` after its \
name) is the prefecture-level subdivision: 县级市 / prefecture-city in China (`三亚市`, \
`文昌市`, `Haikou Shi`), special ward in Tokyo (`Minato-ku`, `Chūō Ku`), county / county \
equivalent in the US, département in France, etc. This is the level at which most user \
queries naturally land. It is shown to you for free — you do NOT need to recall which \
populated places belong to which admin2; the vocabulary already tells you.\n\
- `city` is the noisy GeoNames populated-place entry: a neighbourhood (`Quinze-Vingts`), a \
district / township (`Haitangwan`), a sub-district / street office (`Báigōng Jiēdào`), a \
science park (`Keji yuan`), a park / landmark (`Happy Harbour`), or anything else a \
contributor put on the map.\n\
- The bucket labelled `(uncategorized)` inside a state collects cities whose admin2 the \
preprocessing step could NOT identify. For those, you DO have to fall back on your own \
geographic knowledge — but only those, not the whole vocabulary.\n\n\
Important: the admin2 names are themselves vocabulary metadata, NOT output values. They \
help you decide WHICH cities to return. The matches you emit only use country / state / \
city.\n\n\
Your job: the user is asking about a real-world region; figure out which vocabulary \
entries lie inside it. With admin2 grouping shown above, this is usually a straightforward \
lookup: if the user names an admin2, every city in that bucket is in scope.\n\n\
Vocabulary (country → state → admin2 → cities; cities are comma-separated after the \
admin2 name; `(uncategorized)` holds cities with no joined admin2):\n\n\
{VOCAB}\n\n\
Output strict JSON, no prose:\n\
{\"matches\": [{\"country\": \"...\", \"state\": \"...\", \"city\": \"...\"}, ...]}\n\n\
CORE PRINCIPLE — match the user's precision, not coarser, not finer.\n\
The level of detail in the output (country only / country+state / country+state+city) must \
match the level the user actually targeted. NEVER \"fall back to a broader level\" if a finer \
match isn't possible — broadening returns photos from places the user did NOT ask about. If \
you cannot identify the user's place at the precision they specified, return \
{\"matches\": []}. Empty is the correct answer when there is no overlap; do not substitute a \
neighbouring or enclosing region.\n\n\
How to pick the level:\n\
- User names a COUNTRY (`日本`, `Japan`, `中国`, `US`) → one match, only `country` set. Do NOT \
list every state.\n\
- User names a STATE / PROVINCE / direct-administered municipality that exists as a \
vocabulary state (`海南`, `Hainan`, `上海`, `Tokyo`) → one match with `country` + `state`. Do \
NOT enumerate its admin2s or cities.\n\
- User names an ADMIN2 — a prefecture-level city / county / ward (`三亚市`, `文昌市`, \
`Minato-ku`, `Haidian Qu`). Find that admin2 in the vocabulary and emit ONE match per city \
under it. This is the most common case; the vocabulary already lists them for you, no \
guessing needed.\n\
- User names a VOCABULARY CITY directly (`Pudong`, `Haitangwan`) → one match with all three \
fields.\n\
- User input doesn't match any admin2 verbatim but is a sub-state region you can identify \
(an alias / Chinese name / informal name like `三亚` for `三亚市`, `海淀` for `Haidian Qu`, \
`Shibuya` for `Shibuya-ku`). Same as the admin2 case — return every city under the matched \
admin2.\n\
- User names a sub-state region with NO matching admin2 in the vocabulary — only then fall \
back on your geographic knowledge of which `(uncategorized)` cities lie inside it. Do NOT \
enumerate cities from OTHER admin2s of the same state; that would broaden the result.\n\
- If after all the above you cannot identify ANY vocabulary city as inside the user's \
region, return {\"matches\": []}. Empty is the correct answer when there is no overlap.\n\n\
Return MULTIPLE matches when (a) the user's sub-state region contains multiple vocabulary \
cities (the common case for any prefecture / district / county / metro area), or (b) the \
same city name appears under multiple states.\n\n\
Generic level-of-detail examples (substitute the actual country/state/city names from the \
vocabulary):\n\
- Whole country → {\"country\": \"<long-form country name from the vocabulary>\"}.\n\
- A vocabulary state → {\"country\": \"...\", \"state\": \"<state name>\"}. Do NOT enumerate \
its cities.\n\
- A vocabulary city → {\"country\": \"...\", \"state\": \"...\", \"city\": \"<city name>\"}.\n\
- A sub-state region (district / prefecture / county / neighbourhood) → one match per \
vocabulary city you can identify as lying inside that region, possibly several.\n\
- A region that has no overlap with the vocabulary → {\"matches\": []}.\n\n\
FORMAT: omit `state` and `city` entirely when they are not used — do NOT include them as empty \
strings. `{\"country\": \"...\", \"state\": \"Hainan\"}` is correct; \
`{\"country\": \"...\", \"state\": \"Hainan\", \"city\": \"\"}` is WRONG (the library would \
then exact-match on `city = \"\"` and return zero photos).\n";

/// Resolve a free-form `--place` input to one or more [`PlaceMatch`]es.
///
/// The vocabulary is fetched fresh on every call; callers that resolve
/// multiple places in one CLI invocation should cache it themselves.
/// `lookup` is the admin2 enrichment table; pass an empty map to skip
/// enrichment entirely. When `verbose` is true, a step-by-step trace
/// (vocabulary size, full prompt, raw LLM reply, parsed matches) goes
/// to stderr.
pub fn resolve_place<P, L>(
    places_be: &P,
    llm: &L,
    input: &str,
    lookup: &Admin2Lookup,
    verbose: bool,
) -> Result<Vec<PlaceMatch>>
where
    P: PlacesBackend,
    L: ChatBackend,
{
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("--place is empty");
    }
    let mut vocab = places_be.cities_vocabulary()?;
    enrich_with_admin2(&mut vocab, lookup);
    if verbose {
        let with_a2 = vocab.iter().filter(|e| e.admin2.is_some()).count();
        eprintln!(
            "[verbose] places: vocabulary has {} distinct (city, state, country) tuples ({} \
             enriched with admin2, {} orphans)",
            vocab.len(),
            with_a2,
            vocab.len() - with_a2
        );
    }
    resolve_with_vocab(llm, trimmed, &vocab, verbose)
}

/// Same as [`resolve_place`], but the vocabulary is supplied directly.
/// Caller is responsible for applying `enrich_with_admin2` first if
/// desired.
pub fn resolve_with_vocab<L: ChatBackend>(
    llm: &L,
    input: &str,
    vocab: &[CityVocabEntry],
    verbose: bool,
) -> Result<Vec<PlaceMatch>> {
    if vocab.is_empty() {
        bail!("the library has no geocoded assets; --place cannot be resolved");
    }
    let prompt = SYSTEM_PROMPT_TEMPLATE.replace("{VOCAB}", &format_vocabulary(vocab));
    if verbose {
        eprintln!(
            "[verbose] places: user input = {input:?}, system prompt = {} chars",
            prompt.len()
        );
        eprintln!("[verbose] places: ---- system prompt begin ----");
        eprintln!("{prompt}");
        eprintln!("[verbose] places: ---- system prompt end ----");
    }
    let reply = llm.chat_json(&[Message::system(prompt), Message::user(input.to_string())])?;
    if verbose {
        eprintln!("[verbose] places: raw LLM reply = {reply}");
    }

    #[derive(Deserialize)]
    struct Reply {
        matches: Vec<PlaceMatch>,
    }
    let parsed: Reply = serde_json::from_str(&reply)
        .with_context(|| format!("LLM returned non-JSON for place resolution: {reply}"))?;

    let mut out: Vec<PlaceMatch> = Vec::with_capacity(parsed.matches.len());
    for mut m in parsed.matches {
        m.normalize_empties();
        if m.is_empty() {
            continue;
        }
        if !out.contains(&m) {
            out.push(m);
        }
    }
    if verbose {
        eprintln!("[verbose] places: parsed {} match(es):", out.len());
        for m in &out {
            eprintln!(
                "[verbose]   country={:?} state={:?} city={:?}",
                m.country, m.state, m.city
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use std::cell::RefCell;

    fn entry(country: &str, state: &str, city: &str) -> CityVocabEntry {
        CityVocabEntry {
            country: country.into(),
            state: state.into(),
            city: city.into(),
            admin2: None,
        }
    }

    fn sample_vocab() -> Vec<CityVocabEntry> {
        vec![
            entry("People's Republic of China", "Shanghai", "Pudong"),
            entry("People's Republic of China", "Shanghai", "Anting"),
            entry("People's Republic of China", "Shanghai", "Baoshan"),
            entry("People's Republic of China", "Beijing", "Baizhifang"),
            entry("People's Republic of China", "Zhejiang", "Andong"),
            entry("Japan", "Tokyo", "Azabu-jūban"),
            entry("Japan", "Shizuoka", "Arai"),
        ]
    }

    // ---- format_vocabulary -------------------------------------------------

    #[test]
    fn format_vocabulary_groups_orphans_under_uncategorized() {
        // sample_vocab() has no admin2 set, so every city lands in the
        // (uncategorized) bucket of its state.
        let formatted = format_vocabulary(&sample_vocab());
        assert_eq!(
            formatted,
            "Japan\n  \
               Shizuoka\n    \
                 (uncategorized): Arai\n  \
               Tokyo\n    \
                 (uncategorized): Azabu-jūban\n\
             People's Republic of China\n  \
               Beijing\n    \
                 (uncategorized): Baizhifang\n  \
               Shanghai\n    \
                 (uncategorized): Anting, Baoshan, Pudong\n  \
               Zhejiang\n    \
                 (uncategorized): Andong\n"
        );
    }

    #[test]
    fn format_vocabulary_uses_admin2_when_set() {
        let mut v = vec![
            entry("PRC", "Hainan", "Wenchang"),
            entry("PRC", "Hainan", "Dongjiao"),
            entry("PRC", "Hainan", "Wenjiao"),
            entry("PRC", "Hainan", "Haitangwan"), // orphan
        ];
        for e in v.iter_mut() {
            if matches!(e.city.as_str(), "Wenchang" | "Dongjiao" | "Wenjiao") {
                e.admin2 = Some("Wenchang Shi".into());
            }
        }
        let formatted = format_vocabulary(&v);
        assert_eq!(
            formatted,
            "PRC\n  \
               Hainan\n    \
                 (uncategorized): Haitangwan\n    \
                 Wenchang Shi: Dongjiao, Wenchang, Wenjiao\n"
        );
    }

    #[test]
    fn format_vocabulary_dedupes_repeated_city_in_same_state() {
        let v = vec![
            entry("A", "B", "C"),
            entry("A", "B", "C"),
            entry("A", "B", "D"),
        ];
        assert_eq!(format_vocabulary(&v), "A\n  B\n    (uncategorized): C, D\n");
    }

    #[test]
    fn format_vocabulary_handles_empty() {
        assert_eq!(format_vocabulary(&[]), "");
    }

    // ---- admin2 lookup load + enrich ---------------------------------------

    #[test]
    fn load_admin2_lookup_missing_file_is_empty_ok() {
        let got = load_admin2_lookup(Path::new("/no/such/file/places_index.tsv")).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn load_admin2_lookup_parses_tsv() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            "People's Republic of China\tHainan\tWenchang\tWenchang Shi\n\
             People's Republic of China\tHainan\tDongjiao\tWenchang Shi\n\
             Japan\tTokyo\tAzabu-jūban\tMinato-ku\n",
        )
        .unwrap();
        let lookup = load_admin2_lookup(tmp.path()).unwrap();
        assert_eq!(lookup.len(), 3);
        assert_eq!(
            lookup
                .get(&(
                    "People's Republic of China".to_string(),
                    "Hainan".to_string(),
                    "Wenchang".to_string(),
                ))
                .map(String::as_str),
            Some("Wenchang Shi")
        );
        assert_eq!(
            lookup
                .get(&(
                    "Japan".to_string(),
                    "Tokyo".to_string(),
                    "Azabu-jūban".to_string(),
                ))
                .map(String::as_str),
            Some("Minato-ku")
        );
    }

    #[test]
    fn enrich_with_admin2_fills_only_matches() {
        let mut v = vec![
            entry("PRC", "Hainan", "Wenchang"),
            entry("PRC", "Hainan", "Haitangwan"),
        ];
        let mut lookup = Admin2Lookup::new();
        lookup.insert(
            ("PRC".into(), "Hainan".into(), "Wenchang".into()),
            "Wenchang Shi".into(),
        );
        enrich_with_admin2(&mut v, &lookup);
        assert_eq!(v[0].admin2.as_deref(), Some("Wenchang Shi"));
        assert!(v[1].admin2.is_none(), "orphans stay un-enriched");
    }

    // ---- resolve_place: LLM fake -------------------------------------------

    /// A ChatBackend that captures the prompts and replays canned JSON.
    struct FakeLlm {
        replies: RefCell<Vec<String>>,
        prompts: RefCell<Vec<Vec<Message>>>,
    }
    impl FakeLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: RefCell::new(replies.iter().map(|s| s.to_string()).collect()),
                prompts: RefCell::new(vec![]),
            }
        }
    }
    impl ChatBackend for FakeLlm {
        fn chat_json(&self, messages: &[Message]) -> Result<String> {
            self.prompts.borrow_mut().push(messages.to_vec());
            Ok(self.replies.borrow_mut().remove(0))
        }
    }

    /// A fake PlacesBackend yielding a fixed vocabulary.
    struct FakePlaces(Vec<CityVocabEntry>);
    impl PlacesBackend for FakePlaces {
        fn cities_vocabulary(&self) -> Result<Vec<CityVocabEntry>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn resolve_place_returns_parsed_matches() {
        let llm = FakeLlm::new(&[r#"{"matches":[{"country":"People's Republic of China","state":"Shanghai","city":"Pudong"}]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "上海浦东", &Admin2Lookup::new(), false).unwrap();
        assert_eq!(
            got,
            vec![PlaceMatch {
                country: Some("People's Republic of China".into()),
                state: Some("Shanghai".into()),
                city: Some("Pudong".into()),
            }]
        );
    }

    #[test]
    fn resolve_place_handles_country_only() {
        let llm =
            FakeLlm::new(&[r#"{"matches":[{"country":"People's Republic of China"}]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "中国", &Admin2Lookup::new(), false).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].country.as_deref(),
            Some("People's Republic of China")
        );
        assert!(got[0].state.is_none());
        assert!(got[0].city.is_none());
    }

    #[test]
    fn resolve_place_returns_multiple_matches() {
        let llm = FakeLlm::new(&[
            r#"{"matches":[
                {"country":"People's Republic of China","state":"Shanghai","city":"Pudong"},
                {"country":"People's Republic of China","state":"Zhejiang","city":"Andong"}
            ]}"#,
        ]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "ambiguous", &Admin2Lookup::new(), false).unwrap();
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn resolve_place_drops_all_none_matches() {
        // The LLM might return a row with all three fields missing
        // ("matches": [{}]). Treat that as empty (= no filter), and we
        // explicitly do not want to issue an unfiltered Immich query, so
        // drop the entry.
        let llm = FakeLlm::new(&[r#"{"matches":[{}]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "nonsense", &Admin2Lookup::new(), false).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn resolve_place_normalizes_empty_string_fields_to_none() {
        // LLM (clumsily) returns `"city": ""` instead of omitting the key.
        // We must treat that as `None` — otherwise Immich would do an
        // exact-match on the empty string and return zero photos.
        let llm = FakeLlm::new(&[r#"{"matches":[{
            "country": "People's Republic of China",
            "state":   "Hainan",
            "city":    ""
        }]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "海南", &Admin2Lookup::new(), false).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].country.as_deref(),
            Some("People's Republic of China")
        );
        assert_eq!(got[0].state.as_deref(), Some("Hainan"));
        assert!(
            got[0].city.is_none(),
            "empty-string `city` must be normalized to None"
        );
    }

    #[test]
    fn resolve_place_dedupes_repeated_matches() {
        let llm = FakeLlm::new(&[r#"{"matches":[
            {"country":"Japan","state":"Tokyo","city":"Azabu-jūban"},
            {"country":"Japan","state":"Tokyo","city":"Azabu-jūban"}
        ]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "Azabu", &Admin2Lookup::new(), false).unwrap();
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn resolve_place_empty_matches_is_ok() {
        // LLM gracefully says "no idea". Caller decides whether to bail.
        let llm = FakeLlm::new(&[r#"{"matches":[]}"#]);
        let places = FakePlaces(sample_vocab());
        let got = resolve_place(&places, &llm, "Antarctica", &Admin2Lookup::new(), false).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn resolve_place_rejects_empty_input() {
        let llm = FakeLlm::new(&[]);
        let places = FakePlaces(sample_vocab());
        let err = resolve_place(&places, &llm, "   ", &Admin2Lookup::new(), false).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn resolve_place_errors_on_empty_vocabulary() {
        let llm = FakeLlm::new(&[]);
        let places = FakePlaces(vec![]);
        let err = resolve_place(&places, &llm, "anywhere", &Admin2Lookup::new(), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no geocoded"), "got: {err}");
    }

    #[test]
    fn resolve_place_errors_on_malformed_llm_reply() {
        let llm = FakeLlm::new(&[r#"this is not JSON"#]);
        let places = FakePlaces(sample_vocab());
        let err = resolve_place(&places, &llm, "上海", &Admin2Lookup::new(), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-JSON"), "got: {err}");
    }

    #[test]
    fn resolve_place_prompt_includes_full_vocabulary() {
        let llm = FakeLlm::new(&[r#"{"matches":[]}"#]);
        let places = FakePlaces(sample_vocab());
        resolve_place(&places, &llm, "上海", &Admin2Lookup::new(), false).unwrap();
        let prompts = llm.prompts.borrow();
        assert_eq!(prompts.len(), 1);
        let system = &prompts[0][0];
        assert_eq!(system.role, "system");
        // Every country, state, and city name from sample_vocab() appears.
        for needle in [
            "People's Republic of China",
            "Japan",
            // With no admin2 enrichment, all cities under each state
            // land in the `(uncategorized)` bucket.
            "Shanghai\n    (uncategorized): Anting, Baoshan, Pudong",
            "Azabu-jūban",
            "Output strict JSON",
        ] {
            assert!(
                system.content.contains(needle),
                "system prompt missing `{needle}`:\n{}",
                system.content
            );
        }
        // The user prompt is the verbatim input.
        let user = &prompts[0][1];
        assert_eq!(user.role, "user");
        assert_eq!(user.content, "上海");
    }
}
