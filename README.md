# immich-cli

A small, focused Rust CLI that talks to an Immich server and prints the
**local NFS paths** of matching assets, rather than thumbnails or downloads.

Designed for the case where the Immich library lives on an NFS share that is
also mounted on the machine running the CLI: Immich indexes the photos,
this tool finds them, and you operate on the actual files locally.

## Build

```bash
cargo build --release
# binary lands in target/release/immich-cli
```

## Configure

Defaults to `~/.config/immich-cli/config.toml` (override with `-c`):

```toml
server_url = "http://10.42.12.2:2283"
api_key    = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Map Immich server-side paths to where the same NFS share is mounted locally.
# Multiple entries are allowed; the first prefix match wins.
[[path_map]]
server = "/mnt/qnap-photos"
local  = "~/QNAP-Photos"
```

## Usage

### `search`

`search` requires at least one filter (`--query`, `--taken-after`,
`--taken-before`, `--place`, `--ocr`, or `--type`); running it bare —
or with only empty/whitespace flag values — is rejected to avoid
accidentally listing the entire library.

```bash
# Natural-language query. With [llm] configured, this collects candidates
# from Immich CLIP smart search plus LLM-expanded description keyword
# searches, then asks the LLM to rerank the combined candidate pool.
# Without [llm], it gracefully degrades to CLIP only.
immich-cli search -q "孩子戴生日帽" --limit 20

# Skip the CLIP path entirely; description-only semantic search.
# Useful once your descriptions are comprehensive enough to subsume
# what CLIP would have surfaced. Requires [llm].
immich-cli search -q "穿带兔子图案毛衣的小孩" --description-only

# Time window only (YYYY-MM-DD or full ISO 8601)
immich-cli search --taken-after 2025-01-01 --taken-before 2025-12-31

# Geo location: free-form natural language, resolved against the
# library's actual geocoded vocabulary by the LLM. Chinese, English,
# mixed, abbreviated, all fine. Requires [llm].
immich-cli search --place "上海浦东"
immich-cli search --place "Japan"
immich-cli search --place "三亚"      # prefecture-level
immich-cli search --place "文昌"      # returns photos from every Wenchang Shi city

# Substring match against OCR-detected text in the image (Unicode-aware)
immich-cli search --ocr "DELL"
immich-cli search --ocr "老年"

# Combine everything, restrict to videos, print a table
immich-cli search -q person --place "Shanghai Kangqiao" \
  --taken-after 2025-01-01 --type video \
  --format table

# Newline-delimited JSON (one asset per line) for piping into jq/xargs
immich-cli search -q "beach" --format json | jq -r .localPath

# Confirm the mapped local files actually exist; warns on stderr
immich-cli search -q "people" --verify
```

Output formats:
- `paths` (default) — one local path per line. Unmapped/missing rows are
  prefixed with `UNMAPPED\t` / `MISSING\t` when explicitly included.
- `json` — newline-delimited JSON, one object per asset.
- `table` — aligned `TYPE / TAKEN / LOCATION / PATH`.

The CLI walks Immich's pagination internally; `--limit` is the overall cap.
For `-q` searches it defaults to 36 and cannot exceed 64. For filter-only
searches it defaults to 1000 and has no CLI-enforced upper bound. Per-request
page size is hard-coded to the API maximum for filter-only searches.
When the server has more matches than `--limit` allowed through, the
output ends with a `......` marker (or `{"truncated":true}` in
`--format json`, so NDJSON stays parseable).

#### `--place` precision via GeoNames admin2

`--place` resolves through the LLM, which sees the library's geocoded
vocabulary grouped by GeoNames admin2 (prefecture / 县级市 / county /
district / ward). The CLI looks for a cached index file at
`~/.cache/immich-cli/places_index.tsv`; without one it still works, just
with a flatter vocabulary and more reliance on the LLM's own geography
knowledge.

Build / refresh the index by running:

```bash
scripts/build_places_index.sh
```

The script reads `~/.config/immich-cli/config.toml` for credentials,
downloads the relevant per-country GeoNames dumps (cached under
`~/.cache/immich-cli/geonames/`), and writes the joined TSV. Re-run when
your library acquires assets in a new country / region. With the index
present, `--place "文昌"` and `--place "三亚"` resolve via the joined
admin2 — the first returns photos from `Wenchang`, `Dongjiao`, and
`Wenjiao` simultaneously; the second falls back to the LLM identifying
`Haitangwan` as a Sanya district within the small `(uncategorized)`
bucket.

### `info`

`info` takes the **local NFS path** of a photo or video and prints
everything Immich knows about it: file info, taken/modified times, GPS
location, camera/EXIF, recognized people and faces, tags, albums, and
the Immich-internal fields (visibility, favorite, owner, checksum,
description, duplicate id, stack, etc.).

```bash
# Default: structured text, readable by humans, LLMs, and grep
immich-cli info ~/QNAP-Photos/Family/2018年/IMG_20180908_185429.jpg

# Full raw asset detail as pretty JSON, for automation
immich-cli info ~/QNAP-Photos/Family/2018年/IMG_20180908_185429.jpg --format json | \
    jq '{id, localPath, lat: .exifInfo.latitude, people: [.people[].name]}'
```

The text format groups data under `File`, `Times`, `Location`, `Camera`,
`People`, `Tags`, `OCR`, `Albums`, and `Immich` sections. Each
`Key: value` pair is indented under its section so a quick
`grep -A1 '^Location'` or similar pattern-match works for both shell
pipelines and LLMs scanning the output.

OCR text regions (when the server has OCR enabled) are listed with a
`[NN%] text` prefix so you can skim or grep by confidence. The raw
4-corner bounding boxes, both scores, and the visibility flag are
preserved in `--format json`.

The JSON format is the full `/api/assets/{id}` body plus three
top-level fields we add: `localPath` (resolved NFS path), `albums`
(membership list), and `ocr` (text regions). Nothing is dropped, so
it's safe to query for anything Immich exposes.

### `update-descriptions`

Caption photos with a vision LLM and write the result back to Immich's
`description` field. Designed to be re-run safely — already-captioned
assets are detected and skipped. The caption prompt includes objective
context from Immich (`localDateTime`, city/state/country, optional
GeoNames admin2, and configured recognized people), while instructing
the model not to infer unseen visual details from that metadata.

```bash
# One asset (good for trying out)
immich-cli update-descriptions ~/QNAP-Photos/Family/2025-12-13/foo.jpg

# Whole library, 4 workers in parallel (default)
immich-cli update-descriptions

# Subset: only 2024 photos, cap this run at 200 captions
immich-cli update-descriptions --taken-after 2024-01-01 --limit 200

# Preview what would change without calling the LLM
immich-cli update-descriptions --dry-run

# Print the per-asset prompt facts sent to the LLM (omits fixed instructions and image bytes)
immich-cli update-descriptions --verbose --dry-run
```

#### Idempotency

Each generated description gets a trailing footer line:

```
—generated by immich-cli/v2 sha=kL2A3+Icry1a
```

On a future run we parse the footer and decide per asset:

| Current description | Action |
|---|---|
| Empty | Caption |
| Footer present, sha matches asset.checksum, version matches | **Skip** |
| Footer present, sha differs | Re-caption (file changed) |
| Footer present, older version | Re-caption (prompt/model upgraded) |
| Non-empty without footer (camera/app noise like `cof`, `PixCake`, `Exif_JPEG_420`) | Caption (overwrite) |

`--force` ignores the sha check, useful when iterating on the prompt
without bumping `CURRENT_VERSION`.

#### Required permissions

The configured Immich API token must have the **`asset.update`** scope,
or every PUT comes back 403. Create a fresh token in Immich's Web UI →
User Settings → API Keys if your current one lacks it.

#### Configuration

```toml
[llm]
base_url     = "http://your-llm-gateway:port"
api_key      = "sk-..."
model        = "deepseek-v4-flash"   # text model, for `ask`
vision_model = "mimo-v2.5"           # vision model, for this command
timeout_secs = 120

[people]
"ImmichPersonSpouse" = ["妻子", "妈妈"]
"ImmichPersonSelf" = ["丈夫", "爸爸"]
"ImmichPersonChild" = ["女儿", "孩子"]
"ImmichPersonMotherInLaw" = ["岳母", "外婆"]
"ImmichPersonMother" = ["母亲", "妈妈", "奶奶"]
"ImmichPersonFather" = ["父亲", "爸爸", "爷爷"]
```

`[people]` maps Immich's face-recognition names to relationship labels
that should be searchable in generated descriptions. Only names present
in this table are sent to the captioning prompt; other detected faces are
ignored. If this table is non-empty, each captioned asset requires one
extra `GET /api/assets/{id}` call. A failure to fetch that full asset
detail is logged as an error for that asset and the batch continues with
the next one.
