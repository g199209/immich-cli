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
`--taken-before`, `--city`, `--state`, `--country`, `--ocr`, or
`--type`); running it bare — or with only empty/whitespace flag
values — is rejected to avoid accidentally listing the entire library.

```bash
# Smart (CLIP) query
immich-cli search -q "child playing" --limit 20

# Time window only (YYYY-MM-DD or full ISO 8601)
immich-cli search --taken-after 2025-01-01 --taken-before 2025-12-31

# Geo location (EXIF city / state / country, exact match)
immich-cli search --country "People's Republic of China" --city Kangqiao

# Substring match against OCR-detected text in the image (Unicode-aware)
immich-cli search --ocr "DELL"
immich-cli search --ocr "老年"

# Combine everything, restrict to videos, print a table
immich-cli search -q person --city Kangqiao \
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

The CLI walks Immich's pagination internally; `--limit` is the overall cap
(default 1000). Per-request page size is hard-coded to the API maximum.
When the server has more matches than `--limit` allowed through, the
output ends with a `......` marker (or `{"truncated":true}` in
`--format json`, so NDJSON stays parseable).

### `info`

`info` takes the **local NFS path** of a photo or video and prints
everything Immich knows about it: file info, taken/modified times, GPS
location, camera/EXIF, recognized people and faces, tags, albums, and
the Immich-internal fields (visibility, favorite, owner, checksum,
description, duplicate id, stack, etc.).

```bash
# Default: structured text, readable by humans, LLMs, and grep
immich-cli info ~/QNAP-Photos/PYL/2018年/IMG_20180908_185429.jpg

# Full raw asset detail as pretty JSON, for automation
immich-cli info ~/QNAP-Photos/PYL/2018年/IMG_20180908_185429.jpg --format json | \
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

### `ask`

`ask` does **natural-language semantic search over photo descriptions**,
mediated by an LLM. The CLI itself stays stateless — there is no local
vector store. Each query runs three stages:

1. **Keyword expansion**: the LLM extracts up to 16 substring keywords
   (including synonyms and lexical variants) from the natural-language
   query.
2. **Substring search**: each keyword is fanned out against Immich's
   metadata `description` filter; the union (deduped by asset id) forms
   the candidate set (capped at 100).
3. **Rerank**: the LLM is shown the query and candidate descriptions and
   returns the relevant ids in order.

```bash
# Add an [llm] section to config.toml first (see config.example.toml).
immich-cli ask "我想看看非洲草原上的一大群大象聚集在一起的照片"
immich-cli ask "sunset over the ocean with sailing boats" --format table
```

Requires the `[llm]` section in `config.toml`. Without it `ask` errors
out — there is intentionally no fallback so the user knows the feature
is unavailable.

`ask` only matches against the `exifInfo.description` field. If your
assets have no descriptions yet, every query will report "no description
matched any keyword" — populate descriptions via your own pipeline first
(e.g., a vision-language-model captioning step writing back to the file
EXIF or sidecar). Immich's external libraries are read-only via the API
(PUT to `/api/assets/{id}` returns 403 for description), so descriptions
have to be written to disk and re-indexed.
