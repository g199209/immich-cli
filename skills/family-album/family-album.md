---
name: family-album
description: Search and inspect photos in the user's family photo library. Use whenever the user asks to find, recall, list, count, or look up family photos/videos by content ("孩子戴生日帽"), time range, place ("上海浦东", "Japan"), text visible in the image (OCR), or asset type — or asks for metadata about a specific photo in the library.
---

# family-album

Search and inspect photos in the family photo library via `immich-cli`.
The library lives on an NFS share that is mounted locally; the CLI talks
to the Immich server and prints the **local file paths** of matching
assets so you can read, show, or hand them to other tools directly.

You have exactly two subcommands available: `search` and `info`. Do not
invoke any other subcommand of `immich-cli`.

## `immich-cli search` — find photos

`search` requires at least one filter. Running it bare is rejected on
purpose, so it cannot accidentally list the whole library. Combine
filters freely.

| Flag | Meaning |
|---|---|
| `-q "..."` | Natural-language semantic search. Chinese and English both work. |
| `--taken-after YYYY-MM-DD` | Earliest local-time. ISO 8601 also accepted. |
| `--taken-before YYYY-MM-DD` | Latest local-time. |
| `--place "..."` | Free-form natural-language place. Chinese, English, mixed, abbreviations all work. Resolved by LLM against the library's actual geocoded vocabulary, so you can be vague (`"中国"` → all of China), state-level (`"上海"` → all of Shanghai), or precise (`"上海浦东"` → just Pudong). Multiple matches (e.g. a city name spanning provinces) are queried in parallel and merged. |
| `--ocr "text"` | Substring match against text Immich's OCR detected in the image. Case-sensitive, Unicode-aware. |
| `--type image\|video` | Restrict to one asset type. |
| `--limit N` | Cap results. Defaults to 24 for `-q` searches (max 48), and 1000 for filter-only searches. The `-q` cap also bounds vision-rerank input, so raising it costs extra thumbnail fetches and LLM tokens. |
| `--format paths\|json\|table` | Output. Default `paths`. |

### Picking the right query mode

- User describes what's **visible** ("穿红裙子的小女孩", "雪山", "生日蛋糕") → plain `-q`. CLIP and description candidates are deduped, then a vision model reranks the pool by looking at thumbnails plus per-candidate metadata (time, place, people, matched keywords).
- User describes a **scene, event, or feeling** that is more verbal than visual ("外婆在厨房包饺子", "搬家那天") → also `-q`, but consider adding `--description-only` if CLIP results look noisy. Descriptions tend to capture context, the vision pass plus CLIP captures appearance.
- User gives a **constraint** (date range, place, OCR text, image vs. video) → use the appropriate filter alone, or combine with `-q`.
- Never pass `-q ""` or whitespace-only — it's rejected.

### Output formats

- `paths` (default) — one local file path per line. Feed directly to other tools.
- `json` — newline-delimited JSON, one asset per line. Safe to stream into `jq`.
- `table` — aligned `TYPE / TAKEN / LOCATION / PATH`. For showing to the user; do not parse.

### Truncation

When matches exceed `--limit`, the output ends with a sentinel:
- `paths` / `table`: a final `......` line.
- `json`: a final `{"truncated":true}` line (NDJSON stays parseable).

If you see the marker, either raise `--limit` or tighten the filters.

### Recipes

```bash
# Birthday photos of children, recent two years, top 30
immich-cli search -q "孩子戴生日帽" --taken-after 2024-01-01 --limit 30

# Find a document by visible text
immich-cli search --ocr "毕业证"

# Everything taken in 上海浦东 during 2023
immich-cli search --place "上海浦东" --taken-after 2023-01-01 --taken-before 2023-12-31

# Photos from anywhere in Japan
immich-cli search --place "Japan" --limit 50 --format table

# Top 10 sunset-at-the-sea photos, formatted for the user to skim
immich-cli search -q "海边日落" --limit 10 --format table

# Structured pipeline: grab just the local paths
immich-cli search -q "猫" --format json | jq -r .localPath

# Verbal scene, skip CLIP
immich-cli search -q "外婆在厨房包饺子" --description-only --limit 20
```

## `immich-cli info <local-path>` — inspect one photo

Pass a local NFS path (one that came out of `search`, or one the user
gave you). The default text output is grep-friendly and grouped under
`File`, `Times`, `Location`, `Camera`, `People`, `Tags`, `OCR`, `Albums`.

```bash
immich-cli info ~/QNAP-Photos/Family/2018年/IMG_20180908_185429.jpg
```

For automation, use `--format json`. The body is the full Immich asset
detail plus three added top-level fields: `localPath`, `albums`, `ocr`.

```bash
immich-cli info ~/QNAP-Photos/Family/2018年/IMG_20180908_185429.jpg --format json \
  | jq '{id, localPath, gps: .exifInfo.latitude, people: [.people[].name]}'
```

OCR regions are prefixed with `[NN%] text` in text mode (you can grep by
confidence). The raw 4-corner bounding boxes and scores are preserved in
JSON.

## How to chain results

The whole point of returning local paths is that you can act on them
right away. Common patterns:

- Show one to the user: read the file directly (it's a real local image).
- Inspect metadata before deciding: pipe `search --format json` into
  `info` per id, or simply call `info` on the path.
- Build a slideshow / report: collect paths from `search`, then loop
  with `info --format json` to gather metadata.

## Pitfalls

- **`--place` is LLM-resolved.** It maps to the actual EXIF values
  Immich has stored. If nothing in the library matches, the command
  errors out — try a broader phrasing, or check whether the photos are
  geotagged at all. Don't try to be exhaustive ("中国 上海 浦东 张江");
  the resolver does better with one or two anchors.
- **OCR is substring, not semantic.** `--ocr "DELL"` only matches images
  whose detected text contains literal `DELL`. For fuzzy matches, combine
  with `-q`.
- **Date filters use `localDateTime`** (the wall-clock time at the place
  the photo was taken), not UTC. `--taken-after 2023-12-31` is correct
  for "after the end of 2023 in the user's local timezone".
- **`--limit` is the total cap across all pages.** With `-q`, the default
  is 24 and the maximum is 48 (the vision-rerank step's input scales
  with this number). Without `-q`, the default is 1000 and the CLI walks
  Immich pagination internally; you do not need to loop.
- **Some assets may be `UNMAPPED` or `MISSING`** if the path mapping is
  incomplete or the file was moved on disk. These are reported on stderr
  by default and silently dropped from stdout; pass `--include-unmapped`
  or `--include-missing` if you actually want them in the output.
- **Do not invoke any other `immich-cli` subcommand.** Only `search` and
  `info` are part of this skill.
