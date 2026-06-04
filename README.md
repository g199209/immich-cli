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

`search` requires at least one filter (`--query`, `--taken-after`,
`--taken-before`, `--city`, `--state`, `--country`, or `--type`); running
it bare — or with only empty/whitespace flag values — is rejected to
avoid accidentally listing the entire library.

```bash
# Smart (CLIP) query
immich-cli search -q "child playing" --limit 20

# Time window only (YYYY-MM-DD or full ISO 8601)
immich-cli search --taken-after 2025-01-01 --taken-before 2025-12-31

# Geo location (EXIF city / state / country, exact match)
immich-cli search --country "People's Republic of China" --city Kangqiao

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
