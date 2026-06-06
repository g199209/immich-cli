#!/usr/bin/env bash
#
# Build a (country, state, city) → admin2 lookup for the live Immich
# library, using GeoNames as the source of truth for administrative
# hierarchy. Each library is small (~hundreds of entries); the per-
# country GeoNames dumps are cached in $CACHE_DIR/geonames so re-runs
# are fast.
#
# Output: $CACHE_DIR/places_index.tsv with one line per vocab entry we
# could enrich:
#
#   country<TAB>state<TAB>city<TAB>admin2_name
#
# Vocab entries with no admin2 in GeoNames (Haitangwan-style orphans)
# are NOT written here — the CLI falls back to LLM reasoning for those.
#
# Run after major library changes. Idempotent.

set -euo pipefail

CONFIG="${1:-$HOME/.config/immich-cli/config.toml}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/immich-cli}"
GN_DIR="$CACHE_DIR/geonames"
OUT="$CACHE_DIR/places_index.tsv"
mkdir -p "$GN_DIR"

# ----- Read minimal config (no TOML parser needed) -----------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "error: config file not found: $CONFIG" >&2
  exit 1
fi
SERVER_URL=$(awk -F'=' '/^server_url[[:space:]]*=/ { gsub(/[ \t"]/, "", $2); print $2; exit }' "$CONFIG" | sed 's:/*$::')
API_KEY=$(awk -F'=' '/^api_key[[:space:]]*=/ { gsub(/^[ \t"]+|[ \t"]+$/, "", $2); print $2; exit }' "$CONFIG")
if [[ -z "$SERVER_URL" || -z "$API_KEY" ]]; then
  echo "error: could not parse server_url / api_key from $CONFIG" >&2
  exit 1
fi

# ----- Country name (GeoNames long form) → ISO 3166-1 alpha-2 -----------
country_code() {
  case "$1" in
    "France")                              echo FR ;;
    "Germany")                             echo DE ;;
    "Holy See (Vatican City State)")       echo VA ;;
    "Hong Kong")                           echo HK ;;
    "Italy")                               echo IT ;;
    "Japan")                               echo JP ;;
    "People's Republic of China")          echo CN ;;
    "Russian Federation")                  echo RU ;;
    "Switzerland")                         echo CH ;;
    "United Arab Emirates")                echo AE ;;
    "United States")                       echo US ;;
    "United Kingdom")                      echo GB ;;
    "Canada")                              echo CA ;;
    "Australia")                           echo AU ;;
    "Singapore")                           echo SG ;;
    "South Korea" | "Korea, Republic of")  echo KR ;;
    "Thailand")                            echo TH ;;
    "Vietnam" | "Viet Nam")                echo VN ;;
    "Spain")                               echo ES ;;
    "Netherlands")                         echo NL ;;
    *)                                     echo "" ;;
  esac
}

# ----- Fetch the library vocab -----------------------------------------
echo "fetching Immich vocabulary from $SERVER_URL ..." >&2
VOCAB=$(mktemp)
trap "rm -f $VOCAB" EXIT
curl -fsS -H "x-api-key: $API_KEY" "$SERVER_URL/api/search/cities" \
  | jq -r '.[] | select(.exifInfo.city != null and .exifInfo.state != null and .exifInfo.country != null) | [.exifInfo.country, .exifInfo.state, .exifInfo.city] | @tsv' \
  | sort -u > "$VOCAB"
echo "  $(wc -l < "$VOCAB") distinct (country, state, city) tuples" >&2

# ----- Download per-country dumps (cached) -----------------------------
ensure_cached() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then return; fi
  echo "  downloading $(basename "$url") ..." >&2
  curl -fsSL --max-time 180 -o "$dest" "$url"
}

ensure_cached "https://download.geonames.org/export/dump/admin1CodesASCII.txt" "$GN_DIR/admin1CodesASCII.txt"
ensure_cached "https://download.geonames.org/export/dump/admin2Codes.txt"      "$GN_DIR/admin2Codes.txt"

declare -A NEEDED_CCS=()
while IFS= read -r cn; do
  cc=$(country_code "$cn")
  if [[ -z "$cc" ]]; then
    echo "warn: no ISO code mapping for country \"$cn\" — skipping" >&2
    continue
  fi
  NEEDED_CCS["$cc"]=1
done < <(awk -F'\t' '{ print $1 }' "$VOCAB" | sort -u)

for cc in "${!NEEDED_CCS[@]}"; do
  zip="$GN_DIR/$cc.zip"
  txt="$GN_DIR/$cc.txt"
  if [[ ! -f "$txt" ]]; then
    ensure_cached "https://download.geonames.org/export/dump/$cc.zip" "$zip"
    unzip -q -o "$zip" -d "$GN_DIR"
  fi
done

# ----- Join (Python, in-memory) ----------------------------------------
# Bash + awk loops were O(N_vocab × |CN.txt|) — minutes per run on a
# 130MB CN.txt. Python loads each country file once and looks each
# vocab entry up via dict; total runtime ~few seconds.
echo "joining vocabulary against GeoNames ..." >&2

python3 - "$GN_DIR" "$VOCAB" "$OUT" <<'PY'
import os, sys
from collections import defaultdict

gn_dir, vocab_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Country name (GeoNames long form) → ISO alpha-2. Mirrors the bash
# function above; keep them in sync if you extend the list.
country_code = {
    "France": "FR", "Germany": "DE",
    "Holy See (Vatican City State)": "VA",
    "Hong Kong": "HK", "Italy": "IT", "Japan": "JP",
    "People's Republic of China": "CN",
    "Russian Federation": "RU", "Switzerland": "CH",
    "United Arab Emirates": "AE", "United States": "US",
    "United Kingdom": "GB", "Canada": "CA", "Australia": "AU",
    "Singapore": "SG", "South Korea": "KR", "Korea, Republic of": "KR",
    "Thailand": "TH", "Vietnam": "VN", "Viet Nam": "VN",
    "Spain": "ES", "Netherlands": "NL",
}

# admin1: "{cc}.{code}" → state name
admin1 = {}
with open(os.path.join(gn_dir, "admin1CodesASCII.txt"), encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            admin1[parts[0]] = parts[1]

# Reverse-direction: (cc, state_name) → admin1_code
state_to_a1 = {}
for k, v in admin1.items():
    cc, a1 = k.split(".", 1)
    state_to_a1[(cc, v)] = a1

# admin2: "{cc}.{a1}.{a2}" → admin2 name
admin2 = {}
with open(os.path.join(gn_dir, "admin2Codes.txt"), encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            admin2[parts[0]] = parts[1]

# Per-country: (admin1_code, name_or_asciiname) → (admin2_code, population)
# We keep the highest-population row when a (name, admin1) collides.
per_country = {}
def load_country(cc):
    if cc in per_country:
        return per_country[cc]
    path = os.path.join(gn_dir, f"{cc}.txt")
    if not os.path.exists(path):
        per_country[cc] = None
        return None
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 15:
                continue
            name, asciiname = cols[1], cols[2]
            a1, a2 = cols[10], cols[11]
            try:
                pop = int(cols[14] or "0")
            except ValueError:
                pop = 0
            for key_name in (name, asciiname):
                if not key_name:
                    continue
                key = (a1, key_name)
                prev = table.get(key)
                if prev is None or pop > prev[1]:
                    table[key] = (a2, pop)
    per_country[cc] = table
    return table

# Iterate the vocab. Emit a TSV row only when we resolved admin2 to
# a human-readable name.
emit = 0
total = 0
with open(vocab_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line in f:
        total += 1
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 3:
            continue
        country, state, city = cols
        cc = country_code.get(country)
        if not cc:
            continue
        table = load_country(cc)
        if table is None:
            continue
        a1 = state_to_a1.get((cc, state))
        if not a1:
            continue
        row = table.get((a1, city))
        if not row:
            continue
        a2_code = row[0]
        if not a2_code:
            continue
        a2_name = admin2.get(f"{cc}.{a1}.{a2_code}")
        if not a2_name:
            continue
        out.write(f"{country}\t{state}\t{city}\t{a2_name}\n")
        emit += 1

print(f"  enriched {emit} / {total} vocab entries with admin2", file=sys.stderr)
PY

echo "wrote $OUT" >&2
