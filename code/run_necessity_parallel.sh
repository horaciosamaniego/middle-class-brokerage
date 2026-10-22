#!/usr/bin/env bash
# run_necessity_parallel.sh -- city-level parallel necessity sweep.
#
# Each city is an independent process (BFS is memory-bandwidth bound, so threading
# WITHIN a city saturates early; separate processes on separate cities is the win).
# Cities are processed longest-job-first and tiered by graph size so that a handful
# of megacities never run concurrently enough to exhaust RAM.
#
# Each city writes its OWN part-file (no concurrent writes to one CSV), and a city
# whose part-file already exists is skipped -- so the run is resumable and you can
# Ctrl-C freely. Parts are concatenated at the end.
#
#   ./run_necessity_parallel.sh <graph_dir> <cities_file> <vp> <outfile> [jobs_small] [jobs_big]
#
# e.g.
#   ./run_necessity_parallel.sh ../grafos_ciudades_dirigidos_v4/ powered_cities_25.txt \
#        quintile_nation_population necessity_population.csv 20 4
set -euo pipefail

DATA_DIR="${1:?graph dir}"
CITIES_FILE="${2:?cities file, one name per line}"
VP="${3:?vertex property, e.g. quintile_nation_population}"
OUT="${4:?final output csv}"
JOBS_SMALL="${5:-16}"      # concurrency for ordinary cities
JOBS_BIG="${6:-4}"         # concurrency for megacities (RAM-bound)
BIG_BYTES="${BIG_BYTES:-2000000000}"   # >2 GB .gt  => "big"
MODULE="${MODULE:-brokerage_constraint_pipeline}"
NEC="${NEC:-necessity_sweep.py}"

PARTS="$(dirname "$OUT")/parts_$(basename "${OUT%.csv}")"

# --- output-name safety (matches run_two_nulls_parallel.sh) -------------------
# If a DIRECTORY named like OUT exists (leftover from an older harness that used the
# .csv name as its parts dir), migrate its per-city CSVs into PARTS and free the name.
# Refuse if the directory holds anything other than parts/scratch -- never delete
# unknown user data.
if [[ -d "$OUT" ]]; then
  shopt -s nullglob dotglob
  stray=0
  for f in "$OUT"/*; do
    b="$(basename "$f")"
    case "$b" in
      *.csv|.big.txt|.small.txt|.big|.small) : ;;
      *) echo "ERROR: '$OUT' is a directory containing a non-part file ('$b'); refusing to touch it. Move it aside or pick a different output name." >&2
         stray=1 ;;
    esac
  done
  shopt -u nullglob dotglob
  (( stray )) && exit 2
  echo "NOTE: '$OUT' exists as a directory of per-city CSVs (old parts dir). Migrating into $PARTS/ and freeing the name." >&2
  mkdir -p "$PARTS"
  shopt -s nullglob
  for f in "$OUT"/*.csv; do cp -n "$f" "$PARTS/"; done
  shopt -u nullglob
  rm -rf "$OUT"
fi
if [[ -e "$OUT" && ! -f "$OUT" ]]; then
  echo "ERROR: output '$OUT' exists and is not a regular file; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "$PARTS"
export DATA_DIR VP MODULE NEC PARTS

# ---- order cities by graph size, descending (longest-job-first) ----------------
BIGLIST="$PARTS/.big.txt"; SMALLLIST="$PARTS/.small.txt"
: > "$BIGLIST"; : > "$SMALLLIST"
missing=0
while IFS= read -r city; do
  [[ -z "$city" ]] && continue
  f="$DATA_DIR/grafo_${city}.gt"
  if [[ ! -f "$f" ]]; then
    echo "WARN: no graph file for '$city' -- skipped" >&2; missing=$((missing+1)); continue
  fi
  sz=$(stat -c%s "$f")
  if (( sz > BIG_BYTES )); then printf '%s\t%s\n' "$sz" "$city" >> "$BIGLIST"
  else                          printf '%s\t%s\n' "$sz" "$city" >> "$SMALLLIST"; fi
done < "$CITIES_FILE"
sort -rn "$BIGLIST"   | cut -f2- > "$BIGLIST.sorted"
sort -rn "$SMALLLIST" | cut -f2- > "$SMALLLIST.sorted"
nbig=$(wc -l < "$BIGLIST.sorted"); nsmall=$(wc -l < "$SMALLLIST.sorted")
echo "cities: $nbig big (>$((BIG_BYTES/1000000)) MB, ${JOBS_BIG}x) + $nsmall small (${JOBS_SMALL}x); $missing missing"

# ---- worker -------------------------------------------------------------------
run_one() {
  city="$1"
  safe="${city//\//_}"
  out="$PARTS/${safe}.csv"
  if [[ -s "$out" ]]; then echo "  skip (done): $city"; return 0; fi
  echo "  start: $city"
  if python "$NEC" --data_dir "$DATA_DIR" --module "$MODULE" --vp "$VP" \
        --cities "$city" --out "$out" > "$PARTS/${safe}.log" 2>&1; then
      echo "  done : $city"
  else
      echo "  FAIL : $city (see $PARTS/${safe}.log)" >&2
      rm -f "$out"      # don't leave a partial file that would be 'resumed' as done
  fi
}
export -f run_one

# ---- big cities first, at low concurrency; then the tail at high concurrency ----
if (( nbig > 0 )); then
  echo "=== phase 1: $nbig large graphs, ${JOBS_BIG} at a time ==="
  xargs -d '\n' -I{} -P "$JOBS_BIG" bash -c 'run_one "$@"' _ {} < "$BIGLIST.sorted"
fi
if (( nsmall > 0 )); then
  echo "=== phase 2: $nsmall remaining graphs, ${JOBS_SMALL} at a time ==="
  xargs -d '\n' -I{} -P "$JOBS_SMALL" bash -c 'run_one "$@"' _ {} < "$SMALLLIST.sorted"
fi

# ---- concatenate ---------------------------------------------------------------
if [[ -e "$OUT" && ! -f "$OUT" ]]; then
  echo "ERROR: '$OUT' is not a regular file; refusing to write." >&2; exit 2
fi
first=$(ls -1 "$PARTS"/*.csv 2>/dev/null | head -1 || true)
if [[ -z "$first" ]]; then echo "no part files produced" >&2; exit 1; fi
tmp="$(mktemp "${OUT}.XXXXXX.tmp")"
trap 'rm -f "$tmp"' EXIT
head -1 "$first" > "$tmp"
tail -n +2 -q "$PARTS"/*.csv >> "$tmp"
mv -f "$tmp" "$OUT"; trap - EXIT
echo "wrote $OUT ($(($(wc -l < "$OUT")-1)) rows from $(ls -1 "$PARTS"/*.csv | wc -l) cities)"
