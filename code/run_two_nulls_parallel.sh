#!/usr/bin/env bash
# run_two_nulls_parallel.sh -- city-level parallel two-null mechanism sweep.
#
# run_two_nulls.py already runs ONE city per invocation (--cities <name>) and writes
# a per-city part-file into --out_dir, skipping cities already done. So we drive it
# city-by-city in parallel, exactly like the necessity harness, then concatenate.
# The script has NO --out flag (it has --out_dir and --master); passing --out is
# silently ignored -- that was the bug that made the three modes overwrite each other.
#
# NAME-CLASH SAFETY: the working (parts) directory is "parts_<mode>" -- it has NO
# .csv suffix, so it can never collide with the final <mode>.csv output file. If a
# DIRECTORY named like the output file is found (a leftover from an older harness
# version, e.g. two_null_antenna.csv/ full of per-city CSVs), its parts are migrated
# into parts_<mode>/ and the directory is removed, freeing the name for the file.
# The final write is atomic (temp file + rename) and refuses any non-regular-file OUT.
#
#   ./run_two_nulls_parallel.sh <graph_dir> <cities_file> <vp> <out_csv> [jobs_small] [jobs_big]
#
# e.g. one mode:
#   ./run_two_nulls_parallel.sh ../grafos_ciudades_dirigidos_v4/ powered_antenna.txt \
#        quintile_nation_antenna two_null_antenna.csv 12 3
set -euo pipefail

DATA_DIR="${1:?graph dir}"
CITIES_FILE="${2:?cities file, one name per line}"
VP="${3:?vertex property, e.g. quintile_nation_antenna}"
OUT="${4:?final concatenated csv, e.g. two_null_antenna.csv}"
JOBS_SMALL="${5:-12}"      # ordinary cities
JOBS_BIG="${6:-3}"         # megacities (mixing null is RAM-heavy: keep low)
BIG_BYTES="${BIG_BYTES:-2000000000}"
RUNNER="${RUNNER:-run_two_nulls.py}"
N_PERM="${N_PERM:-500}"; N_MIX="${N_MIX:-300}"; N_ITER="${N_ITER:-2}"
MIN_CAPABLE="${MIN_CAPABLE:-5000}"

# --- output-name safety -------------------------------------------------------
# The recurring bug: a *directory* named like the intended output file (e.g. an
# earlier run's part-dir "two_null_antenna.csv/") makes the final `> "$OUT"` fail
# with a cryptic pandas/`head` "Is a directory" error. Guard against it here.
TAG="$(basename "${OUT%.csv}")"          # mode stem, e.g. two_null_antenna
OUTDIR="parts_${TAG}"                    # working dir has NO .csv suffix -> cannot
                                         # collide with the OUT file's name
# If OUT itself exists as a DIRECTORY, it is almost certainly a parts-dir from an
# older harness version. Salvage its per-city CSVs into our OUTDIR, then remove the
# directory so OUT is free to become a file. Never delete a directory with unknown
# non-CSV contents; only migrate .csv parts.
if [[ -d "$OUT" ]]; then
  # Refuse if the directory holds anything other than per-city CSV parts + the
  # harness's own scratch files -- we must never delete unknown user data.
  shopt -s nullglob dotglob
  stray=0
  for f in "$OUT"/*; do
    b="$(basename "$f")"
    case "$b" in
      *.csv|.master_*|.big|.small|.big.s|.small.s) : ;;   # known/expected
      *) echo "ERROR: '$OUT' is a directory containing a non-part file ('$b'); refusing to touch it. Move it aside or pick a different output name." >&2
         stray=1 ;;
    esac
  done
  shopt -u nullglob dotglob
  (( stray )) && exit 2
  echo "NOTE: '$OUT' exists as a directory of per-city CSVs (old parts dir). Migrating into $OUTDIR/ and freeing the name." >&2
  mkdir -p "$OUTDIR"
  shopt -s nullglob
  for f in "$OUT"/*.csv; do
    case "$(basename "$f")" in .master_*) continue;; esac
    cp -n "$f" "$OUTDIR/"
  done
  shopt -u nullglob
  rm -rf "$OUT"          # safe now: verified to contain only parts/scratch
fi
if [[ -e "$OUT" && ! -f "$OUT" ]]; then
  echo "ERROR: output '$OUT' exists and is not a regular file; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "$OUTDIR"
export DATA_DIR VP RUNNER OUTDIR N_PERM N_MIX N_ITER MIN_CAPABLE

# ---- tier cities by graph size (longest-job-first) ----------------------------
BIGL="$OUTDIR/.big"; SMALLL="$OUTDIR/.small"; : > "$BIGL"; : > "$SMALLL"; miss=0
while IFS= read -r city; do
  [[ -z "$city" ]] && continue
  f="$DATA_DIR/grafo_${city}.gt"
  if [[ ! -f "$f" ]]; then echo "WARN: no graph for '$city'" >&2; miss=$((miss+1)); continue; fi
  sz=$(stat -c%s "$f")
  if (( sz > BIG_BYTES )); then printf '%s\t%s\n' "$sz" "$city" >> "$BIGL"
  else                          printf '%s\t%s\n' "$sz" "$city" >> "$SMALLL"; fi
done < "$CITIES_FILE"
sort -rn "$BIGL"   | cut -f2- > "$BIGL.s"
sort -rn "$SMALLL" | cut -f2- > "$SMALLL.s"
nb=$(wc -l < "$BIGL.s"); ns=$(wc -l < "$SMALLL.s")
echo "mode=$TAG vp=$VP : $nb big (${JOBS_BIG}x) + $ns small (${JOBS_SMALL}x); $miss missing"

# ---- worker: one city, its own part-file, resumable ---------------------------
run_one() {
  city="$1"; safe="${city//\//_}"
  part="$OUTDIR/${safe}.csv"
  if [[ -s "$part" ]]; then echo "  skip(done): $city"; return 0; fi
  # run_two_nulls.py writes the part-file itself into --out_dir; we give it a private
  # --master so its end-of-run concat can't clobber another mode. One city => --cities.
  if python "$RUNNER" --data_dir "$DATA_DIR" --vp "$VP" --cities "$city" \
        --out_dir "$OUTDIR" --master "$OUTDIR/.master_${safe}.csv" \
        --n_perm "$N_PERM" --n_mix "$N_MIX" --n_iter "$N_ITER" \
        --min_capable "$MIN_CAPABLE" > "$OUTDIR/${safe}.log" 2>&1; then
      if [[ -s "$part" ]]; then echo "  done: $city"; else
        echo "  WARN: $city produced no part-file (see log)" >&2; fi
  else
      echo "  FAIL: $city (see $OUTDIR/${safe}.log)" >&2; rm -f "$part"
  fi
  rm -f "$OUTDIR/.master_${safe}.csv"
}
export -f run_one

(( nb > 0 )) && { echo "=== phase 1: $nb megacities, ${JOBS_BIG}x ==="; \
  xargs -d '\n' -I{} -P "$JOBS_BIG" bash -c 'run_one "$@"' _ {} < "$BIGL.s"; }
(( ns > 0 )) && { echo "=== phase 2: $ns cities, ${JOBS_SMALL}x ==="; \
  xargs -d '\n' -I{} -P "$JOBS_SMALL" bash -c 'run_one "$@"' _ {} < "$SMALLL.s"; }

# ---- concatenate all per-city parts into the final summary --------------------
if [[ -e "$OUT" && ! -f "$OUT" ]]; then
  echo "ERROR: '$OUT' is not a regular file; refusing to write." >&2; exit 2
fi
first=$(ls -1 "$OUTDIR"/*.csv 2>/dev/null | grep -v '\.master_\|\.big\|\.small' | head -1 || true)
if [[ -z "$first" ]]; then echo "no part files in $OUTDIR" >&2; exit 1; fi
tmp="$(mktemp "${OUT}.XXXXXX.tmp")"      # write to temp, then atomic rename
trap 'rm -f "$tmp"' EXIT
head -1 "$first" > "$tmp"
for f in "$OUTDIR"/*.csv; do
  case "$f" in *.master_*|*.big|*.small) continue;; esac
  tail -n +2 "$f" >> "$tmp"
done
mv -f "$tmp" "$OUT"; trap - EXIT
echo "wrote $OUT ($(($(wc -l < "$OUT")-1)) rows from $(ls -1 "$OUTDIR"/*.csv | grep -vc '\.master_\|\.big\|\.small') cities)"
