#!/usr/bin/env bash
# run_reciprocal.sh -- re-run the pole census and the ladder census on the FULL call
# graphs under the reciprocity-preserving mixing null (mixing_null.py,
# reciprocity_preserving_null): mutual ties are rewired as undirected edges and
# one-way calls as directed arcs, each with the strata as blocks, and a->b->a
# return paths are removed from the ladder. Position and the removal test do not
# use the mixing null and are not re-run.
#
# Run from an empty working folder, with the graph-tool environment active:
#
#   conda activate gtbb
#   mkdir -p rp_run && cd rp_run
#   GRAPHS=... LISTS=... bash /path/to/code/run_reciprocal.sh check
#   bash /path/to/code/run_reciprocal.sh all
#
# Steps (each resumable, in this order):
#   check    scripts, inputs, machine
#   pole     run_two_nulls.py --reciprocal, all cities  -> rp_two_null_{population,antenna}.csv
#   ladder   mixing_null_5x5.py --reciprocal            -> rp_r5x5_*.csv -> rp_gf_roles_*.csv
#   compare  backbone_compare.py --prefix rp_ (needs MAIN_DERIVED; optional)
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$CODE")"

GRAPHS="${GRAPHS:?set GRAPHS to the folder with the original grafo_<city>.gt files}"
LISTS="${LISTS:?set LISTS to the folder with powered_population.txt and powered_antenna.txt}"
LIST_POP="${LIST_POP:-$LISTS/powered_population.txt}"   # ladder, equal-population (110)
LIST_ANT="${LIST_ANT:-$LISTS/powered_antenna.txt}"      # ladder, equal-area (35)
MAIN_DERIVED="${MAIN_DERIVED:-$REPO/data/derived}"      # only for 'compare'

JOBS_SMALL="${JOBS_SMALL:-12}"   # ordinary cities in parallel
JOBS_BIG="${JOBS_BIG:-4}"        # megacities in parallel (natri can hold all four)
N_PERM="${N_PERM:-500}"; N_MIX="${N_MIX:-300}"; N_SAMPLES="${N_SAMPLES:-300}"
MIN_CAPABLE="${MIN_CAPABLE:-5000}"; export MIN_CAPABLE
export EXTRA_ARGS="--reciprocal"

STEP="${1:-all}"
MODES=(population antenna)
vp_of()   { echo "quintile_nation_$1"; }
list_of() { [[ "$1" == population ]] && echo "$LIST_POP" || echo "$LIST_ANT"; }
say()     { echo; echo "=== $* ($(date '+%F %T'))"; }

step_check() {
  say "check"
  python - "$CODE" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
import graph_tool, mixing_null as mn, mixing_null_5x5 as m5
assert hasattr(mn, "reciprocity_preserving_null"), "mixing_null.py is an older version"
assert hasattr(m5, "r5x5_one_city_reciprocal"), "mixing_null_5x5.py is an older version"
print("  scripts ok, graph-tool", graph_tool.__version__)
PY
  local bad=0 f
  for f in run_two_nulls.py mixing_null.py mixing_null_5x5.py gf_roles.py backbone_compare.py \
           run_two_nulls_parallel.sh run_parallel_5x5.sh; do
    [[ -f "$CODE/$f" ]] || { echo "  MISSING  $CODE/$f"; bad=1; }
  done
  grep -q EXTRA_ARGS "$CODE/run_two_nulls_parallel.sh" && grep -q EXTRA_ARGS "$CODE/run_parallel_5x5.sh" \
    || { echo "  OLD      the two run_*_parallel.sh runners do not pass --reciprocal; unpack the new ones"; bad=1; }
  for f in "$GRAPHS" "$LIST_POP" "$LIST_ANT"; do
    if [[ -e "$f" ]]; then echo "  found    $f"; else echo "  MISSING  $f"; bad=1; fi
  done
  [[ -d "$MAIN_DERIVED" ]] && echo "  found    $MAIN_DERIVED" \
                           || echo "  missing  $MAIN_DERIVED  (only the 'compare' step needs it)"
  echo "  $(ls "$GRAPHS"/grafo_*.gt 2>/dev/null | wc -l) graphs for the pole census;" \
       "$(grep -c . "$LIST_POP" 2>/dev/null) + $(grep -c . "$LIST_ANT" 2>/dev/null) cities for the ladder"
  echo "  $(nproc) cores; jobs $JOBS_SMALL + $JOBS_BIG megacities"
  (( bad )) && { echo "fix the items above"; exit 2; }
  return 0
}

step_pole() {
  ls -1 "$GRAPHS" | sed -n 's/^grafo_\(.*\)\.gt$/\1/p' > all_cities.txt
  for mode in "${MODES[@]}"; do
    say "pole census, reciprocal null, $mode ($(wc -l < all_cities.txt) cities)"
    RUNNER="$CODE/run_two_nulls.py" N_PERM="$N_PERM" N_MIX="$N_MIX" \
      bash "$CODE/run_two_nulls_parallel.sh" "$GRAPHS" all_cities.txt "$(vp_of "$mode")" \
           "rp_two_null_${mode}.csv" "$JOBS_SMALL" "$JOBS_BIG"
  done
}

step_ladder() {
  for mode in "${MODES[@]}"; do
    say "ladder census, reciprocal null, $mode"
    RUNNER="$CODE/mixing_null_5x5.py" N_SAMPLES="$N_SAMPLES" \
      bash "$CODE/run_parallel_5x5.sh" "$GRAPHS" "$(list_of "$mode")" "$(vp_of "$mode")" \
           "rp_r5x5_${mode}.csv" "$JOBS_SMALL" "$JOBS_BIG"
    python "$CODE/gf_roles.py" --r5x5 "rp_r5x5_${mode}.csv" --exclude_returns \
        --min_capable "$MIN_CAPABLE" --out "rp_gf_roles_${mode}.csv"
  done
}

step_compare() {
  say "compare with the published results"
  python "$CODE/backbone_compare.py" --prefix rp_ --main_dir "$MAIN_DERIVED" --bb_dir . \
      --min_capable "$MIN_CAPABLE" --out rp_comparison.csv | tee rp_report.txt
}

case "$STEP" in
  all)   step_check; step_pole; step_ladder; step_compare ;;
  check|pole|ladder|compare) "step_$STEP" ;;
  *) echo "unknown step '$STEP' (all|check|pole|ladder|compare)"; exit 2 ;;
esac
