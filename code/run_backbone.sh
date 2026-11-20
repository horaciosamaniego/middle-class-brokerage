#!/usr/bin/env bash
# run_backbone.sh -- every analysis of the paper, re-run on the RECIPROCATED BACKBONE
# (mutual ties only), plus the comparison with the published results.
#
# Run from an empty working folder, with the graph-tool environment active:
#
#   conda activate <env with graph-tool>
#   mkdir -p ~/backbone_run && cd ~/backbone_run
#   bash /path/to/middle-class-brokerage/code/run_backbone.sh all
#
# Steps can be run one at a time, in this order (each is resumable):
#   check      confirm the patched scripts and the inputs are in place
#   build      make_backbone.py          -> $BB/grafo_*.gt, backbone_stats.csv, returns_diag.csv
#   returns    gf_roles.py on the ORIGINAL ladder runs, return paths removed
#   pole       run_two_nulls.py          -> bb_two_null_{population,antenna}.csv
#   ladder     mixing_null_5x5.py        -> bb_r5x5_*.csv -> bb_gf_roles_*.csv
#   position   sweep_brokerage.py        -> bb_sweep_*.csv -> bb_position_summary_*.csv
#   necessity  necessity_sweep.py        -> bb_necessity_*.csv
#   compare    backbone_compare.py       -> backbone_report.txt, backbone_comparison.csv
#
# Every setting below can be overridden from the environment, e.g.
#   GRAPHS=/data/grafos_v4 JOBS_BUILD=3 bash run_backbone.sh build
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$CODE")"

GRAPHS="${GRAPHS:-$HOME/grafos_ciudades_dirigidos_v4}"   # original directed graphs
BB="${BB:-$HOME/grafos_backbone_v4}"                      # backbone graphs go here
LISTS="${LISTS:-$HOME/brokerage_lists}"                   # folder with the city lists
LIST_POP="${LIST_POP:-$LISTS/powered_population.txt}"    # ladder, equal-population (110)
LIST_ANT="${LIST_ANT:-$LISTS/powered_antenna.txt}"       # ladder, equal-area (35)
LIST_POS="${LIST_POS:-$LISTS/powered_cities.txt}"        # necessity (95)
MAIN_R5X5_POP="${MAIN_R5X5_POP:-$LISTS/r5x5_population.csv}"   # original ladder runs
MAIN_R5X5_ANT="${MAIN_R5X5_ANT:-$LISTS/r5x5_antenna.csv}"
MAIN_DERIVED="${MAIN_DERIVED:-$REPO/data/derived}"        # published derived files

JOBS_BUILD="${JOBS_BUILD:-2}"     # backbones built in parallel (each loads a full ORIGINAL graph)
JOBS_SMALL="${JOBS_SMALL:-12}"    # null-model jobs, ordinary cities
JOBS_BIG="${JOBS_BIG:-3}"         # null-model jobs, megacities
N_PERM="${N_PERM:-500}"; N_MIX="${N_MIX:-300}"; N_SAMPLES="${N_SAMPLES:-300}"
MIN_CAPABLE="${MIN_CAPABLE:-5000}"; export MIN_CAPABLE   # paper: 5000 for poles and ladder cells
# RAM the occupancy sweep may plan for: cities whose estimated peak exceeds it are
# skipped and logged (bb_sweep_*.csv.failures.log), never OOM-killed. Default 90% of RAM.
MEM_GB="${MEM_GB:-$(awk '/MemTotal/ {printf "%d", $2 * 0.9 / 1048576}' /proc/meminfo)}"

STEP="${1:-all}"
MODES=(population antenna)
vp_of()   { echo "quintile_nation_$1"; }
list_of() { [[ "$1" == population ]] && echo "$LIST_POP" || echo "$LIST_ANT"; }
r5_of()   { [[ "$1" == population ]] && echo "$MAIN_R5X5_POP" || echo "$MAIN_R5X5_ANT"; }
say()     { echo; echo "=== $* ($(date '+%F %T'))"; }

step_check() {
  say "check"
  python - "$CODE" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
import graph_tool, mixing_null as mn, mixing_null_5x5 as m5
assert hasattr(mn, "is_symmetric"), "mixing_null.py is the old version"
assert hasattr(m5, "returns_matrix"), "mixing_null_5x5.py is the old version"
print("  scripts ok, graph-tool", graph_tool.__version__)
PY
  local bad=0
  for f in make_backbone.py backbone_compare.py gf_roles.py run_two_nulls.py mixing_null.py brokerage_position_only.py \
           mixing_null_5x5.py sweep_brokerage.py brokerage_constraint_pipeline.py \
           position_tidy.py necessity_sweep.py run_two_nulls_parallel.sh run_parallel_5x5.sh; do
    [[ -f "$CODE/$f" ]] || { echo "  MISSING  $CODE/$f  (unpack the tarball into the repo, next to the other scripts)"; bad=1; }
  done
  for f in two_null_population_summary.csv gf_roles_population.csv necessity_population.csv \
           position_summary_population.csv; do
    [[ -f "$MAIN_DERIVED/$f" ]] || echo "  missing  $MAIN_DERIVED/$f  (only the 'compare' step needs it)"
  done
  for f in "$GRAPHS" "$LIST_POP" "$LIST_ANT" "$LIST_POS"; do
    if [[ -e "$f" ]]; then echo "  found    $f"; else echo "  MISSING  $f"; bad=1; fi
  done
  for f in "$MAIN_R5X5_POP" "$MAIN_R5X5_ANT"; do
    [[ -e "$f" ]] && echo "  found    $f" || echo "  missing  $f  (only the 'returns' step needs it)"
  done
  for l in "$LIST_POP" "$LIST_ANT" "$LIST_POS"; do
    [[ -f "$l" ]] && echo "  $(grep -c . "$l") cities in $(basename "$l")"
  done
  (( bad )) && { echo "fix the paths above (see the settings at the top of this script)"; exit 2; }
  echo "  backbones will be written to $BB"
  echo "  $(nproc) cores, RAM budget ${MEM_GB} GiB; parallel jobs: build $JOBS_BUILD," \
       "nulls $JOBS_SMALL (+$JOBS_BIG megacities), occupancy and removal ${JOBS_SERIAL:-6}"
}

step_build() {
  say "build backbones"
  python "$CODE/make_backbone.py" --data_dir "$GRAPHS" --out_dir "$BB" \
      --cities_file "$LIST_POP" "$LIST_ANT" "$LIST_POS" \
      --vps quintile_nation_population quintile_nation_antenna \
      --jobs "$JOBS_BUILD" --stats backbone_stats.csv --returns returns_diag.csv
  # every city with a backbone, for the pole and position steps
  ls -1 "$BB" | sed -n 's/^grafo_\(.*\)\.gt$/\1/p' > bb_cities.txt
  echo "  $(wc -l < bb_cities.txt) backbones in $BB"
}

step_returns() {
  say "original ladder, return paths removed"
  for mode in "${MODES[@]}"; do
    local r5; r5="$(r5_of "$mode")"
    if [[ ! -f "$r5" ]]; then echo "  skip $mode: $r5 not found"; continue; fi
    python "$CODE/gf_roles.py" --r5x5 "$r5" --returns returns_diag.csv --vp "$(vp_of "$mode")" \
        --exclude_returns --min_capable "$MIN_CAPABLE" --out "gf_roles_${mode}_noreturns.csv"
  done
}

step_pole() {
  for mode in "${MODES[@]}"; do
    say "pole two-null census, $mode"
    RUNNER="$CODE/run_two_nulls.py" N_PERM="$N_PERM" N_MIX="$N_MIX" \
      bash "$CODE/run_two_nulls_parallel.sh" "$BB" bb_cities.txt "$(vp_of "$mode")" \
           "bb_two_null_${mode}.csv" "$JOBS_SMALL" "$JOBS_BIG"
  done
}

step_ladder() {
  for mode in "${MODES[@]}"; do
    say "ladder census, $mode"
    RUNNER="$CODE/mixing_null_5x5.py" N_SAMPLES="$N_SAMPLES" \
      bash "$CODE/run_parallel_5x5.sh" "$BB" "$(list_of "$mode")" "$(vp_of "$mode")" \
           "bb_r5x5_${mode}.csv" "$JOBS_SMALL" "$JOBS_BIG"
    python "$CODE/gf_roles.py" --r5x5 "bb_r5x5_${mode}.csv" --exclude_returns \
        --min_capable "$MIN_CAPABLE" --out "bb_gf_roles_${mode}.csv"
    python "$CODE/gf_roles.py" --r5x5 "bb_r5x5_${mode}.csv" \
        --min_capable "$MIN_CAPABLE" --out "bb_gf_roles_${mode}_withreturns.csv" > /dev/null
  done
}

# The occupancy sweep and the removal test run one city per process. They are run as
# JOBS_SERIAL processes side by side. Graphs above BIG_BYTES (backbone .gt size) all go
# to process 0, one after another, with most of the RAM budget; the rest share the
# smaller cities. Each process writes its own part file; the parts are then merged.
JOBS_SERIAL="${JOBS_SERIAL:-6}"
# brokerage_position_only skips Burt's constraint, which the paper does not use and which
# is the slow, memory-hungry part of brokerage_constraint_pipeline.run_city on big graphs.
# The occupancy columns are identical either way.
POSITION_MODULE="${POSITION_MODULE:-brokerage_position_only}"
BIG_BYTES="${BIG_BYTES:-2000000000}"; export BIG_BYTES

merge_parts() {   # merge_parts <out.csv> <part files...>
  local out="$1"; shift
  python - "$out" "$@" <<'PY'
import sys, os, pandas as pd
out, parts = sys.argv[1], [p for p in sys.argv[2:] if os.path.exists(p) and os.path.getsize(p) > 0]
if not parts:
    sys.exit(f"no part files for {out}")
d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
key = [c for c in ("city", "stratum") if c in d.columns]
d = d.drop_duplicates(key, keep="last")
d.to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out)
print(f"  merged {len(parts)} parts -> {out} ({d.city.nunique()} cities)")
PY
}

step_position() {
  local P="$JOBS_SERIAL"
  local mem0 memk
  mem0=$(( MEM_GB * 6 / 10 )); memk=$(( MEM_GB * 4 / 10 / (P > 1 ? P - 1 : 1) ))
  for mode in "${MODES[@]}"; do
    say "occupancy sweep, $mode ($P processes; megacities in process 0, ${mem0} GiB)"
    local pids=() p
    for (( p = 0; p < P; p++ )); do
      python "$CODE/sweep_brokerage.py" --data_dir "$BB" --vp "$(vp_of "$mode")" \
          --module "$POSITION_MODULE" \
          --n_perm "$N_PERM" --n_partitions "$P" --partition "$p" --max_bytes "$BIG_BYTES" \
          --mem_budget_gb "$(( p == 0 ? mem0 : memk ))" \
          --out "bb_sweep_${mode}.part${p}.csv" > "bb_sweep_${mode}.part${p}.log" 2>&1 &
      pids+=($!)
    done
    local fail=0
    for p in "${!pids[@]}"; do wait "${pids[$p]}" || { echo "  process $p failed: see bb_sweep_${mode}.part${p}.log"; fail=1; }; done
    cat bb_sweep_${mode}.part*.csv.failures.log 2>/dev/null | grep -i "skip\|fail" | sed 's/^/  /' || true
    merge_parts "bb_sweep_${mode}.csv" bb_sweep_${mode}.part*.csv
    python "$CODE/position_tidy.py" --wide "bb_sweep_${mode}.csv" \
        --out "bb_position_summary_${mode}.csv"
    (( fail )) && echo "  WARN some processes failed; re-run 'position' to finish them (done cities are kept)"
  done
}

step_necessity() {
  local P="$JOBS_SERIAL"
  # deal the cities into P lists by graph size, largest first, snake order
  python - "$LIST_POS" "$BB" "$P" <<'PY'
import sys, os
lst, bb, P = sys.argv[1], sys.argv[2], int(sys.argv[3])
cities = [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]
size = lambda c: os.path.getsize(os.path.join(bb, f"grafo_{c}.gt")) if os.path.exists(os.path.join(bb, f"grafo_{c}.gt")) else 0
cities.sort(key=size, reverse=True)
chunks = [[] for _ in range(P)]
for i, c in enumerate(cities):
    r, k = divmod(i, P)
    chunks[k if r % 2 == 0 else P - 1 - k].append(c)
for k, ch in enumerate(chunks):
    open(f"nec_cities.part{k}.txt", "w", encoding="utf-8").write("\n".join(ch) + ("\n" if ch else ""))
PY
  for mode in "${MODES[@]}"; do
    say "necessity, $mode ($P processes)"
    local pids=() p
    for (( p = 0; p < P; p++ )); do
      [[ -s "nec_cities.part${p}.txt" ]] || continue
      python "$CODE/necessity_sweep.py" --data_dir "$BB" --vp "$(vp_of "$mode")" \
          --cities_file "nec_cities.part${p}.txt" --out "bb_necessity_${mode}.part${p}.csv" \
          > "bb_necessity_${mode}.part${p}.log" 2>&1 &
      pids+=($!)
    done
    local fail=0
    for p in "${!pids[@]}"; do wait "${pids[$p]}" || fail=1; done
    merge_parts "bb_necessity_${mode}.csv" bb_necessity_${mode}.part*.csv
    (( fail )) && echo "  WARN some processes failed; re-run 'necessity' to finish them (done cities are kept)"
  done
}

step_compare() {
  say "compare with the published results"
  python "$CODE/backbone_compare.py" --main_dir "$MAIN_DERIVED" --bb_dir . \
      --min_capable "$MIN_CAPABLE" --out backbone_comparison.csv | tee backbone_report.txt
}

case "$STEP" in
  all)   step_check; step_build; step_returns; step_pole; step_ladder
         step_position; step_necessity; step_compare ;;
  check|build|returns|pole|ladder|position|necessity|compare) "step_$STEP" ;;
  *) echo "unknown step '$STEP' (all|check|build|returns|pole|ladder|position|necessity|compare)"; exit 2 ;;
esac
