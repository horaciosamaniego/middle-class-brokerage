#!/usr/bin/env bash
# run_mechanism.sh -- Section 2 (MECHANISM), both nulls, three stratifications, both poles.
# Runs the existing two-null harness three times (one per quintile property), then
# assembles the paper figures/tables. Independent of Position -- can run on WAWA while
# Position runs on NATRI.  Adjust the run_two_nulls.py flags to match your actual CLI.
set -euo pipefail

DATA_DIR="${1:?usage: run_mechanism.sh <graph_dir> [module]}"
MODULE="${2:-brokerage_constraint_pipeline}"

for MODE in antenna population width; do
  VP="quintile_nation_${MODE}"
  echo "=== TWO-NULL: ${MODE} (${VP}) ==="
  # NOTE: confirm run_two_nulls.py's flag names. It must run BOTH poles (5 and 1) and
  # BOTH nulls (R_perm, R_mix) and emit the summary schema mechanism_pipeline.py expects.
  python run_two_nulls.py --data_dir "$DATA_DIR" --vp "$VP" \
      --out "two_null_${MODE}.csv"
done

echo "=== assembling paper figures + tables ==="
python mechanism_pipeline.py \
    --summary antenna=two_null_antenna.csv \
              population=two_null_population.csv \
              width=two_null_width.csv \
    --min_capable 5000 --outdir mechanism_figs --fmt svg pdf
echo "done -> mechanism_figs/"
