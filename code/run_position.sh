#!/usr/bin/env bash
# run_position.sh -- Section 1 (POSITION) across the three stratifications.
# Runs the existing occupancy sweep + necessity three times (one per quintile prop),
# then assembles the paper figures/tables. Occupancy on ALL cities; necessity on the
# powered set only, and NOT for width (degenerate Q5). Edit paths/module as needed.
set -euo pipefail

DATA_DIR="${1:?usage: run_position.sh <graph_dir> <module>}"
MODULE="${2:-brokerage_constraint_pipeline}"
POWERED="powered_cities.txt"     # produced by pairs_floor.py --emit (fraction>=0.5)

for MODE in antenna population width; do
  VP="quintile_nation_${MODE}"
  echo "=== OCCUPANCY sweep: ${MODE} (${VP}) ==="
  python sweep_brokerage.py --data_dir "$DATA_DIR" --module "$MODULE" \
      --vp "$VP" --out "sweep_${MODE}.csv"
  python position_tidy.py --wide "sweep_${MODE}.csv" --out "position_summary_${MODE}.csv"
done

for MODE in antenna population; do          # width dropped: degenerate poles
  VP="quintile_nation_${MODE}"
  echo "=== NECESSITY: ${MODE} (${VP}) ==="
  python necessity_sweep.py --data_dir "$DATA_DIR" --module "$MODULE" \
      --vp "$VP" --cities_file "$POWERED" --out "necessity_${MODE}.csv"
  # merge necessity Delta into the tidy occupancy summary for the figure
  python position_tidy.py --wide "sweep_${MODE}.csv" \
      --necessity "necessity_${MODE}.csv" --out "position_summary_${MODE}.csv"
done

echo "=== assembling paper figures + tables ==="
python position_pipeline.py \
    --summary antenna=position_summary_antenna.csv \
              population=position_summary_population.csv \
              width=position_summary_width.csv \
    --income antenas_income_areal.csv \
    --min_frac 0.5 --outdir position_figs --fmt svg pdf
echo "done -> position_figs/"
