#!/usr/bin/env python3
"""
pairs_floor.py -- locate the occupancy power floor (--min_pairs) from where rho_k
STABILISES as reachable pole-pairs rise, and emit the powered-city list.

Below some pairs_reached, rho_k is a ratio of tiny integers: it swings across the
whole 0..1 range and its cross-city spread is dominated by sampling noise, not
structure. Above it, rho_k settles onto the median. That transition is the floor --
the occupancy analogue of the boundary-capable gate, read from data rather than
picked. This is NOT a claim about a city's result, only about whether its rho_k is
estimable; so it is a validity gate, chosen on rho stability, not on any outcome.

Signal per capability bin: the cross-city IQR width of rho_obs (median over strata).
It is large in the noisy tail and collapses toward a floor once pairs_reached is
adequate.

  python pairs_floor.py --csv position_summary.csv                 # inspect
  python pairs_floor.py --csv position_summary.csv --min_pairs 1500 \
         --emit powered_cities.txt                                 # write the list
"""
import argparse
import numpy as np
import pandas as pd


def load(path):
    d = pd.read_csv(path)
    if "pairs_reached" not in d.columns:
        raise SystemExit("no 'pairs_reached' column -- re-run position_tidy on a sweep "
                         "whose run_city wrote pairs_reached")
    # one row per city (pairs_reached repeats across strata); keep city-level rho spread
    return d


def binned(d, nbins):
    per_city = d.groupby("city").agg(pairs=("pairs_reached", "max")).reset_index()
    per_city = per_city[per_city.pairs > 0]
    per_city["bin"] = pd.cut(np.log10(per_city.pairs), nbins)
    # width of the cross-city rho distribution within each bin, averaged over strata
    rows = []
    for b, grp in per_city.groupby("bin", observed=True):
        cities = set(grp.city)
        sub = d[d.city.isin(cities)]
        # per stratum: IQR of rho_obs across these cities; then median over strata
        iqr = sub.groupby("stratum")["rho_obs"].agg(lambda x: x.quantile(.75) - x.quantile(.25))
        rows.append(dict(n_cities=len(cities),
                         pairs_lo=int(grp.pairs.min()), pairs_hi=int(grp.pairs.max()),
                         median_rho_iqr=float(np.median(iqr))))
    return pd.DataFrame(rows)


def cumulative(d, thresholds):
    per_city = d.groupby("city").agg(pairs=("pairs_reached", "max")).reset_index()
    rows = []
    for t in thresholds:
        cities = set(per_city[per_city.pairs >= t].city)
        if not cities:
            continue
        sub = d[d.city.isin(cities)]
        iqr = sub.groupby("stratum")["rho_obs"].agg(lambda x: x.quantile(.75) - x.quantile(.25))
        rows.append(dict(min_pairs=t, n_cities=len(cities),
                         median_rho_iqr=round(float(np.median(iqr)), 4)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="position_summary.csv")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--thresholds", type=int, nargs="*",
                    default=[100, 250, 500, 1000, 1500, 2000, 3000, 5000, 8000])
    ap.add_argument("--min_pairs", type=int, default=None,
                    help="if set with --emit, write the cities at/above this floor")
    ap.add_argument("--emit", default=None, help="path for powered_cities.txt")
    a = ap.parse_args()
    d = load(a.csv)
    per_city = d.groupby("city").agg(pairs=("pairs_reached", "max")).reset_index()
    print(f"{len(per_city)} cities; pairs_reached range "
          f"{int(per_city.pairs.min())}..{int(per_city.pairs.max())}\n")
    print("BINNED (rho_k cross-city IQR shrinks as pairs_reached rises):")
    print(binned(d, a.nbins).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCUMULATIVE (cities at/above each floor):")
    print(cumulative(d, a.thresholds).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=> set --min_pairs where median_rho_iqr flattens; cities above it are the "
          "powered occupancy set.")
    if a.emit:
        if a.min_pairs is None:
            raise SystemExit("pass --min_pairs with --emit")
        cities = sorted(per_city[per_city.pairs >= a.min_pairs].city)
        with open(a.emit, "w") as f:
            f.write("\n".join(cities) + "\n")
        print(f"\nwrote {a.emit}: {len(cities)} cities with pairs_reached >= {a.min_pairs}")