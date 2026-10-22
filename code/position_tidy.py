#!/usr/bin/env python3
"""
position_tidy.py -- reshape the wide sweep output(s) into the long tidy
position_summary.csv that position_figure.py consumes.

Input: one or more wide CSVs from sweep_brokerage.py (concatenated across machines
and/or countries), each with per-stratum columns brok_share_Qk and brok_z_Qk.
Output: one row per (city, stratum) with rho_obs (=share) and z.

Note on the figure's null band: the sweep emits the degree-stratified permutation
z, not a rho null interval, so the main-text OCCUPANCY panel uses shares (rho_obs)
with the z carried alongside for the SI z-panel. If you want a rho null ribbon in
the main panel, emit brok_share null mean/CI from brokerage_profile (currently it
returns z only); flagged rather than faked here.

Usage:
  python position_tidy.py --wide sweep_natri.csv sweep_wawa.csv sweep_chile.csv \
         --out position_summary.csv
"""
import argparse
import numpy as np
import pandas as pd


def reshape_one(df):
    need_share = [f"brok_share_Q{k}" for k in range(1, 6)]
    if not set(need_share) <= set(df.columns):
        raise SystemExit(f"missing share columns; have {sorted(df.columns)[:8]}...")
    have_z = all(f"brok_z_Q{k}" in df.columns for k in range(1, 6))
    have_null = all(f"brok_share_null_{s}_Q{k}" in df.columns
                    for s in ("mean", "lo", "hi") for k in range(1, 6))
    has_pairs = "pairs_reached" in df.columns
    has_tried = "pairs_tried" in df.columns
    rows = []
    for _, r in df.iterrows():
        for k in range(1, 6):
            row = dict(city=r["city"], stratum=k, rho_obs=r[f"brok_share_Q{k}"])
            row["z"] = r[f"brok_z_Q{k}"] if have_z else np.nan
            if have_null:
                row["rho_null_mean"] = r[f"brok_share_null_mean_Q{k}"]
                row["rho_null_lo"] = r[f"brok_share_null_lo_Q{k}"]
                row["rho_null_hi"] = r[f"brok_share_null_hi_Q{k}"]
            else:
                row["rho_null_mean"] = row["rho_null_lo"] = row["rho_null_hi"] = np.nan
            row["pairs_reached"] = r["pairs_reached"] if has_pairs else np.nan
            row["pairs_tried"] = r["pairs_tried"] if has_tried else np.nan
            row["delta"] = row["delta_lo"] = row["delta_hi"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", nargs="+", required=True, help="wide sweep CSV(s) to concatenate")
    ap.add_argument("--necessity", default=None,
                    help="optional necessity_sweep.py output (long: city,stratum,delta,...)")
    ap.add_argument("--out", default="position_summary.csv")
    a = ap.parse_args()
    parts = []
    for w in a.wide:
        d = pd.read_csv(w)
        parts.append(reshape_one(d))
        print(f"  {w}: {d.city.nunique()} cities")
    tidy = pd.concat(parts, ignore_index=True).drop_duplicates(["city", "stratum"])

    if a.necessity:
        nec = pd.read_csv(a.necessity)[["city", "stratum", "delta", "delta_lo", "delta_hi"]]
        tidy = tidy.drop(columns=["delta", "delta_lo", "delta_hi"]).merge(
            nec, on=["city", "stratum"], how="left")
        n_nec = nec.city.nunique()
        print(f"  merged necessity for {n_nec} cities (middle strata)")

    tidy.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {tidy.city.nunique()} cities x 5 strata = {len(tidy)} rows")
    