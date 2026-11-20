#!/usr/bin/env python3
"""
brokerage_position_only.py -- run_city for the occupancy sweep WITHOUT Burt's
constraint and without the pole role census.

brokerage_constraint_pipeline.run_city computes three things per city: the
occupancy profile (brokerage_profile), Burt's constraint on the reciprocated
backbone, and a pole role census. The paper uses only the first; the pole
result comes from run_two_nulls.py. The constraint is by far the heaviest part
(a chunked two-hop sparse product whose rows explode around hubs), so on large
graphs it dominates time and memory for nothing the paper reports.

This module returns the same occupancy columns, computed by the same function
with the same seed, so they are identical to run_city's. position_tidy.py reads
only these columns.

  python sweep_brokerage.py --module brokerage_position_only --data_dir ... --vp ...
"""
import os
import time

import numpy as np

from brokerage_constraint_pipeline import brokerage_profile, load_city


def run_city(path, vp="quintile_nation", weight_ep="q_calls", n_perm=500, n_bins=10,
             n_src=400, n_tgt=400, chunk=20000, seed=20270624, label=None, verbose=True):
    name = label or os.path.splitext(os.path.basename(path))[0].replace("grafo_", "", 1)
    t0 = time.time()
    g, q = load_city(path, vp=vp)
    B = brokerage_profile(g, q, n_src=n_src, n_tgt=n_tgt, n_perm=n_perm, n_bins=n_bins, seed=seed)
    row = dict(city=name, quintile_ref=vp, n_nodes=g.num_vertices(), n_edges=g.num_edges(),
               brok_peak=B["peak"], brok_centroid=B["centroid"], brok_no_positive=B["no_positive"],
               pairs_reached=B["pairs_reached"], pairs_tried=B["pairs_tried"])
    for k in (1, 2, 3, 4, 5):
        row[f"brok_z_Q{k}"] = B["z"][k]
        row[f"brok_share_Q{k}"] = B["share"][k]
        row[f"brok_share_null_mean_Q{k}"] = B["share_null_mean"][k]
        row[f"brok_share_null_lo_Q{k}"] = B["share_null_lo"][k]
        row[f"brok_share_null_hi_Q{k}"] = B["share_null_hi"][k]
        row[f"n_Q{k}"] = int((q == k).sum())
    row["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"[{name}] occupancy done in {row['seconds']:.0f}s")
    return row
