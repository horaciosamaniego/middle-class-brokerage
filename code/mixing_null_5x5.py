#!/usr/bin/env python3
"""
mixing_null_5x5.py -- Option 2 (Path A): the full 5x5 stratum brokerage matrix under
the SAME mixing-preserving null as the pole statistic.

Key fact (verified against mixing_null.py): the pole R_mix already rewires with
    random_rewire(model="constrained-configuration", block_membership=<quintile 1..5>)
which preserves the ENTIRE 5x5 count mixing matrix M_ab and every node's in/out degree.
The pole analysis merely reads ONE pair's statistic off that null. So the 5x5 needs NO
new null -- identical rewiring -- only a broader readout. This guarantees the ladder
result is null-identical to the pole result (a referee cannot say otherwise).

Statistic, per ordered pair (i -> j) and middle stratum m, IDENTICAL in form to
restricted_boundary_control's product (directed two-path COUNT, not a binary broker flag):

    B[i,j,m] = sum over brokers b in stratum m of  in_i(b) * out_j(b)

where in_i(b)=# edges into b from stratum i, out_j(b)=# edges from b to stratum j.
(The pole beta^BC for (pole,counter) equals B[counter,pole,pole] + B[pole,counter,pole].)

R_mix[i,j,m] = B_obs[i,j,m] / mean_over_samples B_null[i,j,m].

Usage:
  python mixing_null_5x5.py --data_dir DIR --cities "São Paulo" --vp quintile_nation_population \
     --n_samples 200 --burn_in 10 --out r5x5_saopaulo.csv
"""
import argparse, os
import numpy as np
import mixing_null as mn   # reuse the validated primitives


def brokerage_tensor_product(g, quintile_prop, weight_ep=None):
    """B[i,j,m] = sum_{b: q(b)=m} in_i(b)*out_j(b).  Same product form as the pole BC.
    Returned as a 6x6x6 array (1-indexed strata; row 0/col 0 unused)."""
    src, tgt, w, N = mn._edge_arrays(g, weight_ep)
    q = mn._quintile_array(g, quintile_prop)
    # in_i[b] = # (weighted) edges INTO b whose source is stratum i
    # out_j[b] = # (weighted) edges OUT OF b whose target is stratum j
    in_by = np.zeros((6, N)); out_by = np.zeros((6, N))
    for k in range(1, 6):
        mi = q[src] == k
        in_by[k] = np.bincount(tgt[mi], weights=w[mi], minlength=N)
        mo = q[tgt] == k
        out_by[k] = np.bincount(src[mo], weights=w[mo], minlength=N)
    B = np.zeros((6, 6, 6))
    for m in range(1, 6):
        sel = q == m
        for i in range(1, 6):
            ii = in_by[i][sel]
            if not ii.any():
                continue
            for j in range(1, 6):
                B[i, j, m] = float((ii * out_by[j][sel]).sum())
    return B


def r5x5_one_city(g, quintile_prop, n_samples=200, n_iter=1, burn_in=10,
                  weight_ep=None, seed=None):
    """Observed and null-mean 5x5x5 tensors, and R_mix, using the identical
    constrained-configuration rewiring as the pole null."""
    from graph_tool.generation import random_rewire
    try:
        from graph_tool import seed_rng
        if seed is not None:
            seed_rng(int(seed))
    except Exception:
        pass
    if not g.is_directed():
        raise ValueError("expects a directed graph")

    B_obs = brokerage_tensor_product(g, quintile_prop, weight_ep)

    g2 = g.copy()
    bm = g2.new_vertex_property("int")
    bm.a = mn._quintile_array(g2, quintile_prop)          # full 1..5 blocks == pole null
    rw = dict(model="constrained-configuration", block_membership=bm,
              edge_sweep=True, parallel_edges=False, self_loops=False)

    if burn_in > 0:
        random_rewire(g2, n_iter=int(burn_in), **rw)

    acc = np.zeros((6, 6, 6)); acc2 = np.zeros((6, 6, 6))
    for _ in range(n_samples):
        random_rewire(g2, n_iter=int(n_iter), **rw)
        Bs = brokerage_tensor_product(g2, quintile_prop, weight_ep)
        acc += Bs; acc2 += Bs * Bs
    null_mean = acc / n_samples
    null_sd = np.sqrt(np.maximum(acc2 / n_samples - null_mean**2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(null_mean > 0, B_obs / null_mean, np.nan)
    return B_obs, null_mean, null_sd, R


def tidy_rows(city, B_obs, null_mean, null_sd, R, min_capable=5000, n_nodes=None, n_edges=None):
    rows = []
    for i in range(1, 6):
        for j in range(1, 6):
            for m in range(1, 6):
                rows.append(dict(
                    city=city, i=i, j=j, m=m,
                    B_obs=float(B_obs[i, j, m]),
                    null_mean=float(null_mean[i, j, m]),
                    null_sd=float(null_sd[i, j, m]),
                    R_mix=float(R[i, j, m]),
                    testable=int(null_mean[i, j, m] >= min_capable),
                    n_nodes=n_nodes, n_edges=n_edges,
                ))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--graph")
    ap.add_argument("--cities", nargs="*")
    ap.add_argument("--vp", default="quintile_nation_population")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--n_iter", type=int, default=1)
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out_dir", default=None,
                    help="if set, write one <city>.csv per city here and skip cities already done "
                         "(for parallel/resumable sweeps); --out still writes a combined file if given")
    a = ap.parse_args()
    import pandas as pd
    import graph_tool.all as gt

    if a.graph:
        paths = [a.graph]
    elif a.data_dir and a.cities:
        paths = [os.path.join(a.data_dir, f"grafo_{c}.gt") for c in a.cities]
    else:
        raise SystemExit("give --graph or --data_dir + --cities")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)

    allrows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"skip (missing): {p}"); continue
        city = os.path.basename(p).replace("grafo_", "").replace(".gt", "")
        cpath = os.path.join(a.out_dir, f"{city}.csv") if a.out_dir else None
        if cpath and os.path.exists(cpath):
            print(f"skip (done): {city}")
            allrows += pd.read_csv(cpath).to_dict("records"); continue
        g = gt.load_graph(p)
        B_obs, nm, sd, R = r5x5_one_city(g, a.vp, a.n_samples, a.n_iter, a.burn_in, seed=a.seed)
        rows = tidy_rows(city, B_obs, nm, sd, R, a.min_capable,
                         n_nodes=int(g.num_vertices()), n_edges=int(g.num_edges()))
        allrows += rows
        if cpath:
            pd.DataFrame(rows).to_csv(cpath, index=False)
            print(f"  wrote {cpath}")
        # print the R_mix grid collapsed to the dominant middle per (i,j)
        print(f"\n=== {city}: R_mix per pair, dominant middle stratum ===")
        print("  (cell shows m*=argmax_m R_mix  and  that R_mix, testable cells only)")
        hdr = "   j=" + "".join(f"{j:>10}" for j in range(1, 6)); print(hdr)
        for i in range(1, 6):
            cells = []
            for j in range(1, 6):
                best_m, best_r = 0, -1
                for m in range(1, 6):
                    if nm[i, j, m] >= a.min_capable and R[i, j, m] > best_r:
                        best_m, best_r = m, R[i, j, m]
                cells.append(f"  Q{best_m}:{best_r:5.2f}" if best_m else "     --   ")
            print(f" i={i} " + "".join(cells))
    if a.out and allrows:
        pd.DataFrame(allrows).to_csv(a.out, index=False)
        print(f"\nwrote {a.out} ({len(allrows)} rows = {len(allrows)//125} cities x 125 cells)")