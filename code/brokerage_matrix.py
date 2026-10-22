#!/usr/bin/env python3
"""
brokerage_matrix.py -- OBSERVED 5x5 stratum brokerage tensor (no null). Diagnostic
step for Option 2: shows where cross-stratum intermediation actually concentrates and
which cells are dense enough to test under a mixing null before any rewiring is run.

Consistent with the pole census in run_two_nulls.py: a broker b in stratum m brokers
the ordered pair (i,j) iff b has an inbound edge FROM some node in stratum i AND an
outbound edge TO some node in stratum j (a directed two-path i -> b -> j). Unweighted:
each (i, b, j) broker counted once per (i,j), matching the pole statistic's
"boundary-capable" logic generalised to all pairs.

Outputs, per city:
  B[i,j,m]  brokered-pairs tensor  (# brokers in stratum m mediating i->*->j)
  two collapsed 5x5 views:
    BROK[i,j] = sum_m B[i,j,m]        total brokerage volume for pair (i,j)
    for each pair, the middle stratum m that dominates (argmax_m B[i,j,m])
  capability mask: which (i,j,m) cells have >= min_capable brokers (testable under null)

  python brokerage_matrix.py --graph .../grafo_"São Paulo".gt --vp quintile_nation_population
  python brokerage_matrix.py --data_dir DIR --cities "São Paulo" "Natal" --vp ... --out obs_5x5.csv
"""
import argparse, glob, os
import numpy as np


def quintile_array(g, vp):
    """Map the vertex property (strings 'q1'..'q5' or ints) to int16 bins 1..5, 0=other."""
    raw = g.vp[vp]
    N = g.num_vertices()
    q = np.zeros(N, dtype=np.int16)
    try:
        vals = np.array([raw[v] for v in g.vertices()], dtype="U8")
        for k in range(1, 6):
            q[np.isin(vals, (f"q{k}", str(k), f"{k}.0"))] = k
    except Exception:
        arr = np.array(raw.a, dtype=float)
        for k in range(1, 6):
            q[arr == k] = k
    return q


def brokerage_tensor(g, q):
    """B[i,j,m] = number of stratum-m brokers with an in-edge from stratum i and an
    out-edge to stratum j (directed two-path i->m->j). i,j,m in 1..5; b counted once
    per (i,j) it can broker. Poles/middle all treated symmetrically here."""
    N = g.num_vertices()
    E = g.get_edges()
    s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64)
    qs = q[s]; qt = q[t]
    # for each node: which strata does it receive FROM, and send TO
    in_from  = {i: np.zeros(N, dtype=bool) for i in range(1, 6)}
    out_to   = {j: np.zeros(N, dtype=bool) for j in range(1, 6)}
    for i in range(1, 6):
        m_in = qs == i               # edges whose SOURCE is stratum i
        in_from[i][t[m_in]] = True    # their targets receive from stratum i
    for j in range(1, 6):
        m_out = qt == j              # edges whose TARGET is stratum j
        out_to[j][s[m_out]] = True    # their sources send to stratum j
    B = np.zeros((6, 6, 6), dtype=np.int64)   # 1-indexed, ignore 0
    for m in range(1, 6):
        mem = q == m
        for i in range(1, 6):
            bi = in_from[i] & mem
            if not bi.any():
                continue
            for j in range(1, 6):
                B[i, j, m] = int((bi & out_to[j]).sum())
    return B


def summarize(B, city, sizes, min_capable=5000):
    sz = np.array([np.nan] + [max(sizes.get(k, 0), 1) for k in range(1, 6)])  # 1-indexed
    rows = []
    for i in range(1, 6):
        for j in range(1, 6):
            tot = int(B[i, j, 1:].sum())
            dom = int(np.argmax(B[i, j, 1:]) + 1) if tot else 0
            dom_n = int(B[i, j, dom]) if dom else 0
            # abundance-normalized: brokers per member of the middle stratum
            norm = B[i, j, 1:6].astype(float) / sz[1:6]
            dom_norm = int(np.argmax(norm) + 1) if tot else 0
            rows.append(dict(city=city, i=i, j=j, total_brokers=tot,
                             dom_mid=dom, dom_brokers=dom_n, dom_mid_norm=dom_norm,
                             m1=int(B[i,j,1]), m2=int(B[i,j,2]), m3=int(B[i,j,3]),
                             m4=int(B[i,j,4]), m5=int(B[i,j,5]),
                             r1=round(float(norm[0]),4), r2=round(float(norm[1]),4),
                             r3=round(float(norm[2]),4), r4=round(float(norm[3]),4),
                             r5=round(float(norm[4]),4),
                             testable=int(B[i,j,1:].max() >= min_capable)))
    return rows


def run(path, vp, min_capable):
    import graph_tool.all as gt
    city = os.path.basename(path).replace("grafo_", "").replace(".gt", "")
    g = gt.load_graph(path)
    q = quintile_array(g, vp)
    B = brokerage_tensor(g, q)
    comp = {k: int((q == k).sum()) for k in range(1, 6)}
    return summarize(B, city, comp, min_capable), (city, comp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph"); ap.add_argument("--data_dir"); ap.add_argument("--cities", nargs="*")
    ap.add_argument("--vp", default="quintile_nation_population")
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pandas as pd
    paths = []
    if a.graph:
        paths = [a.graph]
    elif a.data_dir and a.cities:
        paths = [os.path.join(a.data_dir, f"grafo_{c}.gt") for c in a.cities]
    else:
        raise SystemExit("give --graph, or --data_dir + --cities")
    allrows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"skip (missing): {p}"); continue
        rows, (city, comp) = run(p, a.vp, a.min_capable)
        allrows += rows
        d = pd.DataFrame(rows)
        print(f"\n=== {city}  strata sizes {comp} ===")
        piv = d.pivot(index="i", columns="j", values="total_brokers")
        print("total brokers per ordered pair (i=row source stratum, j=col target stratum):")
        print(piv.to_string())
        dom = d.pivot(index="i", columns="j", values="dom_mid")
        print("dominant middle stratum per pair (RAW argmax_m -- abundance-confounded):")
        print(dom.to_string())
        domn = d.pivot(index="i", columns="j", values="dom_mid_norm")
        print("dominant middle stratum per pair (ABUNDANCE-NORMALIZED argmax_m B/size_m):")
        print(domn.to_string())
        nt = int(d.testable.sum())
        print(f"testable cells (dominant-broker count >= {a.min_capable}): {nt}/25")
    if a.out and allrows:
        pd.DataFrame(allrows).to_csv(a.out, index=False)
        print(f"\nwrote {a.out} ({len(allrows)} rows = {len(allrows)//25} cities x 25 pairs)")
        