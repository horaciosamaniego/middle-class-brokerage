#!/usr/bin/env python3
"""
within_stratum_density.py -- rule out (or confirm) that the row-gradient in itinerant/
coordinator enrichment (within-stratum ties are more brokered the higher the stratum's
income rank) is explained by higher strata simply being internally DENSER / more
clustered.

The mixing null holds degree + the 5x5 mixing matrix fixed, so in principle within-
stratum density is already controlled. This is the explicit check a referee will ask
for: does per-stratum internal structure track the enrichment gradient?

Per city, per stratum k, computes:
  n_k             stratum size
  m_kk            # directed within-stratum edges (k->k)
  dens_kk         within-stratum edge density = m_kk / (n_k*(n_k-1))
  mean_wdeg       mean within-stratum out-degree = m_kk / n_k
  clust_kk        (optional) local clustering among k-nodes on the induced subgraph
and joins the endpoint-row enrichment (median R_mix of within-stratum brokerage whose
ENDPOINT is stratum k) from the r5x5 tidy file.

Then reports, across cities, the correlation between the row-enrichment and each density
measure. If enrichment still rises with rank AFTER conditioning on density (partial
correlation), the "higher class -> more brokered internal ties" finding is net of density.

  python within_stratum_density.py --data_dir DIR --cities "Natal" "São Paulo" ... \
      --vp quintile_nation_population --r5x5 r5x5_population.csv --out wsd.csv
"""
import argparse, os
import numpy as np
import pandas as pd


def stratum_density(g, q):
    """Return per-stratum (n_k, m_kk, dens_kk, mean_wdeg) for k=1..5."""
    E = g.get_edges()
    s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64)
    qs = q[s]; qt = q[t]
    out = {}
    for k in range(1, 6):
        n_k = int((q == k).sum())
        m_kk = int(((qs == k) & (qt == k)).sum())      # directed within-k edges
        dens = m_kk / (n_k * (n_k - 1)) if n_k > 1 else np.nan
        mwd = m_kk / n_k if n_k > 0 else np.nan
        out[k] = (n_k, m_kk, dens, mwd)
    return out


def row_enrichment(r5x5_df, city, min_capable):
    """For within-stratum brokerage (endpoints i==j==k), return per-endpoint-stratum k
    the median R_mix SEPARATELY for:
      coordinator : broker is an insider (m == k)
      itinerant   : broker is an outsider (m != k)   [pooled over above/below]
    plus the pooled value, all testable cells only. Returns dict k -> (coord, itin, pooled)."""
    d = r5x5_df[(r5x5_df.city == city) & (r5x5_df.i == r5x5_df.j)]
    d = d[d.null_mean >= min_capable]
    out = {}
    for k in range(1, 6):
        rows_k = d[d.i == k]
        coord = rows_k[rows_k.m == k]
        itin = rows_k[rows_k.m != k]
        out[k] = (
            float(coord.R_mix.median()) if len(coord) else np.nan,
            float(itin.R_mix.median()) if len(itin) else np.nan,
            float(rows_k.R_mix.median()) if len(rows_k) else np.nan,
        )
    return out



def _process_city(args):
    """Worker: load one graph, compute density + role-split enrichment. Returns rows."""
    p, vp, r5_rows, min_capable = args   # r5_rows: this city's r5x5 rows as list-of-dicts
    import graph_tool.all as gt
    import pandas as pd
    import numpy as np
    if not os.path.exists(p):
        return ("missing", p, [])
    city = os.path.basename(p).replace("grafo_", "").replace(".gt", "")
    try:
        g = gt.load_graph(p)
    except Exception as e:
        return ("fail", f"{city}: {e!r}", [])
    # quintile array
    raw = g.vp[vp]; N = g.num_vertices(); q = np.zeros(N, dtype=np.int16)
    try:
        vals = np.array([raw[v] for v in g.vertices()], dtype="U8")
        for k in range(1, 6):
            q[np.isin(vals, (f"q{k}", str(k), f"{k}.0"))] = k
    except Exception:
        arr = np.array(raw.a, dtype=float)
        for k in range(1, 6):
            q[arr == k] = k
    dens = stratum_density(g, q)
    r5c = pd.DataFrame(r5_rows)   # this city's r5x5 rows, passed from parent
    enr = row_enrichment(r5c, city, min_capable)
    out = []
    for k in range(1, 6):
        n_k, m_kk, d_kk, mwd = dens[k]
        coord, itin, pooled = enr[k]
        out.append(dict(city=city, stratum=k, n_k=n_k, m_kk=m_kk,
                        dens_kk=d_kk, mean_wdeg=mwd,
                        coord_Rmix=coord, itin_Rmix=itin, pooled_Rmix=pooled))
    return ("ok", city, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--cities", nargs="*")
    ap.add_argument("--graph")
    ap.add_argument("--vp", default="quintile_nation_population")
    ap.add_argument("--r5x5", required=True)
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--out", default="within_stratum_density.csv")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (one graph per worker). "
                         "Light task (no rewiring): set to ~n_cores; RAM per worker "
                         "is one graph, so keep below RAM/max_graph_size.")
    a = ap.parse_args()

    import graph_tool.all as gt
    r5 = pd.read_csv(a.r5x5)

    if a.graph:
        paths = [a.graph]
    elif a.data_dir and a.cities:
        paths = [os.path.join(a.data_dir, f"grafo_{c}.gt") for c in a.cities]
    elif a.data_dir:
        # default: all cities present in the r5x5 file
        paths = [os.path.join(a.data_dir, f"grafo_{c}.gt") for c in sorted(r5.city.unique())]
    else:
        raise SystemExit("give --graph or --data_dir (+ optional --cities)")

    def qarr(g):
        raw = g.vp[a.vp]; N = g.num_vertices(); q = np.zeros(N, dtype=np.int16)
        try:
            vals = np.array([raw[v] for v in g.vertices()], dtype="U8")
            for k in range(1, 6):
                q[np.isin(vals, (f"q{k}", str(k), f"{k}.0"))] = k
        except Exception:
            arr = np.array(raw.a, dtype=float)
            for k in range(1, 6):
                q[arr == k] = k
        return q

    # pre-slice the r5x5 rows per city in the parent (read the big CSV once)
    r5_by_city = {c: sub.to_dict("records") for c, sub in r5.groupby("city")}
    def _city_of(pp):
        return os.path.basename(pp).replace("grafo_", "").replace(".gt", "")
    tasks = [(p, a.vp, r5_by_city.get(_city_of(p), []), a.min_capable) for p in paths]
    rows = []
    if a.jobs > 1:
        import multiprocessing as mp
        # 'spawn' avoids graph-tool/OpenMP fork issues; each worker loads its own graph
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=a.jobs) as pool:
            for status, tag, out in pool.imap_unordered(_process_city, tasks):
                if status == "ok":
                    rows.extend(out); print(f"  {tag}: done")
                elif status == "missing":
                    print(f"  skip (missing): {tag}")
                else:
                    print(f"  FAIL {tag}")
    else:
        for task in tasks:
            status, tag, out = _process_city(task)
            if status == "ok":
                rows.extend(out); print(f"  {tag}: done")
            elif status == "missing":
                print(f"  skip (missing): {tag}")
            else:
                print(f"  FAIL {tag}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({df.city.nunique()} cities)")

    # ---- analysis, SEPARATELY for coordinator (insider) and itinerant (outsider) ----
    def pearson(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5: return np.nan, int(ok.sum())
        return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())

    def partial(x, y, z):
        X = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
        if len(X) < 6: return np.nan, len(X)
        def resid(aa, bb):
            b1 = np.polyfit(bb, aa, 1); return aa - np.polyval(b1, bb)
        rx = resid(X.x.values, X.z.values); ry = resid(X.y.values, X.z.values)
        return float(np.corrcoef(rx, ry)[0, 1]), len(X)

    for label, col in [("COORDINATOR (insider broker, m==k)", "coord_Rmix"),
                       ("ITINERANT   (outsider broker, m!=k)", "itin_Rmix")]:
        v = df.dropna(subset=[col, "dens_kk", "mean_wdeg"])
        print(f"\n================  {label}  ================")
        by = v.groupby("stratum")[col].median()
        print("  median R_mix by endpoint stratum:",
              {int(k): round(x, 3) for k, x in by.items()})
        r_rank, n1 = pearson(v.stratum, v[col])
        r_dens, n2 = pearson(v.dens_kk, v[col])
        print(f"  corr(income rank, enrichment)    r={r_rank:+.3f}  (n={n1})")
        print(f"  corr(within-density, enrichment) r={r_dens:+.3f}  (n={n2})")
        pr, npr = partial(v.stratum.values, v[col].values, v.dens_kk.values)
        prd, nprd = partial(v.stratum.values, v[col].values, v.mean_wdeg.values)
        print(f"  PARTIAL corr(rank, enrichment | within-density)  r={pr:+.3f} (n={npr})")
        print(f"  PARTIAL corr(rank, enrichment | within-mean-deg) r={prd:+.3f} (n={nprd})")

    print("\n  reading: compare the two roles' rank gradients. If ITINERANT rises with rank")
    print("  but COORDINATOR is flat, the finding is that higher classes increasingly rely on")
    print("  OUTSIDERS to close their internal gaps -- the non-obvious result. If both rise,")
    print("  the gradient is generic within-class mediation. Partial corr ~ raw => net of density.")


if __name__ == "__main__":
    main()
    