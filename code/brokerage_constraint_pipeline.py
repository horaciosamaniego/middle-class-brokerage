#!/usr/bin/env python3
"""
Brokerage profile (degree-stratified null) + Burt constraint (reciprocated backbone)
for income-stratified CDR voice-call networks. One row per city.

Design decisions (see conversation):
  * Quintiles are INCOME-WIDTH bands (not population quintiles); quintile_nation is the
    common cross-city ruler. Bin sizes are intentionally unequal.
  * Brokerage = degree-robust share of a stratum among interior nodes of shortest
    directed Q_lo -> Q_hi paths. Computed on the FULL directed graph.
  * Burt constraint = structural-holes companion, on the RECIPROCATED, q_calls-weighted
    backbone (mutual ties only). A stratum with low constraint *net of degree* occupies
    structural holes -> should coincide with the brokerage peak.
  * Both nulls shuffle quintile labels WITHIN degree bins, holding connectivity fixed.
    The label-independent heavy object (interior set / constraint vector) is computed once.
"""
import os, time, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import graph_tool.all as gt
from graph_tool.topology import shortest_distance

# ----------------------------------------------------------------------------- loading
_QMAP_KEYS = lambda k: (f"q{k}", str(k), f"{k}.0", f"{k}")

def _parse_quintiles_from_graph(g, vp, n_q):
    vals = np.array(list(g.vp[vp]), dtype="U8")         # single unavoidable string pass
    q = np.zeros(g.num_vertices(), dtype=np.int16)
    for k in range(1, n_q + 1):
        q[np.isin(vals, _QMAP_KEYS(k))] = k
    return q

def load_quintiles(path, vp="quintile_nation", n_q=5, cache=True):
    """Return ONLY the quintile array, reading the .npy cache when present (no graph
    deserialization). This is what makes the composition scan cheap after the first pass."""
    cpath = f"{path}.{vp}.npy"
    if cache and os.path.exists(cpath):
        return np.load(cpath)
    g = gt.load_graph(path)
    q = _parse_quintiles_from_graph(g, vp, n_q)
    if cache:
        np.save(cpath, q)
    return q

def load_city(path, vp="quintile_nation", n_q=5, cache=True):
    """Load .gt and return (g, q). q is int16 (1..n_q; 0 = missing). Quintiles cached."""
    g = gt.load_graph(path)
    cpath = f"{path}.{vp}.npy"
    if cache and os.path.exists(cpath):
        q = np.load(cpath)
    else:
        q = _parse_quintiles_from_graph(g, vp, n_q)
        if cache:
            np.save(cpath, q)
    return g, q

def quintile_report(q, name=""):
    vals, cnts = np.unique(q[q > 0], return_counts=True)
    n0 = int((q == 0).sum())
    if n0:
        print(f"  [{name}] excluded (q=0): {n0}")
    return pd.Series(cnts, index=[f"Q{v}" for v in vals], name=name or "count")

# -------------------------------------------------------------- degree-binned permuter
def _degree_bin_permuter(deg, valid_mask, n_bins=10, seed=0):
    """Return (permute_fn, valid_idx). permute_fn shuffles labels only WITHIN equal-count
    bins of `deg`, restricted to `valid_mask` nodes -> holds the degree profile of each
    stratum fixed under the null."""
    idx = np.where(valid_mask)[0]
    order = idx[np.argsort(deg[idx], kind="stable")]
    bins = [b for b in np.array_split(order, n_bins) if b.size]
    rng = np.random.default_rng(seed)
    def permute(labels):
        p = labels.copy()
        for b in bins:
            p[b] = labels[b][rng.permutation(b.size)]
        return p
    return permute, idx

def _reach(d):
    d = np.asarray(d, dtype=float)
    return (d > 0) & (d < 1e9)        # 1e9 excludes graph-tool's unreachable sentinel

# ----------------------------------------------------------------- brokerage profile
def brokerage_profile(g, q, lo=1, hi=5, strata=(1, 2, 3, 4, 5), interior_strata=(2, 3, 4),
                      n_src=400, n_tgt=400, n_perm=500, n_bins=10, seed=20270624):
    """Degree-robust share of each stratum among interior nodes of shortest Q_lo->Q_hi
    paths. Returns per-stratum z, peak (over interior_strata), z-centroid, and a flag."""
    rng = np.random.default_rng(seed)
    deg = g.get_total_degrees(g.get_vertices()).astype(np.int64)
    src = np.where(q == lo)[0]; tgt = np.where(q == hi)[0]
    if src.size > n_src: src = rng.choice(src, n_src, replace=False)
    if tgt.size > n_tgt: tgt = rng.choice(tgt, n_tgt, replace=False)
    tgt = tgt.astype(np.int64)

    # ---- collect interior-node multiset ONCE (label-independent) ----
    interior = []; reached = 0; tried = int(src.size) * int(tgt.size)
    for sv in src:
        sv = int(sv)
        dist, pred = shortest_distance(g, source=g.vertex(sv), pred_map=True)
        d = dist.a; pa = pred.a
        ok = _reach(d[tgt])
        for tv in tgt[ok]:
            reached += 1; v = int(tv)
            while True:
                p = int(pa[v])
                if p == sv or p == v:
                    break
                interior.append(p); v = p
    meta = dict(interior_visits=len(interior), pairs_reached=reached, pairs_tried=tried)
    
    if not interior:
        return dict(metric="brokerage", z={k: np.nan for k in strata},
                    share={k: np.nan for k in strata},
                    share_null_mean={k: np.nan for k in strata},
                    share_null_lo={k: np.nan for k in strata},
                    share_null_hi={k: np.nan for k in strata},
                    peak=None, centroid=np.nan, no_positive=True, **meta)

    obs = {k: float(np.mean(q[interior] == k)) for k in strata}
    permute, _ = _degree_bin_permuter(deg, q > 0, n_bins=n_bins, seed=seed)
    null = {k: np.empty(n_perm) for k in strata}
    base = q.copy()
    for j in range(n_perm):
        pc = permute(base)[interior]
        for k in strata:
            null[k][j] = np.mean(pc == k)
    z = {}; share_null_mean = {}; share_null_lo = {}; share_null_hi = {}
    for k in strata:
        col = null[k]
        mu, sd = col.mean(), col.std(ddof=1)
        z[k] = float((obs[k] - mu) / sd) if sd > 0 else np.nan
        lo, hi = np.percentile(col, [2.5, 97.5])
        share_null_mean[k] = float(mu)
        share_null_lo[k] = float(lo)
        share_null_hi[k] = float(hi)
    # peak & centroid over the *interior* strata only (true intermediaries between poles)
    zi = np.array([z[k] for k in interior_strata])
    w = np.clip(zi, 0, None)
    no_pos = bool(np.all(w <= 0))
    peak = None if no_pos else int(interior_strata[int(np.argmax(zi))])
    centroid = float(np.sum(np.array(interior_strata) * w) / w.sum()) if not no_pos else np.nan
    return dict(metric="brokerage", z=z, share=obs,
                share_null_mean=share_null_mean, share_null_lo=share_null_lo,
                share_null_hi=share_null_hi,
                peak=peak, centroid=centroid, no_positive=no_pos, **meta)

# ----------------------------------------------------- reciprocated backbone + Burt
def reciprocated_backbone(g, weight_ep="q_calls"):
    """Symmetric sparse adjacency over MUTUAL ties only; A[u,v] = w(u->v)+w(v->u).
    Returns (A_csr, strength) where strength = weighted backbone degree (row sums)."""
    N = g.num_vertices()
    E = g.get_edges([g.ep[weight_ep]])
    s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64); w = E[:, 2].astype(float)
    D = sp.csr_matrix((w, (s, t)), shape=(N, N))
    both = D.multiply(D.T > 0)                 # w(u->v) only where v->u also exists
    A = (both + both.T).tocsr()                # symmetric, summed both directions
    A.setdiag(0); A.eliminate_zeros()
    strength = np.asarray(A.sum(1)).ravel()
    return A, strength

def burt_constraint(A, chunk=20000):
    """Burt's constraint on a symmetric weighted graph (no self-loops).
       c_i = sum_{j in N(i)} (p_ij + sum_q p_iq p_qj)^2,  p = row-normalized A.
    Returned as a per-node vector; NaN where the node has no backbone ties.
    Chunked over rows to bound the transient P@P memory; lower `chunk` for mega-cities."""
    N = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.zeros(N); nz = deg > 0; inv[nz] = 1.0 / deg[nz]
    P = (sp.diags(inv) @ A).tocsr()            # row-normalized; same support as A
    c = np.zeros(N)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        Pc = P[a:b]                            # m x N, support = A rows
        PP = (Pc @ P)                          # transient two-hop (k=i,j terms vanish: no loops)
        mask = A[a:b].astype(bool)             # restrict indirect term to actual neighbours
        M = (PP.multiply(mask) + Pc).tocsr()   # (p_ij + q_ij) on support
        M.data **= 2
        c[a:b] = np.asarray(M.sum(1)).ravel()
    c[deg == 0] = np.nan
    return c

def constraint_profile(c, q, strength, strata=(1, 2, 3, 4, 5),
                       n_perm=500, n_bins=10, seed=20270624):
    """Degree-robust mean constraint per stratum. Negative z = lower constraint than
    degree-matched chance = stratum occupies structural holes (brokerage potential)."""
    valid = np.isfinite(c) & (q > 0)
    obs = {k: float(np.nanmean(c[valid & (q == k)])) if (valid & (q == k)).any() else np.nan
           for k in strata}
    permute, idx = _degree_bin_permuter(strength, valid, n_bins=n_bins, seed=seed)
    base = q.copy(); cc = c
    null = {k: np.empty(n_perm) for k in strata}
    for j in range(n_perm):
        p = permute(base)
        for k in strata:
            m = (p == k) & valid
            null[k][j] = cc[m].mean() if m.any() else np.nan
    z = {}
    for k in strata:
        col = null[k][np.isfinite(null[k])]
        mu, sd = col.mean(), col.std(ddof=1)
        z[k] = float((obs[k] - mu) / sd) if (sd > 0 and np.isfinite(obs[k])) else np.nan
    holes_peak = None
    finite = {k: z[k] for k in (2, 3, 4) if np.isfinite(z[k])}
    if finite:
        holes_peak = int(min(finite, key=finite.get))   # most negative z = strongest holes
    return dict(metric="constraint", mean=obs, z=z, holes_peak=holes_peak,
                n_backbone=int(valid.sum()))

# ------------------------------------------------- Gould-Fernandez role census (mechanism)
def gf_pole_role_census(g, q, lo=1, hi=5, mids=(2, 3, 4), n_perm=500, n_bins=10,
                        seed=20270624):
    """Gould-Fernandez (1989) role census restricted to directed 2-paths a->b->c whose
    ENDPOINTS are the two poles (one Q_lo, one Q_hi). With pole endpoints, every mediated
    path is exactly one of:
        LIAISON         broker b in a middle stratum  (poor <-> rich genuine bridge)
        GATEKEEPER      broker b is a pole, b==target (controls inflow into its own group)
        REPRESENTATIVE  broker b is a pole, b==source (controls outflow from its own group)
    (coordinator/consultant cannot occur: they require same-group endpoints).
    Degree-stratified null, computed by recolouring within degree bins."""
    N = g.num_vertices()
    deg = g.get_total_degrees(g.get_vertices()).astype(np.int64)
    E = g.get_edges()
    s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64)

    def role_counts(qq):
        # per node: # in/out neighbours that are Q_lo / Q_hi  (vectorised over edges)
        n_lo_in  = np.bincount(t[qq[s] == lo], minlength=N)
        n_hi_in  = np.bincount(t[qq[s] == hi], minlength=N)
        n_lo_out = np.bincount(s[qq[t] == lo], minlength=N)
        n_hi_out = np.bincount(s[qq[t] == hi], minlength=N)
        p_fwd = n_lo_in * n_hi_out          # Q_lo -> b -> Q_hi
        p_bwd = n_hi_in * n_lo_out          # Q_hi -> b -> Q_lo
        out = {}
        for k in mids:                      # middle broker -> liaison
            m = (qq == k)
            out[("liaison", k)] = int((p_fwd[m] + p_bwd[m]).sum())
        m_hi = (qq == hi)                   # rich broker
        out[("gatekeeper", hi)]     = int(p_fwd[m_hi].sum())   # Q_lo->Q_hi->Q_hi  (gate inflow)
        out[("representative", hi)] = int(p_bwd[m_hi].sum())   # Q_hi->Q_hi->Q_lo  (gate outflow)
        m_lo = (qq == lo)                   # poor broker (mirror)
        out[("representative", lo)] = int(p_fwd[m_lo].sum())   # Q_lo->Q_lo->Q_hi
        out[("gatekeeper", lo)]     = int(p_bwd[m_lo].sum())   # Q_hi->Q_lo->Q_lo
        return out

    obs = role_counts(q); keys = list(obs.keys())
    permute, _ = _degree_bin_permuter(deg, q > 0, n_bins=n_bins, seed=seed)
    null = {key: np.empty(n_perm) for key in keys}; base = q.copy()
    for j in range(n_perm):
        rc = role_counts(permute(base))
        for key in keys:
            null[key][j] = rc[key]
    rows = []
    for (role, k) in keys:
        mu, sd = null[(role, k)].mean(), null[(role, k)].std(ddof=1)
        rows.append(dict(role=role, broker=f"Q{k}", observed=obs[(role, k)],
                         null_mean=float(mu), ratio=obs[(role, k)]/mu if mu > 0 else np.nan,
                         z=float((obs[(role, k)]-mu)/sd) if sd > 0 else np.nan))
    df = pd.DataFrame(rows)
    liaison_tot = int(df.loc[df.role == "liaison", "observed"].sum())
    bound_tot   = int(df.loc[df.role.isin(["gatekeeper", "representative"]), "observed"].sum())
    tot = liaison_tot + bound_tot
    # bridging share: of all pole-connecting 2-path mediation, fraction that is genuine
    # middle-liaison (vs pole boundary-control). Low value = closure-dominated.
    bridging_share = liaison_tot / tot if tot > 0 else np.nan
    return df, dict(liaison_total=liaison_tot, boundary_total=bound_tot,
                    bridging_share=bridging_share)

# ------------------------------------------------------------- composition scan (cheap)
def composition_scan(data_dir, pattern="grafo_*.gt", vp="quintile_nation",
                     thin_pole=100, out="composition_scan.csv", verbose=True):
    """Cheap, quintile-only pass over every city: counts, income composition, pole health.
    First pass parses+caches (slow for big uncached cities); re-runs read .npy only.
    Produces the cross-city composition axis (income_centroid) that becomes the x-axis of
    the mechanism/scaling figures, and flags cities whose scarce pole is too thin to trust."""
    import glob
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if verbose: print(f"scanning {len(paths)} cities ...")
    rows = []
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0].replace("grafo_", "", 1)
        try:
            q = load_quintiles(p, vp=vp)
        except Exception as e:
            print(f"  FAIL {name}: {e!r}"); continue
        n = {k: int((q == k).sum()) for k in (1, 2, 3, 4, 5)}
        tot = sum(n.values())
        if tot == 0:
            continue
        min_pole = min(n[1], n[5])
        rows.append(dict(
            city=name, n_nodes=int(q.size), n_missing=int((q == 0).sum()),
            **{f"n_Q{k}": n[k] for k in (1, 2, 3, 4, 5)},
            income_centroid=sum(k * n[k] for k in (1, 2, 3, 4, 5)) / tot,   # 1(poor)..5(rich)
            pole_ratio_log10=(np.log10(n[1] / n[5]) if n[5] > 0 else np.inf),
            min_pole=min_pole, thin_pole=bool(min_pole < thin_pole)))
        if verbose and (i % 25 == 0 or i == len(paths) - 1):
            print(f"  [{i+1}/{len(paths)}] {name}")
    df = pd.DataFrame(rows).sort_values("income_centroid").reset_index(drop=True)
    df.to_csv(out, index=False)
    if verbose:
        print(f"\nwrote {out}  ({len(df)} cities, {int(df.thin_pole.sum())} flagged thin-pole)")
        print(f"income_centroid range: {df.income_centroid.min():.2f} (poorest) "
              f"-> {df.income_centroid.max():.2f} (wealthiest)")
    return df

def suggest_examples(df, k=5, min_nodes=50000):
    """Pick ~k example cities spanning the income-composition axis, among cities with a
    usable scarce pole and enough size. Returns candidates at evenly spaced quantiles of
    income_centroid (poorest ... wealthiest), preferring larger cities at each."""
    ok = df[(~df.thin_pole) & (df.n_nodes >= min_nodes)].copy()
    if ok.empty:
        print("no cities pass the size/pole filters; relax min_nodes or thin_pole."); return ok
    qs = np.linspace(0, 1, k)
    targets = ok.income_centroid.quantile(qs).values
    picks = []
    for tgt in targets:
        cand = ok.iloc[(ok.income_centroid - tgt).abs().argsort()[:3]]   # 3 nearest in comp
        pick = cand.sort_values("n_nodes", ascending=False).iloc[0]       # largest of them
        if pick.city not in [p.city for p in picks]:
            picks.append(pick)
    out = pd.DataFrame(picks)[["city", "n_nodes", "n_Q1", "n_Q5",
                               "income_centroid", "pole_ratio_log10", "min_pole"]]
    print("Suggested example cities (poorest -> wealthiest composition):")
    print(out.round(2).to_string(index=False))
    return out

def _precache_one(path, vp="quintile_nation"):
    """Picklable worker: parse + cache one city's quintiles; returns (city, status)."""
    try:
        q = load_quintiles(path, vp=vp)
        return (os.path.basename(path), f"ok ({q.size} nodes)")
    except Exception as e:
        return (os.path.basename(path), f"FAIL: {e!r}")

def precache(data_dir, pattern="grafo_*.gt", vp="quintile_nation", workers=8):
    """One-time parallel pre-parse of quintiles, so the scan and the sweep then read .npy
    only. Separate processes (the string parse is GIL-bound). Largest files first => no
    idle tail at the end. Safe to re-run: already-cached cities are skipped."""
    import glob
    from functools import partial
    from multiprocessing import Pool
    paths = [p for p in glob.glob(os.path.join(data_dir, pattern))
             if not os.path.exists(f"{p}.{vp}.npy")]
    paths.sort(key=os.path.getsize, reverse=True)        # biggest first
    if not paths:
        print("nothing to do: all cities already cached.")
        return
    print(f"pre-caching {len(paths)} cities with {workers} workers (largest first) ...")
    fails = []
    with Pool(workers, maxtasksperchild=4) as pool:      # recycle workers to release big graphs
        for i, (city, status) in enumerate(
                pool.imap_unordered(partial(_precache_one, vp=vp), paths), 1):
            if status.startswith("FAIL"):
                fails.append((city, status)); print(f"  [{i}/{len(paths)}] {city}: {status}")
            elif i % 20 == 0 or i == len(paths):
                print(f"  [{i}/{len(paths)}] {city}: {status}")
    print(f"done. {len(fails)} failure(s).")
    for c, s in fails:
        print("  ", c, s)

# ----------------------------------------------------------------------- per city
def run_city(path, vp="quintile_nation", weight_ep="q_calls", n_perm=500, n_bins=10,
             n_src=400, n_tgt=400, chunk=20000, seed=20270624, label=None, verbose=True):
    name = label or os.path.splitext(os.path.basename(path))[0].replace("grafo_", "", 1)
    t0 = time.time(); g, q = load_city(path, vp=vp)
    if verbose: print(f"[{name}] V={g.num_vertices()} E={g.num_edges()}  load {time.time()-t0:.0f}s")

    t = time.time()
    B = brokerage_profile(g, q, n_src=n_src, n_tgt=n_tgt, n_perm=n_perm, n_bins=n_bins, seed=seed)
    if verbose: print(f"  brokerage {time.time()-t:.0f}s")

    t = time.time(); A, strength = reciprocated_backbone(g, weight_ep=weight_ep)
    c = burt_constraint(A, chunk=chunk)
    C = constraint_profile(c, q, strength, n_perm=n_perm, n_bins=n_bins, seed=seed)
    if verbose: print(f"  constraint {time.time()-t:.0f}s (backbone n={C['n_backbone']})")

    t = time.time()
    GF, GFsum = gf_pole_role_census(g, q, n_perm=n_perm, n_bins=n_bins, seed=seed)
    if verbose: print(f"  G-F role census {time.time()-t:.0f}s")

    row = dict(city=name, quintile_ref=vp, n_nodes=g.num_vertices(), n_edges=g.num_edges(),
               n_backbone=C["n_backbone"],
               brok_peak=B["peak"], brok_centroid=B["centroid"], brok_no_positive=B["no_positive"],
               pairs_reached=B["pairs_reached"], pairs_tried=B["pairs_tried"],
               holes_peak=C["holes_peak"],
               coincide=(B["peak"] is not None and B["peak"] == C["holes_peak"]),
               gf_bridging_share=GFsum["bridging_share"],
               gf_liaison_total=GFsum["liaison_total"], gf_boundary_total=GFsum["boundary_total"])
    for k in (1, 2, 3, 4, 5):
        row[f"brok_z_Q{k}"] = B["z"][k]
        row[f"brok_share_Q{k}"] = B["share"][k]
        row[f"brok_share_null_mean_Q{k}"] = B["share_null_mean"][k]   # <-- add
        row[f"brok_share_null_lo_Q{k}"] = B["share_null_lo"][k]       # <-- add
        row[f"brok_share_null_hi_Q{k}"] = B["share_null_hi"][k]       # <-- add
        row[f"constraint_z_Q{k}"] = C["z"][k]
        row[f"constraint_mean_Q{k}"] = C["mean"][k]
        row[f"n_Q{k}"] = int((q == k).sum())
    gfa = GF.set_index(["role", "broker"])
    gz, gr = gfa["z"], gfa["ratio"]
    for k in (2, 3, 4):
        row[f"gf_liaison_z_Q{k}"] = float(gz.get(("liaison", f"Q{k}"), np.nan))
        row[f"gf_liaison_ratio_Q{k}"] = float(gr.get(("liaison", f"Q{k}"), np.nan))
    for role in ("gatekeeper", "representative"):
        for pole in ("Q5", "Q1"):
            row[f"gf_{role}_z_{pole}"] = float(gz.get((role, pole), np.nan))
            row[f"gf_{role}_ratio_{pole}"] = float(gr.get((role, pole), np.nan))
    # scale-free closure summaries: combined pole boundary mediation, observed/null.
    # Ratio (not z) is the cross-city comparable -- z is confounded with city size.
    def _closure_ratio(pole):
        sub = GF[(GF.broker == pole) & (GF.role.isin(["gatekeeper", "representative"]))]
        o, m = sub["observed"].sum(), sub["null_mean"].sum()
        return float(o / m) if m > 0 else np.nan
    row["q5_closure_ratio"] = _closure_ratio("Q5")   # >1 = rich gate the boundary more than chance
    row["q1_closure_ratio"] = _closure_ratio("Q1")

    if verbose:
        print(f"  brokerage peak=Q{B['peak']} centroid={B['centroid']:.2f} | "
              f"holes peak=Q{C['holes_peak']} | coincide={row['coincide']}")
        prof = pd.DataFrame({"broker_z": [B["z"][k] for k in (2,3,4)],
                             "constraint_z": [C["z"][k] for k in (2,3,4)]},
                            index=["Q2","Q3","Q4"]).round(2)
        print(prof, "\n  (broker_z high & constraint_z low at same stratum = coincidence)")
        print(f"\n  G-F pole-anchored role census (degree-null z):")
        print(GF.round(2).to_string(index=False))
        print(f"  bridging share = {GFsum['bridging_share']:.3f}  "
              f"(fraction of pole-connecting mediation that is genuine middle-liaison;\n"
              f"   low => closure-dominated: poles guard their own boundary)")
        print(f"  Q5 closure ratio = {row['q5_closure_ratio']:.2f}  "
              f"(Q1 closure ratio = {row['q1_closure_ratio']:.2f})  "
              f"-- scale-free; >1 = rich gate the rich<->poor boundary above chance\n")
    return row

# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["analyze", "scan", "precache"], default="analyze")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cities", nargs="+", help="city names (analyze mode)")
    ap.add_argument("--vp", default="quintile_nation")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.mode == "scan":
        df = composition_scan(a.data_dir, vp=a.vp, out=a.out or "composition_scan.csv")
        suggest_examples(df)
    elif a.mode == "precache":
        precache(a.data_dir, vp=a.vp, workers=a.workers)
    else:
        rows = []
        for city in (a.cities or []):
            p = os.path.join(a.data_dir, f"grafo_{city}.gt")
            if not os.path.exists(p):
                print("skip (not found):", p); continue
            try:
                rows.append(run_city(p, vp=a.vp, n_perm=a.n_perm, chunk=a.chunk, label=city))
            except Exception as e:
                print(f"  FAILED {city}: {e!r}")
        pd.DataFrame(rows).to_csv(a.out or "brokerage_constraint.csv", index=False)
        print("wrote", a.out or "brokerage_constraint.csv")