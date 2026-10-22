#!/usr/bin/env python3
"""
necessity_sweep.py -- Position Part 2: is a middle stratum LOAD-BEARING on the
pole-to-pole routes, beyond just being that many intermediaries?

For each city we measure the pole-restricted global efficiency
    E(R) = mean over sampled Q1->Q5 pairs of 1/d_{G\\R}(s,t),   1/inf := 0
on the observed graph (R = empty) and after removing a vertex set R, via a
graph-tool VERTEX FILTER (no graph copy). The necessity of middle stratum k is
    Delta_k = E(R_rand) - E(Q_k)
where R_rand is an equal-sized random sample of NON-POLE vertices, averaged over
draws. Delta_k > 0 means removing stratum k degrades cross-pole reach MORE than
removing that many arbitrary intermediaries -> the stratum is load-bearing. The
companion length statistic dlen_k = mean_len(Q_k) - mean_len(R_rand) reports the
same in geodesic-lengthening terms (poles rarely disconnect; distance grows).

Only MIDDLE strata (Q2,Q3,Q4) get a Delta: removing a pole is ill-defined (it is
an endpoint of the very paths we measure). Poles are always retained.

COST is real: each E(R) is n_src BFS traversals; per city we do
    1 (baseline) + 3 strata + 3*n_draws random  removals.
Run it on the POWERED/gated city list only, with modest sampling; use smaller
sampling for megacities via --big_nodes/--big_src. Resumable and failure-logging
like sweep_brokerage.py.

  python necessity_sweep.py --data_dir <dir> --module brokerage_constraint_pipeline \
      --cities_file powered_cities.txt --out necessity_br.csv

Validate the math with no graph-tool needed:
  python necessity_sweep.py --selftest_core
"""
import os, sys, glob, argparse, csv, gc, traceback
import numpy as np


# ============================================================ testable pure-python core
def _bfs_dist(adj, source, removed):
    """Unweighted directed BFS distances from `source`, skipping `removed` nodes."""
    from collections import deque
    dist = {source: 0}; dq = deque([source])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w in removed or w in dist:
                continue
            dist[w] = dist[u] + 1; dq.append(w)
    return dist


def _efficiency_core(adj, src, tgt, removed=frozenset()):
    """Pole-restricted efficiency and mean finite geodesic length over src x tgt.
    1/inf := 0 (unreached pairs contribute 0 to E). Mirrors the graph-tool path."""
    inv = 0.0; ln = 0.0; n_fin = 0; n_pairs = 0
    tgt = list(tgt)
    for s in src:
        dist = _bfs_dist(adj, s, removed)
        for t in tgt:
            n_pairs += 1
            d = dist.get(t)
            if d is not None and d > 0:
                inv += 1.0 / d; ln += d; n_fin += 1
    E = inv / n_pairs if n_pairs else 0.0
    mean_len = ln / n_fin if n_fin else float("inf")
    return E, mean_len, n_fin, n_pairs


def _necessity_core(adj, q, lo, hi, mids, n_nodes, n_draws=3, rng=None):
    """Delta_k and dlen_k for each middle stratum, on a pure-python adjacency.
    q: array of labels (0=missing). Poles never removed; controls drawn from
    non-pole vertices (label != lo and != hi)."""
    rng = rng or np.random.default_rng(0)
    src = [i for i in range(n_nodes) if q[i] == lo]
    tgt = [i for i in range(n_nodes) if q[i] == hi]
    nonpole = np.array([i for i in range(n_nodes) if q[i] != lo and q[i] != hi])
    E0, L0, nf0, npr = _efficiency_core(adj, src, tgt)
    out = {}
    for k in mids:
        rem_k = [i for i in range(n_nodes) if q[i] == k]
        nk = len(rem_k)
        if nk == 0:
            out[k] = dict(delta=np.nan); continue
        Ek, Lk, _, _ = _efficiency_core(adj, src, tgt, removed=frozenset(rem_k))
        Er, Lr = [], []
        for _ in range(n_draws):
            samp = rng.choice(nonpole, size=min(nk, nonpole.size), replace=False)
            e, l, _, _ = _efficiency_core(adj, src, tgt, removed=frozenset(samp.tolist()))
            Er.append(e); Lr.append(l)
        Er = np.array(Er)
        out[k] = dict(delta=float(np.mean(Er) - Ek),
                      delta_lo=float(np.min(Er) - Ek), delta_hi=float(np.max(Er) - Ek),
                      E0=E0, E_stratum=Ek, E_rand=float(np.mean(Er)),
                      dlen=float(Lk - np.mean(Lr)), n_removed=nk)
    return out, E0, L0, nf0


def _selftest_core():
    # 0(Q1) -> 1(Q2) -> 4(Q5) is the length-2 route; 0 -> 2(Q3) -> 3(Q4) -> 4 is length-3.
    # Removing Q2 (node 1) forces the longer detour (load-bearing); removing Q3 (node 2)
    # leaves the length-2 route intact (redundant).
    adj = {0: [1, 2], 1: [4], 2: [3], 3: [4], 4: []}
    q = np.array([1, 2, 3, 4, 5])  # nodes 0..4 -> Q1,Q2,Q3,Q4,Q5
    E0, L0, nf, npr = _efficiency_core(adj, [0], [4])
    assert abs(E0 - 0.5) < 1e-9 and abs(L0 - 2.0) < 1e-9, (E0, L0)  # 0->1->4
    # remove Q2 => only 0->2->3->4 (len 3): E=1/3
    Eq2, Lq2, _, _ = _efficiency_core(adj, [0], [4], removed=frozenset([1]))
    assert abs(Eq2 - 1/3) < 1e-9, Eq2
    # remove Q3 => 0->1->4 intact: E=1/2
    Eq3, _, _, _ = _efficiency_core(adj, [0], [4], removed=frozenset([2]))
    assert abs(Eq3 - 0.5) < 1e-9, Eq3
    out, *_ = _necessity_core(adj, q, 1, 5, (2, 3, 4), 5, n_draws=2,
                              rng=np.random.default_rng(0))
    # Q2 load-bearing (removing it hurts more than a random middle node on average):
    assert out[2]["E_stratum"] < out[2]["E_rand"] or out[2]["delta"] > 0
    # Q3 redundant here (removing it leaves E unchanged at 0.5):
    assert abs(out[3]["E_stratum"] - 0.5) < 1e-9
    print("selftest_core OK: efficiency, removal, and Delta signs verified")
    print(f"  baseline E={E0:.3f}  remove Q2 -> E={Eq2:.3f} (hurts)  "
          f"remove Q3 -> E={Eq3:.3f} (redundant)")


# ============================================================ graph-tool efficiency
def _reach(d):
    d = np.asarray(d, dtype=float)
    return (d > 0) & (d < 1e9)


def pole_efficiency_gt(g, src, tgt, keep_mask=None):
    """Efficiency + mean finite length over sampled Q1->Q5 pairs, optionally on the
    vertex-filtered subgraph (keep_mask True = vertex retained).

    NOTE: we build a fresh GraphView per call rather than calling
    g.set_vertex_filter(). Repeatedly setting/clearing a filter on the same graph
    object does NOT take effect after the first application -- the graph keeps the
    first filter, so every subsequent removal silently measures the first one
    (verified: num_vertices() stays frozen while the mask itself is correct).
    A GraphView creates a new filtered view each time and avoids this entirely.
    The property map must stay referenced for the lifetime of the view."""
    from graph_tool import GraphView
    from graph_tool.topology import shortest_distance
    if keep_mask is not None:
        vfilt = g.new_vertex_property("bool")
        vfilt.a = keep_mask
        gg = GraphView(g, vfilt=vfilt)          # fresh view; vfilt kept alive below
    else:
        vfilt = None
        gg = g
    inv = 0.0; ln = 0.0; n_fin = 0; n_pairs = 0
    for s in src:
        d = shortest_distance(gg, source=gg.vertex(int(s))).a
        dt = np.asarray(d, dtype=float)[tgt]     # d.a stays full length N (verified)
        fin = (dt > 0) & (dt < 1e9)
        inv += float(np.sum(1.0 / dt[fin])); ln += float(np.sum(dt[fin]))
        n_fin += int(fin.sum()); n_pairs += len(tgt)
    _ = vfilt                                    # keep reference until here
    E = inv / n_pairs if n_pairs else 0.0
    mean_len = ln / n_fin if n_fin else np.inf
    return E, mean_len, n_fin, n_pairs


def necessity_profile_gt(g, q, lo=1, hi=5, mids=(2, 3, 4), n_src=150, n_tgt=150,
                         n_draws=3, seed=20270624):
    """graph-tool necessity: mirrors _necessity_core but with BFS via shortest_distance
    and vertex-filter removals. Samples n_src/n_tgt poles for tractability."""
    rng = np.random.default_rng(seed)
    N = g.num_vertices()
    src = np.where(q == lo)[0]; tgt = np.where(q == hi)[0]
    if src.size > n_src: src = rng.choice(src, n_src, replace=False)
    if tgt.size > n_tgt: tgt = rng.choice(tgt, n_tgt, replace=False)
    tgt = tgt.astype(np.int64)
    nonpole = np.where((q != lo) & (q != hi))[0]
    E0, L0, nf0, npr = pole_efficiency_gt(g, src, tgt, keep_mask=None)
    rows = {}
    for k in mids:
        rem_k = np.where(q == k)[0]; nk = int(rem_k.size)
        if nk == 0:
            rows[k] = dict(delta=np.nan); continue
        keep = np.ones(N, bool); keep[rem_k] = False
        Ek, Lk, _, _ = pole_efficiency_gt(g, src, tgt, keep_mask=keep)
        Er, Lr = [], []
        for _ in range(n_draws):
            samp = rng.choice(nonpole, size=min(nk, nonpole.size), replace=False)
            keepr = np.ones(N, bool); keepr[samp] = False
            e, l, _, _ = pole_efficiency_gt(g, src, tgt, keep_mask=keepr)
            Er.append(e); Lr.append(l)
        Er = np.array(Er)
        rows[k] = dict(delta=float(np.mean(Er) - Ek),
                       delta_lo=float(np.min(Er) - Ek), delta_hi=float(np.max(Er) - Ek),
                       E0=E0, E_stratum=Ek, E_rand=float(np.mean(Er)),
                       dlen=float(Lk - np.mean(Lr)), n_removed=nk)
    return rows, dict(E0=E0, mean_len0=L0, pairs_reached=nf0, pairs_tried=npr)


# ============================================================ runner (resumable)
def _load_module(name):
    modname = name[:-3] if name.endswith(".py") else name
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    import importlib
    return importlib.import_module(modname)


def _done_cities(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["city"] for r in csv.DictReader(f) if r.get("city")}


def _append(path, rows, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush(); os.fsync(f.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest_core", action="store_true")
    ap.add_argument("--data_dir")
    ap.add_argument("--module", default="brokerage_constraint_pipeline")
    ap.add_argument("--cities", nargs="+")
    ap.add_argument("--cities_file", help="one city name per line (the powered set)")
    ap.add_argument("--vp", default="quintile_nation")
    ap.add_argument("--n_src", type=int, default=150)
    ap.add_argument("--n_tgt", type=int, default=150)
    ap.add_argument("--n_draws", type=int, default=3)
    ap.add_argument("--big_nodes", type=int, default=500000,
                    help="graphs above this many nodes use --big_src sampling")
    ap.add_argument("--big_src", type=int, default=60)
    ap.add_argument("--out", default="necessity.csv")
    a = ap.parse_args()

    if a.selftest_core:
        _selftest_core(); sys.exit(0)
    if not a.data_dir:
        sys.exit("need --data_dir (or --selftest_core)")

    cities = list(a.cities or [])
    if a.cities_file:
        cities += [ln.strip() for ln in open(a.cities_file) if ln.strip()]
    if not cities:
        cities = [os.path.basename(p)[6:-3] for p in
                  sorted(glob.glob(os.path.join(a.data_dir, "grafo_*.gt")))]
    mod = _load_module(a.module)
    done = _done_cities(a.out); fail_log = a.out + ".failures.log"
    if done:
        print(f"resuming: {len(done)} cities already in {a.out}")
    header = ["city", "stratum", "delta", "delta_lo", "delta_hi", "dlen",
              "E0", "E_stratum", "E_rand", "n_removed", "pairs_reached_nec"]
    for i, city in enumerate(cities, 1):
        if city in done:
            continue
        p = os.path.join(a.data_dir, f"grafo_{city}.gt")
        if not os.path.exists(p):
            print(f"  [{i}/{len(cities)}] skip (not found): {city}"); continue
        try:
            g, q = mod.load_city(p, vp=a.vp)
            ns = a.big_src if g.num_vertices() > a.big_nodes else a.n_src
            print(f"  [{i}/{len(cities)}] {city} V={g.num_vertices()} n_src={ns} ...", flush=True)
            prof, meta = necessity_profile_gt(g, q, n_src=ns, n_tgt=ns, n_draws=a.n_draws)
            rows = []
            for k in (2, 3, 4):
                r = prof.get(k, dict(delta=np.nan))
                rows.append(dict(city=city, stratum=k,
                                 delta=r.get("delta", np.nan), delta_lo=r.get("delta_lo", np.nan),
                                 delta_hi=r.get("delta_hi", np.nan), dlen=r.get("dlen", np.nan),
                                 E0=r.get("E0", meta["E0"]), E_stratum=r.get("E_stratum", np.nan),
                                 E_rand=r.get("E_rand", np.nan), n_removed=r.get("n_removed", 0),
                                 pairs_reached_nec=meta["pairs_reached"]))
            _append(a.out, rows, header)
            del g, q; gc.collect()
        except Exception as e:
            with open(fail_log, "a") as f:
                f.write(f"{city}: {e!r}\n{traceback.format_exc()}\n")
            print(f"      FAILED {city}: {e!r} (logged, continuing)")
    print(f"done -> {a.out}" + (f"  (failures in {fail_log})" if os.path.exists(fail_log) else ""))
    