#!/usr/bin/env python3
"""
make_backbone.py -- build the RECIPROCATED BACKBONE of each city graph, and measure
how much of the original same-group brokerage is a -> b -> a return paths.

Backbone. An arc u -> v is kept only when v -> u also exists (u != v). Vertices left
without any mutual tie are dropped. All vertex properties (the quintile labels) and
edge properties are carried over unchanged, and the file keeps its name, so every
tier-2 script runs on the backbone by pointing --data_dir at the backbone folder.
The result is a symmetric directed graph. mixing_null.py and mixing_null_5x5.py
detect symmetry and then rewire the undirected version, so the null keeps every
tie mutual.

Diagnostic (on the ORIGINAL graph, per stratification). For each broker stratum m
and endpoint stratum i, the same-endpoint ladder cell B[i,i,m] counts the two-paths
a -> b -> c with a, c in i and b in m. When a -> b and b -> a both exist, the path
a -> b -> a is counted too, although it is not brokerage. ret_obs is that count;
ret_null is its expectation under the directed mixing null, which does not keep
reciprocity. gf_roles.py --returns ... --exclude_returns removes both from an
existing r5x5 file, giving the original ladder without returns and without
re-running it.

Outputs
  <out_dir>/grafo_<city>.gt     backbone graphs (written atomically)
  --stats   backbone_stats.csv  per city: reciprocity, retention, calls kept, ...
  --returns returns_diag.csv    per city x stratification x (i, m): B_same, ret_obs, ret_null

Usage (natri)
  python make_backbone.py --data_dir ../grafos_ciudades_dirigidos_v4 \\
      --out_dir ../grafos_backbone_v4 \\
      --cities_file powered_population.txt powered_antenna.txt \\
      --vps quintile_nation_population quintile_nation_antenna --jobs 6

Resumable: a city whose graph and part files exist is skipped. --dry_run lists the
cities and sizes and stops.
"""
import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def city_of(path):
    return os.path.splitext(os.path.basename(path))[0].replace("grafo_", "", 1)


def _parts(out_dir, path):
    safe = city_of(path).replace("/", "_")
    d = os.path.join(out_dir, "_parts")
    return os.path.join(d, f"{safe}.stats.csv"), os.path.join(d, f"{safe}.returns.csv")


def build_one(job):
    path, out_dir, vps, weight_ep, force = job
    import pandas as pd
    import graph_tool.all as gt
    import mixing_null as mn
    import mixing_null_5x5 as m5

    name = city_of(path)
    out = os.path.join(out_dir, os.path.basename(path))
    p_stats, p_ret = _parts(out_dir, path)
    if not force and all(os.path.exists(x) for x in (out, p_stats, p_ret)):
        return name, "skip (done)"
    t0 = time.time()
    try:
        g = gt.load_graph(path)
        N = int(g.num_vertices())
        E = g.get_edges([g.edge_index])
        s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64); eid = E[:, 2].astype(np.int64)
        loop = s == t
        s1, t1, e1 = s[~loop], t[~loop], eid[~loop]
        codes = s1 * N + t1
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        first = np.r_[True, cs[1:] != cs[:-1]] if cs.size else np.zeros(0, bool)
        uniq = cs[first]
        uniq_eid = e1[order][first]
        rev = (uniq % N) * N + uniq // N
        i = np.searchsorted(uniq, rev); i[i == uniq.size] = 0
        mutual = uniq[i] == rev
        mutual_codes = uniq[mutual]

        # ---- backbone ------------------------------------------------------------
        efilt = g.new_edge_property("bool")
        efilt.a[uniq_eid[mutual]] = True            # one arc per ordered mutual pair
        vkeep = np.zeros(N, bool)
        vkeep[mutual_codes // N] = True
        vfilt = g.new_vertex_property("bool")
        vfilt.a = vkeep
        bb = gt.Graph(gt.GraphView(g, vfilt=vfilt, efilt=efilt), prune=True)
        bb.gp["backbone"] = bb.new_graph_property("string", "reciprocated")
        if bb.num_edges() and not mn.is_symmetric(bb):
            raise RuntimeError("backbone is not symmetric -- refusing to write it")
        tmp = os.path.join(out_dir, f".tmp_{os.getpid()}_{name.replace('/', '_')}.gt")
        bb.save(tmp, fmt="gt")
        os.replace(tmp, out)

        # ---- stats ---------------------------------------------------------------
        kb = bb.get_out_degrees(bb.get_vertices()).astype(float)   # = undirected degree
        row = dict(city=name, n_nodes=N, n_arcs=int(E.shape[0]), n_selfloops=int(loop.sum()),
                   n_parallel=int(cs.size - uniq.size),
                   n_mutual_arcs=int(mutual.sum()), n_mutual_pairs=int(mutual.sum() // 2),
                   arc_reciprocity=float(mutual.mean()) if uniq.size else np.nan,
                   bb_nodes=int(vkeep.sum()), node_share=float(vkeep.mean()) if N else np.nan,
                   mean_outdeg_orig=float(uniq.size / N) if N else np.nan,
                   mean_deg_bb=float(kb.mean()) if kb.size else np.nan,
                   k_over_k2_bb=float(kb.mean() / np.mean(kb ** 2)) if kb.size else np.nan)
        if weight_ep and weight_ep in g.ep:
            w = np.asarray(g.ep[weight_ep].a, dtype=float)
            w_all = float(w[e1].sum())
            j = np.searchsorted(mutual_codes, codes); j[j == mutual_codes.size] = 0
            is_mut = mutual_codes[j] == codes if mutual_codes.size else np.zeros(codes.size, bool)
            row["calls_share_bb"] = float(w[e1][is_mut].sum() / w_all) if w_all > 0 else np.nan
        rrows = []
        for vp in vps:
            short = vp.replace("quintile_nation_", "")
            if vp not in g.vp:
                row[f"{short}_missing"] = True
                continue
            q = mn._quintile_array(g, vp)
            for k in range(1, 6):
                nk = int((q == k).sum())
                row[f"{short}_n_Q{k}"] = nk
                row[f"{short}_kept_Q{k}"] = float(vkeep[q == k].mean()) if nk else np.nan
            # return-path diagnostic on the ORIGINAL graph
            B = m5.brokerage_tensor_product(g, vp)
            ret = m5.returns_from_mutual(mutual_codes, N, q)
            rnull = m5.returns_null_analytic(g, vp)
            for ii in range(1, 6):
                for m in range(1, 6):
                    rrows.append(dict(city=name, vp=vp, i=ii, m=m, B_same=float(B[ii, ii, m]),
                                      ret_obs=float(ret[ii, m]), ret_null=float(rnull[ii, m])))
            same = sum(B[ii, ii, m] for ii in range(1, 6) for m in range(1, 6))
            row[f"{short}_returns_share"] = float(ret[1:, 1:].sum() / same) if same > 0 else np.nan
        row["seconds"] = round(time.time() - t0, 1)

        os.makedirs(os.path.dirname(p_stats), exist_ok=True)
        for df, p in ((pd.DataFrame([row]), p_stats), (pd.DataFrame(rrows), p_ret)):
            tmpp = p + f".{os.getpid()}.tmp"
            df.to_csv(tmpp, index=False)
            os.replace(tmpp, p)
        return name, (f"ok  V {N} -> {row['bb_nodes']}  arcs {row['n_arcs']} -> "
                      f"{row['n_mutual_arcs']}  reciprocity {row['arc_reciprocity']:.3f}  "
                      f"({row['seconds']:.0f}s)")
    except Exception as e:
        return name, f"FAIL {e!r}\n{traceback.format_exc()}"


def concat(out_dir, paths, target, which):
    import pandas as pd
    frames = []
    for p in paths:
        part = _parts(out_dir, p)[which]
        if os.path.exists(part):
            frames.append(pd.read_csv(part))
    if not frames:
        print(f"  nothing to write for {target}")
        return
    df = pd.concat(frames, ignore_index=True)
    tmp = target + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(f"wrote {target}  ({df.city.nunique()} cities, {len(df)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="original directed graphs")
    ap.add_argument("--out_dir", required=True, help="folder for the backbone graphs")
    ap.add_argument("--cities_file", nargs="*", default=[],
                    help="one or more lists (one city per line); union is built")
    ap.add_argument("--cities", nargs="*", default=[])
    ap.add_argument("--vps", nargs="+",
                    default=["quintile_nation_population", "quintile_nation_antenna"])
    ap.add_argument("--weight_ep", default="q_calls")
    ap.add_argument("--jobs", type=int, default=4, help="cities built in parallel")
    ap.add_argument("--stats", default="backbone_stats.csv")
    ap.add_argument("--returns", default="returns_diag.csv")
    ap.add_argument("--force", action="store_true", help="rebuild cities already done")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if os.path.realpath(a.out_dir) == os.path.realpath(a.data_dir):
        sys.exit("--out_dir must differ from --data_dir (backbones keep the same file names)")
    names = list(a.cities)
    for f in a.cities_file:
        names += [ln.strip() for ln in open(f, encoding="utf-8") if ln.strip()]
    if names:
        seen, paths, missing = set(), [], []
        for c in names:
            if c in seen:
                continue
            seen.add(c)
            p = os.path.join(a.data_dir, f"grafo_{c}.gt")
            (paths if os.path.exists(p) else missing).append(p)
        for p in missing:
            print(f"WARN no graph: {p}")
    else:
        paths = glob.glob(os.path.join(a.data_dir, "grafo_*.gt"))
    paths.sort(key=os.path.getsize, reverse=True)          # largest first: no idle tail
    print(f"{len(paths)} cities, {sum(map(os.path.getsize, paths)) / 2**30:.1f} GiB of .gt, "
          f"{a.jobs} in parallel")
    if a.dry_run:
        for p in paths:
            print(f"  {os.path.getsize(p) / 2**30:7.2f} GiB  {city_of(p)}")
        return
    os.makedirs(os.path.join(a.out_dir, "_parts"), exist_ok=True)

    jobs = [(p, a.out_dir, a.vps, a.weight_ep, a.force) for p in paths]
    n_fail = 0
    if a.jobs <= 1:
        results = map(build_one, jobs)
        for k, (name, msg) in enumerate(results, 1):
            n_fail += msg.startswith("FAIL")
            print(f"  [{k}/{len(jobs)}] {name}: {msg}", flush=True)
    else:
        from multiprocessing import get_context
        with get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
            for k, (name, msg) in enumerate(pool.imap_unordered(build_one, jobs), 1):
                n_fail += msg.startswith("FAIL")
                print(f"  [{k}/{len(jobs)}] {name}: {msg}", flush=True)
    concat(a.out_dir, paths, a.stats, 0)
    concat(a.out_dir, paths, a.returns, 1)
    if n_fail:
        print(f"{n_fail} cities FAILED (see messages above); re-run to retry them")
        sys.exit(1)


if __name__ == "__main__":
    main()
