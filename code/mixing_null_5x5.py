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

Return paths. In the same-endpoint cells (i == j) the product in_i(b)*out_i(b) also
counts a -> b -> a whenever the tie a <-> b is reciprocated. These are not brokerage.
The output carries ret_obs (their observed count) and ret_null (their expectation
under the null), so gf_roles.py --exclude_returns can remove them from both sides.

Reciprocated backbone. On a fully reciprocated graph the rewiring runs on the
undirected version (mixing_null.undirected_version), so every null tie stays mutual,
and ret_null == ret_obs exactly. The column symmetric_null records which null ran.

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
    return tensor_from_arrays(src, tgt, w, N, q)


def tensor_from_arrays(src, tgt, w, N, q):
    """brokerage_tensor_product on arc arrays (multiplicities counted)."""
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


def returns_matrix(g, quintile_prop):
    """ret[i, m] = number of RETURN two-paths a -> b -> a with broker b in stratum m
    and a in stratum i, i.e. the number of reciprocated arcs b -> a from m to i.
    These are counted inside B[i, i, m] by brokerage_tensor_product but are not
    brokerage (Gould-Fernandez require two distinct endpoints)."""
    E = g.get_edges()
    N = int(g.num_vertices())
    s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64)
    ok = s != t
    s, t = s[ok], t[ok]
    if not g.is_directed():                       # every undirected edge is mutual
        s, t = np.concatenate([s, t]), np.concatenate([t, s])
    fwd = np.unique(s * N + t)
    rev = (fwd % N) * N + fwd // N
    i = np.searchsorted(fwd, rev); i[i == fwd.size] = 0
    mutual = fwd[fwd[i] == rev]                   # arcs whose reverse exists
    return returns_from_mutual(mutual, N, mn._quintile_array(g, quintile_prop))


def returns_from_mutual(mutual, N, q):
    """ret[i, m] from the codes (b*N + a) of the reciprocated arcs b -> a."""
    qb, qa = q[mutual // N], q[mutual % N]        # broker b = source, a = target
    ok = (qb >= 1) & (qb <= 5) & (qa >= 1) & (qa <= 5)
    ret = np.zeros((6, 6))
    np.add.at(ret, (qa[ok], qb[ok]), 1.0)
    return ret


def returns_null_analytic(g, quintile_prop):
    """Expected ret[i, m] under the DIRECTED mixing null (independent in- and
    out-stubs), where reciprocity is not preserved:
        E = e[m,i] e[i,m] S_m S_i / (Kout_m Kin_i Kout_i Kin_m),  S_x = sum kout*kin,
    with the b == a terms removed when i == m. Small in sparse graphs; it lets the
    returns be removed from an existing directed run without re-running it."""
    src, tgt, _, N = mn._edge_arrays(g, None)
    q = mn._quintile_array(g, quintile_prop)
    kout = np.bincount(src, minlength=N).astype(float)
    kin = np.bincount(tgt, minlength=N).astype(float)
    e = np.zeros((6, 6))
    ok = (q[src] >= 1) & (q[src] <= 5) & (q[tgt] >= 1) & (q[tgt] <= 5)
    np.add.at(e, (q[src][ok], q[tgt][ok]), 1.0)
    out = np.zeros((6, 6))
    for x in range(1, 6):
        for y in range(1, 6):
            mx, my = q == x, q == y          # x = broker stratum m, y = endpoint stratum i
            ko_m, ki_m = kout[mx].sum(), kin[mx].sum()
            ko_i, ki_i = kout[my].sum(), kin[my].sum()
            if min(ko_m, ki_m, ko_i, ki_i) <= 0:
                continue
            Sm = float((kout[mx] * kin[mx]).sum()); Si = float((kout[my] * kin[my]).sum())
            pair = Sm * Si
            if x == y:
                pair -= float(((kout[mx] * kin[mx]) ** 2).sum())
            out[y, x] = e[x, y] * e[y, x] * pair / (ko_m * ki_i * ko_i * ki_m)
    return out


def r5x5_one_city_reciprocal(g, quintile_prop, n_samples=200, n_iter=1, burn_in=10,
                             seed=None, returns_every=10):
    """r5x5_one_city under the reciprocity-preserving null (mixing_null.
    reciprocity_preserving_null): mutual ties rewired as undirected edges and
    one-way calls as directed arcs, each with the strata as blocks. Return paths in
    the null are counted exactly, with multiplicities, on every `returns_every`-th
    sample and averaged; they come almost entirely from the mutual part, whose
    block-pair counts the null holds fixed, so the average is very stable."""
    try:
        from graph_tool import seed_rng
        if seed is not None:
            seed_rng(int(seed))
    except Exception:
        pass
    B_obs = brokerage_tensor_product(g, quintile_prop, None)
    ret_obs = returns_matrix(g, quintile_prop)
    M, A, name = mn.split_reciprocal(g, quintile_prop)
    N = int(g.num_vertices())
    q = mn._quintile_array(g, quintile_prop)
    mn.rewire_parts(M, A, name, burn_in)
    acc = np.zeros((6, 6, 6)); acc2 = np.zeros((6, 6, 6)); accR = np.zeros((6, 6)); nR = 0
    diag = []
    for k in range(n_samples):
        mn.rewire_parts(M, A, name, n_iter)
        src, tgt = mn.union_arrays(M, A)
        Bs = tensor_from_arrays(src, tgt, np.ones(src.size), N, q)
        acc += Bs; acc2 += Bs * Bs
        if k % max(int(returns_every), 1) == 0:
            accR += mn.returns_by_strata_multi(src, tgt, N, q); nR += 1
        if k < 10:
            diag.append(mn.union_diagnostics(src, tgt, N, 2 * M.num_edges()))
    null_mean = acc / n_samples
    null_sd = np.sqrt(np.maximum(acc2 / n_samples - null_mean**2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(null_mean > 0, B_obs / null_mean, np.nan)
    d = np.array(diag)
    return (B_obs, null_mean, null_sd, R, ret_obs, accR / max(nR, 1),
            dict(null_model="reciprocal", overlap_frac=float(d[:, 0].mean()),
                 recip_drift=float(d[:, 1].mean())))


def r5x5_one_city(g, quintile_prop, n_samples=200, n_iter=1, burn_in=10,
                  weight_ep=None, seed=None, symmetric=None):
    """Observed and null-mean 5x5x5 tensors, and R_mix, using the identical
    constrained-configuration rewiring as the pole null.

    On a fully reciprocated graph (the reciprocated backbone) the rewiring runs on
    its undirected version so every null tie stays mutual, as in mixing_null.
    Also returns ret_obs and ret_null (6x6, [i, m]): the a->b->a return paths
    inside cell (i, i, m), observed and expected under the null. Under the
    symmetric null ret_null == ret_obs exactly (the block-pair counts are fixed);
    under the directed null ret_null is the analytic expectation."""
    from graph_tool.generation import random_rewire
    try:
        from graph_tool import seed_rng
        if seed is not None:
            seed_rng(int(seed))
    except Exception:
        pass
    if symmetric is None:
        symmetric = mn.is_symmetric(g)
    if not symmetric and not g.is_directed():
        raise ValueError("expects a directed graph")
    if symmetric and weight_ep is not None:
        raise ValueError("the symmetric null is defined for the unweighted statistic only")

    B_obs = brokerage_tensor_product(g, quintile_prop, weight_ep)
    ret_obs = returns_matrix(g, quintile_prop)

    if symmetric:
        g2, qname = mn.undirected_version(g, quintile_prop)
        bm = g2.vp[qname]
        ret_null = ret_obs.copy()
    else:
        g2 = g.copy()
        qname = quintile_prop
        bm = g2.new_vertex_property("int")
        bm.a = mn._quintile_array(g2, quintile_prop)      # full 1..5 blocks == pole null
        ret_null = returns_null_analytic(g, quintile_prop)
    rw = dict(model="constrained-configuration", block_membership=bm,
              edge_sweep=True, parallel_edges=False, self_loops=False)

    if burn_in > 0:
        random_rewire(g2, n_iter=int(burn_in), **rw)

    acc = np.zeros((6, 6, 6)); acc2 = np.zeros((6, 6, 6))
    for _ in range(n_samples):
        random_rewire(g2, n_iter=int(n_iter), **rw)
        Bs = brokerage_tensor_product(g2, qname, weight_ep)
        acc += Bs; acc2 += Bs * Bs
    null_mean = acc / n_samples
    null_sd = np.sqrt(np.maximum(acc2 / n_samples - null_mean**2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(null_mean > 0, B_obs / null_mean, np.nan)
    return B_obs, null_mean, null_sd, R, ret_obs, ret_null, bool(symmetric)


def tidy_rows(city, B_obs, null_mean, null_sd, R, min_capable=5000, n_nodes=None, n_edges=None,
              ret_obs=None, ret_null=None, symmetric=None, extra=None):
    rows = []
    for i in range(1, 6):
        for j in range(1, 6):
            for m in range(1, 6):
                r = dict(
                    city=city, i=i, j=j, m=m,
                    B_obs=float(B_obs[i, j, m]),
                    null_mean=float(null_mean[i, j, m]),
                    null_sd=float(null_sd[i, j, m]),
                    R_mix=float(R[i, j, m]),
                    testable=int(null_mean[i, j, m] >= min_capable),
                    n_nodes=n_nodes, n_edges=n_edges,
                )
                if ret_obs is not None:            # returns live only in same-endpoint cells
                    r["ret_obs"] = float(ret_obs[i, m]) if i == j else 0.0
                    r["ret_null"] = float(ret_null[i, m]) if i == j else 0.0
                    r["symmetric_null"] = bool(symmetric)
                if extra:
                    r.update(extra)
                rows.append(r)
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
    ap.add_argument("--reciprocal", action="store_true",
                    help="reciprocity-preserving null (mutual ties undirected, one-way calls directed)")
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
        if a.reciprocal:
            B_obs, nm, sd, R, ret_o, ret_n, extra = r5x5_one_city_reciprocal(
                g, a.vp, a.n_samples, a.n_iter, a.burn_in, seed=a.seed)
            sym = False
        else:
            B_obs, nm, sd, R, ret_o, ret_n, sym = r5x5_one_city(g, a.vp, a.n_samples, a.n_iter,
                                                                a.burn_in, seed=a.seed)
            extra = None
        rows = tidy_rows(city, B_obs, nm, sd, R, a.min_capable,
                         n_nodes=int(g.num_vertices()), n_edges=int(g.num_edges()),
                         ret_obs=ret_o, ret_null=ret_n, symmetric=sym, extra=extra)
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