#!/usr/bin/env python3
"""
calibrate_null.py -- does the mixing-null pipeline return 1.0 when the truth is 1.0?

Three tests, all on real city graphs, answering three referee objections at once.

TEST A -- CALIBRATION (is R_mix unbiased?)
  A graph rewired heavily under the constrained-configuration model IS a draw from the
  null ensemble, so its true enrichment is exactly 1 by construction. We run the full
  pipeline on such graphs and ask what it returns. If it returns ~1.00, the empirical
  R_mix ~ 0.81 is a fact about the data. If it returns ~0.81, the estimator is biased
  and the substantive claim collapses. There is no modelling assumption here -- ground
  truth is guaranteed by the same machinery used in the paper, with empirical observations.
  Also yields the null SPREAD of R_mix across replicates, hence the MINIMUM DETECTABLE
  EFFECT: the smallest departure from 1 the design could resolve.

TEST B -- SENSITIVITY (can it recover a real effect?)
  Starting from a null-ensemble graph, we plant brokerage concentration with biased
  double-edge swaps that preserve every degree and every entry of the 5x5 mixing matrix
  while increasing the coincidence of in-from-counter and out-to-pole ties on a
  designated subset of "specialist" brokers. Concentration is planted at several
  strengths; the pipeline should recover a monotone, above-unity signal.

TEST C -- SUBSAMPLING (does a ~20% operator sample push R_mix below 1 by itself?)
  Two-paths survive observation only if both legs do, so node subsampling attenuates
  two-path counts faster than edge counts. We subsample a null-ensemble graph (true
  R_mix = 1) at the operator's sampling rate and re-run. If R_mix stays ~1.0, sampling
  does not manufacture the sub-unity result. If it falls below 1, the headline null
  result is confounded with sampling and must be reinterpreted.

Usage
  python calibrate_null.py --graph ../grafos_.../grafo_Natal.gt \
      --vp quintile_nation_population --pole 5 --counter 1 \
      --n_replicates 20 --n_samples 300 --subsample_p 0.2 --out calib_natal.csv
"""
import argparse
import os

import numpy as np


# ----------------------------------------------------------------------
# statistic: pole boundary control, identical in form to the paper's
# ----------------------------------------------------------------------
def _edge_arrays(g):
    E = g.get_edges()
    return E[:, 0].astype(np.int64), E[:, 1].astype(np.int64), int(g.num_vertices())


def quintile_array(g, vp):
    """Map the stratum vertex property to int16 bins 1..5 (0 = unlabelled)."""
    raw = g.vp[vp]
    N = int(g.num_vertices())
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


def boundary_control(src, tgt, q, N, pole, counter):
    """beta^BC = sum over pole brokers of in_counter(b)*out_pole(b) + in_pole(b)*out_counter(b).
    Identical product form to the paper's pole statistic."""
    def in_from(k):
        m = q[src] == k
        return np.bincount(tgt[m], minlength=N).astype(float)

    def out_to(k):
        m = q[tgt] == k
        return np.bincount(src[m], minlength=N).astype(float)

    beta = in_from(counter) * out_to(pole) + in_from(pole) * out_to(counter)
    return float(beta[q == pole].sum())


def capable_count(src, tgt, q, N, pole, counter):
    """Number of boundary-capable brokers on the focal pole (the power gate quantity)."""
    def in_from(k):
        m = q[src] == k
        return np.bincount(tgt[m], minlength=N)

    def out_to(k):
        m = q[tgt] == k
        return np.bincount(src[m], minlength=N)

    cap = ((in_from(counter) > 0) & (out_to(pole) > 0)) | \
          ((in_from(pole) > 0) & (out_to(counter) > 0))
    return int(cap[q == pole].sum())


# ----------------------------------------------------------------------
# the null ensemble (identical call to the paper's)
# ----------------------------------------------------------------------
def rewire_inplace(g, qprop, n_iter):
    """Constrained-configuration rewiring with the 5 strata as blocks. Preserves every
    degree and every entry of the 5x5 mixing matrix. Same call as the paper's null."""
    from graph_tool.generation import random_rewire
    bm = g.new_vertex_property("int")
    bm.a = quintile_array(g, qprop)
    random_rewire(g, model="constrained-configuration", block_membership=bm,
                  edge_sweep=True, parallel_edges=False, self_loops=False,
                  n_iter=int(n_iter))
    return g


def r_mix_pole(g, qprop, pole, counter, n_samples, n_iter=1, burn_in=10):
    """Observed / null-mean pole boundary control, on whatever graph is handed in."""
    q = quintile_array(g, qprop)
    src, tgt, N = _edge_arrays(g)
    obs = boundary_control(src, tgt, q, N, pole, counter)

    g2 = g.copy()
    if burn_in > 0:
        rewire_inplace(g2, qprop, burn_in)
    vals = np.empty(n_samples)
    for s in range(n_samples):
        rewire_inplace(g2, qprop, n_iter)
        s2, t2, N2 = _edge_arrays(g2)
        vals[s] = boundary_control(s2, t2, quintile_array(g2, qprop), N2, pole, counter)
    nullmean = float(vals.mean())
    return (obs / nullmean if nullmean > 0 else np.nan), obs, nullmean, float(vals.std())


# ----------------------------------------------------------------------
# TEST B: plant concentration, preserving degree and mixing exactly
# ----------------------------------------------------------------------
def plant_concentration(g, qprop, pole, counter, frac_specialists=0.10,
                        n_swaps=20000, seed=0):
    """Plant brokerage concentration on a designated specialist subset of pole nodes,
    using ONLY moves the null itself permits.

    The statistic is in_counter(b) * out_pole(b) summed over pole brokers, so BOTH legs
    must be concentrated on the SAME nodes -- concentrating one alone barely moves the
    product. Two swap families do this while preserving every node's total in/out degree
    and every entry of the 5x5 mixing matrix:
      (i)  target-side: exchange targets between a (counter->pole) and an (other->pole)
           edge. Both targets are pole nodes, so in-degrees and the mixing matrix are
           untouched, but the counter-tie moves onto a specialist.
      (ii) source-side: exchange sources between a (pole->pole) and a (pole->other)
           edge. Both sources are pole nodes, so out-degrees and the mixing matrix are
           untouched, but the out-to-pole tie moves onto a specialist.
    Because the constrained-configuration null preserves total degree and block counts
    but NOT each node's stratum-specific degree, this concentration lives exactly in the
    space the null explores -- which is why a working estimator must detect it."""
    import graph_tool.all as gt
    rng = np.random.default_rng(seed)
    q = quintile_array(g, qprop)
    N = int(g.num_vertices())

    pole_nodes = np.where(q == pole)[0]
    if pole_nodes.size == 0:
        return g, np.array([], dtype=int)
    k = max(1, int(round(frac_specialists * pole_nodes.size)))
    specialists = rng.choice(pole_nodes, size=k, replace=False)
    spec = np.zeros(N, dtype=bool)
    spec[specialists] = True

    E = g.get_edges()
    src = E[:, 0].astype(np.int64).copy()
    tgt = E[:, 1].astype(np.int64).copy()

    cp = np.where((q[src] == counter) & (q[tgt] == pole))[0]   # counter -> pole
    op = np.where((q[src] != counter) & (q[tgt] == pole))[0]   # other   -> pole
    if cp.size > 1 and op.size > 1:
        for _ in range(int(n_swaps)):
            a = rng.choice(cp); b = rng.choice(op)
            va, vb = tgt[a], tgt[b]
            if va != vb and spec[vb] and not spec[va]:
                tgt[a], tgt[b] = vb, va

    pp = np.where((q[src] == pole) & (q[tgt] == pole))[0]      # pole -> pole
    po = np.where((q[src] == pole) & (q[tgt] != pole))[0]      # pole -> other
    if pp.size > 1 and po.size > 1:
        for _ in range(int(n_swaps)):
            a = rng.choice(pp); b = rng.choice(po)
            ua, ub = src[a], src[b]
            if ua != ub and spec[ub] and not spec[ua]:
                src[a], src[b] = ub, ua

    gp = gt.Graph(directed=True)
    gp.add_vertex(N)
    gp.add_edge_list(np.column_stack([src, tgt]))
    gp.vp[qprop] = g.vp[qprop].copy()
    return gp, specialists


# ----------------------------------------------------------------------
# TEST C: node subsampling (mimics a partial-market-share operator sample)
# ----------------------------------------------------------------------
def subsample_nodes(g, p, qprop, seed=0):
    """Keep a uniform random p-fraction of NODES and the edges among them, the structure
    of a single-operator sample. Two-paths survive only if both legs do, so this is the
    regime in which two-path statistics attenuate fastest."""
    import graph_tool.all as gt
    rng = np.random.default_rng(seed)
    N = int(g.num_vertices())
    keep = rng.random(N) < p
    vf = g.new_vertex_property("bool")
    vf.a = keep
    sub = gt.GraphView(g, vfilt=vf)
    out = gt.Graph(sub, prune=True)   # materialise so downstream indexing is dense
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--vp", default="quintile_nation_population")
    ap.add_argument("--pole", type=int, default=5)
    ap.add_argument("--counter", type=int, default=1)
    ap.add_argument("--n_replicates", type=int, default=20,
                    help="independent null-ensemble draws for Test A (gives the spread / MDE)")
    ap.add_argument("--n_samples", type=int, default=300, help="rewirings per R_mix estimate")
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--ensemble_burn", type=int, default=50,
                    help="sweeps used to draw the ground-truth null graph (Test A)")
    ap.add_argument("--plant_fracs", nargs="*", type=float, default=[0.0, 0.05, 0.10, 0.20],
                    help="specialist fractions for Test B")
    ap.add_argument("--subsample_p", nargs="*", type=float, default=[0.2, 0.5],
                    help="node-retention rates for Test C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import graph_tool.all as gt
    import pandas as pd

    city = os.path.basename(a.graph).replace("grafo_", "").replace(".gt", "")
    g0 = gt.load_graph(a.graph)
    q0 = quintile_array(g0, a.vp)
    s0, t0, N0 = _edge_arrays(g0)
    print(f"=== {city}: {N0} nodes, {g0.num_edges()} edges ===")
    print(f"    boundary-capable Q{a.pole} brokers (observed): "
          f"{capable_count(s0, t0, q0, N0, a.pole, a.counter)}")

    rows = []

    # ---------------- TEST A: calibration ----------------
    print(f"\n--- TEST A: calibration, {a.n_replicates} null-ensemble graphs "
          f"(true R_mix = 1 by construction) ---")
    vals = []
    for r in range(a.n_replicates):
        gnull = g0.copy()
        rewire_inplace(gnull, a.vp, a.ensemble_burn)      # now a draw from the null
        R, obs, nm, sd = r_mix_pole(gnull, a.vp, a.pole, a.counter,
                                    a.n_samples, burn_in=a.burn_in)
        vals.append(R)
        rows.append(dict(city=city, test="A_calibration", replicate=r,
                         planted=np.nan, subsample_p=np.nan,
                         R_mix=R, obs=obs, null_mean=nm, null_sd=sd))
        print(f"    replicate {r:2d}: R_mix = {R:.4f}")
    vals = np.array(vals, dtype=float)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"\n    mean R_mix under a TRUE null of 1.0 : {vals.mean():.4f}")
    print(f"    bias                                : {vals.mean()-1:+.4f}")
    print(f"    sd across replicates                : {vals.std():.4f}")
    print(f"    95% range                           : [{lo:.4f}, {hi:.4f}]")
    print(f"    => MINIMUM DETECTABLE EFFECT (2 sd) : {2*vals.std():.4f}")
    print(f"    => an observed R_mix of 0.81 sits {(1-0.81)/max(vals.std(),1e-9):.1f} "
          f"sd below the calibrated null")

    # ---------------- TEST B: sensitivity ----------------
    print(f"\n--- TEST B: recovery of planted concentration "
          f"(degree- and mixing-preserving) ---")
    for f in a.plant_fracs:
        gnull = g0.copy()
        rewire_inplace(gnull, a.vp, a.ensemble_burn)
        if f > 0:
            gp, spec = plant_concentration(gnull, a.vp, a.pole, a.counter,
                                           frac_specialists=f, seed=a.seed)
        else:
            gp = gnull
        R, obs, nm, sd = r_mix_pole(gp, a.vp, a.pole, a.counter,
                                    a.n_samples, burn_in=a.burn_in)
        rows.append(dict(city=city, test="B_sensitivity", replicate=0,
                         planted=f, subsample_p=np.nan,
                         R_mix=R, obs=obs, null_mean=nm, null_sd=sd))
        print(f"    specialist fraction {f:4.2f} -> recovered R_mix = {R:.4f}")

    # ---------------- TEST C: subsampling ----------------
    print(f"\n--- TEST C: does node subsampling bias R_mix downward? "
          f"(true R_mix = 1 before sampling) ---")
    print("    NOTE replicated: at low retention the surviving graph is small and a")
    print("    single draw is dominated by noise. Bias is the mean across replicates.")
    n_sub_rep = max(5, a.n_replicates // 2)
    for p in a.subsample_p:
        Rs = []
        for r in range(n_sub_rep):
            gnull = g0.copy()
            rewire_inplace(gnull, a.vp, a.ensemble_burn)
            gs = subsample_nodes(gnull, p, a.vp, seed=a.seed + r)
            R, obs, nm, sd = r_mix_pole(gs, a.vp, a.pole, a.counter,
                                        a.n_samples, burn_in=a.burn_in)
            if np.isfinite(R):
                Rs.append(R)
            rows.append(dict(city=city, test="C_subsample", replicate=r,
                             planted=np.nan, subsample_p=p,
                             R_mix=R, obs=obs, null_mean=nm, null_sd=sd))
        Rs = np.array(Rs, dtype=float)
        if Rs.size:
            se = Rs.std() / np.sqrt(Rs.size)
            verdict = "unbiased" if abs(Rs.mean() - 1) < 2 * se else f"BIASED {Rs.mean()-1:+.3f}"
            print(f"    retain {p:4.2f}: mean R_mix = {Rs.mean():.4f}  "
                  f"sd {Rs.std():.4f}  (n={Rs.size})  -> {verdict}")

    print("\n=== reading ===")
    print("  A: mean ~1.00 => the estimator is unbiased and the empirical 0.81 is real.")
    print("     mean ~0.81 => the estimator is biased and the headline result is an artefact.")
    print("  B: R_mix should rise above 1 with planted concentration -- the measure can")
    print("     detect closure when closure is present.")
    print("  C: if R_mix stays ~1.00 under subsampling, the operator sample does not")
    print("     manufacture sub-unity values. If it falls, the null result is confounded.")

    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
