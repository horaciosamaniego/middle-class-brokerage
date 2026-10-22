#!/usr/bin/env python3
"""
calibrate_null_fast.py -- same three tests as calibrate_null.py, restructured to run faster, and to calibrate the code path the paper actually uses.

It seeks to respond to the question: does the mixing-null pipeline return 1.0 when the truth is 1.0?

Three tests on real city graphs, answering three referee objections.

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

WHAT CHANGED from 'calibrate_null.py'
-------------------------------------
1. POLE SUBGRAPH, NOT THE FULL GRAPH.
   run_two_nulls.py computes R_mix on mn.pole_subgraph(...) with contiguous {0,1} block
   labels. The old calibration rewired the full graph, so it was ~17x slower AND was not
   calibrating the production path. We now do exactly what production does.

2. ONE SHARED NULL DENOMINATOR FOR TESTS A AND B.
   The null ensemble is fixed by (degree sequence, mixing matrix). Constrained-
   configuration rewiring preserves both, so EVERY Test-A replicate has the SAME null
   ensemble as g0 and therefore the SAME E[beta]. The planting in Test B is built from
   degree- and mixing-preserving swaps, so it shares that ensemble too (asserted at
   runtime). We therefore estimate E[beta] once, at high precision, and reuse it. Each
   replicate then costs only its observed statistic, which needs no rewiring at all.
   This is faster AND more precise than re-estimating a common quantity 24 times.

3. CHAIN REUSE INSTEAD OF RE-BURNING.
   The old code re-copied g0 and burned `ensemble_burn` sweeps for every replicate. Once
   the chain is mixed, successive draws a few sweeps apart are already null draws. We
   burn once, then advance `--draw_gap` sweeps per replicate.

4. ANALYTIC DENOMINATOR AVAILABLE (--analytic).
   mn.dcsbm_expected_bc gives E[beta] in closed form. The paper already audits swap vs
   analytic agreement per city (mix_consistent). With --analytic the denominator is free,
   so Tests A and B cost only chain advances. Default is the swap denominator, which is
   the honest production-path test; --analytic is for a fast large-n MDE scan.

5. TEST C REUSES NULL DRAWS ACROSS SUBSAMPLE SEEDS.
   Subsampling changes degrees and the mixing matrix, so each subsample genuinely needs
   its own denominator -- that part stays expensive. But the expensive full-graph burn is
   now shared: --sub_draws null draws x --sub_seeds subsample seeds each.

Usage
  DIR=../grafos_ciudades_dirigidos_v4
  python calibrate_null_fast.py --probe --graph $DIR/grafo_Natal.gt
  python calibrate_null_fast.py --graph $DIR/grafo_Natal.gt \
      --vp quintile_nation_population --pole 5 --counter 1 \
      --n_replicates 40 --n_samples 80 --subsample_p 0.2 0.5 --out calib_natal.csv
"""
import argparse
import os
import time

import numpy as np


# ----------------------------------------------------------------------
# statistic (identical in form to mixing_null.restricted_boundary_control)
# ----------------------------------------------------------------------
def _edge_arrays(g):
    E = g.get_edges()
    return (E[:, 0].astype(np.int64), E[:, 1].astype(np.int64), int(g.num_vertices()))


def boundary_control(src, tgt, q, N, pole, counter):
    """sum over brokers b in `pole` of in_counter(b)*out_pole(b) + in_pole(b)*out_counter(b)"""
    def in_c(X):
        m = q[src] == X
        return np.bincount(tgt[m], minlength=N)

    def out_c(X):
        m = q[tgt] == X
        return np.bincount(src[m], minlength=N)

    beta = in_c(counter) * out_c(pole) + in_c(pole) * out_c(counter)
    return float(beta[q == pole].sum())


def capable_count(src, tgt, q, N, pole, counter):
    in_c = np.bincount(tgt[q[src] == counter], minlength=N) > 0
    out_p = np.bincount(src[q[tgt] == pole], minlength=N) > 0
    in_p = np.bincount(tgt[q[src] == pole], minlength=N) > 0
    out_c = np.bincount(src[q[tgt] == counter], minlength=N) > 0
    m = q == pole
    return int(((in_c & out_p) | (in_p & out_c))[m].sum())


# ----------------------------------------------------------------------
# production-identical pole subgraph with contiguous block labels
# ----------------------------------------------------------------------
def make_pole_sub(g, vp, pole, counter, mn):
    """Exactly what run_two_nulls.py does: induced pole subgraph, then contiguous {0,1}
    block labels (constrained-configuration silently fails to preserve the mixing matrix
    with non-contiguous labels). Returns (sub, 'pole_block', pole=1, counter=0)."""
    sub = mn.pole_subgraph(g, vp, pole=pole, counter=counter)
    q = mn._quintile_array(sub, vp)
    pb = sub.new_vertex_property("int")
    pb.a = np.where(q == pole, 1, np.where(q == counter, 0, -1)).astype(np.int64)
    sub.vp["pole_block"] = pb
    return sub


def rewire(g, blockprop, n_iter):
    from graph_tool.generation import random_rewire
    bm = g.new_vertex_property("int")
    bm.a = np.asarray(g.vp[blockprop].a, dtype=np.int64)
    random_rewire(g, model="constrained-configuration", block_membership=bm,
                  edge_sweep=True, parallel_edges=False, self_loops=False,
                  n_iter=int(n_iter))
    return g


def obs_stat(g, blockprop):
    src, tgt, N = _edge_arrays(g)
    q = np.asarray(g.vp[blockprop].a, dtype=np.int64)
    return boundary_control(src, tgt, q, N, pole=1, counter=0)


def null_mean_swap(g, blockprop, n_samples, burn_in=10, n_iter=1, report=None):
    """High-precision E[beta] over the null ensemble of g. Chain is continued, not
    re-copied. Returns (mean, sd, samples)."""
    g2 = g.copy()
    if burn_in:
        rewire(g2, blockprop, burn_in)
    vals = np.empty(n_samples)
    for s in range(n_samples):
        rewire(g2, blockprop, n_iter)
        vals[s] = obs_stat(g2, blockprop)
        if report and (s + 1) % report == 0:
            print(f"      null sample {s+1}/{n_samples}", flush=True)
    return float(vals.mean()), float(vals.std()), vals


# ----------------------------------------------------------------------
# invariants: the shared-denominator trick depends on these holding
# ----------------------------------------------------------------------
def invariants(g, blockprop):
    src, tgt, N = _edge_arrays(g)
    q = np.asarray(g.vp[blockprop].a, dtype=np.int64)
    din = np.bincount(tgt, minlength=N)
    dout = np.bincount(src, minlength=N)
    M = np.zeros((3, 3), dtype=np.int64)       # blocks are {-1,0,1} -> index +1
    for a in (0, 1):
        sa = q[src] == a
        for b in (0, 1):
            M[a + 1, b + 1] = int((sa & (q[tgt] == b)).sum())
    return din, dout, M


def same_ensemble(ref, g, blockprop):
    din, dout, M = invariants(g, blockprop)
    return (np.array_equal(ref[0], din) and np.array_equal(ref[1], dout)
            and np.array_equal(ref[2], M))


# ----------------------------------------------------------------------
# TEST B planting: degree- and mixing-preserving concentration
# ----------------------------------------------------------------------
def plant(g, blockprop, frac_specialists, strength, rng):
    """Plant brokerage concentration on a random specialist subset of pole nodes, using
    only moves the null itself permits (every degree and every block count preserved).

    WHY THE EARLIER VERSION WAS TOO WEAK. It proposed random (counter->pole, pole->pole)
    edge pairs and accepted only when the receiver already pointed at a specialist and the
    donor did not -- an acceptance rate of f*(1-f) ~ 9%. 20,000 attempts therefore landed
    ~1,800 swaps where saturation needs ~f*M_pp, so the planted signal never cleared the
    noise floor. Here the pools are pre-filtered, so EVERY swap succeeds and strength=1
    saturates.

    WHAT "CONCENTRATION" HAS TO MEAN AGAINST A DEGREE-CORRECTED NULL. The null already
    expects hubs to broker more, so piling ties onto high-degree nodes beats nothing. To
    exceed the null a broker's counter-sourced SHARE of its in-edges and pole-targeted
    SHARE of its out-edges must both rise above the graph-wide baseline. We therefore
    drive the specialists' two shares toward 1:
      (i)  swap targets between a counter->pole edge on a NON-specialist and a pole->pole
           edge ON a specialist  -> the specialist's in-edge becomes counter-sourced
      (ii) swap sources between a pole->pole edge from a NON-specialist and a pole->counter
           edge FROM a specialist -> the specialist's out-edge becomes pole-targeted
    Both legs must move on the SAME nodes: beta sums in_counter(b)*out_pole(b), so lifting
    one leg alone barely shifts the product.

    Returns (planted_graph, diagnostics).
    """
    import graph_tool.all as gt
    src, tgt, N = _edge_arrays(g)
    src = src.copy(); tgt = tgt.copy()
    q = np.asarray(g.vp[blockprop].a, dtype=np.int64)
    pole = np.where(q == 1)[0]
    if pole.size < 4:
        return g, {}
    k = max(1, int(round(frac_specialists * pole.size)))
    spec = np.zeros(N, bool)
    spec[rng.choice(pole, size=k, replace=False)] = True

    def shares():
        """(counter-sourced share of specialists' in-edges, pole-targeted share of their
        out-edges, and the graph-wide baselines those must beat)."""
        din = np.bincount(tgt, minlength=N); dout = np.bincount(src, minlength=N)
        in_c = np.bincount(tgt[q[src] == 0], minlength=N)
        out_p = np.bincount(src[q[tgt] == 1], minlength=N)
        pm = q == 1
        base_in = in_c[pm].sum() / max(din[pm].sum(), 1)
        base_out = out_p[pm].sum() / max(dout[pm].sum(), 1)
        return (in_c[spec].sum() / max(din[spec].sum(), 1),
                out_p[spec].sum() / max(dout[spec].sum(), 1), base_in, base_out)

    si0, so0, base_in, base_out = shares()

    # (i) specialists' in-edges -> counter-sourced
    cp = np.where((q[src] == 0) & (q[tgt] == 1))[0]
    pp = np.where((q[src] == 1) & (q[tgt] == 1))[0]
    donor = cp[~spec[tgt[cp]]]          # counter tie sitting on a non-specialist
    recv = pp[spec[tgt[pp]]]            # specialist in-slot currently pole-sourced
    n1 = int(strength * min(donor.size, recv.size))
    if n1 > 0:
        A = rng.choice(donor, n1, replace=False)
        B = rng.choice(recv, n1, replace=False)
        ta, tb = tgt[A].copy(), tgt[B].copy()
        tgt[A], tgt[B] = tb, ta

    # (ii) specialists' out-edges -> pole-targeted
    pp2 = np.where((q[src] == 1) & (q[tgt] == 1))[0]
    pc = np.where((q[src] == 1) & (q[tgt] == 0))[0]
    donor2 = pp2[~spec[src[pp2]]]       # pole-targeted tie from a non-specialist
    recv2 = pc[spec[src[pc]]]           # specialist out-slot currently counter-targeted
    n2 = int(strength * min(donor2.size, recv2.size))
    if n2 > 0:
        A = rng.choice(donor2, n2, replace=False)
        B = rng.choice(recv2, n2, replace=False)
        sa, sb = src[A].copy(), src[B].copy()
        src[A], src[B] = sb, sa

    si1, so1, _, _ = shares()

    gp = gt.Graph(directed=True)
    gp.add_vertex(N)
    gp.add_edge_list(np.column_stack([src, tgt]))
    # a property map is bound to the graph it was created on, so g.vp[...].copy()
    # cannot be assigned to gp. Create the map on gp and copy the values across.
    pb = gp.new_vertex_property("int")
    pb.a = q
    gp.vp[blockprop] = pb

    # a fully saturated specialist contributes 1/(base_in*base_out) times its null
    # expectation, so this is the per-specialist ceiling the planting is aiming at
    ceiling = 1.0 / max(base_in * base_out, 1e-12)
    return gp, dict(n_specialists=k, swaps_in=n1, swaps_out=n2,
                    share_in_before=si0, share_in_after=si1,
                    share_out_before=so0, share_out_after=so1,
                    base_in=base_in, base_out=base_out, ceiling=ceiling)


def subsample(g, p, blockprop, rng):
    import graph_tool.all as gt
    N = int(g.num_vertices())
    vf = g.new_vertex_property("bool")
    vf.a = rng.random(N) < p
    out = gt.Graph(gt.GraphView(g, vfilt=vf), prune=True)
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--vp", default="quintile_nation_population")
    ap.add_argument("--pole", type=int, default=5)
    ap.add_argument("--counter", type=int, default=1)
    ap.add_argument("--probe", action="store_true",
                    help="report subgraph size and a timed runtime estimate, then exit")
    ap.add_argument("--n_replicates", type=int, default=40)
    ap.add_argument("--n_samples", type=int, default=80,
                    help="rewirings for the SHARED null denominator (was 300 per estimate)")
    ap.add_argument("--draw_gap", type=int, default=5,
                    help="sweeps between successive null draws (chain is reused)")
    ap.add_argument("--burn", type=int, default=30, help="initial burn-in sweeps, once")
    ap.add_argument("--full_replicates", type=int, default=5,
                    help="replicates that get their OWN independent denominator. The "
                         "shared denominator makes Test A's mean nearly tautological, so "
                         "these few end-to-end estimates are the honest unbiasedness "
                         "check. Keep small -- each costs a full null estimate.")
    ap.add_argument("--analytic", action="store_true",
                    help="use the closed-form DCSBM denominator (no rewiring at all)")
    ap.add_argument("--plant_fracs", nargs="*", type=float,
                    default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--plant_strength", type=float, default=1.0,
                    help="fraction of the eligible concentrating swaps to perform. "
                         "1.0 saturates as far as degree and mixing allow.")
    ap.add_argument("--subsample_p", nargs="*", type=float, default=[0.2, 0.5])
    ap.add_argument("--sub_draws", type=int, default=3, help="null draws per rate")
    ap.add_argument("--sub_seeds", type=int, default=3, help="subsample seeds per draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import graph_tool.all as gt
    import pandas as pd
    import mixing_null as mn

    rng = np.random.default_rng(a.seed)
    if not os.path.exists(a.graph):
        raise SystemExit(
            f"graph not found: {a.graph}\n"
            f"  (if the path contains '...', that was a placeholder -- give the real "
            f"directory, e.g. ../grafos_ciudades_dirigidos_v4/grafo_Natal.gt)")
    city = os.path.basename(a.graph).replace("grafo_", "").replace(".gt", "")
    t0 = time.time()
    g0 = gt.load_graph(a.graph)
    q0 = mn._quintile_array(g0, a.vp)
    s0, t0a, N0 = _edge_arrays(g0)
    cap = capable_count(s0, t0a, q0, N0, a.pole, a.counter)
    print(f"=== {city} ===")
    print(f"  full graph      : {N0:,} nodes, {g0.num_edges():,} edges "
          f"({time.time()-t0:.0f}s to load)")
    print(f"  boundary-capable Q{a.pole} brokers: {cap:,}")

    sub = make_pole_sub(g0, a.vp, a.pole, a.counter, mn)
    BP = "pole_block"
    print(f"  pole subgraph   : {sub.num_vertices():,} nodes, {sub.num_edges():,} edges "
          f"({g0.num_edges()/max(sub.num_edges(),1):.1f}x fewer edges)")

    # cross-check: the subgraph reproduces the full-graph statistic
    obs_full = boundary_control(s0, t0a, q0, N0, a.pole, a.counter)
    obs_sub = obs_stat(sub, BP)
    print(f"  observed beta   : full {obs_full:,.0f} | subgraph {obs_sub:,.0f} "
          f"-> {'MATCH' if abs(obs_full-obs_sub) < 1e-6 else 'MISMATCH (stop)'}")
    if abs(obs_full - obs_sub) > 1e-6:
        raise SystemExit("pole subgraph does not reproduce the statistic; aborting")

    # ---- probe: time one sweep, extrapolate, exit ----
    if a.probe:
        gp = sub.copy()
        tt = time.time(); rewire(gp, BP, 1); per = time.time() - tt
        n_sweeps = (a.burn + a.n_samples + a.n_replicates * a.draw_gap
                    + a.full_replicates * (a.draw_gap + 10 + a.n_samples)
                    + len(a.plant_fracs) * a.draw_gap
                    + len(a.subsample_p) * a.sub_draws
                      * (a.draw_gap + a.sub_seeds * (10 + a.n_samples)))
        print(f"\n  one subgraph sweep: {per:.2f}s")
        print(f"  planned sweeps    : ~{n_sweeps:,}")
        print(f"  ESTIMATED RUNTIME : ~{n_sweeps*per/60:.1f} min "
              f"({n_sweeps*per/3600:.1f} h)")
        print("\n  (the old script did ~9,640 sweeps of the FULL graph)")
        return

    rows = []

    # ---- the shared null denominator (Tests A and B) ----
    ref = invariants(sub, BP)
    if a.analytic:
        E_null = mn.dcsbm_expected_bc(sub, BP, pole=1, counter=0, weight_ep=None, exact=True)
        null_sd = float("nan")
        print(f"\n  shared denominator (analytic DCSBM): {E_null:,.1f}")
    else:
        tt = time.time()
        print(f"\n  estimating shared null denominator ({a.n_samples} rewirings) ...")
        E_null, null_sd, _ = null_mean_swap(sub, BP, a.n_samples, burn_in=a.burn,
                                            report=max(a.n_samples // 4, 1))
        analytic = mn.dcsbm_expected_bc(sub, BP, pole=1, counter=0, weight_ep=None, exact=True)
        rel = abs(E_null - analytic) / analytic if analytic > 0 else float("nan")
        print(f"  swap E[beta] = {E_null:,.1f} (sd {null_sd:,.1f}) | "
              f"analytic = {analytic:,.1f} | rel. diff {rel:.4f}  "
              f"[{time.time()-tt:.0f}s]")
        print(f"  observed R_mix on the REAL graph: {obs_sub/E_null:.4f}")

    # ---------------- TEST A ----------------
    print(f"\n--- TEST A: calibration, {a.n_replicates} null draws "
          f"(true R_mix = 1 by construction) ---")
    chain = sub.copy()
    rewire(chain, BP, a.burn)
    vals = []
    for r in range(a.n_replicates):
        rewire(chain, BP, a.draw_gap)
        if r == 0 and not same_ensemble(ref, chain, BP):
            raise SystemExit("rewiring did not preserve degrees/mixing; aborting")
        o = obs_stat(chain, BP)
        R = o / E_null
        vals.append(R)
        rows.append(dict(city=city, test="A_calibration", replicate=r, planted=np.nan,
                         subsample_p=np.nan, R_mix=R, obs=o, null_mean=E_null))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"  mean R_mix under a TRUE null of 1.0 : {vals.mean():.4f}")
    print(f"  bias                                : {vals.mean()-1:+.4f}")
    print(f"  sd across draws                     : {vals.std():.4f}")
    print(f"  95% range                           : [{lo:.4f}, {hi:.4f}]")
    print(f"  MINIMUM DETECTABLE EFFECT (2 sd)    : {2*vals.std():.4f}")
    if vals.std() > 0:
        print(f"  an observed R_mix of 0.81 sits        {(1-0.81)/vals.std():.1f} sd below")

    # ---- TEST A2: fully independent end-to-end replicates ----
    if a.full_replicates > 0:
        print(f"\n--- TEST A2: {a.full_replicates} replicates with their OWN denominator "
              f"(end-to-end, no shared quantity) ---")
        fr = []
        for r in range(a.full_replicates):
            rewire(chain, BP, a.draw_gap)
            gcopy = chain.copy()
            o = obs_stat(gcopy, BP)
            em, esd, _ = null_mean_swap(gcopy, BP, a.n_samples, burn_in=10)
            R = o / em if em > 0 else np.nan
            fr.append(R)
            rows.append(dict(city=city, test="A2_independent", replicate=r,
                             planted=np.nan, subsample_p=np.nan, R_mix=R,
                             obs=o, null_mean=em))
            print(f"  replicate {r}: R_mix = {R:.4f}")
        fr = np.array(fr, dtype=float)
        fr = fr[np.isfinite(fr)]
        if fr.size:
            print(f"  mean {fr.mean():.4f}   bias {fr.mean()-1:+.4f}   sd {fr.std():.4f}")
            print("  (this is the figure to quote for 'the estimator is unbiased')")

    # ---------------- TEST B ----------------
    print(f"\n--- TEST B: recovery of planted concentration ---")
    for f in a.plant_fracs:
        rewire(chain, BP, a.draw_gap)
        if f == 0:
            gp, dg = chain, {}
        else:
            gp, dg = plant(chain, BP, f, a.plant_strength, rng)
        okens = same_ensemble(ref, gp, BP)
        o = obs_stat(gp, BP)
        R = o / E_null
        rows.append(dict(city=city, test="B_sensitivity", replicate=0, planted=f,
                         subsample_p=np.nan, R_mix=R, obs=o, null_mean=E_null))
        print(f"  specialists {f:4.2f} -> R_mix = {R:.4f}   "
              f"[ensemble preserved: {okens}]")
        if dg:
            print(f"      swaps {dg['swaps_in']:,} in / {dg['swaps_out']:,} out  on "
                  f"{dg['n_specialists']:,} specialists")
            print(f"      counter-sourced share of their in-edges : "
                  f"{dg['share_in_before']:.3f} -> {dg['share_in_after']:.3f}  "
                  f"(baseline {dg['base_in']:.3f})")
            print(f"      pole-targeted share of their out-edges  : "
                  f"{dg['share_out_before']:.3f} -> {dg['share_out_after']:.3f}  "
                  f"(baseline {dg['base_out']:.3f})")
            print(f"      per-specialist ceiling vs null          : "
                  f"{dg['ceiling']:.2f}x")
    print("  (ensemble preserved = the planted graph shares g0's null, so the shared")
    print("   denominator is valid. If False, that row needs its own denominator.)")

    # ---------------- TEST C ----------------
    print(f"\n--- TEST C: does node subsampling bias R_mix downward? ---")
    for p in a.subsample_p:
        Rs = []
        for d in range(a.sub_draws):
            rewire(chain, BP, a.draw_gap)          # a fresh null draw
            for s in range(a.sub_seeds):
                gs = subsample(chain, p, BP, np.random.default_rng(a.seed + 1000 * d + s))
                if gs.num_edges() < 100:
                    continue
                o = obs_stat(gs, BP)
                em, esd, _ = null_mean_swap(gs, BP, a.n_samples, burn_in=10)
                R = o / em if em > 0 else np.nan
                if np.isfinite(R):
                    Rs.append(R)
                rows.append(dict(city=city, test="C_subsample", replicate=d * 10 + s,
                                 planted=np.nan, subsample_p=p, R_mix=R,
                                 obs=o, null_mean=em))
        Rs = np.array(Rs)
        if Rs.size:
            se = Rs.std() / np.sqrt(Rs.size)
            verdict = ("unbiased" if abs(Rs.mean() - 1) < 2 * se
                       else f"BIASED {Rs.mean()-1:+.3f}")
            print(f"  retain {p:4.2f}: mean R_mix = {Rs.mean():.4f}  sd {Rs.std():.4f}  "
                  f"(n={Rs.size})  -> {verdict}")

    print("\n=== reading ===")
    print("  A ~1.00 => estimator unbiased, the empirical 0.81 is a fact about the data.")
    print("  B rising above 1 => the measure detects closure when closure is present.")
    print("  C ~1.00 => a 20% operator sample does not manufacture sub-unity values.")

    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
    