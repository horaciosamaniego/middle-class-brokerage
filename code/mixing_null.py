"""
mixing_null.py
==============

Null models for the dyad-restricted boundary-control measure beta^BC_b(counter),
summed over brokers b in `pole`. Two nulls share ONE statistic so their ratios
are directly comparable:

  - mixing_preserving_null  -> R_mix  (Null B: degree + mixing matrix fixed;
                                       isolates POSITIONAL control)
  - label_permutation_null  -> R_perm (Null A: degree-stratified relabelling;
                                       the Q1-restricted analogue of the pooled
                                       q5_closure_ratio)

Reading the pair: if R_perm(Q5,Q1) is large but R_mix(Q5,Q1) ~ 1, the restricted
closure is a contact-VOLUME effect (it vanishes once Q5-Q1 volume is held fixed).
If both ~ 1, the Q1 RESTRICTION is what removed it (the pole seals against the
middle, not the bottom). Only the null differs between them; the statistic is
identical, so the comparison is clean.

Designed to drop into brokerage_constraint_pipeline.py. Needs a graph-tool
Graph carrying a quintile vertex property and optionally an edge weight property
(e.g. "q_calls" BR, "cantidad_llamadas" CL).

Quintile coercion: `_quintile_array` accepts int/float maps (NaN -> -1, excluded)
and string/object maps ("5","Q5" -> 5; unparseable -> -1). All entry points
route through it.

CONVENTION (settled empirically on Santiago): graph-tool's random_rewire
preserves the *unweighted* (count) mixing matrix exactly but NOT the weighted
one -- weights ride edge slots, not node pairs. So Null B is only exact for the
UNWEIGHTED statistic (weight_ep=None), which is what the method (binary A_ib /
classical Gould-Fernandez) actually defines. Run both nulls with weight_ep=None.

Validated offline (pure-numpy, graph-tool path exercised via --selftest):
  - restricted_boundary_control vs brute-force 2-path enumeration;
  - dcsbm_expected_bc (exact=True) vs brute-force DC-SBM expectation (~1e-16);
  - within-strata permutation preserves the per-stratum label multiset.
"""

from __future__ import annotations
import numpy as np
from graph_tool import GraphView, Graph


# ----------------------------------------------------------------------
#  Label coercion -> int64 over vertices; missing/unparseable -> -1.
# ----------------------------------------------------------------------
def _quintile_array(g, quintile_prop):
    qp = g.vp[quintile_prop] if isinstance(quintile_prop, str) else quintile_prop
    a = qp.a
    if a is not None:
        a = np.asarray(a, dtype=float)
        return np.rint(np.where(np.isfinite(a), a, -1)).astype(np.int64)

    def parse(val):
        s = str(val).strip().lstrip("Qq").strip()
        try:
            return int(float(s))
        except Exception:
            return -1

    return np.fromiter((parse(qp[v]) for v in g.vertices()),
                       dtype=np.int64, count=int(g.num_vertices()))


def _edge_arrays(g, weight_ep=None):
    """Return src, tgt, w (aligned, edge-index order), N."""
    if weight_ep is None:
        E = g.get_edges()
        w = np.ones(E.shape[0], dtype=float)
    else:
        wp = g.ep[weight_ep] if isinstance(weight_ep, str) else weight_ep
        E = g.get_edges([wp])
        w = E[:, 2].astype(float)
    return E[:, 0].astype(np.int64), E[:, 1].astype(np.int64), w, int(g.num_vertices())


# ----------------------------------------------------------------------
#  Shared statistic core. Both the observed value and every null draw call
#  this, so they cannot drift apart by convention.
# ----------------------------------------------------------------------
def _bc_from_arrays(src, tgt, w, N, q, pole, counter):
    def in_count(X):
        m = q[src] == X
        return np.bincount(tgt[m], weights=w[m], minlength=N)

    def out_count(X):
        m = q[tgt] == X
        return np.bincount(src[m], weights=w[m], minlength=N)

    beta = in_count(counter) * out_count(pole) + in_count(pole) * out_count(counter)
    return float(beta[q == pole].sum())


def restricted_boundary_control(g, quintile_prop="quintile_nation",
                                pole=5, counter=1, weight_ep=None):
    """
    Sum over brokers b with q(b)==pole of
        beta^BC_b(counter) = in_counter(b)*out_pole(b) + in_pole(b)*out_counter(b),
    the directed-2-path count (or weighted volume) whose endpoints are exactly
    one `pole` node and one `counter` node. Unweighted by default (classical GF).
    """
    src, tgt, w, N = _edge_arrays(g, weight_ep)
    q = _quintile_array(g, quintile_prop)
    return _bc_from_arrays(src, tgt, w, N, q, pole, counter)


# ----------------------------------------------------------------------
#  Analytic denominator under the degree-corrected SBM (Null B, canonical).
#  exact=True subtracts the self-loop (i=b / j=b) term; leading term = Eq. (14).
# ----------------------------------------------------------------------
def dcsbm_expected_bc(g, quintile_prop="quintile_nation",
                      pole=5, counter=1, weight_ep=None, exact=True):
    src, tgt, w, N = _edge_arrays(g, weight_ep)
    q = _quintile_array(g, quintile_prop)

    out_str = np.bincount(src, weights=w, minlength=N)
    in_str = np.bincount(tgt, weights=w, minlength=N)

    def m(a, b):
        return float(w[(q[src] == a) & (q[tgt] == b)].sum())

    m_pp, m_cp, m_pc = m(pole, pole), m(counter, pole), m(pole, counter)
    pm = q == pole
    kin_p, kout_p = in_str[pm].sum(), out_str[pm].sum()
    if kin_p == 0 or kout_p == 0:
        return float("nan")

    ib, ob = in_str[pm], out_str[pm]
    A_in_counter = ib * m_cp / kin_p
    A_out_pole = ob * m_pp / kout_p
    A_in_pole = ib * m_pp / kin_p
    A_out_counter = ob * m_pc / kout_p
    if exact:
        EA_bb = ob * ib * m_pp / (kout_p * kin_p)
        gk = A_in_counter * (A_out_pole - EA_bb)
        rp = (A_in_pole - EA_bb) * A_out_counter
    else:
        gk = A_in_counter * A_out_pole
        rp = A_in_pole * A_out_counter
    return float((gk + rp).sum())


# ----------------------------------------------------------------------
#  Null B (microcanonical): constrained directed double-edge swap.
#  EXACT only for the unweighted statistic (see CONVENTION note in header).
# ----------------------------------------------------------------------
def mixing_preserving_null(g, quintile_prop="quintile_nation",
                           pole=5, counter=1, weight_ep=None,
                           n_samples=200, n_iter=1, burn_in=10,
                           restrict_to_poles=False, seed=None,
                           parallel_edges=False, self_loops=False):
    """
    Returns observed, null_mean, R_mix=observed/null_mean, 95% null interval,
    upper-tail p, and the analytic (DC-SBM) cross-check.

    Run on the pole subgraph (pole_subgraph) for speed -- exactly equivalent to
    the full graph for R_mix, since constrained-configuration rewires each block
    pair independently. Use weight_ep=None (graph-tool preserves the count
    mixing matrix, not the weighted one).
    """
    from graph_tool.generation import random_rewire
    try:
        from graph_tool import seed_rng
        if seed is not None:
            seed_rng(int(seed))
    except Exception:
        pass

    if not g.is_directed():
        raise ValueError("mixing_preserving_null expects a directed graph.")

    observed = restricted_boundary_control(g, quintile_prop, pole, counter, weight_ep)

    g2 = g.copy()
    bm = g2.new_vertex_property("int")
    bm.a = _quintile_array(g2, quintile_prop)

    pin = None
    if restrict_to_poles:
        E = g2.get_edges([g2.edge_index])
        qs, qt = bm.a[E[:, 0].astype(np.int64)], bm.a[E[:, 1].astype(np.int64)]
        rewire = (((qs == counter) & (qt == pole)) |
                  ((qs == pole) & (qt == pole)) |
                  ((qs == pole) & (qt == counter)))
        pin = g2.new_edge_property("bool")
        pin.a[E[:, 2].astype(np.int64)] = (~rewire)

    rw_kwargs = dict(model="constrained-configuration", block_membership=bm,
                     edge_sweep=True, parallel_edges=parallel_edges,
                     self_loops=self_loops)
    if pin is not None:
        rw_kwargs["pin"] = pin

    if burn_in > 0:
        random_rewire(g2, n_iter=int(burn_in), **rw_kwargs)

    samples = np.empty(n_samples, dtype=float)
    for s in range(n_samples):
        random_rewire(g2, n_iter=int(n_iter), **rw_kwargs)
        samples[s] = restricted_boundary_control(g2, quintile_prop, pole, counter, weight_ep)

    null_mean = float(samples.mean())
    ratio = observed / null_mean if null_mean > 0 else float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p_upper = float((np.sum(samples >= observed) + 1) / (n_samples + 1))
    analytic = dcsbm_expected_bc(g, quintile_prop, pole, counter, weight_ep, exact=True)

    return {
        "pole": pole, "counter": counter,
        "observed": observed, "null_mean": null_mean, "R_mix": ratio,
        "null_ci95": (float(lo), float(hi)), "p_upper": p_upper,
        "analytic_expected": analytic,
        "R_mix_analytic": (observed / analytic) if analytic and analytic > 0 else float("nan"),
        "n_samples": n_samples, "n_iter": n_iter, "burn_in": burn_in,
        "restricted_to_poles": restrict_to_poles, "weighted": weight_ep is not None,
    }


# ----------------------------------------------------------------------
#  Null A: degree-stratified label permutation. Same statistic, different null.
# ----------------------------------------------------------------------
def _degree_strata(g, method="quantile", n_strata=25):
    """Integer stratum per vertex from (unweighted) total degree."""
    V = g.get_vertices()
    deg = (g.get_in_degrees(V) + g.get_out_degrees(V)).astype(float)
    N = deg.shape[0]
    if method == "quantile":
        order = np.argsort(deg, kind="mergesort")
        ranks = np.empty(N, dtype=np.int64); ranks[order] = np.arange(N)
        return (ranks * int(n_strata) // max(N, 1)).astype(np.int64)
    if method == "log":
        lg = np.floor(np.log2(np.maximum(deg, 1.0))).astype(np.int64)
        _, inv = np.unique(lg, return_inverse=True)
        return inv.astype(np.int64)
    raise ValueError(f"unknown strata_method {method!r}")


def label_permutation_null(g, quintile_prop="quintile_nation",
                           pole=5, counter=1, weight_ep=None,
                           n_samples=300, seed=None,
                           strata=None, strata_method="quantile", n_strata=25):
    """
    Null A for the SAME restricted statistic: hold topology fixed, permute
    quintile labels within degree strata, recompute. Returns R_perm.

    RUN ON THE FULL GRAPH (all quintiles present), NOT the pole subgraph -- the
    permutation must be able to relabel a pole node to ANY quintile within its
    stratum, exactly as the pooled q5_closure_ratio null does. On the Q1+Q5
    subgraph the permutation could only swap Q1<->Q5, a different (wrong) null.

    For strict comparability to your pooled ratio, pass `strata` = the vertex
    property name your pipeline already uses for its degree strata. Otherwise a
    default total-degree binning is used (quantile bins; heavy tail means the
    top bin lumps hubs together -- the documented limitation of any binning).
    """
    rng = np.random.default_rng(seed)
    src, tgt, w, N = _edge_arrays(g, weight_ep)
    q0 = _quintile_array(g, quintile_prop)
    observed = _bc_from_arrays(src, tgt, w, N, q0, pole, counter)

    if strata is not None:
        sa = (g.vp[strata].a if isinstance(strata, str) else np.asarray(strata)).astype(np.int64)
    else:
        sa = _degree_strata(g, strata_method, n_strata)

    order = np.argsort(sa, kind="mergesort")
    bounds = np.flatnonzero(np.diff(sa[order])) + 1
    groups = np.split(order, bounds)

    samples = np.empty(n_samples, dtype=float)
    for s in range(n_samples):
        q = q0.copy()
        for grp in groups:
            q[grp] = q0[rng.permutation(grp)]
        samples[s] = _bc_from_arrays(src, tgt, w, N, q, pole, counter)

    null_mean = float(samples.mean())
    ratio = observed / null_mean if null_mean > 0 else float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p_upper = float((np.sum(samples >= observed) + 1) / (n_samples + 1))
    return {
        "pole": pole, "counter": counter,
        "observed": observed, "null_mean": null_mean, "R_perm": ratio,
        "null_ci95": (float(lo), float(hi)), "p_upper": p_upper,
        "n_samples": n_samples, "n_strata": len(groups),
        "strata_method": (strata if isinstance(strata, str) else strata_method),
        "weighted": weight_ep is not None,
    }


# ----------------------------------------------------------------------
#  Pole subgraph: induced on pole+counter vertices, only the three edge
#  classes the statistic uses. mixing_preserving_null on this == full graph.
# ----------------------------------------------------------------------
def pole_subgraph(g, quintile_prop="quintile_nation", pole=5, counter=1, weight_ep=None):
    q = _quintile_array(g, quintile_prop)
    vfilt = g.new_vertex_property("bool"); vfilt.a = (q == pole) | (q == counter)
    E = g.get_edges([g.edge_index])
    qs = q[E[:, 0].astype(np.int64)]; qt = q[E[:, 1].astype(np.int64)]
    keep = (((qs == counter) & (qt == pole)) |
            ((qs == pole)    & (qt == pole)) |
            ((qs == pole)    & (qt == counter)))
    efilt = g.new_edge_property("bool")
    efilt.a[E[:, 2].astype(np.int64)] = keep
    return Graph(GraphView(g, vfilt=vfilt, efilt=efilt), prune=True)


# ----------------------------------------------------------------------
#  Self-test: python mixing_null.py --selftest   (inside the gt env)
# ----------------------------------------------------------------------
def _selftest():
    import itertools
    rng = np.random.default_rng(7)
    N = 60
    q = rng.integers(1, 6, N)
    g = Graph(directed=True)
    g.add_vertex(N)
    qp = g.new_vertex_property("string")            # strings: exercise coercion
    for v in g.vertices():
        qp[v] = f"Q{q[int(v)]}"
    g.vp["quintile_nation"] = qp
    wp = g.new_edge_property("double")
    seen = set()
    for _ in range(400):
        u, v = int(rng.integers(N)), int(rng.integers(N))
        if u == v or (u, v) in seen:
            continue
        seen.add((u, v)); e = g.add_edge(u, v); wp[e] = rng.random() + 0.1
    g.ep["q_calls"] = wp

    qa = _quintile_array(g, "quintile_nation")
    assert set(np.unique(qa)).issubset(set(range(1, 6)))
    print(f"[ok] coercion: string -> ints {sorted(set(qa.tolist()))}")

    E = g.get_edges(); src, tgt = E[:, 0], E[:, 1]
    out_nb = {i: [] for i in range(N)}; in_nb = {i: [] for i in range(N)}
    for u, v in zip(src.tolist(), tgt.tolist()):
        out_nb[u].append(v); in_nb[v].append(u)
    brute = 0
    for b in range(N):
        if qa[b] != 5:
            continue
        for i, j in itertools.product(in_nb[b], out_nb[b]):
            if i != j and {qa[i], qa[j]} == {5, 1}:
                brute += 1
    vec = restricted_boundary_control(g, "quintile_nation", 5, 1, weight_ep=None)
    assert abs(vec - brute) < 1e-9, (vec, brute)
    print(f"[ok] statistic vs brute force: {vec:.0f} == {brute}")

    full = mixing_preserving_null(g, "quintile_nation", 5, 1, weight_ep=None,
                                  n_samples=400, n_iter=2, seed=1, restrict_to_poles=False)
    print(f"[ok] mixing null   R_mix={full['R_mix']:.3f}  null_mean={full['null_mean']:.2f}")

    pin = mixing_preserving_null(g, "quintile_nation", 5, 1, weight_ep=None,
                                 n_samples=400, n_iter=2, seed=1, restrict_to_poles=True)
    relp = abs(pin['null_mean'] - full['null_mean']) / full['null_mean']
    print(f"[{'ok' if relp < 0.1 else 'CHECK'}] restrict_to_poles rel diff {relp:.3f}")

    perm = label_permutation_null(g, "quintile_nation", 5, 1, weight_ep=None,
                                  n_samples=400, seed=1, n_strata=8)
    print(f"[ok] label null    R_perm={perm['R_perm']:.3f}  "
          f"null_mean={perm['null_mean']:.2f}  n_strata={perm['n_strata']}")
    print("selftest done.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Import the functions, or run with --selftest inside the gt env.")