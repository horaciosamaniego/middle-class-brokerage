#!/usr/bin/env python3
"""
test_backbone.py -- one-minute check, on synthetic graphs, that the backbone tools
behave as intended in this graph-tool installation. No real data is touched.

The synthetic graphs have the features that matter here: directed, about two thirds
of ties reciprocated, heavy-tailed activity, and income labels drawn independently
of who calls whom. With labels independent of ties, every brokerage ratio should be
one. The test checks that
  1. the backbone is symmetric and keeps the labels;
  2. on the backbone, the symmetric mixing null gives R_mix close to one at the poles
     and for every ladder role, and its analytic expectation matches the swaps;
  3. on the ORIGINAL directed graph, removing the a->b->a return paths brings the
     same-group roles back to about one (the size of what they added is printed).

  conda activate gt
  python code/test_backbone.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import graph_tool.all as gt          # noqa: E402
import mixing_null as mn             # noqa: E402
import mixing_null_5x5 as m5         # noqa: E402
import gf_roles as gr                # noqa: E402
import make_backbone as mb           # noqa: E402

VP = "quintile_nation_population"
SAME = ("coordinator", "itinerant_above", "itinerant_below")


def synth(N=30000, M=120000, recip=0.6, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.pareto(1.6, N) + 1.0
    p = w / w.sum()
    u = rng.choice(N, M, p=p); v = rng.choice(N, M, p=p)
    ok = u != v
    pairs = np.unique(np.sort(np.c_[u[ok], v[ok]], axis=1), axis=0)
    both = rng.random(len(pairs)) < recip
    flip = rng.random(len(pairs)) < 0.5
    one = pairs[~both].copy(); one[flip[~both]] = one[flip[~both]][:, ::-1]
    arcs = np.vstack([pairs[both], pairs[both][:, ::-1], one])
    g = gt.Graph(directed=True); g.add_vertex(N); g.add_edge_list(arcs)
    qs = g.new_vertex_property("string")
    lab = rng.integers(1, 6, N)
    for vv in g.vertices():
        qs[vv] = str(lab[int(vv)])
    g.vp[VP] = qs
    wc = g.new_edge_property("int"); wc.a = rng.integers(1, 40, g.num_edges())
    g.ep["q_calls"] = wc
    return g


def roles(g, n=40):
    out = m5.r5x5_one_city(g, VP, n_samples=n, n_iter=1, burn_in=5, seed=7)
    B, nm, sd, R, ro, rn, sym = out
    d = pd.DataFrame(m5.tidy_rows("t", B, nm, sd, R, 0, ret_obs=ro, ret_null=rn, symmetric=sym))
    raw = gr.per_city_roles(d, 0)
    clean = gr.per_city_roles(gr.remove_returns(d), 0)
    return {k: v[0] for k, v in raw.items()}, {k: v[0] for k, v in clean.items()}, sym


def main():
    fails = []

    def check(cond, msg):
        print(("  ok    " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "g"), os.path.join(tmp, "bb")
        os.makedirs(src); os.makedirs(dst)
        g0 = synth()
        g0.save(os.path.join(src, "grafo_Teste.gt"))
        name, msg = mb.build_one((os.path.join(src, "grafo_Teste.gt"), dst, [VP], "q_calls", False))
        print("build:", msg.splitlines()[0])
        g = gt.load_graph(os.path.join(src, "grafo_Teste.gt"))
        b = gt.load_graph(os.path.join(dst, "grafo_Teste.gt"))

        print("\n1. backbone")
        check(mn.is_symmetric(b) and not mn.is_symmetric(g), "backbone symmetric, original not")
        check(VP in b.vp and "q_calls" in b.ep, "labels and weights carried over")

        print("\n2. symmetric null on the backbone (labels independent: expect ~1)")
        for pole, counter in ((5, 1), (1, 5)):
            sub = mn.pole_subgraph(b, VP, pole=pole, counter=counter)
            r = mn.mixing_preserving_null(sub, VP, pole=pole, counter=counter,
                                          n_samples=80, n_iter=2, burn_in=10, seed=3)
            dev = abs(r["null_mean"] - r["analytic_expected"]) / r["analytic_expected"]
            print(f"     Q{pole}: R_mix {r['R_mix']:.3f}  swap vs analytic {dev:.1%}")
            check(r["symmetric"], f"Q{pole} used the symmetric null")
            check(abs(r["R_mix"] - 1) < 0.05, f"Q{pole} R_mix within 0.05 of 1")
            check(dev < 0.05, f"Q{pole} analytic expectation within 5% of the swaps")
        raw, clean, sym = roles(b)
        print("     roles, returns removed: " + "  ".join(f"{k[:10]} {v:.3f}" for k, v in clean.items()))
        check(sym, "ladder used the symmetric null")
        check(all(abs(v - 1) < 0.06 for v in clean.values()), "every role within 0.06 of 1")

        print("\n3. original directed graph, directed null")
        raw, clean, sym = roles(g)
        for k in SAME:
            print(f"     {k:16s} with returns {raw[k]:.3f}   without {clean[k]:.3f}")
        check(not sym, "directed null used on the original graph")
        check(all(raw[k] > clean[k] for k in SAME), "returns inflate every same-group role")
        check(all(abs(clean[k] - 1) < 0.06 for k in SAME), "same-group roles ~1 once returns are removed")

    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
