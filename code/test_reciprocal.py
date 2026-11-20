#!/usr/bin/env python3
"""
test_reciprocal.py -- two-minute check, on synthetic graphs, that the
reciprocity-preserving mixing null behaves as intended in this graph-tool
installation. No real data is touched.

The synthetic graphs are directed, about 60% of ties reciprocated, with heavy-tailed
activity and income labels drawn independently of who calls whom, so every ratio
should be one. The test checks that
  1. the reciprocal null gives pole R_mix close to one, and its swaps agree with
     the closed-form expectation (reciprocal_expected_bc);
  2. on a fully reciprocated graph it coincides with the symmetric null;
  3. averaged over several graphs, every ladder role is close to one once return
     paths are removed, while the directed null sits measurably below one.

  conda activate gtbb
  python code/test_reciprocal.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mixing_null as mn             # noqa: E402
import mixing_null_5x5 as m5         # noqa: E402
import gf_roles as gr                # noqa: E402
from test_backbone import synth, VP  # noqa: E402


def roles(g, reciprocal, seed):
    if reciprocal:
        B, nm, sd, R, ro, rn, extra = m5.r5x5_one_city_reciprocal(g, VP, n_samples=20, burn_in=5, seed=seed)
        sym = False
    else:
        B, nm, sd, R, ro, rn, sym = m5.r5x5_one_city(g, VP, n_samples=20, burn_in=5, seed=seed)
        extra = None
    d = pd.DataFrame(m5.tidy_rows("t", B, nm, sd, R, 0, ret_obs=ro, ret_null=rn,
                                  symmetric=sym, extra=extra))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        d = gr.remove_returns(d)
    return np.array([v[0] for v in gr.per_city_roles(d, 0).values()])


def main():
    fails = []

    def check(cond, msg):
        print(("  ok    " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    print("1. pole census, partly reciprocated graph (expect ~1)")
    g = synth(N=30000, M=120000, recip=0.6, seed=2)
    for pole, counter in ((5, 1), (1, 5)):
        sub = mn.pole_subgraph(g, VP, pole=pole, counter=counter)
        d = mn.mixing_preserving_null(sub, VP, pole=pole, counter=counter,
                                      n_samples=60, n_iter=2, burn_in=10, seed=3)
        r = mn.mixing_preserving_null(sub, VP, pole=pole, counter=counter,
                                      n_samples=60, n_iter=2, burn_in=10, seed=3, reciprocal=True)
        dev = abs(r["null_mean"] / r["analytic_expected"] - 1)
        print(f"     Q{pole}: directed null {d['R_mix']:.3f}   reciprocal null {r['R_mix']:.3f}"
              f"   swaps vs closed form {dev:.2%}   overlap {r['overlap_frac']:.4f}")
        check(r["null_model"] == "reciprocal", f"Q{pole} used the reciprocal null")
        check(abs(r["R_mix"] - 1) < 0.03, f"Q{pole} reciprocal R_mix within 0.03 of 1")
        check(dev < 0.02, f"Q{pole} swaps within 2% of the closed form")

    print("\n2. fully reciprocated graph: reciprocal null = symmetric null")
    gs = synth(N=20000, M=80000, recip=1.0, seed=4)
    sub = mn.pole_subgraph(gs, VP, pole=5, counter=1)
    s = mn.mixing_preserving_null(sub, VP, pole=5, counter=1, n_samples=60, n_iter=2, burn_in=10, seed=3)
    r = mn.mixing_preserving_null(sub, VP, pole=5, counter=1, n_samples=60, n_iter=2, burn_in=10,
                                  seed=3, reciprocal=True)
    print(f"     symmetric null {s['R_mix']:.3f}   reciprocal null {r['R_mix']:.3f}")
    check(abs(s["R_mix"] - r["R_mix"]) < 0.02, "the two agree within 0.02")
    check(abs(r["analytic_expected"] / s["analytic_expected"] - 1) < 1e-9, "closed forms identical")

    print("\n3. ladder, six graphs, return paths removed (expect ~1)")
    D, R = [], []
    for seed in range(10, 16):
        gg = synth(N=20000, M=80000, recip=0.6, seed=seed)
        D.append(roles(gg, False, seed)); R.append(roles(gg, True, seed))
    D, R = np.array(D).mean(0), np.array(R).mean(0)
    for k, (dd, rr) in enumerate(zip(D, R)):
        print(f"     {gr.ROLES[k]:24s} directed {dd:.3f}   reciprocal {rr:.3f}")
    check(np.all(np.abs(R - 1) < 0.02), "every role within 0.02 of 1 under the reciprocal null")
    check(abs(R.mean() - 1) < abs(D.mean() - 1), "reciprocal null closer to 1 than the directed null")

    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
