#!/usr/bin/env python3
"""
run_two_null.py
===============

Two-null harness for the pole-to-pole boundary-control statistic
(gatekeeper + representative, Q_pole <-> Q_counter), per city:

  * R_perm + p  -- degree-stratified LABEL permutation (== pipeline's
                   q5_closure_ratio; lets contact volume float)
  * R_mix  + p  -- degree + MIXING preserving null (holds Q_pole<->Q_counter
                   contact volume fixed; isolates positional control)

Both poles run (Q5<-Q1 and Q1<-Q5). Unweighted statistic throughout.

POWER + SELF-CONSISTENCY (added after the Chilean run):
  * gk_capable/rp_capable/boundary_capable: brokers that can form a boundary
    2-path at all. These bound what the measure can resolve; a pole with a few
    hundred capable brokers is under-powered and its R_mix is a SPARSITY
    signature, not a mechanism (Puerto Montt: ~230 capable, R_mix~0.39).
  * mix_consistent: swap null mean vs analytic independence expectation. They
    agreed to ~1% on Puerto Montt -> the null samples correctly. Divergence on
    any row = distrust that row.
  * power_ok: boundary_capable >= min_capable. Default floor 5000 is calibrated
    to the Chilean data, where only Santiago (~10k+ capable) is informative and
    even La Serena (~2.6k) still shows sparsity bias. REVISIT against the
    Brazilian distribution -- this floor is a starting point, not a law.

Read R_mix with its p AND its power flag, never the ratio alone.
Crash-safe: per-city CSV in out_dir (skipped if present), then concatenated.

Usage:
  python run_two_null.py --probe --data_dir DIR --cities santiago
  python run_two_null.py --data_dir DIR --cities santiago valparaiso ... --vp quintile_nation
"""
import os, glob, time, argparse
import numpy as np
import pandas as pd
import graph_tool.all as gt
import mixing_null as mn

THIN_POLE = 100        # pipeline's thin-pole flag
MIN_CAPABLE = 5000     # boundary-capable brokers needed to trust R_mix (see header)
CONSISTENCY_TOL = 0.10 # |swap_mean - analytic| / analytic below this = consistent


def _add_pole_block(sub, qprop, pole, counter):
    """Contiguous {0,1} block labels (REQUIRED: constrained-configuration silently
    fails to preserve the mixing matrix with non-contiguous labels). counter->0, pole->1."""
    q = mn._quintile_array(sub, qprop)
    pb = sub.new_vertex_property("int")
    pb.a = np.where(q == pole, 1, np.where(q == counter, 0, -1)).astype(np.int64)
    sub.vp["pole_block"] = pb
    return sub


def capable_broker_counts(g, qprop, pole, counter):
    """Per-pole power denominators: `pole` brokers that can form a boundary 2-path.
      gk_capable : in-counter>0 AND out-pole>0    (Q_counter -> b -> Q_pole)
      rp_capable : in-pole>0   AND out-counter>0  (Q_pole -> b -> Q_counter)
      boundary_capable : gk OR rp (brokers contributing >=1 boundary 2-path)
    These -- not the pole size -- bound what the measure can resolve."""
    q = mn._quintile_array(g, qprop)
    E = g.get_edges(); s = E[:, 0].astype(np.int64); t = E[:, 1].astype(np.int64)
    N = g.num_vertices()
    in_c  = np.bincount(t[q[s] == counter], minlength=N) > 0
    out_p = np.bincount(s[q[t] == pole],    minlength=N) > 0
    in_p  = np.bincount(t[q[s] == pole],    minlength=N) > 0
    out_c = np.bincount(s[q[t] == counter], minlength=N) > 0
    m = q == pole
    gk = in_c & out_p
    rp = in_p & out_c
    return (int(gk[m].sum()), int(rp[m].sum()), int((gk | rp)[m].sum()))


def two_null_one_pole(g, qprop, pole, counter,
                      n_perm=500, perm_seed=20270624, n_strata=10,
                      n_mix=300, n_iter=2, burn_in=20, mix_seed=1,
                      min_capable=MIN_CAPABLE, consistency_tol=CONSISTENCY_TOL):
    q = mn._quintile_array(g, qprop)
    n_pole = int((q == pole).sum()); n_counter = int((q == counter).sum())
    gk_cap, rp_cap, bnd_cap = capable_broker_counts(g, qprop, pole, counter)

    # label null on the FULL graph
    perm = mn.label_permutation_null(g, qprop, pole=pole, counter=counter,
                                     weight_ep=None, n_samples=n_perm,
                                     seed=perm_seed, n_strata=n_strata)
    observed = perm["observed"]

    # mixing null on the pole subgraph, contiguous blocks
    sub = mn.pole_subgraph(g, qprop, pole=pole, counter=counter)
    _add_pole_block(sub, qprop, pole, counter)
    obs_sub = mn.restricted_boundary_control(sub, "pole_block", pole=1, counter=0, weight_ep=None)
    mix = mn.mixing_preserving_null(sub, "pole_block", pole=1, counter=0, weight_ep=None,
                                    n_samples=n_mix, n_iter=n_iter, burn_in=burn_in,
                                    seed=mix_seed, restrict_to_poles=False)

    analytic = mix["analytic_expected"]
    consistent = bool(np.isfinite(analytic) and analytic > 0 and
                      abs(mix["null_mean"] - analytic) / analytic < consistency_tol)
    power_ok = bool(bnd_cap >= min_capable)

    return dict(
        pole=pole, counter=counter, n_pole=n_pole, n_counter=n_counter,
        min_pole=min(n_pole, n_counter), thin_pole=bool(min(n_pole, n_counter) < THIN_POLE),
        gk_capable=gk_cap, rp_capable=rp_cap, boundary_capable=bnd_cap, power_ok=power_ok,
        observed=observed,
        R_perm=perm["R_perm"], p_perm=perm["p_upper"],
        perm_ci_lo=perm["null_ci95"][0], perm_ci_hi=perm["null_ci95"][1],
        perm_null_mean=perm["null_mean"],
        R_mix=mix["R_mix"], p_mix=mix["p_upper"],
        mix_ci_lo=mix["null_ci95"][0], mix_ci_hi=mix["null_ci95"][1],
        mix_null_mean=mix["null_mean"], analytic_expected=analytic,
        R_mix_analytic=mix["R_mix_analytic"], mix_consistent=consistent,
        obs_subgraph_ok=bool(abs(obs_sub - observed) < 1e-6),
        mix_significant=bool(mix["p_upper"] < 0.05),
    )


def _verdict(r):
    if not r["mix_consistent"]:
        return "NULL-SUSPECT"          # swap != analytic: distrust this row
    if not r["power_ok"]:
        return "underpowered"          # too few capable brokers: R_mix is sparsity
    if r["thin_pole"]:
        return "provisional(thin)"
    return "REAL closure" if r["mix_significant"] else "no closure"


def run_city(path, qprop="quintile_nation", poles=((5, 1), (1, 5)), **kw):
    name = os.path.splitext(os.path.basename(path))[0].replace("grafo_", "", 1)
    g = gt.load_graph(path)
    rows = []
    for pole, counter in poles:
        r = two_null_one_pole(g, qprop, pole, counter, **kw)
        r["city"] = name
        r["n_nodes"] = int(g.num_vertices()); r["n_edges"] = int(g.num_edges())
        rows.append(r)
    return rows


def probe(path, qprop="quintile_nation"):
    g = gt.load_graph(path)
    print("vertex props:", list(g.vp.keys()))
    print("edge props  :", list(g.ep.keys()))
    q = mn._quintile_array(g, qprop)
    vals, cnts = np.unique(q, return_counts=True)
    print(f"quintile '{qprop}' composition:", {int(v): int(c) for v, c in zip(vals, cnts)})
    print("  (expect bins 1..5 populated; -1/0 = missing/excluded)")


def power_scan(data_dir, cities=None, pattern="grafo_*.gt", qprop="quintile_nation",
               out="power_scan.csv", verbose=True):
    """Cheap triage -- NO nulls. Per city: pole sizes and boundary-capable broker
    counts for both poles. Tells you which cities are powered for the two-null
    measure before committing to the expensive label null, and supplies the
    boundary_capable x-axis for the R_mix-vs-capability adjudication figure.
    Loads each graph once; report 'MISSING' flags filename/slug mismatches."""
    if cities:
        paths = [os.path.join(data_dir, f"grafo_{c}.gt") for c in cities]
    else:
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("grafo_", "", 1)
        if not os.path.exists(p):
            print(f"  MISSING: {name}  ({p})"); continue
        try:
            g = gt.load_graph(p); q = mn._quintile_array(g, qprop)
        except Exception as e:
            print(f"  FAIL {name}: {e!r}"); continue
        gk5, rp5, bnd5 = capable_broker_counts(g, qprop, 5, 1)
        gk1, rp1, bnd1 = capable_broker_counts(g, qprop, 1, 5)
        n1, n5 = int((q == 1).sum()), int((q == 5).sum())
        rows.append(dict(city=name, n_nodes=int(g.num_vertices()), n_edges=int(g.num_edges()),
                         n_Q1=n1, n_Q5=n5, min_pole=min(n1, n5),
                         thin_pole=bool(min(n1, n5) < THIN_POLE),
                         gk5_capable=gk5, rp5_capable=rp5, q5_boundary_capable=bnd5,
                         gk1_capable=gk1, rp1_capable=rp1, q1_boundary_capable=bnd1))
        if verbose:
            r = rows[-1]
            print(f"  {name:18s} N={r['n_nodes']:>9d}  Q5n={n5:>7d}  "
                  f"Q5cap={bnd5:>7d}  Q1cap={bnd1:>7d}  "
                  f"{'[THIN]' if r['thin_pole'] else ''}")
    if not rows:
        print("no cities scanned (check --data_dir and slugs)"); return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("q5_boundary_capable", ascending=False)
    df.to_csv(out, index=False)
    if verbose:
        print(f"\nwrote {out}  ({len(df)} cities, sorted by Q5 boundary-capable)")
    return df


def run_cities(data_dir, cities=None, pattern="grafo_*.gt", qprop="quintile_nation",
               out_dir="two_null_out", master="two_null_summary.csv", verbose=True, **kw):
    os.makedirs(out_dir, exist_ok=True)
    if cities:
        paths = [os.path.join(data_dir, f"grafo_{c}.gt") for c in cities]
    else:
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    frames = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("grafo_", "", 1)
        cpath = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(cpath):
            if verbose: print(f"skip (done): {name}")
            frames.append(pd.read_csv(cpath)); continue
        if not os.path.exists(p):
            print(f"skip (missing): {p}"); continue
        t0 = time.time()
        try:
            rows = run_city(p, qprop=qprop, **kw)
        except Exception as e:
            print(f"FAIL {name}: {e!r}"); continue
        df = pd.DataFrame(rows)
        df.to_csv(cpath, index=False)
        frames.append(df)
        if verbose:
            for r in rows:
                print(f"  {name}  Q{r['pole']}<-Q{r['counter']}: "
                      f"obs={r['observed']:.3e}  cap={r['boundary_capable']:>6d}  "
                      f"R_perm={r['R_perm']:.2f}(p={r['p_perm']:.2f})  "
                      f"R_mix={r['R_mix']:.3f}(p={r['p_mix']:.2f})  -> {_verdict(r)}")
            print(f"  [{name} done in {time.time()-t0:.0f}s]")
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    if "boundary_capable" in summary.columns:        # derived verdict column
        summary["verdict"] = summary.apply(_verdict, axis=1)
    lead = ["city", "pole", "counter", "n_nodes", "n_edges", "n_pole", "n_counter",
            "min_pole", "thin_pole", "gk_capable", "rp_capable", "boundary_capable",
            "power_ok", "observed",
            "R_perm", "p_perm", "perm_ci_lo", "perm_ci_hi", "perm_null_mean",
            "R_mix", "p_mix", "mix_ci_lo", "mix_ci_hi", "mix_null_mean",
            "analytic_expected", "R_mix_analytic", "mix_consistent",
            "mix_significant", "obs_subgraph_ok", "verdict"]
    summary = summary[[c for c in lead if c in summary.columns]]
    summary.to_csv(master, index=False)
    if verbose:
        print(f"\nwrote {master}  ({len(summary)} rows)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cities", nargs="+")
    ap.add_argument("--vp", default="quintile_nation")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--power_scan", action="store_true",
                    help="cheap capable-broker triage over --cities (or all), no nulls")
    ap.add_argument("--out_dir", default="two_null_out")
    ap.add_argument("--master", default="two_null_summary.csv")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--n_mix", type=int, default=300)
    ap.add_argument("--n_iter", type=int, default=2)
    ap.add_argument("--min_capable", type=int, default=MIN_CAPABLE)
    a = ap.parse_args()

    if a.probe:
        c = (a.cities or [None])[0]
        p = os.path.join(a.data_dir, f"grafo_{c}.gt") if c else \
            sorted(glob.glob(os.path.join(a.data_dir, "grafo_*.gt")))[0]
        probe(p, qprop=a.vp)
    elif a.power_scan:
        power_scan(a.data_dir, cities=a.cities, qprop=a.vp,
                   out=a.master.replace("summary", "power_scan"))
    else:
        run_cities(a.data_dir, cities=a.cities, qprop=a.vp,
                   out_dir=a.out_dir, master=a.master,
                   n_perm=a.n_perm, n_mix=a.n_mix, n_iter=a.n_iter,
                   min_capable=a.min_capable)