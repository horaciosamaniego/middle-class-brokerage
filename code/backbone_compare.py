#!/usr/bin/env python3
"""
backbone_compare.py -- put the reciprocated-backbone results next to the published
ones, analysis by analysis, on the same cities.

For each stratification and each analysis it reports
  main       the published value (all cities that pass the paper's gate)
  main@bb    the published value restricted to the cities evaluable on the backbone
  backbone   the value on the reciprocated backbone, same gates applied to the backbone
so a difference between main and backbone can be told apart from a change in the city
set. For the ladder it also reports the published analysis with the a->b->a return
paths removed (gf_roles.py --returns ... --exclude_returns), when that file exists.

  python backbone_compare.py --main_dir ../data/derived --bb_dir . \\
      --out backbone_comparison.csv

Missing inputs are reported and skipped, so it can be run after any subset of steps.
"""
import argparse
import os
import unicodedata

import numpy as np
import pandas as pd

GATE_REACH = 0.5
MIN_CAPABLE = 5000
ROLES = ["coordinator", "itinerant_above", "itinerant_below", "gatekeeper", "representative",
         "liaison_intermediate", "liaison_extremal_above", "liaison_extremal_below"]
POS_NAME = {"population": "population", "antenna": "area"}
PREFIX, LABEL = "bb_", "backbone"   # repo naming of position files

OUT = []


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def read(path, what):
    if not os.path.exists(path):
        print(f"  [skip] {what}: {path} not found")
        return None
    d = pd.read_csv(path)
    if "city" in d.columns:
        d["city"] = d["city"].map(nfc)
    return d


def rec(mode, analysis, item, quantity, main=np.nan, main_at_bb=np.nan, bb=np.nan,
        n_main=np.nan, n_bb=np.nan, extra=np.nan):
    OUT.append(dict(mode=mode, analysis=analysis, item=item, quantity=quantity,
                    main=main, main_at_bb=main_at_bb, backbone=bb,
                    n_main=n_main, n_bb=n_bb, main_noreturns=extra))


def fmt(x, nd=3):
    return "   --" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- position
def position(mode, main_dir, bb_dir):
    m = read(os.path.join(main_dir, f"position_summary_{POS_NAME[mode]}.csv"), "main position")
    b = read(os.path.join(bb_dir, f"{PREFIX}position_summary_{mode}.csv"), "backbone position")
    if m is None or b is None:
        return

    def gated(d):
        reach = (d.pairs_reached / d.pairs_tried.replace(0, np.nan)).groupby(d.city).first()
        return set(reach[reach >= GATE_REACH].index)

    S_main = gated(m)
    S_bb = gated(b) & S_main
    print(f"\n[position, {mode}] cities: main {len(S_main)}, of which pass the gate on "
          f"the backbone {len(S_bb)}")
    print(f"  {'stratum':8s}{'main':>8s}{'main@bb':>9s}{LABEL:>10s}"
          f"{'%below main':>13s}{'%below bb':>11s}")
    for k in range(1, 6):
        def ratio(d, S):
            x = d[(d.stratum == k) & d.city.isin(S) & (d.rho_null_mean > 0)]
            r = x.rho_obs / x.rho_null_mean
            return float(r.median()) if len(r) else np.nan, \
                float(100 * (x.rho_obs < x.rho_null_mean).mean()) if len(x) else np.nan
        rm, bm = ratio(m, S_main)
        rmb, _ = ratio(m, S_bb)
        rb, bb_ = ratio(b, S_bb)
        print(f"  Q{k:<7d}{fmt(rm, 2):>8s}{fmt(rmb, 2):>9s}{fmt(rb, 2):>10s}"
              f"{fmt(bm, 0):>13s}{fmt(bb_, 0):>11s}")
        rec(mode, "position", f"Q{k}", "median obs/null", rm, rmb, rb, len(S_main), len(S_bb))
        rec(mode, "position", f"Q{k}", "% cities below null", bm, np.nan, bb_, len(S_main), len(S_bb))


# --------------------------------------------------------------------------- necessity
def necessity(mode, main_dir, bb_dir):
    m = read(os.path.join(main_dir, f"necessity_{mode}.csv"), "main necessity")
    b = read(os.path.join(bb_dir, f"{PREFIX}necessity_{mode}.csv"), "backbone necessity")
    if m is None or b is None:
        return
    S = set(m.city) & set(b.city)
    print(f"\n[necessity, {mode}] cities: main {m.city.nunique()}, on backbone {len(S)}")
    print(f"  {'stratum':8s}{'main':>9s}{'main@bb':>9s}{LABEL:>10s}{'%>0 main':>10s}{'%>0 bb':>8s}")
    for k in (2, 3, 4):
        def stat(d, S=None):
            x = d[(d.stratum == k)]
            if S is not None:
                x = x[x.city.isin(S)]
            return float(x.delta.median()), float(100 * (x.delta > 0).mean())
        dm, pm = stat(m)
        dmb, _ = stat(m, S)
        db, pb = stat(b, S)
        print(f"  Q{k:<7d}{dm:>9.4f}{dmb:>9.4f}{db:>10.4f}{pm:>10.0f}{pb:>8.0f}")
        rec(mode, "necessity", f"Q{k}", "median Delta", dm, dmb, db, m.city.nunique(), len(S))
        rec(mode, "necessity", f"Q{k}", "% cities Delta>0", pm, np.nan, pb, m.city.nunique(), len(S))
    e0m = float(m.groupby("city").E0.first().median())
    e0b = float(b[b.city.isin(S)].groupby("city").E0.first().median())
    print(f"  baseline efficiency E0: main {e0m:.3f}, backbone {e0b:.3f}")
    rec(mode, "necessity", "E0", "median baseline efficiency", e0m, np.nan, e0b, m.city.nunique(), len(S))


# --------------------------------------------------------------------------- two nulls
def _bools(d):
    for c in ("power_ok", "mix_consistent", "mix_symmetric"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower().isin(["true", "1"])
    return d


def _powered(d, pole):
    x = d[d.pole == pole]
    return x[x.power_ok & x.mix_consistent & (x.boundary_capable >= MIN_CAPABLE)]


def set_min_capable(v):
    global MIN_CAPABLE
    MIN_CAPABLE = int(v)


def two_null(mode, main_dir, bb_dir):
    m = read(os.path.join(main_dir, f"two_null_{mode}_summary.csv"), "main two-null")
    b = read(os.path.join(bb_dir, f"{PREFIX}two_null_{mode}.csv"), "backbone two-null")
    if m is None or b is None:
        return
    m, b = _bools(m), _bools(b)
    if PREFIX == "bb_" and "mix_symmetric" in b.columns and not b.mix_symmetric.all():
        print("  WARN some backbone rows were not run with the symmetric null -- "
              "was the patched mixing_null.py on the path?")
    n_incons = int((~b.mix_consistent).sum())
    if n_incons:
        print(f"  note: {n_incons} backbone rows fail the swap-vs-analytic check and are excluded")
    for pole in (5, 1):
        pm, pb = _powered(m, pole), _powered(b, pole)
        S = set(pb.city) & set(pm.city)
        pmb = pm[pm.city.isin(S)]
        print(f"\n[pole Q{pole}, {mode}] powered cities: main {len(pm)}, {LABEL} {len(pb)} "
              f"(both {len(S)})")
        print(f"  {'':22s}{'main':>8s}{'main@bb':>9s}{LABEL:>11s}")
        for q, f in (("median R_perm", lambda x: x.R_perm.median()),
                     ("median R_mix", lambda x: x.R_mix.median()),
                     ("% cities R_mix < 1", lambda x: 100 * (x.R_mix < 1).mean()),
                     ("% cities p_mix < 0.05", lambda x: 100 * (x.p_mix < 0.05).mean())):
            vm = float(f(pm)) if len(pm) else np.nan
            vmb = float(f(pmb)) if len(pmb) else np.nan
            vb = float(f(pb)) if len(pb) else np.nan
            nd = 0 if q.startswith("%") else 3
            print(f"  {q:22s}{fmt(vm, nd):>8s}{fmt(vmb, nd):>9s}{fmt(vb, nd):>11s}")
            rec(mode, "pole", f"Q{pole}", q, vm, vmb, vb, len(pm), len(pb))


# --------------------------------------------------------------------------- roles
def roles(mode, main_dir, bb_dir):
    m = read(os.path.join(main_dir, f"gf_roles_{mode}.csv"), "main roles")
    b = read(os.path.join(bb_dir, f"{PREFIX}gf_roles_{mode}.csv"), "backbone roles")
    c = read(os.path.join(bb_dir, f"gf_roles_{mode}_noreturns.csv"), "main roles, returns removed")
    if m is None or b is None:
        return
    print(f"\n[ladder roles, {mode}] cities: main {m.city.nunique()}, {LABEL} {b.city.nunique()}"
          + (f", main without returns {c.city.nunique()}" if c is not None else ""))
    print(f"  {'role':24s}{'main':>7s}{'noret':>7s}{'main@bb':>8s}{LABEL:>11s}"
          f"{'%>1 main':>9s}{'%>1 noret':>10s}{'%>1 bb':>8s}{'n bb':>6s}")
    for r in ROLES:
        col = f"Rmix_{r}"
        vm = m[col].dropna(); vb = b[col].dropna()
        S = set(b.loc[b[col].notna(), "city"])
        vmb = m.loc[m.city.isin(S), col].dropna()
        vc = c[col].dropna() if c is not None else pd.Series(dtype=float)
        line = (f"  {r:24s}{fmt(vm.median()):>7s}{fmt(vc.median() if len(vc) else np.nan):>7s}"
                f"{fmt(vmb.median() if len(vmb) else np.nan):>8s}{fmt(vb.median() if len(vb) else np.nan):>11s}"
                f"{100 * (vm > 1).mean():>9.0f}"
                f"{(100 * (vc > 1).mean() if len(vc) else np.nan):>10.0f}"
                f"{(100 * (vb > 1).mean() if len(vb) else np.nan):>8.0f}{len(vb):>6d}")
        print(line)
        rec(mode, "roles", r, "median R_mix", float(vm.median()),
            float(vmb.median()) if len(vmb) else np.nan,
            float(vb.median()) if len(vb) else np.nan, len(vm), len(vb),
            float(vc.median()) if len(vc) else np.nan)
        rec(mode, "roles", r, "% cities R_mix > 1", float(100 * (vm > 1).mean()), np.nan,
            float(100 * (vb > 1).mean()) if len(vb) else np.nan, len(vm), len(vb),
            float(100 * (vc > 1).mean()) if len(vc) else np.nan)


# --------------------------------------------------------------------------- stats
def stats(bb_dir):
    s = read(os.path.join(bb_dir, "backbone_stats.csv"), "backbone stats")
    if s is None:
        return
    print(f"\n[backbone construction] {len(s)} cities")
    for col, lab in (("arc_reciprocity", "share of arcs reciprocated"),
                     ("node_share", "share of users kept"),
                     ("calls_share_bb", "share of calls on reciprocated ties"),
                     ("mean_outdeg_orig", "mean out-degree, original"),
                     ("mean_deg_bb", "mean degree, backbone"),
                     ("population_returns_share", "return paths, share of same-group 2-paths (pop)"),
                     ("antenna_returns_share", "return paths, share of same-group 2-paths (area)")):
        if col in s.columns:
            v = s[col].dropna()
            print(f"  {lab:50s} median {v.median():.3f}   IQR {v.quantile(.25):.3f}-{v.quantile(.75):.3f}")
            rec("all", "construction", col, "median", bb=float(v.median()), n_bb=len(v))
    for mode, short in (("population", "population"), ("antenna", "antenna")):
        kept = [f"{short}_kept_Q{k}" for k in range(1, 6)]
        if all(c in s.columns for c in kept):
            print(f"  users kept by stratum ({mode}): " +
                  "  ".join(f"Q{k} {s[c].median():.2f}" for k, c in enumerate(kept, 1)))
    if "n_parallel" in s.columns and s.n_parallel.sum() > 0:
        print(f"  NOTE {int((s.n_parallel > 0).sum())} graphs contain parallel arcs "
              f"(counted once in the backbone)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main_dir", default="../data/derived",
                    help="folder with the published derived files (the repo's data/derived)")
    ap.add_argument("--bb_dir", default=".", help="folder with the bb_* outputs")
    ap.add_argument("--modes", nargs="+", default=["population", "antenna"])
    ap.add_argument("--out", default="backbone_comparison.csv")
    ap.add_argument("--prefix", default="bb_",
                    help="file prefix of the run to compare: bb_ (backbone) or rp_ (reciprocal null)")
    ap.add_argument("--label", default=None, help="column label (default: backbone or reciprocal)")
    ap.add_argument("--min_capable", type=int, default=MIN_CAPABLE,
                    help="boundary-capable brokers for a pole to count as powered (paper: 5000)")
    a = ap.parse_args()
    set_min_capable(a.min_capable)
    global PREFIX, LABEL
    PREFIX = a.prefix
    LABEL = a.label or ("backbone" if PREFIX == "bb_" else "reciprocal")
    if PREFIX == "bb_":
        stats(a.bb_dir)
    for mode in a.modes:
        print(f"\n{'=' * 78}\n{mode.upper()}\n{'=' * 78}")
        if PREFIX == "bb_":               # the reciprocal run re-does only pole and ladder
            position(mode, a.main_dir, a.bb_dir)
            necessity(mode, a.main_dir, a.bb_dir)
        two_null(mode, a.main_dir, a.bb_dir)
        roles(mode, a.main_dir, a.bb_dir)
    if OUT:
        pd.DataFrame(OUT).to_csv(a.out, index=False)
        print(f"\nwrote {a.out} ({len(OUT)} rows)")


if __name__ == "__main__":
    main()
