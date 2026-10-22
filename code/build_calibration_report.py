#!/usr/bin/env python3
"""
build_calibration_report.py -- regenerate every calibration number from the raw CSVs.

WHY THIS EXISTS
The calibration figures are coupled. The MDE envelope constant K = max[sd(R_mix)*sqrt(E[beta])]
is set by whichever city happens to have the worst ratio, and K then propagates into the
all-city bound (worst-case MDE, the min ratio and which city holds it). Adding one city can
therefore change a dozen numbers spread across si_calibration.tex, si_calibration.md and
RESULTS_calibration.md. Hand-patching them is how stale numbers get into a paper.

This script is the single source of truth. Point it at the pooled battery CSV plus any new
per-city files and it emits:
  - RESULTS_calibration.md   (full numeric record)
  - calibration_table.tex    (the stab:calibration tabular body, ready to paste)
  - calibration_numbers.md   (CHECKLIST of every prose figure, old vs new, with the
                              sentence it appears in -- so nothing is missed in the .tex)

USAGE
  # regenerate from what is already pooled
  python3 build_calibration_report.py \
      --battery calibration_all_cities.csv \
      --production ../two_null_population_summary.csv

  # fold in a new city and regenerate everything
  python3 build_calibration_report.py \
      --battery calibration_all_cities.csv \
      --add calib_recife.csv --add-city Recife \
      --production ../two_null_population_summary.csv \
      --write-battery              # updates calibration_all_cities.csv in place

NOTES
  --add accepts a per-city output of calibrate_null_fast.py. If that file lacks a 'city'
  column, pass --add-city. Repeat --add/--add-city in matching order to fold in several.
  Rows duplicated by (city,test,replicate,planted,subsample_p) are de-duplicated keeping
  the LAST occurrence, so re-running a corrected test (e.g. Natal's Test B) supersedes the
  earlier one rather than polluting the pool.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

KEY = ["city", "test", "replicate", "planted", "subsample_p"]
GATE = 5000


# --------------------------------------------------------------------------- load

def resolve_runs(d):
    """Separate re-runs of the same city BEFORE de-duplicating on replicate index.

    A second calibration run of a city re-estimates the null denominator, so its rows
    carry a different null_mean while reusing replicate indices 0..n and the same planted
    levels. De-duplicating on (city,test,replicate,planted) alone would therefore splice
    the two runs together -- taking some draws from one denominator and some from the
    other -- which is silently wrong. We resolve each (city,test) by denominator block:

      A_calibration  keep the LARGEST block. Draws under one denominator are what the
                     shared-denominator statistic is defined over, and the biggest block
                     is the primary run. Mixing blocks would inflate the dispersion with
                     between-run denominator noise and corrupt the MDE.
      B_sensitivity  keep the LAST block wholesale. A re-run of Test B supersedes the
                     earlier one, and taking it whole avoids pairing a control from the
                     superseded run with planted levels from the corrected one.

    Everything else (A2_independent, C_subsample) legitimately carries its own denominator
    per row and is left alone.
    """
    if "null_mean" not in d.columns:
        return d
    d = d.copy()
    d["_ord"] = np.arange(len(d))
    d["_den"] = d.null_mean.round(3)
    keep, notes = [], []
    for (c, t), g in d.groupby(["city", "test"], sort=False):
        if t in ("A_calibration", "B_sensitivity") and g._den.nunique() > 1:
            blocks = list(g.groupby("_den", sort=False))
            if t == "A_calibration":
                pick = max(blocks, key=lambda kv: (len(kv[1]), -kv[1]._ord.min()))
            else:
                pick = max(blocks, key=lambda kv: kv[1]._ord.max())
            dropped = sum(len(b) for k, b in blocks if k != pick[0])
            notes.append(f"    {c} {t}: {len(blocks)} denominator blocks "
                         f"{[f'{k:,.0f}(n={len(b)})' for k, b in blocks]} "
                         f"-> kept {pick[0]:,.0f} (n={len(pick[1])}), dropped {dropped}")
            keep.append(pick[1])
        else:
            keep.append(g)
    if notes:
        print("  resolved re-runs by denominator block:")
        for n in notes:
            print(n)
    out = pd.concat(keep).sort_values("_ord").drop(columns=["_ord", "_den"])
    return out.reset_index(drop=True)


def load_battery(path, adds, add_cities):
    frames = []
    if path and os.path.exists(path):
        frames.append(pd.read_csv(path))
    for i, p in enumerate(adds or []):
        if not os.path.exists(p):
            sys.exit(f"--add file not found: {p}")
        f = pd.read_csv(p)
        if "city" not in f.columns:
            if not add_cities or i >= len(add_cities):
                sys.exit(f"{p} has no 'city' column; supply a matching --add-city")
            f["city"] = add_cities[i]
        frames.append(f)
        print(f"  folded in {p}  ({len(f)} rows, city={sorted(f.city.unique())})")
    if not frames:
        sys.exit("nothing to read: give --battery and/or --add")
    d = pd.concat(frames, ignore_index=True)
    d = resolve_runs(d)
    for k in KEY:
        if k not in d.columns:
            d[k] = np.nan
    n0 = len(d)
    d = d.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    if len(d) < n0:
        print(f"  de-duplicated {n0 - len(d)} row(s), keeping the last occurrence")
    return d


def load_production(path, pole):
    """city -> (boundary_capable, observed, mix_null_mean, R_mix, mix_consistent)"""
    if not path or not os.path.exists(path):
        print(f"  WARNING: production summary not found ({path}); "
              f"capability and all-city bound will be omitted")
        return None
    p = pd.read_csv(path)
    if "pole" in p.columns and pole is not None:
        p = p[p.pole == pole]
    return p


# --------------------------------------------------------------------------- stats

def per_city(d, prod):
    """One row per city with every derived quantity."""
    rows = []
    for c in d.city.unique():
        A = d[(d.city == c) & (d.test == "A_calibration")]
        A2 = d[(d.city == c) & (d.test == "A2_independent")]
        if len(A) < 3:
            print(f"  skip {c}: only {len(A)} Test A draws")
            continue
        eb = float(A.null_mean.mean())
        sd = float(A.R_mix.std(ddof=1))
        mde = 2 * sd
        r = dict(city=c, n_A=len(A), mean_A=float(A.R_mix.mean()), sd_A=sd,
                 E_beta=eb, MDE=mde, k=mde * np.sqrt(eb),
                 n_A2=len(A2), mean_A2=float(A2.R_mix.mean()) if len(A2) else np.nan)
        # planted
        B = d[(d.city == c) & (d.test == "B_sensitivity")].sort_values("planted")
        if len(B):
            ctrl = B[B.planted == 0].R_mix
            top = B.iloc[-1]
            r.update(plant_f=float(top.planted), plant_R=float(top.R_mix),
                     plant_eff=float(top.R_mix) - 1.0,
                     plant_ctrl=float(ctrl.iloc[0]) if len(ctrl) else np.nan)
            r["plant_xMDE"] = r["plant_eff"] / mde
        # subsample
        C = d[(d.city == c) & (d.test == "C_subsample")]
        r["sub"] = {}
        for p_ in sorted(C.subsample_p.dropna().unique()):
            s = C[C.subsample_p == p_].R_mix
            se = s.std(ddof=1) / np.sqrt(len(s))
            r["sub"][float(p_)] = dict(n=len(s), mean=float(s.mean()),
                                       lo=float(s.mean() - 1.96 * se),
                                       hi=float(s.mean() + 1.96 * se))
        # production join
        if prod is not None:
            m = prod[prod.city == c]
            if len(m):
                m = m.iloc[0]
                r.update(capable=int(m.boundary_capable), obs=float(m.observed),
                         E_beta_prod=float(m.mix_null_mean), R_obs=float(m.R_mix),
                         mix_consistent=bool(m.mix_consistent))
                r["reldiff"] = abs(eb - r["E_beta_prod"]) / r["E_beta_prod"]
                r["admitted"] = r["capable"] >= GATE
                r["dep"] = 1.0 - r["R_obs"]
                r["dep_xMDE"] = r["dep"] / mde
        rows.append(r)
    out = pd.DataFrame(rows)
    sort_col = "capable" if "capable" in out.columns else "E_beta"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def ci(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() - 1.0, v.mean() - 1 - 1.96 * se, v.mean() - 1 + 1.96 * se


def gate_test(d, pc):
    """Bias above vs below the admission threshold."""
    if "admitted" not in pc.columns:
        return None
    up = pc[pc.admitted].city.tolist()
    lo = pc[~pc.admitted].city.tolist()
    if not up or not lo:
        return dict(up=up, lo=lo, insufficient=True)
    a = d[(d.test == "A2_independent") & (d.city.isin(up))].R_mix
    b = d[(d.test == "A2_independent") & (d.city.isin(lo))].R_mix
    res = dict(up=up, lo=lo, insufficient=False,
               up_n=len(a), lo_n=len(b),
               up_bias=ci(a), lo_bias=ci(b))
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        res.update(t=float(t), p=float(p))
    except Exception:
        res.update(t=np.nan, p=np.nan)
    return res


def all_city_bound(prod, K):
    """Apply MDE <= K/sqrt(E[beta]) to every powered city."""
    if prod is None:
        return None
    p = prod.copy()
    need = {"boundary_capable", "mix_null_mean", "R_mix", "mix_consistent"}
    if not need <= set(p.columns):
        return None
    p = p[(p.boundary_capable >= GATE) & (p.mix_consistent) & (p.mix_null_mean > 0)]
    p = p.dropna(subset=["R_mix"])
    if not len(p):
        return None
    p = p.assign(bound=K / np.sqrt(p.mix_null_mean),
                 dep=(1.0 - p.R_mix).abs())
    p = p.assign(ratio=p.dep / p.bound)
    worst = p.loc[p.ratio.idxmin()]
    return dict(n=len(p), worst_bound=float(p.bound.max()),
                min_ratio=float(p.ratio.min()), min_city=str(worst.city),
                frac_gt5=float((p.ratio > 5).mean()),
                frac_gt1=float((p.ratio > 1).mean()))


# --------------------------------------------------------------------------- emit

def fmt_ci(t):
    b, lo, hi = t
    return f"{b:+.5f}", f"[{lo:+.5f}, {hi:+.5f}]"


def write_results_md(path, pc, d, gt, bound, K, Kadm, slope, krange):
    L = []
    P = L.append
    n = len(pc)
    has_cap = "capable" in pc.columns
    P("# Calibration of the mixing-null brokerage estimator")
    P("")
    P(f"{n} cities, equal-population stratification, Q5 pole. Every number is read from the")
    P("pooled battery CSV and, for the production quantities, from the two-null summary.")
    P("Regenerated by `build_calibration_report.py` -- do not hand-edit.")
    P("")
    if has_cap:
        nu, nl = int(pc.admitted.sum()), int((~pc.admitted).sum())
        P(f"**{nu} of the {n} cities clear the boundary-capable admission threshold of "
          f"{GATE:,}; {nl} fall below it.**")
        P("")
        P("## Production reproducibility")
        P("")
        P("| city | capable | admitted | E[beta] calib | E[beta] prod | rel. diff | observed R_mix | mix_consistent |")
        P("|---|---:|:---:|---:|---:|---:|---:|:---:|")
        for r in pc.itertuples():
            P(f"| {r.city} | {r.capable:,} | {'yes' if r.admitted else 'no'} | "
              f"{r.E_beta:,.0f} | {r.E_beta_prod:,.0f} | {100*r.reldiff:.2f}% | "
              f"{r.R_obs:.4f} | {r.mix_consistent} |")
        P("")
        P(f"Null expectation reproduces production to within {100*pc.reldiff.max():.2f}% everywhere.")
        P("")
    P("## TEST A -- calibration on graphs whose true R_mix is 1 by construction")
    P("")
    P("| city | n | mean (shared) | bias | sd | MDE | mean (indep.) | bias |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in pc.itertuples():
        P(f"| {r.city} | {r.n_A} | {r.mean_A:.5f} | {r.mean_A-1:+.5f} | {r.sd_A:.5f} | "
          f"{r.MDE:.4f} | {r.mean_A2:.5f} | {r.mean_A2-1:+.5f} |")
    pool = d[d.test == "A2_independent"].R_mix
    b, lo, hi = ci(pool)
    P("")
    P(f"**Pooled independent-denominator bias (n={len(pool)}): {b:+.5f}, "
      f"95% CI [{lo:+.5f}, {hi:+.5f}]** -- {'covers' if lo <= 0 <= hi else 'DOES NOT COVER'} zero.")
    P("")
    if gt and not gt.get("insufficient"):
        P("## The gate is a precision threshold, not an accuracy threshold")
        P("")
        P("| group | cities | n | A2 bias | 95% CI |")
        P("|---|---|---:|---:|---|")
        for lab, keyn, keyb, cities in [("admitted", "up_n", "up_bias", gt["up"]),
                                        ("below threshold", "lo_n", "lo_bias", gt["lo"])]:
            bb, cc = fmt_ci(gt[keyb])
            P(f"| {lab} | {', '.join(cities)} | {gt[keyn]} | {bb} | {cc} |")
        P("")
        P(f"Welch t-test across the gate: **t = {gt['t']:.2f}, p = {gt['p']:.2f}**"
          f" -- {'no detectable difference' if gt['p'] > 0.05 else 'DIFFERENCE DETECTED, revisit the claim'}.")
        P("")
        P(f"MDE spans {pc.MDE.min():.4f} to {pc.MDE.max():.4f}, a "
          f"{pc.MDE.max()/pc.MDE.min():.0f}-fold loss of precision over a range where accuracy is flat.")
        if "R_obs" in pc.columns:
            ex = pc[~pc.admitted]
            P(f"The excluded cities return {', '.join(f'{v:.2f}' for v in ex.R_obs)}, "
              f"inside the admitted range of {pc[pc.admitted].R_obs.min():.2f} to "
              f"{pc[pc.admitted].R_obs.max():.2f}.")
        P("")
    P("## MDE envelope")
    P("")
    P(f"Log-log slope of sd(R_mix) on E[beta]: **{slope:.2f}** (count-limited predicts -0.50).")
    P(f"The constant below varies by {krange:.1f}x, so we take an envelope, not a fit.")
    P("")
    P("| city | admitted | MDE | MDE x sqrt(E[beta]) |")
    P("|---|:---:|---:|---:|")
    for r in pc.itertuples():
        adm = ('yes' if r.admitted else 'no') if has_cap else '--'
        P(f"| {r.city} | {adm} | {r.MDE:.4f} | {r.k:.3f} |")
    P("")
    kcity = pc.loc[pc.k.idxmax()].city
    P(f"Envelope over all = **{K:.3f}** (set by {kcity}).")
    if Kadm is not None:
        P(f"Envelope over admitted only = {Kadm:.3f}.")
    P("")
    P(f"    MDE <= {K:.2f} / sqrt(E[beta])")
    P("")
    if bound:
        P(f"Applied to the {bound['n']} powered cities: **worst-case MDE {bound['worst_bound']:.4f}; "
          f"{100*bound['frac_gt5']:.0f}% exceed 5x their own bound; smallest ratio "
          f"{bound['min_ratio']:.1f}, at {bound['min_city']}.**")
        P("")
    if "dep_xMDE" in pc.columns:
        P("Where MDE is measured directly rather than bounded:")
        P("")
        P("| city | observed R_mix | departure | MDE | x MDE |")
        P("|---|---:|---:|---:|---:|")
        for r in pc.itertuples():
            P(f"| {r.city} | {r.R_obs:.4f} | {r.dep:.4f} | {r.MDE:.4f} | {r.dep_xMDE:.0f}x |")
        P("")
    P("## TEST B -- recovery of planted concentration")
    P("")
    P("| city | largest planted f | observed R_mix | effect (R-1) | x MDE | control (f=0) |")
    P("|---|---:|---:|---:|---:|---:|")
    for r in pc.itertuples():
        if not hasattr(r, "plant_f") or not np.isfinite(getattr(r, "plant_f", np.nan)):
            continue
        ctrl = f"{r.plant_ctrl:.3f}" if np.isfinite(r.plant_ctrl) else "not run"
        P(f"| {r.city} | {r.plant_f:.2f} | {r.plant_R:.3f} | {r.plant_eff:.3f} | "
          f"{r.plant_xMDE:.0f}x | {ctrl} |")
    P("")
    P("## TEST C -- operator-scale subsampling")
    P("")
    P("| city | retain | n | mean | 95% CI | covers 1 |")
    P("|---|---:|---:|---:|---|---|")
    ncov = ntot = 0
    for r in pc.itertuples():
        for p_, s in sorted(r.sub.items()):
            ok = s["lo"] <= 1 <= s["hi"]
            ncov += ok
            ntot += 1
            P(f"| {r.city} | {p_:.1f} | {s['n']} | {s['mean']:.3f} | "
              f"[{s['lo']:.3f}, {s['hi']:.3f}] | {'yes' if ok else 'NO'} |")
    P("")
    P(f"**{ncov} of {ntot} intervals cover unity.**")
    allm = [s["mean"] for r in pc.itertuples() for s in r.sub.values()]
    if allm:
        P(f"Means span {min(allm):.3f} to {max(allm):.3f}.")
    P("")
    open(path, "w").write("\n".join(L) + "\n")
    return ncov, ntot, allm


def write_table_tex(path, pc, pooled_bias):
    has_cap = "capable" in pc.columns
    L = []
    P = L.append
    P("% generated by build_calibration_report.py -- do not hand-edit")
    P(r"\begin{tabular}{l" + ("r" if has_cap else "") + "rrrrrrr}")
    P(r"\toprule")
    ncol = 9 if has_cap else 8
    # leading blank cells: city, [capable], E[beta]  -> the literal "& " below closes the last
    pad = "& " * (2 if has_cap else 1)
    P(pad + r"& \multicolumn{2}{c}{bias} & & \multicolumn{1}{c}{planted} &")
    P(r"   \multicolumn{2}{c}{subsampled mean} \\")
    s = 4 if has_cap else 3
    P(rf"\cmidrule(lr){{{s}-{s+1}}}\cmidrule(lr){{{s+3}-{s+3}}}\cmidrule(lr){{{s+4}-{s+5}}}")
    P("city & " + ("capable & " if has_cap else "") +
      r"$E[\beta]$ & shared & independent & MDE & largest ($\times$MDE) &")
    P(r"  retain $0.2$ & retain $0.5$ \\")
    P(r"\midrule")

    def num(x):
        return f"{x:,.0f}".replace(",", "{,}")

    def row(r):
        sub = {float(k): v for k, v in r.sub.items()}
        s2 = f"{sub[0.2]['mean']:.3f}" if 0.2 in sub else "---"
        s5 = f"{sub[0.5]['mean']:.3f}" if 0.5 in sub else "---"
        pl = (f"{r.plant_eff:.3f} ({r.plant_xMDE:.0f})"
              if np.isfinite(getattr(r, "plant_eff", np.nan)) else "---")
        cap = f"{num(r.capable)} & " if has_cap else ""
        return (f"{r.city} & {cap}{num(r.E_beta)} & ${r.mean_A-1:+.4f}$ & "
                f"${r.mean_A2-1:+.4f}$ & {r.MDE:.4f} & {pl} & {s2} & {s5} \\\\")

    if has_cap:
        P(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{admitted}} "
          rf"(boundary-capable $\geq {GATE:,}$)}} \\".replace(",", "{,}"))
        for r in pc[pc.admitted].itertuples():
            P(row(r))
        if (~pc.admitted).any():
            P(r"\addlinespace")
            P(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{below threshold}}}} \\")
            for r in pc[~pc.admitted].itertuples():
                P(row(r))
    else:
        for r in pc.itertuples():
            P(row(r))
    P(r"\midrule")
    # the pooled independent-denominator bias sits in its own column, the rest dashed
    icol = 5 if has_cap else 4
    cells = [f"${pooled_bias:+.4f}$" if i == icol else "---" for i in range(2, ncol + 1)]
    P("pooled & " + " & ".join(cells) + r" \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    open(path, "w").write("\n".join(L) + "\n")


def write_checklist(path, pc, d, gt, bound, K, Kadm, slope, krange, cov, prev):
    """Every prose figure in si_calibration.tex, with its current value."""
    pool = d[d.test == "A2_independent"].R_mix
    b, lo, hi = ci(pool)
    ncov, ntot, allm = cov
    n = len(pc)
    cur = {
        "n_cities": n,
        "range_factor": pc.E_beta.max() / pc.E_beta.min(),
        "E_beta_max": pc.E_beta.max(),
        "E_beta_min": pc.E_beta.min(),
        "city_max": pc.loc[pc.E_beta.idxmax()].city,
        "city_min": pc.loc[pc.E_beta.idxmin()].city,
        "A_mean_lo": pc.mean_A.min(), "A_mean_hi": pc.mean_A.max(),
        "A_maxbias": (pc.mean_A - 1).abs().max(),
        "pooled_bias": b, "pooled_lo": lo, "pooled_hi": hi, "pooled_n": len(pool),
        "slope": slope, "k_range": krange, "K": K, "K_admitted": Kadm,
        "MDE_min": pc.MDE.min(), "MDE_max": pc.MDE.max(),
        "plant_xMDE_min": pc.plant_xMDE.min() if "plant_xMDE" in pc else np.nan,
        "plant_xMDE_max": pc.plant_xMDE.max() if "plant_xMDE" in pc else np.nan,
        "sub_n": ntot, "sub_mean_lo": min(allm) if allm else np.nan,
        "sub_mean_hi": max(allm) if allm else np.nan,
    }
    if "reldiff" in pc.columns:
        cur["reldiff_max_pct"] = 100 * pc.reldiff.max()
    if "dep_xMDE" in pc.columns:
        cur["dep_xMDE_min"] = pc.dep_xMDE.min()
        cur["dep_xMDE_max"] = pc.dep_xMDE.max()
    if gt and not gt.get("insufficient"):
        cur.update(gate_up_bias=gt["up_bias"][0], gate_lo_bias=gt["lo_bias"][0],
                   gate_t=gt["t"], gate_p=gt["p"],
                   n_admitted=len(gt["up"]), n_below=len(gt["lo"]))
    if bound:
        cur.update(bound_worst=bound["worst_bound"], bound_minratio=bound["min_ratio"],
                   bound_mincity=bound["min_city"], bound_n=bound["n"],
                   bound_frac5=100 * bound["frac_gt5"])

    SENT = {
        "n_cities": "\"We ran the full battery on N cities\" / \"across five cities\" (several places)",
        "range_factor": "\"spanning a factor of 152 in the null-expected two-path count\"",
        "E_beta_max": "\"from Natal at 1.03e6\"",
        "E_beta_min": "\"down to Macapa at 6.8e3\"",
        "A_mean_lo": "\"returned means between 0.9982 and 1.0018\"",
        "A_maxbias": "\"a largest absolute bias of 0.0018\"",
        "pooled_bias": "\"gives a bias of -0.0023\"",
        "pooled_n": "\"Pooling the resulting 25 replicates\"",
        "reldiff_max_pct": "\"reproduces the production value to within 0.22%\"",
        "n_admitted": "\"Three of the five clear the boundary-capable threshold\"",
        "gate_up_bias": "\"bias is -0.0016 ... for the three admitted cities\"",
        "gate_lo_bias": "\"and -0.0032 ... for the two excluded ones\"",
        "gate_t": "\"(Welch t=0.53, p=0.60)\"",
        "slope": "\"at a log-log slope of -0.45\"",
        "k_range": "\"still varies by a factor of 2.4\"",
        "K": "eq:mde-bound \"MDE <= 3.82/sqrt(E[beta])\"  **propagates to bound_* below**",
        "K_admitted": "\"the largest value among admitted cities alone is 2.56\"",
        "MDE_min": "\"rises from 0.0020 at Natal\"",
        "MDE_max": "\"to 0.0195 at Macapa\"",
        "bound_worst": "\"the worst-case bound at any city is 0.0186\"",
        "bound_minratio": "\"the smallest ratio anywhere is 8.3\"",
        "bound_mincity": "\"... at Balsas\"",
        "dep_xMDE_min": "\"the observed departures run from 15 to 76 times it\"",
        "plant_xMDE_min": "\"by factors of 12 to 210\"",
        "sub_n": "\"all ten mean values\"",
        "sub_mean_lo": "\"lie between 0.982 and 1.020\"",
    }
    L = []
    P = L.append
    P("# Calibration prose-figure checklist")
    P("")
    P("Every number that appears in prose in `si_calibration.tex` / `si_calibration.md`.")
    P("Regenerate after adding a city, then patch any row marked CHANGED.")
    P("")
    P("| key | previous | current | changed | sentence in the .tex |")
    P("|---|---|---|:---:|---|")
    for k, v in cur.items():
        if k in ("city_max", "city_min", "E_beta_max", "E_beta_min",
                 "A_mean_hi", "pooled_lo", "pooled_hi", "MDE_max",
                 "plant_xMDE_max", "dep_xMDE_max", "sub_mean_hi",
                 "n_below", "bound_n", "bound_frac5"):
            continue
        old = prev.get(k)
        fv = f"{v:.5g}" if isinstance(v, (int, float, np.floating)) else str(v)
        fo = ("--" if old is None else
              (f"{old:.5g}" if isinstance(old, (int, float)) else str(old)))
        same = (old is not None and
                (abs(float(v) - float(old)) < 1e-9
                 if isinstance(v, (int, float, np.floating)) else v == old))
        P(f"| `{k}` | {fo} | **{fv}** | {'' if same else '**CHANGED**'} | "
          f"{SENT.get(k, '')} |")
    P("")
    P("Paired values held in this run (not listed above, update alongside their partner):")
    for k in ("E_beta_max", "E_beta_min", "A_mean_hi", "MDE_max",
              "plant_xMDE_max", "dep_xMDE_max", "sub_mean_hi"):
        if k in cur:
            P(f"  - `{k}` = {cur[k]:.5g}")
    open(path, "w").write("\n".join(L) + "\n")
    return cur


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="calibration_all_cities.csv")
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--add-city", action="append", default=[])
    ap.add_argument("--production", default="../two_null_population_summary.csv")
    ap.add_argument("--pole", type=int, default=5)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--write-battery", action="store_true",
                    help="write the merged battery back to --battery")
    ap.add_argument("--state", default="calibration_numbers.json",
                    help="previous run's figures, for the CHANGED column")
    a = ap.parse_args()

    print("reading inputs")
    d = load_battery(a.battery, a.add, a.add_city)
    prod = load_production(a.production, a.pole)
    print(f"  battery: {len(d)} rows, {d.city.nunique()} cities")

    pc = per_city(d, prod)
    if not len(pc):
        sys.exit("no city had enough Test A draws")

    K = float(pc.k.max())
    Kadm = float(pc[pc.admitted].k.max()) if "admitted" in pc.columns and pc.admitted.any() else None
    slope = float(np.polyfit(np.log10(pc.E_beta), np.log10(pc.MDE), 1)[0])
    krange = float(pc.k.max() / pc.k.min())
    gt = gate_test(d, pc)
    bound = all_city_bound(prod, K)

    os.makedirs(a.outdir, exist_ok=True)
    rp = os.path.join(a.outdir, "RESULTS_calibration.md")
    tp = os.path.join(a.outdir, "calibration_table.tex")
    cp = os.path.join(a.outdir, "calibration_numbers.md")
    sp = os.path.join(a.outdir, a.state)

    prev = {}
    if os.path.exists(sp):
        try:
            prev = json.load(open(sp))
        except Exception:
            pass

    cov = write_results_md(rp, pc, d, gt, bound, K, Kadm, slope, krange)
    write_table_tex(tp, pc, ci(d[d.test == 'A2_independent'].R_mix)[0])
    cur = write_checklist(cp, pc, d, gt, bound, K, Kadm, slope, krange, cov, prev)
    json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in cur.items()}, open(sp, "w"), indent=1)

    print(f"\nwrote {rp}\n      {tp}\n      {cp}\n      {sp}")
    if a.write_battery:
        d.to_csv(a.battery, index=False)
        print(f"      {a.battery} (merged, {len(d)} rows)")

    print(f"\nenvelope K = {K:.3f} (set by {pc.loc[pc.k.idxmax()].city})"
          + (f", admitted-only {Kadm:.3f}" if Kadm else ""))
    if bound:
        print(f"all-city bound: worst {bound['worst_bound']:.4f}, "
              f"min ratio {bound['min_ratio']:.1f} at {bound['min_city']}, "
              f"{100*bound['frac_gt5']:.0f}% above 5x")
    changed = [k for k in cur if k in prev and
               isinstance(cur[k], (int, float, np.floating)) and
               abs(float(cur[k]) - float(prev[k])) > 1e-9]
    if changed:
        print(f"\n{len(changed)} figure(s) CHANGED since the last run -- see {cp}")
        for k in changed:
            print(f"  {k}: {prev[k]:.5g} -> {float(cur[k]):.5g}")


if __name__ == "__main__":
    main()
    