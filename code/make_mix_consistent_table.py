#!/usr/bin/env python3
"""
make_mix_consistent_table.py -- build Supplementary Table stab:mix-consistent.

WHAT IT ANSWERS
The calibration battery (Supplementary Section sec:si-calibration) shows the swap-based
null denominator agrees with the closed-form degree-corrected expectation to within 0.9%
in six cities. A referee will ask whether that holds across the WHOLE analysis, not only
the calibrated subset. This builds the table that answers it, summarising the distribution
of the relative swap-versus-analytic discrepancy per stratification and pole.

It emits a compact distribution table rather than one row per city, because a per-city
listing would run to hundreds of rows and says less.

USAGE (on wawa, from the Analisis2026 directory)

  python3 make_mix_consistent_table.py \
      --summary two_null_population_summary.csv equal-population \
      --summary two_null_antenna_summary.csv    equal-area \
      --out stab_mix_consistent.tex

  # if only one stratification is to hand
  python3 make_mix_consistent_table.py --summary two_null_population_summary.csv equal-population

Pass each file with the label it should carry in the table. The script prints a plain-text
version to the terminal and writes the LaTeX table, which already carries
\\label{stab:mix-consistent}, so pasting it into the Supplement restores the reference.

COLUMN DETECTION
The discrepancy may be stored directly or may need computing from the two denominators.
The script looks for, in order:
  1. an explicit relative-discrepancy column (mix_reldiff, swap_analytic_reldiff, ...)
  2. an analytic denominator column (analytic_mean, mix_null_analytic, ...) alongside
     mix_null_mean, from which it computes |swap - analytic| / analytic
If neither is found it prints the available columns and stops, rather than guessing.
"""
import argparse
import sys

import numpy as np
import pandas as pd

RELDIFF_NAMES = ["mix_reldiff", "swap_analytic_reldiff", "reldiff", "mix_rel_diff",
                 "analytic_reldiff", "swap_vs_analytic", "mix_consistency_reldiff"]
ANALYTIC_NAMES = ["analytic_expected", "analytic_mean", "mix_null_analytic", "analytic",
                  "dcsbm_mean", "mix_analytic_mean", "analytic_expectation"]
SWAP_NAMES = ["mix_null_mean", "swap_mean", "mix_null_swap"]


def get_reldiff(d, path):
    for c in RELDIFF_NAMES:
        if c in d.columns:
            print(f"  {path}: using existing column '{c}'")
            return d[c].abs()
    an = next((c for c in ANALYTIC_NAMES if c in d.columns), None)
    sw = next((c for c in SWAP_NAMES if c in d.columns), None)
    if an and sw:
        print(f"  {path}: computing |{sw} - {an}| / {an}")
        return (d[sw] - d[an]).abs() / d[an].replace(0, np.nan)
    if "R_mix" in d.columns and "R_mix_analytic" in d.columns:
        # R_mix = obs/swap and R_mix_analytic = obs/analytic, so their ratio is
        # analytic/swap and |1 - ratio| is the same relative discrepancy.
        print(f"  {path}: computing |1 - R_mix/R_mix_analytic| (equivalent form)")
        return (1.0 - d.R_mix / d.R_mix_analytic.replace(0, np.nan)).abs()
    print(f"\nERROR: no discrepancy column found in {path}.")
    print("Columns present:")
    for c in d.columns:
        print(f"    {c}")
    print("\nRe-run passing the right name via --reldiff-col or --analytic-col.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs=2, action="append", metavar=("CSV", "LABEL"),
                    required=True, help="summary CSV and the stratification label for it")
    ap.add_argument("--reldiff-col", default=None)
    ap.add_argument("--analytic-col", default=None)
    ap.add_argument("--gate", type=int, default=5000)
    ap.add_argument("--out", default="stab_mix_consistent.tex")
    a = ap.parse_args()

    if a.reldiff_col:
        RELDIFF_NAMES.insert(0, a.reldiff_col)
    if a.analytic_col:
        ANALYTIC_NAMES.insert(0, a.analytic_col)

    rows = []
    print("reading inputs")
    for path, label in a.summary:
        d = pd.read_csv(path)
        d["_rd"] = get_reldiff(d, path)
        if "boundary_capable" in d.columns:
            d = d[d.boundary_capable >= a.gate]
        for pole in sorted(d.pole.dropna().unique()):
            g = d[(d.pole == pole) & d._rd.notna()]
            if not len(g):
                continue
            cons = g.mix_consistent.sum() if "mix_consistent" in g.columns else np.nan
            # the R_mix comparison is taken over the ANALYSIS set (gated and consistent),
            # while the discrepancy distribution covers every gated city so the reader
            # can see what the consistency flag is excluding.
            aset = g[g.mix_consistent] if "mix_consistent" in g.columns else g
            rs = ra = pb = np.nan
            if {"R_mix", "R_mix_analytic"} <= set(aset.columns) and len(aset):
                rs, ra = aset.R_mix.median(), aset.R_mix_analytic.median()
                pb = 100 * ((aset.R_mix < 1) & (aset.R_mix_analytic < 1)).mean()
                mxa = aset.R_mix_analytic.max()
            else:
                mxa = np.nan
            rows.append(dict(
                strat=label, pole=int(pole), n=len(g),
                med=g._rd.median(), q25=g._rd.quantile(.25), q75=g._rd.quantile(.75),
                p90=g._rd.quantile(.90), mx=g._rd.max(),
                n_cons=int(cons) if cons == cons else np.nan,
                pct_cons=100 * cons / len(g) if cons == cons else np.nan,
                r_swap=rs, r_an=ra, pct_both=pb, max_an=mxa))
    t = pd.DataFrame(rows)
    if not len(t):
        sys.exit("no rows produced")

    print("\n=== swap-versus-analytic relative discrepancy, powered cities ===")
    print(f"{'stratification':<18}{'pole':>5}{'n':>5}{'median':>9}{'p90':>9}{'max':>9}"
          f"{'consistent':>13}{'R swap':>9}{'R analy':>9}{'both<1':>8}")
    for r in t.itertuples():
        cons = f"{r.n_cons}/{r.n} ({r.pct_cons:.0f}%)" if r.n_cons == r.n_cons else "--"
        rr = (f"{r.r_swap:>9.3f}{r.r_an:>9.3f}{r.pct_both:>7.0f}%"
              if r.r_swap == r.r_swap else " " * 25)
        print(f"{r.strat:<18}{r.pole:>5}{r.n:>5}{r.med:>9.4f}{r.p90:>9.4f}{r.mx:>9.4f}"
              f"{cons:>13}{rr}")
    print(f"\nworst discrepancy anywhere : {t.mx.max():.4f}")
    if t.pct_both.notna().any():
        print(f"below unity under BOTH     : {t.pct_both.min():.0f}% (minimum across groups)")
        print(f"largest analytic R_mix     : {t.max_an.max():.3f}")
        print("\nThe conclusion does not depend on which denominator is used."
              if t.pct_both.min() == 100 else
              "\nWARNING: the conclusion is NOT denominator-invariant. Do not paste.")

    L = [
        r"% generated by make_mix_consistent_table.py -- do not hand-edit",
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{\textbf{Agreement between the swap-based and closed-form null "
        r"expectations across the whole analysis.} The mixing null is sampled by "
        r"constrained-configuration rewiring, and its expected cross-pole two-path count is "
        r"also available in closed form under the degree-corrected stochastic block model, "
        r"Eq.~\eqref{eq:dcsbm-bc}. The relative discrepancy between the two is computed for "
        r"every city and pole that clears the boundary-capable threshold, and summarised here "
        r"by stratification and pole. Agreement establishes that the sampler has mixed and "
        r"that the denominator is not an artefact of the rewiring chain. Cities in which the "
        r"two disagree are treated as unmixed and excluded (Methods, "
        r"Section~\ref{sec:power}). Agreement is closest in the calibrated cities of "
        r"Table~\ref{stab:calibration}, which agree to within $0.9\%$, and looser across "
        r"the analysis as a whole. The final columns show that this looseness does not "
        r"affect the result. The median \Rmix{} computed against each denominator is "
        r"reported for the analysis set, together with the share of cities falling below "
        r"unity under both, so the reader can see that the conclusion does not depend on "
        r"which expectation is used. Discrepancy statistics cover every city clearing the "
        r"capability threshold, while the \Rmix{} columns cover the analysis set, that is "
        r"those cities that also pass the consistency check.}",
        r"\label{stab:mix-consistent}",
        r"\begin{tabular}{lrrrrrcrrr}", r"\toprule",
        r" & & & \multicolumn{3}{c}{swap-vs-analytic discrepancy} & &"
        r" \multicolumn{3}{c}{median \Rmix{}} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){8-10}",
        r"stratification & pole & $n$ & median & p90 & max & consistent &"
        r" swap & analytic & both $<1$ \\",
        r"\midrule",
    ]
    prev = None
    for r in t.itertuples():
        if prev is not None and r.strat != prev:
            L.append(r"\addlinespace")
        prev = r.strat
        cons = (rf"{r.n_cons}/{r.n} ({r.pct_cons:.0f}\%)"
                if r.n_cons == r.n_cons else "---")
        rr = (f"{r.r_swap:.3f} & {r.r_an:.3f} & {r.pct_both:.0f}\\%"
              if r.r_swap == r.r_swap else "--- & --- & ---")
        L.append(f"{r.strat} & $Q_{r.pole}$ & {r.n} & {r.med:.4f} & "
                 f"{r.p90:.4f} & {r.mx:.4f} & {cons} & {rr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {a.out}")
    print("\nPaste it into the Supplement, then restore the cross-reference by replacing")
    print("the closing sentence of sec:si-calibration with:\n")
    print(r"  swap-versus-analytic agreement that underpins the first of these checks is "
          r"likewise computed")
    print(r"  for every city in the analysis and reported in "
          r"Table~\ref{stab:mix-consistent}.")


if __name__ == "__main__":
    main()
    