#!/usr/bin/env python3
"""
make_si_tables.py -- build the two Supplement tables the main text points at.

Supplementary Table S4, per-band position (--position)
    Per-stratum share of the pole-to-pole geodesic interior, its ratio to the
    label-permutation null, and the fraction of cities falling below the null, for each
    stratification. This is the table cited twice in the main text, from the Fig. 1
    caption and from the body at l.126.

Supplementary Table S5, the ladder under equal-area (--ladder)
    The eight Gould-Fernandez roles under the equal-area stratification, matching Table 2
    of the main text. Figure 4 already plots this series in amber, so the data exists and
    only the tabulation is missing.

USAGE
    # 1. one tidy file per stratification, from the Brazilian sweep CSVs only
    python3 position_tidy.py --wide <equal-population sweeps> --out position_summary_population.csv
    python3 position_tidy.py --wide <equal-area sweeps>       --out position_summary_area.csv

    # 2. both tables, checked against the numbers already in the main text
    python3 make_si_tables.py \\
        --position position_summary_population.csv equal-population \\
        --position position_summary_area.csv       equal-area \\
        --ladder   gf_roles/gf_roles_population.csv equal-population \\
        --ladder   gf_roles/gf_roles_antenna.csv    equal-area \\
        --check

Each --position and --ladder takes a CSV and the stratification label to print for it.
Labels must be exactly "equal-population" or "equal-area". The ladder table tabulates the
equal-area file; the equal-population file is used only by --check, against Table 2.
Position rows are filtered by the same reachability gate as the paper (>= 50% of sampled
Q1->Q5 pairs mutually reachable).

The emitted files carry \\label{stab:position} and \\label{stab:ladder-area}, so pasting
them into the Supplement resolves \\SItab{4} and \\SItab{5}.
"""
import argparse
import sys

import numpy as np
import pandas as pd

ROLES = [
    ("coordinator", "coordinator (insider broker)", "same"),
    ("itinerant_above", "itinerant, broker above", "same"),
    ("itinerant_below", "itinerant, broker below", "same"),
    ("gatekeeper", "gatekeeper", "diff"),
    ("representative", "representative", "diff"),
    ("liaison_intermediate", "liaison, intermediate broker", "diff"),
    ("liaison_extremal_above", "liaison, broker above both", "diff"),
    ("liaison_extremal_below", "liaison, broker below both", "diff"),
]


def pick(df, *names, required=True, what=""):
    for n in names:
        if n in df.columns:
            return n
    if required:
        print(f"\nERROR: no column for {what}. Tried {names}.")
        print("Columns present:")
        for c in df.columns:
            print(f"    {c}")
        sys.exit(1)
    return None


# ------------------------------------------------------------------ Table S4

# Values the main text already reports (Results, 'No stratum is over-represented'),
# used by --check to confirm the table is built from the same data as the paper.
MAIN_TEXT_POSITION = {
    "equal-area":       dict(n=92, ratio=[1.01, 0.98, 0.98, 1.00, 1.00],
                             below={2: 62, 3: 65}),
    "equal-population": dict(n=95, ratio=[1.01, 0.97, 0.97, 0.99, 1.01],
                             below={2: 71, 3: 68, 4: 56}),
}
GATE = 0.5   # cities enter when >= half of sampled Q1->Q5 pairs are mutually reachable


def position_table(specs, check=False):
    rows = []
    for path, label in specs:
        d = pd.read_csv(path)
        print(f"\n{path}  ({len(d)} rows, {d['city'].nunique() if 'city' in d else '?'} cities)")
        c_band = pick(d, "stratum", "band", "quintile", "k", "q", what="stratum")
        c_obs = pick(d, "rho_obs", "rho", "share", "observed", what="observed share")
        c_null = pick(d, "rho_null_mean", "rho_null", "null_mean", "null", "perm_mean",
                      what="null expectation")
        c_city = pick(d, "city", required=False, what="city")
        print(f"  stratum={c_band}  observed={c_obs}  null={c_null}  city={c_city}")

        # position_tidy.py writes rho_null_mean = NaN when the sweep CSV carried only
        # z-scores. Without the null mean there is no ratio and no '% below null'.
        if not np.isfinite(d[c_null]).any():
            sys.exit(
                f"\nERROR: every value of '{c_null}' in {path} is empty.\n"
                "position_tidy.py fills it only when the sweep CSV has ALL fifteen columns\n"
                "brok_share_null_{mean,lo,hi}_Q1..Q5 -- if lo or hi is missing, the mean is\n"
                "dropped too, even when present. Check with:\n"
                "    head -1 <sweep>.csv | tr ',' '\\n' | grep -c 'brok_share_null_'   # want 15\n"
                "If they are absent, the sweep was run before brokerage_profile emitted the\n"
                "null mean, and the sweep has to be re-run (or the null mean recovered from\n"
                "wherever Fig. 1's teal band was computed).")

        # the reachability gate used throughout the paper
        if {"pairs_reached", "pairs_tried"} <= set(d.columns):
            reach = d.pairs_reached / d.pairs_tried.replace(0, np.nan)
            per_city = reach.groupby(d[c_city]).first()
            keep = set(per_city[per_city >= GATE].index)
            print(f"  reachability gate (>= {GATE:.0%} of pairs): "
                  f"{len(keep)} of {per_city.size} cities pass")
            d = d[d[c_city].isin(keep)]
        else:
            print("  WARNING: no pairs_reached/pairs_tried columns -- gate NOT applied")

        d = d[np.isfinite(d[c_obs]) & np.isfinite(d[c_null]) & (d[c_null] > 0)]
        d = d.assign(_ratio=d[c_obs] / d[c_null], _below=d[c_obs] < d[c_null])
        for b, g in d.groupby(c_band):
            try:
                bi = int(b)
            except (TypeError, ValueError):
                bi = str(b).strip().lstrip("Qq")
                bi = int(bi) if bi.isdigit() else b
            rows.append(dict(strat=label, band=bi, n=len(g),
                             rho=g[c_obs].median(), ratio=g._ratio.median(),
                             below=100 * g._below.mean()))
    t = pd.DataFrame(rows)
    if not len(t):
        sys.exit("no position rows produced")

    print("\n=== Supplementary Table S4, per-band position ===")
    print(f"{'stratification':<20}{'band':>6}{'n':>6}{'rho':>9}{'obs/null':>10}{'% below':>10}")
    for r in t.itertuples():
        print(f"{r.strat:<20}{'Q'+str(r.band):>6}{r.n:>6}{r.rho:>9.3f}"
              f"{r.ratio:>10.2f}{r.below:>10.0f}")

    if check:
        print("\n=== --check: against the numbers already in the main text ===")
        ok = True
        for lab, exp in MAIN_TEXT_POSITION.items():
            sub = t[t.strat == lab].set_index("band")
            if not len(sub):
                print(f"  {lab}: not in this run, skipped")
                continue
            n = int(sub.n.max())
            flag = "ok" if n == exp["n"] else "MISMATCH"
            ok &= flag == "ok"
            print(f"  {lab}: cities {n} (paper {exp['n']})  {flag}")
            for k, want in enumerate(exp["ratio"], 1):
                got = round(float(sub.loc[k, "ratio"]), 2) if k in sub.index else None
                flag = "ok" if got == want else "MISMATCH"
                ok &= flag == "ok"
                print(f"     Q{k} ratio {got} (paper {want:.2f})  {flag}")
            for k, want in exp["below"].items():
                got = round(float(sub.loc[k, "below"])) if k in sub.index else None
                flag = "ok" if got == want else "MISMATCH"
                ok &= flag == "ok"
                print(f"     Q{k} % below {got} (paper {want})  {flag}")
        print("  ALL MATCH -- table and text agree." if ok else
              "  MISMATCHES ABOVE -- table and text disagree; resolve before pasting.")

    L = [
        r"% generated by make_si_tables.py --position",
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{\textbf{Occupancy of the pole-to-pole geodesic interior, per stratum.} "
        r"For each stratification and each stratum, the median across cities of the share "
        r"of the interior of directed $Q_1 \to Q_5$ shortest paths falling on that stratum "
        r"($\rho_k$), its ratio to the degree-stratified label-permutation null, and the "
        r"percentage of cities in which the observed share falls below the null. A ratio "
        r"of one indicates a stratum occupying the connecting routes exactly as often as "
        r"its abundance and connectivity imply. The bridging account predicts a ratio "
        r"above one on the middle strata. We observe ratios within $3\%$ of one in every "
        r"stratum, with the poles at or marginally above one and the lower-middle "
        r"modestly below it. Cities enter a row when at "
        r"least half of the sampled pole pairs are mutually reachable.}",
        r"\label{stab:position}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"stratification & stratum & $n$ & median $\rho_k$ & observed/null & \% below null \\",
        r"\midrule",
    ]
    prev = None
    for r in t.itertuples():
        if prev is not None and r.strat != prev:
            L.append(r"\addlinespace")
        prev = r.strat
        L.append(f"{r.strat} & $Q_{r.band}$ & {r.n} & {r.rho:.3f} & "
                 f"{r.ratio:.2f} & {r.below:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ Table S5

# Main-text Table 2 (equal-population), used by --check.
MAIN_TEXT_LADDER = {
    "coordinator": (96, 1.27, 100, 0.12),
    "itinerant_above": (83, 1.22, 100, 0.11),
    "itinerant_below": (81, 1.38, 100, 0.09),
    "gatekeeper": (93, 0.88, 0, 0.17),
    "representative": (90, 0.92, 0, 0.17),
    "liaison_intermediate": (71, 0.91, 0, 0.11),
    "liaison_extremal_above": (70, 0.92, 0, 0.11),
    "liaison_extremal_below": (72, 0.91, 6, 0.10),
}


def ladder_table(specs, check=False):
    blocks = []
    for path, label in specs:
        d = pd.read_csv(path)
        print(f"\n{path}  ({len(d)} rows)")
        if not any(c.startswith("Rmix_") for c in d.columns):
            print("ERROR: no Rmix_<role> columns found. Columns present:")
            for c in d.columns:
                print(f"    {c}")
            sys.exit("Send me this list and I'll map the columns.")
        rows = []
        for key, pretty, grp in ROLES:
            rc, oc = f"Rmix_{key}", f"obsshare_{key}"
            if rc not in d.columns:
                print(f"  missing column {rc}, skipping role")
                continue
            v = d[rc].dropna()
            o = d[oc].dropna() if oc in d.columns else pd.Series(dtype=float)
            if not len(v):
                continue
            rows.append(dict(key=key, role=pretty, grp=grp, n=len(v),
                             med=v.median(), frac=100 * (v > 1).mean(),
                             share=o.median() if len(o) else np.nan))
        blocks.append((label, pd.DataFrame(rows)))
        print(f"  {'role':<30}{'n':>5}{'med Rmix':>11}{'% >1':>8}{'obs share':>11}")
        for r in rows:
            print(f"  {r['role']:<30}{r['n']:>5}{r['med']:>11.2f}"
                  f"{r['frac']:>8.0f}{r['share']:>11.2f}")

    if not blocks:
        sys.exit("no ladder rows produced")

    if check:
        pop = [t for lab, t in blocks if lab == "equal-population"]
        print("\n=== --check: equal-population ladder against main-text Table 2 ===")
        if not pop:
            print("  give the equal-population file as a second --ladder to run this check")
        else:
            ok = True
            tp = pop[0].set_index("key")
            for key, (n, med, frac, sh) in MAIN_TEXT_LADDER.items():
                if key not in tp.index:
                    print(f"  {key:<24} missing  MISMATCH"); ok = False; continue
                r = tp.loc[key]
                got = (int(r.n), round(float(r.med), 2), int(round(float(r.frac))),
                       round(float(r.share), 2))
                flag = "ok" if got == (n, med, frac, sh) else "MISMATCH"
                ok &= flag == "ok"
                print(f"  {key:<24} {got}  paper {(n, med, frac, sh)}  {flag}")
            print("  ALL MATCH -- the file reproduces Table 2." if ok else
                  "  MISMATCHES ABOVE -- this file is not what Table 2 was built from.")

    # tabulate equal-area; the main text already reports equal-population as Table 2
    pick_area = [b for b in blocks if b[0] == "equal-area"]
    label, t = pick_area[0] if pick_area else blocks[0]
    L = [
        r"% generated by make_si_tables.py --ladder",
        r"\begin{table}[ht]", r"\centering", r"\small",
        rf"\caption{{\textbf{{Brokerage roles across the income ladder under "
        rf"{label} stratification.}} The replication of main-text Table~2 under the "
        r"second interpretable stratification. For each role, the number of cities in "
        r"which it is estimable, the median mixing-adjusted enrichment "
        r"$R_{\mathrm{mix}}$, the fraction of cities in which it exceeds unity, and its "
        r"median share of observed two-paths. Roles are grouped by whether the two "
        r"connected residents share a stratum. The median of every role lies on the same "
        r"side of unity as under equal-population (main-text Table~2), and the "
        r"intermediate liaison, the direct expression of the bridging hypothesis, is "
        r"enriched in none of the cities, so the pattern does not depend on how income "
        r"is banded. Fewer cities are estimable than under equal-population because the "
        r"equal-area top stratum is the scarcest.}",
        r"\label{stab:ladder-area}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r" & role & $n$ & med.\ $R_{\mathrm{mix}}$ & \% $>1$ & obs.\ share \\",
        r"\midrule",
    ]
    for grp, head in (("same", r"\multirow{3}{*}{\shortstack[l]{endpoints\\same stratum}}"),
                      ("diff", r"\multirow{5}{*}{\shortstack[l]{endpoints\\differ}}")):
        sub = t[t.grp == grp]
        if not len(sub):
            continue
        if grp == "diff":
            L.append(r"\midrule")
        L.append(head)
        for r in sub.itertuples():
            sh = f"{r.share:.2f}" if r.share == r.share else "---"
            L.append(f" & {r.role} & {r.n} & {r.med:.2f} & {r.frac:.0f} & {sh} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print(f"\nTabulating the {label} ladder as Supplementary Table S5.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", nargs=2, action="append", metavar=("CSV", "LABEL"))
    ap.add_argument("--ladder", nargs=2, action="append", metavar=("CSV", "LABEL"))
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--check", action="store_true",
                    help="compare against the numbers already printed in the main text")
    a = ap.parse_args()
    if not a.position and not a.ladder:
        sys.exit("give --position and/or --ladder")

    import os
    os.makedirs(a.out_dir, exist_ok=True)
    # Build each table completely BEFORE opening its file, so a run that stops early
    # (bad input, missing null column) never truncates a good table from a previous run.
    if a.position:
        tex = position_table(a.position, check=a.check)
        p = os.path.join(a.out_dir, "stab_position.tex")
        with open(p, "w") as fh:
            fh.write(tex)
        print(f"\nwrote {p}  -> Supplementary Table S4")
    if a.ladder:
        tex = ladder_table(a.ladder, check=a.check)
        p = os.path.join(a.out_dir, "stab_ladder_area.tex")
        with open(p, "w") as fh:
            fh.write(tex)
        print(f"\nwrote {p}  -> Supplementary Table S5")
    print("\nThe emitted tables need \\usepackage{booktabs} and, for S5, "
          "\\usepackage{multirow}.")


if __name__ == "__main__":
    main()


