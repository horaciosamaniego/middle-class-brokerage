#!/usr/bin/env python3
"""
mechanism_pipeline.py -- Section 2 (MECHANISM) across three income stratifications.

Answers Part 2: given that no stratum is a privileged PATH mediator (Position),
what KIND of hands-on intermediation happens at the boundary -- bridging by the
middle, or boundary control by a pole over its own edge? Operationalised as the
Gould-Fernandez pole-anchored two-path census, adjudicated by TWO nulls:
    R_perm  observed / class-blind (label-permutation) null      -> the naive ratio
    R_mix   observed / stratification-preserving (mixing) null    -> the real test
The paper's methodological point is the COLLAPSE: R_perm > 1 ("significant closure")
but R_mix < 1 once contact volume is held fixed -- the naive closure is an artifact.

Run under three stratifications (quintile_nation_antenna/_population/_width) and BOTH
poles (Q5 rich boundary control and Q1 poor), so the reader sees whether the collapse
is robust to how the poles are defined and symmetric across poles.

This module builds FIGURES and TABLES from the per-stratification two-null summary
CSVs emitted by run_two_nulls.py. Expected summary columns (subset):
  city, pole, boundary_capable, power_ok, mix_consistent,
  observed, R_perm, p_perm, R_mix, p_mix, mix_ci_lo, mix_ci_hi, mix_null_mean, verdict

Outputs:
  MAIN  fig_mechanism_collapse.{svg,pdf}   R_perm->R_mix dumbbells, small-multiple x strat (pole 5)
  MAIN  fig_mechanism_bothpoles.{svg,pdf}  R_mix at Q5 vs Q1, per stratification
  MAIN  table_mechanism_summary.csv        per (strat,pole): n powered, median R_perm/R_mix, collapse
  ANNEX table_mechanism_percity.csv        per city: both ratios, p's, capability, consistency

Usage:
  python mechanism_pipeline.py --summary antenna=two_null_antenna.csv \
     population=two_null_population.csv width=two_null_width.csv \
     --min_capable 5000 --outdir mech_figs --fmt svg pdf
"""
import argparse
import os
import numpy as np
import pandas as pd

INK, TEAL, AMBER, GREY = "#1a1a1a", "#2a7f7f", "#d9942b", "#9aa0a6"
MODES = ["antenna", "population", "width"]
BOOLS = ("power_ok", "mix_consistent")


def load(path):
    d = pd.read_csv(path)
    for c in BOOLS:
        if c in d:
            d[c] = d[c].astype(str).str.strip().str.lower().isin(["true", "1"])
    # dedupe (city,pole) preferring mix_consistent
    d["_k"] = d["mix_consistent"].astype(int) if "mix_consistent" in d else 0
    d = (d.sort_values(["city", "pole", "_k"], ascending=[True, True, False])
           .drop_duplicates(["city", "pole"], keep="first").drop(columns="_k"))
    return d


def powered(d, pole, min_capable):
    m = (d.pole == pole)
    if "power_ok" in d:
        m &= d.power_ok
    if "mix_consistent" in d:
        m &= d.mix_consistent
    if "boundary_capable" in d:
        m &= d.boundary_capable >= min_capable
    return d[m].copy()


# ------------------------------------------------------------------ MAIN collapse fig
def fig_collapse(sums, pole, min_capable, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in MODES if m in sums]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.5 * len(modes), 4.6), sharex=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        p = powered(sums[mode], pole, min_capable).sort_values("R_mix")
        if not len(p):
            ax.text(0.5, 0.5, "no powered cities\n(pole too thin)", transform=ax.transAxes,
                    ha="center", va="center", color=GREY, fontsize=9)
            ax.set_title(mode, fontsize=10); ax.set_xlabel("ratio")
            ax.axvline(1.0, color=INK, lw=1.0, ls="--")
            continue
        y = np.arange(len(p))
        for yi, (_, r) in zip(y, p.iterrows()):
            ax.plot([r.R_mix, r.R_perm], [yi, yi], color=GREY, lw=1.0, zorder=1)
        ax.scatter(p.R_perm, y, s=30, color=AMBER, edgecolor=INK, linewidth=0.4,
                   zorder=3, label=r"$R_{\rm perm}$ (class-blind)")
        ax.scatter(p.R_mix, y, s=30, color=TEAL, edgecolor=INK, linewidth=0.4,
                   zorder=3, label=r"$R_{\rm mix}$ (strat-preserving)")
        ax.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
        ax.set_yticks(y); ax.set_yticklabels(p.city, fontsize=6)
        ax.set_title(f"{mode}  (n={len(p)})", fontsize=10)
        ax.set_xlabel("observed / null")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(frameon=True, fontsize=7, loc="lower right")
    fig.suptitle(f"boundary control at Q{pole}: naive closure collapses under the mixing null",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for pth in paths:
        fig.savefig(pth, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ MAIN both-poles fig
def fig_bothpoles(sums, min_capable, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in MODES if m in sums]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.3 * len(modes), 4.0), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
        for xi, pole, col in [(0, 5, TEAL), (1, 1, AMBER)]:
            p = powered(sums[mode], pole, min_capable)
            if not len(p):
                continue
            jitter = (np.random.default_rng(xi).uniform(-0.08, 0.08, len(p)))
            ax.scatter(np.full(len(p), xi) + jitter, p.R_mix, s=24, color=col,
                       edgecolor=INK, linewidth=0.3, alpha=0.8, zorder=3)
            ax.hlines(p.R_mix.median(), xi - 0.2, xi + 0.2, color=INK, lw=2, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Q5 (rich)", "Q1 (poor)"])
        ax.set_xlim(-0.5, 1.5); ax.set_title(mode, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$R_{\rm mix}$ (boundary control / mixing null)")
    fig.tight_layout()
    for pth in paths:
        fig.savefig(pth, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables
def table_summary(sums, min_capable):
    rows = []
    for mode, d in sums.items():
        for pole in (5, 1):
            p = powered(d, pole, min_capable)
            if not len(p):
                rows.append(dict(mode=mode, pole=f"Q{pole}", n_powered=0)); continue
            rows.append(dict(mode=mode, pole=f"Q{pole}", n_powered=len(p),
                             median_R_perm=round(float(p.R_perm.median()), 3),
                             median_R_mix=round(float(p.R_mix.median()), 3),
                             frac_Rperm_gt1=round(float((p.R_perm > 1).mean()), 3),
                             frac_Rmix_lt1=round(float((p.R_mix < 1).mean()), 3),
                             frac_pmix_sig=round(float((p.p_mix < 0.05).mean()), 3)
                             if "p_mix" in p else np.nan))
    return pd.DataFrame(rows)


def table_percity(sums, min_capable):
    rows = []
    for mode, d in sums.items():
        for pole in (5, 1):
            p = powered(d, pole, min_capable)
            for _, r in p.iterrows():
                rows.append(dict(mode=mode, pole=f"Q{pole}", city=r.city,
                                 boundary_capable=int(r.get("boundary_capable", 0)),
                                 R_perm=round(float(r.R_perm), 3), p_perm=r.get("p_perm", np.nan),
                                 R_mix=round(float(r.R_mix), 3), p_mix=r.get("p_mix", np.nan),
                                 mix_consistent=bool(r.get("mix_consistent", True))))
    return pd.DataFrame(rows)


def _paths(outdir, name, fmts):
    return [os.path.join(outdir, f"{name}.{f}") for f in fmts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", required=True, help="mode=two_null_<mode>.csv")
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--fmt", nargs="*", default=["svg", "pdf"], choices=["png", "svg", "pdf"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    specs = dict(s.split("=", 1) for s in a.summary)
    sums = {m: load(p) for m, p in specs.items()}
    for m, d in sums.items():
        n5 = len(powered(d, 5, a.min_capable)); n1 = len(powered(d, 1, a.min_capable))
        print(f"  {m:10s}: powered Q5={n5}  Q1={n1}  (min_capable={a.min_capable})")

    fig_collapse(sums, 5, a.min_capable, _paths(a.outdir, "fig_mechanism_collapse", a.fmt))
    print("wrote fig_mechanism_collapse (main, pole Q5)")
    fig_bothpoles(sums, a.min_capable, _paths(a.outdir, "fig_mechanism_bothpoles", a.fmt))
    print("wrote fig_mechanism_bothpoles (main)")
    table_summary(sums, a.min_capable).to_csv(
        os.path.join(a.outdir, "table_mechanism_summary.csv"), index=False)
    print("wrote table_mechanism_summary.csv (main)")
    table_percity(sums, a.min_capable).to_csv(
        os.path.join(a.outdir, "table_mechanism_percity.csv"), index=False)
    print("wrote table_mechanism_percity.csv (annex)")
    print("\ndone.")
