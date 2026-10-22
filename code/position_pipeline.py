#!/usr/bin/env python3
"""
position_pipeline.py -- Section 1 (POSITION) across three income stratifications.

Answers Part 1 of the research question -- does a single stratum disproportionately
sit on the routes linking the income poles, and is it the middle? -- and tests the
robustness of the answer to how the poles are DEFINED, by running it under three
quintile assignments now carried on the graph nodes:
    antenna     equal number of neighbourhoods per band   (quintile_nation_antenna)
    population  equal number of people per band            (quintile_nation_population)
    width       equal income ranges per band               (quintile_nation_width)

Two analyses (each already implemented in Brokerage_constraint_pipeline.py and
necessity_sweep.py; this file only ORCHESTRATES them across the three vp's and
assembles paper outputs):
    OCCUPANCY  rho_k : each stratum's share of interior nodes on Q1->Q5 geodesics,
                       vs the degree-stratified permutation null.
    NECESSITY  Delta_k : loss of pole-to-pole efficiency when a middle stratum is
                       removed, vs an equal-sized random removal.

This module builds the FIGURES and TABLES from the per-stratification tidy CSVs.
The heavy sweeps are launched by the accompanying run_position.sh (below), which
just calls the existing sweep/necessity code three times with different --vp.

Outputs:
  MAIN  fig_position_occupancy.{svg,pdf}   3-panel small multiple (antenna/pop/width)
  MAIN  fig_position_necessity.{svg,pdf}   3-panel (width dropped -> greyed if degenerate)
  MAIN  table_band_income_ranges.csv       mode x band -> income min/max + pop share
  ANNEX table_position_summary.csv         per stratification: n powered, median rho, z sign
  ANNEX fig_position_bipolarity.{svg,pdf}  fraction-reached distribution (near-bipolar tail)

Usage (after the three sweeps have produced position_summary_<mode>.csv):
  python position_pipeline.py --summary antenna=position_summary_antenna.csv \
       population=position_summary_population.csv width=position_summary_width.csv \
       --income antenas_income_areal.csv --min_frac 0.5 --outdir figs --fmt svg pdf
"""
import argparse
import os
import numpy as np
import pandas as pd

INK, TEAL, AMBER, GREY = "#1a1a1a", "#2a7f7f", "#d9942b", "#9aa0a6"
QLAB = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"}
MODES = ["antenna", "population", "width"]


# ------------------------------------------------------------------ gating + loading
def gate(d, min_frac, min_pairs):
    """Keep well-sampled cities: fraction_reached >= min_frac if pairs_tried present,
    else pairs_reached >= min_pairs. Returns (gated_df, n_kept, n_all, how)."""
    n_all = d.city.nunique()
    if {"pairs_reached", "pairs_tried"} <= set(d.columns) and d.pairs_tried.notna().any():
        frac = (d.pairs_reached / d.pairs_tried).replace([np.inf, -np.inf], np.nan)
        keep = d.assign(_f=frac).groupby("city")["_f"].transform("max") >= min_frac
        return d[keep], d[keep].city.nunique(), n_all, f"fraction>={min_frac}"
    if "pairs_reached" in d.columns:
        keep = d.groupby("city")["pairs_reached"].transform("max") >= min_pairs
        return d[keep], d[keep].city.nunique(), n_all, f"pairs>={min_pairs} (no pairs_tried)"
    return d, n_all, n_all, "ungated"


def load_summaries(specs, min_frac, min_pairs):
    out = {}
    for mode, path in specs.items():
        d = pd.read_csv(path)
        g, nk, na, how = gate(d, min_frac, min_pairs)
        print(f"  {mode:10s}: {nk}/{na} cities gated ({how})")
        out[mode] = g
    return out


# ------------------------------------------------------------------ MAIN occupancy fig
def fig_occupancy(sums, paths, bridging=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in MODES if m in sums]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.3 * len(modes), 4.0), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        d = sums[mode]
        nb = d.groupby("stratum").agg(lo=("rho_null_lo", "median"), hi=("rho_null_hi", "median"),
                                      mean=("rho_null_mean", "median"))
        ax.fill_between(nb.index, nb.lo, nb.hi, color=TEAL, alpha=0.15, zorder=0,
                        label="perm. null 95%")
        ax.plot(nb.index, nb["mean"], color=TEAL, lw=1.0, ls="--", alpha=0.85, zorder=2,
                label="null expectation")
        grp = d.groupby("stratum")["rho_obs"]
        med, q1, q3 = grp.median(), grp.quantile(.25), grp.quantile(.75)
        ax.fill_between(med.index, q1.values, q3.values, color=AMBER, alpha=0.18, zorder=1,
                        label="obs. IQR")
        ax.plot(med.index, med.values, color=AMBER, lw=2.2, marker="o", markersize=5,
                markeredgecolor=INK, markeredgewidth=0.5, zorder=4, label="obs. median")
        if bridging:
            xs = np.linspace(1, 5, 100); hump = np.exp(-((xs - 3) ** 2) / 0.9)
            base, span = float(med.min()), float(med.max() - med.min())
            ax.plot(xs, base + 0.9 * span * hump, color=INK, lw=0.9, ls=":", alpha=0.5, zorder=3,
                    label="bridging prediction")
        ax.set_xticks(list(QLAB)); ax.set_xticklabels(list(QLAB.values()))
        ax.set_title(f"{mode}", fontsize=10)
        ax.set_xlabel("income stratum")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"share of pole-to-pole geodesic interior, $\rho_k$")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ MAIN necessity fig
def fig_necessity(sums, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in MODES if m in sums]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.3 * len(modes), 4.0), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        d = sums[mode]
        dn = d[d.stratum.isin([2, 3, 4])].dropna(subset=["delta"]) if "delta" in d.columns \
            else d.iloc[0:0]
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        if len(dn):
            for i, (_, g) in enumerate(dn.sort_values("stratum").groupby("city")):
                ax.plot(g.stratum, g.delta, color=GREY, lw=0.4, alpha=0.18, zorder=1,
                        label="individual cities" if i == 0 else None)
            gg = dn.groupby("stratum")["delta"]
            m, lo, hi = gg.median(), gg.quantile(.25), gg.quantile(.75)
            ax.fill_between(m.index, lo.values, hi.values, color=AMBER, alpha=0.18, zorder=1,
                            label="IQR across cities")
            ax.plot(m.index, m.values, color=AMBER, lw=2.2, marker="o", markersize=5,
                    markeredgecolor=INK, markeredgewidth=0.5, zorder=4,
                    label="median across cities")
            # rescale ylims to 8 times the largest absolute value of the IQR, just fo r aesthetics
            ax.set_ylim(np.array([-1, 1]) * (8 * np.max(np.abs(hi.values))))
        else:
            ax.text(0.5, 0.5, "no necessity\n(degenerate poles)", transform=ax.transAxes,
                    ha="center", va="center", color=GREY, fontsize=9)
        ax.set_xticks([2, 3, 4]); ax.set_xticklabels(["Q2", "Q3", "Q4"])
        ax.set_title(f"{mode}", fontsize=10); ax.set_xlabel("removed stratum")

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"excess pole-to-pole efficiency, $\Delta_k$")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ MAIN income-range table
def table_band_income_ranges(income_specs):
    """mode x band -> income min/max/median + population share.
    income_specs: {mode: path} where each CSV is that mode's antenas_income_areal[_mode].csv
    (its own quintile_nac column holds THAT mode's bands). Population is de-duplicated to
    physical sites (LONG,LAT) so co-located sectors are not double-counted."""
    rows = []
    for mode, path in income_specs.items():
        if not os.path.exists(path):
            print(f"  (band-range table: skipping {mode}, not found: {path})")
            continue
        d = pd.read_csv(path)
        if "quintile_nac" not in d.columns or "income_pc" not in d.columns:
            print(f"  (band-range table: {path} lacks quintile_nac/income_pc; skipped)")
            continue
        site_cols = [c for c in ("LONG", "LAT") if c in d.columns]
        if len(site_cols) == 2:
            d = d.drop_duplicates(subset=site_cols + ["income_pc", "served_pop10"]) \
                if "served_pop10" in d.columns else d.drop_duplicates(subset=site_cols + ["income_pc"])
        d = d.dropna(subset=["quintile_nac", "income_pc"])
        d["quintile_nac"] = d["quintile_nac"].astype(int)
        for k in range(1, 6):
            s = d[d.quintile_nac == k]
            if not len(s):
                continue
            pop = s["served_pop10"].sum() if "served_pop10" in s else len(s)
            rows.append(dict(mode=mode, band=f"Q{k}", n_sites=len(s),
                             income_min=round(s.income_pc.min(), 1),
                             income_max=round(s.income_pc.max(), 1),
                             income_median=round(s.income_pc.median(), 1),
                             pop=int(pop)))
    t = pd.DataFrame(rows)
    if len(t):
        t["pop_share"] = t.groupby("mode")["pop"].transform(lambda x: (x / x.sum()).round(3))
    return t


# ------------------------------------------------------------------ ANNEX summary table
def table_position_summary(sums):
    rows = []
    for mode, d in sums.items():
        for k in range(1, 6):
            s = d[d.stratum == k]
            z = s["z"].dropna() if "z" in s else pd.Series(dtype=float)
            ratio = (s.rho_obs / s.rho_null_mean).replace([np.inf, -np.inf], np.nan)
            rows.append(dict(mode=mode, band=f"Q{k}", n_cities=s.city.nunique(),
                             median_rho=round(s.rho_obs.median(), 4),
                             median_ratio=round(float(ratio.median()), 4),
                             median_z=round(float(z.median()), 3) if len(z) else np.nan,
                             frac_z_below0=round(float((z < 0).mean()), 3) if len(z) else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ANNEX bipolarity fig
def fig_bipolarity(sums, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # use the antenna summary (any works) at the CITY level for fraction reached
    d = sums.get("antenna", next(iter(sums.values())))
    if not ({"pairs_reached", "pairs_tried"} <= set(d.columns)):
        return False
    pc = d.groupby("city").agg(pr=("pairs_reached", "max"), pt=("pairs_tried", "max"))
    frac = (pc.pr / pc.pt).replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(frac, bins=30, color=TEAL, alpha=0.6, edgecolor=INK, linewidth=0.4)
    ax.axvline(frac.median(), color=AMBER, lw=1.5, ls="--",
               label=f"median {frac.median():.2f}")
    ax.set_xlabel("fraction of Q1$\\to$Q5 pole-pairs reached")
    ax.set_ylabel("cities")
    ax.text(0.02, 0.96, "left mass = near-bipolar cities\n(poles barely reach each other)",
            transform=ax.transAxes, fontsize=7.5, color=GREY, va="top")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def _fig_paths(outdir, name, fmts):
    return [os.path.join(outdir, f"{name}.{f}") for f in fmts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", required=True,
                    help="mode=position_summary_<mode>.csv for antenna/population/width")
    ap.add_argument("--income", nargs="*", default=None,
                    help="mode=path per stratification, e.g. "
                         "antenna=antenas_income_areal.csv "
                         "population=antenas_income_areal_population.csv "
                         "width=antenas_income_areal_width.csv")
    ap.add_argument("--min_frac", type=float, default=0.5)
    ap.add_argument("--min_pairs", type=int, default=2000)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--fmt", nargs="*", default=["svg", "pdf"], choices=["png", "svg", "pdf"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    specs = dict(s.split("=", 1) for s in a.summary)
    print("gating cities per stratification:")
    sums = load_summaries(specs, a.min_frac, a.min_pairs)

    fig_occupancy(sums, _fig_paths(a.outdir, "fig_position_occupancy", a.fmt))
    print("wrote fig_position_occupancy")
    fig_necessity(sums, _fig_paths(a.outdir, "fig_position_necessity", a.fmt))
    print("wrote fig_position_necessity")

    tps = table_position_summary(sums)
    tps.to_csv(os.path.join(a.outdir, "table_position_summary.csv"), index=False)
    print("wrote table_position_summary.csv (annex)")

    if fig_bipolarity(sums, _fig_paths(a.outdir, "fig_position_bipolarity", a.fmt)):
        print("wrote fig_position_bipolarity (annex)")

    if a.income:
        income_specs = dict(s.split("=", 1) for s in a.income)
        tbr = table_band_income_ranges(income_specs)
        tbr.to_csv(os.path.join(a.outdir, "table_band_income_ranges.csv"), index=False)
        print(f"wrote table_band_income_ranges.csv (main, {len(tbr)} rows)")
    print("\ndone.")
