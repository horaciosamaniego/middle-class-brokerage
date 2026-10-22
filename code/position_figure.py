#!/usr/bin/env python3
"""
position_figure.py -- single-panel Position figure (the negative result: no middle peak).

Occupancy: each stratum's share of interior nodes on shortest pole-to-pole paths
(rho_k), summarised ACROSS cities as a median + interquartile band, against the
degree-stratified permutation null. A schematic "bridging prediction" hump marks the
peak the textbook hypothesis expects on the middle -- so its absence, and the
observed line sitting on the null, read the result directly.

Only well-sampled cities enter: cities with fewer than --min_pairs reachable
pole-to-pole pairs give rho_k as a ratio of tiny integers (unstable, 0..1 noise) and
are excluded from the main panel (the occupancy analogue of the boundary-capable
gate). Pass --all_cities to draw every city faintly for the SI.

Necessity (efficiency-under-removal, Delta_k) is a separate, compute-heavy analysis,
not plotted here; it is reported as a supplementary robustness check.

Input -- position_summary.csv (one row per city x stratum):
  city, stratum(1..5), rho_obs, z,
  rho_null_mean, rho_null_lo, rho_null_hi,   # from the patched brokerage_profile
  pairs_reached                               # reachable pole pairs (power gate)

Usage:
  python position_figure.py --csv position_summary.csv --min_pairs 2000 \
         --fig fig_position --fmt svg pdf
"""
import argparse
import os
import numpy as np
import pandas as pd

INK, TEAL, AMBER, GREY = "#1a1a1a", "#2a7f7f", "#d9942b", "#9aa0a6"
QLAB = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"}


def diagnostic(d):
    """Per middle stratum: median z and fraction of cities with z<0 -- 'at chance'
    (median z ~ 0) vs 'actively depleted' (median z < 0 in most cities)."""
    print("occupancy z by stratum (gated cities):")
    print(f"  {'stratum':8s} {'n':>4s} {'median z':>9s} {'frac z<0':>9s} {'median rho':>11s}")
    for k in range(1, 6):
        s = d[d.stratum == k]
        z = s["z"].dropna() if "z" in s else pd.Series(dtype=float)
        zt = f"{z.median():+.2f}" if len(z) else "  --"
        ft = f"{(z < 0).mean():.2f}" if len(z) else " --"
        tag = ""
        if k in (2, 3, 4) and len(z):
            tag = "  <-- depleted" if (z.median() < 0 and (z < 0).mean() > 0.6) else "  <-- ~chance"
        print(f"  Q{k:<7d} {len(s):>4d} {zt:>9s} {ft:>9s} {s.rho_obs.median():>11.3f}{tag}")


def make_fig(d, paths, all_cities=False, bridging=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_nec = ("delta" in d.columns and
                d[d.stratum.isin([2, 3, 4])]["delta"].notna().any())
    if have_nec:
        fig, (ax, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.2)); axB = None

    nb = d.groupby("stratum").agg(lo=("rho_null_lo", "median"),
                                  hi=("rho_null_hi", "median"),
                                  mean=("rho_null_mean", "median"))
    ax.fill_between(nb.index, nb.lo, nb.hi, color=TEAL, alpha=0.15, zorder=0,
                    label="permutation null (95%)")
    ax.plot(nb.index, nb["mean"], color=TEAL, lw=1.0, ls="--", alpha=0.85, zorder=2)

    grp = d.groupby("stratum")["rho_obs"]
    med = grp.median(); q1 = grp.quantile(0.25); q3 = grp.quantile(0.75)
    ax.fill_between(med.index, q1.values, q3.values, color=AMBER, alpha=0.18, zorder=1,
                    label="observed IQR (cities)")
    ax.plot(med.index, med.values, color=AMBER, lw=2.4, marker="o", markersize=6,
            markeredgecolor=INK, markeredgewidth=0.6, zorder=4, label="observed median")

    if all_cities:
        for _, g in d.groupby("city"):
            g = g.sort_values("stratum")
            ax.plot(g.stratum, g.rho_obs, color=GREY, lw=0.4, alpha=0.15, zorder=1)

    if bridging:
        xs = np.linspace(1, 5, 100); hump = np.exp(-((xs - 3) ** 2) / 0.9)
        base = float(med.min()); span = float(med.max() - med.min())
        ax.plot(xs, base + 0.9 * span * hump, color=INK, lw=1.0, ls=":", alpha=0.55,
                zorder=3, label="bridging prediction (schematic)")

    ax.set_xticks(list(QLAB)); ax.set_xticklabels(list(QLAB.values()))
    ax.set_xlabel("income stratum")
    ax.set_ylabel(r"share of pole-to-pole geodesic interior, $\rho_k$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    if have_nec:
        ax.text(0.02, 0.98, "a", transform=ax.transAxes, fontweight="bold",
                fontsize=12, va="top")

    if axB is not None:
        # necessity: cross-city median + IQR of Delta_k for the middle strata; zero line
        dn = d[d.stratum.isin([2, 3, 4])].dropna(subset=["delta"])
        axB.axhline(0, color=INK, lw=1.0, zorder=2)
        gg = dn.groupby("stratum")["delta"]
        m = gg.median(); lo = gg.quantile(0.25); hi = gg.quantile(0.75)
        axB.fill_between(m.index, lo.values, hi.values, color=AMBER, alpha=0.18, zorder=1)
        axB.plot(m.index, m.values, color=AMBER, lw=2.4, marker="o", markersize=6,
                 markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
        axB.set_xticks([2, 3, 4]); axB.set_xticklabels(["Q2", "Q3", "Q4"])
        axB.set_xlabel("income stratum removed")
        axB.set_ylabel(r"excess pole-to-pole efficiency, $\Delta_k$")
        axB.text(0.02, 0.98, "b", transform=axB.transAxes, fontweight="bold",
                 fontsize=12, va="top")
        axB.text(0.5, 0.02, "above 0 = load-bearing", transform=axB.transAxes,
                 fontsize=7.5, color=GREY, ha="center", va="bottom")
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="position_summary.csv")
    ap.add_argument("--min_frac", type=float, default=0.5,
                    help="min fraction of pole-pairs reached (used when pairs_tried present)")
    ap.add_argument("--min_pairs", type=int, default=2000,
                    help="fallback gate on raw pairs_reached when pairs_tried is absent")
    ap.add_argument("--all_cities", action="store_true", help="faint overlay of every city (SI)")
    ap.add_argument("--no_bridging", action="store_true")
    ap.add_argument("--fig", default="fig_position")
    ap.add_argument("--fmt", nargs="*", default=["png"], choices=["png", "svg", "pdf"])
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    need = {"city", "stratum", "rho_obs", "rho_null_lo", "rho_null_hi"}
    if need - set(d.columns):
        raise SystemExit(f"missing columns: {sorted(need - set(d.columns))}")

    n_all = d.city.nunique()
    if "pairs_reached" in d.columns and "pairs_tried" in d.columns and d.pairs_tried.notna().any():
        frac = (d.pairs_reached / d.pairs_tried).replace([np.inf, -np.inf], np.nan)
        d = d.assign(frac_reached=frac)
        keep = d.groupby("city")["frac_reached"].transform("max") >= a.min_frac
        d = d[keep]
        print(f"power gate: {d.city.nunique()}/{n_all} cities with fraction_reached "
              f">= {a.min_frac} enter the panel")
    elif "pairs_reached" in d.columns:
        keep = d.groupby("city")["pairs_reached"].transform("max") >= a.min_pairs
        d = d[keep]
        print(f"power gate (FALLBACK, no pairs_tried): {d.city.nunique()}/{n_all} cities "
              f"with pairs_reached >= {a.min_pairs}. Add pairs_tried for a proper fraction gate.")
    else:
        print("WARNING: no pairs columns -- cannot gate; plotting all cities.")
    diagnostic(d)
    base = os.path.splitext(a.fig)[0]
    paths = [f"{base}.{fmt}" for fmt in a.fmt]
    make_fig(d, paths, all_cities=a.all_cities, bridging=not a.no_bridging)
    for p in paths:
        print(f"wrote {p}")