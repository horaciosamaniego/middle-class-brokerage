#!/usr/bin/env python3
"""
fig_mechanism_capable.py -- the "absence of closure is not a power artifact" figure.

R_mix (rich pole, Q5) vs boundary-capable Q5 brokers, per city, log-x. Filled markers =
powered (>= min_capable); open = underpowered (shown for range, not interpreted). Uses
the FULL per-city two_null summary (596 rows = 298 cities x 2 poles), so the capability
axis spans the whole hierarchy from cap~1 to Sao Paulo (~1.5e5) -- the strongest form of
the argument that R_mix<1 holds across three orders of magnitude of pole size.

  python fig_mechanism_capable.py --summary two_null_population_summary.csv \
      --min_capable 5000 --out mech_figs/fig_mechanism_capable --fmt pdf svg png
"""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="per-city two_null summary (596-row) CSV")
    ap.add_argument("--pole", type=int, default=5, help="focal pole to plot (default Q5)")
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--out", default="fig_mechanism_capable")
    ap.add_argument("--fmt", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--label", default="equal-population",
                    help="stratification name for the caption corner")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    INK, TEAL, GREY = "#1a1a1a", "#2a7f7f", "#9aa0a6"
    d = pd.read_csv(a.summary)
    d = d[d.pole == a.pole].copy()
    # keep only rows with a positive capability and a finite R_mix
    d = d[(d.boundary_capable > 0) & np.isfinite(d.R_mix)]
    if "mix_consistent" in d.columns:
        # keep inconsistent rows in the plot but we could grey them; here we keep all
        pass
    powered = d[d.boundary_capable >= a.min_capable]
    under = d[d.boundary_capable < a.min_capable]

    # correlation over the full range (what the text cites)
    r = np.corrcoef(np.log10(d.boundary_capable), d.R_mix)[0, 1]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(1.0, color=INK, lw=1.1, ls="--", zorder=2, label="null expectation")
    ax.axvline(a.min_capable, color=GREY, lw=0.9, ls=":", zorder=1)
    # underpowered: open markers
    ax.scatter(under.boundary_capable, under.R_mix, s=26, facecolors="none",
               edgecolors=GREY, linewidths=0.9, zorder=3, label="underpowered")
    # powered: filled teal
    ax.scatter(powered.boundary_capable, powered.R_mix, s=30, color=TEAL,
               edgecolors=INK, linewidths=0.4, zorder=4, label="powered")
    ax.set_xscale("log")
    ax.set_xlabel("boundary-capable $Q_5$ brokers (log scale)")
    ax.set_ylabel(r"mixing-adjusted closure  $R_{\mathrm{mix}}$")
    ax.set_ylim(min(0.55, d.R_mix.min() - 0.03), max(1.15, d.R_mix.max() + 0.03))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
#    ax.text(0.02, 0.03,
    ax.text(0.7, 0.03,
            f"{a.label}\nn={len(d)} cities  |  powered={len(powered)}\n"
            fr"$r(\log_{{10}}\,\mathrm{{cap}},\,R_{{\mathrm{{mix}}}})={r:+.2f}$",
            transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in a.fmt:
        fig.savefig(f"{a.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}.{ext}")
    print(f"n={len(d)}  powered={len(powered)}  r(log10 cap, R_mix)={r:+.3f}  "
          f"all R_mix<1: {bool((powered.R_mix<1).all())}")


if __name__ == "__main__":
    main()
    
