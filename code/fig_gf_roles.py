#!/usr/bin/env python3
"""
fig_gf_roles.py -- the definitive Option 2 figure: the eight Gould-Fernandez brokerage
roles as a forest plot of mixing-adjusted enrichment R_mix, grouped by whether the two
connected individuals (endpoints) share a stratum. Same-endpoint roles are enriched
(right of 1); different-endpoint roles, including the intermediate-liaison "bridging"
role, sit at or below chance (left of 1). Reads the gf_roles per-city table so the
spread across cities can be shown, not just the median.

  python fig_gf_roles.py --percity gf_roles_population.csv --out fig_gf_roles --fmt pdf svg png
  (optionally --percity2 gf_roles_antenna.csv to overlay the robustness stratification)
"""
import argparse, os
import numpy as np
import pandas as pd

# display order + human labels; grouped same-endpoint (top) then different-endpoint
ORDER = [
    ("coordinator",            "coordinator", "same"),
    ("itinerant_above",        "itinerant (broker above)", "same"),
    ("itinerant_below",        "itinerant (broker below)", "same"),
    ("gatekeeper",             "gatekeeper", "diff"),
    ("representative",         "representative", "diff"),
    ("liaison_intermediate",   "liaison (intermediate broker)  \u2014 bridging", "diff"),
    ("liaison_extremal_above", "liaison (broker above both)", "diff"),
    ("liaison_extremal_below", "liaison (broker below both)", "diff"),
]


def load_role_dist(percity_csv):
    """Return dict role -> array of per-city R_mix (dropna)."""
    d = pd.read_csv(percity_csv)
    out = {}
    for key, _, _ in ORDER:
        col = f"Rmix_{key}"
        out[key] = d[col].dropna().values if col in d.columns else np.array([])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percity", required=True, help="gf_roles per-city table (primary stratification)")
    ap.add_argument("--percity2", default=None, help="optional second stratification to overlay")
    ap.add_argument("--label1", default="equal-population")
    ap.add_argument("--label2", default="equal-area")
    ap.add_argument("--out", default="fig_gf_roles")
    ap.add_argument("--fmt", nargs="+", default=["pdf", "svg", "png"])
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, TEAL, AMBER, GREY = "#1a1a1a", "#2a7f7f", "#d9942b", "#9aa0a6"
    d1 = load_role_dist(a.percity)
    d2 = load_role_dist(a.percity2) if a.percity2 else None

    n = len(ORDER)
    ys = np.arange(n)[::-1]            # first role at top
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axvline(1.0, color=INK, lw=1.2, zorder=2)   # null / chance

    def draw(dist, yoff, color, marker, lab):
        for (key, _, grp), y in zip(ORDER, ys):
            v = dist.get(key, np.array([]))
            if v.size == 0:
                continue
            med = np.median(v)
            lo, hi = np.percentile(v, 25), np.percentile(v, 75)
            ax.plot([lo, hi], [y + yoff, y + yoff], color=color, lw=2.2, alpha=0.5, zorder=3,
                    solid_capstyle="round")
            ax.plot(med, y + yoff, marker=marker, color=color, markersize=7,
                    markeredgecolor=INK, markeredgewidth=0.5, zorder=4,
                    label=lab if key == "coordinator" else None)

    if d2 is not None:
        draw(d1, +0.12, TEAL, "o", a.label1)
        draw(d2, -0.12, AMBER, "s", a.label2)
    else:
        draw(d1, 0.0, TEAL, "o", a.label1)

    # group shading + labels
    same_ys = [y for (k, _, g), y in zip(ORDER, ys) if g == "same"]
    diff_ys = [y for (k, _, g), y in zip(ORDER, ys) if g == "diff"]
    ax.axhspan(min(same_ys) - 0.5, max(same_ys) + 0.5, color=TEAL, alpha=0.05, zorder=0)
    ax.text(ax.get_xlim()[0], max(same_ys) + 0.62, "endpoints in the SAME stratum",
            fontsize=8, color=TEAL, va="bottom", style="italic")
    ax.text(ax.get_xlim()[0], max(diff_ys) + 0.62, "endpoints in DIFFERENT strata",
            fontsize=8, color=GREY, va="bottom", style="italic")

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in ORDER], fontsize=9)
    # bold the bridging role's tick label
    for tl, (_, _, _) in zip(ax.get_yticklabels(), ORDER):
        if "bridging" in tl.get_text():
            tl.set_fontweight("bold")
    ax.set_xlabel(r"mixing-adjusted brokerage enrichment  $R_{\mathrm{mix}}$")
    ax.set_ylim(-0.7, n - 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.text(1.0, -0.62, "chance", ha="center", va="top", fontsize=7.5, color=INK)
    if d2 is not None:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()

    for ext in a.fmt:
        fig.savefig(f"{a.out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}.{ext}")


if __name__ == "__main__":
    main()
    