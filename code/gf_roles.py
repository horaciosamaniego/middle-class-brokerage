#!/usr/bin/env python3
"""
gf_roles.py -- collapse the 5x5x5 brokerage tensor into the five Gould-Fernandez
brokerage roles, and compute R_mix per role from the r5x5 sweep output.

The 5x5x5 tensor B[i,j,m] (source stratum i -> broker stratum m -> target stratum j)
IS the full Gould-Fernandez cross-classification. The five named roles are sums over
specific cells (Gould & Fernandez 1989), here with ORDERED income strata:

  coordinator    i == m == j            broker mediates WITHIN its own stratum
  itinerant      i == j, m != i         outsider mediates between two of another stratum,
                                        split by broker position vs that stratum:
                                          itinerant_above  m > i  (higher-income outsider)
                                          itinerant_below  m < i  (lower-income outsider)
  gatekeeper     i != m, m == j         broker mediates an INBOUND tie to its stratum
  representative i == m, m != j         broker mediates an OUTBOUND tie from its stratum
  liaison        i, j, m all distinct   broker belongs to neither endpoint stratum
                                        (this is the BRIDGING role the paper tests)

Because strata are ordered, liaison is further split by broker position:
  liaison_intermediate    min(i,j) < m < max(i,j)   broker income BETWEEN endpoints
                                                     (genuine bridging)
  liaison_extremal_above  m > max(i,j)              broker HIGHER than both (top-down)
  liaison_extremal_below  m < min(i,j)              broker LOWER  than both (bottom-up)
This split is the sharp bridging-vs-control question that plain G-F cannot pose, and the
above/below asymmetry asks whether extremal brokerage is an elite-reaching-down or a
poor-reaching-up phenomenon.

Consumes the tidy r5x5 output (columns: city,i,j,m,B_obs,null_mean,null_sd,R_mix,testable).
Per city and role: R_mix_role = sum(B_obs over role cells) / sum(null_mean over role cells),
using only cells whose null_mean >= min_capable (so a role is reported only where the
null actually places mass). Emits per-city and cross-city-median role tables.

  python gf_roles.py --r5x5 r5x5_population.csv --min_capable 5000 --out gf_roles_population.csv
"""
import argparse
import numpy as np
import pandas as pd


def role_of(i, j, m):
    """Return the G-F role name for an ordered-strata (i,j,m) cell."""
    if i == m == j:
        return "coordinator"
    if i == j and m != i:
        # outsider brokering WITHIN one stratum; split by broker position vs that stratum
        return "itinerant_above" if m > i else "itinerant_below"
    if i != m and m == j:
        return "gatekeeper"
    if i == m and m != j:
        return "representative"
    # all distinct -> liaison; split by ordered position of the broker relative
    # to the endpoint pair. With ordered income strata these are distinct phenomena:
    #   intermediate   : broker income BETWEEN the endpoints        (genuine bridging)
    #   extremal_above : broker HIGHER than both endpoints          (top-down brokerage)
    #   extremal_below : broker LOWER than both endpoints           (bottom-up brokerage)
    lo, hi = min(i, j), max(i, j)
    if lo < m < hi:
        return "liaison_intermediate"
    return "liaison_extremal_above" if m > hi else "liaison_extremal_below"


ROLES = ["coordinator", "itinerant_above", "itinerant_below",
         "gatekeeper", "representative",
         "liaison_intermediate", "liaison_extremal_above", "liaison_extremal_below"]


def per_city_roles(df, min_capable):
    """df: one city's 125 rows. Returns dict role -> (R_mix, n_cells, obs_share).

    R_mix is size-adjusted (the mixing null holds strata sizes + mixing fixed, so
    R_mix>1 means enrichment BEYOND abundance). obs_share is the raw fraction of all
    observed two-paths falling in this role -- deliberately NOT size-adjusted, so the
    contrast between obs_share (abundance-driven) and R_mix (abundance-removed) shows
    the null doing its work. A role can have a large obs_share yet R_mix ~ 1: prevalent
    only because the strata involved are large, not because it is structurally favoured."""
    df = df.copy()
    df["role"] = [role_of(int(r.i), int(r.j), int(r.m)) for r in df.itertuples()]
    total_obs = float(df.B_obs.sum())
    use = df[df.null_mean >= min_capable]          # only cells the null can resolve
    out = {}
    for role in ROLES:
        cells = use[use.role == role]
        obs_share = float(df[df.role == role].B_obs.sum()) / total_obs if total_obs > 0 else np.nan
        if len(cells) == 0 or cells.null_mean.sum() <= 0:
            out[role] = (np.nan, 0, obs_share)
            continue
        R = cells.B_obs.sum() / cells.null_mean.sum()
        out[role] = (float(R), int(len(cells)), obs_share)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r5x5", required=True, help="tidy r5x5 sweep CSV (city,i,j,m,B_obs,null_mean,R_mix,...)")
    ap.add_argument("--min_capable", type=int, default=5000)
    ap.add_argument("--out", default="gf_roles.csv", help="per-city role R_mix table")
    ap.add_argument("--summary_out", default=None, help="cross-city median table (default: <out>_summary.csv)")
    ap.add_argument("--sizes", default=None,
                    help="optional CSV with columns city,n_nodes to join network size when the "
                         "r5x5 file lacks it (e.g. two_null_population_summary.csv)")
    ap.add_argument("--scaling_out", default=None,
                    help="optional output for the role-vs-network-size regression "
                         "(default: <out>_scaling.csv). Requires n_nodes available.")
    a = ap.parse_args()

    d = pd.read_csv(a.r5x5)
    need = {"city", "i", "j", "m", "B_obs", "null_mean"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"r5x5 file missing columns: {missing}")

    # network size per city: from the r5x5 file if present, else joined from --sizes
    size_map = {}
    if "n_nodes" in d.columns and d["n_nodes"].notna().any():
        size_map = d.dropna(subset=["n_nodes"]).groupby("city")["n_nodes"].first().to_dict()
    elif a.sizes:
        sz = pd.read_csv(a.sizes)
        if {"city", "n_nodes"} <= set(sz.columns):
            size_map = sz.groupby("city")["n_nodes"].first().to_dict()
        else:
            print(f"WARN: --sizes file lacks city/n_nodes columns; scaling analysis skipped")

    rows = []
    for city, g in d.groupby("city"):
        rr = per_city_roles(g, a.min_capable)
        rec = {"city": city}
        for role in ROLES:
            R, n, obs_share = rr[role]
            rec[f"Rmix_{role}"] = R
            rec[f"ncells_{role}"] = n
            rec[f"obsshare_{role}"] = obs_share
        rows.append(rec)
    per = pd.DataFrame(rows)
    if size_map:
        per["n_nodes"] = per.city.map(size_map)
    per.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(per)} cities)")

    # cross-city summary: median R_mix per role, and fraction of cities with R>1
    srows = []
    for role in ROLES:
        col = f"Rmix_{role}"
        v = per[col].dropna()
        obs = per[f"obsshare_{role}"].dropna()
        srows.append(dict(
            role=role,
            n_cities=int(v.size),
            median_Rmix=round(float(v.median()), 4) if v.size else np.nan,
            iqr_lo=round(float(v.quantile(.25)), 4) if v.size else np.nan,
            iqr_hi=round(float(v.quantile(.75)), 4) if v.size else np.nan,
            frac_gt1=round(float((v > 1).mean()), 3) if v.size else np.nan,
            median_obs_share=round(float(obs.median()), 4) if obs.size else np.nan,
        ))
    summ = pd.DataFrame(srows)
    so = a.summary_out or a.out.replace(".csv", "_summary.csv")
    summ.to_csv(so, index=False)
    print(f"wrote {so}")
    print("\n=== cross-city median R_mix per Gould-Fernandez role ===")
    print(summ.to_string(index=False))
    print("\nreading: coordinator (within-stratum) is the diagonal; "
          "liaison_intermediate is genuine bridging.\n"
          "bridging hypothesis predicts liaison_intermediate > 1; "
          "boundary-control/homophily predicts coordinator > 1, liaisons <= 1.")

    # -------- scaling: does role prevalence / enrichment vary with CITY size? --------
    if "n_nodes" in per.columns and per["n_nodes"].notna().sum() >= 5:
        x = np.log10(per["n_nodes"].astype(float))
        srows2 = []
        for role in ROLES:
            for kind, col in [("obs_share", f"obsshare_{role}"), ("R_mix", f"Rmix_{role}")]:
                y = per[col].astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5:
                    srows2.append(dict(role=role, quantity=kind, n=int(ok.sum()),
                                       slope=np.nan, r=np.nan, p=np.nan)); continue
                xx, yy = x[ok], y[ok]
                slope, intercept = np.polyfit(xx, yy, 1)
                r = np.corrcoef(xx, yy)[0, 1]
                # two-sided p for Pearson r
                n = ok.sum()
                if abs(r) < 1 - 1e-12 and n > 2:
                    tstat = r * np.sqrt((n - 2) / (1 - r * r))
                    try:
                        from scipy import stats
                        p = 2 * stats.t.sf(abs(tstat), n - 2)
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                srows2.append(dict(role=role, quantity=kind, n=int(n),
                                   slope=round(float(slope), 4), r=round(float(r), 3),
                                   p=round(float(p), 4) if np.isfinite(p) else np.nan))
        scal = pd.DataFrame(srows2)
        sco = a.scaling_out or a.out.replace(".csv", "_scaling.csv")
        scal.to_csv(sco, index=False)
        print(f"\nwrote {sco}")
        print("\n=== role prevalence / enrichment vs log10(network size) ===")
        print(scal.to_string(index=False))
        print("\nreading: 'obs_share' slope>0 -> larger cities allocate more brokerage to this role;")
        print("         'R_mix'    slope>0 -> this role's enrichment INTENSIFIES with city size")
        print("         (R_mix slope is the size-confound-free scaling claim).")
    else:
        print("\n(scaling analysis skipped: no n_nodes -- pass --sizes two_null_population_summary.csv)")


if __name__ == "__main__":
    main()