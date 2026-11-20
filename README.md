# Brokerage without bridging

Code and derived data for

> Samaniego, H., Menezes, R., Cantillán, R., Otero, G., Bucca, M. & Evsukoff, A.
> *Brokerage without bridging: middle-income residents do not connect rich and poor in
> Brazilian cities.*

Archived release: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23606008.svg)](https://doi.org/10.5281/zenodo.23606008)

## Two levels of reproducibility

The analysis runs on call-detail records (CDR) from a Brazilian operator. Those records,
and the per-city contact graphs built from them, are confidential and **are not in this
repository**. So the code is organised in two tiers.

| tier | who can run it | input | output |
|---|---|---|---|
| **1. Figures and tables** | anyone | `data/derived/` (in this repo) | every figure and table in the paper and Supplement |
| **2. Network analysis** | holders of the CDR graphs | per-city `.gt` graphs (not distributed) | the files in `data/derived/` |

`data/derived/` holds city- and stratum-level aggregates only: no subscriber, node or edge
appears in any file.

## The null model

Brokerage is measured against a mixing-preserving null that holds every user's degree and
the contact between every pair of income strata fixed. Since version 2.0.0 the null also
preserves **reciprocity**: mutual ties are rewired as undirected edges and one-way calls as
directed arcs, each with the strata as blocks (`mixing_null.reciprocity_preserving_null`,
`mixing_null_5x5.py --reciprocal`). Two-paths that return to their origin (a→b→a, which a
mutual tie creates) are excluded from the ladder census. Version 1.x used a directed null
that did not preserve reciprocity; see the release notes.

## Paper → code

| in the paper | script | reads | tier |
|---|---|---|:---:|
| Fig. 1, brokerage typology | drawn by hand | — | — |
| Fig. 2, position occupancy | `position_figure.py` | `position_summary_population.csv`, `position_summary_area.csv` | 1 |
| Fig. 3, removal of middle strata | `position_pipeline.py` | `position_summary_*.csv`, `necessity_*.csv` | 1 |
| Fig. 4, closure collapse; Table 1 | `mechanism_pipeline.py` | `two_null_*_summary.csv` | 1 |
| Fig. 5, Gould–Fernandez roles | `fig_gf_roles.py` | `gf_roles_*.csv` | 1 |
| Table 2, ladder census | `make_si_tables.py --check` | `gf_roles_population.csv` | 1 |
| Supp. Fig. S1, power | `fig_mechanism_capable.py` | `two_null_population_summary.csv` | 1 |
| Supp. Table S2, calibration | `build_calibration_report.py` | `calibration_all_cities.csv` | 1 |
| Supp. Table S3, swap vs analytic | `make_mix_consistent_table.py` | `two_null_*_summary.csv` | 1 |
| Supp. Tables S4–S5 | `make_si_tables.py` | `position_summary_*.csv`, `gf_roles_*.csv` | 1 |
| Supp. Note 5, rank and size | `within_stratum_density.py`, `gf_roles.py` | `within_stratum_density.csv`, `gf_roles_*_scaling.csv` | 1 |
| Supp. Note 7 and Tables S7–S8, reciprocated backbone | `backbone_compare.py` | `data/derived/backbone/` and the files above | 1 |
| position sweep (degree-stratified null) | `sweep_brokerage.py` → `position_tidy.py` | `.gt` graphs | 2 |
| pole and ladder censuses, reciprocity-preserving null | `run_reciprocal.sh` (`run_two_nulls.py`, `mixing_null_5x5.py`, `gf_roles.py --exclude_returns`) | `.gt` graphs | 2 |
| reciprocated backbone, every analysis | `run_backbone.sh` (`make_backbone.py` and the scripts above) | `.gt` graphs | 2 |
| estimator calibration (directed form of the null) | `calibrate_null_fast.py` | `.gt` graphs | 2 |

`test_reciprocal.py` and `test_backbone.py` check the null models on synthetic graphs in
about a minute each, with no data needed.

## Running tier 1

```bash
conda env create -f environment.yml
conda activate brokerage
cd data/derived                       # scripts read and write relative to here
python ../../code/make_si_tables.py \
    --position position_summary_population.csv equal-population \
    --position position_summary_area.csv       equal-area \
    --ladder   gf_roles_population.csv          equal-population \
    --ladder   gf_roles_antenna.csv             equal-area \
    --check                            # confirms every value against the paper
python ../../code/mechanism_pipeline.py \
    --summary population=two_null_population_summary.csv antenna=two_null_antenna_summary.csv \
    --min_capable 5000 --outdir mech_figs --fmt pdf png
python ../../code/backbone_compare.py --main_dir . --bb_dir backbone
```

## Data availability

Derived city-level data sufficient to reproduce every figure and table are included in
`data/derived/` and archived on Zenodo (DOI above). The underlying call-detail records
were provided under a confidentiality agreement and cannot be shared; requests should be
addressed to the corresponding author.

## Licence

Code: MIT. Derived data: CC BY 4.0.
