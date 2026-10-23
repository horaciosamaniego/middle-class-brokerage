# Brokerage without bridging

Code and derived data for

> Samaniego, H., Menezes, R., Cantillán, R., Otero, G., Bucca, M. & Evsukoff, A.
> *Brokerage without bridging: the middle class does not connect rich and poor in
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

## Paper → code

<!-- Fill from section 1 of repo_audit_report.md. One row per figure/table. -->

| in the paper | script | reads | tier |
|---|---|---|:---:|
| Fig. 1, position occupancy | `position_figure.py` | `position_summary_population.csv`, `position_summary_area.csv` | 1 |
| Fig. 2, removal of middle strata | `...` | `necessity_*.csv` | 1 |
| Fig. 3, closure collapse | `...` | `two_null_*_summary.csv` | 1 |
| Fig. 4, Gould–Fernandez roles | `fig_gf_roles.py` | `gf_roles_*.csv` | 1 |
| Table 1, pole closure | `...` | `two_null_*_summary.csv` | 1 |
| Table 2, ladder census | `gf_roles.py` | `r5x5_population.csv` | 1 |
| Supp. Fig. S1, power | `fig_mechanism_capable.py` | `two_null_population_summary.csv` | 1 |
| Supp. Table S2, calibration | `build_calibration_report.py` | `calibration_all_cities.csv` | 1 |
| Supp. Table S3, swap vs analytic | `make_mix_consistent_table.py` | `two_null_*_summary.csv` | 1 |
| Supp. Tables S4–S5 | `make_si_tables.py` | `position_summary_*.csv`, `gf_roles_*.csv` | 1 |
| position sweep (degree-stratified null) | `sweep_brokerage.py` → `position_tidy.py` | `.gt` graphs | 2 |
| two-null mechanism census | `run_two_nulls.py` (`run_two_nulls_parallel.sh`) | `.gt` graphs | 2 |
| ladder-wide census | `mixing_null_5x5.py` (`run_parallel_5x5.sh`) | `.gt` graphs | 2 |
| estimator calibration | `calibrate_null.py` | `.gt` graphs | 2 |

## Running tier 1

```bash
conda env create -f environment.yml
conda activate brokerage
cd data/derived                       # scripts read and write relative to here
python ../../code/position_figure.py
python ../../code/make_si_tables.py \
    --position position_summary_population.csv equal-population \
    --position position_summary_area.csv       equal-area \
    --ladder   gf_roles_population.csv          equal-population \
    --ladder   gf_roles_antenna.csv             equal-area \
    --check                            # confirms every value against the paper
```

## Data availability

Derived city-level data sufficient to reproduce every figure and table are included in
`data/derived/` and archived on Zenodo (DOI above). The underlying call-detail records
were provided under a confidentiality agreement and cannot be shared; requests for access
should be directed to [data provider / A. Evsukoff, COPPE/UFRJ].

## Licence

Code: MIT. Derived data: CC BY 4.0.
