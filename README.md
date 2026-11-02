# kazakhstan-biosafety-risk-pipeline

Reference implementation of a disease-agnostic spatio-temporal-species data model and a
pluggable module interface (`RiskModule`) for surveillance risk analytics. Four analytical
methods run behind the same interface:

| Module | Input kind | Output kind | File |
|---|---|---|---|
| Space–time permutation scan | `point_events` | `cluster_table` | `pipeline/modules/space_time_scan.py` |
| MaxEnt-equivalent suitability model | `point_events` | `suitability_surface` | `pipeline/modules/maxent.py` |
| Principal component analysis | `farm_survey` | `components` | `pipeline/modules/pca.py` |
| Bayesian beta-binomial prevalence | `farm_survey` | `posterior` | `pipeline/modules/bayes_beta_binomial.py` |

This repository accompanies the article *"A Disease-Agnostic Data Model and Pluggable Module
Interface for Spatio-Temporal Risk Analytics: An Open Reference Implementation Evaluated on
Veterinary Surveillance Data"* (Kadyrov, Satybaldina, Tashatov, Sagindykov, Aidynov;
*Information*, manuscript information-4650898, under review). Table and figure numbers below
refer to the revised manuscript.

## Status

Implemented:

- Unified `focus` schema and ETL from raw registry spreadsheets (`etl/normalize.py`), with
  coordinate, date and duplicate checks and a quality report.
- Module contract `RunContext` / `RiskResult` / `RiskModule` (`pipeline/core.py`).
- All four modules listed above.
- MaxEnt-equivalent module: AUC over repeated random splits **and** spatial block
  cross-validation (square blocks, k folds, repeated block-to-fold assignment); background
  points restricted to the national territory by a boundary mask
  (`data/boundaries/kazakhstan.geojson`, Natural Earth 1:50m, public domain).
- REST API (FastAPI, `api/main.py`) and a minimal web client (`web/index.html`, Leaflet) for
  the space–time scan module.

Not yet implemented (target architecture described in the article):

- PostgreSQL/PostGIS storage (state is currently kept in memory and flat CSV files).
- Exposure of the MaxEnt-equivalent, PCA and Bayesian modules through the REST API
  (they are run from the command-line scripts below).
- Validated container deployment (`Dockerfile` and `docker-compose.yml` are provided but
  have not been tested as a deployment).

## Installation

```bash
pip install -r requirements.txt
```

## Data

The raw anthrax, rabies and foot-and-mouth-disease registries are institutional data and are **not** included in
this repository. To regenerate the unified tables from the raw spreadsheets:

```bash
python3 etl/normalize.py --raw-dir /path/to/raw/registries --out-dir data/normalized
```

This writes `anthrax_normalized.csv`, `rabies_normalized.csv`, `fmd_normalized.csv`, a
combined `focus_unified.csv` and `QUALITY_REPORT.md`.

Included in the repository and usable without the institutional data:

- `data/literature/` — input tables for the PCA and Bayesian modules, reconstructed from
  published studies (see `data/literature/README.md` for provenance and caveats).
- `data/covariates/kazakhstan_bioclim.tif` — 19 WorldClim 2.1 bioclimatic variables and
  elevation at 10 arc-minutes, clipped to the study region
  (regenerate with `etl/export_covariates.py`).
- `data/boundaries/kazakhstan.geojson` — national boundary used for background masking;
  `data/boundaries/kazakhstan_admin1.geojson` — first-level regions for Figure 3.

## Reproducing the reported runs

```bash
# MaxEnt-equivalent module: random splits + spatial block CV, background inside the boundary
python3 run_maxent.py --disease anthrax --n-iterations 100
python3 run_maxent.py --disease rabies --reservoir-category livestock --n-iterations 100
python3 run_maxent.py --disease rabies --reservoir-category companion --n-iterations 100
python3 run_maxent.py --disease rabies --reservoir-category wildlife  --n-iterations 100 --cv-folds 3

# Space–time permutation scan
python3 run_space_time_scan.py --disease anthrax
python3 run_space_time_scan.py --disease rabies

# PCA and Bayesian modules on the literature-derived tables (no institutional data needed)
python3 run_literature_modules.py
```

Useful options of `run_maxent.py`: `--block-deg` (block size in degrees, `0` disables the
block CV), `--cv-folds`, `--cv-repeats`, `--boundary` (empty string disables the mask).

## Reproducing the paper

Everything except the PCA and Bayesian modules needs the normalized institutional registries in
`data/normalized/` (not in git; see *Data*). Scripts write their outputs to the paths below, and
the committed copies of those outputs are in `results/`. Install the extra dependencies first:

```bash
pip install -r requirements.txt -r requirements-benchmarks.txt
pip install geopandas matplotlib psutil httpx uvicorn threadpoolctl   # experiments and figures
```

| Paper item | Script and command | Output (committed copy) |
|---|---|---|
| Table 2 inputs, Table 7 record counts | `python3 etl/normalize.py --raw-dir <raw> --out-dir data/normalized` | `results/main/QUALITY_REPORT.md` |
| Table 6, anthrax and rabies suitability AUC | `python3 run_maxent.py --disease anthrax --n-iterations 100` (and the three rabies commands under *Reproducing the reported runs*) | `results/main/maxent_*/` |
| Table 6 and Section 6.2, scan clusters | `python3 run_space_time_scan.py --disease anthrax --n-permutations 999` (same for `rabies`) | `results/main/scan_anthrax/`, `results/main/scan_rabies/` |
| Table 6, PCA and Bayesian reproduction | `python3 run_literature_modules.py` | printed to stdout |
| Table 7, stage counts, three duplicate keys, mask effect | `python3 experiments/revision_experiments.py models`, then `scans`, then `python3 experiments/additional_models.py` | `results/revision/audit.json`, `quality_auc.csv`, `quality_clusters.csv`, `conservative_duplicates_*.json` |
| Table 8, timings, workload, memory, concurrent API requests, 9 robustness checks | `python3 experiments/system_experiments.py` | `results/revision/workstation_scaling.csv`, `local_api_concurrency.csv`, `schema_robustness.csv`, `local_poc.json` |
| Table 10, module vs elapid MaxEnt | `python3 benchmarks/maxent_reference.py --disease anthrax` (and the rabies commands in `benchmarks/README.md`) | `results/benchmarks/elapid_*/` |
| Section 6.5, regularization analysis | `python3 experiments/revision_experiments.py models` (module, C grid) and `... elapid` (MaxEnt β grid) | `results/revision/regularization_logistic.csv`, `regularization_elapid.csv` |
| Table 11, module vs SaTScan 10.3.3 | `python3 benchmarks/satscan_export.py --disease anthrax`, run SaTScan with the written `SETTINGS.txt`, then `python3 benchmarks/satscan_compare.py --disease anthrax` (same for `rabies`) | `results/benchmarks/satscan_*/` |
| Table 12, 15 window combinations | `python3 experiments/revision_experiments.py scans` | `results/revision/scan_grid.csv`, `scan_grid_*.csv/json` |
| Section 6.2, rabies geocoding error | `python3 experiments/revision_experiments.py scans` | `results/revision/rabies_error_sensitivity.json` |
| Section 6.3 and Table A1, prior sensitivity | `python3 experiments/revision_experiments.py models` | `results/revision/prior_sensitivity.csv` |
| Section 6.6, fifth disease (FMD) | `python3 benchmarks/reuse_fifth_disease.py --disease fmd` | `results/main/reuse_fmd/` |
| Section 6.6 and Table A1, 19-line adapter and loader sizes | `python3 experiments/reuse_extension.py` (no institutional data needed) | `results/revision/reuse_extension.json`, `prior_grid_plugin_result.csv` |
| Section 6.7, retrospective temporal transfer | `python3 experiments/revision_experiments.py models` | `results/revision/temporal_transfer.csv` |
| Section 7.3 and Tables A2–A3, correlations, ablation, jackknife | `python3 experiments/revision_experiments.py models` | `results/revision/predictor_correlations.json`, `predictor_ablation.csv`, `jackknife_paired.csv` |
| Figures 3–4 | `python3 experiments/make_revision_figures.py` (after the `models` run) | `figures/fig3_anthrax_clusters_map.png`, `figures/fig4_predictor_importance.png` |
| Figures 5–6 | `python3 make_figures.py` | `figures/fig_pca_biplot.png`, `figures/fig_bayesian_posterior.png` |

`experiments/revision_experiments.py` takes one argument, `models`, `scans` or `elapid`. On a
laptop (Apple M3), `scans` takes about 10 minutes and `elapid` about 25 minutes. Each run also
writes `environment_<part>.json` with package versions and the SHA-256 of the normalized inputs.
Block cross-validation uses five fold assignments in the revision experiments, ten in
`run_maxent.py` (Table 6) and three in the regularization grids; the manuscript explains the
resulting small differences in block AUC. The `--sensitivity` option of
`run_space_time_scan.py` runs the earlier 5 × 2 grid (199 permutations), which Table 12
supersedes.

Boundaries in `data/boundaries/`: `kazakhstan.geojson` (Natural Earth 1:50m Admin 0, public
domain; background masking) and `kazakhstan_admin1.geojson` (Natural Earth 1:10m Admin 1, public
domain; region outlines in Figure 3).

## Benchmarks

`benchmarks/` contains the like-for-like comparisons with a full MaxEnt implementation and
with SaTScan, and the fifth-disease reuse experiment; see `benchmarks/README.md`.

## API and web client

```bash
./run_local.sh
```

Starts the API on http://127.0.0.1:8000 (interactive docs at `/docs`) and the web client on
http://127.0.0.1:8080. Upload a CSV in the `focus` schema and run the space–time scan module.
`python3 -m api.smoke_test` runs an end-to-end API check without a live server.

## Adding a module

Subclass `RiskModule` in `pipeline/core.py`, declare `name`, `input_kind` and `produces`,
implement `validate(ctx)` and `run(ctx)`, and return a `RiskResult`. No change to the ETL
stage or to the contract is required.

## License

See `LICENSE`.
