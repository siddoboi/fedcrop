# fedcrop

Federated and explainable AI for rice yield prediction across Indian districts.
Stages A and B of the plan in `Implementation_Roadmap_v3.md`.

## Setup, Windows 11 + Python 3.12

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is not in this repo. Download it from
https://data.mendeley.com/datasets/ywp3y5j9vv/1 and put the `.xls` in
`data/raw/`. The expected filename is in `config/base.yaml`.

## Run, in order

```
python scripts/00_verify.py          # Gate A. Must pass before anything else.
python scripts/01_build_features.py  # cleaning, confound table, meta.json
python scripts/02_baselines.py       # the three baselines, baselines.json
python -m pytest tests -q            # leakage and naming guards
```

`00_verify.py` reproduces sixteen documented constants from the loader itself.
If it fails, the loader is wrong and nothing downstream is meaningful.

## What is here

| Module | Purpose |
|---|---|
| `columns.py` | Source column names. Handles the windspeed abbreviations and the `PERCIPITATION` typo. `feature_names()` is the single source of truth for attribution labels. |
| `io_layer.py` | Reads the `.xls` once, caches to parquet. |
| `cleaning.py` | The cleaning cascade with a per-step audit table. Covariates are imputed per client; climate never is. |
| `features.py` | Assembles the `(N, 12, 5)` sequence tensor and `(N, 9)` covariate matrix into a `FeatureBundle`. |
| `splits.py` | Temporal splitting, client partitioning, per-client deficit years. |
| `scaling.py` | Per-client standardisation with `assert_no_global_scaler`. |
| `baselines.py` | Global mean, district mean, district linear trend. |
| `metrics.py` | RMSE, MAE, R², skill score, worst-client RMSE, cross-client std. |

## Measured results

Cleaned panel: 10,223 rows, 507 districts, 20 clients. Train 7,579,
validate 878, test 1,766.

| Baseline | Test R² | RMSE (kg/ha) |
|---|---|---|
| Global mean | −0.263 | 1005 |
| District mean | 0.304 | 746 |
| **District trend** | **0.348** | **722** |

`district_trend` is the reference for `skill_score`. Every model added later
is reported against it.

## Rules that are easy to break silently

1. Per-client scaling only. A global scaler does not crash, it invalidates
   every number. `assert_no_global_scaler` catches it.
2. Fit scalers on training years only.
3. Never a random split. Temporal structure.
4. Never include `RICE PRODUCTION` (yield = production / area).
5. Never both `TOTAL FERTILISER` and N/P/K (correlation exactly 1.0).
6. Never impute climate. Drop those rows.
7. Never report R² without `district_trend` beside it.
