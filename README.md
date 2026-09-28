# fedcrop

**Federated and Explainable AI Framework for Predictive Crop Yield Analytics
under Climate Change Conditions**

District-level rice yield prediction across 19 Indian states (1990-2015), with
each state acting as a federated client that never shares its data. The project
compares centralised, local-only and federated training (FedAvg, FedProx,
FedPer), explains every model with SHAP, and tests robustness to each state's
own drought years.

## Headline results (5 seeds, test years 2012-2015)

| Method | R² | Skill vs district trend [95% CI] |
|---|---|---|
| Centralised (upper bound, not private) | 0.448 | +0.136 [0.115, 0.158] |
| Local only | 0.391 | +0.047 [0.023, 0.073] |
| **FedPer** | 0.364 | +0.005 [-0.015, 0.022] |
| FedAvg | 0.295 | -0.103 [-0.116, -0.092] |
| FedProx | 0.292 | -0.108 [-0.119, -0.097] |
| District trend baseline | 0.348 | 0 (reference) |

- **Heterogeneity dominates.** FedAvg and FedProx are significantly worse than
  centralised (Wilcoxon p = 0.0002). FedPer, which keeps a personal head per
  state, is not distinguishable from centralised per state (p = 0.44).
- **Explanations change under federation.** Feature-ranking agreement with the
  centralised model (Kendall τ) is 0.51 for FedPer and 0.38 to 0.39 for
  FedAvg/FedProx, against a seed-to-seed control of 0.72.
- **Drought robustness.** With each state's own driest years withheld from
  training, federated models show no measurable loss of R² on those years;
  local-only models lose 0.076.
- **Climate signal is confounded.** Pooled and within-district correlations
  flip sign for several climate variables (Simpson's paradox). Climate helps
  only in combination with farm covariates.

## Pipeline

| Stage | Script | Gate |
|---|---|---|
| A Data verification | `00_verify.py` | 16 checks |
| B Cleaning, features, baselines | `01_*`, `02_*` | baselines reproduce |
| C-D Training (5 arms x 5 seeds) | `03_train_all.py` | ablation table |
| E Explainability | `04_explain.py` | seed-control τ |
| F-G Robustness, freeze | `05_evaluate.py` | Gate G: 11 result files |
| H Backend | `backend/app.py`, FastAPI, read-only | 52 API tests |
| I Frontend | being rebuilt as a separate app | |

## Setup (Windows, Python 3.12)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the ICRISAT workbook (`.xls`, Mendeley DOI 10.17632/ywp3y5j9vv.1) in
`data/raw/`. It is not committed.

## Reproduce

```powershell
python scripts\00_verify.py
python scripts\03_train_all.py
python scripts\03_train_all.py --protocol climate_stress
python scripts\04_explain.py
python scripts\05_evaluate.py
python -m pytest -q
```

Training is seeded and deterministic: a full retrain reproduces the committed
ablation table exactly.

## Key protocol choices

- Temporal split: train 1990-2009, validation 2010-2011, test 2012-2015.
- Scaling fitted per state on training years only; never across states.
- Drought years are chosen per state, not nationally, because a national mean
  cancels opposite regional signals (2015: Karnataka -27.6%, Himachal +21.6%).
- Drought robustness is measured as R² drop. RMSE falls in drought years for
  every model because yield varies less, so it is not a fair measure.
- Climate scenarios are a sensitivity analysis over observed conditions, not a
  climate projection.

## Known limitations

- Climate variables are district-wide means with no cropland mask.
- Rice yield is annual, not season-specific.
- Windspeed behaves like a proxy for something else.
- Telangana has no training years, so 19 of 20 states are federated clients.
