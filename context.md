# fedcrop context (read this first in any new chat)

Federated and Explainable AI Framework for Predictive Crop Yield Analytics under
Climate Change Conditions. Final-year PBL, Sem 7. Solo developer (Siddhesh) with
Claude. Repo: github.com/siddoboi/fedcrop. Windows 11, Python 3.12, venv, Node 24.

If anything in a chat contradicts this file, this file wins. Numbers here come
from committed JSON in artifacts/results/, not from memory.

## Status
- Stages A-G DONE, tested (59 tests pass), pushed (commit 6c540f3).
- Gate G passed: 11 of 11 required result files present and valid.
- Stage H DONE: backend/app.py, pure JSON API, no torch, 52 API tests.
- Next: Stage I frontend, a separate app calling the API on port 8000.

## Data (verified)
- ICRISAT District Level Database, Mendeley DOI 10.17632/ywp3y5j9vv.1.
- Rice yield is ANNUAL, not kharif-only (ICRISAT crops.html).
- Climate = unmasked district-wide TerraClimate means (no cropland mask).
- 12,803 raw rows -> 10,223 clean, 501 districts, 1990-2015.
- Cleaning cascade: target present, yield > 0, rice area >= 2.0k ha
  (empirical noise breakpoint), climate complete, |z| <= 3 on training years.
- Dropped: rice production (leakage), total fertiliser (collinear with N+P+K),
  agricultural labour (88.8% missing).
- Features: 12 months x 5 climate vars (max_temp, min_temp, precipitation,
  evapotranspiration, windspeed) + 9 annual covariates = 69 named features.
  columns.py is the single source of truth for names and order.

## Protocol (locked)
- Temporal split: train 1990-2009 (7,579), val 2010-2011 (878),
  test 2012-2015 (1,766).
- Clients = states. 19 federated clients. Telangana has test rows but no
  training rows, so it has no scaler and no model (1,748 usable test rows).
- Per-client scaling fitted on that client's training years only. Never global.
- Imputation per client, training-year medians, never on climate variables.
- Model: GRU(5->64) + 9 covariates -> Dense(32) -> 1. 16,033 params;
  GRU 13,632 shared under FedPer (85%), head 2,401 local.
- FL: FedAvg, FedProx (mu 0.01), FedPer. 120 rounds max, patience 20.
- Seeds 42-46. Every reported number is a 5-seed mean with bootstrap 95% CI.
- Reference baseline for skill: district_trend.

## Results (5 seeds, test 2012-2015)
Baselines R2: global_mean -0.263, district_mean 0.304, district_trend 0.348.

| arm | R2 | skill vs trend [95% CI] | worst-state RMSE |
|---|---|---|---|
| centralised | 0.448 | +0.136 [0.115, 0.158] | 1005 |
| local | 0.391 | +0.047 [0.023, 0.073] | 1150 |
| fedper | 0.364 | +0.005 [-0.015, 0.022] | 1301 |
| fedavg | 0.295 | -0.103 [-0.116, -0.092] | 1110 |
| fedprox | 0.292 | -0.108 [-0.119, -0.097] | 1113 |

- Wilcoxon paired by state vs centralised (test set, RMSE mean over seeds):
  fedavg p=0.0002, fedprox p=0.0002 (worse); fedper p=0.44, local p=0.89
  (not distinguishable).
- FedProx mu sweep: proximal term is active (L2 drift from FedAvg rises
  0.51 -> 2.60) but hurts skill at every mu. State heterogeneity dominates.
- Feature ablation: climate alone and covariates alone do not beat the trend
  baseline; only together. The signal is interactive.
- Confound: pooled vs within-district correlation flips sign for several
  climate variables (Simpson's paradox). Windspeed is likely a proxy.

## Explainability (Stage E)
- GradientSHAP + Integrated Gradients, per-client baselines.
- Kendall tau vs centralised: seed control 0.719 [0.684, 0.753];
  fedper 0.508; fedavg 0.387; fedprox 0.380. No CI overlaps the control, so the
  gap is due to federation, not seed noise.
- Agronomic check at top-10: PASS (3 monsoon months, 0 post-monsoon).
- AOPC fidelity passes for all arms; FedAvg/FedProx high ratios reflect an
  unresponsive model, not better explanations.

## Drought robustness (Stage F)
- Each client's OWN driest years withheld from training (climate_stress
  protocol, *_stress checkpoints). Metric: R2 drop vs the same model on its own
  non-deficit rows. RMSE change is not reported: deficit years have lower
  variance, so RMSE falls for every arm.
- R2 drop, 5 seeds: fedper -0.198 [-0.207, -0.188], fedavg -0.105,
  fedprox -0.093, centralised -0.021, local +0.076 [0.056, 0.100].
- Claim only "no measurable loss for federated models". The reference set is
  small (10-73 rows per state) and the worst state drops 1.0-1.9 in every arm.

## Climate scenarios (Stage F)
- FedPer, 3x3 grid (0/+1/+2 C x -20/0/+20 % rain), mean of 5 seeds.
- Federation mean at +2 C, -20% rain: -4.0%.
- Plausible: Jharkhand -33%, Chhattisgarh -26%, Maharashtra -24%.
- Not agronomic: Rajasthan +9%, West Bengal +12% under warming. Present as
  "what the model responds to", never as a climate projection.

## Lessons (do not repeat)
- Feature-ablation runs once overwrote the normal checkpoints and produced a
  flat scenario grid. Fixed: checkpoints and results carry _feat-<group> and
  _stress suffixes. 05_evaluate.py now fails loudly on a flat grid.
- Earlier docs had wrong baselines (0.367/0.415). Correct: 0.304/0.348.
- District count is 501, not 507.

## Results contract (artifacts/results/, all required)
meta, confound, baselines, ablation, federation_history, complexity,
attributions, agreement, ood, scenarios, predictions (.json).
Optional: fidelity, heterogeneity, mu_sweep, significance.

## Commands
    python scripts\00_verify.py            # Gate A
    python scripts\01_build_features.py
    python scripts\02_baselines.py
    python scripts\03_train_all.py         # normal, all seeds
    python scripts\03_train_all.py --protocol climate_stress
    python scripts\04_explain.py
    python scripts\05_evaluate.py          # Stage F + Gate G
    python -m pytest -q
