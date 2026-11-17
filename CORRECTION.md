# Correction: the source's −1 missing-value code

This branch (`fix-missing-codes`) corrects a data-cleaning flaw in the version
on `main`, which is the version submitted with project report v5. `main` is
left unchanged on purpose, so the submitted work stays reproducible.

## What was wrong

The source dataset writes **−1** for "not recorded" in some farm columns. The
cleaning on `main` treated it as a real number.

| Column | Values of −1 in the raw panel |
|---|---|
| Gross irrigated area | 730 |
| Nitrogen consumption | 144 |
| Phosphate consumption | 144 |
| Potash consumption | 145 |

No other column holds a negative value, so −1 is the only code. Climate
columns are not affected.

The effect was concentrated in two states. Assam's and West Bengal's median
irrigated area was −1, and every ratio derived from it (irrigation ratio,
fertiliser per hectare) was negative for those rows.

## What changed

- `config/base.yaml`: `missing_codes: [-1]` and
  `missing_code_strategy: impute` (options: `impute`, `drop`, `keep`).
  `keep` reproduces `main` exactly.
- `src/fedcrop/cleaning.py`: `recode_missing_codes()` turns −1 into missing
  **before** the derived ratios are computed. The existing per-state
  imputation (training-year median) then fills it.
- `scripts/01_build_features.py`: records the recoding and, per state, how
  many real training values each imputed median comes from (in `meta.json`).
- `backend/predict.py`: an uploaded −1 is named as the missing-value code
  instead of "cannot be negative".
- 9 new tests (`tests/test_missing_codes.py`, `tests/test_upload_missing_code.py`).
- The full pipeline (steps 01 to 05, the μ sweep, the climate-stress run and the
  feature-group ablation) was rerun and the demo files were regenerated.

Row counts are unchanged: 10,223 rows, split 7,579 / 878 / 1,766.

## Results before and after (5 seeds)

| Measure | `main` (report v5) | This branch |
|---|---|---|
| Centralised R² | 0.448 | 0.424 |
| Centralised skill vs trend | +0.136 | +0.099 |
| Local-only R² | 0.391 | 0.392 |
| FedPer R² | 0.364 | 0.357 |
| FedAvg R² | 0.295 | 0.295 |
| FedProx R² | 0.292 | 0.294 |
| FedPer vs centralised, Wilcoxon p | 0.44 | 0.29 |
| Local-only vs centralised, Wilcoxon p | 0.89 | 0.62 |
| Explanation agreement τ, seed control | 0.719 | 0.725 |
| τ centralised vs FedPer / FedAvg | 0.508 / 0.387 | 0.546 / 0.399 |
| Top-10 Jaccard retained, FedPer / FedAvg | 87% / 73% | 91% / 89% |
| Agronomic plausibility check (top 10) | PASS | PASS |
| Fidelity (all arms) | PASS | PASS |
| Drought R² drop, FedPer | −0.20 | −0.154 |
| +2 °C, −20% rain: Jharkhand / Punjab | −33.3% / +8.5% | −36.4% / +9.4% |

### Feature-group ablation, skill vs trend

| Features | Centralised | FedPer |
|---|---|---|
| Climate + covariates | +0.099 | −0.006 |
| Climate only | −0.065 | −0.161 |
| Covariates only | −0.060 | +0.015 |
| Neither | −0.425 | −0.445 |

## Which claims hold

**Hold**
- FedPer is statistically on par with centralised per state (p = 0.29).
- FedPer's explanations agree most closely with centralised (τ), and the gap to the
  seed control is still attributable to federation.
- The agronomic plausibility check still passes: 4 of the centralised model's top 10
  features are monsoon months, none post-monsoon.
- FedPer is the most robust arm on each state's driest monsoons.
- Plain federation (FedAvg, FedProx) falls below the trend baseline.
- For the centralised model, neither feature group alone beats the trend; only
  both together do. For FedPer, climate adds nothing (as in `main`).

**Weaken**
- Centralised accuracy is lower (R² 0.424 instead of 0.448; skill +0.099 instead
  of +0.136). Part of the earlier accuracy came from the −1 values acting as a
  state marker.
- The FedProx sweep no longer falls at every step of μ: no μ helps, and strong μ hurts.
- On top-10 overlap alone, FedPer's lead over FedAvg and FedProx is now small
  (91% against 89% of the seed control).

## Remaining limitation

Imputation can only use the real values a state has. West Bengal's irrigated
area is the median of **12** real training values and Assam's of **45**, so for
those states the column is close to a constant. Results for those two states,
including their +10% to +11% scenario response, should not be quoted.

Demo files: 412 test rows now carry an imputed farm figure and are left out of
the demos, so Maharashtra and West Bengal have no demo rows.

## Reproduce

```
git checkout fix-missing-codes
python scripts/01_build_features.py
python scripts/02_baselines.py
python scripts/03_train_all.py
python scripts/03_train_all.py --mu-sweep
python scripts/03_train_all.py --protocol climate_stress
python scripts/03_train_all.py --features climate     # and covariates, none
python scripts/03_train_all.py                        # restores the full-feature checkpoints and scalers
python scripts/04_explain.py --topk 10
python scripts/05_evaluate.py
python scripts/10_make_demo_data.py
```

Set `missing_code_strategy: keep` in `config/base.yaml` to reproduce `main`.
