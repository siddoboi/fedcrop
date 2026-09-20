# End-to-end verification

Run 2026-09-20 23:32 UTC · Python 3.11.15 · Linux x86_64

Produced by `scripts/08_verify_e2e.py`. Every line below is the recorded result of an executed check.

## 1. Test suite

```
$ python -m pytest tests -q
48 passed in 4.11s
```

- pipeline: 22 passed in 3.68s
- API: 26 passed in 0.42s

## 2. Pipeline gate A

```
$ python scripts/00_verify.py
GATE A PASSED: 16 checks
```

Gate A re-derives the row counts, the pooled and within-district correlations, the JJAS deficit figures, the fertiliser collinearity and the baseline R² values from the source `.xls` and compares each against the documented constant. It is the check that the reported numbers were not transcribed by hand.

## 3. API routes

| Route | Expected | Got | ms | Bytes |
|---|---:|---:|---:|---:|
| `/api/health` | 200 | 200 | 5 | 988 |
| `/api/headline` | 200 | 200 | 3 | 309 |
| `/api/ablation` | 200 | 200 | 3 | 3,045 |
| `/api/agreement` | 200 | 200 | 1 | 1,527 |
| `/api/complexity` | 200 | 200 | 1 | 1,656 |
| `/api/mu-sweep` | 200 | 200 | 1 | 846 |
| `/api/fidelity` | 200 | 200 | 1 | 704 |
| `/api/confound` | 200 | 200 | 1 | 1,030 |
| `/api/clients` | 200 | 200 | 1 | 1,767 |
| `/api/bundle` | 200 | 200 | 6 | 9,487 |
| `/api/results/meta` | 200 | 200 | 1 | 3,921 |
| `/api/results/ood` | 409 | 409 | 1 | 95 |
| `/api/results/nonsense` | 404 | 404 | 1 | 40 |
| `/` | 200 | 200 | 1 | 23,364 |

## 4. Artifact inventory as reported by the service

Service status: **ok**

| Artifact | Required | Present | KB |
|---|---|---|---:|
| meta | yes | yes | 6.0 |
| baselines | yes | yes | 3.9 |
| ablation | yes | yes | 9.3 |
| agreement | yes | yes | 2.0 |
| complexity | yes | yes | 2.4 |
| mu_sweep | yes | yes | 1.1 |
| fidelity | yes | yes | 4.3 |
| confound | yes | yes | 6.6 |
| heterogeneity | no | yes | 16.1 |
| attributions | no | yes | 271.2 |
| federation_history | no | yes | 130.8 |
| ood | no | **no** | 0.0 |
| perturbation | no | **no** | 0.0 |
| significance | no | **no** | 0.0 |

Missing optional (pipeline stages not run): ood, perturbation, significance. The service reports these rather than serving empty responses, and a direct request for one returns 409.

## 5. Screenshots

| File | KB | Status |
|---|---:|---|
| dashboard_full.png | 1109 | ok |
| 01_header.png | 70 | ok |
| 02_predictive.png | 176 | ok |
| 03_explanation.png | 226 | ok |
| 04_performance.png | 195 | ok |
| 05_supporting.png | 286 | ok |
| 06_status.png | 116 | ok |

## Verdict

All checks passed: test suite green, every route returned its expected status, the service reported its own artifact inventory, and the full screenshot set is present and non-trivial.
