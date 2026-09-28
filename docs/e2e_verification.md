# End-to-end verification

Run 2026-09-28 14:51 UTC · Python 3.12.10 · Windows AMD64

Produced by `scripts/08_verify_e2e.py`. Every line below is the recorded result of an executed check.

## 1. Test suite

```
$ python -m pytest tests -q
78 passed, 1 skipped in 11.14s
```

- pipeline: 27 passed in 8.00s
- API: 51 passed, 1 skipped in 2.61s

## 2. Pipeline gate A

```
$ python scripts/00_verify.py
GATE A PASSED: 16 checks
```

Gate A re-derives the row counts, the pooled and within-district correlations, the JJAS deficit figures, the fertiliser collinearity and the baseline R² values from the source `.xls` and compares each against the documented constant. It is the check that the reported numbers were not transcribed by hand.

## 3. API routes

| Route | Expected | Got | ms | Bytes |
|---|---:|---:|---:|---:|
| `/api/health` | 200 | 200 | 141 | 1,038 |
| `/api/headline` | 200 | 200 | 29 | 309 |
| `/api/ablation` | 200 | 200 | 26 | 3,045 |
| `/api/agreement` | 200 | 200 | 3 | 1,527 |
| `/api/complexity` | 200 | 200 | 2 | 1,656 |
| `/api/mu-sweep` | 200 | 200 | 4 | 846 |
| `/api/fidelity` | 200 | 200 | 21 | 704 |
| `/api/confound` | 200 | 200 | 3 | 1,030 |
| `/api/clients` | 200 | 200 | 26 | 1,767 |
| `/api/bundle` | 200 | 200 | 69 | 79,893 |
| `/api/ood` | 200 | 200 | 57 | 25,502 |
| `/api/scenarios` | 200 | 200 | 17 | 39,583 |
| `/api/scenarios?client=Jharkhand` | 200 | 200 | 3 | 3,552 |
| `/api/predictions` | 200 | 200 | 47 | 456,345 |
| `/api/predictions?client=Atlantis` | 404 | 404 | 2 | 42 |
| `/api/significance` | 200 | 200 | 23 | 5,235 |
| `/api/baselines` | 200 | 200 | 3 | 2,739 |
| `/api/federation` | 200 | 200 | 17 | 88,624 |
| `/api/attributions` | 200 | 200 | 39 | 221,488 |
| `/api/state/Uttar%20Pradesh` | 200 | 200 | 8 | 3,409 |
| `/api/state/Nowhere` | 404 | 404 | 5 | 36 |
| `/api/results/meta` | 200 | 200 | 23 | 3,921 |
| `/api/results/ood` | 200 | 200 | 7 | 182,955 |
| `/api/results/nonsense` | 404 | 404 | 22 | 40 |
| `/` | 200 | 200 | 5 | 381 |
| `/results` | 404 | 404 | 3 | 22 |

## 4. Artifact inventory as reported by the service

Service status: **ok**

| Artifact | Required | Present | KB |
|---|---|---|---:|
| meta | yes | yes | 6.3 |
| confound | yes | yes | 6.6 |
| baselines | yes | yes | 4.0 |
| ablation | yes | yes | 9.6 |
| federation_history | yes | yes | 134.8 |
| complexity | yes | yes | 2.5 |
| attributions | yes | yes | 276.8 |
| agreement | yes | yes | 2.0 |
| ood | yes | yes | 259.7 |
| scenarios | yes | yes | 280.7 |
| predictions | yes | yes | 609.7 |
| mu_sweep | yes | yes | 1.1 |
| fidelity | yes | yes | 4.5 |
| significance | yes | yes | 6.6 |
| heterogeneity | no | yes | 16.7 |

## 5. Screenshots

| File | KB | Status |
|---|---:|---|
| public_full.png | 533 | ok |
| public_01_header.png | 65 | ok |
| public_02_profile.png | 32 | ok |
| public_03_drivers.png | 195 | ok |
| public_04_reliability.png | 134 | ok |
| dashboard_full.png | 1109 | ok |
| 01_header.png | 70 | ok |
| 02_predictive.png | 176 | ok |
| 03_explanation.png | 226 | ok |
| 04_performance.png | 195 | ok |
| 05_supporting.png | 286 | ok |
| 06_status.png | 116 | ok |

## Verdict

All checks passed: test suite green, every route returned its expected status, the service reported its own artifact inventory, and the full screenshot set is present and non-trivial.
