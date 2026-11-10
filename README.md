# fedcrop

**Federated and Explainable AI Framework for Predictive Crop Yield Analytics under Climate Change Conditions**

Final-year Project Based Learning (PBL), B.Tech CSE, Symbiosis Institute of Technology, Nagpur.

fedcrop predicts district-level rice yield (kg/ha) across Indian states. It treats each state as a separate data owner (a federated "client"), trains models without pooling the states' data, explains every prediction, and tests how the models behave under climate stress. Results are served through a FastAPI service and a React dashboard. Users can also upload their own district records as a CSV file to get predictions and explanations.

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Tools, technologies and software](#2-tools-technologies-and-software)
3. [Repository layout](#3-repository-layout)
4. [Option A: run with Docker (quickest)](#4-option-a-run-with-docker-quickest)
5. [Option B: run from source](#5-option-b-run-from-source)
6. [Reproduce the full pipeline](#6-reproduce-the-full-pipeline)
7. [Using the web app](#7-using-the-web-app)
8. [API reference](#8-api-reference)
9. [Results](#9-results)
10. [Rules that are easy to break silently](#10-rules-that-are-easy-to-break-silently)
11. [Limitations](#11-limitations)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What the project does

| Stage | What happens |
|---|---|
| Data | The ICRISAT district-level dataset (Mendeley Data) is cleaned into a panel of 10,223 district-years, 507 districts and 20 state clients. Each row has 12 months × 5 climate variables plus 9 farm covariates (irrigation, fertiliser, area and similar). |
| Baselines | Global mean, district mean and district linear trend. Every model is scored against the district trend ("skill score"). |
| Model | A small GRU (16,033 parameters): a shared recurrent base (13,632) and a prediction head (2,401). |
| Training arms | Centralised (pooled data, the upper reference), local-only (each state alone), FedAvg, FedProx and FedPer (shared base, private per-state head). |
| Explanation | GradientSHAP (Captum) with client-local baselines. Explanation agreement is measured against a seed-to-seed control, and fidelity is checked by perturbation (AOPC). |
| Climate stress | Each state's own driest monsoon years are held out, plus a what-if scenario of +2 °C and −20% rainfall. |
| Delivery | A FastAPI service, a 7-view React dashboard and live CSV prediction, packaged as one Docker image. |

---

## 2. Tools, technologies and software

### Languages and runtimes

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12 | Data pipeline, models, federated training, explanations, API |
| Node.js | 20 LTS | Building the React frontend |
| JavaScript (JSX) | ES2020+ | Frontend |

### Python libraries (`requirements.txt`)

| Library | Purpose |
|---|---|
| **PyTorch** (≥ 2.0) | GRU model, training loops, federated parameter averaging |
| **Captum** (≥ 0.7) | GradientSHAP attributions |
| **pandas**, **NumPy** | Data cleaning, feature tensors |
| **SciPy** | Kendall's tau, Wilcoxon signed-rank test, bootstrap intervals |
| **scikit-learn** | Baseline regressions, metrics |
| **xlrd**, **pyarrow** | Reading the source `.xls`, caching to Parquet |
| **PyYAML** | Configuration (`config/base.yaml`) |
| **FastAPI**, **Uvicorn** | REST API and web server |
| **python-multipart** | CSV file uploads |
| **pytest**, **httpx** | Test suite and API tests |
| **python-pptx** | Generating presentation material |

### Frontend (`frontend/package.json`)

| Library | Purpose |
|---|---|
| **React** 18 | User interface |
| **React Router** 6 | Page navigation (hash routing) |
| **Recharts** 2 | Charts |
| **Vite** 5 | Development server and production build |

### Software and services

| Software | Purpose |
|---|---|
| **Git** and **GitHub** | Version control (`siddoboi/fedcrop`) |
| **Docker Desktop** | Building and running the container image |
| **Visual Studio Code** | Editor |
| **Windows 11 + PowerShell** | Development environment (commands below are for PowerShell; on Linux or macOS use `source .venv/bin/activate` and forward slashes) |
| **Mendeley Data** | Source of the ICRISAT district dataset |

### Methods

Federated Averaging (FedAvg), FedProx (proximal term, μ sweep), FedPer (personalised layers), GRU sequence models, GradientSHAP, Kendall's tau and top-k Jaccard agreement, AOPC fidelity, Wilcoxon signed-rank tests and per-client standardisation.

---

## 3. Repository layout

```
fedcrop/
├── config/base.yaml          # paths, hyperparameters, seeds, dataset filename
├── src/fedcrop/              # the library
│   ├── columns.py            # source column names, feature_names()
│   ├── io_layer.py           # reads the .xls once, caches to Parquet
│   ├── cleaning.py           # cleaning cascade with a per-step audit table
│   ├── features.py           # (N, 12, 5) climate tensor + (N, 9) covariates
│   ├── splits.py             # temporal splits, clients, deficit years
│   ├── scaling.py            # per-client standardisation
│   ├── baselines.py          # global mean, district mean, district trend
│   ├── metrics.py            # RMSE, MAE, R², skill score, worst client
│   ├── model.py              # GRU model (shared base + head)
│   ├── train.py, train_models.py, datasets.py
│   ├── fl/                   # federated loop, aggregation (FedAvg/FedProx/FedPer)
│   ├── xai/                  # attribution, aggregation, agreement, fidelity
│   ├── evaluation/           # perturbation, out-of-distribution, significance
│   ├── export/results_json.py
│   └── complexity.py         # parameters, FLOPs, communication cost
├── scripts/                  # numbered pipeline steps (section 6)
├── backend/
│   ├── app.py                # FastAPI app, results endpoints, serves the frontend
│   └── predict.py            # CSV upload: validate, predict, explain
├── frontend/                 # React + Vite dashboard
├── demo_data/                # ready-made CSV files for the upload page
├── tests/                    # pytest suite
├── artifacts/                # results JSON, scalers, model checkpoints (generated)
├── data/raw/                 # put the source .xls here (not in the repo)
├── Dockerfile, .dockerignore, requirements-docker.txt
└── requirements.txt
```

`artifacts/models/*.pt` and `artifacts/scalers/scalers.json` are **not committed**. They are created by `scripts/03_train_all.py`. Both the web app and the Docker image need them.

---

## 4. Option A: run with Docker (quickest)

This needs a checkout that already has the trained checkpoints (see the note above).

**Prerequisite:** Docker Desktop installed and running.

```powershell
docker build -t fedcrop .
```
The first build takes about 5 minutes. It downloads CPU-only PyTorch and the npm packages.

```powershell
docker run --rm -p 8000:8000 fedcrop
```

Open **http://localhost:8000**. Stop the container with `Ctrl+C`.

- The image is about 430 MB compressed (about 2 GB unpacked).
- It runs as an unprivileged user (uid 1000) and has a health check on `/api/health`.
- It listens on the port in the `PORT` environment variable (default 8000), so a hosting platform can assign its own port:
  ```powershell
  docker run --rm -e PORT=7860 -p 7860:7860 fedcrop
  ```

---

## 5. Option B: run from source

### 5.1 Prerequisites

- Python 3.12
- Node.js 20 LTS (only for the web dashboard)
- Git

### 5.2 Clone and set up Python

```powershell
git clone https://github.com/siddoboi/fedcrop.git
cd fedcrop
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Get the dataset

The dataset is not in this repository. Download it from
https://data.mendeley.com/datasets/ywp3y5j9vv/1 and place the `.xls` file in `data/raw/`. The expected filename is set in `config/base.yaml`.

### 5.4 Run the tests

```powershell
python -m pytest -q
```
Expected result: all tests pass (one is skipped when an optional dependency is absent).

### 5.5 Start the API

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
The interactive API documentation is at http://127.0.0.1:8000/docs.

### 5.6 Start the dashboard (second terminal)

```powershell
cd frontend
npm ci
npm run dev
```
Open **http://localhost:5173**. Vite forwards `/api` requests to the API on port 8000.

To serve a production build from the API itself instead (as the Docker image does):

```powershell
cd frontend
npm run build
cd ..
$env:FEDCROP_FRONTEND_DIR = "frontend\dist"
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Then open http://127.0.0.1:8000.

---

## 6. Reproduce the full pipeline

Run the scripts in order from the repository root with the virtual environment active.

| Step | Command | Output |
|---|---|---|
| Gate A: verify the loader | `python scripts/00_verify.py` | Reproduces sixteen documented constants. **If it fails, the loader is wrong and nothing downstream is meaningful.** |
| Build features | `python scripts/01_build_features.py` | Cleaned panel, confound table, `meta.json` |
| Baselines | `python scripts/02_baselines.py` | `baselines.json` |
| Train all arms (5 seeds) | `python scripts/03_train_all.py` | Checkpoints, scalers, `ablation.json` |
| FedProx μ sweep | `python scripts/03_train_all.py --mu-sweep` | `mu_sweep.json` |
| Climate-stress training | `python scripts/03_train_all.py --protocol climate_stress` | `*_stress_*` checkpoints |
| Feature-group ablation | `python scripts/03_train_all.py --features climate` (also `covariates`, `none`) | `ablation_<group>.json` |
| Explanations | `python scripts/04_explain.py` | Attributions, agreement, fidelity |
| Evaluation and freeze | `python scripts/05_evaluate.py` | Robustness, significance, scenarios, complexity |
| Demo CSV files | `python scripts/10_make_demo_data.py` | `demo_data/*.csv` |

Useful options:

- `03_train_all.py --seeds 42 --arms fedper centralised` trains a subset.
- `04_explain.py --max-rows 300 --topk 5` sets the attribution sample size and top-k.
- `05_evaluate.py --validate-only` checks the frozen results without recomputing them.

Use `--help` on any script for the full list.

Full training on a CPU takes several hours. All results are written to `artifacts/results/` as JSON, and the API reads them from there.

---

## 7. Using the web app

The dashboard has seven views:

| View | Shows |
|---|---|
| **Overview** | Headline results and the method comparison |
| **Yield Prediction** | Accuracy of every arm against the baselines, predicted vs actual |
| **Federated** | FedAvg, FedProx and FedPer compared, the μ sweep, communication cost |
| **Explainable** | Feature attributions, agreement with the centralised model, fidelity |
| **Climate** | Drought hold-out robustness and the +2 °C / −20% rainfall scenario |
| **By State** | Per-state results for any of the 20 clients |
| **Predict from CSV** | Upload your own district records |

### Predicting from a CSV file

1. Open **Predict from CSV**.
2. Download the template, or use a file from `demo_data/`.
3. Upload the file. Every row is validated, and errors are reported per row with the reason.
4. Each valid row is predicted with its own state's FedPer model. If the file has an `actual_yield` column, accuracy is reported too.
5. Click a row to see its explanation: which months and covariates pushed the prediction up or down, in kg/ha.

`demo_data/demo_jharkhand.csv` reproduces the pipeline's own test predictions. `demo_data/demo_with_errors.csv` shows the validation messages. See `demo_data/README.md` for every file.

---

## 8. API reference

All endpoints return JSON. Full schema: `/docs`.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | Service status |
| GET | `/api/bundle` | All dashboard data in one response |
| GET | `/api/headline`, `/api/ablation`, `/api/baselines` | Accuracy results |
| GET | `/api/federation`, `/api/mu-sweep`, `/api/complexity` | Federated learning results |
| GET | `/api/attributions`, `/api/agreement`, `/api/fidelity` | Explanation results |
| GET | `/api/scenarios`, `/api/ood`, `/api/significance` | Climate and robustness results |
| GET | `/api/predictions`, `/api/confound` | Test-set predictions, confound table |
| GET | `/api/clients`, `/api/state/{name}` | Client list, one state's results |
| GET | `/api/results/{name}` | Any results file by name |
| GET | `/api/upload/status`, `/api/upload/columns`, `/api/upload/template` | Upload readiness, required columns, CSV template |
| POST | `/api/upload/predict` | Predictions for an uploaded CSV |
| POST | `/api/upload/explain` | Explanation for one uploaded row |

---

## 9. Results

Test years 2012 to 2015, 5 seeds (mean ± sd). Skill is the R² gain over the district-trend baseline.

| Arm | R² | Skill vs trend |
|---|---|---|
| District trend (baseline) | 0.348 | 0 |
| Centralised | 0.448 ± 0.018 | +0.136 ± 0.029 |
| Local-only | 0.391 ± 0.020 | +0.047 ± 0.031 |
| **FedPer** | 0.364 ± 0.016 | +0.005 ± 0.024 |
| FedAvg | 0.295 ± 0.010 | −0.103 ± 0.015 |
| FedProx | 0.292 ± 0.009 | −0.108 ± 0.015 |

Key findings:

- Under strong differences between states, plain federation (FedAvg, FedProx) falls below the trend baseline. Personalisation (FedPer) makes federation viable: per-state accuracy is not significantly different from centralised (Wilcoxon p = 0.18).
- FedPer's explanations agree most closely with the centralised model's (Kendall's tau 0.508, against a seed-to-seed control of 0.719).
- FedPer is the most robust arm when each state's driest monsoons are held out.
- Under +2 °C and −20% rainfall, predicted yield falls sharply in rain-fed states (Jharkhand) and changes little in heavily irrigated ones (Punjab).

The project report has full tables, confidence intervals and the complexity analysis.

---

## 10. Rules that are easy to break silently

1. Per-client scaling only. A global scaler does not crash, but it invalidates every number. `assert_no_global_scaler` catches it.
2. Fit scalers on training years only.
3. Never use a random split. Keep the temporal structure.
4. Never include `RICE PRODUCTION` (yield = production / area).
5. Never use both `TOTAL FERTILISER` and N/P/K (correlation exactly 1.0).
6. Never impute climate. Drop those rows.
7. Never report R² without `district_trend` beside it.

---

## 11. Limitations

- One crop (rice) and one data source, ending in 2015.
- The source uses −1 as a missing-value code for gross irrigated area (about 10% of rows, mostly Assam and West Bengal) and in a few N/P/K values. The current cleaning treats it as a number. This should be re-cleaned in future work.
- Privacy comes from data locality only. There is no secure aggregation or differential privacy.
- The climate scenario is a sensitivity analysis, not an emission-scenario projection.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `docker` is not recognised | Add Docker Desktop's `resources\bin` folder to PATH, then reopen PowerShell. |
| Port 8000 already in use | Stop the running `uvicorn`, or run with `-p 8001:8000` and open port 8001. |
| Upload page says models are unavailable | `artifacts/models/*_seed42.pt` or `artifacts/scalers/scalers.json` is missing. Run `scripts/03_train_all.py`. |
| Dashboard shows no data | Start the API first, and check that `artifacts/results/` contains the JSON files. |
| `00_verify.py` fails | Check that the `.xls` in `data/raw/` matches the filename in `config/base.yaml`. |
| Slow requests on Windows in development | Use `127.0.0.1` instead of `localhost`. |

---

## Team

Siddhesh Singh, Parth Chaudhari and Shambhavii V. Jaiswal, under the guidance of Dr. Smita Nirkhi Singh (Associate Professor), Symbiosis Institute of Technology, Nagpur.
