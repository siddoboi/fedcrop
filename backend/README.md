# Stage H + I — results API and dashboard

Minimum-scope build: one FastAPI service and one page. No model is loaded and no
inference runs; every figure is read from the frozen artifacts in
`artifacts/results/`.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload          # http://127.0.0.1:8000
```

The dashboard is served at `/`; the API is under `/api`.

## Endpoints

| Route | Returns |
|---|---|
| `GET /api/health` | which artifacts exist, and which required ones are missing |
| `GET /api/headline` | the figures in the page header |
| `GET /api/ablation` | per-arm aggregate over the seed runs (computed, not stored) |
| `GET /api/agreement` | attribution agreement table and the agronomic check |
| `GET /api/complexity` | parameters, inference, communication |
| `GET /api/mu-sweep` | FedProx proximal sweep |
| `GET /api/fidelity` | AOPC results, curves stripped |
| `GET /api/confound` | month-variable pairs ranked by pooled-vs-within gap |
| `GET /api/clients` | per-client rows, mean yield, deficit years |
| `GET /api/results/{name}` | any artifact raw |
| `GET /api/bundle` | everything the dashboard needs, one request |

## Design notes

**Aggregation happens server-side.** `ablation.json` stores one row per arm per seed.
The API collapses those into mean and sample standard deviation so the page does not
re-implement the statistics, and so any other consumer gets the same numbers.

**Missing stages are reported, not hidden.** `ood`, `perturbation` and `significance`
are declared optional in `ARTIFACTS`. When absent, `/api/health` names them and the
dashboard renders an explicit "not run" state rather than an empty panel. Requesting
one directly returns 409 with an instruction, not a 500.

**No build step on the frontend.** One HTML file, no framework, no CDN, inline SVG
charts. It renders offline and will still render unchanged in six months. The original
plan was React + Vite with five views; that was cut to one page to fit the schedule,
and the cut is deliberate rather than incomplete.

**Palette** is taken from `Literature_Review_Deck.pptx` so the dashboard, the deck and
the report read as one piece of work: forest `#1C2B1C` / `#2C5F2D`, sage `#D8DED8`,
moss `#97BC62` for positive, clay `#B85042` for negative, Cambria headings over
Calibri body.

## Regenerating the screenshots

```bash
uvicorn backend.app:app --port 8000 &
python scripts/07_screenshots.py          # -> docs/screenshots/
```
