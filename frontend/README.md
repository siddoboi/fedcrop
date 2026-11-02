# fedcrop frontend

React 18 + Vite + Recharts dashboard over the read-only results API in `backend/`.
It never loads a model: every number comes from the frozen files in `artifacts/results/`.

## Run (two terminals)

```powershell
# terminal 1, repository root
uvicorn backend.app:app --port 8000

# terminal 2
cd frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173. Vite forwards `/api/...` to `http://127.0.0.1:8000`.

## Views

| Route | View | API routes |
|---|---|---|
| `#/overview` | Overview | `/api/bundle` |
| `#/federated` | Federated learning | `/api/federation`, `/api/complexity`, `/api/mu-sweep`, `/api/significance` |
| `#/yield` | Yield prediction | `/api/predictions`, `/api/baselines` |
| `#/explainable` | Explainable AI | `/api/attributions`, `/api/agreement`, `/api/fidelity` |
| `#/climate` | Climate conditions | `/api/scenarios`, `/api/ood`, `/api/confound` |
| `#/state` | By state | `/api/clients`, `/api/state/{name}` |

## Build

```powershell
npm run build      # writes dist/
npm run preview    # serves dist/ on http://localhost:4173, same /api proxy
```

If the built site is served from somewhere other than the API, set
`VITE_API_BASE` before building (see `.env.example`).

## Notes

- Validation RMSE on the Federated learning view is in scaled units (yield
  standardised per state), not kg/ha.
- Climate controls offer only the nine computed combinations
  (0/+1/+2 °C × −20/0/+20 % rainfall); there is no answer for other values.
- Restart the API after regenerating results: it caches each file on first read.
