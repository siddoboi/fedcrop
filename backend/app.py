"""
FastAPI service for the fedcrop results.

Stage H. A read-only JSON API over the frozen evaluation artifacts produced by
the pipeline in `scripts/`. No model is loaded and no inference runs here:
every figure was computed offline and written to artifacts/results/*.json.

The frontend is a separate application. This service serves no HTML; it only
answers /api/* and allows cross-origin GET so a frontend on another port can
call it.

The service does four things:
  1. exposes each raw artifact under /api/results/{name}
  2. computes small, testable aggregates (per-arm means, per-state metrics,
     federation-wide scenario means) so the frontend does no statistics
  3. reports which artifacts are present, so a frontend can degrade honestly
     rather than render blanks
  4. refuses to report "ok" unless every file Gate G requires is present

Run:
    uvicorn backend.app:app --reload --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import json
import math
import statistics as stats
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
RESULTS_DIR = REPO_ROOT / "artifacts" / "results"

# Artifact name -> required. The required set is Gate G's eleven files
# (src/fedcrop/export/results_json.py REQUIRED_EXPORTS) plus the files this
# API's own routes depend on. A test enforces that Gate G is a subset.
ARTIFACTS: dict[str, bool] = {
    # Gate G contract
    "meta": True,
    "confound": True,
    "baselines": True,
    "ablation": True,
    "federation_history": True,
    "complexity": True,
    "attributions": True,
    "agreement": True,
    "ood": True,
    "scenarios": True,
    "predictions": True,
    # read directly by routes below
    "mu_sweep": True,
    "fidelity": True,
    "significance": True,
    # present in the pipeline, not used by any route yet
    "heterogeneity": False,
}

ARM_LABELS = {
    "global_mean": "Global mean",
    "district_mean": "District mean",
    "district_trend": "District trend",
    "centralised": "Centralised",
    "local": "Local-only",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedper": "FedPer",
}

# Display order: baselines first, then the learned arms.
ARM_ORDER = [
    "global_mean",
    "district_mean",
    "district_trend",
    "centralised",
    "local",
    "fedper",
    "fedavg",
    "fedprox",
]
LEARNED_ARMS = ["centralised", "local", "fedper", "fedavg", "fedprox"]
BASELINE_ARMS = {"global_mean", "district_mean", "district_trend"}


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_artifact(name: str) -> Any:
    """Read one artifact from disk. Cached for the process lifetime, so
    restart the server after regenerating results."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(name)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def require(name: str) -> Any:
    """load_artifact for route handlers: a missing file is a 409, not a 500."""
    try:
        return load_artifact(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail=(f"artifact '{name}' has not been generated. "
                    "Run the corresponding pipeline stage first."),
        )


def artifact_status() -> dict[str, dict[str, Any]]:
    """Which artifacts exist on disk, and how big they are."""
    out: dict[str, dict[str, Any]] = {}
    for name, required in ARTIFACTS.items():
        path = RESULTS_DIR / f"{name}.json"
        out[name] = {
            "present": path.exists(),
            "required": required,
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return out


# --------------------------------------------------------------------------
# small statistics, pure Python so the API needs no numpy
# --------------------------------------------------------------------------


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = stats.mean(clean)
    sd = stats.stdev(clean) if len(clean) > 1 else 0.0
    return mean, sd


def regression_metrics(actual: list[float], pred: list[float]) -> dict[str, Any]:
    """RMSE, MAE and R2 in kg/ha. R2 is None when actual has no variance."""
    pairs = [(a, p) for a, p in zip(actual, pred) if a is not None and p is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "rmse": None, "mae": None, "r2": None}
    err = [p - a for a, p in pairs]
    sse = sum(e * e for e in err)
    mean_a = sum(a for a, _ in pairs) / n
    sst = sum((a - mean_a) ** 2 for a, _ in pairs)
    return {
        "n": n,
        "rmse": math.sqrt(sse / n),
        "mae": sum(abs(e) for e in err) / n,
        "r2": (1.0 - sse / sst) if sst > 0 else None,
    }


# --------------------------------------------------------------------------
# Stage C-E aggregates (unchanged behaviour)
# --------------------------------------------------------------------------


def ablation_summary() -> dict[str, Any]:
    """
    Collapse the per-seed ablation runs into one row per arm.

    Baselines are deterministic and carry a single run with skill_vs_trend
    null, since they are what skill is measured against. Learned arms carry
    one run per seed and are reported as mean +/- sample sd.
    """
    raw = require("ablation")
    runs: list[dict[str, Any]] = raw["runs"]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["arm"], []).append(run)

    rows = []
    for arm in ARM_ORDER:
        if arm not in grouped:
            continue
        group = grouped[arm]
        r2_mean, r2_sd = _mean_sd([r["r2"] for r in group])
        rmse_mean, rmse_sd = _mean_sd([r["rmse"] for r in group])
        mae_mean, _ = _mean_sd([r["mae"] for r in group])
        skill_mean, skill_sd = _mean_sd([r["skill_vs_trend"] for r in group])
        worst_mean, _ = _mean_sd([r["worst_client_rmse"] for r in group])
        xstd_mean, _ = _mean_sd([r["cross_client_rmse_std"] for r in group])
        seconds, _ = _mean_sd([r.get("seconds") or 0.0 for r in group])

        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS.get(arm, arm),
                "is_baseline": arm in BASELINE_ARMS,
                "is_reference": arm == raw.get("reference_model"),
                "n_seeds": len(group),
                "r2": r2_mean,
                "r2_sd": r2_sd,
                "rmse": rmse_mean,
                "rmse_sd": rmse_sd,
                "mae": mae_mean,
                "skill_vs_trend": skill_mean,
                "skill_sd": skill_sd,
                "worst_client_rmse": worst_mean,
                "cross_client_rmse_std": xstd_mean,
                "seconds": seconds,
            }
        )

    return {
        "rows": rows,
        "seeds": raw.get("seeds", []),
        "reference_model": raw.get("reference_model"),
        "feature_group": raw.get("feature_group"),
        "n_runs": len(runs),
    }


def headline_numbers() -> dict[str, Any]:
    """The handful of figures a dashboard header shows."""
    meta = require("meta")
    complexity = require("complexity")
    summary = ablation_summary()

    by_arm = {row["arm"]: row for row in summary["rows"]}
    comms = {c["algorithm"]: c for c in complexity["communication"]}

    fedavg_kb = comms.get("fedavg", {}).get("kb_per_round")
    fedper_kb = comms.get("fedper", {}).get("kb_per_round")
    saving = None
    if fedavg_kb and fedper_kb:
        saving = (fedavg_kb - fedper_kb) / fedavg_kb * 100.0

    # Telangana is dropped from federated runs: no training years, no scaler.
    trainable_clients = sum(1 for c in meta["clients"].values() if c["rows"] >= 50)

    return {
        "rows": meta["rows"],
        "districts": meta["districts"],
        "states": meta["states"],
        "clients_federated": trainable_clients,
        "year_min": meta["year_min"],
        "year_max": meta["year_max"],
        "n_features": meta["n_features"],
        "parameters": complexity["model"]["total_parameters"],
        "state_dict_kb": complexity["model"]["state_dict_kb"],
        "ms_per_sample": complexity["inference"]["ms_per_sample"],
        "fedper_shared_pct": complexity["model"]["fedper_shared_pct"],
        "comm_saving_pct": saving,
        "best_r2": by_arm.get("centralised", {}).get("r2"),
        "best_skill": by_arm.get("centralised", {}).get("skill_vs_trend"),
        "seeds": len(summary["seeds"]),
    }


def confound_highlights(limit: int = 8) -> list[dict[str, Any]]:
    """
    The month/variable pairs where pooling most distorts the relationship,
    ranked by the gap between pooled and within-district correlation.
    """
    rows = require("confound")
    scored = []
    for row in rows:
        pooled = row["pooled_r"]
        within = row["within_district_r"]
        scored.append(
            {
                **row,
                "gap": abs(pooled - within),
                "sign_flip": (pooled * within) < 0,
            }
        )
    scored.sort(key=lambda r: r["gap"], reverse=True)
    return scored[:limit]


# --------------------------------------------------------------------------
# Stage F aggregates (new)
# --------------------------------------------------------------------------


def _numeric_mean_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Group rows by `keys` and average every numeric field across the group.
    Used to collapse per-seed rows without hard-coding their column names."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out = []
    for key, grp in groups.items():
        row: dict[str, Any] = dict(zip(keys, key))
        fields = {f for r in grp for f, v in r.items()
                  if f not in keys and f != "seed"
                  and isinstance(v, (int, float)) and not isinstance(v, bool)}
        for f in sorted(fields):
            row[f], _ = _mean_sd([r.get(f) for r in grp])
        row["n_seeds"] = len({r.get("seed") for r in grp})
        out.append(row)
    return out


def ood_view() -> dict[str, Any]:
    """Drought robustness: R2 drop on each client's own deficit years."""
    raw = require("ood")
    # Copies, never mutate the cached artifact.
    summary = sorted(({**r, "label": ARM_LABELS.get(r["arm"], r["arm"])}
                      for r in raw["summary"]),
                     key=lambda r: r.get("mean_r2_drop") or 0.0)
    per_client = _numeric_mean_by(raw.get("degradation", []), ("arm", "client"))
    per_client.sort(key=lambda r: (r["arm"], r["client"]))
    return {
        "summary": summary,
        "per_client": per_client,
        "deficit_years": raw["deficit_years"],
        "seeds": raw.get("seeds", []),
        "metric": "r2_drop",
        "note": raw.get("note"),
    }


def scenarios_view(client_name: str | None = None) -> dict[str, Any]:
    """Counterfactual climate grid, optionally for one state."""
    raw = require("scenarios")
    responses = raw["responses"]
    clients_present = sorted({r["client"] for r in responses})
    if client_name is not None:
        if client_name not in clients_present:
            raise HTTPException(status_code=404,
                                detail=f"no scenario responses for '{client_name}'")
        responses = [r for r in responses if r["client"] == client_name]

    federation = []
    for g in raw["grid"]:
        vals = [r["mean_pct_change"] for r in raw["responses"]
                if r["delta_temp"] == g["delta_temp"] and r["rain_pct"] == g["rain_pct"]]
        mean, _ = _mean_sd(vals)
        federation.append({**g, "mean_pct_change": mean, "n_clients": len(vals)})

    return {
        "arm": raw.get("arm"),
        "seeds": raw.get("seeds", []),
        "grid": raw["grid"],
        "clients": clients_present,
        "responses": responses,
        "federation_mean": federation,
        "note": raw.get("note"),
    }


def predictions_view(client_name: str | None = None) -> dict[str, Any]:
    """Per-row test predictions plus per-state and overall metrics per arm."""
    raw = require("predictions")
    arms: list[str] = raw["arms"]
    rows = raw["rows"]
    clients_present = sorted({r["client"] for r in rows})
    if client_name is not None:
        if client_name not in clients_present:
            raise HTTPException(status_code=404,
                                detail=f"no predictions for '{client_name}'")
        rows = [r for r in rows if r["client"] == client_name]

    def metrics_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        actual = [r["actual"] for r in subset]
        return {arm: regression_metrics(actual, [r.get(f"pred_{arm}") for r in subset])
                for arm in arms}

    per_client = []
    for name in sorted({r["client"] for r in rows}):
        subset = [r for r in rows if r["client"] == name]
        per_client.append({"client": name, "n": len(subset),
                           "metrics": metrics_for(subset)})

    return {
        "seed": raw["seed"],
        "arms": arms,
        "units": raw.get("units"),
        "split": raw.get("split"),
        "clients": clients_present,
        "overall": metrics_for(rows),
        "per_client": per_client,
        "rows": rows,
        "note": raw.get("note"),
    }


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

app = FastAPI(
    title="fedcrop results API",
    version="2.0.0",
    description=(
        "Read-only JSON API over the frozen evaluation artifacts of the "
        "federated and explainable crop yield framework. No model is loaded."
    ),
)

# Read-only public results, so any origin may GET. The frontend runs as a
# separate app on its own port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> dict[str, Any]:
    """Route index, so the bare address says what the service offers."""
    return {
        "service": "fedcrop results API",
        "docs": "/docs",
        "routes": sorted({r.path for r in app.routes
                          if getattr(r, "path", "").startswith("/api/")}),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = artifact_status()
    missing_required = [n for n, s in status.items() if s["required"] and not s["present"]]
    return {
        "status": "ok" if not missing_required else "degraded",
        "results_dir": str(RESULTS_DIR),
        "artifacts": status,
        "missing_required": missing_required,
        "missing_optional": [
            n for n, s in status.items() if not s["required"] and not s["present"]
        ],
    }


@app.get("/api/headline")
def headline() -> dict[str, Any]:
    return headline_numbers()


@app.get("/api/ablation")
def ablation() -> dict[str, Any]:
    return ablation_summary()


@app.get("/api/baselines")
def baselines() -> Any:
    return require("baselines")


@app.get("/api/federation")
def federation() -> Any:
    """Per-round validation history for each federated arm and seed."""
    return require("federation_history")


@app.get("/api/attributions")
def attributions() -> Any:
    return require("attributions")


@app.get("/api/agreement")
def agreement() -> Any:
    return require("agreement")


@app.get("/api/complexity")
def complexity() -> Any:
    return require("complexity")


@app.get("/api/mu-sweep")
def mu_sweep() -> Any:
    return require("mu_sweep")


@app.get("/api/fidelity")
def fidelity() -> Any:
    """Fidelity without the AOPC curves, which are large and not plotted."""
    return [
        {k: v for k, v in row.items() if not k.startswith("curve_")}
        for row in require("fidelity")
    ]


@app.get("/api/confound")
def confound(limit: int = 8) -> dict[str, Any]:
    return {"rows": confound_highlights(limit), "n_total": len(require("confound"))}


@app.get("/api/ood")
def ood() -> dict[str, Any]:
    return ood_view()


@app.get("/api/scenarios")
def scenarios(client: str | None = None) -> dict[str, Any]:
    return scenarios_view(client)


@app.get("/api/predictions")
def predictions(client: str | None = None) -> dict[str, Any]:
    return predictions_view(client)


@app.get("/api/significance")
def significance() -> Any:
    return require("significance")


@app.get("/api/clients")
def clients() -> dict[str, Any]:
    meta = require("meta")
    rows = [
        {"client": name, **values} for name, values in meta["clients"].items()
    ]
    rows.sort(key=lambda r: r["rows"], reverse=True)
    return {"rows": rows, "n": len(rows)}


@app.get("/api/state/{name}")
def state_view(name: str) -> dict[str, Any]:
    """Everything a per-state page shows for one state.

    Reported, not predicted. The service holds no model, so this returns what
    was measured on 1990-2015 records for that client: its yield profile, the
    driver ranking learned there, how predictable it was, and how the model
    responded to the climate grid.
    """
    meta = require("meta")
    if name not in meta["clients"]:
        raise HTTPException(status_code=404, detail=f"unknown state '{name}'")

    profile = meta["clients"][name]
    attrs = require("attributions")

    # FedPer keeps a local head, so its per-client ranking reflects this state
    # rather than the federation average.
    drivers, source = [], None
    for arm in ("fedper", "centralised", "fedavg"):
        per_client = attrs.get("per_client", {}).get(arm, {})
        if name in per_client:
            ranked = sorted(per_client[name].items(), key=lambda kv: -kv[1])[:8]
            total = sum(v for _, v in per_client[name].items()) or 1.0
            drivers = [{"feature": f, "attribution": v, "share": v / total}
                       for f, v in ranked]
            source = arm
            break

    trend = next((t for t in require("baselines")["per_client_trend"]
                  if t["client"] == name), None)

    national = attrs.get("global", {}).get(source or "centralised", {})
    national_top = [f for f, _ in sorted(national.items(), key=lambda kv: -kv[1])[:8]]

    yields = [c["mean_yield"] for c in meta["clients"].values()]
    rank = sorted(yields, reverse=True).index(profile["mean_yield"]) + 1

    scen = require("scenarios")
    climate = [r for r in scen["responses"] if r["client"] == name]

    return {
        "state": name,
        "profile": {**profile, "yield_rank": rank, "n_states": len(yields)},
        "drivers": drivers,
        "driver_source": source,
        "attribution_available": bool(drivers),
        "trend_baseline": trend,
        "national_top_features": national_top,
        "climate_response": climate,
        "climate_available": bool(climate),
        "record": {"year_min": meta["year_min"], "year_max": meta["year_max"]},
    }


@app.get("/api/results/{name}")
def raw_artifact(name: str) -> Any:
    if name not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"unknown artifact '{name}'")
    return require(name)


@app.get("/api/bundle")
def bundle() -> dict[str, Any]:
    """Everything a results dashboard needs for its first paint, in one request.
    Per-row predictions are excluded for size; fetch /api/predictions."""
    return {
        "headline": headline_numbers(),
        "ablation": ablation_summary(),
        "agreement": require("agreement"),
        "complexity": require("complexity"),
        "mu_sweep": require("mu_sweep"),
        # Same shape as GET /api/confound, so the two are interchangeable.
        "confound": {
            "rows": confound_highlights(8),
            "n_total": len(require("confound")),
        },
        "ood": ood_view(),
        "scenarios": scenarios_view(),
        "significance": require("significance"),
        "health": health(),
    }