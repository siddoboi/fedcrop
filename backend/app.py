"""
FastAPI service for the fedcrop results dashboard.

Stage H. Serves the frozen evaluation artifacts produced by the pipeline in
`scripts/`. No model is loaded and no inference runs here: every figure the
dashboard shows was computed offline and written to artifacts/results/*.json.

The service does three things:
  1. exposes each raw artifact under /api/results/{name}
  2. computes per-arm aggregates over the seed runs in ablation.json
  3. reports which artifacts are present, so the UI can degrade honestly
     rather than render blanks for stages that have not been run
"""

from __future__ import annotations

import json
import statistics as stats
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
RESULTS_DIR = REPO_ROOT / "artifacts" / "results"
FRONTEND_DIR = REPO_ROOT / "frontend"

# Artifacts the pipeline can emit. `required` marks the ones the dashboard
# needs to render its core views; the rest are optional and their absence is
# reported rather than hidden.
ARTIFACTS: dict[str, bool] = {
    "meta": True,
    "baselines": True,
    "ablation": True,
    "agreement": True,
    "complexity": True,
    "mu_sweep": True,
    "fidelity": True,
    "confound": True,
    "heterogeneity": False,
    "attributions": False,
    "federation_history": False,
    # Stage F/G outputs. Not yet committed to the repo; the dashboard shows an
    # explicit "not run" state for these instead of empty panels.
    "ood": False,
    "perturbation": False,
    "significance": False,
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

BASELINE_ARMS = {"global_mean", "district_mean", "district_trend"}


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_artifact(name: str) -> Any:
    """Read one artifact from disk. Cached for the process lifetime."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(name)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


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


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = stats.mean(clean)
    sd = stats.stdev(clean) if len(clean) > 1 else 0.0
    return mean, sd


def ablation_summary() -> dict[str, Any]:
    """
    Collapse the per-seed ablation runs into one row per arm.

    Baselines are deterministic and carry a single run with skill_vs_trend
    null, since they are what skill is measured against. Learned arms carry
    one run per seed and are reported as mean +/- sample sd.
    """
    raw = load_artifact("ablation")
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
    """The handful of figures the dashboard header shows."""
    meta = load_artifact("meta")
    complexity = load_artifact("complexity")
    summary = ablation_summary()

    by_arm = {row["arm"]: row for row in summary["rows"]}
    comms = {c["algorithm"]: c for c in complexity["communication"]}

    fedavg_kb = comms.get("fedavg", {}).get("kb_per_round")
    fedper_kb = comms.get("fedper", {}).get("kb_per_round")
    saving = None
    if fedavg_kb and fedper_kb:
        saving = (fedavg_kb - fedper_kb) / fedavg_kb * 100.0

    # Telangana is dropped from federated runs: too few rows to fit a scaler.
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
    The month/variable pairs where pooling most distorts the relationship.

    Ranked by how far the pooled correlation sits from the within-district
    one, which is the quantity the project's G2 argument rests on.
    """
    rows = load_artifact("confound")
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
# app
# --------------------------------------------------------------------------

app = FastAPI(
    title="fedcrop results API",
    version="1.0.0",
    description=(
        "Serves frozen evaluation artifacts for the federated and explainable "
        "crop yield framework. Read-only; no model is loaded."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


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


@app.get("/api/agreement")
def agreement() -> Any:
    return load_artifact("agreement")


@app.get("/api/complexity")
def complexity() -> Any:
    return load_artifact("complexity")


@app.get("/api/mu-sweep")
def mu_sweep() -> Any:
    return load_artifact("mu_sweep")


@app.get("/api/fidelity")
def fidelity() -> Any:
    """Fidelity without the AOPC curves, which the dashboard does not plot."""
    return [
        {k: v for k, v in row.items() if not k.startswith("curve_")}
        for row in load_artifact("fidelity")
    ]


@app.get("/api/confound")
def confound(limit: int = 8) -> dict[str, Any]:
    return {"rows": confound_highlights(limit), "n_total": len(load_artifact("confound"))}


@app.get("/api/clients")
def clients() -> dict[str, Any]:
    meta = load_artifact("meta")
    rows = [
        {"client": name, **values} for name, values in meta["clients"].items()
    ]
    rows.sort(key=lambda r: r["rows"], reverse=True)
    return {"rows": rows, "n": len(rows)}


@app.get("/api/state/{name}")
def state_view(name: str) -> dict[str, Any]:
    """Everything the public page shows for one state.

    Reported, not predicted. The service holds no model, so this endpoint
    returns what was measured on 1990-2015 records for that client: its yield
    profile, the driver ranking the federated model learned there, and how
    predictable the state was for the trend baseline.
    """
    meta = load_artifact("meta")
    if name not in meta["clients"]:
        raise HTTPException(status_code=404, detail=f"unknown state '{name}'")

    profile = meta["clients"][name]
    attrs = load_artifact("attributions")

    # FedPer is the federated arm that keeps a local head, so its per-client
    # ranking is the one that reflects this state rather than the average.
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

    trend = next((t for t in load_artifact("baselines")["per_client_trend"]
                  if t["client"] == name), None)

    national = attrs.get("global", {}).get(source or "centralised", {})
    national_top = [f for f, _ in sorted(national.items(), key=lambda kv: -kv[1])[:8]]

    yields = [c["mean_yield"] for c in meta["clients"].values()]
    rank = sorted(yields, reverse=True).index(profile["mean_yield"]) + 1

    return {
        "state": name,
        "profile": {**profile, "yield_rank": rank, "n_states": len(yields)},
        "drivers": drivers,
        "driver_source": source,
        "attribution_available": bool(drivers),
        "trend_baseline": trend,
        "national_top_features": national_top,
        "record": {"year_min": meta["year_min"], "year_max": meta["year_max"]},
    }


@app.get("/api/results/{name}")
def raw_artifact(name: str) -> Any:
    if name not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"unknown artifact '{name}'")
    try:
        return load_artifact(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail=(
                f"artifact '{name}' has not been generated. "
                "Run the corresponding pipeline stage first."
            ),
        )


@app.get("/api/bundle")
def bundle() -> dict[str, Any]:
    """Everything the dashboard needs, in one request."""
    return {
        "headline": headline_numbers(),
        "ablation": ablation_summary(),
        "agreement": load_artifact("agreement"),
        "complexity": load_artifact("complexity"),
        "mu_sweep": load_artifact("mu_sweep"),
        # Same shape as GET /api/confound, so the two are interchangeable.
        "confound": {
            "rows": confound_highlights(8),
            "n_total": len(load_artifact("confound")),
        },
        "health": health(),
    }


# Frontend last, so /api/* wins on any path collision.
if FRONTEND_DIR.exists():

    @app.get("/")
    def index() -> FileResponse:
        """Public-facing page: what the model learned, per state."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/results")
    def results_page() -> FileResponse:
        """Project-team view: the full evaluation dashboard."""
        return FileResponse(FRONTEND_DIR / "results.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
