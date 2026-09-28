"""Stage G. Freeze the results.

Eleven JSON files form the contract the backend reads. Nothing is computed
here that was not already computed upstream; this module only serialises and
then verifies.

validate_exports is the actual Gate G. It fails loudly on a missing or
malformed file rather than letting the backend discover the problem later,
because a partially-written contract is worse than an absent one: the API
starts, serves incomplete data, and nobody notices until the dashboard shows
a blank panel.

Scenario responses are precomputed here deliberately. The API then never
loads torch and never runs Captum at request time, which turns deployment
into a static-file problem and removes the latency question entirely.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# filename -> required top-level keys (empty tuple means "any non-empty JSON")
REQUIRED_EXPORTS: dict[str, tuple[str, ...]] = {
    "meta.json": ("rows", "districts", "states", "clients", "cleaning_audit"),
    "confound.json": (),
    "baselines.json": ("reference_model", "table"),
    "ablation.json": ("runs", "seeds", "reference_model"),
    "federation_history.json": (),
    "complexity.json": ("model", "parameters", "asymptotics"),
    "attributions.json": ("global", "per_client", "feature_names"),
    "agreement.json": ("table", "agronomic_check", "baseline_strategy"),
    "ood.json": ("degradation", "summary", "deficit_years"),
    # Promoted from optional: two dashboard views read these directly.
    "scenarios.json": ("grid", "responses"),
    "predictions.json": ("rows", "arms", "seed"),
}

OPTIONAL_EXPORTS = ("fidelity.json", "heterogeneity.json", "mu_sweep.json",
                    "significance.json")


def _clean(obj):
    """Make numpy and pandas types JSON-serialisable, NaN -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_clean(payload), fh, indent=2)
    log.info("wrote %s", path.name)


def export_ood(degradation: pd.DataFrame, summary: pd.DataFrame,
               deficit: dict[str, list[int]], ood_metrics: pd.DataFrame,
               path: Path, per_seed_summary: pd.DataFrame | None = None,
               seeds: list[int] | None = None) -> None:
    payload = {
        "degradation": degradation,
        "summary": summary.reset_index() if summary.index.name else summary,
        "per_client_ood": ood_metrics,
        "deficit_years": deficit,
        "note": "deficit years are per client, not a national list; a national "
                "mean cancels opposite regional signs in the same year. "
                "Report r2_drop; absolute RMSE falls in deficit years because "
                "yield variance is lower.",
    }
    if per_seed_summary is not None:
        payload["per_seed_summary"] = per_seed_summary
    if seeds is not None:
        payload["seeds"] = list(seeds)
    write_json(payload, path)


def export_scenarios(response: pd.DataFrame, grid: list, path: Path,
                     per_seed: pd.DataFrame | None = None,
                     arm: str | None = None,
                     seeds: list[int] | None = None) -> None:
    payload = {
        "grid": [{"delta_temp": dt, "rain_pct": rp} for dt, rp in grid],
        "responses": response,
        "note": "sensitivity analysis over observed conditions, not a climate "
                "projection; no emission pathway is present in the 1990-2015 panel",
    }
    if arm is not None:
        payload["arm"] = arm
    if seeds is not None:
        payload["seeds"] = list(seeds)
    if per_seed is not None:
        payload["per_seed"] = per_seed
    write_json(payload, path)


def export_predictions(preds: pd.DataFrame, arms: list[str], seed: int,
                       path: Path) -> None:
    """One record per test row: keys, actual yield, one pred_<arm> per arm."""
    write_json({
        "rows": preds,
        "arms": list(arms),
        "seed": int(seed),
        "units": "kg per ha",
        "split": "test (2012-2015)",
        "note": "predictions from a single seed; accuracy tables elsewhere "
                "report the mean over all seeds",
    }, path)


def export_significance(ablation_ci: pd.DataFrame, pairwise: pd.DataFrame,
                        path: Path) -> None:
    write_json({"ablation_with_ci": ablation_ci, "pairwise_wilcoxon": pairwise,
                "note": "Wilcoxon is paired by client on the 2012-2015 test set, "
                        "per-client RMSE averaged over seeds"},
               path)


def validate_exports(results_dir: Path, strict: bool = True) -> tuple[bool, list[str]]:
    """Gate G. Every required file must exist, parse, and carry its keys."""
    problems: list[str] = []

    for name, keys in REQUIRED_EXPORTS.items():
        path = results_dir / name
        if not path.exists():
            problems.append(f"MISSING  {name}")
            continue
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            problems.append(f"MALFORMED {name}: {exc}")
            continue
        if not payload:
            problems.append(f"EMPTY    {name}")
            continue
        if keys and isinstance(payload, dict):
            missing = [k for k in keys if k not in payload]
            if missing:
                problems.append(f"KEYS     {name} is missing {missing}")

    present_optional = [n for n in OPTIONAL_EXPORTS if (results_dir / n).exists()]

    ok = not problems
    log.info("\n=== Gate G: results freeze ===")
    log.info("  required files: %d of %d present and valid",
             len(REQUIRED_EXPORTS) - len(problems), len(REQUIRED_EXPORTS))
    log.info("  optional files present: %s",
             ", ".join(present_optional) if present_optional else "none")
    if problems:
        for p in problems:
            log.error("  %s", p)
        log.error("  GATE G FAILED - do not start the backend")
    else:
        log.info("  GATE G PASSED - results frozen, backend may proceed")

    if strict and problems:
        raise RuntimeError(f"Gate G failed with {len(problems)} problem(s)")
    return ok, problems


def manifest(results_dir: Path) -> dict:
    """A checksum-style summary so the API can report which results it serves."""
    import hashlib
    entries = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        raw = path.read_bytes()
        entries[path.name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
        }
    return entries