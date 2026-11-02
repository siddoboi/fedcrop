"""
Predictions on uploaded CSV files.

Unlike the rest of the API, these routes run the trained models at request
time. An uploaded row goes through the same functions as training, imported
from the fedcrop package rather than reimplemented here:

    canonical CSV columns  -> source column names   (columns.py)
    6 raw farm covariates  -> 3 derived ratios       (cleaning.add_derived)
    flat columns           -> (N, 12, 5) + (N, 9)    (features.build_*)
    raw units              -> that state's z-scores  (scaling.apply_scaler)
    model output           -> kg/ha                  (scaling.inverse_transform_target)

Each state is predicted with its own scaler and, for FedPer, its own head, so
only the 19 federated states are accepted. Nothing is imputed: the training
medians are not stored, and climate is never imputed anywhere in the project.

PyTorch and Captum load on first use, so the read-only results routes still
start without them.
"""

from __future__ import annotations

import io
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fedcrop import columns as C  # noqa: E402
from fedcrop.cleaning import add_derived  # noqa: E402
from fedcrop.features import build_covariate_matrix, build_sequence_tensor  # noqa: E402
from fedcrop.scaling import ScalerParams, apply_scaler, inverse_transform_target  # noqa: E402
from fedcrop.splits import ClientData  # noqa: E402

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# Module-level so tests can point them at synthetic artifacts.
PATHS: dict[str, Path] = {
    "models": REPO_ROOT / "artifacts" / "models",
    "scalers": REPO_ROOT / "artifacts" / "scalers" / "scalers.json",
    "meta": REPO_ROOT / "artifacts" / "results" / "meta.json",
}

SEED = 42                      # same seed as predictions.json, so uploads can be checked against it
ARMS = ["fedper", "centralised", "local", "fedavg", "fedprox"]
MAX_ROWS = 2000
MAX_BYTES = 2 * 1024 * 1024
MIN_RICE_AREA = 2.0            # thousand ha, config cleaning.min_rice_area
Z_WARN = 4.0                   # standard deviations from the state's training mean
TRAIN_YEARS = (1990, 2009)
SHAP_SAMPLES = 50

MONTHS = C.MONTHS_FULL
SEQ_VARS = C.SEQUENTIAL_VARS
SEQ_SOURCE = C.sequential_columns(SEQ_VARS, MONTHS)              # 60 source names
SEQ_NAMES = C.feature_names(SEQ_VARS, MONTHS, [])                # 60 canonical names, same order
RAW_COV_SOURCE = list(C.BASE_COVARIATES)                         # 6 source names
# feature_names treats an empty list as "use the defaults", so slice instead of passing [].
RAW_COV_NAMES = C.feature_names(SEQ_VARS, MONTHS, C.BASE_COVARIATES)[len(SEQ_SOURCE):]  # 6 canonical names
ALL_FEATURE_NAMES = C.feature_names(SEQ_VARS, MONTHS, C.ANNUAL_COVARIATES)  # 69, model order

KEY_COLUMNS = ["state", "district_code", "district_name", "year"]
REQUIRED = ["state"] + SEQ_NAMES + RAW_COV_NAMES
OPTIONAL = ["district_code", "district_name", "year", "actual_yield"]
INPUT_ORDER = SEQ_NAMES + RAW_COV_NAMES                          # what a client sends back for /explain

UNITS = {
    "max_temp": "°C", "min_temp": "°C", "precipitation": "mm",
    "evapotranspiration": "mm", "windspeed": "m/s",
    "rice_area": "thousand ha", "gross_cropped_area": "thousand ha",
    "gross_irrigated_area": "thousand ha", "nitrogen_consumption": "tonnes",
    "phosphate_consumption": "tonnes", "potash_consumption": "tonnes",
}

router = APIRouter(prefix="/api/upload", tags=["upload"])


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_scalers() -> dict[str, ScalerParams]:
    with open(PATHS["scalers"], encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: ScalerParams(**v) for k, v in raw.items()}


@lru_cache(maxsize=None)
def load_meta() -> dict:
    with open(PATHS["meta"], encoding="utf-8") as fh:
        return json.load(fh)


def ckpt_path(arm: str) -> Path:
    return PATHS["models"] / f"{arm}_seed{SEED}.pt"


def readiness() -> dict[str, Any]:
    """What is missing before predictions can run, in plain words."""
    problems: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        problems.append("PyTorch is not installed in the API's environment.")

    scalers = None
    if not PATHS["scalers"].exists():
        problems.append("artifacts/scalers/scalers.json is missing. Run scripts/03_train_all.py.")
    else:
        scalers = load_scalers()

    if scalers is not None and PATHS["meta"].exists():
        # 03_train_all.py rewrites scalers.json on every run, including
        # --protocol climate_stress, which fits on a different split. Scaling an
        # upload with those would give plausible, wrong predictions.
        expected = load_meta().get("split_sizes", {}).get("train")
        fitted = sum(p.n_train_rows for p in scalers.values())
        if expected is not None and fitted != expected:
            problems.append(
                f"scalers.json was fitted on {fitted} training rows, not the {expected} of the "
                "normal temporal split. It was probably overwritten by a climate-stress run. "
                "Rerun: python scripts/03_train_all.py")

    arms = [a for a in ARMS if ckpt_path(a).exists()]
    if "fedper" not in arms:
        problems.append(f"No FedPer checkpoint ({ckpt_path('fedper').name}). Run scripts/03_train_all.py.")

    return {
        "ready": not problems,
        "problems": problems,
        "arms_available": arms,
        "states": sorted(scalers) if scalers else [],
        "seed": SEED,
        "max_rows": MAX_ROWS,
        "max_mb": MAX_BYTES / 1024 / 1024,
    }


def require_ready() -> None:
    r = readiness()
    if not r["ready"]:
        raise HTTPException(status_code=503, detail=" ".join(r["problems"]))


@lru_cache(maxsize=None)
def _blob(arm: str):
    import torch
    return torch.load(ckpt_path(arm), map_location="cpu", weights_only=False)


@lru_cache(maxsize=None)
def model_for(arm: str, state: str):
    """The model a state would use under this arm, loaded exactly as
    evaluation/perturbation.py does. Returns None when the arm has no model
    for the state (local-only is trained per state)."""
    from fedcrop.config import load_config
    from fedcrop.fl.param_utils import set_parameters
    from fedcrop.model import build_model

    blob = _blob(arm)
    model = build_model(cfg=load_config(), n_seq_vars=len(SEQ_VARS),
                        n_covariates=len(C.ANNUAL_COVARIATES))
    if arm == "centralised":
        model.load_state_dict(blob)
    elif arm == "local":
        if state not in blob:
            return None
        model.load_state_dict(blob[state])
    else:
        model.load_state_dict(blob["global"])
        if blob.get("heads", {}).get(state):
            set_parameters(model, blob["heads"][state])
    model.eval()
    return model


def reset_caches() -> None:
    for f in (load_scalers, load_meta, _blob, model_for):
        f.cache_clear()


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""


def read_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_BYTES // 1024 // 1024} MB.")
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as exc:  # pandas raises several types for malformed CSV
        raise HTTPException(status_code=400, detail=f"Could not read the file as CSV: {exc}")
    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        raise HTTPException(status_code=400, detail="The file has a header but no rows.")
    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"{len(df)} rows; the limit is {MAX_ROWS}. Split the file.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
        raise HTTPException(
            status_code=400,
            detail=f"{len(missing)} required column(s) missing: {shown}. Download the template for the exact header.")
    return df


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict[int, list[str]], list[str]]:
    """Split rows into accepted and rejected, with a reason for every rejection.

    Returns (accepted frame with numeric columns, rejected rows, warnings by
    accepted row number, file-level notices). Row numbers are spreadsheet line
    numbers: the header is line 1.
    """
    scalers = load_scalers()
    meta_clients = load_meta().get("clients", {}) if PATHS["meta"].exists() else {}
    notices: list[str] = []

    extra = [c for c in df.columns if c not in REQUIRED + OPTIONAL]
    if extra:
        notices.append(f"Ignored {len(extra)} column(s) not in the template: {', '.join(extra[:5])}"
                       + ("…" if len(extra) > 5 else ""))
    has_actual = "actual_yield" in df.columns

    numeric_cols = INPUT_ORDER + (["actual_yield"] if has_actual else []) + (["year"] if "year" in df.columns else [])
    num = {c: pd.to_numeric(df[c].str.strip(), errors="coerce") for c in numeric_cols}

    keep_idx: list[int] = []
    rejected: list[dict] = []
    for i in range(len(df)):
        line = i + 2
        state = df.at[i, "state"].strip()
        reasons: list[str] = []

        if state not in scalers:
            if state in meta_clients:
                reasons.append(f"{state} has no training years after cleaning, so no model exists for it.")
            elif state == "":
                reasons.append("State is empty.")
            else:
                reasons.append(f"Unknown state '{state}'. Expected one of the {len(scalers)} federated states.")

        def bad(cols: list[str]) -> tuple[list[str], list[str]]:
            empty, text = [], []
            for c in cols:
                raw = df.at[i, c]
                if _blank(raw):
                    empty.append(c)
                elif pd.isna(num[c].iat[i]):
                    text.append(c)
            return empty, text

        def short(cols: list[str]) -> str:
            return ", ".join(cols[:3]) + (f" and {len(cols) - 3} more" if len(cols) > 3 else "")

        e, t = bad(SEQ_NAMES)
        if e:
            reasons.append(f"Missing climate value in {short(e)}. Climate is never filled in.")
        if t:
            reasons.append(f"Not a number in {short(t)}.")
        e, t = bad(RAW_COV_NAMES)
        if e:
            reasons.append(f"Missing farm figure in {short(e)}. Uploads cannot be filled in, "
                           "because the training medians are not stored.")
        if t:
            reasons.append(f"Not a number in {short(t)}.")

        gca = num["gross_cropped_area"].iat[i]
        if not pd.isna(gca) and gca <= 0:
            reasons.append("Gross cropped area must be above 0: the derived ratios divide by it.")
        rice = num["rice_area"].iat[i]
        if not pd.isna(rice) and rice < MIN_RICE_AREA:
            reasons.append(f"Rice area {rice:g} thousand ha is below the {MIN_RICE_AREA:g} the models were "
                           "trained on; smaller districts were removed as too noisy.")
        for c in RAW_COV_NAMES:
            v = num[c].iat[i]
            if not pd.isna(v) and v < 0:
                reasons.append(f"{c} cannot be negative.")
        if has_actual:
            raw = df.at[i, "actual_yield"]
            v = num["actual_yield"].iat[i]
            if not _blank(raw) and pd.isna(v):
                reasons.append("actual_yield is not a number.")
            elif not pd.isna(v) and v <= 0:
                reasons.append("actual_yield must be above 0 kg/ha.")

        if reasons:
            rejected.append({"row": line, "state": state, "reasons": reasons})
        else:
            keep_idx.append(i)

    out = pd.DataFrame({c: num[c].iloc[keep_idx].to_numpy() for c in numeric_cols})
    out.insert(0, "row", [i + 2 for i in keep_idx])
    out.insert(1, "state", [df.at[i, "state"].strip() for i in keep_idx])
    for k in ("district_code", "district_name"):
        out[k] = [df.at[i, k].strip() if k in df.columns else "" for i in keep_idx]
    if "year" not in out:
        out["year"] = np.nan
    if not has_actual:
        out["actual_yield"] = np.nan

    in_train = int(((out["year"] >= TRAIN_YEARS[0]) & (out["year"] <= TRAIN_YEARS[1])).sum())
    if in_train:
        notices.append(
            f"{in_train} row(s) fall in {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]}, the training period. The models "
            "may have seen those exact rows, so accuracy on them is not a fair test. Use 2012–2015 for that.")
    return out, rejected, {}, notices


# --------------------------------------------------------------------------
# features and prediction
# --------------------------------------------------------------------------


def to_tensors(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Canonical columns -> the exact (N,12,5) and (N,9) arrays training used."""
    src = pd.DataFrame({s: rows[c].to_numpy(dtype=float) for s, c in zip(SEQ_SOURCE, SEQ_NAMES)})
    for s, c in zip(RAW_COV_SOURCE, RAW_COV_NAMES):
        src[s] = rows[c].to_numpy(dtype=float)
    src = add_derived(src)
    return (build_sequence_tensor(src, MONTHS, SEQ_VARS),
            build_covariate_matrix(src, C.ANNUAL_COVARIATES))


def scale(state: str, x_seq: np.ndarray, x_cov: np.ndarray) -> ClientData:
    n = len(x_seq)
    return apply_scaler(ClientData(state, x_seq, x_cov, np.zeros(n, dtype=np.float32), pd.DataFrame()),
                        load_scalers()[state], scale_target=False)


def range_warnings(state: str, scaled: ClientData) -> list[list[str]]:
    """Per row, the inputs more than Z_WARN SDs from this state's training mean."""
    flat = np.concatenate([scaled.x_seq.reshape(len(scaled.x_seq), -1), scaled.x_cov], axis=1)
    out = []
    for r in np.abs(flat):
        idx = np.where(r > Z_WARN)[0]
        if len(idx):
            worst = idx[np.argmax(r[idx])]
            out.append([f"{len(idx)} value(s) far outside {state}'s training range "
                        f"(largest: {ALL_FEATURE_NAMES[worst]}, {r[worst]:.1f} SD). "
                        "The prediction is an extrapolation."])
        else:
            out.append([])
    return out


def predict_scaled(arm: str, state: str, scaled: ClientData) -> np.ndarray | None:
    import torch
    model = model_for(arm, state)
    if model is None:
        return None
    with torch.no_grad():
        p = model(torch.from_numpy(np.ascontiguousarray(scaled.x_seq)).float(),
                  torch.from_numpy(np.ascontiguousarray(scaled.x_cov)).float()).numpy()
    return inverse_transform_target(p, load_scalers()[state])


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    m = ~(np.isnan(actual) | np.isnan(pred))
    a, p = actual[m], pred[m]
    if len(a) == 0:
        return {"n": 0, "rmse": None, "mae": None, "r2": None}
    err = p - a
    sst = float(((a - a.mean()) ** 2).sum())
    return {"n": int(len(a)), "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err ** 2).sum() / sst) if sst > 0 else None}


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def run_predictions(df: pd.DataFrame, arms: list[str]) -> dict[str, Any]:
    accepted, rejected, _, notices = validate(df)
    rows_out: list[dict] = []
    preds = {a: np.full(len(accepted), np.nan) for a in arms}
    warnings: list[list[str]] = [[] for _ in range(len(accepted))]

    for state, idx in accepted.groupby("state").groups.items():
        idx = np.asarray(idx)
        part = accepted.loc[idx]
        x_seq, x_cov = to_tensors(part)
        scaled = scale(state, x_seq, x_cov)
        for j, w in zip(idx, range_warnings(state, scaled)):
            warnings[j].extend(w)
        for a in arms:
            p = predict_scaled(a, state, scaled)
            if p is None:
                for j in idx:
                    warnings[j].append(f"No {a} model exists for {state}.")
            else:
                preds[a][idx] = p

    for j, r in accepted.reset_index(drop=True).iterrows():
        rows_out.append({
            "row": int(r["row"]), "state": r["state"],
            "district_code": r["district_code"], "district_name": r["district_name"],
            "year": None if pd.isna(r["year"]) else int(r["year"]),
            "actual": _num(r["actual_yield"]),
            "pred": {a: _num(preds[a][j]) for a in arms},
            "warnings": warnings[j],
            "x": [float(r[c]) for c in INPUT_ORDER],
        })

    has_actual = bool(accepted["actual_yield"].notna().any()) if len(accepted) else False
    actual = accepted["actual_yield"].to_numpy(dtype=float) if len(accepted) else np.array([])
    out_metrics, per_state = None, []
    if has_actual:
        out_metrics = {a: metrics(actual, preds[a]) for a in arms}
        for state, idx in accepted.groupby("state").groups.items():
            idx = np.asarray(idx)
            per_state.append({"state": state, "n": int(len(idx)),
                              "metrics": {a: metrics(actual[idx], preds[a][idx]) for a in arms}})

    return {
        "summary": {
            "rows_total": int(len(df)), "accepted": int(len(accepted)), "rejected": len(rejected),
            "states": sorted(accepted["state"].unique().tolist()) if len(accepted) else [],
            "has_actual": has_actual, "primary": arms[0], "compared": arms[1:], "seed": SEED,
            "rows_with_warnings": sum(1 for w in warnings if w),
        },
        "input_order": INPUT_ORDER,
        "rows": rows_out,
        "rejected": rejected,
        "metrics": out_metrics,
        "per_state": per_state,
        "notices": notices,
    }


def explain_row(state: str, arm: str, x: list[float]) -> dict[str, Any]:
    """GradientSHAP for one row against that state's average training year.

    The pipeline's client_mean baseline averages the rows being explained; for
    a single row that is the row itself and every attribution collapses to
    zero. In scaled space the state's training mean is exactly the origin, so
    the zero baseline means "a typical year in this state". It uses only that
    state's own scaler, so it stays federated-legal.
    """
    import torch
    from fedcrop.xai.attribution import gradient_shap

    if len(x) != len(INPUT_ORDER):
        raise HTTPException(status_code=400, detail=f"Expected {len(INPUT_ORDER)} values, got {len(x)}.")
    row = pd.DataFrame([dict(zip(INPUT_ORDER, x))])
    x_seq, x_cov = to_tensors(row)
    scaled = scale(state, x_seq, x_cov)
    model = model_for(arm, state)
    if model is None:
        raise HTTPException(status_code=404, detail=f"No {arm} model exists for {state}.")

    inputs = (torch.from_numpy(scaled.x_seq).float(), torch.from_numpy(scaled.x_cov).float())
    base = (torch.zeros(1, *scaled.x_seq.shape[1:]), torch.zeros(1, scaled.x_cov.shape[1]))
    seq_a, cov_a = gradient_shap(model, inputs, base, n_samples=SHAP_SAMPLES, seed=0)

    p = load_scalers()[state]
    contrib = np.concatenate([seq_a.reshape(-1), cov_a.reshape(-1)]) * p.y_std   # kg/ha
    with torch.no_grad():
        pred = float(inverse_transform_target(model(*inputs).numpy(), p)[0])
        typical = float(inverse_transform_target(model(*base).numpy(), p)[0])

    items = [{"feature": f, "contribution": float(c)} for f, c in zip(ALL_FEATURE_NAMES, contrib)]
    items.sort(key=lambda d: -abs(d["contribution"]))
    return {
        "state": state, "arm": arm, "prediction": pred, "typical_year_prediction": typical,
        "explained_gap": float(contrib.sum()), "actual_gap": pred - typical,
        "contributions": items,
        "baseline": f"{state}'s average conditions over its training years (1990–2009)",
        "units": "kg/ha",
    }


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@router.get("/status")
def status() -> dict[str, Any]:
    return readiness()


@router.get("/columns")
def columns() -> dict[str, Any]:
    """Every template column with its unit, so the page can document the format."""
    out = [
        {"name": "state", "required": True, "unit": "", "note": "One of the 19 federated states, spelt as in the dataset"},
        {"name": "district_code", "required": False, "unit": "", "note": "Carried through to the results"},
        {"name": "district_name", "required": False, "unit": "", "note": "Carried through to the results"},
        {"name": "year", "required": False, "unit": "", "note": "Use 2012–2015 for a fair test"},
    ]
    for c in SEQ_NAMES:
        var = c.split("_", 1)[1]
        out.append({"name": c, "required": True, "unit": UNITS[var], "note": "monthly climate"})
    for c in RAW_COV_NAMES:
        out.append({"name": c, "required": True, "unit": UNITS[c], "note": "annual farm figure"})
    out.append({"name": "actual_yield", "required": False, "unit": "kg/ha",
                "note": "Optional. When present, accuracy is reported"})
    return {"columns": out, "derived_by_server": ["irrigation_ratio", "fertiliser_per_ha", "rice_area_share"]}


@router.get("/template")
def template() -> Response:
    header = ",".join(KEY_COLUMNS + SEQ_NAMES + RAW_COV_NAMES + ["actual_yield"]) + "\n"
    return Response(header, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="fedcrop_template.csv"'})


def _parse_arms(models: str) -> list[str]:
    arms = [a.strip() for a in models.split(",") if a.strip()]
    arms = list(dict.fromkeys(arms))                     # de-duplicate, keep order
    bad = [a for a in arms if a not in ARMS]
    if bad or not arms:
        raise HTTPException(status_code=400, detail=f"Unknown model(s) {bad}. Choose from {ARMS}.")
    missing = [a for a in arms if not ckpt_path(a).exists()]
    if missing:
        raise HTTPException(status_code=409, detail=f"No checkpoint for {missing} (seed {SEED}).")
    return arms


class ExplainRequest(BaseModel):
    state: str
    arm: str = "fedper"
    x: list[float]


@router.post("/explain")
def explain(req: ExplainRequest) -> dict[str, Any]:
    require_ready()
    if req.state not in load_scalers():
        raise HTTPException(status_code=400, detail=f"Unknown state '{req.state}'.")
    _parse_arms(req.arm)
    return explain_row(req.state, req.arm, req.x)


# The upload route needs python-multipart. Without it FastAPI refuses to even
# define the route, which would take the whole API down; instead the status
# route reports the problem and everything else keeps working.
try:
    try:
        import python_multipart  # noqa: F401  (package python-multipart, 0.0.13+)
    except ImportError:
        import multipart  # noqa: F401  (older python-multipart releases)
    from fastapi import File, Form, UploadFile

    @router.post("/predict")
    async def predict(file: UploadFile = File(...), models: str = Form("fedper")) -> dict[str, Any]:
        require_ready()
        arms = _parse_arms(models)
        content = await file.read()
        return run_predictions(read_csv(content), arms)

    MULTIPART_OK = True
except ImportError:  # pragma: no cover - environment problem, reported by /status
    MULTIPART_OK = False
    _readiness = readiness

    def readiness() -> dict[str, Any]:  # noqa: F811
        r = _readiness()
        r["problems"].append("python-multipart is not installed: pip install python-multipart")
        r["ready"] = False
        return r
