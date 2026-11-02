"""Tests for the CSV upload routes.

The dangerous failure here does not crash: an upload scaled with the wrong
statistics, or fed features in the wrong order, returns plausible yields that
are simply wrong. So the central tests compare the upload path against the
pipeline's own prediction path, row for row.

Most tests build small synthetic artifacts (random models, plausible scalers)
in a temporary folder, so they run anywhere. The last test checks the real
checkpoints against artifacts/results/predictions.json and skips when the
gitignored checkpoints or the demo files are absent.

    python -m pytest tests/test_predict.py -q
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch", reason="PyTorch is needed for prediction tests")
pytest.importorskip("captum", reason="Captum is needed for explanation tests")
try:
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart", reason="python-multipart is needed for upload tests")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import predict as P  # noqa: E402

REAL_PATHS = dict(P.PATHS)        # captured before any fixture swaps in synthetic artifacts
from fedcrop import columns as C  # noqa: E402
from fedcrop.cleaning import add_derived  # noqa: E402
from fedcrop.config import load_config  # noqa: E402
from fedcrop.model import build_model  # noqa: E402
from fedcrop.scaling import ScalerParams  # noqa: E402

STATES = ["Alpha", "Beta"]
NO_MODEL_STATE = "Gamma"          # plays Telangana: in the panel, no training years
SEQ_MEAN = [30.0, 20.0, 100.0, 50.0, 2.0]
SEQ_STD = [3.0, 3.0, 80.0, 40.0, 0.5]
RAW_MEAN = {"rice_area": 50.0, "nitrogen_consumption": 5000.0, "phosphate_consumption": 2000.0,
            "potash_consumption": 1000.0, "gross_cropped_area": 200.0, "gross_irrigated_area": 100.0}


# --------------------------------------------------------------- fixtures

def _model(seed: int):
    torch.manual_seed(seed)
    return build_model(cfg=load_config(), n_seq_vars=5, n_covariates=9)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic artifacts with the real file layouts, and a client bound to them."""
    d = tmp_path_factory.mktemp("artifacts")
    (d / "models").mkdir()

    scalers = {}
    for i, s in enumerate(STATES):
        scalers[s] = ScalerParams(
            client=s, seq_mean=SEQ_MEAN, seq_std=SEQ_STD,
            cov_mean=[50, 5000, 2000, 1000, 200, 100, 0.5, 40, 0.25],
            cov_std=[25, 2500, 1000, 500, 100, 50, 0.2, 20, 0.1],
            y_mean=2500.0 + 300 * i, y_std=600.0 + 50 * i, n_train_rows=400 + i)
    (d / "scalers.json").write_text(json.dumps({k: asdict(v) for k, v in scalers.items()}))
    meta = {"split_sizes": {"train": sum(p.n_train_rows for p in scalers.values())},
            "clients": {**{s: {"rows": 500} for s in STATES}, NO_MODEL_STATE: {"rows": 18}}}
    (d / "meta.json").write_text(json.dumps(meta))

    a, b, g = _model(1), _model(2), _model(3)
    heads = {"Alpha": {k: v for k, v in _model(4).state_dict().items() if k.startswith("head.")},
             "Beta": {k: v for k, v in _model(5).state_dict().items() if k.startswith("head.")}}
    torch.save(a.state_dict(), d / "models" / "centralised_seed42.pt")
    torch.save({"Alpha": b.state_dict()}, d / "models" / "local_seed42.pt")      # no Beta: tests the gap
    torch.save({"global": g.state_dict(), "heads": heads}, d / "models" / "fedper_seed42.pt")
    torch.save({"global": a.state_dict(), "heads": {}}, d / "models" / "fedavg_seed42.pt")
    torch.save({"global": b.state_dict(), "heads": {}}, d / "models" / "fedprox_seed42.pt")

    saved = dict(P.PATHS)
    P.PATHS.update(models=d / "models", scalers=d / "scalers.json", meta=d / "meta.json")
    P.reset_caches()
    app = FastAPI()
    app.include_router(P.router)
    yield SimpleNamespace(dir=d, client=TestClient(app), scalers=scalers)
    P.PATHS.update(saved)
    P.reset_caches()


def rows(n: int = 6, state: str = "Alpha", seed: int = 0, actual: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"state": [state] * n, "district_code": [str(100 + i) for i in range(n)],
                        "district_name": [f"D{i}" for i in range(n)], "year": [2012 + i % 4 for i in range(n)]})
    for name in P.SEQ_NAMES:
        v = C.SEQUENTIAL_VARS.index(name.split("_", 1)[1])
        out[name] = np.round(SEQ_MEAN[v] + SEQ_STD[v] * rng.normal(size=n) * 0.8, 3).clip(min=0)
    for name, m in RAW_MEAN.items():
        out[name] = np.round(m * rng.uniform(0.6, 1.4, size=n), 3)
    out["gross_irrigated_area"] = np.minimum(out["gross_irrigated_area"], out["gross_cropped_area"])
    if actual:
        out["actual_yield"] = np.round(2500 + 500 * rng.normal(size=n), 1)
    return out


def upload(client, frame: pd.DataFrame, models: str = "fedper"):
    buf = frame.to_csv(index=False).encode()
    return client.post("/api/upload/predict", files={"file": ("x.csv", buf, "text/csv")},
                       data={"models": models})


# ------------------------------------------------------------- readiness

def test_status_is_ready_with_complete_artifacts(env):
    s = env.client.get("/api/upload/status").json()
    assert s["ready"], s["problems"]
    assert set(s["arms_available"]) == set(P.ARMS)
    assert s["states"] == STATES


def test_status_refuses_scalers_from_a_different_split(env):
    """03_train_all.py rewrites scalers.json on every run, including climate-
    stress runs that fit on a different split. That must stop predictions."""
    meta = json.loads((env.dir / "meta.json").read_text())
    meta["split_sizes"]["train"] += 1
    bad = env.dir / "meta_bad.json"
    bad.write_text(json.dumps(meta))
    P.PATHS["meta"] = bad; P.reset_caches()
    try:
        s = env.client.get("/api/upload/status").json()
        assert not s["ready"]
        assert any("climate-stress" in p for p in s["problems"])
        assert upload(env.client, rows()).status_code == 503
    finally:
        P.PATHS["meta"] = env.dir / "meta.json"; P.reset_caches()


# ------------------------------------------------------------- the format

def test_template_header_is_the_model_feature_order(env):
    header = env.client.get("/api/upload/template").text.strip().split(",")
    assert header[:4] == P.KEY_COLUMNS
    assert header[4:64] == C.feature_names(C.SEQUENTIAL_VARS, C.MONTHS_FULL, [])
    assert header[64:70] == ["rice_area", "nitrogen_consumption", "phosphate_consumption",
                             "potash_consumption", "gross_cropped_area", "gross_irrigated_area"]
    assert header[-1] == "actual_yield"


def test_canonical_names_map_to_the_right_source_columns():
    """If this mapping were off by one, every prediction would be silently wrong."""
    assert P.SEQ_NAMES[0] == "jan_max_temp" and P.SEQ_SOURCE[0] == "JANUARY MAXIMUM TEMPERATURE (Centigrate)"
    assert P.SEQ_NAMES[2] == "jan_precipitation" and P.SEQ_SOURCE[2] == "JANUARY PERCIPITATION (Millimeters)"
    assert P.SEQ_NAMES[59] == "dec_windspeed" and P.SEQ_SOURCE[59] == "DEC WINDSPEED (Meter per second)"
    assert P.SEQ_NAMES[44] == "sep_windspeed" and P.SEQ_SOURCE[44] == "SEPT WINDSPEED (Meter per second)"


def test_missing_columns_are_listed_before_any_row_is_read(env):
    res = upload(env.client, rows().drop(columns=["jul_precipitation", "rice_area"]))
    assert res.status_code == 400
    assert "jul_precipitation" in res.json()["detail"] and "rice_area" in res.json()["detail"]


def test_row_limit(env):
    big = pd.concat([rows(1)] * (P.MAX_ROWS + 1), ignore_index=True)
    assert upload(env.client, big).status_code == 413


def test_unknown_model_is_rejected(env):
    assert upload(env.client, rows(), models="fedper,xgboost").status_code == 400


# ------------------------------------------------------------- validation

def test_every_bad_row_is_rejected_with_its_reason(env):
    df = rows(9).astype(object)
    df.loc[1, "state"] = NO_MODEL_STATE
    df.loc[2, "state"] = "Alfa"
    df.loc[3, "jul_precipitation"] = ""
    df.loc[4, "nitrogen_consumption"] = "n/a"
    df.loc[5, "rice_area"] = 1.5
    df.loc[6, "gross_cropped_area"] = 0
    df.loc[7, "actual_yield"] = -5
    body = upload(env.client, df).json()
    reasons = {r["row"]: " ".join(r["reasons"]) for r in body["rejected"]}
    assert body["summary"]["accepted"] == 2                  # rows 0 and 8
    assert "no training years" in reasons[3]
    assert "Unknown state 'Alfa'" in reasons[4]
    assert "Missing climate value in jul_precipitation" in reasons[5]
    assert "Not a number in nitrogen_consumption" in reasons[6]
    assert "below the 2" in reasons[7]
    assert "Gross cropped area must be above 0" in reasons[8]
    assert "actual_yield must be above 0" in reasons[9]


def test_rejected_rows_report_spreadsheet_line_numbers(env):
    df = rows(3).astype(object); df.loc[0, "state"] = "Nowhere"
    assert upload(env.client, df).json()["rejected"][0]["row"] == 2    # header is line 1


def test_actual_yield_is_optional(env):
    body = upload(env.client, rows(actual=False)).json()
    assert body["summary"]["accepted"] == 6
    assert body["summary"]["has_actual"] is False
    assert body["metrics"] is None
    assert all(r["actual"] is None for r in body["rows"])


def test_extreme_value_is_predicted_but_warned(env):
    df = rows(2); df.loc[0, "jul_precipitation"] = 100 + 80 * 9
    body = upload(env.client, df).json()
    assert body["summary"]["accepted"] == 2
    w = body["rows"][0]["warnings"]
    assert w and "jul_precipitation" in w[0] and "extrapolation" in w[0]
    assert body["rows"][1]["warnings"] == []


def test_training_year_rows_are_flagged(env):
    df = rows(3); df.loc[0, "year"] = 2005
    notices = " ".join(upload(env.client, df).json()["notices"])
    assert "1 row(s) fall in 1990–2009" in notices


def test_missing_local_model_is_a_warning_not_a_failure(env):
    body = upload(env.client, rows(2, state="Beta"), models="fedper,local").json()
    assert all(r["pred"]["local"] is None for r in body["rows"])
    assert all(r["pred"]["fedper"] is not None for r in body["rows"])
    assert "No local model exists for Beta" in body["rows"][0]["warnings"][0]


# ------------------------------------------- equivalence with the pipeline

def test_ratios_are_computed_from_the_raw_figures(env):
    df = rows(4)
    _, cov = P.to_tensors(df)
    gca = df["gross_cropped_area"].to_numpy()
    npk = df[["nitrogen_consumption", "phosphate_consumption", "potash_consumption"]].sum(axis=1).to_numpy()
    np.testing.assert_allclose(cov[:, 6], df["gross_irrigated_area"] / gca, rtol=1e-5)
    np.testing.assert_allclose(cov[:, 7], npk / gca, rtol=1e-5)
    np.testing.assert_allclose(cov[:, 8], df["rice_area"] / gca, rtol=1e-5)


@pytest.mark.parametrize("arm", ["fedper", "centralised", "local", "fedavg", "fedprox"])
def test_upload_matches_the_pipeline_prediction_path(env, arm):
    """Predict the same rows through evaluation/perturbation.py's loader and
    train.predict, which are the functions that produced the frozen results."""
    from fedcrop.datasets import make_loader
    from fedcrop.evaluation.perturbation import _load_model_for
    from fedcrop.scaling import apply_scaler, inverse_transform_target
    from fedcrop.splits import ClientData
    from fedcrop.train import predict

    df = rows(8, state="Alpha", seed=3)
    body = upload(env.client, df, models=arm).json()
    got = np.array([r["pred"][arm] for r in body["rows"]])

    src = pd.DataFrame({s: df[c] for s, c in zip(P.SEQ_SOURCE, P.SEQ_NAMES)})
    for s, c in zip(C.BASE_COVARIATES, P.RAW_COV_NAMES):
        src[s] = df[c]
    src = add_derived(src)
    seq = src[C.sequential_columns()].to_numpy(dtype=np.float32).reshape(len(df), 12, 5)
    cov = src[C.ANNUAL_COVARIATES].to_numpy(dtype=np.float32)
    p = env.scalers["Alpha"]
    scaled = apply_scaler(ClientData("Alpha", seq, cov, np.zeros(len(df), np.float32), df), p, scale_target=False)
    shape = SimpleNamespace(x_seq=np.zeros((1, 12, 5)), x_cov=np.zeros((1, 9)))
    model = _load_model_for(arm, P.ckpt_path(arm), shape, load_config(), "Alpha")
    pred, _ = predict(model, make_loader(scaled, shuffle=False))
    np.testing.assert_allclose(got, inverse_transform_target(pred, p), atol=1e-2)


def test_each_state_uses_its_own_scaler_and_head(env):
    """The same numbers uploaded as Alpha and as Beta must give different yields."""
    a = upload(env.client, rows(3, state="Alpha", seed=9)).json()
    b = upload(env.client, rows(3, state="Beta", seed=9)).json()
    assert not np.allclose([r["pred"]["fedper"] for r in a["rows"]],
                           [r["pred"]["fedper"] for r in b["rows"]])


def test_metrics_are_recomputed_from_the_rows(env):
    body = upload(env.client, rows(10), models="fedper,centralised").json()
    for arm in ("fedper", "centralised"):
        err = np.array([r["pred"][arm] - r["actual"] for r in body["rows"]])
        assert body["metrics"][arm]["rmse"] == pytest.approx(float(np.sqrt((err ** 2).mean())), rel=1e-9)


# ------------------------------------------------------------ explanation

def test_explanation_covers_every_input_in_kg_per_ha(env):
    body = upload(env.client, rows(1)).json()
    r = body["rows"][0]
    ex = env.client.post("/api/upload/explain", json={"state": r["state"], "arm": "fedper", "x": r["x"]}).json()
    assert len(ex["contributions"]) == 69
    assert {c["feature"] for c in ex["contributions"]} == set(C.feature_names())
    mags = [abs(c["contribution"]) for c in ex["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert ex["units"] == "kg/ha"
    assert ex["prediction"] == pytest.approx(r["pred"]["fedper"], abs=1e-2)


def test_explanation_is_not_collapsed_to_zero(env):
    """The pipeline's client_mean baseline on a single row is the row itself,
    which zeroes every attribution. The upload explainer must not do that."""
    r = upload(env.client, rows(1, seed=5)).json()["rows"][0]
    ex = env.client.post("/api/upload/explain", json={"state": "Alpha", "x": r["x"]}).json()
    assert max(abs(c["contribution"]) for c in ex["contributions"]) > 1.0


def test_explanation_rejects_a_wrong_length_input(env):
    res = env.client.post("/api/upload/explain", json={"state": "Alpha", "x": [1.0, 2.0]})
    assert res.status_code == 400


# ---------------------------------------------------- the real artifacts

REAL_DEMO = ROOT / "demo_data" / "demo_jharkhand.csv"
REAL_PRED = ROOT / "artifacts" / "results" / "predictions.json"
REAL_CKPT = ROOT / "artifacts" / "models" / "fedper_seed42.pt"


@pytest.mark.skipif(not (REAL_DEMO.exists() and REAL_PRED.exists() and REAL_CKPT.exists()),
                    reason="needs local checkpoints and scripts/10_make_demo_data.py output")
def test_uploading_the_jharkhand_demo_reproduces_the_frozen_predictions():
    """The upload path and the pipeline must give the same yield for the same
    district-year, to within float rounding."""
    saved = dict(P.PATHS)
    P.PATHS.update(REAL_PATHS)    # the module-scoped synthetic fixture may still be active
    P.reset_caches()
    try:
        app = FastAPI(); app.include_router(P.router)
        client = TestClient(app)
        status = client.get("/api/upload/status").json()
        assert status["ready"], status["problems"]
        frame = pd.read_csv(REAL_DEMO, dtype=str, keep_default_na=False)
        body = upload(client, frame, models="fedper,centralised").json()
    finally:
        P.PATHS.update(saved)
        P.reset_caches()
    assert body["summary"]["rejected"] == 0, body["rejected"][:3]

    frozen = json.loads(REAL_PRED.read_text())
    first = frozen["rows"][0]
    dist_key = next((k for k in ("Dist Code", "district_code", "dist_code", "District Code") if k in first), None)
    year_key = next((k for k in ("Year", "year") if k in first), None)
    assert dist_key and year_key, f"predictions.json rows have no district/year key; keys are {sorted(first)}"
    ref = {(str(r[dist_key]), int(r[year_key])): r for r in frozen["rows"] if r.get("client") == "Jharkhand"}
    matched = 0
    for r in body["rows"]:
        f = ref.get((str(r["district_code"]), r["year"]))
        if f is None:
            continue
        matched += 1
        assert r["pred"]["fedper"] == pytest.approx(f["pred_fedper"], abs=0.5)
        assert r["pred"]["centralised"] == pytest.approx(f["pred_centralised"], abs=0.5)
    assert matched == len(body["rows"]), "every demo row should exist in the frozen test predictions"


def test_backend_app_serves_the_upload_routes():
    """Checked through the OpenAPI schema and a live request: recent FastAPI
    versions keep included routers lazily, so app.routes may not list them."""
    from backend.app import app
    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/api/upload/predict", "/api/upload/explain", "/api/upload/status"} <= paths
    assert client.get("/api/upload/status").status_code == 200
