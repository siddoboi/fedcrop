"""Tests for the API failures that do not crash.

An endpoint that returns 200 with a silently wrong mean, or two endpoints that
disagree about the shape of the same data, will render a plausible dashboard
and mislead everyone reading it. These are the cases worth testing; a 500 would
announce itself.

One of these tests exists because the bug actually happened: /api/bundle
returned the confound block as a bare list while /api/confound returned it
wrapped, and the page rendered blank with a console error rather than failing
loudly.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is not installed; backend tests skipped"
)
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402

RESULTS = ROOT / "artifacts" / "results"

client = TestClient(app)


def raw(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


# ------------------------------------------------------------------ health

def test_health_reports_every_declared_artifact():
    body = client.get("/api/health").json()
    from backend.app import ARTIFACTS

    assert set(body["artifacts"]) == set(ARTIFACTS)


def test_health_does_not_claim_ok_while_a_required_artifact_is_missing():
    body = client.get("/api/health").json()
    if body["missing_required"]:
        assert body["status"] == "degraded"
    else:
        assert body["status"] == "ok"


def test_unbuilt_stages_are_reported_rather_than_silently_empty():
    """Stage F/G outputs are absent. The API must say so, not serve nothing."""
    body = client.get("/api/health").json()
    for name in ("ood", "perturbation", "significance"):
        present = body["artifacts"][name]["present"]
        if not present:
            assert name in body["missing_optional"]


def test_requesting_an_unbuilt_artifact_returns_409_not_500():
    body = client.get("/api/health").json()
    missing = [n for n, s in body["artifacts"].items() if not s["present"]]
    if not missing:
        pytest.skip("every artifact is present")
    res = client.get(f"/api/results/{missing[0]}")
    assert res.status_code == 409
    assert "pipeline stage" in res.json()["detail"]


def test_unknown_artifact_name_is_404():
    assert client.get("/api/results/not_a_real_artifact").status_code == 404


# ------------------------------------------------- aggregation correctness

def test_ablation_means_match_the_underlying_runs():
    """A wrong mean is the definition of a failure that does not crash."""
    api = {r["arm"]: r for r in client.get("/api/ablation").json()["rows"]}
    grouped: dict[str, list] = {}
    for run in raw("ablation")["runs"]:
        grouped.setdefault(run["arm"], []).append(run)

    for arm, runs in grouped.items():
        expected = stats.mean(r["r2"] for r in runs)
        assert api[arm]["r2"] == pytest.approx(expected, abs=1e-9), arm
        assert api[arm]["n_seeds"] == len(runs), arm


def test_learned_arms_aggregate_over_every_seed():
    ab = client.get("/api/ablation").json()
    n_seeds = len(ab["seeds"])
    for row in ab["rows"]:
        if not row["is_baseline"]:
            assert row["n_seeds"] == n_seeds, row["arm"]


def test_baselines_are_single_runs_and_carry_no_skill():
    for row in client.get("/api/ablation").json()["rows"]:
        if row["is_baseline"]:
            assert row["n_seeds"] == 1, row["arm"]
            assert row["skill_vs_trend"] is None, row["arm"]


def test_the_reference_arm_is_the_one_skill_is_measured_against():
    ab = client.get("/api/ablation").json()
    refs = [r for r in ab["rows"] if r["is_reference"]]
    assert len(refs) == 1
    assert refs[0]["arm"] == raw("ablation")["reference_model"]


def test_standard_deviation_is_zero_only_for_single_run_arms():
    for row in client.get("/api/ablation").json()["rows"]:
        if row["n_seeds"] == 1:
            assert row["r2_sd"] == 0.0, row["arm"]
        else:
            assert row["r2_sd"] > 0.0, row["arm"]


# --------------------------------------------------- endpoint consistency

@pytest.mark.parametrize(
    "key, route",
    [
        ("ablation", "/api/ablation"),
        ("agreement", "/api/agreement"),
        ("complexity", "/api/complexity"),
        ("mu_sweep", "/api/mu-sweep"),
        ("confound", "/api/confound"),
        ("headline", "/api/headline"),
    ],
)
def test_bundle_agrees_with_the_individual_endpoint(key, route):
    """The confound block once differed in shape between the two. It rendered
    a blank page rather than raising, which is why this is parametrised over
    every block instead of spot-checked."""
    bundle = client.get("/api/bundle").json()
    assert bundle[key] == client.get(route).json(), key


def test_headline_communication_saving_matches_the_complexity_artifact():
    head = client.get("/api/headline").json()
    comms = {c["algorithm"]: c for c in raw("complexity")["communication"]}
    expected = (
        (comms["fedavg"]["kb_per_round"] - comms["fedper"]["kb_per_round"])
        / comms["fedavg"]["kb_per_round"]
        * 100
    )
    assert head["comm_saving_pct"] == pytest.approx(expected, abs=1e-9)


def test_headline_best_r2_is_the_centralised_arm():
    head = client.get("/api/headline").json()
    rows = {r["arm"]: r for r in client.get("/api/ablation").json()["rows"]}
    assert head["best_r2"] == pytest.approx(rows["centralised"]["r2"], abs=1e-9)


# ------------------------------------------------------------ derived data

def test_confound_is_ranked_by_distance_between_pooled_and_within():
    rows = client.get("/api/confound?limit=60").json()["rows"]
    gaps = [r["gap"] for r in rows]
    assert gaps == sorted(gaps, reverse=True)


def test_confound_sign_flip_flag_matches_the_correlations():
    for r in client.get("/api/confound?limit=60").json()["rows"]:
        expected = (r["pooled_r"] * r["within_district_r"]) < 0
        assert r["sign_flip"] is expected, r


def test_confound_limit_is_respected():
    assert len(client.get("/api/confound?limit=3").json()["rows"]) == 3


def test_fidelity_curves_are_stripped_from_the_response():
    """The AOPC curves are large and unplotted; shipping them bloats every load."""
    for row in client.get("/api/fidelity").json():
        assert not any(k.startswith("curve_") for k in row)
        assert "ratio" in row and "verdict" in row


def test_clients_are_ordered_by_size_and_carry_deficit_years():
    rows = client.get("/api/clients").json()["rows"]
    assert [r["rows"] for r in rows] == sorted((r["rows"] for r in rows), reverse=True)
    assert all(len(r["deficit_years"]) == 3 for r in rows)


def test_federated_client_count_excludes_the_untrainable_state():
    """Telangana holds 18 rows after cleaning and cannot fit a scaler."""
    head = client.get("/api/headline").json()
    meta = raw("meta")
    tiny = [c for c, v in meta["clients"].items() if v["rows"] < 50]
    assert head["clients_federated"] == meta["states"] - len(tiny)


# ------------------------------------------------------------------ pages

def test_dashboard_is_served_at_the_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Evaluation Results" in res.text or "<title>" in res.text


def test_api_routes_win_over_the_static_mount():
    assert client.get("/api/health").json()["status"] in {"ok", "degraded"}
