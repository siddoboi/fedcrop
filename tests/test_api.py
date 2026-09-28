"""Tests for the API failures that do not crash.

An endpoint that returns 200 with a silently wrong mean, or two endpoints that
disagree about the shape of the same data, will render a plausible dashboard
and mislead everyone reading it. These are the cases worth testing; a 500 would
announce itself.

Two of these tests exist because the bug actually happened:
  - /api/bundle once returned the confound block as a bare list while
    /api/confound returned it wrapped, and the page rendered blank.
  - the climate grid once came out flat (every change 0.0%) because feature
    ablation runs had overwritten the normal checkpoints with models that
    ignored their inputs. Nothing crashed.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import math
import statistics as stats
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is not installed; backend tests skipped"
)
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import ARTIFACTS, LEARNED_ARMS, app  # noqa: E402

RESULTS = ROOT / "artifacts" / "results"

client = TestClient(app)


def raw(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


# ------------------------------------------------------------------ health

def test_health_reports_every_declared_artifact():
    body = client.get("/api/health").json()
    assert set(body["artifacts"]) == set(ARTIFACTS)


def test_health_does_not_claim_ok_while_a_required_artifact_is_missing():
    body = client.get("/api/health").json()
    if body["missing_required"]:
        assert body["status"] == "degraded"
    else:
        assert body["status"] == "ok"


def test_every_gate_g_file_is_required_by_the_api():
    """If Gate G requires a file, the API must not start 'ok' without it."""
    from fedcrop.export.results_json import REQUIRED_EXPORTS
    gate_g = {name.removesuffix(".json") for name in REQUIRED_EXPORTS}
    required = {n for n, req in ARTIFACTS.items() if req}
    assert gate_g <= required, gate_g - required


def test_committed_results_make_the_api_healthy():
    """Gate G passed on the committed results, so the API must report ok."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok", body["missing_required"]


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


def test_the_retired_perturbation_name_is_gone():
    assert "perturbation" not in ARTIFACTS
    assert client.get("/api/results/perturbation").status_code == 404


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
        ("ood", "/api/ood"),
        ("scenarios", "/api/scenarios"),
        ("significance", "/api/significance"),
    ],
)
def test_bundle_agrees_with_the_individual_endpoint(key, route):
    """The confound block once differed in shape between the two. It rendered
    a blank page rather than raising, which is why this is parametrised over
    every block instead of spot-checked."""
    bundle = client.get("/api/bundle").json()
    assert bundle[key] == client.get(route).json(), key


def test_bundle_leaves_out_per_row_predictions():
    assert "predictions" not in client.get("/api/bundle").json()


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
    """Telangana has no training years and cannot fit a scaler."""
    head = client.get("/api/headline").json()
    meta = raw("meta")
    tiny = [c for c, v in meta["clients"].items() if v["rows"] < 50]
    assert head["clients_federated"] == meta["states"] - len(tiny)


# ------------------------------------------------------ drought robustness

def test_ood_reports_every_learned_arm_once():
    arms = [r["arm"] for r in client.get("/api/ood").json()["summary"]]
    assert sorted(arms) == sorted(LEARNED_ARMS)


def test_ood_is_sorted_most_robust_first():
    drops = [r["mean_r2_drop"] for r in client.get("/api/ood").json()["summary"]]
    assert drops == sorted(drops)


def test_ood_confidence_interval_brackets_the_mean():
    body = client.get("/api/ood").json()
    if len(body["seeds"]) < 2:
        pytest.skip("single-seed OOD run has no interval")
    for r in body["summary"]:
        assert r["mean_r2_drop_ci_lo"] <= r["mean_r2_drop"] <= r["mean_r2_drop_ci_hi"], r


def test_ood_per_client_rows_average_over_every_seed():
    body = client.get("/api/ood").json()
    assert body["per_client"], "no per-client degradation rows"
    for r in body["per_client"]:
        assert r["client"] is not None and r["arm"] is not None, r
        assert r["n_seeds"] == len(body["seeds"]), r


# --------------------------------------------------------- climate grid

def test_scenario_grid_is_the_full_three_by_three():
    grid = client.get("/api/scenarios").json()["grid"]
    points = {(g["delta_temp"], g["rain_pct"]) for g in grid}
    assert points == {(t, r) for t in (0.0, 1.0, 2.0) for r in (-20.0, 0.0, 20.0)}


def test_every_state_has_a_response_at_every_grid_point():
    body = client.get("/api/scenarios").json()
    for name in body["clients"]:
        pts = {(r["delta_temp"], r["rain_pct"]) for r in body["responses"]
               if r["client"] == name}
        assert len(pts) == len(body["grid"]), name


def test_the_unperturbed_point_changes_nothing():
    for r in client.get("/api/scenarios").json()["responses"]:
        if r["delta_temp"] == 0 and r["rain_pct"] == 0:
            assert r["mean_pct_change"] == pytest.approx(0.0, abs=1e-9), r


def test_scenario_grid_is_not_flat():
    """Guards the dead-checkpoint failure: a model that ignores its inputs
    produces 0.0% everywhere, and every chart still renders."""
    changes = [abs(r["mean_pct_change"]) for r in client.get("/api/scenarios").json()["responses"]
               if r["delta_temp"] or r["rain_pct"]]
    assert max(changes) > 1.0, "every perturbation moves yield by under 1%"


def test_federation_mean_is_the_mean_over_states():
    body = client.get("/api/scenarios").json()
    for g in body["federation_mean"]:
        vals = [r["mean_pct_change"] for r in body["responses"]
                if r["delta_temp"] == g["delta_temp"] and r["rain_pct"] == g["rain_pct"]]
        assert g["mean_pct_change"] == pytest.approx(stats.mean(vals), abs=1e-9)
        assert g["n_clients"] == len(body["clients"])


def test_scenario_state_filter_and_unknown_state():
    name = client.get("/api/scenarios").json()["clients"][0]
    body = client.get(f"/api/scenarios?client={name}").json()
    assert {r["client"] for r in body["responses"]} == {name}
    assert client.get("/api/scenarios?client=Atlantis").status_code == 404


# ----------------------------------------------------------- predictions

def test_predictions_cover_every_trainable_state_and_arm():
    body = client.get("/api/predictions").json()
    meta = raw("meta")
    trainable = {c for c, v in meta["clients"].items() if v["rows"] >= 50}
    assert set(body["clients"]) == trainable
    assert sorted(body["arms"]) == sorted(LEARNED_ARMS)
    for r in body["rows"]:
        assert r["actual"] is not None
        for arm in body["arms"]:
            assert r[f"pred_{arm}"] is not None, (arm, r)


def test_prediction_metrics_are_recomputed_from_the_rows():
    body = client.get("/api/predictions").json()
    for arm in body["arms"]:
        err = [r[f"pred_{arm}"] - r["actual"] for r in body["rows"]]
        rmse = math.sqrt(sum(e * e for e in err) / len(err))
        assert body["overall"][arm]["rmse"] == pytest.approx(rmse, rel=1e-9), arm
        assert body["overall"][arm]["n"] == len(body["rows"])


def test_prediction_rmse_matches_the_ablation_run_for_that_seed():
    """The dashboard's scatter and the ablation table must describe the same
    model. If these disagree, one of them was built from a stale checkpoint."""
    body = client.get("/api/predictions").json()
    seed = body["seed"]
    runs = {r["arm"]: r for r in raw("ablation")["runs"] if r.get("seed") == seed}
    for arm in body["arms"]:
        if arm in runs:
            assert body["overall"][arm]["rmse"] == pytest.approx(
                runs[arm]["rmse"], rel=5e-3), arm


def test_prediction_state_filter_and_unknown_state():
    name = client.get("/api/predictions").json()["clients"][0]
    body = client.get(f"/api/predictions?client={name}").json()
    assert {r["client"] for r in body["rows"]} == {name}
    assert len(body["per_client"]) == 1
    assert client.get("/api/predictions?client=Atlantis").status_code == 404


# ------------------------------------------------------ separation

def test_root_is_a_json_route_index_not_a_page():
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "/api/health" in body["routes"]


def test_no_html_is_served():
    assert client.get("/results").status_code == 404
    assert client.get("/static/index.html").status_code == 404


def test_cross_origin_get_is_allowed():
    res = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert res.headers.get("access-control-allow-origin") in {"*", "http://localhost:5173"}


# ------------------------------------------------------- per-state view

def test_state_view_returns_profile_drivers_and_baseline():
    meta = raw("meta")
    name = max(meta["clients"], key=lambda c: meta["clients"][c]["rows"])
    body = client.get(f"/api/state/{name}").json()
    assert body["state"] == name
    assert body["profile"]["rows"] == meta["clients"][name]["rows"]
    assert body["trend_baseline"]["client"] == name
    assert body["attribution_available"] is True
    assert body["climate_available"] is True
    assert 1 <= body["profile"]["yield_rank"] <= body["profile"]["n_states"]


def test_state_driver_shares_are_fractions_of_that_state_total():
    """A share above 1 would mean the page is dividing by the wrong total."""
    meta = raw("meta")
    name = max(meta["clients"], key=lambda c: meta["clients"][c]["rows"])
    for d in client.get(f"/api/state/{name}").json()["drivers"]:
        assert 0.0 <= d["share"] <= 1.0, d


def test_state_drivers_are_ranked():
    meta = raw("meta")
    name = max(meta["clients"], key=lambda c: meta["clients"][c]["rows"])
    vals = [d["attribution"] for d in client.get(f"/api/state/{name}").json()["drivers"]]
    assert vals == sorted(vals, reverse=True)


def test_state_without_a_model_says_so_rather_than_inventing_results():
    """Telangana has no client model. The page must not fall back to national
    attributions or another state's climate response and present them as local."""
    meta = raw("meta")
    tiny = [c for c, v in meta["clients"].items() if v["rows"] < 50]
    if not tiny:
        pytest.skip("every client is trainable")
    body = client.get(f"/api/state/{tiny[0]}").json()
    assert body["attribution_available"] is False
    assert body["drivers"] == []
    assert body["driver_source"] is None
    assert body["climate_available"] is False
    assert body["climate_response"] == []


def test_unknown_state_is_404():
    assert client.get("/api/state/Atlantis").status_code == 404