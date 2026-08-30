"""Tests for the failures that do not crash.

A global scaler or a leaked test year produces a good-looking number that is
wrong. These four groups of tests catch exactly those cases.

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import columns as C, scaling, splits
from fedcrop.config import load_config
from fedcrop.features import FeatureBundle
from fedcrop.splits import ClientData, Splits


# ---------------------------------------------------------------- columns

def test_windspeed_uses_abbreviations():
    assert C.build_column_name("windspeed", "SEPTEMBER") == "SEPT WINDSPEED (Meter per second)"
    assert C.build_column_name("windspeed", "AUGUST") == "AUG WINDSPEED (Meter per second)"
    assert C.build_column_name("windspeed", "JANUARY") == "JAN WINDSPEED (Meter per second)"


def test_other_variables_use_full_month_names():
    assert C.build_column_name("max_temp", "SEPTEMBER").startswith("SEPTEMBER ")
    assert C.build_column_name("precipitation", "AUGUST").startswith("AUGUST ")


def test_precipitation_typo_is_preserved():
    assert "PERCIPITATION" in C.build_column_name("precipitation", "JUNE")


def test_sequential_columns_count_and_order():
    cols = C.sequential_columns()
    assert len(cols) == 60
    assert len(set(cols)) == 60
    assert cols[0] == C.build_column_name("max_temp", "JANUARY")


def test_feature_names_align_with_flattened_length():
    names = C.feature_names()
    assert len(names) == 60 + len(C.ANNUAL_COVARIATES) == 69
    assert names[:5] == ["jan_max_temp", "jan_min_temp", "jan_precipitation",
                         "jan_evapotranspiration", "jan_windspeed"]


def test_excluded_columns_are_not_model_inputs():
    inputs = set(C.sequential_columns()) | set(C.ANNUAL_COVARIATES)
    for bad in C.LEAKAGE_COLUMNS + C.COLLINEAR_COLUMNS + C.HIGH_MISSING_COLUMNS:
        assert bad not in inputs


def test_validate_columns_reports_all_misses():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError) as exc:
        C.validate_columns(df, ["a", "b", "c"])
    assert "b" in str(exc.value) and "c" in str(exc.value)


# ----------------------------------------------------------------- splits

def _toy_bundle(cfg, years=range(1990, 2016), districts=("d1", "d2"),
                states=("s1", "s2")):
    rows = [(d, s, y) for d, s in zip(districts, states) for y in years]
    keys = pd.DataFrame(rows, columns=[cfg.district_key, cfg.state_key, cfg.year_key])
    n = len(keys)
    rng = np.random.default_rng(0)
    return FeatureBundle(
        x_seq=rng.normal(size=(n, 12, 5)).astype(np.float32),
        x_cov=rng.normal(size=(n, 9)).astype(np.float32),
        y=rng.normal(2000, 400, size=n).astype(np.float32),
        keys=keys, feature_names=C.feature_names(),
        months=C.MONTHS_FULL, seq_vars=C.SEQUENTIAL_VARS,
        covariates=C.ANNUAL_COVARIATES,
    )


def test_temporal_split_is_ordered_and_disjoint():
    cfg = load_config()
    b = _toy_bundle(cfg)
    sp = splits.temporal_split(b, cfg)
    assert not (sp.train & sp.val).any()
    assert not (sp.val & sp.test).any()
    assert not (sp.train & sp.test).any()
    years = b.keys[cfg.year_key].to_numpy()
    assert years[sp.train].max() < years[sp.val].min() < years[sp.test].min()


def test_leak_assertion_fires_on_overlapping_split():
    cfg = load_config()
    b = _toy_bundle(cfg)
    years = b.keys[cfg.year_key].to_numpy()
    bad = Splits(train=years <= 2013, val=(years == 2010), test=years >= 2012)
    with pytest.raises(AssertionError):
        splits.assert_no_temporal_leak(b, bad, cfg)


def test_partition_keeps_clients_separate():
    cfg = load_config()
    b = _toy_bundle(cfg)
    clients = splits.partition_by_client(b, cfg)
    assert set(clients) == {"s1", "s2"}
    for name, c in clients.items():
        assert (c.keys[cfg.state_key] == name).all()
    assert sum(len(c) for c in clients.values()) == len(b)


# ---------------------------------------------------------------- scaling

def _toy_clients(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(["s1", "s2"]):
        n = 40
        out[name] = ClientData(
            name=name,
            x_seq=rng.normal(10 * (i + 1), 2, size=(n, 12, 5)).astype(np.float32),
            x_cov=rng.normal(5 * (i + 1), 1, size=(n, 9)).astype(np.float32),
            y=rng.normal(1000 * (i + 1), 200, size=n).astype(np.float32),
            keys=pd.DataFrame({"Dist Code": range(n)}),
        )
    return out


def test_each_client_gets_its_own_scaler():
    clients = _toy_clients()
    params = scaling.fit_all_client_scalers(clients)
    assert params["s1"].y_mean != params["s2"].y_mean
    scaling.assert_no_global_scaler(params, clients)


def test_global_scaler_is_detected():
    clients = _toy_clients()
    params = scaling.fit_all_client_scalers(clients)
    shared = params["s1"]
    faked = {"s1": shared, "s2": scaling.ScalerParams(
        client="s2", seq_mean=shared.seq_mean, seq_std=shared.seq_std,
        cov_mean=shared.cov_mean, cov_std=shared.cov_std,
        y_mean=shared.y_mean, y_std=shared.y_std,
        n_train_rows=len(clients["s2"]))}
    with pytest.raises(AssertionError):
        scaling.assert_no_global_scaler(faked, clients)


def test_scaling_round_trips():
    clients = _toy_clients()
    p = scaling.fit_client_scaler(clients["s1"])
    scaled = scaling.apply_scaler(clients["s1"], p)
    restored = scaling.inverse_transform_target(scaled.y, p)
    np.testing.assert_allclose(restored, clients["s1"].y, rtol=1e-4)


def test_scaler_fitted_on_train_only_does_not_see_test():
    """Applying a train-fitted scaler to shifted test data must not recentre it."""
    clients = _toy_clients()
    train = clients["s1"]
    p = scaling.fit_client_scaler(train)
    test = ClientData("s1", train.x_seq + 50, train.x_cov, train.y + 500, train.keys)
    scaled = scaling.apply_scaler(test, p)
    assert abs(float(scaled.y.mean())) > 0.5, (
        "test data was recentred to zero, so the scaler saw the test split"
    )


# ------------------------------------------------------- model and federation

def test_model_shapes_and_fedper_split():
    import torch
    from fedcrop.model import YieldGRU, count_parameters
    m = YieldGRU(n_seq_vars=5, n_covariates=9, hidden=64)
    out = m(torch.randn(7, 12, 5), torch.randn(7, 9))
    assert out.shape == (7,)
    base, head = m.base_parameter_keys(), m.head_parameter_keys()
    assert set(base) | set(head) == set(m.state_dict())
    assert not (set(base) & set(head))
    assert count_parameters(m) == 16033


def test_aggregate_is_sample_weighted():
    import torch
    from fedcrop.fl.aggregate import aggregate
    a = {"w": torch.tensor([0.0])}
    b = {"w": torch.tensor([10.0])}
    merged = aggregate([a, b], [90.0, 10.0])
    assert abs(float(merged["w"]) - 1.0) < 1e-6


def test_fedper_leaves_heads_untouched():
    import torch
    from fedcrop.fl.aggregate import aggregate
    from fedcrop.fl.param_utils import set_parameters
    from fedcrop.model import YieldGRU
    m = YieldGRU()
    base = m.base_parameter_keys()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    other = YieldGRU()
    merged = aggregate([m.state_dict(), other.state_dict()], [1.0, 1.0], keys=base)
    set_parameters(m, merged, keys=base)
    after = m.state_dict()
    for k in m.head_parameter_keys():
        assert torch.equal(before[k], after[k]), f"head key {k} was modified"
    assert not torch.equal(before[base[0]], after[base[0]])


def test_proximal_term_changes_the_loss():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from fedcrop.model import YieldGRU
    from fedcrop.train import train_epoch
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(16, 12, 5), torch.randn(16, 9), torch.randn(16))
    loader = DataLoader(ds, batch_size=8)
    losses = []
    for mu in (0.0, 10.0):
        torch.manual_seed(0)
        m = YieldGRU()
        anchor = {k: torch.zeros_like(v) for k, v in m.named_parameters()}
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses.append(train_epoch(m, loader, opt, global_params=anchor, mu=mu))
    assert losses[1] > losses[0], "FedProx proximal term had no effect on the loss"
