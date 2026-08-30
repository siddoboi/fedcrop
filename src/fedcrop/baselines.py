"""Baselines.

The district trend is the number every model in this project is measured
against. Reporting R2 without it beside it overstates every result by
roughly the between-district variance, which dominates this problem.

Fallback chain for districts the trend cannot be fitted on:
  enough training years -> per-district linear trend
  too few training years -> district training mean
  absent from training   -> state training mean
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import FeatureBundle
from .metrics import regression_metrics, per_client_metrics
from .splits import Splits

log = logging.getLogger(__name__)


def _frame(bundle: FeatureBundle, cfg) -> pd.DataFrame:
    df = bundle.keys.copy()
    df["y"] = bundle.y
    return df


def global_mean(bundle: FeatureBundle, splits: Splits, cfg) -> np.ndarray:
    df = _frame(bundle, cfg)
    value = df.loc[splits.train, "y"].mean()
    return np.full(len(df), value, dtype=float)


def district_mean(bundle: FeatureBundle, splits: Splits, cfg) -> np.ndarray:
    df = _frame(bundle, cfg)
    train = df[splits.train]
    by_district = train.groupby(cfg.district_key)["y"].mean()
    by_state = train.groupby(cfg.state_key)["y"].mean()
    pred = df[cfg.district_key].map(by_district)
    pred = pred.fillna(df[cfg.state_key].map(by_state))
    return pred.fillna(train["y"].mean()).to_numpy(dtype=float)


def district_trend(bundle: FeatureBundle, splits: Splits, cfg) -> np.ndarray:
    df = _frame(bundle, cfg)
    train = df[splits.train]
    min_years = cfg["baselines"]["min_train_years_for_trend"]

    fits: dict = {}
    for district, grp in train.groupby(cfg.district_key):
        if grp[cfg.year_key].nunique() >= min_years:
            fits[district] = np.polyfit(grp[cfg.year_key], grp["y"], 1)

    by_district_mean = train.groupby(cfg.district_key)["y"].mean()
    by_state_mean = train.groupby(cfg.state_key)["y"].mean()
    global_train_mean = train["y"].mean()

    pred = np.empty(len(df), dtype=float)
    n_trend = n_dmean = n_smean = n_global = 0
    for i, (district, state, year) in enumerate(
        zip(df[cfg.district_key], df[cfg.state_key], df[cfg.year_key])
    ):
        if district in fits:
            pred[i] = np.polyval(fits[district], year)
            n_trend += 1
        elif district in by_district_mean.index:
            pred[i] = by_district_mean[district]
            n_dmean += 1
        elif state in by_state_mean.index:
            pred[i] = by_state_mean[state]
            n_smean += 1
        else:
            pred[i] = global_train_mean
            n_global += 1

    log.info("trend baseline coverage: %d trend, %d district mean, "
             "%d state mean, %d global mean", n_trend, n_dmean, n_smean, n_global)
    return pred


BASELINES = {
    "global_mean": global_mean,
    "district_mean": district_mean,
    "district_trend": district_trend,
}


def baseline_table(bundle: FeatureBundle, splits: Splits, cfg) -> pd.DataFrame:
    """All three baselines evaluated on the test split."""
    y_true = bundle.y[splits.test]
    clients = bundle.keys.loc[splits.test, cfg.state_key].to_numpy()
    rows = []
    for name, fn in BASELINES.items():
        pred = fn(bundle, splits, cfg)[splits.test]
        m = regression_metrics(y_true, pred)
        pc = per_client_metrics(y_true, pred, clients)
        m["model"] = name
        m["worst_client_rmse"] = float(pc["rmse"].max())
        m["cross_client_rmse_std"] = float(pc["rmse"].std())
        rows.append(m)
    cols = ["model", "rmse", "mae", "r2", "worst_client_rmse",
            "cross_client_rmse_std", "n"]
    return pd.DataFrame(rows)[cols]


def reference_predictions(bundle: FeatureBundle, splits: Splits, cfg) -> np.ndarray:
    """The trend baseline, for use as the denominator in skill_score."""
    return district_trend(bundle, splits, cfg)
