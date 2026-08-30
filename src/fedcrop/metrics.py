"""Metrics. All reported in kg/ha, never in scaled units."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                y_baseline: np.ndarray) -> float:
    """1 - MSE(model)/MSE(baseline). Positive means the model beat the baseline.

    Report this beside every R2. R2 against the raw mean flatters every model
    in this problem, because most of the variance is between-district level.
    """
    y_true = np.asarray(y_true, dtype=float)
    num = float(((y_true - np.asarray(y_pred, dtype=float)) ** 2).sum())
    den = float(((y_true - np.asarray(y_baseline, dtype=float)) ** 2).sum())
    return 1.0 - num / den if den > 0 else float("nan")


def per_client_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       clients: np.ndarray) -> pd.DataFrame:
    rows = []
    for client in sorted(set(clients)):
        sel = clients == client
        m = regression_metrics(y_true[sel], y_pred[sel])
        m["client"] = client
        rows.append(m)
    return pd.DataFrame(rows).set_index("client").sort_values("rmse")


def worst_client_rmse(per_client: pd.DataFrame) -> float:
    return float(per_client["rmse"].max())


def cross_client_rmse_std(per_client: pd.DataFrame) -> float:
    return float(per_client["rmse"].std())


def summarise(y_true: np.ndarray, y_pred: np.ndarray, clients: np.ndarray,
              y_baseline: np.ndarray | None = None,
              label: str = "") -> dict[str, float]:
    out = regression_metrics(y_true, y_pred)
    pc = per_client_metrics(y_true, y_pred, clients)
    out["worst_client_rmse"] = worst_client_rmse(pc)
    out["cross_client_rmse_std"] = cross_client_rmse_std(pc)
    if y_baseline is not None:
        out["skill_vs_trend"] = skill_score(y_true, y_pred, y_baseline)
    if label:
        out["model"] = label
    return out
