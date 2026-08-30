"""Per-client standardisation.

A global scaler leaks cross-client statistics, and it bakes in the
between-district level differences that cause the pooled sign inversion.
Fitting and applying are separate functions so fitting on test data is
structurally impossible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .splits import ClientData

log = logging.getLogger(__name__)

EPS = 1e-8


@dataclass
class ScalerParams:
    client: str
    seq_mean: list
    seq_std: list
    cov_mean: list
    cov_std: list
    y_mean: float
    y_std: float
    n_train_rows: int


def fit_client_scaler(train: ClientData) -> ScalerParams:
    """Fit on this client's training rows only. Never call on test data."""
    if len(train) == 0:
        raise ValueError(f"client {train.name} has no training rows")
    return ScalerParams(
        client=train.name,
        seq_mean=train.x_seq.mean(axis=(0, 1)).tolist(),
        seq_std=(train.x_seq.std(axis=(0, 1)) + EPS).tolist(),
        cov_mean=train.x_cov.mean(axis=0).tolist(),
        cov_std=(train.x_cov.std(axis=0) + EPS).tolist(),
        y_mean=float(train.y.mean()),
        y_std=float(train.y.std() + EPS),
        n_train_rows=len(train),
    )


def apply_scaler(data: ClientData, p: ScalerParams,
                 scale_target: bool = True) -> ClientData:
    seq = (data.x_seq - np.asarray(p.seq_mean)) / np.asarray(p.seq_std)
    cov = (data.x_cov - np.asarray(p.cov_mean)) / np.asarray(p.cov_std)
    y = (data.y - p.y_mean) / p.y_std if scale_target else data.y
    return ClientData(data.name, seq.astype(np.float32), cov.astype(np.float32),
                      y.astype(np.float32), data.keys)


def inverse_transform_target(y_scaled: np.ndarray, p: ScalerParams) -> np.ndarray:
    """Back to kg/ha. RMSE must be reported in real units."""
    return y_scaled * p.y_std + p.y_mean


def fit_all_client_scalers(train_clients: dict[str, ClientData],
                           out_dir: Path | None = None) -> dict[str, ScalerParams]:
    params = {name: fit_client_scaler(c) for name, c in train_clients.items()}
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "scalers.json", "w") as fh:
            json.dump({k: asdict(v) for k, v in params.items()}, fh, indent=2)
        log.info("wrote %d client scalers to %s", len(params), out_dir)
    return params


def assert_no_global_scaler(params: dict[str, ScalerParams],
                            train_clients: dict[str, ClientData]) -> None:
    """Fail if any scaler was fitted across more than one client.

    Cheap insurance against the most likely silent bug in the project: a
    global scaler does not crash, it just quietly invalidates every number.
    """
    for name, p in params.items():
        if p.client != name:
            raise AssertionError(f"scaler {name} carries client label {p.client}")
        if p.n_train_rows != len(train_clients[name]):
            raise AssertionError(
                f"scaler for {name} was fitted on {p.n_train_rows} rows but the "
                f"client has {len(train_clients[name])} training rows"
            )
    means = [p.y_mean for p in params.values()]
    if len(set(round(m, 6) for m in means)) == 1 and len(means) > 1:
        raise AssertionError(
            "every client has an identical target mean, which means a global "
            "scaler was fitted and broadcast"
        )
