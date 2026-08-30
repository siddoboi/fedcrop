"""Feature assembly. Everything downstream takes a FeatureBundle, not a frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import columns as C

log = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    x_seq: np.ndarray        # (N, T, V)
    x_cov: np.ndarray        # (N, C)
    y: np.ndarray            # (N,)
    keys: pd.DataFrame       # Dist Code, State Name, Year, aligned by row
    feature_names: list[str]
    months: list[str]
    seq_vars: list[str]
    covariates: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_seq_features(self) -> int:
        return self.x_seq.shape[1] * self.x_seq.shape[2]

    @property
    def n_features(self) -> int:
        return self.n_seq_features + self.x_cov.shape[1]


def resolve_months(window: str) -> list[str]:
    if window not in C.WINDOWS:
        raise ValueError(
            f"window {window!r} not implemented here; available: {list(C.WINDOWS)}"
        )
    return C.WINDOWS[window]


def build_sequence_tensor(df: pd.DataFrame, months: list[str],
                          seq_vars: list[str]) -> np.ndarray:
    """(N, T, V). Month-major ordering, matching columns.sequential_columns."""
    cols = C.sequential_columns(seq_vars, months)
    C.validate_columns(df, cols)
    arr = df[cols].to_numpy(dtype=np.float32)
    n = len(df)
    tensor = arr.reshape(n, len(months), len(seq_vars))
    if np.isnan(tensor).any():
        raise ValueError(
            f"{int(np.isnan(tensor).sum())} NaN values in the sequence tensor. "
            "Climate must be dropped, never imputed - check the cleaning cascade."
        )
    return tensor


def build_covariate_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    C.validate_columns(df, covariates)
    arr = df[covariates].to_numpy(dtype=np.float32)
    if np.isnan(arr).any():
        raise ValueError(
            f"{int(np.isnan(arr).sum())} NaN values in covariates. "
            "Run cleaning.impute_per_client before assembling."
        )
    return arr


def assemble(df: pd.DataFrame, cfg) -> FeatureBundle:
    months = resolve_months(cfg["features"]["window"])
    seq_vars = cfg["features"]["sequential_vars"]
    covariates = C.ANNUAL_COVARIATES

    bundle = FeatureBundle(
        x_seq=build_sequence_tensor(df, months, seq_vars),
        x_cov=build_covariate_matrix(df, covariates),
        y=df[cfg.target].to_numpy(dtype=np.float32),
        keys=df[[cfg.district_key, cfg.state_key, cfg.year_key]].reset_index(drop=True),
        feature_names=C.feature_names(seq_vars, months, covariates),
        months=months,
        seq_vars=seq_vars,
        covariates=covariates,
    )
    log.info("assembled bundle: x_seq %s, x_cov %s, %d named features",
             bundle.x_seq.shape, bundle.x_cov.shape, len(bundle.feature_names))
    assert len(bundle.feature_names) == bundle.n_features, (
        "feature_names length must equal flattened feature count"
    )
    return bundle
