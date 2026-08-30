"""Aggregating explanations rather than data.

A global federated explanation is a sample-weighted mean of per-client
attributions. No raw data crosses a client boundary at any point, which is
the mechanism the methodology deck names as the contribution.

Normalisation matters more than it looks: attribution magnitudes inherit the
scale of each client's target, so without it a high-yield state dominates the
aggregate for reasons of units rather than importance.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .attribution import ClientAttribution

log = logging.getLogger(__name__)


def normalise(series: pd.Series) -> pd.Series:
    total = float(np.abs(series).sum())
    return series / total if total > 0 else series


def aggregate_attributions(attrs: dict[str, ClientAttribution],
                           weights: dict[str, float] | None = None,
                           normalise_first: bool = True) -> pd.Series:
    if not attrs:
        raise ValueError("no client attributions to aggregate")
    weights = weights or {n: float(a.n_samples) for n, a in attrs.items()}
    total = float(sum(weights[n] for n in attrs))

    acc = None
    for name, attr in attrs.items():
        s = attr.mean_abs()
        if normalise_first:
            s = normalise(s)
        contribution = s * (weights[name] / total)
        acc = contribution if acc is None else acc.add(contribution, fill_value=0.0)
    return acc.sort_values(ascending=False)


def per_client_matrix(attrs: dict[str, ClientAttribution],
                      normalise_first: bool = True) -> pd.DataFrame:
    """Clients as rows, features as columns."""
    rows = {}
    for name, attr in attrs.items():
        s = attr.mean_abs()
        rows[name] = normalise(s) if normalise_first else s
    return pd.DataFrame(rows).T


def heterogeneity(attrs: dict[str, ClientAttribution]) -> pd.DataFrame:
    """Per-feature spread of attribution across clients.

    Nearly free to compute, and it makes the heterogeneity claim something to
    point at rather than assert: it shows which climate responses differ by
    state, not merely that they do.
    """
    mat = per_client_matrix(attrs)
    out = pd.DataFrame({
        "mean": mat.mean(axis=0),
        "std": mat.std(axis=0),
        "min": mat.min(axis=0),
        "max": mat.max(axis=0),
        "argmax_client": mat.idxmax(axis=0),
    })
    out["cv"] = out["std"] / out["mean"].replace(0, np.nan)
    return out.sort_values("std", ascending=False)


def month_group_share(series: pd.Series) -> dict[str, float]:
    """Share of total attribution falling in each month group.

    Used by the agronomic check: monsoon months should dominate, and Oct-Dec
    dominance would mean the between-district confound has reappeared inside
    the model.
    """
    groups = {
        "pre_monsoon": ["jan", "feb", "mar", "apr", "may"],
        "monsoon": ["jun", "jul", "aug", "sep"],
        "post_monsoon": ["oct", "nov", "dec"],
    }
    total = float(np.abs(series).sum())
    out = {}
    for label, months in groups.items():
        mask = series.index.str.split("_").str[0].isin(months)
        out[label] = float(np.abs(series[mask]).sum() / total) if total else 0.0
    climate_months = sum(len(v) for v in groups.values())
    is_climate = series.index.str.split("_").str[0].isin(
        [m for v in groups.values() for m in v])
    out["covariates"] = float(np.abs(series[~is_climate]).sum() / total) if total else 0.0
    out["_n_climate_months"] = climate_months
    return out
