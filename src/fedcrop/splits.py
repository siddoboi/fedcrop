"""Splitting and client partitioning.

Two protocols, never mixed in one run:
  1. temporal_split   - train 1990-2009, val 2010-2011, test 2012-2015
  2. climate_stress   - each client's own driest years withheld

Nothing downstream of partition_by_client may see more than one client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import columns as C
from .features import FeatureBundle

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class ClientData:
    name: str
    x_seq: np.ndarray
    x_cov: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)


def _year_mask(years: pd.Series, lo: int, hi: int) -> np.ndarray:
    return ((years >= lo) & (years <= hi)).to_numpy()


def temporal_split(bundle: FeatureBundle, cfg) -> Splits:
    years = bundle.keys[cfg.year_key]
    splits = Splits(
        train=_year_mask(years, *cfg.train_years),
        val=_year_mask(years, *cfg.val_years),
        test=_year_mask(years, *cfg.test_years),
    )
    assert_no_temporal_leak(bundle, splits, cfg)
    log.info("split sizes: train %d, val %d, test %d",
             splits.train.sum(), splits.val.sum(), splits.test.sum())
    return splits


def assert_no_temporal_leak(bundle: FeatureBundle, splits: Splits, cfg) -> None:
    years = bundle.keys[cfg.year_key].to_numpy()
    tr, va, te = years[splits.train], years[splits.val], years[splits.test]
    if len(tr) and len(va) and tr.max() >= va.min():
        raise AssertionError(f"train max {tr.max()} >= val min {va.min()}")
    if len(va) and len(te) and va.max() >= te.min():
        raise AssertionError(f"val max {va.max()} >= test min {te.min()}")
    if len(tr) and len(te) and tr.max() >= te.min():
        raise AssertionError(f"train max {tr.max()} >= test min {te.min()}")


def jjas_by_state_year(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Mean JJAS precipitation per state per year."""
    cols = [C.build_column_name("precipitation", m) for m in C.JJAS]
    C.validate_columns(df, cols)
    work = df[[cfg.state_key, cfg.year_key]].copy()
    work["jjas"] = df[cols].sum(axis=1)
    return work.groupby([cfg.state_key, cfg.year_key])["jjas"].mean().unstack()


def per_client_deficit_years(df: pd.DataFrame, cfg, k: int = 3,
                             mode: str = "rank",
                             threshold_pct: float = -15.0) -> dict[str, list[int]]:
    """Each client's own deficit years.

    A national list is the wrong construct: 2015 is -2.7% for India as a whole
    and -27.6% for Karnataka, because opposite signs cancel in the mean.

    mode="rank"      the k driest years per state
    mode="threshold" every year below threshold_pct departure from that
                     state's own mean. Agronomically more defensible, and it
                     does not force every client to have exactly k events.

    Note on k: at k=3 Karnataka's 2015 (-27.6%) is excluded because three
    other years were drier. Rank cutoffs are arbitrary; report sensitivity.
    """
    table = jjas_by_state_year(df, cfg)
    out: dict[str, list[int]] = {}
    for state, row in table.iterrows():
        if mode == "rank":
            years = row.nsmallest(k).index
        elif mode == "threshold":
            dep = (row - row.mean()) / row.mean() * 100.0
            years = dep[dep < threshold_pct].index
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out[state] = sorted(int(y) for y in years)
    return out


def climate_stress_split(bundle: FeatureBundle, cfg,
                         deficit: dict[str, list[int]]) -> Splits:
    """Withhold each client's own deficit years from that client's training."""
    states = bundle.keys[cfg.state_key].to_numpy()
    years = bundle.keys[cfg.year_key].to_numpy()
    is_deficit = np.array([y in deficit.get(s, []) for s, y in zip(states, years)])
    return Splits(train=~is_deficit, val=np.zeros(len(years), bool), test=is_deficit)


def partition_by_client(bundle: FeatureBundle, cfg,
                        mask: np.ndarray | None = None) -> dict[str, ClientData]:
    """Split a bundle into per-state clients. The federation boundary."""
    mask = np.ones(len(bundle), bool) if mask is None else mask
    states = bundle.keys[cfg.state_key].to_numpy()
    out: dict[str, ClientData] = {}
    for state in sorted(set(states[mask])):
        sel = mask & (states == state)
        out[state] = ClientData(
            name=state,
            x_seq=bundle.x_seq[sel],
            x_cov=bundle.x_cov[sel],
            y=bundle.y[sel],
            keys=bundle.keys[sel].reset_index(drop=True),
        )
    log.info("partitioned into %d clients, %d to %d rows",
             len(out), min(len(c) for c in out.values()),
             max(len(c) for c in out.values()))
    return out
