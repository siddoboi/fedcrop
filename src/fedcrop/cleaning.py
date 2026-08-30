"""Cleaning cascade.

Order matters and is fixed. Every step logs rows removed, per state, into an
audit table that belongs in the report appendix: disproportionate loss in
specific states is a result, not a footnote.

Two rules that are easy to get wrong and fail silently:
  - climate is never imputed (it is the variable under study)
  - the outlier threshold is computed on training years only
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import columns as C

log = logging.getLogger(__name__)


def drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage, collinear, high-missing and non-predictor columns."""
    keep = set(C.KEY_COLUMNS) | set(C.BASE_COVARIATES) | {C.TARGET}
    keep |= set(C.sequential_columns())

    drop = [c for c in df.columns if c not in keep]
    log.info("dropping %d columns, keeping %d", len(drop), len(df.columns) - len(drop))
    for name, group in [
        ("leakage", C.LEAKAGE_COLUMNS),
        ("collinear", C.COLLINEAR_COLUMNS),
        ("high-missing", C.HIGH_MISSING_COLUMNS),
    ]:
        hit = [c for c in group if c in drop]
        if hit:
            log.info("  %s: %s", name, hit)
    return df.drop(columns=drop)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios, which are comparable across districts of different size."""
    df = df.copy()
    gca = df["GROSS CROPPED AREA (1000 ha)"].replace(0, np.nan)
    npk = df[[
        "NITROGEN CONSUMPTION (tons)",
        "PHOSPHATE CONSUMPTION (tons)",
        "POTASH CONSUMPTION (tons)",
    ]].sum(axis=1, min_count=3)
    df["irrigation_ratio"] = df["GROSS IRRIGATED AREA (1000 ha)"] / gca
    df["fertiliser_per_ha"] = npk / gca
    df["rice_area_share"] = df["RICE AREA (1000 ha)"] / gca
    return df


def outlier_mask(df: pd.DataFrame, cfg) -> pd.Series:
    """|z| > threshold on the target, with mean and sd from training years only."""
    z_thresh = cfg["cleaning"]["outlier_z"]
    lo, hi = cfg.train_years
    train = df[(df[cfg.year_key] >= lo) & (df[cfg.year_key] <= hi)][cfg.target]
    mu, sd = train.mean(), train.std()
    z = (df[cfg.target] - mu) / sd
    return z.abs() > z_thresh


def apply_cascade(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the cleaning steps in fixed order.

    Returns the cleaned frame and a per-step audit table.
    """
    target = cfg.target
    state = cfg.state_key
    audit_rows: list[dict] = []

    def record(step: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        lost = before.groupby(state).size().sub(
            after.groupby(state).size(), fill_value=0
        )
        audit_rows.append({
            "step": step,
            "rows_before": len(before),
            "rows_after": len(after),
            "removed": len(before) - len(after),
            "worst_state": lost.idxmax() if lost.max() > 0 else "",
            "worst_state_removed": int(lost.max()) if lost.max() > 0 else 0,
        })
        log.info("%-24s %6d -> %6d  (-%d)", step, len(before), len(after),
                 len(before) - len(after))

    work = df
    audit_rows.append({"step": "raw", "rows_before": len(work), "rows_after": len(work),
                       "removed": 0, "worst_state": "", "worst_state_removed": 0})

    before, work = work, work[work[target].notna()]
    record("target not null", before, work)

    if cfg["cleaning"]["drop_zero_yield"]:
        before, work = work, work[work[target] > 0]
        record("yield > 0", before, work)

    before, work = work, work[work["RICE AREA (1000 ha)"] > 0]
    record("rice area > 0", before, work)

    min_area = cfg["cleaning"]["min_rice_area"]
    before, work = work, work[work["RICE AREA (1000 ha)"] >= min_area]
    record(f"rice area >= {min_area}k ha", before, work)

    if cfg["cleaning"]["drop_missing_climate"]:
        seq = C.sequential_columns()
        before, work = work, work.dropna(subset=seq)
        record("climate complete", before, work)

    before, work = work, work[~outlier_mask(work, cfg)]
    record(f"|z| <= {cfg['cleaning']['outlier_z']}", before, work)

    audit = pd.DataFrame(audit_rows)
    return work.reset_index(drop=True), audit


def impute_per_client(df: pd.DataFrame, cfg,
                      cols: list[str] | None = None) -> pd.DataFrame:
    """Fill missing covariates with the client's own training-year median.

    Two leakage guards in one function: no cross-client statistics, and no
    future information. Climate is never passed here.
    """
    cols = cols or C.ANNUAL_COVARIATES
    df = df.copy()
    lo, hi = cfg.train_years
    train_mask = (df[cfg.year_key] >= lo) & (df[cfg.year_key] <= hi)

    for col in cols:
        if col not in df.columns:
            continue
        flag = f"{col}__was_imputed"
        df[flag] = df[col].isna()
        medians = df[train_mask].groupby(cfg.state_key)[col].median()
        fill = df[cfg.state_key].map(medians)
        df[col] = df[col].fillna(fill)
        # Any client with no training observation at all falls back to the
        # global training median. Logged, because it is a cross-client value.
        still = df[col].isna()
        if still.any():
            log.warning("%s: %d rows fell back to global training median",
                        col, int(still.sum()))
            df.loc[still, col] = df.loc[train_mask, col].median()
    return df


def report_missingness(df: pd.DataFrame, cfg) -> pd.DataFrame:
    cols = [c for c in C.BASE_COVARIATES + C.sequential_columns() if c in df.columns]
    return (
        df[cols].isna().mean().mul(100).round(2)
        .rename("missing_pct").to_frame()
        .sort_values("missing_pct", ascending=False)
    )
