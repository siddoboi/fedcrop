"""Statistical significance.

Paired tests, paired by client, because that is the correct pairing: two
algorithms' RMSE on the same client are not independent samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_across_clients(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Paired signed-rank test on per-client RMSE, paired by client index."""
    common = a.index.intersection(b.index)
    if len(common) < 6:
        return (float("nan"), float("nan"))
    stat, p = stats.wilcoxon(a[common].to_numpy(), b[common].to_numpy())
    return float(stat), float(p)


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def ablation_table(df: pd.DataFrame, metric_cols: list[str] | None = None
                   ) -> pd.DataFrame:
    """Mean, sd and 95% bootstrap CI per arm, across seeds.

    df is the long-format runs table from ablation.json: one row per
    (arm, seed) with the metric columns.
    """
    metric_cols = metric_cols or ["rmse", "mae", "r2", "skill_vs_trend",
                                  "worst_client_rmse", "cross_client_rmse_std"]
    rows = []
    for arm, grp in df.groupby("arm"):
        row = {"arm": arm, "n_seeds": len(grp)}
        for col in metric_cols:
            if col not in grp:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sd"] = float(vals.std()) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (np.nan, np.nan)
            row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_significance(per_client: dict[str, pd.Series],
                          reference: str) -> pd.DataFrame:
    """Wilcoxon signed-rank of each arm's per-client RMSE against a reference."""
    rows = []
    if reference not in per_client:
        raise KeyError(f"reference arm {reference!r} not in per_client results")
    for arm, series in per_client.items():
        if arm == reference:
            continue
        stat, p = wilcoxon_across_clients(per_client[reference], series)
        rows.append({"arm": arm, "reference": reference,
                    "wilcoxon_stat": stat, "p_value": p,
                    "significant_at_0.05": bool(p < 0.05) if np.isfinite(p) else None})
    return pd.DataFrame(rows)
