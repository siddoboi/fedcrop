"""Attribution agreement. This is the contribution.

Kendall's tau and top-k Jaccard between federated and centralised feature
rankings. The number nobody in the reviewed corpus reports.

The seed-stability control is what makes it interpretable. "Federated and
centralised agree at tau = 0.6" means nothing on its own, because two seeds of
the same centralised model might also agree at 0.6. Without the control, any
observed disagreement could be attributable to optimisation noise rather than
to federation, and the headline claim is unsupported.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


def rank_features(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False)


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        log.warning("ranking lengths differ (%d vs %d); comparing %d shared "
                    "features", len(a), len(b), len(common))
    return a[common], b[common]


def kendall_tau(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a, b = _align(a, b)
    tau, p = stats.kendalltau(a.to_numpy(), b.to_numpy())
    return float(tau), float(p)


def spearman_rho(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a, b = _align(a, b)
    rho, p = stats.spearmanr(a.to_numpy(), b.to_numpy())
    return float(rho), float(p)


def top_k_jaccard(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    top_a = set(a.sort_values(ascending=False).head(k).index)
    top_b = set(b.sort_values(ascending=False).head(k).index)
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else float("nan")


def compare(a: pd.Series, b: pd.Series, k: int = 5) -> dict[str, float]:
    tau, tau_p = kendall_tau(a, b)
    rho, rho_p = spearman_rho(a, b)
    return {"kendall_tau": tau, "kendall_p": tau_p,
            "spearman_rho": rho, "spearman_p": rho_p,
            f"top{k}_jaccard": top_k_jaccard(a, b, k)}


def seed_stability(attrs_by_seed: dict[int, pd.Series], k: int = 5) -> pd.DataFrame:
    """Agreement between seeds of the SAME model. The control condition.

    Every pairwise comparison among seeds. If this sits at the same level as
    the federated-versus-centralised comparison, then federation is not what
    is costing you attribution fidelity; optimisation noise is.
    """
    seeds = sorted(attrs_by_seed)
    rows = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            m = compare(attrs_by_seed[s1], attrs_by_seed[s2], k)
            m.update({"seed_a": s1, "seed_b": s2})
            rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def agreement_report(central: dict[int, pd.Series],
                     federated: dict[str, dict[int, pd.Series]],
                     k: int = 5) -> pd.DataFrame:
    """The headline table.

    One row per federated algorithm plus a control row for centralised
    seed-to-seed agreement. Each metric is the mean across seeds with a
    bootstrap 95% interval.
    """
    rows = []

    ctrl = seed_stability(central, k)
    if len(ctrl):
        lo, hi = bootstrap_ci(ctrl["kendall_tau"])
        rows.append({
            "comparison": "centralised vs centralised (seed control)",
            "kendall_tau": ctrl["kendall_tau"].mean(),
            "kendall_tau_sd": ctrl["kendall_tau"].std(),
            "kendall_tau_lo": lo, "kendall_tau_hi": hi,
            f"top{k}_jaccard": ctrl[f"top{k}_jaccard"].mean(),
            "spearman_rho": ctrl["spearman_rho"].mean(),
            "n_pairs": len(ctrl),
        })

    for algo, by_seed in federated.items():
        seeds = sorted(set(by_seed) & set(central))
        if not seeds:
            continue
        per_seed = [compare(central[s], by_seed[s], k) for s in seeds]
        taus = [m["kendall_tau"] for m in per_seed]
        lo, hi = bootstrap_ci(taus)
        rows.append({
            "comparison": f"centralised vs {algo}",
            "kendall_tau": float(np.mean(taus)),
            "kendall_tau_sd": float(np.std(taus)),
            "kendall_tau_lo": lo, "kendall_tau_hi": hi,
            f"top{k}_jaccard": float(np.mean([m[f"top{k}_jaccard"] for m in per_seed])),
            "spearman_rho": float(np.mean([m["spearman_rho"] for m in per_seed])),
            "n_pairs": len(seeds),
        })

    return pd.DataFrame(rows)


def agronomic_check(series: pd.Series, k: int = 5) -> dict:
    """Mechanise the plausibility test rather than leave it to judgement.

    Monsoon months (Jun-Sep) driving the ranking is agronomically correct for
    rice. Oct-Dec on top would mean the between-district confound documented
    in the data has reappeared inside the model.
    """
    from .aggregate import month_group_share
    shares = month_group_share(series)
    top = series.sort_values(ascending=False).head(k)
    prefixes = top.index.str.split("_").str[0]
    n_monsoon = int(prefixes.isin(["jun", "jul", "aug", "sep"]).sum())
    n_post = int(prefixes.isin(["oct", "nov", "dec"]).sum())

    if n_post > n_monsoon:
        verdict = "FAIL"
        reason = ("post-monsoon months outrank monsoon months in the top "
                  f"{k}; this is the between-district confound reappearing")
    elif n_monsoon > 0:
        verdict = "PASS"
        reason = f"{n_monsoon} of the top {k} features are monsoon months"
    else:
        verdict = "INCONCLUSIVE"
        reason = (f"no climate months in the top {k}; the ranking is driven by "
                  "annual covariates, so the agronomic test does not apply")

    return {"verdict": verdict, "reason": reason,
            "top_features": top.index.tolist(),
            "n_monsoon_in_top": n_monsoon, "n_post_monsoon_in_top": n_post,
            "share_monsoon": shares["monsoon"],
            "share_post_monsoon": shares["post_monsoon"],
            "share_covariates": shares["covariates"]}
