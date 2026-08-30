"""Explanation fidelity.

An attribution ranking is only meaningful if removing the features it ranks
highly degrades the model more than removing random ones. Without this check,
a SHAP bar chart is decoration: the method will always return a confident
ranking, and neither the number nor the plot warns you when it describes
nothing durable.

Write the interpretation before running it, so a null result reads as a
prediction confirmed rather than an excuse constructed afterwards.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from ..splits import ClientData

log = logging.getLogger(__name__)


def _predict(model, seq: np.ndarray, cov: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(seq).float(),
                     torch.from_numpy(cov).float()).numpy()


def _mask_features(seq: np.ndarray, cov: np.ndarray, names: list[str],
                   targets: list[str], baseline_seq: np.ndarray,
                   baseline_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace the named features with their baseline value."""
    n_seq = seq.shape[1] * seq.shape[2]
    seq_flat = seq.reshape(len(seq), -1).copy()
    cov_out = cov.copy()
    index = {name: i for i, name in enumerate(names)}
    for t in targets:
        i = index[t]
        if i < n_seq:
            seq_flat[:, i] = baseline_seq.reshape(-1)[i]
        else:
            cov_out[:, i - n_seq] = baseline_cov.reshape(-1)[i - n_seq]
    return seq_flat.reshape(seq.shape), cov_out


def deletion_curve(model, data: ClientData, ranking: pd.Series,
                   feature_names: list[str], k_max: int = 15) -> np.ndarray:
    """RMSE change as the top-k features are replaced by their baseline."""
    base_seq = data.x_seq.mean(axis=0, keepdims=True)
    base_cov = data.x_cov.mean(axis=0, keepdims=True)
    reference = _predict(model, data.x_seq, data.x_cov)
    order = ranking.sort_values(ascending=False).index.tolist()

    out = [0.0]
    for k in range(1, min(k_max, len(order)) + 1):
        seq, cov = _mask_features(data.x_seq, data.x_cov, feature_names,
                                  order[:k], base_seq, base_cov)
        pred = _predict(model, seq, cov)
        out.append(float(np.sqrt(((pred - reference) ** 2).mean())))
    return np.asarray(out)


def random_deletion_curve(model, data: ClientData, feature_names: list[str],
                          k_max: int = 15, n_trials: int = 20,
                          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_trials):
        shuffled = pd.Series(rng.permutation(len(feature_names)),
                             index=feature_names)
        curves.append(deletion_curve(model, data, shuffled, feature_names, k_max))
    return np.mean(curves, axis=0)


def aopc(curve: np.ndarray) -> float:
    """Area over the perturbation curve. Higher means more informative."""
    return float(np.mean(curve[1:])) if len(curve) > 1 else 0.0


def fidelity_report(model, data: ClientData, ranking: pd.Series,
                    feature_names: list[str], k_max: int = 15,
                    n_trials: int = 20, seed: int = 0) -> dict:
    top = deletion_curve(model, data, ranking, feature_names, k_max)
    rnd = random_deletion_curve(model, data, feature_names, k_max, n_trials, seed)
    a_top, a_rnd = aopc(top), aopc(rnd)
    ratio = a_top / a_rnd if a_rnd > 0 else float("nan")

    if not np.isfinite(ratio):
        verdict = "INCONCLUSIVE"
    elif ratio > 1.2:
        verdict = "PASS"
    elif ratio < 0.9:
        verdict = "FAIL (top-k less informative than random)"
    else:
        verdict = "WEAK (top-k indistinguishable from random)"

    return {"client": data.name, "aopc_topk": a_top, "aopc_random": a_rnd,
            "ratio": ratio, "verdict": verdict,
            "curve_topk": top.tolist(), "curve_random": rnd.tolist(),
            "k_max": k_max, "n_rows": len(data)}
