"""Counterfactual climate perturbations.

+1C, +2C temperature and -20% rainfall, applied in RAW units before scaling
with the client's own training scaler. Applying a perturbation to already-
scaled data would mean the delta is in standard-deviation units, not degrees
or percent, and every client's perturbation would silently mean something
different.

This is a sensitivity analysis over the observed range, not a climate
projection: the panel is 1990-2015 with no emission pathway. Framed as
testing robustness to climate analogues, using historical extremes as a
proxy for conditions projected to become more frequent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from ..datasets import make_loader
from ..fl.param_utils import set_parameters
from ..model import build_model
from ..scaling import ScalerParams, apply_scaler, inverse_transform_target
from ..splits import ClientData
from ..train import predict

log = logging.getLogger(__name__)

# Index of each variable within the 5-variable sequence, matching
# columns.SEQUENTIAL_VARS order: max_temp, min_temp, precipitation,
# evapotranspiration, windspeed.
TEMP_IDX = 0
PRECIP_IDX = 2


def perturb(data: ClientData, delta_temp: float = 0.0,
           rain_pct: float = 0.0) -> ClientData:
    """Apply a perturbation in raw units. data.x_seq must be UNSCALED."""
    seq = data.x_seq.copy()
    if delta_temp != 0.0:
        seq[:, :, TEMP_IDX] += delta_temp
    if rain_pct != 0.0:
        seq[:, :, PRECIP_IDX] *= (1.0 + rain_pct / 100.0)
        seq[:, :, PRECIP_IDX] = np.maximum(seq[:, :, PRECIP_IDX], 0.0)
    return ClientData(data.name, seq, data.x_cov.copy(), data.y, data.keys)


def _load_model_for(arm: str, ckpt_path, bundle, cfg, client_name: str):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(bundle, cfg)
    if arm in ("centralised",):
        model.load_state_dict(blob)
    elif arm == "local":
        if client_name not in blob:
            return None
        model.load_state_dict(blob[client_name])
    else:
        model.load_state_dict(blob["global"])
        if blob.get("heads", {}).get(client_name):
            set_parameters(model, blob["heads"][client_name])
    model.eval()
    return model


def counterfactual_response(arm: str, ckpt_path, bundle, cfg,
                            raw_clients: dict[str, ClientData],
                            scalers: dict[str, ScalerParams],
                            grid: list[tuple[float, float]]
                            ) -> pd.DataFrame:
    """Predicted mean yield change per client across a perturbation grid.

    raw_clients must hold UNSCALED x_seq (apply the scaler after perturbing,
    never before), and grid is a list of (delta_temp, rain_pct) pairs.
    """
    rows = []
    for name, raw in raw_clients.items():
        if name not in scalers:
            continue
        model = _load_model_for(arm, ckpt_path, bundle, cfg, name)
        if model is None:
            continue

        base_scaled = apply_scaler(raw, scalers[name])
        base_p, _ = predict(model, make_loader(base_scaled, shuffle=False))
        base_yield = inverse_transform_target(base_p, scalers[name])

        for dt, rp in grid:
            pert = perturb(raw, delta_temp=dt, rain_pct=rp)
            pert_scaled = apply_scaler(pert, scalers[name], scale_target=False)
            p, _ = predict(model, make_loader(
                ClientData(name, pert_scaled.x_seq, pert_scaled.x_cov,
                          np.zeros(len(pert)), pert.keys), shuffle=False))
            pert_yield = inverse_transform_target(p, scalers[name])
            rows.append({
                "arm": arm, "client": name, "delta_temp": dt, "rain_pct": rp,
                "mean_base_yield": float(base_yield.mean()),
                "mean_pert_yield": float(pert_yield.mean()),
                "mean_pct_change": float(
                    (pert_yield.mean() - base_yield.mean()) / base_yield.mean() * 100),
            })
    return pd.DataFrame(rows)


def attribution_shift(base_attr: pd.Series, pert_attr: pd.Series,
                      perturbed_features: list[str]) -> dict:
    """Does SHAP move toward the perturbed variable when it is perturbed?

    A model whose predictions move under a perturbation but whose
    attributions do not shift toward that variable has learned a proxy
    correlated with it, not a mechanism responding to it.
    """
    base_rank = base_attr.rank(ascending=False)
    pert_rank = pert_attr.rank(ascending=False)
    moved = {f: {"rank_before": float(base_rank.get(f, np.nan)),
                 "rank_after": float(pert_rank.get(f, np.nan))}
             for f in perturbed_features if f in base_attr.index}
    improved = sum(1 for f in moved
                   if moved[f]["rank_after"] < moved[f]["rank_before"])
    return {"per_feature": moved, "n_features": len(moved),
            "n_moved_up": improved,
            "verdict": "shifted toward perturbed feature" if improved > len(moved) / 2
                       else "did not consistently shift"}
