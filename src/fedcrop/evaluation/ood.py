"""Out-of-distribution evaluation on each client's own deficit years.

A national deficit-year list is the wrong construct: 2015 was -2.7% for
India as a whole and -27.6% for Karnataka in the same year, because opposite
signs cancel in the national mean. Every OOD test in this module withholds
each client's OWN driest years, never a shared national list.

This directly tests the claim that clients drift at different times and
rates, which a national list cannot expose by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from ..datasets import ClientSplits, make_loader
from ..fl.param_utils import set_parameters
from ..model import build_model
from ..scaling import ScalerParams, apply_scaler
from ..splits import ClientData
from ..train import predict

log = logging.getLogger(__name__)


def build_ood_clients(bundle, cfg, clean_df: pd.DataFrame,
                      deficit: dict[str, list[int]],
                      scalers: dict[str, ScalerParams]) -> dict[str, ClientData]:
    """Each client's rows restricted to its OWN deficit years, scaled with
    that client's already-fitted training scaler.

    Reusing the training scaler, not refitting on deficit years, is
    deliberate: it tests how the model trained on normal years performs when
    conditions shift, which is the actual robustness question. Refitting
    would silently answer a different, easier question.
    """
    from ..features import FeatureBundle
    states = bundle.keys[cfg.state_key].to_numpy()
    years = bundle.keys[cfg.year_key].to_numpy()

    out: dict[str, ClientData] = {}
    for name, years_list in deficit.items():
        if name not in scalers or not years_list:
            continue
        mask = (states == name) & np.isin(years, years_list)
        if not mask.any():
            continue
        raw = ClientData(name, bundle.x_seq[mask], bundle.x_cov[mask],
                         bundle.y[mask], bundle.keys[mask].reset_index(drop=True))
        out[name] = apply_scaler(raw, scalers[name])
    log.info("built OOD sets for %d clients", len(out))
    return out


def _predict_arm(arm: str, ckpt_path, bundle, cfg,
                 ood_clients: dict[str, ClientData]) -> dict[str, np.ndarray]:
    """Predict with whichever checkpoint layout the arm uses.

    The arm name may carry a protocol suffix such as "_stress", so the layout
    is decided by inspecting the loaded blob rather than by string matching.
    A federated checkpoint stores {"global", "heads"}; local stores a dict of
    client states; centralised stores a bare state dict.
    """
    preds = {}
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_federated = isinstance(blob, dict) and "global" in blob
    is_local = arm.startswith("local")

    if is_federated:
        global_state, heads = blob["global"], blob.get("heads", {})
        for name, data in ood_clients.items():
            model = build_model(bundle, cfg)
            model.load_state_dict(global_state)
            if heads.get(name):
                set_parameters(model, heads[name])
            model.eval()
            p, _ = predict(model, make_loader(data, shuffle=False))
            preds[name] = p
    else:
        for name, data in ood_clients.items():
            if is_local:
                if name not in blob:
                    continue
                state = blob[name]
            else:
                state = blob
            model = build_model(bundle, cfg)
            model.load_state_dict(state)
            model.eval()
            p, _ = predict(model, make_loader(data, shuffle=False))
            preds[name] = p
    return preds


def run_ood(bundle, cfg, ood_clients: dict[str, ClientData],
           models_dir, arms: list[str], seed: int) -> pd.DataFrame:
    """RMSE and R2 per client per arm, on deficit years only, in real units."""
    rows = []
    for arm in arms:
        ckpt = models_dir / f"{arm}_seed{seed}.pt"
        if not ckpt.exists():
            log.warning("missing %s, skipping", ckpt.name)
            continue
        preds = _predict_arm(arm, ckpt, bundle, cfg, ood_clients)
        for name, p in preds.items():
            data = ood_clients[name]
            s = data.keys  # scaled ClientData keeps keys untouched
            y_true = data.y  # still scaled; unscale via the stored mean/std
            rows.append({"arm": arm, "client": name, "n": len(p),
                        "pred_scaled": p, "true_scaled": y_true})
    return pd.DataFrame(rows)


def ood_metrics(bundle, cfg, ood_clients: dict[str, ClientData],
                scalers: dict, models_dir, arms: list[str],
                seed: int) -> pd.DataFrame:
    """Per-client, per-arm RMSE/R2 on deficit years, in kg/ha."""
    from ..metrics import regression_metrics
    from ..scaling import inverse_transform_target

    rows = []
    for arm in arms:
        ckpt = models_dir / f"{arm}_seed{seed}.pt"
        if not ckpt.exists():
            continue
        preds = _predict_arm(arm, ckpt, bundle, cfg, ood_clients)
        for name, p_scaled in preds.items():
            data = ood_clients[name]
            p = inverse_transform_target(p_scaled, scalers[name])
            y = inverse_transform_target(data.y, scalers[name])
            m = regression_metrics(y, p)
            m.update({"arm": arm, "client": name})
            rows.append(m)
    return pd.DataFrame(rows)


def degradation_table(normal: pd.DataFrame, ood: pd.DataFrame) -> pd.DataFrame:
    """Compare performance on deficit years against non-deficit years.

    Raw percentage RMSE change is reported but is NOT the headline metric.
    Deficit years have lower and less variable yields, so absolute RMSE is
    smaller there regardless of model quality, and a naive RMSE comparison
    reports "improvement" under drought.

    r2_drop is the primary robustness number: R2 on non-deficit minus R2 on
    deficit. Positive means the model explains less of the variance under
    drought, which is the thing the test is actually asking about.
    """
    keys = ["arm", "client"]
    n = normal.set_index(keys)
    o = ood.set_index(keys)
    common = n.index.intersection(o.index)

    out = pd.DataFrame({
        "rmse_normal": n.loc[common, "rmse"],
        "rmse_ood": o.loc[common, "rmse"],
        "r2_normal": n.loc[common, "r2"],
        "r2_ood": o.loc[common, "r2"],
    })
    out["pct_rmse_change"] = (
        (out["rmse_ood"] - out["rmse_normal"]) / out["rmse_normal"] * 100)
    out["r2_drop"] = out["r2_normal"] - out["r2_ood"]
    return out.reset_index().sort_values(["arm", "r2_drop"])


def summarise_degradation(degradation: pd.DataFrame) -> pd.DataFrame:
    """One row per arm, sorted by r2_drop.

    Expectation stated in advance: FedPer degrades least, because
    personalised heads absorb local drought response that a shared global
    model cannot. If the data does not show this, that is still a reportable
    result rather than a failed experiment.
    """
    g = degradation.groupby("arm")
    return pd.DataFrame({
        "mean_r2_drop": g["r2_drop"].mean(),
        "median_r2_drop": g["r2_drop"].median(),
        "worst_client_r2_drop": g["r2_drop"].max(),
        "mean_pct_rmse_change": g["pct_rmse_change"].mean(),
        "n_clients": g["r2_drop"].count(),
    }).sort_values("mean_r2_drop")
