"""Centralised ceiling and local-only floor.

Both use the identical architecture, optimiser and epoch budget as the
federated clients, and both use per-client scaling. The only difference
between the three arms is the optimisation procedure.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .datasets import ClientSplits, make_loader
from .model import build_model
from .train import TrainHistory, fit, predict

log = logging.getLogger(__name__)


def _fit_kwargs(cfg) -> dict:
    t = cfg.raw.get("training", {})
    return {
        "epochs": t.get("epochs", 200),
        "lr": t.get("lr", 1e-3),
        "weight_decay": t.get("weight_decay", 1e-4),
        "patience": t.get("patience", 20),
    }


def train_centralised(clients: ClientSplits, bundle, cfg, seed: int = 42):
    """Pooled training rows, per-client scaling retained."""
    torch.manual_seed(seed)
    model = build_model(bundle, cfg)
    batch = cfg.raw.get("training", {}).get("batch_size", 64)
    train_loader = make_loader(clients.pooled("train"), batch, shuffle=True)
    val_loader = make_loader(clients.pooled("val"), batch, shuffle=False)
    hist = fit(model, train_loader, val_loader, **_fit_kwargs(cfg))
    log.info("centralised: best epoch %d, val RMSE %.4f",
             hist.best_epoch, hist.best_val_rmse)
    return model, hist


def train_all_local(clients: ClientSplits, bundle, cfg, seed: int = 42):
    """One model per client, no communication. The floor."""
    models: dict = {}
    histories: dict[str, TrainHistory] = {}
    batch = cfg.raw.get("training", {}).get("batch_size", 64)
    for i, name in enumerate(clients.names):
        torch.manual_seed(seed + i)
        model = build_model(bundle, cfg)
        train_loader = make_loader(clients.train[name], batch, shuffle=True)
        val = clients.val[name]
        val_loader = make_loader(val, batch, shuffle=False) if len(val) else None
        histories[name] = fit(model, train_loader, val_loader, **_fit_kwargs(cfg))
        models[name] = model
    log.info("local-only: trained %d client models", len(models))
    return models, histories


def predict_centralised(model, clients: ClientSplits,
                        which: str = "test") -> dict[str, np.ndarray]:
    out = {}
    for name in clients.names:
        data = getattr(clients, which)[name]
        if len(data) == 0:
            continue
        p, _ = predict(model, make_loader(data, shuffle=False))
        out[name] = p
    return out


def predict_local(models: dict, clients: ClientSplits,
                  which: str = "test") -> dict[str, np.ndarray]:
    out = {}
    for name in clients.names:
        data = getattr(clients, which)[name]
        if len(data) == 0 or name not in models:
            continue
        p, _ = predict(models[name], make_loader(data, shuffle=False))
        out[name] = p
    return out
