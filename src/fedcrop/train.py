"""Shared training primitives.

train_epoch and evaluate are used identically by centralised, local-only and
federated client code, so no method gets a different optimiser, loss or epoch
budget than another. That is what makes the ablation table a fair comparison.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_rmse: float = float("inf")


def train_epoch(model: nn.Module, loader: DataLoader,
                optimiser: torch.optim.Optimizer,
                loss_fn: nn.Module | None = None,
                global_params: dict | None = None,
                mu: float = 0.0) -> float:
    """One pass. When mu > 0 the FedProx proximal term is added to the loss.

    That term is the entirety of FedProx. Without it, a FedProx run is FedAvg
    with extra logging.
    """
    loss_fn = loss_fn or nn.MSELoss()
    model.train()
    total, n = 0.0, 0
    for x_seq, x_cov, y in loader:
        optimiser.zero_grad()
        loss = loss_fn(model(x_seq, x_cov), y)
        if mu > 0 and global_params is not None:
            prox = sum(((p - global_params[k].detach()) ** 2).sum()
                       for k, p in model.named_parameters()
                       if k in global_params)
            loss = loss + (mu / 2.0) * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predictions, targets) in scaled space. Unscale in the caller."""
    model.eval()
    preds, targets = [], []
    for x_seq, x_cov, y in loader:
        preds.append(model(x_seq, x_cov).numpy())
        targets.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(targets)


def scaled_rmse(model: nn.Module, loader: DataLoader) -> float:
    p, t = predict(model, loader)
    if len(t) == 0:
        return float("nan")
    return float(np.sqrt(((p - t) ** 2).mean()))


def fit(model: nn.Module, train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
        patience: int = 20, min_delta: float = 1e-4,
        restore_best: bool = True) -> TrainHistory:
    """Train with early stopping on validation RMSE, restoring the best epoch.

    Reporting the final epoch instead of the best understates every result,
    and does so inconsistently across methods.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = TrainHistory()
    best_state, stale = None, 0

    for epoch in range(epochs):
        hist.train_loss.append(train_epoch(model, train_loader, optimiser))
        if val_loader is None or len(val_loader.dataset) == 0:
            continue
        rmse = scaled_rmse(model, val_loader)
        hist.val_rmse.append(rmse)
        if rmse < hist.best_val_rmse - min_delta:
            hist.best_val_rmse, hist.best_epoch, stale = rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return hist
