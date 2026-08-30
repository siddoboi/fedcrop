"""The federated loop.

A sequential simulation of 20 clients on one machine. No Ray, no framework:
FedAvg is a weighted mean of state dicts, FedProx adds one term to the client
loss, FedPer filters which keys are averaged. All three fit here.

Convergence is a stated criterion, not an eyeballed curve: stop when weighted
validation RMSE has not improved by min_delta for `patience` rounds, and
restore the best round rather than the last.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..datasets import ClientSplits, make_loader
from ..model import build_model
from ..train import predict, train_epoch
from .aggregate import aggregate
from .param_utils import get_parameters, parameters_to_bytes, set_parameters

log = logging.getLogger(__name__)

ALGORITHMS = ("fedavg", "fedprox", "fedper")


@dataclass
class FederationResult:
    algorithm: str
    global_state: dict
    client_heads: dict[str, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    best_round: int = -1
    best_val_rmse: float = float("inf")
    bytes_per_round: int = 0
    wall_clock_s: float = 0.0
    rounds_run: int = 0


def _weighted(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")


def client_update(model: nn.Module, data, cfg_fl: dict,
                  global_params: dict | None = None) -> tuple[dict, int, dict]:
    """Train one client for local_epochs. Returns (state, n_rows, metrics)."""
    loader = make_loader(data, batch_size=cfg_fl.get("batch_size", 64), shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg_fl.get("lr", 1e-3),
                                 weight_decay=cfg_fl.get("weight_decay", 1e-4))
    mu = float(cfg_fl.get("mu", 0.0))
    anchor = None
    if mu > 0 and global_params is not None:
        anchor = {k: v.detach().clone() for k, v in global_params.items()}

    losses = []
    for _ in range(cfg_fl.get("local_epochs", 2)):
        losses.append(train_epoch(model, loader, optimiser,
                                  global_params=anchor, mu=mu))
    return model.state_dict(), len(data), {"train_loss": float(np.mean(losses))}


def evaluate_clients(model_factory, global_state: dict, splits_pool: dict,
                     client_heads: dict[str, dict] | None,
                     base_keys: list[str] | None) -> dict[str, float]:
    """Per-client scaled RMSE. Under FedPer each client uses its own head."""
    out: dict[str, float] = {}
    for name, data in splits_pool.items():
        if len(data) == 0:
            continue
        model = model_factory()
        model.load_state_dict(global_state)
        if client_heads and name in client_heads:
            set_parameters(model, client_heads[name])
        p, t = predict(model, make_loader(data, shuffle=False))
        out[name] = float(np.sqrt(((p - t) ** 2).mean())) if len(t) else float("nan")
    return out


def run_federation(clients: ClientSplits, bundle, cfg,
                   algorithm: str = "fedavg", seed: int = 42,
                   overrides: dict | None = None) -> FederationResult:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected {ALGORITHMS}")

    fl_cfg = dict(cfg.raw.get("federated", {}))
    fl_cfg.update(overrides or {})
    if algorithm != "fedprox":
        fl_cfg["mu"] = 0.0

    torch.manual_seed(seed)
    factory = lambda: build_model(bundle, cfg)  # noqa: E731
    template = factory()
    base_keys = template.base_parameter_keys()
    head_keys = template.head_parameter_keys()
    agg_keys = base_keys if algorithm == "fedper" else None

    global_state = copy.deepcopy(template.state_dict())
    client_heads = {n: get_parameters(template, head_keys) for n in clients.names} \
        if algorithm == "fedper" else {}

    result = FederationResult(algorithm=algorithm, global_state=global_state)
    max_rounds = int(fl_cfg.get("rounds", 60))
    patience = int(fl_cfg.get("patience", 12))
    min_delta = float(fl_cfg.get("min_delta", 1e-4))
    best_state, best_heads, stale = None, None, 0
    started = time.time()

    for rnd in range(max_rounds):
        updates, weights, losses = [], [], []
        for name in clients.names:
            data = clients.train[name]
            if len(data) == 0:
                continue
            model = factory()
            model.load_state_dict(global_state)
            if algorithm == "fedper":
                set_parameters(model, client_heads[name])
            state, n, m = client_update(model, data, fl_cfg,
                                        global_params=dict(model.named_parameters())
                                        if fl_cfg.get("mu", 0) > 0 else None)
            if algorithm == "fedper":
                client_heads[name] = get_parameters(model, head_keys)
            updates.append({k: v.detach().clone() for k, v in state.items()})
            weights.append(float(n))
            losses.append(m["train_loss"])

        merged = aggregate(updates, weights, keys=agg_keys)
        for k, v in merged.items():
            global_state[k] = v

        if rnd == 0:
            result.bytes_per_round = parameters_to_bytes(
                {k: v for k, v in updates[0].items()
                 if agg_keys is None or k in agg_keys}
            ) * len(updates)

        val_rmse = evaluate_clients(factory, global_state, clients.val,
                                    client_heads, base_keys)
        w = [float(len(clients.train[n])) for n in val_rmse]
        weighted = _weighted(list(val_rmse.values()), w)
        result.history.append({
            "round": rnd,
            "train_loss": float(np.mean(losses)),
            "val_rmse_weighted": weighted,
            "val_rmse_worst": float(np.nanmax(list(val_rmse.values()))),
            "val_rmse_std": float(np.nanstd(list(val_rmse.values()))),
        })

        if weighted < result.best_val_rmse - min_delta:
            result.best_val_rmse, result.best_round, stale = weighted, rnd, 0
            best_state = copy.deepcopy(global_state)
            best_heads = copy.deepcopy(client_heads)
        else:
            stale += 1
            if stale >= patience:
                log.info("%s: stopping at round %d, best was %d",
                         algorithm, rnd, result.best_round)
                break

    result.rounds_run = len(result.history)
    result.wall_clock_s = time.time() - started
    if best_state is not None:
        result.global_state = best_state
        result.client_heads = best_heads or {}
    else:
        result.client_heads = client_heads
    log.info("%s: %d rounds, best round %d, val RMSE %.4f, %.1fs",
             algorithm, result.rounds_run, result.best_round,
             result.best_val_rmse, result.wall_clock_s)
    return result
