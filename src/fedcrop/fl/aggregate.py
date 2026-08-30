"""Aggregation.

FedAvg, FedProx and FedPer all use this one function. FedAvg and FedProx pass
every key; FedPer passes only the base keys. The algorithms differ in what
travels and what the client loss contains, not in how averaging works.
"""

from __future__ import annotations

import torch


def aggregate(updates: list[dict[str, torch.Tensor]], weights: list[float],
              keys: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Sample-count-weighted mean of client tensors."""
    if not updates:
        raise ValueError("no client updates to aggregate")
    if len(updates) != len(weights):
        raise ValueError("updates and weights must be the same length")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError("client weights sum to zero")
    keys = keys if keys is not None else list(updates[0])

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        present = [(u[k], w) for u, w in zip(updates, weights) if k in u]
        if not present:
            continue
        denom = float(sum(w for _, w in present))
        stacked = torch.stack([t.float() * (w / denom) for t, w in present])
        out[k] = stacked.sum(dim=0).to(present[0][0].dtype)
    return out
