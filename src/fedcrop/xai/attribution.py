"""Feature attribution.

The hardest design choice in this project lives in make_baseline.

Centralised SHAP can draw its reference distribution from the pooled training
set. Federated SHAP structurally cannot: no client may see another client's
data, so each client must attribute against its own baseline. Centralised and
federated attributions are therefore computed against different reference
points by construction.

That is not a flaw to be corrected. It is part of what the agreement metric
measures, and it must be stated explicitly wherever tau is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap, IntegratedGradients

from ..datasets import ClientSplits
from ..splits import ClientData

log = logging.getLogger(__name__)

BASELINE_STRATEGIES = ("client_mean", "client_sample", "zero", "global_mean")


@dataclass
class ClientAttribution:
    client: str
    method: str
    baseline_strategy: str
    seq: np.ndarray          # (n, T, V) per-sample attributions
    cov: np.ndarray          # (n, C)
    feature_names: list[str]
    n_samples: int

    def mean_abs(self) -> pd.Series:
        """One score per named feature, time axis collapsed."""
        seq = np.abs(self.seq).mean(axis=0).reshape(-1)
        cov = np.abs(self.cov).mean(axis=0)
        return pd.Series(np.concatenate([seq, cov]), index=self.feature_names)

    def signed_mean(self) -> pd.Series:
        seq = self.seq.mean(axis=0).reshape(-1)
        cov = self.cov.mean(axis=0)
        return pd.Series(np.concatenate([seq, cov]), index=self.feature_names)


def make_baseline(data: ClientData, strategy: str = "client_mean",
                  n_samples: int = 32, seed: int = 0,
                  global_pool: ClientData | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the reference input for attribution.

    client_mean    that client's own mean input. Federated-legal.
    client_sample  a random draw from that client's rows. Federated-legal.
    zero           the scaled-space origin, i.e. the client mean after scaling.
    global_mean    pooled mean across clients. Centralised only; passing this
                   for a federated model would violate the privacy premise.
    """
    if strategy not in BASELINE_STRATEGIES:
        raise ValueError(f"unknown baseline strategy {strategy!r}")

    if strategy == "global_mean":
        if global_pool is None:
            raise ValueError("global_mean baseline requires the pooled data, "
                             "which is not available to a federated client")
        src = global_pool
    else:
        src = data

    if strategy == "client_sample":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(src), size=min(n_samples, len(src)), replace=False)
        seq, cov = src.x_seq[idx], src.x_cov[idx]
    elif strategy == "zero":
        seq = np.zeros((1, *src.x_seq.shape[1:]), dtype=np.float32)
        cov = np.zeros((1, src.x_cov.shape[1]), dtype=np.float32)
    else:
        seq = src.x_seq.mean(axis=0, keepdims=True)
        cov = src.x_cov.mean(axis=0, keepdims=True)
        seq = np.repeat(seq, n_samples, axis=0)
        cov = np.repeat(cov, n_samples, axis=0)

    return (torch.from_numpy(np.ascontiguousarray(seq)).float(),
            torch.from_numpy(np.ascontiguousarray(cov)).float())


def _to_tensors(data: ClientData) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(np.ascontiguousarray(data.x_seq)).float(),
            torch.from_numpy(np.ascontiguousarray(data.x_cov)).float())


def gradient_shap(model, inputs, baselines, n_samples: int = 50,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model.eval()
    attr = GradientShap(model)
    seq_a, cov_a = attr.attribute(inputs, baselines=baselines,
                                  n_samples=n_samples, stdevs=0.09)
    return seq_a.detach().numpy(), cov_a.detach().numpy()


def integrated_gradients(model, inputs, baselines,
                         steps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Second opinion. Broad disagreement with GradientShap means one is
    misconfigured, not that the model is ambiguous."""
    model.eval()
    base = tuple(b.mean(dim=0, keepdim=True) for b in baselines)
    attr = IntegratedGradients(model)
    seq_a, cov_a = attr.attribute(inputs, baselines=base, n_steps=steps)
    return seq_a.detach().numpy(), cov_a.detach().numpy()


def attribute_client(model, data: ClientData, feature_names: list[str],
                     method: str = "gradient_shap",
                     baseline_strategy: str = "client_mean",
                     max_rows: int = 400, seed: int = 0,
                     global_pool: ClientData | None = None) -> ClientAttribution:
    if len(data) == 0:
        raise ValueError(f"client {data.name} has no rows to attribute")

    if len(data) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(data), size=max_rows, replace=False))
        data = ClientData(data.name, data.x_seq[idx], data.x_cov[idx],
                          data.y[idx], data.keys.iloc[idx].reset_index(drop=True))

    inputs = _to_tensors(data)
    baselines = make_baseline(data, baseline_strategy, seed=seed,
                              global_pool=global_pool)

    if method == "gradient_shap":
        seq_a, cov_a = gradient_shap(model, inputs, baselines, seed=seed)
    elif method == "integrated_gradients":
        seq_a, cov_a = integrated_gradients(model, inputs, baselines)
    else:
        raise ValueError(f"unknown method {method!r}")

    return ClientAttribution(client=data.name, method=method,
                             baseline_strategy=baseline_strategy,
                             seq=seq_a, cov=cov_a,
                             feature_names=feature_names, n_samples=len(data))


def attribute_all_clients(model_for, clients: ClientSplits,
                          feature_names: list[str], which: str = "test",
                          **kwargs) -> dict[str, ClientAttribution]:
    """model_for(client_name) returns the model that client would use.

    Under FedPer that is the global base plus the client's own head, so this
    aggregates over twenty differently-headed models. Legitimate, and it is a
    second reason federated and centralised attributions diverge.
    """
    out = {}
    for name in clients.names:
        data = getattr(clients, which)[name]
        if len(data) == 0:
            continue
        out[name] = attribute_client(model_for(name), data, feature_names, **kwargs)
    log.info("attributed %d clients (%s, %s baseline)", len(out),
             kwargs.get("method", "gradient_shap"),
             kwargs.get("baseline_strategy", "client_mean"))
    return out
