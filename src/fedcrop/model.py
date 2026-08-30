"""The yield model.

GRU over the 12-month climate sequence, final hidden state concatenated with
the 9 annual covariates, then a small dense head.

The forward signature takes two arguments deliberately. Captum needs separate
input groups to attribute over the sequence and the covariates independently,
and merging them into one tensor would make that impossible.

base_parameter_keys and head_parameter_keys declare the FedPer split once,
here, rather than scattering the decision through the federated code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class YieldGRU(nn.Module):
    def __init__(self, n_seq_vars: int = 5, n_covariates: int = 9,
                 hidden: int = 64, dense: int = 32, dropout: float = 0.2,
                 num_layers: int = 1):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=n_seq_vars, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_covariates, dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_seq)
        return self.head(torch.cat([h[-1], x_cov], dim=1)).squeeze(-1)

    def base_parameter_keys(self) -> list[str]:
        """Shared under FedPer: the GRU. Averaged across clients."""
        return [k for k in self.state_dict() if k.startswith("gru.")]

    def head_parameter_keys(self) -> list[str]:
        """Kept local under FedPer. Never transmitted."""
        return [k for k in self.state_dict() if k.startswith("head.")]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def build_model(bundle=None, cfg=None, **kwargs) -> YieldGRU:
    """Construct a model sized from the feature bundle."""
    if bundle is not None:
        kwargs.setdefault("n_seq_vars", bundle.x_seq.shape[2])
        kwargs.setdefault("n_covariates", bundle.x_cov.shape[1])
    if cfg is not None:
        m = cfg.raw.get("model", {})
        for key in ("hidden", "dense", "dropout", "num_layers"):
            if key in m:
                kwargs.setdefault(key, m[key])
    return YieldGRU(**kwargs)
