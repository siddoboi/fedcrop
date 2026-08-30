"""Weight exchange helpers.

Passing `keys` restricts the exchange to a subset of the state dict, which is
the entirety of how FedPer is implemented: the head simply never travels.
"""

from __future__ import annotations

import copy

import torch


def get_parameters(model, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    keys = keys if keys is not None else list(sd)
    return {k: sd[k].detach().clone() for k in keys}


def set_parameters(model, params: dict[str, torch.Tensor],
                   keys: list[str] | None = None) -> None:
    """Load a subset of tensors, leaving every other tensor untouched.

    With keys set, local heads survive aggregation.
    """
    keys = keys if keys is not None else list(params)
    sd = model.state_dict()
    for k in keys:
        if k in params:
            sd[k] = params[k].detach().clone()
    model.load_state_dict(sd)


def clone_parameters(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in params.items()}


def named_parameter_view(model, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Live parameter references, for the FedProx proximal term."""
    out = dict(model.named_parameters())
    if keys is None:
        return out
    return {k: v for k, v in out.items() if k in keys}


def parameters_to_bytes(params: dict[str, torch.Tensor]) -> int:
    """Payload size of one client-to-server message, for the complexity table."""
    return int(sum(v.numel() * v.element_size() for v in params.values()))


def deepcopy_state(model) -> dict:
    return copy.deepcopy(model.state_dict())
