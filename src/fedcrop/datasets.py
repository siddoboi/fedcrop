"""Torch dataset wrappers and the shared client-preparation path.

prepare_clients is the single place where partitioning and scaling happen.
Centralised, local-only and federated training all go through it, so the only
difference between them is the optimisation procedure, not the preprocessing.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import FeatureBundle
from .scaling import ScalerParams, apply_scaler, fit_all_client_scalers
from .splits import ClientData, Splits, partition_by_client

log = logging.getLogger(__name__)


class YieldDataset(Dataset):
    def __init__(self, data: ClientData):
        self.x_seq = torch.from_numpy(np.ascontiguousarray(data.x_seq)).float()
        self.x_cov = torch.from_numpy(np.ascontiguousarray(data.x_cov)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(data.y)).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x_seq[i], self.x_cov[i], self.y[i]


def make_loader(data: ClientData, batch_size: int = 64,
                shuffle: bool = False) -> DataLoader:
    """Shuffling within a client is fine; the temporal boundary is a mask."""
    return DataLoader(YieldDataset(data), batch_size=batch_size, shuffle=shuffle)


def _concat(parts: list[ClientData], name: str = "pooled") -> ClientData:
    import pandas as pd
    return ClientData(
        name=name,
        x_seq=np.concatenate([p.x_seq for p in parts]),
        x_cov=np.concatenate([p.x_cov for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        keys=pd.concat([p.keys for p in parts], ignore_index=True),
    )


class ClientSplits:
    """Per-client train/val/test data, already scaled with that client's scaler."""

    def __init__(self, train: dict[str, ClientData], val: dict[str, ClientData],
                 test: dict[str, ClientData], scalers: dict[str, ScalerParams]):
        self.train, self.val, self.test, self.scalers = train, val, test, scalers
        self.names = sorted(train)

    def sample_counts(self) -> dict[str, int]:
        return {n: len(self.train[n]) for n in self.names}

    def pooled(self, which: str) -> ClientData:
        source = getattr(self, which)
        return _concat([source[n] for n in self.names if len(source[n])], which)


def prepare_clients(bundle: FeatureBundle, splits: Splits, cfg,
                    out_dir=None) -> ClientSplits:
    """Partition by state, fit each client's scaler on its own training rows.

    Every split is transformed with the scaler of the client it belongs to.
    No global statistics are computed anywhere in this function.
    """
    raw_train = partition_by_client(bundle, cfg, splits.train)
    raw_val = partition_by_client(bundle, cfg, splits.val)
    raw_test = partition_by_client(bundle, cfg, splits.test)

    usable = [n for n, c in raw_train.items() if len(c) > 1]
    dropped = sorted(set(raw_train) - set(usable))
    if dropped:
        detail = ", ".join(f"{n} ({len(raw_train[n])} train rows)" for n in dropped)
        log.warning(
            "EXCLUDED FROM FEDERATION: %d of %d clients have too few training "
            "rows to fit a scaler: %s. This is a data limitation, not a bug - "
            "report it in the methods section rather than letting a reader "
            "notice the client count silently drop.",
            len(dropped), len(raw_train), detail)

    scalers = fit_all_client_scalers({n: raw_train[n] for n in usable}, out_dir)

    def scale(pool: dict[str, ClientData]) -> dict[str, ClientData]:
        out = {}
        for n in usable:
            c = pool.get(n)
            if c is None or len(c) == 0:
                c = ClientData(n, bundle.x_seq[:0], bundle.x_cov[:0],
                               bundle.y[:0], bundle.keys.iloc[:0])
            out[n] = apply_scaler(c, scalers[n])
        return out

    cs = ClientSplits(scale(raw_train), scale(raw_val), scale(raw_test), scalers)
    log.info("prepared %d clients; train rows %d to %d", len(cs.names),
             min(cs.sample_counts().values()), max(cs.sample_counts().values()))
    return cs


def unscale_predictions(preds: dict[str, np.ndarray],
                        scalers: dict[str, ScalerParams]) -> dict[str, np.ndarray]:
    return {n: p * scalers[n].y_std + scalers[n].y_mean for n, p in preds.items()}
