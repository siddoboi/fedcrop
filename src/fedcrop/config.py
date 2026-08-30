"""Configuration loading and global seeding."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        """Resolve a configured path against the repo root."""
        return REPO_ROOT / self.raw["paths"][name]

    @property
    def target(self) -> str:
        return self.raw["data"]["target"]

    @property
    def district_key(self) -> str:
        return self.raw["data"]["district_key"]

    @property
    def state_key(self) -> str:
        return self.raw["data"]["partition_key"]

    @property
    def year_key(self) -> str:
        return self.raw["data"]["year_key"]

    @property
    def train_years(self) -> tuple[int, int]:
        return tuple(self.raw["split"]["train_years"])

    @property
    def val_years(self) -> tuple[int, int]:
        return tuple(self.raw["split"]["val_years"])

    @property
    def test_years(self) -> tuple[int, int]:
        return tuple(self.raw["split"]["test_years"])


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(experiment: str | None = None) -> Config:
    """Load base.yaml, optionally merged with an experiment override."""
    base_path = REPO_ROOT / "config" / "base.yaml"
    with open(base_path) as fh:
        cfg = yaml.safe_load(fh)

    if experiment:
        exp_path = REPO_ROOT / "config" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"no experiment config at {exp_path}")
        with open(exp_path) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})

    _validate(cfg)
    return Config(cfg)


def _validate(cfg: dict) -> None:
    tr = cfg["split"]["train_years"]
    va = cfg["split"]["val_years"]
    te = cfg["split"]["test_years"]
    if not (tr[1] < va[0] <= va[1] < te[0]):
        raise ValueError(
            f"split years must be strictly ordered and non-overlapping: "
            f"train {tr}, val {va}, test {te}"
        )
    window = cfg["features"]["window"]
    if window not in ("jan_dec", "jun_nov", "agri_year"):
        raise ValueError(f"unknown window {window!r}")


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness. Call at the top of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
