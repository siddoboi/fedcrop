"""Reading the source workbook. Read the .xls once, then use parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

EXPECTED_SHAPE = (12803, 107)


def resolve_workbook(path: str | Path) -> Path:
    """Find the workbook, tolerating filename variation.

    The Mendeley download ships with spaces and parentheses in its name, and
    people rename it. Rather than demand an exact match, fall back to any
    single .xls in the same directory.
    """
    path = Path(path)
    if path.exists():
        return path
    candidates = sorted(path.parent.glob("*.xls"))
    if len(candidates) == 1:
        log.info("configured name not found; using %s", candidates[0].name)
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{len(candidates)} .xls files in {path.parent}; cannot choose. "
            f"Set paths.raw_xls in config/base.yaml to one of: "
            + ", ".join(c.name for c in candidates)
        )
    raise FileNotFoundError(
        f"no .xls found in {path.parent}. Download the dataset from "
        "https://data.mendeley.com/datasets/ywp3y5j9vv/1 and place it there."
    )


def load_raw(path: str | Path, strict: bool = True) -> pd.DataFrame:
    """Read the ICRISAT .xls. Requires xlrd; openpyxl cannot read legacy .xls."""
    path = resolve_workbook(path)
    df = pd.read_excel(path, sheet_name=0, engine="xlrd")
    if df.shape != EXPECTED_SHAPE:
        msg = f"expected shape {EXPECTED_SHAPE}, got {df.shape}"
        if strict:
            raise ValueError(msg + " - this is not the expected dataset version")
        log.warning(msg)
    log.info("loaded raw workbook: %s rows, %s columns", *df.shape)
    return df


def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s", path)


def load_interim(path: str | Path) -> pd.DataFrame:
    """Load the cached panel. Everything downstream calls this, not load_raw."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached panel at {path}. Run scripts/01_build_features.py first."
        )
    return pd.read_parquet(path)


def ensure_interim(cfg) -> pd.DataFrame:
    """Load the parquet cache, building it from the .xls if absent."""
    interim = cfg.path("interim")
    if interim.exists():
        return load_interim(interim)
    df = load_raw(cfg.path("raw_xls"))
    to_parquet(df, interim)
    return df
