"""The source's -1 "not recorded" code in farm columns.

Treated as a number, -1 gives negative irrigation ratios and fertiliser per
hectare that look like real data. These tests pin each strategy down on a
small synthetic panel, and check that "keep" reproduces the old pipeline
exactly, so the submitted results stay reproducible.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedcrop import cleaning, columns as C  # noqa: E402
from fedcrop.config import Config, load_config  # noqa: E402

IRR = "GROSS IRRIGATED AREA (1000 ha)"
NIT = "NITROGEN CONSUMPTION (tons)"
STATES = ["Assam", "Punjab", "Kerala"]


def cfg_with(strategy: str) -> Config:
    raw = copy.deepcopy(load_config().raw)
    raw["cleaning"]["missing_code_strategy"] = strategy
    raw["cleaning"]["missing_codes"] = [-1]
    return Config(raw)


@pytest.fixture()
def panel() -> pd.DataFrame:
    """3 states x 4 districts x 1990-2015, climate complete, plausible farm figures.

    Assam has -1 irrigated area in most rows (as in the source); Punjab has a
    single -1 nitrogen value; Kerala is clean.
    """
    rng = np.random.default_rng(0)
    rows = []
    for s_i, state in enumerate(STATES):
        for d in range(4):
            for year in range(1990, 2016):
                rows.append({"Dist Code": s_i * 10 + d, "Dist Name": f"{state}{d}",
                             "State Code": s_i, "State Name": state, "Year": year})
    df = pd.DataFrame(rows)
    n = len(df)
    for col in C.sequential_columns():
        df[col] = rng.uniform(1, 100, n)
    df["RICE AREA (1000 ha)"] = rng.uniform(20, 80, n)
    df["GROSS CROPPED AREA (1000 ha)"] = rng.uniform(150, 300, n)
    df[IRR] = rng.uniform(10, 140, n)
    df[NIT] = rng.uniform(1000, 9000, n)
    df["PHOSPHATE CONSUMPTION (tons)"] = rng.uniform(500, 4000, n)
    df["POTASH CONSUMPTION (tons)"] = rng.uniform(200, 2000, n)
    df[C.TARGET] = rng.normal(2500, 300, n).clip(800)
    df[C.LEAKAGE_COLUMNS[0]] = 1.0                       # dropped by drop_excluded

    assam = df.index[df["State Name"] == "Assam"]
    df.loc[assam[:90], IRR] = -1                         # 90 of 104 Assam rows
    df.loc[df.index[df["State Name"] == "Punjab"][5], NIT] = -1
    return df


def old_pipeline(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """The order of operations before this change (report v5)."""
    work = cleaning.add_derived(cleaning.drop_excluded(raw))
    clean, _ = cleaning.apply_cascade(work, cfg)
    return cleaning.impute_per_client(clean, cfg)


def new_pipeline(raw: pd.DataFrame, cfg):
    work, report = cleaning.recode_missing_codes(cleaning.drop_excluded(raw), cfg)
    work = cleaning.add_derived(work)
    clean, audit = cleaning.apply_cascade(work, cfg)
    return cleaning.impute_per_client(clean, cfg), audit, report


def test_keep_reproduces_the_old_pipeline_exactly(panel):
    cfg = cfg_with("keep")
    new, _, report = new_pipeline(panel, cfg)
    pd.testing.assert_frame_equal(new, old_pipeline(panel, cfg))
    assert (new[IRR] == -1).sum() > 0                    # still there, by design
    assert report["rows_affected"] == 91


def test_impute_leaves_no_code_and_no_negative_ratio(panel):
    clean, audit, report = new_pipeline(panel, cfg_with("impute"))
    farm = cleaning.MISSING_CODE_COLUMNS + C.DERIVED_COVARIATES
    assert len(clean) == len(old_pipeline(panel, cfg_with("keep")))   # no rows lost
    assert not clean[farm].isin([-1]).any().any()
    assert (clean[C.DERIVED_COVARIATES] >= 0).all().all()
    assert clean[C.ANNUAL_COVARIATES].notna().all().all()
    assert "no missing-value code" not in set(audit["step"])
    assert report["values_by_column"][IRR] == 90 and report["values_by_column"][NIT] == 1
    assert report["rows_affected_by_state"] == {"Assam": 90, "Punjab": 1}


def test_impute_fills_from_the_states_own_training_values(panel):
    cfg = cfg_with("impute")
    clean, _, _ = new_pipeline(panel, cfg)
    assert clean[f"{IRR}__was_imputed"].sum() == 90
    # Assam keeps 8 real training-year values (one district, 2002-2009): the fill is their median
    real = panel[(panel["State Name"] == "Assam") & (panel[IRR] != -1)
                 & panel["Year"].between(*cfg.train_years)][IRR]
    filled = clean.loc[clean[f"{IRR}__was_imputed"], IRR]
    assert np.allclose(filled, real.median())


def test_imputation_support_counts_real_training_values(panel):
    cfg = cfg_with("impute")
    work, _ = cleaning.recode_missing_codes(cleaning.drop_excluded(panel), cfg)
    clean, _ = cleaning.apply_cascade(cleaning.add_derived(work), cfg)
    support = cleaning.imputation_support(clean, cfg)
    assert support["Kerala"][IRR] == support["Kerala"]["train_rows"] == 80
    assert support["Assam"][IRR] < 20


def test_drop_removes_the_rows_and_logs_the_step(panel):
    clean, audit, _ = new_pipeline(panel, cfg_with("drop"))
    step = audit[audit["step"] == "no missing-value code"].iloc[0]
    assert step["removed"] == 91
    assert step["worst_state"] == "Assam" and step["worst_state_removed"] == 90
    assert cleaning.MISSING_CODE_FLAG not in clean.columns
    assert not clean[cleaning.MISSING_CODE_COLUMNS].isin([-1]).any().any()


def test_climate_is_never_recoded(panel):
    col = C.sequential_columns()[0]
    panel.loc[0, col] = -1                               # e.g. a real -1 °C minimum
    work, _ = cleaning.recode_missing_codes(cleaning.drop_excluded(panel), cfg_with("impute"))
    assert work.loc[0, col] == -1


def test_unknown_strategy_is_refused(panel):
    with pytest.raises(ValueError, match="missing_code_strategy"):
        cleaning.recode_missing_codes(panel, cfg_with("median"))


def test_base_config_uses_impute():
    assert cleaning.missing_code_settings(load_config()) == ([-1.0], "impute")
