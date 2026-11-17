"""An uploaded -1 farm figure is named as the source's missing-value code.

Reuses the synthetic artifacts of test_predict.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_predict import env, rows, upload  # noqa: E402,F401  (env is a fixture)


def test_missing_value_code_is_named_not_called_negative(env):  # noqa: F811
    df = rows(3).astype(object)
    df.loc[0, "gross_irrigated_area"] = -1
    df.loc[1, "potash_consumption"] = -3
    body = upload(env.client, df).json()
    reasons = {r["row"]: " ".join(r["reasons"]) for r in body["rejected"]}
    assert body["summary"]["accepted"] == 1
    assert "missing-value code" in reasons[2] and "cannot be negative" not in reasons[2]
    assert "potash_consumption cannot be negative" in reasons[3]
