"""Gate A. Reproduce every documented constant from the loader itself.

If any check fails, the loader is wrong and nothing downstream is meaningful.
Run this before writing or trusting any model code.

    python scripts/00_verify.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import baselines, cleaning, columns as C, features, io_layer, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("verify")

TOL = 0.003
results: list[tuple[str, float, float, bool]] = []


def check(name: str, actual: float, expected: float, tol: float = TOL) -> None:
    ok = abs(actual - expected) <= tol
    results.append((name, actual, expected, ok))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<46} {actual:>10.3f}   expected {expected:>8.3f}")


def within_district_corr(df: pd.DataFrame, x_col: str, y_col: str,
                         group: str) -> float:
    sub = df[[x_col, y_col, group]].dropna()
    x = sub[x_col] - sub.groupby(group)[x_col].transform("mean")
    y = sub[y_col] - sub.groupby(group)[y_col].transform("mean")
    return float(x.corr(y))


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["run"]["seed"])

    print("\n=== 1. raw workbook ===")
    raw = io_layer.load_raw(cfg.path("raw_xls"))
    check("rows", float(raw.shape[0]), 12803.0, tol=0)
    check("columns", float(raw.shape[1]), 107.0, tol=0)
    check("districts", float(raw["Dist Code"].nunique()), 560.0, tol=0)
    check("states", float(raw["State Name"].nunique()), 20.0, tol=0)

    print("\n=== 2. column naming ===")
    C.validate_columns(raw, C.sequential_columns())
    print("  [PASS] all 60 sequential columns resolve")
    assert C.build_column_name("windspeed", "SEPTEMBER") == "SEPT WINDSPEED (Meter per second)"
    assert C.build_column_name("precipitation", "SEPTEMBER") == "SEPTEMBER PERCIPITATION (Millimeters)"
    print("  [PASS] windspeed abbreviation and PERCIPITATION typo handled")

    print("\n=== 3. correlations on the full yield-present panel ===")
    panel = raw[raw[C.TARGET].notna()]
    check("yield-present rows", float(len(panel)), 12751.0, tol=0)
    for month, pooled_exp, within_exp in [
        ("SEPTEMBER", -0.050, 0.105),
        ("NOVEMBER", 0.306, -0.020),
        ("JULY", -0.095, 0.084),
    ]:
        col = C.build_column_name("precipitation", month)
        check(f"pooled r, {month.lower()} precip",
              float(panel[col].corr(panel[C.TARGET])), pooled_exp)
        check(f"within-district r, {month.lower()} precip",
              within_district_corr(panel, col, C.TARGET, "Dist Code"), within_exp)

    print("\n=== 4. JJAS precipitation ===")
    # Computed on all rows, not just yield-present ones: that is how the
    # documented 734.6 figure was derived, and climate exists for rows where
    # yield does not.
    jjas_cols = [C.build_column_name("precipitation", m) for m in C.JJAS]
    jjas = pd.DataFrame({"Year": raw["Year"], "jjas": raw[jjas_cols].sum(axis=1)})
    by_year = jjas.groupby("Year")["jjas"].mean()
    check("2002 JJAS mm", float(by_year.loc[2002]), 734.6, tol=0.5)
    check("2015 JJAS departure pct",
          float((by_year.loc[2015] - by_year.mean()) / by_year.mean() * 100), -2.7, tol=0.3)

    print("\n=== 5. collinearity and leakage ===")
    npk = raw[["NITROGEN CONSUMPTION (tons)", "PHOSPHATE CONSUMPTION (tons)",
               "POTASH CONSUMPTION (tons)"]].sum(axis=1)
    tot = raw["TOTAL FERTILISER CONSUMPTION (tons)"]
    ok = npk.notna() & tot.notna()
    check("corr(N+P+K, total fertiliser)", float(np.corrcoef(npk[ok], tot[ok])[0, 1]), 1.0)

    print("\n=== 6. cleaning cascade ===")
    work = cleaning.drop_excluded(raw)
    work = cleaning.add_derived(work)
    clean, audit = cleaning.apply_cascade(work, cfg)
    print(audit.to_string(index=False))
    clean = cleaning.impute_per_client(clean, cfg)

    print("\n=== 7. baselines on the cleaned panel ===")
    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)
    table = baselines.baseline_table(bundle, sp, cfg)
    print(table.to_string(index=False))

    r2 = dict(zip(table["model"], table["r2"]))
    check("district_mean R2", r2["district_mean"], 0.304, tol=0.03)
    check("district_trend R2", r2["district_trend"], 0.348, tol=0.03)
    if r2["district_trend"] <= r2["district_mean"]:
        print("  [WARN] trend does not beat the mean - check the trend fallback chain")
    if r2["district_trend"] > 0.6:
        print("  [FAIL] R2 above 0.6 suggests zero-yield rows survived cleaning")

    print("\n=== 8. leakage guards ===")
    splits.assert_no_temporal_leak(bundle, sp, cfg)
    print("  [PASS] no temporal leak between train, val and test")
    deficit = splits.per_client_deficit_years(clean, cfg)
    n_2015 = sum(1 for years in deficit.values() if 2015 in years)
    print(f"  [INFO] 2015 is a deficit year for {n_2015} of {len(deficit)} clients")

    failed = [r for r in results if not r[3]]
    print("\n" + "=" * 72)
    if failed:
        print(f"GATE A FAILED: {len(failed)} of {len(results)} checks did not pass")
        for name, actual, expected, _ in failed:
            print(f"  {name}: got {actual:.4f}, expected {expected:.4f}")
        return 1
    print(f"GATE A PASSED: {len(results)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
