"""Stage A. Build the cleaned panel and cache it, plus the confound table.

    python scripts/01_build_features.py

Writes:
    data/interim/panel.parquet     raw workbook cached
    data/interim/clean.parquet     after the cleaning cascade
    artifacts/results/meta.json    client sizes, cleaning audit
    artifacts/results/confound.json  pooled vs within-district correlations
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import cleaning, columns as C, features, io_layer, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build")


def confound_table(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Pooled and within-district correlation of yield with each month's climate."""
    rows = []
    for var in C.SEQUENTIAL_VARS:
        for month in C.MONTHS_FULL:
            col = C.build_column_name(var, month)
            sub = panel[[col, cfg.target, cfg.district_key]].dropna()
            x_dm = sub[col] - sub.groupby(cfg.district_key)[col].transform("mean")
            y_dm = sub[cfg.target] - sub.groupby(cfg.district_key)[cfg.target].transform("mean")
            rows.append({
                "variable": var,
                "month": C.MONTH_SHORT[month],
                "pooled_r": round(float(sub[col].corr(sub[cfg.target])), 4),
                "within_district_r": round(float(x_dm.corr(y_dm)), 4),
            })
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["run"]["seed"])
    results_dir = cfg.path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = io_layer.ensure_interim(cfg)

    log.info("\n--- confound table ---")
    panel = raw[raw[cfg.target].notna()]
    confound = confound_table(panel, cfg)
    precip = confound[confound.variable == "precipitation"]
    log.info("\n%s", precip[["month", "pooled_r", "within_district_r"]].to_string(index=False))
    confound.to_json(results_dir / "confound.json", orient="records", indent=2)

    log.info("\n--- cleaning ---")
    work = cleaning.add_derived(cleaning.drop_excluded(raw))
    clean, audit = cleaning.apply_cascade(work, cfg)
    clean = cleaning.impute_per_client(clean, cfg)
    io_layer.to_parquet(clean, cfg.path("interim").with_name("clean.parquet"))

    log.info("\n--- assembling ---")
    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)
    clients = splits.partition_by_client(bundle, cfg)
    deficit = splits.per_client_deficit_years(clean, cfg)

    meta = {
        "rows": int(len(clean)),
        "districts": int(clean[cfg.district_key].nunique()),
        "states": int(clean[cfg.state_key].nunique()),
        "year_min": int(clean[cfg.year_key].min()),
        "year_max": int(clean[cfg.year_key].max()),
        "split_sizes": {"train": int(sp.train.sum()), "val": int(sp.val.sum()),
                        "test": int(sp.test.sum())},
        "n_features": int(bundle.n_features),
        "feature_names": bundle.feature_names,
        "clients": {
            name: {
                "rows": len(c),
                "mean_yield": round(float(c.y.mean()), 1),
                "deficit_years": deficit.get(name, []),
            }
            for name, c in clients.items()
        },
        "cleaning_audit": audit.to_dict(orient="records"),
    }
    with open(results_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("wrote %s and %s", results_dir / "meta.json", results_dir / "confound.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
