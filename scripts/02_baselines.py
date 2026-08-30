"""Stage B. Evaluate the three baselines and freeze them.

    python scripts/02_baselines.py

Every model in this project is measured against district_trend. Reporting R2
without it beside it overstates results by roughly the between-district
variance, which dominates this problem.

Writes artifacts/results/baselines.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import baselines, features, io_layer, metrics, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("baselines")


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["run"]["seed"])

    clean_path = cfg.path("interim").with_name("clean.parquet")
    if not clean_path.exists():
        log.error("no cleaned panel. Run scripts/01_build_features.py first.")
        return 1
    clean = io_layer.load_interim(clean_path)

    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)

    table = baselines.baseline_table(bundle, sp, cfg)
    log.info("\n%s", table.to_string(index=False))

    y_true = bundle.y[sp.test]
    clients = bundle.keys.loc[sp.test, cfg.state_key].to_numpy()
    trend = baselines.district_trend(bundle, sp, cfg)[sp.test]
    per_client = metrics.per_client_metrics(y_true, trend, clients)
    log.info("\nper-client RMSE, district_trend baseline:\n%s",
             per_client[["rmse", "mae", "r2", "n"]].round(1).to_string())

    r2 = dict(zip(table["model"], table["r2"]))
    log.info("\nreference for skill_score: district_trend, R2 = %.3f", r2["district_trend"])
    if r2["district_trend"] > 0.6:
        log.warning("R2 above 0.6 suggests zero-yield rows survived cleaning")

    out = {
        "reference_model": "district_trend",
        "table": table.to_dict(orient="records"),
        "per_client_trend": per_client.reset_index().to_dict(orient="records"),
    }
    path = cfg.path("results") / "baselines.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
