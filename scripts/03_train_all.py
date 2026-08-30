"""Stages C and D. Train every arm, across seeds, and build the ablation table.

    python scripts\\03_train_all.py                 # all five arms, all seeds
    python scripts\\03_train_all.py --seeds 42      # one seed, quick check
    python scripts\\03_train_all.py --arms centralised local

Arms: centralised (ceiling), local (floor), fedavg, fedprox, fedper.
Every arm is reported against the district_trend baseline via skill_score.

Writes:
    artifacts/results/ablation.json
    artifacts/results/federation_history.json
    artifacts/results/complexity.json
    artifacts/models/<arm>_seed<N>.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import baselines, complexity, features, io_layer, metrics, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402
from fedcrop.datasets import make_loader, prepare_clients  # noqa: E402
from fedcrop.fl import ALGORITHMS, run_federation  # noqa: E402
from fedcrop.fl.param_utils import set_parameters  # noqa: E402
from fedcrop.model import build_model  # noqa: E402
from fedcrop.train import predict  # noqa: E402
from fedcrop.train_models import (predict_centralised, predict_local,  # noqa: E402
                                  train_all_local, train_centralised)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

ARMS = ["centralised", "local", "fedavg", "fedprox", "fedper"]


def flatten(preds: dict, clients, which: str = "test"):
    """Unscale per client and stack into aligned arrays."""
    y_true, y_pred, client_ids, keys = [], [], [], []
    for name, p in preds.items():
        s = clients.scalers[name]
        data = getattr(clients, which)[name]
        y_true.append(data.y * s.y_std + s.y_mean)
        y_pred.append(p * s.y_std + s.y_mean)
        client_ids.append(np.full(len(p), name))
        keys.append(data.keys)
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(client_ids), pd.concat(keys, ignore_index=True))


def trend_lookup(bundle, sp, cfg) -> dict:
    """district_trend predictions keyed by (district, year), for skill_score."""
    pred = baselines.district_trend(bundle, sp, cfg)
    k = bundle.keys
    return dict(zip(zip(k[cfg.district_key], k[cfg.year_key]), pred))


def federated_predictions(result, clients, bundle, cfg, which="test"):
    out = {}
    for name in clients.names:
        data = getattr(clients, which)[name]
        if len(data) == 0:
            continue
        model = build_model(bundle, cfg)
        model.load_state_dict(result.global_state)
        if result.client_heads.get(name):
            set_parameters(model, result.client_heads[name])
        p, _ = predict(model, make_loader(data, shuffle=False))
        out[name] = p
    return out


def mu_sweep(clients, bundle, cfg, trend, seed: int, results_dir: Path) -> int:
    """Find where FedProx stops behaving like FedAvg.

    At mu = 0 FedProx is exactly FedAvg. If the chosen mu produces the same
    numbers as FedAvg, the proximal term is present but inert, and claiming
    FedProx as a distinct algorithm is not supported by the run. This sweep
    locates the value where the two actually diverge.
    """
    grid = cfg.raw["federated"].get("mu_grid", [0.001, 0.01, 0.1, 1.0])
    rows = []

    ref = run_federation(clients, bundle, cfg, algorithm="fedavg", seed=seed)
    ref_state = {k: v.clone() for k, v in ref.global_state.items()}
    preds = federated_predictions(ref, clients, bundle, cfg, "test")
    y_true, y_pred, cid, keys = flatten(preds, clients, "test")
    r = np.array([trend[(d, y)] for d, y in
                  zip(keys[cfg.district_key], keys[cfg.year_key])])
    m = metrics.summarise(y_true, y_pred, cid, r)
    rows.append({"mu": 0.0, "algorithm": "fedavg", "rmse": m["rmse"],
                 "r2": m["r2"], "skill_vs_trend": m["skill_vs_trend"],
                 "rounds": ref.rounds_run, "param_l2_vs_fedavg": 0.0})
    log.info("\n  mu=0.000 (fedavg)   RMSE %7.1f  R2 %6.3f  skill %+6.3f",
             m["rmse"], m["r2"], m["skill_vs_trend"])

    for mu in grid:
        res = run_federation(clients, bundle, cfg, algorithm="fedprox", seed=seed,
                             overrides={"mu": float(mu)})
        preds = federated_predictions(res, clients, bundle, cfg, "test")
        y_true, y_pred, cid, keys = flatten(preds, clients, "test")
        r = np.array([trend[(d, y)] for d, y in
                      zip(keys[cfg.district_key], keys[cfg.year_key])])
        m = metrics.summarise(y_true, y_pred, cid, r)
        drift = float(sum(((res.global_state[k].float() - v.float()) ** 2).sum()
                          for k, v in ref_state.items()) ** 0.5)
        rows.append({"mu": float(mu), "algorithm": "fedprox", "rmse": m["rmse"],
                     "r2": m["r2"], "skill_vs_trend": m["skill_vs_trend"],
                     "rounds": res.rounds_run, "param_l2_vs_fedavg": drift})
        log.info("  mu=%-9.3f        RMSE %7.1f  R2 %6.3f  skill %+6.3f  "
                 "L2 from FedAvg %.4f", mu, m["rmse"], m["r2"],
                 m["skill_vs_trend"], drift)

    df = pd.DataFrame(rows)
    log.info("\n=== FedProx mu sweep, seed %d ===\n%s", seed,
             df.round(4).to_string(index=False))

    prox = df[df.algorithm == "fedprox"]
    inert = prox[prox.param_l2_vs_fedavg < 1e-3]["mu"].tolist()
    if inert:
        log.info("\n  mu values indistinguishable from FedAvg: %s", inert)
    best = prox.loc[prox.skill_vs_trend.idxmax()]
    log.info("  best skill at mu=%.3f (skill %+0.3f, L2 drift %.4f)",
             best["mu"], best["skill_vs_trend"], best["param_l2_vs_fedavg"])
    log.info("  set federated.mu in config/base.yaml to the smallest mu that "
             "both diverges from FedAvg and does not hurt skill")

    with open(results_dir / "mu_sweep.json", "w") as fh:
        json.dump(df.to_dict(orient="records"), fh, indent=2)
    log.info("\nwrote %s", results_dir / "mu_sweep.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--mu-sweep", action="store_true",
                    help="sweep the FedProx proximal strength and exit")
    args = ap.parse_args()

    cfg = load_config()
    seeds = args.seeds if args.seeds is not None else cfg["run"]["seeds"]
    results_dir = cfg.path("results")
    models_dir = cfg.path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    clean_path = cfg.path("interim").with_name("clean.parquet")
    if not clean_path.exists():
        log.error("no cleaned panel. Run scripts/01_build_features.py first.")
        return 1
    clean = io_layer.load_interim(clean_path)
    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)
    clients = prepare_clients(bundle, sp, cfg, out_dir=cfg.path("scalers"))

    trend = trend_lookup(bundle, sp, cfg)
    base_table = baselines.baseline_table(bundle, sp, cfg)

    if args.mu_sweep:
        return mu_sweep(clients, bundle, cfg, trend, seeds[0], results_dir)

    rows: list[dict] = []
    fed_histories: dict[str, list] = {}
    fed_results_for_complexity: dict = {}

    for _, r in base_table.iterrows():
        rows.append({"arm": r["model"], "seed": None, "rmse": r["rmse"],
                     "mae": r["mae"], "r2": r["r2"],
                     "worst_client_rmse": r["worst_client_rmse"],
                     "cross_client_rmse_std": r["cross_client_rmse_std"],
                     "skill_vs_trend": np.nan, "n": r["n"], "seconds": 0.0})

    for seed in seeds:
        set_global_seed(seed)
        log.info("\n%s seed %d %s", "=" * 26, seed, "=" * 26)

        for arm in args.arms:
            started = time.time()
            if arm == "centralised":
                model, _ = train_centralised(clients, bundle, cfg, seed)
                preds = predict_centralised(model, clients, "test")
                torch.save(model.state_dict(), models_dir / f"centralised_seed{seed}.pt")
            elif arm == "local":
                models, _ = train_all_local(clients, bundle, cfg, seed)
                preds = predict_local(models, clients, "test")
                torch.save({k: m.state_dict() for k, m in models.items()},
                           models_dir / f"local_seed{seed}.pt")
            else:
                res = run_federation(clients, bundle, cfg, algorithm=arm, seed=seed)
                preds = federated_predictions(res, clients, bundle, cfg, "test")
                torch.save({"global": res.global_state, "heads": res.client_heads},
                           models_dir / f"{arm}_seed{seed}.pt")
                fed_histories.setdefault(arm, []).append(
                    {"seed": seed, "best_round": res.best_round,
                     "rounds": res.rounds_run, "history": res.history})
                fed_results_for_complexity[arm] = res

            y_true, y_pred, client_ids, keys = flatten(preds, clients, "test")
            ref = np.array([trend[(d, y)] for d, y in
                            zip(keys[cfg.district_key], keys[cfg.year_key])])
            m = metrics.summarise(y_true, y_pred, client_ids, ref, label=arm)
            m.update({"arm": arm, "seed": seed, "seconds": round(time.time() - started, 1)})
            m.pop("model", None)
            rows.append(m)
            log.info("  %-12s RMSE %7.1f  R2 %6.3f  skill %+6.3f  worst %7.1f  (%.0fs)",
                     arm, m["rmse"], m["r2"], m["skill_vs_trend"],
                     m["worst_client_rmse"], m["seconds"])

    df = pd.DataFrame(rows)
    cols = ["arm", "rmse", "mae", "r2", "skill_vs_trend",
            "worst_client_rmse", "cross_client_rmse_std"]
    summary = (df.groupby("arm", dropna=False)[cols[1:]]
               .agg(["mean", "std"]).round(3))
    log.info("\n=== ablation, mean and sd across %d seed(s) ===\n%s",
             len(seeds), summary.to_string())

    with open(results_dir / "ablation.json", "w") as fh:
        json.dump({"runs": df.replace({np.nan: None}).to_dict(orient="records"),
                   "seeds": list(seeds),
                   "reference_model": "district_trend"}, fh, indent=2)
    if fed_histories:
        with open(results_dir / "federation_history.json", "w") as fh:
            json.dump(fed_histories, fh, indent=2)

    report = complexity.complexity_report(build_model(bundle, cfg),
                                          fed_results_for_complexity or None)
    with open(results_dir / "complexity.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("\n=== complexity ===")
    for k, v in report["model"].items():
        log.info("  %-24s %s", k, v)
    if "communication" in report:
        log.info("\n%s", pd.DataFrame(report["communication"]).to_string(index=False))

    log.info("\nwrote ablation.json, federation_history.json, complexity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
