"""Stage E. Attribution, aggregation, agreement, fidelity.

    python scripts\\04_explain.py                  # all seeds from config
    python scripts\\04_explain.py --seeds 42       # quick check

Requires checkpoints from scripts/03_train_all.py.

Writes:
    artifacts/results/attributions.json
    artifacts/results/agreement.json
    artifacts/results/fidelity.json
    artifacts/results/heterogeneity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import features, io_layer, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402
from fedcrop.datasets import prepare_clients  # noqa: E402
from fedcrop.fl.param_utils import set_parameters  # noqa: E402
from fedcrop.model import build_model  # noqa: E402
from fedcrop.xai import aggregate as agg  # noqa: E402
from fedcrop.xai import agreement as agr  # noqa: E402
from fedcrop.xai import fidelity as fid  # noqa: E402
from fedcrop.xai.attribution import attribute_all_clients  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("explain")

FED_ARMS = ["fedavg", "fedprox", "fedper"]


def load_centralised(path: Path, bundle, cfg):
    model = build_model(bundle, cfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return lambda _name: model


def load_federated(path: Path, bundle, cfg):
    """Returns a factory. Under FedPer each client gets its own head, so the
    aggregate is taken over differently-headed models."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    global_state, heads = blob["global"], blob.get("heads", {})

    def factory(name: str):
        model = build_model(bundle, cfg)
        model.load_state_dict(global_state)
        if heads.get(name):
            set_parameters(model, heads[name])
        model.eval()
        return model
    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--baseline", default="client_mean",
                    choices=["client_mean", "client_sample", "zero"])
    ap.add_argument("--max-rows", type=int, default=300)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    seeds = args.seeds if args.seeds is not None else cfg["run"]["seeds"]
    results_dir, models_dir = cfg.path("results"), cfg.path("models")

    clean_path = cfg.path("interim").with_name("clean.parquet")
    if not clean_path.exists():
        log.error("no cleaned panel. Run scripts/01_build_features.py first.")
        return 1
    clean = io_layer.load_interim(clean_path)
    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)
    clients = prepare_clients(bundle, sp, cfg)
    names = bundle.feature_names

    central_by_seed: dict[int, pd.Series] = {}
    fed_by_seed: dict[str, dict[int, pd.Series]] = {a: {} for a in FED_ARMS}
    per_client_store: dict[str, dict] = {}
    het_store, fidelity_rows = {}, []

    for seed in seeds:
        set_global_seed(seed)
        log.info("\n%s seed %d %s", "=" * 24, seed, "=" * 24)

        ckpt = models_dir / f"centralised_seed{seed}.pt"
        if not ckpt.exists():
            log.warning("missing %s, skipping seed", ckpt.name)
            continue

        factory = load_centralised(ckpt, bundle, cfg)
        attrs = attribute_all_clients(
            factory, clients, names, which="test",
            baseline_strategy=args.baseline, max_rows=args.max_rows, seed=seed)
        central = agg.aggregate_attributions(attrs)
        central_by_seed[seed] = central
        log.info("  centralised top 5: %s", ", ".join(central.head(5).index))

        if seed == seeds[0]:
            het_store = agg.heterogeneity(attrs).round(6).reset_index().rename(
                columns={"index": "feature"}).to_dict(orient="records")
            per_client_store["centralised"] = {
                n: agg.normalise(a.mean_abs()).round(6).to_dict()
                for n, a in attrs.items()}
            biggest = max(attrs, key=lambda n: attrs[n].n_samples)
            fidelity_rows.append({
                "arm": "centralised",
                **fid.fidelity_report(factory(biggest), clients.test[biggest],
                                      central, names, seed=seed)})

        for arm in FED_ARMS:
            ckpt = models_dir / f"{arm}_seed{seed}.pt"
            if not ckpt.exists():
                log.warning("missing %s, skipping", ckpt.name)
                continue
            factory = load_federated(ckpt, bundle, cfg)
            attrs = attribute_all_clients(
                factory, clients, names, which="test",
                baseline_strategy=args.baseline, max_rows=args.max_rows, seed=seed)
            series = agg.aggregate_attributions(attrs)
            fed_by_seed[arm][seed] = series
            tau, _ = agr.kendall_tau(central, series)
            log.info("  %-8s top 5: %-52s tau vs central %+.3f", arm,
                     ", ".join(series.head(5).index), tau)
            if seed == seeds[0]:
                per_client_store[arm] = {
                    n: agg.normalise(a.mean_abs()).round(6).to_dict()
                    for n, a in attrs.items()}
                fidelity_rows.append({
                    "arm": arm,
                    **fid.fidelity_report(factory(biggest), clients.test[biggest],
                                          series, names, seed=seed)})

    if not central_by_seed:
        log.error("no centralised checkpoints found. Run 03_train_all.py first.")
        return 1

    report = agr.agreement_report(central_by_seed,
                                  {a: v for a, v in fed_by_seed.items() if v},
                                  k=args.topk)
    log.info("\n=== agreement, centralised vs federated ===\n%s",
             report.round(4).to_string(index=False))

    ctrl = report[report.comparison.str.contains("seed control")]
    if len(ctrl):
        c = float(ctrl["kendall_tau"].iloc[0])
        log.info("\n  seed-to-seed control tau = %.3f", c)
        for _, r in report[~report.comparison.str.contains("control")].iterrows():
            verdict = ("below the control, so federation costs fidelity"
                       if r["kendall_tau"] < c else
                       "at or above the control, so the difference is not "
                       "attributable to federation")
            log.info("  %-34s tau %+.3f  -> %s", r["comparison"],
                     r["kendall_tau"], verdict)

    mean_central = pd.concat(central_by_seed.values(), axis=1).mean(axis=1)
    check = agr.agronomic_check(mean_central.sort_values(ascending=False),
                                k=args.topk)
    log.info("\n=== agronomic plausibility ===")
    log.info("  verdict: %s", check["verdict"])
    log.info("  %s", check["reason"])
    log.info("  attribution share  monsoon %.3f  post-monsoon %.3f  covariates %.3f",
             check["share_monsoon"], check["share_post_monsoon"],
             check["share_covariates"])

    if fidelity_rows:
        fdf = pd.DataFrame(fidelity_rows)[
            ["arm", "client", "aopc_topk", "aopc_random", "ratio", "verdict"]]
        log.info("\n=== fidelity, top-k vs random deletion ===\n%s",
                 fdf.round(4).to_string(index=False))

    with open(results_dir / "agreement.json", "w") as fh:
        json.dump({"table": report.round(6).to_dict(orient="records"),
                   "agronomic_check": check,
                   "baseline_strategy": args.baseline,
                   "seeds": list(central_by_seed),
                   "note": "federated attributions use per-client baselines "
                           "because no client may access another's data; "
                           "centralised uses the same strategy for a fair "
                           "comparison"}, fh, indent=2)
    with open(results_dir / "attributions.json", "w") as fh:
        json.dump({"global": {"centralised": mean_central.round(6).to_dict(),
                              **{a: pd.concat(v.values(), axis=1).mean(axis=1)
                                 .round(6).to_dict()
                                 for a, v in fed_by_seed.items() if v}},
                   "per_client": per_client_store,
                   "feature_names": names}, fh, indent=2)
    with open(results_dir / "heterogeneity.json", "w") as fh:
        json.dump(het_store, fh, indent=2)
    with open(results_dir / "fidelity.json", "w") as fh:
        json.dump(fidelity_rows, fh, indent=2)

    log.info("\nwrote agreement.json, attributions.json, heterogeneity.json, "
             "fidelity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
