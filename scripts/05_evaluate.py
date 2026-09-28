"""Stages F and G. Robustness evaluation, significance, then the results freeze.

    python scripts\\05_evaluate.py                 # all seeds from config
    python scripts\\05_evaluate.py --seeds 42      # quick check (no CIs)
    python scripts\\05_evaluate.py --validate-only # re-run Gate G alone

Requires checkpoints from 03_train_all.py (normal AND --protocol
climate_stress, for every seed) and results from 04_explain.py.

The script ends with Gate G. If validation fails, it exits non-zero and the
backend must not be started: a partially written contract is worse than an
absent one, because the API would come up and serve incomplete data silently.

Writes:
    artifacts/results/ood.json          drought robustness, mean and CI over seeds
    artifacts/results/scenarios.json    3x3 climate grid, mean and sd over seeds
    artifacts/results/predictions.json  per-row test predictions, every arm
    artifacts/results/significance.json skill CIs + Wilcoxon on the test set
    artifacts/results/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedcrop import features, io_layer, metrics, splits  # noqa: E402
from fedcrop.config import load_config, set_global_seed  # noqa: E402
from fedcrop.datasets import prepare_clients  # noqa: E402
from fedcrop.evaluation import ood as ood_mod  # noqa: E402
from fedcrop.evaluation import perturbation as pert_mod  # noqa: E402
from fedcrop.evaluation import significance as sig_mod  # noqa: E402
from fedcrop.export import results_json as exp  # noqa: E402
from fedcrop.scaling import inverse_transform_target  # noqa: E402
from fedcrop.splits import ClientData  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("evaluate")

ARMS = ["centralised", "local", "fedavg", "fedprox", "fedper"]
# Full 3x3 grid: temperature {0, +1, +2} C  x  rainfall {-20, 0, +20} %.
GRID = [(dt, rp) for dt in (0.0, 1.0, 2.0) for rp in (-20.0, 0.0, 20.0)]


# --------------------------------------------------------------------- helpers
def boot_ci(values, n_boot: int = 10_000, seed: int = 0):
    """Mean and bootstrap 95% CI over seeds. With fewer than 2 values the
    interval is undefined, so it comes back as NaN rather than a fake zero."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan
    if len(v) < 2:
        return float(v.mean()), np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(v.std(ddof=1)),
            float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def raw_client_data(bundle, cfg, mask) -> dict[str, ClientData]:
    """Unscaled per-client data. Perturbation must happen before scaling."""
    states = bundle.keys[cfg.state_key].to_numpy()
    out = {}
    for name in sorted(set(states[mask])):
        sel = mask & (states == name)
        out[name] = ClientData(name, bundle.x_seq[sel], bundle.x_cov[sel],
                               bundle.y[sel],
                               bundle.keys[sel].reset_index(drop=True))
    return out


def _usable_pool(clients, which: str) -> dict[str, ClientData]:
    """Clients on a split that are non-empty AND have a trained scaler.
    A client with test rows but no training rows has no scaler and no model."""
    pool = getattr(clients, which)
    return {n: d for n, d in pool.items() if len(d) and n in clients.scalers}


def per_client_predictions(arm_file: str, bundle, cfg, clients, models_dir,
                           seed: int, which: str):
    """{client: (actual_kg, predicted_kg)} for one checkpoint, or None."""
    ckpt = models_dir / f"{arm_file}_seed{seed}.pt"
    if not ckpt.exists():
        return None
    pool = _usable_pool(clients, which)
    preds = ood_mod._predict_arm(arm_file, ckpt, bundle, cfg, pool)
    out = {}
    for name, p_scaled in preds.items():
        s = clients.scalers[name]
        out[name] = (inverse_transform_target(pool[name].y, s),
                     inverse_transform_target(p_scaled, s))
    return out


def normal_per_client(bundle, cfg, clients, models_dir, seed,
                      arms=None, which="test") -> pd.DataFrame:
    """Per-client metrics per arm on a chosen split."""
    rows = []
    for arm in arms or ARMS:
        res = per_client_predictions(arm, bundle, cfg, clients, models_dir,
                                     seed, which)
        if res is None:
            continue
        for name, (y, p) in res.items():
            m = metrics.regression_metrics(y, p)
            m.update({"arm": arm, "client": name, "seed": seed})
            rows.append(m)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--stress-tag", default="_stress",
                    help="checkpoint suffix for climate-stress-trained models")
    ap.add_argument("--scenario-arm", default="fedper",
                    choices=ARMS, help="arm used for the counterfactual grid")
    args = ap.parse_args()

    cfg = load_config()
    results_dir, models_dir = cfg.path("results"), cfg.path("models")

    if args.validate_only:
        ok, _ = exp.validate_exports(results_dir, strict=False)
        return 0 if ok else 1

    seeds = list(args.seeds if args.seeds is not None else cfg["run"]["seeds"])
    set_global_seed(seeds[0])

    clean_path = cfg.path("interim").with_name("clean.parquet")
    if not clean_path.exists():
        log.error("no cleaned panel. Run scripts/01_build_features.py first.")
        return 1
    clean = io_layer.load_interim(clean_path)
    bundle = features.assemble(clean, cfg)
    sp = splits.temporal_split(bundle, cfg)
    clients = prepare_clients(bundle, sp, cfg)

    # ---------------------------------------------------------- Stage F: OOD
    log.info("\n=== out-of-distribution: each client's own deficit years ===")
    deficit = splits.per_client_deficit_years(clean, cfg)
    n_years = {y for v in deficit.values() for y in v}
    log.info("  %d clients, %d distinct deficit years across the federation",
             len(deficit), len(n_years))

    tag = args.stress_tag
    stress_seeds = [s for s in seeds
                    if (models_dir / f"centralised{tag}_seed{s}.pt").exists()]
    missing = [s for s in seeds if s not in stress_seeds]
    if not stress_seeds:
        log.error(
            "\n  No climate-stress checkpoints found for seeds %s.\n"
            "  The OOD test requires models trained WITHOUT each client's\n"
            "  deficit years. Run this first:\n"
            "      python scripts\\03_train_all.py --protocol climate_stress\n",
            seeds)
        return 1
    if missing:
        log.warning("  WARNING: no stress checkpoints for seeds %s; OOD uses %s only",
                    missing, stress_seeds)

    # The stress split and its scalers do not depend on the seed.
    stress_sp = splits.climate_stress_split(bundle, cfg, deficit)
    stress_clients = prepare_clients(bundle, stress_sp, cfg)
    ood_clients = {n: stress_clients.test[n] for n in stress_clients.names
                   if len(stress_clients.test[n]) and n in stress_clients.scalers}
    stress_arms = [f"{a}{tag}" for a in ARMS]
    log.info("  reference: the SAME stress-trained models on their own "
             "held-out NON-deficit rows.")

    degr_all, summ_all, ood_all = [], [], []
    for seed in stress_seeds:
        ood_rows = ood_mod.ood_metrics(bundle, cfg, ood_clients,
                                       stress_clients.scalers, models_dir,
                                       stress_arms, seed)
        normal_rows = normal_per_client(bundle, cfg, stress_clients, models_dir,
                                        seed, arms=stress_arms, which="val")
        if ood_rows.empty or normal_rows.empty:
            log.warning("  seed %d: incomplete stress checkpoints, skipped", seed)
            continue
        for df in (ood_rows, normal_rows):
            df["arm"] = df["arm"].str.replace(tag, "", regex=False)
        normal_rows = normal_rows.drop(columns=["seed"], errors="ignore")
        ood_rows = ood_rows.drop(columns=["seed"], errors="ignore")

        degradation = ood_mod.degradation_table(normal_rows, ood_rows)
        summary = ood_mod.summarise_degradation(degradation).reset_index()
        degradation["seed"] = seed
        summary["seed"] = seed
        ood_rows["seed"] = seed
        degr_all.append(degradation)
        summ_all.append(summary)
        ood_all.append(ood_rows)
        log.info("  seed %d: mean R2 drop  %s", seed, "  ".join(
            f"{r.arm} {r.mean_r2_drop:+.3f}" for r in summary.itertuples()))

    if not summ_all:
        log.error("no usable stress checkpoints. Run 03_train_all.py "
                  "--protocol climate_stress first.")
        return 1

    degradation = pd.concat(degr_all, ignore_index=True)
    per_seed_summary = pd.concat(summ_all, ignore_index=True)
    ood_rows = pd.concat(ood_all, ignore_index=True)

    num_cols = [c for c in per_seed_summary.columns
                if c not in ("arm", "seed")
                and pd.api.types.is_numeric_dtype(per_seed_summary[c])]
    summary = per_seed_summary.groupby("arm")[num_cols].mean()
    for arm, grp in per_seed_summary.groupby("arm"):
        mean, sd, lo, hi = boot_ci(grp["mean_r2_drop"])
        summary.loc[arm, "mean_r2_drop_sd"] = sd
        summary.loc[arm, "mean_r2_drop_ci_lo"] = lo
        summary.loc[arm, "mean_r2_drop_ci_hi"] = hi
        summary.loc[arm, "n_seeds"] = grp["seed"].nunique()
    summary = summary.sort_values("mean_r2_drop")
    summary.index.name = "arm"

    show = ["mean_r2_drop", "mean_r2_drop_ci_lo", "mean_r2_drop_ci_hi",
            "median_r2_drop", "worst_client_r2_drop", "n_seeds"]
    log.info("\nR2 drop on each client's own deficit years, mean over %d seed(s):\n%s",
             len(summ_all), summary[[c for c in show if c in summary]].round(3).to_string())
    best = summary.index[0]
    log.info("\n  most robust under drought: %s (mean R2 drop %+.3f)",
             best, summary.loc[best, "mean_r2_drop"])
    log.info("  note: r2_drop is the metric to report. Absolute RMSE falls in "
             "deficit years for every arm because yield variance is lower.")
    if len(summ_all) < 2:
        log.info("  note: one seed only, so the CI columns are empty. "
                 "Run without --seeds for intervals.")

    # ---------------------------------------------- Stage F: perturbation grid
    log.info("\n=== counterfactual climate perturbations (%s), 3x3 grid ===",
             args.scenario_arm)
    raw = raw_client_data(bundle, cfg, sp.test)
    raw = {k: v for k, v in raw.items() if k in clients.scalers}
    scen_all = []
    for seed in seeds:
        ckpt = models_dir / f"{args.scenario_arm}_seed{seed}.pt"
        if not ckpt.exists():
            log.warning("  no %s checkpoint for seed %d, skipped", ckpt.name, seed)
            continue
        s = pert_mod.counterfactual_response(
            args.scenario_arm, ckpt, bundle, cfg, raw, clients.scalers, GRID)
        s["seed"] = seed
        scen_all.append(s)
    if not scen_all:
        log.error("no %s checkpoints for the scenario grid.", args.scenario_arm)
        return 1
    scen_seeds = pd.concat(scen_all, ignore_index=True)

    keys = ["arm", "client", "delta_temp", "rain_pct"]
    scenarios = (scen_seeds.groupby(keys)
                 .agg(mean_base_yield=("mean_base_yield", "mean"),
                      mean_pert_yield=("mean_pert_yield", "mean"),
                      mean_pct_change=("mean_pct_change", "mean"),
                      sd_pct_change=("mean_pct_change", "std"),
                      n_seeds=("seed", "nunique"))
                 .reset_index())

    # Guard against the dead-checkpoint failure seen before: a grid that
    # does not move at all means the model ignores its inputs.
    moving = scenarios.loc[(scenarios.delta_temp != 0) | (scenarios.rain_pct != 0),
                           "mean_pct_change"].abs()
    if moving.max() < 1e-3:
        log.error("\n  SCENARIO GRID IS FLAT (max |change| %.2e %%). The loaded "
                  "checkpoints ignore their inputs. Retrain with "
                  "python scripts\\03_train_all.py and rerun.", moving.max())
        return 1

    pivot = scenarios.pivot_table(index="client",
                                  columns=["delta_temp", "rain_pct"],
                                  values="mean_pct_change", observed=False)
    log.info("\nmean predicted yield change %%, mean of %d seed(s) "
             "(rows: client, cols: dT / rain%%)", len(scen_all))
    log.info("%s", pivot.round(2).to_string())
    overall = scenarios.groupby(["delta_temp", "rain_pct"])["mean_pct_change"].mean()
    log.info("\nfederation-wide mean response:\n%s", overall.round(3).to_string())

    # ------------------------------------------ Stage G: per-row predictions
    log.info("\n=== test-set predictions for the dashboard ===")
    pred_seed = seeds[0]
    tables = []
    for arm in ARMS:
        res = per_client_predictions(arm, bundle, cfg, clients, models_dir,
                                     pred_seed, "test")
        if res is None:
            log.warning("  no %s checkpoint for seed %d", arm, pred_seed)
            continue
        parts = []
        for name, (y, p) in res.items():
            k = clients.test[name].keys.reset_index(drop=True).copy()
            k["client"] = name
            k["row"] = np.arange(len(k))
            k["actual"] = np.asarray(y, dtype=float)
            k[f"pred_{arm}"] = np.asarray(p, dtype=float)
            parts.append(k)
        tables.append(pd.concat(parts, ignore_index=True))
    if not tables:
        log.error("no normal checkpoints for seed %d.", pred_seed)
        return 1
    preds = tables[0]
    for t in tables[1:]:
        pcol = [c for c in t.columns if c.startswith("pred_")]
        preds = preds.merge(t[["client", "row"] + pcol], on=["client", "row"],
                            how="outer", validate="one_to_one")
    preds = preds.drop(columns=["row"])
    arms_present = [c[5:] for c in preds.columns if c.startswith("pred_")]
    log.info("  %d test rows, %d clients, arms %s, seed %d",
             len(preds), preds["client"].nunique(), arms_present, pred_seed)

    # -------------------------------------------------- Stage F: significance
    log.info("\n=== significance ===")
    ablation_path = results_dir / "ablation.json"
    ablation_ci = pd.DataFrame()
    if ablation_path.exists():
        with open(ablation_path) as fh:
            runs = pd.DataFrame(json.load(fh)["runs"])
        ablation_ci = sig_mod.ablation_table(runs[runs.seed.notna()])
        cols = ["arm", "n_seeds", "skill_vs_trend_mean",
                "skill_vs_trend_ci_lo", "skill_vs_trend_ci_hi"]
        log.info("\nskill vs district_trend, bootstrap 95%% CI:\n%s",
                 ablation_ci[cols].round(4).to_string(index=False))

    # Wilcoxon on the models the ablation table actually reports: the normal
    # checkpoints on the 2012-2015 test set, per-client RMSE averaged over seeds.
    test_rows = pd.concat([normal_per_client(bundle, cfg, clients, models_dir, s,
                                             arms=ARMS, which="test")
                           for s in seeds], ignore_index=True)
    per_client_rmse = {arm: grp.groupby("client")["rmse"].mean()
                       for arm, grp in test_rows.groupby("arm")}
    pairwise = pd.DataFrame()
    if "centralised" in per_client_rmse:
        pairwise = sig_mod.pairwise_significance(per_client_rmse, "centralised")
        log.info("\nWilcoxon paired by client (test set, mean RMSE over %d seed(s)), "
                 "vs centralised:\n%s", len(seeds),
                 pairwise.round(4).to_string(index=False))

    # -------------------------------------------------------- Stage G: freeze
    exp.export_ood(degradation, summary.reset_index(), deficit, ood_rows,
                   results_dir / "ood.json",
                   per_seed_summary=per_seed_summary, seeds=stress_seeds)
    exp.export_scenarios(scenarios, GRID, results_dir / "scenarios.json",
                         per_seed=scen_seeds, arm=args.scenario_arm,
                         seeds=sorted(scen_seeds["seed"].unique().tolist()))
    exp.export_predictions(preds, arms_present, pred_seed,
                           results_dir / "predictions.json")
    exp.export_significance(ablation_ci, pairwise,
                            results_dir / "significance.json")
    exp.write_json(exp.manifest(results_dir), results_dir / "manifest.json")

    ok, problems = exp.validate_exports(results_dir, strict=False)
    if not ok:
        log.error("\n  %d problem(s). Fix these before Stage H.", len(problems))
        return 1
    log.info("\n  Stages F and G complete. The backend may now be built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())