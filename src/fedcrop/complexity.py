"""Complexity analysis.

Three axes an examiner is likely to ask about:
  space         parameter count and payload size
  communication bytes per round and cumulative
  time          wall clock, and asymptotic cost per round

The asymptotic statement for one federated round is
    O(R * K * E * N_k * T * H^2)
for R rounds, K clients, E local epochs, N_k rows per client, T timesteps and
H hidden units, because a GRU step is O(H^2) in its recurrent weights.
Communication is O(R * K * P) for P shared parameters, independent of data size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .model import YieldGRU, count_parameters


def parameter_breakdown(model: YieldGRU) -> pd.DataFrame:
    rows = []
    for name, p in model.named_parameters():
        rows.append({"parameter": name, "shape": tuple(p.shape),
                     "count": int(p.numel())})
    df = pd.DataFrame(rows)
    df["share_pct"] = (df["count"] / df["count"].sum() * 100).round(1)
    return df


def model_complexity(model: YieldGRU, seq_len: int = 12) -> dict:
    base = model.base_parameter_keys()
    head = model.head_parameter_keys()
    sd = model.state_dict()
    n_base = int(sum(sd[k].numel() for k in base))
    n_head = int(sum(sd[k].numel() for k in head))
    total = count_parameters(model)
    bytes_per = int(sum(v.numel() * v.element_size() for v in sd.values()))
    h = model.hidden
    return {
        "total_parameters": total,
        "base_parameters": n_base,
        "head_parameters": n_head,
        "fedper_shared_pct": round(100 * n_base / max(total, 1), 1),
        "state_dict_bytes": bytes_per,
        "state_dict_kb": round(bytes_per / 1024, 1),
        "gru_flops_per_sample": int(6 * seq_len * h * h),
        "forward_cost": f"O(T*H^2) = O({seq_len}*{h}^2)",
    }


def communication_table(results: dict) -> pd.DataFrame:
    """Per-round and cumulative payload for each federated run."""
    rows = []
    for algo, res in results.items():
        per_round = res.bytes_per_round
        rows.append({
            "algorithm": algo,
            "rounds_run": res.rounds_run,
            "bytes_per_round_all_clients": per_round,
            "kb_per_round": round(per_round / 1024, 1),
            "total_mb": round(per_round * res.rounds_run / 1024 / 1024, 2),
            "wall_clock_s": round(res.wall_clock_s, 1),
            "s_per_round": round(res.wall_clock_s / max(res.rounds_run, 1), 2),
        })
    return pd.DataFrame(rows)


def measure_inference_latency(model: YieldGRU, seq_len: int = 12,
                              n_cov: int = 9, n: int = 200) -> dict:
    import time
    x_seq = torch.randn(n, seq_len, model.gru.input_size)
    x_cov = torch.randn(n, n_cov)
    model.eval()
    with torch.no_grad():
        model(x_seq[:1], x_cov[:1])
        start = time.time()
        model(x_seq, x_cov)
        batch_s = time.time() - start
    return {
        "batch_size": n,
        "batch_seconds": round(batch_s, 4),
        "ms_per_sample": round(batch_s / n * 1000, 4),
    }


def complexity_report(model: YieldGRU, fed_results: dict | None = None,
                      seq_len: int = 12) -> dict:
    out = {
        "model": model_complexity(model, seq_len),
        "parameters": parameter_breakdown(model).to_dict(orient="records"),
        "inference": measure_inference_latency(model, seq_len),
        "asymptotics": {
            "training_per_round": "O(K * E * N_k * T * H^2)",
            "communication_per_round": "O(K * P)",
            "total_training": "O(R * K * E * N_k * T * H^2)",
            "note": "communication is independent of dataset size, which is the "
                    "argument for a compact recurrent model over a transformer",
        },
    }
    if fed_results:
        out["communication"] = communication_table(fed_results).to_dict(orient="records")
    return out
