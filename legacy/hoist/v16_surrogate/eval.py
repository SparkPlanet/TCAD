from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from .circuit import CircuitSpec, SpiceRunner
from .data import SdatDataset, SdatStats
from .model import CircuitSurrogateV16, resolve_device


def flat_z_to_dense(circuit: CircuitSpec, flat_z: np.ndarray) -> np.ndarray:
    flat_z = np.asarray(flat_z, dtype=np.float64)
    if flat_z.ndim != 2 or flat_z.shape[1] != circuit.total_params:
        raise ValueError(
            f"flat_z must be [N, {circuit.total_params}], got {flat_z.shape}"
        )
    dense = np.zeros(
        (flat_z.shape[0], circuit.num_mos, circuit.max_n_params),
        dtype=np.float32,
    )
    offset = 0
    for mi, mos in enumerate(circuit.mos_list):
        n_params = mos.n_params
        dense[:, mi, :n_params] = flat_z[:, offset : offset + n_params]
        offset += n_params
    return dense


def predict_dense_pool(
    model: CircuitSurrogateV16,
    ds: SdatDataset,
    dense_z: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    resolved_device = resolve_device(device)
    if batch_size <= 0:
        dense_values = ds.num_mos * ds.max_n_params
        target_values = 2_000_000 if resolved_device.type == "cuda" else 500_000
        batch_size = int(np.clip(target_values // max(1, dense_values), 256, 65_536))
        print(f"auto eval_batch={batch_size}")

    mask = torch.zeros(1, ds.num_mos, ds.max_n_params, device=resolved_device)
    for m, n_params in enumerate(ds.mos_n_params.tolist()):
        mask[0, m, : int(n_params)] = 1.0
    type_ids = ds.type_ids.to(resolved_device).long()
    model = model.to(resolved_device).eval()

    preds = []
    with torch.inference_mode():
        for start in range(0, dense_z.shape[0], batch_size):
            chunk = torch.from_numpy(dense_z[start : start + batch_size]).to(
                resolved_device
            )
            cur = chunk.size(0)
            pred = model.forward_dense(
                chunk,
                mask.expand(cur, -1, -1),
                type_ids.unsqueeze(0).expand(cur, -1),
            )
            preds.append(pred.view(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def simulate_dense_pool(
    circuit: CircuitSpec,
    dense_z: np.ndarray,
    timeout_s: int,
) -> np.ndarray:
    runner = SpiceRunner(circuit, timeout_s=timeout_s)
    runner.write_models()
    measures = np.full(dense_z.shape[0], np.nan, dtype=np.float64)
    for i in range(dense_z.shape[0]):
        measure = runner.run_with_params(runner.zscore_to_params(dense_z[i]))
        if measure is not None:
            measures[i] = measure
    return measures


def top_indices(values: np.ndarray, k: int, objective: str) -> np.ndarray:
    k = min(k, len(values))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    utility = values if objective == "max" else -values
    return np.argsort(utility)[::-1][:k]


def finite_pool_metrics(
    pred_norm: np.ndarray,
    measures: np.ndarray,
    stats: SdatStats,
    objective: str,
    top_ks: Sequence[int],
    failure_threshold: float | None = None,
) -> dict[str, object]:
    valid = np.isfinite(measures)
    pred_norm = pred_norm[valid]
    measures = measures[valid]
    measure_norm = (measures - stats.nom_measure) / stats.dev_std
    err_norm = pred_norm - measure_norm
    pred_phys = pred_norm * stats.dev_std + stats.nom_measure
    metrics: dict[str, object] = {
        "valid_n": int(valid.sum()),
        "failed_n": int((~valid).sum()),
        "objective": objective,
        "mse_norm": float(np.mean(err_norm * err_norm)),
        "mae_norm": float(np.mean(np.abs(err_norm))),
        "mae_phys": float(np.mean(np.abs(pred_phys - measures))),
        "pearson": float(np.corrcoef(pred_norm, measure_norm)[0, 1])
        if len(measures) > 1
        else float("nan"),
        "spearman": float(spearmanr(pred_norm, measure_norm)[0])
        if len(measures) > 1
        else float("nan"),
        "topk": {},
    }
    if failure_threshold is not None:
        fail = measures >= failure_threshold if objective == "max" else measures <= failure_threshold
        metrics["failure_threshold"] = failure_threshold
        metrics["failure_rate"] = float(np.mean(fail))

    for k in top_ks:
        pred_top = top_indices(pred_norm, k, objective)
        true_top = top_indices(measures, k, objective)
        overlap = len(set(pred_top.tolist()) & set(true_top.tolist()))
        selected_true = measures[pred_top]
        true_best = float(measures[true_top[0]]) if len(true_top) else float("nan")
        selected_best = (
            float(selected_true[top_indices(selected_true, 1, objective)[0]])
            if len(selected_true)
            else float("nan")
        )
        row: dict[str, float] = {
            "recall": float(overlap / max(1, len(true_top))),
            "overlap": float(overlap),
            "selected_true_mean": float(np.mean(selected_true)),
            "selected_true_best": selected_best,
            "golden_best": true_best,
        }
        if failure_threshold is not None:
            selected_fail = (
                selected_true >= failure_threshold
                if objective == "max"
                else selected_true <= failure_threshold
            )
            base_rate = float(metrics["failure_rate"])
            selected_rate = float(np.mean(selected_fail))
            row["selected_failure_rate"] = selected_rate
            row["enrichment"] = selected_rate / max(base_rate, 1e-12)
        metrics["topk"][str(k)] = row
    return metrics


def discovery_metrics(
    selected: Sequence[dict[str, float]],
    random_rows: Sequence[dict[str, float]],
    objective: str,
    failure_threshold: float | None,
) -> dict[str, object]:
    selected_summary = summarize_discovery_rows(selected, objective, failure_threshold)
    random_summary = summarize_discovery_rows(random_rows, objective, failure_threshold)
    out: dict[str, object] = {
        "objective": objective,
        "selected": selected_summary,
        "random": random_summary,
    }
    if failure_threshold is not None:
        sel_rate = float(selected_summary.get("failure_rate", 0.0))
        rnd_rate = float(random_summary.get("failure_rate", 0.0))
        out["failure_enrichment"] = sel_rate / max(rnd_rate, 1e-12)
    return out


def summarize_discovery_rows(
    rows: Sequence[dict[str, float]],
    objective: str,
    failure_threshold: float | None = None,
) -> dict[str, float]:
    measures = np.asarray([r["measure"] for r in rows], dtype=np.float64)
    if len(measures) == 0:
        return {"n": 0.0}
    utility = measures if objective == "max" else -measures
    out = {
        "n": float(len(measures)),
        "best": float(measures[int(np.argmax(utility))]),
        "mean": float(np.mean(measures)),
        "p90_utility": float(np.quantile(utility, 0.9)),
        "p99_utility": float(np.quantile(utility, 0.99)),
    }
    if failure_threshold is not None:
        fail = (
            measures >= failure_threshold
            if objective == "max"
            else measures <= failure_threshold
        )
        out["failure_count"] = float(np.count_nonzero(fail))
        out["failure_rate"] = float(np.mean(fail))
    return out


def estimate_pf_from_tail_rows(
    rows: Sequence[dict[str, float | str]],
    objective: str,
    threshold: float,
    predict_n: int,
) -> dict[str, float]:
    """Estimate Pf / yield from a fixed threshold on validated tail rows.

    This is a calibrated tail-rank estimate: the validated tail rows come from
    the surrogate-selected top region of a fixed Monte Carlo pool, and the count
    of rows crossing the fixed threshold is extrapolated to the full pool size.
    """
    if predict_n <= 0:
        raise ValueError("predict_n must be positive")
    measures = np.asarray(
        [float(r["measure"]) for r in rows if np.isfinite(float(r["measure"]))],
        dtype=np.float64,
    )
    if measures.size == 0:
        raise ValueError("cannot estimate Pf: no valid SPICE rows")
    fail = measures >= threshold if objective == "max" else measures <= threshold
    fail_count = int(np.count_nonzero(fail))
    pf = float(fail_count / predict_n)
    se = float(np.sqrt(max(pf * max(1.0 - pf, 0.0), 0.0) / predict_n))
    fom = float(se / pf) if pf > 0.0 else float("inf")
    return {
        "threshold": float(threshold),
        "predict_n": int(predict_n),
        "valid_spice": int(measures.size),
        "failed_spice": int(len(rows) - measures.size),
        "tail_failure_count": fail_count,
        "failure_probability": pf,
        "yield": float(1.0 - pf),
        "standard_error": se,
        "fom": fom,
        "objective": objective,
    }


def write_metrics_log(
    path: str,
    metrics: dict[str, object],
    sections: Sequence[str] = (),
) -> None:
    lines = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{key}:\n")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    lines.append(f"  {sub_key}:\n")
                    for leaf_key, leaf_value in sub_value.items():
                        lines.append(f"    {leaf_key}: {leaf_value}\n")
                else:
                    lines.append(f"  {sub_key}: {sub_value}\n")
        else:
            lines.append(f"{key}: {value}\n")
    for section in sections:
        lines.append("\n")
        lines.append(section.rstrip())
        lines.append("\n")
    from pathlib import Path

    Path(path).write_text("".join(lines), encoding="utf-8")
