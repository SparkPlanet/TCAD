from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from .circuit import CircuitTxtParser, SpiceRunner
from .common import DEFAULT_CHECKPOINT, DEFAULT_LOG_DIR, DEFAULT_PRED_DIR, DEFAULT_TRAIN_DATA
from .data import SdatDataset, SdatStats, SobolZSampler, SpiceDatasetBuilder
from .eval import (
    discovery_metrics,
    finite_pool_metrics,
    flat_z_to_dense,
    predict_dense_pool,
    simulate_dense_pool,
    summarize_discovery_rows,
    top_indices,
    write_metrics_log,
)
from .mc import MCTopKPredictor, ResidualCorrector, SpiceValidator, TopEntry
from .model import (
    CircuitSurrogateV16,
    FUSION_MODES,
    MODEL_ARCHITECTURES,
    TrainConfig,
    V16Trainer,
    build_surrogate_model,
    resolve_device,
)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def artifact_stem(circuit_name: str) -> str:
    return circuit_name.replace("/", "_").replace(" ", "_")


def add_model_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-architecture",
        choices=sorted(MODEL_ARCHITECTURES),
        default="v16",
        help="Surrogate architecture used for training.",
    )
    parser.add_argument(
        "--no-type-embedding",
        dest="use_type_embedding",
        action="store_false",
        help="Ablation: remove MOS type embedding from the structured v16 encoder.",
    )
    parser.add_argument(
        "--no-global-context",
        dest="use_global_context",
        action="store_false",
        help="Ablation: remove the global context MLP from the structured v16 encoder.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=sorted(FUSION_MODES),
        default="multiplicative",
        help=(
            "Structured fusion: multiplicative uses [h,g,h*g], additive uses h+g, "
            "and concat uses [h,g]."
        ),
    )
    parser.set_defaults(use_type_embedding=True, use_global_context=True)


def dataset_stats(ds: SdatDataset) -> SdatStats:
    return SdatStats(
        nom_measure=float(ds.stats["nom_measure"]),
        dev_std=float(ds.stats["dev_std"]),
        multiplier=float(ds.stats["multiplier"]),
        y_mean=float(ds.stats["y_mean"]),
        y_std=float(ds.stats["y_std"]),
        num_mos=ds.num_mos,
        num_mos_types=ds.num_mos_types,
        max_n_params=ds.max_n_params,
    )


def save_predictions(
    entries: Sequence[TopEntry],
    stats: SdatStats,
    output_path: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(
        [entry.score_norm * stats.dev_std + stats.nom_measure for entry in entries]
    )
    z_path = output.with_suffix(".topz.npy")
    np.save(output, values)
    np.save(z_path, np.stack([entry.z for entry in entries]))
    print(f"wrote {output} and {z_path}")
    return output, z_path


def validate_predictions(
    entries: Sequence[TopEntry],
    circuit_path: str | Path,
    stats: SdatStats,
    top_k: int,
    timeout_s: int,
    objective: str,
    log_path: str | Path,
    calibration_results: Sequence[dict[str, float | str]] | None = None,
    data_path: str | Path | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    circuit = CircuitTxtParser.parse(circuit_path)
    validator = SpiceValidator(
        circuit,
        stats,
        timeout_s=timeout_s,
        objective=objective,
    )
    if data_path:
        results = validator.validate_dataset(
            entries,
            min(top_k, len(entries)),
            data_path,
            metadata=metadata,
        )
    else:
        results = validator.validate(entries, min(top_k, len(entries)))
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    validator.report(
        results,
        log,
        calibration_results=calibration_results or (),
    )


def parse_mc_batch(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return 0
    batch = int(text)
    if batch <= 0:
        raise argparse.ArgumentTypeError("--mc-batch must be positive or 'auto'")
    return batch


def resolve_objective(value: str, circuit_objective: str = "max") -> str:
    objective = circuit_objective if value == "auto" else value
    if objective not in {"max", "min"}:
        raise ValueError("objective must be 'auto', 'max', or 'min'")
    return objective


def _checkpoint_state_and_config(checkpoint: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint state_dict must be a dict")
        config = checkpoint.get("model_config", {})
        return state, dict(config) if isinstance(config, dict) else {}
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    raise TypeError("unsupported checkpoint format")


def make_model_from_sdat(
    path: str | Path,
    ckpt_path: str | Path,
    device: str,
    *,
    model_architecture: str = "v16",
    use_type_embedding: bool = True,
    use_global_context: bool = True,
    fusion_mode: str = "multiplicative",
) -> Tuple[CircuitSurrogateV16, SdatDataset]:
    resolved_device = resolve_device(device)
    ds = SdatDataset(path)
    checkpoint = torch.load(ckpt_path, map_location=resolved_device, weights_only=True)
    state, saved_config = _checkpoint_state_and_config(checkpoint)
    model = build_surrogate_model(
        max_n_params=ds.max_n_params,
        num_mos=ds.num_mos,
        num_mos_types=ds.num_mos_types,
        mos_n_params=ds.mos_n_params.tolist(),
        model_architecture=str(saved_config.get("model_architecture", model_architecture)),
        hidden=(
            int(saved_config["hidden"])
            if "hidden" in saved_config and saved_config["hidden"] not in (None, "")
            else None
        ),
        use_type_embedding=bool(saved_config.get("use_type_embedding", use_type_embedding)),
        use_global_context=bool(saved_config.get("use_global_context", use_global_context)),
        fusion_mode=str(saved_config.get("fusion_mode", fusion_mode)),
    ).to(resolved_device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint does not match the requested model: {ckpt_path}"
        ) from exc
    model.eval()
    return model, ds


def command_build_data(args: argparse.Namespace) -> None:
    circuit = CircuitTxtParser.parse(args.circuit)
    if args.train_z_scale <= 0:
        raise ValueError("train_z_scale must be positive")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    samples = SobolZSampler.sample(
        n_samples=args.train_n,
        dim=circuit.total_params,
        sigma_range=args.sigma_range,
        seed=args.seed,
    ) * args.train_z_scale
    if args.samples_out:
        SobolZSampler.write_binary(samples, args.samples_out)
    stats = SpiceDatasetBuilder(circuit, timeout_s=args.spice_timeout).build_from_flat_z(
        samples,
        args.output,
        allow_failed_nominal=args.allow_failed_nominal,
    )
    print(
        f"wrote {args.output}: nom={stats.nom_measure:.6e}, "
        f"dev_std={stats.dev_std:.6e}, y_std={stats.y_std:.4f}"
    )


def command_train(args: argparse.Namespace) -> None:
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    ds = SdatDataset(args.train)
    threshold_norm = None
    if args.threshold is not None:
        threshold_norm = (
            float(args.threshold) - float(ds.stats["nom_measure"])
        ) / float(ds.stats["dev_std"])
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        val_ratio=args.val_ratio,
        objective=args.objective,
        model_architecture=args.model_architecture,
        use_type_embedding=args.use_type_embedding,
        use_global_context=args.use_global_context,
        fusion_mode=args.fusion_mode,
        threshold_norm=threshold_norm,
        tail_fail_weight=args.tail_fail_weight,
        boundary_weight=args.boundary_weight,
        boundary_sigma=args.boundary_sigma,
    )
    V16Trainer(config).train(args.train, args.ckpt, val_path=args.val or None)


def command_sdat_eval(args: argparse.Namespace) -> None:
    start = time.time()
    circuit = CircuitTxtParser.parse(args.circuit)
    objective = resolve_objective(args.objective, circuit.objective)
    model, train_ds = make_model_from_sdat(args.train, args.ckpt, args.device)
    eval_ds = SdatDataset(args.eval)
    if (
        eval_ds.num_mos != train_ds.num_mos
        or eval_ds.max_n_params != train_ds.max_n_params
        or not torch.equal(eval_ds.type_ids, train_ds.type_ids)
        or not torch.equal(eval_ds.mos_n_params, train_ds.mos_n_params)
    ):
        raise ValueError("evaluation .sdat circuit structure does not match training data")

    dense_z = eval_ds.all_params.numpy()
    measures = (
        eval_ds.all_labels.numpy().astype(np.float64)
        * float(eval_ds.stats["dev_std"])
        + float(eval_ds.stats["nom_measure"])
    )
    pred_norm = predict_dense_pool(
        model,
        train_ds,
        dense_z,
        device=args.device,
        batch_size=args.eval_batch,
    )
    metrics = finite_pool_metrics(
        pred_norm=pred_norm,
        measures=measures,
        stats=dataset_stats(train_ds),
        objective=objective,
        top_ks=args.top_ks,
    )
    metrics.update(
        {
            "circuit": circuit.name,
            "split": args.split,
            "sampling": "independent_scrambled_sobol",
            "train": str(args.train),
            "eval": str(args.eval),
            "ckpt": str(args.ckpt),
            "elapsed_seconds": time.time() - start,
        }
    )
    sections = finite_golden_detail_sections(
        pred_norm=pred_norm,
        measures=measures,
        stats=dataset_stats(train_ds),
        objective=objective,
        top_ks=args.detail_top_ks,
    )
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    write_metrics_log(args.log, metrics, sections=sections)
    print(
        f"{args.split}: n={metrics['valid_n']} mse={metrics['mse_norm']:.6e} "
        f"pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}"
    )
    print(f"sdat-eval log: {args.log}")


def command_predict(args: argparse.Namespace) -> None:
    model, ds = make_model_from_sdat(args.train, args.ckpt, args.device)
    predictor = MCTopKPredictor(
        model,
        ds.type_ids,
        ds.mos_n_params,
        device=args.device,
        objective=args.objective,
        residual_model_path=args.residual_model_path or None,
    )
    entries = predictor.predict_topk(
        n_mc=args.predict_n,
        top_k=args.top_k,
        batch_size=parse_mc_batch(args.mc_batch),
        seed=args.seed,
    )
    stats = dataset_stats(ds)
    output_path, _ = save_predictions(entries, stats, args.output)

    if args.validate_top_k > 0:
        if not args.circuit:
            raise ValueError("--circuit is required when --validate-top-k > 0")
        if args.validation_log:
            log_path = Path(args.validation_log)
        else:
            log_path = DEFAULT_LOG_DIR / (
                f"{output_path.stem}_validate_{args.predict_n}.log"
            )
        validate_predictions(
            entries,
            circuit_path=args.circuit,
            stats=stats,
            top_k=args.validate_top_k,
            timeout_s=args.spice_timeout,
            objective=args.objective,
            log_path=log_path,
            data_path=args.validation_data or None,
            metadata={
                "predict_n": args.predict_n,
                "top_k": args.validate_top_k,
                "seed": args.seed,
                "train": str(args.train),
                "ckpt": str(args.ckpt),
                "sampling": "standard_normal_mc_surrogate_screened",
                "reference_type": "surrogate_screened_tail_not_crude_mc",
            },
        )


def command_tail_threshold(args: argparse.Namespace) -> None:
    loaded = np.load(args.data, allow_pickle=False)
    measures = np.asarray(loaded["measure"], dtype=np.float64)
    valid = np.isfinite(measures)
    if not np.any(valid):
        raise ValueError("tail reference contains no valid SPICE measures")
    objective = (
        args.objective
        if args.objective != "auto"
        else str(np.asarray(loaded["objective"]).item())
    )
    values = measures[valid]
    ordered = np.sort(values)
    if objective == "max":
        ordered = ordered[::-1]

    print(f"data: {args.data}")
    print(f"reference_type: surrogate_screened_tail_not_crude_mc")
    print(f"objective: {objective}")
    print(f"valid_spice: {len(values)}")
    print(f"failed_spice: {len(measures) - len(values)}")
    if args.target_probability:
        predict_n = int(
            args.predict_n
            or np.asarray(loaded.get("meta_predict_n", 0)).item()
        )
        if predict_n <= 0:
            raise ValueError(
                "--predict-n is required when metadata has no predict_n"
            )
        for probability in args.target_probability:
            rank = max(1, int(np.ceil(probability * predict_n)))
            if rank > len(ordered):
                print(
                    f"target_probability={probability:.6e}: rank={rank} "
                    f"unavailable (only {len(ordered)} validated candidates)"
                )
                continue
            print(
                f"target_probability={probability:.6e}: rank={rank} "
                f"threshold={ordered[rank - 1]:.12e}"
            )
    if args.threshold is not None:
        failures = (
            values >= args.threshold
            if objective == "max"
            else values <= args.threshold
        )
        predict_n = int(
            args.predict_n
            or np.asarray(loaded.get("meta_predict_n", 0)).item()
        )
        print(f"threshold: {args.threshold:.12e}")
        print(f"validated_tail_failures: {int(np.sum(failures))}")
        if predict_n > 0:
            print(
                "screened_failure_probability: "
                f"{np.sum(failures) / predict_n:.12e}"
            )


def command_finite_golden(args: argparse.Namespace) -> None:
    start = time.time()
    circuit = CircuitTxtParser.parse(args.circuit)
    objective = resolve_objective(args.objective, circuit.objective)
    model, ds = make_model_from_sdat(args.train, args.ckpt, args.device)
    stats = dataset_stats(ds)

    rng = np.random.default_rng(args.seed)
    if args.pool_z:
        flat_z = np.load(args.pool_z)
    else:
        flat_z = rng.standard_normal(
            (args.pool_n, circuit.total_params),
            dtype=np.float64,
        )
    dense_z = flat_z_to_dense(circuit, flat_z)

    if args.golden_npz and Path(args.golden_npz).exists() and not args.force_spice:
        loaded = np.load(args.golden_npz)
        measures = loaded["measure"].astype(np.float64)
        print(f"loaded finite golden: {args.golden_npz}")
    else:
        measures = simulate_dense_pool(circuit, dense_z, args.spice_timeout)
        if args.golden_npz:
            Path(args.golden_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.golden_npz,
                z=flat_z.astype(np.float32),
                measure=measures,
            )
            print(f"wrote finite golden: {args.golden_npz}")

    pred_norm = predict_dense_pool(
        model,
        ds,
        dense_z,
        device=args.device,
        batch_size=args.eval_batch,
    )
    metrics = finite_pool_metrics(
        pred_norm=pred_norm,
        measures=measures,
        stats=stats,
        objective=objective,
        top_ks=args.top_ks,
        failure_threshold=args.failure_threshold,
    )
    metrics.update(
        {
            "circuit": circuit.name,
            "pool_n": int(flat_z.shape[0]),
            "train": str(args.train),
            "ckpt": str(args.ckpt),
            "elapsed_seconds": time.time() - start,
            "elapsed_minutes": (time.time() - start) / 60.0,
        }
    )
    sections = finite_golden_detail_sections(
        pred_norm=pred_norm,
        measures=measures,
        stats=stats,
        objective=objective,
        top_ks=args.detail_top_ks,
    )
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_log(str(log_path), metrics, sections=sections)
    print(f"finite-golden log: {log_path}")
    print(
        f"valid={metrics['valid_n']} failed={metrics['failed_n']} "
        f"pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}"
    )
    print(f"elapsed={metrics['elapsed_minutes']:.2f} min")


def command_tail_discovery(args: argparse.Namespace) -> None:
    start = time.time()
    circuit = CircuitTxtParser.parse(args.circuit)
    objective = resolve_objective(args.objective, circuit.objective)
    model, ds = make_model_from_sdat(args.train, args.ckpt, args.device)
    stats = dataset_stats(ds)
    first_pass_top_k = args.validate_top_k
    if args.residual:
        first_pass_top_k = min(
            args.predict_n,
            max(
                args.validate_top_k,
                args.residual_candidate_k,
                args.residual_tail_n + args.residual_uncertainty_n + args.validate_top_k,
            ),
        )
    predictor = MCTopKPredictor(
        model,
        ds.type_ids,
        ds.mos_n_params,
        device=args.device,
        objective=objective,
        residual_model_path=args.residual_model_path or None,
    )
    entries = predictor.predict_topk(
        n_mc=args.predict_n,
        top_k=first_pass_top_k,
        batch_size=parse_mc_batch(args.mc_batch),
        seed=args.seed,
        threshold_norm=(
            (float(args.failure_threshold) - stats.nom_measure) / stats.dev_std
            if args.failure_threshold is not None
            else None
        ),
    )
    calibration_results = []
    residual_model_path = ""
    if args.residual:
        residual_model_path = (
            args.residual_model_path
            or str(
                DEFAULT_CHECKPOINT.parent
                / f"{artifact_stem(circuit.name)}_tail_residual_attn.pth"
            )
        )
        residual_corrector = ResidualCorrector(
            circuit,
            stats,
            type_ids=ds.type_ids,
            mos_n_params=ds.mos_n_params,
            base_model=model,
            timeout_s=args.spice_timeout,
            device=args.device,
            objective=objective,
        )
        entries = residual_corrector.fit_and_apply(
            entries,
            model_path=residual_model_path,
            tail_n=args.residual_tail_n,
            uncertainty_n=args.residual_uncertainty_n,
            epochs=args.residual_epochs,
            lr=args.residual_lr,
            seed=args.seed,
            val_ratio=args.residual_val_ratio,
            threshold_norm=(
                (float(args.failure_threshold) - stats.nom_measure) / stats.dev_std
                if args.failure_threshold is not None
                else None
            ),
        )
        calibration_results = residual_corrector.calibration_results

    validator = SpiceValidator(
        circuit,
        stats,
        timeout_s=args.spice_timeout,
        objective=objective,
    )
    if args.validation_data:
        selected_raw = validator.validate_dataset(
            entries,
            args.validate_top_k,
            args.validation_data,
            metadata={
                "circuit": args.circuit,
                "predict_n": args.predict_n,
                "validate_top_k": args.validate_top_k,
                "first_pass_top_k": first_pass_top_k,
                "mode": "residual" if args.residual else "base",
                "objective": objective,
            },
            checkpoint_interval=args.validation_checkpoint_interval,
        )
    else:
        selected_raw = validator.validate(entries, args.validate_top_k)
    selected = tag_discovery_rows(selected_raw, source="final_validation")
    calibration_rows = normalize_calibration_rows(calibration_results)
    discovered_rows = calibration_rows + selected

    rng = np.random.default_rng(args.seed + 1)
    random_n = args.validate_top_k + len(calibration_rows)
    random_flat = rng.standard_normal(
        (random_n, circuit.total_params),
        dtype=np.float64,
    )
    random_dense = flat_z_to_dense(circuit, random_flat)
    random_entries = [
        TopEntry(
            score_norm=0.0,
            sample_idx=i,
            z=random_dense[i],
            base_norm=0.0,
        )
        for i in range(random_dense.shape[0])
    ]
    random_rows = validator.validate(random_entries, random_n)
    metrics = discovery_metrics(
        selected=discovered_rows,
        random_rows=random_rows,
        objective=objective,
        failure_threshold=args.failure_threshold,
    )
    method_spice_budget = len(ds) + len(calibration_rows) + len(selected)
    metrics.update(
        {
            "circuit": circuit.name,
            "predict_n": args.predict_n,
            "validate_top_k": args.validate_top_k,
            "discovered_count": len(discovered_rows),
            "selected_scope": (
                "residual_calibration_plus_final_validation"
                if calibration_rows
                else "final_validation"
            ),
            "first_pass_top_k": first_pass_top_k,
            "residual_candidate_k": args.residual_candidate_k if args.residual else 0,
            "mode": "residual" if args.residual else "base",
            "residual_model_path": residual_model_path,
            "residual_calibration_spice": len(calibration_results),
            "final_validation_spice": len(selected),
            "random_baseline_spice": len(random_rows),
            "method_spice_budget": method_spice_budget,
            "total_spice_budget": method_spice_budget + len(random_rows),
            "residual_calibration": summarize_discovery_rows(
                calibration_rows,
                objective,
                args.failure_threshold,
            ),
            "final_validation": summarize_discovery_rows(
                selected,
                objective,
                args.failure_threshold,
            ),
            "train": str(args.train),
            "ckpt": str(args.ckpt),
            "elapsed_seconds": time.time() - start,
            "elapsed_minutes": (time.time() - start) / 60.0,
        }
    )
    sections = tail_discovery_detail_sections(
        selected=discovered_rows,
        random_rows=random_rows,
        stats=stats,
        objective=objective,
        top_ks=args.detail_top_ks,
    )
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_log(str(log_path), metrics, sections=sections)
    if args.threshold_json:
        if args.target_probability <= 0.0:
            raise ValueError("--target-probability must be > 0 when --threshold-json is set")
        threshold_summary = tail_threshold_summary_from_rows(
            selected,
            objective=objective,
            predict_n=args.predict_n,
            target_probability=args.target_probability,
            source="final_validation",
            validation_data=args.validation_data,
            log=str(log_path),
        )
        threshold_path = Path(args.threshold_json)
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(
            json_dumps_pretty(threshold_summary) + "\n",
            encoding="utf-8",
        )
        print(f"threshold json: {threshold_path}")
    print(f"tail-discovery log: {log_path}")
    print(
        "selected_best="
        f"{metrics['selected'].get('best', float('nan')):.6e} "
        "random_best="
        f"{metrics['random'].get('best', float('nan')):.6e}"
    )
    print(f"elapsed={metrics['elapsed_minutes']:.2f} min")


def command_tail_coverage(args: argparse.Namespace) -> None:
    start = time.time()
    circuit = CircuitTxtParser.parse(args.circuit)
    objective = resolve_objective(args.objective, circuit.objective)
    if args.threshold is None:
        raise ValueError("--threshold is required")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_top_k <= 0 or args.max_top_k > args.predict_n:
        raise ValueError("--max-top-k must be in (0, predict_n]")
    if args.audit_strata < 0 or args.audit_per_layer < 0:
        raise ValueError("audit arguments must be non-negative")
    if args.tail_global_audit_stable_rounds < 1:
        raise ValueError("--tail-global-audit-stable-rounds must be positive")
    tail_closure_limit = (
        float(args.eta) if args.eta is not None else float(args.tail_closure_ratio)
    )
    if tail_closure_limit < 0.0:
        raise ValueError("--tail-closure-ratio must be non-negative")
    if args.tail_closure_z < 0.0:
        raise ValueError("--tail-closure-z must be non-negative")
    if args.tail_closure_weight < 0.0:
        raise ValueError("--tail-closure-weight must be non-negative")
    if args.tail_closure_fom_cap < 0.0:
        raise ValueError("--tail-closure-fom-cap must be non-negative")
    if not (0.0 <= args.tail_closure_decay <= 1.0):
        raise ValueError("--tail-closure-decay must be in [0, 1]")

    model, ds = make_model_from_sdat(args.train, args.ckpt, args.device)
    stats = dataset_stats(ds)
    threshold_norm = (float(args.threshold) - stats.nom_measure) / stats.dev_std
    predictor = MCTopKPredictor(
        model,
        ds.type_ids,
        ds.mos_n_params,
        device=args.device,
        objective=objective,
    )
    base_retained_k = args.max_top_k + args.audit_strata * args.batch_size
    ratio_retained_k = 0
    if args.retained_ratio > 0.0:
        ratio_retained_k = int(np.ceil(args.predict_n * args.retained_ratio))
    final_retained_k = min(
        args.predict_n,
        max(base_retained_k, ratio_retained_k),
    )
    retained_k = final_retained_k
    if args.residual:
        retained_k = min(
            args.predict_n,
            max(
                final_retained_k,
                args.residual_candidate_k,
                args.residual_tail_n + args.residual_uncertainty_n + final_retained_k,
            ),
        )
    entries = predictor.predict_topk(
        n_mc=args.predict_n,
        top_k=retained_k,
        batch_size=parse_mc_batch(args.mc_batch),
        seed=args.seed,
        threshold_norm=threshold_norm,
        failure_weight=args.failure_weight,
        boundary_weight=args.boundary_weight,
    )
    residual_calibration_spice = 0
    residual_model_path = ""
    calibration_measure_by_sample: dict[int, float] = {}
    if args.residual:
        residual_model_path = (
            args.residual_model_path
            or str(
                DEFAULT_CHECKPOINT.parent
                / f"{artifact_stem(circuit.name)}_coverage_residual_attn.pth"
            )
        )
        residual_corrector = ResidualCorrector(
            circuit,
            stats,
            type_ids=ds.type_ids,
            mos_n_params=ds.mos_n_params,
            base_model=model,
            timeout_s=args.spice_timeout,
            device=args.device,
            objective=objective,
        )
        entries = residual_corrector.fit_and_apply(
            entries,
            model_path=residual_model_path,
            tail_n=args.residual_tail_n,
            uncertainty_n=args.residual_uncertainty_n,
            epochs=args.residual_epochs,
            lr=args.residual_lr,
            seed=args.seed,
            val_ratio=args.residual_val_ratio,
            exclude_calibration=False,
            threshold_norm=threshold_norm,
            failure_weight=args.failure_weight,
            boundary_weight=args.boundary_weight,
            uncertainty_weight=args.uncertainty_weight,
        )
        residual_calibration_spice = len(residual_corrector.calibration_results)
        calibration_measure_by_sample = {
            int(float(row["sample_idx"])): float(row["measure"])
            for row in residual_corrector.calibration_results
        }
        entries = entries[:final_retained_k]

    runner = SpiceRunner(circuit, timeout_s=args.spice_timeout)
    runner.write_models()
    measures = np.full(len(entries), np.nan, dtype=np.float64)
    for pos, entry in enumerate(entries):
        cached_measure = calibration_measure_by_sample.get(int(entry.sample_idx))
        if cached_measure is not None:
            measures[pos] = cached_measure
    coverage_spice_calls = 0
    rng = np.random.default_rng(args.seed + 911)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    trace = Path(args.trace_tsv) if args.trace_tsv else None
    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)

    def is_failure(values: np.ndarray) -> np.ndarray:
        return values >= args.threshold if objective == "max" else values <= args.threshold

    rounds: list[dict[str, object]] = []
    status = "max_top_k"
    stop_round: dict[str, object] | None = None

    def write_trace_snapshot() -> None:
        if trace is None:
            return
        keys = list(rounds[0].keys()) if rounds else []
        lines = ["\t".join(keys)] if keys else []
        for row in rounds:
            lines.append("\t".join(str(row[key]) for key in keys))
        trace.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_coverage_snapshot(
        *,
        snapshot_status: str,
        current_label: str = "",
    ) -> None:
        summary = {
            "status": snapshot_status,
            "circuit": circuit.name,
            "objective": objective,
            "threshold": float(args.threshold),
            "predict_n": int(args.predict_n),
            "max_top_k": int(args.max_top_k),
            "retained_ratio": float(args.retained_ratio),
            "batch_size": int(args.batch_size),
            "audit_strata": int(args.audit_strata),
            "audit_per_layer": int(args.audit_per_layer),
            "tail_global_audit_stable_rounds": int(args.tail_global_audit_stable_rounds),
            "fom_target": float(args.fom_target),
            "unverified_beta_alpha": float(args.unverified_beta_alpha),
            "unverified_beta_beta": float(args.unverified_beta_beta),
            "tail_strata_prior_strength": float(args.tail_strata_prior_strength),
            "frontier_beta_alpha": float(args.frontier_beta_alpha),
            "frontier_beta_beta": float(args.frontier_beta_beta),
            "frontier_variance_weight": float(args.frontier_variance_weight),
            "tail_closure_ratio_limit": tail_closure_limit,
            "tail_closure_z": float(args.tail_closure_z),
            "tail_closure_weight": float(args.tail_closure_weight),
            "tail_closure_fom_cap": float(args.tail_closure_fom_cap),
            "tail_closure_decay": float(args.tail_closure_decay),
            "unverified_variance_mode": "frontier_gated_beta_binomial",
            "stopping_variance_mode": "three_term_fom_with_bounded_tail_excess_mass",
            "train_spice": int(len(ds)),
            "coverage_spice_calls": int(coverage_spice_calls),
            "residual_calibration_spice": int(residual_calibration_spice),
            "method_spice_calls": int(coverage_spice_calls) + int(residual_calibration_spice),
            "spice_calls": int(len(ds)) + int(coverage_spice_calls) + int(residual_calibration_spice),
            "result": stop_round or (rounds[-1] if rounds else {}),
            "mode": "residual" if args.residual else "base",
            "residual_model_path": residual_model_path,
            "elapsed_seconds": time.time() - start,
            "progress": {
                "current_label": current_label,
                "validated_positions": int(np.sum(np.isfinite(measures))),
                "retained_entries": int(len(entries)),
                "completed_rounds": int(len(rounds)),
            },
        }
        out_json.write_text(json_dumps_pretty(summary) + "\n", encoding="utf-8")
        write_trace_snapshot()

    def validate_positions(positions: Sequence[int], label: str) -> None:
        nonlocal coverage_spice_calls
        pending = [int(p) for p in positions if 0 <= int(p) < len(entries) and not np.isfinite(measures[int(p)])]
        if not pending:
            return
        for done, pos in enumerate(pending, start=1):
            measure = runner.run_with_params(runner.zscore_to_params(entries[pos].z))
            if measure is not None:
                measures[pos] = measure
                coverage_spice_calls += 1
            if done == 1 or done % 10 == 0 or done == len(pending):
                elapsed = time.time() - start
                print(
                    f"{label}: {done}/{len(pending)} "
                    f"spice={len(ds) + coverage_spice_calls + residual_calibration_spice} "
                    f"elapsed={elapsed / 60.0:.2f} min"
                )
            if done % 25 == 0 or done == len(pending):
                write_coverage_snapshot(
                    snapshot_status="running",
                    current_label=f"{label} {done}/{len(pending)}",
                )
        print(f"{label}: validated {len(pending)} new SPICE rows")

    def tail_audit_edges(k: int) -> list[tuple[int, int]]:
        """Partition every retained, unvalidated rank into audit strata.

        The first strata remain narrow close to the coverage frontier, while
        the final stratum samples the whole residual retained tail.  The old
        implementation audited only the first few bands and silently omitted
        all later observed failures from the Pf point estimate.
        """
        if k >= len(entries):
            return []
        strata = max(1, int(args.audit_strata))
        edges: list[tuple[int, int]] = []
        start = k
        width = max(1, int(args.batch_size))
        for _ in range(strata - 1):
            end = min(len(entries), start + width)
            if start < end:
                edges.append((start, end))
            start = end
            width *= 2
            if start >= len(entries):
                return edges
        if start < len(entries):
            edges.append((start, len(entries)))
        return edges

    # Frontier-gated unverified-tail uncertainty:
    # the current SPICE-validated frontier failure rate sets the absolute
    # scale of remaining failure mass, while residual flip probability only
    # redistributes relative risk across deeper rank strata.
    def estimate_frontier_gated_tail(
        k: int,
        strata_edges: list[tuple[int, int]],
    ) -> tuple[float, float, float, int, int]:
        """Estimate retained-tail failure mass and its predictive variance.

        The prior mean of each stratum's failure probability is anchored to the
        SPICE-observed fail rate in the current focused-validation frontier batch,
        scaled by the stratum's average residual flip probability.  Audited
        failures contribute to the point estimate; the remaining mass is a
        posterior expectation rather than an implicitly zero contribution.
        """
        variance = 0.0
        expected_failures = 0.0
        unresolved_expected = 0.0
        audit_valid = 0
        audit_failures = 0
        prior_strength = max(0.0, float(args.tail_strata_prior_strength))
        eps = 1e-9

        # ---- step 1: frontier batch fail rate (Laplace smoothed) ----
        frontier_start = max(0, k - args.batch_size)
        frontier_end = k
        frontier_measures = measures[frontier_start:frontier_end]
        frontier_valid = np.isfinite(frontier_measures)
        b_t = int(np.sum(frontier_valid))
        if b_t == 0:
            return 0.0, 0.0, 0.0, 0, 0
        c_t = int(np.sum(is_failure(frontier_measures[frontier_valid])))
        alpha0 = float(args.frontier_beta_alpha)
        beta0 = float(args.frontier_beta_beta)
        p_hat_t_bd = (c_t + alpha0) / (b_t + alpha0 + beta0)

        # ---- step 2: average flip probability in the frontier batch ----
        pi_frontier = float(
            np.mean([entries[pos].flip_probability for pos in range(frontier_start, frontier_end)])
        )
        if pi_frontier < eps:
            pi_frontier = eps

        # ---- step 3: estimate every residual retained-tail stratum ----
        for start_rank, end_rank in strata_edges:
            n_weight = end_rank - start_rank
            if n_weight <= 0:
                continue

            layer_measures = measures[start_rank:end_rank]
            valid = np.isfinite(layer_measures)
            n_h = int(np.sum(valid))
            m_h = int(np.sum(is_failure(layer_measures[valid]))) if n_h > 0 else 0
            M_h = n_weight - n_h  # unverified samples in this stratum
            if M_h <= 0:
                expected_failures += m_h
                audit_valid += n_h
                audit_failures += m_h
                continue

            # ---- step 4: stratum-average flip probability ----
            bar_pi_h = float(
                np.mean([entries[pos].flip_probability for pos in range(start_rank, end_rank)])
            )
            bar_pi_h = max(bar_pi_h, eps)

            # ---- step 5: frontier-anchored prior mean for this stratum ----
            # absolute scale from frontier fail rate, relative risk from flip-prob ratio
            p_hat_t_h = p_hat_t_bd * bar_pi_h / pi_frontier
            p_hat_t_h = min(max(p_hat_t_h, eps), 1.0 - eps)

            # ---- step 6: Beta prior → posterior with audit observations ----
            kappa = prior_strength
            if kappa > 0.0:
                alpha_h0 = kappa * p_hat_t_h
                beta_h0 = kappa * (1.0 - p_hat_t_h)
            else:
                alpha_h0 = float(args.unverified_beta_alpha)
                beta_h0 = float(args.unverified_beta_beta)

            alpha_h = max(eps, alpha_h0) + m_h
            beta_h = max(eps, beta_h0) + n_h - m_h

            mu_h = alpha_h / (alpha_h + beta_h)
            sigma2_h = (alpha_h * beta_h) / (
                (alpha_h + beta_h) ** 2 * (alpha_h + beta_h + 1.0)
            )

            # ---- step 7: Beta-binomial posterior-predictive variance ----
            var_Fh = M_h * mu_h * (1.0 - mu_h) + M_h * (M_h - 1.0) * sigma2_h
            variance += var_Fh / (args.predict_n ** 2)
            expected_failures += m_h + M_h * mu_h
            unresolved_expected += M_h * mu_h
            audit_valid += n_h
            audit_failures += m_h

        return (
            float(expected_failures),
            float(unresolved_expected),
            float(variance),
            audit_valid,
            audit_failures,
        )

    previous_k = 0
    global_tail_audit_enabled = False
    fom_stable_rounds = 0
    smoothed_tail_fom = 0.0

    for round_idx, k in enumerate(range(args.batch_size, args.max_top_k + 1, args.batch_size), start=1):
        k = min(k, args.max_top_k, len(entries))
        validate_positions(range(previous_k, k), f"coverage round {round_idx}")
        previous_k = k

        top_measures = measures[:k]
        top_valid = np.isfinite(top_measures)
        top_failures = int(np.sum(is_failure(top_measures[top_valid])))
        pf_top = float(top_failures / args.predict_n)

        audit_variance = 0.0
        residual_variance = 0.0
        audit_edges = tail_audit_edges(k)
        # The final broad stratum monitors the unexplored retained tail.  It
        # participates in the tail-closure certificate, rather than inflating
        # the local frontier FOM throughout the entire run.
        frontier_edges = audit_edges[:-1] if len(audit_edges) > 1 else []
        sampled_audit_edges = audit_edges if global_tail_audit_enabled else frontier_edges
        for layer, (start_rank, end_rank) in enumerate(sampled_audit_edges, start=1):
            layer_positions = np.arange(start_rank, end_rank)
            sample_n = min(args.audit_per_layer, len(layer_positions))
            if sample_n <= 0:
                continue
            sampled = rng.choice(layer_positions, size=sample_n, replace=False)
            validate_positions(sampled.tolist(), f"audit round {round_idx} layer {layer + 1}")
        # FOM uses frontier_edges variance (tail uncertainty); Pf uses only
        # verified SPICE failures (no unverified prior mass in point estimate).
        (
            _,
            _,
            frontier_tail_variance,
            _,
            frontier_audit_failures,
        ) = estimate_frontier_gated_tail(k, frontier_edges)
        (
            _,
            unresolved_tail_failures,
            global_tail_variance,
            audit_valid,
            _,
        ) = estimate_frontier_gated_tail(k, audit_edges)
        confirmed_tail_failures = max(0, frontier_audit_failures)

        # Keep the directly sampled residual-error component separate from the
        # Beta-binomial predictive term.  It is intentionally conservative.
        for start_rank, end_rank in frontier_edges:
            layer_measures = measures[start_rank:end_rank]
            valid = np.isfinite(layer_measures)
            n_h = int(np.sum(valid))
            if n_h == 0:
                continue
            m_h = int(np.sum(is_failure(layer_measures[valid])))
            n_weight = end_rank - start_rank
            p_tilde = (m_h + 1.0) / (n_h + 2.0)
            audit_variance += (
                (n_weight / args.predict_n) ** 2
                * p_tilde
                * (1.0 - p_tilde)
                / max(n_h, 1)
            )
            sampled_positions = np.flatnonzero(valid) + start_rank
            pi = np.asarray(
                [entries[int(pos)].flip_probability for pos in sampled_positions],
                dtype=np.float64,
            )
            residual_variance += (
                args.residual_variance_weight
                / (args.predict_n**2)
                * n_weight
                / max(n_h, 1)
                * float(np.sum(pi * (1.0 - pi)))
            )
        # The local FOM controls when the broad tail audit begins.  The
        # published/stopping FOM below also includes the retained-tail closure
        # mass, so convergence is expressed by a single FOM threshold.
        unverified_variance = frontier_tail_variance * args.frontier_variance_weight
        # Pf uses only SPICE-verified failures; unverified tail mass goes into FOM
        verified_tail_failures = top_failures + confirmed_tail_failures
        pf = float(verified_tail_failures / args.predict_n)
        v_mc = pf * max(1.0 - pf, 0.0) / args.predict_n
        local_v_total = float(v_mc + audit_variance + residual_variance + unverified_variance)
        local_fom = float(np.sqrt(local_v_total) / pf) if pf > 0.0 else float("inf")
        local_fom_ok = local_fom <= args.fom_target
        tail_upper_failures = unresolved_tail_failures + (
            args.tail_closure_z * np.sqrt(global_tail_variance) * args.predict_n
        )
        tail_closure_ratio = tail_upper_failures / max(verified_tail_failures, 1.0)
        if not global_tail_audit_enabled:
            fom_stable_rounds = fom_stable_rounds + 1 if local_fom_ok else 0
            if fom_stable_rounds >= args.tail_global_audit_stable_rounds:
                global_tail_audit_enabled = True
        tail_fom_scale = args.tail_closure_weight / max(tail_closure_limit, 1e-300)
        tail_fom_raw = min(
            float(args.tail_closure_fom_cap),
            tail_closure_ratio * tail_fom_scale,
        )
        tail_fom_excess = min(
            float(args.tail_closure_fom_cap),
            max(0.0, tail_closure_ratio - tail_closure_limit) * tail_fom_scale,
        )
        # Let retained-tail risk increase immediately, but release it
        # gradually as repeated coverage rounds support tail closure.  This
        # prevents a single clean frontier/audit round from causing a step-like
        # drop in the reported stopping FoM.
        smoothed_tail_fom = max(
            float(tail_fom_excess),
            float(smoothed_tail_fom) * float(args.tail_closure_decay),
        )
        tail_closure_variance = (
            smoothed_tail_fom * pf
        ) ** 2
        reported_audit_variance = float(audit_variance + residual_variance)
        tail_variance = float(unverified_variance + tail_closure_variance)
        v_total = float(v_mc + reported_audit_variance + tail_variance)
        fom = float(np.sqrt(v_total) / pf) if pf > 0.0 else float("inf")
        fom_ok = fom <= args.fom_target
        tail_closed = tail_closure_ratio <= tail_closure_limit
        converged = fom_ok and round_idx >= args.min_rounds

        row: dict[str, object] = {
            "round": round_idx,
            "top_k": k,
            "valid_top_spice": int(np.sum(top_valid)),
            "tail_failure_count": top_failures,
            "audit_valid_spice": audit_valid,
            "audit_failure_count": confirmed_tail_failures,
            "tail_estimated_failures": confirmed_tail_failures,
            "tail_unresolved_failures": unresolved_tail_failures,
            "tail_upper_failures": tail_upper_failures,
            "tail_closure_ratio": tail_closure_ratio,
            "tail_closed": int(tail_closed),
            "global_tail_audit": int(global_tail_audit_enabled),
            "pf_top_only": pf_top,
            "failure_probability": pf,
            "yield": 1.0 - pf,
            "v_mc": v_mc,
            "v_audit": reported_audit_variance,
            "v_tail": tail_variance,
            "v_audit_strata": audit_variance,
            "v_residual_diag": residual_variance,
            "v_tail_beta_diag": unverified_variance,
            "v_tail_closure_diag": tail_closure_variance,
            "v_local_total": local_v_total,
            "v_total": v_total,
            "local_fom": local_fom,
            "tail_fom": tail_fom_excess,
            "tail_fom_smoothed": smoothed_tail_fom,
            "tail_fom_raw": tail_fom_raw,
            "fom": fom,
            "local_fom_ok": int(local_fom_ok),
            "fom_ok": int(fom_ok),
            "converged": int(converged),
            "method_spice_calls": int(coverage_spice_calls + residual_calibration_spice),
            "spice_calls": int(len(ds) + coverage_spice_calls + residual_calibration_spice),
        }
        rounds.append(row)
        write_coverage_snapshot(snapshot_status="running", current_label=f"round {round_idx}")
        print(
            f"round={round_idx} K={k} C={top_failures} "
            f"pf={pf:.6e} fom={fom:.4f}"
        )
        if converged:
            status = "converged"
            stop_round = row
            break
        if (
            args.max_spice > 0
            and int(len(ds) + coverage_spice_calls + residual_calibration_spice) >= args.max_spice
        ):
            status = "budget_exhausted"
            stop_round = row
            break

    if stop_round is None and rounds:
        stop_round = rounds[-1]
    write_coverage_snapshot(snapshot_status=status, current_label="final")
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    if trace is not None:
        print(f"coverage trace: {trace}")
    print(f"coverage summary: {out_json}")
    print(
        f"status={status} pf={float((stop_round or {}).get('failure_probability', float('nan'))):.6e} "
        f"fom={float((stop_round or {}).get('fom', float('nan'))):.4f} "
        f"spice={summary['spice_calls']}"
    )


def finite_golden_detail_sections(
    pred_norm: np.ndarray,
    measures: np.ndarray,
    stats: SdatStats,
    objective: str,
    top_ks: Sequence[int],
) -> list[str]:
    valid_indices = np.flatnonzero(np.isfinite(measures))
    valid_measures = measures[valid_indices]
    valid_pred_norm = pred_norm[valid_indices]
    valid_pred_phys = valid_pred_norm * stats.dev_std + stats.nom_measure
    measure_norm = (valid_measures - stats.nom_measure) / stats.dev_std
    max_k = min(max(top_ks, default=0), len(valid_indices))
    sections: list[str] = []
    if max_k <= 0:
        return sections

    pred_order = top_indices(valid_pred_norm, max_k, objective)
    true_order = top_indices(valid_measures, max_k, objective)

    def make_table(title: str, order: np.ndarray) -> str:
        lines = [
            title,
            "rank | sample_idx | pred_norm | pred_phys | spice_norm | spice",
        ]
        for rank, local_idx in enumerate(order, start=1):
            lines.append(
                f"{rank} | {int(valid_indices[local_idx])} | "
                f"{float(valid_pred_norm[local_idx]):.6e} | "
                f"{float(valid_pred_phys[local_idx]):.6e} | "
                f"{float(measure_norm[local_idx]):.6e} | "
                f"{float(valid_measures[local_idx]):.6e}"
            )
        return "\n".join(lines)

    for k in top_ks:
        k = min(k, max_k)
        if k <= 0:
            continue
        sections.append(make_table(f"predicted_top_{k}", pred_order[:k]))
        sections.append(make_table(f"spice_true_top_{k}", true_order[:k]))
    return sections


def tag_discovery_rows(
    rows: Sequence[dict[str, float]],
    source: str,
) -> list[dict[str, float | str]]:
    tagged = []
    for row in rows:
        tagged_row: dict[str, float | str] = dict(row)
        tagged_row["source"] = source
        tagged.append(tagged_row)
    return tagged


def normalize_calibration_rows(
    rows: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    normalized = []
    for rank, row in enumerate(rows, start=1):
        normalized_row: dict[str, float | str] = dict(row)
        normalized_row["rank"] = float(rank)
        normalized_row["source"] = str(row.get("source", "residual_calibration"))
        normalized.append(normalized_row)
    return normalized


def json_dumps_pretty(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def tail_threshold_summary_from_rows(
    rows: Sequence[dict[str, float | str]],
    *,
    objective: str,
    predict_n: int,
    target_probability: float,
    source: str,
    validation_data: str | Path | None = None,
    log: str | Path | None = None,
) -> dict[str, object]:
    valid_measures = np.asarray(
        [float(row["measure"]) for row in rows if np.isfinite(float(row["measure"]))],
        dtype=np.float64,
    )
    if valid_measures.size == 0:
        raise ValueError("cannot select threshold: no valid SPICE rows")
    ordered = np.sort(valid_measures)
    if objective == "max":
        ordered = ordered[::-1]
    target_rank = max(1, int(np.ceil(target_probability * predict_n)))
    if ordered.size < target_rank:
        raise ValueError(
            "not enough verified SPICE rows for target probability: "
            f"rank={target_rank}, valid={ordered.size}, predict_n={predict_n}, "
            f"target_probability={target_probability}"
        )
    threshold = float(ordered[target_rank - 1])
    return {
        "reference_type": "our_method_tail_spice_verified",
        "source": source,
        "objective": objective,
        "predict_n": int(predict_n),
        "target_probability": float(target_probability),
        "target_rank": int(target_rank),
        "threshold": threshold,
        "valid_spice": int(ordered.size),
        "failed_spice": int(len(rows) - ordered.size),
        "top_measure": float(ordered[0]),
        "lowest_verified_tail_measure": float(ordered[-1]),
        "validation_data": str(validation_data or ""),
        "log": str(log or ""),
    }


def tail_discovery_detail_sections(
    selected: Sequence[dict[str, float | str]],
    random_rows: Sequence[dict[str, float | str]],
    stats: SdatStats,
    objective: str,
    top_ks: Sequence[int],
) -> list[str]:
    sections: list[str] = []

    def row_measure_norm(row: dict[str, float | str]) -> float:
        return (float(row["measure"]) - stats.nom_measure) / stats.dev_std

    def sorted_by_spice(
        rows: Sequence[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        return sorted(
            rows,
            key=lambda row: float(row["measure"]),
            reverse=objective == "max",
        )

    def sorted_by_pred(
        rows: Sequence[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        return sorted(
            rows,
            key=lambda row: float(row["pred_norm"]),
            reverse=objective == "max",
        )

    def make_table(
        title: str,
        rows: Sequence[dict[str, float | str]],
        include_pred: bool,
    ) -> str:
        if include_pred:
            header = (
                "rank | source | source_rank | sample_idx | pred_norm | pred_phys | "
                "base_norm | base_phys | spice_norm | spice"
            )
        else:
            header = "rank | source | source_rank | sample_idx | spice_norm | spice"
        lines = [title, header]
        for rank, row in enumerate(rows, start=1):
            source = str(row.get("source", "random"))
            source_rank = int(float(row.get("rank", rank)))
            sample_idx = int(float(row["sample_idx"]))
            spice = float(row["measure"])
            spice_norm = row_measure_norm(row)
            if include_pred:
                pred_norm = float(row["pred_norm"])
                base_norm = float(row["base_norm"])
                pred_phys = pred_norm * stats.dev_std + stats.nom_measure
                base_phys = base_norm * stats.dev_std + stats.nom_measure
                lines.append(
                    f"{rank} | {source} | {source_rank} | {sample_idx} | "
                    f"{pred_norm:.6e} | {pred_phys:.6e} | "
                    f"{base_norm:.6e} | {base_phys:.6e} | "
                    f"{spice_norm:.6e} | {spice:.6e}"
                )
            else:
                lines.append(
                    f"{rank} | {source} | {source_rank} | {sample_idx} | "
                    f"{spice_norm:.6e} | {spice:.6e}"
                )
        return "\n".join(lines)

    selected_by_pred = sorted_by_pred(selected)
    selected_by_spice = sorted_by_spice(selected)
    random_by_spice = sorted_by_spice(random_rows)
    for k in top_ks:
        k_selected = min(k, len(selected))
        k_random = min(k, len(random_rows))
        if k_selected > 0:
            sections.append(
                make_table(
                    f"selected_predicted_order_top_{k_selected}",
                    selected_by_pred[:k_selected],
                    include_pred=True,
                )
            )
            sections.append(
                make_table(
                    f"selected_spice_order_top_{k_selected}",
                    selected_by_spice[:k_selected],
                    include_pred=True,
                )
            )
        if k_random > 0:
            sections.append(
                make_table(
                    f"random_spice_order_top_{k_random}",
                    random_by_spice[:k_random],
                    include_pred=False,
                )
            )
    return sections


def command_all(args: argparse.Namespace) -> None:
    start = time.time()
    circuit = CircuitTxtParser.parse(args.circuit)
    objective = resolve_objective(args.objective, circuit.objective)
    print(f"objective: {objective}")
    stem = artifact_stem(circuit.name)
    result_stem = stem if objective == "max" else f"{stem}_min"
    train_path = (
        Path(args.train_file)
        if args.train_file
        else DEFAULT_TRAIN_DATA.parent / f"{stem}_train_{args.train_n}.sdat"
    )
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt
        else DEFAULT_CHECKPOINT.parent / f"{result_stem}_v16.pth"
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        if args.train_z_scale <= 0:
            raise ValueError("train_z_scale must be positive")
        samples = SobolZSampler.sample(
            n_samples=args.train_n,
            dim=circuit.total_params,
            sigma_range=args.sigma_range,
            seed=args.seed,
        ) * args.train_z_scale
        SpiceDatasetBuilder(circuit, timeout_s=args.spice_timeout).build_from_flat_z(
            samples,
            train_path,
            allow_failed_nominal=args.allow_failed_nominal,
        )

    if not args.no_train:
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            val_ratio=args.val_ratio,
            objective=objective,
            model_architecture=args.model_architecture,
            use_type_embedding=args.use_type_embedding,
            use_global_context=args.use_global_context,
            fusion_mode=args.fusion_mode,
            threshold_norm=(
                (float(args.threshold) - float(SdatDataset(train_path).stats["nom_measure"]))
                / float(SdatDataset(train_path).stats["dev_std"])
                if args.threshold is not None
                else None
            ),
            tail_fail_weight=args.tail_fail_weight,
            boundary_weight=args.boundary_weight,
            boundary_sigma=args.boundary_sigma,
        )
        model = V16Trainer(config).train(train_path, ckpt_path)
        ds = SdatDataset(train_path)
    else:
        model, ds = make_model_from_sdat(
            train_path,
            ckpt_path,
            args.device,
            model_architecture=args.model_architecture,
            use_type_embedding=args.use_type_embedding,
            use_global_context=args.use_global_context,
            fusion_mode=args.fusion_mode,
        )

    top_k = min(max(1, args.validate_top_k, args.predict_n // 100), 100_000)
    first_pass_top_k = top_k
    if args.residual:
        first_pass_top_k = min(
            args.predict_n,
            max(
                top_k,
                args.residual_tail_n
                + args.residual_uncertainty_n
                + args.validate_top_k,
            ),
        )
    predictor = MCTopKPredictor(
        model,
        ds.type_ids,
        ds.mos_n_params,
        device=args.device,
        objective=objective,
    )
    entries = predictor.predict_topk(
        n_mc=args.predict_n,
        top_k=first_pass_top_k,
        batch_size=parse_mc_batch(args.mc_batch),
        seed=args.seed,
        threshold_norm=(
            (float(args.threshold) - float(ds.stats["nom_measure"]))
            / float(ds.stats["dev_std"])
            if args.threshold is not None
            else None
        ),
    )

    stats = dataset_stats(ds)
    calibration_results = []
    if args.residual:
        residual_path = (
            args.residual_model_path
            or str(DEFAULT_CHECKPOINT.parent / f"{result_stem}_residual_attn.pth")
        )
        residual_corrector = ResidualCorrector(
            circuit,
            stats,
            type_ids=ds.type_ids,
            mos_n_params=ds.mos_n_params,
            base_model=model,
            timeout_s=args.spice_timeout,
            device=args.device,
            objective=objective,
        )
        entries = residual_corrector.fit_and_apply(
            entries,
            model_path=residual_path,
            tail_n=args.residual_tail_n,
            uncertainty_n=args.residual_uncertainty_n,
            epochs=args.residual_epochs,
            lr=args.residual_lr,
            seed=args.seed,
            val_ratio=args.residual_val_ratio,
            threshold_norm=(
                (float(args.threshold) - float(ds.stats["nom_measure"]))
                / float(ds.stats["dev_std"])
                if args.threshold is not None
                else None
            ),
        )
        calibration_results = residual_corrector.calibration_results

    pred_path = DEFAULT_PRED_DIR / f"{result_stem}_mc_pred_{args.predict_n}.npy"
    save_predictions(entries, stats, pred_path)

    if args.validate_top_k > 0:
        validate_predictions(
            entries,
            circuit_path=args.circuit,
            stats=stats,
            top_k=args.validate_top_k,
            timeout_s=args.spice_timeout,
            objective=objective,
            log_path=DEFAULT_LOG_DIR / f"{result_stem}_validate_{args.predict_n}.log",
            calibration_results=calibration_results,
        )

    if not args.keep_data and train_path.exists():
        train_path.unlink()
        print(f"removed {train_path}")

    elapsed = time.time() - start
    print(f"done in {elapsed / 60.0:.1f} min")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Python v16 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data", help="Sobol + Python SPICE -> .sdat")
    build.add_argument("--circuit", required=True)
    build.add_argument("--train-n", type=int, default=1024)
    build.add_argument("--sigma-range", type=float, default=5.0)
    build.add_argument(
        "--train-z-scale",
        type=float,
        default=1.0,
        help="Scale Sobol z-scores used for training only; MC remains unchanged.",
    )
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--spice-timeout", type=int, default=60)
    build.add_argument("--output", default=str(DEFAULT_TRAIN_DATA.parent / "train_python.sdat"))
    build.add_argument("--samples-out", default="")
    build.add_argument(
        "--allow-failed-nominal",
        action="store_true",
        help="Keep old behavior: replace failed SPICE runs with the nominal measure.",
    )
    build.set_defaults(func=command_build_data)

    train = sub.add_parser("train", help="Train v16 from .sdat")
    train.add_argument("--train", required=True)
    train.add_argument(
        "--val",
        default="",
        help="Independent .sdat validation set used for checkpoint selection.",
    )
    train.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT))
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    train.add_argument("--threshold", type=float, default=None)
    train.add_argument("--tail-fail-weight", type=float, default=2.0)
    train.add_argument("--boundary-weight", type=float, default=1.0)
    train.add_argument("--boundary-sigma", type=float, default=1.0)
    train.add_argument("--val-ratio", type=float, default=0.0)
    train.add_argument("--objective", choices=["max", "min"], default="max")
    add_model_ablation_args(train)
    train.set_defaults(func=command_train)

    sdat_eval = sub.add_parser(
        "sdat-eval",
        help="Evaluate a checkpoint on an independent SPICE-labeled Sobol .sdat",
    )
    sdat_eval.add_argument("--circuit", required=True)
    sdat_eval.add_argument("--train", required=True)
    sdat_eval.add_argument("--eval", required=True)
    sdat_eval.add_argument("--ckpt", required=True)
    sdat_eval.add_argument("--split", choices=["validation", "test"], required=True)
    sdat_eval.add_argument("--top-ks", nargs="+", type=int, default=[10, 50, 100])
    sdat_eval.add_argument("--detail-top-ks", nargs="+", type=int, default=[10, 100])
    sdat_eval.add_argument(
        "--objective",
        choices=["auto", "max", "min"],
        default="auto",
    )
    sdat_eval.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    sdat_eval.add_argument("--eval-batch", type=int, default=0)
    sdat_eval.add_argument("--log", required=True)
    sdat_eval.set_defaults(func=command_sdat_eval)

    predict = sub.add_parser("predict", help="MC top-k prediction")
    predict.add_argument("--train", required=True, help=".sdat file for stats/type info")
    predict.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT))
    predict.add_argument("--predict-n", type=int, default=100000)
    predict.add_argument("--top-k", type=int, default=1000)
    predict.add_argument("--mc-batch", default="auto", help="Positive integer or auto")
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    predict.add_argument("--objective", choices=["max", "min"], default="max")
    predict.add_argument(
        "--residual-model-path",
        default="",
        help="Apply a trained attention residual model to every MC sample.",
    )
    predict.add_argument(
        "--circuit",
        default="",
        help="Circuit descriptor required for optional SPICE validation.",
    )
    predict.add_argument("--validate-top-k", type=int, default=0)
    predict.add_argument("--spice-timeout", type=int, default=60)
    predict.add_argument(
        "--validation-log",
        default="",
        help="Validation log path; defaults under artifacts/logs.",
    )
    predict.add_argument(
        "--validation-data",
        default="",
        help="Optional resumable .npz containing z, predictions, and SPICE measures.",
    )
    predict.add_argument("--output", default=str(DEFAULT_PRED_DIR / "mc_pred_python.npy"))
    predict.set_defaults(func=command_predict)

    tail_threshold = sub.add_parser(
        "tail-threshold",
        help="Recompute thresholds/counts from a saved screened-tail data set.",
    )
    tail_threshold.add_argument("--data", required=True)
    tail_threshold.add_argument(
        "--objective", choices=["auto", "max", "min"], default="auto"
    )
    tail_threshold.add_argument("--predict-n", type=int, default=0)
    tail_threshold.add_argument(
        "--target-probability", type=float, nargs="*", default=[]
    )
    tail_threshold.add_argument("--threshold", type=float)
    tail_threshold.set_defaults(func=command_tail_threshold)

    finite = sub.add_parser(
        "finite-golden",
        help="Evaluate surrogate accuracy on a fully SPICE-simulated finite MC pool",
    )
    finite.add_argument("--circuit", required=True)
    finite.add_argument("--train", required=True)
    finite.add_argument("--ckpt", required=True)
    finite.add_argument("--pool-n", type=int, default=10000)
    finite.add_argument("--pool-z", default="", help="Optional .npy flat-z pool")
    finite.add_argument("--golden-npz", default="", help="Cache/load SPICE golden")
    finite.add_argument("--force-spice", action="store_true")
    finite.add_argument("--top-ks", nargs="+", type=int, default=[10, 50, 100, 1000])
    finite.add_argument(
        "--detail-top-ks",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="Detailed prediction/SPICE table sizes written to the text log.",
    )
    finite.add_argument(
        "--failure-threshold",
        type=float,
        default=None,
        help="Optional physical spec threshold for failure-rate/enrichment metrics.",
    )
    finite.add_argument(
        "--objective",
        choices=["auto", "max", "min"],
        default="auto",
        help="Tail direction; auto reads '# objective:' from the circuit descriptor.",
    )
    finite.add_argument("--seed", type=int, default=42)
    finite.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    finite.add_argument("--eval-batch", type=int, default=0)
    finite.add_argument("--spice-timeout", type=int, default=60)
    finite.add_argument(
        "--log",
        default=str(DEFAULT_LOG_DIR / "finite_golden.log"),
        help="Plain-text metrics output.",
    )
    finite.set_defaults(func=command_finite_golden)

    discovery = sub.add_parser(
        "tail-discovery",
        help="Validate surrogate-selected high-sigma candidates against random MC",
    )
    discovery.add_argument("--circuit", required=True)
    discovery.add_argument("--train", required=True)
    discovery.add_argument("--ckpt", required=True)
    discovery.add_argument("--predict-n", type=int, default=1000000)
    discovery.add_argument("--validate-top-k", type=int, default=100)
    discovery.add_argument(
        "--detail-top-ks",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="Detailed selected/random table sizes written to the text log.",
    )
    discovery.add_argument("--mc-batch", default="auto", help="Positive integer or auto")
    discovery.add_argument(
        "--failure-threshold",
        type=float,
        default=None,
        help="Optional physical spec threshold for failure discovery metrics.",
    )
    discovery.add_argument(
        "--objective",
        choices=["auto", "max", "min"],
        default="auto",
        help="Tail direction; auto reads '# objective:' from the circuit descriptor.",
    )
    discovery.add_argument("--seed", type=int, default=42)
    discovery.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    discovery.add_argument("--spice-timeout", type=int, default=60)
    discovery.add_argument("--residual", action="store_true")
    discovery.add_argument(
        "--residual-model-path",
        default="",
        help="Load/save the attention residual model for tail calibration.",
    )
    discovery.add_argument("--residual-tail-n", type=int, default=300)
    discovery.add_argument("--residual-uncertainty-n", type=int, default=200)
    discovery.add_argument(
        "--residual-candidate-k",
        type=int,
        default=100000,
        help=(
            "Base-model candidate heap kept for residual reranking. "
            "Residual calibration points are sampled from this heap, then the "
            "remaining candidates are reranked by the residual model."
        ),
    )
    discovery.add_argument("--residual-epochs", type=int, default=300)
    discovery.add_argument("--residual-lr", type=float, default=2e-3)
    discovery.add_argument(
        "--residual-val-ratio",
        type=float,
        default=0.2,
        help="Fraction of residual calibration SPICE points reserved for validation.",
    )
    discovery.add_argument(
        "--log",
        default=str(DEFAULT_LOG_DIR / "tail_discovery.log"),
        help="Plain-text metrics output.",
    )
    discovery.add_argument(
        "--validation-data",
        default="",
        help="Optional resumable .npz for final validated tail samples.",
    )
    discovery.add_argument(
        "--validation-checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint interval for --validation-data SPICE validation.",
    )
    discovery.add_argument(
        "--target-probability",
        type=float,
        default=0.0,
        help="If >0, choose this Pf threshold from final SPICE validation rows.",
    )
    discovery.add_argument(
        "--threshold-json",
        default="",
        help="Write threshold JSON selected from final validation SPICE rows.",
    )
    discovery.set_defaults(func=command_tail_discovery)

    coverage = sub.add_parser(
        "tail-coverage",
        help="Estimate fixed-threshold Pf with expanding Top-K coverage checks",
    )
    coverage.add_argument("--circuit", required=True)
    coverage.add_argument("--train", required=True)
    coverage.add_argument("--ckpt", required=True)
    coverage.add_argument("--threshold", type=float, required=True)
    coverage.add_argument("--predict-n", type=int, default=1000000)
    coverage.add_argument("--max-top-k", type=int, default=5000)
    coverage.add_argument(
        "--retained-ratio",
        type=float,
        default=0.0,
        help=(
            "If >0, retain at least this fraction of the MC pool for residual "
            "reranking and fixed tail-strata uncertainty."
        ),
    )
    coverage.add_argument("--batch-size", type=int, default=100)
    coverage.add_argument("--audit-strata", type=int, default=3)
    coverage.add_argument("--audit-per-layer", type=int, default=100)
    coverage.add_argument(
        "--tail-global-audit-stable-rounds",
        type=int,
        default=3,
        help="Consecutive local-FOM passes required before broad tail auditing begins.",
    )
    coverage.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Deprecated alias for --tail-closure-ratio.",
    )
    coverage.add_argument("--fom-target", type=float, default=0.1)
    coverage.add_argument(
        "--unverified-beta-alpha",
        type=float,
        default=0.5,
        help="Beta prior alpha for variance of unverified audit-stratum failure mass.",
    )
    coverage.add_argument(
        "--unverified-beta-beta",
        type=float,
        default=0.5,
        help="Beta prior beta for variance of unverified audit-stratum failure mass.",
    )
    coverage.add_argument(
        "--tail-strata-prior-strength",
        type=float,
        default=20.0,
        help="Pseudo-count strength for residual-probability prior in fixed tail-strata variance.",
    )
    coverage.add_argument(
        "--frontier-beta-alpha",
        type=float,
        default=1.0,
        help="Beta prior alpha for Laplace smoothing of frontier batch fail rate.",
    )
    coverage.add_argument(
        "--frontier-beta-beta",
        type=float,
        default=1.0,
        help="Beta prior beta for Laplace smoothing of frontier batch fail rate.",
    )
    coverage.add_argument(
        "--frontier-variance-weight",
        type=float,
        default=1.0,
        help="Weight λ_frontier for frontier-gated unverified-tail variance in FoM.",
    )
    coverage.add_argument(
        "--tail-closure-ratio",
        type=float,
        default=0.1,
        help=(
            "Require the posterior expected failure mass still unverified in the "
            "retained tail to be at most this fraction of the Pf estimate before stopping."
        ),
    )
    coverage.add_argument(
        "--tail-closure-z",
        type=float,
        default=1.96,
        help="One-sided tail-closure uncertainty multiplier.",
    )
    coverage.add_argument(
        "--tail-closure-weight",
        type=float,
        default=0.03,
        help="FoM-scale weight for retained-tail closure excess.",
    )
    coverage.add_argument(
        "--tail-closure-fom-cap",
        type=float,
        default=0.12,
        help="Maximum FoM contribution from retained-tail closure excess.",
    )
    coverage.add_argument(
        "--tail-closure-decay",
        type=float,
        default=0.75,
        help=(
            "Round-to-round decay for the retained-tail FoM penalty.  Risk can "
            "increase immediately, but decreases are smoothed by this factor; "
            "0 restores the unsmoothed behavior."
        ),
    )
    coverage.add_argument("--failure-weight", type=float, default=3.0)
    coverage.add_argument("--boundary-weight", type=float, default=1.0)
    coverage.add_argument("--uncertainty-weight", type=float, default=0.3)
    coverage.add_argument("--residual-variance-weight", type=float, default=1.0)
    coverage.add_argument(
        "--max-spice",
        type=int,
        default=10000,
        help="Stop after this many total paper-counted SPICE calls; <=0 disables the budget cap.",
    )
    coverage.add_argument(
        "--min-rounds",
        type=int,
        default=1,
        help="Minimum coverage rounds before convergence can be declared.",
    )
    coverage.add_argument("--mc-batch", default="auto", help="Positive integer or auto")
    coverage.add_argument("--residual", action="store_true")
    coverage.add_argument("--residual-model-path", default="")
    coverage.add_argument("--residual-tail-n", type=int, default=1000)
    coverage.add_argument("--residual-uncertainty-n", type=int, default=500)
    coverage.add_argument(
        "--residual-candidate-k",
        type=int,
        default=10000,
        help="Base-model candidates kept before residual reranking.",
    )
    coverage.add_argument("--residual-epochs", type=int, default=300)
    coverage.add_argument("--residual-lr", type=float, default=2e-3)
    coverage.add_argument("--residual-val-ratio", type=float, default=0.2)
    coverage.add_argument(
        "--objective",
        choices=["auto", "max", "min"],
        default="auto",
        help="Tail direction; auto reads '# objective:' from the circuit descriptor.",
    )
    coverage.add_argument("--seed", type=int, default=42)
    coverage.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    coverage.add_argument("--spice-timeout", type=int, default=60)
    coverage.add_argument(
        "--output-json",
        default=str(DEFAULT_LOG_DIR / "tail_coverage_summary.json"),
    )
    coverage.add_argument(
        "--trace-tsv",
        default=str(DEFAULT_LOG_DIR / "tail_coverage_trace.tsv"),
    )
    coverage.set_defaults(func=command_tail_coverage)

    all_cmd = sub.add_parser("all", help="Build data, train, predict, and validate")
    all_cmd.add_argument("--circuit", required=True)
    all_cmd.add_argument("--train-n", type=int, default=1024)
    all_cmd.add_argument("--predict-n", type=int, default=100000)
    all_cmd.add_argument("--epochs", type=int, default=500)
    all_cmd.add_argument("--batch-size", type=int, default=128)
    all_cmd.add_argument("--mc-batch", default="auto", help="Positive integer or auto")
    all_cmd.add_argument("--validate-top-k", type=int, default=1000)
    all_cmd.add_argument(
        "--sigma-range",
        type=float,
        default=5.0,
        help="Truncated Gaussian range used to build the Sobol training set.",
    )
    all_cmd.add_argument(
        "--train-z-scale",
        type=float,
        default=1.0,
        help="Scale Sobol z-scores used for training only; MC remains unchanged.",
    )
    all_cmd.add_argument("--seed", type=int, default=42)
    all_cmd.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    all_cmd.add_argument("--threshold", type=float, default=None)
    all_cmd.add_argument("--tail-fail-weight", type=float, default=2.0)
    all_cmd.add_argument("--boundary-weight", type=float, default=1.0)
    all_cmd.add_argument("--boundary-sigma", type=float, default=1.0)
    all_cmd.add_argument("--val-ratio", type=float, default=0.0)
    add_model_ablation_args(all_cmd)
    all_cmd.add_argument(
        "--objective",
        choices=["auto", "max", "min"],
        default="auto",
        help="Tail direction; auto reads '# objective:' from the circuit descriptor.",
    )
    all_cmd.add_argument("--spice-timeout", type=int, default=60)
    all_cmd.add_argument(
        "--allow-failed-nominal",
        action="store_true",
        help="Keep old behavior: replace failed SPICE runs with the nominal measure.",
    )
    all_cmd.add_argument("--train-file", default="")
    all_cmd.add_argument("--ckpt", default="")
    all_cmd.add_argument("--no-build", action="store_true")
    all_cmd.add_argument("--no-train", action="store_true")
    all_cmd.add_argument("--keep-data", action="store_true")
    all_cmd.add_argument("--residual", action="store_true")
    all_cmd.add_argument("--residual-model-path", default="")
    all_cmd.add_argument("--residual-tail-n", type=int, default=1000)
    all_cmd.add_argument("--residual-uncertainty-n", type=int, default=500)
    all_cmd.add_argument("--residual-epochs", type=int, default=300)
    all_cmd.add_argument("--residual-lr", type=float, default=2e-3)
    all_cmd.add_argument(
        "--residual-val-ratio",
        type=float,
        default=0.2,
        help="Fraction of residual calibration SPICE points reserved for validation.",
    )
    all_cmd.set_defaults(func=command_all)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
