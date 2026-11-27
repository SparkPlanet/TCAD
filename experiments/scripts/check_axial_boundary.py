"""Independent SPICE check of failure probability along a frozen proposal axis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.special import ndtr
import torch

from validate_frozen_feedback import LEGACY, ROOT, OracleParser, SpiceOracle, rebuild


@torch.no_grad()
def estimate(flow, fit, threshold, n_lines, rng, simulate):
    axis = flow.axis.double().numpy()
    axis = axis / np.linalg.norm(axis)
    z = rng.normal(size=(n_lines // 2, len(axis)))
    perpendicular = z - (z @ axis)[:, None] * axis
    perpendicular = np.stack([perpendicular, -perpendicular], axis=1).reshape(n_lines, -1)
    intercept, slope = fit["probit_intercept"], fit["probit_slope"]
    direction = np.sign(slope)
    center = -intercept / slope
    step = np.clip(1.5 / abs(slope), .1, 1.)

    def physical_at(t):
        x = (perpendicular[:, None, :] + t[:, :, None] * axis).reshape(-1, len(axis))
        return simulate(x).reshape(n_lines, 2)

    probes = np.broadcast_to(center + np.array([-step, step]), (n_lines, 2))
    y = physical_at(probes)
    gradient = (y[:, 1] - y[:, 0]) / (2 * step)
    crossing = np.full(n_lines, center)
    active = gradient * direction > 0
    crossing[active] = np.clip(center + (threshold - y[active].mean(1)) / gradient[active],
                               center - 3 * step, center + 3 * step)

    # Draw one sample from each proposal. The mixture in the weight is shared.
    normal = torch.from_numpy(rng.normal(size=n_lines)).float()
    tilted = torch.from_numpy(rng.random(n_lines) >= .1)
    mapped, _ = flow.spline(normal)
    broad = torch.where(tilted, mapped + flow.shift, normal).double().numpy()
    width = step / 2
    t = np.stack([broad, crossing + width * rng.normal(size=n_lines)], axis=1)
    truth = physical_at(t)
    broad_ratio = flow.axial_log_ratio(torch.from_numpy(t.flatten()).float()).double().numpy().reshape(n_lines, 2)
    local_ratio = -np.log(width) + .5 * (t*t - ((t - crossing[:, None]) / width)**2)
    weight = np.exp(np.log(2.) - np.logaddexp(broad_ratio, local_ratio))

    # The physical correction remains valid for multiple crossings on a line.
    soft = width / 4
    control = ndtr(direction * (t - crossing[:, None]) / soft)
    nominal = ndtr(-direction * crossing / np.hypot(1., soft))
    residual = (weight * ((truth > threshold) - control)).mean(axis=1)
    pairs = (nominal + residual).reshape(-1, 2).mean(axis=1)
    se = pairs.std(ddof=1) / np.sqrt(len(pairs))
    pf = float(pairs.mean())
    result = dict(pf=pf, se=float(se), fom=float(se / pf) if pf > 0 else None,
                  mean_line_control=float(nominal.mean()), mean_physical_correction=float(residual.mean()),
                  pilot_bracket_rate=float(((y[:, 0] > threshold) != (y[:, 1] > threshold)).mean()),
                  n_lines=n_lines, independent_pairs=len(pairs))
    records = dict(perp=perpendicular, pilot_t=probes, pilot_measure=y, evalu_t=t, evalu_measure=truth,
                   weight=weight, control=control, nominal_control=nominal, pair_pf=pairs, crossing=crossing)
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lines", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    report = json.loads((args.source / "summary.json").read_text())
    fit = report["pilot_fit"]
    if fit.get("method") != "conditional_probit" or abs(fit["probit_slope"]) <= 1e-6:
        parser.error("requires a fitted probit with a nonzero slope")
    if args.seed == report["config"]["seed"] or args.lines < 4 or args.lines % 2:
        parser.error("use a fresh seed and an even number of lines, at least four")
    if args.out.exists() or not 1 <= args.workers <= 2 * args.lines:
        parser.error("use a new output directory and a valid worker count")
    case = json.loads((ROOT / "data/legacy_manifest.json").read_text())["cases"][report["case"]]
    spec = OracleParser.parse(LEGACY / case["circuit"])
    with tempfile.TemporaryDirectory() as name:
        scratch = Path(name)
        flow = rebuild(args.source, scratch, report)
        workers = []
        for index in range(args.workers):
            directory = scratch / f"worker_{index}"
            shutil.copytree(spec.spice_dir, directory)
            workers.append(SpiceOracle(replace(spec, spice_dir=directory), 4 * args.lines))

        def simulate(x):
            def evaluate(item):
                oracle, points = item
                batch = oracle.evaluate(points, "line SPICE")
                if not batch.valid.all():
                    raise RuntimeError("nonfinite line measurement")
                return batch.measure
            with ThreadPoolExecutor(args.workers) as pool:
                return np.concatenate(list(pool.map(evaluate, zip(workers, np.array_split(x, args.workers)))))

        result, records = estimate(flow, fit, report["threshold"], args.lines, np.random.default_rng(args.seed), simulate)
        calls = sum(worker.calls for worker in workers)
        if calls != 4 * args.lines:
            raise RuntimeError("incomplete line simulation")
    feedback = report["rounds"][-1]["calls"]
    result.update(case=report["case"], threshold=report["threshold"], seed=args.seed, pilot_seed=report["config"]["seed"],
                  training=case["train_spice"], feedback=feedback, physical_calls=calls,
                  charged_calls=calls + case["train_spice"] + feedback)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "records.npz", **records)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result)


if __name__ == "__main__":
    main()
