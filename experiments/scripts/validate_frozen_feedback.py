"""Rebuild a recorded feedback proposal and estimate Pf with fresh fixed-budget SPICE."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from feedback_flow import LEGACY, OracleParser, SpiceOracle, run
from baselines.shared.oracle import OracleBatch


def rebuild(source, scratch, report):
    config = argparse.Namespace(**{"axis": "mean", **report["config"]})

    class RecordedOracle:
        def __init__(self, spec, budget):
            self.calls = 0
            self.round = 0

        def evaluate(self, x, description):
            self.round += 1
            with np.load(source / f"round_{self.round}.npz") as old:
                np.testing.assert_array_equal(x, old["x"])
                self.calls += len(x)
                return OracleBatch(z=x, measure=old["measure"])

    flow = run(config, scratch, RecordedOracle).eval().requires_grad_(False)
    replay = json.loads((scratch / "summary.json").read_text())
    if (replay["rounds"] != report["rounds"] or replay["status"] != report["status"]
            or replay.get("pilot_fit") != report.get("pilot_fit")):
        raise ValueError("feedback replay differs from source")
    return flow


def off_axis_check(source, report, axis, points, contributions):
    """Choose a direction with feedback labels alone, then test it on the holdout."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    moment = np.zeros_like(axis)
    mass = 0.
    for row in report["rounds"]:
        with np.load(source / f"round_{row['round']}.npz") as data:
            v = data["weight"] * (data["measure"] > report["threshold"])
            moment += v @ data["x"]
            mass += v.sum()
    residual = moment - (moment @ axis) * axis
    norm = np.linalg.norm(residual)
    if norm < 1e-12 or mass == 0 or contributions.sum() == 0:
        return {"status": "insufficient_direction_or_hits"}
    projected = points @ (residual / norm)
    average = np.average(projected, weights=contributions)
    influence = contributions * (projected - average) / contributions.mean()
    return dict(feedback_projection=float(norm/mass), holdout_projection=float(average),
                holdout_se=float(influence.std(ddof=1)/math.sqrt(len(influence))))


STRATIFIED_BLOCK = 80


@torch.no_grad()
def stratified_points(flow, n):
    """Independent blocks with stratified axial normals and antithetic orthogonal pairs."""
    from scipy.special import ndtri

    blocks = n // STRATIFIED_BLOCK
    # A block covers both mixture components in the exact 10:90 proportions.
    axis = flow.axis.double()
    axis = axis / axis.norm()
    positions = []
    for count, tilted in ((4, False), (36, True)):
        uniform = (torch.arange(count) + torch.rand(blocks, count, dtype=torch.float64)) / count
        uniform.clamp_(min=np.nextafter(0., 1.), max=np.nextafter(1., 0.))
        t = torch.from_numpy(ndtri(uniform.numpy()))
        if tilted:
            mapped, _ = flow.spline(t.float().flatten())
            t = (mapped + flow.shift).double().reshape_as(t)
        positions.append(t)
    t = torch.cat(positions, dim=1).flatten()
    z = torch.randn(len(t), len(axis), dtype=torch.float64)
    orthogonal = z - (z @ axis)[:, None] * axis
    center = t[:, None] * axis
    points = torch.stack((center+orthogonal, center-orthogonal), dim=1).flatten(0, 1)
    return points, t.repeat_interleave(2)


def controlled_estimate(t, hit, weights, fit):
    """Integrate the frozen probit analytically and correct it with real SPICE."""
    from scipy.special import ndtr

    intercept, slope = fit["probit_intercept"], fit["probit_slope"]
    probability = ndtr(intercept + slope * t)
    nominal_mean = float(ndtr(intercept / math.hypot(1., slope)))
    corrections = weights * (hit - probability)
    block_pf = nominal_mean + corrections.reshape(-1, STRATIFIED_BLOCK).mean(axis=1)
    se = float(block_pf.std(ddof=1) / math.sqrt(len(block_pf)))
    return float(block_pf.mean()), se, dict(control_probability=probability,
                control_nominal_mean=nominal_mean, axial_coordinate=t, block_pf=block_pf)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--samples", type=int, default=1024)
    cli.add_argument("--seed", type=int, default=43)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--stratified-control", action="store_true",
                     help="use 80-point independent blocks and a frozen axial probit control variate")
    args = cli.parse_args()
    report = json.loads((args.source / "summary.json").read_text())
    case_name = report["case"]
    if args.seed == report["config"]["seed"] or args.samples < 2 or not 1 <= args.workers <= args.samples:
        cli.error("require a new seed, at least two samples and a valid worker count")
    if args.output.exists():
        cli.error("select a new output directory")
    if args.stratified_control and (args.samples % STRATIFIED_BLOCK or args.samples < 2*STRATIFIED_BLOCK
                                    or report.get("pilot_fit", {}).get("method") != "conditional_probit"):
        cli.error("stratified control requires a fitted conditional probit and at least two full blocks")
    with tempfile.TemporaryDirectory() as name:
        scratch = Path(name)
        flow = rebuild(args.source, scratch, report)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            if args.stratified_control:
                x, axial = stratified_points(flow, args.samples)
                log_weight = -flow.axial_log_ratio(axial.float()).double().numpy()
            else:
                x = flow.sample(args.samples)
                log_weight = -flow.log_ratio(x).double().numpy()
        points = x.numpy()
        case = json.loads((ROOT / "data/legacy_manifest.json").read_text())["cases"][case_name]
        spec = OracleParser.parse(LEGACY / case["circuit"])

        def simulate(item):
            i, batch = item
            folder = scratch / f"worker_{i}"
            shutil.copytree(spec.spice_dir, folder)
            oracle = SpiceOracle(replace(spec, spice_dir=folder), len(batch))
            physical = oracle.evaluate(batch, f"holdout {i+1}")
            if not physical.valid.all() or oracle.calls != len(batch):
                raise RuntimeError("incomplete SPICE holdout")
            return physical.measure

        with ThreadPoolExecutor(args.workers) as workers:
            measurements = list(workers.map(simulate, enumerate(np.array_split(points, args.workers))))
    measure = np.concatenate(measurements)
    hit = measure > report["threshold"]
    values = np.exp(log_weight) * hit
    pf = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    extra = {}
    if args.stratified_control:
        pf, se, extra = controlled_estimate(axial.numpy(), hit, np.exp(log_weight), report["pilot_fit"])
    result = dict(source=args.source.name, case=case_name, seed=args.seed, threshold=report["threshold"],
                  historical_training_calls=report["historical_training_calls"],
                  adaptation_calls=report["rounds"][-1]["calls"], holdout_calls=len(values),
                  verified_replay_rounds=len(report["rounds"]), hits=int(hit.sum()), pf=pf, se=se,
                  fom=se/pf if pf > 0 else None,
                  tail_ess=float(values.sum()**2 / (values @ values)) if values.any() else 0.)
    if args.stratified_control:
        result.update(estimator="stratified_probit_control", raw_stratified_pf=float(values.mean()),
                      independent_blocks=len(extra["block_pf"]), control_nominal_mean=extra["control_nominal_mean"])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "holdout.npz", x=points, measure=measure, log_weight=log_weight, **extra)
    diagnostic = off_axis_check(args.source, report, flow.axis.numpy(), points, values)
    (args.output / "axis_check.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result)


if __name__ == "__main__":
    main()
