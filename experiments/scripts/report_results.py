"""Recompute the three estimators from saved SPICE measurements and compare references."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "results/conditional_triplet_46_47"
STRATIFIED = ROOT / "results/stratified_control_46_48"


def evaluate(directory, method, reference):
    run = json.loads((directory / "summary.json").read_text())
    threshold = run["threshold"]
    if threshold != reference["threshold"]:
        raise ValueError(f"threshold mismatch for {directory}")

    if method == "line":
        with np.load(directory / "records.npz") as data:
            physical = data["evalu_measure"]
            if not np.isfinite(physical).all() or not np.isfinite(data["pilot_measure"]).all():
                raise ValueError(f"invalid line probes in {directory}")
            correction = data["weight"] * ((physical > threshold) - data["control"])
            lines = data["nominal_control"] + correction.mean(axis=1)
            observations = lines.reshape(-1, 2).mean(axis=1)
            np.testing.assert_allclose(observations, data["pair_pf"])
            calls = run["training"] + run["feedback"] + physical.size + data["pilot_measure"].size
    else:
        with np.load(directory / "holdout.npz") as data:
            physical = data["measure"]
            if not np.isfinite(physical).all():
                raise ValueError(f"invalid holdout measurements in {directory}")
            weights = np.exp(data["log_weight"])
            hits = physical > threshold
            observations = weights * hits
            if method == "stratified":
                corrections = weights * (hits - data["control_probability"])
                blocks = corrections.reshape(run["independent_blocks"], -1)
                observations = data["control_nominal_mean"] + blocks.mean(axis=1)
                np.testing.assert_allclose(observations, data["block_pf"])
            calls = run["historical_training_calls"] + run["adaptation_calls"] + physical.size

    pf = float(observations.mean())
    se = float(observations.std(ddof=1) / np.sqrt(len(observations)))
    np.testing.assert_allclose([pf, se], [run["pf"], run["se"]], rtol=1e-10, atol=0)
    return pf, se, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=ROOT / "data/references.json")
    args = parser.parse_args()
    references = json.loads(args.references.read_text())["cases"]
    print("case      method          Pf          SE       FOM   SPICE   difference vs MC")
    for case in ("case18", "case243", "case1203"):
        reference = references[case]
        mc_pf = reference["published_mc_pf"]
        if not 0 < mc_pf <= 1:
            raise ValueError(f"invalid reference Pf for {case}")
        baseline = reference["original_hoist_pf_paper_rounded"]
        print(f"{case:<9} {'HOIST':<10} {baseline:.6e}           -         -   "
              f"{reference['original_hoist_charged_calls']:4d}   {100*(baseline/mc_pf-1):+.2f}%")
        for method, directory in (("direct", DIRECT / case), ("stratified", STRATIFIED / case),
                                  ("line", STRATIFIED / f"line_check_{case}_seed49")):
            pf, se, calls = evaluate(directory, method, reference)
            print(f"{case:<9} {method:<10} {pf:.6e} {se:.3e}   {se/pf:.4f}   "
                  f"{calls:4d}   {100*(pf/mc_pf-1):+.2f}%")


if __name__ == "__main__":
    main()
