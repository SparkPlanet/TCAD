"""Check a completed feedback run against the per-batch physical records."""
import json
import math
from pathlib import Path
import sys

import numpy as np


def summarize(directory):
    report = json.loads((directory / "summary.json").read_text())
    # Historical HOIST Table 1 reference and Table 3 charged SPICE cost.
    reference, historical_cost = {"case18": (5.8e-5, 3748),
                                  "case243": (6.18e-5, 2848),
                                  "case1203": (5.238e-5, 5336)}[report["case"]]
    contributions = []
    worst_log_error = 0.
    for row in report["rounds"]:
        with np.load(directory / f"round_{row['round']}.npz") as batch:
            x = batch["x"].astype(np.float64)
            log_p = -.5 * (np.square(x).sum(axis=1) + x.shape[1]*math.log(2*math.pi))
            # log_q is stored in float32, so subtracting high-dimensional
            # log densities loses precision; the stored ratio is more accurate.
            log_error = np.abs(log_p - batch["log_q"] - np.log(batch["weight"]))
            worst_log_error = max(worst_log_error, float(log_error.max()))
            physical = batch["measure"]
            assert np.isfinite(physical).all()
            fail = physical > report["threshold"]
            assert fail.sum() == row["hits"]
            contributions.extend(fail * batch["weight"])
            v = np.asarray(contributions)
            pf = float(v.mean())
            assert np.isclose(pf, row["pf"], rtol=1e-10, atol=0)
            if pf > 0:
                assert np.isclose(v.std(ddof=1)/math.sqrt(len(v))/pf, row["fom"])
    total = len(v) + report["historical_training_calls"]
    assert worst_log_error < .002
    return dict(case=report["case"], status=report["status"], rounds=len(report["rounds"]),
                fresh_spice=len(v), total_spice=total, physical_hits=int(np.count_nonzero(v)),
                pf=pf, fom=row["fom"], tail_ess=row["tail_ess"], mc_reference=reference,
                relative_difference_pct=100*(pf/reference-1), original_hoist_spice=historical_cost,
                cost_change_pct=100*(total/historical_cost-1), max_log_weight_check_error=worst_log_error)


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, allow_nan=False))
