"""Small CPU entry point for physical, proposal-versioned HOIST feedback IS."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy/hoist"
sys.path.insert(0, str(LEGACY))
from baselines.shared.circuit import CircuitTxtParser as OracleParser
from baselines.shared.oracle import SpiceOracle
from v16_surrogate.circuit import CircuitTxtParser
from v16_surrogate.cli import make_model_from_sdat
from v16_surrogate.eval import flat_z_to_dense


class Proposal(nn.Module):
    """Rank-one spline with a nominal orthogonal complement and 10% defensive mass."""
    def __init__(self, center):
        super().__init__()
        self.register_buffer("axis", F.normalize(center, dim=0))
        self.shift = nn.Parameter(center.norm())
        self.width = nn.Parameter(torch.zeros(8))
        self.height = nn.Parameter(torch.zeros(8))
        self.slope = nn.Parameter(torch.full((7,), math.log(math.expm1(.999))))

    def spline(self, x, inverse=False):
        params = [a.expand(len(x), -1) for a in (self.width, self.height, self.slope)]
        return unconstrained_rational_quadratic_spline(x, *params, inverse=inverse, tail_bound=6.)

    def log_ratio(self, x):
        """log(q/p), including the nominal component."""
        return self.axial_log_ratio(x @ self.axis)

    def axial_log_ratio(self, t):
        z, logdet = self.spline(t - self.shift, inverse=True)
        relative = .5 * (t.square() - z.square()) + logdet
        return torch.logaddexp(relative + math.log(.9), relative.new_tensor(math.log(.1)))

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, len(self.axis))
        t = z @ self.axis
        mapped, _ = self.spline(t)
        active = torch.rand(n) >= .1
        return z + ((mapped + self.shift - t) * active)[:, None] * self.axis


class ResidualHead(nn.Module):
    """Conditional logistic residual in units of the training response scale."""
    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        median, raw = self.linear(h).unbind(-1)
        return median, .01 + .15 * F.softplus(raw)


def feedback_axis_displacement(axis, ascent, shift, pf, iteration, gain):
    """Tangent ascent, scaled by failure mass and by the cost of rotating the shift."""
    if pf <= 0:
        return torch.zeros_like(axis)
    tangent = ascent - torch.dot(axis, ascent) * axis
    return (gain / math.sqrt(iteration)) * tangent / (pf * (1 + shift.square()))


def fit_pilot_variance(flow, failures, log_weights):
    """Fit the final axial density using the original generating weights of pilot hits."""
    x = torch.cat(failures)
    log_w = torch.cat(log_weights)
    if len(x) < 2:
        return dict(hits=len(x), fitted=False)
    with torch.no_grad():
        t = x @ flow.axis
        initial = [p.detach().clone() for p in flow.parameters()]
        before = torch.logsumexp(log_w - flow.axial_log_ratio(t), dim=0)
    # E_qi[I (p/qi) (p/q)] is the second moment of the final IS contribution
    # under q. Omitting safe points and the fixed denominator preserves its minimizer.
    optimizer = torch.optim.Adam(flow.parameters(), lr=.015)
    for _ in range(64):
        log_moment = torch.logsumexp(log_w - flow.axial_log_ratio(t), dim=0)
        penalty = sum((p-base).square().mean() for p, base in zip(flow.parameters(), initial))
        optimizer.zero_grad()
        (log_moment - before + .02 * penalty).backward()
        optimizer.step()
    with torch.no_grad():
        after = torch.logsumexp(log_w - flow.axial_log_ratio(t), dim=0)
        if not torch.isfinite(after) or after > before:
            for p, base in zip(flow.parameters(), initial):
                p.copy_(base)
            return dict(hits=len(x), fitted=False)
        return dict(hits=len(x), fitted=True, pilot_second_moment_ratio=float((after-before).exp()))


@torch.no_grad()
def fit_conditional_variance(flow, samples, labels):
    """Approximate the optimal axial density p(t) sqrt(P(failure | t))."""
    from scipy.optimize import minimize
    from scipy.special import log_ndtr, ndtr
    from nflows.transforms.splines.rational_quadratic import (
        DEFAULT_MIN_BIN_HEIGHT, DEFAULT_MIN_BIN_WIDTH, DEFAULT_MIN_DERIVATIVE,
    )

    t = (torch.cat(samples) @ flow.axis).double().numpy()
    event = torch.cat(labels).numpy()
    if min(event.sum(), (~event).sum()) < 2:
        return None
    center, spread = t.mean(), t.std()
    if spread < 1e-8:
        return None
    u = (t - center) / spread
    sign = np.where(event, 1., -1.)

    def loss(coeff):
        return -log_ndtr(sign * (coeff[0] + coeff[1] * u)).sum() + .001 * (coeff @ coeff)

    fit = minimize(loss, np.zeros(2), method="L-BFGS-B")
    if not fit.success:
        return None
    intercept, slope = fit.x
    # Quantile matching sets the knots of the existing invertible spline.
    # Its nominal mixture and its exact IS density remain part of Proposal.
    grid = np.linspace(-14., 14., 14001)
    log_target = -.5 * grid**2 + .5 * log_ndtr(intercept + slope * (grid-center)/spread)
    density = np.exp(log_target - log_target.max())
    cdf = np.r_[0., np.cumsum(.5 * (density[:-1] + density[1:]) * (grid[1]-grid[0]))]
    density /= cdf[-1]
    cdf /= cdf[-1]
    knots = np.arange(-3., 4.)
    quantiles = np.interp(ndtr(knots), cdf, grid)
    shift = quantiles[len(knots)//2]
    width = np.diff(np.r_[-6., knots, 6.])/12 - DEFAULT_MIN_BIN_WIDTH
    height = np.diff(np.r_[-6., quantiles-shift, 6.])/12 - DEFAULT_MIN_BIN_HEIGHT
    derivative = (np.exp(-.5 * knots**2) / math.sqrt(2 * math.pi)
                  / np.interp(quantiles, grid, density)) - DEFAULT_MIN_DERIVATIVE
    for v in (width, height, derivative):
        if not np.isfinite(v).all() or np.any(v <= 0):
            return None
    flow.shift.fill_(shift)
    flow.width.copy_(torch.from_numpy(np.log(width)))
    flow.height.copy_(torch.from_numpy(np.log(height)))
    flow.slope.copy_(torch.from_numpy(derivative + np.log(-np.expm1(-derivative))))
    return dict(method="conditional_probit", fitted=True, hits=int(event.sum()),
                probit_intercept=float(intercept - slope*center/spread),
                probit_slope=float(slope/spread), proposal_median=float(shift))


def run(args, out, oracle_factory=SpiceOracle):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    case = json.loads((ROOT / "data/legacy_manifest.json").read_text())["cases"][args.case]
    circuit = CircuitTxtParser.parse(LEGACY / case["circuit"])
    spec = OracleParser.parse(LEGACY / case["circuit"])
    shutil.copytree(spec.spice_dir, out / "spice")
    oracle = oracle_factory(replace(spec, spice_dir=out / "spice"), args.rounds * args.batch)
    model, ds = make_model_from_sdat(LEGACY / case["train"], LEGACY / case["ckpt"], "cpu")
    model.eval().requires_grad_(False)
    nom, scale = float(ds.stats["nom_measure"]), float(ds.stats["dev_std"])
    threshold = (case["threshold"] - nom) / scale
    mask = torch.arange(ds.max_n_params)[None, :] < ds.mos_n_params[:, None]

    @torch.no_grad()
    def encode(x):
        features = []
        for part in x.split(256):
            dense = torch.from_numpy(flat_z_to_dense(circuit, part.numpy()))
            features.append(model.encode_pooled(dense, mask.float().expand(len(part), -1, -1),
                                                ds.type_ids.long().expand(len(part), -1)))
        h = torch.cat(features)
        return h, model.regressor(h).flatten()

    pool = torch.randn(args.pool, spec.dimension)
    h, pred = encode(pool)
    # Freeze feature normalization so that historical residuals remain comparable.
    mean, std = h.mean(0), h.std(0).clamp_min(.05)
    head = ResidualHead(h.size(1))
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    h = (h - mean) / std
    initial = torch.topk(pred, args.k).indices
    flow = Proposal(pool[initial].mean(0))
    gradients = {}

    def candidate_axis(indices):
        mean_axis = F.normalize(pool[indices].mean(0), dim=0)
        if args.axis == "mean":
            return mean_axis
        ids = indices.tolist()
        missing = [i for i in ids if i not in gradients]
        if missing:
            with torch.enable_grad():
                inputs = torch.from_numpy(flat_z_to_dense(circuit, pool[missing].numpy())).requires_grad_()
                values = model.forward_dense(inputs, mask.float().expand(len(missing), -1, -1),
                                             ds.type_ids.long().expand(len(missing), -1))
                dense_grad, = torch.autograd.grad(values.sum(), inputs)
            gradients.update(zip(missing, F.normalize(dense_grad[:, mask], dim=1)))
        direction = torch.stack([gradients[i] for i in ids]).mean(0)
        return F.normalize(direction, dim=0) if direction.norm() > 1e-8 else mean_axis

    joint_axis = args.axis == "feedback"
    if args.axis in ("gradient", "feedback"):
        flow.axis.copy_(candidate_axis(initial))
    pilot_fit = getattr(args, "pilot_fit", False)
    conditional_fit = getattr(args, "conditional_fit", False)
    pilot_samples, pilot_labels = [], []
    pilot_hits, pilot_log_weights = [], []
    if pilot_fit:
        with torch.no_grad():
            initial_norm = flow.shift.item()
            flow.shift.copy_(pool[initial].mean(0) @ flow.axis)
            initial_projection = flow.shift.item()
    if joint_axis:
        # A leaf buffer permits a separate, sphere-constrained update. Adam only
        # handles the shift and spline; it must not take a Euclidean axis step.
        flow.axis.requires_grad_(True)
    flow_opt = torch.optim.Adam(flow.parameters(), lr=.025)
    with torch.no_grad():
        probe = torch.linspace(-5, 5, 101)
        mapped, jac = flow.spline(probe)
        back, invjac = flow.spline(mapped, inverse=True)
        assert torch.allclose(probe, back, atol=2e-5) and (jac + invjac).abs().max() < 2e-5
    history_h, history_e, contributions = [], [], []
    stable = 0
    version = 0
    rows = []
    fixed_updates = getattr(args, "adapt_rounds", None)
    if pilot_fit and fixed_updates is None:
        fixed_updates = args.rounds - 1
    status = "budget_exhausted" if fixed_updates is None else "fixed_budget_complete"
    for t in range(args.rounds):
        x = flow.sample(args.batch)
        with torch.no_grad():
            ratio = flow.log_ratio(x)
            weight = (-ratio.double()).exp()
            assert torch.isfinite(weight).all() and weight.max() <= 10.0001
            hx, y = encode(x)
            hx = (hx - mean) / std
            b, s = head(hx)
            score = torch.sigmoid((y + b - threshold) / s)
        physical = oracle.evaluate(x.numpy(), f"feedback round {t+1}")
        if not physical.valid.all():
            raise RuntimeError(f"Invalid ngspice response in round {t+1}, run directory {out}")
        truth = torch.tensor(physical.measure, dtype=torch.float64)
        hits = truth > case["threshold"]
        if pilot_fit:
            pilot_hits.append(x[hits])
            pilot_log_weights.append(-ratio[hits].double())
        if conditional_fit:
            pilot_samples.append(x)
            pilot_labels.append(hits)
        contributions.append(weight * hits)
        vals = torch.cat(contributions)
        pf = vals.mean().item()
        fom = (vals.std() / math.sqrt(len(vals)) / pf).item() if pf else None
        ess = (vals.sum().square() / vals.square().sum()).item() if pf else 0.
        ok = fom is not None and fom <= args.fom and ess >= 10 and (vals > 0).sum() >= 10
        stable = stable + 1 if ok else 0
        np.savez(out / f"round_{t+1}.npz", x=x.numpy(), measure=physical.measure,
                 prediction=(y*scale+nom).numpy(), residual_median=(b*scale).numpy(),
                 residual_q90=((b+math.log(9)*s)*scale).numpy(), weight=weight.numpy(),
                 log_q=(ratio-.5*(x.square().sum(1)+x.size(1)*math.log(2*math.pi))).numpy())
        row = dict(round=t+1, version=version, hits=int(hits.sum()), calls=oracle.calls,
                   pf=pf, fom=fom, tail_ess=ess)
        rows.append(row)
        if fixed_updates is None:
            if stable >= 2:
                status = "fom_reached"
                break
            if stable or t+1 == args.rounds:
                continue
        elif t >= fixed_updates:
            continue
        # Use the pre-feedback head for BOTH parts of the forward-KL gradient.
        cheap = torch.randn(args.pool, spec.dimension)
        hc, yc = encode(cheap)
        with torch.no_grad():
            bc, sc = head((hc-mean)/std)
            soft = torch.sigmoid((yc+bc-threshold)/sc)
            correction = weight.float() * (hits.float()-score)
        objective = (soft * flow.log_ratio(cheap)).mean() + (correction * flow.log_ratio(x)).mean()
        flow_opt.zero_grad()
        if joint_axis:
            flow.axis.grad = None
        (-objective).backward()
        if joint_axis:
            with torch.no_grad():
                a = flow.axis
                delta = feedback_axis_displacement(a, -a.grad, flow.shift, pf, t+1,
                                                   getattr(args, "axis_gain", .1))
                delta *= torch.clamp(math.tan(math.radians(2)) / delta.norm().clamp_min(1e-20), max=1.)
                row["axis_step_degrees"] = math.degrees(math.atan(float(delta.norm())))
                a.copy_(F.normalize(a + delta, dim=0))
        flow_opt.step()
        history_h.append(hx)
        history_e.append(((truth-nom)/scale-y).float())
        features, residual = torch.cat(history_h), torch.cat(history_e)
        for _ in range(12):
            loc, spread = head(features)
            z = (residual-loc)/spread
            loss = (spread.log()+F.softplus(z)+F.softplus(-z)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Replay the pool using the updated dangerous-side upper quantile.
        with torch.no_grad():
            loc, spread = head(h)
            promoted = torch.topk(pred+loc+math.log(9)*spread, args.k).indices
            if not joint_axis:
                direction = candidate_axis(promoted)
                flow.axis.copy_(F.normalize(.8*flow.axis+.2*direction, dim=0))
            row["new_topk_vs_initial"] = int((~torch.isin(promoted, initial)).sum())
            row["density_change"] = float((flow.log_ratio(x)-ratio).abs().max())
        version += 1
    report = dict(case=args.case, threshold=case["threshold"], config=vars(args),
                  status=status, historical_training_calls=case["train_spice"], rounds=rows)
    if pilot_fit:
        fitted = fit_conditional_variance(flow, pilot_samples, pilot_labels) if conditional_fit else None
        if fitted is None:
            fitted = fit_pilot_variance(flow, pilot_hits, pilot_log_weights)
        report["pilot_fit"] = dict(initial_norm=initial_norm, initial_projection=initial_projection,
                                   **fitted)
    (out/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(out, status, rows[-1])
    flow.axis.requires_grad_(False)
    return flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case18","case243","case1203"], default="case18")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fom", type=float, default=.1)
    parser.add_argument("--axis", choices=["mean", "gradient", "feedback"], default="mean")
    parser.add_argument("--adapt-rounds", type=int, help="fixed number of updates, followed by frozen rounds")
    parser.add_argument("--axis-gain", type=float, default=.1, help="step multiplier for the feedback axis")
    parser.add_argument("--pilot-fit", action="store_true",
                        help="use a projected initial shift and fit the final density after a fixed pilot")
    parser.add_argument("--conditional-fit", action="store_true",
                        help="with --pilot-fit, smooth the axial failure probability using all pilot labels")
    args = parser.parse_args()
    if args.conditional_fit and not args.pilot_fit:
        parser.error("--conditional-fit requires --pilot-fit")
    if not (args.rounds > 0 and args.batch >= 2 and 0 < args.k < args.pool and args.fom > 0):
        parser.error("Require positive rounds/FOM, batch >= 2 and 0 < k < pool")
    if args.adapt_rounds is not None and not 0 <= args.adapt_rounds < args.rounds:
        parser.error("--adapt-rounds must be between zero and rounds-1")
    if not math.isfinite(args.axis_gain) or args.axis_gain <= 0:
        parser.error("--axis-gain must be positive and finite")
    run(args, Path(tempfile.mkdtemp(prefix="feedback_", dir=ROOT/"results")))
