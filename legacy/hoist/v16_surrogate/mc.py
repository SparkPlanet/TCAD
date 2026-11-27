from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .circuit import CircuitSpec, SpiceRunner
from .data import SdatStats
from .model import CircuitSurrogateV16, resolve_device, set_seed


# ---------------------------------------------------------------------------
# MC top-k prediction, residual correction, and validation
# ---------------------------------------------------------------------------


@dataclass
class TopEntry:
    score_norm: float
    sample_idx: int
    z: np.ndarray
    base_norm: float
    delta_norm: float = 0.0
    margin_norm: float = 0.0
    flip_probability: float = 0.0
    acquisition_score: float = 0.0


class AttentionResidualModel(nn.Module):
    """Small attention model for top-region residual correction.

    It predicts delta = SPICE_norm - base_norm from frozen base-surrogate
    device features.
    """

    def __init__(
        self,
        hidden: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        self.hidden = hidden

        self.base_query = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        valid: torch.Tensor,
        base_norm: torch.Tensor,
    ) -> torch.Tensor:
        h = token_features
        query = self.base_query(base_norm).unsqueeze(1)
        key_padding_mask = valid.squeeze(-1) <= 0
        attn_out, _ = self.cross_attn(query, h, h, key_padding_mask=key_padding_mask)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        max_pooled = h.masked_fill(valid == 0, -1e9).max(dim=1).values
        max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))

        features = torch.cat([attn_out.squeeze(1), pooled, max_pooled, base_norm], dim=-1)
        return self.regressor(features)


class MCTopKPredictor:
    def __init__(
        self,
        model: CircuitSurrogateV16,
        type_ids: torch.Tensor,
        mos_n_params: torch.Tensor,
        device: str = "auto",
        objective: str = "max",
        residual_model_path: str | Path | None = None,
    ):
        if objective not in {"max", "min"}:
            raise ValueError("objective must be 'max' or 'min'")
        self.device = resolve_device(device)
        self.objective = objective
        self.objective_sign = 1.0 if objective == "max" else -1.0
        self.model = model.to(self.device).eval()
        self.type_ids = type_ids.to(self.device).long()
        self.mos_n_params = mos_n_params

        mask = torch.zeros(1, model.num_mos, model.max_n_params, device=self.device)
        for m, n_params in enumerate(mos_n_params.tolist()):
            mask[0, m, : int(n_params)] = 1.0
        self.mask = mask
        self.active_mos = self.model.active_mos.to(self.device)
        self.active_param = self.model.active_param.to(self.device)
        self.residual_model = (
            self._load_residual_model(residual_model_path)
            if residual_model_path
            else None
        )

    def _load_residual_model(self, model_path: str | Path) -> AttentionResidualModel:
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        hidden = int(checkpoint.get("hidden", self.model.hidden))
        if hidden != int(self.model.hidden):
            raise ValueError(
                f"residual hidden={hidden} does not match base hidden={self.model.hidden}"
            )
        residual = AttentionResidualModel(hidden=hidden).to(self.device)
        residual.load_state_dict(checkpoint["state_dict"])
        residual.eval()
        self.residual_alpha = float(checkpoint.get("alpha", 1.0))
        print(f"loaded residual model: {model_path}")
        print(f"residual alpha: {self.residual_alpha:.3f}")
        return residual

    def suggest_batch_size(self) -> int:
        dense_dim = max(1, int(self.model.num_mos * self.model.max_n_params))
        if self.residual_model is not None:
            target_dense_values = 750_000 if self.device.type == "cuda" else 250_000
            batch = target_dense_values // dense_dim
            return int(np.clip(batch, 256, 4_096))
        if self.device.type == "cuda":
            target_dense_values = 2_000_000
            batch = target_dense_values // dense_dim
            return int(np.clip(batch, 1_024, 65_536))
        target_dense_values = 1_000_000
        batch = target_dense_values // dense_dim
        return int(np.clip(batch, 1_024, 16_384))

    def _active_to_dense(self, active_params: torch.Tensor) -> torch.Tensor:
        dense = torch.zeros(
            active_params.shape[0],
            self.model.num_mos,
            self.model.max_n_params,
            device=self.device,
        )
        dense[:, self.active_mos, self.active_param] = active_params
        return dense

    def predict_topk(
        self,
        n_mc: int,
        top_k: int,
        batch_size: int,
        seed: int = 42,
        threshold_norm: float | None = None,
        failure_weight: float = 1.0,
        boundary_weight: float = 1.0,
    ) -> List[TopEntry]:
        if n_mc <= 0:
            raise ValueError("n_mc must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        top_k = min(top_k, n_mc)
        if batch_size <= 0:
            batch_size = self.suggest_batch_size()
            print(f"auto mc_batch={batch_size}")
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        heap: List[Tuple[float, int, np.ndarray, float, float, float, float]] = []
        global_idx = 0
        n_batches = (n_mc + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="MC top-k"):
            cur = min(batch_size, n_mc - batch_idx * batch_size)
            active_params = torch.randn(
                cur,
                self.model.total_active_params,
                device=self.device,
            )
            with torch.no_grad():
                params = self._active_to_dense(active_params)
                mask_b = self.mask.expand(cur, -1, -1)
                type_b = self.type_ids.unsqueeze(0).expand(cur, -1)
                base_pred = self.model.forward_dense(params, mask_b, type_b)
                sample_idx_np = np.arange(cur)
                z_np = params.cpu().numpy()

                if self.residual_model is not None:
                    token_features = self.model.encode_tokens(params, mask_b, type_b)
                    valid = (mask_b.sum(dim=-1, keepdim=True) > 0).float()
                    delta_pred = self.residual_model(
                        token_features,
                        valid,
                        base_pred.view(-1, 1),
                    ) * self.residual_alpha
                    pred = base_pred + delta_pred
                else:
                    delta_pred = torch.zeros_like(base_pred)
                    pred = base_pred

            pred_np = pred.view(-1).cpu().numpy()
            base_np = base_pred.view(-1).cpu().numpy()
            delta_np = delta_pred.view(-1).cpu().numpy()
            if threshold_norm is None:
                margin_np = pred_np * self.objective_sign
                utility_np = margin_np
            else:
                margin_np = self.objective_sign * (pred_np - float(threshold_norm))
                utility_np = failure_weight * margin_np + boundary_weight * (-np.abs(margin_np))
            threshold = heap[0][0] if len(heap) >= top_k else -np.inf
            for i in np.where(utility_np > threshold)[0]:
                score = float(pred_np[i])
                utility = float(utility_np[i])
                entry = (
                    utility,
                    global_idx + int(sample_idx_np[i]),
                    z_np[i].copy(),
                    score,
                    float(base_np[i]),
                    float(delta_np[i]),
                    float(margin_np[i]),
                )
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif utility > heap[0][0]:
                    heapq.heapreplace(heap, entry)
            global_idx += cur

        sorted_entries = sorted(heap, key=lambda item: item[0], reverse=True)
        return [
            TopEntry(
                score_norm=e[3],
                sample_idx=e[1],
                z=e[2],
                base_norm=e[4],
                delta_norm=e[5],
                margin_norm=e[6],
                acquisition_score=e[0],
            )
            for e in sorted_entries
        ]


class ResidualCorrector:
    """Optional v16 + attention residual correction for top-k entries."""

    def __init__(
        self,
        circuit: CircuitSpec,
        stats: SdatStats,
        type_ids: torch.Tensor,
        mos_n_params: torch.Tensor,
        base_model: CircuitSurrogateV16,
        timeout_s: int = 60,
        device: str = "auto",
        objective: str | None = None,
    ):
        self.circuit = circuit
        self.stats = stats
        self.runner = SpiceRunner(circuit, timeout_s=timeout_s)
        self.device = resolve_device(device)
        self.type_ids = type_ids.long()
        self.mos_n_params = mos_n_params.long()
        self.base_model = base_model.to(self.device).eval()

        mask = torch.zeros(circuit.num_mos, circuit.max_n_params)
        for m, n_params in enumerate(self.mos_n_params.tolist()):
            mask[m, : int(n_params)] = 1.0
        self.mask = mask
        self.objective = objective or circuit.objective
        if self.objective not in {"max", "min"}:
            raise ValueError("objective must be 'max' or 'min'")
        self.objective_sign = 1.0 if self.objective == "max" else -1.0
        self.calibration_results: List[Dict[str, float | str]] = []

    def fit_and_apply(
        self,
        entries: List[TopEntry],
        model_path: str | Path,
        tail_n: int = 1000,
        uncertainty_n: int = 500,
        epochs: int = 300,
        lr: float = 2e-3,
        batch_size: int = 64,
        seed: int = 42,
        val_ratio: float = 0.2,
        exclude_calibration: bool = True,
        threshold_norm: float | None = None,
        failure_weight: float = 1.0,
        boundary_weight: float = 1.0,
        uncertainty_weight: float = 1.0,
    ) -> List[TopEntry]:
        if len(entries) < 10:
            return entries
        tail_n = min(tail_n, len(entries))
        rng = np.random.default_rng(seed)
        body_pool = np.arange(tail_n, len(entries))
        body_pick = rng.choice(body_pool, size=min(uncertainty_n, len(body_pool)), replace=False)
        selected_idx = np.concatenate([np.arange(tail_n), body_pick])

        self.runner.write_models()
        z_rows: List[np.ndarray] = []
        base_rows: List[float] = []
        y_rows: List[float] = []
        weights: List[float] = []
        calibration_entries: List[TopEntry] = []
        for idx in tqdm(selected_idx, desc="SPICE residual"):
            entry = entries[int(idx)]
            measure = self.runner.run_with_params(self.runner.zscore_to_params(entry.z))
            if measure is None:
                continue
            spice_norm = (measure - self.stats.nom_measure) / self.stats.dev_std
            z_rows.append(entry.z.astype(np.float32))
            base_rows.append(float(entry.base_norm))
            y_rows.append(float(spice_norm - entry.base_norm))
            weights.append(3.0 if idx < tail_n else 1.0)
            calibration_entries.append(entry)

        if len(z_rows) < 10:
            return entries

        weights_np = np.asarray(weights, dtype=np.float32)
        train_idx, val_idx = self._split_calibration(weights_np, val_ratio, rng)

        set_seed(seed)
        model = AttentionResidualModel(
            hidden=self.base_model.hidden,
            num_heads=4,
            dropout=0.1,
        ).to(self.device)
        z_np = np.stack(z_rows)
        base_np = np.asarray(base_rows, dtype=np.float32)
        target_np = np.asarray(y_rows, dtype=np.float32)
        feature_np, valid_np = self._features_for_arrays(
            z_np,
            batch_size=512,
            desc="residual feature build",
        )
        self._train_residual_model(
            model=model,
            features=feature_np[train_idx],
            valid=valid_np[train_idx],
            base=base_np[train_idx],
            target=target_np[train_idx],
            weights=weights_np[train_idx],
            val_features=feature_np[val_idx],
            val_valid=valid_np[val_idx],
            val_base=base_np[val_idx],
            val_target=target_np[val_idx],
            val_weights=weights_np[val_idx],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        val_delta = self._predict_feature_arrays(
            model,
            feature_np[val_idx],
            valid_np[val_idx],
            base_np[val_idx],
            batch_size=512,
            desc="residual val predict",
        )
        val_target = target_np[val_idx]
        alpha = self._select_alpha(val_delta, val_target)
        calibration_delta = self._predict_feature_arrays(
            model,
            feature_np,
            valid_np,
            base_np,
            batch_size=512,
            desc="residual calibration predict",
        ) * alpha
        flip_edges: np.ndarray | None = None
        flip_probs: np.ndarray | None = None
        if threshold_norm is not None and len(val_idx) > 0:
            val_cal = base_np[val_idx] + calibration_delta[val_idx]
            val_margin = self.objective_sign * (val_cal - float(threshold_norm))
            val_residual = self.objective_sign * (
                target_np[val_idx] - calibration_delta[val_idx]
            )
            flip_edges, flip_probs = self._fit_flip_bins(val_margin, val_residual)
        val_members = {int(i) for i in val_idx.tolist()}
        self.calibration_results = []
        for i, (entry, base_norm, target_delta, delta_norm) in enumerate(
            zip(calibration_entries, base_rows, y_rows, calibration_delta)
        ):
            self.calibration_results.append(
                {
                    "source": "calibration_val" if i in val_members else "calibration_train",
                    "sample_idx": float(entry.sample_idx),
                    "pred_norm": float(base_norm + delta_norm),
                    "base_norm": float(base_norm),
                    "measure": float(
                        (base_norm + target_delta) * self.stats.dev_std
                        + self.stats.nom_measure
                    ),
                }
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": self.base_model.hidden,
                "alpha": alpha,
                "flip_edges": flip_edges if flip_edges is not None else np.asarray([]),
                "flip_probs": flip_probs if flip_probs is not None else np.asarray([]),
            },
            model_path,
        )

        delta = self._predict_delta(
            model,
            entries,
            batch_size=512,
            desc="residual full rerank",
        ) * alpha
        calibrated = np.asarray([entry.base_norm for entry in entries], dtype=np.float64) + delta
        if threshold_norm is None:
            margins = calibrated * self.objective_sign
            flip_probability = np.zeros_like(margins)
            acquisition = margins
        else:
            margins = self.objective_sign * (calibrated - float(threshold_norm))
            flip_probability = self._predict_flip_probability(margins, flip_edges, flip_probs)
            acquisition = self._rank_acquisition(
                margins,
                flip_probability,
                failure_weight=failure_weight,
                boundary_weight=boundary_weight,
                uncertainty_weight=uncertainty_weight,
            )
        calibration_positions = {int(i) for i in selected_idx.tolist()}
        corrected = [
            TopEntry(
                score_norm=float(score_i),
                sample_idx=entry.sample_idx,
                z=entry.z,
                base_norm=entry.base_norm,
                delta_norm=float(delta_i),
                margin_norm=float(margin_i),
                flip_probability=float(pi_i),
                acquisition_score=float(acq_i),
            )
            for position, (entry, delta_i, score_i, margin_i, pi_i, acq_i) in enumerate(
                zip(entries, delta, calibrated, margins, flip_probability, acquisition)
            )
            if (not exclude_calibration) or position not in calibration_positions
        ]
        print(
            f"residual calibration: train={len(train_idx)}, val={len(val_idx)}, "
            f"excluded={len(calibration_positions)}, remaining={len(corrected)}"
        )
        return sorted(
            corrected,
            key=lambda entry: entry.acquisition_score,
            reverse=True,
        )

    @staticmethod
    def _fit_flip_bins(
        margins: np.ndarray,
        residuals: np.ndarray,
        n_bins: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        if margins.size == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        edges = np.unique(np.quantile(margins, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            eps = max(1e-6, abs(float(margins[0])) * 1e-6)
            edges = np.asarray([float(margins[0]) - eps, float(margins[0]) + eps])
        bin_ids = np.clip(np.searchsorted(edges, margins, side="right") - 1, 0, edges.size - 2)
        counts = np.zeros(edges.size - 1, dtype=np.float64)
        flips = np.zeros(edges.size - 1, dtype=np.float64)
        for bin_id, margin, residual in zip(bin_ids, margins, residuals):
            counts[int(bin_id)] += 1.0
            flips[int(bin_id)] += float(residual >= -margin)
        return edges.astype(np.float64), ((flips + 1.0) / (counts + 2.0)).astype(np.float64)

    @staticmethod
    def _predict_flip_probability(
        margins: np.ndarray,
        edges: np.ndarray | None,
        probs: np.ndarray | None,
    ) -> np.ndarray:
        if edges is None or probs is None or len(edges) < 2 or len(probs) == 0:
            return np.full_like(margins, 0.5, dtype=np.float64)
        ids = np.clip(np.searchsorted(edges, margins, side="right") - 1, 0, len(probs) - 1)
        return probs[ids].astype(np.float64)

    @staticmethod
    def _rank_acquisition(
        margins: np.ndarray,
        flip_probability: np.ndarray,
        *,
        failure_weight: float,
        boundary_weight: float,
        uncertainty_weight: float,
    ) -> np.ndarray:
        n = len(margins)
        if n == 0:
            return np.asarray([], dtype=np.float64)
        denom = max(n - 1, 1)
        failure_rank = np.empty(n, dtype=np.float64)
        boundary_rank = np.empty(n, dtype=np.float64)
        uncertainty_rank = np.empty(n, dtype=np.float64)
        failure_rank[np.argsort(margins)] = np.arange(n, dtype=np.float64) / denom
        boundary_rank[np.argsort(-np.abs(margins))] = np.arange(n, dtype=np.float64) / denom
        uncertainty_rank[np.argsort(flip_probability)] = np.arange(n, dtype=np.float64) / denom
        return (
            failure_weight * failure_rank
            + boundary_weight * boundary_rank
            + uncertainty_weight * uncertainty_rank
        )

    @staticmethod
    def _split_calibration(
        weights: np.ndarray,
        val_ratio: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("residual val_ratio must be in (0, 1)")
        train_parts = []
        val_parts = []
        for weight in np.unique(weights):
            group = np.flatnonzero(weights == weight)
            group = rng.permutation(group)
            val_n = min(len(group) - 1, max(1, int(round(len(group) * val_ratio))))
            val_parts.append(group[:val_n])
            train_parts.append(group[val_n:])
        return np.concatenate(train_parts), np.concatenate(val_parts)

    @staticmethod
    def _select_alpha(pred_delta: np.ndarray, target_delta: np.ndarray) -> float:
        alphas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        losses = np.asarray(
            [np.mean((alpha * pred_delta - target_delta) ** 2) for alpha in alphas]
        )
        alpha = float(alphas[int(np.argmin(losses))])
        print(
            "residual alpha validation: "
            + ", ".join(
                f"{float(a):.2f}={float(loss):.6e}"
                for a, loss in zip(alphas, losses)
            )
        )
        print(f"selected residual alpha={alpha:.2f}")
        return alpha

    def _train_residual_model(
        self,
        model: AttentionResidualModel,
        features: np.ndarray,
        valid: np.ndarray,
        base: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        val_features: np.ndarray,
        val_valid: np.ndarray,
        val_base: np.ndarray,
        val_target: np.ndarray,
        val_weights: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> None:
        feature_t = torch.from_numpy(features).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        base_t = torch.from_numpy(base).view(-1, 1).to(self.device)
        target_t = torch.from_numpy(target).view(-1, 1).to(self.device)
        weight_t = torch.from_numpy(weights).view(-1, 1).to(self.device)
        val_feature_t = torch.from_numpy(val_features).to(self.device)
        val_valid_t = torch.from_numpy(val_valid).to(self.device)
        val_base_t = torch.from_numpy(val_base).view(-1, 1).to(self.device)
        val_target_t = torch.from_numpy(val_target).view(-1, 1).to(self.device)
        val_weight_t = torch.from_numpy(val_weights).view(-1, 1).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=lr * 0.05
        )
        n = feature_t.size(0)
        best_loss = float("inf")
        best_state = None
        patience = max(30, epochs // 5)
        wait = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                pred = model(feature_t[idx], valid_t[idx], base_t[idx])
                loss = ((pred - target_t[idx]) ** 2 * weight_t[idx]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.item()) * idx.numel()
            scheduler.step()
            avg_loss = epoch_loss / n
            model.eval()
            with torch.no_grad():
                val_pred = model(val_feature_t, val_valid_t, val_base_t)
                val_loss = float(
                    (((val_pred - val_target_t) ** 2) * val_weight_t).mean().item()
                )
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if epoch % 50 == 0:
                print(
                    f"residual epoch {epoch:03d} | train={avg_loss:.6e} | "
                    f"val={val_loss:.6e}"
                )
            if wait >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        print(
            f"attention residual best_val_loss={best_loss:.6e}, "
            f"train={n}, val={len(val_features)}"
        )

    def _predict_delta(
        self,
        model: AttentionResidualModel,
        entries: Sequence[TopEntry],
        batch_size: int,
        desc: str = "residual apply",
    ) -> np.ndarray:
        model.eval()
        out: List[np.ndarray] = []
        total = len(entries)
        with torch.no_grad():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start : start + batch_size]
                z_np = np.stack([e.z for e in chunk]).astype(np.float32)
                features_np, valid_np = self._features_for_arrays(
                    z_np,
                    batch_size=len(chunk),
                    desc="",
                )
                features = torch.from_numpy(features_np).to(self.device)
                valid = torch.from_numpy(valid_np).to(self.device)
                base = torch.tensor([[e.base_norm] for e in chunk], dtype=torch.float32, device=self.device)
                pred = model(features, valid, base)
                out.append(pred.view(-1).cpu().numpy())
                done = min(start + len(chunk), total)
                batch_idx = start // batch_size
                if batch_idx == 0 or done == total or (batch_idx + 1) % 10 == 0:
                    print(f"{desc}: {done}/{total}")
        return np.concatenate(out)

    def _features_for_arrays(
        self,
        z: np.ndarray,
        batch_size: int,
        desc: str = "base feature",
    ) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        mask = self.mask.to(self.device)
        type_ids = self.type_ids.to(self.device)
        with torch.no_grad():
            for start in range(0, z.shape[0], batch_size):
                chunk = torch.from_numpy(z[start : start + batch_size].astype(np.float32)).to(self.device)
                mask_b = mask.unsqueeze(0).expand(chunk.size(0), -1, -1)
                type_b = type_ids.unsqueeze(0).expand(chunk.size(0), -1)
                token = self.base_model.encode_tokens(chunk, mask_b, type_b)
                valid = (mask_b.sum(dim=-1, keepdim=True) > 0).float()
                features.append(token.cpu().numpy().astype(np.float32))
                valids.append(valid.cpu().numpy().astype(np.float32))
                if desc:
                    done = min(start + chunk.size(0), z.shape[0])
                    print(f"{desc}: {done}/{z.shape[0]}")
        return np.concatenate(features, axis=0), np.concatenate(valids, axis=0)

    def _predict_feature_arrays(
        self,
        model: AttentionResidualModel,
        features: np.ndarray,
        valid: np.ndarray,
        base: np.ndarray,
        batch_size: int,
        desc: str = "residual eval",
    ) -> np.ndarray:
        out: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                feature_t = torch.from_numpy(features[start : start + batch_size]).to(self.device)
                valid_t = torch.from_numpy(valid[start : start + batch_size]).to(self.device)
                base_t = torch.from_numpy(base[start : start + batch_size]).view(-1, 1).to(self.device)
                out.append(model(feature_t, valid_t, base_t).view(-1).cpu().numpy())
                if desc:
                    done = min(start + batch_size, features.shape[0])
                    print(f"{desc}: {done}/{features.shape[0]}")
        return np.concatenate(out)


class SpiceValidator:
    def __init__(
        self,
        circuit: CircuitSpec,
        stats: SdatStats,
        timeout_s: int = 60,
        objective: str | None = None,
    ):
        self.circuit = circuit
        self.stats = stats
        self.runner = SpiceRunner(circuit, timeout_s=timeout_s)
        self.objective = objective or circuit.objective
        if self.objective not in {"max", "min"}:
            raise ValueError("objective must be 'max' or 'min'")

    def validate(self, entries: Sequence[TopEntry], top_k: int) -> List[Dict[str, float]]:
        self.runner.write_models()
        results: List[Dict[str, float]] = []
        for rank, entry in enumerate(tqdm(entries[:top_k], desc="SPICE validate"), start=1):
            measure = self.runner.run_with_params(self.runner.zscore_to_params(entry.z))
            if measure is None:
                continue
            results.append(
                {
                    "rank": float(rank),
                    "sample_idx": float(entry.sample_idx),
                    "pred_norm": float(entry.score_norm),
                    "base_norm": float(entry.base_norm),
                    "measure": float(measure),
                }
            )
        return results

    def validate_dataset(
        self,
        entries: Sequence[TopEntry],
        top_k: int,
        output_path: str | Path,
        metadata: Dict[str, Any] | None = None,
        checkpoint_interval: int = 100,
    ) -> List[Dict[str, float]]:
        """Validate and persist a reusable surrogate-screened tail data set.

        Existing rows are resumed by sample index. Every requested candidate is
        retained; failed SPICE runs have a NaN measure and can be retried by
        deleting the output or by passing a fresh output path.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = list(entries[: min(top_k, len(entries))])
        saved: Dict[int, float] = {}
        if output.exists():
            loaded = np.load(output, allow_pickle=False)
            if "sample_idx" in loaded and "measure" in loaded:
                current_indices = np.asarray(
                    [entry.sample_idx for entry in selected], dtype=np.int64
                )
                saved_indices = np.asarray(loaded["sample_idx"], dtype=np.int64)
                if not np.array_equal(saved_indices, current_indices):
                    raise ValueError(
                        f"{output} belongs to a different screened candidate "
                        "set; use a new --validation-data path"
                    )
                saved = {
                    int(index): float(measure)
                    for index, measure in zip(
                        loaded["sample_idx"], loaded["measure"]
                    )
                    if np.isfinite(measure)
                }
            print(f"resuming {output}: {len(saved)} valid SPICE rows")

        measures = np.full(len(selected), np.nan, dtype=np.float64)
        for position, entry in enumerate(selected):
            if entry.sample_idx in saved:
                measures[position] = saved[entry.sample_idx]

        self.runner.write_models()
        pending = np.flatnonzero(~np.isfinite(measures))
        completed_since_save = 0
        for position in tqdm(pending, desc="SPICE tail reference"):
            entry = selected[int(position)]
            measure = self.runner.run_with_params(self.runner.zscore_to_params(entry.z))
            if measure is not None:
                measures[position] = measure
            completed_since_save += 1
            if completed_since_save >= checkpoint_interval:
                self._write_validation_dataset(
                    output, selected, measures, metadata or {}
                )
                completed_since_save = 0
        self._write_validation_dataset(output, selected, measures, metadata or {})

        results: List[Dict[str, float]] = []
        for rank, (entry, measure) in enumerate(
            zip(selected, measures), start=1
        ):
            if not np.isfinite(measure):
                continue
            results.append(
                {
                    "rank": float(rank),
                    "sample_idx": float(entry.sample_idx),
                    "pred_norm": float(entry.score_norm),
                    "base_norm": float(entry.base_norm),
                    "measure": float(measure),
                }
            )
        return results

    def _write_validation_dataset(
        self,
        output: Path,
        entries: Sequence[TopEntry],
        measures: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        if not entries:
            raise ValueError("cannot save an empty validation data set")
        temporary = output.with_name(output.name + ".tmp.npz")
        pred_norm = np.asarray(
            [entry.score_norm for entry in entries], dtype=np.float64
        )
        payload = {
            "format_version": np.asarray(1, dtype=np.int64),
            "sample_idx": np.asarray(
                [entry.sample_idx for entry in entries], dtype=np.int64
            ),
            "pred_rank": np.arange(1, len(entries) + 1, dtype=np.int64),
            "z": np.stack([entry.z for entry in entries]).astype(
                np.float32, copy=False
            ),
            "pred_norm": pred_norm,
            "base_norm": np.asarray(
                [entry.base_norm for entry in entries], dtype=np.float64
            ),
            "delta_norm": np.asarray(
                [entry.delta_norm for entry in entries], dtype=np.float64
            ),
            "pred_measure": (
                pred_norm * self.stats.dev_std + self.stats.nom_measure
            ),
            "measure": np.asarray(measures, dtype=np.float64),
            "valid": np.isfinite(measures),
            "nom_measure": np.asarray(self.stats.nom_measure),
            "dev_std": np.asarray(self.stats.dev_std),
            "objective": np.asarray(self.objective),
        }
        for key, value in metadata.items():
            if value is not None:
                payload[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(temporary, **payload)
        temporary.replace(output)

    def report(
        self,
        results: Sequence[Dict[str, float]],
        log_path: str | Path,
        calibration_results: Sequence[Dict[str, float | str]] = (),
    ) -> None:
        if len(results) < 2:
            print(f"Not enough valid SPICE runs: {len(results)}")
            return

        pred_norm = np.asarray([r["pred_norm"] for r in results])
        base_norm = np.asarray([r["base_norm"] for r in results])
        measure = np.asarray([r["measure"] for r in results])
        measure_norm = (measure - self.stats.nom_measure) / self.stats.dev_std
        pred_phys = pred_norm * self.stats.dev_std + self.stats.nom_measure
        base_phys = base_norm * self.stats.dev_std + self.stats.nom_measure

        pearson = float(np.corrcoef(pred_norm, measure_norm)[0, 1])
        spearman = float(spearmanr(pred_norm, measure_norm)[0])
        mae = float(np.mean(np.abs(pred_phys - measure)))
        base_pearson = float(np.corrcoef(base_norm, measure_norm)[0, 1])
        base_spearman = float(spearmanr(base_norm, measure_norm)[0])
        base_mae = float(np.mean(np.abs(base_phys - measure)))
        real_order = np.argsort(measure)
        if self.objective == "max":
            real_order = real_order[::-1]
        real_rank = {int(i): rank + 1 for rank, i in enumerate(real_order)}

        combined_by_sample: Dict[int, Dict[str, float | str]] = {}
        for i, row in enumerate(results):
            sample_idx = int(float(row["sample_idx"]))
            combined_by_sample[sample_idx] = {
                "source": "evaluation",
                "sample_idx": sample_idx,
                "base_phys": float(base_phys[i]),
                "pred_phys": float(pred_phys[i]),
                "measure": float(row["measure"]),
            }
        for row in calibration_results:
            sample_idx = int(float(row["sample_idx"]))
            combined_by_sample[sample_idx] = {
                "source": str(row["source"]),
                "sample_idx": sample_idx,
                "base_phys": (
                    float(row["base_norm"]) * self.stats.dev_std
                    + self.stats.nom_measure
                ),
                "pred_phys": (
                    float(row["pred_norm"]) * self.stats.dev_std
                    + self.stats.nom_measure
                ),
                "measure": float(row["measure"]),
            }
        combined = sorted(
            combined_by_sample.values(),
            key=lambda row: float(row["measure"]),
            reverse=self.objective == "max",
        )

        lines = [
            f"log_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"evaluation_spice: {len(results)}\n",
            f"calibration_spice: {len(calibration_results)}\n",
            f"discovered_union: {len(combined)}\n",
            f"objective: {self.objective}\n",
            "metrics_scope: evaluation_only\n",
            f"base_pearson: {base_pearson:.6f}\n",
            f"base_spearman: {base_spearman:.6f}\n",
            f"base_mae_phys: {base_mae:.6e}\n",
            f"corrected_pearson: {pearson:.6f}\n",
            f"corrected_spearman: {spearman:.6f}\n",
            f"corrected_mae_phys: {mae:.6e}\n\n",
        ]

        if combined:
            lines.append(
                "combined_real_rank | source | sample_idx | base_phys | "
                "pred_phys | spice\n"
            )
            for rank, row in enumerate(combined, start=1):
                lines.append(
                    f"{rank:18d} | {str(row['source']):18s} | "
                    f"{int(row['sample_idx']):10d} | "
                    f"{float(row['base_phys']):.6e} | "
                    f"{float(row['pred_phys']):.6e} | "
                    f"{float(row['measure']):.6e}\n"
                )

        lines.append(
            "\nevaluation_pred_rank | sample_idx | base_phys | pred_phys | "
            "spice | evaluation_real_rank\n"
        )
        for i, row in enumerate(results):
            lines.append(
                f"{int(row['rank']):9d} | {int(row['sample_idx']):10d} | "
                f"{base_phys[i]:.6e} | {pred_phys[i]:.6e} | {measure[i]:.6e} | "
                f"{real_rank[i]}\n"
            )

        if calibration_results:
            lines.append(
                "\ncalibration_source | sample_idx | base_phys | pred_phys | spice\n"
            )
            for row in calibration_results:
                base_phys_i = (
                    float(row["base_norm"]) * self.stats.dev_std
                    + self.stats.nom_measure
                )
                pred_phys_i = (
                    float(row["pred_norm"]) * self.stats.dev_std
                    + self.stats.nom_measure
                )
                lines.append(
                    f"{str(row['source']):18s} | {int(float(row['sample_idx'])):10d} | "
                    f"{base_phys_i:.6e} | {pred_phys_i:.6e} | "
                    f"{float(row['measure']):.6e}\n"
                )
        Path(log_path).write_text("".join(lines), encoding="utf-8")
        print(
            f"Base Pearson={base_pearson:.4f} Spearman={base_spearman:.4f} "
            f"MAE={base_mae:.4e}"
        )
        print(
            f"Corrected Pearson={pearson:.4f} Spearman={spearman:.4f} "
            f"MAE={mae:.4e}"
        )
        print(f"Validation log: {log_path}")
