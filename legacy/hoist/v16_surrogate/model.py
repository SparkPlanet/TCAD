from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .data import SdatDataset, collate_sdat


# ---------------------------------------------------------------------------
# v16 model and training
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve CLI device string.

    `auto` prefers CUDA when available and falls back to CPU. Explicit CUDA
    requests still fail early if CUDA is unavailable, which avoids silently
    running long jobs on CPU by accident.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return resolved


def suggest_hidden_dim(num_mos: int, max_n_params: int) -> int:
    total = num_mos * max_n_params
    raw_dim = 8.0 * math.log2(total / 48.0)
    clamped_dim = max(16, min(64, int(raw_dim)))
    return max(8, (clamped_dim // 8) * 8)


MODEL_ARCHITECTURES = {"v16", "flat_mlp"}
FUSION_MODES = {"multiplicative", "additive", "concat"}


class EncoderV16(nn.Module):
    def __init__(
        self,
        max_n_params: int,
        hidden: int,
        num_mos_types: int,
        num_mos: int,
        *,
        use_type_embedding: bool = True,
        use_global_context: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.use_type_embedding = use_type_embedding
        self.use_global_context = use_global_context
        self.param_encoder = nn.Sequential(
            nn.Linear(max_n_params + hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.type_emb = (
            nn.Embedding(num_mos_types + 4, hidden) if use_type_embedding else None
        )
        self.compress = nn.Linear(hidden, 16)
        self.global_mlp = (
            nn.Sequential(
                nn.Linear(num_mos * 16, hidden),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden, hidden),
            )
            if use_global_context
            else None
        )

    def forward_parts(
        self, params: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = params * mask
        batch_size, num_mos, max_params = params.shape
        if self.type_emb is not None:
            type_features = self.type_emb(type_ids.long())
        else:
            type_features = torch.zeros(
                batch_size,
                num_mos,
                self.hidden,
                dtype=params.dtype,
                device=params.device,
            )
        token_input = torch.cat([params, type_features], dim=-1)
        h = self.param_encoder(token_input.reshape(batch_size * num_mos, max_params + self.hidden))
        h = h.reshape(batch_size, num_mos, self.hidden)
        h_compressed = self.compress(h.reshape(batch_size * num_mos, self.hidden))
        h_flat = h_compressed.reshape(batch_size, num_mos * 16)
        if self.global_mlp is not None:
            global_ctx = self.global_mlp(h_flat)
        else:
            global_ctx = h.new_zeros((batch_size, self.hidden))
        return h, global_ctx


class CircuitSurrogateV16(nn.Module):
    def __init__(
        self,
        max_n_params: int,
        num_mos: int,
        num_mos_types: int,
        mos_n_params: Optional[Sequence[int]] = None,
        hidden: Optional[int] = None,
        use_type_embedding: bool = True,
        use_global_context: bool = True,
        fusion_mode: str = "multiplicative",
    ):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {sorted(FUSION_MODES)}")
        self.num_mos = num_mos
        self.max_n_params = max_n_params
        self.num_mos_types = num_mos_types
        self.model_architecture = "v16"
        self.use_type_embedding = use_type_embedding
        self.use_global_context = use_global_context
        self.fusion_mode = fusion_mode
        if mos_n_params is None:
            mos_n_params = [max_n_params] * num_mos
        active_mos = []
        active_param = []
        for mos_idx, n_params in enumerate(mos_n_params):
            for param_idx in range(int(n_params)):
                active_mos.append(mos_idx)
                active_param.append(param_idx)
        self.total_active_params = len(active_mos)
        self.register_buffer("active_mos", torch.tensor(active_mos, dtype=torch.long))
        self.register_buffer("active_param", torch.tensor(active_param, dtype=torch.long))
        self.hidden = hidden or suggest_hidden_dim(num_mos, max_n_params)
        self.encoder = EncoderV16(
            max_n_params,
            self.hidden,
            num_mos_types,
            num_mos,
            use_type_embedding=use_type_embedding,
            use_global_context=use_global_context,
        )
        fusion_input_dim = {
            "multiplicative": self.hidden * 3,
            "additive": self.hidden,
            "concat": self.hidden * 2,
        }[fusion_mode]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden * 2, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_dense(batch["params"], batch["mask"], batch["type_ids"])

    def encode_tokens(
        self, params: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        h, global_ctx = self.encoder.forward_parts(params, mask, type_ids)
        g = global_ctx.unsqueeze(1).expand_as(h)
        if self.fusion_mode == "multiplicative":
            fusion_input = torch.cat([h, g, h * g], dim=-1)
        elif self.fusion_mode == "additive":
            fusion_input = h + g
        else:
            fusion_input = torch.cat([h, g], dim=-1)
        u = self.fusion(fusion_input)
        valid = (mask.sum(dim=-1, keepdim=True) > 0).to(u.dtype)
        return u * valid

    def encode_pooled(
        self, params: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        u = self.encode_tokens(params, mask, type_ids)
        valid = (mask.sum(dim=-1, keepdim=True) > 0).to(u.dtype)
        mean_pool = u.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        max_pool = u.masked_fill(valid == 0, -torch.finfo(u.dtype).max).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward_dense(
        self, params: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.encode_pooled(params, mask, type_ids)
        return self.regressor(pooled)

    def active_flat(self, params: torch.Tensor) -> torch.Tensor:
        return params[:, self.active_mos, self.active_param]

    def model_config(self) -> Dict[str, object]:
        return {
            "model_architecture": self.model_architecture,
            "hidden": int(self.hidden),
            "surrogate_version": "hoist_paper_v1",
            "use_type_embedding": bool(self.use_type_embedding),
            "use_global_context": bool(self.use_global_context),
            "fusion_mode": self.fusion_mode,
        }


class FlatMLPSurrogate(nn.Module):
    """Plain dense MLP baseline with the v16 MC/prediction interface."""

    def __init__(
        self,
        max_n_params: int,
        num_mos: int,
        num_mos_types: int,
        mos_n_params: Optional[Sequence[int]] = None,
        hidden: Optional[int] = None,
    ):
        super().__init__()
        self.num_mos = num_mos
        self.max_n_params = max_n_params
        self.num_mos_types = num_mos_types
        self.model_architecture = "flat_mlp"
        self.use_type_embedding = False
        self.use_global_context = False
        if mos_n_params is None:
            mos_n_params = [max_n_params] * num_mos
        active_mos = []
        active_param = []
        for mos_idx, n_params in enumerate(mos_n_params):
            for param_idx in range(int(n_params)):
                active_mos.append(mos_idx)
                active_param.append(param_idx)
        self.total_active_params = len(active_mos)
        self.register_buffer("active_mos", torch.tensor(active_mos, dtype=torch.long))
        self.register_buffer("active_param", torch.tensor(active_param, dtype=torch.long))
        self.flat_input_dim = num_mos * max_n_params
        base_hidden = hidden or max(128, min(512, 2 * self.flat_input_dim))
        self.hidden = int(base_hidden)
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_input_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def flat_params(self, params: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (params * mask).reshape(params.size(0), -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_dense(batch["params"], batch["mask"], batch["type_ids"])

    def forward_dense(
        self, params: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        del type_ids
        return self.regressor(self.flat_params(params, mask))

    def model_config(self) -> Dict[str, object]:
        return {
            "model_architecture": self.model_architecture,
            "hidden": int(self.hidden),
            "surrogate_version": "flat_mlp_v1",
            "use_type_embedding": False,
            "use_global_context": False,
        }


def build_surrogate_model(
    *,
    max_n_params: int,
    num_mos: int,
    num_mos_types: int,
    mos_n_params: Sequence[int],
    model_architecture: str = "v16",
    hidden: Optional[int] = None,
    use_type_embedding: bool = True,
    use_global_context: bool = True,
    fusion_mode: str = "multiplicative",
) -> nn.Module:
    if model_architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"model_architecture must be one of {sorted(MODEL_ARCHITECTURES)}"
        )
    if model_architecture == "flat_mlp":
        return FlatMLPSurrogate(
            max_n_params=max_n_params,
            num_mos=num_mos,
            num_mos_types=num_mos_types,
            mos_n_params=mos_n_params,
            hidden=hidden,
        )
    return CircuitSurrogateV16(
        max_n_params=max_n_params,
        num_mos=num_mos,
        num_mos_types=num_mos_types,
        mos_n_params=mos_n_params,
        hidden=hidden,
        use_type_embedding=use_type_embedding,
        use_global_context=use_global_context,
        fusion_mode=fusion_mode,
    )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    patience: int = 100
    grad_clip: float = 0.5
    seed: int = 42
    device: str = "auto"
    val_ratio: float = 0.0
    objective: str = "max"
    model_architecture: str = "v16"
    use_type_embedding: bool = True
    use_global_context: bool = True
    fusion_mode: str = "multiplicative"
    threshold_norm: float | None = None
    tail_fail_weight: float = 2.0
    boundary_weight: float = 1.0
    boundary_sigma: float = 1.0


class V16Trainer:
    def __init__(self, config: TrainConfig):
        if config.objective not in {"max", "min"}:
            raise ValueError("objective must be 'max' or 'min'")
        self.config = config

    @property
    def objective_sign(self) -> float:
        return 1.0 if self.config.objective == "max" else -1.0

    def train(
        self,
        train_path: str | Path,
        ckpt_path: str | Path,
        val_path: str | Path | None = None,
    ) -> CircuitSurrogateV16:
        set_seed(self.config.seed)
        device = resolve_device(self.config.device)
        dataset = SdatDataset(train_path)
        if val_path:
            if self.config.val_ratio > 0.0:
                raise ValueError("--val and --val-ratio cannot be used together")
            train_dataset = dataset
            val_dataset = NormalizedValidationDataset(
                SdatDataset(val_path),
                reference=dataset,
            )
        else:
            train_dataset, val_dataset = self._split_dataset(dataset)
        loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True,
            collate_fn=collate_sdat,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=min(self.config.batch_size, len(val_dataset)),
                shuffle=False,
                collate_fn=collate_sdat,
            )
            if val_dataset is not None
            else None
        )
        model = build_surrogate_model(
            max_n_params=dataset.max_n_params,
            num_mos=dataset.num_mos,
            num_mos_types=dataset.num_mos_types,
            mos_n_params=dataset.mos_n_params.tolist(),
            model_architecture=self.config.model_architecture,
            use_type_embedding=self.config.use_type_embedding,
            use_global_context=self.config.use_global_context,
            fusion_mode=self.config.fusion_mode,
        ).to(device)
        print(
            f"v16 python: arch={getattr(model, 'model_architecture', 'v16')}, "
            f"hidden={model.hidden}, params="
            f"{sum(p.numel() for p in model.parameters()):,}, device={device}, "
            f"train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, "
            f"objective={self.config.objective}, "
            f"threshold_norm={self.config.threshold_norm}, "
            f"use_type_embedding={getattr(model, 'use_type_embedding', False)}, "
            f"use_global_context={getattr(model, 'use_global_context', False)}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        best_loss = float("inf")
        wait = 0

        for epoch in range(self.config.epochs):
            model.train()
            if epoch < self.config.warmup_epochs:
                scale = (epoch + 1) / self.config.warmup_epochs
                for group in optimizer.param_groups:
                    group["lr"] = self.config.lr * scale

            total_loss = 0.0
            total_supervised = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                y = batch["y"].view(-1, 1)
                pred = model.forward_dense(
                    batch["params"], batch["mask"], batch["type_ids"]
                )
                supervised_loss = self._base_loss(pred, y)
                loss = supervised_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
                total_loss += float(loss.item())
                total_supervised += float(supervised_loss.item())

            avg_loss = total_loss / max(1, len(loader))
            avg_supervised = total_supervised / max(1, len(loader))
            if epoch >= self.config.warmup_epochs:
                scheduler.step()

            val_metrics = self._evaluate(model, val_loader, device) if val_loader else None
            checkpoint_loss = val_metrics["mse"] if val_metrics else avg_loss
            if checkpoint_loss < best_loss - 1e-7:
                best_loss = checkpoint_loss
                wait = 0
                self._save_checkpoint(model, ckpt_path)
            else:
                wait += 1

            if epoch % 10 == 0:
                val_text = (
                    f" | val_mse={val_metrics['mse']:.6e}"
                    if val_metrics
                    else ""
                )
                print(
                    f"epoch {epoch:03d} | loss={avg_loss:.6e} | "
                    f"base={avg_supervised:.6e}{val_text} | wait={wait}"
                )
            if wait >= self.config.patience:
                print(f"early stop @ epoch {epoch}, best_loss={best_loss:.6e}")
                break

        model.load_state_dict(
            self._checkpoint_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
        )
        model.eval()
        if val_loader:
            metrics = self._evaluate(model, val_loader, device)
            print(
                "validation: "
                f"mse={metrics['mse']:.6e}, mae={metrics['mae']:.6e}, "
                f"pearson={metrics['pearson']:.4f}, spearman={metrics['spearman']:.4f}"
            )
        return model

    @staticmethod
    def _save_checkpoint(model: nn.Module, ckpt_path: str | Path) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": (
                    model.model_config()
                    if hasattr(model, "model_config")
                    else {"model_architecture": "v16"}
                ),
            },
            ckpt_path,
        )

    @staticmethod
    def _checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            if isinstance(state, dict):
                return state
        if isinstance(checkpoint, dict):
            return checkpoint  # old checkpoints were raw state_dicts
        raise TypeError("unsupported checkpoint format")

    def _split_dataset(self, dataset: SdatDataset) -> Tuple[object, Optional[object]]:
        if self.config.val_ratio <= 0.0:
            return dataset, None
        if not 0.0 < self.config.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        val_count = max(1, int(round(len(dataset) * self.config.val_ratio)))
        val_count = min(val_count, len(dataset) - 1)
        generator = torch.Generator().manual_seed(self.config.seed)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        return Subset(dataset, order[val_count:]), Subset(dataset, order[:val_count])

    @staticmethod
    def _evaluate(
        model: CircuitSurrogateV16,
        loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        predictions = []
        targets = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                predictions.append(model(batch).view(-1).cpu())
                targets.append(batch["y"].view(-1).cpu())
        pred = torch.cat(predictions).numpy().astype(np.float64)
        target = torch.cat(targets).numpy().astype(np.float64)
        error = pred - target
        pearson = float(np.corrcoef(pred, target)[0, 1]) if len(target) > 1 else float("nan")
        pred_rank = np.argsort(np.argsort(pred))
        target_rank = np.argsort(np.argsort(target))
        spearman = (
            float(np.corrcoef(pred_rank, target_rank)[0, 1])
            if len(target) > 1
            else float("nan")
        )
        return {
            "mse": float(np.mean(error * error)),
            "mae": float(np.mean(np.abs(error))),
            "pearson": pearson,
            "spearman": spearman,
        }

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.threshold_norm is None:
            return F.mse_loss(pred, target)
        signed_margin = self.objective_sign * (target - float(self.config.threshold_norm))
        weights = torch.ones_like(target)
        if self.config.tail_fail_weight != 0.0:
            weights = weights + self.config.tail_fail_weight * (signed_margin > 0.0).to(target.dtype)
        if self.config.boundary_weight != 0.0:
            sigma = max(float(self.config.boundary_sigma), 1e-12)
            weights = weights + self.config.boundary_weight * torch.exp(-signed_margin.abs() / sigma)
        return ((pred - target).pow(2) * weights.detach()).mean()


class NormalizedValidationDataset(Dataset):
    """Expose an independent .sdat using the training set's label scale."""

    def __init__(self, dataset: SdatDataset, reference: SdatDataset):
        if (
            dataset.num_mos != reference.num_mos
            or dataset.max_n_params != reference.max_n_params
            or dataset.num_mos_types != reference.num_mos_types
            or not torch.equal(dataset.type_ids, reference.type_ids)
            or not torch.equal(dataset.mos_n_params, reference.mos_n_params)
        ):
            raise ValueError("validation .sdat circuit structure does not match training data")
        self.dataset = dataset
        measure = (
            dataset.all_labels.double() * float(dataset.stats["dev_std"])
            + float(dataset.stats["nom_measure"])
        )
        self.labels = (
            (measure - float(reference.stats["nom_measure"]))
            / float(reference.stats["dev_std"])
        ).float()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        sample["y"] = self.labels[idx].view(1)
        return sample
