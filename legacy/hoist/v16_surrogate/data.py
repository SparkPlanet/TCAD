from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import norm, qmc
from tqdm import tqdm

from .circuit import CircuitSpec, SpiceRunner
from .common import SDAT_MAGIC, SDAT_VERSION


# ---------------------------------------------------------------------------
# Sampling and .sdat IO
# ---------------------------------------------------------------------------


class SobolZSampler:
    @staticmethod
    def sample(n_samples: int, dim: int, sigma_range: float, seed: int) -> np.ndarray:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        raw = sampler.random(n=n_samples)
        lo = norm.cdf(-sigma_range)
        hi = norm.cdf(sigma_range)
        return norm.ppf(lo + (hi - lo) * raw).astype(np.float64)

    @staticmethod
    def write_binary(samples: np.ndarray, path: str | Path) -> None:
        np.asarray(samples, dtype=np.float64).tofile(path)

    @staticmethod
    def read_binary(path: str | Path, dim: int) -> np.ndarray:
        flat = np.fromfile(path, dtype=np.float64)
        if flat.size % dim != 0:
            raise ValueError(f"Sample file size is not divisible by dim={dim}")
        return flat.reshape(-1, dim)


@dataclass(frozen=True)
class SdatStats:
    nom_measure: float
    dev_std: float
    multiplier: float
    y_mean: float
    y_std: float
    num_mos: int
    num_mos_types: int
    max_n_params: int


class SdatWriter:
    @staticmethod
    def write(
        path: str | Path,
        circuit: CircuitSpec,
        params: np.ndarray,
        labels: np.ndarray,
        stats: SdatStats,
    ) -> None:
        params = np.asarray(params, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        n_samples = int(params.shape[0])
        if params.shape != (n_samples, circuit.num_mos, circuit.max_n_params):
            raise ValueError(
                "params must have shape "
                f"({n_samples}, {circuit.num_mos}, {circuit.max_n_params})"
            )
        if labels.shape[0] != n_samples:
            raise ValueError("labels length does not match params")

        with open(path, "wb") as f:
            f.write(
                struct.pack(
                    "<6I8I",
                    SDAT_MAGIC,
                    SDAT_VERSION,
                    n_samples,
                    circuit.num_mos,
                    circuit.max_n_params,
                    circuit.num_mos_types,
                    *([0] * 8),
                )
            )
            f.write(
                struct.pack(
                    "<5d",
                    stats.nom_measure,
                    stats.dev_std,
                    stats.multiplier,
                    stats.y_mean,
                    stats.y_std,
                )
            )
            circuit.type_ids.astype(np.int32).tofile(f)
            circuit.mos_n_params.astype(np.int32).tofile(f)
            params.tofile(f)
            labels.tofile(f)


class SdatDataset(torch.utils.data.Dataset):
    """In-memory .sdat reader compatible with the original v16 format."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic, version, ns, nm, mp, nt, *_ = struct.unpack("<6I8I", f.read(56))
            if magic != SDAT_MAGIC:
                raise ValueError(f"Invalid .sdat magic: {magic:#x}")
            if version != SDAT_VERSION:
                raise ValueError(f"Unsupported .sdat version: {version}")

            stats_values = struct.unpack("<5d", f.read(40))
            self.stats = {
                "nom_measure": stats_values[0],
                "dev_std": stats_values[1],
                "multiplier": stats_values[2],
                "y_mean": stats_values[3],
                "y_std": stats_values[4],
                "num_mos": nm,
                "num_mos_types": nt,
                "max_n_params": mp,
            }
            self.n_samples = ns
            self.num_mos = nm
            self.max_n_params = mp
            self.num_mos_types = nt

            self.type_ids = torch.from_numpy(
                np.fromfile(f, dtype=np.int32, count=nm).copy()
            )
            self.mos_n_params = torch.from_numpy(
                np.fromfile(f, dtype=np.int32, count=nm).copy()
            )
            n_param_values = ns * nm * mp
            params_np = np.fromfile(f, dtype=np.float32, count=n_param_values)
            self.all_params = torch.from_numpy(params_np.copy()).view(ns, nm, mp)
            labels_np = np.fromfile(f, dtype=np.float32, count=ns)
            self.all_labels = torch.from_numpy(labels_np.copy())

        self._mask = torch.zeros(self.num_mos, self.max_n_params)
        for m, n_params in enumerate(self.mos_n_params.tolist()):
            self._mask[m, : int(n_params)] = 1.0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "params": self.all_params[idx],
            "mask": self._mask,
            "type_ids": self.type_ids,
            "y": self.all_labels[idx].view(1),
        }


def collate_sdat(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "params": torch.stack([s["params"] for s in batch]),
        "mask": torch.stack([s["mask"] for s in batch]),
        "type_ids": torch.stack([s["type_ids"] for s in batch]),
        "y": torch.stack([s["y"] for s in batch]),
    }


class SpiceDatasetBuilder:
    """Pure Python replacement for simgen train."""

    def __init__(self, circuit: CircuitSpec, timeout_s: int = 60):
        self.circuit = circuit
        self.runner = SpiceRunner(circuit, timeout_s=timeout_s)

    def build_from_flat_z(
        self,
        flat_z: np.ndarray,
        output_path: str | Path,
        allow_failed_nominal: bool = False,
    ) -> SdatStats:
        flat_z = np.asarray(flat_z, dtype=np.float64)
        if flat_z.ndim != 2 or flat_z.shape[1] != self.circuit.total_params:
            raise ValueError(
                f"flat_z must be [N, {self.circuit.total_params}], got {flat_z.shape}"
            )

        self.runner.write_models()
        nom_measure = self.runner.run_with_params(self.runner.nominal_params())
        if nom_measure is None:
            raise RuntimeError("Nominal SPICE simulation failed")

        params = np.zeros(
            (flat_z.shape[0], self.circuit.num_mos, self.circuit.max_n_params),
            dtype=np.float32,
        )
        measures = np.empty(flat_z.shape[0], dtype=np.float64)

        print(f"Nominal measure: {nom_measure:.6e}")
        failed_indices: List[int] = []
        for i in tqdm(range(flat_z.shape[0]), desc="SPICE train data"):
            row_dense = self._flat_to_dense(flat_z[i])
            params[i] = row_dense.astype(np.float32)
            measure = self.runner.run_with_params(self.runner.zscore_to_params(row_dense))
            if measure is None:
                failed_indices.append(i)
                measures[i] = nom_measure
            else:
                measures[i] = measure

        if failed_indices and not allow_failed_nominal:
            preview = ", ".join(str(i) for i in failed_indices[:20])
            if len(failed_indices) > 20:
                preview += ", ..."
            raise RuntimeError(
                f"{len(failed_indices)} SPICE simulations failed or produced no "
                f"measure; failed indices: {preview}. Re-run with a longer "
                "--spice-timeout, smaller --train-z-scale, or pass "
                "--allow-failed-nominal to keep the old nominal-substitution behavior."
            )
        if failed_indices:
            print(
                f"warning: substituted nominal measure for {len(failed_indices)} "
                "failed SPICE simulations"
            )

        delta = measures - nom_measure
        dev_std = float(np.sqrt(np.mean(delta * delta)))
        if dev_std == 0.0:
            dev_std = 1.0
        labels = (measures - nom_measure) / dev_std

        stats = SdatStats(
            nom_measure=float(nom_measure),
            dev_std=dev_std,
            multiplier=1.0 / dev_std,
            y_mean=float(labels.mean()),
            y_std=float(labels.std()),
            num_mos=self.circuit.num_mos,
            num_mos_types=self.circuit.num_mos_types,
            max_n_params=self.circuit.max_n_params,
        )
        SdatWriter.write(output_path, self.circuit, params, labels, stats)
        return stats

    def _flat_to_dense(self, flat: np.ndarray) -> np.ndarray:
        dense = np.zeros((self.circuit.num_mos, self.circuit.max_n_params), dtype=np.float64)
        offset = 0
        for mi, mos in enumerate(self.circuit.mos_list):
            n = mos.n_params
            dense[mi, :n] = flat[offset : offset + n]
            offset += n
        return dense
