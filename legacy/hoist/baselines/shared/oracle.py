from __future__ import annotations

import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fcntl
import numpy as np
from tqdm import tqdm

from .circuit import CircuitSpec


@dataclass(frozen=True)
class OracleBatch:
    z: np.ndarray
    measure: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.measure)


class SpiceOracle:
    """Budgeted ngspice evaluator shared by all standalone baselines."""

    def __init__(self, circuit: CircuitSpec, budget: int, timeout_s: int = 60):
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.circuit = circuit
        self.budget = int(budget)
        self.timeout_s = int(timeout_s)
        self.calls = 0
        self.failed_calls = 0
        self.params_path = circuit.spice_dir / "params.sp"
        self.models_path = circuit.spice_dir / "models.sp"
        self.lock_path = circuit.spice_dir / ".spice_eval.lock"
        with self._sample_lock():
            self._write_models()

    @contextmanager
    def _sample_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def _write_models(self) -> None:
        lines = ["* Automatically generated baseline models\n"]
        for mos in self.circuit.mos_list:
            model_type = "nmos" if "nmos" in mos.type_name.lower() else "pmos"
            line = (
                f".model m_local_{mos.mos_id} {model_type} "
                "level=8 version=3.3.0"
            )
            for name in mos.params:
                line += f" {name}={{M{mos.mos_id}_{name}}}"
            lines.append(line + "\n")
        self.models_path.write_text("".join(lines), encoding="utf-8")

    def _write_parameters(self, z: np.ndarray) -> None:
        values = self.circuit.flat_z_to_parameters(z)
        lines = ["* Current baseline sample\n"]
        lines.extend(f".param {key} = {value:.12e}\n" for key, value in values.items())
        self.params_path.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def _ngspice_threads() -> int:
        raw = os.environ.get("NGSPICE_NUM_THREADS") or os.environ.get("NGSPICE_THREADS")
        if not raw:
            return 1
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    @contextmanager
    def _threaded_netlist(self) -> Iterator[str]:
        threads = self._ngspice_threads()
        if threads <= 1:
            yield self.circuit.active_sp
            return

        source = self.circuit.spice_dir / self.circuit.active_sp
        text = source.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        control_index = next(
            (idx for idx, line in enumerate(lines) if line.strip().lower() == ".control"),
            None,
        )
        if control_index is None:
            yield self.circuit.active_sp
            return

        end_index = next(
            (
                idx
                for idx, line in enumerate(lines[control_index + 1 :], start=control_index + 1)
                if line.strip().lower() == ".endc"
            ),
            len(lines),
        )
        if any("num_threads" in line.lower() for line in lines[control_index:end_index]):
            yield self.circuit.active_sp
            return

        temp_name = f".{Path(self.circuit.active_sp).stem}.ngspice_threads_{os.getpid()}.sp"
        temp_path = self.circuit.spice_dir / temp_name
        injected = (
            lines[: control_index + 1]
            + [f"    set num_threads={threads}\n"]
            + lines[control_index + 1 :]
        )
        temp_path.write_text("".join(injected), encoding="utf-8")
        try:
            yield temp_name
        finally:
            temp_path.unlink(missing_ok=True)

    def _run_one(self, z: np.ndarray) -> float:
        with self._sample_lock():
            self._write_models()
            self._write_parameters(z)
            try:
                with self._threaded_netlist() as active_sp:
                    process = subprocess.run(
                        ["ngspice", "-b", active_sp],
                        cwd=self.circuit.spice_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_s,
                        check=False,
                    )
            except (OSError, subprocess.TimeoutExpired):
                return float("nan")
        output = process.stdout + "\n" + process.stderr
        match = re.search(
            r"Final Measure\s*[:=]\s*"
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            output,
            re.IGNORECASE,
        )
        if match is None and self.circuit.measure_name:
            match = re.search(
                rf"(?<![\w]){re.escape(self.circuit.measure_name)}\s*=\s*"
                r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
                output,
                re.IGNORECASE,
            )
        return float(match.group(1)) if match else float("nan")

    def evaluate(self, z: np.ndarray, description: str) -> OracleBatch:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.circuit.dimension:
            raise ValueError(
                f"expected z=[N,{self.circuit.dimension}], got {z.shape}"
            )
        z = z[: max(0, self.remaining)]
        measure = np.full(len(z), np.nan, dtype=np.float64)
        indices = (
            tqdm(range(len(z)), desc=description)
            if len(z) > 1
            else range(len(z))
        )
        for index in indices:
            measure[index] = self._run_one(z[index])
            self.calls += 1
            if not np.isfinite(measure[index]):
                self.failed_calls += 1
        return OracleBatch(z=z, measure=measure)
