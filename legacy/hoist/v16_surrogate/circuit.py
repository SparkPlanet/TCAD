from __future__ import annotations

import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import fcntl

import numpy as np

from .common import ROOT


# ---------------------------------------------------------------------------
# Circuit parsing and SPICE execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MosSpec:
    mos_id: int
    type_name: str
    type_id: int
    params: Tuple[str, ...]
    nominal: Tuple[float, ...]
    std: Tuple[float, ...]

    @property
    def n_params(self) -> int:
        return len(self.params)


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    circuit_path: Path
    spice_dir: Path
    active_sp: str
    param_order: Tuple[str, ...]
    global_std: float
    mos_list: Tuple[MosSpec, ...]
    measure_name: str = ""
    objective: str = "max"

    @property
    def num_mos(self) -> int:
        return len(self.mos_list)

    @property
    def num_mos_types(self) -> int:
        return len({m.type_id for m in self.mos_list})

    @property
    def max_n_params(self) -> int:
        return max((m.n_params for m in self.mos_list), default=0)

    @property
    def total_params(self) -> int:
        return sum(m.n_params for m in self.mos_list)

    @property
    def type_ids(self) -> np.ndarray:
        return np.asarray([m.type_id for m in self.mos_list], dtype=np.int32)

    @property
    def mos_n_params(self) -> np.ndarray:
        return np.asarray([m.n_params for m in self.mos_list], dtype=np.int32)


class CircuitTxtParser:
    """Parser for data_gen/circuit/*.txt circuit descriptions."""

    @staticmethod
    def parse(path: str | Path, root: Path = ROOT) -> CircuitSpec:
        circuit_path = Path(path)
        if not circuit_path.is_absolute():
            circuit_path = root / circuit_path
        circuit_path = circuit_path.resolve()

        name = circuit_path.stem
        spice_dir = Path(".")
        active_sp = ""
        param_order: List[str] = []
        global_std = 0.01
        measure_name = ""
        objective = "max"
        current_type = "unknown"
        type_map: Dict[str, int] = {}
        mos_list: List[MosSpec] = []

        for raw in circuit_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("# spice_dir:"):
                spice_dir = Path(line.split(":", 1)[1].strip())
            elif line.startswith("# active_sp:"):
                active_sp = line.split(":", 1)[1].strip()
            elif line.startswith("# param_order:"):
                param_order = line.split(":", 1)[1].strip().split()
            elif line.startswith("# std:"):
                token = line.split(":", 1)[1].strip().split()[0]
                global_std = float(token)
            elif line.startswith("# measure:"):
                measure_name = line.split(":", 1)[1].strip()
            elif line.startswith("# objective:"):
                objective = line.split(":", 1)[1].strip().lower()
                if objective not in {"max", "min"}:
                    raise ValueError(
                        f"Invalid objective '{objective}' in {circuit_path}; "
                        "expected 'max' or 'min'"
                    )
            elif line.startswith("# ---"):
                match = re.search(r"# --- (.+?) ---", line)
                if match:
                    current_type = match.group(1).strip()
            elif line.startswith("M"):
                colon = line.find("：")
                if colon < 0:
                    colon = line.find(":")
                if colon < 0:
                    continue

                mos_id = int(line[1:colon])
                rest = line[colon + 1 :]
                parsed: Dict[str, float] = {}
                for token in rest.split():
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    parsed[key] = float(value)

                ordered_params = [p for p in param_order if p in parsed]
                if not ordered_params:
                    ordered_params = list(parsed.keys())
                if current_type not in type_map:
                    type_map[current_type] = len(type_map)

                nominal = tuple(parsed[p] for p in ordered_params)
                std = tuple(global_std for _ in ordered_params)
                mos_list.append(
                    MosSpec(
                        mos_id=mos_id,
                        type_name=current_type,
                        type_id=type_map[current_type],
                        params=tuple(ordered_params),
                        nominal=nominal,
                        std=std,
                    )
                )

        if not active_sp:
            raise ValueError(f"Missing '# active_sp:' in {circuit_path}")
        if not mos_list:
            raise ValueError(f"No MOS rows parsed from {circuit_path}")

        if not spice_dir.is_absolute():
            spice_dir = root / spice_dir

        return CircuitSpec(
            name=name,
            circuit_path=circuit_path,
            spice_dir=spice_dir.resolve(),
            active_sp=active_sp,
            param_order=tuple(param_order),
            global_std=global_std,
            mos_list=tuple(mos_list),
            measure_name=measure_name,
            objective=objective,
        )


class SpiceRunner:
    """Writes params/models files and runs ngspice for one circuit."""

    def __init__(self, circuit: CircuitSpec, timeout_s: int = 60):
        self.circuit = circuit
        self.timeout_s = timeout_s
        self.params_path = circuit.spice_dir / "params.sp"
        self.models_path = circuit.spice_dir / "models.sp"
        self.lock_path = circuit.spice_dir / ".spice_eval.lock"

    @contextmanager
    def sample_lock(self) -> Iterator[None]:
        """Serialize params.sp/model writes and ngspice execution per circuit.

        The netlists include shared files in ``spice_dir``. Multiple scripts
        evaluating the same circuit can otherwise overwrite params.sp while an
        ngspice process is starting, which corrupts measured tails.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def write_models(self) -> None:
        with self.sample_lock():
            self._write_models_unlocked()

    def _write_models_unlocked(self) -> None:
        lines = ["* --- Automatically Generated Models ---\n"]
        for mos in self.circuit.mos_list:
            mtype = "nmos" if "nmos" in mos.type_name.lower() else "pmos"
            parts = [
                f".model m_local_{mos.mos_id} {mtype} level=8 version=3.3.0"
            ]
            for pname in mos.params:
                parts.append(f" {pname}={{M{mos.mos_id}_{pname}}}")
            lines.append("".join(parts) + "\n")
        self.models_path.write_text("".join(lines), encoding="utf-8")

    def write_params(self, values: Dict[str, float]) -> None:
        lines = ["* Current Sample Parameters\n"]
        for key, value in values.items():
            lines.append(f".param {key} = {value:.6e}\n")
        self.params_path.write_text("".join(lines), encoding="utf-8")

    def nominal_params(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for mos in self.circuit.mos_list:
            for pname, nominal in zip(mos.params, mos.nominal):
                values[f"M{mos.mos_id}_{pname}"] = nominal
        return values

    def zscore_to_params(self, z_row: np.ndarray) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for mi, mos in enumerate(self.circuit.mos_list):
            for pi, (pname, nominal, std) in enumerate(
                zip(mos.params, mos.nominal, mos.std)
            ):
                z = float(z_row[mi, pi])
                values[f"M{mos.mos_id}_{pname}"] = nominal + abs(nominal * std) * z
        return values

    def run_with_params(self, values: Dict[str, float]) -> Optional[float]:
        with self.sample_lock():
            self.write_params(values)
            return self._run_unlocked()

    def run(self) -> Optional[float]:
        with self.sample_lock():
            return self._run_unlocked()

    @staticmethod
    def ngspice_threads() -> int:
        raw = os.environ.get("NGSPICE_NUM_THREADS") or os.environ.get("NGSPICE_THREADS")
        if not raw:
            return 1
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    @contextmanager
    def threaded_netlist(self) -> Iterator[str]:
        threads = self.ngspice_threads()
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

    def _run_unlocked(self) -> Optional[float]:
        try:
            with self.threaded_netlist() as active_sp:
                result = subprocess.run(
                    ["ngspice", "-b", active_sp],
                    cwd=self.circuit.spice_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired):
            return None

        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(
            r"Final Measure\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
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
        return float(match.group(1)) if match else None
