from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MosSpec:
    mos_id: int
    type_name: str
    params: tuple[str, ...]
    nominal: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    descriptor: Path
    spice_dir: Path
    active_sp: str
    mos_list: tuple[MosSpec, ...]
    measure_name: str
    objective: str

    @property
    def dimension(self) -> int:
        return sum(len(mos.params) for mos in self.mos_list)

    def flat_z_to_parameters(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.dimension,):
            raise ValueError(f"expected z=[{self.dimension}], got {z.shape}")
        values: dict[str, float] = {}
        offset = 0
        for mos in self.mos_list:
            for name, nominal, relative_std in zip(
                mos.params,
                mos.nominal,
                mos.std,
            ):
                values[f"M{mos.mos_id}_{name}"] = (
                    nominal + abs(nominal * relative_std) * z[offset]
                )
                offset += 1
        return values


class CircuitTxtParser:
    @staticmethod
    def parse(path: str | Path) -> CircuitSpec:
        descriptor = Path(path).resolve()
        root = Path(__file__).resolve().parents[2]
        name = descriptor.stem
        spice_dir = Path(".")
        active_sp = ""
        param_order: list[str] = []
        global_std = 0.01
        measure_name = ""
        objective = "max"
        current_type = "unknown"
        mos_list: list[MosSpec] = []

        for raw_line in descriptor.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("# name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("# spice_dir:"):
                spice_dir = Path(line.split(":", 1)[1].strip())
            elif line.startswith("# active_sp:"):
                active_sp = line.split(":", 1)[1].strip()
            elif line.startswith("# param_order:"):
                param_order = line.split(":", 1)[1].split()
            elif line.startswith("# std:"):
                global_std = float(line.split(":", 1)[1].split()[0])
            elif line.startswith("# measure:"):
                measure_name = line.split(":", 1)[1].strip()
            elif line.startswith("# objective:"):
                objective = line.split(":", 1)[1].strip().lower()
            elif line.startswith("# ---"):
                match = re.search(r"# --- (.+?) ---", line)
                if match:
                    current_type = match.group(1).strip()
            elif line.startswith("M"):
                separator = line.find("：")
                if separator < 0:
                    separator = line.find(":")
                if separator < 0:
                    continue
                mos_id = int(line[1:separator])
                parsed: dict[str, float] = {}
                for token in line[separator + 1 :].split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        parsed[key] = float(value)
                names = [name for name in param_order if name in parsed]
                if not names:
                    names = list(parsed)
                mos_list.append(
                    MosSpec(
                        mos_id=mos_id,
                        type_name=current_type,
                        params=tuple(names),
                        nominal=tuple(parsed[name] for name in names),
                        std=tuple(global_std for _ in names),
                    )
                )

        if objective not in {"max", "min"}:
            raise ValueError("circuit objective must be 'max' or 'min'")
        if not active_sp or not mos_list:
            raise ValueError(f"incomplete circuit descriptor: {descriptor}")
        if not spice_dir.is_absolute():
            spice_dir = (root / spice_dir).resolve()
        return CircuitSpec(
            name=name,
            descriptor=descriptor,
            spice_dir=spice_dir,
            active_sp=active_sp,
            mos_list=tuple(mos_list),
            measure_name=measure_name,
            objective=objective,
        )
