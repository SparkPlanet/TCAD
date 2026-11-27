from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class YieldEstimate:
    method: str
    failure_probability: float
    standard_error: float
    spice_calls: int
    failed_spice_calls: int
    elapsed_seconds: float
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def as_text(self) -> str:
        radius = 1.96 * self.standard_error
        rows = [
            f"method: {self.method}",
            f"failure_probability: {self.failure_probability:.12e}",
            f"yield: {1.0 - self.failure_probability:.12e}",
            f"standard_error: {self.standard_error:.12e}",
            f"ci95_low: {max(0.0, self.failure_probability - radius):.12e}",
            f"ci95_high: {min(1.0, self.failure_probability + radius):.12e}",
            f"spice_calls: {self.spice_calls}",
            f"failed_spice_calls: {self.failed_spice_calls}",
            f"elapsed_seconds: {self.elapsed_seconds:.6f}",
        ]
        rows.extend(f"{key}: {value}" for key, value in self.metadata.items())
        return "\n".join(rows) + "\n"


def write_result(path: str | Path, result: YieldEstimate) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.as_text(), encoding="utf-8")
