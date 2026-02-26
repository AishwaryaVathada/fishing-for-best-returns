from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict


class SafetyConstraint(Protocol):
    """Maps env info dict to a non-negative cost signal and an optional boolean violation."""

    def cost(self, info: Dict) -> float:
        ...

    def violated(self, info: Dict) -> bool:
        ...


@dataclass(frozen=True)
class HardPopulationConstraint:
    salmon_min: float = 3_000.0
    shark_min: float = 150.0

    def violated(self, info: Dict) -> bool:
        salmon = float(info.get("salmon", 0.0))
        shark = float(info.get("shark", 0.0))
        return (salmon < self.salmon_min) or (shark < self.shark_min)

    def cost(self, info: Dict) -> float:
        return 1.0 if self.violated(info) else 0.0
