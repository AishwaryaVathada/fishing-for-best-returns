from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional

import numpy as np


class Dynamics(Protocol):
    def step(
        self,
        salmon: float,
        shark: float,
        effort: float,
        month_1_indexed: int,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        """Return (salmon_caught, salmon_next, shark_next)."""


@dataclass(frozen=True)
class ToyDynamicsConfig:
    # Salmon logistic growth
    salmon_r0: float = 0.20
    salmon_K: float = 500_000.0

    # Predation (Lotka-Volterra style)
    predation_rate: float = 2.0e-7
    shark_conversion: float = 1.5e-3
    shark_mortality: float = 0.10
    shark_K: float = 5_000.0

    # Effort -> catch saturation
    catchability: float = 2.0e-6

    # Seasonality and long super-cycles
    seasonal_amp: float = 0.25
    supercycle_amp: float = 0.15
    supercycle_period_months: int = 120

    # Process noise
    noise_scale: float = 0.02


class OceanRLDynamics:
    """Adapter for the provided black-box transition function (oceanrl.query)."""

    def __init__(self):
        try:
            from oceanrl import query as ocean_query  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "oceanrl is not installed. Install the wheel provided by the course staff."
            ) from e
        self._query = ocean_query

    def step(
        self,
        salmon: float,
        shark: float,
        effort: float,
        month_1_indexed: int,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # oceanrl handles randomness internally; rng not used.
        salmon_caught, salmon_next, shark_next = self._query(
            float(salmon), float(shark), float(effort), int(month_1_indexed)
        )
        return float(salmon_caught), float(salmon_next), float(shark_next)


class ToyPredPreyDynamics:
    """A configurable predator-prey model with seasonality, saturation, and noise.

    This is NOT intended to match the hidden oceanrl dynamics exactly.
    It is a controllable stand-in to test algorithms and instrumentation locally.
    """

    def __init__(self, cfg: Optional[ToyDynamicsConfig] = None):
        self.cfg = cfg or ToyDynamicsConfig()

    def _seasonal_multiplier(self, month_1_indexed: int) -> float:
        m = (month_1_indexed - 1) % 12
        seasonal = math.sin(2.0 * math.pi * m / 12.0)
        supercycle = math.sin(
            2.0 * math.pi * (month_1_indexed - 1) / float(self.cfg.supercycle_period_months)
        )
        return 1.0 + self.cfg.seasonal_amp * seasonal + self.cfg.supercycle_amp * supercycle

    def step(
        self,
        salmon: float,
        shark: float,
        effort: float,
        month_1_indexed: int,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        salmon = max(0.0, float(salmon))
        shark = max(0.0, float(shark))
        effort = max(0.0, float(effort))

        mult = self._seasonal_multiplier(month_1_indexed)

        # Saturating catch: salmon * (1 - exp(-q * effort))
        catch_frac = 1.0 - math.exp(-self.cfg.catchability * effort)
        salmon_caught = salmon * catch_frac

        # Predator-prey interaction
        predation = self.cfg.predation_rate * salmon * shark

        # Logistic growth for salmon
        salmon_growth = (self.cfg.salmon_r0 * mult) * salmon * (1.0 - salmon / self.cfg.salmon_K)

        salmon_next = salmon + salmon_growth - predation - salmon_caught

        # Shark dynamics: growth from predation, mortality, carrying capacity
        shark_growth = (self.cfg.shark_conversion * predation) * mult
        shark_next = shark + shark_growth - self.cfg.shark_mortality * shark * (1.0 + shark / self.cfg.shark_K)

        # Multiplicative lognormal-ish noise
        if self.cfg.noise_scale > 0:
            eps_s = rng.normal(0.0, self.cfg.noise_scale)
            eps_k = rng.normal(0.0, self.cfg.noise_scale)
            salmon_next *= math.exp(eps_s)
            shark_next *= math.exp(eps_k)

        salmon_next = float(max(0.0, salmon_next))
        shark_next = float(max(0.0, shark_next))
        return float(salmon_caught), salmon_next, shark_next


def make_dynamics(backend: str):
    backend = backend.lower()
    if backend in ("auto", "oceanrl"):
        if backend == "oceanrl":
            return OceanRLDynamics()
        # auto mode
        try:
            return OceanRLDynamics()
        except Exception:
            return ToyPredPreyDynamics()
    if backend == "toy":
        return ToyPredPreyDynamics()
    raise ValueError(f"Unknown dynamics backend: {backend}")
