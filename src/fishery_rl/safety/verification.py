from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from fishery_rl.safety.constraints import SafetyConstraint, HardPopulationConstraint
from fishery_rl.evaluation.rollout import rollout_episode
from fishery_rl.evaluation.metrics import summarize_returns


@dataclass
class PolicyVerificationReport:
    n_rollouts: int
    mean_return: float
    std_return: float
    q05: float
    q50: float
    q95: float
    cvar10: float
    violation_rate: float
    mean_time_to_violation: Optional[float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def verify_policy(
    env_fn: Callable[[], object],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    n_rollouts: int = 100,
    seed: int = 0,
    constraint: Optional[SafetyConstraint] = None,
) -> PolicyVerificationReport:
    """Monte Carlo verification under stochasticity and random initial conditions.

    Returns a report with distributional return statistics and safety violation rates.
    """
    constraint = constraint or HardPopulationConstraint()

    returns: List[float] = []
    violations: List[bool] = []
    ttv: List[int] = []

    for i in range(n_rollouts):
        ep = rollout_episode(env_fn, policy_fn, seed=seed + i, constraint=constraint)
        returns.append(ep["return"])
        violations.append(ep["violated"])
        if ep["time_to_violation"] is not None:
            ttv.append(int(ep["time_to_violation"]))

    stats = summarize_returns(np.array(returns, dtype=np.float64))
    violation_rate = float(np.mean(violations))
    mean_ttv = float(np.mean(ttv)) if len(ttv) > 0 else None

    return PolicyVerificationReport(
        n_rollouts=int(n_rollouts),
        mean_return=float(stats["mean"]),
        std_return=float(stats["std"]),
        q05=float(stats["q05"]),
        q50=float(stats["q50"]),
        q95=float(stats["q95"]),
        cvar10=float(stats["cvar10"]),
        violation_rate=violation_rate,
        mean_time_to_violation=mean_ttv,
    )
