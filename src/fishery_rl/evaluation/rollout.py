from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from fishery_rl.safety.constraints import SafetyConstraint, HardPopulationConstraint


def rollout_episode(
    env_fn: Callable[[], object],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    seed: int = 0,
    constraint: Optional[SafetyConstraint] = None,
) -> Dict:
    """Roll out one full episode and return structured metrics."""
    constraint = constraint or HardPopulationConstraint()
    env = env_fn()

    obs, info = env.reset(seed=seed)
    done = False

    ep_return = 0.0
    violated = False
    time_to_violation = None
    t = 0

    while not done:
        act = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        done = bool(terminated) or bool(truncated)
        ep_return += float(reward)

        if (not violated) and constraint.violated(info):
            violated = True
            time_to_violation = t

        t += 1

    return {
        "return": float(ep_return),
        "violated": bool(violated),
        "time_to_violation": time_to_violation,
        "length": int(t),
    }
