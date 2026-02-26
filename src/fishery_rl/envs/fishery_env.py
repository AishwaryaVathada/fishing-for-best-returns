from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fishery_rl.envs.dynamics import make_dynamics, Dynamics


@dataclass(frozen=True)
class FisheryEnvConfig:
    # Reward coefficients from the assignment spec
    K1: float = 0.001
    K2: float = 0.01
    K3: float = 100.0
    K4: float = 100.0

    # Episode
    horizon_months: int = 900
    max_effort: float = 1e6  # Numerical safety cap; action is conceptually unbounded.

    # Initial ranges
    salmon_init_min: int = 10_000
    salmon_init_max: int = 30_000
    shark_init_min: int = 400
    shark_init_max: int = 600

    # Safety boundaries (hard constraints)
    salmon_min_safe: float = 3_000.0
    shark_min_safe: float = 150.0

    # If reward_shaping=True, violations add -penalty_per_violation to reward
    reward_shaping: bool = True
    penalty_per_violation: float = 200.0

    # Observation design
    use_log_obs: bool = True


class FisheryEnv(gym.Env):
    """Long-horizon sustainable fishery environment.

    State:
        (salmon_t, shark_t, month_t)

    Action:
        non-negative fishing effort

    Reward (per step):
        r_t = K1 * salmon_caught_t - K2 * effort_t
      + optional shaping penalties for safety violations

    Terminal bonus:
        K3 * log(salmon_T) + K4 * log(shark_T)

    Notes:
        - The month index passed to the black-box dynamics is 1-indexed (1..horizon).
        - For algorithm stability, effort is capped at max_effort (default 1e6).
    """

    metadata = {"render_modes": []}

    def __init__(self, backend: str = "auto", seed: int = 0, cfg: Optional[FisheryEnvConfig] = None):
        super().__init__()
        self.cfg = cfg or FisheryEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.dyn: Dynamics = make_dynamics(backend)

        # Observation: [salmon, shark, sin(month), cos(month)] in log-space by default
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.cfg.max_effort], dtype=np.float32),
            dtype=np.float32,
        )

        self.salmon: float = 0.0
        self.shark: float = 0.0
        self.month0: int = 0  # internal 0-index for convenience
        self.t: int = 0

    def _encode(self, salmon: float, shark: float, month0: int) -> np.ndarray:
        if self.cfg.use_log_obs:
            s = math.log1p(max(0.0, salmon))
            k = math.log1p(max(0.0, shark))
        else:
            s, k = float(salmon), float(shark)
        m = month0 % 12
        sin_m = math.sin(2.0 * math.pi * m / 12.0)
        cos_m = math.cos(2.0 * math.pi * m / 12.0)
        return np.array([s, k, sin_m, cos_m], dtype=np.float32)

    def _violation(self, salmon: float, shark: float) -> bool:
        return (salmon < self.cfg.salmon_min_safe) or (shark < self.cfg.shark_min_safe)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.salmon = float(self.rng.integers(self.cfg.salmon_init_min, self.cfg.salmon_init_max))
        self.shark = float(self.rng.integers(self.cfg.shark_init_min, self.cfg.shark_init_max))
        self.month0 = 0
        self.t = 0
        obs = self._encode(self.salmon, self.shark, self.month0)
        info = {"salmon": self.salmon, "shark": self.shark, "month": self.month0, "cost": 0.0}
        return obs, info

    def step(self, action):
        effort = float(np.clip(float(action[0]), 0.0, self.cfg.max_effort))
        month_1_indexed = self.month0 + 1

        caught, salmon_next, shark_next = self.dyn.step(
            self.salmon, self.shark, effort, month_1_indexed, self.rng
        )

        reward = self.cfg.K1 * caught - self.cfg.K2 * effort

        violation = self._violation(salmon_next, shark_next)
        cost = 1.0 if violation else 0.0

        if self.cfg.reward_shaping and violation:
            reward -= self.cfg.penalty_per_violation

        self.salmon, self.shark = float(salmon_next), float(shark_next)
        self.month0 += 1
        self.t += 1

        terminated = self.t >= self.cfg.horizon_months

        if terminated:
            reward += self.cfg.K3 * math.log(max(1e-12, self.salmon))
            reward += self.cfg.K4 * math.log(max(1e-12, self.shark))

        obs = self._encode(self.salmon, self.shark, self.month0)
        info: Dict[str, float] = {
            "salmon": self.salmon,
            "shark": self.shark,
            "month": float(self.month0),
            "salmon_caught": float(caught),
            "effort": effort,
            "cost": float(cost),
        }
        return obs, float(reward), bool(terminated), False, info
