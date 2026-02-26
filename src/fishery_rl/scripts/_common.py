from __future__ import annotations

import os
from typing import Callable, Optional

from fishery_rl.envs.fishery_env import FisheryEnv, FisheryEnvConfig
from fishery_rl.utils.imports import import_from_path


def make_env_fn(
    backend: str,
    horizon: int,
    seed: int,
    reward_shaping: bool = True,
    penalty_per_violation: float = 200.0,
    custom_env: Optional[str] = None,
) -> Callable[[], object]:
    if custom_env is not None:
        EnvClass = import_from_path(custom_env)
        def _fn():
            return EnvClass()
        return _fn

    cfg = FisheryEnvConfig(
        horizon_months=int(horizon),
        reward_shaping=bool(reward_shaping),
        penalty_per_violation=float(penalty_per_violation),
    )

    def _fn():
        return FisheryEnv(backend=backend, seed=seed, cfg=cfg)

    return _fn


def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
