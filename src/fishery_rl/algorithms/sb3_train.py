from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import TQC  # type: ignore
    _TQC_AVAILABLE = True
except Exception:  # pragma: no cover
    _TQC_AVAILABLE = False
    TQC = None


def _make_vec_env(env_fn: Callable[[], object]):
    def _init():
        return Monitor(env_fn())
    return DummyVecEnv([_init])


def train_sb3(
    algo: str,
    env_fn: Callable[[], object],
    total_timesteps: int,
    out_dir: str,
    seed: int = 0,
    device: str = "auto",
    algo_kwargs: Optional[Dict] = None,
) -> str:
    """Train an SB3 algorithm and save model + VecNormalize stats."""
    os.makedirs(out_dir, exist_ok=True)
    algo = algo.lower()
    algo_kwargs = algo_kwargs or {}

    vec_env = _make_vec_env(env_fn)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    common = dict(
        env=vec_env,
        seed=seed,
        device=device,
        verbose=1,
        gamma=1.0,  # no discounting by default (long-horizon objective)
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    if algo == "ppo":
        model = PPO("MlpPolicy", **common, **algo_kwargs)
    elif algo == "sac":
        model = SAC("MlpPolicy", **common, **algo_kwargs)
    elif algo == "td3":
        model = TD3("MlpPolicy", **common, **algo_kwargs)
    elif algo in ("tqc", "distributional"):
        if not _TQC_AVAILABLE:
            raise RuntimeError("TQC requires sb3-contrib. Install: pip install sb3-contrib")
        model = TQC("MlpPolicy", **common, **algo_kwargs)  # type: ignore
    else:
        raise ValueError(f"Unsupported SB3 algo: {algo}")

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": algo,
                "total_timesteps": int(total_timesteps),
                "seed": int(seed),
                "device": device,
                "algo_kwargs": algo_kwargs,
            },
            f,
            indent=2,
        )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)

    model_path = os.path.join(out_dir, "model.zip")
    model.save(model_path)

    vec_path = os.path.join(out_dir, "vecnormalize.pkl")
    vec_env.save(vec_path)

    return model_path


def load_sb3_policy(
    algo: str,
    model_path: str,
    vecnormalize_path: Optional[str] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load an SB3 policy as a pure function obs -> action."""
    algo = algo.lower()
    if vecnormalize_path is None:
        vecnormalize_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")

    # Create a dummy VecEnv for VecNormalize (required by SB3)
    from gymnasium import spaces
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1e6], dtype=np.float32))

        def reset(self, *, seed=None, options=None):
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((4,), dtype=np.float32), 0.0, True, False, {}

    dummy = DummyVecEnv([lambda: _DummyEnv()])

    vec = None
    if os.path.exists(vecnormalize_path):
        vec = VecNormalize.load(vecnormalize_path, dummy)
        vec.training = False
        vec.norm_reward = False

    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    elif algo == "td3":
        model = TD3.load(model_path)
    elif algo in ("tqc", "distributional"):
        if not _TQC_AVAILABLE:
            raise RuntimeError("TQC requires sb3-contrib.")
        model = TQC.load(model_path)  # type: ignore
    else:
        raise ValueError(f"Unsupported SB3 algo: {algo}")

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32)
        if vec is not None:
            o = vec.normalize_obs(o[None, :])[0]
        action, _ = model.predict(o, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return policy_fn
