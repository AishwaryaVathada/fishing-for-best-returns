from __future__ import annotations

import argparse
import json
import os

from fishery_rl.algorithms.sb3_train import train_sb3
from fishery_rl.algorithms.cppo import train_cppo, CPPOConfig
from fishery_rl.algorithms.es import train_es, ESConfig
from fishery_rl.scripts._common import make_env_fn, ensure_out_dir


def main():
    p = argparse.ArgumentParser(description="Train RL agents for the fishery task.")
    p.add_argument("--algo", type=str, required=True, choices=["sac", "td3", "ppo", "tqc", "distributional", "cppo", "es"])
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "toy", "oceanrl"])
    p.add_argument("--timesteps", type=int, default=50_000, help="Training steps (SB3) or total_steps (C-PPO) or iterations proxy (ES).")
    p.add_argument("--horizon", type=int, default=900, help="Episode length in months.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="runs/run")
    p.add_argument("--no-shaping", action="store_true", help="Disable reward shaping penalties. Cost signal remains available.")
    p.add_argument("--penalty", type=float, default=200.0, help="Reward penalty per violation (if shaping enabled).")
    p.add_argument("--custom-env", type=str, default=None, help="Optional env import path 'module:Class'. Overrides backend.")
    args = p.parse_args()

    out = ensure_out_dir(args.out)
    env_fn = make_env_fn(
        backend=args.backend,
        horizon=args.horizon,
        seed=args.seed,
        reward_shaping=not args.no_shaping,
        penalty_per_violation=args.penalty,
        custom_env=args.custom_env,
    )

    algo = args.algo.lower()

    if algo in ("sac", "td3", "ppo", "tqc", "distributional"):
        model_path = train_sb3(
            algo="tqc" if algo == "distributional" else algo,
            env_fn=env_fn,
            total_timesteps=args.timesteps,
            out_dir=out,
            seed=args.seed,
            device=args.device,
            algo_kwargs={},
        )
    elif algo == "cppo":
        cfg = CPPOConfig(total_steps=args.timesteps)
        model_path = train_cppo(env_fn, out, cfg=cfg, seed=args.seed, device="cpu" if args.device == "auto" else args.device)
    elif algo == "es":
        cfg = ESConfig(total_iterations=max(10, args.timesteps // 500), seed=args.seed)
        model_path = train_es(env_fn, out, cfg=cfg, device="cpu" if args.device == "auto" else args.device)
    else:
        raise ValueError(algo)

    print(model_path)


if __name__ == "__main__":
    main()
