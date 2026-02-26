from __future__ import annotations

import argparse
import json
import os

from fishery_rl.scripts._common import make_env_fn, ensure_out_dir
from fishery_rl.tuning.optuna_tune import tune


def main():
    p = argparse.ArgumentParser(description="Optuna tuning for fishery RL agents.")
    p.add_argument("--algo", type=str, required=True, choices=["sac", "td3", "ppo", "tqc", "distributional", "cppo", "es"])
    p.add_argument("--backend", type=str, default="toy", choices=["auto", "toy", "oceanrl"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="optuna_results/study")
    p.add_argument("--safety-penalty", type=float, default=1e6)
    p.add_argument("--custom-env", type=str, default=None, help="Optional env import path 'module:Class'. Overrides backend.")
    args = p.parse_args()

    out = ensure_out_dir(args.out)
    env_fn = make_env_fn(args.backend, args.horizon, args.seed, custom_env=args.custom_env)

    best = tune(
        algo=args.algo,
        env_fn=env_fn,
        out_dir=out,
        trials=args.trials,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        safety_penalty=args.safety_penalty,
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
