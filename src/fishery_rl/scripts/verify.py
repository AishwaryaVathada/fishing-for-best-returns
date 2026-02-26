from __future__ import annotations

import argparse
import json
import os

from fishery_rl.algorithms.sb3_train import load_sb3_policy
from fishery_rl.algorithms.cppo import load_cppo_policy
from fishery_rl.algorithms.es import load_es_policy
from fishery_rl.safety.verification import verify_policy
from fishery_rl.scripts._common import make_env_fn, ensure_out_dir


def main():
    p = argparse.ArgumentParser(description="Policy verification under uncertainty.")
    p.add_argument("--algo", type=str, required=True, choices=["sac", "td3", "ppo", "tqc", "distributional", "cppo", "es"])
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "toy", "oceanrl"])
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--vecnorm", type=str, default=None)
    p.add_argument("--n-rollouts", type=int, default=100)
    p.add_argument("--horizon", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None, help="Optional output directory to save report.json")
    p.add_argument("--custom-env", type=str, default=None, help="Optional env import path 'module:Class'. Overrides backend.")
    args = p.parse_args()

    env_fn = make_env_fn(args.backend, args.horizon, args.seed, custom_env=args.custom_env)

    algo = args.algo.lower()
    if algo in ("sac", "td3", "ppo", "tqc", "distributional"):
        policy = load_sb3_policy("tqc" if algo == "distributional" else algo, args.model, args.vecnorm)
    elif algo == "cppo":
        policy = load_cppo_policy(args.model, device=args.device)
    elif algo == "es":
        policy = load_es_policy(args.model, device=args.device)
    else:
        raise ValueError(algo)

    report = verify_policy(env_fn, policy, n_rollouts=args.n_rollouts, seed=args.seed + 10_000)
    print(report.to_json())

    if args.out is not None:
        out = ensure_out_dir(args.out)
        with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
            f.write(report.to_json())


if __name__ == "__main__":
    main()
