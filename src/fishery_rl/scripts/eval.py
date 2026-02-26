from __future__ import annotations

import argparse
import os

from fishery_rl.algorithms.sb3_train import load_sb3_policy
from fishery_rl.algorithms.cppo import load_cppo_policy
from fishery_rl.algorithms.es import load_es_policy
from fishery_rl.safety.verification import verify_policy
from fishery_rl.scripts._common import make_env_fn


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained agent.")
    p.add_argument("--algo", type=str, required=True, choices=["sac", "td3", "ppo", "tqc", "distributional", "cppo", "es"])
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "toy", "oceanrl"])
    p.add_argument("--model", type=str, required=True, help="Path to model file (SB3 model.zip or model.pt).")
    p.add_argument("--vecnorm", type=str, default=None, help="Optional VecNormalize path (SB3 only).")
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--horizon", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
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

    report = verify_policy(env_fn, policy, n_rollouts=args.n_eval, seed=args.seed + 999)
    print(report.to_json())


if __name__ == "__main__":
    main()
