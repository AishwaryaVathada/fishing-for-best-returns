Reinforcement Learning for Long-Horizon Safety: Policy Verification Under Uncertainty

This project implements a modular RL experimentation stack for sustainable fishery management with long-horizon objectives and explicit safety constraints. The reference task is a two-species ecosystem (salmon and sharks) where the agent chooses a continuous fishing effort each month to balance harvest yield against sustainability over a 75-year horizon (900 months). The reward structure and terminal sustainability bonuses follow the project specification.

Key properties:
- Long-horizon episodic control (default horizon: 900 steps, gamma = 1.0)
- Continuous action space (non-negative fishing effort)
- Multiple algorithms:
  - SAC, TD3, PPO (Stable-Baselines3)
  - Distributional RL: TQC (sb3-contrib, quantile critics for continuous control)
  - Constrained PPO (C-PPO): primal-dual Lagrangian PPO implemented in PyTorch
  - Evolution Strategies (ES): gradient-free policy search (OpenAI-style, antithetic sampling)
- Optuna hyperparameter search with multi-seed validation
- Policy verification under uncertainty via Monte Carlo rollouts and safety violation statistics

Environment backends:
- oceanrl backend: uses the provided black-box transition function (if installed)
- toy backend: a research-grade predator-prey model with seasonality, saturation effects for effort, and process noise
  (useful for local testing when the wheel or target runtime is unavailable)

Repository layout

src/fishery_rl/
  envs/         Environment + transition dynamics backends (oceanrl / toy)
  safety/       Constraints, cost functions, and verification harness
  algorithms/   Training backends (SB3), C-PPO (PyTorch), ES (PyTorch)
  tuning/       Optuna tuning driver (shared across algorithms)
  evaluation/   Rollout utilities and metrics
  scripts/      CLI entrypoints for train / eval / tune / verify

Install

    pip install -r requirements.txt
    pip install -e .

Windows quick env setup (named `fishery`)

    python -m venv fishery
    .\fishery\Scripts\activate
    python -m pip install -U pip
    pip install -r requirements.txt
    pip install -e .

Running a fast local smoke test (toy backend)

    python -m fishery_rl.scripts.train --algo ppo --backend toy --timesteps 3000 --horizon 60 --out runs/smoke
    python -m fishery_rl.scripts.verify --algo ppo --backend toy --model runs/smoke/model.zip --n-rollouts 30 --horizon 60

Run all six requested algorithms + Optuna + optional LLM analysis

    python -m fishery_rl.scripts.benchmark --backend toy --train-steps 6000 --horizon 90 --verify-rollouts 20 --optuna-trials 3 --optuna-train-steps 4000 --optuna-eval-episodes 3 --out runs/benchmark_all

Optional: add deep LLM interpretation for already-generated results

    set OPENROUTER_API_KEY=your_key_here
    python -m fishery_rl.scripts.llm_analyze --summary runs/benchmark_all/summary.json

Training on the oceanrl backend

    pip install path/to/oceanrl-0.1.0-py3-none-any.whl
    python -m fishery_rl.scripts.train --algo sac --backend oceanrl --timesteps 200000 --out runs/sac_oceanrl

Hyperparameter tuning

    python -m fishery_rl.scripts.tune --algo tqc --backend toy --trials 30 --train-steps 20000 --eval-episodes 5

Unknown/pluggable environment support

- `train`, `eval`, `verify`, `tune`, and `benchmark` support `--custom-env module:Class`.
- The target class must follow Gymnasium env conventions (`reset`/`step`, `observation_space`, `action_space`).
- Example:

    python -m fishery_rl.scripts.train --algo ppo --custom-env my_pkg.my_env:MyEnv --timesteps 10000 --out runs/custom_env_ppo

Policy verification

Verification is implemented as a reproducible evaluation protocol:
- multi-seed Monte Carlo rollouts
- return distribution summary (mean, std, quantiles, CVaR)
- safety violation probability and time-to-violation statistics

    python -m fishery_rl.scripts.verify --algo cppo --backend toy --model runs/cppo/model.pt --n-rollouts 100

Notes on safety constraints

The environment provides a cost signal based on hard safety boundaries (salmon >= salmon_min, sharks >= shark_min).
- In reward-shaping mode, violations add a penalty term to reward.
- In constrained mode (C-PPO), the optimizer uses a Lagrangian dual variable to enforce an expected-cost limit.

Reproducibility and rigor

All scripts support:
- deterministic seeding
- run directories with config.json, metrics.csv, and checkpoints
- multi-seed evaluation

License: MIT
