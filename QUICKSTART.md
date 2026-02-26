Quickstart

This repository provides a modular RL framework for long-horizon sustainable fishery management with safety constraints.

1) Create environment and install dependencies

Windows (PowerShell)
    python -m venv fishery
    .\fishery\Scripts\activate
    pip install -U pip
    pip install -r requirements.txt
    pip install -e .

macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    pip install -e .

2) Smoke-test on the toy dynamics (fast, no wheel needed)

    python -m fishery_rl.scripts.train --algo ppo --backend toy --timesteps 3000 --horizon 60 --out runs/smoke
    python -m fishery_rl.scripts.eval  --algo ppo --backend toy --model runs/smoke/model.zip --n-eval 5 --horizon 60

3) If you have the provided wheel (oceanrl), install it and run a small training job

    pip install path/to/oceanrl-0.1.0-py3-none-any.whl
    python -m fishery_rl.scripts.train --algo sac --backend oceanrl --timesteps 20000 --horizon 120 --out runs/oceanrl_small

4) Hyperparameter tuning with Optuna (toy backend is recommended for fast iteration)

    python -m fishery_rl.scripts.tune --algo td3 --backend toy --trials 10 --train-steps 8000 --eval-episodes 3 --horizon 90

5) Benchmark all requested algorithms (SAC, PPO, TD3, C-PPO, Distributional/TQC, ES) with Optuna outputs

    python -m fishery_rl.scripts.benchmark --backend toy --train-steps 6000 --horizon 90 --verify-rollouts 20 --optuna-trials 3 --optuna-train-steps 4000 --optuna-eval-episodes 3 --out runs/benchmark_all

6) Safety verification under uncertainty (multi-seed rollouts + violation statistics)

    python -m fishery_rl.scripts.verify --algo ppo --backend toy --model runs/smoke/model.zip --n-rollouts 30 --horizon 60

7) Optional LLM reasoning over benchmark outputs

    set OPENROUTER_API_KEY=your_key_here
    python -m fishery_rl.scripts.llm_analyze --summary runs/benchmark_all/summary.json

8) TensorBoard

    tensorboard --logdir runs

9) Unknown/pluggable env support for all scripts

    python -m fishery_rl.scripts.train --algo ppo --custom-env my_pkg.my_env:MyEnv --timesteps 10000 --out runs/custom
