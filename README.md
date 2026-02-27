# Reinforcement Learning for Long-Horizon Safety
## Policy Verification Under Uncertainty for Sustainable Fishery Management

This repository implements a modular reinforcement learning (RL) experimentation stack for **long-horizon control with explicit safety constraints**, using a sustainable fishery management task as the reference domain.

The original project specification assumes an opaque transition function (a provided wheel). Because the exact runtime environment may be unavailable during development, this repo is designed so that **any compatible environment/dynamics backend can be plugged in** while keeping training, tuning, and verification unchanged. Your current draft is preserved conceptually but has been reformatted and aligned to standard software best practices. fileciteturn1file0

---

## Problem and task definition

**State:**  
A 2-species ecosystem state observed monthly:
- salmon population
- shark population
- month index (encoded as seasonal features)

**Action:**  
A continuous **non-negative fishing effort**.

**Objective (long-horizon):**  
Maximize harvest yield while maintaining ecosystem sustainability over **900 months (75 years)**, with safety boundaries on minimum viable populations.

### Reward and terminal bonus

Per-step reward:

- `r_t = K1 * salmon_caught_t - K2 * effort_t`
- Optional reward shaping adds a penalty when safety constraints are violated.

Terminal bonus at the end of the horizon:

- `K3 * log(salmon_T) + K4 * log(shark_T)`

### Safety constraints and cost signal

Safety boundaries are encoded as a **cost signal**:
- `cost = 1` if `(salmon < salmon_min_safe) OR (shark < shark_min_safe)` else `0`.

This supports:
- **reward shaping** (penalize violations in the reward), and
- **constrained optimization** (C-PPO optimizes expected return subject to expected cost limits).

---

## Algorithms implemented

This repo supports the requested algorithm set:

### Stable-Baselines3 (SB3)
- **PPO**
- **SAC**
- **TD3**

### Distributional RL (continuous control)
- **TQC** (Truncated Quantile Critics, via `sb3-contrib`)

### Constrained RL
- **Constrained PPO (C-PPO)** implemented in PyTorch  
  A **primal-dual Lagrangian PPO** variant with a learnable dual multiplier λ, using a separate cost value function.

### Gradient-free optimization
- **Evolution Strategies (ES)** with antithetic sampling and rank-based shaping.

---

## Environment backends (pluggable)

The environment is a Gymnasium-compatible `FisheryEnv` with interchangeable transition dynamics:

- `backend=oceanrl`: calls the provided black-box transition function (if the wheel is installed).
- `backend=toy`: a research-grade predator–prey model with seasonality, effort saturation, and process noise.
- `backend=auto`: tries `oceanrl`, falls back to `toy`.

This design supports development when the true environment is unavailable, while maintaining the same algorithm and evaluation interfaces.

---

## Repository layout

```text
src/fishery_rl/
  envs/         FisheryEnv + transition backends (oceanrl / toy)
  safety/       Constraints, cost signal, and verification harness
  algorithms/   SB3 training, C-PPO (PyTorch), ES (PyTorch)
  tuning/       Optuna tuning driver (shared across algorithms)
  evaluation/   Rollout utilities and return-risk metrics
  scripts/      CLI entrypoints: train / eval / tune / verify
tests/          Minimal smoke test
```

---

## Installation

### Requirements
- Python 3.10+
- (Optional) CUDA-capable GPU for faster training with SB3/Torch

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional: if you have the staff-provided wheel:

```bash
pip install path/to/oceanrl-0.1.0-py3-none-any.whl
```

---

## Usage

See **QUICKSTART.md** for a minimal smoke test and common workflows.

### Train

```bash
python -m fishery_rl.scripts.train --algo ppo --backend toy --timesteps 50000 --horizon 900 --out runs/ppo_toy
```

### Evaluate (distributional safety/return stats)

```bash
python -m fishery_rl.scripts.eval --algo ppo --backend toy --model runs/ppo_toy/model.zip --n-eval 30 --horizon 900
```

### Verify under uncertainty (Monte Carlo policy verification)

```bash
python -m fishery_rl.scripts.verify --algo ppo --backend toy --model runs/ppo_toy/model.zip --n-rollouts 100 --horizon 900 --out runs/ppo_toy
```

### Hyperparameter tuning (Optuna)

```bash
python -m fishery_rl.scripts.tune --algo td3 --backend toy --trials 25 --train-steps 20000 --eval-episodes 5 --horizon 90 --out optuna_results/td3_toy
```

The Optuna objective is:

- `score = mean_return - safety_penalty * violation_rate`

This explicitly trades off performance and safety.

---

## Extending to new environments

### Plugging in a custom environment (Gymnasium)

All entrypoints support `--custom-env module:Class` to inject a Gymnasium-compatible env.

Example:

```bash
python -m fishery_rl.scripts.train --algo ppo --custom-env my_pkg.my_env:MyEnv --timesteps 10000 --out runs/custom_env_ppo
```

**Requirements for custom envs:**
- Implements `reset()` and `step()`
- Defines `observation_space` and `action_space`
- Returns an `info` dict with a numeric `cost` field for constrained/verification workflows (recommended)

### Plugging in a new dynamics backend

Add a new dynamics class in `fishery_rl/envs/dynamics.py` matching:

- `step(salmon, shark, effort, month_1_indexed, rng) -> (salmon_caught, salmon_next, shark_next)`

---

## Reproducibility

- Deterministic seeding is supported in all scripts.
- Each run directory writes:
  - `config.json`
  - model checkpoint (`model.zip` or `model.pt`)
  - TensorBoard logs (`runs/.../tb`)

TensorBoard:

```bash
tensorboard --logdir runs
```

---

## License

MIT. See `LICENSE`.
