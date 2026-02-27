# Quickstart
## Reinforcement Learning for Long-Horizon Safety (Fishery)

This guide gets you from a clean machine to a verified end-to-end run in a few minutes, using the **toy backend** (no staff wheel required). It also shows how to swap in the opaque `oceanrl` backend when available.

---

## 1) Prerequisites

- Python **3.10+**
- Git
- (Optional) CUDA GPU for faster training

---

## 2) Setup and install

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## 3) Smoke test (fast local run)

Use a short horizon and few timesteps to validate that:
- the environment runs,
- the algorithm trains,
- the model saves,
- verification produces metrics.

Train:

```bash
python -m fishery_rl.scripts.train \
  --algo ppo \
  --backend toy \
  --timesteps 3000 \
  --horizon 60 \
  --out runs/smoke_ppo_toy
```

Verify:

```bash
python -m fishery_rl.scripts.verify \
  --algo ppo \
  --backend toy \
  --model runs/smoke_ppo_toy/model.zip \
  --n-rollouts 10 \
  --horizon 60 \
  --out runs/smoke_ppo_toy
```

You should see a JSON report including mean return and violation rate.

---

## 4) Try each requested algorithm (short runs)

SB3 algorithms:

```bash
python -m fishery_rl.scripts.train --algo sac --backend toy --timesteps 5000 --horizon 60 --out runs/smoke_sac_toy
python -m fishery_rl.scripts.train --algo td3 --backend toy --timesteps 5000 --horizon 60 --out runs/smoke_td3_toy
python -m fishery_rl.scripts.train --algo ppo --backend toy --timesteps 5000 --horizon 60 --out runs/smoke_ppo_toy
```

Distributional RL (TQC):

```bash
python -m fishery_rl.scripts.train --algo tqc --backend toy --timesteps 5000 --horizon 60 --out runs/smoke_tqc_toy
```

Constrained PPO (PyTorch):

```bash
python -m fishery_rl.scripts.train --algo cppo --backend toy --timesteps 8000 --horizon 60 --out runs/smoke_cppo_toy --device cpu
python -m fishery_rl.scripts.verify --algo cppo --backend toy --model runs/smoke_cppo_toy/model.pt --n-rollouts 10 --horizon 60
```

Evolution Strategies:

```bash
python -m fishery_rl.scripts.train --algo es --backend toy --timesteps 10000 --horizon 60 --out runs/smoke_es_toy --device cpu
python -m fishery_rl.scripts.verify --algo es --backend toy --model runs/smoke_es_toy/model.pt --n-rollouts 10 --horizon 60
```

Notes:
- SB3 will use GPU if Torch + CUDA are available and you pass `--device cuda`.
- ES/C-PPO are implemented in PyTorch and default to CPU for portability.

---

## 5) Optuna tuning (small study)

Recommended: tune with a shorter horizon for speed, then re-train best params at full horizon.

```bash
python -m fishery_rl.scripts.tune \
  --algo td3 \
  --backend toy \
  --trials 10 \
  --train-steps 8000 \
  --eval-episodes 3 \
  --horizon 90 \
  --out optuna_results/td3_toy_small
```

The `best.json` file is written to the output directory.

---

## 6) Switch to the opaque oceanrl backend (if available)

If you have the staff wheel:

```bash
pip install path/to/oceanrl-0.1.0-py3-none-any.whl
```

Then run training against the true dynamics:

```bash
python -m fishery_rl.scripts.train --algo sac --backend oceanrl --timesteps 20000 --horizon 120 --out runs/sac_oceanrl_small
```

---

## 7) TensorBoard

```bash
tensorboard --logdir runs
```

Open the URL printed by TensorBoard to inspect learning curves.

---

## Troubleshooting

### TQC import error
If `tqc` fails to import, install `sb3-contrib`:

```bash
pip install sb3-contrib
```

### Gymnasium vs gym
This repo uses `gymnasium`. If you have legacy `gym` conflicts, start from a clean venv.

### Windows execution policy
If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
