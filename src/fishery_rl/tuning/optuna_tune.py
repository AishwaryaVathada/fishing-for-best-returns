from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional

import numpy as np
import optuna

from fishery_rl.algorithms.sb3_train import train_sb3, load_sb3_policy
from fishery_rl.algorithms.cppo import train_cppo, load_cppo_policy, CPPOConfig
from fishery_rl.algorithms.es import train_es, load_es_policy, ESConfig
from fishery_rl.safety.verification import verify_policy


def _suggest_sb3_kwargs(trial: optuna.Trial, algo: str) -> Dict:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    net = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net == "small":
        policy_kwargs = dict(net_arch=[128, 128])
    elif net == "medium":
        policy_kwargs = dict(net_arch=[256, 256])
    else:
        policy_kwargs = dict(net_arch=[512, 512, 512])

    if algo == "ppo":
        return {
            "learning_rate": lr,
            "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 15),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
            "policy_kwargs": policy_kwargs,
        }
    # off-policy
    return {
        "learning_rate": lr,
        "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 1_000_000]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "policy_kwargs": policy_kwargs,
    }


def tune(
    algo: str,
    env_fn: Callable[[], object],
    out_dir: str,
    trials: int = 25,
    train_steps: int = 20_000,
    eval_episodes: int = 5,
    seed: int = 0,
    device: str = "auto",
    safety_penalty: float = 1e6,
):
    """Unified Optuna tuning for SB3 algorithms, C-PPO, and ES.

    Objective = mean_return - safety_penalty * violation_rate
    """
    os.makedirs(out_dir, exist_ok=True)
    algo = algo.lower()

    def objective(trial: optuna.Trial) -> float:
        trial_dir = os.path.join(out_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        if algo in ("sac", "td3", "ppo", "tqc", "distributional"):
            kwargs = _suggest_sb3_kwargs(trial, "ppo" if algo == "ppo" else "off")
            model_path = train_sb3(
                algo=algo if algo != "distributional" else "tqc",
                env_fn=env_fn,
                total_timesteps=train_steps,
                out_dir=trial_dir,
                seed=seed + trial.number,
                device=device,
                algo_kwargs=kwargs,
            )
            policy = load_sb3_policy(algo if algo != "distributional" else "tqc", model_path)

        elif algo == "cppo":
            cfg = CPPOConfig(
                total_steps=train_steps,
                lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                clip_range=trial.suggest_float("clip_range", 0.10, 0.30),
                lambda_lr=trial.suggest_float("lambda_lr", 0.01, 0.2, log=True),
                cost_limit=trial.suggest_float("cost_limit", 0.0, 0.05),
            )
            model_path = train_cppo(env_fn, trial_dir, cfg=cfg, seed=seed + trial.number, device="cpu" if device == "auto" else device)
            policy = load_cppo_policy(model_path, device="cpu" if device == "auto" else device)

        elif algo == "es":
            cfg = ESConfig(
                total_iterations=max(10, train_steps // 500),
                population=trial.suggest_categorical("population", [16, 32, 48]),
                sigma=trial.suggest_float("sigma", 0.01, 0.2, log=True),
                lr=trial.suggest_float("lr", 0.005, 0.1, log=True),
                seed=seed + trial.number,
            )
            model_path = train_es(env_fn, trial_dir, cfg=cfg, device="cpu" if device == "auto" else device)
            policy = load_es_policy(model_path, device="cpu" if device == "auto" else device)

        else:
            raise ValueError(f"Unknown algo: {algo}")

        report = verify_policy(env_fn, policy, n_rollouts=eval_episodes, seed=seed + 10_000 + trial.number)
        score = report.mean_return - safety_penalty * report.violation_rate
        trial.set_user_attr("mean_return", report.mean_return)
        trial.set_user_attr("violation_rate", report.violation_rate)
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(trials), show_progress_bar=True)

    best = {
        "algo": algo,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_attrs": dict(study.best_trial.user_attrs),
    }

    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return best
