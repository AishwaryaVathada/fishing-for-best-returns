from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from fishery_rl.algorithms.sb3_train import train_sb3, load_sb3_policy
from fishery_rl.algorithms.cppo import train_cppo, load_cppo_policy, CPPOConfig
from fishery_rl.algorithms.es import train_es, load_es_policy, ESConfig
from fishery_rl.safety.verification import verify_policy
from fishery_rl.scripts._common import make_env_fn, ensure_out_dir
from fishery_rl.tuning.optuna_tune import tune


ALGOS_DEFAULT = ["sac", "ppo", "td3", "cppo", "distributional", "es"]


def _train_and_load_policy(
    algo: str,
    env_fn,
    train_steps: int,
    seed: int,
    device: str,
    out_dir: str,
):
    algo = algo.lower()
    if algo in ("sac", "td3", "ppo", "tqc", "distributional"):
        sb3_algo = "tqc" if algo == "distributional" else algo
        model_path = train_sb3(
            algo=sb3_algo,
            env_fn=env_fn,
            total_timesteps=train_steps,
            out_dir=out_dir,
            seed=seed,
            device=device,
            algo_kwargs={},
        )
        policy = load_sb3_policy(sb3_algo, model_path)
        return model_path, policy
    if algo == "cppo":
        model_path = train_cppo(
            env_fn=env_fn,
            out_dir=out_dir,
            cfg=CPPOConfig(total_steps=train_steps),
            seed=seed,
            device="cpu" if device == "auto" else device,
        )
        policy = load_cppo_policy(model_path, device="cpu" if device == "auto" else device)
        return model_path, policy
    if algo == "es":
        model_path = train_es(
            env_fn=env_fn,
            out_dir=out_dir,
            cfg=ESConfig(total_iterations=max(10, train_steps // 500), seed=seed),
            device="cpu" if device == "auto" else device,
        )
        policy = load_es_policy(model_path, device="cpu" if device == "auto" else device)
        return model_path, policy
    raise ValueError(f"Unsupported algo: {algo}")


def _call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int = 120,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _render_llm_prompt(summary: Dict) -> str:
    return (
        "Analyze these RL benchmark results for fishery sustainability and safety.\n"
        "Focus on long-horizon return, violation-rate tradeoffs, risk metrics (CVaR), and why "
        "different algorithms behaved differently.\n"
        "Provide concrete recommendations for next experiments.\n\n"
        f"RESULTS_JSON:\n{json.dumps(summary, indent=2)}"
    )


def main():
    p = argparse.ArgumentParser(description="Benchmark all requested RL algorithms + optional Optuna + optional LLM analysis.")
    p.add_argument("--algos", nargs="+", default=ALGOS_DEFAULT, choices=["sac", "td3", "ppo", "tqc", "distributional", "cppo", "es"])
    p.add_argument("--backend", type=str, default="toy", choices=["auto", "toy", "oceanrl"])
    p.add_argument("--custom-env", type=str, default=None, help="Optional env import path 'module:Class'. Overrides backend.")
    p.add_argument("--train-steps", type=int, default=10_000)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--verify-rollouts", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="runs/benchmark")

    p.add_argument("--optuna-trials", type=int, default=5)
    p.add_argument("--optuna-train-steps", type=int, default=8_000)
    p.add_argument("--optuna-eval-episodes", type=int, default=5)
    p.add_argument("--skip-optuna", action="store_true")

    p.add_argument("--llm-base-url", type=str, default="https://openrouter.ai/api/v1")
    p.add_argument("--llm-model", type=str, default="deepseek/deepseek-chat-v3-0324:free")
    p.add_argument("--llm-api-key-env", type=str, default="OPENROUTER_API_KEY")
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args()

    out = ensure_out_dir(args.out)
    env_fn = make_env_fn(
        backend=args.backend,
        horizon=args.horizon,
        seed=args.seed,
        custom_env=args.custom_env,
    )

    summary: Dict[str, object] = {
        "backend": args.backend,
        "custom_env": args.custom_env,
        "train_steps": args.train_steps,
        "horizon": args.horizon,
        "verify_rollouts": args.verify_rollouts,
        "results": {},
        "optuna": {},
    }

    for i, algo in enumerate(args.algos):
        algo_out = ensure_out_dir(os.path.join(out, algo))
        start = time.time()
        model_path, policy = _train_and_load_policy(
            algo=algo,
            env_fn=env_fn,
            train_steps=args.train_steps,
            seed=args.seed + i,
            device=args.device,
            out_dir=algo_out,
        )
        report = verify_policy(env_fn, policy, n_rollouts=args.verify_rollouts, seed=args.seed + 10_000 + i)
        elapsed = time.time() - start
        summary["results"][algo] = {
            "model_path": model_path,
            "train_and_verify_seconds": elapsed,
            "verification": json.loads(report.to_json()),
        }

        if not args.skip_optuna:
            tune_out = ensure_out_dir(os.path.join(out, "optuna", algo))
            best = tune(
                algo=algo,
                env_fn=env_fn,
                out_dir=tune_out,
                trials=args.optuna_trials,
                train_steps=args.optuna_train_steps,
                eval_episodes=args.optuna_eval_episodes,
                seed=args.seed + i,
                device=args.device,
            )
            summary["optuna"][algo] = best

        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if not args.skip_llm:
        api_key = os.getenv(args.llm_api_key_env)
        if api_key:
            try:
                analysis = _call_openai_compatible(
                    base_url=args.llm_base_url,
                    model=args.llm_model,
                    api_key=api_key,
                    system_prompt="You are an expert RL researcher for safety-critical control.",
                    user_prompt=_render_llm_prompt(summary),
                )
                summary["llm_analysis"] = {
                    "base_url": args.llm_base_url,
                    "model": args.llm_model,
                    "api_key_env": args.llm_api_key_env,
                    "analysis": analysis,
                }
            except (KeyError, IndexError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                summary["llm_analysis"] = {"error": str(e)}
        else:
            summary["llm_analysis"] = {
                "error": f"Missing API key in environment variable: {args.llm_api_key_env}"
            }

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
