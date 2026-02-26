from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ESConfig:
    total_iterations: int = 400
    population: int = 32  # will be doubled via antithetic sampling if even
    sigma: float = 0.05
    lr: float = 0.02
    elite_frac: float = 0.2
    max_effort: float = 1e6
    horizon_override: Optional[int] = None  # for fast debug
    seed: int = 0


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]),
            nn.Tanh(),
            nn.Linear(hidden[1], act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_params(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i : i + n].view_as(p))
        i += n


def _rollout_return(env_fn: Callable[[], object], policy: nn.Module, seed: int, max_effort: float) -> Tuple[float, float]:
    env = env_fn()
    obs, info = env.reset(seed=seed)
    done = False
    G = 0.0
    cost_sum = 0.0
    while not done:
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).view(1, -1)
        with torch.no_grad():
            a = policy(o).view(-1).cpu().numpy().astype(np.float32)
        effort = np.log1p(np.exp(a))  # softplus
        effort = np.clip(effort, 0.0, max_effort)
        obs, r, terminated, truncated, info = env.step(effort)
        done = bool(terminated) or bool(truncated)
        G += float(r)
        cost_sum += float(info.get("cost", 0.0))
    return float(G), float(cost_sum)


def train_es(
    env_fn: Callable[[], object],
    out_dir: str,
    cfg: Optional[ESConfig] = None,
    device: str = "cpu",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or ESConfig()

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Infer dims
    env = env_fn()
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = int(np.prod(np.asarray(obs).shape))
    act_dim = int(np.prod(np.asarray(env.action_space.shape)))

    policy = DeterministicPolicy(obs_dim, act_dim).to(device)
    theta = _flatten_params(policy).to(device)

    # Simple Adam on parameter vector
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    pop = int(cfg.population)
    if pop < 2:
        raise ValueError("population must be >= 2")

    # Antithetic sampling: use half noise vectors and mirror
    half = pop // 2
    pop_effective = half * 2

    for it in range(cfg.total_iterations):
        noises = rng.normal(0.0, 1.0, size=(half, theta.numel())).astype(np.float32)
        noises = torch.from_numpy(noises).to(device)

        returns = []
        costs = []

        for i in range(half):
            for sign_idx, sign in enumerate((+1.0, -1.0)):
                theta_pert = theta + sign * cfg.sigma * noises[i]
                _set_params(policy, theta_pert)
                rollout_seed = int(cfg.seed + it * 1000 + i * 2 + sign_idx)
                G, C = _rollout_return(env_fn, policy, seed=rollout_seed, max_effort=cfg.max_effort)
                returns.append(G)
                costs.append(C)

        returns = np.asarray(returns, dtype=np.float64)
        costs = np.asarray(costs, dtype=np.float64)

        # Rank-based shaping (robust to outliers)
        ranks = returns.argsort().argsort().astype(np.float64)
        shaped = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

        # Gradient estimate
        grad = torch.zeros_like(theta)
        k = 0
        for i in range(half):
            grad += float(shaped[k]) * noises[i]
            k += 1
            grad += float(shaped[k]) * (-noises[i])
            k += 1
        grad /= (pop_effective * cfg.sigma)

        # Adam update
        t = it + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta + cfg.lr * m_hat / (torch.sqrt(v_hat) + eps)

        # Set updated params
        _set_params(policy, theta)

        writer.add_scalar("es/mean_return", float(returns.mean()), it)
        writer.add_scalar("es/std_return", float(returns.std()), it)
        writer.add_scalar("es/mean_cost", float(costs.mean()), it)

    model_path = os.path.join(out_dir, "model.pt")
    torch.save({"theta": theta.detach().cpu(), "cfg": asdict(cfg), "obs_dim": obs_dim, "act_dim": act_dim}, model_path)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"algo": "es", "cfg": asdict(cfg)}, f, indent=2)

    writer.close()
    return model_path


def load_es_policy(model_path: str, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ESConfig(**ckpt["cfg"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    policy = DeterministicPolicy(obs_dim, act_dim).to(device)
    theta = ckpt["theta"].to(device)
    _set_params(policy, theta)
    policy.eval()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).view(1, -1).to(device)
        with torch.no_grad():
            a = policy(o).view(-1).cpu().numpy().astype(np.float32)
        effort = np.log1p(np.exp(a))
        effort = np.clip(effort, 0.0, cfg.max_effort)
        return effort.reshape(1).astype(np.float32)

    return policy_fn
