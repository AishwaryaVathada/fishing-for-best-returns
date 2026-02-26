from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class CPPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 256

    lr: float = 3e-4
    vf_lr: float = 3e-4
    cost_vf_lr: float = 3e-4

    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    gamma: float = 1.0
    gae_lambda: float = 0.95

    # Constraint handling
    cost_limit: float = 0.0  # expected per-episode cost limit (0 means no violations desired)
    lambda_lr: float = 0.05
    lambda_init: float = 1.0

    # Action scaling (continuous effort)
    max_effort: float = 1e6

    # Network
    hidden_sizes: Tuple[int, int] = (256, 256)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(obs)
        mu = self.mu(h)
        log_std = self.log_std.clamp(-5.0, 2.0)
        return mu, log_std

    def dist(self, obs: torch.Tensor) -> Normal:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        return Normal(mu, std)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def _compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns


def train_cppo(
    env_fn: Callable[[], object],
    out_dir: str,
    cfg: Optional[CPPOConfig] = None,
    seed: int = 0,
    device: str = "cpu",
) -> str:
    """Constrained PPO (C-PPO) with a primal-dual Lagrangian update.

    The environment is expected to provide a 'cost' field in info (0/1 by default).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or CPPOConfig()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = env_fn()
    obs, info = env.reset(seed=seed)
    obs_dim = int(np.prod(np.asarray(obs).shape))
    act_dim = int(np.prod(np.asarray(env.action_space.shape)))

    policy = GaussianPolicy(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    v_r = ValueNet(obs_dim, cfg.hidden_sizes).to(device)
    v_c = ValueNet(obs_dim, cfg.hidden_sizes).to(device)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_vr = torch.optim.Adam(v_r.parameters(), lr=cfg.vf_lr)
    opt_vc = torch.optim.Adam(v_c.parameters(), lr=cfg.cost_vf_lr)

    lam = float(cfg.lambda_init)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # Buffers (rollout_steps + 1 values for bootstrap)
    def alloc(T):
        return {
            "obs": np.zeros((T, obs_dim), dtype=np.float32),
            "act": np.zeros((T, act_dim), dtype=np.float32),
            "logp": np.zeros((T,), dtype=np.float32),
            "rew": np.zeros((T,), dtype=np.float32),
            "cost": np.zeros((T,), dtype=np.float32),
            "done": np.zeros((T,), dtype=np.float32),
            "vr": np.zeros((T + 1,), dtype=np.float32),
            "vc": np.zeros((T + 1,), dtype=np.float32),
        }

    buf = alloc(cfg.rollout_steps)

    global_step = 0
    episode = 0
    ep_return = 0.0
    ep_cost = 0.0

    while global_step < cfg.total_steps:
        # Collect rollouts
        for t in range(cfg.rollout_steps):
            o = np.asarray(obs, dtype=np.float32).reshape(-1)
            buf["obs"][t] = o

            with torch.no_grad():
                ot = torch.from_numpy(o).to(device)
                dist = policy.dist(ot)
                a = dist.sample()
                logp = dist.log_prob(a).sum().cpu().item()
                vr = v_r(ot).cpu().item()
                vc = v_c(ot).cpu().item()

            # Scale to non-negative effort: softplus then cap
            a_np = a.cpu().numpy().astype(np.float32)
            effort = np.log1p(np.exp(a_np))  # softplus
            effort = np.clip(effort, 0.0, cfg.max_effort)

            next_obs, rew, terminated, truncated, info = env.step(effort)
            done = bool(terminated) or bool(truncated)
            cost = float(info.get("cost", 0.0))

            buf["act"][t] = a_np
            buf["logp"][t] = float(logp)
            buf["rew"][t] = float(rew)
            buf["cost"][t] = float(cost)
            buf["done"][t] = float(done)
            buf["vr"][t] = float(vr)
            buf["vc"][t] = float(vc)

            ep_return += float(rew)
            ep_cost += float(cost)

            global_step += 1
            obs = next_obs

            if done:
                episode += 1
                writer.add_scalar("episode/return", ep_return, global_step)
                writer.add_scalar("episode/cost", ep_cost, global_step)
                obs, info = env.reset(seed=int(seed + episode))
                ep_return = 0.0
                ep_cost = 0.0

            if global_step >= cfg.total_steps:
                break

        # Bootstrap last value
        o_last = np.asarray(obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            ot = torch.from_numpy(o_last).to(device)
            buf["vr"][-1] = float(v_r(ot).cpu().item())
            buf["vc"][-1] = float(v_c(ot).cpu().item())

        # Compute advantages/returns (reward and cost)
        adv_r, ret_r = _compute_gae(buf["rew"], buf["vr"], buf["done"], cfg.gamma, cfg.gae_lambda)
        adv_c, ret_c = _compute_gae(buf["cost"], buf["vc"], buf["done"], cfg.gamma, cfg.gae_lambda)

        # Normalize advantages
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        # Lagrangian advantage
        adv = adv_r - lam * adv_c

        # Update lambda (dual ascent) using episode-average cost estimate
        # Here we approximate expected cost per rollout (normalized to per-episode scale)
        rollout_cost = float(np.sum(buf["cost"]))
        rollout_cost_rate = rollout_cost / max(1.0, float(cfg.rollout_steps))
        lam = max(0.0, lam + cfg.lambda_lr * (rollout_cost_rate - cfg.cost_limit))
        writer.add_scalar("constraint/lambda", lam, global_step)
        writer.add_scalar("constraint/rollout_cost_rate", rollout_cost_rate, global_step)

        # Flatten rollout data
        obs_t = torch.from_numpy(buf["obs"]).to(device)
        act_t = torch.from_numpy(buf["act"]).to(device)
        logp_old = torch.from_numpy(buf["logp"]).to(device)
        adv_t = torch.from_numpy(adv).to(device)
        ret_r_t = torch.from_numpy(ret_r).to(device)
        ret_c_t = torch.from_numpy(ret_c).to(device)

        n = cfg.rollout_steps
        idx = np.arange(n)

        for epoch in range(cfg.update_epochs):
            rng.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                o_mb = obs_t[mb]
                a_mb = act_t[mb]
                logp_old_mb = logp_old[mb]
                adv_mb = adv_t[mb]
                ret_r_mb = ret_r_t[mb]
                ret_c_mb = ret_c_t[mb]

                dist = policy.dist(o_mb)
                logp = dist.log_prob(a_mb).sum(-1)
                ratio = torch.exp(logp - logp_old_mb)

                # PPO clipped objective
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                pi_loss = -(torch.min(surr1, surr2)).mean()

                ent = dist.entropy().sum(-1).mean()
                pi_loss = pi_loss - cfg.ent_coef * ent

                opt_pi.zero_grad(set_to_none=True)
                pi_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt_pi.step()

                # Value losses
                vr_pred = v_r(o_mb)
                vc_pred = v_c(o_mb)
                vr_loss = ((vr_pred - ret_r_mb) ** 2).mean()
                vc_loss = ((vc_pred - ret_c_mb) ** 2).mean()

                opt_vr.zero_grad(set_to_none=True)
                vr_loss.backward()
                nn.utils.clip_grad_norm_(v_r.parameters(), cfg.max_grad_norm)
                opt_vr.step()

                opt_vc.zero_grad(set_to_none=True)
                vc_loss.backward()
                nn.utils.clip_grad_norm_(v_c.parameters(), cfg.max_grad_norm)
                opt_vc.step()

        writer.add_scalar("loss/pi", float(pi_loss.detach().cpu().item()), global_step)
        writer.add_scalar("loss/vr", float(vr_loss.detach().cpu().item()), global_step)
        writer.add_scalar("loss/vc", float(vc_loss.detach().cpu().item()), global_step)

    # Save artifacts
    ckpt = {
        "policy": policy.state_dict(),
        "v_r": v_r.state_dict(),
        "v_c": v_c.state_dict(),
        "cfg": asdict(cfg),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(ckpt, model_path)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"algo": "cppo", "seed": seed, "device": device, "cfg": asdict(cfg)}, f, indent=2)

    writer.close()
    return model_path


def load_cppo_policy(model_path: str, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = CPPOConfig(**ckpt["cfg"])
    obs_dim = ckpt.get("obs_dim")
    act_dim = ckpt.get("act_dim")
    if obs_dim is None or act_dim is None:
        # Backward compatibility: infer from first layer and log_std.
        obs_dim = int(ckpt["policy"]["mlp.0.weight"].shape[1])
        act_dim = int(ckpt["policy"]["log_std"].shape[0])

    policy = GaussianPolicy(int(obs_dim), int(act_dim), cfg.hidden_sizes).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            ot = torch.from_numpy(o).to(device)
            dist = policy.dist(ot)
            a = dist.mean  # deterministic for evaluation
            a_np = a.cpu().numpy().astype(np.float32)
            effort = np.log1p(np.exp(a_np))
            effort = np.clip(effort, 0.0, cfg.max_effort)
            return effort.reshape(1).astype(np.float32)

    return policy_fn
