"""Compact Soft Actor-Critic agent (PLAN.md sections 5, 14, 27).

One ``SACAgent`` == one policy in the population. It owns a separate actor, twin
critic, target critic, and an auto-tuned entropy temperature. The agent is
backend-agnostic to *where* its training batches come from: local buffer only,
or a mix of local + routed data (PLAN.md section 14) -- routing never touches
this class.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .actor import GaussianActor
from .critic import TwinCritic


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_entropy: float | None = None  # default: -action_dim
    seed: int = 0


class SACAgent:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, cfg.hidden)
        self.critic = TwinCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.target_entropy = (
            cfg.target_entropy if cfg.target_entropy is not None else -float(cfg.action_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.act(obs, deterministic=deterministic)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One gradient step from a sampled batch. Returns loss diagnostics."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        # --- critic update ---
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_a)
            target_q = torch.min(tq1, tq2) - self.alpha * next_logp
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor + temperature update ---
        new_a, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- target soft update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
        }

    @torch.no_grad()
    def q_value(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Min-of-twin Q used by the QMP-style receiver-Q baseline (B5)."""
        o = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        a = torch.as_tensor(np.atleast_2d(action), dtype=torch.float32)
        q1, q2 = self.critic(o, a)
        return float(torch.min(q1, q2).squeeze().item())
