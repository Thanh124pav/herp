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

from ..utils.normalization import RunningNormalizer
from .actor import GaussianActor
from .critic import TwinCritic


def resolve_device(device: str) -> torch.device:
    """Pick a torch device, falling back to CPU if CUDA is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


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
    device: str = "cpu"
    obs_norm: bool = False  # running observation normalization (Meta-World stability)


class SACAgent:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.critic = TwinCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.target_entropy = (
            cfg.target_entropy if cfg.target_entropy is not None else -float(cfg.action_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        # running observation normalizer (identity until it has data)
        self.obs_rms = RunningNormalizer(cfg.obs_dim) if cfg.obs_norm else None

    # -- observation normalization ----------------------------------------
    def observe_obs(self, obs) -> None:
        """Update running obs stats from freshly collected observations."""
        if self.obs_rms is not None:
            self.obs_rms.update_batch(np.atleast_2d(np.asarray(obs, dtype=np.float64)))

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs) if self.obs_rms is not None else obs

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.act(self._norm(obs), deterministic=deterministic)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One gradient step from a sampled batch. Returns loss diagnostics."""
        dev = self.device
        obs = torch.as_tensor(self._norm(batch["obs"]), dtype=torch.float32, device=dev)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=dev)
        next_obs = torch.as_tensor(self._norm(batch["next_obs"]), dtype=torch.float32, device=dev)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=dev)

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
        dev = self.device
        o = torch.as_tensor(np.atleast_2d(self._norm(obs)), dtype=torch.float32, device=dev)
        a = torch.as_tensor(np.atleast_2d(action), dtype=torch.float32, device=dev)
        q1, q2 = self.critic(o, a)
        return float(torch.min(q1, q2).squeeze().item())

    @torch.no_grad()
    def td_error(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """|target - Q| per transition (SUPER-style priority, PLAN.md 2.3)."""
        dev = self.device
        obs = torch.as_tensor(self._norm(batch["obs"]), dtype=torch.float32, device=dev)
        act = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)
        rew = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=dev)
        nobs = torch.as_tensor(self._norm(batch["next_obs"]), dtype=torch.float32, device=dev)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32, device=dev)
        na, nlp = self.actor.sample(nobs)
        tq1, tq2 = self.critic_target(nobs, na)
        target = rew + self.cfg.gamma * (1 - done) * (torch.min(tq1, tq2) - self.alpha * nlp)
        q1, q2 = self.critic(obs, act)
        return torch.abs(target - torch.min(q1, q2)).squeeze(-1).cpu().numpy()

    # -- checkpoint / restore (PLAN.md section 16 Milestone 0, Experiment 3) ----
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        if self.obs_rms is not None and sd.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(sd["obs_rms"])

    def snapshot(self) -> dict:
        """A detached deep copy usable to restore this exact policy state."""
        return copy.deepcopy(self.state_dict())
