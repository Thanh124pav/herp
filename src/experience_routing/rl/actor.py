"""Squashed-Gaussian actor for SAC (PLAN.md section 5, ported concept from QMP).

Kept deliberately small (2-layer MLP) so training runs on CPU. Policies keep
their actor/critic objects separate, mirroring QMP's separated-policy structure
(PLAN.md section 2.1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Return ``(action, log_prob)`` with tanh-squashing correction."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, log_std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mu)
        else:
            std = log_std.exp()
            x = torch.distributions.Normal(mu, std).sample()
            action = torch.tanh(x)
        return action.squeeze(0).cpu().numpy()
