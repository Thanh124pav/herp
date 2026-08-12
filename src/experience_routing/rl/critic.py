"""Twin Q critic for SAC (PLAN.md section 5).

Two Q-heads (clipped-double-Q) sharing no parameters, matching the standard SAC
recipe. The receiver-side critic is also what the QMP-style baseline (PLAN.md
section B5) uses to score candidate actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden)
        self.q2 = QNetwork(obs_dim, action_dim, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)
