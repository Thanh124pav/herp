"""Minimal EnvAdapter interface per IMPLEMENTATION.md section 2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class EnvAdapter(ABC):
    """HERP core talks to environments only through this interface."""

    benchmark: str = ""
    env_id: str = ""
    env: Any = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        if self.env is not None:
            self.env.close()

    def action_low(self):
        return torch.as_tensor(self.env.action_space.low, dtype=torch.float32)

    def action_high(self):
        return torch.as_tensor(self.env.action_space.high, dtype=torch.float32)

    @abstractmethod
    def save_state(self) -> Any:
        """Deep-copy every simulator variable needed for exact restoration."""

    @abstractmethod
    def restore_state(self, snapshot) -> tuple:
        """Return (obs, info) after restoring state from a stored snapshot."""

    @abstractmethod
    def obs_tensor(self, obs, device="cpu") -> torch.Tensor:
        """Flatten one observation into a 1-D policy-ready tensor."""

    def region_features(self, obs) -> torch.Tensor:
        """Default: the same tensor used as policy input (post regionizer normalization)."""
        return self.obs_tensor(obs, device="cpu")

    @abstractmethod
    def elapsed_steps(self) -> int:
        """Episode clock, used to restore truncation timing."""

    @abstractmethod
    def success_from_info(self, info) -> bool:
        """Task success flag from a step's info dict."""
