"""Batched EnvAdapter interface (IMPLEMENTATION.md §4.1).

Every method is batched. Observations, actions, rewards, dones, and rewards
are torch tensors of shape ``(num_envs, ...)`` on ``self.device``. Snapshot
save/restore is per-slot: ``env_ids`` is a LongTensor of slot indices.

Single-env benchmarks (Meta-World, Fetch) implement this contract by holding a
python list of independent MuJoCo envs and looping serially (§4.3, §4.4). The
interface is what matters; the vectorization strategy is per-adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class EnvAdapter(ABC):
    """HERP core talks to environments only through this interface."""

    benchmark: str = ""
    env_id: str = ""
    num_envs: int = 1
    obs_dim: int = 0
    action_dim: int = 0
    max_episode_steps: int = 0
    device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------ life-cycle

    @abstractmethod
    def make(self, num_envs: int = 1, seed: int = 0, **kwargs) -> "EnvAdapter":
        """Build the underlying env(s). Idempotent when already built."""

    def close(self) -> None:  # pragma: no cover - trivial
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self.env = None

    # ------------------------------------------------------------------ stepping

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Batched reset. Returns (obs (num_envs, obs_dim), info)."""

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Batched step.

        Returns
        -------
        obs         : (num_envs, obs_dim) float32
        reward      : (num_envs,)         float32
        terminated  : (num_envs,)         bool
        truncated   : (num_envs,)         bool
        info        : dict of per-env info; scalar keys promoted to tensors
        """

    # ------------------------------------------------------------------ snapshots

    @abstractmethod
    def save_state(self, env_ids: torch.Tensor) -> list:
        """Deep-copy the sim state for the given env slots.

        Returns a python list of length ``len(env_ids)`` containing one opaque
        Snapshot object per requested slot. Snapshot is a per-benchmark
        dataclass; treat it as opaque outside the adapter.
        """

    @abstractmethod
    def restore_state(
        self, env_ids: torch.Tensor, snapshots: list
    ) -> torch.Tensor:
        """Restore ``snapshots[i]`` into slot ``env_ids[i]``.

        Returns the observation of the restored slots, shape
        ``(len(env_ids), obs_dim)``.
        """

    # ------------------------------------------------------------------ features

    def obs_tensor(self, raw_obs) -> torch.Tensor:
        """Convert whatever ``reset``/``step`` returns into a policy-ready tensor.

        Default: pass through ``torch.as_tensor``. Adapters with dict obs must
        override.
        """
        if isinstance(raw_obs, torch.Tensor):
            return raw_obs.float()
        return torch.as_tensor(raw_obs, device=self.device).float()

    def region_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Slice/transform obs → region-feature vector. Default: identity."""
        return obs

    # ------------------------------------------------------------------ clocks

    @abstractmethod
    def elapsed_steps(self) -> torch.Tensor:
        """Per-env elapsed-step counter, shape ``(num_envs,)`` LongTensor."""

    @abstractmethod
    def success_from_info(self, info) -> torch.Tensor:
        """Per-env success flag from a step's info, shape ``(num_envs,)`` bool."""

    # ------------------------------------------------------------------ spaces

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def action_low(self) -> torch.Tensor:
        low = self.env.action_space.low
        return torch.as_tensor(low, dtype=torch.float32, device=self.device)

    def action_high(self) -> torch.Tensor:
        high = self.env.action_space.high
        return torch.as_tensor(high, dtype=torch.float32, device=self.device)
