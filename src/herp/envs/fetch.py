"""Gymnasium Robotics Fetch adapter (IMPLEMENTATION.md sections 5 and 36 Phase 4).

Fetch observations are dicts; flatten as [observation, achieved_goal, desired_goal].
Snapshot must include goal state so restore is exact; verify with
``scripts/check_restore.py --benchmark fetch --env-id FetchPush-v4``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base import EnvAdapter


@dataclass
class FetchSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    desired_goal: np.ndarray
    elapsed_steps: int = 0
    extra: dict = field(default_factory=dict)


class FetchAdapter(EnvAdapter):
    benchmark = "fetch"

    def __init__(self, env_id: str, reward_mode: str = "dense", **_):
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.env = None

    def make(self) -> "FetchAdapter":
        try:
            import gymnasium_robotics  # noqa: F401
        except ImportError as exc:
            raise NotImplementedError(
                "Fetch adapter is a scaffolded stub; install gymnasium_robotics and complete "
                "make/save_state/restore_state before running Fetch experiments (Phase 4)."
            ) from exc
        raise NotImplementedError(
            "Fetch adapter body not yet implemented; extend before Phase 4 runs."
        )

    def save_state(self):
        raise NotImplementedError

    def restore_state(self, snapshot):
        raise NotImplementedError

    def obs_tensor(self, obs, device="cpu"):
        if isinstance(obs, dict):
            parts = [obs.get(k) for k in ("observation", "achieved_goal", "desired_goal") if k in obs]
            return torch.cat([torch.as_tensor(p, device=device).float().flatten() for p in parts])
        return torch.as_tensor(obs, device=device).float().flatten()

    def elapsed_steps(self) -> int:
        raise NotImplementedError

    def success_from_info(self, info) -> bool:
        return bool(info.get("is_success", False))
