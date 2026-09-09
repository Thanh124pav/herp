"""Meta-World MuJoCo adapter (IMPLEMENTATION.md sections 4 and 36 Phase 3).

Snapshots must cover ``qpos``, ``qvel``, mocap targets, and task-specific state
(random goal, object configuration). The paper-ready gate is the exact one-step
replay test in ``scripts/check_restore.py``; import metaworld only on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base import EnvAdapter


@dataclass
class MetaWorldSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    task_state: dict = field(default_factory=dict)
    elapsed_steps: int = 0


class MetaWorldAdapter(EnvAdapter):
    benchmark = "metaworld"

    def __init__(self, env_id: str, reward_mode: str = "dense", **_):
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.env = None

    def make(self) -> "MetaWorldAdapter":
        try:
            import metaworld  # noqa: F401
        except ImportError as exc:
            raise NotImplementedError(
                "Meta-World adapter is a scaffolded stub; install metaworld and complete "
                "make/save_state/restore_state before running Meta-World experiments (Phase 3)."
            ) from exc
        raise NotImplementedError(
            "Meta-World adapter body not yet implemented; extend before Phase 3 runs."
        )

    def save_state(self):
        raise NotImplementedError

    def restore_state(self, snapshot):
        raise NotImplementedError

    def obs_tensor(self, obs, device="cpu"):
        return torch.as_tensor(obs, device=device).float().flatten()

    def elapsed_steps(self) -> int:
        raise NotImplementedError

    def success_from_info(self, info) -> bool:
        return bool(info.get("success", False))
