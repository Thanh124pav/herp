"""Environment interface (PLAN.md sections 3, 5).

All environments used by the population expose the same minimal Gym-like API plus
an ``EnvSpec`` describing the task-relevant state layout the segmenter needs
(PLAN.md sections 6.1, 9). The synthetic env and the (optional) Meta-World adapter
both implement this so routing code never depends on which backend is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvSpec:
    """Static description of an environment and its task-relevant state layout.

    ``task_feature_slices`` maps a semantic name to a slice of the observation
    vector. The segmenter/grouper operate only on the concatenation of these
    slices (the "task-relevant" dims of PLAN.md sections 6.1 / 7.1), never the raw
    observation, so the pipeline is robust to padding/irrelevant dims.
    """

    obs_dim: int
    action_dim: int
    max_steps: int
    task_feature_slices: dict[str, slice] = field(default_factory=dict)

    @property
    def task_dim(self) -> int:
        return sum(sl.stop - sl.start for sl in self.task_feature_slices.values())

    def task_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract and concatenate task-relevant dims from an observation.

        Accepts a single obs [obs_dim] or a batch [T, obs_dim].
        """
        obs = np.asarray(obs, dtype=np.float64)
        parts = [obs[..., sl] for sl in self.task_feature_slices.values()]
        if not parts:
            return obs
        return np.concatenate(parts, axis=-1)


class BaseEnv:
    """Minimal environment protocol used across the codebase."""

    spec: EnvSpec

    def reset(self, seed: int | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def step(self, action: np.ndarray):  # pragma: no cover
        """Return ``(obs, reward, terminated, truncated, info)``."""
        raise NotImplementedError
