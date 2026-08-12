"""Core data structures for the experience-routing pipeline (PLAN.md section 4).

These are plain dataclasses holding NumPy arrays. Deliberately no learned latent
vectors (PLAN.md section 4.3) -- the whole descriptor is state-transition based.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    """A full episode collected by one policy (PLAN.md section 4.1)."""

    policy_id: int
    episode_id: int
    states: np.ndarray  # [T+1, state_dim]
    actions: np.ndarray  # [T, action_dim]
    rewards: np.ndarray  # [T]
    dones: np.ndarray  # [T]
    success: bool
    env_context: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class Chunk:
    """A contiguous sub-trajectory (candidate experience instance), section 4.2.

    ``pre_state``/``post_state``/``effect`` are computed on *normalized
    task-relevant* state dimensions (PLAN.md section 4.2, 7.1). ``effect`` is
    ``post_state - pre_state``.
    """

    chunk_id: int
    policy_id: int
    episode_id: int
    start_t: int
    end_t: int  # exclusive index into states beyond the last action

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    pre_state: np.ndarray
    post_state: np.ndarray
    effect: np.ndarray

    success_episode: bool
    experience_id: int | None = None

    @property
    def length(self) -> int:
        return int(self.end_t - self.start_t)

    def descriptor(self) -> np.ndarray:
        """Concatenated (precondition, effect) descriptor (section 7.1)."""
        return np.concatenate([self.pre_state, self.effect])


@dataclass
class Experience:
    """A discovered functional-experience prototype (PLAN.md section 4.3)."""

    experience_id: int

    precondition_center: np.ndarray
    effect_center: np.ndarray

    n_chunks: int = 0
    n_success_chunks: int = 0
    n_success_episodes: int = 0

    valid: bool = False

    # donor policy_id -> list of contributed chunk_ids
    donor_chunk_ids: dict[int, list[int]] = field(default_factory=dict)

    def descriptor(self) -> np.ndarray:
        return np.concatenate([self.precondition_center, self.effect_center])


@dataclass
class Route:
    """A single donor->receiver routing decision for one experience (section 4.5)."""

    donor_id: int
    receiver_id: int
    experience_id: int
    chunk_ids: list[int]
    mass: float
