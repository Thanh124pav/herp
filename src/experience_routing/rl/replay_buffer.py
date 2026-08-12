"""Replay buffers (PLAN.md sections 4, 14).

Each policy owns a ``local_buffer`` (its own interactions) and a
``routed_buffer`` (chunks routed in from donors). Training mixes them at a
controlled ``route_batch_fraction`` (PLAN.md section 14).
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions, sampled uniformly."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_many(self, obs, actions, rewards, next_obs, dones) -> None:
        for o, a, r, no, d in zip(obs, actions, rewards, next_obs, dones):
            self.add(o, a, r, no, d)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> dict[str, np.ndarray] | None:
        if self.size == 0:
            return None
        idx = self._rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


def mixed_batch(
    local: ReplayBuffer,
    routed: ReplayBuffer,
    batch_size: int,
    route_fraction: float,
) -> dict[str, np.ndarray] | None:
    """Sample a training batch mixing local and routed data (PLAN.md section 14).

    Falls back to whichever buffer has data if the other is empty.
    """
    if len(local) == 0 and len(routed) == 0:
        return None
    n_route = int(round(batch_size * route_fraction)) if len(routed) > 0 else 0
    n_local = batch_size - n_route
    parts = []
    if n_local > 0 and len(local) > 0:
        parts.append(local.sample(n_local))
    if n_route > 0 and len(routed) > 0:
        parts.append(routed.sample(n_route))
    if not parts:  # e.g. only routed data exists but n_route rounded to 0
        return (routed if len(routed) > 0 else local).sample(batch_size)
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
