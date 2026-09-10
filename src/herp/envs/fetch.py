"""Gymnasium-Robotics Fetch adapter (IMPLEMENTATION.md §4.4).

Serial vectorization over a list of independent Fetch envs. Obs is a dict of
``observation``/``achieved_goal``/``desired_goal``; flatten by concatenation
into a single per-env vector. Snapshots cover qpos/qvel/mocap and the sampled
goal so restore is exact.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base import EnvAdapter
from ._serial import SerialVectorState, close_all, collect_success, serial_reset, serial_step


_OBS_KEYS = ("observation", "achieved_goal", "desired_goal")


@dataclass
class FetchSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    desired_goal: np.ndarray
    elapsed_steps: int = 0
    extra: dict = field(default_factory=dict)


def _flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.asarray(obs[k], dtype=np.float32).flatten() for k in _OBS_KEYS if k in obs]
        return np.concatenate(parts) if parts else np.asarray(obs, dtype=np.float32).flatten()
    return np.asarray(obs, dtype=np.float32).flatten()


def _build_env(env_id: str, seed: int):
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    env = gym.make(env_id).unwrapped
    return env


class FetchAdapter(EnvAdapter):
    benchmark = "fetch"

    def __init__(
        self, env_id: str, reward_mode: str = "dense", device: str | torch.device = "cpu", **_
    ):
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.device = torch.device(device)
        self.env = None
        self._state: SerialVectorState | None = None
        self.max_episode_steps = 50

    # ------------------------------------------------------------------ life-cycle

    def make(self, num_envs: int = 1, seed: int = 0, **_) -> "FetchAdapter":
        if self._state is not None:
            return self
        envs = [_build_env(self.env_id, seed + i) for i in range(num_envs)]
        # Prime obs / spaces
        first_obs = None
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=seed + i)
            if first_obs is None:
                first_obs = obs
        self.num_envs = num_envs
        self._state = SerialVectorState(envs, self.device)
        self.env = envs[0]
        self.obs_dim = int(_flatten_obs(first_obs).size)
        self.action_dim = int(np.prod(envs[0].action_space.shape))
        self.max_episode_steps = int(getattr(envs[0].spec, "max_episode_steps", 50) or 50)
        return self

    def close(self) -> None:
        if self._state is not None:
            close_all(self._state)
            self._state = None
            self.env = None

    # ------------------------------------------------------------------ stepping

    def reset(self, seed: int | None = None):
        obs_np, info = serial_reset(self._state, _flatten_obs)
        return torch.as_tensor(obs_np, device=self.device).float(), info

    def step(self, actions: torch.Tensor):
        obs_np, rew, term, trunc, info = serial_step(
            self._state, actions, _flatten_obs, self.max_episode_steps
        )
        return (
            torch.as_tensor(obs_np, device=self.device).float(),
            torch.as_tensor(rew, device=self.device).float(),
            torch.as_tensor(term, device=self.device).bool(),
            torch.as_tensor(trunc, device=self.device).bool(),
            info,
        )

    # ------------------------------------------------------------------ snapshots

    def _snap_one(self, i: int) -> FetchSnapshot:
        env = self._state.envs[i]
        data = getattr(env, "data", None) or env.sim.data
        goal = np.asarray(getattr(env, "goal", np.zeros(3)), dtype=np.float64).copy()
        return FetchSnapshot(
            qpos=np.array(data.qpos, dtype=np.float64, copy=True),
            qvel=np.array(data.qvel, dtype=np.float64, copy=True),
            mocap_pos=np.array(data.mocap_pos, dtype=np.float64, copy=True),
            mocap_quat=np.array(data.mocap_quat, dtype=np.float64, copy=True),
            desired_goal=goal,
            elapsed_steps=int(self._state.elapsed[i]),
        )

    def save_state(self, env_ids: torch.Tensor) -> list[FetchSnapshot]:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        return [self._snap_one(i) for i in env_ids]

    def _restore_one(self, i: int, snap: FetchSnapshot):
        env = self._state.envs[i]
        data = getattr(env, "data", None) or env.sim.data
        model = getattr(env, "model", None) or env.sim.model
        data.qpos[:] = snap.qpos
        data.qvel[:] = snap.qvel
        if data.mocap_pos.size:
            data.mocap_pos[:] = snap.mocap_pos
        if data.mocap_quat.size:
            data.mocap_quat[:] = snap.mocap_quat
        try:
            env.goal = snap.desired_goal.copy()
        except Exception:
            pass
        try:
            import mujoco

            mujoco.mj_forward(model, data)
        except Exception:
            if hasattr(env, "sim"):
                env.sim.forward()
        obs = env._get_obs() if hasattr(env, "_get_obs") else {}
        self._state.last_obs[i] = obs
        self._state.elapsed[i] = int(snap.elapsed_steps)
        return _flatten_obs(obs)

    def restore_state(
        self, env_ids: torch.Tensor, snapshots: list[FetchSnapshot]
    ) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        obs = [self._restore_one(i, s) for i, s in zip(env_ids, snapshots)]
        return torch.as_tensor(np.stack(obs, axis=0), device=self.device).float()

    # ------------------------------------------------------------------ features

    def obs_tensor(self, raw_obs) -> torch.Tensor:
        if isinstance(raw_obs, dict):
            return torch.as_tensor(_flatten_obs(raw_obs), device=self.device).float()
        if torch.is_tensor(raw_obs):
            return raw_obs.to(self.device).float()
        return torch.as_tensor(np.asarray(raw_obs, dtype=np.float32), device=self.device).float()

    def elapsed_steps(self) -> torch.Tensor:
        return self._state.elapsed.detach().clone()

    def success_from_info(self, info) -> torch.Tensor:
        return torch.as_tensor(collect_success(info, "is_success"), device=self.device).bool()
