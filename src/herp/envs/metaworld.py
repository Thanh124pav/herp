"""Meta-World V3 adapter (IMPLEMENTATION.md §4.3).

Serial vectorization: ``num_envs`` is implemented as a python list of
independent MuJoCo envs stepped one after another. Snapshots cover
``qpos``/``qvel``, mocap targets, task state, and an elapsed-step clock (V3
does not expose one in a stable API, so we maintain it here). Restore uses
``mj_forward`` so the returned observation is exact.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base import EnvAdapter
from ._serial import SerialVectorState, close_all, collect_success, serial_reset, serial_step


@dataclass
class MetaWorldSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    task_state: dict = field(default_factory=dict)
    elapsed_steps: int = 0
    env_state: Any = None  # optional get_env_state() blob for envs that support it
    prev_obs: Any = None
    last_stable_obs: Any = None


def _flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([_flatten_obs(v) for v in obs.values()])
    return np.asarray(obs, dtype=np.float32).flatten()


def _build_env(env_id: str, seed: int):
    """Create a single Meta-World V3 env for ``env_id``, seeded and task-set."""
    try:
        import metaworld
    except ImportError as exc:  # pragma: no cover - env layer error
        raise RuntimeError("metaworld is required") from exc

    # Prefer V3 MT1 API; fall back to V2 gymnasium make.
    if hasattr(metaworld, "MT1"):
        try:
            mt1 = metaworld.MT1(env_id, seed=seed)
            env = mt1.train_classes[env_id]()
            env.set_task(mt1.train_tasks[seed % len(mt1.train_tasks)])
            return env
        except Exception:
            pass
    import gymnasium as gym
    return gym.make(env_id).unwrapped


class MetaWorldAdapter(EnvAdapter):
    benchmark = "metaworld"

    def __init__(
        self, env_id: str, reward_mode: str = "dense", device: str | torch.device = "cpu", **_
    ):
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.device = torch.device(device)
        self.env = None
        self._state: SerialVectorState | None = None
        self.max_episode_steps = 500

    # ------------------------------------------------------------------ life-cycle

    def make(self, num_envs: int = 1, seed: int = 0, **_) -> "MetaWorldAdapter":
        if self._state is not None:
            return self
        envs = [_build_env(self.env_id, seed + i) for i in range(num_envs)]
        # Warm each env so obs / action space are populated.
        for env in envs:
            env.reset()
        first = envs[0]
        self.num_envs = num_envs
        self._state = SerialVectorState(envs, self.device)
        self.env = first  # spaces come from here (they match across seeds)
        self.obs_dim = int(np.prod(first.observation_space.shape))
        self.action_dim = int(np.prod(first.action_space.shape))
        try:
            self.max_episode_steps = int(getattr(first, "max_path_length", 500))
        except Exception:
            self.max_episode_steps = 500
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

    def _get_env(self, i: int):
        return self._state.envs[i]

    def _snap_one(self, i: int) -> MetaWorldSnapshot:
        env = self._get_env(i)
        data = getattr(env, "data", None)
        if data is None:
            # Very old gyms — fall back to sim
            data = env.sim.data
        try:
            env_state = env.get_env_state() if hasattr(env, "get_env_state") else None
        except Exception:
            env_state = None
        task_state = {}
        for key in ("_target_pos", "_random_reset_space"):
            if hasattr(env, key):
                try:
                    task_state[key] = copy.deepcopy(getattr(env, key))
                except Exception:
                    pass
        # Meta-World's ``_get_obs`` returns [curr_obs, prev_obs, goal] and then
        # advances ``_prev_obs = curr_obs``. To reproduce the last observation
        # bit-exact on restore we need the pre-step ``_prev_obs``, which the env
        # has already overwritten — but that value survives inside the last obs
        # itself as bytes 18..36 (see sawyer_xyz_env.py). Recover it from there.
        last_obs = self._state.last_obs[i]
        prev = None
        if isinstance(last_obs, np.ndarray) and last_obs.size >= 36:
            prev = last_obs[18:36].copy()
        else:
            prev = getattr(env, "_prev_obs", None)
            if prev is not None:
                prev = np.array(prev, copy=True)
        last_stable = getattr(env, "_last_stable_obs", None)
        task_state["curr_path_length"] = getattr(env, "curr_path_length", None)
        task_state["_target_pos"] = copy.deepcopy(getattr(env, "_target_pos", None))
        return MetaWorldSnapshot(
            qpos=np.array(data.qpos, dtype=np.float64, copy=True),
            qvel=np.array(data.qvel, dtype=np.float64, copy=True),
            mocap_pos=np.array(data.mocap_pos, dtype=np.float64, copy=True),
            mocap_quat=np.array(data.mocap_quat, dtype=np.float64, copy=True),
            task_state=task_state,
            elapsed_steps=int(self._state.elapsed[i]),
            env_state=copy.deepcopy(env_state) if env_state is not None else None,
            prev_obs=prev,
            last_stable_obs=copy.deepcopy(last_stable) if last_stable is not None else None,
        )

    def save_state(self, env_ids: torch.Tensor) -> list[MetaWorldSnapshot]:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        return [self._snap_one(i) for i in env_ids]

    def _restore_one(self, i: int, snap: MetaWorldSnapshot):
        env = self._get_env(i)
        data = getattr(env, "data", None) or env.sim.data
        model = getattr(env, "model", None) or env.sim.model
        # Prefer the env's own set_env_state when the snapshot has one — it
        # touches sim state the ad-hoc qpos/qvel/mocap path would miss. Fall
        # back to the raw path when the env does not expose set_env_state.
        used_env_state = False
        if snap.env_state is not None and hasattr(env, "set_env_state"):
            try:
                env.set_env_state(snap.env_state)
                used_env_state = True
            except Exception:
                used_env_state = False
        if not used_env_state:
            data.qpos[:] = snap.qpos
            data.qvel[:] = snap.qvel
            if data.mocap_pos.size:
                data.mocap_pos[:] = snap.mocap_pos
            if data.mocap_quat.size:
                data.mocap_quat[:] = snap.mocap_quat
        # Restore task state (target pos, path length, ...).
        for key, value in (snap.task_state or {}).items():
            if value is None:
                continue
            try:
                setattr(env, key, copy.deepcopy(value))
            except Exception:
                pass
        try:
            import mujoco

            mujoco.mj_forward(model, data)
        except Exception:
            # Old MuJoCo fallback
            if hasattr(env, "sim"):
                env.sim.forward()
        # Restore obs-history state so _get_obs is a pure function of the sim.
        if snap.prev_obs is not None and hasattr(env, "_prev_obs"):
            try:
                env._prev_obs = np.array(snap.prev_obs, copy=True)
            except Exception:
                pass
        if snap.last_stable_obs is not None and hasattr(env, "_last_stable_obs"):
            try:
                env._last_stable_obs = copy.deepcopy(snap.last_stable_obs)
            except Exception:
                pass
        # Refresh observation
        if hasattr(env, "_get_obs"):
            obs = env._get_obs()
        else:
            # Last-resort: rebuild via a no-op step of the last action zero
            obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._state.last_obs[i] = obs
        self._state.elapsed[i] = int(snap.elapsed_steps)
        return _flatten_obs(obs)

    def restore_state(
        self, env_ids: torch.Tensor, snapshots: list[MetaWorldSnapshot]
    ) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        obs = [self._restore_one(i, s) for i, s in zip(env_ids, snapshots)]
        return torch.as_tensor(np.stack(obs, axis=0), device=self.device).float()

    # ------------------------------------------------------------------ features

    def obs_tensor(self, raw_obs) -> torch.Tensor:
        if isinstance(raw_obs, np.ndarray):
            return torch.as_tensor(raw_obs, device=self.device).float()
        if torch.is_tensor(raw_obs):
            return raw_obs.to(self.device).float()
        return torch.as_tensor(_flatten_obs(raw_obs), device=self.device).float()

    def elapsed_steps(self) -> torch.Tensor:
        return self._state.elapsed.detach().clone()

    def success_from_info(self, info) -> torch.Tensor:
        return torch.as_tensor(collect_success(info, "success"), device=self.device).bool()
