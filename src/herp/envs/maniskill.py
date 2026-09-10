"""Batched ManiSkill 3.x adapter (IMPLEMENTATION.md §4.2).

Wraps ``ManiSkillVectorEnv`` — the same wrapper the upstream ``ppo.py`` uses —
and exposes the batched EnvAdapter contract from ``base.py``. Snapshots use
``env.unwrapped.get_state_dict()`` / ``set_state_dict()`` which ManiSkill 3
supports as an atomic per-batch tensor dict; per-slot save/restore is handled
here by slicing / scattering that dict.

The controller state and internal ``_elapsed_steps`` clock must round-trip so
that the §5.2 restore-precision gate (obs L∞ ≤ 1e-7) can pass on state obs.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .base import EnvAdapter


@dataclass
class ManiSkillSnapshot:
    """Opaque per-slot snapshot for ManiSkill.

    ``state_dict`` is a nested dict of CPU tensors of shape ``(1, ...)`` (a
    single-slot slice of the batched ``get_state_dict`` output). Restore
    scatters that back into the corresponding slot.
    """

    state_dict: Any
    elapsed_steps: int
    return_so_far: float = 0.0


def _detach_clone(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: _detach_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_detach_clone(v) for v in x)
    return copy.deepcopy(x)


def _slice_state_dict(sd, i):
    """Extract slot ``i`` from a batched ManiSkill state_dict."""
    if torch.is_tensor(sd):
        return sd[i : i + 1].detach().cpu().clone()
    if isinstance(sd, dict):
        return {k: _slice_state_dict(v, i) for k, v in sd.items()}
    return _detach_clone(sd)


def _scatter_state_dict(dst, src, i, device):
    """Write slot ``i`` of ``dst`` from a single-slot ``src`` in-place."""
    if torch.is_tensor(dst):
        s = src if torch.is_tensor(src) else torch.as_tensor(src)
        dst[i : i + 1] = s.to(device=dst.device, dtype=dst.dtype)
        return dst
    if isinstance(dst, dict):
        for k, v in src.items():
            _scatter_state_dict(dst[k], v, i, device)
        return dst
    return dst


class ManiSkillAdapter(EnvAdapter):
    benchmark = "maniskill"

    def __init__(
        self,
        env_id: str,
        control_mode: str = "pd_joint_delta_pos",
        obs_mode: str = "state",
        reward_mode: str = "normalized_dense",
        sim_backend: str = "physx_cuda",
        render_backend: str = "gpu",
        device: str | torch.device = "cuda",
        reconfiguration_freq: int | None = None,
        ignore_terminations: bool = False,
        **_,
    ):
        self.env_id = env_id
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.sim_backend = sim_backend
        self.render_backend = render_backend
        self.device = torch.device(device)
        self.reconfiguration_freq = reconfiguration_freq
        self.ignore_terminations = ignore_terminations
        self.env = None
        self._counter = None  # per-env step counter (num_envs,)

    def make(self, num_envs: int = 1, seed: int = 0, **_) -> "ManiSkillAdapter":
        if self.env is not None:
            return self
        if self.sim_backend == "physx_cpu":
            # WSL / CPU rendering needs an llvmpipe Vulkan ICD.
            os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
            os.environ.setdefault("MESA_VK_DEVICE_SELECT", "llvmpipe")
            os.environ.setdefault("SAPIEN_DISABLE_RAY_TRACING", "1")
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        # Match upstream ManiSkill ppo.py env construction — the physx_cuda
        # backend appears to behave slightly differently when render_mode
        # is None vs "rgb_array" (renderer init path), so mirror the recipe
        # exactly to avoid divergence from the upstream baseline.
        render_mode = "rgb_array" if self.sim_backend == "physx_cuda" else None
        env = gym.make(
            self.env_id,
            num_envs=num_envs,
            obs_mode=self.obs_mode,
            reward_mode=self.reward_mode,
            control_mode=self.control_mode,
            render_mode=render_mode,
            sim_backend=self.sim_backend,
            render_backend=self.render_backend,
            reconfiguration_freq=self.reconfiguration_freq,
        )
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        self.env = ManiSkillVectorEnv(
            env, num_envs,
            ignore_terminations=self.ignore_terminations,
            record_metrics=True,
        )
        self.num_envs = num_envs
        self.obs_dim = int(np.prod(self.env.single_observation_space.shape))
        self.action_dim = int(np.prod(self.env.single_action_space.shape))
        try:
            from mani_skill.utils import gym_utils
            self.max_episode_steps = int(gym_utils.find_max_episode_steps_value(self.env._env))
        except Exception:
            self.max_episode_steps = 200
        self._counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._seed = seed
        return self

    # ------------------------------------------------------------------ stepping

    def reset(self, seed: int | None = None):
        # When the caller omits ``seed``, use the env's own advancing RNG (no
        # fixed seed) rather than replaying ``self._seed`` — the old behaviour
        # made every eval reset land on the SAME task setup, so success stayed
        # 0 in eval even when training was solving the task.
        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)
        self._counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return obs.to(self.device), info

    def step(self, actions: torch.Tensor):
        # ManiSkillVectorEnv on ``physx_cuda`` expects GPU tensors. The old
        # code inferred device from ``env.action_space.low`` which is a numpy
        # array (device="cpu"), silently moving every action to the CPU on
        # every step — that broke value-function convergence during long runs.
        obs, reward, terminated, truncated, info = self.env.step(actions.to(self.device))
        self._counter = self._counter + 1
        # reset the counter for slots that just ended an episode
        done_mask = torch.as_tensor(terminated, device=self.device) | torch.as_tensor(
            truncated, device=self.device
        )
        if done_mask.any():
            self._counter = torch.where(done_mask, torch.zeros_like(self._counter), self._counter)
        return (
            obs.to(self.device).float(),
            torch.as_tensor(reward, device=self.device).float(),
            torch.as_tensor(terminated, device=self.device).bool(),
            torch.as_tensor(truncated, device=self.device).bool(),
            info,
        )

    # ------------------------------------------------------------------ snapshots

    def save_state(self, env_ids: torch.Tensor) -> list:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        base = self.env.unwrapped
        # ManiSkill 3: get_state_dict() returns a batched nested dict for all slots.
        sd = base.get_state_dict()
        elapsed = self._counter.detach().cpu().tolist()
        out = []
        for idx in env_ids.tolist():
            out.append(
                ManiSkillSnapshot(
                    state_dict=_slice_state_dict(sd, idx),
                    elapsed_steps=int(elapsed[idx]),
                )
            )
        return out

    def restore_state(self, env_ids: torch.Tensor, snapshots: list) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        base = self.env.unwrapped
        # Round-trip through a full-batch state dict so ManiSkill's
        # set_state_dict scatters everything atomically to the sim.
        full = base.get_state_dict()
        for i, snap in zip(env_ids.tolist(), snapshots):
            _scatter_state_dict(full, snap.state_dict, i, self.device)
        base.set_state_dict(full)
        # Recompute observations from restored state.
        obs = base.get_obs()
        # Restore elapsed clocks (best-effort — some ManiSkill wrappers own this).
        for i, snap in zip(env_ids.tolist(), snapshots):
            self._counter[i] = int(snap.elapsed_steps)
            if hasattr(base, "_elapsed_steps"):
                try:
                    base._elapsed_steps[i] = int(snap.elapsed_steps)
                except Exception:
                    pass
        obs_t = obs.to(self.device).float() if torch.is_tensor(obs) else torch.as_tensor(obs, device=self.device).float()
        return obs_t[env_ids.to(obs_t.device)]

    # ------------------------------------------------------------------ features

    def obs_tensor(self, raw_obs) -> torch.Tensor:
        if isinstance(raw_obs, dict):
            parts = [self.obs_tensor(v) for v in raw_obs.values()]
            return torch.cat(parts, dim=-1)
        return torch.as_tensor(raw_obs, device=self.device).float()

    def region_features(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def elapsed_steps(self) -> torch.Tensor:
        return self._counter.detach().clone()

    def success_from_info(self, info) -> torch.Tensor:
        s = info.get("success") if isinstance(info, dict) else None
        if s is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.as_tensor(s, device=self.device).bool().view(-1)
