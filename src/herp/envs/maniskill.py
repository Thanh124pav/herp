"""ManiSkill 3.x adapter with controller + clock restoration.

Wraps the previously verified ManiSkill logic (near-zero restore error on
PickCube-v1) so HERP core no longer imports ``mani_skill`` directly.
"""
from __future__ import annotations

import copy
import os

import gymnasium as gym
import torch

from .base import EnvAdapter


def _flatten_obs(obs, device="cpu"):
    if isinstance(obs, dict):
        return torch.cat([_flatten_obs(v, device) for v in obs.values()])
    return torch.as_tensor(obs, device=device).float().flatten()


class ManiSkillAdapter(EnvAdapter):
    benchmark = "maniskill"

    def __init__(
        self,
        env_id: str,
        control_mode: str = "pd_ee_delta_pose",
        obs_mode: str = "state",
        reward_mode: str = "dense",
        sim_backend: str = "physx_cpu",
        render_backend: str = "cpu",
    ):
        self.env_id = env_id
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.sim_backend = sim_backend
        self.render_backend = render_backend
        self.env = None

    def make(self) -> "ManiSkillAdapter":
        # WSL/CPU rendering needs an llvmpipe Vulkan ICD; do this BEFORE sapien import.
        if self.render_backend == "cpu":
            os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
            os.environ.setdefault("MESA_VK_DEVICE_SELECT", "llvmpipe")
            os.environ.setdefault("SAPIEN_DISABLE_RAY_TRACING", "1")
        import mani_skill.envs  # noqa: F401
        self.env = gym.make(
            self.env_id,
            obs_mode=self.obs_mode,
            reward_mode=self.reward_mode,
            control_mode=self.control_mode,
            render_mode=None,
            sim_backend=self.sim_backend,
            render_backend=self.render_backend,
        )
        return self

    def save_state(self):
        return copy.deepcopy(self.env.unwrapped.get_state_dict())

    def restore_state(self, snapshot):
        obs, info = self.env.reset(
            options={"reset_to_env_states": {"env_states": snapshot.env_state}}
        )
        target = getattr(self.env, "unwrapped", self.env)
        controller = snapshot.env_state.get("controller") if isinstance(snapshot.env_state, dict) else None
        if controller and hasattr(target, "agent"):
            target.agent.set_controller_state(controller)
        if hasattr(target, "_elapsed_steps"):
            target._elapsed_steps[:] = snapshot.elapsed_steps
        wrapper = self.env
        while wrapper is not target:
            if "_elapsed_steps" in vars(wrapper):
                wrapper._elapsed_steps = snapshot.elapsed_steps
            wrapper = wrapper.env
        if hasattr(target, "get_obs"):
            info = target.get_info()
            obs = target.get_obs(info)
        return obs, info

    def obs_tensor(self, obs, device="cpu"):
        return _flatten_obs(obs, device)

    def region_features(self, obs):
        return self.obs_tensor(obs, "cpu")

    def elapsed_steps(self) -> int:
        target = getattr(self.env, "unwrapped", self.env)
        return int(torch.as_tensor(getattr(target, "elapsed_steps", 0)).flatten()[0])

    def success_from_info(self, info) -> bool:
        value = info.get("success", False)
        return bool(torch.as_tensor(value).any())
