"""Optional Meta-World adapter (PLAN.md sections 2.5, 3).

A drop-in swap for the synthetic env behind the same ``BaseEnv``/``EnvSpec``
interface, so the whole routing pipeline runs unchanged on a real Meta-World
manipulation task. Heavy deps (mujoco, metaworld, gymnasium) are imported lazily
and are NOT installed by default -- install with ``pip install -e '.[metaworld]'``.

Meta-World observation layout (v2, 39-dim) exposes the task-relevant fields the
segmenter needs: end-effector xyz (0:3), gripper (3:4), first object xyz (4:7),
and the goal (36:39). We expose those as ``task_feature_slices`` including an
object->goal vector proxy via the goal slice.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEnv, EnvSpec


def _metaworld_spec(obs_dim: int, action_dim: int, max_steps: int) -> EnvSpec:
    return EnvSpec(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_steps=max_steps,
        task_feature_slices={
            "ee": slice(0, 3),
            "gripper": slice(3, 4),
            "object": slice(4, 7),
            "goal": slice(36, 39),
        },
    )


class MetaWorldEnv(BaseEnv):
    def __init__(self, task_name: str = "reach-v2", seed: int = 0):
        try:
            import metaworld  # noqa: F401
            import gymnasium as gym  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Meta-World support requires optional deps. Install with:\n"
                "  pip install -e '.[metaworld]'\n"
                "and clone the suite per PLAN.md section 2.5."
            ) from exc
        from metaworld.envs import (  # type: ignore
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS,
        )

        env_cls = ENVS[f"{task_name}-goal-observable"]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        obs, _ = self._env.reset()
        self.spec = _metaworld_spec(obs.shape[0], self._env.action_space.shape[0],
                                    getattr(self._env, "max_path_length", 500))

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float64)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        info = dict(info)
        info["success"] = bool(info.get("success", 0.0) > 0.5)
        return np.asarray(obs, dtype=np.float64), float(reward), bool(terminated), bool(truncated), info


def make_metaworld_factory(task_name: str = "reach-v2"):
    """Return an ``env_factory(policy_id, n_policies)`` for the population.

    Each policy gets a different seed (learning-history diversity); Meta-World's
    own goal randomization plays the role of the mild environment perturbation.
    """
    def factory(policy_id: int, n_policies: int) -> BaseEnv:
        return MetaWorldEnv(task_name=task_name, seed=1000 + policy_id)

    return factory
