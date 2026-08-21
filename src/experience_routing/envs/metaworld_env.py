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


def _resolve_env_class(task_name: str):
    """Look up a goal-observable env class across Farama Meta-World versions.

    Accepts ``reach``, ``reach-v2`` or ``reach-v3`` and prefers the newest
    ``-v3`` suite present in the installed package (falls back to ``-v2``).
    Returns ``(env_cls, resolved_key)``.
    """
    import metaworld  # type: ignore

    base = task_name
    for suffix in ("-v3", "-v2"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    for ver, attr in (("v3", "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE"),
                      ("v2", "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE")):
        envs = getattr(metaworld, attr, None)
        if envs is None:
            continue
        key = f"{base}-{ver}-goal-observable"
        if key in envs:
            return envs[key], key
    raise KeyError(
        f"task '{task_name}' not found in installed Meta-World "
        f"(tried {base}-v3/-v2-goal-observable)."
    )


class MetaWorldEnv(BaseEnv):
    def __init__(self, task_name: str = "reach-v3", seed: int = 0, horizon: int | None = None):
        try:
            import metaworld  # noqa: F401
            import gymnasium as gym  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Meta-World support requires optional deps. Install with:\n"
                "  pip install metaworld mujoco gymnasium\n"
                "(PLAN.md section 2.5)."
            ) from exc

        env_cls, self._key = _resolve_env_class(task_name)
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False  # re-randomize goal each reset (perturbation)
        reset_out = self._env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        env_max = int(getattr(self._env, "max_path_length", 500))
        # optional shorter horizon: more episodes per budget for a flow smoke
        self._horizon = min(int(horizon), env_max) if horizon else env_max
        self._t = 0
        self.spec = _metaworld_spec(np.asarray(obs).shape[0],
                                    self._env.action_space.shape[0], self._horizon)

    def reset(self, seed: int | None = None) -> np.ndarray:
        reset_out = self._env.reset(seed=seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self._t = 0
        return np.asarray(obs, dtype=np.float64)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        self._t += 1
        info = dict(info)
        info["success"] = bool(float(info.get("success", 0.0)) > 0.5)
        # terminate the episode on success or at the (possibly shortened) horizon
        terminated = bool(terminated) or info["success"]
        truncated = bool(truncated) or self._t >= self._horizon
        return np.asarray(obs, dtype=np.float64), float(reward), terminated, bool(truncated), info


def make_metaworld_factory(task_name: str = "reach-v3", horizon: int | None = None):
    """Return an ``env_factory(policy_id, n_policies)`` for the population.

    Each policy gets a different seed (learning-history diversity); Meta-World's
    own goal randomization plays the role of the mild environment perturbation.
    ``horizon`` optionally shortens episodes (more episodes per interaction
    budget) -- handy for a fast flow smoke.
    """
    def factory(policy_id: int, n_policies: int) -> BaseEnv:
        return MetaWorldEnv(task_name=task_name, seed=1000 + policy_id, horizon=horizon)

    return factory
