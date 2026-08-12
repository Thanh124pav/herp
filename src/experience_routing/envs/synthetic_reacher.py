"""Lightweight synthetic pick-and-place environment (PLAN.md sections 1, 5).

A fast, dependency-free continuous-control task standing in for a Meta-World
manipulation task so the *full* online loop + routing pipeline runs on CPU in
minutes. It deliberately exposes the task-relevant state fields the segmenter
needs (PLAN.md section 6.1): end-effector position, gripper state, object
position, object->goal vector, and a contact/grasp indicator.

The task has three visibly different functional phases -- reach, grasp,
transport -- so state-transition segmentation (Module B) and experience grouping
(Module C) have real structure to discover. Mild per-policy dynamics
perturbations (``PerturbationSpec``) make different policies good at different
phases, which is exactly the complementary-competence structure Gate C
(PLAN.md section 25) looks for.

Observation layout (obs_dim = 8), all task-relevant:
    [0:2] end-effector position
    [2:3] gripper state in [0, 1]  (1 = closed)
    [3:5] object position
    [5:7] object->goal vector
    [7:8] contact/grasp indicator in {0, 1}

Action layout (action_dim = 3), each in [-1, 1]:
    [0:2] end-effector velocity command
    [2:3] gripper open/close command
"""

from __future__ import annotations

import numpy as np

from .base import BaseEnv, EnvSpec
from .perturbations import PerturbationSpec

_SPEC = EnvSpec(
    obs_dim=8,
    action_dim=3,
    max_steps=60,
    task_feature_slices={
        "ee": slice(0, 2),
        "gripper": slice(2, 3),
        "object": slice(3, 5),
        "object_goal": slice(5, 7),
        "contact": slice(7, 8),
    },
)


class SyntheticPickPlace(BaseEnv):
    """A minimal 2D reach-grasp-transport task."""

    def __init__(
        self,
        perturbation: PerturbationSpec | None = None,
        arena: float = 1.0,
        grasp_radius: float = 0.20,
        success_radius: float = 0.18,
    ):
        self.spec = _SPEC
        self.pert = perturbation or PerturbationSpec()
        self.arena = float(arena)
        self.grasp_radius = float(grasp_radius)
        self.success_radius = float(success_radius)
        self._rng = np.random.default_rng(0)
        self._reset_state()

    # -- internal state ----------------------------------------------------
    def _reset_state(self) -> None:
        self.ee = np.zeros(2)
        self.grip = 0.0
        self.obj = np.zeros(2)
        self.goal = np.zeros(2)
        self.grasped = False
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.ee,
                [self.grip],
                self.obj,
                self.goal - self.obj,
                [1.0 if self.grasped else 0.0],
            ]
        ).astype(np.float64)

    # -- API ---------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        a = self.arena
        self.ee = self._rng.uniform(-a, a, size=2)
        self.obj = self._rng.uniform(-a, a, size=2)
        self.goal = self._rng.uniform(-a, a, size=2)
        # ensure a non-trivial transport distance
        while np.linalg.norm(self.goal - self.obj) < 0.5 * a:
            self.goal = self._rng.uniform(-a, a, size=2)
        self.grip = 0.0
        self.grasped = False
        self.t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        move = action[:2] * self.pert.move_speed * 0.20
        # perturbed control gain adds mild per-policy dynamics differences
        self.ee = np.clip(self.ee + move, -self.arena * 1.5, self.arena * 1.5)
        self.grip = float(np.clip(self.grip + action[2] * 0.5, 0.0, 1.0))

        near_obj = np.linalg.norm(self.ee - self.obj) < self.grasp_radius * self.pert.grasp_ease
        if not self.grasped and near_obj and self.grip > 0.6:
            self.grasped = True
        if self.grasped and self.grip < 0.4:
            self.grasped = False  # dropped
        if self.grasped:
            self.obj = self.ee.copy()

        # reward shaping across the three phases
        d_ee_obj = np.linalg.norm(self.ee - self.obj)
        d_obj_goal = np.linalg.norm(self.obj - self.goal)
        reach_term = -d_ee_obj
        transport_term = -d_obj_goal
        grasp_bonus = 0.5 if self.grasped else 0.0
        reward = 0.5 * reach_term + 1.0 * transport_term + grasp_bonus

        self.t += 1
        success = bool(d_obj_goal < self.success_radius)
        if success:
            reward += 5.0
        terminated = success
        truncated = self.t >= self.spec.max_steps
        info = {"success": success, "grasped": self.grasped}
        return self._obs(), float(reward), terminated, truncated, info
