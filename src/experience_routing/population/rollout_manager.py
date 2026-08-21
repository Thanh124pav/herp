"""Rollout coordination for the population (PLAN.md section 15 loop, step 1-2).

Steps every worker once and trains it, returning any finished trajectories so the
online loop can feed them into the experience pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from ..experience.trajectory import Trajectory
from .population import Population


class RolloutManager:
    """Advance the population one env step at a time.

    ``behavior_selector`` optionally overrides how each worker acts during
    (non-warmup) collection. It is called ``selector(worker, population) ->
    action`` and enables QMP-style receiver-Q behavior sharing (PLAN.md B5)
    without touching the SAC backbone.
    """

    def __init__(
        self,
        population: Population,
        warmup_steps: int = 1000,
        batch_size: int = 128,
        behavior_selector: Callable | None = None,
    ):
        self.population = population
        self.warmup_steps = int(warmup_steps)
        self.batch_size = int(batch_size)
        self.behavior_selector = behavior_selector

    def step(self, train: bool = True) -> list[Trajectory]:
        """Advance every worker one env step; train each; collect finished trajs."""
        finished: list[Trajectory] = []
        for worker in self.population:
            warmup = worker.env_steps < self.warmup_steps
            action = None
            if not warmup and self.behavior_selector is not None:
                action = self.behavior_selector(worker, self.population)
            traj = worker.collect_step(warmup=warmup, action=action)
            if traj is not None:
                finished.append(traj)
            if train and not warmup:
                worker.train_step(self.batch_size)
        return finished
