"""Rollout coordination for the population (PLAN.md section 15 loop, step 1-2).

Steps every worker once and trains it, returning any finished trajectories so the
online loop can feed them into the experience pipeline.
"""

from __future__ import annotations

from ..experience.trajectory import Trajectory
from .population import Population


class RolloutManager:
    def __init__(self, population: Population, warmup_steps: int = 1000, batch_size: int = 128):
        self.population = population
        self.warmup_steps = int(warmup_steps)
        self.batch_size = int(batch_size)

    def step(self, train: bool = True) -> list[Trajectory]:
        """Advance every worker one env step; train each; collect finished trajs."""
        finished: list[Trajectory] = []
        for worker in self.population:
            warmup = worker.env_steps < self.warmup_steps
            traj = worker.collect_step(warmup=warmup)
            if traj is not None:
                finished.append(traj)
            if train and not warmup:
                worker.train_step(self.batch_size)
        return finished
