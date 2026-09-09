"""The policy population (PLAN.md Module A, sections 5, 20).

Holds ``N`` independent workers, each with its own seed and mild dynamics
perturbation. Provides matched-budget accounting across the population.
"""

from __future__ import annotations

from collections.abc import Callable

from ..envs.base import BaseEnv
from ..envs.perturbations import PerturbationSpec
from ..envs.synthetic_reacher import SyntheticPickPlace
from .worker import Worker


def default_env_factory(policy_id: int, n_policies: int) -> BaseEnv:
    """Build a synthetic env with a mild per-policy perturbation (PLAN.md 1.1)."""
    return SyntheticPickPlace(
        perturbation=PerturbationSpec.for_policy(policy_id, n_policies)
    )


class Population:
    def __init__(
        self,
        size: int = 4,
        env_factory: Callable[[int, int], BaseEnv] = default_env_factory,
        route_batch_fraction: float = 0.25,
        base_seed: int = 0,
        hidden: int = 128,
        utd: int = 1,
        device: str = "cpu",
        obs_norm: bool = False,
        lr: float = 3e-4,
    ):
        self.size = int(size)
        self.workers: list[Worker] = [
            Worker(
                policy_id=i,
                env=env_factory(i, size),
                route_batch_fraction=route_batch_fraction,
                seed=base_seed + 100 * i,
                hidden=hidden, utd=utd, device=device, obs_norm=obs_norm, lr=lr,
            )
            for i in range(size)
        ]

    def __iter__(self):
        return iter(self.workers)

    def __getitem__(self, i: int) -> Worker:
        return self.workers[i]

    @property
    def total_env_steps(self) -> int:
        """Total environment interactions across the population (PLAN.md section 20)."""
        return sum(w.env_steps for w in self.workers)

    def evaluate_all(self, episodes: int = 20) -> dict[int, dict[str, float]]:
        return {w.policy_id: w.evaluate(episodes=episodes) for w in self.workers}
