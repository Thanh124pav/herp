"""Dynamic policy x experience competence matrix (PLAN.md Module E, section 9).

Tracks opportunities and successes per (policy, experience) and produces a
Beta-smoothed competence estimate plus an evidence count. The experience
vocabulary can grow, so the matrix is built on demand from the current set of
experience ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.base import EnvSpec
from ..experience.trajectory import Experience, Trajectory
from .opportunity import scan_trajectory


@dataclass
class CompetenceConfig:
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    min_opportunities: int = 5
    horizon: int = 8
    eps_pre: float = 1.0
    eps_effect: float = 1.0


class CompetenceTracker:
    def __init__(self, num_policies: int, spec: EnvSpec, normalizer, config: CompetenceConfig | None = None):
        self.num_policies = int(num_policies)
        self.spec = spec
        self.normalizer = normalizer
        self.config = config or CompetenceConfig()
        # (policy_id, experience_id) -> counts
        self._opportunities: dict[tuple[int, int], int] = {}
        self._successes: dict[tuple[int, int], int] = {}

    def observe(self, traj: Trajectory, experiences: dict[int, Experience]) -> None:
        cfg = self.config
        counts = scan_trajectory(
            traj, experiences, self.spec, self.normalizer,
            cfg.eps_pre, cfg.eps_effect, cfg.horizon,
        )
        for eid, (opp, succ) in counts.items():
            self._opportunities[(traj.policy_id, eid)] = (
                self._opportunities.get((traj.policy_id, eid), 0) + opp
            )
            self._successes[(traj.policy_id, eid)] = (
                self._successes.get((traj.policy_id, eid), 0) + succ
            )

    def competence(self, policy_id: int, experience_id: int) -> float:
        cfg = self.config
        o = self._opportunities.get((policy_id, experience_id), 0)
        s = self._successes.get((policy_id, experience_id), 0)
        return (s + cfg.beta_alpha) / (o + cfg.beta_alpha + cfg.beta_beta)

    def is_uncertain(self, policy_id: int, experience_id: int) -> bool:
        """Too few opportunities -> uncertain rather than forced to zero (section 9.1)."""
        return self._opportunities.get((policy_id, experience_id), 0) < self.config.min_opportunities

    def matrix(self, experience_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (competence_mean, trials, successes) of shape [N, K]."""
        N, K = self.num_policies, len(experience_ids)
        mean = np.zeros((N, K))
        trials = np.zeros((N, K), dtype=int)
        succ = np.zeros((N, K), dtype=int)
        for i in range(N):
            for j, eid in enumerate(experience_ids):
                mean[i, j] = self.competence(i, eid)
                trials[i, j] = self._opportunities.get((i, eid), 0)
                succ[i, j] = self._successes.get((i, eid), 0)
        return mean, trials, succ
