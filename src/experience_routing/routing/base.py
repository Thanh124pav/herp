"""Router interface and shared routing context (PLAN.md sections 12-13, 17).

Two families of router share one interface:

  * transition-level baselines (no-share, share-all, random, TD-priority) move
    raw transitions between policy buffers;
  * experience-level routers (greedy, UOT) use the competence matrix + deficit +
    experience bank to route successful exemplar chunks from donors to receivers.

Both implement ``route(population, ctx)`` which mutates receiver routed buffers
and returns the ``Route`` records actually applied (for logging, PLAN.md section
21). Experience routers additionally expose a pure, unit-testable ``assign(...)``
returning ``Route`` objects (PLAN.md section 22 OT test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..experience.bank import ExperienceBank
from ..experience.trajectory import Chunk, Route
from ..experience.vocabulary import ExperienceVocabulary


@dataclass
class RoutingContext:
    """Everything a router needs at a routing round (PLAN.md section 15)."""

    competence: np.ndarray  # [N, K]
    experience_ids: list[int]
    n_success_chunks: np.ndarray  # [N, K]
    bank: ExperienceBank
    vocabulary: ExperienceVocabulary
    budget_chunks: int = 32
    budget_transitions: int = 256  # matched bandwidth for transition-level baselines
    max_chunks_per_experience: int = 200
    demand_power: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def add_chunks_to_receiver(worker, chunks: list[Chunk]) -> int:
    """Push every transition of ``chunks`` into a worker's routed buffer.

    Returns the number of transitions added.
    """
    n = 0
    for c in chunks:
        S, A, R = c.states, c.actions, c.rewards
        for t in range(len(A)):
            done = 1.0 if (t == len(A) - 1 and c.success_episode) else 0.0
            worker.add_routed_transitions([S[t]], [A[t]], [R[t]], [S[t + 1]], [done])
            n += 1
    return n


class Router:
    name = "base"

    def route(self, population, ctx: RoutingContext) -> list[Route]:  # pragma: no cover
        raise NotImplementedError
