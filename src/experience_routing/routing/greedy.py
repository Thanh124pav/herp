"""B6 / Module H -- Greedy best-donor experience routing (PLAN.md section 12).

For each receiver deficit ``(r, e)``, pick the highest-competence donor
``d* = argmax_d competence[d, e]`` and route chunks ``(d*, e) -> (r, e)``,
subject to the routing budget. No optimal transport. This is the essential
baseline: if UOT can't beat greedy, OT may not be necessary (PLAN.md section 12).
"""

from __future__ import annotations

import numpy as np

from ..competence.deficit import deficit_matrix
from ..experience.trajectory import Route
from ._materialize import materialize
from .base import Router, RoutingContext


def assign_greedy(
    competence: np.ndarray,
    experience_ids: list[int],
    n_success_chunks: np.ndarray,
    budget_chunks: int,
    min_deficit: float = 1e-3,
) -> list[Route]:
    """Pure, unit-testable greedy assignment. Returns Routes with integer mass."""
    N, K = competence.shape
    deficit = deficit_matrix(competence)
    # rank (receiver, experience) cells by deficit, high first
    cells = [
        (deficit[r, e], r, e)
        for r in range(N)
        for e in range(K)
        if deficit[r, e] > min_deficit
    ]
    cells.sort(key=lambda x: -x[0])
    total_def = sum(c[0] for c in cells) or 1.0

    routes: list[Route] = []
    for dval, r, e in cells:
        donor = int(np.argmax(competence[:, e]))
        if donor == r:
            continue
        if n_success_chunks[donor, e] <= 0:
            continue
        share = dval / total_def
        n_chunks = int(round(budget_chunks * share))
        n_chunks = min(n_chunks, int(n_success_chunks[donor, e]))
        if n_chunks <= 0:
            continue
        routes.append(
            Route(donor_id=donor, receiver_id=r, experience_id=experience_ids[e],
                  chunk_ids=[], mass=float(n_chunks))
        )
    return routes


class GreedyRouter(Router):
    name = "greedy"

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        if ctx.competence.size == 0:
            return []
        routes = assign_greedy(
            ctx.competence, ctx.experience_ids, ctx.n_success_chunks, ctx.budget_chunks
        )
        return materialize(population, ctx, routes)
