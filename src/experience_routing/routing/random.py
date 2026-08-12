"""B3 -- Random routing baseline (PLAN.md section 17).

Uses the exact same routing bandwidth as the proposed method but selects random
donor transitions and random receivers. Tests whether the routing *decision*
matters or any cross-policy data suffices.
"""

from __future__ import annotations

from ..experience.trajectory import Route
from .base import Router, RoutingContext


class RandomRouter(Router):
    name = "random"

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        workers = list(population)
        n = len(workers)
        if n <= 1:
            return []
        routes: list[Route] = []
        remaining = ctx.budget_transitions
        block = max(1, ctx.budget_transitions // (2 * n))
        while remaining > 0:
            d, r = ctx.rng.integers(0, n, size=2)
            if d == r:
                continue
            donor, receiver = workers[int(d)], workers[int(r)]
            take = min(block, remaining, len(donor.local_buffer))
            if take <= 0:
                remaining -= block
                continue
            batch = donor.local_buffer.sample(take)
            receiver.add_routed_transitions(
                batch["obs"], batch["actions"], batch["rewards"],
                batch["next_obs"], batch["dones"],
            )
            routes.append(Route(donor.policy_id, receiver.policy_id, -1, [], float(take)))
            remaining -= take
        return routes
