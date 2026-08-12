"""B2 -- Share-All / shared replay baseline (PLAN.md section 17).

Every policy can learn from every other policy's experience. We copy a matched
number of random transitions from each donor's local buffer into each receiver's
routed buffer (bounded by ``budget_transitions`` per receiver for fair bandwidth).
"""

from __future__ import annotations

from ..experience.trajectory import Route
from .base import Router, RoutingContext


class ShareAllRouter(Router):
    name = "share_all"

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        workers = list(population)
        n = len(workers)
        if n <= 1:
            return []
        routes: list[Route] = []
        per_donor = max(1, ctx.budget_transitions // (n - 1))
        for receiver in workers:
            for donor in workers:
                if donor.policy_id == receiver.policy_id:
                    continue
                batch = donor.local_buffer.sample(min(per_donor, len(donor.local_buffer)))
                if batch is None:
                    continue
                receiver.add_routed_transitions(
                    batch["obs"], batch["actions"], batch["rewards"],
                    batch["next_obs"], batch["dones"],
                )
                routes.append(
                    Route(donor.policy_id, receiver.policy_id, -1, [], float(len(batch["obs"])))
                )
        return routes
