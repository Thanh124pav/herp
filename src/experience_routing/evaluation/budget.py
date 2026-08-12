"""Fair budget accounting (PLAN.md section 20).

Primary population comparisons must match total environment interactions across
methods. This tracker records per-policy and total interaction counts plus routed
bandwidth so runs can be compared on matched budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetAccountant:
    routed_transitions: int = 0
    routing_rounds: int = 0
    per_policy_steps: dict[int, int] = field(default_factory=dict)

    def record_population(self, population) -> None:
        for w in population:
            self.per_policy_steps[w.policy_id] = w.env_steps

    def record_routing(self, routes) -> None:
        self.routing_rounds += 1
        self.routed_transitions += sum(len(r.chunk_ids) for r in routes) if routes else 0

    @property
    def total_env_steps(self) -> int:
        return sum(self.per_policy_steps.values())

    def summary(self) -> dict:
        return {
            "total_env_steps": self.total_env_steps,
            "per_policy_steps": dict(self.per_policy_steps),
            "routing_rounds": self.routing_rounds,
            "routed_chunks_total": self.routed_transitions,
        }
