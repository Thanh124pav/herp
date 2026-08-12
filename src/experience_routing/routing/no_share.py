"""B1 -- Independent population: no routing at all (PLAN.md section 17)."""

from __future__ import annotations

from ..experience.trajectory import Route
from .base import Router, RoutingContext


class NoShareRouter(Router):
    name = "no_share"

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        return []
