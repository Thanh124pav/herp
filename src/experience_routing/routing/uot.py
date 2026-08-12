"""B7 / Module I -- Unbalanced Optimal Transport experience router (PLAN.md section 13).

Source nodes are ``(donor d, experience e)``, target nodes ``(receiver r,
experience e)``. For the MVP we only route the *same* discovered experience
(``e_source == e_target``); cross-experience and self-transfer pairs get a very
large cost. Unbalanced OT lets the router ignore low-quality donor supply and
leave small deficits unfilled -- digital experience is not conserved
(PLAN.md section 13.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot

from ..competence.deficit import deficit_matrix, supply_matrix
from ..experience.trajectory import Route
from ._materialize import materialize
from .base import Router, RoutingContext


@dataclass
class UOTConfig:
    reg: float = 0.05  # entropic regularization
    reg_m: float = 1.0  # marginal relaxation (unbalancedness)
    big_cost: float = 1000.0
    lambda_quality: float = 1.0
    lambda_redundancy: float = 1.0
    max_chunks_per_experience: int = 200
    demand_power: float = 1.0


def build_cost(competence: np.ndarray, cfg: UOTConfig) -> np.ndarray:
    """Cost over flattened (policy, experience) source/target nodes (section 13.4)."""
    N, K = competence.shape
    M = N * K
    cost = np.full((M, M), cfg.big_cost, dtype=np.float64)
    for d in range(N):
        for e in range(K):
            s = d * K + e
            for r in range(N):
                t = r * K + e  # same experience only
                if d == r:
                    continue  # self-transfer prohibited
                cost[s, t] = (
                    cfg.lambda_quality * (1.0 - competence[d, e])
                    + cfg.lambda_redundancy * competence[r, e]
                )
    return cost


def solve_transport(
    competence: np.ndarray,
    n_success_chunks: np.ndarray,
    cfg: UOTConfig,
) -> np.ndarray:
    """Return the UOT plan gamma of shape [N*K, N*K] (unbalanced Sinkhorn)."""
    N, K = competence.shape
    supply = supply_matrix(competence, n_success_chunks, cfg.max_chunks_per_experience)
    demand = deficit_matrix(competence) ** cfg.demand_power

    a = supply.reshape(-1).astype(np.float64)
    b = demand.reshape(-1).astype(np.float64)
    eps = 1e-6
    a = a + eps
    b = b + eps
    a = a / a.sum()
    b = b / b.sum()

    cost = build_cost(competence, cfg)
    gamma = ot.unbalanced.sinkhorn_unbalanced(a, b, cost, cfg.reg, cfg.reg_m)
    return np.asarray(gamma)


def assign_uot(
    competence: np.ndarray,
    experience_ids: list[int],
    n_success_chunks: np.ndarray,
    budget_chunks: int,
    cfg: UOTConfig | None = None,
    mass_threshold: float = 1e-3,
) -> list[Route]:
    """Pure, unit-testable UOT assignment (PLAN.md section 22 OT test)."""
    cfg = cfg or UOTConfig()
    if competence.size == 0:
        return []
    N, K = competence.shape
    gamma = solve_transport(competence, n_success_chunks, cfg)
    total = gamma.sum()
    if total <= 0:
        return []
    routes: list[Route] = []
    for d in range(N):
        for e in range(K):
            s = d * K + e
            for r in range(N):
                if d == r:
                    continue
                t = r * K + e
                mass = gamma[s, t]
                if mass < mass_threshold:
                    continue
                n_chunks = int(round(budget_chunks * mass / total))
                n_chunks = min(n_chunks, int(n_success_chunks[d, e]))
                if n_chunks <= 0:
                    continue
                routes.append(
                    Route(donor_id=d, receiver_id=r, experience_id=experience_ids[e],
                          chunk_ids=[], mass=float(n_chunks))
                )
    return routes


class UOTRouter(Router):
    name = "uot"

    def __init__(self, config: UOTConfig | None = None):
        self.config = config or UOTConfig()

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        if ctx.competence.size == 0:
            return []
        cfg = self.config
        cfg.max_chunks_per_experience = ctx.max_chunks_per_experience
        cfg.demand_power = ctx.demand_power
        routes = assign_uot(
            ctx.competence, ctx.experience_ids, ctx.n_success_chunks, ctx.budget_chunks, cfg
        )
        return materialize(population, ctx, routes)
