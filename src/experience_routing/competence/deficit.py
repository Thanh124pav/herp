"""Minimax-inspired receiver deficit / demand (PLAN.md Module F, section 10).

For each valid experience, the frontier is the best competence present in the
population; a policy's deficit is how far it is below that frontier. The minimax
stage answers only "who is missing what" -- it does NOT choose the donor.
"""

from __future__ import annotations

import numpy as np


def frontier(competence: np.ndarray) -> np.ndarray:
    """Best competence per experience: max over policies. Shape [K]."""
    if competence.size == 0:
        return np.zeros(competence.shape[1] if competence.ndim == 2 else 0)
    return competence.max(axis=0)


def deficit_matrix(competence: np.ndarray) -> np.ndarray:
    """deficit[i, e] = max(0, frontier[e] - competence[i, e]). Shape [N, K]."""
    fr = frontier(competence)
    return np.maximum(0.0, fr[None, :] - competence)


def demand_matrix(competence: np.ndarray, demand_power: float = 1.0) -> np.ndarray:
    """OT demand = deficit ** demand_power (PLAN.md section 10)."""
    return deficit_matrix(competence) ** float(demand_power)


def supply_matrix(competence: np.ndarray, n_success_chunks: np.ndarray, max_chunks_per_experience: int) -> np.ndarray:
    """Donor supply (PLAN.md Module G, section 11):

        supply[d, e] = competence[d, e] * min(n_success_chunks[d, e], cap).
    """
    capped = np.minimum(n_success_chunks, max_chunks_per_experience)
    return competence * capped
