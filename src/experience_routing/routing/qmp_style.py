"""B5 -- QMP-style receiver-Q behavior sharing (PLAN.md sections 2.1, 17).

QMP's core idea, ported to our SAC backbone: at a receiver state, each population
policy proposes an action, the receiver's own critic scores the candidates, and
the highest-Q candidate is selected. This is a *behavior-selection* mechanism
(acts at rollout time), not a replay-routing mechanism -- so ``route()`` is a
no-op and the meaningful entry point is ``select_action`` (a hook the rollout
loop can call). Kept as an interface hook for the MVP (PLAN.md deferred items).
"""

from __future__ import annotations

import numpy as np

from ..experience.trajectory import Route
from .base import Router, RoutingContext


def select_action(receiver_agent, candidate_agents, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    """Return the candidate action with the highest receiver-Q (PLAN.md B5)."""
    best_a, best_q = None, -np.inf
    for agent in candidate_agents:
        a = agent.act(obs, deterministic=deterministic)
        q = receiver_agent.q_value(obs, a)
        if q > best_q:
            best_q, best_a = q, a
    if best_a is None:
        best_a = receiver_agent.act(obs, deterministic=deterministic)
    return best_a


def qmp_behavior_selector(worker, population) -> np.ndarray:
    """RolloutManager hook: pick the receiver-Q-best candidate action (B5).

    Every population policy proposes an action at the receiver's current state;
    the receiver's own critic scores them and the best is executed. Exploration
    is preserved by sampling (``deterministic=False``) from each candidate.
    """
    candidates = [w.agent for w in population]
    return select_action(
        worker.agent, candidates, worker.current_obs, deterministic=False
    )


class QMPStyleRouter(Router):
    """Registry placeholder: QMP-style sharing routes no replay data."""

    name = "qmp_style"

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        return []
