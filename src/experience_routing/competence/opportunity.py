"""Opportunity / success detection for competence (PLAN.md Module E, section 9.1).

An *opportunity* for experience ``e`` occurs when a policy reaches a state close
to ``e``'s precondition prototype. The experience is *executed successfully* if,
within horizon ``H_e``, the realized state change matches ``e``'s effect
prototype. This deliberately avoids using raw chunk frequency as competence
(PLAN.md section 9.1).
"""

from __future__ import annotations

import numpy as np

from ..envs.base import EnvSpec
from ..experience.trajectory import Experience, Trajectory


def scan_trajectory(
    traj: Trajectory,
    experiences: dict[int, Experience],
    spec: EnvSpec,
    normalizer,
    eps_pre: float,
    eps_effect: float,
    horizon: int,
) -> dict[int, tuple[int, int]]:
    """Count (opportunities, successes) per experience in a single trajectory.

    Returns ``{experience_id: (n_opportunities, n_successes)}``. Each timestep
    contributes at most one opportunity per experience.
    """
    tf = normalizer.normalize(spec.task_features(traj.states))  # [T+1, task_dim]
    T = len(tf) - 1
    out: dict[int, tuple[int, int]] = {}
    if T <= 0 or not experiences:
        return {eid: (0, 0) for eid in experiences}
    states = tf[:T]  # [T, d] potential opportunity states
    for eid, exp in experiences.items():
        # opportunities: states within eps_pre of the precondition prototype
        d_pre = np.linalg.norm(states - exp.precondition_center, axis=1)  # [T]
        opp_idx = np.where(d_pre <= eps_pre)[0]
        if opp_idx.size == 0:
            out[eid] = (0, 0)
            continue
        successes = 0
        for t in opp_idx:
            hi = min(t + horizon, T)
            realized = tf[t + 1 : hi + 1] - tf[t]
            if len(realized) and np.min(np.linalg.norm(realized - exp.effect_center, axis=1)) <= eps_effect:
                successes += 1
        out[eid] = (int(opp_idx.size), int(successes))
    return out
