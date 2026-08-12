"""OT router unit test (PLAN.md section 22) -- the hard gate before routing.

Competence: A strong on e1, B weak on e1; B strong on e2, A weak on e2.
Expected transport:  A:e1 -> B:e1  and  B:e2 -> A:e2.
"""

import numpy as np

from experience_routing.routing.greedy import assign_greedy
from experience_routing.routing.uot import UOTConfig, assign_uot, solve_transport


# policies A=0, B=1 ; experiences e1=0, e2=1
COMPETENCE = np.array([
    [0.9, 0.1],   # A: strong e1, weak e2
    [0.1, 0.9],   # B: weak e1, strong e2
])
N_SUCCESS = np.array([
    [20, 20],
    [20, 20],
])
EXP_IDS = [101, 102]


def _dominant_route(routes):
    return {(r.donor_id, r.receiver_id, r.experience_id) for r in routes}


def test_uot_routes_complementary_experiences():
    routes = assign_uot(COMPETENCE, EXP_IDS, N_SUCCESS, budget_chunks=32,
                        cfg=UOTConfig(reg=0.05, reg_m=1.0))
    keys = _dominant_route(routes)
    # A teaches e1 to B, B teaches e2 to A
    assert (0, 1, 101) in keys
    assert (1, 0, 102) in keys
    # no self-transfer and no cross-experience wrong-direction dominant routes
    assert all(r.donor_id != r.receiver_id for r in routes)


def test_transport_mass_concentrates_on_correct_pairs():
    gamma = solve_transport(COMPETENCE, N_SUCCESS, UOTConfig(reg=0.05, reg_m=1.0))
    N, K = COMPETENCE.shape
    # index (policy, experience) -> flat
    a_e1 = 0 * K + 0  # A:e1 source
    b_e1 = 1 * K + 0  # B:e1 target
    b_e2 = 1 * K + 1  # B:e2 source
    a_e2 = 0 * K + 1  # A:e2 target
    # correct transfers carry more mass than the self/blocked directions
    assert gamma[a_e1, b_e1] > gamma[b_e1, a_e1]
    assert gamma[b_e2, a_e2] > gamma[a_e2, b_e2]


def test_greedy_picks_best_donor():
    routes = assign_greedy(COMPETENCE, EXP_IDS, N_SUCCESS, budget_chunks=32)
    keys = _dominant_route(routes)
    assert (0, 1, 101) in keys  # A (best on e1) -> B
    assert (1, 0, 102) in keys  # B (best on e2) -> A
