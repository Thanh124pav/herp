"""Deficit unit tests (PLAN.md section 22).

frontier = max competence; best policy deficit = 0; weaker policy deficit > 0.
"""

import numpy as np

from experience_routing.competence.deficit import deficit_matrix, frontier, supply_matrix


def test_frontier_and_deficit():
    # 3 policies, 2 experiences
    C = np.array([
        [0.9, 0.2],
        [0.4, 0.8],
        [0.1, 0.3],
    ])
    fr = frontier(C)
    assert np.allclose(fr, [0.9, 0.8])
    D = deficit_matrix(C)
    # best policy on each experience has zero deficit
    assert D[0, 0] == 0.0  # policy 0 is best on e0
    assert D[1, 1] == 0.0  # policy 1 is best on e1
    # weaker policies have positive deficit
    assert D[2, 0] > 0 and D[1, 0] > 0
    assert np.isclose(D[2, 0], 0.9 - 0.1)


def test_supply_scales_with_competence_and_chunks():
    C = np.array([[0.8, 0.1]])
    n_success = np.array([[10, 100]])
    S = supply_matrix(C, n_success, max_chunks_per_experience=20)
    assert np.isclose(S[0, 0], 0.8 * 10)
    assert np.isclose(S[0, 1], 0.1 * 20)  # capped at 20
