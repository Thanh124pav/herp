"""Competence unit test (PLAN.md section 22).

policy A succeeds 8/10 opportunities, policy B succeeds 2/10 -> C[A,e] > C[B,e].
"""

from experience_routing.competence.tracker import CompetenceConfig, CompetenceTracker


def _tracker():
    # spec/normalizer unused when we inject counts directly
    return CompetenceTracker(num_policies=2, spec=None, normalizer=None,
                             config=CompetenceConfig(beta_alpha=1.0, beta_beta=1.0))


def test_more_successful_policy_has_higher_competence():
    tr = _tracker()
    tr._opportunities[(0, 5)] = 10
    tr._successes[(0, 5)] = 8
    tr._opportunities[(1, 5)] = 10
    tr._successes[(1, 5)] = 2
    ca = tr.competence(0, 5)
    cb = tr.competence(1, 5)
    assert ca > cb
    # Beta(1,1) smoothing: (8+1)/(10+2)=0.75, (2+1)/12=0.25
    assert abs(ca - 0.75) < 1e-9
    assert abs(cb - 0.25) < 1e-9


def test_few_opportunities_flagged_uncertain():
    tr = CompetenceTracker(num_policies=2, spec=None, normalizer=None,
                           config=CompetenceConfig(min_opportunities=5))
    tr._opportunities[(0, 3)] = 2  # below min_opportunities -> uncertain
    tr._successes[(0, 3)] = 1
    tr._opportunities[(0, 4)] = 10  # plenty of evidence -> certain
    tr._successes[(0, 4)] = 6
    assert tr.is_uncertain(0, 3)
    assert not tr.is_uncertain(0, 4)
