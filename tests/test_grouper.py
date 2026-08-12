"""Grouper unit tests (PLAN.md section 22)."""

import numpy as np

from experience_routing.experience.grouper import experience_distance
from experience_routing.experience.vocabulary import ExperienceVocabulary
from tests.conftest import make_chunk


def test_identical_descriptor_same_experience():
    vocab = ExperienceVocabulary(eps_experience=0.5)
    c1 = make_chunk(0, pre=[0, 0, 0, 0], effect=[1, 0, 0, 0])
    c2 = make_chunk(1, pre=[0.01, 0, 0, 0], effect=[1.01, 0, 0, 0])
    e1 = vocab.assign(c1)
    e2 = vocab.assign(c2)
    assert e1 == e2
    assert len(vocab) == 1


def test_large_difference_new_experience():
    vocab = ExperienceVocabulary(eps_experience=0.5)
    c1 = make_chunk(0, pre=[0, 0, 0, 0], effect=[1, 0, 0, 0])
    c2 = make_chunk(1, pre=[5, 5, 5, 5], effect=[-1, 0, 0, 0])
    e1 = vocab.assign(c1)
    e2 = vocab.assign(c2)
    assert e1 != e2
    assert len(vocab) == 2


def test_prototype_update_deterministic():
    v1 = ExperienceVocabulary(eps_experience=1.0)
    v2 = ExperienceVocabulary(eps_experience=1.0)
    chunks = [make_chunk(i, pre=[0.1 * i, 0, 0, 0], effect=[1, 0, 0, 0]) for i in range(5)]
    for c in chunks:
        v1.assign(c)
    for c in [make_chunk(i, pre=[0.1 * i, 0, 0, 0], effect=[1, 0, 0, 0]) for i in range(5)]:
        v2.assign(c)
    e1 = v1.experiences[0]
    e2 = v2.experiences[0]
    assert np.allclose(e1.precondition_center, e2.precondition_center)
    assert np.allclose(e1.effect_center, e2.effect_center)


def test_distance_weights():
    d = experience_distance(
        np.zeros(2), np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 2.0]),
        alpha_pre=0.5, alpha_eff=0.5,
    )
    assert np.isclose(d, 0.5 * 1.0 + 0.5 * 2.0)
