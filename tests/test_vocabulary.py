"""Vocabulary merge + validity unit tests (PLAN.md section 22)."""

from experience_routing.experience.validity import ValidityConfig, ValidityFilter
from experience_routing.experience.vocabulary import ExperienceVocabulary
from tests.conftest import make_chunk


def test_merge_reduces_fragmentation():
    # Two prototypes closer than eps_merge should collapse into one.
    vocab = ExperienceVocabulary(eps_experience=0.2, eps_merge=0.5)
    a = make_chunk(0, pre=[0, 0, 0, 0], effect=[1, 0, 0, 0])
    # pre distance 0.5 -> weighted 0.25 > eps_experience (0.2): separate prototypes,
    # but <= eps_merge (0.5): mergeable.
    b = make_chunk(1, pre=[0.5, 0, 0, 0], effect=[1, 0, 0, 0])
    vocab.assign(a)
    vocab.assign(b)
    assert len(vocab) == 2
    merges = vocab.merge_close()
    assert merges == 1
    assert len(vocab) == 1


def test_max_experiences_caps_vocabulary():
    # With a hard cap of 2, a third distinct chunk is absorbed into its nearest
    # prototype instead of spawning a new experience (bounds K).
    vocab = ExperienceVocabulary(eps_experience=0.01, eps_merge=0.005, max_experiences=2)
    a = vocab.assign(make_chunk(0, pre=[0, 0, 0, 0], effect=[0, 0, 0, 0]))
    b = vocab.assign(make_chunk(1, pre=[5, 0, 0, 0], effect=[0, 0, 0, 0]))
    assert len(vocab) == 2 and a != b
    # a far-away third chunk would normally create a 3rd cluster, but the cap
    # forces absorption into the nearest existing prototype.
    c = vocab.assign(make_chunk(2, pre=[10, 0, 0, 0], effect=[0, 0, 0, 0]))
    assert len(vocab) == 2
    assert c in (a, b)


def test_repeated_success_becomes_valid():
    vf = ValidityFilter(ValidityConfig(min_success_support=3, success_support_threshold=0.05))
    # experience 7 appears in 4 distinct successful episodes
    for ep in range(4):
        vf.observe_chunk(experience_id=7, policy_id=0, episode_id=ep, success=True)
    # a rare experience appears in only 1
    vf.observe_chunk(experience_id=9, policy_id=0, episode_id=0, success=True)
    assert vf.is_valid(7)
    assert not vf.is_valid(9)  # below min_success_support


def test_failure_episodes_do_not_count():
    vf = ValidityFilter(ValidityConfig(min_success_support=1, success_support_threshold=0.01))
    vf.observe_chunk(experience_id=1, policy_id=0, episode_id=0, success=False)
    assert vf.n_success_support(1) == 0
    assert not vf.is_valid(1)
