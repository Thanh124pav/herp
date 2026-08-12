import numpy as np
import pytest

from experience_routing.envs.base import EnvSpec
from experience_routing.experience.trajectory import Chunk, Trajectory


@pytest.fixture
def spec():
    # 4 task-relevant dims, all of the observation
    return EnvSpec(obs_dim=4, action_dim=2, max_steps=40,
                   task_feature_slices={"all": slice(0, 4)})


def make_phase_trajectory(policy_id=0, episode_id=0, success=True, seed=0):
    """A trajectory with three visibly different state-transition regimes."""
    rng = np.random.default_rng(seed)
    segs = []
    # phase 1: move in +x
    p = np.zeros(4)
    for _ in range(12):
        p = p + np.array([0.3, 0.0, 0.0, 0.0]) + rng.normal(0, 0.01, 4)
        segs.append(p.copy())
    # phase 2: move in +y
    for _ in range(12):
        p = p + np.array([0.0, 0.3, 0.0, 0.0]) + rng.normal(0, 0.01, 4)
        segs.append(p.copy())
    # phase 3: gripper/contact channel rises
    for _ in range(12):
        p = p + np.array([0.0, 0.0, 0.3, 0.3]) + rng.normal(0, 0.01, 4)
        segs.append(p.copy())
    states = np.array([np.zeros(4), *segs])
    T = len(states) - 1
    return Trajectory(
        policy_id=policy_id, episode_id=episode_id,
        states=states, actions=np.zeros((T, 2)),
        rewards=np.ones(T), dones=np.zeros(T), success=success,
    )


def make_chunk(chunk_id, pre, effect, policy_id=0, success=True):
    pre = np.asarray(pre, dtype=float)
    effect = np.asarray(effect, dtype=float)
    post = pre + effect
    states = np.array([pre, post])
    return Chunk(
        chunk_id=chunk_id, policy_id=policy_id, episode_id=0,
        start_t=0, end_t=1, states=states, actions=np.zeros((1, 2)),
        rewards=np.ones(1), pre_state=pre, post_state=post, effect=effect,
        success_episode=success,
    )
