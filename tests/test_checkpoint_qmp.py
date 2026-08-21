"""Tests for SAC checkpoint/restore and QMP-style receiver-Q selection.

Covers the machinery Experiment 3 (PLAN.md 19.3) and the B5 baseline (17) rely
on: a policy snapshot must round-trip exactly, and QMP must pick the candidate
action the receiver's own critic scores highest.
"""

from __future__ import annotations

import numpy as np

from experience_routing.envs.synthetic_reacher import SyntheticPickPlace
from experience_routing.population.population import Population
from experience_routing.population.worker import Worker
from experience_routing.rl.sac import SACAgent, SACConfig
from experience_routing.routing.qmp_style import qmp_behavior_selector, select_action


def _agent(seed: int) -> SACAgent:
    return SACAgent(SACConfig(obs_dim=6, action_dim=2, seed=seed))


def test_snapshot_restore_roundtrip():
    agent = _agent(0)
    obs = np.zeros(6, dtype=np.float32)
    snap = agent.snapshot()
    before = agent.act(obs, deterministic=True)

    # perturb the policy with several updates on random data
    rng = np.random.default_rng(1)
    for _ in range(20):
        batch = {
            "obs": rng.normal(size=(32, 6)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(32, 2)).astype(np.float32),
            "rewards": rng.normal(size=(32, 1)).astype(np.float32),
            "next_obs": rng.normal(size=(32, 6)).astype(np.float32),
            "dones": np.zeros((32, 1), dtype=np.float32),
        }
        agent.update(batch)
    after = agent.act(obs, deterministic=True)
    assert not np.allclose(before, after)  # training actually moved the policy

    agent.load_state_dict(snap)
    restored = agent.act(obs, deterministic=True)
    assert np.allclose(before, restored, atol=1e-6)  # exact restore


def test_snapshot_is_independent_copy():
    agent = _agent(0)
    snap = agent.snapshot()
    rng = np.random.default_rng(2)
    batch = {
        "obs": rng.normal(size=(16, 6)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(16, 2)).astype(np.float32),
        "rewards": rng.normal(size=(16, 1)).astype(np.float32),
        "next_obs": rng.normal(size=(16, 6)).astype(np.float32),
        "dones": np.zeros((16, 1), dtype=np.float32),
    }
    agent.update(batch)
    # the snapshot must not have been mutated by the post-snapshot update
    agent2 = _agent(0)
    agent2.load_state_dict(snap)
    obs = np.ones(6, dtype=np.float32)
    fresh = _agent(0)
    assert np.allclose(agent2.act(obs, deterministic=True), fresh.act(obs, deterministic=True), atol=1e-6)


def test_qmp_selects_highest_receiver_q():
    obs = np.zeros(6, dtype=np.float32)
    receiver = _agent(0)
    candidates = [_agent(1), _agent(2), _agent(3)]
    a = select_action(receiver, candidates, obs, deterministic=True)
    # the selected action must have receiver-Q >= every candidate's own action Q
    q_sel = receiver.q_value(obs, a)
    for c in candidates:
        q_c = receiver.q_value(obs, c.act(obs, deterministic=True))
        assert q_sel >= q_c - 1e-6


def test_qmp_behavior_selector_runs_over_population():
    pop = Population(size=3)
    w = pop[0]
    action = qmp_behavior_selector(w, pop)
    assert action.shape == (w.env.spec.action_dim,)
    assert np.all(np.isfinite(action))


def test_reset_routed_buffer_isolates():
    env = SyntheticPickPlace()
    w = Worker(policy_id=0, env=env, seed=0)
    w.add_routed_transitions(
        [np.zeros(env.spec.obs_dim)], [np.zeros(env.spec.action_dim)],
        [0.0], [np.zeros(env.spec.obs_dim)], [0.0],
    )
    assert len(w.routed_buffer) == 1
    w.reset_routed_buffer()
    assert len(w.routed_buffer) == 0
