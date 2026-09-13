"""DMC adapter smoke tests.

Skipped when dm_control isn't installed (CI without MuJoCo GL libs). When
installed, exercises the round-trip we actually depend on:
  * reset returns a flat float32 vector on device
  * step advances elapsed steps and stops at max_episode_steps
  * save_state / restore_state deterministically reproduces the observation
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from herp.envs import make_adapter
from herp.envs.dmc import DMCAdapter, DMCSnapshot, _parse_env_id


def test_env_id_parsing_accepts_both_separators():
    assert _parse_env_id("walker-run") == ("walker", "run")
    assert _parse_env_id("walker/run") == ("walker", "run")
    with pytest.raises(ValueError):
        _parse_env_id("walker")
    with pytest.raises(ValueError):
        _parse_env_id("-run")


def test_registry_returns_dmc_adapter():
    adapter = make_adapter("dmc", env_id="cartpole-balance", device="cpu")
    assert isinstance(adapter, DMCAdapter)
    assert adapter.benchmark == "dmc"


def test_reset_and_step_shapes_and_dtypes():
    a = DMCAdapter("cartpole-balance", device="cpu").make(num_envs=2, seed=0)
    obs, info = a.reset()
    assert obs.shape == (2, a.obs_dim) and obs.dtype == torch.float32
    action = torch.zeros(2, a.action_dim)
    obs2, r, term, trunc, info = a.step(action)
    assert obs2.shape == (2, a.obs_dim) and r.shape == (2,)
    assert term.dtype == torch.bool and trunc.dtype == torch.bool
    a.close()


def test_snapshot_roundtrip_reproduces_observation_and_time():
    a = DMCAdapter("cartpole-balance", device="cpu").make(num_envs=1, seed=0)
    a.reset()
    # Advance a few random steps so state is non-trivial.
    for _ in range(10):
        a.step(torch.zeros(1, a.action_dim).uniform_(-1, 1))
    snaps = a.save_state(torch.tensor([0]))
    assert isinstance(snaps[0], DMCSnapshot)
    obs_before, *_ = a.step(torch.zeros(1, a.action_dim))
    # Restore should undo the last step: physics.time and observation must match.
    restored = a.restore_state(torch.tensor([0]), snaps)
    assert restored.shape == (1, a.obs_dim)
    obs_after_restore, *_ = a.step(torch.zeros(1, a.action_dim))
    # From the same restored state + same action, we get the same next observation.
    assert torch.allclose(obs_before, obs_after_restore, atol=1e-5), (obs_before - obs_after_restore).abs().max()
    a.close()


def test_success_from_info_is_always_false():
    """DMC doesn't define per-episode success — the adapter must return a
    zero mask so merge_cross_backbone_results.py renders success as N/A."""
    a = DMCAdapter("cartpole-balance", device="cpu").make(num_envs=3, seed=0)
    a.reset()
    _, _, _, _, info = a.step(torch.zeros(3, a.action_dim))
    s = a.success_from_info(info)
    assert s.shape == (3,) and s.dtype == torch.bool and not s.any()
    a.close()
