"""AllocationController is the backbone-independent bridge between any
Learner (PPO/SAC/MBRL) and HERP's region-based acquisition. These tests
lock the fair-comparison invariants: normalizer freezes after calibration,
gradient-signature accounting is actor-only and RNG-isolated, and
finish_round produces finite p/sigma even in degenerate regimes.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from herp.allocation_controller import AllocationController
from herp.config import HERPV3Config
from herp.archive import Snapshot


class _StubAdapter:
    """Minimal adapter matching the AllocationController's needs.

    Never touches a real simulator so tests stay CPU-only and offline.
    """
    num_envs = 1
    action_dim = 2
    obs_dim = 3

    def __init__(self):
        self.device = torch.device('cpu')
        self._counter = torch.zeros(1, dtype=torch.long)

    def save_state(self, env_ids):
        return [SimpleNamespace(state_dict={}, elapsed_steps=0) for _ in env_ids]

    def elapsed_steps(self):
        return self._counter

    def action_low(self):
        return torch.tensor([-1., -1.])

    def action_high(self):
        return torch.tensor([1., 1.])


class _RecordingLearner:
    """Records every gradient_signature() invocation to catch leaks.

    Returns a deterministic signature per batch so finish_round is
    numerically reproducible even without real network parameters.
    """
    def __init__(self):
        self.device = torch.device('cpu')
        self.calls = []
        # Fake parameter so gradient_signature callers can assert no grad leak.
        self.param = torch.nn.Parameter(torch.zeros(4))

    def gradient_signature(self, batch):
        obs = batch['obs']
        self.calls.append(int(obs.shape[0]))
        # Deterministic pseudo-gradient — length matches actor params so the
        # cosine ratio in finish_round exercises a real vector-space update.
        return torch.tensor([float(obs.sum()), float(obs.mean()), 1.0, -1.0])

    def policy_parameters(self):
        return [self.param]


def _controller():
    adapter = _StubAdapter()
    cfg = HERPV3Config(future_horizon=4, min_common_steps=2, predictor_min_labels=2,
                       max_regions=8, chain_radius=2., sigma_predictor_kappa=8.)
    return AllocationController(adapter, cfg, seed=0, sigma_mode='shrinkage'), cfg, adapter


def test_choose_falls_back_to_root_until_min_non_root_regions_seen():
    """New archives have only root; choose() must not sample from empty archives."""
    controller, cfg, _ = _controller()
    rid, snapshot, probs = controller.choose('herp')
    assert rid == 0 and snapshot is None
    # priority_distribution over just root gives a point mass at index 0.
    assert probs.numel() == 1 and torch.isclose(probs.sum(), torch.tensor(1., dtype=torch.float64))


def test_calibrate_updates_normalizer_and_activation_flag():
    """Normalizer must accept counted warmup samples and stay unfrozen until
    the controller activates — freezing early breaks THEORY §17 (predictor
    labels use post-calibration features)."""
    controller, _, _ = _controller()
    assert not controller.active
    obs = torch.randn(64, 3)
    controller.calibrate(obs)
    assert controller.normalizer.count == 64
    controller.active = True
    with pytest.raises(RuntimeError, match='already frozen'):
        controller.calibrate(torch.randn(4, 3))


def test_fragment_features_use_action_and_state_normalization():
    """fragment() produces the padded trajectory features consumed by sigma
    estimators. Actions are min/max-normalized to [-1, 1]; features must
    respect the (future_horizon, feature_dim) shape regardless of collect length."""
    controller, cfg, adapter = _controller()
    controller.calibrate(torch.zeros(8, 3))  # unit normalizer
    obs = torch.zeros(2, 3)
    next_obs = torch.ones(2, 3)
    actions = torch.zeros(2, adapter.action_dim)  # midpoint -> normalized 0
    frag = controller.fragment(0, obs, next_obs, actions, version=0, root_started=True)
    assert frag.traj_features.shape == (cfg.future_horizon, 3 + adapter.action_dim)
    assert frag.valid_mask.tolist() == [True, True, False, False]
    # Un-normalized actions at low/high should hit -1 / +1 exactly.
    frag_low = controller.fragment(0, obs, next_obs, torch.full_like(actions, -1.), 0, True)
    frag_high = controller.fragment(0, obs, next_obs, torch.full_like(actions, 1.), 0, True)
    assert torch.allclose(frag_low.traj_features[0, 3:], -torch.ones(adapter.action_dim))
    assert torch.allclose(frag_high.traj_features[0, 3:], torch.ones(adapter.action_dim))


def test_finish_round_forks_rng_and_leaves_no_gradients_on_actor_params():
    """finish_round() computes actor-gradient signatures for p_v. It MUST NOT
    (a) leak gradients onto learner.policy_parameters() (would poison next
    optimizer step), (b) advance the training RNG stream (would break
    reproducibility of learner.act after diagnostics)."""
    controller, cfg, adapter = _controller()
    controller.calibrate(torch.zeros(16, 3))
    controller.active = True
    # Force a non-root region to exist so finish_round has both root+child
    # fragment groups.
    controller.archive.add_region(torch.zeros(6), step=0)
    controller.archive.regions[1].snapshots.append(
        Snapshot(env_state={}, obs=torch.zeros(3), timestep=0, elapsed_steps=0)
    )
    fragments = []
    for rid in (0, 1):
        obs = torch.randn(cfg.future_horizon, 3)
        next_obs = torch.randn(cfg.future_horizon, 3)
        actions = torch.zeros(cfg.future_horizon, adapter.action_dim)
        f = controller.fragment(rid, obs, next_obs, actions, version=0, root_started=rid == 0)
        fragments.append(f)
    reference_obs = torch.randn(cfg.future_horizon, 3)
    learner = _RecordingLearner()
    torch.manual_seed(1234)
    reference_rng_before = torch.get_rng_state()
    diagnostics = controller.finish_round(fragments, reference_obs, learner, version=0,
                                          pre_features={r.region_id: torch.zeros(5, dtype=torch.float64)
                                                        for r in controller.archive})
    reference_rng_after = torch.get_rng_state()
    assert learner.param.grad is None, 'gradient_signature leaked into learner params'
    assert torch.equal(reference_rng_before, reference_rng_after), \
        'finish_round advanced the training RNG stream (must fork_rng for diagnostics)'
    assert diagnostics['num_regions'] == 2
    # gradient_signature called: 1 for reference + 1 per region group.
    assert len(learner.calls) == 3
    for r in controller.archive:
        assert math.isfinite(r.sigma_raw) and r.sigma_raw >= cfg.sigma_floor
        assert -1 - 1e-6 <= r.p_ema <= 1 + 1e-6


def test_finish_round_survives_missing_reference_and_zero_fragments():
    """Corner: no reference observations (early rounds) and one fragment per
    region (below min_common_steps). Must still populate q_pred via predictor
    prior and not raise."""
    controller, cfg, _ = _controller()
    controller.calibrate(torch.zeros(4, 3))
    controller.active = True
    controller.archive.add_region(torch.zeros(6), step=0)
    fragments = [
        controller.fragment(1, torch.randn(cfg.future_horizon, 3),
                            torch.randn(cfg.future_horizon, 3),
                            torch.zeros(cfg.future_horizon, 2), version=0, root_started=False)
    ]
    learner = _RecordingLearner()
    diagnostics = controller.finish_round(fragments, torch.empty(0, 3), learner, version=0,
                                          pre_features={r.region_id: torch.zeros(5, dtype=torch.float64)
                                                        for r in controller.archive})
    assert diagnostics['num_regions'] == 2
    # No reference -> no gradient calls at all.
    assert learner.calls == []
    for r in controller.archive:
        assert math.isfinite(r.sigma_raw) and r.sigma_raw >= cfg.sigma_floor
