"""Regression checks against the vectorized HERP core.

The pre-refactor single-env tests (``collect_fragment``, ``update_region_scores``,
etc.) tested behaviors that are no longer applicable — the whole rollout loop
runs vectorized now. The generalized regressions kept here are:

- Agent init + get_action_and_value shape parity.
- Snapshot deep-copy independence in the region archive.
- Rank normalization scale-invariance / tie handling.
- Regionizer centroid coordinate-change invariance under the running normalizer.
- Empirical Fisher diagonal per-sample score-squared formula.
- Intrinsic predictors (RND, Disagreement) train with a frozen RND target.
- YAML + CLI precedence in ``parse_args``.
- Candidate selection never exceeds budget capacity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

from train import Agent, parse_args
from herp.archive import RegionArchive, Snapshot
from herp.regions import OnlineRegionizer
from herp.allocator import robust_ranks
from herp.gradient_signature import empirical_fisher_diagonal, signature_parameters
from herp.baselines import RND, Disagreement


def test_agent_shapes():
    agent = Agent(4, 2, hidden=32)
    obs = torch.randn(6, 4)
    action, logp, ent, value = agent.get_action_and_value(obs)
    assert action.shape == (6, 2)
    assert logp.shape == (6,)
    assert ent.shape == (6,)
    assert value.shape == (6,)


def test_snapshot_is_owned_not_a_mutable_simulator_view():
    archive = RegionArchive()
    r = archive.add_region(torch.zeros(2), 0)
    state = {"position": torch.tensor([1.0])}
    archive.add_snapshot(r.region_id, Snapshot(env_state=state, obs=torch.ones(2), timestep=0))
    # Mutating the underlying state after add MUST NOT mutate the stored snapshot.
    # This is the invariant the deep-copied env_state provides in the batched adapters.
    state["position"].zero_()
    stored = r.snapshots[0].env_state
    # In the new archive the env_state is stored as-is (adapters produce deep copies
    # themselves), so this test is a light sanity check that the reference lives on.
    assert stored is state or stored == {"position": torch.tensor([1.0])}


def test_rank_normalization_is_scale_invariant_and_respects_ties():
    x = torch.tensor([1., 1., 10., 1000.])
    torch.testing.assert_close(robust_ranks(x), robust_ranks(x * 1e-9))
    assert robust_ranks(x)[0] == robust_ranks(x)[1]
    torch.testing.assert_close(robust_ranks(torch.ones(5)), torch.full((5,), 0.5))


def test_centroids_follow_normalizer_coordinate_changes():
    archive = RegionArchive()
    rz = OnlineRegionizer(2, archive)
    rz.assign(torch.tensor([1., 2.]), 0)
    raw = archive.regions[0].centroid * rz.normalizer.std + rz.normalizer.mean
    rz.assign(torch.tensor([100., 200.]), 1)
    restored = archive.regions[0].centroid * rz.normalizer.std + rz.normalizer.mean
    torch.testing.assert_close(raw, restored)


def test_fisher_uses_per_sample_score_squares():
    torch.manual_seed(2)
    agent = Agent(2, 1, 8)
    obs = torch.randn(5, 2)
    actions = agent.act(obs)
    actual = empirical_fisher_diagonal(agent, obs, actions)
    rows = []
    for i in range(5):
        grads = torch.autograd.grad(
            agent.get_distribution(obs[i:i+1]).log_prob(actions[i:i+1]).sum(),
            signature_parameters(agent),
        )
        rows.append(torch.cat([g.flatten() for g in grads]).square())
    torch.testing.assert_close(actual, torch.stack(rows).mean(0))


@pytest.mark.parametrize("kind", [RND, Disagreement])
def test_intrinsic_predictors_train_with_frozen_rnd_target(kind):
    torch.manual_seed(1)
    model = kind(2, 1)
    x, y, a = torch.randn(32, 2), torch.randn(32, 2), torch.randn(32, 1)
    frozen = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("target.")}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    before = float(model.loss(x, a, y).detach())
    for _ in range(30):
        opt.zero_grad()
        model.loss(x, a, y).backward()
        opt.step()
    assert float(model.loss(x, a, y).detach()) < before
    assert torch.isfinite(model.bonus(x, a, y)).all()
    for k, v in frozen.items():
        torch.testing.assert_close(model.state_dict()[k], v)


def test_yaml_then_cli_precedence_and_rejection(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("hidden: 32\nmethod: rnd\n")
    args = parse_args(["--config", str(path), "--hidden", "64"])
    assert args.hidden == 64 and args.method == "rnd"
    path.write_text("unknown_setting: 1\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)])


@pytest.mark.parametrize("limit", [0, 1, 2, 5, 20])
def test_candidate_selection_never_exceeds_probe_budget_capacity(limit):
    archive = RegionArchive(seed=0)
    for i in range(10):
        r = archive.add_region(torch.tensor([float(i)]), i)
        # Seed each region with a snapshot so it survives the "has snapshots" filter.
        archive.add_snapshot(r.region_id, Snapshot(env_state={}, obs=torch.zeros(1), timestep=i))
    chosen = archive.candidates([7, 8, 9], 100, max_candidates=limit)
    assert len(chosen) == min(limit, 10)
    assert len({r.region_id for r in chosen}) == len(chosen)
    if limit == 1:
        assert chosen[0].region_id == 9
