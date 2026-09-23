from __future__ import annotations

from types import SimpleNamespace

import torch

from herp.archive import RegionArchive, Snapshot
from herp.video_events import (
    CandidateStore,
    RecordedFragment,
    VideoEvent,
    event_is_valid,
    radius_cluster,
    score_outcomes,
)
from scripts.render_mira_video_events import upload_event


def recorded(fragment_id, value, success=False):
    state = torch.tensor([[value, value], [value, value]], dtype=torch.float32)
    return RecordedFragment(
        source_region_id=1,
        policy_version=3,
        fragment_id=fragment_id,
        slot_id=0,
        category="REGION_ACQUISITION",
        states=state.clone(),
        state_features=state.clone(),
        actions=torch.zeros(2, 1),
        next_states=state.clone(),
        rewards=torch.tensor([value, value]),
        terminated=torch.tensor([False, True]),
        truncated=torch.zeros(2, dtype=torch.bool),
        chain_region_id=torch.ones(2, dtype=torch.long),
        successes=torch.tensor([False, success]),
        restart_region_id=1,
        restart_snapshot_index=fragment_id,
        restart_snapshot_timestep=10 + fragment_id,
        restart_snapshot_elapsed_steps=4,
        restart_snapshot=None,
    )


def event(event_id, fragments):
    metrics = score_outcomes(fragments, radius=1.0)
    return VideoEvent(
        event_id=event_id,
        training_step=100,
        policy_version=3,
        region_id=1,
        p_raw=.4,
        p_ema=.5,
        sigma_raw=.6,
        q_direct=.2,
        q_pred=.3,
        q_combined=.25,
        allocation_prob=.4,
        allocated_fragments=len(fragments),
        root_fraction=.2,
        allocation_entropy=.8,
        outcome_radius=1.,
        rollout_fragments=fragments,
        outcome_features=metrics["features"],
        cluster_assignments=metrics["assignments"],
        diversity_score=metrics["ranking_score"],
        pairwise_diversity=metrics["pairwise_diversity"],
        distinct_outcome_count=metrics["distinct_outcomes"],
        success_count=metrics["success_count"],
    )


def test_radius_clustering_has_transparent_fixed_threshold():
    features = torch.tensor([[0., 0.], [.1, .1], [3., 0.], [3.1, .1]])
    assignments = radius_cluster(features, radius=.5)
    assert assignments == [0, 0, 1, 1]


def test_outcome_metrics_and_ranking_reward_distinct_results(tmp_path):
    diverse = event("diverse", [recorded(i, float(i % 3)) for i in range(6)])
    collapsed = event("collapsed", [recorded(i, 0.) for i in range(6)])
    assert diverse.distinct_outcome_count >= 3
    assert diverse.pairwise_diversity > collapsed.pairwise_diversity
    assert event_is_valid(diverse)
    assert not event_is_valid(collapsed)

    store = CandidateStore(tmp_path, max_candidates=1)
    assert store.consider(collapsed)
    assert store.consider(diverse)
    ranked = store.rank()
    assert [item.event_id for item in ranked] == ["diverse"]
    assert (tmp_path / "event_diverse.pt").exists()
    assert not (tmp_path / "event_collapsed.pt").exists()


def test_snapshot_identity_does_not_add_rng_draws():
    left = RegionArchive(seed=17)
    right = RegionArchive(seed=17)
    for archive in (left, right):
        archive.ensure_root()
        region = archive.add_region(torch.zeros(1), 0)
        for index in range(5):
            archive.add_snapshot(region.region_id, Snapshot(index, torch.tensor([float(index)]), index))
    plain = [left.sample_snapshot(left.regions[1]).timestep for _ in range(20)]
    identified = [right.sample_snapshot_with_index(right.regions[1]) for _ in range(20)]
    assert plain == [snapshot.timestep for snapshot, _ in identified]
    assert all(right.regions[1].snapshots[index] is snapshot for snapshot, index in identified)
    assert torch.equal(left._generator.get_state(), right._generator.get_state())


def test_wandb_payload_contains_video_and_traceability(tmp_path):
    clip = tmp_path / "tiny.mp4"
    clip.write_bytes(b"not decoded by this unit test")
    candidate = event("candidate", [recorded(i, float(i)) for i in range(3)])
    candidate.replay_max_state_error = 1e-5

    class Wandb:
        @staticmethod
        def Video(path, **kwargs):
            return {"path": path, **kwargs}

    class Run:
        def __init__(self):
            self.payload = None

        def log(self, payload):
            self.payload = payload

    run = Run()
    upload_event(run, Wandb, candidate, clip, "mira_events")
    assert "mira_events/candidate" in run.payload
    assert run.payload["video_event/candidate/distinct_outcomes"] == candidate.distinct_outcome_count
    assert run.payload["video_event/candidate/region_id"] == 1


def test_capture_toggle_preserves_vector_allocations_and_transitions():
    from herp.chain_features import RunningFeatureNormalizer
    from herp.vector_acquisition import VectorFragmentCollector

    class Env:
        num_envs = 3
        action_dim = 1
        device = torch.device("cpu")

        def __init__(self):
            self.x = torch.zeros(3, 1)
            self.t = torch.zeros(3, dtype=torch.long)

        def reset(self):
            self.x.zero_()
            self.t.zero_()
            return self.x.clone(), {}

        def reset_indices(self, ids):
            self.x[ids] = 0
            self.t[ids] = 0
            return self.x.clone(), {}

        def save_state(self, ids):
            return [self.x[index].clone() for index in ids]

        def elapsed_steps(self):
            return self.t.clone()

        def restore_state(self, ids, snapshots):
            for index, snapshot in zip(ids, snapshots):
                self.x[index] = snapshot
                self.t[index] = 0
            return self.x[ids].clone()

        def action_low(self):
            return torch.full((3, 1), -1.)

        def action_high(self):
            return torch.ones(3, 1)

        def success_from_info(self, _info):
            return torch.zeros(3, dtype=torch.bool)

        def step(self, _action):
            self.x += 1
            self.t += 1
            done = self.t == 2
            final = self.x.clone()
            self.x[done] = 0
            self.t[done] = 0
            return (
                self.x.clone(),
                torch.ones(3),
                torch.zeros(3, dtype=torch.bool),
                done,
                {"final_observation": final},
            )

    class Policy:
        logstd = torch.zeros(1, 1)

        def get_distribution(self, obs):
            return torch.distributions.Normal(torch.zeros_like(obs), torch.ones_like(obs))

        def get_action_and_value(self, obs):
            zeros = torch.zeros(len(obs))
            return torch.zeros_like(obs), zeros, zeros, self.get_value(obs)

        def get_value(self, obs):
            return obs[:, 0]

    def run(enabled):
        archive = RegionArchive(seed=91)
        archive.ensure_root()
        region = archive.add_region(torch.zeros(2), 0)
        archive.add_snapshot(region.region_id, Snapshot(torch.tensor([7.]), torch.tensor([7.]), 11))
        collector = VectorFragmentCollector(Env(), Policy(), RunningFeatureNormalizer(), 4)
        collector.capture_diagnostics = enabled
        generator = torch.Generator().manual_seed(123)
        fragments = collector.collect_round(
            archive, torch.tensor([.25, .75]), 4, generator, reference_slots=0)
        return collector, fragments, generator.get_state(), archive._generator.get_state()

    off_collector, off, off_allocator_rng, off_archive_rng = run(False)
    on_collector, on, on_allocator_rng, on_archive_rng = run(True)
    assert off_collector.counters == on_collector.counters
    assert torch.equal(off_allocator_rng, on_allocator_rng)
    assert torch.equal(off_archive_rng, on_archive_rng)
    assert [f.source_region_id for f in off] == [f.source_region_id for f in on]
    assert [f.restart_snapshot_index for f in off] == [f.restart_snapshot_index for f in on]
    for left, right in zip(off, on):
        torch.testing.assert_close(left.states, right.states)
        torch.testing.assert_close(left.actions, right.actions)
        torch.testing.assert_close(left.next_states, right.next_states)
    assert all(f.restart_snapshot is None for f in off)
    assert any(f.restart_snapshot is not None for f in on)
