"""Scientific event packages and outcome-diversity ranking for MIRA training.

This module is deliberately simulator-agnostic.  Training records exact actions
and the already selected restart snapshot; rendering is a separate post-training
operation so event selection cannot affect policy or environment transitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import copy
import json
import math

import torch


@dataclass
class RecordedFragment:
    source_region_id: int
    policy_version: int
    fragment_id: int
    slot_id: int
    category: str
    states: torch.Tensor
    state_features: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    chain_region_id: torch.Tensor | None
    successes: torch.Tensor | None
    restart_region_id: int | None
    restart_snapshot_index: int | None
    restart_snapshot_timestep: int | None
    restart_snapshot_elapsed_steps: int | None
    restart_snapshot: object | None

    @classmethod
    def from_rollout(cls, fragment) -> "RecordedFragment":
        def cpu_clone(value):
            return value.detach().cpu().clone() if torch.is_tensor(value) else value

        return cls(
            source_region_id=int(fragment.source_region_id),
            policy_version=int(fragment.policy_version),
            fragment_id=int(fragment.fragment_id),
            slot_id=int(getattr(fragment, "slot_id", 0)),
            category=str(fragment.category),
            states=cpu_clone(fragment.states),
            state_features=cpu_clone(fragment.state_features),
            actions=cpu_clone(fragment.actions),
            next_states=cpu_clone(fragment.next_states),
            rewards=cpu_clone(fragment.rewards),
            terminated=cpu_clone(fragment.terminated),
            truncated=cpu_clone(fragment.truncated),
            chain_region_id=cpu_clone(getattr(fragment, "chain_region_id", None)),
            successes=cpu_clone(getattr(fragment, "successes", None)),
            restart_region_id=getattr(fragment, "restart_region_id", None),
            restart_snapshot_index=getattr(fragment, "restart_snapshot_index", None),
            restart_snapshot_timestep=getattr(fragment, "restart_snapshot_timestep", None),
            restart_snapshot_elapsed_steps=getattr(fragment, "restart_snapshot_elapsed_steps", None),
            # Strip trajectory provenance from the snapshot itself: origins are
            # stored once in VideoEvent.pre_fragments, not duplicated recursively
            # inside every rollout snapshot.
            restart_snapshot=_copy_snapshot_without_origins(
                getattr(fragment, "restart_snapshot", None)),
        )


def _copy_snapshot_without_origins(snapshot):
    if snapshot is None:
        return None
    clean = copy.copy(snapshot)
    if hasattr(clean, "origin_fragments"):
        clean.origin_fragments = []
    return copy.deepcopy(clean)


@dataclass
class VideoEvent:
    event_id: str
    training_step: int
    policy_version: int
    region_id: int
    p_raw: float
    p_ema: float
    sigma_raw: float
    q_direct: float
    q_pred: float
    q_combined: float
    allocation_prob: float
    allocated_fragments: int
    root_fraction: float
    allocation_entropy: float
    outcome_radius: float
    pre_fragments: list[RecordedFragment] = field(default_factory=list)
    rollout_fragments: list[RecordedFragment] = field(default_factory=list)
    outcome_features: torch.Tensor | None = None
    cluster_assignments: list[int] = field(default_factory=list)
    diversity_score: float = 0.0
    pairwise_diversity: float = 0.0
    distinct_outcome_count: int = 0
    success_count: int = 0
    source: dict[str, Any] = field(default_factory=dict)
    replay_max_state_error: float | None = None
    replay_exact: bool | None = None

    def metadata(self) -> dict[str, Any]:
        data = {
            "event_id": self.event_id,
            "training_step": self.training_step,
            "policy_version": self.policy_version,
            "region_id": self.region_id,
            "p_raw": self.p_raw,
            "p_ema": self.p_ema,
            "sigma_raw": self.sigma_raw,
            "q_direct": self.q_direct,
            "q_pred": self.q_pred,
            "q_combined": self.q_combined,
            "allocation_prob": self.allocation_prob,
            "allocated_fragments": self.allocated_fragments,
            "root_fraction": self.root_fraction,
            "allocation_entropy": self.allocation_entropy,
            "outcome_radius": self.outcome_radius,
            "distinct_outcomes": self.distinct_outcome_count,
            "pairwise_diversity": self.pairwise_diversity,
            "ranking_score": self.diversity_score,
            "success_count": self.success_count,
            "pre_fragment_count": len(self.pre_fragments),
            "rollout_fragment_count": len(self.rollout_fragments),
            "fragment_ids": [f.fragment_id for f in self.rollout_fragments],
            "restart_snapshots": [
                {
                    "fragment_id": f.fragment_id,
                    "archive_index": f.restart_snapshot_index,
                    "timestep": f.restart_snapshot_timestep,
                    "elapsed_steps": f.restart_snapshot_elapsed_steps,
                }
                for f in self.rollout_fragments
            ],
            "source": self.source,
            "replay_max_state_error": self.replay_max_state_error,
            "replay_exact": self.replay_exact,
        }
        return _json_finite(data)


def _json_finite(value):
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def outcome_features(fragments: list[RecordedFragment], tail_steps: int = 5) -> torch.Tensor:
    """Build normalized state-primary outcome features for one restart event."""
    if not fragments:
        return torch.empty(0, 0, dtype=torch.float64)
    rows = []
    state_width = None
    for fragment in fragments:
        features = torch.as_tensor(fragment.state_features, dtype=torch.float64)
        if len(features) == 0:
            raise ValueError("Outcome fragments must contain at least one transition")
        k = min(tail_steps, len(features))
        state = torch.cat([features[-1], features[-k:].mean(0)])
        state_width = len(state)
        success = (
            torch.as_tensor(fragment.successes, dtype=torch.bool)
            if fragment.successes is not None
            else torch.zeros(len(features), dtype=torch.bool)
        )
        extras = torch.tensor(
            [
                float(torch.as_tensor(fragment.rewards).sum()),
                float(bool(torch.as_tensor(fragment.terminated)[-1])),
                float(bool(success.any())),
                float(bool(success[-1])),
            ],
            dtype=torch.float64,
        )
        rows.append(torch.cat([state, extras]))
    matrix = torch.stack(rows)
    # Normalize continuous state and return components within the event. Binary
    # terminal/success flags retain their direct interpretation.
    continuous = matrix[:, : state_width + 1]
    mean = continuous.mean(0)
    scale = continuous.std(0, unbiased=False).clamp_min(1e-6)
    matrix[:, : state_width + 1] = (continuous - mean) / scale
    return matrix


def radius_cluster(features: torch.Tensor, radius: float) -> list[int]:
    if radius <= 0:
        raise ValueError("Outcome radius must be positive")
    if len(features) == 0:
        return []
    centers: list[torch.Tensor] = []
    members: list[list[torch.Tensor]] = []
    assignments = []
    for row in features:
        if not centers:
            centers.append(row.clone())
            members.append([row])
            assignments.append(0)
            continue
        distances = torch.stack([(row - center).norm() for center in centers])
        nearest = int(distances.argmin())
        if float(distances[nearest]) > radius:
            assignments.append(len(centers))
            centers.append(row.clone())
            members.append([row])
        else:
            assignments.append(nearest)
            members[nearest].append(row)
            centers[nearest] = torch.stack(members[nearest]).mean(0)
    return assignments


def score_outcomes(fragments: list[RecordedFragment], radius: float) -> dict[str, Any]:
    features = outcome_features(fragments)
    assignments = radius_cluster(features, radius)
    n = len(features)
    if n > 1:
        pairwise = float(torch.pdist(features, p=2).mean())
    else:
        pairwise = 0.0
    distinct = len(set(assignments))
    ranking = math.log1p(distinct) * pairwise * math.log1p(n)
    success_count = sum(
        int(f.successes is not None and bool(torch.as_tensor(f.successes).any()))
        for f in fragments
    )
    return {
        "features": features,
        "assignments": assignments,
        "distinct_outcomes": distinct,
        "pairwise_diversity": pairwise,
        "ranking_score": ranking,
        "success_count": success_count,
    }


def build_event(*, metadata: dict[str, Any], fragments, pre_fragments, radius: float) -> VideoEvent:
    recorded = [RecordedFragment.from_rollout(f) for f in fragments]
    pre = [RecordedFragment.from_rollout(f) for f in pre_fragments]
    metrics = score_outcomes(recorded, radius)
    event = VideoEvent(
        **metadata,
        outcome_radius=float(radius),
        pre_fragments=pre,
        rollout_fragments=recorded,
    )
    event.outcome_features = metrics["features"]
    event.cluster_assignments = metrics["assignments"]
    event.distinct_outcome_count = metrics["distinct_outcomes"]
    event.pairwise_diversity = metrics["pairwise_diversity"]
    event.diversity_score = metrics["ranking_score"]
    event.success_count = metrics["success_count"]
    return event


def save_event(event: VideoEvent, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f"event_{event.event_id}.pt"
    metadata = directory / f"event_{event.event_id}.json"
    temporary = payload.with_suffix(".pt.tmp")
    torch.save(event, temporary)
    temporary.replace(payload)
    metadata.write_text(json.dumps(event.metadata(), indent=2))
    return payload, metadata


def load_event(path: str | Path) -> VideoEvent:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class CandidateStore:
    """Disk-backed bounded top-score event set, safe to resume after a crash."""

    def __init__(self, directory: str | Path, max_candidates: int):
        self.directory = Path(directory)
        self.max_candidates = int(max_candidates)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events: dict[str, VideoEvent] = {}
        for path in self.directory.glob("event_*.pt"):
            try:
                event = load_event(path)
                self.events[event.event_id] = event
            except Exception:
                continue

    def consider(self, event: VideoEvent) -> bool:
        if self.max_candidates <= 0:
            return False
        self.events[event.event_id] = event
        ranked = sorted(self.events.values(), key=lambda x: x.diversity_score, reverse=True)
        kept = ranked[: self.max_candidates]
        keep_ids = {event.event_id for event in kept}
        accepted = event.event_id in keep_ids
        for old_id in list(self.events):
            if old_id not in keep_ids:
                self.events.pop(old_id)
                for suffix in (".pt", ".json", ".mp4"):
                    path = self.directory / f"event_{old_id}{suffix}"
                    if path.exists():
                        path.unlink()
        if accepted:
            save_event(event, self.directory)
        return accepted

    def rank(self) -> list[VideoEvent]:
        ranked = sorted(self.events.values(), key=lambda x: x.diversity_score, reverse=True)
        rows = []
        for rank, event in enumerate(ranked, 1):
            row = event.metadata()
            row["rank"] = rank
            mp4 = self.directory / f"event_{event.event_id}.mp4"
            row["mp4"] = mp4.name if mp4.exists() else None
            rows.append(row)
        (self.directory / "ranking.json").write_text(json.dumps(rows, indent=2))
        return ranked


def event_is_valid(event: VideoEvent, min_fragments: int = 3, min_distinct: int = 2,
                   min_diversity: float = 0.0) -> bool:
    return (
        len(event.rollout_fragments) >= min_fragments
        and event.distinct_outcome_count >= min_distinct
        and event.pairwise_diversity > min_diversity
    )
