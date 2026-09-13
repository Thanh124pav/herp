"""Region archive with per-slot batched snapshots (IMPLEMENTATION.md §4.7).

The archive is process-local; snapshots are held per-region and sampled by the
probe scheduler. ``add(env_ids, snapshots, region_ids)`` accepts the batched
output of an adapter's ``save_state``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Snapshot:
    """A previously reached simulator state that can seed short rollouts."""

    env_state: Any
    obs: torch.Tensor
    timestep: int
    episode_id: int = 0
    return_so_far: float = 0.0
    elapsed_steps: int = 0


@dataclass
class Region:
    region_id: int
    centroid: torch.Tensor
    count: int = 0
    snapshots: list[Snapshot] = field(default_factory=list)
    last_seen_step: int = 0
    last_probed_step: int = 0
    sigma_raw: float = 0.0
    sigma_ema: float = 0.0
    p_raw: float = 0.0
    p_ema: float = 0.0
    priority: float = 0.0
    gradient_norm: float = 0.0
    snapshot_count: int = 0
    is_root: bool = False
    num_chains: int = 0
    num_entries: int = 0
    mean_chain_len: float = 0.
    mean_policy_change: float = 0.
    mean_action_entropy: float = 0.
    mean_state_change: float = 0.
    q_direct: float = float("nan")
    q_pred: float = 0.
    q_combined: float = 0.
    sigma_sample_count: int = 0
    allocated_fragments: int = 0
    last_scored_step: int = 0
    td_error_ema: float = 0.
    value_change: float = 0.
    previous_value: float | None = None


class RegionArchive:
    """Regions + bounded per-region state snapshots."""

    def __init__(self, max_snapshots_per_region: int = 8, seed: int = 0):
        self.max_snapshots_per_region = int(max_snapshots_per_region)
        self.regions: list[Region] = []
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    def ensure_root(self):
        if not self.regions:
            self.regions.append(Region(0, torch.empty(0), is_root=True))
        if not self.regions[0].is_root:
            raise ValueError("Cannot reinterpret a legacy archive as v3")
        return self.regions[0]

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def add_region(self, centroid: torch.Tensor, step: int) -> Region:
        region = Region(
            region_id=len(self.regions),
            centroid=centroid.detach().clone().float().flatten(),
            count=1,
            last_seen_step=int(step),
        )
        self.regions.append(region)
        return region

    def add_snapshot(self, region_id: int, snapshot: Snapshot) -> None:
        region = self.regions[region_id]
        region.count += 1
        region.snapshot_count += 1
        region.last_seen_step = int(snapshot.timestep)
        if self.max_snapshots_per_region <= 0:
            return
        if len(region.snapshots) < self.max_snapshots_per_region:
            region.snapshots.append(snapshot)
            return
        draw = torch.randint(region.snapshot_count, (1,), generator=self._generator).item()
        if draw < self.max_snapshots_per_region:
            region.snapshots[draw] = snapshot

    def add(
        self,
        env_ids: torch.Tensor,
        snapshots: list,
        region_ids: list[int],
        obs: torch.Tensor,
        step: int,
    ) -> None:
        """Batched add: ``snapshots[i]`` belongs to slot ``env_ids[i]`` in region ``region_ids[i]``."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).tolist()
        for env_idx, snap, rid in zip(env_ids, snapshots, region_ids):
            packaged = Snapshot(
                env_state=snap,
                obs=obs[env_idx].detach().cpu().clone(),
                timestep=int(step),
                episode_id=0,
                return_so_far=0.0,
                elapsed_steps=getattr(snap, "elapsed_steps", 0),
            )
            self.add_snapshot(int(rid), packaged)

    def sample_snapshot(self, region: Region) -> Snapshot:
        if not region.snapshots:
            raise ValueError(f"region {region.region_id} has no snapshots")
        idx = torch.randint(len(region.snapshots), (1,), generator=self._generator).item()
        return region.snapshots[idx]

    def candidates(
        self,
        recent_region_ids: list[int] | None,
        step: int,
        max_candidates: int = 16,
        stale_fraction: float = 0.25,
    ) -> list[Region]:
        if not self.regions or max_candidates <= 0:
            return []
        selected: dict[int, Region] = {}
        n_recent = max(1, max_candidates // 2)
        for rid in reversed(recent_region_ids or []):
            if 0 <= rid < len(self.regions) and self.regions[rid].snapshots:
                selected[rid] = self.regions[rid]
            if len(selected) >= max_candidates:
                return list(selected.values())
            if len(selected) >= n_recent:
                break

        n_stale = max(1, int(round(max_candidates * stale_fraction)))
        stale = sorted(
            [r for r in self.regions if r.snapshots],
            key=lambda r: (step - r.last_probed_step),
            reverse=True,
        )
        for region in stale[:n_stale]:
            selected.setdefault(region.region_id, region)
            if len(selected) >= max_candidates:
                return list(selected.values())

        order = torch.randperm(len(self.regions), generator=self._generator).tolist()
        for rid in order:
            r = self.regions[rid]
            if r.snapshots:
                selected.setdefault(rid, r)
            if len(selected) >= max_candidates:
                break
        return list(selected.values())
