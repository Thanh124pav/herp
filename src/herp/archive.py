from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy

import torch


@dataclass
class Snapshot:
    """A previously reached simulator state that can seed short rollouts."""

    env_state: Any
    obs: torch.Tensor
    timestep: int
    episode_id: int
    return_so_far: float
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


class RegionArchive:
    """Stores regions and bounded per-region state snapshots."""

    def __init__(self, max_snapshots_per_region: int = 8, seed: int = 0):
        self.max_snapshots_per_region = int(max_snapshots_per_region)
        self.regions: list[Region] = []
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

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
        snapshot = deepcopy(snapshot)
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
            if 0 <= rid < len(self.regions):
                selected[rid] = self.regions[rid]
            if len(selected) >= max_candidates:
                return list(selected.values())
            if len(selected) >= n_recent:
                break

        n_stale = max(1, int(round(max_candidates * stale_fraction)))
        stale = sorted(self.regions, key=lambda r: (step - r.last_probed_step), reverse=True)
        for region in stale[:n_stale]:
            selected.setdefault(region.region_id, region)
            if len(selected) >= max_candidates:
                return list(selected.values())

        order = torch.randperm(len(self.regions), generator=self._generator).tolist()
        for rid in order:
            selected.setdefault(rid, self.regions[rid])
            if len(selected) >= max_candidates:
                break
        return list(selected.values())
