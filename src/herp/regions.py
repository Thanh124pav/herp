from __future__ import annotations

from dataclasses import dataclass

import torch

from .archive import RegionArchive, Snapshot


@dataclass
class RegionizerConfig:
    region_radius: float = 0.5
    max_regions: int = 128
    centroid_tau: float = 0.05
    obs_clip: float = 10.0
    eps: float = 1e-6


class RunningNormalizer:
    def __init__(self, shape: int | tuple[int, ...], eps: float = 1e-6):
        self.count = 0
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.m2 = torch.zeros(shape, dtype=torch.float32)
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().cpu().float().reshape(-1, *self.mean.shape)
        for item in x:
            self.count += 1
            delta = item - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (item - self.mean)

    @property
    def std(self) -> torch.Tensor:
        if self.count < 32:
            return torch.ones_like(self.mean)
        return torch.sqrt(self.m2 / max(1, self.count - 1) + self.eps)

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return torch.clamp((x.detach().float() - self.mean) / self.std, -clip, clip)


class OnlineRegionizer:
    """Radius/k-center region assignment over normalized state features."""

    def __init__(self, obs_dim: int, archive: RegionArchive, cfg: RegionizerConfig | None = None):
        self.cfg = cfg or RegionizerConfig()
        self.archive = archive
        self.normalizer = RunningNormalizer(obs_dim, eps=self.cfg.eps)

    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.detach().cpu().float().flatten()
        return self.normalizer.normalize(obs, clip=self.cfg.obs_clip).flatten()

    def assign(self, obs: torch.Tensor, step: int, update_stats: bool = True) -> int:
        obs = obs.detach().cpu().float().flatten()
        if update_stats:
            # Keep centroids in the same coordinate system as the moving normalizer.
            old_mean, old_std = self.normalizer.mean.clone(), self.normalizer.std.clone()
            self.normalizer.update(obs)
            for region in self.archive:
                raw = region.centroid * old_std + old_mean
                region.centroid = (raw - self.normalizer.mean) / self.normalizer.std
        z = self.phi(obs)
        if len(self.archive.regions) == 0:
            return self.archive.add_region(z, step).region_id

        centroids = torch.stack([r.centroid.to(z.device) for r in self.archive.regions])
        distances = torch.cdist(z[None], centroids).squeeze(0)
        min_dist, idx = distances.min(dim=0)
        if (
            min_dist.item() > self.cfg.region_radius
            and len(self.archive.regions) < self.cfg.max_regions
        ):
            return self.archive.add_region(z, step).region_id

        region = self.archive.regions[int(idx)]
        tau = self.cfg.centroid_tau
        region.centroid = (1.0 - tau) * region.centroid + tau * z.cpu()
        region.last_seen_step = int(step)
        return region.region_id

    def observe_snapshot(
        self,
        env_state,
        obs: torch.Tensor,
        timestep: int,
        episode_id: int,
        return_so_far: float,
        elapsed_steps: int = 0,
    ) -> int:
        region_id = self.assign(obs, timestep, update_stats=True)
        self.archive.add_snapshot(
            region_id,
            Snapshot(env_state, obs.detach().cpu(), timestep, episode_id, return_so_far, elapsed_steps),
        )
        return region_id
