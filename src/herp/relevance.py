from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .archive import Region


def cosine_relevance(region_gradient: torch.Tensor, reference_gradient: torch.Tensor) -> float:
    if region_gradient.numel() == 0 or reference_gradient.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(region_gradient.flatten(), reference_gradient.flatten(), dim=0))


@dataclass
class RelevanceConfig:
    ema_tau: float = 0.9
    eps_p: float = 0.05
    alpha_p: float = 1.0


class RelevanceTracker:
    def __init__(self, cfg: RelevanceConfig | None = None):
        self.cfg = cfg or RelevanceConfig()

    def update_cosine(
        self,
        region: Region,
        region_gradient: torch.Tensor,
        reference_gradient: torch.Tensor,
    ) -> float:
        raw = cosine_relevance(region_gradient, reference_gradient)
        positive = max(0.0, raw)
        region.p_raw = raw
        region.p_ema = self.cfg.ema_tau * region.p_ema + (1.0 - self.cfg.ema_tau) * positive
        return region.p_ema

    def normalized(self, regions: list[Region]) -> dict[int, float]:
        if not regions:
            return {}
        weights = torch.tensor(
            [(max(0.0, r.p_ema) + self.cfg.eps_p) ** self.cfg.alpha_p for r in regions],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-8)
        return {r.region_id: float(weights[i]) for i, r in enumerate(regions)}
