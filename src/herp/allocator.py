from __future__ import annotations

from dataclasses import dataclass

import torch

from .archive import Region


@dataclass
class AllocationConfig:
    alpha: float = 1.0
    beta: float = 1.0
    eps_p: float = 0.05
    eps_sigma: float = 0.05
    uniform_mix: float = 0.10
    n_min: int = 0
    staleness_mix: float = 0.05
    current_step: int = 0
    normalize_sigma: bool = True


def robust_ranks(values: torch.Tensor) -> torch.Tensor:
    """Average ranks in [0,1]; ties receive the same rank (constant -> 0.5)."""
    if values.numel() < 2:
        return torch.ones_like(values) * 0.5
    less = (values[:, None] > values[None, :]).sum(1)
    equal = (values[:, None] == values[None, :]).sum(1)
    return (less + (equal - 1) / 2) / (values.numel() - 1)


def priority_distribution(regions: list[Region], cfg: AllocationConfig | None = None) -> torch.Tensor:
    cfg = cfg or AllocationConfig()
    if not regions:
        return torch.zeros(0)
    p = torch.tensor([max(0.0, r.p_ema) for r in regions], dtype=torch.float32)
    sigma = torch.tensor([max(0.0, r.sigma_ema) for r in regions], dtype=torch.float32)
    if cfg.normalize_sigma:
        sigma = robust_ranks(sigma)
    score = (p + cfg.eps_p).pow(cfg.alpha) * (sigma + cfg.eps_sigma).pow(cfg.beta)
    if not torch.isfinite(score).all() or score.sum() <= 0:
        q = torch.full_like(score, 1.0 / len(regions))
    else:
        q = score / score.sum()
    if cfg.staleness_mix > 0:
        age = torch.tensor([max(0, cfg.current_step - r.last_probed_step) + 1 for r in regions], dtype=torch.float32)
        q = (1 - cfg.staleness_mix) * q + cfg.staleness_mix * age / age.sum()
    if cfg.uniform_mix > 0:
        q = (1.0 - cfg.uniform_mix) * q + cfg.uniform_mix / len(regions)
    return q / q.sum().clamp_min(1e-8)


def allocate_budget(
    regions: list[Region],
    budget: int,
    cfg: AllocationConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict[int, int]:
    cfg = cfg or AllocationConfig()
    if budget <= 0 or not regions:
        return {r.region_id: 0 for r in regions}
    base = min(cfg.n_min, max(0, budget // len(regions)))
    allocation = torch.full((len(regions),), base, dtype=torch.long)
    remaining = int(budget - allocation.sum().item())
    if remaining > 0:
        draws = torch.multinomial(
            priority_distribution(regions, cfg),
            num_samples=remaining,
            replacement=True,
            generator=generator,
        )
        allocation += torch.bincount(draws, minlength=len(regions))
    return {region.region_id: int(allocation[i].item()) for i, region in enumerate(regions)}
