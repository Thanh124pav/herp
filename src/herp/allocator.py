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

# HERP v3 allocator: no quota / mixture heuristic. p and sigma are each
# normalized to the same [0,1] scale before multiplication (see cfg.score_normalize).
def _normalize(values, mode, floor):
    import torch
    v = torch.as_tensor(values, dtype=torch.float64)
    if mode == 'none' or v.numel() < 2:
        return v
    if mode == 'rank':
        # Average-rank in [0,1]; ties share; constant vector -> 0.5.
        less = (v[:, None] > v[None, :]).sum(1).double()
        equal = (v[:, None] == v[None, :]).sum(1).double()
        return (less + (equal - 1) / 2) / (v.numel() - 1)
    if mode == 'zscore':
        mu, sd = v.mean(), v.std(unbiased=False).clamp_min(1e-8)
        return torch.sigmoid((v - mu) / sd).clamp_min(floor)
    raise ValueError(f'unknown score_normalize mode {mode!r}')


def v3_priority_distribution(regions, cfg, mode='herp'):
    import torch
    p_raw = torch.tensor([max(0., r.p_ema) for r in regions], dtype=torch.float64)
    sigma_raw = torch.tensor([max(cfg.sigma_floor, r.sigma_raw) for r in regions], dtype=torch.float64)
    if mode == 'plr_region':
        s_raw = torch.tensor([max(0., r.td_error_ema) for r in regions], dtype=torch.float64)
    elif mode == 'sacl_style':
        s_raw = torch.tensor([max(0., r.value_change) for r in regions], dtype=torch.float64)
    else:
        s_raw = None
    p_norm = _normalize(p_raw + cfg.relevance_floor, cfg.score_normalize, cfg.relevance_floor)
    sigma_norm = _normalize(sigma_raw, cfg.score_normalize, cfg.sigma_floor)
    p_norm = (p_norm + cfg.relevance_floor).pow(cfg.relevance_alpha)
    if mode in ('uniform', 'state_radius_uniform'):
        scores = torch.ones_like(p_norm)
    elif mode == 'herp_sigma':
        scores = sigma_norm
    elif mode == 'herp_p':
        scores = p_norm
    elif mode == 'plr_region':
        scores = _normalize(s_raw, cfg.score_normalize, cfg.relevance_floor) + cfg.relevance_floor
    elif mode == 'sacl_style':
        scores = _normalize(s_raw, cfg.score_normalize, cfg.relevance_floor) + cfg.relevance_floor
    else:
        scores = p_norm * sigma_norm
    if cfg.score_temperature != 1.0:
        scores = scores.clamp_min(1e-12) ** float(cfg.score_temperature)
    if not len(scores) or not torch.isfinite(scores).all() or (scores < 0).any() or scores.sum() <= 0:
        raise ValueError('Allocation requires finite positive scores')
    return scores / scores.sum()


def allocate_fragments(probabilities, num_fragments, generator=None):
    import torch
    ids = torch.multinomial(probabilities,num_fragments,replacement=True,generator=generator)
    return torch.bincount(ids,minlength=len(probabilities))
