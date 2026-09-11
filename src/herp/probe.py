"""Vectorized probe rollouts and σ estimation (IMPLEMENTATION.md §4.5/§4.8).

A probe batch is a set of (region, action_probe_index) pairs assigned to env
slots; all slots restore to their region snapshot at t=0 and then roll H steps.
The compressed future ``Ψ(τ⁺)`` is computed per slot and grouped by region for
the σ estimator.
"""
from __future__ import annotations

from typing import Any

import torch

from .archive import Region, Snapshot
from .sigma import branch_decomposition_sigma, discounted_future_feature, pairwise_sigma


def default_obs_tensor(obs, device="cpu"):
    if isinstance(obs, dict):
        return torch.cat([default_obs_tensor(v, device) for v in obs.values()])
    return torch.as_tensor(obs, device=device).float().flatten()


@torch.no_grad()
def sample_probe_assignment(
    candidates: list[Region],
    q: torch.Tensor,
    num_slots: int,
    probes_per_region: int,
    generator: torch.Generator,
) -> tuple[list[Region], list[Snapshot], list[int]]:
    """Return the region / snapshot / probe-index list of length ``num_slots``.

    Each entry is one env-slot's probe assignment. Slots are drawn from ``q``
    with replacement; ``probes_per_region`` action probes are grouped per
    region for the branch-decomposition σ.
    """
    if not candidates or num_slots <= 0:
        return [], [], []
    draws = torch.multinomial(q, num_samples=num_slots, replacement=True, generator=generator)
    regions, snaps, probe_idx = [], [], []
    counts: dict[int, int] = {}
    for d in draws.tolist():
        region = candidates[int(d)]
        regions.append(region)
        counts[region.region_id] = counts.get(region.region_id, 0) + 1
        probe_idx.append(counts[region.region_id] - 1)
        snaps.append(region.snapshots[0] if len(region.snapshots) == 1 else region.snapshots[
            int(torch.randint(len(region.snapshots), (1,), generator=generator).item())
        ])
    return regions, snaps, probe_idx


@torch.no_grad()
def probe_rollout(
    adapter,
    agent,
    env_ids: torch.Tensor,
    snapshots: list,
    probe_horizon: int,
    device: torch.device,
    probe_scale: float = 1.0,
    first_action_noise: torch.Tensor | None = None,
    gamma_branch: float = 0.95,
) -> tuple[torch.Tensor, int]:
    """Restore ``snapshots`` into ``env_ids`` and roll H steps with the policy.

    First action is ``μ + probe_scale·σ·ε`` where ε is provided per slot in
    ``first_action_noise`` (shape ``(len(env_ids), action_dim)``). Returns the
    per-slot compressed future ``Ψ(τ⁺)`` — one row per slot, plus the total
    number of environment transitions consumed.
    """
    obs = adapter.restore_state(env_ids, snapshots)
    n = obs.shape[0]
    feats: list[torch.Tensor] = []
    steps = 0
    for h in range(probe_horizon):
        dist = agent.get_distribution(obs.to(device))
        if h == 0 and first_action_noise is not None:
            action = dist.mean + dist.stddev * first_action_noise.to(device) * probe_scale
        else:
            action = dist.sample()
        action = action.clamp(adapter.action_low().to(action.device), adapter.action_high().to(action.device))
        obs, _rew, _term, _trunc, _info = adapter.step(action)
        feats.append(obs.detach().cpu())
        steps += n
    stacked = torch.stack(feats, dim=0)  # [H, N, D]
    # Discount + average over H per slot.
    weights = torch.tensor(
        [gamma_branch ** i for i in range(stacked.shape[0])],
        dtype=stacked.dtype,
    )
    weights = weights / weights.sum().clamp_min(1e-8)
    z = (stacked * weights[:, None, None]).sum(dim=0)  # [N, D]
    return z, steps


def sigma_from_features(
    z: torch.Tensor,
    region_of_slot: list[int],
    probe_of_slot: list[int],
    lambda_dyn: float = 0.0,
    estimator: str = "pairwise",
) -> dict[int, float]:
    """Group per-slot future features by region and compute σ per region.

    ``region_of_slot[i]`` is the region_id for slot ``i``; ``probe_of_slot[i]``
    is the action-probe index inside that region.
    """
    by_region: dict[int, list[torch.Tensor]] = {}
    for slot_idx, rid in enumerate(region_of_slot):
        by_region.setdefault(rid, []).append(z[slot_idx])
    out: dict[int, float] = {}
    for rid, feats in by_region.items():
        if len(feats) < 2:
            out[rid] = 0.0
            continue
        stack = torch.stack(feats, dim=0).float()
        if estimator == "branch":
            # Shape [A, 1, d] as a degenerate branch decomposition when
            # env repeats are not run — falls back to pairwise for A>=2.
            total, _b, _d = branch_decomposition_sigma(stack.unsqueeze(1), lambda_dyn)
            out[rid] = float(total)
        else:
            out[rid] = float(pairwise_sigma(stack))
    return out
