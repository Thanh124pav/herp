"""Policy gradient signature with full actor parameters and shared adv scaling.

Per IMPLEMENTATION.md sections 7-9:
- signature covers the full actor plus logstd (never the critic),
- advantages are scaled by a shared running std, not per-batch mean/std normalized,
- loss is the unclipped score function ``-E[log pi(a|s) * A_scaled]``.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


def flatten_gradients(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    parts = [g.detach().flatten() for g in grads if g is not None]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def signature_parameters(policy: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Full actor + logstd parameters; explicitly excludes the critic."""

    picked: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def _add(param):
        if param is None or not param.requires_grad:
            return
        key = id(param)
        if key in seen:
            return
        seen.add(key)
        picked.append(param)

    for attr in ("actor", "actor_head"):
        module = getattr(policy, attr, None)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                _add(p)
    for attr in ("logstd", "log_std"):
        _add(getattr(policy, attr, None))
    if picked:
        return picked

    # Fallback: everything trainable that is not obviously in the critic.
    for name, param in policy.named_parameters():
        if param.requires_grad and "critic" not in name.lower() and "value" not in name.lower():
            _add(param)
    if not picked:
        picked = [p for p in policy.parameters() if p.requires_grad]
    return picked


def policy_gradient_signature(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    adv_scale: float | torch.Tensor | None = None,
    params: list[torch.nn.Parameter] | None = None,
) -> torch.Tensor:
    """Score-function gradient with shared advantage scaling (no per-batch mean subtract)."""

    params = params or signature_parameters(policy)
    dist = policy.get_distribution(obs)
    log_prob = dist.log_prob(actions)
    if log_prob.ndim > 1:
        log_prob = log_prob.sum(dim=-1)
    adv = advantages.detach().flatten().float()
    if adv_scale is None:
        scale = 1.0
    elif torch.is_tensor(adv_scale):
        scale = float(adv_scale)
    else:
        scale = float(adv_scale)
    adv = adv / (scale + 1e-8)
    loss = -(log_prob.flatten() * adv).mean()
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    return torch.cat(
        [(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(params, grads)]
    )


def empirical_fisher_diagonal(policy, obs, actions, max_samples: int = 128):
    """Mean squared per-sample score gradients over ``obs``/``actions``."""
    params = signature_parameters(policy)
    n = len(obs)
    idx = torch.linspace(0, n - 1, min(max_samples, n)).long()
    diagonal = torch.zeros(sum(p.numel() for p in params), device=obs.device)
    for i in idx:
        loss = policy.get_distribution(obs[i : i + 1]).log_prob(actions[i : i + 1]).sum()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        score = torch.cat(
            [(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(params, grads)]
        )
        diagonal += score.square()
    return diagonal / len(idx)
