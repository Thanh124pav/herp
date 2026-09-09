from __future__ import annotations

from collections.abc import Iterable

import torch


def flatten_gradients(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    parts = [g.detach().flatten() for g in grads if g is not None]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def signature_parameters(policy: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Prefer actor-head-like parameters, falling back to all trainable params."""

    named = list(policy.named_parameters())
    preferred_tokens = ("actor_head", "action_head", "mean", "logstd", "fc_mean")
    picked = [
        p for name, p in named
        if p.requires_grad and any(token in name.lower() for token in preferred_tokens)
    ]
    if picked:
        return picked
    return [p for _, p in named if p.requires_grad]


def policy_gradient_signature(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    params: list[torch.nn.Parameter] | None = None,
) -> torch.Tensor:
    """Unclipped PPO score-function signature: -E[log pi(a|s) A]."""

    params = params or signature_parameters(policy)
    dist = policy.get_distribution(obs)
    log_prob = dist.log_prob(actions)
    if log_prob.ndim > 1:
        log_prob = log_prob.sum(dim=-1)
    adv = advantages.detach().flatten()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8) if adv.numel() > 1 else adv
    loss = -(log_prob.flatten() * adv).mean()
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(params, grads)])


def empirical_fisher_diagonal(policy, obs, actions, max_samples=128):
    """Mean squared per-transition score gradients, without advantage weighting."""
    params = signature_parameters(policy)
    idx = torch.linspace(0, len(obs) - 1, min(max_samples, len(obs))).long()
    diagonal = torch.zeros(sum(p.numel() for p in params), device=obs.device)
    for i in idx:
        loss = policy.get_distribution(obs[i:i+1]).log_prob(actions[i:i+1]).sum()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        score = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(params, grads)])
        diagonal += score.square()
    return diagonal / len(idx)
