from __future__ import annotations

import torch


def pairwise_sigma(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """sqrt(0.5 * mean pairwise squared distance), matching THEORY.md."""

    z = features.float()
    if z.ndim != 2:
        raise ValueError(f"expected [K, d] features, got shape {tuple(z.shape)}")
    k = z.shape[0]
    if k < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    diff = z[:, None, :] - z[None, :, :]
    d2 = diff.pow(2).sum(dim=-1)
    mask = ~torch.eye(k, dtype=torch.bool, device=z.device)
    return torch.sqrt(0.5 * d2[mask].mean() + eps)


def branch_decomposition_sigma(
    features: torch.Tensor,
    lambda_dyn: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, controllable-branch, and dynamics-noise sigma values.

    ``features`` is shaped [num_action_probes, num_env_repeats, dim].
    """

    z = features.float()
    if z.ndim != 3:
        raise ValueError(f"expected [A, M, d] features, got shape {tuple(z.shape)}")
    mean_per_action = z.mean(dim=1)
    global_mean = mean_per_action.mean(dim=0)
    branch2 = (mean_per_action - global_mean).pow(2).sum(dim=-1).mean()
    dyn2 = (z - mean_per_action[:, None, :]).pow(2).sum(dim=-1).mean()
    total = torch.sqrt(branch2 + float(lambda_dyn) * dyn2 + eps)
    return total, torch.sqrt(branch2 + eps), torch.sqrt(dyn2 + eps)


def discounted_future_feature(
    future_features: torch.Tensor,
    gamma_branch: float = 0.95,
) -> torch.Tensor:
    """Discounted mean of phi(s_1), ..., phi(s_H)."""

    if future_features.ndim != 2:
        raise ValueError("future_features must be [H, d]")
    h = future_features.shape[0]
    weights = torch.as_tensor(
        [gamma_branch**i for i in range(h)],
        dtype=future_features.dtype,
        device=future_features.device,
    )
    weights = weights / weights.sum().clamp_min(1e-8)
    return (future_features * weights[:, None]).sum(dim=0)

# HERP v3: fixed-window estimators. Earlier functions are legacy-v2 ablations.
def fixed_window_distance(x, y, min_common_steps=8):
    common = x.valid_mask & y.valid_mask
    if int(common.sum()) < min_common_steps:
        return None
    return (x.traj_features[common]-y.traj_features[common]).square().sum(-1).mean()


def direct_q_estimate(fragments, min_common_steps=8):
    if len(fragments)>=2 and all(bool(f.valid_mask.all()) and len(f.valid_mask)>=min_common_steps for f in fragments):
        features=torch.stack([f.traj_features for f in fragments])
        return full_window_q(features),len(fragments)*(len(fragments)-1)//2
    pairs = []
    for i,x in enumerate(fragments):
        for y in fragments[i+1:]:
            d = fixed_window_distance(x,y,min_common_steps)
            if d is not None:
                pairs.append(d)
    if not pairs:
        return float('nan'), 0
    return float(.5*torch.stack(pairs).mean()),len(pairs)


def full_window_q(features):
    """O(K M D) U-statistic, equivalent to all pairs for complete windows."""
    if len(features)<2:
        return float('nan')
    return float(features.double().var(dim=0,unbiased=True).sum(-1).mean())


def root_total_variance(weights, means, q):
    """Means must be flattened fixed-M representations divided by sqrt(M)."""
    if not weights:
        return 0.
    if set(weights)-set(means):
        return float('nan')
    mu = sum(weights[k]*means[k] for k in weights)
    return sum(weights[k]*(q[k]+float((means[k]-mu).square().sum())) for k in weights)
