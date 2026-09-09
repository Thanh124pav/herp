"""Restored-state action probes and future-trajectory sigma estimation."""
from __future__ import annotations

from collections.abc import Callable

import torch

from .archive import Snapshot
from .regions import OnlineRegionizer
from .sigma import branch_decomposition_sigma, discounted_future_feature, pairwise_sigma


def default_obs_tensor(obs, device="cpu"):
    """Fallback obs flatten when no adapter is provided (tests only)."""
    if isinstance(obs, dict):
        return torch.cat([default_obs_tensor(v, device) for v in obs.values()])
    return torch.as_tensor(obs, device=device).float().flatten()


def elapsed_steps(env, adapter=None):
    if adapter is not None:
        return adapter.elapsed_steps()
    target = getattr(env, "unwrapped", env)
    return int(torch.as_tensor(getattr(target, "elapsed_steps", 0)).flatten()[0])


def reset_to_snapshot(env, snapshot: Snapshot, adapter=None):
    """Restore an archived state. ``adapter`` is preferred; env-level fallback for tests."""
    if adapter is not None:
        return adapter.restore_state(snapshot)
    obs, info = env.reset(options={"reset_to_env_states": {"env_states": snapshot.env_state}})
    target = getattr(env, "unwrapped", env)
    controller = snapshot.env_state.get("controller") if isinstance(snapshot.env_state, dict) else None
    if controller and hasattr(target, "agent"):
        target.agent.set_controller_state(controller)
    if hasattr(target, "_elapsed_steps"):
        target._elapsed_steps[:] = snapshot.elapsed_steps
    wrapper = env
    while wrapper is not target:
        if "_elapsed_steps" in vars(wrapper):
            wrapper._elapsed_steps = snapshot.elapsed_steps
        wrapper = wrapper.env
    if hasattr(target, "get_obs"):
        info = target.get_info()
        obs = target.get_obs(info)
    return obs, info


def _to_tensor(adapter, obs, device):
    if adapter is not None:
        return adapter.obs_tensor(obs, device)
    return default_obs_tensor(obs, device)


@torch.no_grad()
def probe_region(
    env,
    snapshot,
    policy,
    regionizer,
    num_action_probes: int = 4,
    num_env_repeats: int = 1,
    probe_horizon: int = 8,
    probe_scale: float = 1.0,
    gamma_branch: float = 0.95,
    lambda_dyn: float = 0.0,
    obs_to_tensor: Callable | None = None,
    device: str = "cpu",
    estimator: str = "pairwise",
    gamma: float = 0.99,
    adapter=None,
):
    """Estimate branching sigma from short probe rollouts after a snapshot restore.

    Env randomness is intentionally not rewound between repeats. For ``branch``
    estimation, continuation policy noise is shared across action groups so that
    between-group variance isolates the first-action perturbation.
    """
    if min(num_action_probes, num_env_repeats, probe_horizon) < 1:
        raise ValueError("Probe dimensions must be positive")
    if obs_to_tensor is None:
        obs_to_tensor = lambda obs, device="cpu": _to_tensor(adapter, obs, device)  # noqa: E731

    futures, returns, trajectories = [], [], []
    steps = 0
    action_dim = policy.get_distribution(snapshot.obs.to(device)[None]).mean.shape[-1]
    common_noise = torch.randn(probe_horizon, action_dim, device=device)
    first_noise = torch.randn(num_action_probes, action_dim, device=device)
    for a in range(num_action_probes):
        action_features, action_returns = [], []
        noise = common_noise if estimator == "branch" else torch.randn_like(common_noise)
        for m in range(num_env_repeats):
            obs, _ = reset_to_snapshot(env, snapshot, adapter=adapter)
            feats, path = [], [obs_to_tensor(obs).cpu()]
            total_return = 0.0
            terminated = False
            for h in range(probe_horizon):
                obs_t = obs_to_tensor(obs, device)
                dist = policy.get_distribution(obs_t[None])
                eps = first_noise[a] * probe_scale if h == 0 else noise[h]
                action = (dist.mean + dist.stddev * eps).squeeze(0)
                low = torch.as_tensor(env.action_space.low, device=device)
                high = torch.as_tensor(env.action_space.high, device=device)
                obs, reward, terminated, truncated, _ = env.step(action.clamp(low, high).cpu().numpy())
                total_return += gamma ** h * float(torch.as_tensor(reward).mean())
                steps += 1
                obs_t = obs_to_tensor(obs, device)
                feats.append(regionizer.phi(obs_t))
                path.append(obs_t.cpu())
                if bool(terminated) or bool(truncated):
                    break
            if not bool(terminated):
                total_return += gamma ** len(feats) * float(policy.value(obs_t[None]))
            action_features.append(discounted_future_feature(torch.stack(feats), gamma_branch))
            action_returns.append(total_return)
            trajectories.append(torch.stack(path))
        futures.append(torch.stack(action_features))
        returns.append(action_returns)
    z = torch.stack(futures)
    branch_total, branch, dyn = branch_decomposition_sigma(z, lambda_dyn)
    sigma = branch_total if estimator == "branch" else pairwise_sigma(z.flatten(0, 1))
    if estimator == "return":
        sigma = (
            torch.tensor(returns).flatten().std(unbiased=True)
            if num_action_probes * num_env_repeats > 1
            else torch.tensor(0.0)
        )
    return dict(
        features=z.cpu(),
        sigma=float(sigma),
        sigma_branch=float(branch),
        sigma_dyn=float(dyn),
        returns=torch.tensor(returns),
        steps=steps,
        trajectories=trajectories,
    )
