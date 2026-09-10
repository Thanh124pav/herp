"""Reference batch collection (IMPLEMENTATION.md §4.9).

A separate small vectorized env, stepped by the *current* policy in deterministic
mode. Its trajectories feed the p_v cosine/dot/fisher/hybrid estimators. It does
NOT update policy weights and does NOT share buffers with the main rollout, but
its transitions ARE counted against the interaction budget (§4.6).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReferenceBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    steps: int


def collect_reference(
    adapter,
    agent,
    horizon: int,
    device: torch.device,
    adv_scale: float | None = None,
    seed: int | None = None,
) -> ReferenceBatch:
    """Roll out ``horizon`` steps in the reference vectorized env.

    Deterministic action, per-slot advantages via 1-step TD (a reference batch
    is not meant to be a long GAE; only its (obs, action, adv) shape matters).
    """
    obs, _ = adapter.reset(seed=seed)
    n = adapter.num_envs
    obs_buf, act_buf, rew_buf, val_buf = [], [], [], []
    for t in range(horizon):
        with torch.no_grad():
            action, _logp, _ent, value = agent.get_action_and_value(obs.to(device), deterministic=True)
        applied = action.clamp(adapter.action_low(), adapter.action_high())
        next_obs, reward, term, trunc, _info = adapter.step(applied)
        obs_buf.append(obs.detach().cpu())
        act_buf.append(action.detach().cpu())
        rew_buf.append(reward.detach().cpu())
        val_buf.append(value.detach().cpu().reshape(-1))
        obs = next_obs
    with torch.no_grad():
        last_value = agent.get_value(obs.to(device)).detach().cpu().reshape(-1)
    obs_t = torch.stack(obs_buf, dim=0)  # [T, N, D]
    act_t = torch.stack(act_buf, dim=0)  # [T, N, A]
    rew_t = torch.stack(rew_buf, dim=0)  # [T, N]
    val_t = torch.stack(val_buf, dim=0)  # [T, N]
    # 1-step TD residual as a proxy advantage (unclipped, unnormalized).
    next_values = torch.cat([val_t[1:], last_value[None, :]], dim=0)
    adv = rew_t + 0.99 * next_values - val_t
    if adv_scale:
        adv = adv / max(float(adv_scale), 1e-6)
    return ReferenceBatch(
        obs=obs_t.reshape(-1, obs_t.shape[-1]),
        actions=act_t.reshape(-1, act_t.shape[-1]),
        advantages=adv.reshape(-1),
        rewards=rew_t.reshape(-1),
        steps=int(horizon * n),
    )
