"""Deterministic policy evaluation with fixed seeds; RNG isolated from training."""
from __future__ import annotations

import numpy as np
import torch

from .probe import default_obs_tensor


@torch.no_grad()
def evaluate_policy(env, policy, episodes: int = 50, device: str = "cpu",
                    seed_base: int = 2_000_000, adapter=None):
    returns, successes, finals = [], [], []
    steps = 0
    to_tensor = adapter.obs_tensor if adapter is not None else default_obs_tensor
    success_from_info = adapter.success_from_info if adapter is not None else (
        lambda info: bool(torch.as_tensor(info.get("success", False)).any())
    )
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed_base + episode)
            done, success, ep_return = False, False, 0.0
            info = {}
            while not done:
                action = policy.act(to_tensor(obs, device)[None], deterministic=True).squeeze(0)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)
                ep_return += float(torch.as_tensor(reward).mean())
                success |= success_from_info(info)
                steps += 1
            final = success_from_info(info)
            returns.append(ep_return)
            successes.append(float(success))
            finals.append(float(final))
    return dict(
        eval_return=float(np.mean(returns)),
        eval_success=float(np.mean(successes)),
        eval_success_final=float(np.mean(finals)),
        eval_steps=steps,
        eval_episodes=episodes,
    )
