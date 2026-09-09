from __future__ import annotations
import numpy as np
import torch
from .probe import default_obs_tensor


@torch.no_grad()
def evaluate_policy(env, policy, episodes=50, device='cpu', seed_base=2_000_000):
    returns, successes, finals = [], [], []
    steps = 0
    # Isolate policy/training RNG from evaluation and reset-seed side effects.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed_base + episode)
            done, success, ep_return = False, False, 0.
            while not done:
                action = policy.act(default_obs_tensor(obs, device)[None], deterministic=True).squeeze(0)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)
                ep_return += float(torch.as_tensor(reward).mean())
                final = bool(torch.as_tensor(info.get('success', False)).any())
                success |= final
                steps += 1
            returns.append(ep_return)
            successes.append(float(success))
            finals.append(float(final))
    return dict(eval_return=float(np.mean(returns)), eval_success=float(np.mean(successes)),
                eval_success_final=float(np.mean(finals)), eval_steps=steps,
                eval_episodes=episodes)
