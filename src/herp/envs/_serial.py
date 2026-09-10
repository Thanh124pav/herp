"""Serial-vector helpers shared by Meta-World and Fetch adapters.

MuJoCo envs are step-serial and the paper does not need process-parallelism on
those benchmarks (the manual, §4.3/§4.4, calls this explicitly). This module
holds the common list-of-envs stepping and snapshot round-trip logic so both
adapters stay small.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch


def stack_np(items):
    return np.stack([np.asarray(x) for x in items], axis=0)


class SerialVectorState:
    """State bag for a list-of-envs serial vectorizer.

    The adapter owns instantiation; this class only stores the envs, their
    per-env elapsed-step counters, and last observation, so ``step`` can update
    them uniformly across benchmarks.
    """

    __slots__ = ("envs", "elapsed", "last_obs", "device")

    def __init__(self, envs: list, device: torch.device):
        self.envs = envs
        self.device = device
        self.elapsed = torch.zeros(len(envs), dtype=torch.long, device=device)
        self.last_obs: list = [None] * len(envs)

    def __len__(self) -> int:
        return len(self.envs)


def serial_reset(
    state: SerialVectorState, obs_fn: Callable, reset_kwargs_fn: Callable | None = None
):
    """Reset every env; return stacked obs and merged info dict."""
    obs_list, info_list = [], []
    for i, env in enumerate(state.envs):
        kwargs = reset_kwargs_fn(i) if reset_kwargs_fn is not None else {}
        obs, info = env.reset(**kwargs)
        state.last_obs[i] = obs
        obs_list.append(obs_fn(obs))
        info_list.append(info if isinstance(info, dict) else {})
    state.elapsed = torch.zeros(len(state), dtype=torch.long, device=state.device)
    return np.stack(obs_list, axis=0), {"per_env": info_list}


def serial_step(
    state: SerialVectorState, actions: torch.Tensor, obs_fn: Callable, max_ep: int
):
    """Step every env with a per-env action row; auto-reset on episode end."""
    acts = actions.detach().cpu().numpy()
    obs_list, rew, term, trunc, info_list = [], [], [], [], []
    for i, env in enumerate(state.envs):
        obs, reward, terminated, truncated, info = env.step(acts[i])
        state.elapsed[i] += 1
        # Meta-World does not always emit truncation; enforce it here.
        if int(state.elapsed[i]) >= max_ep:
            truncated = True
        state.last_obs[i] = obs
        obs_list.append(obs_fn(obs))
        rew.append(float(reward))
        term.append(bool(terminated))
        trunc.append(bool(truncated))
        info_list.append(info if isinstance(info, dict) else {})
        if terminated or truncated:
            state.last_obs[i], _ = env.reset()
            obs_list[-1] = obs_fn(state.last_obs[i])
            state.elapsed[i] = 0
    return (
        np.stack(obs_list, axis=0),
        np.array(rew, dtype=np.float32),
        np.array(term, dtype=bool),
        np.array(trunc, dtype=bool),
        {"per_env": info_list},
    )


def collect_success(info: dict, key: str = "success") -> np.ndarray:
    per = info.get("per_env", [])
    return np.array([bool(x.get(key, False)) for x in per], dtype=bool)


def close_all(state: SerialVectorState) -> None:
    for env in state.envs:
        try:
            env.close()
        except Exception:
            pass
