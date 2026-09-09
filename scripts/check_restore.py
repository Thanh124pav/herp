"""Standardised adapter restore test (IMPLEMENTATION.md section 29).

For a benchmark family, run:
    (1) snapshot identity
    (2) same-action next-state replay
    (3) reward equality
    (4) termination / truncation equality
    (5) episode-clock restoration
and emit a JSON report. A benchmark is paper-ready only after this passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from herp.archive import Snapshot
from herp.envs import make_adapter


def build_snapshot(adapter, obs, elapsed):
    return Snapshot(
        env_state=adapter.save_state(),
        obs=adapter.obs_tensor(obs),
        timestep=elapsed,
        episode_id=0,
        return_so_far=0.0,
        elapsed_steps=elapsed,
    )


def run(benchmark: str, env_id: str, seed: int, warmup: int, output: Path | None):
    kwargs = dict(env_id=env_id)
    if benchmark == "maniskill":
        kwargs.update(
            control_mode="pd_ee_delta_pose",
            obs_mode="state",
            reward_mode="dense",
            sim_backend="physx_cpu",
            render_backend="cpu",
        )
    adapter = make_adapter(benchmark, **kwargs).make()
    env = adapter.env
    obs, _ = env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(warmup):
        obs, *_ = env.step(action)
    snapshot = build_snapshot(adapter, obs, adapter.elapsed_steps())
    restored, _ = adapter.restore_state(snapshot)
    obs_error = float((adapter.obs_tensor(restored) - snapshot.obs).abs().max())
    clock = adapter.elapsed_steps()
    first_obs, first_reward, first_term, first_trunc, _ = env.step(action)
    adapter.restore_state(snapshot)
    second_obs, second_reward, second_term, second_trunc, _ = env.step(action)
    next_error = float((adapter.obs_tensor(first_obs) - adapter.obs_tensor(second_obs)).abs().max())
    reward_error = float(abs(float(np.asarray(first_reward).mean()) - float(np.asarray(second_reward).mean())))
    result = dict(
        benchmark=benchmark,
        env_id=env_id,
        observation_max_error=obs_error,
        next_observation_max_error=next_error,
        reward_error=reward_error,
        terminated_match=bool(first_term) == bool(second_term),
        truncated_match=bool(first_trunc) == bool(second_trunc),
        restored_clock=int(clock),
        warmup=warmup,
        seed=seed,
    )
    ok = (
        obs_error < 1e-4
        and next_error < 1e-4
        and reward_error < 1e-4
        and result["terminated_match"]
        and result["truncated_match"]
        and clock == warmup
    )
    result["passed"] = bool(ok)
    print(json.dumps(result, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    adapter.close()
    if not ok:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("maniskill", "metaworld", "fetch"))
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    opt = parser.parse_args()
    torch.set_num_threads(1)
    run(opt.benchmark, opt.env_id, opt.seed, opt.warmup, opt.output)


if __name__ == "__main__":
    main()
