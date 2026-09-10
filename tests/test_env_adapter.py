"""Snapshot-restore precision tests (IMPLEMENTATION.md §5.2).

For each benchmark adapter, reset, step randomly, snapshot, drift, restore,
and check obs L∞ error vs. the pre-drift observation.

- ManiSkill (state obs): ≤ 1e-6 tolerance. IMPLEMENTATION.md §5.2 asks for
  1e-7, but ManiSkill 3 state observations are float32 (~1 ulp ≈ 1.2e-7 on
  values near 1), so 1e-7 is not reachable through a scatter+forward round
  trip. 1e-6 empirically covers all four §7 ManiSkill tasks.
- Meta-World, Fetch (MuJoCo): ≤ 1e-6 tolerance.
- ``elapsed_steps`` after restore must equal the count at snapshot time.

Adapters that cannot be imported on the test host (no CUDA, no metaworld …)
are skipped rather than failed — CI on a CPU-only box still gets useful signal
from the Meta-World and Fetch cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _random_step(adapter, rng):
    low = adapter.action_low().cpu().numpy()
    high = adapter.action_high().cpu().numpy()
    a = rng.uniform(low, high, size=(adapter.num_envs, adapter.action_dim)).astype(np.float32)
    return torch.as_tensor(a, device=adapter.device)


def _restore_precision(adapter, tol: float):
    rng = np.random.default_rng(0)
    obs, _ = adapter.reset(seed=0)
    for _ in range(int(rng.integers(0, 20))):
        obs, *_ = adapter.step(_random_step(adapter, rng))
    env_ids = torch.arange(adapter.num_envs, dtype=torch.long)
    snapshots = adapter.save_state(env_ids)
    baseline_obs = obs.detach().cpu().clone()
    baseline_elapsed = adapter.elapsed_steps().detach().cpu().clone()
    for _ in range(int(rng.integers(0, 20))):
        obs, *_ = adapter.step(_random_step(adapter, rng))
    restored = adapter.restore_state(env_ids, snapshots).detach().cpu()
    err = (restored - baseline_obs).abs().amax().item()
    assert err <= tol, f"obs L∞ = {err:.2e} exceeds tolerance {tol:.0e}"
    elapsed_after = adapter.elapsed_steps().detach().cpu()
    assert torch.equal(elapsed_after, baseline_elapsed), (
        f"elapsed_steps mismatch: {elapsed_after.tolist()} vs {baseline_elapsed.tolist()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ManiSkill GPU sim requires CUDA")
def test_maniskill_snapshot_restore_precision():
    from herp.envs.maniskill import ManiSkillAdapter

    try:
        adapter = ManiSkillAdapter(
            "PickCube-v1", sim_backend="physx_cuda", device="cuda"
        ).make(num_envs=4, seed=0)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"ManiSkill unavailable: {exc}")
    try:
        _restore_precision(adapter, tol=1e-6)
    finally:
        adapter.close()


def test_metaworld_snapshot_restore_precision():
    try:
        import metaworld  # noqa: F401
    except Exception:
        pytest.skip("metaworld not installed")
    from herp.envs.metaworld import MetaWorldAdapter

    adapter = MetaWorldAdapter("reach-v3", device="cpu")
    try:
        adapter.make(num_envs=2, seed=0)
    except Exception as exc:
        pytest.skip(f"metaworld env unavailable: {exc}")
    try:
        _restore_precision(adapter, tol=1e-6)
    finally:
        adapter.close()


def test_fetch_snapshot_restore_precision():
    try:
        import gymnasium_robotics  # noqa: F401
    except Exception:
        pytest.skip("gymnasium_robotics not installed")
    from herp.envs.fetch import FetchAdapter

    try:
        adapter = FetchAdapter("FetchPush-v4", device="cpu").make(num_envs=2, seed=0)
    except Exception as exc:
        pytest.skip(f"fetch env unavailable: {exc}")
    try:
        _restore_precision(adapter, tol=1e-6)
    finally:
        adapter.close()
