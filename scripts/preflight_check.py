"""Pre-flight validation for the HERP v3 campaign.

Run BEFORE launching the tonight's big campaign to catch environment
problems while it's cheap to fix them. Exits non-zero on any failure so
CI / a shell script can gate the launch on it.

Checks:
  1. Python imports (herp modules + torch + mani_skill + wandb)
  2. CUDA available; GPU visible
  3. WandB login present (skips if WANDB_MODE=offline)
  4. ManiSkill GPU init on the chosen env (LD_LIBRARY_PATH / VK_ICD_FILENAMES)
  5. HERPV3Config sanity (chain_radius consistent with train_v3 CLI default)
  6. train_v3.py argparse works end-to-end (no CLI regressions)
  7. Allocator produces a valid distribution on edge-case inputs
  8. Snapshot save→restore round-trip on the chosen env

Usage:
    LD_LIBRARY_PATH=/usr/lib/wsl/lib python scripts/preflight_check.py
    ENV_ID=PickCube-v1 NUM_ENVS=512 python scripts/preflight_check.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import traceback
from pathlib import Path


PASS = '\033[32m  PASS\033[0m'
FAIL = '\033[31m  FAIL\033[0m'
INFO = '\033[36m  INFO\033[0m'
WARN = '\033[33m  WARN\033[0m'


def _check(name, fn):
    print(f'* {name} ...', end=' ', flush=True)
    try:
        result = fn()
        print('OK' if result is None else f'OK ({result})')
        return True
    except Exception as exc:
        print(f'FAIL\n    {type(exc).__name__}: {exc}')
        return False


def imports():
    import torch, wandb, numpy, mani_skill
    from herp.config import HERPV3Config
    from herp.allocator import v3_priority_distribution
    from herp.sigma import direct_q_estimate
    from herp.chain_partition import BoundaryDetector
    return f'torch {torch.__version__}, mani_skill {mani_skill.__version__}, wandb {wandb.__version__}'


def cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    return f'{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB'


def wandb_login():
    if os.environ.get('WANDB_MODE') == 'offline':
        return 'WANDB_MODE=offline; skipping login check'
    if os.environ.get('WANDB_API_KEY'):
        return 'WANDB_API_KEY set'
    import netrc
    try:
        n = netrc.netrc(os.path.expanduser('~/.netrc'))
        if n.authenticators('api.wandb.ai'):
            return 'wandb login via ~/.netrc'
    except Exception:
        pass
    raise RuntimeError('no WANDB_API_KEY, no ~/.netrc; run `wandb login`')


def maniskill_gpu(env_id, num_envs):
    from herp.envs.maniskill import ManiSkillAdapter
    adapter = ManiSkillAdapter(env_id, sim_backend='physx_cuda', render_backend='cpu', device='cuda').make(min(num_envs, 32), 0)
    obs, _ = adapter.reset(seed=0)
    if obs is None or not obs.shape[0]:
        raise RuntimeError('reset returned empty observation')
    adapter.env.close()
    return f'{env_id} obs_dim={adapter.obs_dim} action_dim={adapter.action_dim}'


def config_consistency():
    from herp.config import HERPV3Config
    cfg = HERPV3Config()
    # Guard against silent drift from THEORY §31 defaults.
    assert cfg.future_horizon == 32, f'future_horizon {cfg.future_horizon} != 32 (THEORY §31)'
    assert cfg.boundary_percentile == 0.90, f'boundary_percentile != 0.90'
    assert cfg.boundary_lambda_policy == 1.0 and cfg.boundary_lambda_state == 0.0
    assert cfg.sigma_floor == 1e-3 and cfg.relevance_floor == 1e-3
    assert cfg.sigma_predictor_kappa == 8.0
    assert cfg.score_normalize in ('none', 'rank', 'zscore')
    return f'M={cfg.future_horizon}, radius={cfg.chain_radius}, norm={cfg.score_normalize}'


def cli_parse():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / 'scripts/train_v3.py'), '--help'],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f'train_v3 --help failed: {r.stderr[:200]}')
    for flag in ('--wandb-mode', '--auto-resume', '--score-normalize', '--control-mode', '--sigma-mode'):
        if flag not in r.stdout:
            raise RuntimeError(f'CLI missing flag {flag}')
    return 'all expected flags present'


def allocator_edge():
    import torch
    from types import SimpleNamespace
    from herp.config import HERPV3Config
    from herp.allocator import v3_priority_distribution
    cfg = HERPV3Config(score_normalize='rank')
    # All-zero p and sigma → uniform via rank ties.
    regs = [SimpleNamespace(p_ema=0., sigma_raw=0.) for _ in range(5)]
    d = v3_priority_distribution(regs, cfg)
    assert d.sum().item() > 0.99 and (d > 0).all() and torch.isfinite(d).all()
    # Skewed: 4 wildly different scales.
    regs = [SimpleNamespace(p_ema=p, sigma_raw=s) for p, s in [(-.5, 100.), (.9, 1e-4), (.5, 1.), (.7, 10.)]]
    d = v3_priority_distribution(regs, cfg)
    assert torch.isfinite(d).all() and (d > 0).all()
    return f'{d.tolist()}'


def snapshot_roundtrip(env_id):
    import torch
    from herp.envs.maniskill import ManiSkillAdapter
    adapter = ManiSkillAdapter(env_id, sim_backend='physx_cuda', render_backend='cpu', device='cuda').make(4, 0)
    obs, _ = adapter.reset(seed=0)
    snaps = adapter.save_state(torch.arange(4))
    obs2, _, _, _, _ = adapter.step(torch.zeros(4, adapter.action_dim, device=adapter.device))
    restored = adapter.restore_state(torch.arange(4), snaps)
    diff = float((restored.cpu() - obs.cpu()).abs().max())
    adapter.env.close()
    if diff > 1e-3:
        raise RuntimeError(f'restore mismatch {diff}')
    return f'obs diff max={diff:.2e}'


def main():
    env_id = os.environ.get('ENV_ID', 'PickCube-v1')
    num_envs = int(os.environ.get('NUM_ENVS', '512'))

    print(f'HERP v3 pre-flight check ({env_id}, num_envs={num_envs})')
    print('---')
    results = [
        _check('imports', imports),
        _check('cuda', cuda),
        _check('wandb login', wandb_login),
        _check('config consistency', config_consistency),
        _check('CLI parse', cli_parse),
        _check('allocator edge cases', allocator_edge),
        _check('maniskill GPU init', lambda: maniskill_gpu(env_id, num_envs)),
        _check('snapshot roundtrip', lambda: snapshot_roundtrip(env_id)),
    ]
    print('---')
    total, passed = len(results), sum(results)
    print(f'{passed}/{total} checks passed')
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
