"""HERP v3 — 1M-step campaign orchestrator (server-ready, resumable).

Order of operations, mirroring the user's request:

  1. Sigma-mechanism ablations (fast, small budget) to validate the estimator.
  2. HERP main + Tier-A same-backbone baselines at 1M steps.
  3. Adapted close baselines (PLR-Region, SACL-style).

Each individual run streams to W&B (if enabled) and can be safely re-launched.
Restartability rules:

  * A run with ``summary.json`` present is treated as completed and skipped.
  * Otherwise, ``--auto-resume`` picks the newest ``checkpoint_*.pt`` inside
    ``output_dir`` and continues.
  * The orchestrator writes ``manifest.json`` after every run and rewrites the
    same file on resume, so a fresh invocation picks up where the last one
    stopped.

Environment prerequisites (see ``scripts/server_run.sh``):
  * conda env ``deeplearning`` active with wandb/torch/mani-skill installed
  * ``LD_LIBRARY_PATH=/usr/lib/wsl/lib`` for PhysX GPU on WSL (harmless elsewhere)
  * ``WANDB_API_KEY`` exported for online mode (offline needs no key)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# --- Phase definitions ------------------------------------------------------

# Sigma ablations run at a small budget (default 200k) to validate the
# estimator before we spend a full 1M-step run on any main method. Each entry
# is (nickname, method, extra_cli_kwargs).
SIGMA_ABLATIONS = [
    ('sigma_direct_M8',    'herp', dict(sigma_mode='direct',     future_horizon=8)),
    ('sigma_direct_M32',   'herp', dict(sigma_mode='direct',     future_horizon=32)),
    ('sigma_predictor_M32','herp', dict(sigma_mode='predictor',  future_horizon=32)),
    ('sigma_shrinkage_M32','herp', dict(sigma_mode='shrinkage',  future_horizon=32)),
]

# Tier-A main runs at 1M steps. HERP first (user's stated priority), then the
# closest exploration / revisit baselines. External SAC-based baselines
# (RFCL/ActiveRL/BRO/MaxInfoRL) are launched via their own third_party runners
# — this orchestrator only covers same-backbone PPO methods.
MAIN_METHODS = [
    'herp',                   # p * sigma  (Contribution 3)
    'herp_sigma',             # sigma-only ablation
    'herp_p',                 # p-only ablation
    'ppo',                    # A0
    'rnd',                    # A1
    'disagreement',           # A2
    'uniform',                # A4  Uniform Region Revisit
    'state_radius_uniform',   # A5.1
    'state_radius_psigma',    # A5.2
    'plr_region',             # A6 (HERP-setting PLR)
    'sacl_style',             # A7 (HERP-setting SACL)
]


def _extra_args(kwargs: dict) -> list[str]:
    out: list[str] = []
    for key, value in kwargs.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value: out.append(flag)
        else:
            out += [flag, str(value)]
    return out


def _load_manifest(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return []


def _save_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=2, default=str))


def _summary_completed(dir_: Path) -> bool:
    summary = dir_ / 'summary.json'
    if not summary.exists():
        return False
    try:
        return json.loads(summary.read_text()).get('status') == 'completed'
    except json.JSONDecodeError:
        return False


def _run_one(run_dir: Path, base_cli: list[str], extra_cli: list[str],
             env: dict, deadline: float | None, log_stream) -> tuple[int, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/train_v3.py'),
           *base_cli, *extra_cli,
           '--output-dir', str(run_dir), '--auto-resume']
    record = dict(command=cmd, run_dir=str(run_dir), started_iso=datetime.utcnow().isoformat())
    log_stream.write(f'\n>>> {" ".join(cmd)}\n'); log_stream.flush()
    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_stream, stderr=subprocess.STDOUT)
    timeout = None if deadline is None else max(1, deadline - time.time())
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); code = -9
        code = -15
    record.update(exit_code=code, finished_iso=datetime.utcnow().isoformat(),
                  status='completed' if code == 0 else 'deadline' if code == -15 else 'failed')
    return code, record


def _round_up(n: int, k: int) -> int:
    """Round n up to the nearest multiple of k (train_v3 requires divisibility)."""
    return ((int(n) + int(k) - 1) // int(k)) * int(k)


def _build_cli(env_id: str, seed: int, timesteps: int, num_envs: int,
               num_eval_envs: int, eval_interval: int, eval_episodes: int,
               checkpoint_interval: int, wandb_mode: str, wandb_project: str,
               wandb_entity: str, wandb_group: str, extra_wandb_tags: list[str]) -> list[str]:
    tags = ','.join(extra_wandb_tags)
    # train_v3 asserts total_timesteps % num_envs == 0 when vectorized.
    total = _round_up(timesteps, num_envs) if num_envs > 1 else int(timesteps)
    eval_iv = _round_up(eval_interval, num_envs) if num_envs > 1 else int(eval_interval)
    ckpt_iv = _round_up(checkpoint_interval, num_envs) if num_envs > 1 else int(checkpoint_interval)
    return [
        '--env-id', env_id, '--seed', str(seed),
        '--num-envs', str(num_envs), '--num-eval-envs', str(num_eval_envs),
        '--total-timesteps', str(total),
        '--eval-interval', str(eval_iv), '--eval-episodes', str(eval_episodes),
        '--checkpoint-interval', str(ckpt_iv),
        '--wandb-mode', wandb_mode,
        '--wandb-project', wandb_project,
        *(['--wandb-entity', wandb_entity] if wandb_entity else []),
        *(['--wandb-group', wandb_group] if wandb_group else []),
        *(['--wandb-tags', tags] if tags else []),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True, help='campaign root; per-run subdirs live here')
    p.add_argument('--env-id', default='PickCube-v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--num-eval-envs', type=int, default=32)
    p.add_argument('--sigma-timesteps', type=int, default=200_000,
                   help='per-run budget for sigma ablations (small; validates estimator)')
    p.add_argument('--main-timesteps', type=int, default=1_000_000,
                   help='per-run budget for main comparison runs (1M steps per user)')
    p.add_argument('--eval-interval', type=int, default=100_000)
    p.add_argument('--eval-episodes', type=int, default=64)
    p.add_argument('--checkpoint-interval', type=int, default=100_000)
    p.add_argument('--wandb-mode', choices=['disabled','offline','online'], default='online')
    p.add_argument('--wandb-project', default='herp-v3')
    p.add_argument('--wandb-entity', default='')
    p.add_argument('--wandb-group', default='', help='defaults to run-dir basename if empty')
    p.add_argument('--skip-sigma', action='store_true', help='skip sigma ablations phase')
    p.add_argument('--only', default='',
                   help='comma-separated method names to include (e.g. herp,ppo,rnd); overrides MAIN_METHODS')
    p.add_argument('--priority-first', default='herp',
                   help='comma-separated methods to promote to the front of the main queue')
    p.add_argument('--deadline-hours', type=float, default=0.0,
                   help='wall-clock deadline in hours; 0 disables')
    args = p.parse_args(argv)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    group = args.wandb_group or out.name
    deadline = None if args.deadline_hours <= 0 else time.time() + args.deadline_hours * 3600

    env = os.environ.copy()
    env.setdefault('LD_LIBRARY_PATH', '/usr/lib/wsl/lib')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')

    manifest_path = out / 'manifest.json'
    manifest = _load_manifest(manifest_path)
    known = {r['run_dir']: i for i, r in enumerate(manifest)}

    def record_run(record: dict) -> None:
        if record['run_dir'] in known:
            manifest[known[record['run_dir']]] = record
        else:
            known[record['run_dir']] = len(manifest); manifest.append(record)
        _save_manifest(manifest_path, manifest)

    log_path = out / 'runner.log'
    log_stream = open(log_path, 'a', buffering=1)

    # Phase 1 — sigma ablations. Small budget, validates the estimator before
    # we sink a 1M-step run on the wrong shrinkage / horizon setting.
    if not args.skip_sigma:
        for nickname, method, extras in SIGMA_ABLATIONS:
            run_dir = out / 'sigma_ablation' / nickname
            if _summary_completed(run_dir):
                record_run(dict(phase='sigma_ablation', nickname=nickname, method=method,
                                run_dir=str(run_dir), status='completed', reused=True))
                continue
            if deadline is not None and time.time() >= deadline: break
            base = _build_cli(args.env_id, args.seed, args.sigma_timesteps,
                              args.num_envs, args.num_eval_envs, args.eval_interval,
                              args.eval_episodes, args.checkpoint_interval,
                              args.wandb_mode, args.wandb_project, args.wandb_entity,
                              f'{group}-sigma', ['phase=sigma', f'ablation={nickname}'])
            base = ['--method', method] + base
            code, record = _run_one(run_dir, base, _extra_args(extras), env, deadline, log_stream)
            record.update(phase='sigma_ablation', nickname=nickname, method=method)
            record_run(record)

    # Phase 2 — main methods at 1M. HERP first, then RND/Disagreement/PPO/rest.
    priority = [m.strip() for m in args.priority_first.split(',') if m.strip()]
    methods = [m.strip() for m in args.only.split(',') if m.strip()] or list(MAIN_METHODS)
    methods = [m for m in priority if m in methods] + [m for m in methods if m not in priority]

    for method in methods:
        run_dir = out / 'main' / method
        if _summary_completed(run_dir):
            record_run(dict(phase='main', method=method, run_dir=str(run_dir),
                            status='completed', reused=True))
            continue
        if deadline is not None and time.time() >= deadline: break
        base = _build_cli(args.env_id, args.seed, args.main_timesteps,
                          args.num_envs, args.num_eval_envs, args.eval_interval,
                          args.eval_episodes, args.checkpoint_interval,
                          args.wandb_mode, args.wandb_project, args.wandb_entity,
                          f'{group}-main', ['phase=main', f'method={method}'])
        base = ['--method', method] + base
        code, record = _run_one(run_dir, base, [], env, deadline, log_stream)
        record.update(phase='main', method=method)
        record_run(record)

    log_stream.close()
    print(json.dumps({'campaign_root': str(out), 'runs': len(manifest),
                      'completed': sum(1 for r in manifest if r.get('status') == 'completed')}))


if __name__ == '__main__':
    main()
