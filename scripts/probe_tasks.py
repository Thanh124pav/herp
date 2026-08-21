"""Task-selection probe: which standard Meta-World task does the upgraded backbone
actually learn? (PLAN.md Gate A on Meta-World, section 25.)

Trains a single upgraded SAC (hidden/UTD/obs-norm/GPU) on each candidate task for
a fixed budget and reports success over time, so we pick a task that is neither
too easy (reach) nor intractable for the population + baseline experiment.

    python scripts/probe_tasks.py --tasks push-v3 window-open-v3 door-open-v3 \
        --steps 60000 --hidden 256 --utd 4 --device cuda --obs-norm
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experience_routing.envs.metaworld_env import MetaWorldEnv
from experience_routing.population.worker import Worker


def probe_task(task, steps, hidden, utd, device, obs_norm, horizon, warmup,
               eval_every, episodes, seed):
    env = MetaWorldEnv(task, seed=seed, horizon=horizon)
    w = Worker(policy_id=0, env=env, seed=seed, hidden=hidden, utd=utd,
               device=device, obs_norm=obs_norm)
    t0 = time.time()
    curve = []
    for step in range(1, steps + 1):
        warm = w.env_steps < warmup
        w.collect_step(warmup=warm)
        if not warm:
            w.train_step(256)
        if step % eval_every == 0:
            ev = w.evaluate(episodes=episodes, base_seed=10_000)
            curve.append((step, ev["success_rate"], ev["mean_return"]))
            print(f"  [{task}] step={step:>7} success={ev['success_rate']:.2f} "
                  f"return={ev['mean_return']:.1f} ({(time.time()-t0)/60:.1f} min)")
    best = max(c[1] for c in curve) if curve else 0.0
    return {"task": task, "curve": curve, "best_success": best,
            "final_success": curve[-1][1] if curve else 0.0,
            "minutes": (time.time() - t0) / 60}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+",
                   default=["push-v3", "window-open-v3", "door-open-v3", "drawer-open-v3"])
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--obs-norm", action="store_true", default=True)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"probing {args.tasks} | hidden={args.hidden} utd={args.utd} "
          f"device={args.device} obs_norm={args.obs_norm} steps={args.steps}")
    results = []
    for task in args.tasks:
        print(f"\n=== {task} ===")
        try:
            results.append(probe_task(
                task, args.steps, args.hidden, args.utd, args.device, args.obs_norm,
                args.horizon, args.warmup, args.eval_every, args.episodes, args.seed))
        except Exception as e:
            print(f"  [{task}] FAILED: {e}")

    print("\n=== RANKING (by best success) ===")
    for r in sorted(results, key=lambda r: -r["best_success"]):
        print(f"  {r['task']:22s} best={r['best_success']:.2f} "
              f"final={r['final_success']:.2f} ({r['minutes']:.1f} min)")
    if results:
        pick = max(results, key=lambda r: r["best_success"])
        print(f"\nRECOMMEND: {pick['task']} (best success {pick['best_success']:.2f})")


if __name__ == "__main__":
    main()
