"""Run MetaWorld button-press-v3 RQ1 campaign (CPU, all PPO-group methods).

Budget: 500K steps (from pilot). 4 envs, 3 seeds.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

TASKS = {
    "button-press-v3": {"budget": 500_000, "num_envs": 4},
}
METHODS = ["ppo", "rnd", "disagreement", "herp"]
SEEDS = [0, 1]
MAX_PARALLEL = 4


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def run_one(env_id, method, seed):
    task = TASKS[env_id]
    out_dir = OUT / "rq1" / f"metaworld_{env_id}_{method}_s{seed}"
    summary = out_dir / "summary.json"

    if summary.exists():
        print(f"  [skip] {env_id} {method} s{seed}", flush=True)
        return env_id, method, seed, "skip"

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]
    ne = task["num_envs"]
    budget = -(-budget // ne) * ne

    cmd = [
        PYTHON, "-u", TRAIN,
        "--phase", "performance", "--benchmark", "metaworld",
        "--env-id", env_id, "--method", method, "--root-floor", "0.15",
        "--seed", str(seed), "--total-timesteps", str(budget),
        "--output-dir", str(out_dir),
        "--num-envs", str(ne), "--num-eval-envs", "2",
        "--eval-interval", str(min(50_000, budget // 10)),
        "--eval-episodes", "20",
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-metaworld",
        "--wandb-run-name", f"rq1-mw-{env_id}-{method}-s{seed}",
        "--wandb-tags", f"rq1,metaworld,{env_id},{method},seed{seed}",
    ]

    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "MUJOCO_GL": "egl"}

    print(f"  [start] {env_id} {method} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(out_dir / "console.log", "w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=7200)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR: {e}"

    print(f"  [{status}] {env_id} {method} s{seed} ({elapsed:.0f}s)", flush=True)
    return env_id, method, seed, status


def main():
    jobs = [(env_id, method, seed)
            for env_id in TASKS
            for method in METHODS
            for seed in SEEDS]

    print("=" * 60)
    print("  MetaWorld Campaign (CPU)")
    print(f"  Tasks: {list(TASKS.keys())}")
    print(f"  Methods: {METHODS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total jobs: {len(jobs)}, Max parallel: {MAX_PARALLEL}")
    print("=" * 60, flush=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, e, m, s): (e, m, s) for e, m, s in jobs}
        results = [f.result() for f in as_completed(futures)]

    ok = sum(1 for r in results if r[3] in ("ok", "skip"))
    failed = [r for r in results if r[3] not in ("ok", "skip")]
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{len(jobs)} succeeded")
    if failed:
        for r in failed:
            print(f"  FAILED: {r[0]} {r[1]} s{r[2]}: {r[3]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
