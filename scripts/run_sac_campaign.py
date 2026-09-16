"""Run SAC group (SAC + HERP-SAC) on DMC and MetaWorld (CPU).

Fair comparison: same SAC backbone, same budget. Only the allocation differs.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_herp_sac_cpu.py")
OUT = ROOT / "outputs" / "final_campaign"

TASKS = {
    # DMC
    ("dmc", "walker-run"): {"budget": 500_000, "seeds": 2, "gamma": 0.99},
    ("dmc", "cheetah-run"): {"budget": 500_000, "seeds": 2, "gamma": 0.99},
    # MetaWorld
    ("metaworld", "button-press-v3"): {"budget": 500_000, "seeds": 2, "gamma": 0.99},
}

METHODS = ["sac", "sac_herp"]
MAX_PARALLEL = 6


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def out_dir_name(benchmark, env_id, method, seed):
    prefix = {"dmc": "dmc", "metaworld": "metaworld", "maniskill": "maniskill"}[benchmark]
    return OUT / "rq1" / f"{prefix}_{env_id}_{method}_s{seed}"


def run_one(benchmark, env_id, method, seed):
    task = TASKS[(benchmark, env_id)]
    odir = out_dir_name(benchmark, env_id, method, seed)
    summary = odir / "summary.json"

    if summary.exists():
        print(f"  [skip] {method} {benchmark}/{env_id} s{seed}", flush=True)
        return benchmark, env_id, method, seed, "skip"

    odir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]

    cmd = [
        PYTHON, "-u", TRAIN,
        "--benchmark", benchmark,
        "--env-id", env_id,
        "--method", method,
        "--seed", str(seed),
        "--total-timesteps", str(budget),
        "--output-dir", str(odir),
        "--root-floor", "0.15",
        "--gamma", str(task["gamma"]),
        "--eval-interval", str(min(25_000, budget // 10)),
        "--eval-episodes", "10",
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-sac",
        "--wandb-run-name", f"rq1-{method}-{env_id}-s{seed}",
        "--wandb-tags", f"rq1,sac,{benchmark},{env_id},{method},seed{seed}",
    ]

    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "MUJOCO_GL": "egl"}

    print(f"  [start] {method} {benchmark}/{env_id} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(odir / "console.log", "w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=14400)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR: {e}"

    print(f"  [{status}] {method} {benchmark}/{env_id} s{seed} ({elapsed:.0f}s)", flush=True)
    return benchmark, env_id, method, seed, status


def main():
    jobs = []
    for (benchmark, env_id), task in TASKS.items():
        for method in METHODS:
            for seed in range(task["seeds"]):
                jobs.append((benchmark, env_id, method, seed))

    print("=" * 60)
    print("  SAC Group Campaign (CPU)")
    print(f"  Tasks: {list(TASKS.keys())}")
    print(f"  Methods: {METHODS}")
    print(f"  Total jobs: {len(jobs)}, Max parallel: {MAX_PARALLEL}")
    print("=" * 60, flush=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        results = [f.result() for f in as_completed(futures)]

    ok = sum(1 for r in results if r[4] in ("ok", "skip"))
    failed = [r for r in results if r[4] not in ("ok", "skip")]
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{len(jobs)} succeeded")
    if failed:
        for r in failed:
            print(f"  FAILED: {r[2]} {r[0]}/{r[1]} s{r[3]}: {r[4]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
