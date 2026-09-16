"""Run remaining ManiSkill RQ1 GPU jobs in parallel (3-4 concurrent).

Supplements continue_campaign.py which runs 1 at a time.
Skips any job that already has summary.json or is in continue_campaign's queue.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

MANISKILL_TASKS = {
    "PickCube-v1":       dict(num_envs=256, control_mode="pd_ee_delta_pose", reward_mode="dense", budget=2_000_128),
    "LiftPegUpright-v1": dict(num_envs=256, control_mode="pd_joint_delta_pos", reward_mode="normalized_dense", budget=1_490_944),
}

MAX_GPU_PARALLEL = 3  # alongside 1 from continue_campaign = 4 total


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def is_done(env_id, method, seed):
    for parent in ["rq3/maniskill", "rq1"]:
        candidates = [
            OUT / parent / f"{env_id}_{method}_s{seed}" / "summary.json",
            OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}" / "summary.json",
        ]
        for c in candidates:
            if c.exists():
                return True
    return False


def run_job(env_id, method, seed):
    if is_done(env_id, method, seed):
        print(f"  [skip] {env_id} {method} s{seed}", flush=True)
        return env_id, method, seed, "skip"

    tc = MANISKILL_TASKS[env_id]
    budget = tc["budget"]
    ne = tc["num_envs"]
    budget = -(-budget // ne) * ne

    out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "summary.json").exists():
        print(f"  [skip] {env_id} {method} s{seed}", flush=True)
        return env_id, method, seed, "skip"

    cmd = [
        PYTHON, "-u", TRAIN,
        "--phase", "performance", "--benchmark", "maniskill",
        "--env-id", env_id, "--method", method, "--root-floor", "0.15",
        "--seed", str(seed), "--total-timesteps", str(budget),
        "--output-dir", str(out_dir),
        "--num-envs", str(ne), "--num-eval-envs", "16",
        "--control-mode", tc["control_mode"],
        "--reward-mode", tc["reward_mode"],
        "--eval-interval", str(min(100_000, budget // 10)),
        "--eval-episodes", "50",
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-maniskill",
        "--wandb-run-name", f"rq1-maniskill-{env_id}-{method}-s{seed}",
        "--wandb-tags", f"rq1,maniskill,{env_id},{method},seed{seed}",
        "--auto-resume",
    ]

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}

    print(f"  [start] {env_id} {method} s{seed}", flush=True)
    t0 = time.time()
    with open(out_dir / "console.log", "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        code = proc.wait()
    elapsed = time.time() - t0
    ok = code == 0 and (out_dir / "summary.json").exists()
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {env_id} {method} s{seed} ({elapsed:.0f}s)", flush=True)
    return env_id, method, seed, status


def main():
    # Build job list: skip what continue_campaign handles (disagreement-s0 currently running)
    # and what's already done
    jobs = []
    for env_id in ["PickCube-v1", "LiftPegUpright-v1"]:
        for method in ["ppo", "rnd", "disagreement", "herp"]:
            for seed in [0, 1, 2]:
                if not is_done(env_id, method, seed):
                    jobs.append((env_id, method, seed))

    # Remove the job currently running in continue_campaign (disagreement-s0)
    running = ("PickCube-v1", "disagreement", 0)
    jobs = [j for j in jobs if j != running]

    print("=" * 60)
    print(f"  Parallel GPU Campaign — {len(jobs)} jobs, {MAX_GPU_PARALLEL} concurrent")
    print("=" * 60)
    for j in jobs:
        print(f"    {j[0]} {j[1]} s{j[2]}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_GPU_PARALLEL) as pool:
        futures = {pool.submit(run_job, *j): j for j in jobs}
        for f in as_completed(futures):
            results.append(f.result())

    ok = sum(1 for r in results if r[3] in ("ok", "skip"))
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{len(jobs)} ({elapsed:.0f}s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
