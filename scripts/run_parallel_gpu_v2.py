"""Run remaining ManiSkill RQ1 GPU jobs with controlled parallelism.

Polls for GPU slot availability by counting running train_v3 processes
via /proc/*/cmdline (ps aux truncates long lines).
"""
import json, os, subprocess, sys, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

MANISKILL_TASKS = {
    "PickCube-v1":       dict(num_envs=256, control_mode="pd_ee_delta_pose", reward_mode="dense", budget=2_000_128),
    "LiftPegUpright-v1": dict(num_envs=256, control_mode="pd_joint_delta_pos", reward_mode="normalized_dense", budget=1_490_944),
}

MAX_TOTAL_GPU = 4


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def count_running_gpu_jobs():
    """Count running GPU processes (train_v3, tdmpc2, sac_vector) using /proc."""
    count = 0
    try:
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
                if ("train_v3.py" in cmdline or "train_herp_tdmpc2" in cmdline
                    or "train_herp_sac_vector" in cmdline) and "maniskill" in cmdline.lower():
                    count += 1
                elif "run_mbrl_campaign" in cmdline:
                    count += 1
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    except Exception:
        pass
    return count


def get_running_output_dirs():
    """Get output dirs of running train_v3 processes."""
    dirs = set()
    try:
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
                if "train_v3.py" in cmdline and "--output-dir" in cmdline:
                    parts = cmdline.replace("\x00", " ").split()
                    for i, p in enumerate(parts):
                        if p == "--output-dir" and i + 1 < len(parts):
                            dirs.add(parts[i + 1])
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    except Exception:
        pass
    return dirs


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


def launch_job(env_id, method, seed):
    out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
    tc = MANISKILL_TASKS[env_id]
    budget = tc["budget"]
    ne = tc["num_envs"]
    budget = -(-budget // ne) * ne
    out_dir.mkdir(parents=True, exist_ok=True)

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
    lf = open(out_dir / "console.log", "a")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launched] {env_id} {method} s{seed} (PID {proc.pid})", flush=True)
    return proc, lf, (env_id, method, seed)


def main():
    # Build job queue
    queue = []
    for env_id in ["PickCube-v1", "LiftPegUpright-v1"]:
        for method in ["ppo", "rnd", "disagreement", "herp"]:
            for seed in [0, 1]:
                if not is_done(env_id, method, seed):
                    queue.append((env_id, method, seed))

    running_dirs = get_running_output_dirs()

    # Skip jobs that are already running
    new_queue = []
    for job in queue:
        env_id, method, seed = job
        out_dir = str(OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}")
        if out_dir in running_dirs:
            print(f"  [already running] {env_id} {method} s{seed}", flush=True)
        else:
            new_queue.append(job)
    queue = new_queue

    current_running = count_running_gpu_jobs()
    print("=" * 60)
    print(f"  Parallel GPU Campaign v2")
    print(f"  {len(queue)} new jobs to launch, {current_running} already running")
    print(f"  Max total GPU jobs: {MAX_TOTAL_GPU}")
    print("=" * 60)
    for j in queue:
        print(f"    {j[0]} {j[1]} s{j[2]}")
    print(flush=True)

    active = []
    done = 0
    failed = 0

    while queue or active:
        # Check for completed processes
        still_active = []
        for proc, lf, job in active:
            ret = proc.poll()
            if ret is not None:
                lf.close()
                env_id, method, seed = job
                out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
                if ret == 0 and (out_dir / "summary.json").exists():
                    print(f"  [done] {env_id} {method} s{seed}", flush=True)
                    done += 1
                else:
                    print(f"  [FAIL] {env_id} {method} s{seed} (rc={ret})", flush=True)
                    failed += 1
            else:
                still_active.append((proc, lf, job))
        active = still_active

        # Launch new jobs if slots available
        running = count_running_gpu_jobs()
        slots = MAX_TOTAL_GPU - running
        while slots > 0 and queue:
            job = queue.pop(0)
            if is_done(*job):
                done += 1
                continue
            active.append(launch_job(*job))
            slots -= 1
            time.sleep(2)

        if queue or active:
            time.sleep(30)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {done} ok, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
