"""MBRL group campaign: TD-MPC2 + HERP-TD-MPC2 on ManiSkill (GPU).

Runs up to MAX_PARALLEL jobs concurrently on GPU (~3GB VRAM each).
Also launches SAC GPU jobs when MBRL finishes.
Uses train_herp_tdmpc2.py for tdmpc2/tdmpc2_herp.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TDMPC2_PYTHON = str(ROOT / ".venvs" / "tdmpc2_min" / "bin" / "python")
TDMPC2_TRAIN = str(ROOT / "scripts" / "train_herp_tdmpc2.py")
SAC_PYTHON = sys.executable
SAC_TRAIN = str(ROOT / "scripts" / "train_herp_sac_vector.py")
OUT = ROOT / "outputs" / "final_campaign"

VRAM_PER_JOB_MB = 3200
TOTAL_VRAM_MB = 12288
MAX_PARALLEL = 3  # ~3GB each => 9GB, leaves headroom

TDMPC2_TASKS = {
    "PickCube-v1": {"budget": 2_000_000, "control_mode": "pd_ee_delta_pose",
                    "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
    "LiftPegUpright-v1": {"budget": 1_490_944, "control_mode": "pd_joint_delta_pos",
                          "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
}

SAC_TASKS = {
    "PickCube-v1": {"budget": 2_000_000, "control_mode": "pd_ee_delta_pose",
                    "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
    "LiftPegUpright-v1": {"budget": 1_490_944, "control_mode": "pd_joint_delta_pos",
                          "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
}


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def is_done(env_id, method, seed):
    return (OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}" / "summary.json").exists()


def is_already_running(env_id, method, seed):
    """Check if a training process for this job is already running."""
    target = f"maniskill_{env_id}_{method}_s{seed}"
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if ("train_herp_tdmpc2" in cmdline or "train_herp_sac_vector" in cmdline) \
               and env_id in cmdline and f"--method\x00{method}" in cmdline \
               and f"--seed\x00{seed}" in cmdline:
                return int(pid_dir.name)
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return 0


def get_gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


def count_ppo_gpu_jobs():
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "train_v3.py" in cmdline and "maniskill" in cmdline.lower():
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def launch_tdmpc2(env_id, method, seed):
    task = TDMPC2_TASKS[env_id]
    out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]
    ne = task["num_envs"]
    budget = -(-budget // ne) * ne

    cmd = [
        TDMPC2_PYTHON, "-u", TDMPC2_TRAIN,
        "--env-id", env_id, "--method", method,
        "--seed", str(seed), "--total-timesteps", str(budget),
        "--output-dir", str(out_dir),
        "--num-envs", str(ne), "--num-eval-envs", "8",
        "--control-mode", task["control_mode"],
        "--reward-mode", task["reward_mode"],
        "--gamma", str(task["gamma"]),
        "--eval-interval", str(min(100_000, budget // 10)),
        "--eval-episodes", "50",
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-mbrl",
        "--wandb-run-name", f"rq1-{method}-{env_id}-s{seed}",
        "--wandb-tags", f"rq1,mbrl,{env_id},{method},seed{seed}",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MUJOCO_GL": "egl",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json"}
    lf = open(out_dir / "console.log", "a")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {method} {env_id} s{seed} (PID {proc.pid})", flush=True)
    return proc, lf, (env_id, method, seed)


def launch_sac(env_id, method, seed):
    task = SAC_TASKS[env_id]
    out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]
    ne = task["num_envs"]
    budget = -(-budget // ne) * ne

    cmd = [
        SAC_PYTHON, "-u", SAC_TRAIN,
        "--env-id", env_id, "--method", method,
        "--seed", str(seed), "--total-timesteps", str(budget),
        "--output-dir", str(out_dir),
        "--num-envs", str(ne), "--num-eval-envs", "8",
        "--control-mode", task["control_mode"],
        "--reward-mode", task["reward_mode"],
        "--gamma", str(task["gamma"]),
        "--eval-freq", str(min(50_000, budget // 10)),
        "--eval-episodes", "20",
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project-name", wandb_project(env_id),
        "--wandb-group", "rq1-sac-maniskill",
        "--wandb-run-name", f"rq1-{method}-{env_id}-s{seed}",
        "--phase", "performance",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    lf = open(out_dir / "console.log", "a")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {method} {env_id} s{seed} (PID {proc.pid})", flush=True)
    return proc, lf, (env_id, method, seed)


def run_parallel(queue, launcher_fn, label, max_parallel=MAX_PARALLEL):
    active = []
    done = failed = 0

    while queue or active:
        still_active = []
        for entry in active:
            proc_or_pid, lf, job = entry
            env_id, method, seed = job
            out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"

            if isinstance(proc_or_pid, int):
                # Adopted orphan — check if PID still alive
                try:
                    os.kill(proc_or_pid, 0)
                    still_active.append(entry)
                    continue
                except ProcessLookupError:
                    pass
                if (out_dir / "summary.json").exists():
                    print(f"  [done] {method} {env_id} s{seed} (adopted)", flush=True)
                    done += 1
                else:
                    print(f"  [FAIL] {method} {env_id} s{seed} (adopted, no summary)", flush=True)
                    failed += 1
            else:
                ret = proc_or_pid.poll()
                if ret is not None:
                    if lf:
                        lf.close()
                    if ret == 0 and (out_dir / "summary.json").exists():
                        print(f"  [done] {method} {env_id} s{seed}", flush=True)
                        done += 1
                    else:
                        print(f"  [FAIL] {method} {env_id} s{seed} (rc={ret})", flush=True)
                        failed += 1
                else:
                    still_active.append(entry)
        active = still_active

        while len(active) < max_parallel and queue:
            job = queue.pop(0)
            if is_done(*job):
                print(f"  [skip] {job[1]} {job[0]} s{job[2]}", flush=True)
                done += 1
                continue
            orphan_pid = is_already_running(*job)
            if orphan_pid:
                print(f"  [adopt] {job[1]} {job[0]} s{job[2]} (PID {orphan_pid})", flush=True)
                active.append((orphan_pid, None, job))  # track as (pid, logfile, job)
                continue
            active.append(launcher_fn(*job))
            time.sleep(3)

        if queue or active:
            time.sleep(30)

    print(f"  {label}: {done} ok, {failed} failed", flush=True)
    return done, failed


def main():
    print("=" * 60)
    print("  GPU Campaign: MBRL + SAC (parallel, up to 3 concurrent)")
    print("=" * 60, flush=True)

    # Wait for PPO GPU to drain
    while True:
        n = count_ppo_gpu_jobs()
        if n == 0:
            break
        print(f"  Waiting for {n} PPO GPU jobs...", flush=True)
        time.sleep(60)

    # Phase 1: MBRL (TD-MPC2 + HERP-TD-MPC2)
    mbrl_queue = []
    for env_id in TDMPC2_TASKS:
        for method in ["tdmpc2", "tdmpc2_herp"]:
            for seed in [0, 1]:
                mbrl_queue.append((env_id, method, seed))

    print(f"\n  Phase 1: MBRL — {len(mbrl_queue)} jobs, max {MAX_PARALLEL} parallel", flush=True)
    run_parallel(mbrl_queue, launch_tdmpc2, "MBRL")

    # Phase 2: SAC GPU
    sac_queue = []
    for env_id in SAC_TASKS:
        for method in ["sac", "sac_herp"]:
            for seed in [0, 1]:
                sac_queue.append((env_id, method, seed))

    print(f"\n  Phase 2: SAC GPU — {len(sac_queue)} jobs, max {MAX_PARALLEL} parallel", flush=True)
    run_parallel(sac_queue, launch_sac, "SAC GPU")

    print(f"\n{'=' * 60}")
    print(f"  ALL GPU CAMPAIGNS DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
