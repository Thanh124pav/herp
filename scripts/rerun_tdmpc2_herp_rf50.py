"""Rerun TD-MPC2+HERP with root_floor=0.50 (was 0.15).

Same issue as SAC: root_floor=0.15 starved the replay buffer/world model
of on-distribution data (only 15% root, 85% region).

Waits for GPU slots to free up, then runs sequentially (each job ~3GB VRAM).
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venvs" / "tdmpc2_min" / "bin" / "python")
TRAIN = str(ROOT / "scripts" / "train_herp_tdmpc2.py")
OUT = ROOT / "outputs" / "final_campaign" / "rq1"

ROOT_FLOOR = 0.50

TASKS = {
    "PickCube-v1": {"budget": 2_000_000, "seeds": [0, 1, 2]},
}
NUM_ENVS = 64
NUM_EVAL_ENVS = 16
MAX_GPU_JOBS = 3


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def count_gpu_jobs():
    """Count running GPU training jobs."""
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if ("train_herp_tdmpc2" in cmdline or "train_v3.py" in cmdline) \
               and "python" in cmdline:
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def clean_output(env_id, seed):
    import shutil
    d = OUT / f"maniskill_{env_id}_tdmpc2_herp_s{seed}"
    if not d.exists():
        return
    for f in d.iterdir():
        if f.name == "provenance.json":
            continue
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
    print(f"  [clean] {d.name}", flush=True)


def launch(env_id, seed, budget):
    d = OUT / f"maniskill_{env_id}_tdmpc2_herp_s{seed}"
    d.mkdir(parents=True, exist_ok=True)

    if (d / "summary.json").exists():
        print(f"  [skip] {env_id} s{seed} (done)", flush=True)
        return None

    cmd = [
        PYTHON, "-u", TRAIN,
        "--benchmark", "maniskill",
        "--env-id", env_id,
        "--method", "tdmpc2_herp",
        "--seed", str(seed),
        "--total-timesteps", str(budget),
        "--output-dir", str(d),
        "--num-envs", str(NUM_ENVS),
        "--num-eval-envs", str(NUM_EVAL_ENVS),
        "--root-floor", str(ROOT_FLOOR),
        "--eval-interval", "100000",
        "--eval-episodes", "50",
        "--control-mode", "pd_ee_delta_pose",
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-tdmpc2-herp-rf50",
        "--wandb-run-name", f"tdmpc2_herp-{env_id}-s{seed}-rf50",
        "--wandb-tags", f"rq1,maniskill,{env_id},tdmpc2_herp,seed{seed},rf50",
    ]
    env = {**os.environ, "MUJOCO_GL": "egl",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json"}
    lf = open(d / "console.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {env_id} s{seed} PID {proc.pid} (rf={ROOT_FLOOR})", flush=True)
    return proc, lf, (env_id, seed)


def main():
    print("=" * 60)
    print(f"  TD-MPC2+HERP rerun: root_floor={ROOT_FLOOR}")
    print(f"  Tasks: {list(TASKS.keys())}")
    print(f"  Waits for GPU slots (max {MAX_GPU_JOBS} total)")
    print("=" * 60, flush=True)

    # Build queue
    queue = []
    for env_id, cfg in TASKS.items():
        for seed in cfg["seeds"]:
            clean_output(env_id, seed)
            queue.append((env_id, seed, cfg["budget"]))

    active = []
    done = failed = 0

    while queue or active:
        # Reap
        still = []
        for proc, lf, info in active:
            ret = proc.poll()
            if ret is not None:
                lf.close()
                eid, s = info
                od = OUT / f"maniskill_{eid}_tdmpc2_herp_s{s}"
                if ret == 0 and (od / "summary.json").exists():
                    print(f"  [done] {eid} s{s}", flush=True)
                    done += 1
                else:
                    print(f"  [FAIL] {eid} s{s} (rc={ret})", flush=True)
                    failed += 1
            else:
                still.append((proc, lf, info))
        active = still

        # Launch if GPU available
        if queue and len(active) < MAX_GPU_JOBS:
            gpu_used = count_gpu_jobs()
            free = MAX_GPU_JOBS - gpu_used
            while free > 0 and queue:
                args = queue.pop(0)
                result = launch(*args)
                if result is None:
                    done += 1
                else:
                    active.append(result)
                    free -= 1

        if active or queue:
            time.sleep(60)

    print(f"\n{'=' * 60}")
    print(f"  TD-MPC2+HERP rf={ROOT_FLOOR}: {done} ok, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
