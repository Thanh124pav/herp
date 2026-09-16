"""Rerun PickCube-v1 PPO-group with num_minibatches=8 (fix collapse).

Original runs collapsed: PPO s1 peaked 94% then dropped to 0%.
Root cause: num_minibatches=32 → minibatch_size=256 too aggressive.
Fix: num_minibatches=8 → minibatch_size=1024, stable training.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

NEW_BUDGET = 4_000_000  # 4M steps
NUM_ENVS = 256
BUDGET_ALIGNED = -(-NEW_BUDGET // NUM_ENVS) * NUM_ENVS  # ceil to num_envs

METHODS = ["ppo", "herp", "rnd", "disagreement"]
SEEDS = [0, 1]
MAX_PARALLEL = 3  # ~3GB VRAM each on GPU


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def clean_output_dir(method, seed):
    """Remove old checkpoints and summary for a fresh rerun."""
    out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_{method}_s{seed}"
    if out_dir.exists():
        for f in out_dir.glob("checkpoint_*.pt"):
            f.unlink()
            print(f"  [rm] {f.name}", flush=True)
        summary = out_dir / "summary.json"
        if summary.exists():
            summary.unlink()
            print(f"  [rm] summary.json for {method} s{seed}", flush=True)


def launch(method, seed):
    out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_{method}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-u", TRAIN,
        "--phase", "performance", "--benchmark", "maniskill",
        "--env-id", "PickCube-v1", "--method", method, "--root-floor", "0.15",
        "--seed", str(seed), "--total-timesteps", str(BUDGET_ALIGNED),
        "--output-dir", str(out_dir),
        "--num-envs", str(NUM_ENVS), "--num-eval-envs", "16",
        "--num-minibatches", "8",
        "--control-mode", "pd_ee_delta_pose",
        "--reward-mode", "dense",
        "--eval-interval", "100000",
        "--eval-episodes", "50",
        "--checkpoint-interval", str(BUDGET_ALIGNED // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project("PickCube-v1"),
        "--wandb-group", "rq1-pickcube-4M-mb8",
        "--wandb-run-name", f"rq1-{method}-PickCube-v1-s{seed}-mb8",
        "--wandb-tags", f"rq1,maniskill,PickCube-v1,{method},seed{seed},4M,mb8",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    lf = open(out_dir / "console.log", "a")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {method} s{seed} (PID {proc.pid}, budget {BUDGET_ALIGNED:,})", flush=True)
    return proc, lf, (method, seed)


def count_non_pickcube_gpu():
    """Count GPU jobs that are NOT PickCube PPO-group (MBRL, SAC GPU, etc)."""
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "run_mbrl_campaign" in cmdline:
                count += 1
            elif ("train_herp_tdmpc2" in cmdline
                  or "train_herp_sac_vector" in cmdline) and "maniskill" in cmdline.lower():
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def is_already_running(method, seed):
    """Check if this specific job is already running."""
    target_dir = f"maniskill_PickCube-v1_{method}_s{seed}"
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "train_v3.py" in cmdline and "PickCube" in cmdline \
               and f"--method\x00{method}" in cmdline and f"--seed\x00{seed}" in cmdline:
                return int(pid_dir.name)
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return 0


def count_my_running():
    """Count running PickCube PPO-group jobs."""
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "train_v3.py" in cmdline and "PickCube" in cmdline:
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def main():
    print("=" * 60)
    print(f"  PickCube-v1 Rerun: num_minibatches=8 (fix collapse)")
    print(f"  Budget: {BUDGET_ALIGNED:,} steps, num_envs={NUM_ENVS}")
    print(f"  Methods: {METHODS}, Seeds: {SEEDS}")
    print("=" * 60, flush=True)

    # Clean old collapsed runs
    for m in METHODS:
        for s in SEEDS:
            clean_output_dir(m, s)

    # Wait for MBRL/other GPU campaigns to finish
    while True:
        n = count_non_pickcube_gpu()
        if n == 0:
            break
        print(f"  Waiting for {n} MBRL/other GPU jobs to finish...", flush=True)
        time.sleep(60)

    # Build queue, skip already-running or completed jobs
    queue = []
    adopted = 0
    for m in METHODS:
        for s in SEEDS:
            out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_{m}_s{s}"
            if (out_dir / "summary.json").exists():
                print(f"  [skip] {m} s{s} (completed)", flush=True)
                continue
            pid = is_already_running(m, s)
            if pid:
                print(f"  [already running] {m} s{s} (PID {pid})", flush=True)
                adopted += 1
                continue
            queue.append((m, s))

    print(f"  Queue: {len(queue)} to launch, {adopted} already running", flush=True)

    active = []
    done = failed = 0

    while queue or active:
        still_active = []
        for proc, lf, job in active:
            ret = proc.poll()
            if ret is not None:
                lf.close()
                method, seed = job
                out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_{method}_s{seed}"
                if ret == 0 and (out_dir / "summary.json").exists():
                    print(f"  [done] {method} s{seed}", flush=True)
                    done += 1
                else:
                    print(f"  [FAIL] {method} s{seed} (rc={ret})", flush=True)
                    failed += 1
            else:
                still_active.append((proc, lf, job))
        active = still_active

        my_running = count_my_running()
        slots = MAX_PARALLEL - my_running
        while slots > 0 and queue:
            job = queue.pop(0)
            active.append(launch(*job))
            slots -= 1
            time.sleep(3)

        if queue or active:
            time.sleep(30)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {done} ok, {failed} failed (+ {adopted} already running)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
