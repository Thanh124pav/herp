"""Run SAC + HERP-SAC on ManiSkill (GPU, vector mode).

Waits for PPO-group GPU queue to drain, then runs SAC group sequentially.
Uses train_herp_sac_vector.py for both sac and sac_herp.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_herp_sac_vector.py")
OUT = ROOT / "outputs" / "final_campaign"

TASKS = {
    "PickCube-v1": {"budget": 2_000_000, "control_mode": "pd_ee_delta_pose",
                    "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
    "LiftPegUpright-v1": {"budget": 1_490_944, "control_mode": "pd_joint_delta_pos",
                          "reward_mode": "normalized_dense", "num_envs": 16, "gamma": 0.8},
}
METHODS = ["sac", "sac_herp"]
SEEDS = [0, 1]


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def count_maniskill_gpu():
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "run_mbrl_campaign" in cmdline or "continue_pickcube" in cmdline:
                count += 1
            elif ("train_v3.py" in cmdline or "train_herp_sac_vector" in cmdline
                  or "train_herp_tdmpc2" in cmdline) and "maniskill" in cmdline.lower():
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def run_one(env_id, method, seed):
    task = TASKS[env_id]
    out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
    summary = out_dir / "summary.json"

    if summary.exists():
        print(f"  [skip] {method} {env_id} s{seed}", flush=True)
        return "skip"

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]
    ne = task["num_envs"]
    budget = -(-budget // ne) * ne

    cmd = [
        PYTHON, "-u", TRAIN,
        "--env-id", env_id,
        "--method", method,
        "--seed", str(seed),
        "--total-timesteps", str(budget),
        "--output-dir", str(out_dir),
        "--num-envs", str(ne),
        "--num-eval-envs", "8",
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

    print(f"\n  [start] {method} {env_id} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(out_dir / "console.log", "w") as lf:
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

    print(f"  [{status}] {method} {env_id} s{seed} ({elapsed:.0f}s)", flush=True)
    return status


def main():
    print("=" * 60)
    print("  SAC Group GPU Campaign (ManiSkill)")
    print(f"  Waiting for PPO-group GPU queue + MBRL group to finish...")
    print("=" * 60, flush=True)

    while True:
        n = count_maniskill_gpu()
        if n == 0:
            print("  GPU free! Starting SAC group.", flush=True)
            break
        print(f"  Waiting for {n} GPU jobs to finish...", flush=True)
        time.sleep(120)

    total = ok = 0
    for env_id in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                total += 1
                status = run_one(env_id, method, seed)
                if status in ("ok", "skip"):
                    ok += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{total} succeeded")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
