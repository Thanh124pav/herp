"""Rerun HERP-only on PickCube-v1 with num_minibatches=8 (fix collapse).

Original HERP runs collapsed: peak 20% then dropped to 0%.
Root cause: num_minibatches=32 too aggressive. Fix: num_minibatches=8.
Only reruns HERP — PPO/RND/Disagreement are kept as-is.
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

BUDGET = 4_000_000
NUM_ENVS = 256
BUDGET_ALIGNED = -(-BUDGET // NUM_ENVS) * NUM_ENVS

SEEDS = [0, 1]


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def launch(seed):
    out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_herp_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-u", TRAIN,
        "--phase", "performance", "--benchmark", "maniskill",
        "--env-id", "PickCube-v1", "--method", "herp", "--root-floor", "0.15",
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
        "--wandb-group", "rq1-herp-pickcube-4M-mb8",
        "--wandb-run-name", f"rq1-herp-PickCube-v1-s{seed}-mb8",
        "--wandb-tags", f"rq1,maniskill,PickCube-v1,herp,seed{seed},4M,mb8",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    lf = open(out_dir / "console.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[launch] herp s{seed} (PID {proc.pid}, budget {BUDGET_ALIGNED:,}, mb=8)", flush=True)
    return proc, lf


def main():
    print("=" * 60)
    print(f"  HERP PickCube-v1 Rerun: num_minibatches=8")
    print(f"  Budget: {BUDGET_ALIGNED:,} steps, num_envs={NUM_ENVS}")
    print(f"  Seeds: {SEEDS} (sequential — only 1 GPU slot free)")
    print("=" * 60, flush=True)

    for seed in SEEDS:
        out_dir = OUT / "rq1" / f"maniskill_PickCube-v1_herp_s{seed}"
        if (out_dir / "summary.json").exists():
            print(f"[skip] herp s{seed} already done", flush=True)
            continue

        proc, lf = launch(seed)
        proc.wait()
        lf.close()

        if proc.returncode == 0 and (out_dir / "summary.json").exists():
            print(f"[done] herp s{seed}", flush=True)
        else:
            print(f"[FAIL] herp s{seed} (rc={proc.returncode})", flush=True)
            break

    print(f"\n{'=' * 60}")
    print(f"  HERP PickCube rerun complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
