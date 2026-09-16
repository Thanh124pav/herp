"""Run TD-MPC2 on ManiSkill tasks (GPU, fair budget).

Uses locked ManiSkill TD-MPC2 via tdmpc2_official.py.
Requires GPU — run after Phase 3 completes.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_campaign"
TDMPC2_PYTHON = str(ROOT / ".venvs" / "tdmpc2_min" / "bin" / "python")
TDMPC2_SCRIPT = "/workspace/herp/scripts/tdmpc2_official.py"

TASKS = {
    "PickCube-v1": {"budget": 2_000_000, "control_mode": "pd_ee_delta_pose"},
    "LiftPegUpright-v1": {"budget": 1_490_944, "control_mode": "pd_joint_delta_pos"},
}

SEEDS = [0, 1, 2]


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def run_one(env_id, seed):
    task = TASKS[env_id]
    out_dir = OUT / "rq1" / f"maniskill_{env_id}_tdmpc2_s{seed}"
    summary = out_dir / "summary.json"

    if summary.exists():
        print(f"  [skip] TD-MPC2 {env_id} s{seed}", flush=True)
        return "skip"

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]
    num_envs = 256

    # Round budget to multiple of num_envs
    budget = -(-budget // num_envs) * num_envs

    cmd = [
        TDMPC2_PYTHON, TDMPC2_SCRIPT,
        "--output-dir", str(out_dir),
        "--env-id", env_id,
        "--phase", "performance",
        "--seed", str(seed),
        "--total-timesteps", str(budget),
        "--num-envs", str(num_envs),
        "--eval-episodes", "50",
        "--eval-interval", str(min(100_000, budget // 10)),
        "--wandb-project", wandb_project(env_id),
        "--wandb-mode", "online",
    ]

    env = {**os.environ, "MUJOCO_GL": "egl", "WANDB_MODE": "online",
           "HERP_PHASE": "performance"}

    print(f"\n  [start] TD-MPC2 {env_id} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(out_dir / "console.log", "w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=14400)
        elapsed = time.time() - t0

        # Create summary from metrics.jsonl if available
        metrics_file = out_dir / "native_source" / "logs" / "metrics.jsonl"
        if not metrics_file.exists():
            # Try alternate location
            for mf in out_dir.rglob("metrics.jsonl"):
                metrics_file = mf
                break

        if metrics_file.exists():
            lines = [json.loads(l) for l in metrics_file.read_text().strip().split("\n") if l.strip()]
            if lines:
                last = lines[-1]
                summary_data = {
                    "method": "tdmpc2", "task": env_id, "seed": seed,
                    "total_timesteps": budget,
                    "final_evaluation": {
                        "eval_return": last.get("eval_return", 0),
                        "eval_success": last.get("success_once", 0) or 0,
                        "step": last.get("env_steps", budget),
                    },
                    "wall_seconds": elapsed,
                    "source": "locked_tdmpc2",
                }
                summary.write_text(json.dumps(summary_data, indent=2))

        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR: {e}"

    print(f"  [{status}] TD-MPC2 {env_id} s{seed} ({elapsed:.0f}s)", flush=True)
    return status


def main():
    print("=" * 60)
    print("  TD-MPC2 Campaign (GPU, ManiSkill)")
    print(f"  Tasks: {list(TASKS.keys())}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60)

    if not Path(TDMPC2_PYTHON).exists():
        print(f"  ERROR: TD-MPC2 venv not found at {TDMPC2_PYTHON}")
        return

    # Sequential — GPU can only run one at a time
    total = 0
    ok_count = 0
    for env_id in TASKS:
        for seed in SEEDS:
            total += 1
            status = run_one(env_id, seed)
            if status in ("ok", "skip"):
                ok_count += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok_count}/{total} succeeded")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
