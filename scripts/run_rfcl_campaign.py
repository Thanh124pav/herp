"""Run RFCL on ManiSkill PickCube (CPU mode, fair budget).

RFCL is a demo-based curriculum SAC method. train.steps = raw env steps.
Uses .venvs/rfcl with JAX CPU.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_campaign"
RFCL_PYTHON = str(ROOT / ".venvs" / "rfcl" / "bin" / "python")
RFCL_TRAIN = str(ROOT / "third_party" / "rfcl" / "train.py")
RFCL_REPO = ROOT / "third_party" / "rfcl"

TASKS = {
    "PickCube-v1": {
        "budget": 2_000_000,
        "config": "configs/ms3-cpu/sac_ms3_pickcube.yml",
        "benchmark": "maniskill",
    },
}

SEEDS = [0, 1]
MAX_PARALLEL = 3


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def run_one(env_id, seed):
    task = TASKS[env_id]
    out_dir = OUT / "rq1" / f"{task['benchmark']}_{env_id}_rfcl_s{seed}"
    summary = out_dir / "summary.json"

    if summary.exists():
        print(f"  [skip] RFCL {env_id} s{seed}", flush=True)
        return env_id, seed, "skip"

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = task["budget"]

    config_path = str(RFCL_REPO / task["config"])
    cmd = [
        RFCL_PYTHON, RFCL_TRAIN,
        config_path,
        f"seed={seed}",
        f"train.steps={budget}",
        f"env.env_type=gym:cpu",
        f"env.num_envs=4",
        f"env.env_kwargs.render_backend=cpu",
        f"eval_env.num_envs=2",
        f"save_eval_video=false",
        f"sac.eval_freq={min(100_000, budget // 10)}",
        f"logger.wandb=True",
        f"logger.wandb_cfg.group=rq1-rfcl",
        f"logger.workspace={str(out_dir)}",
        f"logger.exp_name=rq1-rfcl-{env_id}-s{seed}",
        f"logger.project_name={wandb_project(env_id)}",
    ]

    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "MUJOCO_GL": "egl", "WANDB_MODE": "online",
           "JAX_PLATFORMS": "cpu"}

    print(f"  [start] RFCL {env_id} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(out_dir / "console.log", "w") as lf:
            proc = subprocess.run(cmd, cwd=RFCL_REPO, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=14400)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            eval_files = sorted(out_dir.rglob("eval_*.json"))
            if not eval_files:
                eval_files = sorted(out_dir.rglob("*eval*"))
            tb_files = sorted(out_dir.rglob("events.out.tfevents.*"))
            summary_data = {
                "method": "rfcl", "task": env_id, "seed": seed,
                "total_timesteps": budget,
                "wall_seconds": elapsed,
                "source": "locked_rfcl",
            }
            summary.write_text(json.dumps(summary_data, indent=2))

        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR: {e}"

    print(f"  [{status}] RFCL {env_id} s{seed} ({elapsed:.0f}s)", flush=True)
    return env_id, seed, status


def main():
    print("=" * 60)
    print("  RFCL Campaign (CPU, ManiSkill PickCube)")
    print(f"  Budget: {TASKS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60)

    if not Path(RFCL_PYTHON).exists():
        print(f"  ERROR: RFCL venv not found at {RFCL_PYTHON}")
        return

    jobs = [(env_id, seed) for env_id in TASKS for seed in SEEDS]
    print(f"\n  Total jobs: {len(jobs)}")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, e, s): (e, s) for e, s in jobs}
        results = [f.result() for f in as_completed(futures)]

    ok = sum(1 for r in results if r[2] in ("ok", "skip"))
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{len(jobs)} succeeded")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
