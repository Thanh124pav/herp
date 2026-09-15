"""RQ3 SAC ablation: sac_herp_p and sac_herp_sigma on DMC tasks.

sac and sac_herp already running in RQ1 campaign — reuse those results.
This script adds the two missing ablation variants.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_herp_sac_cpu.py")
OUT = ROOT / "outputs" / "final_campaign"

TASKS = {
    "walker-run": {"budget": 500_000, "gamma": 0.99},
    "cheetah-run": {"budget": 500_000, "gamma": 0.99},
}
METHODS = ["sac_herp_p", "sac_herp_sigma"]
SEEDS = [0, 1]
MAX_PARALLEL = 4


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def launch(env_id, method, seed):
    task = TASKS[env_id]
    out_dir = OUT / "rq3" / "sac" / f"dmc_{env_id}_{method}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "summary.json").exists():
        print(f"  [skip] {method} {env_id} s{seed} (done)", flush=True)
        return None

    cmd = [
        PYTHON, "-u", TRAIN,
        "--benchmark", "dmc",
        "--env-id", env_id,
        "--method", method,
        "--seed", str(seed),
        "--total-timesteps", str(task["budget"]),
        "--output-dir", str(out_dir),
        "--gamma", str(task["gamma"]),
        "--eval-interval", "10000",
        "--eval-episodes", "10",
        "--checkpoint-interval", str(task["budget"] // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq3-sac-ablation",
        "--wandb-run-name", f"rq3-{method}-{env_id}-s{seed}",
        "--phase", "performance",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    lf = open(out_dir / "console.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {method} {env_id} s{seed} (PID {proc.pid})", flush=True)
    return proc, lf, (method, env_id, seed)


def main():
    print("=" * 60)
    print("  RQ3 SAC Ablation: sac_herp_p + sac_herp_sigma on DMC")
    print(f"  Tasks: {list(TASKS.keys())}, Seeds: {SEEDS}")
    print("=" * 60, flush=True)

    # Also symlink RQ1 sac/sac_herp results into rq3/sac/ for easy comparison
    rq3_sac = OUT / "rq3" / "sac"
    rq3_sac.mkdir(parents=True, exist_ok=True)
    for env_id in TASKS:
        for method in ["sac", "sac_herp"]:
            for seed in SEEDS:
                src = OUT / "rq1" / f"dmc_{env_id}_{method}_s{seed}"
                dst = rq3_sac / f"dmc_{env_id}_{method}_s{seed}"
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)
                    print(f"  [link] {dst.name} -> rq1/", flush=True)

    active = []
    done = failed = 0

    queue = [(env_id, method, seed)
             for env_id in TASKS for method in METHODS for seed in SEEDS]

    for job in queue:
        result = launch(*job)
        if result is None:
            done += 1
            continue
        active.append(result)
        if len(active) >= MAX_PARALLEL:
            # Wait for one to finish
            while True:
                for i, (proc, lf, info) in enumerate(active):
                    ret = proc.poll()
                    if ret is not None:
                        lf.close()
                        m, e, s = info
                        od = OUT / "rq3" / "sac" / f"dmc_{e}_{m}_s{s}"
                        if ret == 0 and (od / "summary.json").exists():
                            print(f"  [done] {m} {e} s{s}", flush=True)
                            done += 1
                        else:
                            print(f"  [FAIL] {m} {e} s{s} (rc={ret})", flush=True)
                            failed += 1
                        active.pop(i)
                        break
                else:
                    time.sleep(30)
                    continue
                break

    # Wait for remaining
    for proc, lf, info in active:
        proc.wait()
        lf.close()
        m, e, s = info
        od = OUT / "rq3" / "sac" / f"dmc_{e}_{m}_s{s}"
        if proc.returncode == 0 and (od / "summary.json").exists():
            print(f"  [done] {m} {e} s{s}", flush=True)
            done += 1
        else:
            print(f"  [FAIL] {m} {e} s{s} (rc={proc.returncode})", flush=True)
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RQ3 SAC ablation: {done} ok, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
