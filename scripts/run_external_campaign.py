"""Run external baselines (BRO, MaxInfoRL) on DMC tasks in parallel on CPU.

Fair budget: same raw env steps as PPO group per task.
BRO: no action repeat, 1 step = 1 raw step.
MaxInfoRL: action_repeat=2, script handles conversion internally.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_campaign"

VENVS = {
    "BRO": ROOT / ".venvs" / "bro" / "bin" / "python",
    "MaxInfoRL": ROOT / ".venvs" / "maxinforl_min" / "bin" / "python",
}

TASKS = {
    "walker-run": 500_000,
    "cheetah-run": 500_000,
}

SEEDS = list(range(2))
MAX_PARALLEL_BRO = 3      # BRO: JAX uses many threads per process, limit concurrency
MAX_PARALLEL_MAXINFO = 5   # MaxInfoRL: lighter, can run more


def wandb_project(task):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in task).strip("-")
    return f"herp-{safe}"


def run_one(method, task, seed):
    out_dir = OUT / "rq1" / f"dmc_{task}_{method.lower()}_s{seed}"
    summary = out_dir / "summary.json"
    metrics = out_dir / "metrics.jsonl"

    if summary.exists():
        print(f"  [skip] {method} {task} s{seed}", flush=True)
        return method, task, seed, "skip"

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = TASKS[task]
    venv_python = str(VENVS[method])
    runner = str(ROOT / "scripts" / "run_external_native.py")

    cmd = [
        venv_python, runner,
        "--method", method,
        "--task", task,
        "--seed", str(seed),
        "--env-steps", str(budget),
        "--output-dir", str(out_dir),
        "--wandb-project", wandb_project(task),
    ]

    env = {**os.environ,
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
           "JAX_PLATFORMS": "cpu",
           "JAX_COMPILATION_CACHE_DIR": str(ROOT / ".cache" / "jax_compilation"),
           "MUJOCO_GL": "egl", "WANDB_MODE": "online"}

    print(f"  [start] {method} {task} s{seed}", flush=True)
    t0 = time.time()
    try:
        with open(out_dir / "console.log", "w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=36000)
        elapsed = time.time() - t0

        if metrics.exists():
            lines = [json.loads(l) for l in metrics.read_text().strip().split("\n") if l.strip()]
            if lines:
                last = lines[-1]
                summary_data = {
                    "method": method, "task": task, "seed": seed,
                    "total_timesteps": budget,
                    "final_evaluation": {
                        "eval_return": last.get("eval_return", 0),
                        "eval_success": last.get("success_once", 0) or 0,
                        "step": last.get("env_steps", budget),
                    },
                    "wall_seconds": elapsed,
                    "source": f"locked_{method}",
                }
                summary.write_text(json.dumps(summary_data, indent=2))

        ok = summary.exists()
        status = "ok" if ok else "FAIL"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERROR: {e}"

    print(f"  [{status}] {method} {task} s{seed} ({elapsed:.0f}s)", flush=True)
    return method, task, seed, status


def main():
    print("=" * 60)
    print("  External Baselines Campaign (CPU parallel)")
    print("  BRO + MaxInfoRL on DMC walker-run + cheetah-run")
    print(f"  Budget: {TASKS}")
    print(f"  Seeds: {SEEDS}, BRO parallel: {MAX_PARALLEL_BRO}, MaxInfo parallel: {MAX_PARALLEL_MAXINFO}")
    print("=" * 60)

    # Verify venvs exist
    for method, vpy in VENVS.items():
        if not vpy.exists():
            print(f"  ERROR: {method} venv not found at {vpy}")
            return

    # Run BRO with limited parallelism (JAX is thread-heavy)
    bro_jobs = [(m, t, s) for m in ["BRO"] for t in TASKS for s in SEEDS]
    maxinfo_jobs = [(m, t, s) for m in ["MaxInfoRL"] for t in TASKS for s in SEEDS]

    all_results = []

    print(f"\n  BRO jobs: {len(bro_jobs)}, max parallel: {MAX_PARALLEL_BRO}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BRO) as pool:
        futures = {pool.submit(run_one, m, t, s): (m, t, s) for m, t, s in bro_jobs}
        for future in as_completed(futures):
            all_results.append(future.result())

    print(f"\n  MaxInfoRL jobs: {len(maxinfo_jobs)}, max parallel: {MAX_PARALLEL_MAXINFO}")
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MAXINFO) as pool:
        futures = {pool.submit(run_one, m, t, s): (m, t, s) for m, t, s in maxinfo_jobs}
        for future in as_completed(futures):
            all_results.append(future.result())

    elapsed = time.time() - t0
    ok = sum(1 for r in all_results if r[3] in ("ok", "skip"))
    total = len(all_results)
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok}/{total} succeeded ({elapsed:.0f}s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
