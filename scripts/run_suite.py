"""Bounded, restartable, equal-budget benchmark suite (IMPLEMENTATION.md §32).

Each (benchmark, task, method, seed) cell is a subprocess call to train.py. A cell
whose complete.json exists is skipped, so re-running after a crash resumes.
Aggregation runs after every returned cell.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ML_TASKS = {
    "maniskill": ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"],
    "metaworld": ["button-press-v3", "drawer-open-v3", "pick-place-v3", "peg-insert-side-v3"],
    "fetch": ["FetchPush-v4", "FetchPickAndPlace-v4"],
}


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=["maniskill"], choices=list(ML_TASKS))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Explicit task list; overrides benchmark defaults.")
    p.add_argument("--methods", nargs="+",
                   default=["ppo", "rnd", "disagreement", "herp_sigma", "herp_p", "herp"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sim-backend", default=None,
                   help="Override sim backend; default is per-benchmark (§4.16).")
    p.add_argument("--num-envs", type=int, default=None,
                   help="Override vector-env count; default is per-benchmark (§4.16).")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags forwarded to train.py verbatim.")
    p.add_argument("--output-dir", default="outputs/herp_suite")
    return p.parse_args()


def main():
    opt = parse()
    root = Path(opt.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(vars(opt))
    protocol.pop("workers")
    protocol_path = root / "suite.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Protocol at {protocol_path} differs; use a new output directory.")
    protocol_path.write_text(json.dumps(protocol, indent=2, default=str))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def cells():
        for benchmark in opt.benchmarks:
            tasks = opt.tasks or ML_TASKS[benchmark]
            for task in tasks:
                for method in opt.methods:
                    for seed in opt.seeds:
                        yield benchmark, task, method, seed

    # §4.16 per-benchmark defaults
    defaults = {
        "maniskill": dict(num_envs=1024, sim_backend="physx_cuda"),
        "metaworld": dict(num_envs=8, sim_backend="cpu"),
        "fetch": dict(num_envs=8, sim_backend="cpu"),
    }

    def run(benchmark, task, method, seed):
        cell = root / benchmark / f"{task}_{method}_s{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        completed = list(cell.glob("*/complete.json"))
        if completed:
            return dict(benchmark=benchmark, task=task, method=method, seed=seed,
                        status="already_complete")
        d = defaults.get(benchmark, {})
        num_envs = opt.num_envs or d.get("num_envs", 1)
        sim_backend = opt.sim_backend or d.get("sim_backend", "physx_cpu")
        command = [
            sys.executable, "train.py",
            "--benchmark", benchmark, "--env-id", task, "--method", method, "--seed", str(seed),
            "--total-timesteps", str(opt.steps), "--eval-interval", str(opt.eval_interval),
            "--eval-episodes", str(opt.eval_episodes), "--device", opt.device,
            "--sim-backend", sim_backend, "--num-envs", str(num_envs),
            "--output-dir", str(cell),
            *opt.extra,
        ]
        with (cell / "console.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        return dict(benchmark=benchmark, task=task, method=method, seed=seed,
                    status="complete" if result.returncode == 0 else "failed",
                    returncode=result.returncode)

    failed = []
    with ThreadPoolExecutor(max_workers=opt.workers) as pool:
        futures = [pool.submit(run, *cell) for cell in cells()]
        for future in as_completed(futures):
            status = future.result()
            print(json.dumps(status), flush=True)
            if status["status"] == "failed":
                failed.append(status)
            subprocess.run(
                [sys.executable, "analysis/aggregate.py", "--root", str(root),
                 "--output-dir", str(root / "analysis")],
                env=env, check=False,
            )
    (root / "suite_complete.json").write_text(json.dumps(dict(failures=failed, finished_at=time.time()), indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
