"""Bounded, restartable, seed-outer benchmark suite (IMPLEMENTATION.md §4.16, §7).

Each ``(seed, benchmark, task, method)`` cell is a subprocess call to train.py.
A cell whose ``complete.json`` exists is skipped, so re-running after a crash
resumes. Aggregation runs after every returned cell.

Ordering (seed-outer per the user request): the outer loop is ``seed`` — for
each seed, all benchmarks × tasks × methods finish before the next seed
starts. Rationale: a first-seed sweep of every method makes it easy to spot
regressions early without waiting for later seeds.

Per-benchmark budgets follow §7:
  ManiSkill:  5 000 000  (75 000 000 for PegInsertionSide-v1)
  Meta-World: 2 000 000
  Fetch:      1 000 000
Override with ``--steps`` (uniform) or ``--task-steps TASK=STEPS`` (per-task).

After every cell finishes, a plateau watchdog inspects the cell's
``metrics.csv``: if the last few ``eval_success`` values are all below a
threshold and their trend is flat, print a ``⚠ PLATEAU WARNING`` line so the
user notices it in ``suite.log``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time


ML_TASKS = {
    "maniskill": ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"],
    "metaworld": ["button-press-v3", "drawer-open-v3", "pick-place-v3", "peg-insert-side-v3"],
    "fetch": ["FetchPush-v4", "FetchPickAndPlace-v4"],
}

# §7 per-benchmark / per-task defaults.
DEFAULT_STEPS_PER_BENCHMARK = {
    "maniskill": 5_000_000,
    "metaworld": 2_000_000,
    "fetch": 1_000_000,
}
DEFAULT_STEPS_PER_TASK = {
    "PegInsertionSide-v1": 75_000_000,
}

# §4.16 per-benchmark sim defaults.
SIM_DEFAULTS = {
    "maniskill": dict(num_envs=1024, sim_backend="physx_cuda"),
    "metaworld": dict(num_envs=8, sim_backend="cpu"),
    "fetch": dict(num_envs=8, sim_backend="cpu"),
}

PLATEAU_THRESHOLD = 0.10   # eval_success below this is "low"
PLATEAU_WINDOW = 5         # ...for at least this many of the tail evals
PLATEAU_MIN_EVALS = 3      # only warn if we have at least this many eval rows


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=["maniskill", "metaworld", "fetch"],
                   choices=list(ML_TASKS))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Explicit task list; overrides benchmark defaults.")
    p.add_argument("--exclude-tasks", nargs="*", default=[],
                   help="Task IDs to skip (applied after per-benchmark expansion). "
                        "E.g. run the matrix without PegInsertionSide-v1 when that "
                        "task is being handled by a separate staged runner.")
    p.add_argument("--methods", nargs="+",
                   default=["ppo", "rnd", "disagreement", "herp_sigma", "herp_p", "herp"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=None,
                   help="Uniform step budget override; default is per-benchmark (§7).")
    p.add_argument("--task-steps", nargs="*", default=[],
                   help="Per-task overrides, e.g. --task-steps PegInsertionSide-v1=25000000")
    p.add_argument("--eval-interval", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel cells. ManiSkill state-obs GPU sim uses only "
                        "~4GB/cell, so 3 cells fit a 16GB card (~2.1x aggregate "
                        "throughput, ~12GB peak). num_envs stays at the recipe "
                        "value (1024); parallelism is the efficiency lever, not "
                        "a bigger num_envs (which would cut PPO updates/budget).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sim-backend", default=None,
                   help="Override sim backend; default is per-benchmark (§4.16).")
    p.add_argument("--num-envs", type=int, default=None,
                   help="Override vector-env count; default is per-benchmark (§4.16).")
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default="herp")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-group", default="",
                   help="Group name; defaults to the suite output directory name.")
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--wandb-log-every", type=int, default=1,
                   help="Log every N policy updates (eval updates are always logged).")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags forwarded to train.py verbatim.")
    p.add_argument("--output-dir", default="outputs/herp_suite")
    return p.parse_args()


def resolve_steps(opt, benchmark: str, task: str) -> int:
    if opt.steps is not None:
        return int(opt.steps)
    if task in opt.task_steps_map:
        return int(opt.task_steps_map[task])
    if task in DEFAULT_STEPS_PER_TASK:
        return int(DEFAULT_STEPS_PER_TASK[task])
    return int(DEFAULT_STEPS_PER_BENCHMARK.get(benchmark, 1_000_000))


def eval_success_from_metrics(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        return []
    out: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("eval_success", "")
            if v not in ("", None):
                try:
                    out.append(float(v))
                except ValueError:
                    continue
    return out


def check_plateau(cell_dir: Path) -> tuple[str, dict] | tuple[None, None]:
    """Return ("plateau", info) if the run finished with low + flat eval_success."""
    metrics_csvs = list(cell_dir.glob("*/metrics.csv"))
    if not metrics_csvs:
        return None, None
    values = eval_success_from_metrics(metrics_csvs[0])
    if len(values) < PLATEAU_MIN_EVALS:
        return None, None
    tail = values[-PLATEAU_WINDOW:] if len(values) >= PLATEAU_WINDOW else values
    max_tail = max(tail)
    if max_tail >= PLATEAU_THRESHOLD:
        return None, None
    # Flat = all-below-threshold. Also compute a rough slope to distinguish
    # "converged low" from "still improving slowly."
    trend = tail[-1] - tail[0]
    return "plateau", dict(
        max_tail=max_tail,
        last=tail[-1],
        first=tail[0],
        trend=trend,
        tail=tail,
        num_evals=len(values),
    )


def main():
    opt = parse()
    opt.task_steps_map = {}
    for entry in opt.task_steps:
        if "=" not in entry:
            raise SystemExit(f"--task-steps expects TASK=STEPS, got {entry!r}")
        key, val = entry.split("=", 1)
        opt.task_steps_map[key.strip()] = int(val)
    root = Path(opt.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {k: v for k, v in vars(opt).items() if k not in ("workers",)}
    protocol_path = root / "suite.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text())
        mismatches = {
            key: (existing_protocol[key], protocol.get(key))
            for key in existing_protocol
            if key in protocol and existing_protocol[key] != protocol[key]
        }
        if mismatches:
            raise ValueError(
                f"Protocol at {protocol_path} differs for {mismatches}; use a new output directory."
            )
    protocol_path.write_text(json.dumps(protocol, indent=2, default=str))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    # Seed-outer ordering: for each seed, all benchmarks × tasks × methods.
    def cells():
        for seed in opt.seeds:
            for benchmark in opt.benchmarks:
                tasks = opt.tasks or ML_TASKS[benchmark]
                tasks = [t for t in tasks if t not in opt.exclude_tasks]
                for task in tasks:
                    for method in opt.methods:
                        yield seed, benchmark, task, method

    def run(seed, benchmark, task, method):
        cell = root / benchmark / f"{task}_{method}_s{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        completed = list(cell.glob("*/complete.json"))
        if completed:
            plateau_kind, plateau_info = check_plateau(cell)
            return dict(seed=seed, benchmark=benchmark, task=task, method=method,
                        status="already_complete",
                        plateau=plateau_info if plateau_kind else None)
        d = SIM_DEFAULTS.get(benchmark, {})
        num_envs = opt.num_envs or d.get("num_envs", 1)
        sim_backend = opt.sim_backend or d.get("sim_backend", "physx_cpu")
        steps = resolve_steps(opt, benchmark, task)
        checkpoints = list(cell.glob("*/checkpoint_*.pt"))
        resume_args = []
        if checkpoints:
            def checkpoint_step(path: Path) -> int:
                try:
                    return int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    return -1
            latest_checkpoint = max(checkpoints, key=checkpoint_step)
            if checkpoint_step(latest_checkpoint) >= 0:
                resume_args = ["--resume-from", str(latest_checkpoint)]
        command = [
            sys.executable, "train.py",
            "--benchmark", benchmark, "--env-id", task, "--method", method, "--seed", str(seed),
            "--total-timesteps", str(steps),
            "--eval-interval", str(min(opt.eval_interval, max(1, steps // 20))),
            "--eval-episodes", str(opt.eval_episodes),
            "--device", opt.device,
            "--sim-backend", sim_backend, "--num-envs", str(num_envs),
            "--output-dir", str(cell),
            "--wandb-mode", opt.wandb_mode,
            "--wandb-project", opt.wandb_project,
            "--wandb-entity", opt.wandb_entity,
            "--wandb-group", opt.wandb_group or root.name,
            "--wandb-tags", opt.wandb_tags,
            "--wandb-log-every", str(opt.wandb_log_every),
            *resume_args,
            *opt.extra,
        ]
        started = time.time()
        with (cell / "console.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        duration = time.time() - started
        plateau_kind, plateau_info = check_plateau(cell)
        return dict(seed=seed, benchmark=benchmark, task=task, method=method,
                    status="complete" if result.returncode == 0 else "failed",
                    returncode=result.returncode,
                    duration_seconds=round(duration, 1),
                    plateau=plateau_info if plateau_kind else None)

    failed = []
    plateaus = []
    with ThreadPoolExecutor(max_workers=opt.workers) as pool:
        futures = [pool.submit(run, *cell) for cell in cells()]
        for future in as_completed(futures):
            status = future.result()
            print(json.dumps(status), flush=True)
            if status["status"] == "failed":
                failed.append(status)
            if status.get("plateau"):
                # Highly visible line so the user notices it in tail -f.
                tag = f"{status['benchmark']}/{status['task']}/{status['method']}/s{status['seed']}"
                info = status["plateau"]
                print(
                    f"⚠ PLATEAU WARNING [{tag}] eval_success stuck < {PLATEAU_THRESHOLD}: "
                    f"tail={info['tail']} (max={info['max_tail']:.3f}, trend={info['trend']:+.3f})",
                    flush=True,
                )
                plateaus.append(status)
            subprocess.run(
                [sys.executable, "analysis/aggregate.py", "--root", str(root),
                 "--output-dir", str(root / "analysis")],
                env=env, check=False,
            )
    (root / "suite_complete.json").write_text(
        json.dumps(
            dict(failures=failed, plateaus=plateaus, finished_at=time.time()),
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
