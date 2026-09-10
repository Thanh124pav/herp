#!/usr/bin/env python3
"""Staged, resume-based budget escalation for a single (task, method) cell.

Motivation: PegInsertionSide-v1 has a large paper budget (75M in §7), but we do
not want to commit 75M up front. Instead run a short budget, check whether
eval_success has plateaued, and only then extend — reusing the previous
checkpoint via train.py's --resume-from (total_timesteps is not resume-guarded,
and anneal_lr defaults to False, so extending is equivalent to one long run).

Each stage k trains up to STAGES[k] cumulative env steps, resuming from the
latest checkpoint written by stage k-1. After each stage we read eval_success
from metrics.csv; we stop when the best eval_success gained < --plateau-eps over
the previous stage (learning has flattened — whether high or stuck low), or when
success is already >= --solved, or when the budget cap is reached.

Example:
    python scripts/run_peg_staged.py --env-id PegInsertionSide-v1 --method ppo \
        --stages 5_000_000 10_000_000 15_000_000 20_000_000 25_000_000 \
        --num-envs 1024 --output-dir outputs/peg_staged_ppo --wandb-mode online
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="maniskill")
    p.add_argument("--env-id", default="PegInsertionSide-v1")
    p.add_argument("--method", default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stages", nargs="+", type=int,
                   default=[5_000_000, 10_000_000, 15_000_000, 20_000_000,
                            25_000_000, 35_000_000, 50_000_000, 75_000_000],
                   help="Cumulative total_timesteps targets, ascending.")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--sim-backend", default="physx_cuda")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-interval", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--plateau-eps", type=float, default=0.02,
                   help="Stop when best eval_success gains less than this vs the "
                        "previous stage.")
    p.add_argument("--min-success-for-plateau", type=float, default=0.05,
                   help="Only treat a low gain as a plateau once best eval_success "
                        "has climbed above this floor. While the run is still stuck "
                        "below it (e.g. a hard task with delayed take-off), keep "
                        "escalating the budget to the cap instead of concluding.")
    p.add_argument("--solved", type=float, default=0.95,
                   help="Stop early once best eval_success reaches this.")
    p.add_argument("--output-dir", default="outputs/peg_staged")
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default="herp")
    p.add_argument("--wandb-group", default="peg-staged")
    p.add_argument("--wandb-tags", default="peg-staged")
    return p.parse_args()


def best_eval_success(stage_dir: Path) -> float | None:
    """Max eval_success across all metrics.csv rows under a stage directory."""
    best = None
    for csv_path in stage_dir.glob("*/metrics.csv"):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                v = row.get("eval_success", "")
                if v not in ("", None):
                    try:
                        val = float(v)
                    except ValueError:
                        continue
                    best = val if best is None else max(best, val)
    return best


def latest_checkpoint(stage_dir: Path) -> Path | None:
    def step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    ckpts = list(stage_dir.glob("*/checkpoint_*.pt"))
    if not ckpts:
        return None
    latest = max(ckpts, key=step)
    return latest if step(latest) >= 0 else None


def main():
    opt = parse()
    root = Path(opt.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stages = sorted(set(opt.stages))
    history: list[dict] = []
    prev_best = None
    prev_ckpt = None

    for i, budget in enumerate(stages):
        stage_dir = root / f"stage_{budget}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        marker = stage_dir / "stage_complete.json"
        own_ckpt = latest_checkpoint(stage_dir)

        if marker.exists() and own_ckpt is not None:
            # Finished in a previous driver run; re-evaluate without re-training
            # so restart resumes at the right point and applies stop conditions.
            best = best_eval_success(stage_dir)
            prev_ckpt = own_ckpt
            print(json.dumps(dict(event="stage_skip", stage=i, budget=budget,
                                  best_eval_success=best)), flush=True)
        else:
            # Resume own partial checkpoint if present, else the previous stage's.
            resume_from = own_ckpt or prev_ckpt
            resume_args = ["--resume-from", str(resume_from)] if resume_from else []
            command = [
                sys.executable, "train.py",
                "--benchmark", opt.benchmark, "--env-id", opt.env_id,
                "--method", opt.method, "--seed", str(opt.seed),
                "--total-timesteps", str(budget),
                "--num-envs", str(opt.num_envs), "--sim-backend", opt.sim_backend,
                "--device", opt.device,
                "--eval-interval", str(opt.eval_interval),
                "--eval-episodes", str(opt.eval_episodes),
                "--output-dir", str(stage_dir),
                "--wandb-mode", opt.wandb_mode, "--wandb-project", opt.wandb_project,
                "--wandb-group", opt.wandb_group,
                "--wandb-tags", opt.wandb_tags,
                *resume_args,
            ]
            print(json.dumps(dict(event="stage_start", stage=i, budget=budget,
                                  resumed_from=str(resume_from) if resume_from else None)),
                  flush=True)
            with (stage_dir / "console.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(json.dumps(dict(event="stage_failed", stage=i, budget=budget,
                                      returncode=result.returncode)), flush=True)
                raise SystemExit(1)
            ckpt = latest_checkpoint(stage_dir)
            if ckpt is None:
                print(json.dumps(dict(event="no_checkpoint", stage=i)), flush=True)
                raise SystemExit(1)
            prev_ckpt = ckpt
            best = best_eval_success(stage_dir)
            marker.write_text(json.dumps(dict(budget=budget, best_eval_success=best)))

        gain = None if (best is None or prev_best is None) else best - prev_best
        rec = dict(event="stage_done", stage=i, budget=budget,
                   best_eval_success=best, gain_vs_prev=gain)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        (root / "staged_summary.json").write_text(json.dumps(history, indent=2))

        # Stop conditions.
        if best is not None and best >= opt.solved:
            print(json.dumps(dict(event="stop", reason="solved", budget=budget,
                                  best_eval_success=best)), flush=True)
            break
        # Only conclude "plateau" once the run has actually learned above the floor;
        # while still stuck low, keep escalating (delayed take-off on hard tasks).
        if (gain is not None and gain < opt.plateau_eps
                and best is not None and best >= opt.min_success_for_plateau):
            print(json.dumps(dict(event="stop", reason="plateau", budget=budget,
                                  gain_vs_prev=gain, plateau_eps=opt.plateau_eps,
                                  best_eval_success=best)), flush=True)
            break
        prev_best = best if best is not None else prev_best
    else:
        print(json.dumps(dict(event="stop", reason="budget_cap",
                              budget=stages[-1])), flush=True)

    print(json.dumps(dict(event="finished", history=history)), flush=True)


if __name__ == "__main__":
    main()
