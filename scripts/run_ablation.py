"""Estimator + hyperparameter ablation runner (IMPLEMENTATION.md §§27-28).

Runs the p_v estimator grid, the sigma_v estimator grid, and (optionally) the
probe-horizon / probe-count / allocated-fraction hyperparameter grid, on one or
two representative tasks. Cells are skipped if a matching ``complete.json`` is
already present.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="maniskill", choices=("maniskill", "metaworld", "fetch"))
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--p-estimators", nargs="+",
                   default=["cosine", "dot", "fisher", "occupancy", "hybrid"])
    p.add_argument("--sigma-estimators", nargs="+",
                   default=["pairwise", "branch", "return"])
    p.add_argument("--hparams", action="store_true",
                   help="Also sweep probe_horizon / num_probes / allocated_frac (§28).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sim-backend", default="physx_cpu")
    p.add_argument("--output-dir", default="outputs/herp_ablation")
    return p.parse_args()


def launch(base_dir: Path, tag: str, extra: list[str], opt) -> int:
    cell = base_dir / tag / f"seed{extra_seed(extra)}"
    cell.mkdir(parents=True, exist_ok=True)
    if list(cell.glob("*/complete.json")):
        print(json.dumps(dict(tag=tag, status="already_complete")), flush=True)
        return 0
    command = [
        sys.executable, "train.py", "--benchmark", opt.benchmark, "--env-id", opt.env_id,
        "--total-timesteps", str(opt.steps), "--eval-interval", str(opt.eval_interval),
        "--device", opt.device, "--sim-backend", opt.sim_backend,
        "--output-dir", str(cell), *extra,
    ]
    with (cell / "console.log").open("w") as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    print(json.dumps(dict(tag=tag, status="complete" if rc == 0 else "failed", rc=rc)), flush=True)
    return rc


def extra_seed(extra: list[str]) -> int:
    return int(extra[extra.index("--seed") + 1])


def main():
    opt = parse()
    base = Path(opt.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "protocol.json").write_text(json.dumps(vars(opt), indent=2))
    for seed in opt.seeds:
        for est in opt.p_estimators:
            launch(base / "p", f"{est}", [
                "--method", "herp", "--seed", str(seed), "--p-estimator", est,
            ], opt)
        for est in opt.sigma_estimators:
            launch(base / "sigma", f"{est}", [
                "--method", "herp", "--seed", str(seed), "--sigma-estimator", est,
            ], opt)
        if opt.hparams:
            for h in (4, 8, 16):
                launch(base / "hparam", f"H{h}", [
                    "--method", "herp", "--seed", str(seed), "--probe-horizon", str(h),
                ], opt)
            for k in (2, 4, 8):
                launch(base / "hparam", f"K{k}", [
                    "--method", "herp", "--seed", str(seed), "--num-probes", str(k),
                ], opt)
            for frac in (0.10, 0.25, 0.50):
                launch(base / "hparam", f"frac{frac:.2f}", [
                    "--method", "herp", "--seed", str(seed), "--allocated-frac", str(frac),
                ], opt)


if __name__ == "__main__":
    main()
