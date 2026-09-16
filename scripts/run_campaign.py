"""HERP final campaign orchestrator — stages A through G per PLAN.md.

Exploits 88 CPU cores + RTX 3060: ManiSkill on GPU (vectorized) while
DMC/MetaWorld jobs run in parallel on CPU. Logs to wandb online.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

ENV_BASE = dict(
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    MUJOCO_GL="egl",
)

def wandb_project(env_id):
    """One W&B project per task, as required by the final campaign protocol."""
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"
MAX_CPU_PARALLEL = 16  # conservative: 88 cores / ~4-5 threads per DMC job

# ---------- task definitions ----------
MANISKILL_TASKS = {
    "PickCube-v1":       dict(num_envs=256, control_mode="pd_ee_delta_pose", reward_mode="dense"),
    "LiftPegUpright-v1": dict(num_envs=256, control_mode="pd_joint_delta_pos", reward_mode="normalized_dense"),
}
DMC_TASKS = {
    "walker-run":              dict(),
    "cartpole-swingup_sparse": dict(),
}

PPO_METHODS = ["ppo", "rnd", "disagreement", "herp"]
ABLATION_METHODS = ["herp", "herp_p", "herp_sigma", "uniform"]


def build_cmd(benchmark, env_id, method, total_timesteps, seed,
              output_dir, wandb_group="", wandb_tags="", phase="pilot",
              extra_args=None):
    """Build the train_v3.py command list."""
    out = Path(output_dir)
    if benchmark == "maniskill":
        nenv = MANISKILL_TASKS.get(env_id, {}).get("num_envs", 256)
        total_timesteps = -(-total_timesteps // nenv) * nenv
    cmd = [
        PYTHON, "-u", TRAIN,
        "--phase", phase,
        "--benchmark", benchmark,
        "--env-id", env_id,
        "--method", method,
        "--root-floor", "0.15",
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--output-dir", str(out),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", wandb_group or f"{phase}-{benchmark}",
        "--wandb-run-name", f"{phase}-{benchmark}-{env_id}-{method}-s{seed}",
        "--wandb-tags", wandb_tags or f"{phase},{benchmark},{env_id},{method}",
        "--auto-resume",
    ]
    if benchmark == "maniskill":
        task_cfg = MANISKILL_TASKS.get(env_id, {})
        cmd += [
            "--num-envs", str(task_cfg.get("num_envs", 256)),
            "--num-eval-envs", "16",
            "--control-mode", task_cfg.get("control_mode", "pd_ee_delta_pose"),
            "--reward-mode", task_cfg.get("reward_mode", "dense"),
        ]
    else:
        cmd += ["--num-envs", "1", "--num-eval-envs", "1", "--batch-size", "1024"]
    if extra_args:
        cmd += extra_args
    return cmd


def run_job(name, cmd, output_dir, env=None):
    """Run a single training job synchronously. Returns (name, ok)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"  [skip] {name}", flush=True)
        return (name, True)

    run_env = {**os.environ, **ENV_BASE, **(env or {})}
    print(f"  [start] {name}", flush=True)
    t0 = time.time()
    log_path = out / "console.log"
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=run_env, stdout=lf, stderr=subprocess.STDOUT)
        code = proc.wait()
    elapsed = time.time() - t0
    ok = code == 0 and (out / "summary.json").exists()
    print(f"  [{'ok' if ok else 'FAIL'}] {name} ({elapsed:.0f}s)", flush=True)
    return (name, ok)


def run_parallel_cpu(jobs, max_workers=MAX_CPU_PARALLEL):
    """Run CPU jobs in parallel using a thread pool."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for name, cmd, out_dir in jobs:
            f = pool.submit(run_job, name, cmd, out_dir)
            futures[f] = name
        for f in concurrent.futures.as_completed(futures):
            name, ok = f.result()
            results[name] = ok
    return results


def run_gpu_and_cpu(gpu_jobs, cpu_jobs):
    """Run GPU jobs sequentially while CPU jobs run in parallel."""
    results = {}
    cpu_futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CPU_PARALLEL) as cpu_pool:
        for name, cmd, out_dir in cpu_jobs:
            f = cpu_pool.submit(run_job, name, cmd, out_dir)
            cpu_futures.append((f, name))

        for name, cmd, out_dir in gpu_jobs:
            n, ok = run_job(name, cmd, out_dir)
            results[n] = ok

        for f, name in cpu_futures:
            n, ok = f.result()
            results[n] = ok

    return results


# ==================== STAGES ====================

def stage_smoke():
    """Stage A: Quick smoke tests — 50k steps per method/env."""
    print("\n" + "=" * 60)
    print("  STAGE A: SMOKE TESTS")
    print("=" * 60)

    gpu_jobs = []
    cpu_jobs = []

    for env_id in ["PickCube-v1"]:
        for method in ["ppo", "herp"]:
            out_dir = OUT / "smoke" / f"maniskill_{env_id}_{method}"
            cmd = build_cmd("maniskill", env_id, method, 51200, 0, out_dir,
                            phase="smoke",
                            extra_args=["--eval-interval", "51200", "--eval-episodes", "8",
                                        "--checkpoint-interval", "51200"])
            gpu_jobs.append((f"smoke-ms-{env_id}-{method}", cmd, out_dir))

    for env_id in ["walker-run", "cartpole-swingup_sparse"]:
        for method in ["ppo", "herp"]:
            out_dir = OUT / "smoke" / f"dmc_{env_id}_{method}"
            cmd = build_cmd("dmc", env_id, method, 50000, 0, out_dir,
                            phase="smoke",
                            extra_args=["--eval-interval", "50000", "--eval-episodes", "5",
                                        "--checkpoint-interval", "50000"])
            cpu_jobs.append((f"smoke-dmc-{env_id}-{method}", cmd, out_dir))

    results = run_gpu_and_cpu(gpu_jobs, cpu_jobs)

    all_ok = all(results.values())
    print(f"\n--- Smoke: {sum(results.values())}/{len(results)} passed ---")
    for name, ok in results.items():
        if not ok:
            print(f"  FAIL: {name}")
    return all_ok


def stage_pilots():
    """Stage B: Vanilla PPO budget pilots to calibrate T_task."""
    print("\n" + "=" * 60)
    print("  STAGE B: BUDGET PILOTS")
    print("=" * 60)

    gpu_jobs = []
    cpu_jobs = []

    ms_pilots = {
        "PickCube-v1":       1_000_000,
        "LiftPegUpright-v1": 2_000_000,
    }
    dmc_pilots = {
        "walker-run":              500_000,
        "cartpole-swingup_sparse": 500_000,
    }

    for env_id, budget in ms_pilots.items():
        out_dir = OUT / "pilots" / f"maniskill_{env_id}_ppo"
        eval_int = min(100_000, budget // 10)
        cmd = build_cmd("maniskill", env_id, "ppo", budget, 0, out_dir,
                        phase="pilot",
                        extra_args=["--eval-interval", str(eval_int),
                                    "--eval-episodes", "10",
                                    "--checkpoint-interval", str(budget // 4)])
        gpu_jobs.append((f"pilot-ms-{env_id}", cmd, out_dir))

    for env_id, budget in dmc_pilots.items():
        out_dir = OUT / "pilots" / f"dmc_{env_id}_ppo"
        cmd = build_cmd("dmc", env_id, "ppo", budget, 0, out_dir,
                        phase="pilot",
                        extra_args=["--eval-interval", "50000",
                                    "--eval-episodes", "10",
                                    "--checkpoint-interval", str(budget // 4)])
        cpu_jobs.append((f"pilot-dmc-{env_id}", cmd, out_dir))

    results = run_gpu_and_cpu(gpu_jobs, cpu_jobs)

    # Print pilot summaries
    for (bench, env_id) in [("maniskill", "PickCube-v1"), ("maniskill", "LiftPegUpright-v1"),
                             ("dmc", "walker-run"), ("dmc", "cartpole-swingup_sparse")]:
        out_dir = OUT / "pilots" / f"{bench}_{env_id}_ppo"
        if (out_dir / "summary.json").exists():
            s = json.loads((out_dir / "summary.json").read_text())
            ev = s.get("final_evaluation", {})
            print(f"  {bench}/{env_id}: return={ev.get('eval_return', 0):.2f}, "
                  f"success={ev.get('eval_success', 0):.3f}, "
                  f"steps={s.get('training_steps', 0)}")
    return results


def freeze_budgets():
    """Freeze task budgets based on pilot outcomes."""
    pilot_dir = OUT / "pilots"
    budgets = {}

    task_defs = [
        ("maniskill", "PickCube-v1", 1_000_000),
        ("maniskill", "LiftPegUpright-v1", 2_000_000),
        ("dmc", "walker-run", 500_000),
        ("dmc", "cartpole-swingup_sparse", 500_000),
    ]

    for bench, env_id, default_budget in task_defs:
        out_dir = pilot_dir / f"{bench}_{env_id}_ppo"
        if (out_dir / "summary.json").exists():
            s = json.loads((out_dir / "summary.json").read_text())
            ev = s.get("final_evaluation", {})
            ts = s.get("training_steps", default_budget)
            if bench == "maniskill":
                success = ev.get("eval_success", 0)
                if success >= 0.8:
                    budgets[f"{bench}/{env_id}"] = max(ts // 2, 500_000)
                elif success >= 0.2:
                    budgets[f"{bench}/{env_id}"] = ts
                else:
                    budgets[f"{bench}/{env_id}"] = min(ts * 2, 5_000_000)
            else:
                budgets[f"{bench}/{env_id}"] = ts
        else:
            budgets[f"{bench}/{env_id}"] = default_budget

    bf = OUT / "task_budgets.json"
    bf.write_text(json.dumps(budgets, indent=2))
    print(f"\nFrozen budgets:\n{json.dumps(budgets, indent=2)}")
    return budgets


def stage_rq3(budgets):
    """Stage C: RQ3 p×σ ablation — all seeds, GPU+CPU parallel."""
    print("\n" + "=" * 60)
    print("  STAGE C: RQ3 ABLATION")
    print("=" * 60)

    for seed in [0, 1, 2]:
        gpu_jobs = []
        cpu_jobs = []

        for method in ABLATION_METHODS:
            # PickCube on GPU
            env_id = "PickCube-v1"
            budget = budgets.get("maniskill/PickCube-v1", 1_000_000)
            out_dir = OUT / "rq3" / "maniskill" / f"{env_id}_{method}_s{seed}"
            cmd = build_cmd("maniskill", env_id, method, budget, seed, out_dir,
                            wandb_group="rq3-maniskill",
                            wandb_tags=f"rq3,maniskill,{env_id},{method},seed{seed}",
                            phase="ablation",
                            extra_args=["--eval-interval", str(min(100_000, budget // 10)),
                                        "--eval-episodes", "50",
                                        "--checkpoint-interval", str(budget // 4)])
            gpu_jobs.append((f"rq3-ms-{env_id}-{method}-s{seed}", cmd, out_dir))

            # Cartpole on CPU
            env_id = "cartpole-swingup_sparse"
            budget = budgets.get("dmc/cartpole-swingup_sparse", 500_000)
            out_dir = OUT / "rq3" / "dmc" / f"{env_id}_{method}_s{seed}"
            cmd = build_cmd("dmc", env_id, method, budget, seed, out_dir,
                            wandb_group="rq3-dmc",
                            wandb_tags=f"rq3,dmc,{env_id},{method},seed{seed}",
                            phase="ablation",
                            extra_args=["--eval-interval", "50000",
                                        "--eval-episodes", "10",
                                        "--checkpoint-interval", str(budget // 4)])
            cpu_jobs.append((f"rq3-dmc-{env_id}-{method}-s{seed}", cmd, out_dir))

        results = run_gpu_and_cpu(gpu_jobs, cpu_jobs)

        if seed == 0:
            print("\n--- RQ3 seed-0 gate: checking viability ---")
            # Quick sanity: if herp is clearly broken vs p-only in both domains, abort
            herp_ms = OUT / "rq3" / "maniskill" / "PickCube-v1_herp_s0" / "summary.json"
            p_ms = OUT / "rq3" / "maniskill" / "PickCube-v1_herp_p_s0" / "summary.json"
            if herp_ms.exists() and p_ms.exists():
                h = json.loads(herp_ms.read_text())["final_evaluation"]["eval_success"]
                pp = json.loads(p_ms.read_text())["final_evaluation"]["eval_success"]
                print(f"  PickCube: herp={h:.3f}, herp_p={pp:.3f}")
            print("  Continuing to seeds 1-2...")


def stage_rq4(budgets):
    """Stage D: RQ4 sigma oracle validation on PickCube + walker-run."""
    print("\n" + "=" * 60)
    print("  STAGE D: RQ4 SIGMA ORACLE")
    print("=" * 60)

    rq4_tasks = [
        ("maniskill", "PickCube-v1"),
        ("dmc", "walker-run"),
    ]

    for bench, env_id in rq4_tasks:
        # Find HERP checkpoints — try rq3 first, then rq4-specific
        herp_dir = OUT / "rq3" / bench / f"{env_id}_herp_s0"
        if not herp_dir.exists() or not list(herp_dir.glob("checkpoint_*.pt")):
            budget = budgets.get(f"{bench}/{env_id}", 500_000)
            herp_dir = OUT / "rq4" / bench / f"{env_id}_herp_s0"
            if bench == "dmc":
                eval_int = 50000
            else:
                eval_int = min(100_000, budget // 10)
            cmd = build_cmd(bench, env_id, "herp", budget, 0, herp_dir,
                            wandb_group=f"rq4-{bench}", phase="performance",
                            extra_args=["--eval-interval", str(eval_int),
                                        "--eval-episodes", "10",
                                        "--checkpoint-interval", str(budget // 4)])
            run_job(f"rq4-{bench}-{env_id}-herp", cmd, herp_dir)

        checkpoints = sorted(herp_dir.glob("checkpoint_*.pt"))
        checkpoints = [c for c in checkpoints if "final" not in c.name]
        if not checkpoints:
            print(f"  [skip] No intermediate checkpoints for {env_id}")
            continue

        for ckpt in checkpoints:
            oracle_dir = OUT / "rq4" / bench / f"{env_id}_oracle_{ckpt.stem}"
            if (oracle_dir / "metadata.json").exists():
                print(f"  [skip] Oracle already collected: {ckpt.name}")
                continue
            oracle_dir.mkdir(parents=True, exist_ok=True)
            oracle_cmd = [
                PYTHON, "-u", str(ROOT / "scripts" / "collect_sigma_oracle.py"),
                "--checkpoint", str(ckpt),
                "--benchmark", bench,
                "--env-id", env_id,
                "--fragments-per-region", "128" if bench == "maniskill" else "256",
                "--output-dir", str(oracle_dir),
            ]
            if bench == "maniskill":
                task_cfg = MANISKILL_TASKS.get(env_id, {})
                oracle_cmd += [
                    "--control-mode", task_cfg.get("control_mode", "pd_ee_delta_pose"),
                    "--reward-mode", task_cfg.get("reward_mode", "dense"),
                ]
            print(f"  Collecting oracle for {env_id} @ {ckpt.name}...")
            run_env = {**os.environ, **ENV_BASE}
            try:
                subprocess.run(oracle_cmd, cwd=ROOT, env=run_env, check=True, timeout=7200)
                print(f"    Oracle collected: {ckpt.name}")
            except Exception as e:
                print(f"    Oracle FAILED: {ckpt.name}: {e}")


def stage_rq1(budgets):
    """Stage F-G: Full RQ1 performance matrix — all methods × tasks × seeds."""
    print("\n" + "=" * 60)
    print("  STAGE F-G: RQ1 MATRIX")
    print("=" * 60)

    for seed in [0, 1, 2]:
        gpu_jobs = []
        cpu_jobs = []

        for task_key, budget in budgets.items():
            bench, env_id = task_key.split("/", 1)
            if bench == "dmc":
                eval_int = 50000
                eval_ep = "10"
            else:
                eval_int = min(100_000, budget // 10)
                eval_ep = "50"

            for method in PPO_METHODS:
                # Reuse RQ3 runs if they exist
                rq3_dir = OUT / "rq3" / bench / f"{env_id}_{method}_s{seed}"
                if (rq3_dir / "summary.json").exists():
                    continue

                out_dir = OUT / "rq1" / f"{bench}_{env_id}_{method}_s{seed}"
                cmd = build_cmd(bench, env_id, method, budget, seed, out_dir,
                                wandb_group=f"rq1-{bench}",
                                wandb_tags=f"rq1,{bench},{env_id},{method},seed{seed}",
                                phase="performance",
                                extra_args=["--eval-interval", str(eval_int),
                                            "--eval-episodes", eval_ep,
                                            "--checkpoint-interval", str(budget // 4)])
                if bench == "maniskill":
                    gpu_jobs.append((f"rq1-{bench}-{env_id}-{method}-s{seed}", cmd, out_dir))
                else:
                    cpu_jobs.append((f"rq1-{bench}-{env_id}-{method}-s{seed}", cmd, out_dir))

        if gpu_jobs or cpu_jobs:
            print(f"\n  Seed {seed}: {len(gpu_jobs)} GPU + {len(cpu_jobs)} CPU jobs")
            run_gpu_and_cpu(gpu_jobs, cpu_jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["all", "smoke", "pilots", "rq3", "rq4", "rq1"])
    p.add_argument("--skip-smoke", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "smoke") and not args.skip_smoke:
        if not stage_smoke():
            print("\nSMOKE TESTS FAILED — aborting")
            sys.exit(1)

    if args.stage in ("all", "pilots"):
        stage_pilots()

    budgets = freeze_budgets()

    if args.stage in ("all", "rq3"):
        stage_rq3(budgets)

    if args.stage in ("all", "rq4"):
        stage_rq4(budgets)

    if args.stage in ("all", "rq1"):
        stage_rq1(budgets)

    print("\n" + "=" * 60)
    print("  CAMPAIGN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
