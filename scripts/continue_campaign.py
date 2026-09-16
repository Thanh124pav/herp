"""Continue the HERP final campaign from where it left off.

Runs remaining jobs in priority order:
1. RQ3 ManiSkill remaining seeds (GPU)
2. RQ4 ManiSkill oracle collection (GPU)
3. RQ1 ManiSkill baselines (GPU)
4. RQ1 LiftPegUpright (GPU)

CPU jobs run in parallel with GPU jobs when possible.
"""
from __future__ import annotations
import json, os, subprocess, sys, time, concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_v3.py")
OUT = ROOT / "outputs" / "final_campaign"

ENV_BASE = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl")

MANISKILL_TASKS = {
    "PickCube-v1":       dict(num_envs=256, control_mode="pd_ee_delta_pose", reward_mode="dense"),
    "LiftPegUpright-v1": dict(num_envs=256, control_mode="pd_joint_delta_pos", reward_mode="normalized_dense"),
}

def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"

def run_job(name, cmd, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"  [skip] {name}", flush=True)
        return True
    run_env = {**os.environ, **ENV_BASE}
    print(f"\n  [start] {name}", flush=True)
    t0 = time.time()
    with open(out / "console.log", "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=run_env, stdout=lf, stderr=subprocess.STDOUT)
        code = proc.wait()
    elapsed = time.time() - t0
    ok = code == 0 and (out / "summary.json").exists()
    print(f"  [{'ok' if ok else 'FAIL'}] {name} ({elapsed:.0f}s)", flush=True)
    return ok


def build_ms_cmd(env_id, method, budget, seed, output_dir, phase="performance",
                 wandb_group="", eval_episodes="50"):
    tc = MANISKILL_TASKS.get(env_id, {})
    ne = tc.get("num_envs", 256)
    budget = -(-budget // ne) * ne  # round up to multiple of num_envs
    return [
        PYTHON, "-u", TRAIN,
        "--phase", phase, "--benchmark", "maniskill",
        "--env-id", env_id, "--method", method, "--root-floor", "0.15",
        "--seed", str(seed), "--total-timesteps", str(budget),
        "--output-dir", str(output_dir),
        "--num-envs", str(ne), "--num-eval-envs", "16",
        "--control-mode", tc.get("control_mode", "pd_ee_delta_pose"),
        "--reward-mode", tc.get("reward_mode", "dense"),
        "--eval-interval", str(min(100_000, budget // 10)),
        "--eval-episodes", eval_episodes,
        "--checkpoint-interval", str(budget // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", wandb_group or f"{phase}-maniskill",
        "--wandb-run-name", f"{phase}-maniskill-{env_id}-{method}-s{seed}",
        "--wandb-tags", f"{phase},maniskill,{env_id},{method},seed{seed}",
        "--auto-resume",
    ]


def main():
    budgets = json.loads((OUT / "task_budgets.json").read_text())
    completed = 0
    total = 0

    # Phase 1: Wait for RQ3 ManiSkill remaining seeds (managed by existing orchestrator)
    print("="*60)
    print("  Phase 1: Waiting for RQ3 ManiSkill seeds to complete")
    print("="*60)
    import time as _time
    for seed in [0, 1, 2]:
        for method in ["herp", "herp_p", "herp_sigma", "uniform"]:
            env_id = "PickCube-v1"
            out_dir = OUT / "rq3" / "maniskill" / f"{env_id}_{method}_s{seed}"
            total += 1
            if (out_dir / "summary.json").exists():
                completed += 1
            else:
                print(f"  Waiting for {method}_s{seed}...")
    # Poll until all RQ3 ManiSkill jobs complete
    while True:
        all_done = True
        for seed in [0, 1, 2]:
            for method in ["herp", "herp_p", "herp_sigma", "uniform"]:
                if not (OUT / "rq3" / "maniskill" / f"PickCube-v1_{method}_s{seed}" / "summary.json").exists():
                    all_done = False
                    break
            if not all_done:
                break
        if all_done:
            print("  All RQ3 ManiSkill runs complete!")
            break
        _time.sleep(60)
        print("  ...still waiting", flush=True)

    # Phase 2: RQ4 ManiSkill oracle
    print("\n" + "="*60)
    print("  Phase 2: RQ4 ManiSkill oracle")
    print("="*60)
    for seed in [0, 1, 2]:
        herp_dir = OUT / "rq3" / "maniskill" / f"PickCube-v1_herp_s{seed}"
        checkpoints = sorted(herp_dir.glob("checkpoint_*.pt")) if herp_dir.exists() else []
        checkpoints = [c for c in checkpoints if "final" not in c.name]
        for ckpt in checkpoints:
            oracle_dir = OUT / "rq4" / "maniskill" / f"PickCube-v1_s{seed}_{ckpt.stem}"
            total += 1
            if (oracle_dir / "metadata.json").exists():
                completed += 1
                continue
            oracle_dir.mkdir(parents=True, exist_ok=True)
            tc = MANISKILL_TASKS["PickCube-v1"]
            cmd = [
                PYTHON, "-u", str(ROOT / "scripts" / "collect_sigma_oracle.py"),
                "--checkpoint", str(ckpt),
                "--benchmark", "maniskill", "--env-id", "PickCube-v1",
                "--fragments-per-region", "128",
                "--output-dir", str(oracle_dir),
                "--control-mode", tc["control_mode"],
                "--reward-mode", tc["reward_mode"],
            ]
            print(f"\n  [oracle] s{seed} {ckpt.name}...")
            try:
                subprocess.run(cmd, cwd=ROOT, env={**os.environ, **ENV_BASE},
                               check=True, timeout=7200)
                completed += 1
            except Exception as e:
                print(f"    FAILED: {e}")

    # Phase 3: RQ1 ManiSkill baselines
    print("\n" + "="*60)
    print("  Phase 3: RQ1 ManiSkill baselines")
    print("="*60)
    for env_id in ["PickCube-v1", "LiftPegUpright-v1"]:
        budget_key = f"maniskill/{env_id}"
        budget = budgets.get(budget_key, 2_000_000)
        for seed in [0, 1, 2]:
            for method in ["ppo", "rnd", "disagreement", "herp"]:
                total += 1
                # Check if already done via rq3
                rq3_dir = OUT / "rq3" / "maniskill" / f"{env_id}_{method}_s{seed}"
                if (rq3_dir / "summary.json").exists():
                    completed += 1
                    continue
                out_dir = OUT / "rq1" / f"maniskill_{env_id}_{method}_s{seed}"
                if (out_dir / "summary.json").exists():
                    completed += 1
                    continue
                cmd = build_ms_cmd(env_id, method, budget, seed, out_dir,
                                   wandb_group=f"rq1-maniskill")
                run_job(f"rq1-ms-{env_id}-{method}-s{seed}", cmd, out_dir)

    print(f"\n{'='*60}")
    print(f"  DONE: {completed}/{total} jobs completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
