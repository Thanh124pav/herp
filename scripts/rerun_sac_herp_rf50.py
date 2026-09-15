"""Rerun SAC+HERP with root_floor=0.50 (was 0.15, caused SAC to underperform).

root_floor=0.15 diverted 85% of budget to region acquisition, starving SAC's
replay buffer of on-distribution data. root_floor=0.50 keeps ~50% root + ~10%
reference ≈ 60% from normal resets.

Covers: DMC (walker-run, cheetah-run) + MetaWorld (button-press-v3) on CPU.
ManiSkill needs GPU — handled separately.
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train_herp_sac_cpu.py")
OUT = ROOT / "outputs" / "final_campaign" / "rq1"

ROOT_FLOOR = 0.50

TASKS = {
    "dmc": {
        "walker-run":  {"budget": 500_000, "gamma": 0.99, "seeds": [0,1,2,3,4]},
        "cheetah-run": {"budget": 500_000, "gamma": 0.99, "seeds": [0,1,2,3,4]},
    },
    "metaworld": {
        "button-press-v3": {"budget": 500_000, "gamma": 0.99, "seeds": [0,1,2]},
    },
}
MAX_PARALLEL = 6


def wandb_project(env_id):
    safe = "".join(c.lower() if c.isalnum() else "-" for c in env_id).strip("-")
    return f"herp-{safe}"


def kill_old_sac_herp():
    """Kill running sac_herp processes (root_floor=0.15)."""
    killed = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "train_herp_sac_cpu.py" in cmdline and "--method\x00sac_herp\x00" in cmdline:
                # Don't kill sac_herp_p or sac_herp_sigma (RQ3 ablation)
                if "sac_herp_p" not in cmdline and "sac_herp_sigma" not in cmdline:
                    pid = int(pid_dir.name)
                    os.kill(pid, signal.SIGTERM)
                    print(f"  [kill] PID {pid}", flush=True)
                    killed += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    if killed:
        time.sleep(3)
    print(f"  Killed {killed} old sac_herp processes", flush=True)


def clean_output(bench, env_id, seed):
    """Remove old run artifacts so we start fresh."""
    d = OUT / f"{bench}_{env_id}_sac_herp_s{seed}"
    if not d.exists():
        return
    for f in d.iterdir():
        if f.name in ("provenance.json",):
            continue
        if f.is_dir():
            import shutil
            shutil.rmtree(f)
        else:
            f.unlink()
    print(f"  [clean] {d.name}", flush=True)


def launch(bench, env_id, task_cfg, seed):
    d = OUT / f"{bench}_{env_id}_sac_herp_s{seed}"
    d.mkdir(parents=True, exist_ok=True)

    if (d / "summary.json").exists():
        print(f"  [skip] {bench}/{env_id} s{seed} (done)", flush=True)
        return None

    cmd = [
        PYTHON, "-u", TRAIN,
        "--benchmark", bench,
        "--env-id", env_id,
        "--method", "sac_herp",
        "--seed", str(seed),
        "--total-timesteps", str(task_cfg["budget"]),
        "--output-dir", str(d),
        "--gamma", str(task_cfg["gamma"]),
        "--root-floor", str(ROOT_FLOOR),
        "--eval-interval", "25000",
        "--eval-episodes", "10",
        "--checkpoint-interval", str(task_cfg["budget"] // 4),
        "--wandb-mode", "online",
        "--wandb-project", wandb_project(env_id),
        "--wandb-group", "rq1-sac-herp-rf50",
        "--wandb-run-name", f"sac_herp-{env_id}-s{seed}-rf50",
        "--wandb-tags", f"rq1,{bench},{env_id},sac_herp,seed{seed},rf50",
    ]
    if bench == "metaworld":
        cmd += ["--reward-mode", "dense"]

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    lf = open(d / "console.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  [launch] {bench}/{env_id} s{seed} PID {proc.pid} (rf={ROOT_FLOOR})", flush=True)
    return proc, lf, (bench, env_id, seed)


def main():
    print("=" * 60)
    print(f"  SAC+HERP rerun: root_floor={ROOT_FLOOR}")
    print(f"  DMC: walker-run, cheetah-run (5 seeds)")
    print(f"  MetaWorld: button-press-v3 (3 seeds)")
    print(f"  Max parallel: {MAX_PARALLEL}")
    print("=" * 60, flush=True)

    kill_old_sac_herp()

    # Clean old outputs
    for bench, tasks in TASKS.items():
        for env_id, cfg in tasks.items():
            for seed in cfg["seeds"]:
                clean_output(bench, env_id, seed)

    # Build queue
    queue = []
    for bench, tasks in TASKS.items():
        for env_id, cfg in tasks.items():
            for seed in cfg["seeds"]:
                queue.append((bench, env_id, cfg, seed))

    active = []
    done = failed = 0

    while queue or active:
        # Reap finished
        still = []
        for proc, lf, info in active:
            ret = proc.poll()
            if ret is not None:
                lf.close()
                b, e, s = info
                od = OUT / f"{b}_{e}_sac_herp_s{s}"
                if ret == 0 and (od / "summary.json").exists():
                    print(f"  [done] {b}/{e} s{s}", flush=True)
                    done += 1
                else:
                    print(f"  [FAIL] {b}/{e} s{s} (rc={ret})", flush=True)
                    failed += 1
            else:
                still.append((proc, lf, info))
        active = still

        # Launch up to MAX_PARALLEL
        while len(active) < MAX_PARALLEL and queue:
            args = queue.pop(0)
            result = launch(*args)
            if result is None:
                done += 1
            else:
                active.append(result)

        if active:
            time.sleep(30)

    print(f"\n{'=' * 60}")
    print(f"  SAC+HERP rf={ROOT_FLOOR}: {done} ok, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
