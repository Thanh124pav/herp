"""Poll BRO output dirs for eval results and create summary.json when done.

BRO saves eval results to {save_dir}/{seed}.txt as (step, return) pairs.
This script polls until all BRO processes finish, then creates summaries.
"""
import json, time, os
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "final_campaign"

TASKS = ["walker-run", "cheetah-run"]
SEEDS = range(5)
BUDGET = 500_000


def bro_running():
    count = 0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
            if "native_train.py" in cmdline and "bro" in cmdline.lower():
                count += 1
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return count


def harvest():
    created = 0
    missing = 0
    for task in TASKS:
        for seed in SEEDS:
            out_dir = OUT / "rq1" / f"dmc_{task}_bro_s{seed}"
            summary = out_dir / "summary.json"

            if summary.exists():
                continue

            eval_file = out_dir / f"{seed}.txt"
            if not eval_file.exists():
                missing += 1
                print(f"  [no eval] {task} s{seed}")
                continue

            data = np.loadtxt(eval_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if len(data) == 0:
                missing += 1
                continue

            final_step = int(data[-1, 0])
            final_return = float(data[-1, 1])

            summary_data = {
                "method": "BRO",
                "task": task,
                "seed": seed,
                "total_timesteps": BUDGET,
                "final_evaluation": {
                    "eval_return": final_return,
                    "eval_success": 0,
                    "step": final_step,
                },
                "source": "locked_BRO",
            }
            summary.write_text(json.dumps(summary_data, indent=2))
            created += 1
            print(f"  [created] {task} s{seed}: eval_return={final_return:.1f} at step {final_step}")

    return created, missing


def main():
    print("=" * 60)
    print("  BRO Result Harvester")
    print("=" * 60, flush=True)

    while True:
        n = bro_running()
        created, missing = harvest()
        print(f"  BRO processes: {n}, created: {created}, missing: {missing}", flush=True)

        if n == 0:
            harvest()
            print("  All BRO processes finished. Final harvest done.", flush=True)
            break

        time.sleep(300)

    print("Done.")


if __name__ == "__main__":
    main()
