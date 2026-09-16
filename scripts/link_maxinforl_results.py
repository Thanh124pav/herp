"""Create summary.json for MaxInfoRL runs in the expected dmc_* paths.

Old script writes to: outputs/final_campaign/rq1/external/maxinforl_{task}_s{seed}
New scripts expect:   outputs/final_campaign/rq1/dmc_{task}_maxinforl_s{seed}

Reads evaluations.npz from the old path and creates summary.json in the new path.
"""
import json, os
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "final_campaign"

TASKS = ["walker-run", "cheetah-run"]
SEEDS = range(5)
BUDGET = 500_000


def main():
    for task in TASKS:
        for seed in SEEDS:
            old_dir = OUT / "rq1" / "external" / f"maxinforl_{task}_s{seed}"
            new_dir = OUT / "rq1" / f"dmc_{task}_maxinforl_s{seed}"

            if (new_dir / "summary.json").exists():
                print(f"  [skip] {task} s{seed} — summary exists")
                continue

            eval_file = old_dir / "evaluations.npz"
            if not eval_file.exists():
                print(f"  [wait] {task} s{seed} — not finished yet")
                continue

            data = np.load(eval_file)
            results = data["results"]  # shape (n_evals, n_eval_episodes)
            timesteps = data["timesteps"]

            if len(results) == 0:
                print(f"  [empty] {task} s{seed}")
                continue

            final_return = float(results[-1].mean())
            final_step = int(timesteps[-1])

            new_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "method": "MaxInfoRL",
                "task": task,
                "seed": seed,
                "total_timesteps": BUDGET,
                "final_evaluation": {
                    "eval_return": final_return,
                    "eval_success": 0,
                    "step": final_step,
                },
                "source": "locked_MaxInfoRL",
            }
            (new_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"  [created] {task} s{seed}: eval_return={final_return:.1f} at step {final_step}")

    print("\nDone.")


if __name__ == "__main__":
    main()
