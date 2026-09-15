"""Generate RQ1 main table: mean±std across seeds, per PLAN.md §15.

Reads summary.json from RQ3/RQ1 run directories. RQ3 HERP runs double
as RQ1 HERP entries (reuse rule §7.3). Outputs JSON + markdown table.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def find_run(campaign, bench, env_id, method, seed):
    """Find a completed run's summary, checking rq3 then rq1."""
    candidates = [
        campaign / "rq3" / bench / f"{env_id}_{method}_s{seed}" / "summary.json",
        campaign / "rq1" / f"{bench}_{env_id}_{method}_s{seed}" / "summary.json",
        campaign / "rq1" / bench / f"{env_id}_{method}_s{seed}" / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", default="outputs/final_campaign")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", default="outputs/final_campaign/rq1/main_table.json")
    args = p.parse_args()

    campaign = Path(args.campaign_dir)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    budgets = json.loads((campaign / "task_budgets.json").read_text())

    methods_by_group = {
        "PPO": ["ppo", "rnd", "disagreement", "herp"],
    }

    tasks = list(budgets.keys())

    table = {}
    for group, methods in methods_by_group.items():
        for method in methods:
            row = {}
            for task_key in tasks:
                bench, env_id = task_key.split("/", 1)
                values = []
                for seed in args.seeds:
                    summary = find_run(campaign, bench, env_id, method, seed)
                    if summary:
                        ev = summary.get("final_evaluation", {})
                        if bench == "dmc":
                            values.append(ev.get("eval_return", 0))
                        else:
                            values.append(ev.get("eval_success", 0))

                if values:
                    row[task_key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "n": len(values),
                        "values": values,
                    }
                else:
                    row[task_key] = None

            table[f"{group}/{method}"] = row

    out_file.write_text(json.dumps(table, indent=2))

    # Print markdown table
    print("\n## RQ1 Main Table\n")
    header = "| Method |"
    sep = "|--------|"
    for t in tasks:
        short = t.split("/")[1]
        header += f" {short} |"
        sep += "--------|"
    print(header)
    print(sep)

    for key, row in table.items():
        group, method = key.split("/")
        line = f"| {method} |"
        for t in tasks:
            cell = row.get(t)
            if cell is None:
                line += " N/A |"
            else:
                line += f" {cell['mean']:.3f}±{cell['std']:.3f} |"
        print(line)

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
