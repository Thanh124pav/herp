"""Experiment 4 -- routing comparison on matched budgets (PLAN.md sections 17, 19.4).

Runs several routers with identical population size, total interactions, and
routing bandwidth, then writes a comparison table + bar chart.

    python scripts/run_baseline.py --routers no_share share_all random greedy uot \
        --total-env-steps 60000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routers", nargs="+",
                   default=["no_share", "share_all", "random", "td_priority", "greedy", "uot"])
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--total-env-steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/comparison")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for router in args.routers:
        print(f"\n===== router: {router} =====")
        cfg = PipelineConfig(
            population_size=args.population_size,
            total_env_steps=args.total_env_steps,
            router=router,
            seed=args.seed,
            segmenter_fit_after=6_000,
        )
        res = OnlineTrainer(cfg).train()
        table[router] = {
            "population_summary": res["population_summary"],
            "budget": res["budget"],
            "routing_time": res["timing"]["routing_time"],
        }

    (out / "comparison.json").write_text(json.dumps(table, indent=2, default=float))

    routers = list(table)
    mean_succ = [table[r]["population_summary"]["mean_success"] for r in routers]
    worst_succ = [table[r]["population_summary"]["worst_success"] for r in routers]
    x = range(len(routers))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.2 for i in x], mean_succ, width=0.4, label="mean success")
    ax.bar([i + 0.2 for i in x], worst_succ, width=0.4, label="worst-policy success")
    ax.set_xticks(list(x))
    ax.set_xticklabels(routers, rotation=20)
    ax.set_ylabel("success rate")
    ax.set_title(f"Routing comparison (N={args.population_size}, "
                 f"{args.total_env_steps} total env steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)
    plt.close(fig)

    print("\n=== SUMMARY (matched budget) ===")
    for r in routers:
        s = table[r]["population_summary"]
        print(f"{r:12s} mean_success={s['mean_success']:.3f} "
              f"worst={s['worst_success']:.3f} mean_return={s['mean_return']:.1f}")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
