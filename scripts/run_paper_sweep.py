"""Paper experiment driver — multi-seed routing comparison (PLAN.md sections 19.4, 20).

Runs each router across several seeds at a *matched budget* (same N, same total
env interactions, same routing bandwidth), then aggregates mean +/- std across
seeds and writes a comparison table (JSON + CSV) and a bar chart with error bars.

    python scripts/run_paper_sweep.py \
        --routers no_share share_all random td_priority greedy uot \
        --population-size 8 --total-env-steps 120000 --seeds 0 1 2 \
        --outdir outputs/paper/headline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402


METRICS = ["mean_success", "worst_success", "best_success", "mean_return", "worst_return"]


def run_one(router: str, n: int, steps: int, seed: int, budget_chunks: int) -> dict:
    cfg = PipelineConfig(
        population_size=n,
        total_env_steps=steps,
        router=router,
        seed=seed,
        budget_chunks=budget_chunks,
        segmenter_fit_after=6_000,
        verbose=False,
    )
    t0 = time.time()
    res = OnlineTrainer(cfg).train()
    wall = time.time() - t0
    row = dict(res["population_summary"])
    row["routing_time"] = float(res["timing"].get("routing_time", 0.0))
    row["routed_chunks_total"] = int(res["budget"].get("routed_chunks_total", 0))
    row["num_valid_experiences"] = int(len(res["experience_ids"]))
    row["wall_time"] = wall
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routers", nargs="+",
                   default=["no_share", "share_all", "random", "td_priority", "greedy", "uot"])
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--total-env-steps", type=int, default=120_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget-chunks", type=int, default=32)
    p.add_argument("--outdir", default="outputs/paper/headline")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw: dict[str, dict[int, dict]] = {r: {} for r in args.routers}
    for router in args.routers:
        for seed in args.seeds:
            print(f"[sweep] router={router:12s} seed={seed} N={args.population_size} "
                  f"steps={args.total_env_steps} ...", flush=True)
            row = run_one(router, args.population_size, args.total_env_steps,
                          seed, args.budget_chunks)
            raw[router][seed] = row
            print(f"        -> mean_success={row['mean_success']:.3f} "
                  f"worst={row['worst_success']:.3f} "
                  f"mean_return={row['mean_return']:.1f} "
                  f"routed={row['routed_chunks_total']} "
                  f"({row['wall_time']:.0f}s)", flush=True)
            # checkpoint after every run so a crash keeps partial results
            (out / "raw.json").write_text(json.dumps(raw, indent=2, default=float))

    # aggregate mean +/- std across seeds
    agg: dict[str, dict] = {}
    for router in args.routers:
        rows = list(raw[router].values())
        agg[router] = {}
        for m in METRICS + ["routing_time", "routed_chunks_total", "num_valid_experiences"]:
            vals = np.array([r[m] for r in rows], dtype=float)
            agg[router][m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, default=float))

    # CSV for the paper table
    lines = ["router,seeds,mean_success,mean_success_std,worst_success,worst_success_std,"
             "mean_return,mean_return_std,routed_chunks,routing_time_s"]
    for router in args.routers:
        a = agg[router]
        lines.append(
            f"{router},{len(args.seeds)},"
            f"{a['mean_success']['mean']:.4f},{a['mean_success']['std']:.4f},"
            f"{a['worst_success']['mean']:.4f},{a['worst_success']['std']:.4f},"
            f"{a['mean_return']['mean']:.3f},{a['mean_return']['std']:.3f},"
            f"{a['routed_chunks_total']['mean']:.1f},{a['routing_time']['mean']:.4f}"
        )
    (out / "table.csv").write_text("\n".join(lines) + "\n")

    # bar chart: mean & worst success with std error bars
    routers = args.routers
    x = np.arange(len(routers))
    ms = [agg[r]["mean_success"]["mean"] for r in routers]
    ms_e = [agg[r]["mean_success"]["std"] for r in routers]
    ws = [agg[r]["worst_success"]["mean"] for r in routers]
    ws_e = [agg[r]["worst_success"]["std"] for r in routers]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, ms, width=0.4, yerr=ms_e, capsize=4, label="mean success")
    ax.bar(x + 0.2, ws, width=0.4, yerr=ws_e, capsize=4, label="worst-policy success")
    ax.set_xticks(x)
    ax.set_xticklabels(routers, rotation=20)
    ax.set_ylabel("success rate")
    ax.set_title(f"Routing comparison  N={args.population_size}, "
                 f"{args.total_env_steps} env steps, {len(args.seeds)} seeds (mean+/-std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)

    print("\n=== AGGREGATE (mean +/- std across seeds) ===")
    for r in routers:
        a = agg[r]
        print(f"{r:12s} mean_succ={a['mean_success']['mean']:.3f}+/-{a['mean_success']['std']:.3f}  "
              f"worst={a['worst_success']['mean']:.3f}+/-{a['worst_success']['std']:.3f}  "
              f"return={a['mean_return']['mean']:.1f}  routed={a['routed_chunks_total']['mean']:.0f}")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
