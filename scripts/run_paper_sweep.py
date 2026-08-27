"""Paper experiment driver — multi-seed routing comparison (PLAN.md sections 19.4, 20).

Runs each (router, seed) cell at a *matched budget* (same N, same total env
interactions, same routing bandwidth). Cells are independent, so they run in a
process pool; each worker pins torch to a single thread so several cells share
the CPU without oversubscription. Aggregates mean +/- std across seeds and
writes a comparison table (JSON + CSV) and a bar chart with error bars.

    python scripts/run_paper_sweep.py \
        --routers no_share share_all random td_priority greedy uot \
        --population-size 8 --total-env-steps 80000 --seeds 0 1 2 \
        --workers 4 --outdir outputs/paper/headline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS = ["mean_success", "worst_success", "best_success", "mean_return", "worst_return"]


def run_cell(router: str, n: int, steps: int, seed: int, budget_chunks: int,
             episodes: int, eval_interval: int) -> dict:
    # single-threaded torch so N workers fit on N cores without contention
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from experience_routing.pipeline import OnlineTrainer, PipelineConfig

    cfg = PipelineConfig(
        population_size=n,
        total_env_steps=steps,
        router=router,
        seed=seed,
        budget_chunks=budget_chunks,
        episodes_per_policy=episodes,
        eval_interval=eval_interval,
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
    return {"router": router, "seed": seed, "row": row}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routers", nargs="+",
                   default=["no_share", "share_all", "random", "td_priority", "greedy", "uot"])
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--total-env-steps", type=int, default=80_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget-chunks", type=int, default=32)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--eval-interval", type=int, default=20_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", default="outputs/paper/headline")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw: dict[str, dict[int, dict]] = {r: {} for r in args.routers}
    cells = [(r, s) for r in args.routers for s in args.seeds]
    print(f"[sweep] {len(cells)} cells, {args.workers} workers, "
          f"N={args.population_size}, steps={args.total_env_steps}, "
          f"seeds={args.seeds}", flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_cell, r, args.population_size, args.total_env_steps, s,
                      args.budget_chunks, args.episodes, args.eval_interval): (r, s)
            for (r, s) in cells
        }
        done = 0
        for fut in as_completed(futs):
            r, s = futs[fut]
            res = fut.result()
            raw[res["router"]][res["seed"]] = res["row"]
            done += 1
            row = res["row"]
            print(f"[{done}/{len(cells)}] router={r:12s} seed={s} "
                  f"mean_success={row['mean_success']:.3f} worst={row['worst_success']:.3f} "
                  f"return={row['mean_return']:.1f} routed={row['routed_chunks_total']} "
                  f"({row['wall_time']:.0f}s)", flush=True)
            (out / "raw.json").write_text(json.dumps(raw, indent=2, default=float))
    print(f"[sweep] all cells done in {time.time() - t_start:.0f}s", flush=True)

    # aggregate mean +/- std across seeds
    agg: dict[str, dict] = {}
    for router in args.routers:
        rows = list(raw[router].values())
        if not rows:
            continue
        agg[router] = {}
        for m in METRICS + ["routing_time", "routed_chunks_total", "num_valid_experiences"]:
            vals = np.array([r[m] for r in rows], dtype=float)
            agg[router][m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, default=float))

    lines = ["router,seeds,mean_success,mean_success_std,worst_success,worst_success_std,"
             "mean_return,mean_return_std,routed_chunks,routing_time_s"]
    for router in args.routers:
        if router not in agg:
            continue
        a = agg[router]
        lines.append(
            f"{router},{len(raw[router])},"
            f"{a['mean_success']['mean']:.4f},{a['mean_success']['std']:.4f},"
            f"{a['worst_success']['mean']:.4f},{a['worst_success']['std']:.4f},"
            f"{a['mean_return']['mean']:.3f},{a['mean_return']['std']:.3f},"
            f"{a['routed_chunks_total']['mean']:.1f},{a['routing_time']['mean']:.4f}"
        )
    (out / "table.csv").write_text("\n".join(lines) + "\n")

    routers = [r for r in args.routers if r in agg]
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
