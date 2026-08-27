"""Paper extras — emergence artifacts + ablations A5 (population size) and A6
(routing budget), PLAN.md sections 19.1, 19.5.

Each run is a full OnlineTrainer run; independent runs go through a single-thread
process pool. Writes:
  * emergence/  competence+deficit heatmap and Gate-C stats from a no_share N=8 run;
  * ablation_popsize.{json,png}   uot mean/worst success + Gate-C vs N in {2,4,8};
  * ablation_budget.{json,png}    uot mean/worst success vs routing budget_chunks.

    python scripts/run_paper_extras.py --total-env-steps 80000 --seeds 0 1 2 --workers 4
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


def _run(kind: str, n: int, steps: int, seed: int, router: str, budget_chunks: int,
         episodes: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from experience_routing.competence.deficit import deficit_matrix  # noqa: F401
    from experience_routing.evaluation.routing_metrics import complementarity_stats
    from experience_routing.pipeline import OnlineTrainer, PipelineConfig

    cfg = PipelineConfig(
        population_size=n, total_env_steps=steps, router=router, seed=seed,
        budget_chunks=budget_chunks, episodes_per_policy=episodes,
        eval_interval=max(20_000, steps // 4), segmenter_fit_after=6_000, verbose=False,
    )
    res = OnlineTrainer(cfg).train()
    C = res["competence"]
    comp = complementarity_stats(C) if getattr(C, "size", 0) else {}
    row = dict(res["population_summary"])
    row["routed_chunks_total"] = int(res["budget"].get("routed_chunks_total", 0))
    row["num_valid_experiences"] = int(len(res["experience_ids"]))
    row["complementarity"] = comp
    return {"kind": kind, "n": n, "seed": seed, "router": router,
            "budget_chunks": budget_chunks, "row": row}


def _agg(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-env-steps", type=int, default=80_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget-chunks", type=int, default=32)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--pop-sizes", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 32, 128])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", default="outputs/paper/extras")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    steps = args.total_env_steps

    jobs = []
    # A5 population size (uot), scale total steps so per-policy budget is matched
    for n in args.pop_sizes:
        per_policy = steps // 8
        for s in args.seeds:
            jobs.append(("popsize", n, per_policy * n, s, "uot", args.budget_chunks))
    # A6 routing budget (uot, N=8)
    for b in args.budgets:
        for s in args.seeds:
            jobs.append(("budget", 8, steps, s, "uot", b))

    print(f"[extras] {len(jobs)} runs, {args.workers} workers", flush=True)
    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, k, n, st, s, r, b, args.episodes)
                for (k, n, st, s, r, b) in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            row = res["row"]
            tag = f"{res['kind']} n={res['n']} b={res['budget_chunks']} seed={res['seed']}"
            print(f"[{i}/{len(jobs)}] {tag:38s} mean={row['mean_success']:.3f} "
                  f"worst={row['worst_success']:.3f} valid={row['num_valid_experiences']}",
                  flush=True)
            (out / "extras_raw.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[extras] done in {time.time() - t0:.0f}s", flush=True)

    # --- A5 aggregate + plot ---
    pop = {}
    for n in args.pop_sizes:
        rows = [r["row"] for r in results if r["kind"] == "popsize" and r["n"] == n]
        if rows:
            pop[n] = {
                "mean_success": _agg(rows, "mean_success"),
                "worst_success": _agg(rows, "worst_success"),
                "frac_distinct_best": _agg(
                    [{"v": r["complementarity"].get("frac_experiences_distinct_best", 0.0)}
                     for r in rows], "v"),
            }
    (out / "ablation_popsize.json").write_text(json.dumps(pop, indent=2, default=float))
    if pop:
        ns = sorted(pop)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(ns, [pop[n]["mean_success"]["mean"] for n in ns],
                    yerr=[pop[n]["mean_success"]["std"] for n in ns], marker="o", label="mean success")
        ax.errorbar(ns, [pop[n]["worst_success"]["mean"] for n in ns],
                    yerr=[pop[n]["worst_success"]["std"] for n in ns], marker="s", label="worst-policy success")
        ax.set_xlabel("population size N"); ax.set_ylabel("success rate")
        ax.set_title("A5 — UOT routing vs population size (per-policy budget matched)")
        ax.set_xticks(ns); ax.legend(); fig.tight_layout()
        fig.savefig(out / "ablation_popsize.png", dpi=130); plt.close(fig)

    # --- A6 aggregate + plot ---
    bud = {}
    for b in args.budgets:
        rows = [r["row"] for r in results if r["kind"] == "budget" and r["budget_chunks"] == b]
        if rows:
            bud[b] = {
                "mean_success": _agg(rows, "mean_success"),
                "worst_success": _agg(rows, "worst_success"),
                "routed_chunks_total": _agg(rows, "routed_chunks_total"),
            }
    (out / "ablation_budget.json").write_text(json.dumps(bud, indent=2, default=float))
    if bud:
        bs = sorted(bud)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(bs, [bud[b]["mean_success"]["mean"] for b in bs],
                    yerr=[bud[b]["mean_success"]["std"] for b in bs], marker="o", label="mean success")
        ax.errorbar(bs, [bud[b]["worst_success"]["mean"] for b in bs],
                    yerr=[bud[b]["worst_success"]["std"] for b in bs], marker="s", label="worst-policy success")
        ax.set_xlabel("routing budget (chunks / interval)"); ax.set_ylabel("success rate")
        ax.set_xscale("log", base=2); ax.set_xticks(bs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title("A6 — UOT routing vs routing budget (N=8)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out / "ablation_budget.png", dpi=130); plt.close(fig)

    print("\n=== A5 population size (uot) ===")
    for n in sorted(pop):
        a = pop[n]
        print(f"N={n}: mean={a['mean_success']['mean']:.3f} worst={a['worst_success']['mean']:.3f} "
              f"frac_distinct_best={a['frac_distinct_best']['mean']:.2f}")
    print("=== A6 routing budget (uot, N=8) ===")
    for b in sorted(bud):
        a = bud[b]
        print(f"budget={b}: mean={a['mean_success']['mean']:.3f} worst={a['worst_success']['mean']:.3f} "
              f"routed={a['routed_chunks_total']['mean']:.0f}")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
