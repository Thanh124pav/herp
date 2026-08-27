"""Complete the interrupted A6 sweep: run the missing budget=128 cells, merge
into extras_raw.json, and (re)build the A5/A6 aggregates + plots."""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/paper/extras")
RAW = OUT / "extras_raw.json"


def run_cell(kind, n, steps, seed, router, budget_chunks, episodes):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch; torch.set_num_threads(1)
    from experience_routing.evaluation.routing_metrics import complementarity_stats
    from experience_routing.pipeline import OnlineTrainer, PipelineConfig
    cfg = PipelineConfig(population_size=n, total_env_steps=steps, router=router, seed=seed,
                         budget_chunks=budget_chunks, episodes_per_policy=episodes,
                         eval_interval=max(20_000, steps // 4), segmenter_fit_after=6_000, verbose=False)
    res = OnlineTrainer(cfg).train()
    C = res["competence"]
    comp = complementarity_stats(C) if getattr(C, "size", 0) else {}
    row = dict(res["population_summary"])
    row["routed_chunks_total"] = int(res["budget"].get("routed_chunks_total", 0))
    row["num_valid_experiences"] = int(len(res["experience_ids"]))
    row["complementarity"] = comp
    return {"kind": kind, "n": n, "seed": seed, "router": router,
            "budget_chunks": budget_chunks, "row": row}


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return {"mean": float(v.mean()), "std": float(v.std())}


def main():
    results = json.loads(RAW.read_text())
    have = {(r["kind"], r["n"], r["budget_chunks"], r["seed"]) for r in results}
    missing = [("budget", 8, 80_000, s, "uot", 128) for s in (0, 1, 2)
               if ("budget", 8, 128, s) not in have]
    print(f"{len(results)} runs loaded; {len(missing)} missing", flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(run_cell, k, n, st, s, r, b, 15) for (k, n, st, s, r, b) in missing]
            for fut in as_completed(futs):
                res = fut.result(); results.append(res)
                print(f"  done budget={res['budget_chunks']} seed={res['seed']} "
                      f"mean={res['row']['mean_success']:.3f}", flush=True)
                RAW.write_text(json.dumps(results, indent=2, default=float))

    # A5
    pop = {}
    for n in (2, 4, 8):
        rows = [r["row"] for r in results if r["kind"] == "popsize" and r["n"] == n]
        if rows:
            pop[str(n)] = {"mean_success": agg(rows, "mean_success"),
                           "worst_success": agg(rows, "worst_success"),
                           "frac_distinct_best": agg(
                               [{"v": r["complementarity"].get("frac_experiences_distinct_best", 0.0)}
                                for r in rows], "v")}
    (OUT / "ablation_popsize.json").write_text(json.dumps(pop, indent=2, default=float))
    ns = sorted(pop, key=int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar([int(n) for n in ns], [pop[n]["mean_success"]["mean"] for n in ns],
                yerr=[pop[n]["mean_success"]["std"] for n in ns], marker="o", label="mean success")
    ax.errorbar([int(n) for n in ns], [pop[n]["worst_success"]["mean"] for n in ns],
                yerr=[pop[n]["worst_success"]["std"] for n in ns], marker="s", label="worst-policy success")
    ax.set_xlabel("population size N"); ax.set_ylabel("success rate")
    ax.set_title("A5 — UOT routing vs population size (per-policy budget matched)")
    ax.set_xticks([int(n) for n in ns]); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "ablation_popsize.png", dpi=130); plt.close(fig)

    # A6
    bud = {}
    for b in (8, 32, 128):
        rows = [r["row"] for r in results if r["kind"] == "budget" and r["budget_chunks"] == b]
        if rows:
            bud[str(b)] = {"mean_success": agg(rows, "mean_success"),
                           "worst_success": agg(rows, "worst_success"),
                           "routed_chunks_total": agg(rows, "routed_chunks_total")}
    (OUT / "ablation_budget.json").write_text(json.dumps(bud, indent=2, default=float))
    bs = sorted(bud, key=int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar([int(b) for b in bs], [bud[b]["mean_success"]["mean"] for b in bs],
                yerr=[bud[b]["mean_success"]["std"] for b in bs], marker="o", label="mean success")
    ax.errorbar([int(b) for b in bs], [bud[b]["worst_success"]["mean"] for b in bs],
                yerr=[bud[b]["worst_success"]["std"] for b in bs], marker="s", label="worst-policy success")
    ax.set_xlabel("routing budget (chunks / interval)"); ax.set_ylabel("success rate")
    ax.set_xscale("log", base=2); ax.set_xticks([int(b) for b in bs])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title("A6 — UOT routing vs routing budget (N=8)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "ablation_budget.png", dpi=130); plt.close(fig)

    print("=== A5 ==="); [print(n, pop[n]) for n in ns]
    print("=== A6 ==="); [print(b, bud[b]) for b in bs]
    print("written ablation_popsize/budget .json + .png")


if __name__ == "__main__":
    main()
