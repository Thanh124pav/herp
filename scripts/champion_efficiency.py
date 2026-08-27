"""Champion sample-efficiency — does competition + learning-from-others forge a
better/faster BEST model than isolated training?

Metric shift: the objective is the single best policy (best_success), not the
population. For each router we log the best-policy success trajectory over env
steps (fine eval), then report steps-to-threshold and area-under-curve of the
champion, averaged over seeds. This is the decisive comparison of routing vs
no_share when the goal is "pick the best model".

    python scripts/champion_efficiency.py --routers no_share uot greedy share_all \
        --population-size 8 --total-env-steps 80000 --eval-interval 8000 \
        --seeds 0 1 2 --workers 4 --outdir outputs/paper/champion
"""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THRESHOLD = 0.30  # champion success level we time to reach


def run_cell(router, n, steps, seed, eval_interval, episodes):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch; torch.set_num_threads(1)
    from experience_routing.pipeline import OnlineTrainer, PipelineConfig
    cfg = PipelineConfig(population_size=n, total_env_steps=steps, router=router, seed=seed,
                         eval_interval=eval_interval, episodes_per_policy=episodes,
                         segmenter_fit_after=6_000, verbose=False)
    res = OnlineTrainer(cfg).train()
    h = res["history"]
    steps_axis = list(h["eval_steps"])
    per_policy = h["success"]  # {pid: [succ per eval]}
    T = len(steps_axis)
    best = [max(per_policy[p][t] for p in per_policy) for t in range(T)]
    mean = [float(np.mean([per_policy[p][t] for p in per_policy])) for t in range(T)]
    return {"router": router, "seed": seed, "eval_steps": steps_axis,
            "best": best, "mean": mean,
            "final_best": float(res["population_summary"]["best_success"])}


def steps_to_threshold(steps_axis, best, thr):
    for s, b in zip(steps_axis, best):
        if b >= thr:
            return s
    return None  # never reached


def auc(steps_axis, curve):
    # normalized area under the curve (mean success-weight over the run)
    return float(np.trapz(curve, steps_axis) / (steps_axis[-1] - steps_axis[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routers", nargs="+", default=["no_share", "uot", "greedy", "share_all"])
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--total-env-steps", type=int, default=80_000)
    p.add_argument("--eval-interval", type=int, default=8_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", default="outputs/paper/champion")
    args = p.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    cells = [(r, s) for r in args.routers for s in args.seeds]
    print(f"[champion] {len(cells)} cells, {args.workers} workers, "
          f"eval every {args.eval_interval}", flush=True)
    raw = {r: [] for r in args.routers}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, r, args.population_size, args.total_env_steps, s,
                          args.eval_interval, args.episodes): (r, s) for (r, s) in cells}
        done = 0
        for fut in as_completed(futs):
            res = fut.result(); raw[res["router"]].append(res); done += 1
            stt = steps_to_threshold(res["eval_steps"], res["best"], THRESHOLD)
            print(f"[{done}/{len(cells)}] {res['router']:12s} seed={res['seed']} "
                  f"final_best={res['final_best']:.3f} "
                  f"steps_to_{THRESHOLD:g}={stt}", flush=True)
            (out / "champion_raw.json").write_text(json.dumps(raw, indent=2, default=float))
    print(f"[champion] done in {time.time()-t0:.0f}s", flush=True)

    # aggregate on a common step axis (assume identical across seeds)
    summary = {}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"no_share": "#3b6fb0", "uot": "#c2453d", "greedy": "#b3841f", "share_all": "#2f8a5b",
              "random": "#7a5bd0", "td_priority": "#5c6470"}
    for r in args.routers:
        runs = raw[r]
        axis = runs[0]["eval_steps"]
        B = np.array([x["best"] for x in runs])   # [seeds, T]
        M = np.array([x["mean"] for x in runs])
        bmean, bstd = B.mean(0), B.std(0)
        stt = [steps_to_threshold(axis, x["best"], THRESHOLD) for x in runs]
        stt_reached = [s for s in stt if s is not None]
        summary[r] = {
            "final_best_mean": float(B[:, -1].mean()), "final_best_std": float(B[:, -1].std()),
            "auc_best": float(np.mean([auc(axis, x["best"]) for x in runs])),
            "auc_mean": float(np.mean([auc(axis, x["mean"]) for x in runs])),
            "steps_to_threshold_mean": (float(np.mean(stt_reached)) if stt_reached else None),
            "seeds_reached_threshold": f"{len(stt_reached)}/{len(runs)}",
        }
        c = colors.get(r, None)
        ax.plot(axis, bmean, marker="o", label=f"{r} (best)", color=c)
        ax.fill_between(axis, bmean - bstd, bmean + bstd, alpha=0.15, color=c)
    ax.axhline(THRESHOLD, ls="--", lw=1, color="#888", label=f"threshold {THRESHOLD:g}")
    ax.set_xlabel("total env steps"); ax.set_ylabel("best-policy (champion) success")
    ax.set_title(f"Champion learning curve — N={args.population_size}, {len(args.seeds)} seeds (mean±std)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out / "champion_curve.png", dpi=130); plt.close(fig)

    (out / "champion_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n=== CHAMPION SUMMARY ===")
    print(f"{'router':12s} {'final_best':>12s} {'AUC(best)':>10s} {'steps→0.3':>12s} {'reached':>9s}")
    for r in args.routers:
        s = summary[r]
        stt = f"{s['steps_to_threshold_mean']:.0f}" if s["steps_to_threshold_mean"] else "—"
        print(f"{r:12s} {s['final_best_mean']:.3f}±{s['final_best_std']:.3f}  "
              f"{s['auc_best']:.4f}   {stt:>10s}   {s['seeds_reached_threshold']:>8s}")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
