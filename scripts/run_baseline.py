"""Experiment 4 -- full B0-B7 routing comparison on matched budgets (PLAN.md 17, 19.4, 20).

Runs the complete baseline suite with identical population size, total
interactions, backbone, and routing bandwidth, over one or more seeds, then
writes a comparison table + bar chart with error bars.

Baselines (PLAN.md section 17):
    B0 single_sac   -- one SAC, matched TOTAL interaction budget (section 20)
    B1 no_share     -- independent population
    B2 share_all    -- shared replay
    B3 random       -- random cross-policy routing
    B4 td_priority  -- SUPER-style global TD-error sharing
    B5 qmp_style    -- QMP-style receiver-Q behavior selection
    B6 greedy       -- greedy best-donor experience routing
    B7 uot          -- full method: UOT experience routing

    python scripts/run_baseline.py --env metaworld:push-v3 --population-size 8 \
        --total-env-steps 800000 --hidden 256 --utd 4 --device cuda --obs-norm \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402

# label -> (router, is_single_sac)
BASELINES = {
    "B0_single_sac": ("no_share", True),
    "B1_no_share": ("no_share", False),
    "B2_share_all": ("share_all", False),
    "B3_random": ("random", False),
    "B4_td_priority": ("td_priority", False),
    "B5_qmp_style": ("qmp_style", False),
    "B6_greedy": ("greedy", False),
    "B7_uot": ("uot", False),
}


def env_factory_for(name: str, horizon: int | None):
    if name == "synthetic":
        from experience_routing.population.population import default_env_factory
        return default_env_factory
    if name.startswith("metaworld"):
        task = name.split(":", 1)[1] if ":" in name else "reach-v3"
        from experience_routing.envs.metaworld_env import make_metaworld_factory
        return make_metaworld_factory(task, horizon=horizon)
    raise ValueError(f"unknown env '{name}'")


def build_config(args, router: str, single_sac: bool, seed: int) -> PipelineConfig:
    return PipelineConfig(
        population_size=1 if single_sac else args.population_size,
        total_env_steps=args.total_env_steps,  # matched TOTAL interactions (section 20)
        router=router,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        routing_interval=args.routing_interval,
        refresh_interval=args.refresh_interval,
        segmenter_fit_after=args.segmenter_fit_after,
        episodes_per_policy=args.episodes,
        eps_experience=args.eps_experience,
        max_experiences=args.max_experiences,
        min_success_support=args.min_success_support,
        eps_pre=args.eps_pre, eps_effect=args.eps_effect, horizon=args.competence_horizon,
        budget_chunks=args.budget_chunks,
        hidden=args.hidden, utd=args.utd, device=args.device, obs_norm=args.obs_norm, lr=args.lr,
        seed=seed, verbose=args.verbose,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines", nargs="+", default=list(BASELINES),
                   help=f"subset of {list(BASELINES)}")
    p.add_argument("--env", default="synthetic")
    p.add_argument("--metaworld-horizon", type=int, default=None)
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--total-env-steps", type=int, default=60_000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--routing-interval", type=int, default=5_000)
    p.add_argument("--refresh-interval", type=int, default=10_000)
    p.add_argument("--segmenter-fit-after", type=int, default=6_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eps-experience", type=float, default=0.8)
    p.add_argument("--max-experiences", type=int, default=None)
    p.add_argument("--min-success-support", type=int, default=3)
    p.add_argument("--eps-pre", type=float, default=2.0)
    p.add_argument("--eps-effect", type=float, default=1.5)
    p.add_argument("--competence-horizon", type=int, default=8)
    p.add_argument("--budget-chunks", type=int, default=32)
    # backbone
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--obs-norm", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--outdir", default="outputs/comparison")
    args = p.parse_args()

    env_factory = env_factory_for(args.env, args.metaworld_horizon)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # per_run[label][seed] = metrics
    table: dict = {}
    for label in args.baselines:
        router, single = BASELINES[label]
        table[label] = {"router": router, "single_sac": single, "seeds": {}}
        for seed in args.seeds:
            print(f"\n===== {label} (router={router}, single={single}) seed={seed} =====")
            cfg = build_config(args, router, single, seed)
            res = OnlineTrainer(cfg, env_factory=env_factory).train()
            s = res["population_summary"]
            table[label]["seeds"][seed] = {
                "mean_success": s["mean_success"],
                "worst_success": s["worst_success"],
                "best_success": s["best_success"],
                "mean_return": s["mean_return"],
                "routed_chunks": res["route_stats"]["chunk_count"],
                "positive_transfer_rate": res["transfer"]["positive_transfer_rate"],
                "num_valid_experiences": len(res["experience_ids"]),
            }
            (out / "comparison.json").write_text(json.dumps(table, indent=2, default=float))

    # aggregate mean +/- std across seeds
    agg = {}
    for label, d in table.items():
        rows = list(d["seeds"].values())
        agg[label] = {
            m: {"mean": float(np.mean([r[m] for r in rows])),
                "std": float(np.std([r[m] for r in rows]))}
            for m in ("mean_success", "worst_success", "mean_return")
        }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, default=float))

    # bar chart: mean success (+/- std over seeds), worst-policy success
    labels = list(table)
    ms = [agg[l]["mean_success"]["mean"] for l in labels]
    ms_e = [agg[l]["mean_success"]["std"] for l in labels]
    ws = [agg[l]["worst_success"]["mean"] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 5))
    ax.bar(x - 0.2, ms, 0.4, yerr=ms_e, capsize=4, label="mean success",
           color=["crimson" if l == "B7_uot" else "tab:blue" for l in labels])
    ax.bar(x + 0.2, ws, 0.4, label="worst-policy success", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("success rate")
    ax.set_title(f"B0-B7 comparison  ({args.env}, N={args.population_size}, "
                 f"{args.total_env_steps} total steps, {len(args.seeds)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)

    print("\n=== SUMMARY (matched budget, mean +/- std over seeds) ===")
    for label in labels:
        a = agg[label]
        print(f"{label:16s} mean_success={a['mean_success']['mean']:.3f}"
              f"+/-{a['mean_success']['std']:.3f}  "
              f"worst={a['worst_success']['mean']:.3f}  "
              f"mean_return={a['mean_return']['mean']:.1f}")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
