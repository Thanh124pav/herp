"""Full MVP run: online routing loop + all section-28 artifacts (PLAN.md section 28).

    python scripts/run_full.py --env synthetic --population-size 4 --router uot

Produces, under ``outputs/<run-name>/``:
  * learning curves (per-policy success + return);
  * experience vocabulary over time (K_t);
  * competence + deficit heatmaps;
  * UOT transport matrix (uot router);
  * routing counts / budget summary (metrics.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experience_routing.evaluation.routing_metrics import (
    complementarity_stats,
    plot_competence_and_deficit,
    plot_learning_curves,
    plot_transport_matrix,
    plot_vocab_over_time,
)
from experience_routing.pipeline import OnlineTrainer, PipelineConfig


def build_config(args) -> PipelineConfig:
    cfg = PipelineConfig(
        population_size=args.population_size,
        total_env_steps=args.total_env_steps,
        router=args.router,
        eval_interval=args.eval_interval,
        routing_interval=args.routing_interval,
        refresh_interval=args.refresh_interval,
        segmenter_fit_after=args.segmenter_fit_after,
        episodes_per_policy=args.episodes,
        eps_experience=args.eps_experience,
        budget_chunks=args.budget_chunks,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return cfg


def env_factory_for(name: str):
    if name == "synthetic":
        from experience_routing.population.population import default_env_factory
        return default_env_factory
    if name.startswith("metaworld"):
        task = name.split(":", 1)[1] if ":" in name else "reach-v2"
        from experience_routing.envs.metaworld_env import make_metaworld_factory
        return make_metaworld_factory(task)
    raise ValueError(f"unknown env '{name}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="synthetic",
                   help="'synthetic' or 'metaworld:<task>' (e.g. metaworld:reach-v2)")
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--router", default="uot",
                   choices=["no_share", "share_all", "random", "td_priority", "greedy", "uot"])
    p.add_argument("--total-env-steps", type=int, default=60_000)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--routing-interval", type=int, default=5_000)
    p.add_argument("--refresh-interval", type=int, default=10_000)
    p.add_argument("--segmenter-fit-after", type=int, default=6_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eps-experience", type=float, default=0.8)
    p.add_argument("--budget-chunks", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--quick", action="store_true", help="short smoke run")
    args = p.parse_args()

    if args.quick:
        args.total_env_steps = 12_000
        args.eval_interval = 3_000
        args.routing_interval = 3_000
        args.refresh_interval = 3_000
        args.segmenter_fit_after = 3_000
        args.episodes = 10

    cfg = build_config(args)
    trainer = OnlineTrainer(cfg, env_factory=env_factory_for(args.env))
    results = trainer.train()

    run_name = f"{args.env.replace(':', '_')}_{args.router}_seed{args.seed}"
    out = Path(args.outdir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(results["history"], out / "learning_curves.png")
    plot_vocab_over_time(results["history"], out / "vocab_over_time.png")
    C = results["competence"]
    if C.size:
        plot_competence_and_deficit(C, results["deficit"], results["experience_ids"],
                                    out / "competence_deficit.png")
    if results["gamma"] is not None:
        plot_transport_matrix(results["gamma"], cfg.population_size, results["experience_ids"],
                              out / "uot_transport.png")

    metrics = {
        "router": args.router,
        "env": args.env,
        "population_summary": results["population_summary"],
        "budget": results["budget"],
        "timing": results["timing"],
        "num_experiences": int(len(trainer.vocabulary)),
        "num_valid_experiences": int(len(trainer.valid_ids)),
        "complementarity": complementarity_stats(C) if C.size else {},
        "final_eval": {str(k): v for k, v in results["final_eval"].items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nArtifacts written to {out}/")
    print(json.dumps(metrics["population_summary"], indent=2))


if __name__ == "__main__":
    main()
