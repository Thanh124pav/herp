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
    plot_loss_curves,
    plot_representative_chunks,
    plot_route_distributions,
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
        eps_merge=args.eps_merge,
        min_success_support=args.min_success_support,
        eps_pre=args.eps_pre,
        eps_effect=args.eps_effect,
        horizon=args.competence_horizon,
        max_experiences=args.max_experiences,
        budget_chunks=args.budget_chunks,
        hidden=args.hidden,
        utd=args.utd,
        device=args.device,
        obs_norm=args.obs_norm,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return cfg


def env_factory_for(name: str, horizon: int | None = None):
    if name == "synthetic":
        from experience_routing.population.population import default_env_factory
        return default_env_factory
    if name.startswith("metaworld"):
        task = name.split(":", 1)[1] if ":" in name else "reach-v3"
        from experience_routing.envs.metaworld_env import make_metaworld_factory
        return make_metaworld_factory(task, horizon=horizon)
    raise ValueError(f"unknown env '{name}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="synthetic",
                   help="'synthetic' or 'metaworld:<task>' (e.g. metaworld:reach-v2)")
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--router", default="uot",
                   choices=["no_share", "share_all", "random", "td_priority",
                            "qmp_style", "greedy", "uot"])
    p.add_argument("--total-env-steps", type=int, default=60_000)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--routing-interval", type=int, default=5_000)
    p.add_argument("--refresh-interval", type=int, default=10_000)
    p.add_argument("--segmenter-fit-after", type=int, default=6_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eps-experience", type=float, default=0.8,
                   help="grouping threshold; calibrate per env (PLAN.md section 24)")
    p.add_argument("--eps-merge", type=float, default=None, help="default 0.7*eps_experience")
    p.add_argument("--min-success-support", type=int, default=3)
    p.add_argument("--eps-pre", type=float, default=2.0, help="competence opportunity radius")
    p.add_argument("--eps-effect", type=float, default=1.5, help="competence effect radius")
    p.add_argument("--competence-horizon", type=int, default=8)
    p.add_argument("--max-experiences", type=int, default=None,
                   help="hard cap on vocabulary size (anti-fragmentation)")
    p.add_argument("--budget-chunks", type=int, default=32)
    # backbone strength (PLAN.md section 5)
    p.add_argument("--hidden", type=int, default=128, help="actor/critic hidden width (Meta-World: 256)")
    p.add_argument("--utd", type=int, default=1, help="gradient updates per env step")
    p.add_argument("--device", default="cpu", help="'cuda' for GPU")
    p.add_argument("--obs-norm", action="store_true", help="running observation normalization")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--quick", action="store_true", help="short smoke run")
    p.add_argument("--metaworld-horizon", type=int, default=None,
                   help="shorten Meta-World episodes (more episodes per budget)")
    # logging (PLAN.md section 21)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="experience-routing")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    if args.quick:
        args.total_env_steps = 12_000
        args.eval_interval = 3_000
        args.routing_interval = 3_000
        args.refresh_interval = 3_000
        args.segmenter_fit_after = 3_000
        args.episodes = 10

    cfg = build_config(args)
    run_name = f"{args.env.replace(':', '_')}_{args.router}_seed{args.seed}"
    out = Path(args.outdir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    logger = None
    wandb_mode = args.wandb_mode
    if args.wandb or wandb_mode in ("offline",):
        from experience_routing.evaluation.logger import RunLogger
        logger = RunLogger(
            project=args.wandb_project, run_name=run_name,
            config={**vars(args), **{k: getattr(cfg, k) for k in ("eps_experience", "budget_chunks")}},
            use_wandb=(wandb_mode != "disabled"), wandb_mode=wandb_mode, outdir=out,
        )

    trainer = OnlineTrainer(
        cfg, env_factory=env_factory_for(args.env, args.metaworld_horizon), logger=logger,
    )
    results = trainer.train()

    plot_learning_curves(results["history"], out / "learning_curves.png")
    plot_loss_curves(results["history"], out / "loss_curves.png")
    plot_vocab_over_time(results["history"], out / "vocab_over_time.png")
    C = results["competence"]
    if C.size:
        plot_competence_and_deficit(C, results["deficit"], results["experience_ids"],
                                    out / "competence_deficit.png")
    if results["gamma"] is not None:
        plot_transport_matrix(results["gamma"], cfg.population_size, results["experience_ids"],
                              out / "uot_transport.png")
    if results["route_stats"]["route_count"] > 0:
        plot_route_distributions(results["route_stats"], cfg.population_size,
                                 out / "route_distributions.png")
    plot_representative_chunks(trainer.bank, trainer.vocabulary, results["experience_ids"],
                               trainer.spec, out / "representative_chunks.png")

    metrics = {
        "router": args.router,
        "env": args.env,
        "population_summary": results["population_summary"],
        "budget": results["budget"],
        "timing": results["timing"],
        "num_experiences": int(len(trainer.vocabulary)),
        "num_valid_experiences": int(len(trainer.valid_ids)),
        "complementarity": complementarity_stats(C) if C.size else {},
        "route_stats": results["route_stats"],
        "transfer": results["transfer"],
        "final_eval": {str(k): v for k, v in results["final_eval"].items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    if logger is not None:
        summ = {f"final/{k}": v for k, v in results["population_summary"].items()}
        summ["final/num_valid_experiences"] = int(len(trainer.valid_ids))
        summ["final/positive_transfer_rate"] = results["transfer"]["positive_transfer_rate"]
        summ["final/routed_chunks_total"] = results["route_stats"]["chunk_count"]
        logger.summary(summ)
        for png in out.glob("*.png"):
            logger.log_artifact_file(png)
        logger.finish()

    print(f"\nArtifacts written to {out}/")
    print(json.dumps(metrics["population_summary"], indent=2))


if __name__ == "__main__":
    main()
