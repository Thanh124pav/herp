"""Milestone 5 diagnostic -- competence + deficit heatmaps (PLAN.md sections 9, 10, 16).

Runs a population loop and renders the policy x experience competence heatmap and
the receiver x experience deficit heatmap, plus Gate-C complementarity stats.

    python scripts/inspect_competence.py --total-env-steps 40000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experience_routing.competence.deficit import deficit_matrix
from experience_routing.evaluation.routing_metrics import (
    complementarity_stats,
    plot_competence_and_deficit,
)
from experience_routing.pipeline import OnlineTrainer, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-env-steps", type=int, default=40_000)
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--outdir", default="outputs/competence_diagnostics")
    args = p.parse_args()

    cfg = PipelineConfig(
        population_size=args.population_size,
        total_env_steps=args.total_env_steps,
        router="no_share",
        segmenter_fit_after=6_000,
        eval_interval=10_000,
        verbose=True,
    )
    trainer = OnlineTrainer(cfg)
    res = trainer.train()
    C = res["competence"]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if C.size:
        plot_competence_and_deficit(C, deficit_matrix(C), res["experience_ids"],
                                    out / "competence_deficit.png")
        stats = complementarity_stats(C)
        print(json.dumps(stats, indent=2, default=float))
        print(f"heatmap written to {out}/")
        gate_c = stats["frac_experiences_distinct_best"] > (1.0 / args.population_size)
        print("Gate C (complementary competence):", "GO" if gate_c else "inspect")
    else:
        print("no valid experiences discovered -- run longer or lower thresholds")


if __name__ == "__main__":
    main()
