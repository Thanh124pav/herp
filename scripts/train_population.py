"""Milestone 1 -- N independent policies, no sharing (PLAN.md section 16).

    python scripts/train_population.py --population-size 4 --total-env-steps 40000

Runs the online loop with the ``no_share`` router (pure independent population)
and reports per-policy curves + matched total interactions.
"""

from __future__ import annotations

import argparse
import json

from experience_routing.pipeline import OnlineTrainer, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--total-env-steps", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = PipelineConfig(
        population_size=args.population_size,
        total_env_steps=args.total_env_steps,
        router="no_share",
        seed=args.seed,
    )
    trainer = OnlineTrainer(cfg)
    results = trainer.train()
    print(json.dumps({
        "population_summary": results["population_summary"],
        "budget": results["budget"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
