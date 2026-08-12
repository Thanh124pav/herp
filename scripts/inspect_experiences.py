"""Milestone 4 diagnostic -- experience vocabulary inspection (PLAN.md sections 7, 16).

Runs a short population loop, then reports vocabulary size K_t, cluster-support
histogram, and the largest experiences' prototypes.

    python scripts/inspect_experiences.py --total-env-steps 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-env-steps", type=int, default=20_000)
    p.add_argument("--outdir", default="outputs/experience_diagnostics")
    args = p.parse_args()

    cfg = PipelineConfig(total_env_steps=args.total_env_steps, router="no_share",
                         eval_interval=5_000, segmenter_fit_after=5_000, verbose=True)
    trainer = OnlineTrainer(cfg)
    trainer.train()

    vocab = trainer.vocabulary
    sizes = sorted((e.n_chunks for e in vocab.experiences.values()), reverse=True)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(sizes)), sizes)
    ax.set_title(f"Cluster-support histogram (K={len(vocab)})")
    ax.set_xlabel("experience (sorted by size)")
    ax.set_ylabel("n_chunks")
    fig.tight_layout()
    fig.savefig(out / "cluster_support_hist.png", dpi=110)
    plt.close(fig)

    top = sorted(vocab.experiences.values(), key=lambda e: -e.n_chunks)[:8]
    print(f"K_t (total experiences) = {len(vocab)}; valid = {len(trainer.valid_ids)}")
    print("merge_count =", vocab.merge_count)
    for e in top:
        print(f"  exp {e.experience_id}: n_chunks={e.n_chunks} "
              f"n_success={e.n_success_chunks} valid={e.valid} "
              f"pre[:3]={np.round(e.precondition_center[:3], 2)} "
              f"eff[:3]={np.round(e.effect_center[:3], 2)}")
    print(f"histogram written to {out}/")


if __name__ == "__main__":
    main()
