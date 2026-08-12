"""Milestone 3 diagnostic -- chunk boundary plots (PLAN.md sections 6.3, 16).

Collects trajectories from a short independent rollout, fits the segmenter, and
renders per-trajectory boundary diagnostics: reward, cluster label, chunk
boundaries, and selected state deltas. Sanity check that the chunker neither
splits at every step nor returns one chunk per trajectory.

    python scripts/inspect_chunks.py --num 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experience_routing.experience.segmenter import Segmenter, step_features  # noqa: E402
from experience_routing.population.population import Population  # noqa: E402
from experience_routing.population.rollout_manager import RolloutManager  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=12, help="trajectories to plot")
    p.add_argument("--collect-rounds", type=int, default=1500)
    p.add_argument("--outdir", default="outputs/chunk_diagnostics")
    args = p.parse_args()

    pop = Population(size=4)
    rm = RolloutManager(pop, warmup_steps=800)
    trajs = []
    for _ in range(args.collect_rounds):
        trajs += rm.step(train=False)
    seg = Segmenter(pop[0].env.spec, k_seg=6, median_window=7, min_chunk_len=4)
    seg.fit(trajs)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    successes = [t for t in trajs if t.success] or trajs
    n_chunks_all = []
    for t in trajs:
        n_chunks_all.append(len(seg.segment(t)))

    for i, traj in enumerate(successes[: args.num]):
        chunks = seg.segment(traj)
        labels = seg._labels_for(traj)
        tf = seg.spec.task_features(traj.states)
        g = step_features(tf)
        fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        a0.plot(traj.rewards, color="tab:green")
        a0.set_ylabel("reward")
        a0.set_title(f"traj p{traj.policy_id} ep{traj.episode_id} success={traj.success} "
                     f"({len(chunks)} chunks)")
        a1.plot(labels, drawstyle="steps-post", color="tab:blue")
        a1.set_ylabel("cluster")
        for c in chunks:
            for a in (a0, a1, a2):
                a.axvline(c.start_t, color="red", alpha=0.35, lw=0.8)
        for d in range(min(g.shape[1], 5)):
            a2.plot(g[:, d], alpha=0.7, label=f"d{d}")
        a2.set_ylabel("state delta")
        a2.set_xlabel("t")
        a2.legend(fontsize=6, ncol=5)
        fig.tight_layout()
        fig.savefig(out / f"traj_{i:03d}.png", dpi=100)
        plt.close(fig)

    import numpy as np
    print(f"wrote {min(args.num, len(successes))} boundary plots to {out}/")
    print(f"chunks/traj mean={np.mean(n_chunks_all):.2f} "
          f"min={min(n_chunks_all)} max={max(n_chunks_all)}")


if __name__ == "__main__":
    main()
