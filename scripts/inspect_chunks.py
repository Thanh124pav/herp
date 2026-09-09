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
    p.add_argument("--num", type=int, default=100, help="trajectories to plot (PLAN.md 6.3)")
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
    # plot successes first (more interesting), then pad with any trajectories
    ordered = [t for t in trajs if t.success] + [t for t in trajs if not t.success]
    n_chunks_all = []
    for t in trajs:
        n_chunks_all.append(len(seg.segment(t)))

    for i, traj in enumerate(ordered[: args.num]):
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

    # summary diagnostic: chunks-per-trajectory histogram (PLAN.md section 6.3)
    n_chunks_all = np.asarray(n_chunks_all)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, n_chunks_all.max() + 2) - 0.5
    ax.hist(n_chunks_all, bins=bins, color="tab:purple", alpha=0.8)
    ax.set_title(f"Chunks per trajectory (N={len(n_chunks_all)} trajs)")
    ax.set_xlabel("num chunks")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "chunks_per_traj_hist.png", dpi=110)
    plt.close(fig)

    mean_c = float(n_chunks_all.mean())
    frac_single = float(np.mean(n_chunks_all <= 1))
    # a per-step-splitter would produce ~max_steps chunks; flag if mean is huge
    over_split = mean_c > 0.5 * pop[0].env.spec.max_steps
    n_plotted = min(args.num, len(ordered))
    print(f"wrote {n_plotted} boundary plots + summary histogram to {out}/")
    print(f"chunks/traj mean={mean_c:.2f} min={int(n_chunks_all.min())} "
          f"max={int(n_chunks_all.max())} frac_single_chunk={frac_single:.2f}")
    ok = (not over_split) and frac_single < 0.5 and mean_c > 1.0
    verdict = ("PASS" if ok else "CHECK") + " (Gate B sanity, PLAN.md 6.3/25)"
    print(f"segmenter sanity: {verdict} -- "
          f"{'not per-step, not one-chunk-per-traj' if ok else 'inspect thresholds'}")


if __name__ == "__main__":
    main()
