"""Slide-ready competence / deficit heatmaps (PLAN.md sections 9-10, 19.1).

Runs a short synthetic UOT pipeline and renders annotated, presentation-quality
heatmaps: competence C[i,e], deficit[i,e], with the frontier (best policy per
experience) highlighted -- the visual that shows *complementary* competence
(Gate C). Output: outputs/slides/{competence,deficit,competence_deficit}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experience_routing.competence.deficit import deficit_matrix  # noqa: E402
from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402


def _annotated_heatmap(ax, M, title, cmap, best_row=None, vmax=None, cbar_label=""):
    N, K = M.shape
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("experience", fontsize=11)
    ax.set_ylabel("policy", fontsize=11)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"e{j}" for j in range(K)], fontsize=9)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"P{i}" for i in range(N)], fontsize=10)
    # value annotations, contrast-aware text color
    thr = (vmax or M.max() or 1.0) * 0.55
    for i in range(N):
        for j in range(K):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > thr else "black", fontsize=8)
    # mark the frontier policy per experience (complementarity highlight)
    if best_row is not None:
        for j, i in enumerate(best_row):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="crimson", lw=2.2))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=10)
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-env-steps", type=int, default=48_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/slides")
    args = p.parse_args()

    cfg = PipelineConfig(
        population_size=4, total_env_steps=args.total_env_steps, router="uot",
        eval_interval=12_000, routing_interval=6_000, refresh_interval=12_000,
        segmenter_fit_after=6_000, eps_experience=0.8, seed=args.seed, verbose=True,
    )
    trainer = OnlineTrainer(cfg)
    res = trainer.train()
    C = res["competence"]
    if C.size == 0:
        print("no valid experiences; increase --total-env-steps")
        return
    D = deficit_matrix(C)
    best = C.argmax(axis=0)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # combined figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(1.6 + 1.1 * C.shape[1], 4.2))
    _annotated_heatmap(a1, C, "Competence  C[i, e]", "viridis", best_row=best,
                       vmax=1.0, cbar_label="P(success | opportunity)")
    _annotated_heatmap(a2, D, "Receiver deficit[i, e]", "magma",
                       vmax=float(D.max()) or 1.0, cbar_label="frontier − competence")
    fig.suptitle(f"Policy × experience competence and deficit "
                 f"(N={C.shape[0]}, {C.shape[1]} valid experiences)\n"
                 f"red box = strongest policy per experience (complementary competence)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "competence_deficit.png", dpi=150)
    plt.close(fig)

    # individual figures for flexible slide layout
    for name, M, cmap, vmax, lbl, mark in [
        ("competence", C, "viridis", 1.0, "P(success | opportunity)", best),
        ("deficit", D, "magma", float(D.max()) or 1.0, "frontier − competence", None),
    ]:
        fig, ax = plt.subplots(figsize=(1.4 + 1.0 * C.shape[1], 4.0))
        _annotated_heatmap(ax, M, name.capitalize(), cmap, best_row=mark, vmax=vmax,
                           cbar_label=lbl)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    n_distinct = len(set(best.tolist()))
    print(f"\nwrote slide heatmaps to {out}/ "
          f"({C.shape[1]} experiences, {n_distinct}/{C.shape[0]} policies lead)")
    print(f"best_policy_per_experience = {best.tolist()}")


if __name__ == "__main__":
    main()
