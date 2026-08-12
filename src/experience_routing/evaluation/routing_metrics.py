"""Routing metrics + artifact plotting (PLAN.md sections 21, 28).

Produces the diagnostic artifacts the Definition-of-Done requires: learning
curves, experience-vocabulary-over-time, competence/deficit heatmaps, and the
UOT transport matrix. Matplotlib with the non-interactive Agg backend so it runs
headless on CPU.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _heatmap(ax, M, title, xlabel, ylabel):
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def plot_learning_curves(history: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    steps = history["eval_steps"]
    for pid, curve in history["success"].items():
        ax1.plot(steps, curve, marker="o", label=f"policy {pid}")
    ax1.set_title("Success rate")
    ax1.set_xlabel("total env steps")
    ax1.set_ylabel("success rate")
    ax1.legend(fontsize=8)
    for pid, curve in history["return"].items():
        ax2.plot(steps, curve, marker="o", label=f"policy {pid}")
    ax2.set_title("Mean return")
    ax2.set_xlabel("total env steps")
    ax2.set_ylabel("mean return")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_vocab_over_time(history: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["eval_steps"], history["num_experiences"], marker="o", label="total")
    ax.plot(history["eval_steps"], history["num_valid_experiences"], marker="s", label="valid")
    ax.set_title("Experience vocabulary over time (K_t)")
    ax.set_xlabel("total env steps")
    ax.set_ylabel("num experiences")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_competence_and_deficit(competence, deficit, experience_ids, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    im1 = _heatmap(ax1, competence, "Competence C[i,e]", "experience", "policy")
    fig.colorbar(im1, ax=ax1)
    im2 = _heatmap(ax2, deficit, "Deficit[i,e]", "experience", "policy")
    fig.colorbar(im2, ax=ax2)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(experience_ids)))
        ax.set_xticklabels(experience_ids, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_transport_matrix(gamma, n_policies, experience_ids, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = _heatmap(ax, gamma, "UOT transport gamma (src->dst)",
                  "target (policy,experience)", "source (policy,experience)")
    fig.colorbar(im)
    labels = [f"p{p}:e{experience_ids[e]}" for p in range(n_policies) for e in range(len(experience_ids))]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def complementarity_stats(competence: np.ndarray) -> dict:
    """Gate C diagnostics (PLAN.md sections 19.1, 25)."""
    if competence.size == 0:
        return {"num_experiences": 0, "frac_experiences_distinct_best": 0.0, "mean_frontier_gap": 0.0}
    best = competence.argmax(axis=0)
    frontier = competence.max(axis=0)
    second = np.sort(competence, axis=0)[-2] if competence.shape[0] > 1 else frontier
    return {
        "num_experiences": int(competence.shape[1]),
        "frac_experiences_distinct_best": float(len(set(best.tolist())) / max(1, competence.shape[0])),
        "mean_frontier_gap": float(np.mean(frontier - second)),
        "best_policy_per_experience": best.tolist(),
    }
