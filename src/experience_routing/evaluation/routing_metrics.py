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


def plot_loss_curves(history: dict, out: Path) -> None:
    """Per-policy critic/actor loss over time (PLAN.md section 21 RL logging)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    steps = history["eval_steps"]
    for pid, curve in history.get("critic_loss", {}).items():
        ax1.plot(steps, curve, marker="o", label=f"policy {pid}")
    ax1.set_title("Critic loss")
    ax1.set_xlabel("total env steps")
    ax1.set_ylabel("critic loss")
    ax1.legend(fontsize=8)
    for pid, curve in history.get("actor_loss", {}).items():
        ax2.plot(steps, curve, marker="o", label=f"policy {pid}")
    ax2.set_title("Actor loss")
    ax2.set_xlabel("total env steps")
    ax2.set_ylabel("actor loss")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_route_distributions(route_stats: dict, n_policies: int, out: Path) -> None:
    """Donor / receiver / experience routed-chunk distributions (section 21)."""
    donor = route_stats.get("donor_counts", {})
    recv = route_stats.get("receiver_counts", {})
    exp = route_stats.get("experience_counts", {})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pids = list(range(n_policies))
    axes[0].bar(pids, [donor.get(p, 0) for p in pids], color="tab:blue")
    axes[0].set_title("Donor distribution")
    axes[0].set_xlabel("policy")
    axes[0].set_ylabel("routed chunks")
    axes[1].bar(pids, [recv.get(p, 0) for p in pids], color="tab:orange")
    axes[1].set_title("Receiver distribution")
    axes[1].set_xlabel("policy")
    eids = sorted(exp)
    axes[2].bar(range(len(eids)), [exp[e] for e in eids], color="tab:green")
    axes[2].set_title("Experience distribution")
    axes[2].set_xlabel("experience id")
    axes[2].set_xticks(range(len(eids)))
    axes[2].set_xticklabels(eids, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_representative_chunks(bank, vocabulary, experience_ids, spec, out: Path,
                               n_experiences: int = 6, n_chunks: int = 4) -> None:
    """Render representative exemplar chunks per experience (DoD section 28 item 4).

    For each of the largest valid experiences, show a few nearest-prototype
    exemplar chunks as their normalized task-feature trajectories with the
    pre-state (start) and post-state (effect) marked -- the visual sanity check
    that grouped chunks realize approximately the same state transition (Exp 2).
    """
    from ..experience.grouper import chunk_experience_distance

    # rank experiences by how many exemplar chunks exist across donors
    def n_exemplars(eid):
        return sum(bank.n_successful(p, eid) for p in range(64))

    ranked = sorted(experience_ids, key=lambda e: -n_exemplars(e))
    ranked = [e for e in ranked if n_exemplars(e) > 0][:n_experiences]
    if not ranked:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no exemplar chunks banked yet", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(ranked), 1, figsize=(9, 2.4 * len(ranked)), squeeze=False)
    for row, eid in enumerate(ranked):
        exp = vocabulary.experiences.get(eid)
        # gather exemplars across donors, nearest-to-prototype first
        cands = []
        for p in range(64):
            cands += bank.successful_chunks(p, eid)
        if exp is not None:
            cands.sort(key=lambda c: chunk_experience_distance(c, exp))
        cands = cands[:n_chunks]
        ax = axes[row][0]
        for k, c in enumerate(cands):
            # normalized task-feature trajectory, first dim, offset per exemplar
            tf = spec.task_features(c.states)
            ax.plot(range(len(tf)), tf[:, 0] + k * 0.0, alpha=0.7,
                    label=f"p{c.policy_id} ep{c.episode_id}")
            ax.scatter([0], [tf[0, 0]], color="green", zorder=5, s=20)
            ax.scatter([len(tf) - 1], [tf[-1, 0]], color="red", zorder=5, s=20)
        ax.set_title(f"experience e{eid}: {len(cands)} representative chunks "
                     f"(green=pre, red=post; task-feat[0])", fontsize=9)
        ax.set_xlabel("t within chunk")
        ax.legend(fontsize=6, ncol=4)
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
