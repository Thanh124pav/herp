"""Experiment 3 / Milestone 14 -- controlled transfer validation (PLAN.md 19.3).

The *key diagnostic before full online routing*: does the receiver deficit
identify experience that actually helps the receiver? For selected
``(receiver, experience)`` deficit pairs we run a checkpoint / route / update /
evaluate / restore probe and compare the receiver-deficit choice against three
controls that use the **same routing budget**:

    control    -- K updates, no routed data (isolates "more updates" from "routed data")
    deficit    -- route donor chunks for the identified deficit experience  (proposed)
    mastered   -- route donor chunks for an experience the receiver already masters
    random     -- route donor chunks for a random other valid experience
    high_td    -- route the globally highest-TD-error transitions (SUPER-style)

Procedure per (pair, condition):

    snapshot receiver policy
      -> reset routed buffer, inject the condition's chunks
      -> K gradient updates (local + routed mix)
      -> evaluate receiver success/return
    restore snapshot (+ clear routed buffer)

The hypothesis is supported if ``deficit`` yields the largest gain over
``control`` -- larger than routing already-mastered / random / globally-high-TD
experience.

    python scripts/experiment_controlled_transfer.py --warmup-steps 24000 --pairs 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experience_routing.competence.deficit import deficit_matrix  # noqa: E402
from experience_routing.pipeline import OnlineTrainer, PipelineConfig  # noqa: E402
from experience_routing.routing.base import add_chunks_to_receiver  # noqa: E402

CONDITIONS = ["control", "deficit", "mastered", "random", "high_td"]


def fill_experience_chunks(receiver, donor_id, experience_id, bank, vocabulary, budget):
    chunks = bank.select(donor_id, experience_id, vocabulary, budget)
    if not chunks:
        return 0
    return add_chunks_to_receiver(receiver, chunks)


def fill_high_td(receiver, population, budget_transitions, candidate_pool=512):
    """SUPER-style: inject the globally highest receiver-TD transitions."""
    pooled = []  # (td, donor_idx, j)
    donor_batches = {}
    for wi, w in enumerate(population):
        if w.policy_id == receiver.policy_id or len(w.local_buffer) == 0:
            continue
        take = min(candidate_pool, len(w.local_buffer))
        batch = w.local_buffer.sample(take)
        td = receiver.agent.td_error(batch)  # score with the *receiver's* critic
        donor_batches[wi] = batch
        for j in range(take):
            pooled.append((td[j], wi, j))
    if not pooled:
        return 0
    pooled.sort(key=lambda x: -x[0])
    n = 0
    for _, wi, j in pooled[:budget_transitions]:
        b = donor_batches[wi]
        receiver.add_routed_transitions(
            [b["obs"][j]], [b["actions"][j]], [b["rewards"][j]],
            [b["next_obs"][j]], [b["dones"][j]],
        )
        n += 1
    return n


def probe(receiver, population, condition, target, C, ids, bank, vocabulary,
          budget, budget_transitions, k_updates, batch_size, eval_episodes, eval_seed, rng):
    """Run one checkpoint/route/update/evaluate/restore probe. Returns eval dict."""
    r, e_idx = target["receiver"], target["e_idx"]
    snap = receiver.snapshot_policy()
    receiver.reset_routed_buffer()

    injected = 0
    if condition == "deficit":
        injected = fill_experience_chunks(receiver, target["donor"], ids[e_idx], bank, vocabulary, budget)
    elif condition == "mastered":
        # experience the receiver already masters best (donor = frontier owner)
        mastered_e = int(np.argmax(C[r]))
        donor = int(np.argmax(C[:, mastered_e]))
        if donor == r:
            donor = int(np.argsort(C[:, mastered_e])[-2]) if C.shape[0] > 1 else r
        injected = fill_experience_chunks(receiver, donor, ids[mastered_e], bank, vocabulary, budget)
    elif condition == "random":
        others = [j for j in range(len(ids)) if j != e_idx]
        if others:
            e_rand = int(rng.choice(others))
            donor = int(np.argmax(C[:, e_rand]))
            if donor == r and C.shape[0] > 1:
                donor = int(np.argsort(C[:, e_rand])[-2])
            injected = fill_experience_chunks(receiver, donor, ids[e_rand], bank, vocabulary, budget)
    elif condition == "high_td":
        injected = fill_high_td(receiver, population, budget_transitions)
    # "control": nothing injected

    for _ in range(k_updates):
        receiver.train_step(batch_size)
    result = receiver.evaluate(episodes=eval_episodes, base_seed=eval_seed)
    result["injected"] = injected

    receiver.restore_policy(snap)
    receiver.reset_routed_buffer()
    return result


def select_targets(C, ids, bank, n_pairs, min_deficit=0.02):
    """Pick high-deficit (receiver, experience) cells with a real donor supply."""
    D = deficit_matrix(C)
    N, K = C.shape
    cells = []
    for r in range(N):
        for e in range(K):
            if D[r, e] < min_deficit:
                continue
            donor = int(np.argmax(C[:, e]))
            if donor == r or bank.n_successful(donor, ids[e]) <= 0:
                continue
            cells.append((D[r, e], r, e, donor))
    cells.sort(key=lambda x: -x[0])
    targets = []
    for dval, r, e, donor in cells[:n_pairs]:
        targets.append({"receiver": r, "e_idx": e, "experience_id": int(ids[e]),
                        "donor": donor, "deficit": float(dval)})
    return targets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup-steps", type=int, default=24_000,
                   help="env steps to develop differentiated competence + banks")
    p.add_argument("--pairs", type=int, default=4, help="target (receiver,experience) pairs")
    p.add_argument("--k-updates", type=int, default=200, help="gradient updates per probe")
    p.add_argument("--repeats", type=int, default=3, help="probe repeats per (pair, condition)")
    p.add_argument("--budget-chunks", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/controlled_transfer")
    args = p.parse_args()

    # 1. develop a population with differentiated competence + banked exemplars.
    cfg = PipelineConfig(
        population_size=args.population_size,
        total_env_steps=args.warmup_steps,
        router="no_share",  # no online routing -- clean isolation for the probe
        routing_interval=5_000,
        refresh_interval=8_000,
        eval_interval=8_000,
        segmenter_fit_after=6_000,
        budget_chunks=args.budget_chunks,
        seed=args.seed,
        verbose=True,
    )
    print("=== developing population (no_share) ===")
    trainer = OnlineTrainer(cfg)
    trainer.train()

    ids, C, n_success = trainer._valid_matrix()
    if C.size == 0 and trainer._last_matrix is not None:
        ids, C, n_success = trainer._last_matrix
    if C.size == 0:
        print("No valid experiences discovered; increase --warmup-steps. Aborting.")
        return

    bank, vocabulary = trainer.bank, trainer.vocabulary
    budget_transitions = args.budget_chunks * cfg.min_chunk_len * 2
    targets = select_targets(C, ids, bank, args.pairs)
    if not targets:
        print("No high-deficit (receiver, experience) pairs with donor supply. Aborting.")
        return

    print(f"\n=== {len(targets)} target deficit pairs ===")
    for t in targets:
        print(f"  receiver p{t['receiver']} <- donor p{t['donor']} "
              f"on e{t['experience_id']} (deficit={t['deficit']:.3f})")

    rng = np.random.default_rng(args.seed)
    # 2. run controlled probes.
    # results[condition] = list of delta_success (over control) and raw success
    raw = {c: [] for c in CONDITIONS}
    per_target = []
    for ti, t in enumerate(targets):
        receiver = trainer.population[t["receiver"]]
        eval_seed = 2_000_000 + 1000 * ti
        cond_success = {c: [] for c in CONDITIONS}
        for rep in range(args.repeats):
            for cond in CONDITIONS:
                res = probe(receiver, trainer.population, cond, t, C, ids, bank, vocabulary,
                            args.budget_chunks, budget_transitions, args.k_updates,
                            args.batch_size, args.eval_episodes, eval_seed + rep, rng)
                cond_success[cond].append(res["success_rate"])
        control_mean = float(np.mean(cond_success["control"]))
        row = {"target": t, "control_success": control_mean}
        for cond in CONDITIONS:
            m = float(np.mean(cond_success[cond]))
            raw[cond].append(m)
            row[f"{cond}_success"] = m
            row[f"{cond}_gain"] = m - control_mean
        per_target.append(row)
        print(f"\n[pair {ti}] receiver p{t['receiver']} e{t['experience_id']} "
              f"control={control_mean:.3f}")
        for cond in CONDITIONS:
            if cond == "control":
                continue
            print(f"    {cond:9s} success={np.mean(cond_success[cond]):.3f} "
                  f"gain={np.mean(cond_success[cond]) - control_mean:+.3f}")

    # 3. aggregate + plot.
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    control_arr = np.array(raw["control"])
    summary = {}
    for cond in CONDITIONS:
        arr = np.array(raw[cond])
        gains = arr - control_arr
        summary[cond] = {
            "mean_success": float(arr.mean()),
            "mean_gain_over_control": float(gains.mean()),
            "std_gain": float(gains.std()),
        }

    conds = [c for c in CONDITIONS if c != "control"]
    means = [summary[c]["mean_gain_over_control"] for c in conds]
    stds = [summary[c]["std_gain"] for c in conds]
    colors = ["tab:red" if c == "deficit" else "tab:gray" for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(conds, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean success gain over control")
    ax.set_title(f"Controlled transfer (Exp 3): deficit vs controls\n"
                 f"{len(targets)} pairs x {args.repeats} repeats, {args.k_updates} updates")
    fig.tight_layout()
    fig.savefig(out / "controlled_transfer.png", dpi=120)
    plt.close(fig)

    best = max(conds, key=lambda c: summary[c]["mean_gain_over_control"])
    verdict = ("SUPPORTED: receiver-deficit routing gives the largest gain"
               if best == "deficit"
               else f"NOT supported: '{best}' beat deficit -- inspect competence signal")
    payload = {
        "verdict": verdict,
        "best_condition": best,
        "summary": summary,
        "targets": [{k: (v if not isinstance(v, np.integer) else int(v))
                     for k, v in t.items()} for t in targets],
        "per_target": per_target,
        "config": {"warmup_steps": args.warmup_steps, "pairs": len(targets),
                   "k_updates": args.k_updates, "repeats": args.repeats,
                   "budget_chunks": args.budget_chunks},
    }
    (out / "controlled_transfer.json").write_text(json.dumps(payload, indent=2, default=float))

    print("\n=== VERDICT ===")
    print(verdict)
    for c in conds:
        print(f"  {c:9s} gain={summary[c]['mean_gain_over_control']:+.3f} "
              f"+/- {summary[c]['std_gain']:.3f}")
    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
