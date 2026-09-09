# Experience Routing (MVP)

Receiver-aware **experience-level routing** in a population of independently
learning robot policies. This is the MVP implementation of the research plan in
[`PLAN.md`](PLAN.md).

> **Hypothesis.** A population of independently learning policies discovers
> *complementary* functional experiences; routing those experiences from strong
> donors to weak receivers improves learning over independent training,
> indiscriminate shared replay, and global-priority sharing.

The scientific core — the **donor→receiver routing problem** — is what this
codebase implements and verifies, exactly as the plan scopes it (no learned
visual/skill embeddings, no VLA/VLM, no RL coordinator).

```
Parallel online SAC
  → state-transition segmentation
  → dynamic experience grouping
  → success-support validity filter
  → policy × experience competence matrix
  → minimax receiver deficit / demand
  → (greedy | unbalanced-OT) donor→receiver routing
  → direct replay → policy updates
```

## What runs here vs. what's deferred

This repo is designed to run the **entire pipeline end-to-end on CPU** against a
fast **synthetic pick-and-place** environment, so the routing signal is fully
observable and unit-tested. The heavy RL backbone (SAC) and the real **Meta-World**
suite are wired behind the same interfaces as **drop-in swaps** — a GPU session
can run the headline B0–B9 experiments unchanged.

| Component | Status |
|---|---|
| SAC backbone (PyTorch, CPU) | ✅ implemented |
| Synthetic env (segmenter-ready state fields) | ✅ implemented |
| Segmentation / grouping / validity / competence / deficit | ✅ implemented + unit-tested |
| Routers: `no_share`, `share_all`, `random`, `td_priority`, `greedy`, `uot` | ✅ implemented |
| QMP-style receiver-Q selection (B5) | ✅ working baseline (collection-time behavior selection) |
| Controlled-transfer validation (Experiment 3, checkpoint/route/update/restore) | ✅ implemented (`scripts/experiment_controlled_transfer.py`) |
| §21 logging (critic/actor loss, donor/receiver/experience dists, transfer rates) | ✅ implemented |
| Meta-World adapter | ✅ drop-in (`envs/metaworld_env.py`, needs `.[metaworld]`) |
| Meta-World convergence runs, SEAC, ManiSkill, WandB dashboards | ⏸ deferred (GPU) |

## Install

```bash
pip install -e .
# RL backbone (CPU wheel; the plan's optimizer/critic use PyTorch):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Optional real environment:
pip install -e '.[metaworld]'
```

## Run

```bash
# Full MVP run + all artifacts (PLAN.md section 28)
python scripts/run_full.py --env synthetic --population-size 4 --router uot

# Fast smoke run
python scripts/run_full.py --router uot --quick

# Routing comparison on matched budgets (Experiment 4, PLAN.md section 19.4)
python scripts/run_baseline.py --routers no_share share_all random td_priority qmp_style greedy uot

# Controlled-transfer validation (Experiment 3 / Milestone 14, PLAN.md section 19.3):
# checkpoint receiver -> route deficit chunks -> update -> evaluate -> restore,
# vs mastered / random / high-TD controls at the same budget.
python scripts/experiment_controlled_transfer.py --warmup-steps 32000 --pairs 4

# Milestone diagnostics
python scripts/train_single.py                 # Gate A: single SAC learns
python scripts/inspect_chunks.py --num 100     # section 6.3: 100 boundary plots + histogram
python scripts/inspect_experiences.py          # section 7 vocabulary
python scripts/inspect_competence.py           # sections 9-10 heatmaps + Gate C
```

`run_full.py` writes, under `outputs/<run>/`:

- `learning_curves.png` — per-policy success + return;
- `loss_curves.png` — per-policy critic/actor loss over time (§21);
- `vocab_over_time.png` — experience vocabulary size `K_t`;
- `competence_deficit.png` — competence and deficit heatmaps;
- `uot_transport.png` — donor→receiver transport matrix (uot router);
- `route_distributions.png` — donor / receiver / experience routed-chunk distributions (§21);
- `representative_chunks.png` — representative exemplar chunks per experience (DoD §28 item 4);
- `metrics.json` — population mean/worst/best, budget, complementarity, route stats, transfer rates.

### Logging (PLAN.md §21)

Every run logs the full §21 metric taxonomy — per-policy `return`/`success`/
`critic_loss`/`actor_loss`, experience-discovery counts, competence
frontier/deficit, routing counts + donor/receiver/experience distributions +
positive/negative-transfer rate, and compute timings. Metrics stream to
**Weights & Biases** *and*, always, to a local `log.jsonl` + `config.json` +
`summary.json` under the run dir — so a run is reviewable later even offline.

```bash
python scripts/run_full.py --router uot --wandb                    # → W&B (online)
python scripts/run_full.py --router uot --wandb --wandb-mode offline  # local W&B dir + JSONL
python scripts/run_full.py --router uot                            # JSONL/summary only, no W&B
```

### Meta-World

The adapter tracks the current **Farama Meta-World V3** suite (falls back to V2
if that's what's installed). It's state-based (39-dim), so **no rendering /
GPU** is involved — the cost is CPU MuJoCo stepping. A flow smoke on the
lightest task:

```bash
python scripts/run_full.py --env metaworld:reach-v3 --router uot --quick \
    --metaworld-horizon 120 --wandb --wandb-mode offline
```

`--metaworld-horizon` shortens episodes (more episodes per interaction budget)
for a fast pipeline check. Full B0–B7 convergence on Meta-World is the
overnight-scale, CPU-bound work the plan defers.

## Validated result (synthetic env, CPU)

A 48k-step UOT run (`--total-env-steps 48000`, N=4) end-to-end on CPU:

- all 4 policies learn the task (population mean success 0.23, worst 0.15);
- the pipeline discovers a dynamic experience vocabulary and marks **9 valid**
  experiences via the success-support filter;
- the competence matrix is **complementary** — different policies are strongest
  on different experiences (`best_policy_per_experience = [0,2,3,1,3,2,2,3,1]`,
  i.e. **all 4** policies each lead on some experience), which is **Gate C**
  (PLAN.md section 25) — the local-comparative-advantage the routing hypothesis
  needs;
- UOT routes **141 exemplar chunks** across 127 donor→receiver routes; the OT
  solve is sub-millisecond thanks to routing over the (capped) valid set only;
- once routing is active, **82%** of per-policy success moves are positive
  (positive-transfer rate, PLAN.md §21).

### Experiment 3 — controlled transfer validates the deficit signal

The **key diagnostic before full online routing** (PLAN.md §19.3 / Milestone 14):
for high-deficit `(receiver, experience)` pairs we checkpoint the receiver, route
donor chunks, run a fixed number of updates, evaluate, then restore — comparing
the receiver-deficit choice against controls at the *same budget*
(`scripts/experiment_controlled_transfer.py`). Mean success gain over a
no-routing control (4 pairs × 3 repeats):

| condition | mean gain over control |
|---|---:|
| **deficit** (proposed) | **+0.083** |
| high-TD (SUPER-style) | +0.033 |
| random experience | −0.021 |
| already-mastered experience | −0.058 |

Routing for the identified **deficit gives the largest gain** — larger than
globally-high-TD sharing, and routing already-mastered or random experience
actually *hurts*. This is the receiver-need signal the routing hypothesis rests
on, confirmed in isolation before any online routing loop.

Numbers vary with seed; the point is that the simple state-transition pipeline
produces a **non-trivial, receiver-specific routing signal** (PLAN.md section 26),
which is the MVP's question to answer before scaling to Meta-World/GPU.

## Tests

```bash
pytest -q
```

Covers the section-22 suite: segmenter (boundaries sorted / no overlap / full
cover / min length), grouper (same vs. different experience, deterministic
prototype update), vocabulary merge, validity, competence (A 8/10 > B 2/10),
deficit (frontier = max, best deficit = 0), and the **OT gate** (A:e1→B:e1,
B:e2→A:e2) that the plan requires to pass before any routing experiment. Plus
SAC checkpoint/restore round-trip and QMP-style receiver-Q selection (the
machinery behind Experiment 3 and the B5 baseline).

## Layout

```
src/experience_routing/
  envs/        synthetic_reacher, metaworld_env (drop-in), perturbations, base(EnvSpec)
  rl/          sac, actor, critic, replay_buffer  (local + routed buffers)
  population/  worker, population, rollout_manager (matched-budget accounting)
  experience/  segmenter, grouper, vocabulary, validity, bank, trajectory
  competence/  tracker, opportunity, deficit  (competence, frontier, supply)
  routing/     base, no_share, share_all, random, td_priority, qmp_style, greedy, uot
  evaluation/  evaluator, routing_metrics (plots), budget
  pipeline.py  OnlineTrainer — the section-15 online loop
scripts/       run_full, run_baseline, train_single, train_population, inspect_*
tests/         section-22 unit tests
configs/       section-23 defaults (env/policy/population/experience/routing)
```

## Prior-work references

Per `PLAN.md` section 2, this is **not** a fork of any one paper; the reference
repos are read-only. Clone them under `external/` if you want to consult them:

```bash
git clone https://github.com/clvrai/qmp                 external/qmp
git clone https://github.com/jesbu1/extract             external/extract
git clone https://github.com/mgerstgrasser/super        external/super
git clone https://github.com/uoe-agents/seac            external/seac
git clone https://github.com/Farama-Foundation/Metaworld external/metaworld
git clone https://github.com/PythonOT/POT               external/POT
```

`POT` (unbalanced Sinkhorn OT) is a normal dependency; the rest are conceptual
references (multi-policy SAC organization, cluster→smooth→chunk segmentation,
TD-priority / receiver-Q baselines).
