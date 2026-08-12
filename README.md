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
| QMP-style receiver-Q selection (B5) | ✅ interface hook (`routing/qmp_style.py`) |
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
python scripts/run_baseline.py --routers no_share share_all random td_priority greedy uot

# Milestone diagnostics
python scripts/train_single.py                 # Gate A: single SAC learns
python scripts/inspect_chunks.py               # section 6.3 boundary plots
python scripts/inspect_experiences.py          # section 7 vocabulary
python scripts/inspect_competence.py           # sections 9-10 heatmaps + Gate C
```

`run_full.py` writes, under `outputs/<run>/`:

- `learning_curves.png` — per-policy success + return;
- `vocab_over_time.png` — experience vocabulary size `K_t`;
- `competence_deficit.png` — competence and deficit heatmaps;
- `uot_transport.png` — donor→receiver transport matrix (uot router);
- `metrics.json` — population mean/worst/best, budget, complementarity stats.

Swap to Meta-World with `--env metaworld:reach-v2` once the extras are installed.

## Tests

```bash
pytest -q
```

Covers the section-22 suite: segmenter (boundaries sorted / no overlap / full
cover / min length), grouper (same vs. different experience, deterministic
prototype update), vocabulary merge, validity, competence (A 8/10 > B 2/10),
deficit (frontier = max, best deficit = 0), and the **OT gate** (A:e1→B:e1,
B:e2→A:e2) that the plan requires to pass before any routing experiment.

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
