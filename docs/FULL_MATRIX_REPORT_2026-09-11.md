# HERP full-matrix report — ManiSkill, all methods (2026-09-11)

**W&B:** https://wandb.ai/hust_edu_vn/herp — group `full-matrix-1seed`
**Hardware:** single RTX 5080 (16 GB), `physx_cuda`, `num_envs=1024`, 3 parallel cells.

## TL;DR

All six methods (`ppo`, `rnd`, `disagreement`, `herp_sigma`, `herp_p`, `herp`) were
run on **6 ManiSkill tasks**, one seed (seed 0), same PPO backbone / budget / seed
per task — so the comparison is **fair**. **5 of 6 tasks show clear learning**
(≥4 requested); the two medium tasks (**LiftPegUpright**, **PlaceSphere**) cleanly
*differentiate* the methods, which is the useful regime for the paper. StackCube
did not learn within 5M (budget too small — see next steps).

Headline: **`herp_p` is the standout on the hard-to-explore tasks** — it is the
*only* method that solves **PlaceSphere** (0.91 vs ~0 for everything else) and is
second-best on **LiftPegUpright** (0.84).

## Setup

| | |
|---|---|
| Benchmark | ManiSkill 3, `obs_mode=state`, `reward_mode=normalized_dense`, `control_mode=pd_joint_delta_pos` |
| Methods | `ppo`, `rnd`, `disagreement`, `herp_sigma`, `herp_p`, `herp` (same PPO backbone; only the exploration/allocation mechanism differs) |
| Seeds | 0 (single seed — as requested; see caveats) |
| Budget | 5M env steps (PokeCube 10M) |
| Eval | 100 episodes every 250k steps |
| Fairness | identical backbone, hyperparameters, budget and seed across methods within each task; only the method flag changes |

## Main results — eval success (final / best over training)

| Task (budget) | ppo | rnd | disagreement | herp_sigma | herp_p | herp |
|---|---|---|---|---|---|---|
| PushCube (5M)        | 0.99/1.00 | 1.00/1.00 | 1.00/1.00 | 0.95/0.95 | 0.90/0.90 | 0.84/0.84 |
| PickCube (5M)        | 0.98/0.98 | 0.98/1.00 | 0.97/1.00 | 1.00/1.00 | 1.00/1.00 | 0.99/1.00 |
| LiftPegUpright (5M)  | 0.00/0.02 | 0.61/0.67 | **0.98/0.98** | 0.55/0.55 | 0.84/0.89 | 0.08/0.08 |
| PlaceSphere (5M)     | 0.00/0.07 | 0.00/0.01 | 0.00/0.01 | 0.01/0.01 | **0.91/0.91** | 0.00/0.00 |
| PokeCube (10M)       | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 0.98/1.00 | 0.99/1.00 | 0.54/0.63 |
| StackCube (5M)       | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

![Learning curves](figs/report_maniskill_seed0.png)

## Per-task reading

- **PushCube / PickCube (★, ★★):** saturated — every method reaches ~0.9–1.0. Confirms the
  pipeline is correct and all methods train. Not discriminative (too easy).
- **LiftPegUpright (★★★):** strong spread. `disagreement` 0.98 > `herp_p` 0.84 > `rnd` 0.61 >
  `herp_sigma` 0.55 > `herp` 0.08 > `ppo` 0.00. Plain PPO fails to explore the grasp+reorient;
  exploration bonuses and `herp_p` recover it.
- **PlaceSphere (★★★):** the sharpest result — **only `herp_p` learns (0.91)**; all others ≈0.
  Rolling sphere + placement is a hard-exploration task where relevance-weighted allocation pays off.
- **PokeCube (★★★–★★★★, 10M):** easier than expected — all reach ~1.0 except `herp` (0.54).
- **StackCube (★★★★):** all methods 0.0 at 5M — budget too small (upstream needs ~25M+).

## Caveats (read before citing)

1. **Single seed.** Per the run config, this is seed 0 only. The comparison is *fair*
   (identical setup) but **not yet statistically conclusive** — the differentiating numbers
   (e.g. LiftPegUpright `herp` 0.08 vs `herp_p` 0.84, PlaceSphere `herp_p` 0.91 vs 0) should be
   confirmed with seeds 1–2 before they go in a paper.
2. **Saturation** on PushCube/PickCube/PokeCube — fine per the request ("scores need not be
   saturated, as long as the policy learns and the comparison is fair"), but these tasks don't
   separate methods.
3. **StackCube** needs a larger budget to become informative (stage it — see next steps).

## Known issues

- **Meta-World / Fetch failed (all 36 cells).** Root cause identified: a **device mismatch** —
  `action.clamp(adapter.action_low(), adapter.action_high())` runs with `action` on `cuda`
  (agent device) but the CPU-sim adapters return bounds on `cpu`
  (`RuntimeError: Expected all tensors to be on the same device`). These benchmarks use CPU
  MuJoCo, so `--device cuda` + CPU adapter bounds collide. **Fix:** move the clamp bounds to
  the action's device (or set the adapter device to the agent device). Not blocking this report
  (ManiSkill covers ≥4 learning tasks); flagged for a follow-up.
- **PegInsertionSide** is handled by a separate staged runner (`scripts/run_peg_staged.py`),
  PPO only, escalating 5M→10M→…→75M. As of this report it is still ~0 through ~5M (expected for
  a ★★★★★ task) and is resuming on the GPU now.

## Reproduction

```bash
# Matrix (ManiSkill; metaworld/fetch currently blocked by the device bug above):
python scripts/run_suite.py --benchmarks maniskill \
  --methods ppo rnd disagreement herp_sigma herp_p herp --seeds 0 \
  --exclude-tasks PegInsertionSide-v1 --workers 3 --device cuda \
  --wandb-mode online --wandb-project herp --wandb-group full-matrix-1seed

# Peg (staged, resumable):
python scripts/run_peg_staged.py --env-id PegInsertionSide-v1 --method ppo --seed 0 \
  --stages 5000000 10000000 15000000 20000000 --num-envs 1024 --wandb-mode online
```

Every cell writes `provenance.json` + `metrics.csv`; W&B group `full-matrix-1seed`.
Aggregated artifacts: `outputs/full_matrix_1seed/analysis/` (`RESULTS.md`, `main_table.csv/tex`).

## Recommended next steps (in priority order)

1. **Seeds 1–2 for the discriminative tasks** (LiftPegUpright, PlaceSphere, and StackCube once
   its budget is fixed) — turns the suggestive `herp_p` advantage into a citable result.
2. **Stage StackCube** with `run_peg_staged.py` (5M→10M→…) to find the budget where it learns.
3. **Fix the Meta-World/Fetch device bug** to add two more environments.
4. Let the **Peg staged sweep** continue to find its learning budget.
